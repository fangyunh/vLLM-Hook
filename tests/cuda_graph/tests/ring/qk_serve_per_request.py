"""Task 13 phase C ONLINE-SERVE validation harness for the per-request QK capture-ring DELIVERY path
(branch `capture_ring`). The QK analogue of serve_per_request.py (the HS Task-12 serve gate, job
584787 PASS). It drives the REAL serve code path — `AsyncLLM.generate` ->
`_hook_plugin._patched_generate` (the request-start router `_decide_ring_route_qk` + `route_ring_to_disk`,
the finalize BLOCK-UNTIL-HELD `get_ring_per_request` / disk `confirm_ring_delivery`, and the abort
`finally` -> `clear_ring_request`) — by firing CONCURRENT generate() coroutines against one AsyncLLM
engine (continuous batching), NO HTTP server. It asserts the same GATE as the HS serve harness, adapted
to QK's two-ring (q + growing-prefix k_all) structure:

  * MID-SERVING delivery: a request is delivered WHILE others are still generating (not at shutdown).
  * BYTE-IDENTITY: each delivered artifact == the SOLO-EAGER reference (rtol=atol=_SERVE_*), for BOTH
    the RPC (host-buffer) route AND the DISK (offload) route, with a strict layer-set gate. QK compares
    the per-layer flat "q" AND the growing-prefix "k_all" LIST entry-for-entry.
  * RESIDENCY -> 0: after every request delivered+freed the worker's PerRequestIndex AND disk staging
    are empty (queried NON-destructively via the `ring_residency` RPC, WITHOUT stopping the drain).
  * ABORT frees: a request CANCELLED mid-generation has its ring state (host index + disk staging)
    released, so residency returns to 0.
  * BOTH ROUTES exercised: short prompts -> small QK artifact -> RPC; long prompts -> large -> DISK
    (VLLM_HOOK_ROUTER_T_RPC set between the two predicted sizes).

QK-specific vs the HS harness:
  * ROUTABLE output_qk is a DICT {layer: [heads]} (from important_heads). `_decide_ring_route_qk`
    returns None for a non-dict output_qk (whole-model `output_qk=True`, e.g. the _alltok.json config)
    -> the safe host-buffer RPC default -> the DISK route would NEVER fire. So this harness uses the
    important-heads all_tokens config (9 distinct layers) and passes the SAME layer->heads dict to the
    graph leg's extra_args.
  * Layer key is 0-based (== eager match_attn / the ring reader), NOT HS's 1-based.
  * The graph (ring) per-request deliverable is ALREADY the ring's flat-q / growing-prefix-list k_all
    form (assemble_qk / load_multilayer_qk_ring_artifact) -> no normalization graph-side. The eager
    probes are PADDED tensors -> un-padded back to that form (`_flatten_eager_q` for q,
    `_split_eager_kall` for k_all, from qk_ring_parity.py). The disk sidecar is `qk_ring_meta.jsonl`.

Reused verbatim from serve_per_request.py: the concurrent AsyncLLM driver structure, the abort probe
(one RPC + one DISK leg so BOTH cleanup paths fire), the residency/mid-serving/routes gates, the
zstd-pickle RPC decode contract, and the wedged-delivery guard (a missing/partial disk dest -> empty
layer store + loud marker -> VERDICT FAIL, never a crash).

    python qk_serve_per_request.py capture --mode graph --out g.pkl
    python qk_serve_per_request.py capture --mode eager --out e.pkl
    python qk_serve_per_request.py compare --graph g.pkl --eager e.pkl
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import asyncio
import getpass
import json
import pickle
import sys
import time

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16
_WORKER_EXT_QK = "vllm_hook_plugins.workers.probe_hookqk_worker.ProbeHookQKWorker"

# The important-heads all_tokens config (dict output_qk -> routable). The SAME layer->heads dict feeds
# the eager (HookLLM) and graph (AsyncLLM extra_args) legs so both capture the identical 9 layers.
_CONFIG_FILE = os.environ.get(
    "VLLM_HOOK_CONFIG_FILE",
    f"model_configs/attention_tracker/{_MODEL.split('/')[-1]}_alltok_heads.json")

# Deterministic generation length for the delivered (main-batch) requests -> a fixed per-request
# artifact shape shared by the graph + eager legs. ignore_eos makes it exact regardless of tokens.
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_SERVE_MAX_TOKENS", "8"))
# Concurrency: N_RPC short prompts (route RPC) + N_DISK long prompts (route DISK), fired together.
_N_RPC = int(os.environ.get("VLLM_HOOK_SERVE_N_RPC", "4"))
_N_DISK = int(os.environ.get("VLLM_HOOK_SERVE_N_DISK", "4"))
# Prompt token lengths chosen so predicted QK bytes straddle VLLM_HOOK_ROUTER_T_RPC (see the .sh):
# short -> small artifact -> RPC; long -> large artifact -> DISK. Uniform hookq_mode=all_tokens +
# hooks_on=both, so predicted bytes scale purely with sequence length (the router's size model).
# For the 14-head/9-layer alltok config: len6+gen8 ~= 98 KB (RPC), len220+gen8 ~= 1.56 MiB (DISK).
_LEN_RPC = int(os.environ.get("VLLM_HOOK_SERVE_LEN_RPC", "6"))
_LEN_DISK = int(os.environ.get("VLLM_HOOK_SERVE_LEN_DISK", "220"))

_HOOKS_ON = "both"

# Abort-probe RPC leg: max_tokens small enough that the request's predicted QK artifact stays under
# VLLM_HOOK_ROUTER_T_RPC so it routes RPC (host index), yet large enough that it is reliably still
# generating when cancelled. len6+gen40 ~= 322 KB < 512 KiB -> RPC. The DISK abort leg keeps a large
# max_tokens (long prompt routes DISK regardless). Retune with T_RPC if a model/config moves the
# crossover, or the abort non-vacuity gate FAILs loudly.
_ABORT_RPC_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_SERVE_ABORT_RPC_MAX_TOKENS", "40"))
_ABORT_DISK_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_SERVE_ABORT_DISK_MAX_TOKENS", "500"))

# Reserved (non-case) key carrying the graph leg's serve-only evidence into compare().
_META_KEY = "__serve_meta__"

# BYTE-IDENTITY tolerance for the serve compare (Gate 5), env-overridable. 5e-2 (vs the shutdown
# oracle's 1e-2) to accept the KNOWN graph-vs-eager numerical divergence at the DEEPEST captured
# decoder layers on LONG sequences (the important-heads set reaches layer 19); a real cross-request
# bleed is orders of magnitude larger (wrong values / wrong shape), which 5e-2 still catches, while the
# layer-SET / SHAPE / step-count / residency / abort gates below stay STRICT.
_SERVE_RTOL = float(os.environ.get("VLLM_HOOK_SERVE_RTOL", "5e-2"))
_SERVE_ATOL = float(os.environ.get("VLLM_HOOK_SERVE_ATOL", "5e-2"))


def _layer_to_heads():
    """{layer:int -> [head:int]} from the config's important_heads (the SAME dict HookLLM builds into
    output_qk), so the graph and eager legs capture the identical layer set."""
    cfg = json.load(open(_CONFIG_FILE))
    l2h = {}
    for L, h in cfg.get("params", {}).get("important_heads", []):
        l2h.setdefault(int(L), []).append(int(h))
    if not l2h:
        raise RuntimeError(
            f"config {_CONFIG_FILE} has no important_heads -> output_qk would be True (whole-model), "
            f"which _decide_ring_route_qk cannot size -> the DISK route never fires. Use an "
            f"important-heads config.")
    return l2h


def _matches_req_id(internal_req_id: str, external_req_id: str) -> bool:
    """v1/legacy req_id match (same as workers/_common.py::iter_matching_req_ids). vLLM serve rewrites
    the external id to internal `{external}-{random8}`; older/offline paths keep it."""
    return internal_req_id == external_req_id or internal_req_id.startswith(f"{external_req_id}-")


def _flatten_eager_q(t):
    """Un-pad/un-stack a driver-side "q" tensor to the FLAT per-emitted-row layout the ring's q_cat
    produces (identical to qk_ring_parity.py). last_token: `(num_steps, q_dim)` already flat.
    all_tokens: `(num_steps, max_len, q_dim)` -> step 0 (full prefill) + row 0 of each later step."""
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.dim() == 2:
        return t
    if t.dim() != 3:
        return None
    num_steps = t.shape[0]
    if num_steps == 0:
        return None
    if num_steps == 1:
        return t[0]
    parts = [t[0]] + [t[i, 0:1] for i in range(1, num_steps)]
    return torch.cat(parts, dim=0)


def _split_eager_kall(k_all, q_raw):
    """Split the driver-side padded growing-prefix "k_all" tensor into the ring's per-step LIST form
    (identical to qk_ring_parity.py). Step i real length = prompt_len + i (unchunked prefill +
    one-new-token-per-decode-step); prompt_len read off the padded "q" shape (all_tokens:
    q_raw.shape[1]). A single-step capture (num_steps==1) needs no derivation."""
    if k_all is None or not isinstance(k_all, torch.Tensor):
        return None
    if k_all.dim() == 2:
        return [k_all]
    if k_all.dim() != 3:
        return None
    num_steps = k_all.shape[0]
    if num_steps == 0:
        return None
    if num_steps == 1:
        return [k_all[0]]
    if not isinstance(q_raw, torch.Tensor) or q_raw.dim() != 3:
        raise RuntimeError(
            "cannot reconstruct per-step k_all lengths without an all_tokens padded 'q' tensor "
            "(need prompt_len = q.shape[1]); multi-step last_token legs are not supported here.")
    prompt_len = int(q_raw.shape[1])
    return [k_all[i, :prompt_len + i, :] for i in range(num_steps)]


def _eager_layer_store(out):
    """{layer_num(0-based) -> {"q": flat cpu f32, "k_all": [per-step cpu f32]}} from a SOLO generate()
    output's probes qk_cache (eager PADDED tensors, un-padded to the ring's flat/list form)."""
    probes = getattr(out[0], "probes", None)
    qk = (probes or {}).get("qk_cache", {})
    store = {}
    for _mod, entry in qk.items():
        if not isinstance(entry, dict):
            continue
        layer_num = entry.get("layer_num")
        if layer_num is None:
            continue
        q_raw = entry.get("q")
        k_all_raw = entry.get("k_all")
        q_flat = _flatten_eager_q(q_raw)
        k_list = _split_eager_kall(k_all_raw, q_raw)
        if q_flat is None or k_list is None:
            continue
        store[int(layer_num)] = {
            "q": q_flat.detach().to(torch.float32).cpu(),
            "k_all": [t.detach().to(torch.float32).cpu() for t in k_list],
        }
    return store


def _rpc_probes_layer_store(probes):
    """{layer_num(0-based) -> {"q": cpu f32, "k_all": [cpu f32]}} from the RPC-delivered output.probes
    qk_cache. The per-request ring RPC ships the ring's flat-q / growing-prefix-LIST k_all form already
    (assemble_qk -> _marshal_perreq_qk), keyed by layer with layer_num inside -> no normalization."""
    qk = (probes or {}).get("qk_cache", {})
    store = {}
    for _k, entry in qk.items():
        if not isinstance(entry, dict):
            continue
        layer_num = entry.get("layer_num")
        q = entry.get("q")
        k_all = entry.get("k_all")
        if layer_num is None or not isinstance(q, torch.Tensor) or not isinstance(k_all, list):
            continue
        store[int(layer_num)] = {
            "q": q.detach().to(torch.float32).cpu(),
            "k_all": [t.detach().to(torch.float32).cpu() for t in k_all],
        }
    return store


def _disk_delivered_layer_store(dest_dir, req_id):
    """{layer_num(0-based) -> {"q": cpu f32, "k_all": [cpu f32]}} for ONE request from the DISK-route
    delivered run_dir (offload copytree'd the per-request QK staging to hook_dir/run_id). Reconstructed
    with the same ring reader the shared-file QK path uses (load_multilayer_qk_ring_artifact).

    WEDGED-DELIVERY GUARD: if the offload never landed the run_dir (dest missing, its
    `qk_ring_meta.jsonl` sidecar absent/empty, or the reader raises on a partial/corrupt copy), return
    an EMPTY layer store + a loud marker instead of raising -> compare()'s layer-set gate FAILs the
    VERDICT loudly (never a stack trace with NO verdict)."""
    from vllm_hook_plugins.graph.ring_reader import load_multilayer_qk_ring_artifact
    sidecar = os.path.join(dest_dir, "qk_ring_meta.jsonl")
    if not os.path.isdir(dest_dir) or not os.path.isfile(sidecar) or os.path.getsize(sidecar) == 0:
        print(f"[qk-serve-per-request:graph] DISK DELIVERY MISSING req_id={req_id!r} dest={dest_dir!r} "
              f"(dir/sidecar absent or empty) -> empty layer store (VERDICT will FAIL loudly)",
              flush=True)
        return {}
    try:
        by_req = load_multilayer_qk_ring_artifact(dest_dir)  # {internal_req_id: {layer: {q, k_all}}}
    except Exception as ex:  # noqa: BLE001 -- a partial/corrupt delivered dir must FAIL, not crash
        print(f"[qk-serve-per-request:graph] DISK DELIVERY UNREADABLE req_id={req_id!r} "
              f"dest={dest_dir!r}: {ex!r} -> empty layer store (VERDICT will FAIL loudly)", flush=True)
        return {}
    matched = None
    for internal in by_req:
        if _matches_req_id(str(internal), req_id):
            matched = internal
            break
    if matched is None:
        return {}
    store = {}
    for layer, d in by_req[matched].items():
        q = d.get("q")
        k_all = d.get("k_all")
        if q is None or k_all is None:
            continue
        store[int(layer)] = {
            "q": q.detach().to(torch.float32).cpu(),
            "k_all": [t.detach().to(torch.float32).cpu() for t in k_all],
        }
    return store


# ---------------------------------------------------------------------------
# Prompt construction (identical approach to serve_per_request.py): natural text -> token ids so the
# request-start router sees a prompt length and can size-route; the SAME ids feed both legs -> a
# byte-identical token sequence -> deterministic greedy -> apples-to-apples. Each prompt gets a UNIQUE
# leading marker so no two concurrent requests share a prefix.
# ---------------------------------------------------------------------------
_BASE_LONG = (" The history of computing spans many decades of steady incremental progress across "
              "hardware architecture, compilers, operating systems, distributed protocols, and the "
              "mathematics of numerical methods that underpin modern large scale machine learning "
              "systems deployed to serve millions of concurrent interactive requests every day. ")


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True, cache_dir="./cache/")


def _encode_to_len(tok, text, target_len):
    ids = tok.encode(text)
    while len(ids) < target_len:
        ids = ids + tok.encode(_BASE_LONG)
    return ids[:target_len]


def _build_prompt_specs(tok):
    specs = []
    for i in range(_N_RPC):
        text = f"Request {i}: The capital of France is"
        specs.append({"name": f"rpc{i}", "token_ids": _encode_to_len(tok, text, _LEN_RPC),
                      "group": "rpc"})
    for i in range(_N_DISK):
        text = f"Request D{i}:" + _BASE_LONG
        specs.append({"name": f"disk{i}", "token_ids": _encode_to_len(tok, text, _LEN_DISK),
                      "group": "disk"})
    return specs


def _sampling(max_tokens, l2h):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, ignore_eos=True,
        extra_args={"output_qk": l2h, "hookq_mode": "all_tokens", "hooks_on": _HOOKS_ON},
    )


# ===========================================================================
# GRAPH leg: concurrent AsyncLLM serve driver
# ===========================================================================
def _boot_async_engine():
    os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
    os.environ["VLLM_HOOK_RING_PER_REQUEST"] = "1"
    os.environ.setdefault("VLLM_HOOK_QK_CAPTURE", "buffer")
    os.environ["VLLM_HOOK_WORKER"] = "qk"
    # Keep the request's save_to_disk ABSENT so _resolve_sink stays 'rpc' and the RING route arms.
    os.environ.setdefault("VLLM_HOOK_STORAGE_ROUTER", "0")

    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    engine_args = AsyncEngineArgs(
        model=_MODEL,
        worker_extension_cls=_WORKER_EXT_QK,
        download_dir="./cache/",
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=False,
        # QK ring path DEFERS prefix-cache (reader raises on first-step num_computed>0). Unique
        # per-request markers already prevent prefix sharing, but disable it to be strictly safe.
        enable_prefix_caching=False,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
        compilation_config={"cudagraph_mode": cudagraph_mode},
    )
    print(f"[qk-serve-per-request:graph] booting AsyncLLM model={_MODEL} FULL={cudagraph_mode} "
          f"per_request=1 T_RPC={os.environ.get('VLLM_HOOK_ROUTER_T_RPC','<default>')}", flush=True)
    return AsyncLLM.from_engine_args(engine_args)


async def _ring_residency(engine):
    """(host_live, disk) from the NON-destructive ring_residency RPC (rank-0). Raises if the QK ring
    per-request path is not installed."""
    res = await engine.collective_rpc("ring_residency")
    for r in res:
        if r is not None:
            return (int(r[0]), int(r[1]))
    raise RuntimeError("ring_residency returned None on every rank -- the per-request QK capture-ring "
                       "path is not installed (check ALLOW_CUDAGRAPH / QK_CAPTURE / RING_PER_REQUEST).")


async def _drive_one(engine, spec, hook_dir, l2h, inflight, completions):
    """Fire ONE serve request end-to-end; record its delivered artifact + route + mid-serving evidence.
    RPC route -> output.probes["qk_cache"]; DISK route -> the delivered run_dir at hook_dir/run_id."""
    from vllm.inputs import TokensPrompt
    name = spec["name"]
    req_id = f"serve-{name}"
    run_id = f"deliver-{name}"
    sp = _sampling(_MAX_TOKENS, l2h)
    sp.extra_args = dict(sp.extra_args)
    sp.extra_args["run_id"] = run_id
    sp.extra_args["hook_dir"] = hook_dir
    final = None
    async for out in engine.generate(TokensPrompt(prompt_token_ids=list(spec["token_ids"])),
                                     sp, req_id):
        if out.finished:
            final = out
    t_done = time.monotonic()
    others = len(inflight - {name})
    inflight.discard(name)
    token_ids = list(final.outputs[0].token_ids) if final is not None else []

    probes = getattr(final, "probes", None) if final is not None else None
    if probes and probes.get("qk_cache"):
        route = "rpc"
        layers = _rpc_probes_layer_store(probes)
    else:
        route = "disk"
        dest = os.path.join(hook_dir, run_id)
        layers = _disk_delivered_layer_store(dest, req_id)
    completions.append({"name": name, "route": route, "others_inflight": others, "t_done": t_done,
                        "layers": layers, "token_ids": token_ids, "group": spec["group"]})
    print(f"[qk-serve-per-request:graph] delivered name={name} route={route} "
          f"others_inflight_at_completion={others} layers={len(layers)} gen={len(token_ids)}",
          flush=True)


async def _abort_probe(engine, tok, hook_dir, l2h):
    """Launch one RPC-bound (short prompt + small max_tokens -> predicted artifact < T_RPC -> host
    index) and one DISK-bound (long prompt -> predicted artifact > T_RPC -> disk staging) request, wait
    until BOTH routes have live ring state, CANCEL both mid-generation, then assert residency returns to
    0 -- proving _patched_generate's abort `finally` -> clear_ring_request frees BOTH the host-index
    entry AND the disk staging. Routing one RPC + one DISK makes the host>=1 AND disk>=1 wait satisfiable
    within a couple poll cycles and forces BOTH abort cleanup paths."""
    from vllm.inputs import TokensPrompt
    specs = [
        {"name": "abort_rpc", "ids": _encode_to_len(tok, "Abort A: The capital of France is", _LEN_RPC),
         "max_tokens": _ABORT_RPC_MAX_TOKENS, "want": "host"},
        {"name": "abort_disk", "ids": _encode_to_len(tok, "Abort B:" + _BASE_LONG, _LEN_DISK),
         "max_tokens": _ABORT_DISK_MAX_TOKENS, "want": "disk"},
    ]

    async def _run(spec):
        sp = _sampling(spec["max_tokens"], l2h)
        sp.extra_args = dict(sp.extra_args)
        sp.extra_args["run_id"] = f"abort-{spec['name']}"
        sp.extra_args["hook_dir"] = hook_dir
        req_id = f"serve-{spec['name']}"
        async for _out in engine.generate(TokensPrompt(prompt_token_ids=list(spec["ids"])),
                                          sp, req_id):
            pass  # never reached to completion -- cancelled below

    tasks = [asyncio.ensure_future(_run(s)) for s in specs]
    during = (0, 0)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        during = await _ring_residency(engine)
        if during[0] >= 1 and during[1] >= 1:
            break
    print(f"[qk-serve-per-request:graph] abort: residency_during=(host={during[0]}, disk={during[1]}) "
          f"(gate: host>=1 AND disk>=1 -> both cleanup paths staged)", flush=True)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    after = await _ring_residency(engine)
    d2 = time.monotonic() + 30.0
    while after != (0, 0) and time.monotonic() < d2:
        await asyncio.sleep(0.05)
        after = await _ring_residency(engine)
    print(f"[qk-serve-per-request:graph] abort: residency_after=(host={after[0]}, disk={after[1]}) "
          f"(gate: ==(0,0))", flush=True)
    return during, after


async def _run_graph(out_path):
    engine = _boot_async_engine()
    tok = _load_tokenizer()
    l2h = _layer_to_heads()
    work = os.path.dirname(os.path.abspath(out_path)) or "."
    hook_dir = os.environ.get("VLLM_HOOK_DELIVER_DIR", os.path.join(work, "delivered"))
    os.makedirs(hook_dir, exist_ok=True)
    result = {}
    try:
        # ---- ABORT phase FIRST (clean slate: residency starts at 0) ----
        during, after = await _abort_probe(engine, tok, hook_dir, l2h)

        # ---- MAIN concurrent batch: mixed RPC + DISK, fired together ----
        specs = _build_prompt_specs(tok)
        inflight = {s["name"] for s in specs}
        completions = []
        await asyncio.gather(*[_drive_one(engine, s, hook_dir, l2h, inflight, completions)
                               for s in specs])

        res_after_main = await _ring_residency(engine)
        print(f"[qk-serve-per-request:graph] residency_after_main=(host={res_after_main[0]}, "
              f"disk={res_after_main[1]}) (gate: ==(0,0))", flush=True)

        for c in completions:
            result[c["name"]] = {"layers": c["layers"], "token_ids": c["token_ids"],
                                 "route": c["route"], "group": c["group"]}
        routes = [c["route"] for c in completions]
        max_others = max((c["others_inflight"] for c in completions), default=0)
        t_done = sorted(c["t_done"] for c in completions)
        spread = (t_done[-1] - t_done[0]) if len(t_done) >= 2 else 0.0
        result[_META_KEY] = {
            "residency_after_main": list(res_after_main),
            "residency_during_abort": list(during),
            "residency_after_abort": list(after),
            "n_rpc": routes.count("rpc"),
            "n_disk": routes.count("disk"),
            "mid_serving_max_others": int(max_others),
            "completion_spread_s": float(spread),
        }
        m = result[_META_KEY]
        print(f"[qk-serve-per-request:graph] ROUTES rpc={m['n_rpc']} disk={m['n_disk']}", flush=True)
        print(f"[qk-serve-per-request:graph] MID-SERVING max_others_inflight_at_completion="
              f"{m['mid_serving_max_others']} completion_spread={spread:.3f}s", flush=True)
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[qk-serve-per-request:graph] wrote {out_path}", flush=True)
    return 0


# ===========================================================================
# EAGER leg: solo ground truth via HookLLM (register_forward_hook path)
# ===========================================================================
def _run_eager(out_path):
    os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
    from vllm.inputs import TokensPrompt
    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    tok = _load_tokenizer()
    specs = _build_prompt_specs(tok)
    _user = os.environ.get("USER") or getpass.getuser()
    print(f"[qk-serve-per-request:eager] booting HookLLM (solo ground truth) config={_CONFIG_FILE}",
          flush=True)
    llm = HookLLM(
        model=_MODEL,
        worker_name="probe_hook_qk",
        analyzer_name="attn_tracker",
        config_file=_CONFIG_FILE,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
    )
    result = {}
    for spec in specs:
        sp = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, ignore_eos=True,
                            extra_args={"hooks_on": _HOOKS_ON})
        out = llm.generate(TokensPrompt(prompt_token_ids=list(spec["token_ids"])), sp,
                           save_to_disk=False)
        store = _eager_layer_store(out)
        token_ids = list(out[0].outputs[0].token_ids)
        result[spec["name"]] = {"layers": store, "token_ids": token_ids}
        print(f"[qk-serve-per-request:eager] name={spec['name']} captured {len(store)} layers "
              f"gen={len(token_ids)}", flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[qk-serve-per-request:eager] wrote {out_path}", flush=True)
    return 0


# ===========================================================================
# COMPARE: byte-identity (both routes, q + k_all list) + all serve gates -> one VERDICT
# ===========================================================================
def compare(graph_path, eager_path, rtol, atol):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    meta = g.get(_META_KEY) or {}

    # --- Gate 1: residency drained to 0 after the main batch.
    rm = tuple(meta.get("residency_after_main", [-1, -1]))
    print(f"[qk-serve-per-request] RESIDENCY after_main={rm} during_abort="
          f"{tuple(meta.get('residency_during_abort', []))} "
          f"after_abort={tuple(meta.get('residency_after_abort', []))} (gate after==(0,0))")
    if rm != (0, 0):
        print("[qk-serve-per-request]   -> FAIL: residency_after_main != (0,0) (a delivered request "
              "leaked host/disk state)")
        overall_ok = False

    # --- Gate 2: ABORT freed -- BOTH routes non-vacuously staged (host>=1 AND disk>=1) then -> 0.
    during = tuple(meta.get("residency_during_abort", [0, 0]))
    after = tuple(meta.get("residency_after_abort", [-1, -1]))
    abort_nonvacuous = (during[0] >= 1 and during[1] >= 1)
    abort_freed = (after == (0, 0))
    print(f"[qk-serve-per-request] ABORT during={during} after={after} "
          f"(nonvacuous={abort_nonvacuous} freed={abort_freed})")
    if not abort_nonvacuous:
        print("[qk-serve-per-request]   -> FAIL: abort probe did not stage BOTH routes (host>=1 AND "
              "disk>=1) -- the cancel did not exercise both clear_ring_request AND clear_request_disk "
              "(check the RPC/DISK abort-leg routing vs VLLM_HOOK_ROUTER_T_RPC)")
        overall_ok = False
    if not abort_freed:
        print("[qk-serve-per-request]   -> FAIL: residency_after_abort != (0,0) (cancelled request "
              "leaked ring state)")
        overall_ok = False

    # --- Gate 3: BOTH routes exercised.
    n_rpc, n_disk = int(meta.get("n_rpc", 0)), int(meta.get("n_disk", 0))
    both_routes = n_rpc >= 1 and n_disk >= 1
    print(f"[qk-serve-per-request] ROUTES rpc={n_rpc} disk={n_disk} (both_exercised={both_routes})")
    if not both_routes:
        print("[qk-serve-per-request]   -> FAIL: not both routes fired (tune VLLM_HOOK_ROUTER_T_RPC / "
              "prompt lengths so short->RPC and long->DISK)")
        overall_ok = False

    # --- Gate 4: MID-SERVING -- at least one request delivered while others were still generating.
    max_others = int(meta.get("mid_serving_max_others", 0))
    mid_serving = max_others >= 1
    print(f"[qk-serve-per-request] MID-SERVING max_others_inflight_at_completion={max_others} "
          f"completion_spread={float(meta.get('completion_spread_s', 0.0)):.3f}s "
          f"(interleaved={mid_serving})")
    if not mid_serving:
        print("[qk-serve-per-request]   -> FAIL: every request completed only after all others (looks "
              "like shutdown-only delivery, not mid-serving)")
        overall_ok = False

    # --- Gate 5: BYTE-IDENTITY per (request, layer) for BOTH routes: q (flat) + k_all (list).
    case_keys = sorted((set(g) & set(e)) - {_META_KEY})
    total = matched = 0
    seen_route = {"rpc": 0, "disk": 0}
    for case in case_keys:
        route = g[case].get("route", "?")
        seen_route[route] = seen_route.get(route, 0) + 1
        g_tok, e_tok = g[case].get("token_ids"), e[case].get("token_ids")
        if g_tok != e_tok:
            print(f"[qk-serve-per-request] case={case} route={route}: TOKEN MISMATCH "
                  f"graph={g_tok} eager={e_tok}")
            overall_ok = False
        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        if set(g_layers) != set(e_layers):
            print(f"[qk-serve-per-request] case={case} route={route}: LAYER-SET MISMATCH "
                  f"graph={sorted(g_layers)} eager={sorted(e_layers)} "
                  f"(missing_in_graph={sorted(set(e_layers) - set(g_layers))}) -> FAIL")
            overall_ok = False
        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[qk-serve-per-request] case={case} route={route}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)}) -> FAIL")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]

            # ---- q: single flat tensor per layer ----
            gq, eq = gv.get("q"), ev.get("q")
            if gq is None or eq is None:
                print(f"[qk-serve-per-request] case={case} route={route} layer={layer} q: MISSING "
                      f"(graph={gq is not None} eager={eq is not None})")
                overall_ok = False
            else:
                total += 1
                if gq.shape != eq.shape:
                    print(f"[qk-serve-per-request] case={case} route={route} layer={layer} q: SHAPE "
                          f"MISMATCH graph={tuple(gq.shape)} eager={tuple(eq.shape)}")
                    overall_ok = False
                else:
                    ok = torch.allclose(gq, eq, rtol=rtol, atol=atol)
                    md = (gq - eq).abs().max().item() if gq.numel() else 0.0
                    matched += int(ok)
                    overall_ok = overall_ok and ok
                    print(f"[qk-serve-per-request] case={case} route={route} layer={layer} q: "
                          f"match={ok} max|Δ|={md:.3e} shape={tuple(gq.shape)}")

            # ---- k_all: LIST of growing-prefix tensors, entry-for-entry ----
            gk, ek = gv.get("k_all"), ev.get("k_all")
            if gk is None or ek is None:
                print(f"[qk-serve-per-request] case={case} route={route} layer={layer} k_all: MISSING "
                      f"(graph={gk is not None} eager={ek is not None})")
                overall_ok = False
                continue
            if len(gk) != len(ek):
                print(f"[qk-serve-per-request] case={case} route={route} layer={layer} k_all: "
                      f"STEP-COUNT MISMATCH graph={len(gk)} eager={len(ek)} (a dropped/extra step)")
                overall_ok = False
                continue
            steps_ok = 0
            for idx, (gkt, ekt) in enumerate(zip(gk, ek)):
                total += 1
                if gkt.shape != ekt.shape:
                    print(f"[qk-serve-per-request] case={case} route={route} layer={layer} "
                          f"k_all[{idx}]: SHAPE MISMATCH graph={tuple(gkt.shape)} "
                          f"eager={tuple(ekt.shape)}")
                    overall_ok = False
                    continue
                ok = torch.allclose(gkt, ekt, rtol=rtol, atol=atol)
                md = (gkt - ekt).abs().max().item() if gkt.numel() else 0.0
                matched += int(ok)
                steps_ok += int(ok)
                overall_ok = overall_ok and ok
            print(f"[qk-serve-per-request] case={case} route={route} layer={layer} k_all: "
                  f"{steps_ok}/{len(gk)} steps within tol")

    print("=" * 64)
    print(f"[qk-serve-per-request] {matched}/{total} tensors within rtol={rtol} atol={atol}; "
          f"delivered rpc={seen_route.get('rpc',0)} disk={seen_route.get('disk',0)}")
    if overall_ok and total > 0:
        print("[qk-serve-per-request] VERDICT: PASS — mid-serving per-request QK delivery matches "
              "eager on BOTH routes, residency drained to 0, abort freed.")
        return 0
    print("[qk-serve-per-request] VERDICT: FAIL — a Task-13C serve gate failed (see the -> FAIL "
          "lines above).")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--mode", choices=["graph", "eager"], required=True)
    pc.add_argument("--out", required=True)
    pk = sub.add_parser("compare")
    pk.add_argument("--graph", required=True)
    pk.add_argument("--eager", required=True)
    pk.add_argument("--rtol", type=float, default=_SERVE_RTOL)
    pk.add_argument("--atol", type=float, default=_SERVE_ATOL)
    args = p.parse_args()
    if args.cmd == "capture":
        if args.mode == "graph":
            sys.exit(asyncio.run(_run_graph(args.out)))
        sys.exit(_run_eager(args.out))
    else:
        sys.exit(compare(args.graph, args.eager, args.rtol, args.atol))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
    main()
