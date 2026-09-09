"""Subprocess-isolated graph-vs-eager QK parity oracle for the capture-RING path
(branch `capture_ring`, plan Task 15).

QK ring analogue of tests/cuda_graph/tests/ring/hs_ring_parity.py (mirror its structure) and the
existing eager-reference harness tests/cuda_graph/tests/qk_graph/qk_parity.py (reuse its q/k_all
normalization idea). Captures the same prompts twice — once under FULL cudagraph + the QK
capture-ring buffer path (scatter -> shared GPU ring -> off-loop drain -> durable per-layer q/k
raw files, see graph/install.py::install_qk_hosts / _build_routing), once under the legacy eager
register_forward_hook path — in SEPARATE processes (exclusive-process GPU), then asserts the
per-(request, layer) q AND growing-prefix k_all match within tolerance.

Key structural differences from qk_parity.py:

1. The ring path never populates the RPC/bank buckets (probe_hookqk_worker.py's graph_install
   docstring: "get_captured_states / marshal_finished / deliver_finished stay inert here"), so
   `out[0].probes["qk_cache"]` is EMPTY under graph mode. The graph leg instead calls the worker's
   `flush_ring()` collective_rpc (by STRING METHOD NAME, mirrors hs_ring_parity.py / the cb_oom
   `_rpc` convention -- no VLLM_ALLOW_INSECURE_SERIALIZATION needed) once, after all cases have
   generated, then reconstructs every request's per-layer q/k_all via
   graph.ring_reader.load_multilayer_qk_ring_artifact(run_dir).

2. Layer-key alignment: QKStepEntry.layer / the ring reader's output key is 0-based (==
   workers/_common.py::match_attn's raw regex-captured layer index -- graph/install.py:
   "layers = [(layer_num, host.q_buf, host.k_buf), ...]; layer_num is 0-based (== eager
   match_attn)"). The eager `probe_hookqk_worker.py` hook passes the SAME 0-based number as
   `layer_num` (`layer_num = match_attn(module_name)`, no +1 -- unlike HS's 1-based
   `layer_num+1`). Both normalize functions below key their output on this 0-based int, so
   compare() diffs matching layers regardless of internal module-name keying.

3. `k_all` SHAPE mismatch between the two paths -- this is the harness's main adaptation vs
   qk_parity.py, and worth spelling out precisely:
     - EAGER (out[0].probes["qk_cache"][layer]["k_all"]) is a PADDED TENSOR
       `(num_steps, max_len, k_dim)`, built by the worker's `_k_all_cpu_list` (per-hook-fire
       growing-prefix reconstruction: `[full[:L] for L in k_prefix_ends]`) then
       `pad_sequence(...)` in `get_captured_states`, and -- if the trajectory-default COMPACT_KALL
       wire format fired -- rebuilt identically by `_hook_plugin.py::_reconstruct_compact_kall`
       (`pad_sequence([full[:L] for L in ends])`). Either way the DRIVER discards the per-step
       real lengths (`k_prefix_ends`) once the padded tensor is built, so this harness must
       re-derive them (see `_split_eager_kall`).
     - RING (graph.ring_reader.load_multilayer_qk_ring_artifact) never pads: `k_all` is a LIST
       of tensors, one per step that recorded a `prefix_end` (every step under all_tokens; only
       the final prefill chunk / each decode step under last_token -- QKStepEntry's emit_q gate),
       each exactly its real growing-prefix length. No reconstruction needed on this side.
   `_split_eager_kall` turns the eager padded tensor back into the ring's per-step list form so
   compare() diffs list-for-list, index-for-index (both sides preserve step/`k_start` order).
   The per-step real length is `prompt_len + i` (steps: step 0 = the unchunked prefill, i.e. every
   captured token; each later step is exactly one new decode token) -- the SAME unchunked-prefill,
   one-new-token-per-decode-step assumption hs_ring_parity.py's `_flatten_eager_hs` documents.
   `prompt_len` is derived from the padded "q" tensor's own shape (all_tokens mode: step 0's Q is
   the full prompt, every decode step's Q is exactly 1 new token, so `max_len == prompt_len`) --
   no separate tokenizer call, no assumption beyond what capture() already guarantees. The
   single-step case (num_steps==1, e.g. leg A: hooks_on=prefill) needs no derivation at all: a
   1-element pad_sequence list adds NO padding, so the whole row is already the real prefix.

4. "q" uses the SAME flattening idea as `_flatten_eager_hs` (last_token: already flat, one row per
   emit; all_tokens: step 0 is the full prompt, every later step contributes exactly its ONE new
   real row) -- because the ring's "q" is likewise a FLAT concatenation of only the emit_q steps'
   real rows (graph/install.py::_build_routing: "q on emit_q ... the op scatters q into all the
   reserved slots regardless, and a non-emit step's q rows are simply never referenced").

5. Prefix-cache is DEFERRED on the QK ring path (v1): `load_multilayer_qk_ring_artifact` raises
   NotImplementedError if a request's first captured step has `num_computed>0`. Both legs below
   use FRESH prefills only (`enable_prefix_caching=False` on both the graph AND eager engines, so
   the comparison is apples-to-apples), matching the task report's scoped-in cases.

    python qk_ring_parity.py capture --mode graph --out g.pkl
    python qk_ring_parity.py capture --mode eager --out e.pkl
    python qk_ring_parity.py compare --graph g.pkl --eager e.pkl
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import getpass
import pickle
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16

# Decode-stage parity knobs (same contract as qk_parity.py / hs_ring_parity.py):
#   VLLM_HOOK_PARITY_MAX_TOKENS > 1  -> generate that many tokens so the ring routing must
#       fire on every decode step, not just prefill.
#   VLLM_HOOK_PARITY_HOOKS_ON=both   -> request decode+prefill capture (default "prefill").
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PARITY_MAX_TOKENS", "1"))
_HOOKS_ON = os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "").strip()

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _matches_req_id(internal_req_id: str, external_req_id: str) -> bool:
    """Same v1/legacy req_id match rule as workers/_common.py::iter_matching_req_ids (vLLM
    v0.12+ uses the SAME id internally; older versions append a random suffix)."""
    return internal_req_id == external_req_id or internal_req_id.startswith(f"{external_req_id}-")


def _flatten_eager_q(t):
    """Un-pad/un-stack a driver-side "q" tensor to the FLAT per-emitted-row layout the ring's
    q_cat produces (see module docstring point 4).

    - last_token mode: `t` is `(num_steps, q_dim)` (torch.stack of single rows, one per emit_q
      fire) -> already flat, return as-is.
    - all_tokens mode: `t` is `(num_steps, max_len, q_dim)` from pad_sequence. Step 0 is always
      the (unchunked) PREFILL -- the longest span, since every DECODE step in this harness
      captures exactly one token -- so `max_len == step 0's real length` and step 0 needs no
      trimming; every later step's only REAL row is row 0 (the rest is pad_sequence's zero
      padding). Concatenating step 0 (full) with row 0 of every later step reproduces the ring's
      flat per-token concatenation exactly.
    """
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
    """Split the driver-side padded growing-prefix "k_all" tensor into the ring's per-step LIST
    form (see module docstring point 3): `[k_all_step0, k_all_step1, ...]`, each trimmed to its
    real (unpadded) length.

    `k_all` is `(num_steps, max_len, k_dim)` (pad_sequence over `[full[:L] for L in
    k_prefix_ends]` -- see probe_hookqk_worker.py::_k_all_cpu_list / append_k_prefix). Step i's
    real length is `prompt_len + i` for an UNCHUNKED prefill + one-new-token-per-decode-step
    trajectory (this harness's contract, same as hs_ring_parity.py's `_flatten_eager_hs`
    docstring). `prompt_len` is NOT directly recoverable from `k_all` alone (step 0 -- the
    prefill -- is the SHORTEST row here, since k_all only grows over time, the OPPOSITE of q/HS's
    padding direction) so it is read off the padded "q" tensor's own shape instead (all_tokens
    mode: `q_raw.shape[1] == prompt_len`, see `_flatten_eager_q`'s docstring).

    A single-step capture (`num_steps == 1`, e.g. leg A hooks_on=prefill) needs no derivation at
    all: a 1-element pad_sequence list adds NO padding, so the whole row is already real.
    """
    if k_all is None or not isinstance(k_all, torch.Tensor):
        return None
    if k_all.dim() == 2:            # defensive: pad_sequence always adds a leading dim in
        return [k_all]              # practice, but handle a bare single row just in case.
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
            "(need prompt_len = q.shape[1]); multi-step last_token legs are not supported by "
            "this harness's normalization (not exercised by run_qk_ring_parity.sh's two legs).")
    prompt_len = int(q_raw.shape[1])
    return [k_all[i, :prompt_len + i, :] for i in range(num_steps)]


def _eager_layer_store(out):
    """{layer_num:int(0-based) -> {"q": flat cpu f32 tensor, "k_all": [per-step cpu f32
    tensors]}} from a generate() output's probes, keyed on the entry's `layer_num` (0-based,
    == eager match_attn / the ring's layer key) so it lines up in compare()."""
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


def _flush_ring(llm):
    """collective_rpc("flush_ring") by STRING METHOD NAME on the worker
    (ProbeHookQKWorker.flush_ring, workers/probe_hookqk_worker.py) -- ships no plain-function
    payload, so VLLM_ALLOW_INSECURE_SERIALIZATION is NOT needed (matches
    tests/cuda_graph/tests/ring/hs_ring_parity.py's `_flush_ring` / the cb_oom `_rpc`
    convention). Final drain + write the shared QK sidecar; returns the per-worker run_dir (None
    if the ring path was not installed). TP=1 in scope -> a single rank-0 result."""
    rows = None
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc("flush_ring")
        except Exception as e:  # noqa: BLE001
            print(f"[qk-ring-parity] collective_rpc('flush_ring') via {type(h).__name__} "
                  f"failed: {e}", flush=True)
            rows = None
            continue
        if rows:
            break
    if not rows:
        return None
    return rows[0]


def _ring_layer_store(run_dir, req_id):
    """{layer_num:int(0-based) -> {"q": cpu f32 tensor, "k_all": [cpu f32 tensors]}} for ONE
    request, reconstructed from the durable QK ring dump
    (graph.ring_reader.load_multilayer_qk_ring_artifact)."""
    from vllm_hook_plugins.graph.ring_reader import load_multilayer_qk_ring_artifact
    data = load_multilayer_qk_ring_artifact(run_dir)
    matched_key = None
    for internal_id in data:
        if _matches_req_id(internal_id, req_id):
            matched_key = internal_id
            break
    if matched_key is None:
        return {}
    per_layer = data[matched_key]
    store = {}
    for layer, d in per_layer.items():
        store[int(layer)] = {
            "q": d["q"].detach().to(torch.float32).cpu(),
            "k_all": [t.detach().to(torch.float32).cpu() for t in d["k_all"]],
        }
    return store


def capture(mode, out_path):
    ring_dir = None
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
        ring_dir = os.environ.get("VLLM_HOOK_RING_DIR")
        if not ring_dir:
            stem = os.path.splitext(os.path.basename(out_path))[0]
            ring_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)) or ".",
                                     f"ring_{stem}")
            os.environ["VLLM_HOOK_RING_DIR"] = ring_dir
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    # Ring capture is the FULL-cudagraph mechanism (compile ON); default to FULL rather than
    # qk_parity.py's PIECEWISE default.
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/attention_tracker/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[qk-ring-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file}"
          + (f" ring_dir={ring_dir}" if ring_dir else ""), flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="probe_hook_qk",
        analyzer_name="attn_tracker",
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        # Prefix-cache reconstruction is DEFERRED on the QK ring path (v1, module docstring
        # point 5): the reader raises NotImplementedError if a request's first captured step
        # has num_computed>0. Fresh prefill only on BOTH engines, so this is apples-to-apples.
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
        **extra,
    )

    # Greedy: the prefill captures the prompt's post-RoPE q/k, which both paths must agree on.
    # With _MAX_TOKENS > 1 + hooks_on=both the ring/hook must ALSO fire on every decode step, so
    # the captured q/k_all spans prefill + decode tokens and the parity check covers decode.
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        extra_args=({"hooks_on": _HOOKS_ON} if _HOOKS_ON else None),
    )
    print(f"[qk-ring-parity:{mode}] sampling max_tokens={_MAX_TOKENS} "
          f"hooks_on={_HOOKS_ON or 'default(prefill)'}", flush=True)

    result = {}
    req_ids_by_case = {}
    for case in _CASES:
        out = llm.generate(case["text"], sp, save_to_disk=False)
        req_ids_by_case[case["name"]] = str(out[0].request_id)
        token_ids = list(out[0].outputs[0].token_ids)
        if mode == "eager":
            store = _eager_layer_store(out)
            result[case["name"]] = {"layers": store, "token_ids": token_ids}
            print(f"[qk-ring-parity:{mode}] case={case['name']} captured {len(store)} layers "
                  f"(eager probes)", flush=True)
        else:
            # Ring path: probes are inert (bank never built) -> deferred to the post-loop
            # flush_ring + load_multilayer_qk_ring_artifact reconstruction (module docstring
            # point 1).
            result[case["name"]] = {"layers": None, "token_ids": token_ids}
            print(f"[qk-ring-parity:{mode}] case={case['name']} generated {len(token_ids)} "
                  f"tokens (ring reconstruction deferred to flush)", flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass

    if mode == "graph":
        run_dir = _flush_ring(llm)
        if not run_dir:
            raise RuntimeError(
                "flush_ring collective_rpc returned no run_dir -- the QK capture-ring path is "
                "not installed on the worker (check VLLM_HOOK_ALLOW_CUDAGRAPH / "
                "VLLM_HOOK_QK_CAPTURE).")
        print(f"[qk-ring-parity:{mode}] flush_ring -> run_dir={run_dir}", flush=True)
        for case in _CASES:
            store = _ring_layer_store(run_dir, req_ids_by_case[case["name"]])
            result[case["name"]]["layers"] = store
            print(f"[qk-ring-parity:{mode}] case={case['name']} reconstructed {len(store)} "
                  f"layers from ring dump (req_id={req_ids_by_case[case['name']]})", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[qk-ring-parity:{mode}] wrote {out_path}", flush=True)
    return 0


def compare(graph_path, eager_path, rtol, atol):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    total = matched = 0
    for case in sorted(set(g) & set(e)):
        g_tok, e_tok = g[case].get("token_ids"), e[case].get("token_ids")
        if g_tok != e_tok:
            print(f"[qk-ring-parity] case={case}: TOKEN MISMATCH graph={g_tok} eager={e_tok} "
                  f"(q/k_all compare below is not apples-to-apples if these diverge)")
            overall_ok = False

        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[qk-ring-parity] case={case}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]

            # ---- q: single flat tensor per layer ----
            gq, eq = gv.get("q"), ev.get("q")
            if gq is None or eq is None:
                print(f"[qk-ring-parity] case={case} layer={layer} q: MISSING "
                      f"(graph={gq is not None} eager={eq is not None})")
                overall_ok = False
            else:
                total += 1
                if gq.shape != eq.shape:
                    print(f"[qk-ring-parity] case={case} layer={layer} q: SHAPE MISMATCH "
                          f"graph={tuple(gq.shape)} eager={tuple(eq.shape)}")
                    overall_ok = False
                else:
                    ok = torch.allclose(gq, eq, rtol=rtol, atol=atol)
                    md = (gq - eq).abs().max().item() if gq.numel() else 0.0
                    mean = (gq - eq).abs().mean().item() if gq.numel() else 0.0
                    matched += int(ok)
                    overall_ok = overall_ok and ok
                    print(f"[qk-ring-parity] case={case} layer={layer} q: match={ok} "
                          f"max|Δ|={md:.3e} mean|Δ|={mean:.3e} shape={tuple(gq.shape)}")

            # ---- k_all: a LIST of growing-prefix tensors, compared entry-for-entry ----
            gk, ek = gv.get("k_all"), ev.get("k_all")
            if gk is None or ek is None:
                print(f"[qk-ring-parity] case={case} layer={layer} k_all: MISSING "
                      f"(graph={gk is not None} eager={ek is not None})")
                overall_ok = False
                continue
            if len(gk) != len(ek):
                print(f"[qk-ring-parity] case={case} layer={layer} k_all: STEP-COUNT "
                      f"MISMATCH graph={len(gk)} eager={len(ek)} (a dropped/extra step)")
                overall_ok = False
                continue
            steps_ok = 0
            for idx, (gkt, ekt) in enumerate(zip(gk, ek)):
                total += 1
                if gkt.shape != ekt.shape:
                    print(f"[qk-ring-parity] case={case} layer={layer} k_all[{idx}]: SHAPE "
                          f"MISMATCH graph={tuple(gkt.shape)} eager={tuple(ekt.shape)}")
                    overall_ok = False
                    continue
                ok = torch.allclose(gkt, ekt, rtol=rtol, atol=atol)
                md = (gkt - ekt).abs().max().item() if gkt.numel() else 0.0
                matched += int(ok)
                steps_ok += int(ok)
                overall_ok = overall_ok and ok
                print(f"[qk-ring-parity] case={case} layer={layer} k_all[{idx}]: match={ok} "
                      f"max|Δ|={md:.3e} shape={tuple(gkt.shape)}")
            print(f"[qk-ring-parity] case={case} layer={layer} k_all: {steps_ok}/{len(gk)} "
                  f"steps within tol")

    print("=" * 60)
    print(f"[qk-ring-parity] {matched}/{total} tensors within rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print("[qk-ring-parity] VERDICT: PASS — ring QK capture matches eager.")
        return 0
    print("[qk-ring-parity] VERDICT: FAIL — ring QK capture diverged from eager.")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--mode", choices=["graph", "eager"], required=True)
    pc.add_argument("--out", required=True)
    pk = sub.add_parser("compare")
    pk.add_argument("--graph", required=True)
    pk.add_argument("--eager", required=True)
    pk.add_argument("--rtol", type=float, default=1e-2)
    pk.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out))
    else:
        sys.exit(compare(args.graph, args.eager, args.rtol, args.atol))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
    main()
