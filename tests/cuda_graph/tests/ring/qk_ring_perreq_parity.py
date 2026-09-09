"""Subprocess-isolated graph-vs-eager QK parity oracle for the PER-REQUEST ring-DEMUX delivery path
(branch `capture_ring`, plan Task 13 phase B). This is the byte-identity GATE for the QK per-request
ring demux (Task 13A wiring, commit 6a3a1e1) — the QK analogue of
tests/cuda_graph/tests/ring/hs_ring_perreq_parity.py (job 575168 PASS).

It fuses two proven oracles:
  * per-request STRUCTURE from hs_ring_perreq_parity.py — the graph leg captures BOTH prompts in ONE
    batched generate([p1, p2], ...) (a real interleaved batch the off-loop consumer must DEMUX by
    req_id into a PerRequestIndex, no shared per-layer file), then reads back via the QK worker's
    flush_ring_per_request() collective_rpc, which stop()s the drain (finalize_all delivers last-step
    stragglers), pops each request's assembled per-layer q/k_all via pop_deliverable_qk/assemble_qk,
    frees it, and returns (deliverables, residency_after). residency_after MUST be 0 (the residency
    gate: every captured request delivered + freed, nothing leaked in the index).
  * q/k_all NORMALIZATION + compare from qk_ring_parity.py — the graph (ring) leg's per-request
    deliverable is ALREADY in the ring's flat-q / growing-prefix-k_all LIST form (assemble_qk =
    {layer: {"q": cat(emit q rows), "k_all": [k_full[:L] for L in prefix_ends]}}); the eager leg's
    out[0].probes["qk_cache"] is a PADDED tensor that must be un-padded back to that form
    (_flatten_eager_q for q, _split_eager_kall for k_all), so compare() diffs q flat-for-flat and
    k_all list-for-list, index-for-index.

Why solo-eager (not batched): HookLLM.generate's convenience probe-merge onto outputs[0] is LOSSY for
a multi-request batch, so a batched eager read would corrupt request 0's ground truth.
Per-request QK is batch-invariant under causal attention + fresh prefill (the M5 property — no
cross-request value bleed), so a SOLO-EAGER reference is exactly the ground truth for the batched-graph
demux. req_ids are matched apples-to-apples via _matches_req_id (external vs internal {ext}-{rand}).

Layer-key alignment (unchanged from qk_ring_parity.py): QKStepEntry.layer / the ring reader / the
per-request assembly key each request's per-layer tensor on the 0-based layer index (== eager
match_attn), NOT HS's 1-based number. compare() diffs matching 0-based layers directly.

Prefix-cache is DEFERRED on the QK ring path (v1): assembly/reader raise NotImplementedError if a
request's first captured step has num_computed>0. Both legs use FRESH prefills only
(enable_prefix_caching=False on BOTH engines), matching qk_ring_parity.py.

    python qk_ring_perreq_parity.py capture --mode graph --out g.pkl
    python qk_ring_perreq_parity.py capture --mode eager --out e.pkl
    python qk_ring_perreq_parity.py compare --graph g.pkl --eager e.pkl
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import getpass
import pickle
import sys

import torch
import zstandard as zstd

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16

# Decode-stage parity knobs (same contract as qk_ring_parity.py / hs_ring_perreq_parity.py):
#   VLLM_HOOK_PARITY_MAX_TOKENS > 1  -> generate that many tokens so the ring routing must fire on
#       every decode step, not just prefill.
#   VLLM_HOOK_PARITY_HOOKS_ON=both   -> request decode+prefill capture (default "prefill").
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PARITY_MAX_TOKENS", "1"))
_HOOKS_ON = os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "").strip()

# Reserved (non-case) key carrying the graph leg's residency-gate result into compare().
_RESIDENCY_KEY = "__residency_after__"

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _matches_req_id(internal_req_id: str, external_req_id: str) -> bool:
    """Same v1/legacy req_id match rule as workers/_common.py::iter_matching_req_ids (vLLM serve
    rewrites the external id to internal `{external}-{random8}`; older/offline paths keep it)."""
    return internal_req_id == external_req_id or internal_req_id.startswith(f"{external_req_id}-")


def _flatten_eager_q(t):
    """Un-pad/un-stack a driver-side "q" tensor to the FLAT per-emitted-row layout the ring's q_cat
    produces (see qk_ring_parity.py point 4).

    - last_token mode: `t` is `(num_steps, q_dim)` (torch.stack of single rows, one per emit_q fire)
      -> already flat, return as-is.
    - all_tokens mode: `t` is `(num_steps, max_len, q_dim)` from pad_sequence. Step 0 is the
      (unchunked) PREFILL — the longest span, since every DECODE step here captures exactly one token
      — so `max_len == step 0's real length` and step 0 needs no trimming; every later step's only
      REAL row is row 0. Concatenating step 0 (full) with row 0 of every later step reproduces the
      ring's flat per-token concatenation exactly.
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
    """Split the driver-side padded growing-prefix "k_all" tensor into the ring's per-step LIST form
    `[k_all_step0, k_all_step1, ...]`, each trimmed to its real (unpadded) length (see qk_ring_parity.py
    point 3). `k_all` is `(num_steps, max_len, k_dim)` (pad_sequence over `[full[:L] for L in
    k_prefix_ends]`). Step i's real length is `prompt_len + i` for an UNCHUNKED prefill + one-new-token-
    per-decode-step trajectory; `prompt_len` is read off the padded "q" tensor's own shape (all_tokens:
    `q_raw.shape[1] == prompt_len`) because k_all's step 0 is the SHORTEST row (k_all only grows). A
    single-step capture (num_steps==1) needs no derivation — a 1-element pad_sequence adds no padding."""
    if k_all is None or not isinstance(k_all, torch.Tensor):
        return None
    if k_all.dim() == 2:            # defensive: pad_sequence always adds a leading dim in practice,
        return [k_all]              # but handle a bare single row just in case.
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
            "(need prompt_len = q.shape[1]); multi-step last_token legs are not supported by this "
            "harness's normalization (not exercised by run_qk_ring_perreq_parity.sh's two legs).")
    prompt_len = int(q_raw.shape[1])
    return [k_all[i, :prompt_len + i, :] for i in range(num_steps)]


def _eager_layer_store(out):
    """{layer_num:int(0-based) -> {"q": flat cpu f32 tensor, "k_all": [per-step cpu f32 tensors]}}
    from a SOLO generate() output's probes, keyed on the entry's 0-based `layer_num` (== the ring's
    per-request layer key)."""
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


def _flush_ring_per_request(llm):
    """collective_rpc("flush_ring_per_request") by STRING METHOD NAME on the QK worker
    (ProbeHookQKWorker.flush_ring_per_request) — ships no plain-function payload, so
    VLLM_ALLOW_INSECURE_SERIALIZATION is NOT needed. Drives the per-request drain's end-of-run
    delivery and returns the rank-0 (deliverables, residency_after) tuple, or None if the per-request
    QK ring path was not installed. The worker serializes the tuple to ZSTD-COMPRESSED PICKLE BYTES
    (collective_rpc drops raw torch tensors -> they arrive as lists), so decompress+unpickle here.
    TP=1 in scope -> a single rank-0 result."""
    rows = None
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc("flush_ring_per_request")
        except Exception as e:  # noqa: BLE001
            print(f"[qk-ring-perreq-parity] collective_rpc('flush_ring_per_request') via "
                  f"{type(h).__name__} failed: {e}", flush=True)
            rows = None
            continue
        if rows and rows[0] is not None:
            break
    if not rows or rows[0] is None:
        return None
    blob = rows[0]  # zstd(pickle((deliverables, residency_after)))
    return pickle.loads(zstd.ZstdDecompressor().decompress(blob))


def _perreq_layer_store(deliverables, req_id):
    """{layer_num:int(0-based) -> {"q": cpu f32 tensor, "k_all": [cpu f32 tensors]}} for ONE request,
    from the flush_ring_per_request deliverables dict (keyed by INTERNAL req_id; matched via
    _matches_req_id). The delivered per-layer entry is ALREADY the ring's flat-q / growing-prefix-list
    k_all form (assemble_qk), so no normalization is needed on this side."""
    matched_key = None
    for internal_id in deliverables:
        if _matches_req_id(internal_id, req_id):
            matched_key = internal_id
            break
    if matched_key is None:
        return {}
    per_layer = deliverables[matched_key]
    store = {}
    for layer, d in per_layer.items():
        q = d.get("q")
        k_all = d.get("k_all")
        if q is None or k_all is None:
            continue
        store[int(layer)] = {
            "q": q.detach().to(torch.float32).cpu(),
            "k_all": [t.detach().to(torch.float32).cpu() for t in k_all],
        }
    return store


def capture(mode, out_path):
    ring_dir = None
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ["VLLM_HOOK_RING_PER_REQUEST"] = "1"   # arm the per-request demux path
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

    # Ring capture is the FULL-cudagraph mechanism (compile ON); default to FULL.
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

    print(f"[qk-ring-perreq-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file}"
          + (f" ring_dir={ring_dir} per_request=1" if ring_dir else ""), flush=True)

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
        # Prefix-cache reconstruction is DEFERRED on the QK ring path (v1): assembly/reader raise
        # NotImplementedError if a request's first captured step has num_computed>0. Fresh prefill
        # only on BOTH engines, so this is apples-to-apples (matches qk_ring_parity.py).
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
        **extra,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        extra_args=({"hooks_on": _HOOKS_ON} if _HOOKS_ON else None),
    )
    print(f"[qk-ring-perreq-parity:{mode}] sampling max_tokens={_MAX_TOKENS} "
          f"hooks_on={_HOOKS_ON or 'default(prefill)'}", flush=True)

    result = {}
    if mode == "graph":
        # ONE batched generate over ALL cases -> a genuine interleaved batch the consumer must demux
        # by req_id. Probes are inert on the ring path (bank never built) -> reconstruction is
        # deferred to flush_ring_per_request below.
        prompts = [c["text"] for c in _CASES]
        outs = llm.generate(prompts, sp, save_to_disk=False)
        req_ids_by_case = {}
        for case, out in zip(_CASES, outs):
            req_ids_by_case[case["name"]] = str(out.request_id)
            token_ids = list(out.outputs[0].token_ids)
            result[case["name"]] = {"layers": None, "token_ids": token_ids}
            print(f"[qk-ring-perreq-parity:{mode}] case={case['name']} req_id={out.request_id} "
                  f"generated {len(token_ids)} tokens (ring demux deferred to flush)", flush=True)

        popped = _flush_ring_per_request(llm)
        if popped is None:
            raise RuntimeError(
                "flush_ring_per_request collective_rpc returned no payload -- the per-request QK "
                "capture-ring path is not installed on the worker (check VLLM_HOOK_ALLOW_CUDAGRAPH "
                "/ VLLM_HOOK_QK_CAPTURE / VLLM_HOOK_RING_PER_REQUEST).")
        deliverables, residency_after = popped
        result[_RESIDENCY_KEY] = int(residency_after)
        print(f"[qk-ring-perreq-parity:{mode}] flush_ring_per_request -> "
              f"{len(deliverables)} delivered request(s), "
              f"RESIDENCY residency_after={residency_after} (gate: ==0)", flush=True)
        if residency_after != 0:
            print(f"[qk-ring-perreq-parity:{mode}] RESIDENCY GATE FAILED: residency_after="
                  f"{residency_after} != 0 (a captured request leaked in the PerRequestIndex).",
                  flush=True)
        for case in _CASES:
            store = _perreq_layer_store(deliverables, req_ids_by_case[case["name"]])
            result[case["name"]]["layers"] = store
            print(f"[qk-ring-perreq-parity:{mode}] case={case['name']} reconstructed {len(store)} "
                  f"layers from per-request delivery (req_id={req_ids_by_case[case['name']]})",
                  flush=True)
    else:
        # Eager ground truth: SOLO generate per prompt -> clean per-request probes (a batched eager
        # call would trip HookLLM's lossy multi-request probe-merge onto outputs[0]).
        for case in _CASES:
            out = llm.generate(case["text"], sp, save_to_disk=False)
            store = _eager_layer_store(out)
            token_ids = list(out[0].outputs[0].token_ids)
            result[case["name"]] = {"layers": store, "token_ids": token_ids}
            print(f"[qk-ring-perreq-parity:{mode}] case={case['name']} captured {len(store)} "
                  f"layers (eager probes)", flush=True)
            try:
                llm.llm_engine.reset_prefix_cache()
            except Exception:  # noqa: BLE001
                pass

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[qk-ring-perreq-parity:{mode}] wrote {out_path}", flush=True)
    return 0


def compare(graph_path, eager_path, rtol, atol):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    total = matched = 0

    # Residency gate (per-request delivery): every captured request must have been delivered AND
    # freed -> the index is empty (residency_after == 0). A non-zero value means a captured request
    # leaked, so FAIL the verdict loudly even if the tensors that DID come back matched.
    residency = g.get(_RESIDENCY_KEY)
    if residency is None:
        print("[qk-ring-perreq-parity] RESIDENCY residency_after=MISSING "
              "(graph leg did not record it) -> FAIL")
        overall_ok = False
    else:
        print(f"[qk-ring-perreq-parity] RESIDENCY residency_after={residency} (gate: ==0)")
        if residency != 0:
            overall_ok = False

    case_keys = sorted((set(g) & set(e)) - {_RESIDENCY_KEY})
    for case in case_keys:
        g_tok, e_tok = g[case].get("token_ids"), e[case].get("token_ids")
        if g_tok != e_tok:
            print(f"[qk-ring-perreq-parity] case={case}: TOKEN MISMATCH graph={g_tok} "
                  f"eager={e_tok} (q/k_all compare below is not apples-to-apples if these diverge)")
            overall_ok = False

        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        # Layer-set completeness gate: a partial-layer silent capture must NOT pass on the common
        # subset. If graph and eager did not capture the SAME layer set, fail the verdict loudly.
        if set(g_layers) != set(e_layers):
            print(f"[qk-ring-perreq-parity] case={case}: LAYER-SET MISMATCH "
                  f"graph={sorted(g_layers)} eager={sorted(e_layers)} "
                  f"(missing_in_graph={sorted(set(e_layers) - set(g_layers))}, "
                  f"missing_in_eager={sorted(set(g_layers) - set(e_layers))}) -> FAIL")
            overall_ok = False
        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[qk-ring-perreq-parity] case={case}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]

            # ---- q: single flat tensor per layer ----
            gq, eq = gv.get("q"), ev.get("q")
            if gq is None or eq is None:
                print(f"[qk-ring-perreq-parity] case={case} layer={layer} q: MISSING "
                      f"(graph={gq is not None} eager={eq is not None})")
                overall_ok = False
            else:
                total += 1
                if gq.shape != eq.shape:
                    print(f"[qk-ring-perreq-parity] case={case} layer={layer} q: SHAPE MISMATCH "
                          f"graph={tuple(gq.shape)} eager={tuple(eq.shape)}")
                    overall_ok = False
                else:
                    ok = torch.allclose(gq, eq, rtol=rtol, atol=atol)
                    md = (gq - eq).abs().max().item() if gq.numel() else 0.0
                    mean = (gq - eq).abs().mean().item() if gq.numel() else 0.0
                    matched += int(ok)
                    overall_ok = overall_ok and ok
                    print(f"[qk-ring-perreq-parity] case={case} layer={layer} q: match={ok} "
                          f"max|Δ|={md:.3e} mean|Δ|={mean:.3e} shape={tuple(gq.shape)}")

            # ---- k_all: a LIST of growing-prefix tensors, compared entry-for-entry ----
            gk, ek = gv.get("k_all"), ev.get("k_all")
            if gk is None or ek is None:
                print(f"[qk-ring-perreq-parity] case={case} layer={layer} k_all: MISSING "
                      f"(graph={gk is not None} eager={ek is not None})")
                overall_ok = False
                continue
            if len(gk) != len(ek):
                print(f"[qk-ring-perreq-parity] case={case} layer={layer} k_all: STEP-COUNT "
                      f"MISMATCH graph={len(gk)} eager={len(ek)} (a dropped/extra step)")
                overall_ok = False
                continue
            steps_ok = 0
            for idx, (gkt, ekt) in enumerate(zip(gk, ek)):
                total += 1
                if gkt.shape != ekt.shape:
                    print(f"[qk-ring-perreq-parity] case={case} layer={layer} k_all[{idx}]: SHAPE "
                          f"MISMATCH graph={tuple(gkt.shape)} eager={tuple(ekt.shape)}")
                    overall_ok = False
                    continue
                ok = torch.allclose(gkt, ekt, rtol=rtol, atol=atol)
                md = (gkt - ekt).abs().max().item() if gkt.numel() else 0.0
                matched += int(ok)
                steps_ok += int(ok)
                overall_ok = overall_ok and ok
                print(f"[qk-ring-perreq-parity] case={case} layer={layer} k_all[{idx}]: match={ok} "
                      f"max|Δ|={md:.3e} shape={tuple(gkt.shape)}")
            print(f"[qk-ring-perreq-parity] case={case} layer={layer} k_all: {steps_ok}/{len(gk)} "
                  f"steps within tol")

    print("=" * 60)
    print(f"[qk-ring-perreq-parity] {matched}/{total} tensors within rtol={rtol} atol={atol}; "
          f"residency_after={residency} (gate ==0)")
    if overall_ok and total > 0:
        print("[qk-ring-perreq-parity] VERDICT: PASS — per-request ring QK delivery matches eager "
              "and residency drained to 0.")
        return 0
    print("[qk-ring-perreq-parity] VERDICT: FAIL — per-request ring QK delivery diverged from "
          "eager or residency gate not satisfied.")
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
