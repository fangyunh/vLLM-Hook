"""Subprocess-isolated graph-vs-eager HIDDEN-STATE parity oracle for the capture-RING path
(branch `capture_ring`, plan Task 10).

HS ring analogue of tests/cuda_graph/tests/hs_graph/hs_parity.py. Captures the same prompts
twice — once under FULL cudagraph + the capture-ring buffer path (scatter -> shared GPU ring ->
synchronous per-step drain -> durable per-layer raw files, see graph/install_hs.py), once under
the legacy eager register_forward_hook path — in SEPARATE processes (exclusive-process GPU), then
asserts the per-(request, layer) hidden states match within tolerance.

Key structural difference from hs_parity.py: the ring path never populates the RPC/bank buckets
(`get_captured_states` is inert on this path — see workers/probe_hidden_states_worker.py's
graph_install docstring), so `out[0].probes["hs_cache"]` is EMPTY under graph mode. The graph
leg instead calls the worker's `flush_ring()` collective_rpc (by STRING METHOD NAME — no
plain-function payload, so no VLLM_ALLOW_INSECURE_SERIALIZATION needed; mirrors
tests/cuda_graph/tests/cb_oom/cb_oom_parity.py's `_rpc` convention) once, after all cases have
generated, then reconstructs every request's per-layer tensors via
graph.ring_reader.load_multilayer_ring_artifact(run_dir).

Layer-key alignment: the ring stores LayerEntry.layer = L+1 (1-based, == HSHookHost's
egress_layer_num). The eager register_forward_hook path passes the SAME 1-based number as
`layer_num` (probe_hidden_states_worker.py: `lambda ... ln=layer_num+1: hs_hook(o, n, ln)`). Both
normalize functions below key their output dict on this integer `layer_num` (NOT the hs_cache
dict's module-name key), so compare() diffs matching layers regardless of internal naming.

    python hs_ring_parity.py capture --mode graph --out g.pkl
    python hs_ring_parity.py capture --mode eager --out e.pkl
    python hs_ring_parity.py compare --graph g.pkl --eager e.pkl
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

# Decode-stage parity knobs (same contract as hs_parity.py):
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


def _flatten_eager_hs(t):
    """Un-pad a driver-side hs_cache tensor to the FLAT per-token-row layout the ring
    reconstruction produces (the ring concatenates one LayerEntry's rows per step, in step
    order, for every step that routed a real slot).

    - last_token mode: `t` is already `(num_steps, hidden)` (torch.stack of single rows, one
      per hook fire) -> already flat (no padding exists in this shape), return as-is.
    - all_tokens mode: `t` is `(num_steps, max_len, hidden)` from `pad_sequence` (worker's
      get_captured_states / marshal_finished). Step 0 is always the PREFILL forward — the
      longest span, because every DECODE step in this harness captures exactly ONE token (no
      chunked prefill, no speculative decoding) — so `max_len == step 0's real length` and step
      0 needs no trimming; every later step's only REAL row is row 0 (the rest is
      pad_sequence's zero padding, added on the right of dim 1). Concatenating step 0 (full)
      with row 0 of every later step reproduces the ring's flat per-token concatenation exactly.
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


def _eager_layer_store(out):
    """{layer_num:int -> flat cpu f32 tensor} from a generate() output's probes, keyed on the
    entry's `layer_num` (1-based) rather than the hs_cache dict's module-name key, so it lines
    up with the ring's `L+1` layer numbering in compare()."""
    probes = getattr(out[0], "probes", None)
    hs = (probes or {}).get("hs_cache", {})
    store = {}
    for _mod, entry in hs.items():
        if not isinstance(entry, dict):
            continue
        layer_num = entry.get("layer_num")
        if layer_num is None:
            continue
        flat = _flatten_eager_hs(entry.get("hidden_states"))
        if flat is None:
            continue
        store[int(layer_num)] = flat.detach().to(torch.float32).cpu()
    return store


def _flush_ring(llm):
    """collective_rpc("flush_ring") by STRING METHOD NAME on the worker
    (ProbeHiddenStatesWorker.flush_ring, workers/probe_hidden_states_worker.py) — ships no
    plain-function payload, so VLLM_ALLOW_INSECURE_SERIALIZATION is NOT needed (matches
    tests/cuda_graph/tests/cb_oom/cb_oom_parity.py's `_rpc` convention, unlike the AOT-probe
    helper which ships an actual callable). Final drain + write the shared sidecar; returns the
    per-worker run_dir (None if the ring path was not installed). TP=1 in scope (assumption 8 in
    the Task 7-9 report) -> a single rank-0 result."""
    rows = None
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc("flush_ring")
        except Exception as e:  # noqa: BLE001
            print(f"[hs-ring-parity] collective_rpc('flush_ring') via {type(h).__name__} "
                  f"failed: {e}", flush=True)
            rows = None
            continue
        if rows:
            break
    if not rows:
        return None
    return rows[0]


def _ring_layer_store(run_dir, req_id):
    """{layer_num:int -> cpu f32 tensor} for ONE request, reconstructed from the durable ring
    dump (graph.ring_reader.load_multilayer_ring_artifact)."""
    from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact
    data = load_multilayer_ring_artifact(run_dir)
    matched_key = None
    for internal_id in data:
        if _matches_req_id(internal_id, req_id):
            matched_key = internal_id
            break
    if matched_key is None:
        return {}
    per_layer = data[matched_key]
    return {int(layer): t.detach().to(torch.float32).cpu() for layer, t in per_layer.items()}


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
    # hs_parity.py's PIECEWISE default.
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[hs-ring-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file}"
          + (f" ring_dir={ring_dir}" if ring_dir else ""), flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
        **extra,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        extra_args=({"hooks_on": _HOOKS_ON} if _HOOKS_ON else None),
    )
    print(f"[hs-ring-parity:{mode}] sampling max_tokens={_MAX_TOKENS} "
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
            print(f"[hs-ring-parity:{mode}] case={case['name']} captured {len(store)} layers "
                  f"(eager probes)", flush=True)
        else:
            # Ring path: probes are inert (bank never built) -> deferred to the post-loop
            # flush_ring + load_multilayer_ring_artifact reconstruction (Task 7-9 assumption 7).
            result[case["name"]] = {"layers": None, "token_ids": token_ids}
            print(f"[hs-ring-parity:{mode}] case={case['name']} generated {len(token_ids)} "
                  f"tokens (ring reconstruction deferred to flush)", flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass

    if mode == "graph":
        run_dir = _flush_ring(llm)
        if not run_dir:
            raise RuntimeError(
                "flush_ring collective_rpc returned no run_dir -- the capture-ring path is "
                "not installed on the worker (check VLLM_HOOK_ALLOW_CUDAGRAPH / "
                "VLLM_HOOK_HS_CAPTURE).")
        print(f"[hs-ring-parity:{mode}] flush_ring -> run_dir={run_dir}", flush=True)

        for case in _CASES:
            store = _ring_layer_store(run_dir, req_ids_by_case[case["name"]])
            result[case["name"]]["layers"] = store
            print(f"[hs-ring-parity:{mode}] case={case['name']} reconstructed {len(store)} "
                  f"layers from ring dump (req_id={req_ids_by_case[case['name']]})", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[hs-ring-parity:{mode}] wrote {out_path}", flush=True)
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
            print(f"[hs-ring-parity] case={case}: TOKEN MISMATCH graph={g_tok} eager={e_tok} "
                  f"(hidden-state compare below is not apples-to-apples if these diverge)")
            overall_ok = False

        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[hs-ring-parity] case={case}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]
            if gv is None or ev is None:
                continue
            total += 1
            if gv.shape != ev.shape:
                print(f"[hs-ring-parity] case={case} layer={layer}: SHAPE MISMATCH "
                      f"graph={tuple(gv.shape)} eager={tuple(ev.shape)}")
                overall_ok = False
                continue
            ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
            md = (gv - ev).abs().max().item() if gv.numel() else 0.0
            mean = (gv - ev).abs().mean().item() if gv.numel() else 0.0
            matched += int(ok)
            overall_ok = overall_ok and ok
            print(f"[hs-ring-parity] case={case} layer={layer}: match={ok} "
                  f"max|Δ|={md:.3e} mean|Δ|={mean:.3e} shape={tuple(gv.shape)}")

    print("=" * 60)
    print(f"[hs-ring-parity] {matched}/{total} tensors within rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print("[hs-ring-parity] VERDICT: PASS — ring HS capture matches eager.")
        return 0
    print("[hs-ring-parity] VERDICT: FAIL — ring HS capture diverged from eager.")
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
