"""Subprocess-isolated graph-vs-eager HIDDEN-STATE parity oracle for the PER-REQUEST ring-DEMUX
delivery path (branch `capture_ring`, plan Task 6). This is the byte-identity GATE for the
per-request ring demux (Task 5, commits ec6c876 + 47ec5be).

Sibling of tests/cuda_graph/tests/ring/hs_ring_parity.py (the SHARED-FILE ring oracle). Same FULL
cudagraph + capture-ring buffer mechanism, same eager register_forward_hook ground truth — the ONE
difference is how the graph leg reads its captured tensors back:

  * shared-file oracle: flush_ring() -> durable per-layer raw files -> load_multilayer_ring_artifact.
  * THIS oracle (per-request): arms VLLM_HOOK_RING_PER_REQUEST=1 so the off-loop consumer DEMUXES
    each step's drained rows BY req_id into a PerRequestIndex (no shared file written), then reads
    back via the worker's NEW flush_ring_per_request() collective_rpc — which stop()s the drain
    (finalize_all delivers last-step stragglers), pops each request's assembled per-layer tensors,
    frees it, and returns (deliverables, residency_after). residency_after MUST be 0 (the residency
    gate: every captured request delivered + freed, nothing leaked in the index).

To genuinely exercise the demux the GRAPH leg captures BOTH prompts in ONE batched
generate([p1, p2], ...) call — a real interleaved batch whose per-step rows the consumer must split
by req_id. The EAGER leg captures each prompt in its OWN solo generate() call: HookLLM.generate's
convenience probe-merge (hook_llm.py) OVERWRITES outputs[0].probes on any len(outputs)>1 batch and
is LOSSY for multi-step HS (it keeps step 0 only), so a batched eager read would corrupt request 0's
ground truth. Per-request HS is batch-invariant under causal attention + prefix-cache reset (the M5
property — no cross-request value bleed), so a solo-eager reference is exactly the
ground truth for the batched-graph demux. req_ids are matched apples-to-apples via _matches_req_id,
same as hs_ring_parity.py.

Layer-key alignment (unchanged from hs_ring_parity.py): the PerRequestIndex keys each request's
per-layer tensor on LayerEntry.layer = L+1 (1-based, == HSHookHost's egress_layer_num); the eager
register_forward_hook path passes the SAME 1-based number as layer_num. compare() diffs matching
1-based layers directly. Per-request assembly (PerRequestIndex.pop_deliverable = torch.cat of a
request's per-step demuxed row slices in step order) produces the SAME flat per-token layout that
_flatten_eager_hs reconstructs from the eager stacked tensor, so the compare is unchanged.

    python hs_ring_perreq_parity.py capture --mode graph --out g.pkl
    python hs_ring_perreq_parity.py capture --mode eager --out e.pkl
    python hs_ring_perreq_parity.py compare --graph g.pkl --eager e.pkl
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

# Decode-stage parity knobs (same contract as hs_ring_parity.py):
#   VLLM_HOOK_PARITY_MAX_TOKENS > 1  -> generate that many tokens so the ring routing must
#       fire on every decode step, not just prefill.
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
    """Same v1/legacy req_id match rule as workers/_common.py::iter_matching_req_ids (vLLM
    v0.12+ uses the SAME id internally; older versions append a random suffix)."""
    return internal_req_id == external_req_id or internal_req_id.startswith(f"{external_req_id}-")


def _flatten_eager_hs(t):
    """Un-pad a driver-side hs_cache tensor to the FLAT per-token-row layout the per-request ring
    assembly produces (PerRequestIndex.pop_deliverable concatenates one request's per-step demuxed
    row slices, in step order — the SAME flat layout the shared-file ring reconstruction produces).

    - last_token mode: `t` is already `(num_steps, hidden)` (torch.stack of single rows, one per
      hook fire) -> already flat, return as-is.
    - all_tokens mode: `t` is `(num_steps, max_len, hidden)` from `pad_sequence`. Step 0 is the
      PREFILL forward (the longest span, since every DECODE step here captures exactly ONE token) so
      it needs no trimming; every later step's only REAL row is row 0. Concatenating step 0 (full)
      with row 0 of each later step reproduces the ring's flat per-token concatenation exactly."""
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
    entry's `layer_num` (1-based), so it lines up with the ring's `L+1` layer numbering."""
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


def _flush_ring_per_request(llm):
    """collective_rpc("flush_ring_per_request") by STRING METHOD NAME on the HS worker
    (ProbeHiddenStatesWorker.flush_ring_per_request) — ships no plain-function payload, so
    VLLM_ALLOW_INSECURE_SERIALIZATION is NOT needed (mirrors hs_ring_parity.py's _flush_ring).
    Drives the per-request drain's end-of-run delivery and returns the rank-0
    (deliverables, residency_after) tuple, or None if the per-request ring path was not installed.
    The worker serializes the tuple to ZSTD-COMPRESSED PICKLE BYTES (collective_rpc drops raw torch
    tensors -> they arrive as lists), so decompress+unpickle here. TP=1 in scope -> a single rank-0
    result."""
    rows = None
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc("flush_ring_per_request")
        except Exception as e:  # noqa: BLE001
            print(f"[hs-ring-perreq-parity] collective_rpc('flush_ring_per_request') via "
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
    """{layer_num:int -> cpu f32 tensor} for ONE request, from the flush_ring_per_request
    deliverables dict (keyed by INTERNAL req_id; matched via _matches_req_id)."""
    matched_key = None
    for internal_id in deliverables:
        if _matches_req_id(internal_id, req_id):
            matched_key = internal_id
            break
    if matched_key is None:
        return {}
    per_layer = deliverables[matched_key]
    return {int(layer): t.detach().to(torch.float32).cpu() for layer, t in per_layer.items()}


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
        f"model_configs/hidden_states/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[hs-ring-perreq-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file}"
          + (f" ring_dir={ring_dir} per_request=1" if ring_dir else ""), flush=True)

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
    print(f"[hs-ring-perreq-parity:{mode}] sampling max_tokens={_MAX_TOKENS} "
          f"hooks_on={_HOOKS_ON or 'default(prefill)'}", flush=True)

    result = {}
    if mode == "graph":
        # ONE batched generate over ALL cases -> a genuine interleaved batch the consumer must
        # demux by req_id. Probes are inert on the ring path (bank never built) -> reconstruction
        # is deferred to flush_ring_per_request below.
        prompts = [c["text"] for c in _CASES]
        outs = llm.generate(prompts, sp, save_to_disk=False)
        req_ids_by_case = {}
        for case, out in zip(_CASES, outs):
            req_ids_by_case[case["name"]] = str(out.request_id)
            token_ids = list(out.outputs[0].token_ids)
            result[case["name"]] = {"layers": None, "token_ids": token_ids}
            print(f"[hs-ring-perreq-parity:{mode}] case={case['name']} req_id={out.request_id} "
                  f"generated {len(token_ids)} tokens (ring demux deferred to flush)", flush=True)

        popped = _flush_ring_per_request(llm)
        if popped is None:
            raise RuntimeError(
                "flush_ring_per_request collective_rpc returned no payload -- the per-request "
                "capture-ring path is not installed on the worker (check VLLM_HOOK_ALLOW_CUDAGRAPH "
                "/ VLLM_HOOK_HS_CAPTURE / VLLM_HOOK_RING_PER_REQUEST).")
        deliverables, residency_after = popped
        result[_RESIDENCY_KEY] = int(residency_after)
        print(f"[hs-ring-perreq-parity:{mode}] flush_ring_per_request -> "
              f"{len(deliverables)} delivered request(s), "
              f"RESIDENCY residency_after={residency_after} (gate: ==0)", flush=True)
        if residency_after != 0:
            # Loud, greppable, but non-fatal here so the OTHER leg still runs; compare() turns
            # this into a FAIL verdict.
            print(f"[hs-ring-perreq-parity:{mode}] RESIDENCY GATE FAILED: residency_after="
                  f"{residency_after} != 0 (a captured request leaked in the PerRequestIndex).",
                  flush=True)
        for case in _CASES:
            store = _perreq_layer_store(deliverables, req_ids_by_case[case["name"]])
            result[case["name"]]["layers"] = store
            print(f"[hs-ring-perreq-parity:{mode}] case={case['name']} reconstructed {len(store)} "
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
            print(f"[hs-ring-perreq-parity:{mode}] case={case['name']} captured {len(store)} "
                  f"layers (eager probes)", flush=True)
            try:
                llm.llm_engine.reset_prefix_cache()
            except Exception:  # noqa: BLE001
                pass

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[hs-ring-perreq-parity:{mode}] wrote {out_path}", flush=True)
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
        print("[hs-ring-perreq-parity] RESIDENCY residency_after=MISSING "
              "(graph leg did not record it) -> FAIL")
        overall_ok = False
    else:
        print(f"[hs-ring-perreq-parity] RESIDENCY residency_after={residency} (gate: ==0)")
        if residency != 0:
            overall_ok = False

    case_keys = sorted((set(g) & set(e)) - {_RESIDENCY_KEY})
    for case in case_keys:
        g_tok, e_tok = g[case].get("token_ids"), e[case].get("token_ids")
        if g_tok != e_tok:
            print(f"[hs-ring-perreq-parity] case={case}: TOKEN MISMATCH graph={g_tok} "
                  f"eager={e_tok} (hidden-state compare below is not apples-to-apples "
                  f"if these diverge)")
            overall_ok = False

        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        # Layer-set completeness gate: a partial-layer silent capture must NOT pass on the common
        # subset. If graph and eager did not capture the SAME layer set, fail the verdict loudly.
        if set(g_layers) != set(e_layers):
            print(f"[hs-ring-perreq-parity] case={case}: LAYER-SET MISMATCH "
                  f"graph={sorted(g_layers)} eager={sorted(e_layers)} "
                  f"(missing_in_graph={sorted(set(e_layers) - set(g_layers))}, "
                  f"missing_in_eager={sorted(set(g_layers) - set(e_layers))}) -> FAIL")
            overall_ok = False
        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[hs-ring-perreq-parity] case={case}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]
            if gv is None or ev is None:
                continue
            total += 1
            if gv.shape != ev.shape:
                print(f"[hs-ring-perreq-parity] case={case} layer={layer}: SHAPE MISMATCH "
                      f"graph={tuple(gv.shape)} eager={tuple(ev.shape)}")
                overall_ok = False
                continue
            ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
            md = (gv - ev).abs().max().item() if gv.numel() else 0.0
            mean = (gv - ev).abs().mean().item() if gv.numel() else 0.0
            matched += int(ok)
            overall_ok = overall_ok and ok
            print(f"[hs-ring-perreq-parity] case={case} layer={layer}: match={ok} "
                  f"max|Δ|={md:.3e} mean|Δ|={mean:.3e} shape={tuple(gv.shape)}")

    print("=" * 60)
    print(f"[hs-ring-perreq-parity] {matched}/{total} tensors within rtol={rtol} atol={atol}; "
          f"residency_after={residency} (gate ==0)")
    if overall_ok and total > 0:
        print("[hs-ring-perreq-parity] VERDICT: PASS — per-request ring HS delivery matches eager "
              "and residency drained to 0.")
        return 0
    print("[hs-ring-perreq-parity] VERDICT: FAIL — per-request ring HS delivery diverged from "
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
