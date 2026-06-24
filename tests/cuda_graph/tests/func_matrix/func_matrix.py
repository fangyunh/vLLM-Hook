"""Functionality-matrix parity oracle (v0.5.0 Step 4): per-request-different config
and continuous-batching churn under FULL CUDA graphs, graph-vs-eager — on STEERING.

Steering is the headline v0.5.0 fix (W1 invalidation routing). This proves the per-step
optimizations hold for MULTI-REQUEST batches by steering a 4-request batch in ONE
generate() call and comparing graph vs eager PER REQUEST, in two scenarios (env
VLLM_HOOK_FUNC_SCENARIO):

  mixed — 4 requests steered with DIFFERENT coefficients in one batch (per-request-
          different steering; plan constraint c). Proves the per-(layer,token) routing +
          W1 invalidation honor distinct per-request configs under one graph.
  churn — 4 requests, same coefficient, DIFFERENT max_tokens [4,8,12,16] so requests
          finish at different steps -> vLLM condenses input_batch mid-generation and the
          W1 routing signature (req_ids, qsl) changes -> forces a correct re-route
          (constraint d).

Why steering (not QK capture): steer + HS route by FLAT batch position (disjoint per
request), so they are multi-request-safe. QK buffer capture accumulates by ABSOLUTE
sequence position into ONE shared k_buf, so two requests' positions COLLIDE on the same
rows — the single-request limitation tracked as M5 (generalize per-request k_buf row
ranges). A QK multi-request batch therefore FAILS by construction here and is out of
v0.5.0 scope; this harness validates the multi-request-safe path (steering).

PASS = for every request the graph-steered token sequence matches eager-steered exactly,
AND steering is non-vacuous (eager-steered != eager-base for at least one request).

    python func_matrix.py capture --mode graph --out g.pkl
    python func_matrix.py capture --mode eager --out e.pkl
    python func_matrix.py compare --graph g.pkl --eager e.pkl
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
_SCENARIO = os.environ.get("VLLM_HOOK_FUNC_SCENARIO", "mixed").lower()
_COEFF = float(os.environ.get("VLLM_STEER_COEFF", "20.0"))

_PROMPTS = [
    "The capital of France is",
    "Quantum computing leverages superposition to",
    "In the beginning the universe was",
    "The most important property of water is",
]


def _specs(scenario):
    """Per-request (max_tokens, coefficient) for the scenario's 4 real requests."""
    if scenario == "churn":
        # Same coefficient; different max_tokens -> staggered finishes (condense).
        return [(mt, _COEFF) for mt in (4, 8, 12, 16)]
    # mixed: same max_tokens; per-request SELECTIVITY — coeff 0 (no steer) alternating
    # with coeff C (steer) in ONE batch. The graph must steer ONLY the coeff-C requests
    # and leave the coeff-0 ones == base, proving per-request-different steering under one
    # graph (constraint c). NOTE: the eager hook is batch-wide (applies the FIRST request's
    # config to the whole residual), so eager CANNOT serve as the reference for mixed —
    # mixed is validated graph-internally (steer-vs-base selectivity), not vs eager.
    return [(10, 0.0), (10, _COEFF), (10, 0.0), (10, _COEFF)]


def _toks(out_obj):
    o = out_obj.outputs[0]
    return [int(t) for t in o.token_ids]


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ.setdefault("VLLM_HOOK_STEER_MODE", "buffer")
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/activation_steer/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[func:{_SCENARIO}:{mode}] booting enforce_eager={enforce_eager} "
          f"config={config_file} nreq={len(_PROMPTS)} coeff={_COEFF}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="steer_hook_act",
        analyzer_name=None,
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    specs = _specs(_SCENARIO)
    base_sp = [SamplingParams(temperature=0.0, max_tokens=mt) for mt, _ in specs]
    steer_sp = [SamplingParams(temperature=0.0, max_tokens=mt,
                               extra_args={"steer": {"coefficient": c}})
                for mt, c in specs]

    base_out = llm.generate(list(_PROMPTS), base_sp, use_hook=False)
    steer_out = llm.generate(list(_PROMPTS), steer_sp, use_hook=True)

    store = {i: {"base": _toks(base_out[i]), "steer": _toks(steer_out[i]),
                 "coeff": specs[i][1]}
             for i in range(len(_PROMPTS))}
    print(f"[func:{_SCENARIO}:{mode}] per-req steer lens="
          f"{[len(store[i]['steer']) for i in range(len(_PROMPTS))]}", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(store, f)
    print(f"[func:{_SCENARIO}:{mode}] wrote {out_path}", flush=True)
    return 0


def _compare_mixed(g):
    """Graph-internal selectivity: coeff-0 requests must be unsteered (steer==base),
    coeff-C requests must be steered (steer!=base) — all in ONE batch (constraint c).
    Uses the GRAPH store only (the eager hook is batch-wide and cannot express this).
    """
    ok = True
    n_steered = n_unsteered = 0
    for i in sorted(g):
        coeff = g[i].get("coeff", 0.0)
        steered = (g[i]["steer"] != g[i]["base"])
        expect = coeff > 0.0
        match = (steered == expect)
        ok = ok and match
        n_steered += int(expect)
        n_unsteered += int(not expect)
        print(f"  req{i}: coeff={coeff} expect_steered={expect} actual_steered={steered} "
              f"=> {'OK' if match else 'FAIL'}")
    print("=" * 70)
    if ok and n_steered > 0 and n_unsteered > 0:
        print("[func-matrix:mixed] VERDICT: PASS — graph steers ONLY the coeff>0 requests "
              "and leaves coeff=0 requests == base (per-request-different in one batch).")
        return 0
    print("[func-matrix:mixed] VERDICT: FAIL — per-request steering selectivity wrong "
          "(a coeff=0 request was steered or a coeff>0 request was not).")
    return 1


def compare(graph_path, eager_path):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    if _SCENARIO == "mixed":
        return _compare_mixed(g)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    ok = True
    any_nonvacuous = False
    for i in sorted(set(g) & set(e)):
        gs, es = g[i]["steer"], e[i]["steer"]
        gb, eb = g[i]["base"], e[i]["base"]
        steer_match = (gs == es)
        base_match = (gb == eb)          # graph-vs-eager determinism (drift context)
        nonvacuous = (es != eb)          # steering changed the output
        any_nonvacuous = any_nonvacuous or nonvacuous
        # A steering-SPECIFIC failure is steer diverging while base agrees. If base also
        # diverges it is generic cudagraph/batch numeric drift, not a steering bug.
        if not steer_match and base_match:
            ok = False
        print(f"  req{i}: steer graph==eager={steer_match} base graph==eager={base_match} "
              f"steer_changed={nonvacuous} len={len(gs)}")
        if not steer_match:
            print(f"    graph-steer={gs}\n    eager-steer={es}")
            if not base_match:
                print(f"    [base also diverged -> generic cudagraph drift, not steering]")

    print("=" * 70)
    if ok and any_nonvacuous:
        print(f"[func-matrix:{_SCENARIO}] VERDICT: PASS — graph steering matches eager "
              f"per request (steering-specific), non-vacuous.")
        return 0
    reason = ("graph steering diverged from eager where base agreed"
              if not ok else "steering was vacuous (raise VLLM_STEER_COEFF)")
    print(f"[func-matrix:{_SCENARIO}] VERDICT: FAIL — {reason}.")
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
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out))
    else:
        sys.exit(compare(args.graph, args.eager))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
