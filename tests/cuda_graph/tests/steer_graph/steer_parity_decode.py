"""Subprocess-isolated graph-vs-eager DECODE-phase steering parity oracle (v0.3.0).

The prefill oracle (steer_parity.py, max_tokens=1) only proves steering on the
single prefill forward. This proves steering across DECODE: the eager path steers
EVERY forward (prefill + each decode step) via its register_forward_hook; under
PIECEWISE the steer_residual splitting op is an eager seam that likewise runs on
every step (proven for the read-only hs_probe at max_tokens=16). The open question
this answers is whether the op's IN-PLACE residual mutation propagates downstream
on cudagraph REPLAY (decode), not just on the live prefill forward.

Each engine boot runs TWO greedy generations of N tokens on the same prompt:
  * base  — use_hook=False (no steering, the control)
  * steer — use_hook=True (steering at the config's optimal_layer, every step)

PASS = graph-steered reproduces eager-steered token-for-token across all N decode
steps (greedy temp=0 is deterministic; HS decode parity showed Qwen2-1.5B fp32
graph-vs-eager logits track to ~1e-3, so the sampled sequences agree). A non-vacuous
steering effect (steer sequence != base sequence) keeps the test honest, and the
graph-vs-eager BASE sequence agreement contextualizes any steered mismatch as
steering-specific rather than generic cudagraph drift.

    python steer_parity_decode.py capture --mode graph --out g.pkl
    python steer_parity_decode.py capture --mode eager --out e.pkl
    python steer_parity_decode.py compare --graph g.pkl --eager e.pkl
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

_COEFF = float(os.environ.get("VLLM_STEER_COEFF", "20.0"))
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PARITY_MAX_TOKENS", "12"))

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _seq(out):
    """Return the generated token-id list + decoded text for a single request."""
    o = out[0].outputs[0]
    return {"token_ids": [int(t) for t in o.token_ids], "text": o.text}


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/activation_steer/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[steer-decode:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file} coeff={_COEFF} max_tokens={_MAX_TOKENS}", flush=True)

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

    # Greedy multi-token generation so steering must fire on every DECODE step in
    # both modes. No logprobs needed — the sampled sequence is the signal.
    sp_base = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
    sp_steer = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS,
                              extra_args={"steer": {"coefficient": _COEFF}})

    result = {}
    for case in _CASES:
        base = _seq(llm.generate(case["text"], sp_base, use_hook=False))
        steer = _seq(llm.generate(case["text"], sp_steer, use_hook=True))
        result[case["name"]] = {"base": base, "steer": steer}
        print(f"[steer-decode:{mode}] case={case['name']}\n"
              f"    base : {base['token_ids']}\n           {base['text']!r}\n"
              f"    steer: {steer['token_ids']}\n           {steer['text']!r}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[steer-decode:{mode}] wrote {out_path}", flush=True)
    return 0


def _lcp(a: list, b: list) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(graph_path, eager_path):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    any_case = False
    for case in sorted(set(g) & set(e)):
        any_case = True
        gs = g[case]["steer"]["token_ids"]
        es = e[case]["steer"]["token_ids"]
        gb = g[case]["base"]["token_ids"]
        eb = e[case]["base"]["token_ids"]

        steer_ok = (gs == es)                 # graph-steer vs eager-steer (the test)
        vacuous = (es == eb)                  # steering changed nothing -> inconclusive
        base_ok = (gb == eb)                  # graph-vs-eager determinism (context)

        n = max(len(gs), len(es))
        lcp = _lcp(gs, es)
        case_ok = steer_ok and not vacuous
        overall_ok = overall_ok and case_ok

        print(f"[steer-decode] case={case}: steer_match={steer_ok} "
              f"(LCP={lcp}/{n}) base_match={base_ok} "
              f"steer_changed_output={not vacuous} => {'OK' if case_ok else 'FAIL'}")
        if not steer_ok:
            print(f"    graph-steer: {gs}")
            print(f"    eager-steer: {es}")
        if vacuous:
            print("    [VACUOUS: steering left the continuation unchanged — raise "
                  "VLLM_STEER_COEFF]")

    print("=" * 70)
    if overall_ok and any_case:
        print("[steer-decode] VERDICT: PASS — graph steering matches eager across decode.")
        return 0
    print("[steer-decode] VERDICT: FAIL — graph decode steering diverged from eager.")
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
