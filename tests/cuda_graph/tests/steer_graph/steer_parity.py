"""Subprocess-isolated graph-vs-eager ACTIVATION-STEERING parity oracle (v0.3.0).

Steering is the mutating counterpart of the read-only QK/HS capture: it adds a
direction to the residual stream at one layer, changing the model's OUTPUT. So
unlike qk_parity / hs_parity (which compare captured tensors), this compares the
EFFECT of steering on the next-token logprob distribution.

Each engine boot runs TWO generations on the same prompt:
  * base  — use_hook=False, no steering (the control)
  * steer — use_hook=True, steering applied at the config's optimal_layer

with ``max_tokens=1`` so only the PREFILL forward runs — graph mode steers prefill
via the steer_residual op (decode under cudagraph replay is out of scope, like the
capture milestones). We boot once per mode in separate processes (exclusive-process
GPU) and compare:

    python steer_parity.py capture --mode graph --out g.pkl
    python steer_parity.py capture --mode eager --out e.pkl
    python steer_parity.py compare --graph g.pkl --eager e.pkl

PASS = graph steering reproduces eager steering: per case, graph-steered's
next-token logprobs match eager-steered's (D_parity) FAR more closely than the
steering effect itself (D_steer), and the argmax token agrees — while a non-vacuous
D_steer confirms steering actually did something. D_parity is judged against the
graph-vs-eager numeric drift floor measured on the unsteered control (D_basedrift).
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

# Steering strength override for the test (the config's dummy vector is unit-norm
# with coefficient 1.0, too weak to shift the output — bump it so the effect is
# unambiguous). Tunable so a different model/vector can dial it in.
_COEFF = float(os.environ.get("VLLM_STEER_COEFF", "20.0"))
_TOPK = int(os.environ.get("VLLM_STEER_LOGPROBS", "20"))

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _logprob_map(out):
    """Extract {token_id: logprob} for the single generated position + the
    sampled token id and decoded text."""
    o = out[0].outputs[0]
    lps = o.logprobs[0] if o.logprobs else {}
    m = {int(tid): float(lp.logprob) for tid, lp in lps.items()}
    return {
        "logprobs": m,
        "token_id": int(o.token_ids[0]) if o.token_ids else None,
        "text": o.text,
    }


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

    print(f"[steer-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file} coeff={_COEFF}", flush=True)

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

    # max_tokens=1: only the prefill forward runs, so steering is prefill-only in
    # BOTH modes — a valid graph-vs-eager comparison (decode steering under graph
    # replay is out of scope). logprobs gives the full next-token distribution.
    sp_base = SamplingParams(temperature=0.0, max_tokens=1, logprobs=_TOPK)
    sp_steer = SamplingParams(temperature=0.0, max_tokens=1, logprobs=_TOPK,
                              extra_args={"steer": {"coefficient": _COEFF}})

    result = {}
    for case in _CASES:
        base = _logprob_map(llm.generate(case["text"], sp_base, use_hook=False))
        steer = _logprob_map(llm.generate(case["text"], sp_steer, use_hook=True))
        result[case["name"]] = {"base": base, "steer": steer}
        print(f"[steer-parity:{mode}] case={case['name']} "
              f"base_tok={base['token_id']}({base['text']!r}) "
              f"steer_tok={steer['token_id']}({steer['text']!r})", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[steer-parity:{mode}] wrote {out_path}", flush=True)
    return 0


def _maxdiff(a: dict, b: dict):
    """Max abs logprob difference over the shared token ids of two distributions.
    Returns (maxdiff, n_shared). Empty intersection -> inf."""
    shared = set(a) & set(b)
    if not shared:
        return float("inf"), 0
    return max(abs(a[t] - b[t]) for t in shared), len(shared)


def compare(graph_path, eager_path, atol, steer_min, ratio):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    any_case = False
    for case in sorted(set(g) & set(e)):
        any_case = True
        gb, gs = g[case]["base"]["logprobs"], g[case]["steer"]["logprobs"]
        eb, es = e[case]["base"]["logprobs"], e[case]["steer"]["logprobs"]

        d_steer, _ = _maxdiff(es, eb)         # eager: effect of steering
        d_parity, n_par = _maxdiff(gs, es)    # graph-steer vs eager-steer (the test)
        d_drift, _ = _maxdiff(gb, eb)         # graph-vs-eager numeric floor (unsteered)

        gtok = g[case]["steer"]["token_id"]
        etok = e[case]["steer"]["token_id"]
        tok_ok = (gtok == etok)

        # Steering is non-vacuous if it moved the distribution meaningfully.
        vacuous = d_steer < steer_min
        # Parity holds if graph-steer tracks eager-steer within the numeric floor
        # (a small absolute atol, or comfortably within the steering effect).
        parity_ok = (d_parity <= max(atol, d_drift * 1.5)
                     or d_parity <= d_steer * ratio)

        case_ok = tok_ok and parity_ok and not vacuous
        overall_ok = overall_ok and case_ok

        print(f"[steer-parity] case={case}: "
              f"argmax graph={gtok} eager={etok} match={tok_ok} | "
              f"D_steer={d_steer:.3e} D_parity={d_parity:.3e} "
              f"D_basedrift={d_drift:.3e} (shared={n_par}) "
              f"=> {'OK' if case_ok else 'FAIL'}"
              + ("  [VACUOUS: steering too weak, raise VLLM_STEER_COEFF]" if vacuous else ""))

    print("=" * 70)
    if overall_ok and any_case:
        print("[steer-parity] VERDICT: PASS — graph steering matches eager steering.")
        return 0
    print("[steer-parity] VERDICT: FAIL — graph steering diverged from eager.")
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
    pk.add_argument("--atol", type=float, default=0.15)
    pk.add_argument("--steer-min", type=float, default=0.1)
    pk.add_argument("--ratio", type=float, default=0.25)
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out))
    else:
        sys.exit(compare(args.graph, args.eager, args.atol, args.steer_min, args.ratio))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
