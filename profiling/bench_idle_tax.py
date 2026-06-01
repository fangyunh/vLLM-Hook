"""bench_idle_tax — quantify the per-layer launch cost of the plugin.

Three configurations (decode-token timing at multiple batch sizes):
  1. baseline       — vLLM, no plugin imported
  2. plugin_loaded  — plugin imported (worker_extension installed) but no
                      hook installed; this is the cost the user pays for
                      having the plugin on the path even when nothing
                      activates it (the "loaded but mostly idle" reality)
  3. hooks_on       — hooks installed but extra_args opted out per request
                      so the hook closures are dispatched but return early

Reports the per-layer overhead (μs) and aggregate idle-tax (%) against the
budgets called out in cuda_graph_plan.html: ≤3 μs per-layer and ≤5%
aggregate at bs=1.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROFILING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = PROFILING_DIR.parent
sys.path.insert(0, str(PROFILING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

from _common import save_csv  # noqa: E402


def _measure(llm, prompts: List[str], n_decode: int, sample_params: dict) -> float:
    """Return median seconds per decode token across the batch."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=n_decode, ignore_eos=True,
                        **sample_params)

    # Warm
    _ = llm.generate(prompts, sp)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.perf_counter() - t0
        total = sum(len(o.outputs[0].token_ids) for o in outs)
        timings.append(dt / max(1, total))
    return statistics.median(timings)


def _run_baseline(model: str, n_decode: int, batches: List[int],
                  enforce_eager: bool, max_model_len: int,
                  gpu_mem_util: float) -> Dict[int, float]:
    from vllm import LLM
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_util,
              enforce_eager=enforce_eager)
    out: Dict[int, float] = {}
    for bs in batches:
        prompts = ["The quick brown fox jumps over the"] * bs
        out[bs] = _measure(llm, prompts, n_decode, sample_params={})
    del llm
    return out


def _run_plugin_loaded(model: str, n_decode: int, batches: List[int],
                       enforce_eager: bool, max_model_len: int,
                       gpu_mem_util: float) -> Dict[int, float]:
    # Importing the plugin registers patches; no hook installation happens
    # because no request opts in.
    import vllm.plugins as vp
    vp.load_general_plugins()
    from vllm import LLM
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_util,
              enforce_eager=enforce_eager)
    out: Dict[int, float] = {}
    for bs in batches:
        prompts = ["The quick brown fox jumps over the"] * bs
        out[bs] = _measure(llm, prompts, n_decode, sample_params={})
    del llm
    return out


def _run_hooks_on_idle(model: str, n_decode: int, batches: List[int],
                       enforce_eager: bool, max_model_len: int,
                       gpu_mem_util: float, hook_dir: str) -> Dict[int, float]:
    """Hooks installed but every request opts out via empty extra_args."""
    import torch
    from vllm_hook_plugins import register_plugins
    from vllm_hook_plugins.hook_llm import HookLLM
    register_plugins()
    llm = HookLLM(
        model=model, worker_name="probe_hidden_states",
        analyzer_name=None, hook_dir=hook_dir,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        enable_hook=False,   # do not activate per-request
        dtype=torch.float16,
    )
    # Force hook installation on the worker even though no request uses it,
    # so the per-fire dispatch cost is paid every step. We do that by
    # generating one request with use_hook=True via a bare extra_args.
    _ = llm.generate(["warmup"], temperature=0.0, max_tokens=1, use_hook=True)

    out: Dict[int, float] = {}
    for bs in batches:
        prompts = ["The quick brown fox jumps over the"] * bs
        # use_hook=False makes HookLLM skip the extra_args dance — the
        # closures still fire (they were installed) but return early because
        # extra_args["output_hidden_states"] is None.
        out[bs] = _measure(llm, prompts, n_decode, sample_params={})
    del llm
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--num-decode-tokens", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--hook-dir", default="/dev/shm/vllm_hook")
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        import torch
        if not torch.cuda.is_available():
            print("[idle_tax] CUDA not available — exiting.")
            return 2
    except ImportError:
        print("[idle_tax] torch not installed.")
        return 2

    print("[idle_tax] === BASELINE ===")
    base = _run_baseline(args.model, args.num_decode_tokens, args.batches,
                         args.enforce_eager, args.max_model_len, args.gpu_mem_util)
    for bs, s in base.items():
        print(f"  bs={bs:3d}  {s*1e6:8.1f} us/tok  ({1/s:8.1f} tok/s)")

    print("[idle_tax] === PLUGIN LOADED (no hook install) ===")
    plug = _run_plugin_loaded(args.model, args.num_decode_tokens, args.batches,
                              args.enforce_eager, args.max_model_len, args.gpu_mem_util)
    for bs, s in plug.items():
        print(f"  bs={bs:3d}  {s*1e6:8.1f} us/tok")

    print("[idle_tax] === HOOKS INSTALLED, REQUEST OPTS OUT ===")
    hooks = _run_hooks_on_idle(args.model, args.num_decode_tokens, args.batches,
                               args.enforce_eager, args.max_model_len,
                               args.gpu_mem_util, args.hook_dir)
    for bs, s in hooks.items():
        print(f"  bs={bs:3d}  {s*1e6:8.1f} us/tok")

    # Tally
    print("\n" + "=" * 78)
    print(f"  {'bs':>4s} {'baseline_us':>12s} {'plugin_us':>12s} {'hooks_us':>12s} "
          f"{'plugin_pct':>10s} {'hooks_pct':>10s}")
    rows = []
    rc = 0
    for bs in args.batches:
        b = base[bs]; pl = plug[bs]; hk = hooks[bs]
        plug_pct  = 100.0 * (pl - b) / b
        hooks_pct = 100.0 * (hk - b) / b
        verdict = "PASS" if hooks_pct <= 5.0 else "FAIL"
        if verdict == "FAIL":
            rc = 1
        print(f"  {bs:>4d} {b*1e6:>10.1f}us {pl*1e6:>10.1f}us {hk*1e6:>10.1f}us "
              f"{plug_pct:>+8.2f}% {hooks_pct:>+8.2f}%  {verdict}")
        rows.append({
            "bs":         bs,
            "baseline_s_per_tok": b,
            "plugin_s_per_tok":   pl,
            "hooks_s_per_tok":    hk,
            "plugin_overhead_pct": plug_pct,
            "hooks_overhead_pct":  hooks_pct,
            "verdict": verdict,
        })
    print("=" * 78)

    if args.output:
        save_csv(rows, args.output)
        print(f"[idle_tax] CSV → {args.output}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
