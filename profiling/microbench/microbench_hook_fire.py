"""microbench_hook_fire — isolate per-fire CPU cost of probe workers.

Strategy: build a HookLLM in probe_hidden_states mode, run the same
fixed batch twice — once with ``use_hook=False`` (closure not installed)
and once with ``output_hidden_states=True`` (closure fires every layer
every step). Subtract to get a hook-only delta.

The profiler's ``hook.fire.hs`` counter and ``captured.bytes.hs`` gauge
break the delta down further (per-fire bytes accumulated, fire count).
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

PROFILING_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT  = PROFILING_DIR.parent
sys.path.insert(0, str(PROFILING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

from _common import gpu_used_mb, profile_snapshot, reset_profiler  # noqa: E402


def _measure(llm, prompts: List[str], use_hook: bool, max_tokens: int,
             reps: int) -> Dict[str, float]:
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    # warmup
    _ = llm.generate(prompts, sp, use_hook=use_hook, save_to_disk=False)
    reset_profiler()
    lats = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sp, use_hook=use_hook, save_to_disk=False)
        lats.append(time.perf_counter() - t0)
    snap = profile_snapshot() or {}
    return {
        "mean_s":   statistics.mean(lats),
        "median_s": statistics.median(lats),
        "min_s":    min(lats),
        "max_s":    max(lats),
        "peak_gpu_mb": gpu_used_mb(),
        "n_hook_fires": (snap.get("counters") or {}).get("hook.fire.hs", 0),
        "captured_bytes_total":
            (snap.get("gauges", {}).get("captured.bytes.hs") or {}).get("sum", 0),
        "cpu_xfer_ms_mean":
            (snap.get("timers", {}).get("worker.cpu_transfer.hs") or {}).get("mean", 0),
        "compress_ms_mean":
            (snap.get("timers", {}).get("worker.compress.hs") or {}).get("mean", 0),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--config", default=None,
                   help="Hidden-states config JSON (layers + mode).")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--hook-dir", default="/dev/shm/vllm_hook")
    p.add_argument("--gpu-mem-util", type=float, default=0.4)
    args = p.parse_args()

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_HOOK_PROFILE", "1")

    import torch
    from vllm_hook_plugins import register_plugins
    from vllm_hook_plugins.hook_llm import HookLLM
    register_plugins()

    llm = HookLLM(
        model=args.model,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=args.config,
        hook_dir=args.hook_dir,
        dtype=torch.float16,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_hook=True,
    )

    from _common import synth_prompts
    prompts = synth_prompts(args.prompt_len, llm.tokenizer, args.batch)

    print(f"[microbench] baseline (use_hook=False)")
    baseline = _measure(llm, prompts, use_hook=False, max_tokens=args.max_tokens,
                        reps=args.reps)
    print(f"  mean={baseline['mean_s']*1000:.2f}ms  "
          f"median={baseline['median_s']*1000:.2f}ms")

    print(f"[microbench] hooks active (output_hidden_states=True)")
    hooked = _measure(llm, prompts, use_hook=True, max_tokens=args.max_tokens,
                      reps=args.reps)
    print(f"  mean={hooked['mean_s']*1000:.2f}ms  "
          f"median={hooked['median_s']*1000:.2f}ms")

    delta_ms = (hooked["median_s"] - baseline["median_s"]) * 1000.0
    fires = hooked["n_hook_fires"]
    per_fire_us = (delta_ms * 1000 / fires) if fires else 0.0
    print()
    print(f"delta_median   = {delta_ms:+.2f}ms")
    print(f"hook_fires     = {fires}")
    print(f"per_fire       = {per_fire_us:.2f}us")
    print(f"captured_bytes = {hooked['captured_bytes_total']:.0f}")
    print(f"cpu_xfer_mean  = {hooked['cpu_xfer_ms_mean']:.2f}ms")
    print(f"compress_mean  = {hooked['compress_ms_mean']:.2f}ms")

    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
