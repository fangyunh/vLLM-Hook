"""Step 0.7 — Idle-tax budget: how much does per-layer wrap-launch cost?

This is the throughput floor for the all-layers-wrapped design. With the
Step 0 counter-op installed on every decoder layer (do_*=False — kernel
just increments a counter, the smallest possible body), we measure
decode token-time delta vs an unhooked baseline. Per-layer overhead is
the metric.

Pass: per-layer overhead <= 3 us; aggregate idle overhead <= 5% at bs=1
on the development model. Report the (batch × ?) matrix.

Failure pivots:
  - per-layer > 5 us: revisit conditional-node future-work, or restrict
    via VLLM_HOOK_ACTIVE_LAYERS (deploy-time list of activatable layers).
  - aggregate > 10% at bs=1: the all-layers-wrapped design is not
    competitive vs eager-island at small batch — document explicitly.

Usage:
    python tests/cuda_graph/step0_7_idle_tax.py \\
        --model google/gemma-3-4b-it \\
        --num-decode-tokens 128 \\
        --batches 1 8 64
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List

import torch

# Keep the vllm-hook plugin out: it forces enforce_eager=True, which would
# disable CUDA graphs and make the "wrapped" timings meaningless. (setdefault
# lets a user override.) Set before vLLM is imported / before subprocesses.
os.environ.setdefault("VLLM_PLUGINS", "")


def measure(llm, prompts: List[str], n_decode: int) -> float:
    """Return median seconds per decode token across the batch."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=n_decode, ignore_eos=True)

    # Warmup pass — let any first-call JIT settle.
    _ = llm.generate(prompts, sp)
    torch.cuda.synchronize()

    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        timings.append(dt / max(1, total_tokens))
    return statistics.median(timings)


def run(model: str, n_decode: int, batches: List[int], with_wrap: bool) -> dict:
    """Boot vLLM (optionally with Step 0's counter wrap) and time decode."""
    from step0_injection import boot_llm, realized_mode

    llm = boot_llm(model, "PIECEWISE")
    print(f"[step0.7] realized cudagraph_mode = {realized_mode(llm)}", flush=True)

    num_layers = 0
    if with_wrap:
        # Inject the Step 0 wrap on every matched decoder-layer class.
        from step0_injection import (
            install_class_wrap, register_counter_op, _get_counter
        )
        register_counter_op()
        inner = llm.llm_engine.model_executor.driver_worker.model_runner.model
        install_class_wrap(inner)

        import re
        patterns = [
            re.compile(r"^model\.layers\.\d+$"),
            re.compile(r"^transformer\.h\.\d+$"),
            re.compile(r"^model\.decoder\.layers\.\d+$"),
        ]
        num_layers = sum(1 for n, _ in inner.named_modules()
                         if any(p.match(n) for p in patterns))

    out = {}
    for bs in batches:
        prompts = ["The quick brown fox jumps over the"] * bs
        s_per_tok = measure(llm, prompts, n_decode)
        out[bs] = s_per_tok
        label = "WRAPPED " if with_wrap else "BASELINE"
        print(f"[step0.7] {label} bs={bs:3d}  "
              f"{s_per_tok * 1e6:8.1f} us/token "
              f"({1.0 / s_per_tok:8.1f} tok/s)")

    # Device-global VRAM in use after the run (model + KV cache + any wrap
    # buffers). mem_get_info() reflects the whole device, so it is valid even
    # when the worker runs in a separate process.
    # NOTE: baseline and wrapped boot sequentially in ONE process here, so this
    # is a SANITY readout, not an apples-to-apples buffer cost. The counter wrap
    # also allocates ~nothing. For the authoritative memory gate — separate
    # processes, real-shaped buffers, KV-block accounting — use
    # step0_9_mem_budget.py.
    torch.cuda.synchronize()
    _free, _total = torch.cuda.mem_get_info()
    used_gib = (_total - _free) / 2**30

    return {"per_bs": out, "num_layers": num_layers, "used_gib": used_gib}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--num-decode-tokens", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[step0.7] CUDA not available.")
        return 2

    print("[step0.7] === BASELINE (unhooked) ===")
    base = run(args.model, args.num_decode_tokens, args.batches, with_wrap=False)

    print("\n[step0.7] === WRAPPED (counter op on every layer, idle) ===")
    wrap = run(args.model, args.num_decode_tokens, args.batches, with_wrap=True)

    # Guard: confirm the wrap actually executed. If the counter is still 0,
    # the op never ran (class-level wrap bypassed, or installed too late), so
    # the "wrapped" timings above measure an ABSENT op, not an idle one — the
    # idle-tax numbers would be meaningless and could read as a false PASS.
    # Settle injection first: step0_injection.py (driver install) or
    # step0_8_plugin_install.py (worker-process install).
    from step0_injection import _get_counter
    torch.cuda.synchronize()
    fired = int(_get_counter().item())
    if fired == 0:
        print("\n[step0.7] FAIL — counter is 0 after the wrapped run.")
        print("          The op never executed, so the 'wrapped' timings do NOT")
        print("          include it; the idle-tax numbers are meaningless. Run")
        print("          step0_injection.py / step0_8_plugin_install.py first to")
        print("          confirm injection lands in the captured graph.")
        return 1
    print(f"[step0.7] wrap fired {fired} times during the wrapped run "
          f"(op confirmed present).")

    b_used = base.get("used_gib")
    w_used = wrap.get("used_gib")
    if b_used is not None and w_used is not None:
        print(f"[step0.7] VRAM (device-global, post-run): baseline {b_used:.2f} GiB "
              f" wrapped {w_used:.2f} GiB  delta {w_used - b_used:+.2f} GiB")
        print("          (sanity readout — sequential single-process boots + a "
              "zero-byte counter wrap;")
        print("           use step0_9_mem_budget.py for the authoritative memory gate.)")

    num_layers = wrap["num_layers"]
    print("\n" + "=" * 70)
    print("[step0.7] IDLE-TAX RESULTS")
    print("=" * 70)
    print(f"  num matched layers: {num_layers}")
    print(f"  {'bs':>4s}  {'baseline':>10s}  {'wrapped':>10s}  "
          f"{'delta':>9s}  {'per-layer':>10s}  {'%reg':>6s}  {'verdict':>10s}")
    rc = 0
    for bs in args.batches:
        b = base["per_bs"][bs]
        w = wrap["per_bs"][bs]
        delta = w - b
        per_layer_us = (delta / max(1, num_layers)) * 1e6
        pct = 100.0 * delta / b
        verdict = "PASS" if per_layer_us <= 3.0 and pct <= 5.0 else "FAIL"
        if verdict == "FAIL":
            rc = 1
        print(f"  {bs:>4d}  {b * 1e6:>8.1f}us  {w * 1e6:>8.1f}us  "
              f"{delta * 1e6:>+7.1f}us  {per_layer_us:>+8.2f}us  "
              f"{pct:>+5.2f}%  {verdict:>10s}")

    print("=" * 70)
    if rc == 0:
        print("[step0.7] PASS — idle-tax within budget on every cell.")
        print("          Proceed to Step 1 (transparency).")
    else:
        print("[step0.7] FAIL — at least one cell over budget.")
        print("          Pivot: introduce VLLM_HOOK_ACTIVE_LAYERS to narrow")
        print("                 the activatable layer set at deploy time, OR")
        print("                 accept the regression and document the floor.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
