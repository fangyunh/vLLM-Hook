"""Step 0.5 — Mode matrix: which CUDAGraphModes survive class-level wrap?

Re-runs Step 0's counter-op injection under each targeted CUDAGraphMode
and tabulates the result. Settles claim C5 per-mode and locks in which
modes are shippable before any mode-specific code is written.

A PIECEWISE pass is mandatory (the plan ships at minimum on this mode).
FULL_DECODE_ONLY is the throughput mode; if it passes, the plan ships
both. FULL_AND_PIECEWISE is checked but not required.

Usage:
    python tests/cuda_graph/step0_5_mode_matrix.py \\
        --model google/gemma-3-4b-it
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
STEP0 = HERE / "step0_injection.py"


def run_one(model: str, mode: str, num_tokens: int) -> tuple[int, str]:
    """Invoke step0_injection.py in a clean subprocess so each mode gets a
    fresh vLLM engine. Returns (rc, last 30 lines of stdout)."""
    print(f"\n{'=' * 70}")
    print(f"[step0.5] MODE = {mode}")
    print('=' * 70)
    proc = subprocess.run(
        [
            sys.executable, str(STEP0),
            "--model", model,
            "--num-decode-tokens", str(num_tokens),
            "--cudagraph-mode", mode,
        ],
        capture_output=True,
        text=True,
    )
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
    print(tail)
    return proc.returncode, tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--num-decode-tokens", type=int, default=32)
    args = parser.parse_args()

    MODES = ["PIECEWISE", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"]
    results: dict[str, int] = {}

    for mode in MODES:
        rc, _ = run_one(args.model, mode, args.num_decode_tokens)
        results[mode] = rc

    print("\n" + "=" * 70)
    print("[step0.5] MODE MATRIX SUMMARY")
    print("=" * 70)
    for mode, rc in results.items():
        verdict = {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE"}.get(rc, "?")
        required = " (required for ship)" if mode == "PIECEWISE" else ""
        print(f"  {mode:25s} -> {verdict}{required}")

    if results.get("PIECEWISE") != 0:
        print("\n[step0.5] GOAL-GATE FIRES: PIECEWISE failed — no pure-plugin")
        print("          path exists on this torch/vLLM version.")
        return 1

    if results.get("FULL_DECODE_ONLY") == 0:
        print("\n[step0.5] BOTH MODES SHIPPABLE — proceed with dual-mode plan.")
        return 0
    else:
        print("\n[step0.5] PIECEWISE-ONLY shippable. Document the throughput")
        print("          ceiling; FULL_DECODE_ONLY is deferred work.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
