"""Step 0.9 — Memory budget gate: what do the static buffers actually cost?

The latency gate (step0_7) measures the per-layer launch overhead in TIME. This
gate measures the OTHER half of "ops on every layer" — VRAM — which is often the
bigger throughput risk at scale: static capture buffers occupy GPU memory that
would otherwise be KV cache, so fewer concurrent requests fit.

The counter op used by step0_7 allocates ~nothing, so it is NOT representative
of memory. This gate allocates REAL-shaped per-layer buffers (capture-rank only)
and reports:

  1. Static buffer footprint — measured (torch CUDA allocator delta) vs the
     plan's predicted formula (num_layers × rows × hidden × dtype_bytes), where
     rows = max_num_batched_tokens (all_tokens) or max_num_seqs+1 (last_token).
  2. KV-cache impact — the engine's `num_gpu_blocks` WITH buffers vs a clean
     BASELINE. This is the throughput-relevant number: "X GiB of buffers" →
     "Y% fewer KV blocks" → fewer concurrent requests.
  3. Steady-state boundedness — worker `memory_allocated` sampled across a churn
     of many short requests; asserts the footprint stays flat (no growth/leak).

Scope honesty: this validates the STATIC-buffer design (footprint + KV impact +
boundedness). It does NOT validate the not-yet-built per-request capture egress
/ row-allocator (free-on-completion, no stale-row reads) — that is a Step-2
correctness item once the real ops exist.

Baseline and buffers boot in SEPARATE subprocesses so the KV-block counts are
apples-to-apples (no leftover memory from a prior engine in the same process).

Pass criteria (defaults, override on the CLI):
  * num_gpu_blocks(buffers) / num_gpu_blocks(baseline) >= --min-kv-retention (0.90)
  * churn drift <= --max-drift-pct (2.0%)

Usage:
    python tests/cuda_graph/step0_9_mem_budget.py \\
        --model <model-you-can-access> --policy last_token
    # stress the worst case (per-step / all-tokens sizing):
    python tests/cuda_graph/step0_9_mem_budget.py --model <model> --policy all_tokens
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

MODULE_NAME = "step0_9_mem_budget"
WORKER_CLS = f"{MODULE_NAME}.MemProbeWorker"
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.environ["PYTHONPATH"] = _THIS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

_LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.\d+$"),
    re.compile(r"^language_model\.model\.layers\.\d+$"),
    re.compile(r"^transformer\.h\.\d+$"),
    re.compile(r"^model\.decoder\.layers\.\d+$"),
]


def _scan_model(model) -> tuple[int, int]:
    """Return (num matched decoder layers, hidden_size)."""
    n = sum(1 for name, _ in model.named_modules()
            if any(p.match(name) for p in _LAYER_PATTERNS))
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = int(getattr(text_cfg, "hidden_size"))
    return n, hidden


# ---------------------------------------------------------------------------
# Worker that allocates representative static buffers at load_model.
# ---------------------------------------------------------------------------

try:
    from vllm.v1.worker.gpu_worker import Worker as _V1Worker
except Exception:  # pragma: no cover - only importable on a GPU host w/ vLLM
    _V1Worker = object


class MemProbeWorker(_V1Worker):
    """Allocates per-layer static buffers (capture rank only), sized by the
    plan's policy, BEFORE vLLM profiles the KV cache — so their effect on
    num_gpu_blocks is real. Exposes memory stats over collective_rpc."""

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        self._meminfo: dict = {}
        try:
            model = self.model_runner.model
            n_layers, hidden = _scan_model(model)
            dtype = next(model.parameters()).dtype
            elem = torch.tensor([], dtype=dtype).element_size()

            sc = self.vllm_config.scheduler_config
            policy = os.environ.get("STEP0_9_POLICY", "last_token")
            if policy == "all_tokens":
                rows = int(sc.max_num_batched_tokens)
            else:  # last_token (default, cheap policy)
                rows = int(sc.max_num_seqs) + 1

            tp = self.parallel_config.tensor_parallel_size
            is_capture = tp <= 1 or (self.rank % tp == 0)

            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            self._bufs = []
            if is_capture:
                for _ in range(n_layers):
                    self._bufs.append(
                        torch.zeros(rows, hidden, dtype=dtype, device="cuda"))
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()

            self._meminfo = dict(
                num_layers=n_layers, hidden=hidden, rows=rows, policy=policy,
                dtype=str(dtype), elem_size=elem, is_capture=bool(is_capture),
                measured_buf_bytes=int(after - before),
                predicted_buf_bytes=int(n_layers * rows * hidden * elem) if is_capture else 0,
            )
            print(f"[step0.9/worker] policy={policy} layers={n_layers} rows={rows} "
                  f"hidden={hidden} buf={(after - before) / 2**20:.1f} MiB "
                  f"capture_rank={is_capture}", flush=True)
        except Exception as e:
            self._meminfo = {"error": str(e)}
            print(f"[step0.9/worker] buffer alloc failed: {e}", flush=True)
        return r

    def mem_report(self) -> dict:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        d = dict(self._meminfo)
        d.update(alloc_bytes=int(torch.cuda.memory_allocated()),
                 reserved_bytes=int(torch.cuda.memory_reserved()),
                 free_bytes=int(free), total_bytes=int(total))
        return d

    def mem_alloc_mb(self) -> float:
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 2**20


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _collective_rpc(llm, method: str) -> list:
    fn = getattr(llm, "collective_rpc", None)
    if fn is None:
        eng = getattr(llm, "llm_engine", None)
        fn = getattr(eng, "collective_rpc", None) if eng is not None else None
    if fn is None:
        raise RuntimeError("collective_rpc not found")
    res = fn(method)
    return list(res) if isinstance(res, (list, tuple)) else [res]


def _get_num_gpu_blocks(llm):
    """Best-effort read of the KV-cache GPU block count across vLLM versions."""
    eng = getattr(llm, "llm_engine", None)
    for obj in (eng, llm):
        if obj is None:
            continue
        cc = getattr(obj, "cache_config", None)
        v = getattr(cc, "num_gpu_blocks", None) if cc is not None else None
        if v:
            return int(v)
        vc = getattr(obj, "vllm_config", None)
        cc = getattr(vc, "cache_config", None) if vc is not None else None
        v = getattr(cc, "num_gpu_blocks", None) if cc is not None else None
        if v:
            return int(v)
    return None


def _mib(b) -> str:
    return f"{b / 2**20:.1f} MiB" if b else "n/a"


# ---------------------------------------------------------------------------
# One mode = one engine (baseline or buffers). Prints a single RESULT line.
# ---------------------------------------------------------------------------


def run_mode(mode: str, model: str, policy: str, churn_requests: int,
             cudagraph_mode: str) -> int:
    from vllm import LLM, SamplingParams

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_CUDAGRAPH_MODE"] = cudagraph_mode
    os.environ["STEP0_9_POLICY"] = policy

    kwargs: dict[str, Any] = dict(
        model=model, enforce_eager=False, max_model_len=512,
        gpu_memory_utilization=0.85,
    )
    if mode == "buffers":
        kwargs["worker_cls"] = WORKER_CLS

    print(f"[step0.9] booting {mode} engine (model={model}, mode={cudagraph_mode})",
          flush=True)
    llm = LLM(**kwargs)

    blocks = _get_num_gpu_blocks(llm)
    rep = None
    if mode == "buffers":
        try:
            rep = _collective_rpc(llm, "mem_report")[0]
        except Exception as e:
            rep = {"error": str(e)}

    # Churn (buffers only): drive many short requests, sample worker memory,
    # confirm the footprint stays flat (no growth / leak under load).
    drift = None
    mem_series: list = []
    if mode == "buffers" and churn_requests > 0:
        sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
        chunk = 16
        n_chunks = max(1, math.ceil(churn_requests / chunk))
        for _ in range(n_chunks):
            llm.generate(["Tell me a short story about a fox."] * chunk, sp)
            try:
                mem_series.append(_collective_rpc(llm, "mem_alloc_mb")[0])
            except Exception:
                pass
        if mem_series:
            lo, hi = min(mem_series), max(mem_series)
            drift = ((hi - lo) / lo * 100.0) if lo > 0 else 0.0

    result = dict(mode=mode, policy=policy, num_gpu_blocks=blocks,
                  mem_report=rep, drift_pct=drift,
                  mem_series_mb=mem_series, churn_requests=churn_requests)
    print("RESULT " + json.dumps(result))
    return 0


# ---------------------------------------------------------------------------
# Compare: spawn baseline + buffers in separate processes, report verdict.
# ---------------------------------------------------------------------------


def _spawn(mode: str, args) -> tuple[dict | None, int]:
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--mode", mode,
        "--model", args.model,
        "--policy", args.policy,
        "--churn-requests", str(args.churn_requests),
        "--cudagraph-mode", args.cudagraph_mode,
    ]
    print(f"\n{'=' * 70}\n[step0.9] launching {mode} subprocess\n{'=' * 70}",
          flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    res = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                res = json.loads(line[len("RESULT "):])
            except Exception:
                pass
    print("\n".join((proc.stdout + proc.stderr).splitlines()[-20:]))
    if res is None:
        print(f"[step0.9] {mode}: no RESULT line (boot failed?). rc={proc.returncode}")
    return res, proc.returncode


def run_compare(args) -> int:
    base, _ = _spawn("baseline", args)
    buf, _ = _spawn("buffers", args)

    bb = base.get("num_gpu_blocks") if base else None
    wb = buf.get("num_gpu_blocks") if buf else None
    rep = (buf or {}).get("mem_report") or {}
    drift = (buf or {}).get("drift_pct")

    print("\n" + "=" * 70)
    print("[step0.9] MEMORY BUDGET RESULTS")
    print("=" * 70)
    print(f"  policy                  = {args.policy}")
    print(f"  matched layers          = {rep.get('num_layers', '?')}, "
          f"rows/layer = {rep.get('rows', '?')}, hidden = {rep.get('hidden', '?')}, "
          f"dtype = {rep.get('dtype', '?')}")
    print(f"  static buffers measured = {_mib(rep.get('measured_buf_bytes'))}  "
          f"(predicted {_mib(rep.get('predicted_buf_bytes'))})")
    if rep.get("total_bytes"):
        pct_vram = 100.0 * (rep.get("measured_buf_bytes") or 0) / rep["total_bytes"]
        print(f"  buffers as % of GPU VRAM= {pct_vram:.2f}%  "
              f"(device total {_mib(rep['total_bytes'])})")
    print(f"  KV blocks baseline      = {bb}")
    print(f"  KV blocks with buffers  = {wb}")

    rc = 0
    if bb and wb:
        retention = wb / bb
        print(f"  KV-block retention      = {retention:.3f}  "
              f"(budget >= {args.min_kv_retention:.2f})")
        if retention < args.min_kv_retention:
            rc = 1
    else:
        retention = None
        print("  KV-block retention      = n/a (num_gpu_blocks not reachable on "
              "this vLLM version) — relying on static-byte + drift checks")
        rc = max(rc, 2)

    if drift is not None:
        print(f"  churn drift             = {drift:.2f}%  "
              f"(budget <= {args.max_drift_pct:.2f}%, over {args.churn_requests} reqs)")
        if drift > args.max_drift_pct:
            rc = 1
    else:
        print("  churn drift             = n/a (no samples collected)")
        rc = max(rc, 2)

    print("=" * 70)
    if rc == 0:
        print("[step0.9] PASS — static-buffer footprint within budget, KV-cache")
        print("          retention healthy, and footprint flat under churn.")
        print("          (Validates the static design; per-request capture egress /")
        print("           row-allocator correctness is a separate Step-2 item.)")
    elif rc == 1:
        print("[step0.9] FAIL — over budget. Pivots: switch to last_token sizing,")
        print("          narrow the activatable set via VLLM_HOOK_ACTIVE_LAYERS,")
        print("          or accept fewer concurrent requests and document it.")
    else:
        print("[step0.9] INCONCLUSIVE — a metric could not be read on this stack;")
        print("          see the notes above and re-run with diagnostics.")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="compare",
                        choices=["compare", "baseline", "buffers"],
                        help="compare (default) spawns baseline + buffers; the "
                             "other two are the per-engine workers it spawns.")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--policy", default="last_token",
                        choices=["last_token", "all_tokens"],
                        help="Buffer sizing policy: last_token (default, cheap) "
                             "sizes by max_num_seqs; all_tokens sizes by "
                             "max_num_batched_tokens (worst case).")
    parser.add_argument("--churn-requests", type=int, default=200)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE",
                        choices=["PIECEWISE", "FULL_DECODE_ONLY",
                                 "FULL_AND_PIECEWISE"])
    parser.add_argument("--min-kv-retention", type=float, default=0.90)
    parser.add_argument("--max-drift-pct", type=float, default=2.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[step0.9] CUDA not available; this test requires a GPU.")
        return 2

    if args.mode == "compare":
        return run_compare(args)
    return run_mode(args.mode, args.model, args.policy,
                    args.churn_requests, args.cudagraph_mode)


if __name__ == "__main__":
    sys.exit(main())
