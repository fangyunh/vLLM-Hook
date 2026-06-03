"""bench_grid — resumable cartesian sweep of cells over the §1 metrics.

Three plans:
  - plan-quick    R0..R5 × {16, 256} prompt_len × {1, 16} batch × {1, 32} max_tokens
                  single storage variant per worker (~150 cells, ≤30 min on A100)
  - plan-storage  HS worker × all 5 storage variants × all modes × prompt_len × batch
                  (~400 cells, ~2 hr)
  - plan-full     all axes (~3k cells, overnight; resume-able)

The first deliverable for the advisor (plan.html §7) is the R0..R5 table
from plan-quick. Run it, point summarize.py at the CSV, and the headline
table drops out.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROFILING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = PROFILING_DIR.parent
sys.path.insert(0, str(PROFILING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

from _common import (  # noqa: E402
    DEFAULT_HOOK_DIR,
    STORAGE_VARIANTS,
    append_jsonl,
    save_csv,
)
from profile_one_run import ROW_TO_WORKER, run_cell  # noqa: E402


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def _plan_quick(args) -> List[Dict[str, Any]]:
    """The R0..R6 seven-row headline. v0.2.0 hooks each get both modes as a
    row so QK and HS are directly comparable.

    Plus optional v0.1.0 peer rows when ``args.include_v010`` is set. The
    v010 rows use the vendored v0.1.0 workers under profiling/peers/v010/
    for a direct hook-vs-hook comparison. Prefix caching is forced off
    inside the v010 engine, since v0.1.0 has no prefix-recon path.

    Row map:
        R0  baseline                        (rpc / last_token, but mode unused)
        R1  plugin_idle                     (rpc / last_token, mode unused)
        R2  probe_hook_qk        last_token
        R3  probe_hook_qk        all_tokens
        R4  probe_hidden_states  last_token   ← new in 2026-06-03 patch;
        R5  probe_hidden_states  all_tokens     symmetric with QK above
        R6  steer_hook_act       (in-place, mode unused)
    """
    # (row, mode) pairs in display order — R0..R6.
    rows: List[tuple] = [
        ("baseline",            "last_token"),
        ("plugin_idle",         "last_token"),
        ("probe_hook_qk",       "last_token"),
        ("probe_hook_qk",       "all_tokens"),
        ("probe_hidden_states", "last_token"),   # R4 — typical-use HS
        ("probe_hidden_states", "all_tokens"),   # R5 — worst-case HS
        ("steer_hook_act",      "last_token"),
    ]
    if getattr(args, "include_v010", False):
        rows.extend([
            ("probe_hook_qk_v010",       "last_token"),
            ("probe_hook_qk_v010",       "all_tokens"),
            ("probe_hidden_states_v010", "last_token"),
            ("probe_hidden_states_v010", "all_tokens"),
            ("steer_hook_act_v010",      "last_token"),
        ])

    # Storage-variant overrides per row. v010 rows use the same storage
    # variants v0.1.0 actually supported in Numerical_Analysis so the
    # apples-to-apples comparison is on the same I/O path.
    overrides = {
        "baseline":                  {"storage_variant": "rpc"},
        "plugin_idle":               {"storage_variant": "rpc"},
        "probe_hook_qk":             {"storage_variant": "disk-st-async"},
        "probe_hidden_states":       {"storage_variant": "disk-st-async"},
        "steer_hook_act":            {"storage_variant": "rpc"},
        "probe_hook_qk_v010":        {"storage_variant": "disk-st-async"},
        "probe_hidden_states_v010":  {"storage_variant": "disk-st-async"},
        "steer_hook_act_v010":       {"storage_variant": "rpc"},
    }

    plan: List[Dict[str, Any]] = []
    for row, mode in rows:
        for prompt_len in args.prompt_lens:
            for batch in args.batch_sizes:
                for max_tok in args.max_tokens:
                    cell = dict(
                        row=row,
                        model=args.model, dtype=args.dtype, tp_size=args.tp_size,
                        prompt_len=prompt_len, batch_size=batch, max_tokens=max_tok,
                        n_layers=args.n_layers,
                        base_config=args.base_config,
                        hook_dir=args.hook_dir,
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_mem_util,
                        enable_prefix_caching=args.prefix_caching,
                        enforce_eager=(not args.no_enforce_eager),
                        reps=args.reps, warmup=args.warmup,
                        hooks_on=args.hooks_on,
                        mode=mode,
                    )
                    cell.update(overrides[row])
                    plan.append(cell)
    return plan


def _plan_storage(args) -> List[Dict[str, Any]]:
    """HS worker × all storage variants × modes × prompt_len × batch."""
    plan = []
    for variant in STORAGE_VARIANTS:
        spec = STORAGE_VARIANTS[variant]
        for mode in ("last_token", "all_tokens"):
            if spec["last_token_only"] and mode != "last_token":
                continue
            for prompt_len in args.prompt_lens:
                for batch in args.batch_sizes:
                    plan.append(dict(
                        row="probe_hidden_states",
                        model=args.model, dtype=args.dtype, tp_size=args.tp_size,
                        storage_variant=variant, mode=mode,
                        prompt_len=prompt_len, batch_size=batch,
                        max_tokens=1,
                        n_layers=args.n_layers,
                        base_config=args.base_config,
                        hook_dir=args.hook_dir,
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_mem_util,
                        enable_prefix_caching=args.prefix_caching,
                        enforce_eager=(not args.no_enforce_eager),
                        reps=args.reps, warmup=args.warmup,
                        hooks_on=args.hooks_on,
                    ))
    return plan


def _plan_full(args) -> List[Dict[str, Any]]:
    """Every axis. Big. Resume from --start-from after a crash."""
    plan: List[Dict[str, Any]] = []
    for row in ("baseline", "plugin_idle", "probe_hidden_states",
                "probe_hook_qk", "steer_hook_act"):
        valid_variants = ["rpc"] if row in ("baseline", "plugin_idle", "steer_hook_act") \
            else list(STORAGE_VARIANTS)
        for variant in valid_variants:
            spec = STORAGE_VARIANTS.get(variant, {"last_token_only": False})
            for mode in ("last_token", "all_tokens"):
                if spec.get("last_token_only", False) and mode != "last_token":
                    continue
                for hooks_on in ("prefill", "decode", "both"):
                    if row in ("baseline", "plugin_idle"):
                        hooks_on = "prefill"  # irrelevant; collapse
                    for prompt_len in args.prompt_lens:
                        for batch in args.batch_sizes:
                            for max_tok in args.max_tokens:
                                plan.append(dict(
                                    row=row,
                                    model=args.model, dtype=args.dtype, tp_size=args.tp_size,
                                    storage_variant=variant, mode=mode, hooks_on=hooks_on,
                                    prompt_len=prompt_len, batch_size=batch, max_tokens=max_tok,
                                    n_layers=args.n_layers,
                                    base_config=args.base_config,
                                    hook_dir=args.hook_dir,
                                    max_model_len=args.max_model_len,
                                    gpu_memory_utilization=args.gpu_mem_util,
                                    enable_prefix_caching=args.prefix_caching,
                                    enforce_eager=(not args.no_enforce_eager),
                                    reps=args.reps, warmup=args.warmup,
                                ))
                    if row in ("baseline", "plugin_idle"):
                        break  # already collapsed; don't iterate hooks_on
    # Deduplicate identical cells that result from the collapse.
    uniq: Dict[str, Dict[str, Any]] = {}
    for c in plan:
        key = json.dumps(c, sort_keys=True, default=str)
        uniq[key] = c
    return list(uniq.values())


PLANS = {
    "plan-quick":   _plan_quick,
    "plan-storage": _plan_storage,
    "plan-full":    _plan_full,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_plan(plan: List[Dict[str, Any]], *, output_csv: str,
             output_jsonl: Optional[str], start_from: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if start_from > 1 and os.path.exists(output_csv):
        with open(output_csv) as f:
            rows = list(csv.DictReader(f))
        print(f"[bench_grid] resume: loaded {len(rows)} prior rows; "
              f"starting cell {start_from}")

    for i, cell in enumerate(plan, start=1):
        if i < start_from:
            continue
        t0 = time.time()
        print(f"[bench_grid] {i}/{len(plan)}  row={cell['row']}  "
              f"storage={cell.get('storage_variant')}  mode={cell.get('mode')}  "
              f"prompt={cell['prompt_len']}  batch={cell['batch_size']}  "
              f"max_tok={cell['max_tokens']}")
        try:
            result = run_cell(cell)
        except Exception as exc:
            print(f"[bench_grid] CELL {i} FAILED: {type(exc).__name__}: {exc}")
            result = {"row": cell["row"], "status": "fail", "error": str(exc)}
        result["plan_idx"] = i
        result["cell_wall_s"] = round(time.time() - t0, 3)
        rows.append(result)

        if output_jsonl:
            append_jsonl(result, output_jsonl)
        # Checkpoint after each cell.
        save_csv(rows, output_csv)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=list(PLANS), default="plan-quick")
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 256])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    p.add_argument("--max-tokens", type=int, nargs="+", default=[1, 32])
    p.add_argument("--n-layers", default=None,
                   help="Comma-separated layer indices, or omit for all.")
    p.add_argument("--hooks-on", default="prefill",
                   choices=["prefill", "decode", "both"])
    p.add_argument("--base-config", default=None)
    p.add_argument("--hook-dir", default=DEFAULT_HOOK_DIR)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.5)
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--no-enforce-eager", action="store_true")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output",      default=None,
                   help="CSV path. Default: results/<plan>-<date>.csv")
    p.add_argument("--output-jsonl", default=None,
                   help="JSONL path. Default: same basename as CSV.")
    p.add_argument("--start-from",  type=int, default=1)
    p.add_argument("--include-v010", action="store_true",
                   help="Add R6..R8 = v0.1.0 peer rows to plan-quick for a "
                        "direct v0.2.0-vs-v0.1.0 comparison (probe_hidden_states, "
                        "probe_hook_qk, steer_hook_act each get a _v010 sibling). "
                        "v0.1.0 prefix caching is forced off inside the engine.")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print the plan without running any cells.")
    args = p.parse_args()

    if args.n_layers is not None and args.n_layers != "all":
        args.n_layers = [int(x) for x in args.n_layers.split(",") if x.strip()]
    elif args.n_layers == "all":
        args.n_layers = None

    if args.output is None:
        stamp = time.strftime("%Y%m%d-%H%M")
        args.output = str(PROFILING_DIR / "results" / f"{args.plan}-{stamp}.csv")
    if args.output_jsonl is None:
        args.output_jsonl = str(Path(args.output).with_suffix(".jsonl"))

    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The harness itself wants profiling on so per-cell snapshots get flattened
    # into the CSV. Explicit env override (=0) still wins.
    os.environ.setdefault("VLLM_HOOK_PROFILE", "1")

    plan = PLANS[args.plan](args)
    print(f"[bench_grid] plan={args.plan}  cells={len(plan)}  "
          f"output={args.output}")

    if args.dry_run:
        for i, cell in enumerate(plan, start=1):
            print(f"  cell {i}: {cell['row']}  "
                  f"storage={cell.get('storage_variant')}  "
                  f"mode={cell.get('mode')}  "
                  f"prompt={cell['prompt_len']}  batch={cell['batch_size']}  "
                  f"max_tok={cell['max_tokens']}")
        return 0

    rows = run_plan(plan, output_csv=args.output, output_jsonl=args.output_jsonl,
                    start_from=args.start_from)
    print(f"[bench_grid] done. {len(rows)} rows → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
