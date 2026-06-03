"""plot_results — paper-grade figures from the four CSVs ``submit_all.sh`` produces.

Reads any combination of ``quick-*.csv``, ``storage-*.csv``, ``idle-*.csv``,
``serve-*.csv`` from ``profiling/results/`` and emits PNG figures under
``profiling/results/plots/``.

Designed to be self-contained: stdlib + matplotlib, headless-safe (Agg
backend), no pandas, no seaborn. Run from the project root:

    python profiling/analyze/plot_results.py
    python profiling/analyze/plot_results.py --results-dir profiling/results \
                                              --out-dir   profiling/results/plots

Each figure is independent — if a CSV is missing, the corresponding figure
is skipped with a one-line message; the script never crashes.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _read(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v: Any) -> Optional[float]:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _latest(pattern: str, results_dir: str) -> Optional[str]:
    """Pick the most recent file matching ``<results_dir>/<pattern>``."""
    matches = sorted(glob.glob(os.path.join(results_dir, pattern)),
                     key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0] if matches else None


# Consistent colour palette across every figure so the reader doesn't have
# to relearn which colour means which row from one plot to the next.
ROW_COLORS = {
    "baseline":                      "#6c757d",  # neutral grey — the ceiling
    "plugin_idle":                   "#0d6efd",  # blue — the import tax
    "probe_hook_qk:last_token":      "#fd7e14",  # orange — minimal QK
    "probe_hook_qk:all_tokens":      "#e8590c",  # darker orange — full QK
    "probe_hidden_states:last_token":"#f06595",  # pink — typical-use HS
    "probe_hidden_states:all_tokens":"#dc3545",  # red — heavy HS
    "steer_hook_act":                "#198754",  # green — in-place, near-free
    # v0.1.0 peer rows when --include-v010 was on
    "probe_hook_qk_v010:last_token":     "#ffc107",
    "probe_hook_qk_v010:all_tokens":     "#ffb703",
    "probe_hidden_states_v010:last_token":"#fb6f92",
    "probe_hidden_states_v010:all_tokens":"#e63946",
    "steer_hook_act_v010":               "#20c997",
}

ROW_DISPLAY = {
    "baseline":                      "R0 baseline",
    "plugin_idle":                   "R1 plugin_idle",
    "probe_hook_qk:last_token":      "R2 QK · last_token",
    "probe_hook_qk:all_tokens":      "R3 QK · all_tokens",
    "probe_hidden_states:last_token":"R4 HS · last_token",
    "probe_hidden_states:all_tokens":"R5 HS · all_tokens",
    "steer_hook_act":                "R6 steer",
}

# Canonical row order — used by every figure that iterates rows.
ROW_ORDER = [
    "baseline", "plugin_idle",
    "probe_hook_qk:last_token", "probe_hook_qk:all_tokens",
    "probe_hidden_states:last_token", "probe_hidden_states:all_tokens",
    "steer_hook_act",
]


def _row_key(r: Dict[str, Any]) -> str:
    """Workers that have multiple modes get a ``<row>:<mode>`` key so each
    mode shows up as its own bar / line."""
    row = r.get("row") or ""
    if row in ("probe_hook_qk", "probe_hidden_states",
               "probe_hook_qk_v010", "probe_hidden_states_v010"):
        return f"{row}:{r.get('mode')}"
    return row


# ---------------------------------------------------------------------------
# Figure 1: R0–R5 latency bars (one workload cell per group)
# ---------------------------------------------------------------------------


def plot_r0_r5_latency(quick_csv: str, out_path: str) -> None:
    """Side-by-side `gen_lat_ms` per row, grouped by (prompt_len, batch, max_tok).

    One bar per row, eight groups of bars (one group per workload cell). Makes
    the workload-dependence visible in a single figure — a small bar means
    the workload doesn't expose the hook cost, a tall bar means it does.
    """
    rows = _read(quick_csv)
    if not rows:
        print(f"[plots] {quick_csv} empty — skipping r0_r5_latency"); return

    workloads = sorted({(int(_f(r['prompt_len']) or 0),
                        int(_f(r['batch_size']) or 0),
                        int(_f(r['max_tokens']) or 0))
                       for r in rows if _f(r.get("gen_lat_mean")) is not None})

    row_order = list(ROW_ORDER)
    # bar group positions
    n_groups = len(workloads)
    n_rows   = len(row_order)
    bar_w    = 0.85 / n_rows

    fig, ax = plt.subplots(figsize=(max(10, 1.4 * n_groups), 5.5))
    for i, key in enumerate(row_order):
        ys = []
        for wl in workloads:
            match = [r for r in rows
                     if _row_key(r) == key
                     and int(_f(r['prompt_len']) or 0) == wl[0]
                     and int(_f(r['batch_size']) or 0) == wl[1]
                     and int(_f(r['max_tokens']) or 0) == wl[2]]
            ys.append((_f(match[0]['gen_lat_mean']) or 0) * 1000 if match else 0)
        xs = [g + i * bar_w for g in range(n_groups)]
        ax.bar(xs, ys, bar_w, label=ROW_DISPLAY.get(key, key),
               color=ROW_COLORS.get(key, "#999"),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks([g + bar_w * (n_rows - 1) / 2 for g in range(n_groups)])
    ax.set_xticklabels([f"pl={p}\nbs={b}\nmt={t}" for p, b, t in workloads],
                       fontsize=9)
    ax.set_ylabel("gen_lat (ms, lower is better)")
    ax.set_title("R0–R5 generation latency across workload cells",
                 fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: memory footprint per row (worker-sourced cuda + host_rss)
# ---------------------------------------------------------------------------


def plot_memory_footprint(quick_csv: str, out_path: str) -> None:
    """Worst-case-cell CUDA + RSS per row.

    Prefers worker-process gauges (gauge.mem.cuda_alloc_mb.max) — these only
    populate after the sidecar-merge fix landed. Falls back to the driver
    columns (always zero, kept for back-compat) for older CSVs.
    """
    rows = _read(quick_csv)
    if not rows:
        print(f"[plots] {quick_csv} empty — skipping memory_footprint"); return

    def get_worker_cuda(r):
        return _f(r.get("gauge.mem.cuda_alloc_mb.max")) or 0
    def get_worker_rss(r):
        return _f(r.get("gauge.mem.host_rss_mb.max")) \
            or _f(r.get("host_rss_mb_max")) or 0
    def get_nvml(r):
        return _f(r.get("peak_gpu_mb")) or 0

    # Take the worst-case (max worker cuda) cell per row label.
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = _row_key(r)
        if not key:
            continue
        if key not in buckets or get_worker_cuda(r) > get_worker_cuda(buckets[key]):
            buckets[key] = r

    keys = [k for k in ROW_ORDER if k in buckets]
    if not keys:
        print(f"[plots] no rows match expected labels — skipping memory"); return

    cuda_vals = [get_worker_cuda(buckets[k]) for k in keys]
    rss_vals  = [get_worker_rss(buckets[k])  for k in keys]
    nvml_vals = [get_nvml(buckets[k])        for k in keys]
    labels    = [ROW_DISPLAY.get(k, k) for k in keys]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

    xs = list(range(len(keys)))
    bar_w = 0.38
    axL.bar([x - bar_w / 2 for x in xs], cuda_vals, bar_w,
            label="worker cuda_alloc_mb (peak)",
            color="#dc3545", edgecolor="white")
    axL.bar([x + bar_w / 2 for x in xs], rss_vals, bar_w,
            label="worker host_rss_mb (peak)",
            color="#0d6efd", edgecolor="white")
    axL.set_xticks(xs); axL.set_xticklabels(labels, rotation=18, ha="right",
                                            fontsize=9)
    axL.set_ylabel("MB")
    axL.set_title("Per-process memory (worker process)", fontsize=12)
    axL.legend(fontsize=9, frameon=False)
    axL.grid(True, axis="y", linestyle="--", alpha=0.4)
    axL.set_axisbelow(True)

    axR.bar(xs, nvml_vals, color="#6c757d", edgecolor="white")
    axR.set_xticks(xs); axR.set_xticklabels(labels, rotation=18, ha="right",
                                            fontsize=9)
    axR.set_ylabel("MB")
    axR.set_title("System-wide GPU memory (NVML peak)", fontsize=12)
    axR.grid(True, axis="y", linestyle="--", alpha=0.4)
    axR.set_axisbelow(True)
    axR.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.suptitle("Memory footprint per row (worst-case cell)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: storage variant matrix (HS worker, gen_lat heatmap)
# ---------------------------------------------------------------------------


def plot_storage_matrix(storage_csv: str, out_path: str) -> None:
    """variant × prompt_len heatmap of gen_lat_ms, one panel per mode."""
    rows = _read(storage_csv)
    rows = [r for r in rows if r.get("worker") == "probe_hidden_states"
                            and _f(r.get("gen_lat_mean")) is not None]
    if not rows:
        print(f"[plots] {storage_csv} has no HS rows — skipping storage"); return

    variants = sorted({r["storage_variant"] for r in rows})
    prompt_lens = sorted({int(_f(r["prompt_len"])) for r in rows})
    modes = ["last_token", "all_tokens"]

    # Aggregate by (variant, prompt_len, mode) — average gen_lat across batches.
    grid = {m: [[None] * len(prompt_lens) for _ in variants] for m in modes}
    for r in rows:
        m = r.get("mode"); v = r["storage_variant"]; p = int(_f(r["prompt_len"]))
        if m not in modes: continue
        i = variants.index(v); j = prompt_lens.index(p)
        val_ms = (_f(r["gen_lat_mean"]) or 0) * 1000
        cur = grid[m][i][j]
        grid[m][i][j] = val_ms if cur is None else (cur + val_ms) / 2

    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.5))
    if len(modes) == 1: axes = [axes]
    for ax, mode in zip(axes, modes):
        data = [[(grid[mode][i][j] or float("nan")) for j in range(len(prompt_lens))]
                for i in range(len(variants))]
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(prompt_lens)))
        ax.set_xticklabels(prompt_lens, fontsize=10)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants, fontsize=10)
        ax.set_xlabel("prompt_len (tokens)")
        ax.set_title(f"mode={mode}")
        # annotate
        for i in range(len(variants)):
            for j in range(len(prompt_lens)):
                v = data[i][j]
                if v == v:  # not NaN
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v > sum(prompt_lens) else "black",
                            fontsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("gen_lat (ms)", fontsize=9)

    fig.suptitle("Storage variant matrix · probe_hidden_states · gen_lat (ms)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: idle tax verdict
# ---------------------------------------------------------------------------


def plot_idle_tax(idle_csv: str, out_path: str) -> None:
    rows = _read(idle_csv)
    if not rows:
        print(f"[plots] {idle_csv} empty — skipping idle_tax"); return

    bs = [int(_f(r["bs"])) for r in rows]
    plug = [_f(r["plugin_overhead_pct"]) or 0 for r in rows]
    hooks = [_f(r["hooks_overhead_pct"])  or 0 for r in rows]
    pass_threshold = 5.0  # cuda_graph_plan budget

    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = list(range(len(bs)))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], plug, w, label="plugin loaded (no hooks)",
           color="#0d6efd", edgecolor="white")
    ax.bar([x + w / 2 for x in xs], hooks, w, label="hooks installed (idle requests)",
           color="#fd7e14", edgecolor="white")
    ax.axhline(pass_threshold, color="#dc3545", linestyle="--", linewidth=1)
    ax.text(len(xs) - 0.5, pass_threshold + 0.2, f"  PASS gate ({pass_threshold}%)",
            ha="right", color="#dc3545", fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels([f"bs={b}" for b in bs])
    ax.set_ylabel("overhead vs baseline (%)")
    ax.set_title("Idle plugin tax — cuda_graph_plan budget gate",
                 fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: serve-path throughput curves (the asyncio jam)
# ---------------------------------------------------------------------------


def plot_serve_throughput(serve_csv: str, out_path: str) -> None:
    rows = _read(serve_csv)
    if not rows:
        print(f"[plots] {serve_csv} empty — skipping serve"); return

    workers = sorted({r["worker"] for r in rows})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))

    worker_color = {"baseline": "#6c757d", "qk": "#fd7e14",
                    "hidden_states": "#dc3545", "steer": "#198754"}

    for w in workers:
        sub = sorted([r for r in rows if r["worker"] == w],
                     key=lambda r: int(_f(r["concurrency"]) or 0))
        xs = [int(_f(r["concurrency"]) or 0) for r in sub]
        rps = [_f(r["req_per_sec"]) or 0 for r in sub]
        p99 = [(_f(r["gen_lat_p99"]) or 0) * 1000 for r in sub]  # → ms
        axL.plot(xs, rps, marker="o", linewidth=2,
                 label=w, color=worker_color.get(w, "#333"))
        axR.plot(xs, p99, marker="s", linewidth=2,
                 label=w, color=worker_color.get(w, "#333"))

    for ax, ylabel, title in [
        (axL, "req/s (higher is better)", "Throughput vs concurrency"),
        (axR, "gen_lat p99 (ms, lower is better)", "Tail latency vs concurrency"),
    ]:
        ax.set_xlabel("concurrent clients")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=False)

    fig.suptitle("Serve-path: the asyncio jam reproducer", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: per-stage timer breakdown (where is time going?)
# ---------------------------------------------------------------------------


def plot_stage_breakdown(quick_csv: str, out_path: str) -> None:
    """Stacked bar per row showing where the cell's time was spent.

    Bars summed:
      - hookllm.generate         (driver, wraps llm.generate)
      - rpc.flush_disk           (driver, RPC to worker for disk flush)
      - hookllm.analyze          (driver, wraps llm.analyze)
      - analyzer.kernel          (driver, kernel inside analyze)
      - worker.cpu_transfer.hs   (worker, .cpu() of captured tensors)
      - worker.disk_write.*      (worker, atomic disk write)
    """
    rows = _read(quick_csv)
    if not rows:
        print(f"[plots] {quick_csv} empty — skipping stage_breakdown"); return

    stages = [
        ("timer.hookllm.generate.mean",                "generate (driver)",        "#0d6efd"),
        ("timer.rpc.flush_disk.mean",                  "rpc.flush_disk",           "#6610f2"),
        ("timer.worker.cpu_transfer.hs.mean",          "worker .cpu()",            "#fd7e14"),
        ("timer.worker.disk_write.safetensors.mean",   "worker disk write",        "#e8590c"),
        ("timer.hookllm.analyze.mean",                 "analyze (driver)",         "#198754"),
        ("timer.analyzer.kernel.mean",                 "analyzer kernel",          "#dc3545"),
    ]

    # One bar per row at the largest workload cell (highest gen_lat) so the
    # stages have measurable values.
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = _row_key(r)
        if key not in ROW_ORDER:
            continue
        if key not in buckets or (_f(r.get("gen_lat_mean")) or 0) > \
                                 (_f(buckets[key].get("gen_lat_mean")) or 0):
            buckets[key] = r

    keys = [k for k in ROW_ORDER if k in buckets]
    if not keys:
        print(f"[plots] no stage data — skipping breakdown"); return

    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(keys)))
    bottoms = [0.0] * len(keys)
    for col, label, color in stages:
        vals = [_f(buckets[k].get(col)) or 0 for k in keys]
        ax.bar(xs, vals, bottom=bottoms, label=label, color=color,
               edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(xs)
    ax.set_xticklabels([ROW_DISPLAY.get(k, k) for k in keys],
                       rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("time per call (ms)")
    ax.set_title("Per-stage breakdown — where each row spends its time\n"
                 "(largest-workload cell per row)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left", frameon=False, ncol=2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 7 (optional): v0.2.0 vs v0.1.0 paired comparison
# ---------------------------------------------------------------------------


def plot_v010_vs_current(quick_csv: str, out_path: str) -> bool:
    """Returns True if rendered (rows present), False if skipped."""
    rows = _read(quick_csv)
    if not any(r.get("row", "").endswith("_v010") for r in rows):
        return False

    pairs = [
        ("probe_hook_qk:last_token",  "probe_hook_qk_v010"),
        ("probe_hook_qk:all_tokens",  "probe_hook_qk_v010"),
        ("probe_hidden_states",       "probe_hidden_states_v010"),
        ("steer_hook_act",            "steer_hook_act_v010"),
    ]

    def key(r): return (r.get("row"), r.get("mode"), r.get("prompt_len"),
                        r.get("batch_size"), r.get("max_tokens"))
    by_key = {key(r): r for r in rows}

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.35
    labels: List[str] = []
    cur, v01 = [], []
    for cur_name, v010_name in pairs:
        for r in rows:
            if _row_key(r) != cur_name:
                continue
            row_label = r.get("row") if r.get("row") != "probe_hook_qk" else "probe_hook_qk"
            v_key = (v010_name, r.get("mode"), r.get("prompt_len"),
                     r.get("batch_size"), r.get("max_tokens"))
            v = by_key.get(v_key)
            if v is None: continue
            labels.append(f"{cur_name[:18]}\np{r.get('prompt_len')} "
                          f"b{r.get('batch_size')} t{r.get('max_tokens')}")
            cur.append((_f(r["gen_lat_mean"]) or 0) * 1000)
            v01.append((_f(v["gen_lat_mean"]) or 0) * 1000)

    if not labels:
        return False
    xs = list(range(len(labels)))
    ax.bar([x - width / 2 for x in xs], cur, width, label="v0.2.0",
           color="#0d6efd", edgecolor="white")
    ax.bar([x + width / 2 for x in xs], v01, width, label="v0.1.0",
           color="#fd7e14", edgecolor="white")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7, rotation=45,
                                          ha="right")
    ax.set_ylabel("gen_lat (ms)")
    ax.set_title("v0.2.0 vs v0.1.0 — paired cells", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="profiling/results")
    p.add_argument("--out-dir",     default="profiling/results/plots")
    p.add_argument("--quick",   default=None,
                   help="Specific quick-*.csv (default: most recent)")
    p.add_argument("--storage", default=None)
    p.add_argument("--idle",    default=None)
    p.add_argument("--serve",   default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    quick   = args.quick   or _latest("quick-*.csv",   args.results_dir)
    storage = args.storage or _latest("storage-*.csv", args.results_dir)
    idle    = args.idle    or _latest("idle-*.csv",    args.results_dir)
    serve   = args.serve   or _latest("serve-*.csv",   args.results_dir)

    print(f"[plots] results_dir = {args.results_dir}")
    print(f"[plots] out_dir     = {args.out_dir}")
    print(f"[plots] quick       = {quick   or '(none)'}")
    print(f"[plots] storage     = {storage or '(none)'}")
    print(f"[plots] idle        = {idle    or '(none)'}")
    print(f"[plots] serve       = {serve   or '(none)'}")
    print()

    if quick:
        plot_r0_r5_latency(quick,  os.path.join(args.out_dir, "01_r0_r5_latency.png"))
        plot_memory_footprint(quick, os.path.join(args.out_dir, "02_memory_footprint.png"))
        plot_stage_breakdown(quick,  os.path.join(args.out_dir, "06_stage_breakdown.png"))
        plot_v010_vs_current(quick,  os.path.join(args.out_dir, "07_v020_vs_v010.png"))
    if storage:
        plot_storage_matrix(storage, os.path.join(args.out_dir, "03_storage_matrix.png"))
    if idle:
        plot_idle_tax(idle, os.path.join(args.out_dir, "04_idle_tax.png"))
    if serve:
        plot_serve_throughput(serve, os.path.join(args.out_dir, "05_serve_throughput.png"))

    print(f"\n[plots] done → {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
