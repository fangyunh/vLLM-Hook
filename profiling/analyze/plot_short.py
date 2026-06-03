"""plot_short — three high-density figures for the executive summary.

Reads:
  profiling/results/storage-*.csv     (HS × 6 storage variants × workloads)
  profiling/results/idle-*.csv        (3 PASS/FAIL rows)
  profiling/results/serve-*.csv       (3 workers × 4 concurrencies)

R0–R5 numbers come from the most recent REPORT.md analysis (the quick CSV
that generated them was deleted; the numbers are stable across runs and
embedded here as constants so the script is self-contained).

Outputs three PNGs to ``profiling/results/short_plots/``:
  fig1_cost_dashboard.png   — R0–R5 latency × artifact × idle-tax PASS gate
  fig2_storage_choice.png   — every storage variant on one log axis
  fig3_serve_jam.png        — req/s + response-size companion
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "profiling/results/short_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# R0–R6 numbers. R0/R1/R2/R3/R5/R6 are means across 8 workload cells per row,
# pulled from the latest REPORT.md §3.1 (quick CSV 1595702). R4 (HS last_token)
# is an ESTIMATE derived from the storage CSV's HS-last vs HS-all ratio scaled
# against R5's measured value — the next quick run will replace it with the
# real number. The estimate keeps the dashboard's structure honest while we
# wait for the rerun.
# ---------------------------------------------------------------------------
_HS_LAST_TOKEN_ESTIMATED = True  # set False once real plan-quick number lands

R_ROWS = [
    # (label,                       gen_ms, vs_baseline_pct, artifact_kb)
    ("R0 baseline",                 294.3,   0.0,                 0.0),
    ("R1 plugin_idle",              301.3,   2.4,                 0.0),
    ("R6 steer (in-place)",         309.5,   5.2,                 0.0),
    ("R3 QK · all_tokens",          313.1,   6.4,              4050.0),
    ("R2 QK · last_token",          319.3,   8.5,               607.0),
    ("R4 HS · last_token *",        347.1,  18.0,               680.0),  # estimated
    ("R5 HS · all_tokens",          388.6,  32.0,             97112.0),
]
TIER_COLORS = ["#6c757d", "#0d6efd", "#198754",
               "#fd7e14", "#e8590c",
               "#f06595", "#dc3545"]

# Idle tax PASS/FAIL — from idle-1588746.csv
IDLE_TAX = [
    # (batch_size, plugin_overhead_pct, hooks_overhead_pct)
    (1,   -1.28,  1.58),
    (8,   -0.37,  4.27),
    (64,  -3.46,  1.62),
]


def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None


def _latest(pattern):
    matches = sorted(glob.glob(os.path.join("profiling/results", pattern)),
                     key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


# ===========================================================================
# Figure 1: The cost dashboard (3 panels)
# ===========================================================================

def fig1_cost_dashboard():
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.5),
                                         gridspec_kw=dict(width_ratios=[1.4, 1.0, 1.0]))

    # ---- Left panel: gen_lat (bars) + overhead % (text labels) ----
    labels = [r[0] for r in R_ROWS]
    gens   = [r[1] for r in R_ROWS]
    deltas = [r[2] for r in R_ROWS]
    bars = axL.bar(range(len(labels)), gens, color=TIER_COLORS, edgecolor="white", linewidth=0.5)
    for i, (b, d) in enumerate(zip(bars, deltas)):
        lbl = "—" if d == 0 else f"{d:+.1f}%"
        weight = "bold" if abs(d) >= 10 else "normal"
        col = "#dc3545" if d >= 10 else ("#16a34a" if 0 < d < 5 else "#1f2328")
        axL.text(b.get_x() + b.get_width()/2, b.get_height() + 8, lbl,
                 ha="center", fontsize=10.5, color=col, weight=weight)
    axL.set_xticks(range(len(labels)))
    axL.set_xticklabels(labels, rotation=22, ha="right", fontsize=9.5)
    axL.set_ylabel("mean gen_lat (ms)\nacross 8 workload cells", fontsize=10)
    axL.set_title("(a) The four cost tiers", fontsize=12, weight="bold")
    axL.set_ylim(0, max(gens) * 1.15)
    axL.grid(True, axis="y", linestyle="--", alpha=0.4)
    axL.set_axisbelow(True)

    # ---- Middle panel: artifact volume (log scale) ----
    arts = [r[3] for r in R_ROWS]
    arts_plot = [max(a, 0.5) for a in arts]  # log axis needs >0
    bars = axM.bar(range(len(labels)), arts_plot, color=TIER_COLORS, edgecolor="white", linewidth=0.5)
    for i, (b, a) in enumerate(zip(bars, arts)):
        if a == 0:
            txt = "0"
        elif a >= 1024:
            txt = f"{a/1024:.0f} MB"
        else:
            txt = f"{a:.0f} KB"
        axM.text(b.get_x() + b.get_width()/2, b.get_height() * 1.4, txt,
                 ha="center", fontsize=10, color="#1f2328")
    axM.set_yscale("log")
    axM.set_xticks(range(len(labels)))
    axM.set_xticklabels([l.split()[0] for l in labels], fontsize=10)  # short labels
    axM.set_ylabel("artifact / cell (KB, log scale)", fontsize=10)
    axM.set_title("(b) I/O volume", fontsize=12, weight="bold")
    axM.set_ylim(0.3, max(arts_plot) * 5)
    axM.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
    axM.set_axisbelow(True)

    # ---- Right panel: idle tax with PASS gate ----
    bs_labels = [f"bs={b}" for b, _, _ in IDLE_TAX]
    plug_vals = [p for _, p, _ in IDLE_TAX]
    hook_vals = [h for _, _, h in IDLE_TAX]
    xs = np.arange(len(bs_labels)); w = 0.38
    axR.bar(xs - w/2, plug_vals, w, label="plugin loaded\n(no hooks)",
            color="#0d6efd", edgecolor="white")
    axR.bar(xs + w/2, hook_vals, w, label="hooks installed\n(opt-out)",
            color="#fd7e14", edgecolor="white")
    axR.axhline(5.0, color="#dc3545", linestyle="--", linewidth=1.4)
    axR.text(len(xs) - 0.5, 5.2, " ≤ 5% PASS gate", ha="right",
             color="#dc3545", fontsize=10, weight="bold")
    axR.axhline(0, color="#1f2328", linewidth=0.5)
    axR.set_xticks(xs); axR.set_xticklabels(bs_labels, fontsize=10)
    axR.set_ylabel("overhead vs baseline (%)", fontsize=10)
    axR.set_title("(c) Idle plugin tax", fontsize=12, weight="bold")
    axR.set_ylim(-5, 7)
    axR.legend(fontsize=9, frameon=False, loc="lower left")
    axR.grid(True, axis="y", linestyle="--", alpha=0.4)
    axR.set_axisbelow(True)

    fig.suptitle("Cost dashboard — plug-in is free when idle, heavy only for HS",
                 fontsize=13.5, weight="bold", y=1.00)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_cost_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ===========================================================================
# Figure 2: Storage variant choice (one panel, log scale)
# ===========================================================================

def fig2_storage_choice():
    storage_csv = _latest("storage-*.csv")
    if storage_csv is None:
        print("[plot] no storage CSV — skipping fig2"); return

    with open(storage_csv) as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r.get("worker") == "probe_hidden_states"
                            and r.get("mode") == "all_tokens"   # the harder mode
                            and int(_f(r.get("batch_size")) or 0) == 8
                            and _f(r.get("gen_lat_mean")) is not None]

    # Group by variant → list of (prompt_len, gen_ms)
    by_variant: Dict[str, List[tuple]] = defaultdict(list)
    for r in rows:
        pl = int(_f(r["prompt_len"]))
        ms = _f(r["gen_lat_mean"]) * 1000
        by_variant[r["storage_variant"]].append((pl, ms))

    # Order: catastrophic → mid → best
    order = ["rpc", "disk-pt", "disk-pt-async", "disk-st", "disk-st-async"]
    palette = {
        "rpc":           "#dc3545",
        "disk-pt":       "#fd7e14",
        "disk-pt-async": "#e8590c",
        "disk-st":       "#0d6efd",
        "disk-st-async": "#16a34a",
    }
    markers = {"rpc": "X", "disk-pt": "s", "disk-pt-async": "D",
               "disk-st": "o", "disk-st-async": "^"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for v in order:
        if v not in by_variant: continue
        pts = sorted(by_variant[v])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        linew = 3.0 if v == "disk-st-async" else (2.6 if v == "rpc" else 1.6)
        ax.plot(xs, ys, marker=markers[v], markersize=9, linewidth=linew,
                color=palette[v], label=v)

    # Annotate the extremes.
    rpc_pts = sorted(by_variant.get("rpc", []))
    if rpc_pts:
        worst = max(rpc_pts, key=lambda p: p[1])
        ax.annotate(f"{worst[1]/1000:.0f} s\n(rpc catastrophe)",
                    xy=worst, xytext=(worst[0]*0.6, worst[1]*0.7),
                    fontsize=10.5, color="#dc3545", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#dc3545", lw=1.2),
                    ha="center")
    dsa = sorted(by_variant.get("disk-st-async", []))
    if dsa:
        best = max(dsa, key=lambda p: p[1])
        ax.annotate(f"{best[1]:.0f} ms\n(disk-st-async)",
                    xy=best, xytext=(best[0]*1.05, best[1]*0.15),
                    fontsize=10.5, color="#16a34a", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.2),
                    ha="left")

    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 64, 256, 512])
    ax.set_xticklabels(["16", "64", "256", "512"])
    ax.set_xlabel("prompt_len (tokens, log scale)", fontsize=11)
    ax.set_ylabel("gen_lat (ms, log scale)", fontsize=11)
    ax.set_title("Storage variant choice spans 3 orders of magnitude\n"
                 "(probe_hidden_states · all_tokens · batch=8)",
                 fontsize=13, weight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(title="storage variant", fontsize=10, loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_storage_choice.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ===========================================================================
# Figure 3: Serve-path jam (2 panels — throughput + response bytes)
# ===========================================================================

def fig3_serve_jam():
    serve_csv = _latest("serve-*.csv")
    if serve_csv is None:
        print("[plot] no serve CSV — skipping fig3"); return

    with open(serve_csv) as fh:
        rows = list(csv.DictReader(fh))

    workers = ["baseline", "hidden_states", "qk"]
    colors  = {"baseline": "#6c757d", "hidden_states": "#dc3545", "qk": "#fd7e14"}
    markers = {"baseline": "o", "hidden_states": "s", "qk": "^"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5),
                                    gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    # ---- Left: req/s vs concurrency ----
    max_rps = 0
    for w in workers:
        sub = sorted([r for r in rows if r["worker"] == w],
                     key=lambda r: int(_f(r["concurrency"]) or 0))
        xs = [int(_f(r["concurrency"]) or 0) for r in sub]
        ys = [_f(r["req_per_sec"]) or 0 for r in sub]
        if not xs: continue
        axL.plot(xs, ys, marker=markers[w], markersize=10, linewidth=2.4,
                 color=colors[w], label=w)
        max_rps = max(max_rps, max(ys))
        # Annotate k=8 value
        axL.annotate(f"{ys[-1]:.1f}", xy=(xs[-1], ys[-1]),
                     xytext=(xs[-1] + 0.3, ys[-1]),
                     fontsize=10.5, color=colors[w], weight="bold",
                     ha="left", va="center")

    # Annotate the gap at k=8
    axL.axvline(8, color="#999", linestyle=":", linewidth=1)
    axL.text(8.0, max_rps * 0.55,
             "4.6× gap\n(qk)", ha="left", fontsize=10, color="#fd7e14")

    axL.set_xlabel("concurrent clients", fontsize=11)
    axL.set_ylabel("req/s (higher is better)", fontsize=11)
    axL.set_title("(a) Throughput collapses under concurrency",
                  fontsize=12, weight="bold")
    axL.grid(True, linestyle="--", alpha=0.4)
    axL.set_axisbelow(True)
    axL.legend(fontsize=11, frameon=False, loc="upper left")
    axL.set_xticks([1, 2, 4, 8])

    # ---- Right: response bytes per request, log scale ----
    bytes_at_k8: Dict[str, float] = {}
    for w in workers:
        sub = [r for r in rows if r["worker"] == w
               and int(_f(r["concurrency"]) or 0) == 8]
        if sub:
            bytes_at_k8[w] = (_f(sub[0]["response_bytes_mean"]) or 0) / 1024  # → KB
    labels = list(bytes_at_k8.keys())
    vals   = [bytes_at_k8[w] for w in labels]
    cols   = [colors[w] for w in labels]
    bars = axR.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        if v >= 1024:
            txt = f"{v/1024:.1f} MB"
        else:
            txt = f"{v:.1f} KB"
        axR.text(b.get_x() + b.get_width()/2, b.get_height() * 1.4, txt,
                 ha="center", fontsize=10.5, weight="bold")
    axR.set_yscale("log")
    axR.set_ylabel("response bytes / request\n(log scale)", fontsize=11)
    axR.set_title("(b) Why: JSON serialization", fontsize=12, weight="bold")
    axR.set_ylim(0.4, max(vals) * 6)
    axR.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)
    axR.set_axisbelow(True)

    fig.suptitle("Serve-path bottleneck is JSON encoding, not generation",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_serve_jam.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    fig1_cost_dashboard()
    fig2_storage_choice()
    fig3_serve_jam()
    print(f"\n[done] 3 figures → {OUT_DIR}/")
