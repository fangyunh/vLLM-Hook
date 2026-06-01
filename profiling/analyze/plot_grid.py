"""plot_grid — heatmaps over (n_layers × prompt_len) for one metric.

Matches the style of ``Numerical_Analysis/plot_grid_results.py``. Operates
on the CSV produced by ``bench_grid.py --plan plan-storage`` or
``--plan plan-full``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Any, Dict, List


def _read(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _plot_one(metric: str, group_by: str, rows: List[Dict[str, Any]],
              out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot_grid] matplotlib missing; install matplotlib to plot")
        return

    groups: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    layers = set(); prompt_lens = set()
    for r in rows:
        g = str(r.get(group_by) or "—")
        try:
            ln = int(float(r.get("n_layers_captured", 0) or 0))
            pl = int(float(r.get("prompt_len", 0) or 0))
            val = _to_float(r.get(metric))
        except (TypeError, ValueError):
            continue
        groups[g][(ln, pl)] = val
        layers.add(ln)
        prompt_lens.add(pl)
    layers_l = sorted(x for x in layers if x > 0)
    prompt_lens_l = sorted(x for x in prompt_lens if x > 0)
    if not groups or not layers_l or not prompt_lens_l:
        print(f"[plot_grid] no data for metric={metric}, group_by={group_by}")
        return

    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (g, data) in zip(axes.flat, sorted(groups.items())):
        grid = np.full((len(layers_l), len(prompt_lens_l)), float("nan"))
        for (ln, pl), v in data.items():
            i = layers_l.index(ln); j = prompt_lens_l.index(pl)
            grid[i, j] = v
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       interpolation="nearest")
        ax.set_xticks(range(len(prompt_lens_l)))
        ax.set_xticklabels(prompt_lens_l)
        ax.set_yticks(range(len(layers_l)))
        ax.set_yticklabels(layers_l)
        ax.set_xlabel("prompt_len")
        ax.set_ylabel("n_layers_captured")
        ax.set_title(f"{group_by}={g}")
        fig.colorbar(im, ax=ax)
    fig.suptitle(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot_grid] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metric", default="gen_lat_mean")
    p.add_argument("--group-by", default="storage_variant")
    p.add_argument("--output", default="grid.png")
    args = p.parse_args()
    rows = _read(args.csv)
    _plot_one(args.metric, args.group_by, rows, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
