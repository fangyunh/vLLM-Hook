"""compare_baselines — diff hook rows against the baseline / plugin_idle rows.

Given a bench_grid CSV, for each unique (model, prompt_len, batch, max_tokens)
cell, computes the relative slowdown of each hook variant against the
baseline row at the matching cell, and emits a markdown table.

This is the table the advisor sees first: "what does turning on each
worker actually cost?"
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def _read(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v):
    if v is None or v == "":
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _key(r: Dict[str, Any]) -> Tuple:
    return (r.get("model"), r.get("prompt_len"), r.get("batch_size"),
            r.get("max_tokens"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metric", default="gen_lat_mean",
                   help="Column to compare. Default gen_lat_mean.")
    args = p.parse_args()

    rows = _read(args.csv)
    base: Dict[Tuple, Dict[str, Any]] = {}
    by_row: Dict[str, Dict[Tuple, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("row") == "baseline":
            base[_key(r)] = r
        else:
            by_row[r.get("row")][_key(r)] = r
    if not base:
        print("[compare] no baseline rows found.")
        return 2

    headers = ["row", "model", "prompt_len", "batch", "max_tok",
               "baseline_ms", f"{args.metric}_ms", "slowdown_x"]
    body: List[List[str]] = []
    for row_name in sorted(by_row):
        for key, r in sorted(by_row[row_name].items()):
            b = base.get(key)
            if not b:
                continue
            try:
                bv = float(b.get(args.metric)) * 1000 if b.get(args.metric) else 0
                rv = float(r.get(args.metric)) * 1000 if r.get(args.metric) else 0
                slow = rv / bv if bv else float("nan")
            except (TypeError, ValueError):
                continue
            body.append([row_name, *map(str, key), _f(bv), _f(rv), _f(slow) + "x"])

    if not body:
        print("[compare] no overlap between baseline and hook rows.")
        return 1
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
