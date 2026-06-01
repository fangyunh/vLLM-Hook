"""summarize — turn bench_grid CSV into a markdown report.

Output sections:
  1. Six-row table   (R0..R5 — plan.html §7 deliverable; only if present)
  2. Storage matrix  (variant × mode × key metrics, if multiple variants seen)
  3. Top per-stage timers (mean ms, by named timer)
  4. Idle tax        (plugin/hooks overhead vs baseline at matched cells)

Standalone — no vLLM / no torch required. Pure stdlib + csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def _f(v: Any) -> str:
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


def _read(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _coerce(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert numeric strings to floats where possible (CSV-roundtrip)."""
    out = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            try:
                nr[k] = float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                nr[k] = v
        out.append(nr)
    return out


def _table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _section_six_rows(rows: List[Dict[str, Any]]) -> Optional[str]:
    """The §7 deliverable. Filter to one rep per row label."""
    target = ["baseline", "plugin_idle", "probe_hook_qk", "probe_hook_qk",
              "probe_hidden_states", "steer_hook_act"]
    pick: Dict[str, Dict[str, Any]] = {}
    # Distinguish R2 (last_token) from R3 (all_tokens) for probe_hook_qk
    for r in rows:
        row = r.get("row")
        if row not in target:
            continue
        if row == "probe_hook_qk":
            key = f"probe_hook_qk:{r.get('mode')}"
        else:
            key = row
        if key in pick:
            continue
        pick[key] = r
    if not pick:
        return None

    labels = [
        ("R0", "baseline"),
        ("R1", "plugin_idle"),
        ("R2", "probe_hook_qk:last_token"),
        ("R3", "probe_hook_qk:all_tokens"),
        ("R4", "probe_hidden_states"),
        ("R5", "steer_hook_act"),
    ]
    headers = ["#", "row", "gen_lat_ms", "prefill_tok/s", "decode_tok/s",
               "peak_gpu_mb", "artifact_kb"]
    body = []
    for tag, key in labels:
        r = pick.get(key)
        if r is None:
            body.append([tag, key, "—", "—", "—", "—", "—"])
            continue
        body.append([
            tag, key,
            _f((r.get("gen_lat_mean") or 0) * 1000),
            _f(r.get("prefill_tok_per_sec")),
            _f(r.get("decode_tok_per_sec")),
            _f(r.get("peak_gpu_mb")),
            _f(r.get("artifact_kb_mean")),
        ])
    return "## R0–R5 six-row summary (plan.html §7)\n\n" + _table(body, headers)


def _section_storage_matrix(rows: List[Dict[str, Any]]) -> Optional[str]:
    storage_rows = [r for r in rows
                    if r.get("worker") == "probe_hidden_states"
                    and r.get("storage_variant")]
    seen_variants = sorted({r.get("storage_variant") for r in storage_rows})
    if len(seen_variants) <= 1:
        return None

    headers = ["variant", "mode", "prompt_len", "batch",
               "gen_lat_ms", "wait_ms", "analyze_ms",
               "peak_gpu_mb", "artifact_kb"]
    body = []
    keyed: Dict[tuple, Dict[str, Any]] = {}
    for r in storage_rows:
        k = (r.get("storage_variant"), r.get("mode"),
             r.get("prompt_len"), r.get("batch_size"))
        keyed.setdefault(k, r)
    for k in sorted(keyed):
        r = keyed[k]
        body.append([
            r.get("storage_variant"), r.get("mode"),
            _f(r.get("prompt_len")), _f(r.get("batch_size")),
            _f((r.get("gen_lat_mean") or 0) * 1000),
            _f((r.get("wait_lat_mean") or 0) * 1000),
            _f((r.get("analyze_lat_mean") or 0) * 1000),
            _f(r.get("peak_gpu_mb")),
            _f(r.get("artifact_kb_mean")),
        ])
    return "## Storage variant matrix (HS worker)\n\n" + _table(body, headers)


def _section_top_timers(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Across all rows, show the top-mean timer per cell label."""
    timer_cols = [k for k in (rows[0] if rows else {}) if k.startswith("timer.")
                  and k.endswith(".mean")]
    if not timer_cols:
        return None
    # For each timer, aggregate the mean across rows (gives a global picture).
    agg: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        for c in timer_cols:
            v = r.get(c)
            if isinstance(v, (int, float)) and v is not None:
                agg[c].append(float(v))
    ranked = sorted(
        ((c, sum(vs) / len(vs)) for c, vs in agg.items() if vs),
        key=lambda x: -x[1],
    )[:15]
    if not ranked:
        return None
    headers = ["timer", "mean_ms_global"]
    body = [[c.replace("timer.", "").replace(".mean", ""), _f(v)] for c, v in ranked]
    return "## Top per-stage timers (global mean ms)\n\n" + _table(body, headers)


def _section_idle_tax(rows: List[Dict[str, Any]]) -> Optional[str]:
    base = {(r.get("prompt_len"), r.get("batch_size"), r.get("max_tokens")):
            r for r in rows if r.get("row") == "baseline"}
    plug = {(r.get("prompt_len"), r.get("batch_size"), r.get("max_tokens")):
            r for r in rows if r.get("row") == "plugin_idle"}
    if not base or not plug:
        return None
    headers = ["prompt_len", "batch", "max_tok",
               "base_gen_ms", "plugin_gen_ms", "delta_pct"]
    body = []
    for key in sorted(set(base) | set(plug)):
        b = base.get(key); pl = plug.get(key)
        if not b or not pl:
            continue
        bv = (b.get("gen_lat_mean") or 0) * 1000
        pv = (pl.get("gen_lat_mean") or 0) * 1000
        delta = 100 * (pv - bv) / bv if bv else 0
        body.append([_f(key[0]), _f(key[1]), _f(key[2]),
                     _f(bv), _f(pv), _f(delta) + "%"])
    if not body:
        return None
    return "## Idle plugin tax (baseline vs plugin_idle)\n\n" + _table(body, headers)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--output", default=None,
                   help="Write markdown to file instead of stdout.")
    args = p.parse_args()
    rows = _coerce(_read(args.csv))
    parts: List[str] = [f"# Profile report — `{args.csv}`",
                       f"_{len(rows)} rows_"]
    for section in (_section_six_rows, _section_storage_matrix,
                    _section_idle_tax, _section_top_timers):
        s = section(rows)
        if s:
            parts.append(s)
    md = "\n\n".join(parts) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        print(f"[summarize] wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
