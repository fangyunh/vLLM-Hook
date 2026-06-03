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


def _section_memory(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Per-row memory comparison. Pulls CUDA stats from the *worker* sidecar
    merge (``gauge.mem.cuda_alloc_mb.*``) when present, since the driver
    process has no CUDA context and its own ``cuda_*_mb_*`` columns are
    structurally zero. Falls back to whatever's there for older CSVs."""

    has_worker_cuda  = bool(rows and "gauge.mem.cuda_alloc_mb.max" in rows[0])
    has_driver_cuda  = bool(rows and "cuda_peak_alloc_mb_max" in rows[0])
    has_host_rss     = bool(rows and ("host_rss_mb_max" in rows[0]
                                      or "gauge.mem.host_rss_mb.max" in rows[0]))
    if not rows or not (has_worker_cuda or has_driver_cuda or has_host_rss):
        return None

    def _v(r, *keys):
        """First non-zero / non-empty value from a list of column names."""
        for k in keys:
            v = r.get(k)
            if v in (None, "", "None"):
                continue
            try:
                if float(v) != 0:
                    return v
            except (TypeError, ValueError):
                return v
        return None

    # Group by (row, mode) — same key the headline table uses — and pick
    # the worst-case (max-CUDA) cell for each.
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rname = r.get("row")
        if rname is None:
            continue
        key = f"{rname}:{r.get('mode')}" if rname == "probe_hook_qk" else rname
        prev = buckets.get(key)
        if prev is None:
            buckets[key] = r
            continue
        prev_peak = _v(prev, "gauge.mem.cuda_alloc_mb.max",
                       "cuda_peak_alloc_mb_max") or 0
        new_peak  = _v(r,    "gauge.mem.cuda_alloc_mb.max",
                       "cuda_peak_alloc_mb_max") or 0
        try:
            if float(new_peak) > float(prev_peak):
                buckets[key] = r
        except (TypeError, ValueError):
            pass

    order = ["baseline", "plugin_idle",
             "probe_hook_qk:last_token", "probe_hook_qk:all_tokens",
             "probe_hidden_states", "steer_hook_act"]
    headers = ["row", "worker cuda_alloc_mb",
               "worker cuda_reserved_mb", "worker host_rss_mb",
               "driver host_rss_mb", "nvml peak (mb)"]
    body = []
    for key in order:
        r = buckets.get(key)
        if r is None:
            continue
        body.append([
            key,
            _f(_v(r, "gauge.mem.cuda_alloc_mb.max",  "cuda_peak_alloc_mb_max")),
            _f(_v(r, "gauge.mem.cuda_reserved_mb.max", "cuda_peak_reserved_mb_max")),
            _f(_v(r, "gauge.mem.host_rss_mb.max")),
            _f(_v(r, "host_rss_mb_max")),
            _f(r.get("peak_gpu_mb")),
        ])
    if not body:
        return None
    note = ("_All values are MB. `worker cuda_alloc_mb` is the high-water "
            "mark of `torch.cuda.memory_allocated()` sampled every 50 ms "
            "inside the EngineCore process — the cleanest signal of what "
            "the hook itself costs in GPU memory._")
    return ("## Memory footprint per row (worst-case cell)\n\n"
            + note + "\n\n"
            + _table(body, headers))


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


def _section_v010_vs_current(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Per-(prompt, batch, max_tok) comparison of v0.2.0 vs v0.1.0 for each
    worker that has both rows. Skipped when no _v010 rows are present."""
    pairs = [
        ("probe_hidden_states", "probe_hidden_states_v010"),
        ("probe_hook_qk",       "probe_hook_qk_v010"),
        ("steer_hook_act",      "steer_hook_act_v010"),
    ]
    has_any_v010 = any(r.get("row", "").endswith("_v010") for r in rows)
    if not has_any_v010:
        return None

    def keyof(r):
        return (r.get("row"), r.get("mode"), r.get("prompt_len"),
                r.get("batch_size"), r.get("max_tokens"))

    by_key = {keyof(r): r for r in rows}

    sections = []
    for current, v010 in pairs:
        # Find every workload cell where BOTH the v0.2.0 row and the matching
        # v0.1.0 row ran in the same mode.
        body = []
        for r in rows:
            if r.get("row") != current:
                continue
            v010_key = (v010, r.get("mode"), r.get("prompt_len"),
                        r.get("batch_size"), r.get("max_tokens"))
            v = by_key.get(v010_key)
            if v is None:
                continue
            cur_ms = (r.get("gen_lat_mean") or 0) * 1000
            v_ms   = (v.get("gen_lat_mean") or 0) * 1000
            delta_pct = ((v_ms - cur_ms) / cur_ms * 100) if cur_ms else 0
            body.append([
                _f(r.get("prompt_len")), _f(r.get("batch_size")),
                _f(r.get("max_tokens")), r.get("mode") or "—",
                _f(cur_ms), _f(v_ms), _f(delta_pct) + "%",
                _f(r.get("artifact_kb_mean")), _f(v.get("artifact_kb_mean")),
            ])
        if not body:
            continue
        headers = ["prompt_len", "batch", "max_tok", "mode",
                   "v0.2.0 gen_ms", "v0.1.0 gen_ms", "v0.1.0 − v0.2.0",
                   "v0.2.0 art_kb", "v0.1.0 art_kb"]
        sections.append(f"### {current}\n\n" + _table(body, headers))

    if not sections:
        return None
    intro = ("## v0.2.0 vs v0.1.0 (paired cells)\n\n"
             "_Negative `v0.1.0 − v0.2.0` means v0.1.0 was faster on that "
             "cell; positive means the current version wins. v0.1.0 has no "
             "prefix-cache prefix-recon path, so its artifact size may "
             "differ when prefix caching is on._\n\n")
    return intro + "\n\n".join(sections)


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
    for section in (_section_six_rows, _section_memory,
                    _section_v010_vs_current,
                    _section_storage_matrix, _section_idle_tax,
                    _section_top_timers):
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
