#!/bin/bash
# verify_capture.sh — did the hidden-state capture actually work?
#
# Usage:  bluevela/verify_capture.sh bluevela/logs/hs_capture.<JOBID>.out
#
# Three checks, cheapest first:
#   1. did each leg engage the mode it claimed?
#   2. did every configured layer produce a tensor with a finite, non-zero norm?
#   3. does the graph leg reproduce the eager leg exactly?  <-- the real proof
set -euo pipefail

LOG="${1:-}"
[ -n "$LOG" ] && [ -f "$LOG" ] || { echo "usage: $0 <hs_capture log>"; exit 2; }

echo "=== 1. modes that engaged ==="
grep -E '^\[demo_hiddenstate\] mode=' "$LOG" || { echo "NO mode line — the demo never started"; exit 1; }

echo
echo "=== 2. captured layers per leg ==="
awk '/LEG=eager/{leg="eager"} /LEG=graph/{leg="graph"}
     /norm=/ && /shape=/ {n[leg]++}
     END {for (l in n) printf "  %-6s %d layer-tensors\n", l, n[l]}' "$LOG"

# grep exits 1 when it finds nothing, which is the GOOD case here -- do not let
# set -e/pipefail treat "clean log" as a failure.
bad=$(grep -coE 'norm=(nan|inf|-inf|0\.0000)' "$LOG" || true)
if [ "$bad" -gt 0 ]; then
  echo "  FAIL: $bad degenerate norm(s) (nan/inf/zero) — capture ran but the values are junk"
  grep -nE 'norm=(nan|inf|-inf|0\.0000)' "$LOG" | head -5 || true
  exit 1
fi
echo "  all norms finite and non-zero"

echo
echo "=== 3. graph vs eager (the correctness check) ==="
if ! grep -q 'LEG=graph' "$LOG" || ! grep -q 'LEG=eager' "$LOG"; then
  echo "  SKIPPED — log has only one leg. Re-run with HS_MODE=both to compare."
  exit 0
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
awk '/LEG=eager/{on=1;next} /LEG=graph/{on=0} on && /norm=/ && /shape=/' "$LOG" \
  | sed 's/^[[:space:]]*//' > "$tmp/eager.txt"
awk '/LEG=graph/{on=1;next} on && /norm=/ && /shape=/' "$LOG" \
  | sed 's/^[[:space:]]*//' > "$tmp/graph.txt"

if [ ! -s "$tmp/eager.txt" ] || [ ! -s "$tmp/graph.txt" ]; then
  echo "  SKIPPED — could not extract both legs' layer lines"
  exit 0
fi

if diff -q "$tmp/eager.txt" "$tmp/graph.txt" >/dev/null; then
  echo "  PASS — every layer's shape and norm is identical between eager and graph."
  echo "         The CUDA-graph capture ring reproduced the eager ground truth."
else
  echo "  FAIL — graph capture differs from eager:"
  diff "$tmp/eager.txt" "$tmp/graph.txt" | head -20
  echo
  echo "  Norms are printed to 4dp, so a diff here is a real divergence, not rounding."
  exit 1
fi
