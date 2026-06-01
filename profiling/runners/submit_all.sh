#!/bin/bash
# submit_all.sh — login-node helper that queues the "first deliverable" set
# with LSF job dependencies. Mirrors tests/cuda_graph/submit_all_gates.sh.
#
# Run this from the project root on the LSF login node, NOT inside a job:
#   bash profiling/runners/submit_all.sh
#
# Topology:
#       run_smoke   (5 min profiler sanity check; gates the rest)
#         └── run_quick    (R0..R5 deliverable; ~30 min)
#         └── run_idle_tax (per-layer launch cost; ~15 min)
#
# run_quick and run_idle_tax run in parallel after run_smoke completes
# cleanly. If smoke FAILS the dependents stay pending forever; bkill
# them and diagnose smoke first (no profiler dump usually means
# VLLM_HOOK_PROFILE didn't propagate to the worker process).
#
# To also add storage / serve / full, append them with the same -w pattern:
#   STORAGE_ID=$(bsub -w "done($SMOKE_ID)" < profiling/runners/run_storage.sh ...)

set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

echo "[submit] smoke (profiler sanity check)..."
SMOKE_ID=$(bsub < profiling/runners/run_smoke.sh \
            | awk '{print $2}' | tr -d '<>')
echo "[submit] smoke job id: $SMOKE_ID"

DEP="done($SMOKE_ID)"

echo "[submit] quick (R0..R5), depends on done($SMOKE_ID)..."
QUICK_ID=$(bsub -w "$DEP" < profiling/runners/run_quick.sh \
            | awk '{print $2}' | tr -d '<>')
echo "[submit] quick job id: $QUICK_ID"

echo "[submit] idle_tax, depends on done($SMOKE_ID)..."
IDLE_ID=$(bsub -w "$DEP" < profiling/runners/run_idle_tax.sh \
           | awk '{print $2}' | tr -d '<>')
echo "[submit] idle_tax job id: $IDLE_ID"

cat <<EOF

[submit] All three queued.
  smoke    : $SMOKE_ID   (runs immediately when a GPU is free)
  quick    : $QUICK_ID   (waits for smoke done)
  idle_tax : $IDLE_ID    (waits for smoke done)

Track:
  bjobs
  bpeek $SMOKE_ID
  tail -f profiling/runners/logs/smoke.${SMOKE_ID}.out

Cancel one:
  bkill <JOBID>

Results land in:
  profiling/results/quick-*-${QUICK_ID}.csv  (+ -summary.md, .jsonl)
  profiling/results/idle-*-${IDLE_ID}.csv

If smoke FAILS the dependents stay pending forever — bkill them, fix the
issue (most often: VLLM_HOOK_PROFILE not reaching the worker subprocess),
then re-run submit_all.sh.
EOF
