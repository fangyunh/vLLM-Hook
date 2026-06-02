#!/bin/bash
# submit_all.sh — login-node helper that queues the profiling deliverable set
# with LSF job dependencies. Every stage waits on smoke succeeding so a broken
# build never burns hours of GPU time on bad runs.
#
# Run from the project root on the LSF login node, NOT inside a job:
#   bash profiling/runners/submit_all.sh
#
# Default topology (STAGES="quick idle storage serve full"):
#
#       run_smoke   (5-min profiler sanity check; gates the rest)
#         ├── run_quick    (R0..R5 deliverable;          ~30 min)
#         ├── run_idle_tax (CUDA-graph budget gate;      ~15 min)
#         ├── run_storage  (5 storage variants × HS;     ~2 h)
#         ├── run_serve    (asyncio-jam reproducer;      ~30 min)
#         └── run_full     (overnight cartesian sweep;   ~6–8 h)
#
# All four post-smoke stages are independent and run in parallel after smoke
# completes cleanly. If smoke FAILS the dependents stay pending forever;
# bkill them and diagnose smoke first (no profiler dump usually means
# VLLM_HOOK_PROFILE didn't propagate to the worker process).
#
# Subset via the STAGES env var (space-separated; order is preserved):
#   STAGES="quick idle"          bash profiling/runners/submit_all.sh  # daily loop
#   STAGES="storage serve"       bash profiling/runners/submit_all.sh  # episodic
#   STAGES="full"                bash profiling/runners/submit_all.sh  # overnight
#   STAGES=""                    bash profiling/runners/submit_all.sh  # smoke only

set -euo pipefail
cd "$(dirname "$0")/../.."   # project root

# ---------------------------------------------------------------------------
# Stage selection
# ---------------------------------------------------------------------------
STAGES_DEFAULT="quick idle storage serve full"
STAGES="${STAGES-$STAGES_DEFAULT}"

VALID_STAGES=(quick idle storage serve full)
is_valid_stage() {
    local s=$1
    for v in "${VALID_STAGES[@]}"; do
        [[ "$s" == "$v" ]] && return 0
    done
    return 1
}

for s in $STAGES; do
    if ! is_valid_stage "$s"; then
        echo "[submit] ERROR: unknown stage '$s'." >&2
        echo "[submit]        valid: ${VALID_STAGES[*]}" >&2
        exit 1
    fi
done

echo "[submit] stages selected: ${STAGES:-(none — smoke only)}"

# ---------------------------------------------------------------------------
# Smoke — gates everything
# ---------------------------------------------------------------------------
echo "[submit] smoke (profiler sanity check)..."
SMOKE_ID=$(bsub < profiling/runners/run_smoke.sh \
            | awk '{print $2}' | tr -d '<>')
echo "[submit] smoke job id: $SMOKE_ID"

DEP="done($SMOKE_ID)"

# ---------------------------------------------------------------------------
# Dependent stages
# ---------------------------------------------------------------------------
QUICK_ID=""
IDLE_ID=""
STORAGE_ID=""
SERVE_ID=""
FULL_ID=""

submit_dep() {
    # $1 = display name, $2 = script path
    # Returns the job id on stdout; log line goes to stderr so the caller's
    # $(...) capture only sees the id.
    local name=$1 script=$2 id
    id=$(bsub -w "$DEP" < "$script" | awk '{print $2}' | tr -d '<>')
    echo "[submit] $name job id: $id  (waits on done($SMOKE_ID))" >&2
    echo "$id"
}

for s in $STAGES; do
    case $s in
        quick)   QUICK_ID=$(submit_dep   "quick   " profiling/runners/run_quick.sh)    ;;
        idle)    IDLE_ID=$(submit_dep    "idle_tax" profiling/runners/run_idle_tax.sh) ;;
        storage) STORAGE_ID=$(submit_dep "storage " profiling/runners/run_storage.sh)  ;;
        serve)   SERVE_ID=$(submit_dep   "serve   " profiling/runners/run_serve.sh)    ;;
        full)    FULL_ID=$(submit_dep    "full    " profiling/runners/run_full.sh)     ;;
    esac
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat <<EOF

[submit] Queued:
  smoke    : $SMOKE_ID   (runs immediately when a GPU is free)
EOF
[[ -n $QUICK_ID   ]] && echo "  quick    : $QUICK_ID   (waits for smoke)  — R0..R5 in ~30 min"
[[ -n $IDLE_ID    ]] && echo "  idle_tax : $IDLE_ID   (waits for smoke)  — CUDA-graph gate in ~15 min"
[[ -n $STORAGE_ID ]] && echo "  storage  : $STORAGE_ID   (waits for smoke)  — 5 storage variants × HS in ~2 h"
[[ -n $SERVE_ID   ]] && echo "  serve    : $SERVE_ID   (waits for smoke)  — asyncio-jam reproducer in ~30 min"
[[ -n $FULL_ID    ]] && echo "  full     : $FULL_ID   (waits for smoke)  — overnight cartesian in ~6–8 h"

cat <<EOF

Track:
  bjobs
  bpeek $SMOKE_ID
  tail -f profiling/runners/logs/smoke.${SMOKE_ID}.out

Cancel one:
  bkill <JOBID>
EOF

# Build the cancel-all and results-list blocks only when at least one
# dependent stage was queued — otherwise these sections render as orphans.
DEP_IDS=""
[[ -n $QUICK_ID   ]] && DEP_IDS+="$QUICK_ID "
[[ -n $IDLE_ID    ]] && DEP_IDS+="$IDLE_ID "
[[ -n $STORAGE_ID ]] && DEP_IDS+="$STORAGE_ID "
[[ -n $SERVE_ID   ]] && DEP_IDS+="$SERVE_ID "
[[ -n $FULL_ID    ]] && DEP_IDS+="$FULL_ID "
DEP_IDS=${DEP_IDS% }

if [[ -n $DEP_IDS ]]; then
    echo
    echo "Cancel everything that hasn't started yet:"
    echo "  bkill $DEP_IDS"
    echo
    echo "Results land in:"
    [[ -n $QUICK_ID   ]] && echo "  profiling/results/quick-*-${QUICK_ID}.{csv,jsonl,summary.md}"
    [[ -n $IDLE_ID    ]] && echo "  profiling/results/idle-*-${IDLE_ID}.csv"
    [[ -n $STORAGE_ID ]] && echo "  profiling/results/storage-*-${STORAGE_ID}.{csv,summary.md,heatmap.png}"
    [[ -n $SERVE_ID   ]] && echo "  profiling/results/serve-*-${SERVE_ID}.csv"
    [[ -n $FULL_ID    ]] && echo "  profiling/results/full-*-${FULL_ID}.{csv,summary.md}"
fi

cat <<EOF

If smoke FAILS the dependents stay pending forever — bkill them, fix the
issue (most often: VLLM_HOOK_PROFILE not reaching the worker subprocess),
then re-run submit_all.sh.

Subset next time with the STAGES env var:
  STAGES="quick idle"    bash profiling/runners/submit_all.sh   # daily loop
  STAGES="storage serve" bash profiling/runners/submit_all.sh   # episodic
  STAGES="full"          bash profiling/runners/submit_all.sh   # overnight
  STAGES=""              bash profiling/runners/submit_all.sh   # smoke only
EOF
