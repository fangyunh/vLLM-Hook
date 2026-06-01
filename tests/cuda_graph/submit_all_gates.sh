#!/bin/bash
# submit_all_gates.sh — login-node helper that submits the three gates
# with LSF job dependencies so the cheaper kill-switches gate the rest.
#
# Run this from the project root on the LSF login node, NOT inside a job:
#   bash tests/cuda_graph/submit_all_gates.sh
#
# Topology:
#       step0  (decisive)
#         ├── step0_5   (mode matrix, runs only if step0 finished cleanly)
#         └── step0_7   (idle-tax, runs only if step0 finished cleanly)
#
# step0_5 and step0_7 run in parallel after step0 completes. Use `bjobs`
# to track; `bpeek <JOBID>` to peek at running output; `bkill <JOBID>` to
# cancel any of them.

set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

# Submit step0 and capture its job ID. bsub prints "Job <NNNN> is submitted..."
echo "[submit] step0 (injection)..."
STEP0_ID=$(bsub < tests/cuda_graph/run_step0_injection.sh \
            | awk '{print $2}' | tr -d '<>')
echo "[submit] step0 job id: $STEP0_ID"

# Both downstream gates wait for step0 to finish. We use done(<id>) (not
# ended() — we don't want the downstream to start if step0 errored).
DEP="done($STEP0_ID)"

echo "[submit] step0_5 (mode matrix), depends on done($STEP0_ID)..."
STEP0_5_ID=$(bsub -w "$DEP" < tests/cuda_graph/run_step0_5_mode_matrix.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_5 job id: $STEP0_5_ID"

echo "[submit] step0_7 (idle-tax), depends on done($STEP0_ID)..."
STEP0_7_ID=$(bsub -w "$DEP" < tests/cuda_graph/run_step0_7_idle_tax.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_7 job id: $STEP0_7_ID"

cat <<EOF

[submit] All three gates queued.
  step0    : $STEP0_ID   (runs immediately when a GPU is free)
  step0_5  : $STEP0_5_ID (waits for step0 done)
  step0_7  : $STEP0_7_ID (waits for step0 done)

Track:
  bjobs
  bpeek $STEP0_ID
  tail -f tests/cuda_graph/logs/step0.${STEP0_ID}.out

Cancel a gate:
  bkill <JOBID>

If step0 FAILS (non-zero exit), the LSF dependency 'done(...)' keeps
step0_5 and step0_7 pending forever — bkill them and diagnose step0
first.
EOF
