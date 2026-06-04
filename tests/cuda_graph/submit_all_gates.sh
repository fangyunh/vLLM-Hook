#!/bin/bash
# submit_all_gates.sh — login-node helper that submits the gates
# with LSF job dependencies so the cheaper kill-switch gates the rest.
#
# Run this from the project root on the LSF login node, NOT inside a job:
#   bash tests/cuda_graph/submit_all_gates.sh
#
# Topology (vLLM v1):
#       step0_8  (DECISIVE — worker-install injection at load_model)
#         ├── step0_5   (mode matrix; drives step0_8 per mode)
#         ├── step0_7   (idle-tax / LATENCY; worker-install)
#         └── step0_9   (memory budget / VRAM + KV impact)
#       step0     (INFORMATIONAL, non-blocking — driver-install probe)
#
# Why step0_8 is decisive, not step0: on vLLM v1, torch.compile + CUDA-graph
# capture happen INSIDE LLM() before the driver regains control, so the
# driver-install probe (step0_injection) is structurally too late — it always
# reports 0 fires. Only worker-install (step0_8: install at load_model, before
# compile) can land the op in the captured graph. We still run step0 for the
# record, but nothing depends on it.
#
# step0_5, step0_7 and step0_9 run in parallel once step0_8 is done. Use `bjobs`
# to track; `bpeek <JOBID>` to peek at output; `bkill <JOBID>` to cancel.

set -euo pipefail

cd "$(dirname "$0")/../.."   # project root

# Decisive gate: worker-install injection. Capture its job ID.
echo "[submit] step0_8 (DECISIVE — worker-install injection)..."
STEP0_8_ID=$(bsub < tests/cuda_graph/run_step0_8_plugin_install.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_8 job id: $STEP0_8_ID"

# Downstream gates wait for step0_8 to finish cleanly (done(), not ended()).
DEP="done($STEP0_8_ID)"

echo "[submit] step0_5 (mode matrix), depends on done($STEP0_8_ID)..."
STEP0_5_ID=$(bsub -w "$DEP" < tests/cuda_graph/run_step0_5_mode_matrix.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_5 job id: $STEP0_5_ID"

echo "[submit] step0_7 (idle-tax), depends on done($STEP0_8_ID)..."
STEP0_7_ID=$(bsub -w "$DEP" < tests/cuda_graph/run_step0_7_idle_tax.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_7 job id: $STEP0_7_ID"

echo "[submit] step0_9 (memory budget), depends on done($STEP0_8_ID)..."
STEP0_9_ID=$(bsub -w "$DEP" < tests/cuda_graph/run_step0_9_mem_budget.sh \
              | awk '{print $2}' | tr -d '<>')
echo "[submit] step0_9 job id: $STEP0_9_ID"

# Informational only — NOT a dependency for anything. Expected to report 0 fires
# on vLLM v1 (driver-install is too late); it documents that finding.
echo "[submit] step0 (informational driver-install probe; non-blocking)..."
STEP0_ID=$(bsub < tests/cuda_graph/run_step0_injection.sh \
            | awk '{print $2}' | tr -d '<>')
echo "[submit] step0 job id: $STEP0_ID"

cat <<EOF

[submit] Gates queued.
  step0_8  : $STEP0_8_ID  (DECISIVE — runs immediately when a GPU is free)
  step0_5  : $STEP0_5_ID  (waits for step0_8 done)  — mode matrix
  step0_7  : $STEP0_7_ID  (waits for step0_8 done)  — latency idle tax
  step0_9  : $STEP0_9_ID  (waits for step0_8 done)  — memory budget / KV impact
  step0    : $STEP0_ID  (informational; non-blocking; expected 0 on v1)

Track:
  bjobs
  bpeek $STEP0_8_ID
  tail -f tests/cuda_graph/logs/step0_8.${STEP0_8_ID}.out

Cancel a gate:
  bkill <JOBID>

If step0_8 FAILS (non-zero exit), the LSF dependency 'done(...)' keeps
step0_5, step0_7 and step0_9 pending forever — bkill them and diagnose
step0_8 first.
EOF
