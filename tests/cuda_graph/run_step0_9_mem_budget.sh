#!/bin/bash
# run_step0_9_mem_budget.sh — memory-budget gate.
# Measures the static-buffer VRAM footprint, its KV-cache impact
# (num_gpu_blocks baseline vs with-buffers), and footprint boundedness under
# request churn. Complements step0_7 (which measures the LATENCY idle tax).
#
# Submit:  bsub < tests/cuda_graph/run_step0_9_mem_budget.sh
# Logs:    tests/cuda_graph/logs/step0_9.<JOBID>.{out,err}
#BSUB -J vllm_hook_step0_9
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/step0_9.%J.out
#BSUB -e tests/cuda_graph/logs/step0_9.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
POLICY="${POLICY:-last_token}"          # last_token (cheap default) | all_tokens (worst case)
CHURN="${CHURN:-200}"
MODE="${MODE:-PIECEWISE}"
MIN_KV_RETENTION="${MIN_KV_RETENTION:-0.90}"
MAX_DRIFT_PCT="${MAX_DRIFT_PCT:-2.0}"

mkdir -p tests/cuda_graph/logs

echo "[step0.9] model=$MODEL policy=$POLICY churn=$CHURN mode=$MODE " \
     "min_kv_retention=$MIN_KV_RETENTION max_drift_pct=$MAX_DRIFT_PCT"

python tests/cuda_graph/step0_9_mem_budget.py \
    --model "$MODEL" \
    --policy "$POLICY" \
    --churn-requests "$CHURN" \
    --cudagraph-mode "$MODE" \
    --min-kv-retention "$MIN_KV_RETENTION" \
    --max-drift-pct "$MAX_DRIFT_PCT"
