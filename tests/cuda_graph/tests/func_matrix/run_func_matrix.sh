#!/bin/bash
# run_func_matrix.sh — v0.5.0 Step 4 functionality matrix under FULL_DECODE_ONLY.
# Proves the per-step optimizations (W1 steer invalidation, W3 capture incremental routing)
# hold for MULTI-REQUEST batches, graph-vs-eager, in two scenarios:
#   mixed — 4 requests, same layers, different (hookq_mode, hooks_on) per request
#           (per-request-different capture in one graph batch; plan constraint c).
#   churn — 4 requests, same config, different max_tokens [4,8,12,16] so requests finish
#           at different steps -> input_batch condense + composition change (constraint d).
# PASS = graph matches eager per (request, layer) within rtol=atol=1e-2 (QK q + k_all).
#
# Submit:  bsub < tests/cuda_graph/tests/func_matrix/run_func_matrix.sh
# Result:  grep 'func-matrix.*VERDICT' tests/cuda_graph/logs/func_matrix.*.out
#
#BSUB -J vllm_hook_func_matrix
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/func_matrix.%J.out
#BSUB -e tests/cuda_graph/logs/func_matrix.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_STEER_COEFF="${VLLM_STEER_COEFF:-20.0}"
export VLLM_HOOK_CONFIG_FILE="model_configs/activation_steer/Qwen2-1.5B-Instruct.json"

WORK="/dev/shm/vllm_hook_${USER}/func_matrix_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/func_matrix/func_matrix.py

# Multi-request STEERING (the W1 headline) — steer routes by flat batch position, so it
# is multi-request-safe (unlike QK absolute-position k_buf accumulation, tracked as M5).
run_scenario () {
  local s="$1"
  local g="$WORK/${s}_graph.pkl" e="$WORK/${s}_eager.pkl"
  echo "============================================================"
  echo "[run_func] SCENARIO=$s"
  echo "============================================================"
  echo "[run_func] === $s 1/3 graph steer (buffer, FULL_DECODE_ONLY) ==="
  VLLM_HOOK_FUNC_SCENARIO="$s" VLLM_HOOK_STEER_MODE=buffer \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_func] === $s 2/3 eager steer (ground truth) ==="
  VLLM_HOOK_FUNC_SCENARIO="$s" VLLM_HOOK_STEER_MODE=op \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_func] === $s 3/3 compare (per-request token-for-token) ==="
  VLLM_HOOK_FUNC_SCENARIO="$s" \
    python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_func] $s FAILED"
}

run_scenario mixed
run_scenario churn

echo "[run_func] DONE — see VERDICT lines above (mixed + churn must both PASS)."
