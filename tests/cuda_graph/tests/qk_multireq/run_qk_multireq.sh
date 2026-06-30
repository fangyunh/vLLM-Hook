#!/bin/bash
# run_qk_multireq.sh — v0.5.6 W1′ QK MULTI-REQUEST parity under FULL_DECODE_ONLY.
# Proves W1′ idle-skip preserves multi-request QK capture: each request in a 4-request
# graph batch matches its OWN solo-eager baseline (q + k_all + per-request layer set).
#
#   prefill — hooks_on=prefill, HETEROGENEOUS per-request layer sets. Every decode step
#             is idle, so W1′ idle-skip fires on all of them; correctness + constraint (c).
#   both    — hooks_on=both, all layers, churn max_tokens [4,8,12,16]. Every step active
#             (idle-skip never fires) + composition churn — active-path regression guard.
#
# Submit:  bsub < tests/cuda_graph/tests/qk_multireq/run_qk_multireq.sh
# Result:  grep 'multireq.*VERDICT' tests/cuda_graph/logs/qk_multireq.*.out
#
#BSUB -J vllm_hook_qk_multireq
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_multireq.%J.out
#BSUB -e tests/cuda_graph/logs/qk_multireq.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_QK_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"

WORK="/dev/shm/vllm_hook_${USER}/qk_multireq_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/qk_multireq/qk_multireq.py

run_scenario () {
  local s="$1"
  local g="$WORK/${s}_graph.pkl" e="$WORK/${s}_eager.pkl"
  echo "============================================================"
  echo "[run_multireq] SCENARIO=$s"
  echo "============================================================"
  echo "[run_multireq] === $s 1/3 graph capture (4-req batch, buffer, FULL_DECODE_ONLY) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" VLLM_HOOK_QK_CAPTURE=buffer \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_multireq] === $s 2/3 solo-eager baseline (per request, ground truth) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" VLLM_HOOK_QK_CAPTURE=op \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_multireq] === $s 3/3 compare (per-request vs solo-eager) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" \
    python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_multireq] $s FAILED"
}

run_scenario prefill
run_scenario both

echo "[run_multireq] DONE — see VERDICT lines above (prefill + both must both PASS)."
