#!/bin/bash
# run_hs_multireq.sh — HS MULTI-REQUEST value parity for the BATCHED egress (the only gather
# mechanism the HS path ships now — the vectorized fast path was folded into it and deleted).
# Each request in a 4-request graph batch must match its OWN solo-eager baseline
# (hidden_states + per-request layer set) — the GPU proof that the batched gather's
# per-request offsets (`red_off`) have no cross-request bleed and survive condense
# (staggered max_tokens).
#
#   both   — hooks_on=both, last_token, ALL layers, churn max_tokens [4,8,12,16]. The core
#            multi-request batched-gather proof (homogeneous, all-layer).
#   subset — hooks_on=both, last_token, every request the SAME subset [1,2,3,4]. Homogeneous
#            but not all-registered — the multi-req subset proof.
#   hetero — hooks_on=both, last_token, per-request DIFFERENT layer subsets. Heterogeneous
#            per-request layer sets under concurrency.
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_multireq.sh
# Result:  grep -E 'hs-multireq.*VERDICT' tests/cuda_graph/logs/hs_multireq.*.out
#
#BSUB -J vllm_hook_hs_multireq
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_multireq.%J.out
#BSUB -e tests/cuda_graph/logs/hs_multireq.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_CONFIG_FILE="model_configs/hidden_states/Qwen2-1.5B-Instruct_lasttok_all.json"

WORK="/dev/shm/vllm_hook_${USER}/hs_multireq_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/hs_graph/hs_multireq.py

run_scenario () {
  local s="$1"
  local g="$WORK/${s}_graph.pkl" e="$WORK/${s}_eager.pkl"
  echo "============================================================"
  echo "[run_hs_multireq] SCENARIO=$s"
  echo "============================================================"
  echo "[run_hs_multireq] === $s 1/3 graph capture (4-req batch, buffer, FULL) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" VLLM_HOOK_HS_CAPTURE=buffer \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_hs_multireq] === $s 2/3 solo-eager baseline (per request, ground truth) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" VLLM_HOOK_HS_CAPTURE=op \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_hs_multireq] === $s 3/3 compare (per-request vs solo-eager) ==="
  VLLM_HOOK_MULTIREQ_SCENARIO="$s" \
    python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_hs_multireq] $s FAILED"
}

run_scenario both      # homogeneous all-layers — the core multi-request batched-gather proof
run_scenario subset    # homogeneous SUBSET (not all-registered)
run_scenario hetero    # heterogeneous per-request layer sets under concurrency

echo "[run_hs_multireq] DONE — see VERDICT lines above (both + subset + hetero must all PASS)."
