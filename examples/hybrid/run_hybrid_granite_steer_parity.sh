#!/bin/bash
# run_hybrid_granite_steer_parity.sh — DEFINITIVE graph-vs-eager steering parity on
# Granite-3.1-8B (the real model the demos use), covering PREFILL and DECODE. Reuses
# the Qwen2 parity harnesses with VLLM_HOOK_DEMO_MODEL pointed at granite, so the
# granite adjust_rs config + dummy vector drive the steer. PASS proves Hybrid-mode
# steering reproduces eager steering on an 8B model, not just Qwen2-1.5B.
#
# Submit:  bsub < examples/hybrid/run_hybrid_granite_steer_parity.sh
# Result:  grep 'VERDICT' tests/cuda_graph/logs/steer_parity*granite*.out
#
#BSUB -J vllm_hook_granite_steer_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o tests/cuda_graph/logs/steer_parity_granite.%J.out
#BSUB -e tests/cuda_graph/logs/steer_parity_granite.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="ibm-granite/granite-3.1-8b-instruct"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/steer_granite_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PRE=tests/cuda_graph/tests/steer_graph/steer_parity.py
DEC=tests/cuda_graph/tests/steer_graph/steer_parity_decode.py

echo "[granite-steer-parity] ===== PREFILL (max_tokens=1) ====="
echo "[granite-steer-parity] 1/3 graph"
python -u "$PRE" capture --mode graph --out "$WORK/pre_graph.pkl"
echo "[granite-steer-parity] 2/3 eager"
python -u "$PRE" capture --mode eager --out "$WORK/pre_eager.pkl"
echo "[granite-steer-parity] 3/3 compare"
python -u "$PRE" compare --graph "$WORK/pre_graph.pkl" --eager "$WORK/pre_eager.pkl"

echo "[granite-steer-parity] ===== DECODE (max_tokens=12) ====="
export VLLM_HOOK_PARITY_MAX_TOKENS=12
echo "[granite-steer-parity] 1/3 graph"
python -u "$DEC" capture --mode graph --out "$WORK/dec_graph.pkl"
echo "[granite-steer-parity] 2/3 eager"
python -u "$DEC" capture --mode eager --out "$WORK/dec_eager.pkl"
echo "[granite-steer-parity] 3/3 compare"
python -u "$DEC" compare --graph "$WORK/dec_graph.pkl" --eager "$WORK/dec_eager.pkl"
