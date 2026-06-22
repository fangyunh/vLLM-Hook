#!/bin/bash
# run_steer_parity_decode.sh — graph-vs-eager activation-steering parity covering
# PREFILL **and DECODE**. Steering analogue of run_hs_parity_decode.sh: generates
# multiple tokens so the steer_residual op must steer on every decode step, then
# compares the graph-steered vs eager-steered token sequences for an exact match.
#
# Submit:  bsub < tests/cuda_graph/tests/steer_graph/run_steer_parity_decode.sh
# Result:  grep 'steer-decode] VERDICT' tests/cuda_graph/logs/steer_parity_decode.*.out
#
#BSUB -J vllm_hook_steer_decode
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/steer_parity_decode.%J.out
#BSUB -e tests/cuda_graph/logs/steer_parity_decode.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_STEER_COEFF="${VLLM_STEER_COEFF:-20.0}"
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-12}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/steer_parity_decode_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"
PY=tests/cuda_graph/tests/steer_graph/steer_parity_decode.py

echo "[run_steer_decode] max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS coeff=$VLLM_STEER_COEFF"
echo "[run_steer_decode] === 1/3 graph capture (prefill+decode) ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_steer_decode] === 2/3 eager capture (prefill+decode) ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_steer_decode] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
