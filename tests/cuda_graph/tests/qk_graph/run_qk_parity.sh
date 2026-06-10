#!/bin/bash
# run_qk_parity.sh — subprocess-isolated graph-vs-eager QK parity (v0.3.0 Step 1).
#
# Runs THREE sequential python processes in ONE LSF job so each engine fully
# releases the exclusive-process GPU before the next boots (the in-process
# VLLM_HOOK_PARITY arm fails with cudaErrorDevicesUnavailable):
#   1. capture --mode graph  (PIECEWISE, op-path)         -> graph.pkl
#   2. capture --mode eager  (legacy register_forward_hook) -> eager.pkl
#   3. compare graph.pkl eager.pkl                          -> PASS/FAIL
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity.sh
# Result:  grep 'parity\] VERDICT' tests/cuda_graph/logs/qk_parity.*.out
#
#BSUB -J vllm_hook_qk_parity
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
# Force a clean compile each graph run so a stale cache cannot mask a regression.
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Per-job artifact dir (node-local tmpfs); namespaced so it never collides.
WORK="/dev/shm/vllm_hook_${USER}/parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"

PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

echo "[run_qk_parity] === 1/3 graph capture ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_qk_parity] === 2/3 eager capture ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_qk_parity] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
