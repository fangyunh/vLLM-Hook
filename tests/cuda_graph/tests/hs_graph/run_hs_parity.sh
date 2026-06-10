#!/bin/bash
# run_hs_parity.sh — subprocess-isolated graph-vs-eager hidden-state parity.
# HS analogue of run_qk_parity.sh: three sequential processes in one job so each
# engine fully releases the exclusive-process GPU before the next boots.
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity.sh
# Result:  grep 'hs-parity] VERDICT' tests/cuda_graph/logs/hs_parity.*.out
#
#BSUB -J vllm_hook_hs_parity
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_parity.%J.out
#BSUB -e tests/cuda_graph/logs/hs_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/hs_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"
PY=tests/cuda_graph/tests/hs_graph/hs_parity.py

echo "[run_hs_parity] === 1/3 graph capture ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_hs_parity] === 2/3 eager capture ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_hs_parity] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
