#!/bin/bash
# run_steer_parity.sh — subprocess-isolated graph-vs-eager activation-steering parity.
# Steering analogue of run_hs_parity.sh: three sequential processes in one job so
# each engine fully releases the exclusive-process GPU before the next boots.
#
# Submit:  bsub < tests/cuda_graph/tests/steer_graph/run_steer_parity.sh
# Result:  grep 'steer-parity] VERDICT' tests/cuda_graph/logs/steer_parity.*.out
#
#BSUB -J vllm_hook_steer_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/steer_parity.%J.out
#BSUB -e tests/cuda_graph/logs/steer_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_STEER_COEFF="${VLLM_STEER_COEFF:-20.0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/steer_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"
PY=tests/cuda_graph/tests/steer_graph/steer_parity.py

echo "[run_steer_parity] === 1/3 graph capture ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_steer_parity] === 2/3 eager capture ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_steer_parity] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
