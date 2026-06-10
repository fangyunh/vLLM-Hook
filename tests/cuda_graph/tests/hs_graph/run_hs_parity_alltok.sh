#!/bin/bash
# run_hs_parity_alltok.sh — graph-vs-eager hidden-state parity in ALL_TOKENS mode
# over ALL layers. Stronger than the default last_token HS parity: compares the
# full per-token hidden-state sequence for every decoder layer (config has empty
# layers -> output_hidden_states=True -> all layers; mode=all_tokens).
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity_alltok.sh
#
#BSUB -J vllm_hook_hs_parity_alltok
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_parity_alltok.%J.out
#BSUB -e tests/cuda_graph/logs/hs_parity_alltok.%J.err
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
# all_tokens, all layers (the only difference from run_hs_parity.sh).
export VLLM_HOOK_CONFIG_FILE="model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"

WORK="/dev/shm/vllm_hook_${USER}/hs_parity_alltok_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"
PY=tests/cuda_graph/tests/hs_graph/hs_parity.py

echo "[run_hs_parity_alltok] config=$VLLM_HOOK_CONFIG_FILE"
echo "[run_hs_parity_alltok] === 1/3 graph capture (all_tokens) ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_hs_parity_alltok] === 2/3 eager capture (all_tokens) ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_hs_parity_alltok] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
