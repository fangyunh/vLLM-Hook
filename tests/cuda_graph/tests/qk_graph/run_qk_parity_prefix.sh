#!/bin/bash
# run_qk_parity_prefix.sh — graph-vs-eager QK parity WITH prefix caching active
# (num_cached > 0), exercising prefix-K reconstruction. A warm prompt populates
# the prefix cache; a target prompt that shares the warm prefix is then captured
# WITHOUT a reset, so its prefill reuses cached blocks and the hook must rebuild
# the cached prefix keys from the paged KV cache. If graph-mode reconstruction is
# broken, target k_all is shorter/different than eager's and the compare FAILs.
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity_prefix.sh
#
#BSUB -J vllm_hook_qk_parity_prefix
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity_prefix.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity_prefix.%J.err
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
# all_tokens, all layers -> full per-token k_all comparison incl. reconstructed prefix.
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"

WORK="/dev/shm/vllm_hook_${USER}/parity_prefix_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"

PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

echo "[run_qk_parity_prefix] === 1/3 graph capture (prefix scenario) ==="
python -u "$PY" capture --mode graph --scenario prefix --out "$GRAPH_PKL"

echo "[run_qk_parity_prefix] === 2/3 eager capture (prefix scenario) ==="
python -u "$PY" capture --mode eager --scenario prefix --out "$EAGER_PKL"

echo "[run_qk_parity_prefix] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
