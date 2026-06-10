#!/bin/bash
# run_qk_prefix_dbg.sh — single-boot diagnostic for the prefix-K device-assert.
# Runs ONLY the graph-mode prefix-scenario capture with prefix-K debug logging
# (block_table geometry + block_ids min/max) and the localizing CUDA sync
# (VLLM_HOOK_QK_PREFIXK_SYNC=1) so an out-of-bounds gather surfaces at our line.
#
#BSUB -J vllm_hook_qk_prefix_dbg
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_prefix_dbg.%J.out
#BSUB -e tests/cuda_graph/logs/qk_prefix_dbg.%J.err
set -uo pipefail   # NOT -e: we EXPECT a crash; we want the log either way.

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
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
# Diagnostics:
export VLLM_HOOK_QK_DEBUG=1
export VLLM_HOOK_QK_PREFIXK_SYNC=1

WORK="/dev/shm/vllm_hook_${USER}/prefix_dbg_${LSB_JOBID:-manual}"
mkdir -p "$WORK"

echo "[run_qk_prefix_dbg] graph-mode prefix capture with prefix-K diagnostics"
python -u tests/cuda_graph/tests/qk_graph/qk_parity.py \
    capture --mode graph --scenario prefix --out "$WORK/graph.pkl" || true
echo "[run_qk_prefix_dbg] done (exit of python ignored; see prefixK debug lines above)"
