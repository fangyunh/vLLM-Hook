#!/bin/bash
# run_qk_noneager.sh — run the QK probe demo under NON-eager (CUDA-graph) mode,
# INTERACTIVELY, so you watch the log live in your terminal.
#
# Two ways to use it:
#
# (A) One-shot interactive job (recommended — submit & watch, auto-exits):
#       bsub -Is -gpu "num=1:mode=exclusive_process" -n 4 \
#            -R "rusage[ngpus=1,mem=32GB]" \
#            tests/cuda_graph/run_qk_noneager.sh
#
# (B) Grab an interactive shell first, then run this script by hand:
#       bsub -Is -gpu "num=1:mode=exclusive_process" -n 4 \
#            -R "rusage[ngpus=1,mem=32GB]" /bin/bash
#       # ... once you land on the GPU node:
#       cd ~/vLLM-Hook && tests/cuda_graph/run_qk_noneager.sh
#
# Override anything via env, e.g.:
#       VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY tests/cuda_graph/run_qk_noneager.sh
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

# disk-st-async storage variant (fastest), matching run_demo.sh.
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1

# CUDA-graph knobs. PIECEWISE is the mandatory/shippable mode; try
# FULL_DECODE_ONLY for the higher-throughput decode path.
export VLLM_HOOK_CUDAGRAPH_MODE="${VLLM_HOOK_CUDAGRAPH_MODE:-PIECEWISE}"
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"

# Make vLLM chatty about compilation/capture so you can SEE the graphs build.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[run_qk_noneager] model=$VLLM_HOOK_DEMO_MODEL  cudagraph_mode=$VLLM_HOOK_CUDAGRAPH_MODE"
echo "[run_qk_noneager] (TORCHDYNAMO_DISABLE is forced to 0 inside the python script)"

# -u = unbuffered stdout so logs stream live in the interactive session.
python -u tests/cuda_graph/qk_noneager_demo.py
