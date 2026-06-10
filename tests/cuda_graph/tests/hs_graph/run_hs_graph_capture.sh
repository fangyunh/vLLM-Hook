#!/bin/bash
# run_hs_graph_capture.sh — v0.3.0 hidden-states CUDA-graph capture verification,
# run as a BATCH LSF job. HS analogue of run_qk_graph_capture.sh.
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_graph_capture.sh
# Result:  grep 'VERDICT' tests/cuda_graph/logs/hs_graph.*.out
#
#BSUB -J vllm_hook_hs_graph
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_graph.%J.out
#BSUB -e tests/cuda_graph/logs/hs_graph.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
# REQUIRED: keep enforce_eager=False AND arm the v0.3.0 graph install path.
export VLLM_HOOK_ALLOW_CUDAGRAPH=1
# Fresh compile each run so a stale cache can't mask a regression.
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[run_hs_graph] model=$VLLM_HOOK_DEMO_MODEL cudagraph_mode=$VLLM_HOOK_CUDAGRAPH_MODE"
python -u tests/cuda_graph/tests/hs_graph/test_hs_graph_capture.py
