#!/bin/bash
# run_qk_noneager.sh — run the QK probe demo under NON-eager (CUDA-graph) mode
# as a BATCH (non-interactive) LSF job. Output goes to log files, not your
# terminal.
#
# Submit:  mkdir -p tests/cuda_graph/logs
#          bsub < tests/cuda_graph/run_qk_noneager.sh
# Watch:   bjobs                                          # status
#          bpeek <JOBID>                                  # peek live output
#          tail -f tests/cuda_graph/logs/qk_noneager.*.out
#
#BSUB -J vllm_hook_qk_noneager
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_noneager.%J.out
#BSUB -e tests/cuda_graph/logs/qk_noneager.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

# disk-st-async storage variant (fastest), matching run_demo.sh.
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1

# CUDA-graph mode. Switch by commenting/uncommenting one of these lines.
# PIECEWISE       = mandatory/shippable mode; attention stays an eager seam.
# FULL_DECODE_ONLY = higher-throughput decode path; decode folds into one graph.
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
# export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY

export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"

# Make vLLM chatty about compilation/capture so the log shows the graphs build.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[run_qk_noneager] model=$VLLM_HOOK_DEMO_MODEL  cudagraph_mode=$VLLM_HOOK_CUDAGRAPH_MODE"
echo "[run_qk_noneager] (TORCHDYNAMO_DISABLE is forced to 0 inside the python script)"

# -u = unbuffered stdout so bpeek / tail -f show progress without buffering lag.
python -u tests/cuda_graph/qk_noneager_demo.py
