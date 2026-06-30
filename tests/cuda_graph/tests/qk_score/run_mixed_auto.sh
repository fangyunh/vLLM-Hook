#!/bin/bash
# run_mixed_auto.sh — v0.5.7 D2 mixed-batch merge + auto-select integration (buffer FULL).
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/qk_score/run_mixed_auto.sh
# Result:  grep 'mixed-auto] VERDICT' tests/cuda_graph/logs/mixed_auto.*.out
#
#BSUB -J vllm_hook_mixed_auto
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/mixed_auto.%J.out
#BSUB -e tests/cuda_graph/logs/mixed_auto.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

PY=tests/cuda_graph/tests/qk_score/mixed_auto_parity.py
echo "[run_mixed_auto] === mixed-batch merge + auto-select (FULL_DECODE_ONLY) ==="
python -u "$PY"
