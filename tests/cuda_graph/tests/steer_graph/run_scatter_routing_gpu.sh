#!/bin/bash
# run_scatter_routing_gpu.sh — Lever A Task A2 GPU gate: graph/steer_routing_gpu.scatter_routing
# (Triton + torch backends) is byte-identical to the host apply_incremental_routing across the
# Phase-0 matrix + churn, and its launch cost is a small constant that does not scale with N/B.
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/steer_graph/run_scatter_routing_gpu.sh
# Result:  grep -E 'VERDICT|launch cost|PASS|FAIL' tests/cuda_graph/logs/scatter_routing_gpu.*.out
#
#BSUB -J vllm_hook_scatter_routing_gpu
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/scatter_routing_gpu.%J.out
#BSUB -e tests/cuda_graph/logs/scatter_routing_gpu.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1

cd tests/cuda_graph/tests/steer_graph
python -u test_scatter_routing_gpu.py
