#!/bin/bash
# run_steer_fused_unit.sh — standalone GPU unit test for the fused steer_buffer
# Triton kernel (VLLM_HOOK_STEER_FUSED path) vs the aten reference _steer_buffer_impl.
# No engine, no cudagraph: builds random routing tensors on the GPU and compares the
# fused kernel against the aten body row-for-row (add_vector byte-identical, adjust_rs
# within the reduction-reorder tolerance, no-op rows untouched). This is the fast
# Phase-3.3.1 gate — run it BEFORE the full parity oracle.
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/steer_graph/run_steer_fused_unit.sh
# Result:  grep 'steer-fused-unit] VERDICT' tests/cuda_graph/logs/steer_fused_unit.*.out
#
#BSUB -J vllm_hook_steer_fused_unit
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/steer_fused_unit.%J.out
#BSUB -e tests/cuda_graph/logs/steer_fused_unit.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

python -u tests/cuda_graph/tests/steer_graph/test_steer_fused_unit.py
