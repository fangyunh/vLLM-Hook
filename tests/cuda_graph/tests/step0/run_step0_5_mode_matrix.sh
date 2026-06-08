#!/bin/bash
# run_step0_5_mode_matrix.sh — Gate 2 of the CUDA-graph plan.
# Sweeps CUDAGraphMode ∈ {PIECEWISE, FULL_DECODE_ONLY, FULL_AND_PIECEWISE}
# to settle C5 per-mode. Run only after Gate 1 (step0) passes.
#
# Submit:  bsub < tests/cuda_graph/run_step0_5_mode_matrix.sh
# Logs:    tests/cuda_graph/logs/step0_5.<JOBID>.{out,err}
#
# Note: this gate boots vLLM 3 times in fresh subprocesses, so wall-clock
# is ~3x step0. Time budget bumped accordingly.
#BSUB -J vllm_hook_step0_5
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/step0_5.%J.out
#BSUB -e tests/cuda_graph/logs/step0_5.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
NUM_TOKENS="${NUM_TOKENS:-32}"

mkdir -p tests/cuda_graph/logs

echo "[step0.5] model=$MODEL  num_decode_tokens=$NUM_TOKENS"
python tests/cuda_graph/step0_5_mode_matrix.py \
    --model "$MODEL" \
    --num-decode-tokens "$NUM_TOKENS"
