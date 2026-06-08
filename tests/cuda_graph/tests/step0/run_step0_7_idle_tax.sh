#!/bin/bash
# run_step0_7_idle_tax.sh — Gate 3 of the CUDA-graph plan.
# Measures per-layer launch overhead in the loaded-but-idle state.
# Independent of Gate 2; can run in parallel with step0_5 once step0 passes.
#
# Submit:  bsub < tests/cuda_graph/run_step0_7_idle_tax.sh
# Logs:    tests/cuda_graph/logs/step0_7.<JOBID>.{out,err}
#
# Boots vLLM twice (baseline + wrapped); wall-clock ~2x step0.
#BSUB -J vllm_hook_step0_7
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/step0_7.%J.out
#BSUB -e tests/cuda_graph/logs/step0_7.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
NUM_TOKENS="${NUM_TOKENS:-128}"
BATCHES="${BATCHES:-1 8 64}"

mkdir -p tests/cuda_graph/logs

echo "[step0.7] model=$MODEL  num_decode_tokens=$NUM_TOKENS  batches=$BATCHES"
# shellcheck disable=SC2086  # $BATCHES intentionally splits on whitespace
python tests/cuda_graph/step0_7_idle_tax.py \
    --model "$MODEL" \
    --num-decode-tokens "$NUM_TOKENS" \
    --batches $BATCHES
