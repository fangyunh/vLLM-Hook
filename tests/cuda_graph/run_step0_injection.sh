#!/bin/bash
# run_step0_injection.sh — Gate 1 (decisive kill-switch) of the CUDA-graph plan.
# Settles C1 (opacity) + C4 (class-level wrap survival).
#
# Submit:  bsub < tests/cuda_graph/run_step0_injection.sh
# Logs:    tests/cuda_graph/logs/step0.<JOBID>.{out,err}
#BSUB -J vllm_hook_step0
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/step0.%J.out
#BSUB -e tests/cuda_graph/logs/step0.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

# Gemma-3-4B is gated on HF; ensure auth is configured on this node.
# (Run `huggingface-cli login` once on the server, or set HF_TOKEN.)

MODEL="${MODEL:-google/gemma-3-4b-it}"
NUM_TOKENS="${NUM_TOKENS:-64}"
MODE="${MODE:-PIECEWISE}"

mkdir -p tests/cuda_graph/logs

echo "[step0] model=$MODEL  mode=$MODE  num_decode_tokens=$NUM_TOKENS"
python tests/cuda_graph/step0_injection.py \
    --model "$MODEL" \
    --num-decode-tokens "$NUM_TOKENS" \
    --cudagraph-mode "$MODE"
