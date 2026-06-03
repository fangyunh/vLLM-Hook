#!/bin/bash
# run_step0_8_plugin_install.sh — production-path injection gate.
# Validates that installing the wrap inside the WORKER at load_model (via a
# worker_cls subclass — the same process/timing the plugin's monkey-patch of
# Worker.load_model uses) lands the op in the captured graph. Complements
# step0_injection.py, which installs from the driver (single-process only).
#
# Submit:  bsub < tests/cuda_graph/run_step0_8_plugin_install.sh
# Logs:    tests/cuda_graph/logs/step0_8.<JOBID>.{out,err}
#BSUB -J vllm_hook_step0_8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/step0_8.%J.out
#BSUB -e tests/cuda_graph/logs/step0_8.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
NUM_TOKENS="${NUM_TOKENS:-64}"
MODE="${MODE:-PIECEWISE}"
ASYNC="${ASYNC:-0}"   # set ASYNC=1 to also exercise the AsyncLLM (serve-shape) path

mkdir -p tests/cuda_graph/logs

echo "[step0.8] model=$MODEL  mode=$MODE  num_decode_tokens=$NUM_TOKENS  async=$ASYNC"

EXTRA=""
if [[ "$ASYNC" == "1" ]]; then EXTRA="--async"; fi

python tests/cuda_graph/step0_8_plugin_install.py \
    --model "$MODEL" \
    --num-decode-tokens "$NUM_TOKENS" \
    --cudagraph-mode "$MODE" \
    $EXTRA
