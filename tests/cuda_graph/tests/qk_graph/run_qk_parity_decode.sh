#!/bin/bash
# run_qk_parity_decode.sh — graph-vs-eager QK parity covering PREFILL **and
# DECODE**. Same subprocess-isolated oracle as run_qk_parity_alltok.sh, but
# generates multiple tokens with hooks_on=both so the capture hook MUST fire on
# every decode step. The compare step switches to STRICT full-shape equality
# (VLLM_HOOK_PARITY_MAX_TOKENS>1), so if graph mode captured prefill only, the
# graph q/k has fewer rows than eager and the run FAILS as a LENGTH MISMATCH.
#
# This is the test that turns "decode capture is implemented" into "decode
# capture is verified" — the splitting-op seam (vllm_hook::qk_probe) must run
# eagerly every decode step under PIECEWISE and match the eager ground truth.
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity_decode.sh
# Result:  grep 'parity\] VERDICT' tests/cuda_graph/logs/qk_parity_decode.*.out
#
#BSUB -J vllm_hook_qk_parity_decode
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity_decode.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity_decode.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# All-tokens, all-layers config so each step's full q/k slice is captured.
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
# Decode-stage knobs: multiple tokens + fire the hook during decode too.
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-16}"
export VLLM_HOOK_PARITY_HOOKS_ON="${VLLM_HOOK_PARITY_HOOKS_ON:-both}"

WORK="/dev/shm/vllm_hook_${USER}/parity_decode_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"

PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

echo "[run_qk_parity_decode] config=$VLLM_HOOK_CONFIG_FILE max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS hooks_on=$VLLM_HOOK_PARITY_HOOKS_ON"
echo "[run_qk_parity_decode] === 1/3 graph capture (prefill+decode) ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_qk_parity_decode] === 2/3 eager capture (prefill+decode) ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_qk_parity_decode] === 3/3 compare (strict full-shape) ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
