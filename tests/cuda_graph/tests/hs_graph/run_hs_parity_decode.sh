#!/bin/bash
# run_hs_parity_decode.sh — graph-vs-eager HIDDEN-STATE parity covering PREFILL
# **and DECODE**. HS analogue of run_qk_parity_decode.sh: generates multiple
# tokens with hooks_on=both so hs_probe must fire on every decode step. hs_parity
# compare() already requires exact full-shape equality, so a graph capture that
# dropped decode tokens FAILS as a SHAPE MISMATCH.
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity_decode.sh
# Result:  grep 'hs-parity\] VERDICT' tests/cuda_graph/logs/hs_parity_decode.*.out
#
#BSUB -J vllm_hook_hs_parity_decode
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_parity_decode.%J.out
#BSUB -e tests/cuda_graph/logs/hs_parity_decode.%J.err
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
# All-tokens, all-layers config so each step's full residual vector is captured.
export VLLM_HOOK_CONFIG_FILE="model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"
# Decode-stage knobs: multiple tokens + fire the hook during decode too.
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-16}"
export VLLM_HOOK_PARITY_HOOKS_ON="${VLLM_HOOK_PARITY_HOOKS_ON:-both}"

WORK="/dev/shm/vllm_hook_${USER}/hs_parity_decode_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"

PY=tests/cuda_graph/tests/hs_graph/hs_parity.py

echo "[run_hs_parity_decode] config=$VLLM_HOOK_CONFIG_FILE max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS hooks_on=$VLLM_HOOK_PARITY_HOOKS_ON"
echo "[run_hs_parity_decode] === 1/3 graph capture (prefill+decode) ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_hs_parity_decode] === 2/3 eager capture (prefill+decode) ==="
python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_hs_parity_decode] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
