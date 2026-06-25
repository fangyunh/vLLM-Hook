#!/bin/bash
# run_prefix_cache_parity.sh — does buffer-mode QK capture restore FULL-context K when
# prefix caching trims a prompt's shared prefix (the continuous-batching regime)?
#
# We boot with enable_prefix_caching=True and run the SAME prompt TWICE: run1 populates
# the KV cache (fresh prefill), run2 hits the cache (num_computed>0, only the tail is
# forwarded). Pre-fix the buffer captures only the tail (UNDER-CAPTURE); post-fix it
# reads the cached prefix back from paged KV at egress, matching the eager ground truth.
#
# Per mode (all_tokens, last_token), in ONE job:
#   1. graph capture (FULL_DECODE_ONLY, buffer)  of run1(fresh)+run2(cached)
#   2. eager capture (op->register_forward_hook ground truth) of the same
#   3. compare: fresh = control (must match), cached = test (full-context K)
# VERDICT per mode: PASS (clean|numeric) | UNDER-CAPTURE | VALUE-DIVERGENCE | INCONCLUSIVE.
#
# Submit:  bsub < tests/cuda_graph/tests/prefix_cache/run_prefix_cache_parity.sh
# Result:  grep -E 'VERDICT|coverage|prefixK|UNDER-CAPTURE' tests/cuda_graph/logs/prefix_cache_parity.*.out
#
#BSUB -J vllm_hook_prefix_cache_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/prefix_cache_parity.%J.out
#BSUB -e tests/cuda_graph/logs/prefix_cache_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_PREFIX_TOKS="${VLLM_HOOK_PREFIX_TOKS:-256}"
export VLLM_HOOK_PREFIX_MNBT="${VLLM_HOOK_PREFIX_MNBT:-2048}"
# Keep the QK debug print on so the log shows the buffer prefixK reconstructions.
export VLLM_HOOK_QK_DEBUG="${VLLM_HOOK_QK_DEBUG:-1}"

WORK="/dev/shm/vllm_hook_${USER}/prefix_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/prefix_cache/prefix_cache_parity.py

run_mode () {
  local tag="$1"
  local cfg="$2"
  echo "============================================================"
  echo "[run_prefix] MODE=$tag  config=$cfg"
  echo "============================================================"
  local G="$WORK/${tag}_graph.pkl"
  local E="$WORK/${tag}_eager.pkl"
  echo "[run_prefix] === $tag 1/3 graph capture (FULL_DECODE_ONLY, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=buffer \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" capture --mode graph --out "$G" \
    || { echo "[run_prefix] MODE=$tag GRAPH CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_prefix] === $tag 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" capture --mode eager --out "$E" \
    || { echo "[run_prefix] MODE=$tag EAGER CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_prefix] === $tag 3/3 compare (fresh=control, cached=test) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" python -u "$PY" compare --graph "$G" --eager "$E" \
    || echo "[run_prefix] MODE=$tag returned non-zero (see VERDICT above)"
}

ALLTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
LASTTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct.json"

run_mode prefix_alltok    "$ALLTOK"
run_mode prefix_lasttoken "$LASTTOK"

echo "[run_prefix] DONE — VERDICT lines above."
