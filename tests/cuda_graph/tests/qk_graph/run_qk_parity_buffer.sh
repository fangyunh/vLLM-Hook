#!/bin/bash
# run_qk_parity_buffer.sh — graph-vs-eager QK parity for BUFFER capture mode under
# PIECEWISE (v0.4.0 M1). Buffer mode (VLLM_HOOK_QK_CAPTURE=buffer) is the static-
# buffer + device-routing scatter path (capture_qk index_copy_) — the FULL-ready
# primitive. It has NEVER been parity-proven, even under PIECEWISE. This job proves
# the routing + scatter + egress chain is numerically correct BEFORE adding FULL's
# replay semantics (M2). Runs two prefill-only legs in ONE job:
#   leg A: last_token (default config, important_heads layers)
#   leg B: all_tokens (all 28 layers, full per-token q/k slice)
# Each leg: graph (PIECEWISE, buffer) vs eager (register_forward_hook) -> compare.
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity_buffer.sh
# Result:  grep 'parity\] VERDICT' tests/cuda_graph/logs/qk_parity_buffer.*.out
#          grep 'QK hosts installed\|QK static buffers' tests/cuda_graph/logs/qk_parity_buffer.*.out
#
#BSUB -J vllm_hook_qk_parity_buffer
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity_buffer.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity_buffer.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
# Force a clean compile each graph run so a stale cache cannot mask a regression
# (also keeps AOT-compile OFF; AOT+buffer is a later milestone).
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
# The path under test: static-buffer + device-routing scatter.
export VLLM_HOOK_QK_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/parity_buffer_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  echo "============================================================"
  echo "[run_qk_parity_buffer] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_qk_parity_buffer] === ${name} 1/3 graph capture (PIECEWISE, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=buffer \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_qk_parity_buffer] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_qk_parity_buffer] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_qk_parity_buffer] LEG $name FAILED"
}

# Leg A: last_token (default config), prefill-only (max_tokens=1).
run_leg last_token "model_configs/attention_tracker/Qwen2-1.5B-Instruct.json" prefill 1

# Leg B: all_tokens over all layers, prefill-only.
run_leg all_tokens "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json" prefill 1

# Leg C: all_tokens, BOTH (prefill + decode) — exercises k_buf accumulation (full
# key history) under PIECEWISE decode replay, isolating it from FULL_DECODE_ONLY.
run_leg both_alltok "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json" both 12

echo "[run_qk_parity_buffer] DONE — see VERDICT lines above per leg."
