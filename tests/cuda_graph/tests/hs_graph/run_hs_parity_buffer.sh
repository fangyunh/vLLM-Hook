#!/bin/bash
# run_hs_parity_buffer.sh — graph-vs-eager HIDDEN-STATE parity for BUFFER capture mode
# under PIECEWISE (v0.4.0 M4 / M1-equivalent for HS). HS had NO buffer-mode path before
# v0.4.0 — this exercises the new capture_hs scatter (HSHookHost static hs_buf + device
# routing), the FULL-ready analogue of capture_qk. NO splitting op is declared (the op is
# absorbed into the cudagraph), so this also confirms the op replays under PIECEWISE decode.
# Unlike QK there is no prefix-K: the residual stream is per-token, so the scattered rows
# ARE the artifact — both hook configs compare cleanly.
#
# Two legs in ONE job:
#   leg A (prefill-only, last_token): default config layers [1,2,3,4].
#   leg B (both, all_tokens):        all layers, prefill + every decode step (max_tokens=12).
# Each leg: graph (PIECEWISE, buffer) vs eager (register_forward_hook) -> compare.
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity_buffer.sh
# Result:  grep 'hs-parity\] VERDICT' tests/cuda_graph/logs/hs_parity_buffer.*.out
#          grep 'buffer-mode HS capture wired\|HS static buffers' tests/cuda_graph/logs/hs_parity_buffer.*.out
#
#BSUB -J vllm_hook_hs_parity_buffer
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_parity_buffer.%J.out
#BSUB -e tests/cuda_graph/logs/hs_parity_buffer.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
# The path under test: static-buffer + device-routing HS scatter.
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/hs_parity_buffer_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/hs_graph/hs_parity.py

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  echo "============================================================"
  echo "[run_hs_parity_buffer] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_hs_parity_buffer] === ${name} 1/3 graph capture (PIECEWISE, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_HS_CAPTURE=buffer \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_hs_parity_buffer] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_HS_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_hs_parity_buffer] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_hs_parity_buffer] LEG $name FAILED"
}

# Leg A: prefill-only, last_token (default config, layers [1,2,3,4]).
run_leg prefillonly_lasttok "model_configs/hidden_states/Qwen2-1.5B-Instruct.json" prefill 1

# Leg B: both (prefill+decode), all_tokens over all layers.
run_leg both_alltok "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json" both 12

echo "[run_hs_parity_buffer] DONE — see VERDICT lines above per leg."
