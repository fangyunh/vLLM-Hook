#!/bin/bash
# run_hs_parity_full.sh — graph-vs-eager HIDDEN-STATE parity under FULL_DECODE_ONLY
# (v0.4.0 M4 FULL): eager prefill + FULL cudagraph decode, BUFFER capture mode. The
# capture_hs scatter declares NO splitting op, so it is absorbed into the single full
# decode cudagraph and replays every decode step; routing/egress is host Python in the
# execute_model wrapper, outside the graph. HS has no prefix-K, so BOTH hook configs
# compare cleanly token-for-token (no isolation crutch needed — contrast QK k_all).
#
# Two hook-activity configs, each its own leg in ONE job:
#   leg A (prefill-only, last_token): prefill captured (eager path); every decode token
#       routes to the sentinel -> the baked scatter is a true no-op. Proves the full-graph
#       decode op costs nothing / corrupts nothing; prefill parity holds. (config 1)
#   leg B (both, all_tokens):        eager-prefill capture + full-graph-decode capture,
#       parity vs eager for every captured token (prefill + 12 decode steps). Strict
#       full-shape compare (MAX_TOKENS>1) -> a dropped decode token FAILS loudly. (config 2)
#
# Submit:  bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity_full.sh
# Result:  grep 'hs-parity\] VERDICT' tests/cuda_graph/logs/hs_parity_full.*.out
#          grep 'buffer-mode HS capture wired\|HS static buffers' tests/cuda_graph/logs/hs_parity_full.*.out
#
#BSUB -J vllm_hook_hs_parity_full
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_parity_full.%J.out
#BSUB -e tests/cuda_graph/logs/hs_parity_full.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill, full cudagraph decode.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/hs_parity_full_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/hs_graph/hs_parity.py

echo "[run_hs_parity_full] cudagraph_mode=FULL_DECODE_ONLY capture=buffer"

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  echo "============================================================"
  echo "[run_hs_parity_full] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_hs_parity_full] === ${name} 1/3 graph capture (FULL_DECODE_ONLY, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_HS_CAPTURE=buffer \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_hs_parity_full] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_HS_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_hs_parity_full] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_hs_parity_full] LEG $name FAILED"
}

# Leg A: prefill-only, last_token — decode is a baked no-op under the full decode graph.
run_leg prefillonly_lasttok "model_configs/hidden_states/Qwen2-1.5B-Instruct.json" prefill 12

# Leg B: both (prefill+decode), all_tokens over all layers — full-graph decode capture.
run_leg both_alltok "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json" both 12

echo "[run_hs_parity_full] DONE — see VERDICT lines above per leg."
