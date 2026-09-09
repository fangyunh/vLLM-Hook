#!/bin/bash
# run_steer_parity_full.sh — graph-vs-eager activation-steering parity under
# FULL (v0.4.0 M3): eager prefill + FULL cudagraph decode, BUFFER steering
# mode. The new vllm_hook::steer_buffer op (masked in-place residual += coeff·vec) is
# NOT a splitting op, so it is absorbed into the single full decode cudagraph and
# replays every decode step; routing (per-request coeff/vec_id) is host Python in the
# _prepare_inputs wrapper, OUTSIDE the graph. This is the milestone that proves a FULL
# graph can honor per-request config AND that an in-place residual mutation propagates
# downstream on cudagraph replay (the central worry: no Python runs on replay).
#
# Each engine boot runs base (no steer) + steer (config's optimal_layer, every step).
# PASS = graph-steered reproduces eager-steered token-for-token across all decode steps,
# AND steering is non-vacuous (steer != base). add_vector method, coeff=VLLM_STEER_COEFF.
#
# Submit:  bsub < tests/cuda_graph/tests/steer_graph/run_steer_parity_full.sh
# Result:  grep 'steer-decode] VERDICT' tests/cuda_graph/logs/steer_parity_full.*.out
#          grep 'buffer-mode steering wired\|dbg-steer' tests/cuda_graph/logs/steer_parity_full.*.out
#
#BSUB -J vllm_hook_steer_full
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/steer_parity_full.%J.out
#BSUB -e tests/cuda_graph/logs/steer_parity_full.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill, full cudagraph decode.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_STEER_COEFF="${VLLM_STEER_COEFF:-20.0}"
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-12}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/steer_parity_full_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/steer_graph/steer_parity_decode.py

echo "[run_steer_full] cudagraph_mode=FULL steer=buffer max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS coeff=$VLLM_STEER_COEFF"

run_leg () {
  local name="$1" cfg="$2"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  echo "============================================================"
  echo "[run_steer_full] LEG=$name config=$cfg"
  echo "============================================================"
  echo "[run_steer_full] === ${name} 1/3 graph capture (FULL, buffer${3:+ ; $3}) ==="
  # $3 is an optional "VAR=VAL" for the graph leg; via `env` so a runtime-expanded assignment
  # is honored (a bare ${3:-} prefix is parsed as a command, not an assignment -> 127).
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_STEER_MODE=buffer env ${3:-} \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_steer_full] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_STEER_MODE=op \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_steer_full] === ${name} 3/3 compare (token-for-token, prefill+decode) ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_steer_full] LEG $name FAILED"
}

# Legs A/B run the SHIPPED default — GPU routing is default-ON for steer
# (VLLM_HOOK_STEER_GPU_ROUTING unset), so these now exercise the GPU scatter.
# Leg A: add_vector (host-supplied coefficient * fixed vector).
run_leg add_vector "model_configs/activation_steer/Qwen2-1.5B-Instruct.json"

# Leg B: adjust_rs (in-kernel coefficient = avg_proj - residual.unit, computed at replay).
run_leg adjust_rs "model_configs/activation_steer/Qwen2-1.5B-Instruct_adjust_rs.json"

# Leg C: HOST-routing control — pins VLLM_HOOK_STEER_GPU_ROUTING=0 so the host
# apply_incremental_routing path stays proven now that GPU routing is the default
# (mirrors the batched-egress leg-C convention).
run_leg add_vector_hostroute "model_configs/activation_steer/Qwen2-1.5B-Instruct.json" \
  "VLLM_HOOK_STEER_GPU_ROUTING=0"

echo "[run_steer_full] DONE — see VERDICT lines above per leg."
