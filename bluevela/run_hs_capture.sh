#!/bin/bash
# run_hs_capture.sh — hidden-state capture demo on BLUEVELA_LSF.
#
# Runs examples/demo_hiddenstate.py: captures hidden states from the configured
# layers and prints each layer's tensor shape and norm.
#
# Two legs in one job (HS_MODE selects):
#   eager  — forward-hook capture path (the shipped default)
#   graph  — FULL CUDA-graph capture ring (VLLM_HOOK_ALLOW_CUDAGRAPH=1)
# Same prompts, same config, so the two legs are directly comparable.
#
# Submit:  bash -lc 'bsub -G grp_exploratory < bluevela/run_hs_capture.sh'
# Watch:   bjobs ; bpeek <JOBID>
# Result:  grep -E 'mode=|model.layers' bluevela/logs/hs_capture.<JOBID>.out
#
#BSUB -J vllm_hook_hs_capture
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o bluevela/logs/hs_capture.%J.out
#BSUB -e bluevela/logs/hs_capture.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p bluevela/logs

# Node-local scratch for the JIT caches; /opt/nvme is not present on every host.
export TMPDIR="/opt/nvme/${USER}/hs_capture_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/tmp/vllm_hook_${USER}/hs_capture_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Model resolution pings the HF API even when cached; a burst of jobs gets 429'd and
# surfaces as "server failed to become ready". Offline once the model is cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export VLLM_HOOK_USE_SAFETENSORS=1
# A stale inductor cache silently poisons a graph-mode run; always compile fresh.
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_CONFIG_FILE="${VLLM_HOOK_CONFIG_FILE:-model_configs/hidden_states/Qwen2-1.5B-Instruct.json}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Counters are no-ops without this; the demo prints them at the end.
export VLLM_HOOK_PROFILE="${VLLM_HOOK_PROFILE:-1}"
export VLLM_HOOK_PROFILE_DIR="${VLLM_HOOK_PROFILE_DIR:-$TMPDIR/profile}"
mkdir -p "$VLLM_HOOK_PROFILE_DIR"

HS_MODE="${HS_MODE:-both}"     # eager | graph | both

echo "[run_hs_capture] host=$(hostname) job=${LSB_JOBID:-manual}"
echo "[run_hs_capture] branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
echo "[run_hs_capture] model=$VLLM_HOOK_DEMO_MODEL config=$VLLM_HOOK_CONFIG_FILE mode=$HS_MODE"

run_leg () {
  local name="$1" allow_graph="$2"
  echo "============================================================"
  echo "[run_hs_capture] LEG=$name VLLM_HOOK_ALLOW_CUDAGRAPH=$allow_graph"
  echo "============================================================"
  VLLM_HOOK_ALLOW_CUDAGRAPH="$allow_graph" \
    python -u examples/demo_hiddenstate.py \
    || echo "[run_hs_capture] LEG $name FAILED"
}

case "$HS_MODE" in
  eager) run_leg eager 0 ;;
  graph) run_leg graph 1 ;;
  both)  run_leg eager 0; run_leg graph 1 ;;
  *)     echo "[run_hs_capture] bad HS_MODE=$HS_MODE (want eager|graph|both)"; exit 2 ;;
esac

echo "[run_hs_capture] DONE — per-layer shapes/norms above; profile JSON in $VLLM_HOOK_PROFILE_DIR"
