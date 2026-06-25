#!/bin/bash
# run_cb_oom.sh — proves the v0.5.x continuous-batching OOM fix (Stage 1 continuous
# async drain; Stage 3 resident-ceiling admission throttle). The feature is OPT-IN
# (default OFF); these probes enable it via explicit envs (DRAIN_BYTES / RESIDENT_CEILING),
# which is what production sets as VLLM_HOOK_CAPTURE_STREAMING=1. See
# docs/v0.5.x_cb_oom_plan.md "Test / verify plan".
#
# Three probes, each engine isolated in its own subprocess (GPU is exclusive_process):
#   (a) PARITY   — capture+eager parity with the Stage-1 drain FIRING, per mode
#                  (all_tokens, last_token). Proves the drain is value-preserving.
#   (b) GPUBOUND — high-concurrency long-decode all_tokens capture; reads PEAK
#                  GPU-resident capture bytes with the drain OFF (held to flush ->
#                  grows) vs ON (continuous drain -> bounded), then contrasts.
#   (c) THROTTLE — tiny resident ceiling; asserts capture.throttled>0 (no crash) and
#                  the admitted request still matches eager (whole-request throttle).
#
# Submit:  bsub < tests/cuda_graph/tests/cb_oom/run_cb_oom.sh
# Result:  grep -E 'VERDICT|peak_GPU|throttled|num_drains' tests/cuda_graph/logs/cb_oom.*.out
#
#BSUB -J vllm_hook_cb_oom
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/cb_oom.%J.out
#BSUB -e tests/cuda_graph/logs/cb_oom.%J.err
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
# Workload sizing (override per node if needed).
export VLLM_HOOK_CB_REQS="${VLLM_HOOK_CB_REQS:-16}"
export VLLM_HOOK_CB_DECODE="${VLLM_HOOK_CB_DECODE:-48}"
export VLLM_HOOK_CB_PROMPT_TOKS="${VLLM_HOOK_CB_PROMPT_TOKS:-96}"
export VLLM_HOOK_CB_THROTTLE_REQS="${VLLM_HOOK_CB_THROTTLE_REQS:-8}"
export VLLM_HOOK_CB_CEILING="${VLLM_HOOK_CB_CEILING:-1024}"
# Keep the QK/drain debug prints so the log shows drains + throttle firing.
export VLLM_HOOK_QK_DEBUG="${VLLM_HOOK_QK_DEBUG:-1}"
export VLLM_HOOK_DRAIN_DEBUG="${VLLM_HOOK_DRAIN_DEBUG:-1}"

WORK="/dev/shm/vllm_hook_${USER}/cb_oom_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/cb_oom/cb_oom_parity.py

ALLTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
LASTTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct.json"

# ============================================================
# (a) PARITY — Stage-1 drain ON is value-preserving (per mode)
# ============================================================
parity_mode () {
  local tag="$1"; local cfg="$2"
  echo "============================================================"
  echo "[run_cb_oom] (a) PARITY  MODE=$tag  config=$cfg"
  echo "============================================================"
  local G="$WORK/${tag}_graph.pkl"
  local E="$WORK/${tag}_eager.pkl"
  echo "[run_cb_oom] === $tag 1/3 graph capture (FULL_DECODE_ONLY, buffer, drain ON) ==="
  # Force a small drain budget so the Stage-1 drain DEFINITELY fires (a superset of the
  # 0.05 auto default), proving value-preservation is non-vacuous.
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=buffer \
    VLLM_HOOK_CAPTURE_DRAIN_BYTES="${VLLM_HOOK_CB_PARITY_DRAIN_BYTES:-65536}" \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" parity --mode graph --out "$G" \
    || { echo "[run_cb_oom] MODE=$tag GRAPH CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_cb_oom] === $tag 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" parity --mode eager --out "$E" \
    || { echo "[run_cb_oom] MODE=$tag EAGER CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_cb_oom] === $tag 3/3 compare ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" python -u "$PY" parity-compare --graph "$G" --eager "$E" \
    || echo "[run_cb_oom] MODE=$tag parity-compare returned non-zero (see VERDICT above)"
}

parity_mode parity_alltok    "$ALLTOK"
parity_mode parity_lasttoken "$LASTTOK"

# ============================================================
# (b) GPUBOUND — peak GPU-resident capture: drain OFF vs ON
# ============================================================
echo "============================================================"
echo "[run_cb_oom] (b) GPUBOUND  reqs=$VLLM_HOOK_CB_REQS decode=$VLLM_HOOK_CB_DECODE all_tokens"
echo "============================================================"
GOFF="$WORK/gpu_off.pkl"
GON="$WORK/gpu_on.pkl"
echo "[run_cb_oom] === gpuprobe stage1=OFF (drain disabled, clones held to flush) ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" VLLM_HOOK_QK_CAPTURE=buffer \
  python -u "$PY" gpuprobe --stage1 off --out "$GOFF" \
  || echo "[run_cb_oom] gpuprobe OFF FAILED"
echo "[run_cb_oom] === gpuprobe stage1=ON (continuous drain -> pinned host) ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" VLLM_HOOK_QK_CAPTURE=buffer \
  python -u "$PY" gpuprobe --stage1 on --out "$GON" \
  || echo "[run_cb_oom] gpuprobe ON FAILED"
echo "[run_cb_oom] === gpu-compare ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" python -u "$PY" gpu-compare --off "$GOFF" --on "$GON" \
  || echo "[run_cb_oom] gpu-compare returned non-zero (see VERDICT above)"

# ============================================================
# (c) THROTTLE — Stage 3 resident-ceiling admission throttle
# ============================================================
echo "============================================================"
echo "[run_cb_oom] (c) THROTTLE  ceiling=$VLLM_HOOK_CB_CEILING B  reqs=$VLLM_HOOK_CB_THROTTLE_REQS"
echo "============================================================"
TGR="$WORK/throttle_graph.pkl"
TEA="$WORK/throttle_eager.pkl"
echo "[run_cb_oom] === throttle (graph, tiny ceiling, profile on) ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" VLLM_HOOK_QK_CAPTURE=buffer \
  python -u "$PY" throttle --out "$TGR" \
  || echo "[run_cb_oom] throttle leg FAILED"
echo "[run_cb_oom] === throttle-eager (ground truth, same single prompt) ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" VLLM_HOOK_QK_CAPTURE=op \
  python -u "$PY" throttle-eager --out "$TEA" \
  || echo "[run_cb_oom] throttle-eager FAILED"
echo "[run_cb_oom] === throttle-compare ==="
VLLM_HOOK_CONFIG_FILE="$ALLTOK" python -u "$PY" throttle-compare --throttle "$TGR" --eager "$TEA" \
  || echo "[run_cb_oom] throttle-compare returned non-zero (see VERDICT above)"

echo "[run_cb_oom] DONE — VERDICT lines above (PARITY x2, GPUBOUND, THROTTLE)."
