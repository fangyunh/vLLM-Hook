#!/bin/bash
# run_perf_v055.sh — v0.5.5 profiling: (1) fixed-shape single-stream NO-REGRESSION check
# vs the v0.5.4 HTML numbers (graph capture-on-decode ms/step, default per-request egress,
# QK 21.7 / HS 14.1 graph), and (2) the Lever A high-concurrency WIN — decode ms/step for
# an N-request concurrent batch with batched egress OFF (per-request) vs ON (Lever A
# reduce-then-free). Lever A only touches the batched path, so (1) confirms the proven
# single-stream path is unchanged and (2) shows where v0.5.5 actually improves.
#
# Submit:  bsub < tests/cuda_graph/tests/capture_perf/run_perf_v055.sh
# Result:  grep -E 'VERDICT|ms/step|Lever A vs' tests/cuda_graph/logs/perf_v055.*.out
#
#BSUB -J vllm_hook_perf_v055
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/perf_v055.%J.out
#BSUB -e tests/cuda_graph/logs/perf_v055.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_PERF_REPEATS="${VLLM_HOOK_PERF_REPEATS:-3}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/perf_v055_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY1=tests/cuda_graph/tests/capture_perf/capture_perf.py        # single-stream, graph vs eager
PY2=tests/cuda_graph/tests/capture_perf/capture_perf_cb.py     # multi-request, batched off vs on

# ============================================================
# (1) FIXED-SHAPE single-stream — NO-REGRESSION vs v0.5.4 (graph vs eager ms/step)
# ============================================================
fixed_worker () {
  local w="$1" cfg="$2"
  local g="$WORK/${w}_graph.json" e="$WORK/${w}_eager.json"
  echo "============================================================"
  echo "[perf_v055] (1) FIXED-SHAPE single-stream  WORKER=$w  (compare graph ms/step to v0.5.4)"
  echo "============================================================"
  VLLM_HOOK_PERF_MAX_TOKENS=128 VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" \
    python -u "$PY1" measure --mode graph --out "$g"
  VLLM_HOOK_PERF_MAX_TOKENS=128 VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" \
    python -u "$PY1" measure --mode eager --out "$e"
  python -u "$PY1" report --graph "$g" --eager "$e" || echo "[perf_v055] $w fixed REPORT FAILED"
}

fixed_worker qk "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
fixed_worker hs "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"

# ============================================================
# (2) MULTI-REQUEST — Lever A WIN: decode ms/step at N concurrent, batched 0 vs 1
# ============================================================
export VLLM_HOOK_PERF_NREQ="${VLLM_HOOK_PERF_NREQ:-16}"
export VLLM_HOOK_PERF_MAX_TOKENS=64

lever_worker () {
  local w="$1" cfg="$2"
  local off="$WORK/${w}_cb_off.json" on="$WORK/${w}_cb_on.json"
  echo "============================================================"
  echo "[perf_v055] (2) LEVER A high-concurrency  WORKER=$w  N=$VLLM_HOOK_PERF_NREQ"
  echo "============================================================"
  echo "[perf_v055] === $w batched=0 (per-request egress) ==="
  VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_BATCHED_EGRESS=0 \
    python -u "$PY2" measure --out "$off"
  echo "[perf_v055] === $w batched=1 (Lever A reduce-then-free) ==="
  VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_BATCHED_EGRESS=1 \
    python -u "$PY2" measure --out "$on"
  echo "[perf_v055] === $w report ==="
  python -u "$PY2" report --off "$off" --on "$on" || echo "[perf_v055] $w lever REPORT FAILED"
}

lever_worker qk "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
lever_worker hs "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"

echo "[perf_v055] DONE — see VERDICT + ms/step lines above (fixed-shape x2, Lever A x2)."
