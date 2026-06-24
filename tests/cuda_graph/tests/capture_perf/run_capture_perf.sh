#!/bin/bash
# run_capture_perf.sh — per-decode-step CAPTURE performance under FULL_DECODE_ONLY
# (v0.5.0 W3/W6d). Measures whether graph capture-on-decode (buffer mode, with Step 1's
# sync-free pinned ring + prefix-width upload and Step 2's drain) beats v0.2.0 eager
# (register_forward_hook every step), for BOTH workers (QK + HS), all_tokens / all_layers
# (the heaviest capture). Complements the parity oracles (correctness). This is the
# MEASURE-FIRST gate for W3: if graph already beats eager, W3 is mostly delivered by
# Step 1; if not, it quantifies the target for incremental routing / egress work.
#
# Submit:  bsub < tests/cuda_graph/tests/capture_perf/run_capture_perf.sh
# Result:  grep 'cap-perf.*VERDICT' tests/cuda_graph/logs/capture_perf.*.out
#          grep 'decode ms/step\|vs eager-capture' tests/cuda_graph/logs/capture_perf.*.out
#
#BSUB -J vllm_hook_capture_perf
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/capture_perf.%J.out
#BSUB -e tests/cuda_graph/logs/capture_perf.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_PERF_MAX_TOKENS="${VLLM_HOOK_PERF_MAX_TOKENS:-128}"
export VLLM_HOOK_PERF_REPEATS="${VLLM_HOOK_PERF_REPEATS:-3}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/capture_perf_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/capture_perf/capture_perf.py

run_worker () {
  local w="$1" cfg="$2"
  local g="$WORK/${w}_graph.json" e="$WORK/${w}_eager.json"
  echo "============================================================"
  echo "[run_cap_perf] WORKER=$w  cfg=$cfg"
  echo "============================================================"
  # capture_perf.py sets the per-worker buffer env (os.environ.setdefault) in graph mode.
  echo "[run_cap_perf] === $w 1/3 graph measure (buffer, FULL_DECODE_ONLY) ==="
  VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" \
    python -u "$PY" measure --mode graph --out "$g"
  echo "[run_cap_perf] === $w 2/3 eager measure (v0.2.0 register_forward_hook) ==="
  VLLM_HOOK_PERF_WORKER="$w" VLLM_HOOK_CONFIG_FILE="$cfg" \
    python -u "$PY" measure --mode eager --out "$e"
  echo "[run_cap_perf] === $w 3/3 report ==="
  python -u "$PY" report --graph "$g" --eager "$e" || echo "[run_cap_perf] $w REPORT FAILED"
}

run_worker qk "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
run_worker hs "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"

echo "[run_cap_perf] DONE — see VERDICT lines above (QK + HS)."
