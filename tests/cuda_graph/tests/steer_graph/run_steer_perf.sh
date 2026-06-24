#!/bin/bash
# run_steer_perf.sh — per-decode-step STEERING performance under FULL_DECODE_ONLY
# (v0.5.0 W1/W2/W6). Proves the steering regression is fixed: graph-steer (buffer mode,
# eager prefill + FULL cudagraph decode, with W1 invalidation routing + W2/W6a
# double-buffered pinned mirror) must be FASTER per decode step than v0.2.0 eager
# (register_forward_hook every step). Complements run_steer_parity_full.sh (correctness).
#
# Each engine boots in its own subprocess and reports decode ms/step for no-hook (floor)
# and steer, computed as 1000*(T_full - T_prefill)/(N-1). VLLM_HOOK_PROFILE surfaces the
# graph.route.skip / .build counters so you can confirm W1 invalidation fires.
#
# Submit:  bsub < tests/cuda_graph/tests/steer_graph/run_steer_perf.sh
# Result:  grep 'steer-perf] VERDICT' tests/cuda_graph/logs/steer_perf.*.out
#          grep 'decode ms/step\|invalidation:' tests/cuda_graph/logs/steer_perf.*.out
#
#BSUB -J vllm_hook_steer_perf
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/steer_perf.%J.out
#BSUB -e tests/cuda_graph/logs/steer_perf.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_STEER_COEFF="${VLLM_STEER_COEFF:-20.0}"
export VLLM_HOOK_PERF_MAX_TOKENS="${VLLM_HOOK_PERF_MAX_TOKENS:-128}"
export VLLM_HOOK_PERF_REPEATS="${VLLM_HOOK_PERF_REPEATS:-3}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Count-only profiling (cheap counters: graph.route.skip / .build); no CUDA timers.
export VLLM_HOOK_PROFILE="${VLLM_HOOK_PROFILE:-1}"

WORK="/dev/shm/vllm_hook_${USER}/steer_perf_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/steer_graph/steer_perf.py
CFG="${VLLM_HOOK_CONFIG_FILE:-model_configs/activation_steer/Qwen2-1.5B-Instruct.json}"
G="$WORK/graph.json" E="$WORK/eager.json"

echo "[run_steer_perf] FULL_DECODE_ONLY N=$VLLM_HOOK_PERF_MAX_TOKENS coeff=$VLLM_STEER_COEFF cfg=$CFG"

echo "[run_steer_perf] === 1/3 graph measure (FULL_DECODE_ONLY, buffer) ==="
VLLM_HOOK_CONFIG_FILE="$CFG" VLLM_HOOK_STEER_MODE=buffer \
  python -u "$PY" measure --mode graph --out "$G"

echo "[run_steer_perf] === 2/3 eager measure (v0.2.0 baseline, register_forward_hook) ==="
VLLM_HOOK_CONFIG_FILE="$CFG" \
  python -u "$PY" measure --mode eager --out "$E"

echo "[run_steer_perf] === 3/3 report ==="
python -u "$PY" report --graph "$G" --eager "$E" || echo "[run_steer_perf] REPORT FAILED"

echo "[run_steer_perf] DONE — see VERDICT line above."
