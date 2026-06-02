#!/bin/bash
# run_serve.sh — serve-path throughput. Reproduces the plan.html §7 asyncio jam.
#
# Auto-launches `vllm serve` per worker, drives k concurrent clients,
# tears down. Watch gen_lat_p99 collapse as concurrency rises for the
# QK+all_tokens worker.
#
# Submit:  bsub < profiling/runners/run_serve.sh
# Logs:    profiling/runners/logs/serve.<JOBID>.{out,err}
#
# Override:
#   WORKERS="baseline qk hidden_states" CONCURRENCY="1 2 4 8" DURATION=30 \
#       bsub < profiling/runners/run_serve.sh
#BSUB -J vllm_hook_prof_serve
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[ngpus=1,mem=48GB] span[hosts=1]"
#BSUB -W 3:00
#BSUB -o profiling/runners/logs/serve.%J.out
#BSUB -e profiling/runners/logs/serve.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs profiling/results

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
WORKERS="${WORKERS:-baseline qk hidden_states}"
CONCURRENCY="${CONCURRENCY:-1 2 4 8}"
DURATION="${DURATION:-30}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PORT="${PORT:-8770}"
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook_${USER:-noname}_${LSB_JOBID:-$$}}"
PROFILE_DIR="${PROFILE_DIR:-/dev/shm/vllm_hook_profile_serve_${USER:-noname}_${LSB_JOBID:-$$}}"

mkdir -p "$HOOK_DIR" "$PROFILE_DIR"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"

TAG="${MODEL//\//_}-${LSB_JOBID:-$(date +%Y%m%d-%H%M)}"
OUT_CSV="profiling/results/serve-${TAG}.csv"

# Comma-join WORKERS for the CLI flag (the CLI expects "a,b,c").
WORKERS_CSV="$(echo "$WORKERS" | tr ' ' ',')"

echo "[serve] model=$MODEL  workers=$WORKERS_CSV  out=$OUT_CSV"
python -u profiling/bench_throughput_serve.py \
    --model       "$MODEL" \
    --workers     "$WORKERS_CSV" \
    --concurrency $CONCURRENCY \
    --duration    "$DURATION" \
    --max-tokens  "$MAX_TOKENS" \
    --port        "$PORT" \
    --hook-dir    "$HOOK_DIR" \
    --output      "$OUT_CSV"

echo
echo "[serve] CSV → $OUT_CSV"
