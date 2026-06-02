#!/bin/bash
# run_idle_tax.sh — measure the per-layer launch cost of the plugin.
# Three configurations × {1, 8, 64} batch sizes, ~15 min total.
#
# PASS criterion (from cuda_graph_plan.html):
#   per-layer overhead ≤ 3 µs, aggregate idle overhead ≤ 5% at bs=1.
# Non-zero exit => FAIL on at least one cell.
#
# Submit:  bsub < profiling/runners/run_idle_tax.sh
# Logs:    profiling/runners/logs/idle.<JOBID>.{out,err}
#BSUB -J vllm_hook_prof_idle
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB] span[hosts=1]"
#BSUB -W 1:00
#BSUB -o profiling/runners/logs/idle.%J.out
#BSUB -e profiling/runners/logs/idle.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs profiling/results

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
BATCHES="${BATCHES:-1 8 64}"
N_DECODE="${N_DECODE:-128}"
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook_${USER:-noname}_${LSB_JOBID:-$$}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"

mkdir -p "$HOOK_DIR"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Memory profiler so idle/loaded/hooks comparisons include RSS + CUDA stats.
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_MEM=1

TAG="${MODEL//\//_}-${LSB_JOBID:-$(date +%Y%m%d-%H%M)}"
OUT_CSV="profiling/results/idle-${TAG}.csv"

echo "[idle] model=$MODEL  out=$OUT_CSV"
python -u profiling/bench_idle_tax.py \
    --model              "$MODEL" \
    --num-decode-tokens  "$N_DECODE" \
    --batches            $BATCHES \
    --hook-dir           "$HOOK_DIR" \
    --gpu-mem-util       "$GPU_MEM_UTIL" \
    --output             "$OUT_CSV"

echo
echo "[idle] CSV → $OUT_CSV"
