#!/bin/bash
# run_full.sh — overnight cartesian sweep (plan-full).
# Every row × every storage variant × every mode × every hooks_on phase
# × prompt_lens × batches × max_tokens. Resumable via --start-from.
#
# Typical wall time on one A100: 4-8 hr depending on prompt_len caps.
#
# Submit:  bsub < profiling/runners/run_full.sh
# Logs:    profiling/runners/logs/full.<JOBID>.{out,err}
#
# Resume after a crash at cell N:
#   START_FROM=N OUT_CSV=profiling/results/full-PREV.csv \
#       bsub < profiling/runners/run_full.sh
#BSUB -J vllm_hook_prof_full
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB] span[hosts=1]"
#BSUB -W 12:00
#BSUB -o profiling/runners/logs/full.%J.out
#BSUB -e profiling/runners/logs/full.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs profiling/results

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
PROMPT_LENS="${PROMPT_LENS:-16 256 1024}"
BATCH_SIZES="${BATCH_SIZES:-1 4 16}"
MAX_TOKENS="${MAX_TOKENS:-1 32}"
REPS="${REPS:-3}"
WARMUP="${WARMUP:-1}"
START_FROM="${START_FROM:-1}"
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook_${USER:-noname}_${LSB_JOBID:-$$}}"
PROFILE_DIR="${PROFILE_DIR:-/dev/shm/vllm_hook_profile_${USER:-noname}_${LSB_JOBID:-$$}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"

mkdir -p "$HOOK_DIR" "$PROFILE_DIR"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"
export VLLM_HOOK_PROFILE_MEM=1

# If resuming, OUT_CSV is the prior run's file. Otherwise mint a new one.
TAG="${MODEL//\//_}-${LSB_JOBID:-$(date +%Y%m%d-%H%M)}"
OUT_CSV="${OUT_CSV:-profiling/results/full-${TAG}.csv}"
OUT_MD="${OUT_CSV%.csv}-summary.md"

echo "[full] model=$MODEL  start_from=$START_FROM  out=$OUT_CSV"
python -u profiling/bench_grid.py \
    --plan         plan-full \
    --model        "$MODEL" \
    --prompt-lens  $PROMPT_LENS \
    --batch-sizes  $BATCH_SIZES \
    --max-tokens   $MAX_TOKENS \
    --hook-dir     "$HOOK_DIR" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --reps         "$REPS" \
    --warmup       "$WARMUP" \
    --start-from   "$START_FROM" \
    --output       "$OUT_CSV"

echo
echo "[full] summarizing..."
python -u profiling/analyze/summarize.py "$OUT_CSV" --output "$OUT_MD" || true

echo
echo "[full] done."
echo "  CSV     : $OUT_CSV"
echo "  Summary : $OUT_MD"
