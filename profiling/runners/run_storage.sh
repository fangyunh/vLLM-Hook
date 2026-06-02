#!/bin/bash
# run_storage.sh — full storage-variant sweep (replaces the prelim table
# in configs.md). Hidden-states worker × {rpc, disk-pt, disk-pt-async,
# disk-st, disk-st-async, shm} × {last_token, all_tokens} × prompt lens
# × batches. ~2 hr on A100.
#
# Submit:  bsub < profiling/runners/run_storage.sh
# Logs:    profiling/runners/logs/storage.<JOBID>.{out,err}
#BSUB -J vllm_hook_prof_storage
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB] span[hosts=1]"
#BSUB -W 4:00
#BSUB -o profiling/runners/logs/storage.%J.out
#BSUB -e profiling/runners/logs/storage.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs profiling/results

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
PROMPT_LENS="${PROMPT_LENS:-16 64 256 512}"
BATCH_SIZES="${BATCH_SIZES:-1 8}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-2}"
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook_${USER:-noname}_${LSB_JOBID:-$$}}"
PROFILE_DIR="${PROFILE_DIR:-/dev/shm/vllm_hook_profile_${USER:-noname}_${LSB_JOBID:-$$}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"

mkdir -p "$HOOK_DIR" "$PROFILE_DIR"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"
export VLLM_HOOK_PROFILE_MEM=1

TAG="${MODEL//\//_}-${LSB_JOBID:-$(date +%Y%m%d-%H%M)}"
OUT_CSV="profiling/results/storage-${TAG}.csv"
OUT_MD="profiling/results/storage-${TAG}-summary.md"
OUT_PNG="profiling/results/storage-${TAG}-heatmap.png"

echo "[storage] model=$MODEL  out=$OUT_CSV"
python -u profiling/bench_grid.py \
    --plan         plan-storage \
    --model        "$MODEL" \
    --prompt-lens  $PROMPT_LENS \
    --batch-sizes  $BATCH_SIZES \
    --hook-dir     "$HOOK_DIR" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --reps         "$REPS" \
    --warmup       "$WARMUP" \
    --output       "$OUT_CSV"

echo
echo "[storage] summarizing..."
python -u profiling/analyze/summarize.py "$OUT_CSV" --output "$OUT_MD"

echo
echo "[storage] plotting gen-lat heatmap..."
python -u profiling/analyze/plot_grid.py "$OUT_CSV" \
    --metric gen_lat_mean --group-by storage_variant --output "$OUT_PNG" || \
    echo "[storage] plot skipped (matplotlib missing?)"

echo
echo "[storage] done."
echo "  CSV     : $OUT_CSV"
echo "  Summary : $OUT_MD"
echo "  Heatmap : $OUT_PNG"
