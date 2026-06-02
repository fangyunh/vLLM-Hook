#!/bin/bash
# run_quick.sh — the R0..R5 six-row deliverable from plan.html §7.
#
# Runs bench_grid.py --plan plan-quick on one GPU. ~30 min on A100;
# longer on H100 with the larger model. Produces:
#   profiling/results/quick-<MODEL>-<JOBID>.csv
#   profiling/results/quick-<MODEL>-<JOBID>.jsonl
#   profiling/results/quick-<MODEL>-<JOBID>-summary.md
#
# Submit:  bsub < profiling/runners/run_quick.sh
# Logs:    profiling/runners/logs/quick.<JOBID>.{out,err}
#
# Override via env vars before bsub:
#   MODEL=ibm-granite/granite-3.1-8b-instruct \
#       bsub < profiling/runners/run_quick.sh
#   PROMPT_LENS="16 64 256" BATCH_SIZES="1 8" \
#       bsub < profiling/runners/run_quick.sh
#BSUB -J vllm_hook_prof_quick
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB] span[hosts=1]"
#BSUB -o profiling/runners/logs/quick.%J.out
#BSUB -e profiling/runners/logs/quick.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs profiling/results

MODEL="${MODEL:-Qwen/Qwen2-1.5B-Instruct}"
PROMPT_LENS="${PROMPT_LENS:-16 256}"
BATCH_SIZES="${BATCH_SIZES:-1 16}"
MAX_TOKENS="${MAX_TOKENS:-1 32}"
REPS="${REPS:-5}"
WARMUP="${WARMUP:-2}"
# Per-job tmpfs paths — see run_smoke.sh for the collision rationale.
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook_${USER:-noname}_${LSB_JOBID:-$$}}"
PROFILE_DIR="${PROFILE_DIR:-/dev/shm/vllm_hook_profile_${USER:-noname}_${LSB_JOBID:-$$}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

mkdir -p "$HOOK_DIR" "$PROFILE_DIR"

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"
# Background NVML + psutil + torch.cuda memory sampler (50 ms).
export VLLM_HOOK_PROFILE_MEM=1

# Job-id suffixed output so concurrent jobs never overwrite each other.
TAG="${MODEL//\//_}-${LSB_JOBID:-$(date +%Y%m%d-%H%M)}"
OUT_CSV="profiling/results/quick-${TAG}.csv"
OUT_MD="profiling/results/quick-${TAG}-summary.md"

echo "[quick] model=$MODEL  out=$OUT_CSV"
python -u profiling/bench_grid.py \
    --plan         plan-quick \
    --model        "$MODEL" \
    --prompt-lens  $PROMPT_LENS \
    --batch-sizes  $BATCH_SIZES \
    --max-tokens   $MAX_TOKENS \
    --hook-dir     "$HOOK_DIR" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --reps         "$REPS" \
    --warmup       "$WARMUP" \
    --output       "$OUT_CSV"

echo
echo "[quick] summarizing into $OUT_MD ..."
python -u profiling/analyze/summarize.py "$OUT_CSV" --output "$OUT_MD"

echo
echo "[quick] done."
echo "  CSV     : $OUT_CSV"
echo "  Summary : $OUT_MD"
echo "  JSONL   : ${OUT_CSV%.csv}.jsonl"
