#!/bin/bash
# run_qk_score.sh — v0.6.0 EAGER score-capture parity (the profiled v0.2.0 path).
#
# Proves the on-GPU per-head attention score (qk_capture='score', eager
# register_forward_hook path) equals the analyzer's recompute from a QK capture
# of the same run. Three sequential processes in one job (GPU is exclusive_process):
#   1. capture --mode score (eager)  -> score.pkl
#   2. capture --mode qk    (eager)  -> qk.pkl   (ground truth)
#   3. compare                       -> PASS/FAIL
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/qk_score/run_qk_score.sh
# Result:  grep 'score-parity] VERDICT' tests/cuda_graph/logs/qk_score.*.out
#
#BSUB -J vllm_hook_qk_score
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_score.%J.out
#BSUB -e tests/cuda_graph/logs/qk_score.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_QK_SCORE_HEAD="${VLLM_HOOK_QK_SCORE_HEAD:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/score_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/qk_score/qk_score_parity.py

echo "[run_qk_score] === 1/3 eager score capture ==="
python -u "$PY" capture --mode score --out "$WORK/score.pkl"
echo "[run_qk_score] === 2/3 eager qk ground truth ==="
python -u "$PY" capture --mode qk --out "$WORK/qk.pkl"
echo "[run_qk_score] === 3/3 compare ==="
python -u "$PY" compare --score "$WORK/score.pkl" --qk "$WORK/qk.pkl"
