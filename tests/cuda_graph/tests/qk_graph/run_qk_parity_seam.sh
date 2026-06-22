#!/bin/bash
# run_qk_parity_seam.sh — graph-vs-eager QK parity for the SEAM capture prototype.
#
# Identical to run_qk_parity.sh but the graph leg runs VLLM_HOOK_QK_CAPTURE=seam:
# capture rides the attention impl's eager seam (NO qk_probe splitting op). The eager
# (ground-truth) leg ignores the var — graph mode is unarmed there, so it uses the
# legacy register_forward_hook path. PASS => seam capture == eager within tol.
#   1. capture --mode graph  (PIECEWISE, SEAM-path)          -> graph.pkl
#   2. capture --mode eager  (legacy register_forward_hook)  -> eager.pkl
#   3. compare graph.pkl eager.pkl                           -> PASS/FAIL
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity_seam.sh
# Result:  grep 'parity\] VERDICT' tests/cuda_graph/logs/qk_parity_seam.*.out
#          grep 'seam-mode QK capture wired' tests/cuda_graph/logs/qk_parity_seam.*.out
#
#BSUB -J vllm_hook_qk_parity_seam
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity_seam.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity_seam.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
# Force a clean compile each graph run so a stale cache cannot mask a regression.
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
# The prototype under test: capture inside the attention impl's eager seam.
export VLLM_HOOK_QK_CAPTURE=seam
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Per-job artifact dir (node-local tmpfs); namespaced so it never collides.
WORK="/dev/shm/vllm_hook_${USER}/parity_seam_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
GRAPH_PKL="$WORK/graph.pkl"
EAGER_PKL="$WORK/eager.pkl"

PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

echo "[run_qk_parity_seam] === 1/3 graph capture (VLLM_HOOK_QK_CAPTURE=seam) ==="
python -u "$PY" capture --mode graph --out "$GRAPH_PKL"

echo "[run_qk_parity_seam] === 2/3 eager capture (ground truth) ==="
# Ground truth must use the eager register_forward_hook path, not seam.
VLLM_HOOK_QK_CAPTURE=op python -u "$PY" capture --mode eager --out "$EAGER_PKL"

echo "[run_qk_parity_seam] === 3/3 compare ==="
python -u "$PY" compare --graph "$GRAPH_PKL" --eager "$EAGER_PKL"
