#!/bin/bash
# run_aot_parity.sh — graph+AOT-compile parity for Hybrid-mode capture/steer.
#
# For each worker, FOUR sequential processes in ONE LSF job (exclusive-process GPU):
#   1. eager  capture  (AOT off, register_forward_hook ground truth)  -> e.pkl
#   2. aot    capture  (graph + VLLM_USE_AOT_COMPILE=1, FRESH compile) -> a.pkl
#   3. reload capture  (graph + AOT, SAME cache root -> loads (2)'s artifact) -> r.pkl
#   4. compare a,r against e
#
# ONE shared VLLM_CACHE_ROOT across ALL workers, so:
#   * each worker's reload step exercises the AOT load-from-disk path (the sink=None
#     crash site), and
#   * cross-worker isolation is tested: if the per-worker splitting-op fix regressed,
#     hs/steer would load QK's artifact (shared key) and crash/mismatch.
# Compile cache is deliberately NOT disabled — AOT compile is the whole point.
#
# Submit:  bsub < tests/cuda_graph/tests/aot_compile/run_aot_parity.sh
# Result:  grep 'VERDICT' tests/cuda_graph/logs/aot_parity.*.out
#
#BSUB -J vllm_hook_aot_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/aot_parity.%J.out
#BSUB -e tests/cuda_graph/logs/aot_parity.%J.err
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# ONE shared cache root for the whole job (see header). Per-job namespaced.
export VLLM_CACHE_ROOT="/dev/shm/vllm_hook_${USER}/aot_parity_${LSB_JOBID:-manual}"
mkdir -p "$VLLM_CACHE_ROOT"

WORK="/dev/shm/vllm_hook_${USER}/aot_parity_pkls_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/aot_compile/aot_parity.py

rc_all=0
for W in qk hs steer; do
  echo "============================================================"
  echo "[run_aot_parity] === worker=$W ==="
  echo "============================================================"
  E="$WORK/${W}_eager.pkl"; A="$WORK/${W}_aot.pkl"; R="$WORK/${W}_reload.pkl"
  echo "[run_aot_parity] $W 1/4 eager";  python -u "$PY" capture --worker "$W" --mode eager  --out "$E" || { echo "[run_aot_parity] $W eager FAILED"; rc_all=1; continue; }
  echo "[run_aot_parity] $W 2/4 aot";    python -u "$PY" capture --worker "$W" --mode aot    --out "$A" || { echo "[run_aot_parity] $W aot FAILED"; rc_all=1; continue; }
  echo "[run_aot_parity] $W 3/4 reload"; python -u "$PY" capture --worker "$W" --mode reload --out "$R" || { echo "[run_aot_parity] $W reload FAILED (regression?)"; rc_all=1; continue; }
  echo "[run_aot_parity] $W 4/4 compare"
  python -u "$PY" compare --worker "$W" --eager "$E" --aot "$A" --reload "$R" || rc_all=1
done

echo "[run_aot_parity] overall rc=$rc_all"
exit $rc_all
