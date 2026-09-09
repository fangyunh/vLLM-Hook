#!/bin/bash
# run_decode_verify.sh — prove the EAGER (main-line) hook path truly CAPTURES
# (QK, HS) and STEERS at the DECODE stage under hooks_on="both".
#
# EAGER, no cudagraph (ALLOW_CUDAGRAPH unset). The eager register_forward_hook
# capture/steer code is byte-identical between `main` and `graph_enable` (git
# diff = comments only), so this result applies to main.
#
# Each worker runs in its own fresh process (GPU is exclusive_process).
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/decode_verify/run_decode_verify.sh
# Result:  grep 'VERDICT' tests/cuda_graph/logs/decode_verify.*.out
#
#BSUB -J vllm_hook_decode_verify
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/decode_verify.%J.out
#BSUB -e tests/cuda_graph/logs/decode_verify.%J.err
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Model cache is offloaded to ./cache (symlink to /proj). The DEFAULT hub cache
# (~/.cache/huggingface) holds a WEIGHTLESS snapshot of the same hash, so point
# HF at ./cache or resolution finds config but no weights.
export HF_HUB_CACHE="$(pwd)/cache"
export VLLM_USE_V1=1
# steer.fire is read from the worker via collective_rpc(callable): allow the
# pickle fallback and put the helper module on the EngineCore's import path.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONPATH="$(pwd)/tests/cuda_graph/tests/decode_verify:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-8}"
export VERIFY_SCRATCH="/dev/shm/vllm_hook_${USER}/decode_verify_${LSB_JOBID:-manual}"
mkdir -p "$VERIFY_SCRATCH"

PY=tests/cuda_graph/tests/decode_verify/verify_both_decode.py
echo "[decode_verify] model=$VLLM_HOOK_DEMO_MODEL max_tokens=$VERIFY_MAX_TOKENS EAGER (no cudagraph)"

rc=0
for w in qk hs steer; do
  echo "============================================================"
  echo "[decode_verify] WORKER=$w"
  echo "============================================================"
  python -u "$PY" --worker "$w"
  if [ $? -ne 0 ]; then rc=1; echo "[decode_verify] $w FAILED"; fi
done

echo "============================================================"
grep -h 'VERDICT' tests/cuda_graph/logs/decode_verify.${LSB_JOBID:-manual}.out 2>/dev/null || true
if [ $rc -eq 0 ]; then
  echo "[decode_verify] OVERALL: PASS"
else
  echo "[decode_verify] OVERALL: FAIL"
fi
exit $rc
