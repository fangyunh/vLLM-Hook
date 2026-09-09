#!/bin/bash
# run_graph_decode_verify.sh — prove the FULL CUDA-graph (buffer) capture/steer
# path is CORRECT at the DECODE stage, against an INDEPENDENT ground truth
# (teacher forcing), NOT by comparing to the eager path.
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/decode_verify/run_graph_decode_verify.sh
# Result:  grep 'VERDICT' tests/cuda_graph/logs/graph_decode_verify.*.out
#
#BSUB -J vllm_hook_graph_decode
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/graph_decode_verify.%J.out
#BSUB -e tests/cuda_graph/logs/graph_decode_verify.%J.err
set -uo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$(pwd)/cache"           # model cache is offloaded to ./cache
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_DISABLE_COMPILE_CACHE=1          # avoid stale-graph poisoning
export VLLM_USE_AOT_COMPILE=0                # mutating steer_buffer op breaks torch AOT
export VLLM_ALLOW_INSECURE_SERIALIZATION=1   # steer.fire read via collective_rpc(callable)
export PYTHONPATH="$(pwd)/tests/cuda_graph/tests/decode_verify:${PYTHONPATH:-}"
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VERIFY_MAX_TOKENS="${VERIFY_MAX_TOKENS:-6}"
export VERIFY_SCRATCH="/dev/shm/vllm_hook_${USER}/graph_decode_${LSB_JOBID:-manual}"
mkdir -p "$VERIFY_SCRATCH"

PY=tests/cuda_graph/tests/decode_verify/verify_graph_decode.py
echo "[graph_decode_verify] model=$VLLM_HOOK_DEMO_MODEL max_tokens=$VERIFY_MAX_TOKENS FULL cudagraph + buffer"

rc=0
for w in qk hs steer; do
  echo "============================================================"
  echo "[graph_decode_verify] WORKER=$w"
  echo "============================================================"
  python -u "$PY" --worker "$w"
  [ $? -ne 0 ] && { rc=1; echo "[graph_decode_verify] $w FAILED"; }
done

echo "============================================================"
if [ $rc -eq 0 ]; then echo "[graph_decode_verify] OVERALL: PASS"; else echo "[graph_decode_verify] OVERALL: FAIL"; fi
exit $rc
