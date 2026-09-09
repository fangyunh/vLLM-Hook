#!/bin/bash
# run_quant_parity.sh — GPU value-parity for artifact quantization (Phase 1, eager path).
#
# For each worker {qk, hs}: capture the NATIVE fp16 reference, then capture int8/int4/int2/
# fp8_e4m3 (each in its own process so the exclusive-process GPU is released between boots),
# and compare each quantized capture to native. PASS iff every captured tensor is within its
# dtype's max|Δ|/amax bound (int8<=1/127, int4<=1/7, int2<=1.0, fp8<=0.125, x1.5 slack).
#
# Submit:  bsub < tests/cuda_graph/tests/quant_parity/run_quant_parity.sh
# Result:  grep 'quant-parity] VERDICT' tests/cuda_graph/logs/quant_parity.*.out
#
#BSUB -J vllm_hook_quant_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -W 03:00
#BSUB -o tests/cuda_graph/logs/quant_parity.%J.out
#BSUB -e tests/cuda_graph/logs/quant_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Prefill-only single pass (clean per_token capture; avoids the lossy offline multi-pass
# merge). Override MAX_TOKENS>1 + HOOKS_ON=both to also exercise decode-step capture.
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-1}"
export VLLM_HOOK_PARITY_HOOKS_ON="${VLLM_HOOK_PARITY_HOOKS_ON:-}"

WORK="/dev/shm/vllm_hook_${USER}/quant_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/quant_parity/quant_parity.py
DTYPES=(int8 int4 int2 fp8_e4m3)

# store=rpc: worker hands off quantized over the RPC hop, driver dequants in generate().
# store=disk: worker writes the quant struct to .pt, the analyzer's disk loader dequants.
# Both are the 1b-full "dequant at the analysis boundary, not the worker" contract.
rc=0
for STORE in rpc disk; do
  for W in qk hs; do
    echo "[run_quant_parity] ===== store=$STORE worker=$W : native reference ====="
    python -u "$PY" capture --store "$STORE" --worker "$W" --dtype none \
        --out "$WORK/${STORE}_${W}_native.pkl"
    for D in "${DTYPES[@]}"; do
        echo "[run_quant_parity] ===== store=$STORE worker=$W dtype=$D : capture ====="
        python -u "$PY" capture --store "$STORE" --worker "$W" --dtype "$D" \
            --out "$WORK/${STORE}_${W}_${D}.pkl"
        echo "[run_quant_parity] ===== store=$STORE worker=$W dtype=$D : compare vs native ====="
        python -u "$PY" compare --store "$STORE" --worker "$W" --dtype "$D" \
            --ref "$WORK/${STORE}_${W}_native.pkl" --test "$WORK/${STORE}_${W}_${D}.pkl" || rc=1
    done
  done
done

echo "[run_quant_parity] ===== ALL VERDICTS ====="
grep -h 'quant-parity] VERDICT' "tests/cuda_graph/logs/quant_parity.${LSB_JOBID:-manual}.out" 2>/dev/null || true
echo "[run_quant_parity] overall rc=$rc"
exit $rc
