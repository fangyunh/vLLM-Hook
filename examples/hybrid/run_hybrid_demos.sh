#!/bin/bash
# run_hybrid_demos.sh — run the three Hybrid-mode (CUDA-graph) demos on Granite-8B,
# back to back in one job (each engine releases the exclusive-process GPU before the
# next boots). Confirms steering, hidden-state capture, and Q/K capture all work with
# CUDA graphs ON, on a real 8B model, across prefill + decode.
#
# Submit:  bsub < examples/hybrid/run_hybrid_demos.sh
# Result:  grep 'VERDICT' examples/hybrid/logs/hybrid_demos.*.out
#
#BSUB -J vllm_hook_hybrid_demos
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o examples/hybrid/logs/hybrid_demos.%J.out
#BSUB -e examples/hybrid/logs/hybrid_demos.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p examples/hybrid/logs

# Hybrid-mode env (the demos also set VLLM_HOOK_ALLOW_CUDAGRAPH themselves).
export VLLM_HOOK_ALLOW_CUDAGRAPH=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-ibm-granite/granite-3.1-8b-instruct}"
# Keep the test runtime sane: 128-token decode for all three (overrides actsteer's 2048
# default; the long-decode demos already default to 128). Still exercises prefill+decode.
export VLLM_HOOK_DEMO_MAX_TOKENS="${VLLM_HOOK_DEMO_MAX_TOKENS:-128}"
export VLLM_HOOK_DEMO_HOOKS_ON="${VLLM_HOOK_DEMO_HOOKS_ON:-both}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

echo "[run_hybrid_demos] model=$VLLM_HOOK_DEMO_MODEL max_tokens=$VLLM_HOOK_DEMO_MAX_TOKENS"

echo "[run_hybrid_demos] === 1/3 activation steering (adjust_rs, layer 15) ==="
python -u examples/hybrid/demo_actsteer_hybrid.py

echo "[run_hybrid_demos] === 2/3 hidden-state capture (long decode, both phases) ==="
python -u examples/hybrid/demo_hiddenstate_longdec_hybrid.py

echo "[run_hybrid_demos] === 3/3 attention-tracker Q/K capture (long decode, both phases) ==="
python -u examples/hybrid/demo_attntracker_longdec_hybrid.py

echo "[run_hybrid_demos] === DONE — grep VERDICT for the three self-checks ==="
