#!/bin/bash
# run_qk_serve_per_request.sh — Task 13 phase C ONLINE-SERVE validation for the per-request QK
# capture-ring DELIVERY path (branch `capture_ring`). QK analogue of run_serve_per_request.sh (the HS
# Task-12 serve gate, job 584787 PASS). Drives the REAL serve code path — AsyncLLM.generate ->
# _hook_plugin._patched_generate (request-start router _decide_ring_route_qk + route_ring_to_disk,
# finalize BLOCK-UNTIL-HELD get_ring_per_request / disk confirm_ring_delivery, abort finally ->
# clear_ring_request) — by firing CONCURRENT generate() coroutines against ONE AsyncLLM engine under
# continuous batching (NO HTTP server).
#
# Asserts (single grep-able VERDICT): mid-serving delivery, byte-identity vs SOLO-EAGER on BOTH the RPC
# and DISK routes (q flat + growing-prefix k_all LIST), residency -> 0 after delivery AND after an
# aborted-mid-stream request (non-destructive ring_residency RPC). BOTH routes forced by prompt size:
# short prompts predict a small QK artifact (< T_RPC -> RPC), long prompts a large one (> T_RPC -> DISK),
# with VLLM_HOOK_ROUTER_T_RPC set between them.
#
# Uses the important-heads all_tokens config (dict output_qk over 9 layers) so the request-start router
# can SIZE the artifact — an _alltok.json (no important_heads) gives output_qk=True (whole-model),
# which _decide_ring_route_qk cannot size -> everything RPC -> the DISK route never fires.
#
# Submit:  bsub < tests/cuda_graph/tests/ring/run_qk_serve_per_request.sh
# Result:  grep 'qk-serve-per-request\] VERDICT'   tests/cuda_graph/logs/qk_serve_per_request.%J.out
#          grep 'qk-serve-per-request\] RESIDENCY' tests/cuda_graph/logs/qk_serve_per_request.%J.out
#          grep 'qk-serve-per-request\] ROUTES'    tests/cuda_graph/logs/qk_serve_per_request.%J.out
#
#BSUB -J vllm_hook_qk_serve_per_request
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_serve_per_request.%J.out
#BSUB -e tests/cuda_graph/logs/qk_serve_per_request.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
# FULL target: eager prefill + full-graph decode, capture-ring buffer, per-request DELIVERY.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_QK_CAPTURE=buffer
export VLLM_HOOK_RING_PER_REQUEST=1
# DIAGNOSTIC: gated disk-pipeline trace (route_to_disk / demux hit-miss / finish submit / offload
# start-done-error). Off in normal runs.
export VLLM_HOOK_RING_DEBUG="${VLLM_HOOK_RING_DEBUG:-0}"
# Keep the request's save_to_disk ABSENT so the finalize sink stays 'rpc' and the RING route arms
# (the ring route then decides rpc-vs-disk itself from the predicted size).
export VLLM_HOOK_STORAGE_ROUTER=0
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# The important-heads all_tokens config (14 heads / 9 layers) — dict output_qk -> routable by size.
export VLLM_HOOK_CONFIG_FILE="${VLLM_HOOK_CONFIG_FILE:-model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok_heads.json}"
# RPC/disk crossover, BYTES. Qwen2-1.5B all_tokens+both, 14-head/9-layer: short (len 6 + gen 8)
# predicts ~98 KB -> RPC; long (len 220 + gen 8) predicts ~1.56 MiB -> DISK. 512 KiB sits between.
export VLLM_HOOK_ROUTER_T_RPC="${VLLM_HOOK_ROUTER_T_RPC:-524288}"
# Deliver timeout: bounded block-until-held / disk-confirm (LOUD on a wedged consumer, never a hang).
export VLLM_HOOK_RING_DELIVER_TIMEOUT_S="${VLLM_HOOK_RING_DELIVER_TIMEOUT_S:-60}"

WORK="/dev/shm/vllm_hook_${USER}/qk_serve_per_request_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
# The DISK route stages per-request run_dirs on NVMe then offloads (copytree) to the client dest.
export VLLM_HOOK_RING_DIR="$WORK/ring"
export VLLM_HOOK_DELIVER_DIR="$WORK/delivered"

PY=tests/cuda_graph/tests/ring/qk_serve_per_request.py
G="$WORK/graph.pkl" E="$WORK/eager.pkl"

echo "[run_qk_serve_per_request] cudagraph_mode=FULL capture=buffer per_request=1 T_RPC=$VLLM_HOOK_ROUTER_T_RPC config=$VLLM_HOOK_CONFIG_FILE (QK serve delivery)"
echo "[run_qk_serve_per_request] === 1/3 graph(concurrent AsyncLLM serve) capture ==="
python -u "$PY" capture --mode graph --out "$G"
echo "[run_qk_serve_per_request] === 2/3 eager (solo ground truth) capture ==="
python -u "$PY" capture --mode eager --out "$E"
echo "[run_qk_serve_per_request] === 3/3 compare ==="
python -u "$PY" compare --graph "$G" --eager "$E" \
  || echo "[run_qk_serve_per_request] SERVE VALIDATION FAILED"

echo "[run_qk_serve_per_request] DONE — see VERDICT + RESIDENCY + ROUTES lines above."
