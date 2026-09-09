#!/bin/bash
# run_serve_per_request.sh — Task 12 ONLINE-SERVE validation for the per-request HS capture-ring
# DELIVERY path (branch `capture_ring`). Drives the REAL serve code path — AsyncLLM.generate ->
# _hook_plugin._patched_generate (request-start router + route_ring_to_disk, finalize BLOCK-UNTIL-
# HELD get_ring_per_request / disk confirm_ring_delivery, abort finally -> clear_ring_request) — by
# firing CONCURRENT generate() coroutines against ONE AsyncLLM engine under continuous batching
# (NO HTTP server). Same FULL-cudagraph + capture-ring buffer + RING_PER_REQUEST mechanism as
# run_hs_ring_perreq_parity.sh; the difference is the AUTHENTIC serve delivery vs that oracle's
# shutdown-time flush_ring_per_request read.
#
# Asserts (single grep-able VERDICT): mid-serving delivery (a request delivered while others still
# generate), byte-identity vs SOLO-EAGER on BOTH the RPC and DISK routes, residency -> 0 after
# delivery (non-destructive ring_residency RPC), and residency -> 0 after an aborted-mid-stream
# request. BOTH routes are FORCED by prompt size: short prompts predict a small artifact (< T_RPC ->
# RPC) and long prompts a large one (> T_RPC -> DISK), with VLLM_HOOK_ROUTER_T_RPC set between them.
#
# Submit:  bsub < tests/cuda_graph/tests/ring/run_serve_per_request.sh
# Result:  grep 'serve-per-request\] VERDICT'   tests/cuda_graph/logs/serve_per_request.%J.out
#          grep 'serve-per-request\] RESIDENCY' tests/cuda_graph/logs/serve_per_request.%J.out
#          grep 'serve-per-request\] ROUTES'    tests/cuda_graph/logs/serve_per_request.%J.out
#
#BSUB -J vllm_hook_serve_per_request
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/serve_per_request.%J.out
#BSUB -e tests/cuda_graph/logs/serve_per_request.%J.err
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
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_RING_PER_REQUEST=1
# DIAGNOSTIC: gated disk-pipeline trace (route_to_disk / demux hit-miss / finish submit / offload
# start-done-error) to pinpoint the disk-route serve delivery break. Off in normal runs.
export VLLM_HOOK_RING_DEBUG="${VLLM_HOOK_RING_DEBUG:-0}"
# Keep the request's save_to_disk ABSENT so the finalize sink stays 'rpc' and the RING route arms
# (the ring route then decides rpc-vs-disk itself from the predicted size).
export VLLM_HOOK_STORAGE_ROUTER=0
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# RPC/disk crossover, BYTES. Qwen2-1.5B all_tokens+both: short (len 6 + gen 8 = seq 14) predicts
# ~1.15 MB -> RPC; long (len 220 + gen 8 = seq 228) predicts ~19.6 MB -> DISK. 4 MiB sits between.
export VLLM_HOOK_ROUTER_T_RPC="${VLLM_HOOK_ROUTER_T_RPC:-4194304}"
# Deliver timeout: bounded block-until-held / disk-confirm (LOUD on a wedged consumer, never a hang).
export VLLM_HOOK_RING_DELIVER_TIMEOUT_S="${VLLM_HOOK_RING_DELIVER_TIMEOUT_S:-60}"

WORK="/dev/shm/vllm_hook_${USER}/serve_per_request_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
# The DISK route stages per-request run_dirs on NVMe then offloads (copytree) to the client dest.
export VLLM_HOOK_RING_DIR="$WORK/ring"
export VLLM_HOOK_DELIVER_DIR="$WORK/delivered"

PY=tests/cuda_graph/tests/ring/serve_per_request.py
G="$WORK/graph.pkl" E="$WORK/eager.pkl"

echo "[run_serve_per_request] cudagraph_mode=FULL capture=buffer per_request=1 T_RPC=$VLLM_HOOK_ROUTER_T_RPC (serve delivery)"
echo "[run_serve_per_request] === 1/3 graph(concurrent AsyncLLM serve) capture ==="
python -u "$PY" capture --mode graph --out "$G"
echo "[run_serve_per_request] === 2/3 eager (solo ground truth) capture ==="
python -u "$PY" capture --mode eager --out "$E"
echo "[run_serve_per_request] === 3/3 compare ==="
python -u "$PY" compare --graph "$G" --eager "$E" \
  || echo "[run_serve_per_request] SERVE VALIDATION FAILED"

echo "[run_serve_per_request] DONE — see VERDICT + RESIDENCY + ROUTES lines above."
