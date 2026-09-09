#!/bin/bash
# run_hs_ring_perreq_parity.sh — graph(capture-ring PER-REQUEST demux)-vs-eager HIDDEN-STATE parity
# oracle (plan Task 6, branch `capture_ring`). The byte-identity GATE for the per-request ring demux
# (Task 5, commits ec6c876 + 47ec5be). Same FULL-cudagraph capture-ring mechanism as
# run_hs_ring_parity.sh, but arms VLLM_HOOK_RING_PER_REQUEST=1 so the off-loop consumer DEMUXES each
# step's drained rows BY req_id into a PerRequestIndex (no shared per-layer file). The graph leg
# captures BOTH prompts in ONE batched generate (a real interleaved batch), then reads back via the
# worker's flush_ring_per_request() collective_rpc — which stop()s the drain (finalize_all delivers
# last-step stragglers), pops each request's per-layer tensors, frees it, and returns
# (deliverables, residency_after). residency_after MUST be 0 (the residency gate). Eager stays the
# legacy register_forward_hook ground truth via out[0].probes (solo per prompt — see the .py docstring).
#
# Four hook-activity/routing configs, each its own leg in ONE job (mirrors run_hs_ring_parity.sh):
#   leg A (prefill-only, last_token, layers [1,2,3,4]): prefill captured; every decode token routes
#       to the ring SENTINEL row -> per-request delivery must still deliver each request's single
#       prefill row and drain residency to 0.
#   leg B (both, all_tokens, ALL layers): every decode step reserves a real ring row and gets demuxed
#       per req_id -> strict full-shape per-request compare across 12 steps for BOTH interleaved
#       requests. A dropped/misrouted/mis-demuxed token FAILS loudly (shape/token mismatch) and a
#       leaked request FAILS the residency gate.
#
# ⚠ This harness has been RED on every leg since ~2026-08-04 for a pre-existing,
# separately-tracked defect in the per-request delivery path (0 delivered, 0/0 tensors). See the
# fuller note above the legs before citing this file as coverage for anything.
#
# Submit:  bsub < tests/cuda_graph/tests/ring/run_hs_ring_perreq_parity.sh
# Result:  grep 'hs-ring-perreq-parity\] VERDICT' tests/cuda_graph/logs/hs_ring_perreq_parity.%J.out
#          grep 'hs-ring-perreq-parity\] RESIDENCY' tests/cuda_graph/logs/hs_ring_perreq_parity.%J.out
#
#BSUB -J vllm_hook_hs_ring_perreq_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_ring_perreq_parity.%J.out
#BSUB -e tests/cuda_graph/logs/hs_ring_perreq_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill (op mechanism removed; buffer covers prefill too), full
# cudagraph decode, capture-ring buffer path, PER-REQUEST demux delivery.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_RING_PER_REQUEST=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/hs_ring_perreq_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/ring/hs_ring_perreq_parity.py

echo "[run_hs_ring_perreq_parity] cudagraph_mode=FULL capture=buffer per_request=1 (capture-ring demux)"

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  local ring_dir="$WORK/${name}_ring"
  echo "============================================================"
  echo "[run_hs_ring_perreq_parity] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_hs_ring_perreq_parity] === ${name} 1/3 graph(ring per-request) capture (FULL, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_RING_DIR="$ring_dir" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_hs_ring_perreq_parity] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_hs_ring_perreq_parity] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" \
    || echo "[run_hs_ring_perreq_parity] LEG $name FAILED"
}

# Leg A: prefill-only, last_token, subset layers [1,2,3,4] — decode is a baked ring no-op (every
# decode step routes to the SENTINEL row) under hooks_on=prefill; per-request delivery still drains.
run_leg prefillonly_lasttok "model_configs/hidden_states/Qwen2-1.5B-Instruct.json" prefill 12

# Leg B: both (prefill+decode), all_tokens over ALL layers — every decode step reserves + drains a
# real ring row, demuxed per req_id. The hard case: strict full-shape per-request compare across 12
# steps for BOTH interleaved requests.
run_leg both_alltok "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json" both 12

# ⚠ NO COVERAGE IS AVAILABLE FROM THIS HARNESS TODAY -- READ BEFORE CITING IT (final-review fix
# M6). This whole script has been RED since ~2026-08-04: EVERY leg reports 0 delivered requests
# and 0/0 tensors. That is a PRE-EXISTING defect in the per-request delivery path this harness
# drives, tracked separately. It cannot report a FALSE pass -- `compare()` in
# hs_ring_perreq_parity.py guards its PASS verdict on `total > 0`, so 0/0 fails loud -- so the
# legs are kept in place, ready to become real coverage the moment the delivery defect is fixed.
# Byte-identity of the shared-file drain is meanwhile covered by run_hs_ring_parity.sh (legs A/B/C).

echo "[run_hs_ring_perreq_parity] DONE — see VERDICT + RESIDENCY lines above per leg."
