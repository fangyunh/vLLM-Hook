#!/bin/bash
# run_hs_ring_parity.sh — graph(capture-ring)-vs-eager HIDDEN-STATE parity oracle (plan Task 10,
# branch `capture_ring`). The ring path (graph/install_hs.py) scatters each captured token's
# residual into a persistent per-layer GPU RING (all layers share ONE logical cursor) at an
# ADVANCING slot; a synchronous per-step drain reads the newly-written region and appends it to a
# durable per-layer raw file + a shared metadata sidecar. No RPC/bank/egress copy-out on this
# path, so the graph leg reads back via the worker's `flush_ring()` collective_rpc +
# graph.ring_reader.load_multilayer_ring_artifact instead of out[0].probes (see hs_ring_parity.py
# docstring). Eager stays the legacy register_forward_hook ground truth via out[0].probes.
#
# Three hook-activity configs, each its own leg in ONE job (mirrors run_hs_parity_full.sh):
#   leg A (prefill-only, last_token, layers [1,2,3,4]): prefill captured; every decode token
#       routes to the ring SENTINEL row -> the baked capture_hs scatter is a true no-op on those
#       steps. Proves the ring costs/corrupts nothing under decode; prefill parity holds.
#   leg B (both, all_tokens, ALL layers): eager-prefill-equivalent + every decode step reserves a
#       real ring row and gets drained -> parity vs eager for every captured token (prefill span +
#       11 decode steps). Strict full-shape compare (MAX_TOKENS>1) -> a dropped/misrouted decode
#       token FAILS loudly (shape or token mismatch). This is the DEFAULT-path control — since
#       VLLM_HOOK_CAPTURE_FUSED flipped default-ON (2026-08-10), an unset flag here now exercises
#       the FUSED hidden+residual scatter (graph/capture_triton.py), not the aten body.
#   leg C (same shape as leg B, VLLM_HOOK_CAPTURE_FUSED=0): pins the ATEN FALLBACK — the kill
#       switch's control leg, so a regression in the pre-fusion `hidden + residual` +
#       `hs_buf.index_copy_()` path (dispatched from ops.py's _capture_hs_impl) does not go
#       unnoticed now that it is no longer the default. Task 3's oracle proved the fused kernel is
#       bit-exact eagerly and ring parity proved it legal inside a captured FULL cudagraph (no host
#       sync / autotune / dynamic launch config smuggled in) before the flip; this leg keeps the
#       OTHER branch — the one the kill switch falls back to — under the same byte-identical
#       graph-vs-eager parity as leg B (mirrors what `a45441d` did when batched-egress flipped).
#
# Submit:  bsub < tests/cuda_graph/tests/ring/run_hs_ring_parity.sh
# Result:  grep 'hs-ring-parity\] VERDICT' tests/cuda_graph/logs/hs_ring_parity.%J.out
#          grep 'HS capture ring\|HS ring drain ON' tests/cuda_graph/logs/hs_ring_parity.%J.out
#
#BSUB -J vllm_hook_hs_ring_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_ring_parity.%J.out
#BSUB -e tests/cuda_graph/logs/hs_ring_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

# The Triton kernel specializations JIT-compile at boot; gcc's assembler scratch for that build
# lands under the compute node's /tmp, which is small and unrelated to the HOME quota; a full /tmp
# aborts the JIT with "error writing to /tmp/ccXXXXXX.s: No space left on device" and has killed
# jobs before. Point TMPDIR/TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR at real per-job
# scratch instead, applied for every leg so compile-cache locality can't land on only one of them.
export TMPDIR="/opt/nvme/${USER}/hsring_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/proj/dmfexp/fangyunh/scratch_hsring/tmp_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf "$TMPDIR"' EXIT

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill (op mechanism removed; buffer covers prefill too), full
# cudagraph decode, capture-ring buffer path.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/hs_ring_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/ring/hs_ring_parity.py

echo "[job] host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
echo "[run_hs_ring_parity] cudagraph_mode=FULL capture=buffer (capture-ring path)"

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  local ring_dir="$WORK/${name}_ring"
  echo "============================================================"
  echo "[run_hs_ring_parity] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_hs_ring_parity] === ${name} 1/3 graph(ring) capture (FULL, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_RING_DIR="$ring_dir" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_hs_ring_parity] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_hs_ring_parity] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_hs_ring_parity] LEG $name FAILED"
}

# Leg A: prefill-only, last_token, subset layers [1,2,3,4] — decode is a baked ring no-op
# (every decode step routes to the SENTINEL row) under hooks_on=prefill.
run_leg prefillonly_lasttok "model_configs/hidden_states/Qwen2-1.5B-Instruct.json" prefill 12

# Leg B: both (prefill+decode), all_tokens over ALL layers — every decode step reserves +
# drains a real ring row. The hard case: strict full-shape compare across 12 steps.
run_leg both_alltok "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json" both 12

# leg C — same shape as leg B, but pinned to the ATEN fallback (VLLM_HOOK_CAPTURE_FUSED=0). Leg B
# (unchanged, above) now exercises the new default (fused), so this leg is what keeps the
# kill-switch's aten path under the same byte-identical graph-vs-eager parity as leg B.
export VLLM_HOOK_CAPTURE_FUSED=0
run_leg aten_both_alltok "model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json" both 12
unset VLLM_HOOK_CAPTURE_FUSED

echo "[run_hs_ring_parity] DONE — see VERDICT lines above per leg."
