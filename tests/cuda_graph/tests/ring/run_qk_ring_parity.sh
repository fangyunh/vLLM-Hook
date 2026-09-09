#!/bin/bash
# run_qk_ring_parity.sh — graph(capture-ring)-vs-eager QK parity oracle (plan Task 15,
# branch `capture_ring`). The ring path (graph/install.py::install_qk_hosts / _build_routing)
# scatters each captured token's post-RoPE q + k into TWO per-layer GPU RINGS sharing ONE
# advancing logical cursor (capture_qk writes q_buf[idx] AND k_buf[idx] at the same index); an
# off-loop consumer thread drains committed rows to durable per-layer q/k raw files + a shared
# metadata sidecar. No RPC/bank/egress copy-out on this path, so the graph leg reads back via the
# worker's `flush_ring()` collective_rpc + graph.ring_reader.load_multilayer_qk_ring_artifact
# instead of out[0].probes (see qk_ring_parity.py docstring). Eager stays the legacy
# register_forward_hook ground truth via out[0].probes.
#
# Prefix-cache reconstruction is DEFERRED on the QK ring path (v1): the reader raises
# NotImplementedError if a request's first captured step has num_computed>0. Both engines run
# with enable_prefix_caching=False (fresh prefill only), so the two legs below stay in scope.
#
# Two hook-activity configs, each its own leg in ONE job (mirrors run_hs_ring_parity.sh /
# run_qk_parity_full.sh):
#   leg A (prefill-only, last_token, the important_heads subset layers): prefill captured; every
#       decode step is SKIPPED entirely by the hooks_on=prefill gate in _build_routing (no
#       reserve, no scatter) -> a true no-op. Proves the ring costs/corrupts nothing under decode
#       churn; prefill last_token q-narrowing + K-completeness parity holds (report assumption 3).
#   leg B (both, all_tokens, ALL layers): eager-prefill-equivalent + every decode step reserves a
#       real ring row and gets drained -> parity vs eager for every captured token (prefill span +
#       11 decode steps), q AND the full growing-prefix k_all. Strict full-shape compare
#       (MAX_TOKENS>1) -> a dropped/misrouted decode token FAILS loudly (shape/step-count
#       mismatch), never silently trimmed.
#
# Submit:  bsub < tests/cuda_graph/tests/ring/run_qk_ring_parity.sh
# Result:  grep 'qk-ring-parity\] VERDICT' tests/cuda_graph/logs/qk_ring_parity.%J.out
#          grep 'QK capture ring\|QK ring drain ON' tests/cuda_graph/logs/qk_ring_parity.%J.out
#
#BSUB -J vllm_hook_qk_ring_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_ring_parity.%J.out
#BSUB -e tests/cuda_graph/logs/qk_ring_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill (buffer covers prefill too), full cudagraph decode, QK
# capture-ring buffer path.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_QK_CAPTURE=buffer
# Force raw q/k capture, never attention-score (score is out of scope on the ring path — v1 raw
# q/k only, graph/install.py fails loud at worker-wide install and skips+warns per-request; this
# pin just documents intent, since VLLM_HOOK_QK_AUTO_SELECT already defaults OFF).
export VLLM_HOOK_QK_AUTO_SELECT=0
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# Pre-size the per-layer q/k mmap raw files explicitly (default is
# max(2 GiB, n_slots*row_bytes) PER q AND k file PER layer — sparse but avoids the
# plain-append overflow tail for this rate-4-ish small workload); confirm no "falling back to
# the plain append path" warning fired in the log.
export VLLM_HOOK_RING_MMAP_BYTES="${VLLM_HOOK_RING_MMAP_BYTES:-2147483648}"  # 2 GiB

WORK="/dev/shm/vllm_hook_${USER}/qk_ring_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/ring/qk_ring_parity.py

echo "[run_qk_ring_parity] cudagraph_mode=FULL capture=buffer (QK capture-ring path)"

run_leg () {
  local name="$1" cfg="$2" hooks_on="$3" mt="$4"
  local g="$WORK/${name}_graph.pkl" e="$WORK/${name}_eager.pkl"
  local ring_dir="$WORK/${name}_ring"
  echo "============================================================"
  echo "[run_qk_ring_parity] LEG=$name config=$cfg hooks_on=$hooks_on max_tokens=$mt"
  echo "============================================================"
  echo "[run_qk_ring_parity] === ${name} 1/3 graph(ring) capture (FULL, buffer) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_RING_DIR="$ring_dir" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode graph --out "$g"
  echo "[run_qk_ring_parity] === ${name} 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CONFIG_FILE="$cfg" \
    VLLM_HOOK_PARITY_HOOKS_ON="$hooks_on" VLLM_HOOK_PARITY_MAX_TOKENS="$mt" \
    python -u "$PY" capture --mode eager --out "$e"
  echo "[run_qk_ring_parity] === ${name} 3/3 compare ==="
  python -u "$PY" compare --graph "$g" --eager "$e" || echo "[run_qk_ring_parity] LEG $name FAILED"
}

# Leg A: prefill-only, last_token, the important_heads subset layers — decode is a true no-op
# under hooks_on=prefill (the routing wrapper's hooks_on gate skips the reserve entirely, so no
# ring row is ever touched on those steps).
run_leg prefillonly_lasttok "model_configs/attention_tracker/Qwen2-1.5B-Instruct.json" prefill 12

# Leg B: both (prefill+decode), all_tokens over ALL layers — every decode step reserves + drains
# a real ring row. The hard case: strict step-count + full-shape compare across 12 steps, q AND
# the full growing-prefix k_all.
run_leg both_alltok "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json" both 12

echo "[run_qk_ring_parity] DONE — see VERDICT lines above per leg."
