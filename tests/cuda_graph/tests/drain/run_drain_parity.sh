#!/bin/bash
# run_drain_parity.sh — v0.5.0 W4/W6c: prove the budget-driven GPU->pinned capture drain
# is VALUE-PRESERVING. Reuses the proven QK + HS FULL-decode parity oracles (both /
# all_tokens — the heaviest capture: 28 layers x per-token q/k slices x 12 decode steps,
# QK accumulating the full k_all history each step) but forces the drain to fire FREQUENTLY
# by setting a tiny absolute byte budget (VLLM_HOOK_CAPTURE_DRAIN_BYTES). The GRAPH leg
# (buffer mode) drains GPU clones to pinned host mid-generation on a dedicated copy stream;
# the EAGER leg (op mode) has no graph install and no drain (the ground truth). PASS =
# graph (drained) reproduces eager (undrained) within rtol=atol=1e-2 -> draining moves WHERE
# the tensors live, never their values, and the multi-stream copy/record_stream/event guard
# has no use-after-free or race.
#
# Submit:  bsub < tests/cuda_graph/tests/drain/run_drain_parity.sh
# Result:  grep -E 'parity\] VERDICT|hs-parity\] VERDICT' tests/cuda_graph/logs/drain_parity.*.out
#          grep 'capture drain enabled' tests/cuda_graph/logs/drain_parity.*.out
#
#BSUB -J vllm_hook_drain_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/drain_parity.%J.out
#BSUB -e tests/cuda_graph/logs/drain_parity.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-12}"
# THE knob under test: a tiny absolute budget forces a drain almost every decode step,
# so the graph leg's buckets become a mix of pinned-host (drained) + GPU (recent) tensors.
export VLLM_HOOK_CAPTURE_DRAIN_BYTES="${VLLM_HOOK_CAPTURE_DRAIN_BYTES:-65536}"

WORK="/dev/shm/vllm_hook_${USER}/drain_parity_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
QK=tests/cuda_graph/tests/qk_graph/qk_parity.py
HS=tests/cuda_graph/tests/hs_graph/hs_parity.py

echo "[run_drain] FULL_DECODE_ONLY drain_bytes=$VLLM_HOOK_CAPTURE_DRAIN_BYTES max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS"

# ---- QK both / all_tokens: graph (buffer, drained) vs eager (op, undrained) ----
QKCFG="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
GQ="$WORK/qk_graph.pkl" EQ="$WORK/qk_eager.pkl"
echo "============================================================"
echo "[run_drain] QK both/all_tokens  cfg=$QKCFG"
echo "============================================================"
echo "[run_drain] === QK 1/3 graph capture (buffer + FORCED drain) ==="
VLLM_HOOK_CONFIG_FILE="$QKCFG" VLLM_HOOK_QK_CAPTURE=buffer \
  VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$QK" capture --mode graph --out "$GQ" 2>&1 | tee "$WORK/qk_graph.log"
echo "[run_drain] === QK 2/3 eager capture (ground truth, no drain) ==="
VLLM_HOOK_CONFIG_FILE="$QKCFG" VLLM_HOOK_QK_CAPTURE=op \
  VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$QK" capture --mode eager --out "$EQ"
echo "[run_drain] === QK 3/3 compare (strict full-shape, q+k_all) ==="
python -u "$QK" compare --graph "$GQ" --eager "$EQ" || echo "[run_drain] QK FAILED"

# ---- HS both / all_tokens: graph (buffer, drained) vs eager (op, undrained) ----
HSCFG="model_configs/hidden_states/Qwen2-1.5B-Instruct_alltok.json"
GH="$WORK/hs_graph.pkl" EH="$WORK/hs_eager.pkl"
echo "============================================================"
echo "[run_drain] HS both/all_tokens  cfg=$HSCFG"
echo "============================================================"
echo "[run_drain] === HS 1/3 graph capture (buffer + FORCED drain) ==="
VLLM_HOOK_CONFIG_FILE="$HSCFG" VLLM_HOOK_HS_CAPTURE=buffer \
  VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$HS" capture --mode graph --out "$GH" 2>&1 | tee "$WORK/hs_graph.log"
echo "[run_drain] === HS 2/3 eager capture (ground truth, no drain) ==="
VLLM_HOOK_CONFIG_FILE="$HSCFG" VLLM_HOOK_HS_CAPTURE=op \
  VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$HS" capture --mode eager --out "$EH"
echo "[run_drain] === HS 3/3 compare (strict full-shape) ==="
python -u "$HS" compare --graph "$GH" --eager "$EH" || echo "[run_drain] HS FAILED"

# ---- non-vacuity: the drain MUST have fired, else PASS proves nothing ----
QK_DRAINS=$(grep -c '\[graph/drain\] drained' "$WORK/qk_graph.log" 2>/dev/null || true)
HS_DRAINS=$(grep -c '\[graph/drain\] drained' "$WORK/hs_graph.log" 2>/dev/null || true)
echo "[run_drain] DRAIN-FIRED: QK=$QK_DRAINS HS=$HS_DRAINS (each must be > 0)"
if [ "$QK_DRAINS" -gt 0 ] && [ "$HS_DRAINS" -gt 0 ]; then
  echo "[run_drain] VACUITY: PASS — drains fired, so the GPU->pinned cross-stream path was exercised."
else
  echo "[run_drain] VACUITY: FAIL — a drain never fired; lower VLLM_HOOK_CAPTURE_DRAIN_BYTES and rerun."
fi

echo "[run_drain] DONE — see VERDICT lines above (QK + HS must both PASS) + VACUITY."
