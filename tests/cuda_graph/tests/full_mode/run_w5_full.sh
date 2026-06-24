#!/bin/bash
# run_w5_full.sh — v0.5.0 Step 5 (W5): FULL (mixed-step) CUDA graphs. Under
# cudagraph_mode=FULL the vLLM dispatcher graphs MIXED prefill+decode steps too (which
# FULL_DECODE_ONLY runs eager) — the win for a continuous-batching server where most
# steps are mixed. Buffer mode needs NO plugin change: the splitting-op gate already
# declares no splitting op for buffer (gated on the capture-mode env, not cudagraph_mode),
# so the scatter/steer op is absorbed into the prefill/mixed graphs too and replays.
#
# This validates feasibility (parity holds under FULL) + measures the cost (bigger
# cudagraph pool -> fewer KV-cache blocks):
#   1. QK capture parity under FULL (both/all_tokens) — graph vs eager.
#   2. Steer parity under FULL (add_vector) — graph vs eager, token-for-token.
#   3. KV-cache delta: # GPU blocks under FULL vs FULL_DECODE_ONLY (INFO log grep).
# PASS = both parity VERDICTs PASS (FULL graphs the prefill/mixed steps without breaking
# capture/steer); the KV-cache line quantifies the memory cost to weigh per deployment.
#
# Submit:  bsub < tests/cuda_graph/tests/full_mode/run_w5_full.sh
# Result:  grep -E 'VERDICT|W5-KV' tests/cuda_graph/logs/w5_full.*.out
#
#BSUB -J vllm_hook_w5_full
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/w5_full.%J.out
#BSUB -e tests/cuda_graph/logs/w5_full.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-12}"

WORK="/dev/shm/vllm_hook_${USER}/w5_full_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
QK=tests/cuda_graph/tests/qk_graph/qk_parity.py
ST=tests/cuda_graph/tests/steer_graph/steer_parity_decode.py

# ---------- 1. QK capture parity under FULL (both / all_tokens) ----------
echo "============================================================"
echo "[w5] 1) QK capture parity under cudagraph_mode=FULL (both/all_tokens)"
echo "============================================================"
QKCFG="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
GQ="$WORK/qk_full_graph.pkl" EQ="$WORK/qk_full_eager.pkl"
VLLM_LOGGING_LEVEL=WARNING VLLM_HOOK_CUDAGRAPH_MODE=FULL VLLM_HOOK_CONFIG_FILE="$QKCFG" \
  VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$QK" capture --mode graph --out "$GQ"
VLLM_LOGGING_LEVEL=WARNING VLLM_HOOK_CONFIG_FILE="$QKCFG" \
  VLLM_HOOK_QK_CAPTURE=op VLLM_HOOK_PARITY_HOOKS_ON=both \
  python -u "$QK" capture --mode eager --out "$EQ"
python -u "$QK" compare --graph "$GQ" --eager "$EQ" || echo "[w5] QK FULL FAILED"

# ---------- 2. Steer parity under FULL ----------
echo "============================================================"
echo "[w5] 2) Steer parity under cudagraph_mode=FULL (add_vector)"
echo "============================================================"
STCFG="model_configs/activation_steer/Qwen2-1.5B-Instruct.json"
GS="$WORK/steer_full_graph.pkl" ES="$WORK/steer_full_eager.pkl"
VLLM_LOGGING_LEVEL=WARNING VLLM_HOOK_CUDAGRAPH_MODE=FULL VLLM_HOOK_CONFIG_FILE="$STCFG" \
  VLLM_HOOK_STEER_MODE=buffer python -u "$ST" capture --mode graph --out "$GS"
VLLM_LOGGING_LEVEL=WARNING VLLM_HOOK_CONFIG_FILE="$STCFG" \
  VLLM_HOOK_STEER_MODE=op python -u "$ST" capture --mode eager --out "$ES"
python -u "$ST" compare --graph "$GS" --eager "$ES" || echo "[w5] steer FULL FAILED"

# ---------- 3. KV-cache delta: FULL vs FULL_DECODE_ONLY (INFO log grep) ----------
echo "============================================================"
echo "[w5] 3) KV-cache cost: # GPU blocks FULL vs FULL_DECODE_ONLY"
echo "============================================================"
kv_for_mode () {
  local mode="$1"
  local log="$WORK/kv_${mode}.log"
  # One throwaway graph boot at INFO so vLLM prints the KV-cache sizing line.
  VLLM_LOGGING_LEVEL=INFO VLLM_HOOK_CUDAGRAPH_MODE="$mode" VLLM_HOOK_CONFIG_FILE="$QKCFG" \
    VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_PARITY_HOOKS_ON=both VLLM_HOOK_PARITY_MAX_TOKENS=2 \
    python -u "$QK" capture --mode graph --out "$WORK/kv_${mode}.pkl" > "$log" 2>&1 || true
  local kv
  kv=$(grep -iE 'GPU KV cache size|# GPU blocks|GPU blocks:|KV cache size|Maximum concurrency|num_gpu_blocks' "$log" | tail -3)
  echo "[w5] W5-KV mode=$mode ::"
  echo "$kv" | sed 's/^/[w5]   /'
}
kv_for_mode FULL_DECODE_ONLY
kv_for_mode FULL

echo "[w5] DONE — QK + steer parity VERDICTs above must PASS; W5-KV lines quantify the KV-cache cost."
