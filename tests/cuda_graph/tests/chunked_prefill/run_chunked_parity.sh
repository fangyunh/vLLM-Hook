#!/bin/bash
# run_chunked_parity.sh — verification (b): does the CURRENT buffer-mode QK capture
# correctly capture Q/K under CHUNKED PREFILL (a single prompt split across prefill
# steps)? We force chunking by booting with max_num_batched_tokens (MNBT) BELOW the
# long prompt's length; the buffer cap == MNBT, so positions past it are the suspected
# under-capture. Eager (paged-cache prefix-K reconstruction) is the ground truth.
#
# Per mode (all_tokens, last_token), in ONE job:
#   1. graph capture (FULL_DECODE_ONLY, buffer) of a SHORT (<MNBT) and a LONG (>MNBT) prompt
#   2. eager capture (op->register_forward_hook ground truth) of the same two prompts
#   3. compare: SHORT is the control (must match); LONG is the test (chunked).
# VERDICT per mode: PASS (chunked capture correct) | UNDER-CAPTURE CONFIRMED | INCONCLUSIVE.
#
# Submit:  bsub < tests/cuda_graph/tests/chunked_prefill/run_chunked_parity.sh
# Result:  grep -E 'VERDICT|coverage|chunks|UNDER-CAPTURE' tests/cuda_graph/logs/chunked_parity.*.out
#          grep 'over_cap' tests/cuda_graph/logs/chunked_parity.*.out   # routing drops past cap
#
#BSUB -J vllm_hook_chunked_parity
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/chunked_parity.%J.out
#BSUB -e tests/cuda_graph/logs/chunked_parity.%J.err
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
# Fixed long-prompt length (independent of MNBT) so it both chunks at MNBT=512 (3
# chunks) AND fits max_model_len=2048 for the MNBT=2048 no-chunk control.
export VLLM_HOOK_CHUNKED_LONG="${VLLM_HOOK_CHUNKED_LONG:-1472}"
export VLLM_HOOK_CHUNKED_SHORT="${VLLM_HOOK_CHUNKED_SHORT:-128}"

WORK="/dev/shm/vllm_hook_${USER}/chunked_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/chunked_prefill/chunked_prefill_parity.py

run_mode () {
  local tag="$1"
  local cfg="$2"
  local mnbt="$3"   # buffer cap == MNBT; long prompt (~1472) chunks iff MNBT<1472
  echo "============================================================"
  echo "[run_chunked] MODE=$tag  config=$cfg  MNBT=$mnbt"
  echo "============================================================"
  local G="$WORK/${tag}_graph.pkl"
  local E="$WORK/${tag}_eager.pkl"
  echo "[run_chunked] === $tag 1/3 graph capture (FULL_DECODE_ONLY, buffer) ==="
  VLLM_HOOK_CHUNKED_MNBT="$mnbt" VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=buffer \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" capture --mode graph --out "$G" \
    || { echo "[run_chunked] MODE=$tag GRAPH CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_chunked] === $tag 2/3 eager capture (ground truth) ==="
  VLLM_HOOK_CHUNKED_MNBT="$mnbt" VLLM_HOOK_CONFIG_FILE="$cfg" VLLM_HOOK_QK_CAPTURE=op \
    VLLM_HOOK_PARITY_HOOKS_ON=prefill python -u "$PY" capture --mode eager --out "$E" \
    || { echo "[run_chunked] MODE=$tag EAGER CAPTURE FAILED — skipping leg"; return 0; }
  echo "[run_chunked] === $tag 3/3 compare (short=control, long=test) ==="
  VLLM_HOOK_CHUNKED_MNBT="$mnbt" VLLM_HOOK_CONFIG_FILE="$cfg" \
    python -u "$PY" compare --graph "$G" --eager "$E" \
    || echo "[run_chunked] MODE=$tag returned non-zero (see VERDICT above)"
}

ALLTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
LASTTOK="model_configs/attention_tracker/Qwen2-1.5B-Instruct.json"

# Chunked legs (MNBT=512 < 1472 -> 3 chunks): the fix under test.
run_mode chunk_alltok    "$ALLTOK"  512
run_mode chunk_lasttoken "$LASTTOK" 512
# No-chunk CONTROL (MNBT=2048 >= 1472 -> single prefill step): isolates baseline
# compiled-vs-eager fp drift on the same long sequence. If the long prompt drifts the
# SAME here as in the chunked leg, the drift is numeric (depth-accumulated), not chunking.
run_mode nochunk_alltok  "$ALLTOK"  2048

echo "[run_chunked] DONE — VERDICT lines above. Compare chunk_alltok vs nochunk_alltok long-prompt drift."
