#!/bin/bash
# run_qk_parity_full.sh — graph-vs-eager QK parity under FULL_DECODE_ONLY (v0.4.0 M2):
# eager prefill + FULL cudagraph decode, BUFFER capture mode. This is THE FULL capture
# milestone. Buffer mode declares NO splitting op, so the capture_qk scatter is absorbed
# into the single full decode cudagraph and replays every decode step; routing/egress is
# host Python in the execute_model wrapper, OUTSIDE the graph, in both phases.
#
# Two hook-activity configs (the user's ask), each its own leg in ONE job:
#   leg A (prefill-only): hooks_on=prefill, max_tokens=12 — prefill captured (eager path),
#       every decode token routes to the sentinel -> the baked scatter is a true no-op.
#       Proves the full-graph decode op costs nothing and corrupts nothing; prefill parity
#       holds. (config 1)
#   leg B (both): hooks_on=both, max_tokens=12 — eager-prefill capture + full-graph-decode
#       capture, parity vs eager for every captured token (q AND k_all). Buffer mode routes
#       by ABSOLUTE sequence position so k_buf ACCUMULATES the full key history across decode
#       steps (the buffer analogue of the eager path's prefix-K reconstruction), so k_all
#       matches eager's full reconstruction with no NO_PREFIXK crutch. (config 2)
#
# Both legs use the all_tokens / all_layers config (strongest: full per-token q/k slice,
# strict full-shape compare via MAX_TOKENS>1 — a dropped decode token FAILS as a mismatch).
#
# Submit:  bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity_full.sh
# Result:  grep 'parity\] VERDICT' tests/cuda_graph/logs/qk_parity_full.*.out
#          grep 'splitting op\|QK static buffers\|cudagraph_mode' tests/cuda_graph/logs/qk_parity_full.*.out
#
#BSUB -J vllm_hook_qk_parity_full
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_parity_full.%J.out
#BSUB -e tests/cuda_graph/logs/qk_parity_full.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill, full cudagraph decode.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_QK_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-12}"

CFG="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"
export VLLM_HOOK_CONFIG_FILE="$CFG"

WORK="/dev/shm/vllm_hook_${USER}/parity_full_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/qk_graph/qk_parity.py

echo "[run_qk_parity_full] cudagraph_mode=FULL_DECODE_ONLY capture=buffer config=$CFG max_tokens=$VLLM_HOOK_PARITY_MAX_TOKENS"

# ---- leg A: prefill-only (decode is a baked no-op under the full decode graph) ----
GA="$WORK/prefillonly_graph.pkl" EA="$WORK/prefillonly_eager.pkl"
echo "============================================================"
echo "[run_qk_parity_full] LEG A = prefill-only (hooks_on=prefill)"
echo "============================================================"
echo "[run_qk_parity_full] === A 1/3 graph capture (FULL_DECODE_ONLY, buffer) ==="
VLLM_HOOK_PARITY_HOOKS_ON=prefill VLLM_HOOK_QK_CAPTURE=buffer \
  python -u "$PY" capture --mode graph --out "$GA"
echo "[run_qk_parity_full] === A 2/3 eager capture (ground truth) ==="
VLLM_HOOK_PARITY_HOOKS_ON=prefill VLLM_HOOK_QK_CAPTURE=op \
  python -u "$PY" capture --mode eager --out "$EA"
echo "[run_qk_parity_full] === A 3/3 compare ==="
python -u "$PY" compare --graph "$GA" --eager "$EA" || echo "[run_qk_parity_full] LEG A FAILED"

# ---- leg B: both (prefill + every decode step); k_buf accumulation gives full k_all ----
GB="$WORK/both_graph.pkl" EB="$WORK/both_eager.pkl"
echo "============================================================"
echo "[run_qk_parity_full] LEG B = both (hooks_on=both; absolute-position k_buf accumulation)"
echo "============================================================"
echo "[run_qk_parity_full] === B 1/3 graph capture (FULL_DECODE_ONLY, buffer, decode) ==="
VLLM_HOOK_PARITY_HOOKS_ON=both VLLM_HOOK_QK_CAPTURE=buffer \
  python -u "$PY" capture --mode graph --out "$GB"
echo "[run_qk_parity_full] === B 2/3 eager capture (ground truth, full prefix-K) ==="
VLLM_HOOK_PARITY_HOOKS_ON=both VLLM_HOOK_QK_CAPTURE=op \
  python -u "$PY" capture --mode eager --out "$EB"
echo "[run_qk_parity_full] === B 3/3 compare (strict full-shape: prefill+decode, q+k_all) ==="
python -u "$PY" compare --graph "$GB" --eager "$EB" || echo "[run_qk_parity_full] LEG B FAILED"

echo "[run_qk_parity_full] DONE — see VERDICT lines above per leg."
