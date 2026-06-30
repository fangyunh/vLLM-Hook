#!/bin/bash
# run_qk_score_graph.sh — v0.6.0 score-capture parity for the GRAPH paths.
#
# Captures the score under op-mode (PIECEWISE) and buffer-mode (FULL_DECODE_ONLY),
# plus the eager QK ground truth, then compares each score against the recompute.
# Five sequential processes in one job (GPU is exclusive_process).
#
# Submit:  bsub -G grp_exploratory < tests/cuda_graph/tests/qk_score/run_qk_score_graph.sh
# Result:  grep 'score-parity] VERDICT' tests/cuda_graph/logs/qk_score_graph.*.out
#
#BSUB -J vllm_hook_qk_score_graph
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_score_graph.%J.out
#BSUB -e tests/cuda_graph/logs/qk_score_graph.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_QK_SCORE_HEAD="${VLLM_HOOK_QK_SCORE_HEAD:-0}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
# >1 so the buffer (FULL_DECODE_ONLY) arm actually REPLAYS decode cudagraphs (not just eager
# prefill) and the score path's multi-pass accumulation is exercised; the harness also
# asserts the capture op fired (no silent eager fallback). compare still checks the prefill pass.
export VLLM_HOOK_PARITY_MAX_TOKENS="${VLLM_HOOK_PARITY_MAX_TOKENS:-8}"

WORK="/dev/shm/vllm_hook_${USER}/score_graph_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/qk_score/qk_score_parity.py

echo "[run_qk_score_graph] === 1/5 op-mode (PIECEWISE) score ==="
VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=op VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE \
    python -u "$PY" capture --mode score --out "$WORK/score_op.pkl"

echo "[run_qk_score_graph] === 2/5 buffer-mode (FULL_DECODE_ONLY) score ==="
VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
    python -u "$PY" capture --mode score --out "$WORK/score_buffer.pkl"

echo "[run_qk_score_graph] === 3/5 eager qk ground truth ==="
python -u "$PY" capture --mode qk --out "$WORK/qk.pkl"

echo "[run_qk_score_graph] === 4/7 compare op vs qk ==="
python -u "$PY" compare --score "$WORK/score_op.pkl" --qk "$WORK/qk.pkl"
echo "[run_qk_score_graph] === 5/7 compare buffer (all_tokens) vs qk ==="
python -u "$PY" compare --score "$WORK/score_buffer.pkl" --qk "$WORK/qk.pkl"

# v0.5.7 D1: buffer-mode LAST_TOKEN score computes the [1,S_k] score ON-GPU at EGRESS from the
# staged last query + the FULL key history read from paged KV (never clones K). Compare vs the
# LAST ROW of the QK recompute.
echo "[run_qk_score_graph] === 6/7 buffer-mode (FULL_DECODE_ONLY) last_token score (D1 on-GPU egress) ==="
VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
    python -u "$PY" capture --mode score --hookq last_token --out "$WORK/score_buffer_lasttok.pkl"
echo "[run_qk_score_graph] === 7/7 compare buffer last_token vs qk (last row) ==="
python -u "$PY" compare --score "$WORK/score_buffer_lasttok.pkl" --qk "$WORK/qk.pkl" --score-lasttok

# ---- v0.5.7 D2: MULTI-HEAD score parity (important_heads -> output_qk {layer:[heads]} dict
# -> score the whole head set, [n,S_q,S_k]). VLLM_HOOK_SCORE_MH=1 selects the *_mh configs;
# each captured head is compared independently against the QK recompute for that head. ----
echo "[run_qk_score_graph] === D2 8/11 op-mode (PIECEWISE) MULTI-HEAD all_tokens score ==="
VLLM_HOOK_SCORE_MH=1 VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=op VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE \
    python -u "$PY" capture --mode score --out "$WORK/score_op_mh.pkl"
echo "[run_qk_score_graph] === D2 9/11 buffer-mode MULTI-HEAD all_tokens score ==="
VLLM_HOOK_SCORE_MH=1 VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
    python -u "$PY" capture --mode score --out "$WORK/score_buffer_mh.pkl"
echo "[run_qk_score_graph] === D2 10/11 buffer-mode MULTI-HEAD last_token score (D1 on-GPU egress) ==="
VLLM_HOOK_SCORE_MH=1 VLLM_HOOK_SCORE_GRAPH=1 VLLM_HOOK_QK_CAPTURE=buffer VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
    python -u "$PY" capture --mode score --hookq last_token --out "$WORK/score_buffer_lasttok_mh.pkl"
echo "[run_qk_score_graph] === D2 11/11 compare multi-head (op, buffer all_tokens, buffer last_token) vs qk ==="
python -u "$PY" compare --score "$WORK/score_op_mh.pkl"             --qk "$WORK/qk.pkl"
python -u "$PY" compare --score "$WORK/score_buffer_mh.pkl"         --qk "$WORK/qk.pkl"
python -u "$PY" compare --score "$WORK/score_buffer_lasttok_mh.pkl" --qk "$WORK/qk.pkl" --score-lasttok
