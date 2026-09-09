#!/bin/bash
# run_writer_process.sh — Rank-1c equivalence gate: VLLM_HOOK_WRITER_PROCESS off (thread) vs
# on (process) must produce IDENTICAL on-disk artifacts, under FULL cudagraph + buffer capture.
#
# Rank 1c moves the disk serialize (pad + safetensors-encode / pickle) off the engine GIL into
# a spawned writer CHILD process. Both the thread path and the process path call the SAME pure
# graph/artifact_writer.write_artifact, and the cpu_cache crosses to the child via a torch.mp
# shm handoff. This oracle captures the same prompts twice in SEPARATE subprocesses
# (exclusive_process GPU), wp=0 (thread) then wp=1 (process), and asserts the on-disk artifacts
# are identical:
#   .safetensors -> raw-byte equal; .json -> equal (profile stripped); .pt -> torch.equal.
# The wp=1 run is proven to have serialized IN the writer process (queue_put>0, in-worker
# disk_write==0) so byte-equality is not a hollow green. Each leg: capture wp=0 -> wp=1 -> compare.
#
# Submit:  bsub < tests/cuda_graph/tests/capture_perf/run_writer_process.sh
# Result:  grep -E 'writer-equiv\] VERDICT' tests/cuda_graph/logs/writer_process.*.out
#
#BSUB -J vllm_hook_writer_proc
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/writer_process.%J.out
#BSUB -e tests/cuda_graph/logs/writer_process.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
# The FULL target: eager prefill, full cudagraph decode, buffer capture.
export VLLM_HOOK_CUDAGRAPH_MODE=FULL
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

WORK="/dev/shm/vllm_hook_${USER}/writer_process_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/capture_perf/writer_process_equiv.py

echo "[run_writer_process] cudagraph_mode=FULL capture=buffer — thread-save vs process-save identical"

run_leg () {
  # args: worker gran fmt
  # Thread control (wp=0) is captured ONCE; then the writer process (wp=1) is captured under
  # BOTH VLLM_HOOK_WRITER_PACK=1 (default single-mapping pack) and =0 (legacy per-tensor
  # handoff), each compared byte-for-byte against the same thread control. Both must be
  # identical: pack-on proves the fix preserves bytes, pack-off proves the escape hatch does too.
  local worker="$1" gran="$2" fmt="$3"
  local name="${worker}_${gran}_${fmt}"
  local off="$WORK/${name}_thread.pkl"
  local hd_off="$WORK/${name}_hd_thread"
  echo "============================================================"
  echo "[run_writer_process] LEG=$name worker=$worker gran=$gran fmt=$fmt"
  echo "============================================================"
  echo "[run_writer_process] === ${name} capture WRITER_PROCESS=0 (thread control) ==="
  python -u "$PY" capture --worker "$worker" --gran "$gran" --fmt "$fmt" --wp 0 \
    --hookdir "$hd_off" --out "$off"
  local pack
  for pack in 1 0; do
    local on="$WORK/${name}_process_pack${pack}.pkl"
    local hd_on="$WORK/${name}_hd_process_pack${pack}"
    echo "[run_writer_process] === ${name} capture WRITER_PROCESS=1 WRITER_PACK=${pack} ==="
    VLLM_HOOK_WRITER_PACK="$pack" python -u "$PY" capture --worker "$worker" --gran "$gran" \
      --fmt "$fmt" --wp 1 --hookdir "$hd_on" --out "$on"
    echo "[run_writer_process] === ${name} compare thread vs process(pack=${pack}) ==="
    python -u "$PY" compare --a "$off" --b "$on" \
      || echo "[run_writer_process] LEG $name pack=$pack FAILED"
  done
}

# ---- safetensors path (pad + safetensors-encode in the writer process) ----
run_leg qk last_token st
run_leg qk all_tokens st
run_leg hs last_token st
run_leg hs all_tokens st

# ---- .pt path (pickle in the writer process) ----
run_leg qk last_token pt
run_leg hs last_token pt

echo "[run_writer_process] DONE — see VERDICT lines above per leg."
echo "grep this: grep -E 'writer-equiv\] VERDICT' tests/cuda_graph/logs/writer_process.*.out"
