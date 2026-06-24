#!/bin/bash
# run_capture_perf_qkdrain.sh — DIAGNOSTIC for the QK capture-on-decode regression
# (graph 49.98 vs eager 42.54 ms/step, all_tokens N=128). Tests whether the W4 drain
# (freeing GPU-resident k_all clones to pinned host) alone closes the gap — i.e. whether
# the bottleneck is GPU allocator pressure from the O(N^2) accumulating clones vs the
# clone/host-loop work itself. Runs QK three ways and prints all three capture ms/step:
#   graph-nodrain (the 49.98 baseline) | graph-drain (8 MiB budget) | eager (42.54).
#
# Submit:  bsub < tests/cuda_graph/tests/capture_perf/run_capture_perf_qkdrain.sh
# Result:  grep -E 'cap-perf|QKDRAIN' tests/cuda_graph/logs/capture_perf_qkdrain.*.out
#
#BSUB -J vllm_hook_qkdrain
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/capture_perf_qkdrain.%J.out
#BSUB -e tests/cuda_graph/logs/capture_perf_qkdrain.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_HOOK_PERF_MAX_TOKENS="${VLLM_HOOK_PERF_MAX_TOKENS:-128}"
export VLLM_HOOK_PERF_REPEATS="${VLLM_HOOK_PERF_REPEATS:-3}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_HOOK_PERF_WORKER=qk
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json"

WORK="/dev/shm/vllm_hook_${USER}/qkdrain_${LSB_JOBID:-manual}"
mkdir -p "$WORK"
PY=tests/cuda_graph/tests/capture_perf/capture_perf.py

echo "[qkdrain] === graph NO-drain (baseline) ==="
python -u "$PY" measure --mode graph --out "$WORK/nodrain.json"

echo "[qkdrain] === graph WITH drain (8 MiB budget) ==="
VLLM_HOOK_CAPTURE_DRAIN_BYTES=8388608 \
  python -u "$PY" measure --mode graph --out "$WORK/drain.json"

echo "[qkdrain] === eager ==="
python -u "$PY" measure --mode eager --out "$WORK/eager.json"

echo "[qkdrain] === summary ==="
python - "$WORK/nodrain.json" "$WORK/drain.json" "$WORK/eager.json" <<'PYEOF'
import json, sys
nd, dr, ea = (json.load(open(p)) for p in sys.argv[1:4])
print(f"QKDRAIN base no-hook (graph): {nd['base_ms_per_step']:.3f} ms/step")
print(f"QKDRAIN capture graph NO-drain : {nd['capture_ms_per_step']:.3f} ms/step")
print(f"QKDRAIN capture graph DRAIN-8M : {dr['capture_ms_per_step']:.3f} ms/step")
print(f"QKDRAIN capture EAGER          : {ea['capture_ms_per_step']:.3f} ms/step")
better = "DRAIN BEATS EAGER" if dr['capture_ms_per_step'] <= ea['capture_ms_per_step'] else "DRAIN STILL > EAGER"
print(f"QKDRAIN VERDICT: {better} "
      f"(drain {dr['capture_ms_per_step']:.1f} vs eager {ea['capture_ms_per_step']:.1f})")
PYEOF
echo "[qkdrain] DONE"
