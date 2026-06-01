#!/bin/bash
# run_smoke.sh — 5-minute end-to-end profiler sanity check.
#
# Runs the existing examples/demo_hiddenstate.py with profiling ON, then
# prints the per-stage timer table from the resulting JSON. Use this once
# after a fresh checkout / new node to confirm the instrumentation actually
# fires before queueing the longer benchmarks.
#
# Submit:  bsub < profiling/runners/run_smoke.sh
# Logs:    profiling/runners/logs/smoke.<JOBID>.{out,err}
#BSUB -J vllm_hook_prof_smoke
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o profiling/runners/logs/smoke.%J.out
#BSUB -e profiling/runners/logs/smoke.%J.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p profiling/runners/logs

PROFILE_DIR="${PROFILE_DIR:-/dev/shm/vllm_hook_profile_smoke}"
mkdir -p "$PROFILE_DIR"
rm -f "$PROFILE_DIR"/profile-*.json

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"

echo "[smoke] running demo_hiddenstate.py with profiling on..."
python examples/demo_hiddenstate.py

echo
echo "[smoke] profile dumps in $PROFILE_DIR:"
ls -la "$PROFILE_DIR"
LATEST=$(ls -t "$PROFILE_DIR"/profile-*.json 2>/dev/null | head -1 || true)
if [[ -z "$LATEST" ]]; then
    echo "[smoke] FAIL — no profile JSON was written."
    echo "         Check VLLM_HOOK_PROFILE is propagating to the worker process."
    exit 1
fi

echo
echo "[smoke] timers from $LATEST:"
python - "$LATEST" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    snap = json.load(f)
timers = snap.get("timers", {})
counters = snap.get("counters", {})
print(f"{'timer':<42s} {'count':>6s} {'mean_ms':>10s} {'p99_ms':>10s}")
print("-" * 74)
for name, stats in sorted(timers.items(), key=lambda kv: -(kv[1].get('mean') or 0)):
    if isinstance(stats, dict) and stats.get('count'):
        print(f"{name:<42s} {stats['count']:>6d} "
              f"{stats.get('mean', 0):>10.2f} {stats.get('p99', 0):>10.2f}")
print()
print("counters:")
for name, val in sorted(counters.items()):
    print(f"  {name:<40s} {val}")
PY

echo
echo "[smoke] PASS — profiler fired and dump is readable. Ready for the real benchmarks."
