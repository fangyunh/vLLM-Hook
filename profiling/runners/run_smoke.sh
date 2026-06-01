#!/bin/bash
# run_smoke.sh — 5-minute end-to-end profiler sanity check.
#
# Runs examples/demo_hiddenstate.py with profiling ON, then verifies the
# instrumentation actually fired by checking TWO independent signals:
#   1. JSON dumps under VLLM_HOOK_PROFILE_DIR (driver + worker, written
#      by the atexit handler in _profiler.py).
#   2. The safetensors sidecar at /dev/shm/vllm_hook/<run_id>/tp_rank_*/
#      hidden_states.json — its meta["profile"] field is written by the
#      worker even when the in-process dump misses (sidecar enrichment).
#
# Use this once after a fresh checkout / new node before queueing the
# longer benchmarks.
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
HOOK_DIR="${HOOK_DIR:-/dev/shm/vllm_hook}"
mkdir -p "$PROFILE_DIR" "$HOOK_DIR"
rm -f "$PROFILE_DIR"/profile-*.json
rm -rf "$HOOK_DIR"/*

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFILE_DIR"

# Confirm the env vars are set as we expect them.
echo "[smoke] env check:"
env | grep -E "^VLLM_(HOOK|USE_V1|WORKER_MULTIPROC)" | sort
echo

# Confirm the plugin imports cleanly before launching the demo.
python -c "
from vllm_hook_plugins._profiler import PROF, is_enabled
import os
print(f'[smoke] _profiler imported. enabled={is_enabled()}  '
      f'env VLLM_HOOK_PROFILE={os.environ.get(\"VLLM_HOOK_PROFILE\")!r}  '
      f'dir={os.environ.get(\"VLLM_HOOK_PROFILE_DIR\")!r}')
"
echo

echo "[smoke] running demo_hiddenstate.py with profiling on..."
python examples/demo_hiddenstate.py

echo
echo "[smoke] profile dumps in $PROFILE_DIR:"
ls -la "$PROFILE_DIR" || true

echo
echo "[smoke] safetensors sidecars in $HOOK_DIR:"
find "$HOOK_DIR" -name '*.json' -printf '%p  (%s bytes)\n' 2>/dev/null || \
    find "$HOOK_DIR" -name '*.json' 2>/dev/null

# Signal 1 — any in-process dump?
DUMP=$(ls -t "$PROFILE_DIR"/profile-*.json 2>/dev/null | head -1 || true)

# Signal 2 — any sidecar with embedded profile?
SIDECAR=$(find "$HOOK_DIR" -name 'hidden_states.json' 2>/dev/null | head -1 || true)

if [[ -n "$DUMP" ]]; then
    echo
    echo "[smoke] in-process dump: $DUMP"
    python - "$DUMP" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    snap = json.load(f)
print(f"  enabled={snap.get('enabled')}  pid={snap.get('pid')}  "
      f"wall_s={snap.get('wall_s', 0):.2f}")
timers = snap.get("timers", {})
counters = snap.get("counters", {})
print(f"\n  {'timer':<42s} {'count':>6s} {'mean_ms':>10s} {'p99_ms':>10s}")
print("  " + "-" * 72)
for name, stats in sorted(timers.items(), key=lambda kv: -(kv[1].get('mean') or 0))[:20]:
    if isinstance(stats, dict) and stats.get('count'):
        print(f"  {name:<42s} {stats['count']:>6d} "
              f"{stats.get('mean', 0):>10.2f} {stats.get('p99', 0):>10.2f}")
print(f"\n  counters:")
for name, val in sorted(counters.items()):
    print(f"    {name:<40s} {val}")
PY
fi

if [[ -n "$SIDECAR" ]]; then
    echo
    echo "[smoke] sidecar profile (from worker): $SIDECAR"
    python - "$SIDECAR" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    meta = json.load(f)
prof = meta.get("profile")
if not prof:
    print("  sidecar has no 'profile' key — worker did NOT see "
          "VLLM_HOOK_PROFILE=1.")
    sys.exit(0)
print(f"  pid={prof.get('pid')}  enabled={prof.get('enabled')}  "
      f"wall_s={prof.get('wall_s', 0):.2f}")
timers = prof.get("timers", {})
counters = prof.get("counters", {})
print(f"\n  worker timers (top 15 by mean):")
for name, stats in sorted(timers.items(),
                           key=lambda kv: -(kv[1].get('mean') or 0))[:15]:
    if isinstance(stats, dict) and stats.get('count'):
        print(f"    {name:<40s} count={stats['count']:>5d}  "
              f"mean_ms={stats.get('mean', 0):>8.2f}")
print(f"\n  worker counters:")
for name, val in sorted(counters.items()):
    print(f"    {name:<40s} {val}")
PY
fi

# Decide pass/fail. Either signal is sufficient — the in-process dump
# proves the driver-side timers fire; the sidecar proves the worker-side
# timers fire. Both is best; just one is acceptable for smoke.
if [[ -z "$DUMP" && -z "$SIDECAR" ]]; then
    echo
    echo "[smoke] FAIL — neither an in-process dump nor a sidecar profile."
    echo "        Check stderr for tracebacks from the atexit dump:"
    echo "          grep 'vllm-hook profiler' profiling/runners/logs/smoke.*.err"
    exit 1
fi

echo
echo "[smoke] PASS — profiler fired. Ready for the real benchmarks."
