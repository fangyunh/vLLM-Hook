#!/bin/bash
# run_profile_churn.sh — localize the CB HS residual plugin tax (v0.5.7).
# Profiles a churning buffer-HS batch; reads the worker PROF dump for graph.route /
# graph.egress timings + upload/noupload/skip counts (does A skip under churn?).
#
#BSUB -J vllm_hook_profile_churn
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[ngpus=1,mem=48GB]"
#BSUB -o tests/cuda_graph/logs/profile_churn.%J.out
#BSUB -e tests/cuda_graph/logs/profile_churn.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export VLLM_HOOK_HS_CAPTURE=buffer
export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_HOOK_QK_DEBUG=0
# Profiling on; dump to a shared path keyed by job id.
PROFDIR="tests/cuda_graph/logs/prof_churn_${LSB_JOBID:-manual}"
mkdir -p "$PROFDIR"
export VLLM_HOOK_PROFILE=1
export VLLM_HOOK_PROFILE_DIR="$PROFDIR"
export CHURN_NREQ="${CHURN_NREQ:-600}"
export CHURN_WORKER="${CHURN_WORKER:-hs}"

PY=tests/cuda_graph/tests/capture_perf/profile_churn.py

echo "=== A ON (default) — VLLM_HOOK_INCREMENTAL_ROUTING=1, EGRESS_STREAM=1 ==="
VLLM_HOOK_INCREMENTAL_ROUTING=1 VLLM_HOOK_EGRESS_STREAM=1 \
  python -u "$PY" || echo "[run] A-ON leg failed"

echo "=== A OFF (legacy full-rebuild control) — INCREMENTAL_ROUTING=0, EGRESS_STREAM=0 ==="
VLLM_HOOK_INCREMENTAL_ROUTING=0 VLLM_HOOK_EGRESS_STREAM=0 \
  VLLM_HOOK_PROFILE_DIR="${PROFDIR}_legacy" \
  python -u "$PY" || echo "[run] A-OFF leg failed"

echo "============================================================"
echo "[run] PROF dumps (worker side):"
python - <<'PY'
import json, glob, os
for tag, d in [("A-ON", os.environ["VLLM_HOOK_PROFILE_DIR"]),
               ("A-OFF(legacy)", os.environ["VLLM_HOOK_PROFILE_DIR"] + "_legacy")]:
    files = sorted(glob.glob(os.path.join(d, "profile-worker*.json")))
    print(f"\n===== {tag} :: {d} ({len(files)} worker dumps) =====")
    # merge all worker dumps (usually 1 for TP=1)
    timers, counters = {}, {}
    for f in files:
        snap = json.load(open(f))
        for k, v in snap.get("timers", {}).items():
            timers.setdefault(k, {"count": 0, "sum": 0.0})
            timers[k]["count"] += v.get("count", 0); timers[k]["sum"] += v.get("sum", 0.0)
        for k, v in snap.get("counters", {}).items():
            counters[k] = counters.get(k, 0) + v
    def show(k):
        t = timers.get(k)
        if t and t["count"]:
            print(f"  {k:22s} calls={t['count']:>7d}  total={t['sum']:8.1f}ms  mean={t['sum']/t['count']*1000:7.1f}us/call")
    for k in ("graph.route", "graph.egress", "kv.prefix_recon"):
        show(k)
    print("  counters:", {k: counters[k] for k in sorted(counters)
                          if any(s in k for s in ("route", "egress", "fire", "throttl"))})
    up = counters.get("graph.route.upload", 0); no = counters.get("graph.route.noupload", 0)
    sk = counters.get("graph.route.skip", 0); tot = up + no + sk
    if tot:
        print(f"  ROUTING STEPS: upload={up} ({100*up/tot:.0f}%)  noupload/skip-via-diff={no} "
              f"({100*no/tot:.0f}%)  key-skip={sk} ({100*sk/tot:.0f}%)  total={tot}")
PY
echo "[run] DONE"
