#!/bin/bash
# Bit-exactness oracle for the Triton-fused capture_hs scatter (fused vs aten, both dst modes).
# Submit from the repo root:  bsub -G grp_exploratory < tests/cuda_graph/tests/capture_perf/run_capture_fused_unit.sh
# Result: grep 'capture-fused] VERDICT' tests/cuda_graph/logs/capture_fused_unit.*.out
#
#BSUB -J vllm_hook_capture_fused_unit
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/capture_fused_unit.%J.out
#BSUB -e tests/cuda_graph/logs/capture_fused_unit.%J.err

source ~/.bashrc
conda activate vllm_hook_env
set -eo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_HOOK_ALLOW_CUDAGRAPH=1

# This oracle JIT-compiles many kernel specializations (dtype x HAS_RESIDUAL x SENTINEL)
# -- pre-empt the recurring
# "fatal error: error writing to /tmp/ccXXXXXX.s: No space left on device" gcc failure from the
# compute node's tiny /tmp by pointing scratch at local NVMe first.
export TMPDIR="/opt/nvme/${USER}/capfused_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR="/proj/dmfexp/fangyunh/scratch_capfused/tmp_${LSB_JOBID:-manual}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

cd /u/fangyunh/vLLM-Hook
echo "[job] host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
python tests/cuda_graph/tests/capture_perf/capture_fused_unit.py
