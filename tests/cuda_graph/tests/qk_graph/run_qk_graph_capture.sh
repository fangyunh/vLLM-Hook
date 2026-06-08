#!/bin/bash
# run_qk_graph_capture.sh — v0.3.0 QK CUDA-graph capture verification (file 5/5),
# run as a BATCH (non-interactive) LSF job. Output goes to log files under
# tests/cuda_graph/logs/, not your terminal.
#
# Submit:  mkdir -p tests/cuda_graph/logs
#          bsub < tests/cuda_graph/tests/qk_graph/run_qk_graph_capture.sh
# Watch:   bjobs                                              # status
#          bpeek <JOBID>                                      # peek live output
#          tail -f tests/cuda_graph/logs/qk_graph.*.out       # follow stdout
# Result:  grep 'VERDICT' tests/cuda_graph/logs/qk_graph.*.out
#          PASS -> exit 0; FAIL -> exit 1 (the python sets the exit code).
#
# Optional parity arm (eager-vs-graph q/k match): export VLLM_HOOK_PARITY=1
# before bsub, or uncomment the export below.
#
#BSUB -J vllm_hook_qk_graph
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/qk_graph.%J.out
#BSUB -e tests/cuda_graph/logs/qk_graph.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook

mkdir -p tests/cuda_graph/logs

# disk-st-async storage variant (fastest), matching run_demo.sh / run_qk_noneager.sh.
export VLLM_HOOK_USE_SAFETENSORS=1
export VLLM_HOOK_ASYNC_SAVE=1

# REQUIRED: the plugin's engine-config patch (_hook_plugin.py) otherwise
# hardcodes enforce_eager=True, silently disabling CUDA graphs no matter what
# cudagraph_mode we pass. This opt-out keeps enforce_eager=False as the caller
# set it AND arms the v0.3.0 graph install path.
export VLLM_HOOK_ALLOW_CUDAGRAPH=1

# Force a clean compile each run so a stale torch.compile cache cannot mask a
# regression in the capture op / class-wrap. (Production cache-signature mixing
# is a later step.)
export VLLM_DISABLE_COMPILE_CACHE=1

# CUDA-graph mode. Switch by commenting/uncommenting one of these lines.
# PIECEWISE        = mandatory/shippable mode; attention stays an eager seam
#                    (the FX split seam where the Attention.forward wrap is traced in).
# FULL_DECODE_ONLY = decode folds into one graph; best-effort capture (fragile).
export VLLM_HOOK_CUDAGRAPH_MODE=PIECEWISE
# export VLLM_HOOK_CUDAGRAPH_MODE=FULL_DECODE_ONLY

# Optional eager-vs-graph per-layer q/k parity arm (extra engine boot).
# export VLLM_HOOK_PARITY=1

export VLLM_HOOK_DEMO_MODEL="${VLLM_HOOK_DEMO_MODEL:-Qwen/Qwen2-1.5B-Instruct}"

# Make vLLM chatty about compilation/capture so the log shows the graphs build.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

echo "[run_qk_graph] model=$VLLM_HOOK_DEMO_MODEL  cudagraph_mode=$VLLM_HOOK_CUDAGRAPH_MODE  parity=${VLLM_HOOK_PARITY:-0}"
echo "[run_qk_graph] (TORCHDYNAMO_DISABLE is forced to 0 inside the python script)"

# -u = unbuffered stdout so bpeek / tail -f show progress without buffering lag.
python -u tests/cuda_graph/tests/qk_graph/test_qk_graph_capture.py
