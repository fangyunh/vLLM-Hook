"""profiling/smoke_test.py — self-contained profiler sanity check.

Builds a HookLLM, runs one generate + analyze, lets the atexit dump fire.
Used by ``profiling/runners/run_smoke.sh``.

Why this exists separately from ``examples/demo_hiddenstate.py``:
  - The demo hardcodes ``hook_dir="/dev/shm/vllm_hook"``. On shared LSF
    nodes that path can survive between jobs (tmpfs is node-local but
    reboot-persistent) and may already exist with permissions that block
    other users from creating subdirs.
  - The smoke job needs to be hermetic: a fresh unique hook_dir per run,
    plus an explicit run_id, plus profile-on. None of those belong in a
    user-facing demo.

Reads ``HOOK_DIR`` from env if set, otherwise picks a unique path under
``/dev/shm`` keyed by USER + PID so concurrent or back-to-back jobs never
collide.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

mp.set_start_method("spawn", force=True)
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

import torch
from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


def _pick_hook_dir() -> str:
    if "HOOK_DIR" in os.environ:
        return os.environ["HOOK_DIR"]
    user = (os.environ.get("USER") or os.environ.get("LOGNAME") or "noname")
    jobid = os.environ.get("LSB_JOBID") or str(os.getpid())
    return f"/dev/shm/vllm_hook_smoke_{user}_{jobid}"


def main() -> int:
    hook_dir = _pick_hook_dir()
    print(f"[smoke_test] hook_dir = {hook_dir}", flush=True)
    os.makedirs(hook_dir, exist_ok=True)

    # Defensive ownership check — fail loudly with the actual cause rather
    # than a deep stack trace from os.makedirs() inside the worker.
    try:
        probe = os.path.join(hook_dir, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        st = os.stat(hook_dir)
        print(f"[smoke_test] FAIL — cannot write into hook_dir.\n"
              f"  path : {hook_dir}\n"
              f"  uid  : owner={st.st_uid} group={st.st_gid} mode={oct(st.st_mode)}\n"
              f"  me   : uid={os.getuid()} gid={os.getgid()}\n"
              f"  err  : {exc}\n"
              f"Pick a different HOOK_DIR (e.g. /dev/shm/vllm_hook_$USER_$$) and retry.",
              file=sys.stderr, flush=True)
        return 2

    llm = HookLLM(
        model="Qwen/Qwen2.5-3B-Instruct",
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file="model_configs/hidden_states/Qwen2.5-3B-Instruct.json",
        hook_dir=hook_dir,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    output = llm.generate(
        "The capital of France is",
        SamplingParams(temperature=0.0, max_tokens=10),
        save_to_disk=True,
    )
    stats = llm.analyze(analyzer_spec={"reduce": "none"})

    print(f"[smoke_test] generated: {output[0].outputs[0].text.strip()!r}",
          flush=True)
    for i, (layer_name, tensors) in enumerate(sorted(stats["hidden_states"].items())):
        if i >= 4:
            print("  ...")
            break
        t = tensors[0]
        print(f"  {layer_name}: shape={tuple(t.shape)}  "
              f"norm={float(torch.norm(t.float())):.2f}")

    print("[smoke_test] done — atexit dump will fire now.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
