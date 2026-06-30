"""Profiled churn load to localize the CB HS residual plugin tax (v0.5.7).

Boots ONE FULL_DECODE_ONLY buffer-HS engine and fires a large concurrent batch with
STAGGERED prompt lengths + max_tokens so the scheduler runs a churning batch (prefills
mixed into decodes + condensation as requests finish) — the open-loop CB condition that
defeats A's per-column routing skip. VLLM_HOOK_PROFILE=1 makes the worker dump
graph.route / graph.egress timers + graph.route.upload/noupload/skip counters at exit;
the report tells us whether A skips uploads under churn or degrades to full-rebuild, and
how much per-step wall time the routing/egress wrappers actually cost.

    VLLM_HOOK_PROFILE=1 VLLM_HOOK_PROFILE_DIR=<dir> python profile_churn.py
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"          # before plugin import (graph mode)
os.environ.setdefault("VLLM_HOOK_ALLOW_CUDAGRAPH", "1")
os.environ.setdefault("VLLM_HOOK_HS_CAPTURE", "buffer")

import getpass
import random
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16
_NREQ = int(os.environ.get("CHURN_NREQ", "600"))          # concurrency -> batch width
_CGMODE = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
_WORKER = os.environ.get("CHURN_WORKER", "hs").lower()    # hs | qk

_WORKERS = {
    "hs": ("probe_hidden_states", "hidden_states",
           "model_configs/hidden_states/Qwen2-1.5B-Instruct.json", "VLLM_HOOK_HS_CAPTURE"),
    "qk": ("probe_hook_qk", "attn_tracker",
           "model_configs/attention_tracker/Qwen2-1.5B-Instruct.json", "VLLM_HOOK_QK_CAPTURE"),
}

_FILLER = ("The quick brown fox jumps over the lazy dog near the quiet river bank while "
           "the morning sun rises slowly over the distant misty hills and valleys below. ")


def main():
    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM
    from vllm_hook_plugins._profiler import PROF, is_enabled

    wname, aname, cfg, capenv = _WORKERS[_WORKER]
    os.environ[capenv] = "buffer"
    user = os.environ.get("USER") or getpass.getuser()
    extra = {"compilation_config": {"cudagraph_mode": _CGMODE}} if _CGMODE.upper() != "NONE" else {}

    llm = HookLLM(
        model=_MODEL, worker_name=wname, analyzer_name=aname, config_file=cfg,
        download_dir="./cache/", hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{user}"),
        gpu_memory_utilization=float(os.environ.get("CHURN_GPU_UTIL", "0.85")),
        max_model_len=1024, trust_remote_code=True, dtype=_DTYPE,
        enforce_eager=False, enable_prefix_caching=False, enable_hook=True,
        tensor_parallel_size=1, **extra,
    )

    rng = random.Random(1234)
    prompts, sps = [], []
    for i in range(_NREQ):
        ptoks = rng.choice([16, 32, 64, 96, 160, 256])      # staggered prefill lengths
        mtoks = rng.choice([8, 16, 32, 48, 64, 96])         # staggered finishes -> churn+condense
        prompts.append((_FILLER * (ptoks // 12 + 4))[:ptoks * 4])
        sps.append(SamplingParams(temperature=0.0, max_tokens=mtoks,
                                  extra_args={"hooks_on": "prefill", "output_hidden_states": True}
                                  if _WORKER == "hs" else
                                  {"hooks_on": "prefill", "output_qk": True}))

    print(f"[churn] {_WORKER} N={_NREQ} model={_MODEL} cgmode={_CGMODE} "
          f"profile={is_enabled()}", flush=True)
    outs = llm.generate(prompts, sps, use_hook=True)
    print(f"[churn] generated {len(outs)} outputs", flush=True)

    # Pull the WORKER's PROF snapshot (the graph.route/egress timers live there) via the
    # string-named dump_profiler RPC — reliable, no code serialization, no atexit dependency.
    if is_enabled():
        try:
            paths = llm.llm.collective_rpc("dump_profiler")
            print("[churn] worker PROF dumped via RPC:", paths, flush=True)
        except Exception as e:  # noqa: BLE001
            print("[churn] RPC PROF dump FAILED:", repr(e), flush=True)
    print("[churn] DONE — read worker PROF dump from VLLM_HOOK_PROFILE_DIR.", flush=True)
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    sys.exit(main())
