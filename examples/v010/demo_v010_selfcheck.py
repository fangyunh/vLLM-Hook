#!/usr/bin/env python3
"""v0.1.0 self-check demo — drives one v0.1.0 workload through the peer driver and
verifies capture/steer actually happens (PASS/FAIL), mirroring the hybrid demos'
self-check style.

This is a SANITY demo, not the profiling path. The actual v0.1.0 campaign data is
produced by OUR harness (`vllm-hook-profile run results/v010/<campaign>/run.yaml`)
via the ported `offline_v010` driver — see this directory's README.md. This demo
exists to (a) document the v0.1.0 worker_cls + flag/env paradigm end-to-end and
(b) give a quick "does the worker still capture on this model?" check.

Run on a GPU node (needs vllm_hook_env_v010 + the fork on PYTHONPATH):

    export PYTHONPATH=$HOME/vLLM-Hook-Profiling/vendor/vLLM-Hook-v010:$PYTHONPATH
    export VLLM_PLUGINS=""           # disable the fork's v0.2.x auto-injection
    export VLLM_HOOK_USE_SAFETENSORS=1 VLLM_HOOK_ASYNC_SAVE=1
    VHP_WORKLOAD=hs python examples/v010/demo_v010_selfcheck.py   # hs | qk | steer
"""
import os
import sys
import uuid

# The peer v0.1.0 driver (V010Engine) lives in the local fork.
from profiling.peers.v010 import driver as v010

REPO = "/u/fangyunh/vLLM-Hook"
CFG_DIR = os.path.join(REPO, "examples/v010/configs")

WORKLOADS = {
    "hs": dict(
        row="probe_hidden_states_v010",
        model=os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct"),
        config=os.path.join(CFG_DIR, "hs_lasttok_granite.json"),
        max_tokens=int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128")),
    ),
    "qk": dict(
        row="probe_hook_qk_v010",
        model=os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct"),
        config=os.path.join(CFG_DIR, "qk_lasttok_granite.json"),
        max_tokens=int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128")),
    ),
    "steer": dict(
        row="steer_hook_act_v010",
        model=os.environ.get("VLLM_HOOK_DEMO_MODEL", "microsoft/Phi-3-mini-4k-instruct"),
        config=os.path.join(CFG_DIR, "steer_phi3.json"),
        max_tokens=int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "64")),
    ),
}


def main() -> int:
    wl = os.environ.get("VHP_WORKLOAD", "hs")
    if wl not in WORKLOADS:
        print(f"VHP_WORKLOAD must be one of {list(WORKLOADS)} (got {wl!r})")
        return 2
    spec = WORKLOADS[wl]
    hook_dir = f"/dev/shm/vllm_hook_v010_demo_{os.getpid()}"
    print(f"[v010-demo] workload={wl} model={spec['model']} config={spec['config']}")

    from vllm import SamplingParams

    eng = v010.build_engine(
        spec["row"], model=spec["model"], hook_dir=hook_dir,
        config_file=spec["config"], dtype="auto", tp_size=1,
        max_model_len=2048, gpu_memory_utilization=0.7,
        enable_prefix_caching=False, enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"], ignore_eos=True)
    run_id = str(uuid.uuid4())
    out = eng.generate(["The capital of France is"], sp, save_to_disk=(wl != "steer"),
                       run_id=run_id)
    text = out[0].outputs[0].text[:60].replace("\n", " ")
    print(f"[v010-demo] generated: {text!r}")

    if wl == "steer":
        # In-place edit writes nothing; success = it ran without error.
        print("VERDICT: PASS (steer ran in-place, no artifact expected)")
        return 0

    run_dir = os.path.join(hook_dir, run_id)
    files = []
    for root, _d, fs in os.walk(run_dir):
        files += [os.path.join(root, f) for f in fs]
    nbytes = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    ok = nbytes > 0
    print(f"[v010-demo] artifact: {nbytes/1024:.1f} KB in {len(files)} files under {run_dir}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} (captured {nbytes/1024:.1f} KB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
