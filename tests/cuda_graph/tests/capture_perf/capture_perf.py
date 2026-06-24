"""Subprocess-isolated per-decode-step CAPTURE perf harness (v0.5.0 W3/W6d).

The QK/HS parity oracles prove capture-on-decode is CORRECT under FULL graphs. This
measures whether it is FAST: the v0.5.0 goal is hooked decode beats v0.2.0 eager. After
Step 1 the capture routing already rides the sync-free pinned ring + prefix-width upload
(no per-step CPU<->GPU sync), and Step 2's drain bounds memory; this harness quantifies
where capture-on-decode stands vs eager BEFORE deciding what W3 still needs.

Worker selected by env VLLM_HOOK_PERF_WORKER = qk | hs. Each engine boots in its own
subprocess and times the DECODE phase by subtracting a prefill-only (max_tokens=1) pass
from a full N-token pass, for two configs in the SAME engine:
    base    — use_hook=False (no capture: the no-hook floor for this mode)
    capture — use_hook=True, hooks_on=both (capture every prefill + decode step)

    graph mode:  VLLM_HOOK_ALLOW_CUDAGRAPH=1 + buffer capture, FULL_DECODE_ONLY
    eager mode:  enforce_eager=True (register_forward_hook every step)

VERDICT (report): PASS iff graph-capture ms/step <= eager-capture ms/step (beats eager).

    VLLM_HOOK_PERF_WORKER=qk python capture_perf.py measure --mode graph --out g.json
    VLLM_HOOK_PERF_WORKER=qk python capture_perf.py measure --mode eager --out e.json
    python capture_perf.py report --graph g.json --eager e.json
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import getpass
import json
import statistics
import sys
import time

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PERF_MAX_TOKENS", "128"))
_REPEATS = int(os.environ.get("VLLM_HOOK_PERF_REPEATS", "3"))
_SLACK = float(os.environ.get("VLLM_HOOK_PERF_SLACK", "0.0"))
_WORKER = os.environ.get("VLLM_HOOK_PERF_WORKER", "qk").lower()
_PROMPT = "The capital of France is"

# (worker_name, analyzer_name, default config dir, graph capture-mode env var)
_WORKERS = {
    "qk": ("probe_hook_qk", "attn_tracker", "attention_tracker", "VLLM_HOOK_QK_CAPTURE"),
    "hs": ("probe_hidden_states", "hidden_states", "hidden_states", "VLLM_HOOK_HS_CAPTURE"),
}


def _decode_ms_per_step(llm, sp_prefill, sp_full, use_hook):
    """median decode ms/step = 1000*(T_full - T_prefill)/(N-1) over _REPEATS runs.

    No driver-side torch.cuda.synchronize() — under exclusive_process the EngineCore
    owns the GPU in a separate subprocess; llm.generate() is blocking so the wall-clock
    around it already measures end-to-end GPU work, and the prefill subtraction cancels
    fixed per-generate overhead.
    """
    _ = llm.generate(_PROMPT, sp_full, use_hook=use_hook)  # warmup (compile/capture)

    def _time(sp):
        samples = []
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            _ = llm.generate(_PROMPT, sp, use_hook=use_hook)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    return 1000.0 * (_time(sp_full) - _time(sp_prefill)) / max(1, _MAX_TOKENS - 1)


def measure(mode, out_path):
    worker_name, analyzer_name, cfg_dir, cap_env = _WORKERS[_WORKER]
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ.setdefault(cap_env, "buffer")
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/{cfg_dir}/{_MODEL.split('/')[-1]}_alltok.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[cap-perf:{_WORKER}:{mode}] booting worker={worker_name} "
          f"enforce_eager={enforce_eager} config={config_file} N={_MAX_TOKENS}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name=worker_name,
        analyzer_name=analyzer_name,
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    sp_prefill_base = SamplingParams(temperature=0.0, max_tokens=1)
    sp_full_base = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
    sp_prefill_cap = SamplingParams(temperature=0.0, max_tokens=1,
                                    extra_args={"hooks_on": "both"})
    sp_full_cap = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS,
                                 extra_args={"hooks_on": "both"})

    base_ms = _decode_ms_per_step(llm, sp_prefill_base, sp_full_base, use_hook=False)
    cap_ms = _decode_ms_per_step(llm, sp_prefill_cap, sp_full_cap, use_hook=True)

    payload = {"mode": mode, "worker": _WORKER, "base_ms_per_step": base_ms,
               "capture_ms_per_step": cap_ms, "n_decode": _MAX_TOKENS - 1}
    print(f"[cap-perf:{_WORKER}:{mode}] base={base_ms:.3f} ms/step  "
          f"capture={cap_ms:.3f} ms/step", flush=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[cap-perf:{_WORKER}:{mode}] wrote {out_path}", flush=True)
    return 0


def report(graph_path, eager_path):
    with open(graph_path) as f:
        g = json.load(f)
    with open(eager_path) as f:
        e = json.load(f)
    w = g.get("worker", "?")
    g_base, g_cap = g["base_ms_per_step"], g["capture_ms_per_step"]
    e_base, e_cap = e["base_ms_per_step"], e["capture_ms_per_step"]

    print("=" * 72)
    print(f"[cap-perf:{w}] decode ms/step (lower is better)")
    print(f"  {'':14s}{'no-hook':>12s}{'capture':>12s}{'hook overhead':>16s}")
    print(f"  {'graph (FULL)':14s}{g_base:>10.3f}ms{g_cap:>10.3f}ms{g_cap - g_base:>+13.3f}ms")
    print(f"  {'eager (v0.2)':14s}{e_base:>10.3f}ms{e_cap:>10.3f}ms{e_cap - e_base:>+13.3f}ms")
    print("-" * 72)
    speedup = e_cap / g_cap if g_cap > 0 else float("inf")
    print(f"  graph-capture vs eager-capture: {g_cap:.3f} vs {e_cap:.3f} ms/step "
          f"({speedup:.2f}x)")
    print("=" * 72)

    if g_cap <= e_cap * (1.0 + _SLACK):
        print(f"[cap-perf:{w}] VERDICT: PASS — graph-capture ({g_cap:.3f} ms/step) "
              f"<= eager-capture ({e_cap:.3f} ms/step). Beats eager.")
        return 0
    print(f"[cap-perf:{w}] VERDICT: FAIL — graph-capture ({g_cap:.3f} ms/step) "
          f"> eager-capture ({e_cap:.3f} ms/step). W3 work needed.")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("measure")
    pm.add_argument("--mode", choices=["graph", "eager"], required=True)
    pm.add_argument("--out", required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--graph", required=True)
    pr.add_argument("--eager", required=True)
    args = p.parse_args()
    if args.cmd == "measure":
        sys.exit(measure(args.mode, args.out))
    else:
        sys.exit(report(args.graph, args.eager))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
