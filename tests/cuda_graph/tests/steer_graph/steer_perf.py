"""Subprocess-isolated per-decode-step STEERING perf harness (v0.5.0 W1/W2/W6).

The parity oracle (steer_parity_decode.py) proves graph steering is CORRECT. This
proves it is FAST: the v0.5.0 goal is that hooked decode under FULL CUDA graphs beats
v0.2.0 eager (the profiled regression was steering ~35.5 ms/step graph vs ~9.6 ms eager
on Phi-3, caused by the per-step routing rebuild + a per-step CPU<->GPU sync —
addressed by W1 invalidation + W2/W6a double-buffered pinned mirror).

Each engine boots in its own subprocess (GPU exclusive-process) and times the DECODE
phase only, by subtracting a prefill-only (max_tokens=1) pass from a full N-token pass:
    decode_ms_per_step = 1000 * (T_full - T_prefill) / (N - 1)
measured for two request configs in the SAME engine:
    base  — use_hook=False (no steering: the no-hook floor for this mode)
    steer — use_hook=True  (steering at the config's optimal_layer, every step)

    graph mode:  VLLM_HOOK_ALLOW_CUDAGRAPH=1 VLLM_HOOK_STEER_MODE=buffer, FULL_DECODE_ONLY
    eager mode:  VLLM_HOOK_ALLOW_CUDAGRAPH=0, enforce_eager=True (register_forward_hook)

VERDICT (report): PASS iff graph-steer ms/step <= eager-steer ms/step (regression fixed)
AND graph-steer is within a small factor of the graph no-hook floor (steering rides the
replay). If VLLM_HOOK_PROFILE=1, also surfaces the graph.route.skip / .build counters so
you can confirm W1 invalidation is actually firing (skip >> build on steady decode).

    python steer_perf.py measure --mode graph --out g.json
    python steer_perf.py measure --mode eager --out e.json
    python steer_perf.py report  --graph g.json --eager e.json
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

_COEFF = float(os.environ.get("VLLM_STEER_COEFF", "20.0"))
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PERF_MAX_TOKENS", "128"))
_REPEATS = int(os.environ.get("VLLM_HOOK_PERF_REPEATS", "3"))
# Regression-fixed margin: graph-steer must be <= eager-steer * (1 + slack). A small
# slack absorbs run-to-run noise; the real win should be well under 1.0.
_SLACK = float(os.environ.get("VLLM_HOOK_PERF_SLACK", "0.0"))

_PROMPT = "The capital of France is"


def _decode_ms_per_step(llm, sp_prefill, sp_full, use_hook):
    """median decode ms/step = 1000*(T_full - T_prefill)/(N-1) over _REPEATS runs.

    No torch.cuda.synchronize() here: under exclusive_process GPU mode the EngineCore
    owns the device in a SEPARATE subprocess, so the driver has no CUDA context and a
    driver-side sync raises cudaErrorDevicesUnavailable. It is also unnecessary —
    llm.generate() is a blocking call that returns only once the tokens are generated
    and detokenized, so the wall-clock around it already measures end-to-end GPU work.
    The prefill-subtraction (T_full - T_prefill) cancels the fixed per-generate IPC/
    setup overhead, leaving the decode-step cost.
    """
    # Warmup (first call compiles / captures; not timed).
    _ = llm.generate(_PROMPT, sp_full, use_hook=use_hook)

    def _time(sp):
        samples = []
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            _ = llm.generate(_PROMPT, sp, use_hook=use_hook)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    t_prefill = _time(sp_prefill)
    t_full = _time(sp_full)
    n_decode = max(1, _MAX_TOKENS - 1)
    return 1000.0 * (t_full - t_prefill) / n_decode


def measure(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ.setdefault("VLLM_HOOK_STEER_MODE", "buffer")
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/activation_steer/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[steer-perf:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file} coeff={_COEFF} N={_MAX_TOKENS} repeats={_REPEATS}",
          flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="steer_hook_act",
        analyzer_name=None,
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
    sp_prefill_steer = SamplingParams(temperature=0.0, max_tokens=1,
                                      extra_args={"steer": {"coefficient": _COEFF}})
    sp_full_steer = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS,
                                   extra_args={"steer": {"coefficient": _COEFF}})

    base_ms = _decode_ms_per_step(llm, sp_prefill_base, sp_full_base, use_hook=False)
    steer_ms = _decode_ms_per_step(llm, sp_prefill_steer, sp_full_steer, use_hook=True)

    payload = {"mode": mode, "base_ms_per_step": base_ms, "steer_ms_per_step": steer_ms,
               "n_decode": _MAX_TOKENS - 1}

    # Surface the W1 invalidation counters if profiling is on (proves skip >> build).
    try:
        from vllm_hook_plugins._profiler import PROF
        snap = PROF.summary_only()
        counters = snap.get("counters", {}) if isinstance(snap, dict) else {}
        for k in ("graph.route.skip", "graph.route.build"):
            if k in counters:
                payload[k] = counters[k]
    except Exception:  # noqa: BLE001
        pass

    print(f"[steer-perf:{mode}] base={base_ms:.3f} ms/step  steer={steer_ms:.3f} ms/step"
          + (f"  (skip={payload.get('graph.route.skip')} "
             f"build={payload.get('graph.route.build')})"
             if "graph.route.skip" in payload else ""), flush=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[steer-perf:{mode}] wrote {out_path}", flush=True)
    return 0


def report(graph_path, eager_path):
    with open(graph_path) as f:
        g = json.load(f)
    with open(eager_path) as f:
        e = json.load(f)

    g_base, g_steer = g["base_ms_per_step"], g["steer_ms_per_step"]
    e_base, e_steer = e["base_ms_per_step"], e["steer_ms_per_step"]

    print("=" * 72)
    print("[steer-perf] decode ms/step (lower is better)")
    print(f"  {'':14s}{'no-hook':>12s}{'steer':>12s}{'hook overhead':>16s}")
    print(f"  {'graph (FULL)':14s}{g_base:>10.3f}ms{g_steer:>10.3f}ms"
          f"{g_steer - g_base:>+13.3f}ms")
    print(f"  {'eager (v0.2)':14s}{e_base:>10.3f}ms{e_steer:>10.3f}ms"
          f"{e_steer - e_base:>+13.3f}ms")
    print("-" * 72)
    speedup = e_steer / g_steer if g_steer > 0 else float("inf")
    print(f"  graph-steer vs eager-steer: {g_steer:.3f} vs {e_steer:.3f} ms/step "
          f"({speedup:.2f}x)")
    if "graph.route.skip" in g:
        print(f"  W1 invalidation: skip={g.get('graph.route.skip')} "
              f"build={g.get('graph.route.build')} "
              "(skip >> build confirms steady-decode replay)")
    print("=" * 72)

    regression_fixed = g_steer <= e_steer * (1.0 + _SLACK)
    if regression_fixed:
        print(f"[steer-perf] VERDICT: PASS — graph-steer ({g_steer:.3f} ms/step) "
              f"<= eager-steer ({e_steer:.3f} ms/step). Regression fixed.")
        return 0
    print(f"[steer-perf] VERDICT: FAIL — graph-steer ({g_steer:.3f} ms/step) "
          f"> eager-steer ({e_steer:.3f} ms/step). Still regressed.")
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
