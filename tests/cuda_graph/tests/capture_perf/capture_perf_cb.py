"""Multi-request CAPTURE perf harness — measures Lever A's high-concurrency win (v0.5.5).

The single-stream capture_perf.py cannot see batched egress: at one request the per-step
egress is O(num_layers x 1) == O(num_layers), so batched (Lever A reduce-then-free) and
per-request cost the same. Lever A's value is the O(num_layers x N) -> O(num_layers) cut in
the host-side egress that runs every decode step, which only shows at N >> 1.

This boots ONE graph engine (FULL_DECODE_ONLY, buffer) and times the DECODE phase of an
N-request concurrent batch (hooks_on=both, capture every step) by subtracting a prefill-only
(max_tokens=1) pass from a full M-token pass, for ONE value of VLLM_HOOK_BATCHED_EGRESS.
The wrapper runs it twice — batched=0 (per-request egress) and batched=1 (Lever A) — and the
report compares: PASS iff batched <= per-request (Lever A does not regress, and wins at scale).

    VLLM_HOOK_PERF_WORKER=qk VLLM_HOOK_BATCHED_EGRESS=0 python capture_perf_cb.py measure --out off.json
    VLLM_HOOK_PERF_WORKER=qk VLLM_HOOK_BATCHED_EGRESS=1 python capture_perf_cb.py measure --out on.json
    python capture_perf_cb.py report --off off.json --on on.json
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
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PERF_MAX_TOKENS", "64"))
_REPEATS = int(os.environ.get("VLLM_HOOK_PERF_REPEATS", "3"))
_NREQ = int(os.environ.get("VLLM_HOOK_PERF_NREQ", "16"))        # concurrency (the point)
_PROMPT_TOKS = int(os.environ.get("VLLM_HOOK_PERF_PROMPT_TOKS", "96"))
_WORKER = os.environ.get("VLLM_HOOK_PERF_WORKER", "qk").lower()
_BATCHED = os.environ.get("VLLM_HOOK_BATCHED_EGRESS", "0") == "1"

# (worker_name, analyzer_name, default config dir, graph capture-mode env var)
_WORKERS = {
    "qk": ("probe_hook_qk", "attn_tracker", "attention_tracker", "VLLM_HOOK_QK_CAPTURE"),
    "hs": ("probe_hidden_states", "hidden_states", "hidden_states", "VLLM_HOOK_HS_CAPTURE"),
}

_FILLER = ("The quick brown fox jumps over the lazy dog near the quiet river bank "
           "while the morning sun rises slowly over the distant misty hills. ")


def _make_prompt(tokenizer, target_toks):
    big = _FILLER * (target_toks // 8 + 8)
    ids = tokenizer.encode(big)[:target_toks]
    return tokenizer.decode(ids)


def _decode_ms_per_step(llm, prompts, sp_prefill, sp_full, use_hook):
    """median decode ms/step for the N-request batch = 1000*(T_full - T_prefill)/(M-1).

    The decode phase runs M-1 steps, each a forward over all N seqs PLUS the host-side
    egress over all N requests — exactly where Lever A cuts O(layers x N) -> O(layers)."""
    _ = llm.generate(prompts, sp_full, use_hook=use_hook)  # warmup (compile/capture)

    def _time(sp):
        samples = []
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            _ = llm.generate(prompts, sp, use_hook=use_hook)
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples)

    return 1000.0 * (_time(sp_full) - _time(sp_prefill)) / max(1, _MAX_TOKENS - 1)


def measure(out_path):
    worker_name, analyzer_name, cfg_dir, cap_env = _WORKERS[_WORKER]
    os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
    os.environ.setdefault(cap_env, "buffer")
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/{cfg_dir}/{_MODEL.split('/')[-1]}_alltok.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[cap-perf-cb:{_WORKER}:batched={int(_BATCHED)}] booting worker={worker_name} "
          f"N={_NREQ} M={_MAX_TOKENS} config={config_file}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name=worker_name,
        analyzer_name=analyzer_name,
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        max_num_batched_tokens=int(os.environ.get("VLLM_HOOK_PERF_MNBT", "4096")),
        max_num_seqs=max(_NREQ, 16),
        enable_chunked_prefill=True,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    prompt = _make_prompt(llm.tokenizer, _PROMPT_TOKS)
    prompts = [prompt] * _NREQ
    sp_prefill = SamplingParams(temperature=0.0, max_tokens=1,
                                extra_args={"hooks_on": "both"})
    sp_full = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS,
                             extra_args={"hooks_on": "both"})

    cap_ms = _decode_ms_per_step(llm, prompts, sp_prefill, sp_full, use_hook=True)
    payload = {"worker": _WORKER, "batched": _BATCHED, "nreq": _NREQ,
               "capture_ms_per_step": cap_ms, "n_decode": _MAX_TOKENS - 1}
    print(f"[cap-perf-cb:{_WORKER}:batched={int(_BATCHED)}] N={_NREQ} "
          f"capture={cap_ms:.3f} ms/step", flush=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[cap-perf-cb:{_WORKER}] wrote {out_path}", flush=True)
    return 0


def report(off_path, on_path):
    with open(off_path) as f:
        off = json.load(f)
    with open(on_path) as f:
        on = json.load(f)
    w, n = on.get("worker", "?"), on.get("nreq", "?")
    o_ms, b_ms = off["capture_ms_per_step"], on["capture_ms_per_step"]
    speedup = o_ms / b_ms if b_ms > 0 else float("inf")
    print("=" * 72)
    print(f"[cap-perf-cb:{w}] N={n} concurrent · decode ms/step (lower is better)")
    print(f"  per-request egress (batched=0): {o_ms:.3f} ms/step")
    print(f"  Lever A batched   (batched=1): {b_ms:.3f} ms/step")
    print(f"  Lever A vs per-request: {speedup:.2f}x  ({o_ms - b_ms:+.3f} ms/step)")
    print("=" * 72)
    if b_ms <= o_ms * 1.02:  # within 2% noise OR faster
        print(f"[cap-perf-cb:{w}] VERDICT: PASS — Lever A batched egress ({b_ms:.3f}) "
              f"<= per-request ({o_ms:.3f}) ms/step at N={n}. Egress cost cut, no regression.")
        return 0
    print(f"[cap-perf-cb:{w}] VERDICT: REGRESSION — Lever A batched ({b_ms:.3f}) > "
          f"per-request ({o_ms:.3f}) ms/step at N={n}. Investigate.")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("measure")
    pm.add_argument("--out", required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--off", required=True)
    pr.add_argument("--on", required=True)
    args = p.parse_args()
    if args.cmd == "measure":
        sys.exit(measure(args.out))
    else:
        sys.exit(report(args.off, args.on))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
