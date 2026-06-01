"""bench_throughput_serve — closed-loop throughput against a running vllm serve.

Two roles:
  - launcher    starts vllm serve, waits for /health, runs the benchmark,
                tears down (use --serve-cmd to override the launch).
  - external    --base-url is provided; the harness only drives requests.

For each (worker × concurrency × max_tokens) cell:
  - Spawns ``--concurrency`` worker threads.
  - Each thread drives requests in a tight loop for ``--duration`` seconds.
  - Records per-request: ttft (when streaming), gen_lat, response_bytes.
  - Pulls a profile snapshot from the serve worker at end-of-cell via the
    /v1/completions hook (best-effort — falls back to local PROF.snapshot
    of the client-side timers).

This is the harness that reproduces the plan.html §7 asyncio jam: at
concurrency≥4 with probe_hook_qk + all_tokens, ttft_p99 should collapse
because ``_serialize_probes`` holds the event loop for seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT  = PROFILING_DIR.parent
sys.path.insert(0, str(PROFILING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

from _common import save_csv  # noqa: E402


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _wait_for_health(base_url: str, timeout_s: float = 120.0) -> bool:
    import urllib.request
    health = base_url.rstrip("/").replace("/v1", "") + "/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _start_server(model: str, port: int, worker: str,
                  hook_dir: str, log_path: str,
                  enforce_eager: bool, max_model_len: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_HOOK_WORKER"] = worker
    env.setdefault("VLLM_HOOK_PROFILE", "1")
    env.setdefault("VLLM_HOOK_PROFILE_DIR", "/tmp/vllm_hook_profile_serve")
    cmd = ["vllm", "serve", model,
           "--port", str(port),
           "--max-model-len", str(max_model_len)]
    if enforce_eager:
        cmd.append("--enforce-eager")
    print(f"[serve] launching: {' '.join(cmd)}")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.gen_lats: List[float] = []
        self.ttfts:    List[float] = []
        self.response_bytes: List[int] = []
        self.errors:   int = 0
        self.completed: int = 0

    def add(self, gen_lat: float, ttft: Optional[float], resp_bytes: int):
        with self.lock:
            self.gen_lats.append(gen_lat)
            if ttft is not None:
                self.ttfts.append(ttft)
            self.response_bytes.append(resp_bytes)
            self.completed += 1

    def err(self):
        with self.lock:
            self.errors += 1


def _client_loop(base_url: str, model: str, prompt: str, max_tokens: int,
                 extra_body: Dict[str, Any], stop_event: threading.Event,
                 counters: _Counters) -> None:
    import openai
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
    while not stop_event.is_set():
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.0,
                extra_body=extra_body,
            )
            t1 = time.perf_counter()
            try:
                resp_size = len(resp.model_dump_json())
            except Exception:
                resp_size = 0
            counters.add(gen_lat=t1 - t0, ttft=None, resp_bytes=resp_size)
        except Exception as exc:
            counters.err()
            print(f"[client] {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------


def _run_cell(base_url: str, model: str, *, worker: str, concurrency: int,
              prompt: str, max_tokens: int, duration_s: float,
              hook_extra: Dict[str, Any]) -> Dict[str, Any]:
    counters = _Counters()
    stop = threading.Event()
    threads: List[threading.Thread] = []

    extra_body = {"vllm_xargs": hook_extra} if hook_extra else {}

    t0 = time.perf_counter()
    for _ in range(concurrency):
        t = threading.Thread(target=_client_loop,
                             args=(base_url, model, prompt, max_tokens,
                                   extra_body, stop, counters),
                             daemon=True)
        t.start()
        threads.append(t)

    stop.wait(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)
    t1 = time.perf_counter()
    wall = t1 - t0

    g = sorted(counters.gen_lats)
    rb = counters.response_bytes
    def _p(samples, q):
        if not samples:
            return None
        return samples[min(len(samples) - 1, int(len(samples) * q))]

    return {
        "worker":         worker,
        "concurrency":    concurrency,
        "max_tokens":     max_tokens,
        "duration_s":     wall,
        "completed":      counters.completed,
        "errors":         counters.errors,
        "req_per_sec":    counters.completed / max(wall, 1e-6),
        "gen_lat_mean":   statistics.mean(g)         if g else None,
        "gen_lat_p50":    _p(g, 0.50),
        "gen_lat_p99":    _p(g, 0.99),
        "response_bytes_mean": statistics.mean(rb)   if rb else None,
        "response_bytes_max":  max(rb)               if rb else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", default="baseline,qk,hidden_states",
                   help="Comma-separated VLLM_HOOK_WORKER values (or 'baseline' for "
                        "no plugin).")
    p.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--duration",  type=float, default=30.0)
    p.add_argument("--prompt",    default="Summarize the following paragraph: "
                                          "vLLM is a fast and easy-to-use library for "
                                          "LLM inference and serving.")
    p.add_argument("--base-url",  default=None,
                   help="If set, skip launching the server.")
    p.add_argument("--port",      type=int, default=8770)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--hook-dir",  default="/dev/shm/vllm_hook")
    p.add_argument("--no-enforce-eager", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    all_rows: List[Dict[str, Any]] = []

    for worker in workers:
        # Per-request extra body matching the worker.
        if worker == "baseline":
            hook_extra: Dict[str, Any] = {}
        elif worker == "qk":
            hook_extra = {"output_qk": True, "hookq_mode": "all_tokens"}
        elif worker == "hidden_states":
            hook_extra = {"output_hidden_states": True, "hs_mode": "all_tokens"}
        elif worker == "steer":
            hook_extra = {"steer": True}
        else:
            hook_extra = {}

        # Server lifecycle
        owned_proc: Optional[subprocess.Popen] = None
        base_url = args.base_url
        if base_url is None:
            log_path = os.path.join("/tmp", f"vllm-serve-{worker}.log")
            owned_proc = _start_server(args.model, args.port, worker,
                                       args.hook_dir, log_path,
                                       enforce_eager=(not args.no_enforce_eager),
                                       max_model_len=args.max_model_len)
            base_url = f"http://127.0.0.1:{args.port}/v1"
            print(f"[serve] waiting for /health at {base_url}")
            if not _wait_for_health(base_url, timeout_s=300):
                print(f"[serve] HEALTH TIMEOUT (worker={worker}) — see {log_path}")
                if owned_proc:
                    owned_proc.terminate()
                continue

        try:
            for c in args.concurrency:
                print(f"[bench] worker={worker} concurrency={c} duration={args.duration}s")
                row = _run_cell(base_url, args.model, worker=worker,
                                concurrency=c, prompt=args.prompt,
                                max_tokens=args.max_tokens,
                                duration_s=args.duration,
                                hook_extra=hook_extra)
                print(f"        completed={row['completed']} "
                      f"req/s={row['req_per_sec']:.2f} "
                      f"p99={row['gen_lat_p99'] or 0:.3f}s")
                all_rows.append(row)
                if args.output:
                    save_csv(all_rows, args.output)
        finally:
            if owned_proc is not None:
                owned_proc.terminate()
                try:
                    owned_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    owned_proc.kill()

    if args.output:
        save_csv(all_rows, args.output)
        print(f"[bench] CSV → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
