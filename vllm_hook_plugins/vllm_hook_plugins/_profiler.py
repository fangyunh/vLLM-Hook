"""Process-local profiler for vLLM-Hook.

Activated by ``VLLM_HOOK_PROFILE=1``. When disabled, every entry point is a
no-op so the production hot path is unchanged.

Env vars
--------
VLLM_HOOK_PROFILE         "1" to enable. Default off.
VLLM_HOOK_PROFILE_FINE    "1" to also record tier-2 timers (more events,
                          more overhead — adds CUDA event syncs).
VLLM_HOOK_PROFILE_CUDA    "1" to record timed_cuda(...) GPU events.
                          Always implied when PROFILE_FINE=1.
VLLM_HOOK_PROFILE_DIR     Where dump() writes JSON. Default
                          /tmp/vllm_hook_profile.
VLLM_HOOK_PROFILE_MEM     "1" to start the background NVML+RSS sampler.

Usage
-----
    from vllm_hook_plugins._profiler import PROF

    with PROF.timed("hookllm.generate"):
        outputs = llm.generate(...)

    PROF.incr("hook.fire")
    PROF.gauge("rpc.payload_bytes", len(payload))

    PROF.dump()   # writes <PROFILE_DIR>/profile-<pid>-<seq>.json
"""
from __future__ import annotations

import atexit
import json
import os
import statistics
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


_ENABLED  = _env_bool("VLLM_HOOK_PROFILE")
_FINE     = _env_bool("VLLM_HOOK_PROFILE_FINE")
_CUDA_EVT = _env_bool("VLLM_HOOK_PROFILE_CUDA") or _FINE
_MEM_SAMP = _env_bool("VLLM_HOOK_PROFILE_MEM")
_DUMP_DIR = os.environ.get("VLLM_HOOK_PROFILE_DIR", "/tmp/vllm_hook_profile")


def is_enabled() -> bool:
    return _ENABLED


class Profiler:
    """Thread-safe, process-local profiler.

    Three collection primitives:
    - ``timed(name)`` — CPU wall-clock context manager, records ms.
    - ``timed_cuda(name)`` — GPU time via CUDA events, records ms.
                              No-op unless VLLM_HOOK_PROFILE_CUDA=1.
    - ``incr(name, n)`` — monotonic counter.
    - ``gauge(name, value)`` — append a numeric sample (bytes, queue depth).

    ``snapshot()`` returns summary stats. ``dump(path)`` writes JSON.
    """

    def __init__(self) -> None:
        self.timers:   Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int]         = defaultdict(int)
        self.gauges:   Dict[str, List[float]] = defaultdict(list)
        self.events:   List[Dict[str, Any]]   = []   # bounded; used for traces
        self._lock = threading.Lock()
        self._dump_seq = 0
        self._start_wall = time.time()

    # ------------------------------------------------------------------
    # Timer primitives
    # ------------------------------------------------------------------

    @contextmanager
    def timed(self, name: str, *, tier: int = 1) -> Iterator[None]:
        if not _ENABLED or (tier == 2 and not _FINE):
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.timers[name].append(dt_ms)

    @contextmanager
    def timed_cuda(self, name: str, *, tier: int = 1) -> Iterator[None]:
        # Tier-2 by default — CUDA events require a sync that distorts timing.
        if not _ENABLED or not _CUDA_EVT or (tier == 2 and not _FINE):
            yield
            return
        try:
            import torch
        except ImportError:
            yield
            return
        if not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            with self._lock:
                self.timers[name].append(ms)

    # ------------------------------------------------------------------
    # Counter / gauge primitives
    # ------------------------------------------------------------------

    def incr(self, name: str, n: int = 1) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self.counters[name] += n

    def gauge(self, name: str, value: float) -> None:
        if not _ENABLED:
            return
        with self._lock:
            self.gauges[name].append(float(value))

    def event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Record a one-shot tagged event with a timestamp.

        Bounded to 10k events to avoid runaway memory in long-running
        servers; older events are dropped.
        """
        if not _ENABLED:
            return
        rec = {"t": time.time() - self._start_wall, "name": name}
        if payload:
            rec["payload"] = payload
        with self._lock:
            self.events.append(rec)
            if len(self.events) > 10_000:
                del self.events[: len(self.events) - 10_000]

    # ------------------------------------------------------------------
    # Reset / snapshot / dump
    # ------------------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self.timers.clear()
            self.counters.clear()
            self.gauges.clear()
            self.events.clear()
            self._start_wall = time.time()

    def _summarize(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"count": 0}
        n = len(samples)
        out = {
            "count": n,
            "sum":   sum(samples),
            "mean":  sum(samples) / n,
            "min":   min(samples),
            "max":   max(samples),
        }
        if n > 1:
            out["std"] = statistics.stdev(samples)
            sorted_s = sorted(samples)
            out["p50"] = sorted_s[n // 2]
            out["p90"] = sorted_s[min(n - 1, int(n * 0.90))]
            out["p99"] = sorted_s[min(n - 1, int(n * 0.99))]
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Return summary statistics for every recorded metric.

        Timer units are ms. Gauge units depend on the call site.
        Includes the raw samples too — JSON is small enough and lets
        callers compute new percentiles later.
        """
        with self._lock:
            return {
                "enabled":  _ENABLED,
                "tier":     2 if _FINE else 1,
                "cuda":     _CUDA_EVT,
                "pid":      os.getpid(),
                "wall_s":   time.time() - self._start_wall,
                "timers":   {k: {**self._summarize(v), "samples_ms": list(v)}
                             for k, v in self.timers.items()},
                "counters": dict(self.counters),
                "gauges":   {k: {**self._summarize(v), "samples": list(v)}
                             for k, v in self.gauges.items()},
                "events":   list(self.events),
            }

    def summary_only(self) -> Dict[str, Any]:
        """Snapshot without per-sample arrays — for sidecar embedding."""
        with self._lock:
            return {
                "enabled":  _ENABLED,
                "tier":     2 if _FINE else 1,
                "pid":      os.getpid(),
                "wall_s":   time.time() - self._start_wall,
                "timers":   {k: self._summarize(v) for k, v in self.timers.items()},
                "counters": dict(self.counters),
                "gauges":   {k: self._summarize(v) for k, v in self.gauges.items()},
            }

    def dump(self, path: Optional[str] = None, *, role: str = "proc") -> Optional[str]:
        """Write the full snapshot as JSON. Returns the path written, or None
        if profiling is disabled.

        ``role`` is a short tag baked into the filename (``driver`` / ``worker``
        / ``proc``) so multi-process runs produce identifiable dumps.
        """
        if not _ENABLED:
            return None
        if path is None:
            os.makedirs(_DUMP_DIR, exist_ok=True)
            self._dump_seq += 1
            path = os.path.join(
                _DUMP_DIR,
                f"profile-{role}-{os.getpid()}-{self._dump_seq}.json",
            )
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2, default=str)
        return path


# Singleton — every wrap call sites the same instance.
PROF = Profiler()


# ---------------------------------------------------------------------------
# atexit dump — runs while the import system is still alive, unlike __del__
# ---------------------------------------------------------------------------

# Best-effort identification of "is this the driver or a worker subprocess?"
# Workers spawn with VLLM_DP_RANK / RANK / LOCAL_RANK set; the driver doesn't.
def _detect_role() -> str:
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK", "PMI_RANK"):
        v = os.environ.get(key)
        if v is not None and v != "":
            return f"worker-r{v}"
    # vLLM v1 sometimes uses VLLM_WORKER_MULTIPROC_METHOD without populating
    # a rank var; fall back to checking the main module name.
    main = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    if "vllm" in main.lower() and "engine" in main.lower():
        return "worker"
    return "driver"


_ROLE = _detect_role()


def _atexit_dump() -> None:
    """Called once per process during normal interpreter exit.

    Safe to invoke even when profiling is disabled — dump() short-circuits.
    Writes a sentinel line to stderr on failure so debugging doesn't depend
    on a silent JSON-missing symptom.
    """
    try:
        path = PROF.dump(role=_ROLE)
        if path is not None:
            print(f"[vllm-hook profiler] wrote {path}", file=sys.stderr, flush=True)
    except Exception:
        try:
            print("[vllm-hook profiler] atexit dump FAILED:", file=sys.stderr)
            traceback.print_exc()
        except Exception:
            pass


if _ENABLED:
    atexit.register(_atexit_dump)


# ---------------------------------------------------------------------------
# Background memory sampler (NVML GPU + driver-process RSS)
# ---------------------------------------------------------------------------


class MemorySampler:
    """Background daemon thread that samples NVML GPU memory + process RSS.

    Activated by VLLM_HOOK_PROFILE_MEM=1 (and PROFILE=1). Updates two
    gauges on PROF: ``mem.gpu_mb`` and ``mem.host_rss_mb``.
    """

    def __init__(self, interval_s: float = 0.05, gpu_index: int = 0) -> None:
        self.interval = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._psutil_proc = None

    def _try_init(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._nvml_handle = None
        try:
            import psutil
            self._psutil_proc = psutil.Process(os.getpid())
        except Exception:
            self._psutil_proc = None
        return self._nvml_handle is not None or self._psutil_proc is not None

    def _loop(self) -> None:
        import pynvml  # may be imported lazily
        while not self._stop.is_set():
            if self._nvml_handle is not None:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    PROF.gauge("mem.gpu_mb", info.used / 1024 ** 2)
                except Exception:
                    pass
            if self._psutil_proc is not None:
                try:
                    rss = self._psutil_proc.memory_info().rss
                    PROF.gauge("mem.host_rss_mb", rss / 1024 ** 2)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not (_ENABLED and _MEM_SAMP):
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if not self._try_init():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="vllm-hook-mem-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


MEM_SAMPLER = MemorySampler()


# Auto-start the memory sampler when the module is imported with PROFILE_MEM=1.
# Driver-side import will catch the driver process; the worker process imports
# happen at install_hooks time, which also covers worker-side sampling.
if _ENABLED and _MEM_SAMP:
    MEM_SAMPLER.start()
