"""Memory-budget-driven adaptive GPU->pinned drains for CUDA-graph capture (v0.5.0 W4/W6c).

Buffer-mode capture egress clones the scattered q/k/hidden rows GPU->GPU each step and
DEFERS the .cpu() to request retrieval/flush, so per-request captured intermediates
accumulate on the GPU for the request's lifetime (worst case: QK ``all_tokens`` decode
re-clones the full ``k_all`` history every step -> O(steps x seq_len) per layer per
request). Under many concurrent heavy-capture requests that is an OOM risk.

``CaptureDrainManager`` bounds it BY CONSTRUCTION (W4): it tracks the bytes of
GPU-resident capture clones and, when they cross a configured budget, drains the
accumulated clones GPU->PINNED-host on a DEDICATED copy stream (W6c) so the transfer
overlaps the next forward instead of blocking it. The drained tensors live in pinned
host memory; the worker's existing ``[t.cpu() for t in ...]`` flush is then a no-op on
them (a CPU tensor's ``.cpu()`` returns itself), so the artifact is byte-identical —
draining only moves WHERE the tensor lives, never its values.

Correctness across streams:
- The clones are produced on the compute stream; the drain stream ``wait_stream``s it
  so it never copies an unmaterialized clone.
- ``record_stream`` marks each drained GPU tensor as in-use on the drain stream, so the
  caching allocator cannot recycle its memory until the async D2H completes (the classic
  multi-stream use-after-free guard).
- The flush waits on the drain event (``synchronize``) before reading a pinned tensor.

v0.5.x adds an OPT-IN streaming mode (VLLM_HOOK_CAPTURE_STREAMING=1, default OFF) that
(a) drains GPU-resident capture continuously into pinned host on a dedicated copy stream so
it stays bounded (not held to flush) under high-concurrency continuous batching, and (b)
enforces a resident-byte CEILING that throttles admission of NEW requests when total
resident capture crosses a hard limit — requests already capturing run to completion (never
half-captured), every throttle is logged + counted (no silent truncation). DEFAULT OFF ->
the manager is inert (no drain, no throttle, defer-to-flush) so the eager/op path and the
existing parity oracles are byte-for-byte unchanged. Setting an explicit budget/ceiling env
enables the respective half on its own (back-compat with the drain parity sweep). The
ceiling is a SOFT gate on NEW requests: in-flight admitted requests still grow with decode
length, so peak ~= ceiling + in-flight overshoot (bounded by concurrency, not the ceiling).
Stage 2 (incremental sink + free, deferred) collapses the pinned host retention window from
request-lifetime to N steps.

Env:
  VLLM_HOOK_CAPTURE_STREAMING   master opt-in (default 0 = OFF). When 1, enables BOTH the
                                auto drain budget AND the resident-ceiling throttle.
  VLLM_HOOK_CAPTURE_GPU_BUDGET  fraction of FREE GPU memory (at install) capture clones may
                                occupy before draining. Auto 0.05 under STREAMING=1, or set
                                explicitly. Explicit 0 = drain DISABLED (escape hatch).
  VLLM_HOOK_CAPTURE_DRAIN_BYTES absolute drain budget; overrides the fraction and enables the
                                drain on its own (used by the parity sweep to FORCE drains).
  VLLM_HOOK_CAPTURE_DRAIN_RING  depth of the drain event ring (default 3) so continuous
                                draining waits a K-steps-stale (already-complete) event,
                                never the in-flight D2H copy.
  VLLM_HOOK_CAPTURE_RESIDENT_CEILING       absolute byte ceiling on total resident capture;
                                           over it, NEW requests are not admitted. Setting it
                                           enables the throttle on its own. <=0 = no ceiling.
  VLLM_HOOK_CAPTURE_RESIDENT_CEILING_FRAC  fraction of total host RAM for the ceiling when no
                                           absolute value is set (default 0.5, generous).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch

# Print the first few drains so a parity job can PROVE the cross-stream path was
# exercised (a PASS with zero drains proves nothing). Default on; cheap (bounded).
_DRAIN_DEBUG = os.environ.get("VLLM_HOOK_DRAIN_DEBUG", "1") == "1"


def _free_gpu_bytes(device: torch.device) -> int:
    """Free bytes on ``device`` right now (one cheap mem_get_info; no kernel sync)."""
    try:
        if device.type == "cuda":
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
    except Exception:  # noqa: BLE001
        pass
    return 0


# Large absolute fallback when host RAM cannot be read (never trips at any sane scale).
_CEILING_FALLBACK = 1 << 40  # 1 TiB


def _total_host_ram_bytes() -> int:
    """Total host RAM in bytes (psutil if present, else sysconf). 0 if unreadable."""
    try:
        import psutil  # optional dependency
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001
        return 0


def _ceiling_from_env() -> int:
    """Resident-byte ceiling for the admission throttle (generous default).

    A non-positive absolute value or fraction means "no ceiling" -> the large fallback,
    symmetric with VLLM_HOOK_CAPTURE_GPU_BUDGET=0 disabling the drain.
    """
    abs_c = os.environ.get("VLLM_HOOK_CAPTURE_RESIDENT_CEILING")
    if abs_c:
        v = int(abs_c)
        return v if v > 0 else _CEILING_FALLBACK
    frac = float(os.environ.get("VLLM_HOOK_CAPTURE_RESIDENT_CEILING_FRAC", "0.5") or "0.5")
    if frac <= 0:
        return _CEILING_FALLBACK
    total = _total_host_ram_bytes()
    return int(frac * total) if total > 0 else _CEILING_FALLBACK


def bucket_bytes(layer_dict: dict) -> int:
    """Sum the bytes of all list-valued tensor entries in a popped request bucket.

    Mirrors the ``_drain`` filter (list-valued keys, ``torch.is_tensor``), so it counts
    exactly the q/k_all/hidden_states tensors and skips scalars (layer_num, hookq_mode)
    and the int list ``k_prefix_ends``. ``.numel()`` on a view returns its reduced
    logical size, so this matches the per-request bytes accrued into ``resident_bytes``
    at egress regardless of batched mode or whether the tensor was drained to pinned.
    """
    total = 0
    for entry in layer_dict.values():
        if not isinstance(entry, dict):
            continue
        for val in entry.values():
            if not isinstance(val, list):
                continue
            for t in val:
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    return total


class CaptureDrainManager:
    """Per-worker budget tracker + dedicated-stream GPU->pinned drainer.

    ``note(nbytes)`` accrues GPU-resident capture bytes; ``maybe_drain(worker)`` (called
    once per step after egress) drains all GPU-resident clones in the worker's capture
    buckets to pinned host when the accrued bytes cross the budget. ``synchronize()``
    (called by the worker flush) waits for the last drain's async copies to land.
    """

    # Capture-bucket attribute names on the worker; each maps req_id -> {module -> {...}}
    # where some values are LISTS of captured tensors (q / k_all / hidden_states).
    _BUCKET_ATTRS = ("_captured_states", "_disk_states")

    def __init__(self, device, budget_bytes: int, throttle_enabled: bool = False) -> None:
        self.device = torch.device(device)
        self.budget_bytes = int(budget_bytes)
        self.enabled = self.budget_bytes > 0 and self.device.type == "cuda"
        self.gpu_bytes = 0
        self._drains = 0
        # SINGLE dedicated copy stream (device clones stay single); only the host-side
        # completion EVENTS ring so continuous draining never stalls the producer on the
        # previous drain's in-flight D2H (mirrors PinnedMirrorRing in registry.py).
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(self.device) if self.enabled else None)
        self._ring_depth = max(1, int(os.environ.get("VLLM_HOOK_CAPTURE_DRAIN_RING", "3")))
        self._ring_idx = 0
        self._events: List[Optional[torch.cuda.Event]] = (
            [torch.cuda.Event() for _ in range(self._ring_depth)] if self.enabled else [])

        # Stage 3: resident-byte ceiling admission throttle. ``resident_bytes`` tracks
        # total held-to-flush capture bytes (accrued at egress, decremented on bucket
        # pop); over ``ceiling_bytes`` we stop admitting NEW requests. Works even when the
        # drain is disabled (budget 0) — the ceiling is a hard OOM guard, not a drain knob.
        self.resident_bytes = 0
        self.throttle_enabled = bool(throttle_enabled) and self.device.type == "cuda"
        self.ceiling_bytes = _ceiling_from_env()
        self._throttled: set = set()  # req_ids refused admission (skip every later step)

    @classmethod
    def from_env(cls, device) -> "CaptureDrainManager":
        """Build the drain/throttle manager. OPT-IN: default OFF.

        ``VLLM_HOOK_CAPTURE_STREAMING=1`` enables BOTH halves with auto defaults (drain
        budget 0.05 * free GPU, ceiling 0.5 * host RAM). An explicit
        ``VLLM_HOOK_CAPTURE_DRAIN_BYTES`` / ``_GPU_BUDGET`` (>0) enables the drain on its
        own; an explicit ``VLLM_HOOK_CAPTURE_RESIDENT_CEILING(_FRAC)`` enables the throttle
        on its own. With nothing set the manager is inert (defer-to-flush, no throttle) ->
        zero behavior change for the eager/op path and the existing parity oracles.
        """
        dev = torch.device(device)
        streaming = os.environ.get("VLLM_HOOK_CAPTURE_STREAMING", "0") == "1"
        abs_bytes = os.environ.get("VLLM_HOOK_CAPTURE_DRAIN_BYTES")
        gpu_budget = os.environ.get("VLLM_HOOK_CAPTURE_GPU_BUDGET")
        if abs_bytes:
            budget = int(abs_bytes)
        elif gpu_budget is not None and gpu_budget != "":
            frac = float(gpu_budget or "0")
            budget = int(frac * _free_gpu_bytes(dev)) if frac > 0 else 0
        elif streaming:
            budget = int(0.05 * _free_gpu_bytes(dev))
        else:
            budget = 0
        # Throttle enables with the master switch OR an explicit ceiling env.
        ceiling_set = bool(os.environ.get("VLLM_HOOK_CAPTURE_RESIDENT_CEILING")
                           or os.environ.get("VLLM_HOOK_CAPTURE_RESIDENT_CEILING_FRAC"))
        return cls(dev, budget, throttle_enabled=streaming or ceiling_set)

    # ------------------------------------------------------------------
    # Accounting (hot path — cheap when disabled).
    # ------------------------------------------------------------------

    def note(self, nbytes: int) -> None:
        """Accrue ``nbytes`` of freshly-cloned GPU capture data this step.

        ``gpu_bytes`` is a soft accrue-and-reset heuristic: it is NOT decremented when a
        request flushes (pops its bucket), so after a request completes the next drain
        may fire slightly early. This affects drain FREQUENCY (perf) only, never captured
        values — a follow-up could decrement on pop or recompute from the live buckets.
        """
        if self.enabled:
            self.gpu_bytes += int(nbytes)

    # ------------------------------------------------------------------
    # Stage 3: resident-ceiling admission throttle (distinct from gpu_bytes).
    # ------------------------------------------------------------------

    def note_resident(self, nbytes: int) -> None:
        """Accrue per-request REDUCED capture bytes for the resident-ceiling throttle.

        Distinct from ``note``/``gpu_bytes`` (the drain budget): this counts the SAVED
        bytes held to flush (batching-invariant — the reduced view, matching ``bucket_bytes``
        at pop). No-op when the throttle is disabled -> zero overhead on the default path.
        """
        if self.throttle_enabled:
            self.resident_bytes += int(nbytes)

    def admit(self, req_id, already_capturing: bool) -> bool:
        """Whether egress may START capturing this request this step.

        Refuses ONLY when total resident capture bytes exceed the ceiling AND the
        request is not already being captured — an admitted request always runs to
        completion (never throttled mid-stream → no half-captured artifact). A refused
        req_id is remembered so every later step/layer skips it uniformly.
        """
        if not self.throttle_enabled or already_capturing:
            return True
        if self.resident_bytes > self.ceiling_bytes:
            self._throttled.add(req_id)
            return False
        return True

    def on_pop(self, req_id, nbytes: int) -> None:
        """Release a request's resident bytes when its bucket is popped at flush/abort.

        Decrement-on-pop closes the soft-accrue gap in ``note`` for the resident
        counter (clamped at 0), so the ceiling reflects only in-flight capture.
        """
        if not self.throttle_enabled:
            return
        self.resident_bytes -= int(nbytes)
        if self.resident_bytes < 0:
            self.resident_bytes = 0
        self._throttled.discard(req_id)

    def forget(self, external_req_id) -> None:
        """Drop any throttled req_ids for a terminated external request.

        A throttled request never created a bucket, so ``on_pop`` is never called for it at
        flush; without this, ``_throttled`` would grow one entry per throttled request under
        sustained throttling. Called from the worker retrieval/cleanup paths. Matches the
        same exact-or-``{id}-``prefix rule as ``iter_matching_req_ids``.
        """
        if not self.throttle_enabled or not self._throttled:
            return
        prefix = f"{external_req_id}-"
        self._throttled = {
            r for r in self._throttled
            if not (r == external_req_id or (isinstance(r, str) and r.startswith(prefix)))
        }

    def maybe_drain(self, worker) -> None:
        """Drain to pinned host if the accrued GPU capture bytes exceed the budget.

        Called once per step after egress. A no-op when disabled or under budget.
        """
        if not self.enabled or self.gpu_bytes <= self.budget_bytes:
            return
        self._drain(worker)

    # ------------------------------------------------------------------
    # The drain (dedicated stream; async D2H into pinned host).
    # ------------------------------------------------------------------

    def _drain(self, worker) -> None:
        compute = torch.cuda.current_stream(self.device)
        # Ring the EVENTS: wait this slot's K-steps-stale event before reusing it. On a
        # warm ring it is already complete (no producer stall); the unconditional wait is
        # the correctness floor (an under-deep ring only stalls, never corrupts) and it
        # bounds in-flight D2H drains to ring_depth. Pinned dst tensors are freshly
        # allocated per drain (the bucket owns each until flush), so no slab aliasing.
        slot = self._ring_idx
        ev = self._events[slot] if slot < len(self._events) else None
        if ev is not None:
            ev.synchronize()
        # The drain stream must not copy a clone before the compute stream materialized it.
        self._stream.wait_stream(compute)
        moved = 0
        with torch.cuda.stream(self._stream):
            for attr in self._BUCKET_ATTRS:
                bucket = getattr(worker, attr, None)
                if not bucket:
                    continue
                for layer_states in bucket.values():
                    for entry in layer_states.values():
                        for key, val in entry.items():
                            if not isinstance(val, list):
                                continue  # skip scalars (layer_num, hookq_mode, ...)
                            for i, t in enumerate(val):
                                if torch.is_tensor(t) and t.is_cuda:
                                    dst = torch.empty(
                                        t.shape, dtype=t.dtype, pin_memory=True)
                                    dst.copy_(t, non_blocking=True)
                                    t.record_stream(self._stream)  # allocator guard
                                    val[i] = dst
                                    moved += 1
        # Guard: flush waits on this slot's event before reading any drained tensor.
        if ev is not None:
            self._stream.record_event(ev)
        self._ring_idx = (slot + 1) % self._ring_depth
        self.gpu_bytes = 0
        self._drains += 1
        if _DRAIN_DEBUG and self._drains <= 12:
            print(f"[graph/drain] drained {moved} tensor(s) "
                  f"(drain #{self._drains}, budget={self.budget_bytes}B)", flush=True)

    def synchronize(self) -> None:
        """Block until ALL pending drains' async copies have landed in pinned host.

        Called by the worker flush/retrieval before it reads the (now-CPU) tensors.
        With continuous draining several drains may be in flight across ring slots, so
        wait every slot's event (a never-recorded event reports complete → cheap).
        """
        for ev in self._events:
            if ev is not None:
                ev.synchronize()

    @property
    def num_drains(self) -> int:
        return self._drains


def drain_barrier(worker) -> None:
    """Worker-flush helper: wait for any pending capture drain before reading buckets."""
    dm = getattr(worker, "_capture_drain", None)
    if dm is not None:
        dm.synchronize()


__all__ = ["CaptureDrainManager", "drain_barrier", "bucket_bytes"]
