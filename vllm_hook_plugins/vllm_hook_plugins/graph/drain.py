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

Env:
  VLLM_HOOK_CAPTURE_GPU_BUDGET  fraction of FREE GPU memory (at install) to allow capture
                                clones to occupy before draining. Default 0.0 = DISABLED
                                (defer-to-flush, the pre-v0.5.0 behavior) -> zero overhead.
  VLLM_HOOK_CAPTURE_DRAIN_BYTES absolute byte budget; overrides the fraction when set
                                (used by the parity sweep to FORCE frequent drains).
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

    def __init__(self, device, budget_bytes: int) -> None:
        self.device = torch.device(device)
        self.budget_bytes = int(budget_bytes)
        self.enabled = self.budget_bytes > 0 and self.device.type == "cuda"
        self.gpu_bytes = 0
        self._drains = 0
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(self.device) if self.enabled else None)
        self._pending_event: Optional[torch.cuda.Event] = None

    @classmethod
    def from_env(cls, device) -> "CaptureDrainManager":
        """Build from VLLM_HOOK_CAPTURE_DRAIN_BYTES / _GPU_BUDGET (fraction of free)."""
        dev = torch.device(device)
        abs_bytes = os.environ.get("VLLM_HOOK_CAPTURE_DRAIN_BYTES")
        if abs_bytes:
            budget = int(abs_bytes)
        else:
            frac = float(os.environ.get("VLLM_HOOK_CAPTURE_GPU_BUDGET", "0") or "0")
            budget = int(frac * _free_gpu_bytes(dev)) if frac > 0 else 0
        return cls(dev, budget)

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
        # Guard: flush waits on this before reading any drained (pinned) tensor.
        self._pending_event = self._stream.record_event()
        self.gpu_bytes = 0
        self._drains += 1
        if _DRAIN_DEBUG and self._drains <= 12:
            print(f"[graph/drain] drained {moved} tensor(s) "
                  f"(drain #{self._drains}, budget={self.budget_bytes}B)", flush=True)

    def synchronize(self) -> None:
        """Block until the last drain's async copies have landed in pinned host.

        Called by the worker flush/retrieval before it reads the (now-CPU) tensors.
        Idempotent and cheap when no drain is pending.
        """
        ev = self._pending_event
        if ev is not None:
            ev.synchronize()
            self._pending_event = None

    @property
    def num_drains(self) -> int:
        return self._drains


def drain_barrier(worker) -> None:
    """Worker-flush helper: wait for any pending capture drain before reading buckets."""
    dm = getattr(worker, "_capture_drain", None)
    if dm is not None:
        dm.synchronize()


__all__ = ["CaptureDrainManager", "drain_barrier"]
