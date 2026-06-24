"""Per-worker device routing slabs for CUDA-graph QK capture."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from vllm_hook_plugins.graph.hosts import QKHookHost

# Depth of the pinned-mirror ring (W2/W6a). >=2 lets reset_pinned() reuse a slot
# whose H2D upload was enqueued >=2 steps ago — its event is already complete, so
# the wait never stalls the decode hot path. 1 == the old single-mirror behaviour
# (the per-step CPU<->GPU serialization the v0.5.0 plan removes). vLLM's async
# scheduler runs the CPU at most ~1 step ahead of the GPU, so 2 is sufficient.
RING_DEPTH = max(1, int(os.environ.get("VLLM_HOOK_PINNED_RING", "2")))


class PinnedMirrorRing:
    """Depth-K ring of pinned-host mirror *sets* + per-slot CUDA upload events.

    The capture/steer routing is staged in pinned host memory, then copied H2D into
    the device slabs each step. With a SINGLE mirror, ``reset_pinned`` must
    ``event.synchronize()`` before zeroing it (zeroing a mirror whose H2D copy is
    still in flight would latch zeros = silent capture/steer loss) — a per-step
    CPU<->GPU serialization that breaks vLLM's async decode overlap. With K>=2 slots
    the slot reused this step was last uploaded K steps ago, so its event is already
    complete and the wait returns immediately (kept only as a correctness guard).

    Each slot holds one pinned tensor per ``(name, shape, dtype)`` spec. The owning
    registry exposes ``registry.<name>_pinned`` as the CURRENT slot's tensor (a
    property), so build code writes the active mirror unchanged; ``upload`` copies the
    current slot to the device slabs (registry-specific) and advances the cursor.
    """

    def __init__(self, specs, pin: bool, depth: int = RING_DEPTH) -> None:
        self.depth = max(1, int(depth))
        self.idx = 0
        self.slots: List[Dict[str, torch.Tensor]] = [
            {
                name: torch.zeros(shape, dtype=dtype, pin_memory=pin)
                for name, shape, dtype in specs
            }
            for _ in range(self.depth)
        ]
        # One event per slot; None off-CUDA (copies are synchronous there).
        self.events: List[Optional[torch.cuda.Event]] = [
            torch.cuda.Event() if pin else None for _ in range(self.depth)
        ]

    def cur(self, name: str) -> torch.Tensor:
        """The current slot's tensor for ``name`` (what build code writes/upload reads)."""
        return self.slots[self.idx][name]

    def wait_current(self) -> None:
        """Block until the current slot's last upload (K steps ago) is done.

        On a warm ring this event is already complete, so this does not stall. The
        wait is an UNCONDITIONAL synchronize, so an under-deep ring (or a CPU that
        runs further ahead than the ring covers) only degrades to a stall — it can
        never overwrite a mirror mid-H2D. VLLM_HOOK_PINNED_RING is thus a pure perf
        knob with no correctness floor; a never-recorded event reports complete, so
        the depth-2 warm-up and any never-reused slot are safe.
        """
        ev = self.events[self.idx]
        if ev is not None:
            ev.synchronize()

    def record_advance(self) -> None:
        """Record the current slot's upload event, then rotate to the next slot."""
        ev = self.events[self.idx]
        if ev is not None:
            ev.record()
        self.idx = (self.idx + 1) % self.depth


class HostRegistry:
    """Per-worker owner of the QK routing slabs and host directory.

    Rows = ``num_layers`` (hosts index by ``layer_num``); columns = ``cap`` token
    capacity (= ``max_num_batched_tokens``). Owns two device slabs, each with a
    pinned mirror: ``capture_index_all`` (num_layers, cap) int64 destination rows,
    and ``any_active`` (num_layers,) int32 per-layer active markers.
    """

    def __init__(
        self,
        num_layers: int,
        cap: int,
        device: torch.device | str,
        should_capture: bool = True,
    ) -> None:
        self.num_layers = int(num_layers)
        self.cap = int(cap)
        self.device = torch.device(device)
        self.should_capture = bool(should_capture)

        # int64 (index_copy_ needs Long rows); any_active int32. Zeroed, so
        # pre-first-upload every token routes to sentinel row 0 -- a safe no-op.
        self.capture_index_all = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, device=self.device
        )
        self.any_active = torch.zeros(
            self.num_layers, dtype=torch.int32, device=self.device
        )

        # Pinned-CPU mirrors for non-blocking uploads, ring-buffered (W2/W6a) so
        # reset_pinned never waits on the in-flight copy; plain CPU off-CUDA so the
        # registry stays constructible for tests. capture_index_pinned /
        # any_active_pinned are properties onto the CURRENT ring slot.
        pin = self.device.type == "cuda"
        self._ring = PinnedMirrorRing(
            [
                ("capture_index", (self.num_layers, self.cap), torch.int64),
                ("any_active", (self.num_layers,), torch.int32),
            ],
            pin=pin,
        )

        # layer_num -> QKHookHost (or HSHookHost — duck-typed: layer_num/cap/bind_views)
        self.hosts: Dict[int, QKHookHost] = {}

        # Plans built by the _prepare_inputs routing wrapper this step, consumed by
        # the execute_model egress wrapper post-forward (the two run in one step).
        self._pending_plans: list = []

        # Forward-context stash: the attention wrap snapshots the live context on
        # its first fire each step (it's only valid inside the forward), so egress
        # can read it post-forward for prefix-K. See graph/install.py.
        self.fwd_ctx = None          # the live ForwardContext object
        self.fwd_attn_metadata = None  # ctx.attn_metadata snapshot
        self._ctx_step_token = 0     # bumped each step so the wrap re-snapshots

    # ------------------------------------------------------------------
    # Host registration + view assignment
    # ------------------------------------------------------------------

    def register_host(self, host: QKHookHost) -> None:
        """Record a host under its ``layer_num`` (slab row); idempotent per layer."""
        if not (0 <= host.layer_num < self.num_layers):
            raise ValueError(
                f"host layer_num {host.layer_num} out of range "
                f"[0, {self.num_layers}) for module {host.module_name!r}"
            )
        if host.cap != self.cap:
            raise ValueError(
                f"host cap {host.cap} != registry cap {self.cap} "
                f"for module {host.module_name!r}"
            )
        self.hosts[host.layer_num] = host

    def assign_views(self) -> None:
        """Bind each host to its slab row-views, so one ``upload()`` reaches all.

        Call once after all hosts register (the load_model patch does). The views
        alias slab storage, so the refresh is visible to every ``capture()`` on
        the next replay.
        """
        for layer_num, host in self.hosts.items():
            host.bind_views(
                self.capture_index_all[layer_num],   # (cap,) int64 view
                self.any_active[layer_num],           # 0-d int32 view
            )

    # ------------------------------------------------------------------
    # Per-step upload (fan-out by view) — called from the execute_model wrapper.
    # ------------------------------------------------------------------

    @property
    def capture_index_pinned(self) -> torch.Tensor:
        """The current ring slot's (num_layers, cap) int64 routing mirror."""
        return self._ring.cur("capture_index")

    @property
    def any_active_pinned(self) -> torch.Tensor:
        """The current ring slot's (num_layers,) int32 active-marker mirror."""
        return self._ring.cur("any_active")

    def routing_key(self, model_runner, qsl_cpu) -> Optional[tuple]:
        """Invalidation key for the routing wrapper (W1). ``None`` = never skip.

        Capture (QK/HS) returns ``None`` so the routing is rebuilt every step — the
        conservative, latency-neutral choice. With FLAT per-step routing the index is
        actually bit-identical on a stable decode batch (each request's column is
        fixed), so a real signature like SteerRegistry's would let W1 collapse decode
        to a pure replay; that is an opt-in perf follow-on, deliberately left off here.
        """
        return None

    def upload(self, width: Optional[int] = None) -> None:
        """Push the current pinned mirror slot to the device slabs: one copy per step.

        ``width`` (when given) copies only the ``[:, :width]`` column prefix of the
        capture_index slab — the prefix the cudagraph-padded forward actually reads
        (a 2D strided async copy, much smaller than the full ``cap`` width on decode).
        Columns beyond ``width`` are never read by the op, so the stale device tail is
        harmless. ``any_active`` is tiny (num_layers,) so it always copies in full.
        Records the slot's upload event and rotates the ring (W2/W6a).
        """
        ci = self._ring.cur("capture_index")
        aa = self._ring.cur("any_active")
        if width is None:
            self.capture_index_all.copy_(ci, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.capture_index_all[:, :w].copy_(ci[:, :w], non_blocking=True)
        self.any_active.copy_(aa, non_blocking=True)
        self._ring.record_advance()

    def reset_pinned(self, width: Optional[int] = None) -> None:
        """Zero the current pinned mirror slot to the sentinel no-op, before writes.

        Defaults any unrequested (layer, token) to row 0 + inactive. ``width`` zeros
        only the ``[:, :width]`` prefix (the part this step writes + uploads); the op
        never reads beyond ``width``, so the tail need not be re-zeroed. Waits on this
        slot's own (K-steps-stale, already-complete) upload event first — a guard that
        no longer stalls the hot path now that the ring gives a fresh slot each step.
        """
        self._ring.wait_current()
        ci = self._ring.cur("capture_index")
        aa = self._ring.cur("any_active")
        if width is None:
            ci.zero_()
        else:
            w = max(1, min(int(width), self.cap))
            ci[:, :w].zero_()
        aa.zero_()

    # ------------------------------------------------------------------
    # Forward-context stash (set by the attention wrap, read by egress).
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Clear the stash and bump the step token so the wrap re-snapshots once.

        Called by the execute_model wrapper before the forward.
        """
        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token += 1

    def stash_forward_context(self, ctx, attn_metadata) -> None:
        """Snapshot the live forward context for post-forward egress reads.

        First call per step wins (wrap guards on the step token), capturing the
        first layer's batch-wide query_start_loc / block_table that egress needs.
        """
        self.fwd_ctx = ctx
        self.fwd_attn_metadata = attn_metadata

    # ------------------------------------------------------------------
    # Egress accessors (read static buffers after the forward returns).
    # ------------------------------------------------------------------

    def iter_hosts(self) -> Iterator[Tuple[int, QKHookHost]]:
        """Yield ``(layer_num, host)`` for every registered host, layer-ordered."""
        for layer_num in sorted(self.hosts):
            yield layer_num, self.hosts[layer_num]

    def get_host(self, layer_num: int) -> QKHookHost | None:
        """Return the host for ``layer_num``, or ``None`` if not registered."""
        return self.hosts.get(layer_num)

    @property
    def num_hosts(self) -> int:
        return len(self.hosts)
