"""Per-worker device routing slabs for CUDA-graph QK capture."""
from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch

from vllm_hook_plugins.graph.hosts import QKHookHost


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

        # Pinned-CPU mirrors for non-blocking uploads; plain CPU off-CUDA so the
        # registry stays constructible for tests.
        pin = self.device.type == "cuda"
        self.capture_index_pinned = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, pin_memory=pin
        )
        self.any_active_pinned = torch.zeros(
            self.num_layers, dtype=torch.int32, pin_memory=pin
        )

        # Recorded after each upload; reset_pinned() waits on it so it can't zero
        # the mirror mid-transfer (would latch zeros = silent capture loss). CUDA
        # only; None elsewhere, where copies are synchronous.
        self._upload_event = (
            torch.cuda.Event() if self.device.type == "cuda" else None
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

    def upload(self, width: Optional[int] = None) -> None:
        """Push the pinned mirrors to the device slabs: one copy each per step.

        ``width`` (when given) copies only the ``[:, :width]`` column prefix of the
        capture_index slab — the prefix the cudagraph-padded forward actually reads
        (a 2D strided async copy, much smaller than the full ``cap`` width on decode).
        Columns beyond ``width`` are never read by the op, so the stale device tail is
        harmless. ``any_active`` is tiny (num_layers,) so it always copies in full.
        """
        if width is None:
            self.capture_index_all.copy_(self.capture_index_pinned, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.capture_index_all[:, :w].copy_(
                self.capture_index_pinned[:, :w], non_blocking=True)
        self.any_active.copy_(self.any_active_pinned, non_blocking=True)
        # So the next reset_pinned() can wait before reusing the mirror.
        if self._upload_event is not None:
            self._upload_event.record()

    def reset_pinned(self, width: Optional[int] = None) -> None:
        """Zero the pinned mirrors to the sentinel no-op, before each step's writes.

        Defaults any unrequested (layer, token) to row 0 + inactive. ``width`` zeros
        only the ``[:, :width]`` prefix (the part this step writes + uploads); the op
        never reads beyond ``width``, so the tail need not be re-zeroed. Waits on the
        prior upload first: zeroing the still-in-flight mirror would race the copy and
        latch zeros (silent capture loss); the event wait avoids a full sync.
        """
        if self._upload_event is not None:
            self._upload_event.synchronize()
        if width is None:
            self.capture_index_pinned.zero_()
        else:
            w = max(1, min(int(width), self.cap))
            self.capture_index_pinned[:, :w].zero_()
        self.any_active_pinned.zero_()

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
