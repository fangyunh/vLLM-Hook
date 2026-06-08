"""Global routing registry for CUDA-graph QK capture (file 2/5).

ROLE (§8.5 — device-tensor routing, never Python branches in the graph):
    The `HostRegistry` is the per-worker singleton that owns the GLOBAL routing
    slabs shared by every `QKHookHost`, and fans the per-step routing decision
    out to all layers with ONE device upload per feature — so the per-step cost
    is independent of the layer count.

    Owned slabs (all on the capture-rank device, sized once at construction):
      capture_index_all : (num_layers, cap) int64 — per-layer per-token dest row
      any_active        : (num_layers,)     int32 — per-layer "any token active"
    Each slab has a PINNED-CPU mirror; the execute_model wrapper (install.py,
    NEVER compiled) writes the next step's routing into the pinned mirror, then
    `upload()` issues a single `copy_(non_blocking=True)` per feature. The hosts
    hold ROW-VIEWS into the device slabs (assigned by `assign_views`), so the
    compiled graph reads fresh routing each replay without re-binding anything.

    Why per-layer rows and not one shared row: different requests may ask for
    different layer subsets via ``extra_args["output_qk"]``. Routing is therefore
    computed per (layer, token); a layer not requested for a given token simply
    gets index 0 (the discard sentinel), which the op's `active`-gate + sentinel
    row make an exact no-op (graph/ops.py caller contract).

CAPTURE-RANK ONLY (§9 / TP):
    Buffers, ops and slabs exist on the capture rank only (tp_size<=1 or
    rank % tp_size == 0). The registry carries a `should_capture` flag; on a
    non-capture rank install.py builds no registry at all, so the wrap's
    `do_capture=False` hosts short-circuit. The flag is kept here so egress and
    diagnostics can assert the invariant.

EXTENSION POINT (HS / Steer, later — QK only is wired now):
    The slab/view/upload machinery is feature-agnostic. A future hidden-states
    or steering feature would add its own slab(s) + host type alongside the QK
    ones; the `register_host` / `assign_views` / `upload` shape is intentionally
    generic so it can be reused. Only QK is populated today (SCOPE BOUNDARY).
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch

from vllm_hook_plugins.graph.hosts import QKHookHost


class HostRegistry:
    """Per-worker owner of the global QK routing slabs and host directory.

    Parameters
    ----------
    num_layers:
        Number of attention layers (= number of hosts that will register).
        Slab row count. Hosts index in by ``layer_num``.
    cap:
        Token capacity = ``max_num_batched_tokens``. Column count of
        ``capture_index_all`` and the per-host ``capture_index`` view length.
    device:
        Capture-rank device the slabs live on.
    should_capture:
        Whether this rank captures (tp_size<=1 or rank % tp_size == 0). The
        registry is only constructed on capture ranks; the flag is recorded so
        egress can assert it.

    Slabs (device, owned)
    ---------------------
    capture_index_all : (num_layers, cap) int64
    any_active        : (num_layers,)     int32

    Pinned mirrors (host, owned) — written by the execute_model wrapper, then
    uploaded with one non-blocking copy each:
    capture_index_pinned : (num_layers, cap) int64
    any_active_pinned    : (num_layers,)     int32
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

        # Global device slabs. int64 destination rows (index_copy_ needs Long);
        # any_active is int32 — an explicit per-layer "any token captured this
        # step" marker (the op no longer multiplies it into q/k; sentinel-row
        # routing supersedes the gate). Both start zeroed -> all tokens routed to
        # the discard sentinel (row 0) and inactive, i.e. a safe no-op until the
        # first upload.
        self.capture_index_all = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, device=self.device
        )
        self.any_active = torch.zeros(
            self.num_layers, dtype=torch.int32, device=self.device
        )

        # Pinned-CPU mirrors for non-blocking H2D uploads. pin_memory only makes
        # sense (and only works) with CUDA; fall back to a plain CPU tensor
        # otherwise so the registry is still constructible off-GPU for tests.
        pin = self.device.type == "cuda"
        self.capture_index_pinned = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, pin_memory=pin
        )
        self.any_active_pinned = torch.zeros(
            self.num_layers, dtype=torch.int32, pin_memory=pin
        )

        # CUDA event recorded AFTER each non-blocking upload, waited on before the
        # NEXT step zeros the pinned mirror — otherwise reset_pinned() could
        # overwrite the host buffer while the prior step's async H2D copy is still
        # in flight (the device slab would then receive zeros -> silent capture
        # loss). Only meaningful on CUDA; None elsewhere (CPU copies are sync).
        self._upload_event = (
            torch.cuda.Event() if self.device.type == "cuda" else None
        )

        # layer_num -> QKHookHost
        self.hosts: Dict[int, QKHookHost] = {}

        # Forward-context stash (set DURING the forward by the attention wrap,
        # where the context is live; read POST-forward by egress for prefix-K).
        # The eager path proves attn_metadata / no_compile_layers are only valid
        # inside the forward; reading them AFTER execute_model returns (when
        # set_forward_context has exited) yields None. So the attention wrap
        # snapshots the live context here on its FIRST fire of each step, and
        # egress reads `fwd_ctx` instead of calling get_forward_context() itself.
        # See graph/install.py _wrap_attn_class / _egress.
        self.fwd_ctx = None          # the live ForwardContext object
        self.fwd_attn_metadata = None  # ctx.attn_metadata snapshot
        self._ctx_step_token = 0     # bumped each step so the wrap re-snapshots

    # ------------------------------------------------------------------
    # Host registration + view assignment
    # ------------------------------------------------------------------

    def register_host(self, host: QKHookHost) -> None:
        """Record a host under its ``layer_num``. Idempotent per layer.

        The host's ``layer_num`` must be in ``[0, num_layers)`` (it indexes the
        slab row). Re-registering the same layer overwrites the entry — harmless
        under a re-entrant ``load_model``.
        """
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
        """Bind each registered host to its row-view into the global slabs.

        ``capture_index_all[layer]`` is a ``(cap,)`` view; ``any_active[layer]``
        is a 0-d view. Both alias the slab storage, so a single ``upload()``
        refresh is visible to every host's ``capture()`` on the next replay.
        Call once after all hosts are registered (the load_model patch does).
        """
        for layer_num, host in self.hosts.items():
            host.bind_views(
                self.capture_index_all[layer_num],   # (cap,) int64 view
                self.any_active[layer_num],           # 0-d int32 view
            )

    # ------------------------------------------------------------------
    # Per-step upload (fan-out by view) — called from the execute_model wrapper.
    # ------------------------------------------------------------------

    def upload(self) -> None:
        """Push the pinned mirrors to the device slabs with one copy each.

        The execute_model wrapper writes the next step's routing into
        ``capture_index_pinned`` / ``any_active_pinned`` (host side), then calls
        this. Exactly two non-blocking H2D copies per step regardless of layer
        count; the hosts' views see the new values on the next graph replay.
        """
        self.capture_index_all.copy_(self.capture_index_pinned, non_blocking=True)
        self.any_active.copy_(self.any_active_pinned, non_blocking=True)
        # Record the upload so the NEXT reset_pinned() can wait for it to finish
        # before reusing (zeroing) the pinned host buffer (see __init__).
        if self._upload_event is not None:
            self._upload_event.record()

    def reset_pinned(self) -> None:
        """Zero the pinned mirrors (all tokens -> sentinel, all layers inactive).

        The execute_model wrapper calls this at the start of each step before
        writing the current batch's routing, so any layer/token not explicitly
        requested defaults to the exact no-op (index 0 + inactive). This keeps
        the op's no-op precise (registry caller contract in graph/ops.py).

        Waits on the prior step's upload event first: the pinned mirror is the
        SAME host buffer the previous step uploaded with a non-blocking copy, so
        zeroing it before that copy completes would race the H2D transfer and the
        device slab could latch zeros (silent capture loss). The event wait makes
        the reuse safe without a full-stream synchronize.
        """
        if self._upload_event is not None:
            self._upload_event.synchronize()
        self.capture_index_pinned.zero_()
        self.any_active_pinned.zero_()

    # ------------------------------------------------------------------
    # Forward-context stash (set by the attention wrap, read by egress).
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Mark the start of a step so the attention wrap re-snapshots context.

        Called by the execute_model wrapper PRE-forward. Clears the stale stash
        and bumps a token; the wrap compares the token to know it is the first
        fire of this step and (re)captures the live forward context exactly once.
        """
        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token += 1

    def stash_forward_context(self, ctx, attn_metadata) -> None:
        """Snapshot the live forward context for POST-forward egress reads.

        Called from inside the attention-class wrap (context is live there). Only
        the first call per step takes effect (the wrap guards on the step token),
        so this stores the metadata from the first attention layer's invocation —
        which carries the batch-wide query_start_loc / block_table egress needs.
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
