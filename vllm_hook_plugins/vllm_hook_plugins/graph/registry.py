"""Global routing registry for CUDA-graph QK capture.

The ``HostRegistry`` is the per-worker singleton that owns the global routing
slabs shared by every ``QKHookHost`` and fans the per-step routing decision out
to all layers. Routing lives in device tensors, never Python branches, so the
compiled graph can read it on replay.

- Owns two slabs on the capture-rank device, sized once at construction:
  ``capture_index_all`` (num_layers, cap) int64 — the per-layer per-token
  destination row — and ``any_active`` (num_layers,) int32 — a per-layer
  "any token active" marker.
- Each slab has a pinned-CPU mirror. The execute_model wrapper (in install.py,
  NEVER compiled) writes the next step's routing into the mirror, then
  ``upload()`` issues a single non-blocking copy per slab — so the per-step
  upload cost is independent of the layer count. The hosts hold row-views into
  the device slabs (assigned by ``assign_views``), so the compiled graph reads
  fresh routing each replay without re-binding anything.
- Routing is per-layer, not one shared row, because different requests may ask
  for different layer subsets via ``extra_args["output_qk"]``. It is computed
  per (layer, token); a layer not requested for a given token gets index 0 (the
  discard sentinel), which the op's active-gate plus sentinel row make an exact
  no-op (the caller contract in graph/ops.py).

Capture-rank only: buffers, ops and slabs exist on the capture rank only
(tp_size<=1 or rank % tp_size == 0). On a non-capture rank install.py builds no
registry at all, so the wrap's no-capture hosts short-circuit. The registry
carries a ``should_capture`` flag so egress and diagnostics can assert the
invariant.

The slab/view/upload machinery is feature-agnostic: a future hidden-states or
steering feature would add its own slab(s) plus host type alongside the QK ones,
reusing the ``register_host`` / ``assign_views`` / ``upload`` shape. Only QK is
populated today.
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch

from vllm_hook_plugins.graph.hosts import QKHookHost


class HostRegistry:
    """Per-worker owner of the global QK routing slabs and host directory.

    ``num_layers`` is the attention-layer count (= the number of hosts that will
    register) and the slab row count; hosts index in by ``layer_num``. ``cap`` is
    the token capacity (= ``max_num_batched_tokens``), the column count of
    ``capture_index_all`` and the length of each host's ``capture_index`` view.
    ``device`` is the capture-rank device the slabs live on. ``should_capture``
    records whether this rank captures (tp_size<=1 or rank % tp_size == 0); the
    registry is only constructed on capture ranks, but the flag is kept so egress
    can assert it.

    Owns two device slabs — ``capture_index_all`` (num_layers, cap) int64 and
    ``any_active`` (num_layers,) int32 — each with a pinned host mirror
    (``capture_index_pinned`` / ``any_active_pinned``) that the execute_model
    wrapper writes and ``upload()`` pushes to the device with one non-blocking
    copy.
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

        # Global device slabs. The index slab is int64 because index_copy_ needs
        # Long destination rows; any_active is int32, an explicit per-layer
        # marker for "any token captured this step" (the op no longer multiplies
        # it into q/k, since sentinel-row routing now supersedes the gate). Both
        # start zeroed, so until the first upload every token routes to the
        # discard sentinel (row 0) and every layer is inactive -- a safe no-op.
        self.capture_index_all = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, device=self.device
        )
        self.any_active = torch.zeros(
            self.num_layers, dtype=torch.int32, device=self.device
        )

        # Pinned-CPU mirrors for non-blocking host-to-device uploads. Pinned
        # memory only makes sense (and only works) with CUDA, so fall back to a
        # plain CPU tensor otherwise to keep the registry constructible off-GPU
        # for tests.
        pin = self.device.type == "cuda"
        self.capture_index_pinned = torch.zeros(
            self.num_layers, self.cap, dtype=torch.int64, pin_memory=pin
        )
        self.any_active_pinned = torch.zeros(
            self.num_layers, dtype=torch.int32, pin_memory=pin
        )

        # CUDA event recorded AFTER each non-blocking upload, waited on before the
        # next step zeros the pinned mirror. Without it reset_pinned() could
        # overwrite the host buffer while the prior step's async upload is still
        # in flight, and the device slab would then latch zeros -- a silent
        # capture loss. Only meaningful on CUDA; None elsewhere, where copies are
        # synchronous.
        self._upload_event = (
            torch.cuda.Event() if self.device.type == "cuda" else None
        )

        # layer_num -> QKHookHost
        self.hosts: Dict[int, QKHookHost] = {}

        # Forward-context stash, set DURING the forward by the attention wrap
        # (where the context is live) and read POST-forward by egress for
        # prefix-K. The eager path proves attn_metadata and no_compile_layers are
        # only valid inside the forward: reading them AFTER execute_model returns,
        # once set_forward_context has exited, yields None. So the attention wrap
        # snapshots the live context here on its FIRST fire of each step, and
        # egress reads the stash instead of calling get_forward_context itself.
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

        ``capture_index_all[layer]`` is a ``(cap,)`` view and ``any_active[layer]``
        is a 0-d view; both alias the slab storage, so a single ``upload()``
        refresh is visible to every host's ``capture()`` on the next replay. Call
        once after all hosts are registered (the load_model patch does).
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
        ``capture_index_pinned`` / ``any_active_pinned`` on the host side, then
        calls this. Exactly two non-blocking host-to-device copies per step
        regardless of layer count; the hosts' views see the new values on the
        next graph replay.
        """
        self.capture_index_all.copy_(self.capture_index_pinned, non_blocking=True)
        self.any_active.copy_(self.any_active_pinned, non_blocking=True)
        # Record the upload so the next reset_pinned() can wait for it to finish
        # before reusing (zeroing) the pinned host buffer (see __init__).
        if self._upload_event is not None:
            self._upload_event.record()

    def reset_pinned(self) -> None:
        """Zero the pinned mirrors: all tokens to the sentinel, all layers inactive.

        The execute_model wrapper calls this at the start of each step before
        writing the current batch's routing, so any layer or token not explicitly
        requested defaults to the exact no-op (index 0 plus inactive). This keeps
        the op's no-op precise (the caller contract in graph/ops.py).

        Waits on the prior step's upload event first. The pinned mirror is the
        SAME host buffer the previous step uploaded with a non-blocking copy, so
        zeroing it before that copy completes would race the transfer and the
        device slab could latch zeros -- a silent capture loss. The event wait
        makes the reuse safe without a full-stream synchronize.
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

        Called by the execute_model wrapper BEFORE the forward. Clears the stale
        stash and bumps a token; the wrap compares the token to know it is the
        first fire of this step and recaptures the live forward context exactly
        once.
        """
        self.fwd_ctx = None
        self.fwd_attn_metadata = None
        self._ctx_step_token += 1

    def stash_forward_context(self, ctx, attn_metadata) -> None:
        """Snapshot the live forward context for post-forward egress reads.

        Called from inside the attention-class wrap, where the context is live.
        Only the first call per step takes effect (the wrap guards on the step
        token), so this stores the metadata from the first attention layer's
        invocation -- which carries the batch-wide query_start_loc and
        block_table that egress needs.
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
