"""Per-attn-layer capture host for CUDA-graph QK capture.

A ``QKHookHost`` owns the static q/k buffers for ONE attention module
(e.g. ``model.layers.0.self_attn.attn``). It is a tiny ``nn.Module`` whose
only job is to:

- hold the static ``q_buf``/``k_buf`` via ``register_buffer``, so their
  ``data_ptr()`` stays fixed across graph replays — the buffers must be
  registered before vLLM allocates its cudagraph pool, or the recorded graph
  would write to the wrong address;
- expose ``capture(q, k)``, which calls the opaque ``vllm_hook::capture_qk``
  custom op against those buffers plus the routing tensors (``capture_index``,
  ``any_active``). The op is the thing Dynamo records as a graph node and
  replays every step (see graph/ops.py for why a custom op, not a hook).

The routing tensors are views, not owned by the host. ``capture_index`` and
``any_active`` are 1-D views into the registry's global slabs
(graph/registry.py), assigned via ``bind_views()`` after construction. The
registry updates the slabs each step with one pinned-to-device copy (fan-out
by view) and the compiled graph only reads them. Keeping them as views is what
makes the per-step upload cost independent of the layer count.

``capture()`` is graph-legal: it does no Python branching on per-request data
and allocates nothing of its own — it is a single op call on fixed-shape
tensors. The op itself slices ``index[:q.shape[0]]`` and may do a defensive
dtype cast; both are graph-constant operations on the captured token width
(see graph/ops.py). The ``do_capture`` flag is an install-time constant, set
once when the host is built on the capture rank; ``install.py`` reads it to
decide whether to call ``capture()`` from the class-level wrap at all, so it
never introduces data-dependent control flow inside the traced region.

``capture_index`` is the full-length ``(cap,)`` view; the op aligns it to the
live q/k row count by slicing its leading entries. The routing layer fills
exactly those leading entries (column = flat token position), so the alignment
holds.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class QKHookHost(nn.Module):
    """Static-buffer host for a single attention module's QK capture.

    Parameters
    ----------
    module_name:
        Fully-qualified attn module name, e.g. ``"model.layers.0.self_attn.attn"``.
        Used as the egress bucket key, matching the eager ``qkv_hook`` exactly so
        analyzers and disk save read the same buckets unchanged (the reuse
        contract).
    layer_num:
        Integer layer index (``match_attn(module_name)``); also the row index
        of this host's view into the registry's global slabs.
    cap:
        Token capacity = ``max_num_batched_tokens``. Buffers are sized
        ``cap + 1`` so row 0 can serve as the discard sentinel.
    q_dim:
        ``num_q_heads * head_dim`` — width of a post-RoPE query row.
    k_dim:
        ``num_kv_heads * head_dim`` — width of a post-RoPE key row.
    dtype:
        Buffer dtype (the model compute dtype).
    device:
        Capture-rank device the buffers live on.
    do_capture:
        Install-time constant, ``True`` only on the capture rank. ``install.py``
        reads it to gate whether the class-level wrap calls ``capture()`` at all;
        it is never used for branching inside the traced op.

    Buffers (registered, fixed data_ptr across replays)
    ---------------------------------------------------
    q_buf : (cap + 1, q_dim)  — row 0 = discard sentinel
    k_buf : (cap + 1, k_dim)  — row 0 = discard sentinel

    Views (assigned by the registry via ``bind_views``; not owned here)
    -------------------------------------------------------------------
    capture_index : (cap,) int64 — per-token destination row for this layer
    any_active    : ()   int/bool — gate: any capture-requested token this step
    """

    def __init__(
        self,
        module_name: str,
        layer_num: int,
        cap: int,
        q_dim: int,
        k_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        do_capture: bool = True,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.layer_num = int(layer_num)
        self.cap = int(cap)
        self.q_dim = int(q_dim)
        self.k_dim = int(k_dim)
        self.do_capture = bool(do_capture)

        # Row 0 is the discard sentinel and is never read by egress. Registering
        # these as buffers (persistent=False keeps them out of the state_dict and
        # checkpoints) pins their data_ptr across graph replays, but only if the
        # host is built BEFORE vLLM's cudagraph pool is allocated — the load_model
        # patch is what guarantees that timing.
        self.register_buffer(
            "q_buf",
            torch.zeros(self.cap + 1, self.q_dim, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "k_buf",
            torch.zeros(self.cap + 1, self.k_dim, dtype=dtype, device=device),
            persistent=False,
        )

        # Routing views are populated later by the registry's bind_views(). They
        # are plain attributes, not buffers, because the registry owns the backing
        # slabs; registering the view here would double-register the same storage.
        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None

        # Keep-alive sink for the impl-runs prefill op (qk_probe). The op mutates
        # this 1-element counter so it survives dead-code elimination; the real
        # capture is a side effect of the op impl. Each host gets its own sink so
        # the per-layer op nodes don't serialize on a shared mutated tensor.
        self.register_buffer(
            "sink", torch.zeros(1, dtype=torch.int64, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # View binding (called by the registry, once, after construction).
    # ------------------------------------------------------------------

    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs.

        ``capture_index`` is a ``(cap,)`` int64 view (one row of
        ``capture_index_all``); ``any_active`` is a 0-d view (one element of the
        per-layer ``any_active`` slab). Both are device tensors the registry
        refreshes each step via a single fan-out copy; ``capture()`` only reads
        them, so the bindings are stable for the host's lifetime.
        """
        self.capture_index = capture_index
        self.any_active = any_active

    # ------------------------------------------------------------------
    # Graph-recorded capture (the only thing the wrap calls).
    # ------------------------------------------------------------------

    def capture(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Scatter post-RoPE ``q``/``k`` rows into the static buffers.

        Single call to the opaque ``vllm_hook::capture_qk`` op — no Python
        branching on per-request data — so it is graph-legal and replays on every
        step. The op slices ``capture_index[:q.shape[0]]`` to align with the live
        token count, then writes ``q_buf[idx[t]] = q[t]`` (and the same for k) for
        every token ``t``; tokens routed to row 0 (the write-only sentinel) are
        discarded. ``any_active`` is passed through for op-schema stability but is
        no longer used as a multiplicative gate — the sentinel-row routing alone
        makes unrequested tokens an exact no-op (see graph/ops.py).
        """
        torch.ops.vllm_hook.capture_qk(
            q, k, self.q_buf, self.k_buf, self.capture_index, self.any_active
        )

    def probe(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Impl-runs-at-runtime capture (the prefill path; see graph/ops.py).

        Calls the opaque ``vllm_hook::qk_probe`` op, which keeps itself alive by
        mutating ``self.sink`` and — at runtime, mid-forward, where the live
        forward context and current input_batch are available — runs the eager
        per-request capture body (install._capture_body) into the worker buckets.
        No static buffers / routing needed: the capture happens in the op impl.
        """
        torch.ops.vllm_hook.qk_probe(q, k, self.sink, self.layer_num)
