"""Per-attention-layer static-buffer host for CUDA-graph QK capture."""
from __future__ import annotations

import torch
import torch.nn as nn


class QKHookHost(nn.Module):
    """Static-buffer host for a single attention module's QK capture.

    Owns static ``q_buf``/``k_buf`` sized ``(cap + 1, dim)`` with row 0 a
    write-only discard sentinel. ``capture_index``/``any_active`` are views into
    the registry slabs (bound later, not owned here). ``do_capture`` is an
    install-time gate (True only on the capture rank) read by install.py to
    decide whether to call ``capture()`` — never a branch inside the op.
    ``module_name`` is the egress bucket key, matching the eager ``qkv_hook``
    so analyzers/disk-save read the same buckets (the reuse contract).
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

        # Row 0 = discard sentinel (never read by egress). register_buffer pins
        # data_ptr across replays, but only if the host is built BEFORE vLLM's
        # cudagraph pool is allocated — the load_model patch guarantees that.
        # persistent=False keeps these out of the state_dict.
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

        # Routing views, populated later by bind_views(). Plain attributes, not
        # buffers: the registry owns the backing slabs, so registering would
        # double-register the same storage.
        self.capture_index: torch.Tensor | None = None
        self.any_active: torch.Tensor | None = None

        # Keep-alive sink for qk_probe: the op mutates this counter so it survives
        # DCE (capture is a side effect of the impl). Per-host so the per-layer op
        # nodes don't serialize on a shared mutated tensor.
        self.register_buffer(
            "sink", torch.zeros(1, dtype=torch.int64, device=device),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # View binding (called by the registry, once, after construction).
    # ------------------------------------------------------------------

    def bind_views(self, capture_index: torch.Tensor, any_active: torch.Tensor) -> None:
        """Attach this layer's row-views into the registry's global slabs.

        ``capture_index`` is a ``(cap,)`` int64 view; ``any_active`` is 0-d. The
        registry refreshes both each step; ``capture()`` only reads them, so the
        bindings hold for the host's lifetime.
        """
        self.capture_index = capture_index
        self.any_active = any_active

    # ------------------------------------------------------------------
    # Graph-recorded capture (the only thing the wrap calls).
    # ------------------------------------------------------------------

    def capture(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Scatter post-RoPE ``q``/``k`` rows into the static buffers.

        Single ``vllm_hook::capture_qk`` op call (graph-legal). The op aligns
        ``capture_index`` to the live token count and writes each row to its
        routed slot; tokens routed to row 0 (the sentinel) are discarded.
        ``any_active`` is passed for schema stability only — sentinel-row routing
        already makes unrequested tokens a no-op (see graph/ops.py).
        """
        torch.ops.vllm_hook.capture_qk(
            q, k, self.q_buf, self.k_buf, self.capture_index, self.any_active
        )

    def probe(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Impl-runs-at-runtime capture (prefill path; see graph/ops.py).

        Calls ``vllm_hook::qk_probe``, which mutates ``self.sink`` to stay alive
        and runs the eager capture body mid-forward into the worker buckets. No
        static buffers/routing — capture happens inside the op impl.
        """
        torch.ops.vllm_hook.qk_probe(q, k, self.sink, self.layer_num)
