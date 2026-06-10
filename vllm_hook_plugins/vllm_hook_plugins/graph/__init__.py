"""CUDA-graph capture subpackage.

Adds CUDA-graph-safe capture for the PIECEWISE / FULL_DECODE_ONLY cudagraph
modes, leaving the v0.2.0 eager ``register_forward_hook`` path in
``workers/probe_hookqk_worker.py`` intact as the fallback.

The eager path is bypassed under ``torch.compile``: Dynamo traces each
``nn.Module.forward`` once and replaces it with compiled code, so forward-hook
callbacks never fire. A custom op is recorded as a graph node instead and
replays on every step. This subpackage builds the real capture op on that
mechanism; ``ops.py`` is the op-registration layer (``register_graph_ops`` plus
the ``capture_qk`` handle), consumed by ``hosts.py`` / ``registry.py`` /
``install.py``.
"""
from __future__ import annotations

from vllm_hook_plugins.graph.ops import capture_qk, register_graph_ops

__all__ = [
    "register_graph_ops",
    "capture_qk",
]
