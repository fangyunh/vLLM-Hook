"""CUDA-graph-safe capture subpackage."""
from __future__ import annotations

from vllm_hook_plugins.graph.ops import capture_qk, register_graph_ops

__all__ = [
    "register_graph_ops",
    "capture_qk",
]
