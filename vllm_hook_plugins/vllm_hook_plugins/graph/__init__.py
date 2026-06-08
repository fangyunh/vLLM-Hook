"""CUDA-graph capture subpackage (v0.3.0).

This package adds CUDA-graph-safe QK capture for PIECEWISE and FULL_DECODE_ONLY
cudagraph modes, while leaving the v0.2.0 eager `register_forward_hook` path in
`workers/probe_hookqk_worker.py` fully intact as the fallback.

The eager path is bypassed under `torch.compile` (Dynamo traces each
`nn.Module.forward` once and replaces it with compiled code, so forward-hook
callbacks never fire). A *custom op*, by contrast, is recorded as a graph node
and replays on every step — proven by `tests/cuda_graph/tests/step0/`. This
subpackage builds the real capture op on top of that lesson.

File 1/5 (this + `ops.py`) provides only the op-registration layer:

    register_graph_ops()  -> idempotent registration of the capture ops
    capture_qk            -> bound handle to torch.ops.vllm_hook.capture_qk

Downstream files (`hosts.py`, `registry.py`, `install.py`) consume these.
"""
from __future__ import annotations

from vllm_hook_plugins.graph.ops import capture_qk, register_graph_ops

__all__ = [
    "register_graph_ops",
    "capture_qk",
]
