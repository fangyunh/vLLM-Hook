"""Custom ops for CUDA-graph QK/HS capture."""
from __future__ import annotations

import os

import torch

_DBG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"
_DBG_NONCAP = [0]  # count of non-capturing (real-forward) impl fires logged

# The dedicated vllm_hook library (held alive); populated by register_graph_ops.
_LIB = None
_OPS_REGISTERED = False

# Real impl executions (prefill + graph-capture passes, NOT cudagraph replays).
# Read via get_fire_count to confirm the op is in the executed graph. Diagnostic.
_FIRE_COUNT = [0]


def get_fire_count() -> int:
    return _FIRE_COUNT[0]


# qk_probe: impl-runs-at-runtime capture (the prefill path). Unlike capture_qk's
# pre-forward routing (which can't see the current step's input_batch), the op
# impl runs mid-forward where the eager hook used to — but only on real
# (non-replayed) passes, so this covers prefill, not decode under replay. Stays
# alive via a sink counter; heavy lifting delegated to a callback to avoid a
# circular import with install.py.
_CAPTURE_FN = None  # set by install.set_capture_fn; signature (q, k, layer_idx)


def set_capture_fn(fn) -> None:
    """Register the runtime callback ``qk_probe``'s impl invokes: ``fn(q, k, layer_idx)``."""
    global _CAPTURE_FN
    _CAPTURE_FN = fn


def _qk_probe_impl(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
                   layer_idx: int) -> None:
    """Run the eager capture body, keeping the op alive via ``sink``.

    Bails while ``is_current_stream_capturing`` (capture/warmup forwards) so they
    don't pollute the buckets; real prefill forwards capture.
    """
    sink.add_(1)
    _FIRE_COUNT[0] += 1
    fn = _CAPTURE_FN
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        pass
    # Log small-batch real fires only; warmup/profiling use huge shapes that flood.
    if _DBG and not capturing and q.shape[0] <= 2048 and _DBG_NONCAP[0] < 30:
        _DBG_NONCAP[0] += 1
        print(f"[graph/dbg] qk_probe impl REAL fire #{_DBG_NONCAP[0]} "
              f"layer={int(layer_idx)} capturing={capturing} "
              f"q={tuple(q.shape)}", flush=True)
    if fn is None:
        return
    if capturing:
        return
    # Best-effort: never let a capture bug crash the forward.
    try:
        fn(q, k, int(layer_idx))
    except Exception as e:  # noqa: BLE001
        if _DBG and _DBG_NONCAP[0] < 12:
            print(f"[graph/dbg] qk_probe capture body raised: {e!r}", flush=True)


def _qk_probe_fake(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
                   layer_idx: int) -> None:
    return None


# hs_probe: like qk_probe but captures the decoder-layer residual stream. vLLM
# layers use a fused-residual pattern returning (hidden, residual) whose sum is
# the residual stream; has_residual=1 for that case, 0 when the layer returned a
# single tensor (then residual is a copy of hidden and ignored).
_HS_CAPTURE_FN = None  # set by install_hs.set_hs_capture_fn; (hidden, residual, layer_idx, has_residual)


def set_hs_capture_fn(fn) -> None:
    """Register the runtime hidden-state capture callback ``hs_probe``'s impl calls."""
    global _HS_CAPTURE_FN
    _HS_CAPTURE_FN = fn


def _hs_probe_impl(hidden: torch.Tensor, residual: torch.Tensor,
                   sink: torch.Tensor, layer_idx: int,
                   has_residual: int) -> None:
    """Run the eager hidden-state capture body, keeping the op alive via ``sink``.

    Bails while capturing (warmup/capture forwards); real forwards capture.
    """
    sink.add_(1)
    _FIRE_COUNT[0] += 1
    fn = _HS_CAPTURE_FN
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        pass
    if _DBG and not capturing and hidden.shape[0] <= 2048 and _DBG_NONCAP[0] < 30:
        _DBG_NONCAP[0] += 1
        print(f"[graph/dbg] hs_probe impl REAL fire #{_DBG_NONCAP[0]} "
              f"layer={int(layer_idx)} capturing={capturing} "
              f"h={tuple(hidden.shape)} has_resid={int(has_residual)}", flush=True)
    if fn is None or capturing:
        return
    try:
        fn(hidden, residual, int(layer_idx), int(has_residual))
    except Exception as e:  # noqa: BLE001
        if _DBG and _DBG_NONCAP[0] < 12:
            print(f"[graph/dbg] hs_probe capture body raised: {e!r}", flush=True)


def _hs_probe_fake(hidden: torch.Tensor, residual: torch.Tensor,
                   sink: torch.Tensor, layer_idx: int,
                   has_residual: int) -> None:
    return None


QK_PROBE_OP_NAME = "qk_probe"
HS_PROBE_OP_NAME = "hs_probe"

# Constants so downstream files import rather than hard-code op names.
LIB_NAMESPACE = "vllm_hook"
CAPTURE_QK_OP_NAME = "capture_qk"


def _import_direct_register_custom_op():
    """Locate ``direct_register_custom_op``; it has moved between vLLM releases."""
    errs = []
    for modpath in (
        "vllm.utils",
        "vllm.utils.torch_utils",
        "vllm.utils._custom_ops",
        "vllm.compilation.decorators",
    ):
        try:
            mod = __import__(modpath, fromlist=["direct_register_custom_op"])
            fn = getattr(mod, "direct_register_custom_op", None)
            if fn is not None:
                return fn
            errs.append(f"  {modpath}: imported but no symbol")
        except ImportError as e:
            errs.append(f"  {modpath}: {e}")
    raise ImportError(
        "Could not locate direct_register_custom_op in any known vLLM path.\n"
        + "\n".join(errs)
    )


# ---------------------------------------------------------------------------
# capture_qk: real (CUDA) and fake (meta) impls
# ---------------------------------------------------------------------------


def _capture_qk_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Scatter post-RoPE q/k rows into the static buffers, in place.

    Vectorized and graph-safe (no host sync). ``index`` is length CAP but q/k have
    only ``n = q.shape[0]`` rows this step, and ``index_copy_`` requires the index
    length to match; the leading ``n`` routing entries align row-for-row with q/k,
    and ``n`` is a graph-constant slice. ``active`` is unused — sentinel-row
    routing already no-ops inactive tokens, and a multiply would allocate a
    per-step temporary and could zero real data on a routing miscompute. The dtype
    cast is a defensive fallback (q/k should already match buf dtype).
    """
    del active  # unused: sentinel-row routing supersedes the multiplicative gate
    _FIRE_COUNT[0] += 1  # diagnostic only
    n = q.shape[0]
    idx = index[:n].to(torch.long)
    q_src = q if q.dtype == q_buf.dtype else q.to(q_buf.dtype)
    k_src = k if k.dtype == k_buf.dtype else k.to(k_buf.dtype)
    q_buf.index_copy_(0, idx, q_src)
    k_buf.index_copy_(0, idx, k_src)
    return None


def _capture_qk_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


# ---------------------------------------------------------------------------
# registration entry point
# ---------------------------------------------------------------------------


def register_graph_ops() -> None:
    """Register all graph-capture custom ops under ``vllm_hook``. Idempotent:
    "already registered" errors are swallowed."""
    global _LIB, _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    from torch.library import Library

    direct_register_custom_op = _import_direct_register_custom_op()

    # Dedicated library so ops live at torch.ops.vllm_hook.*, not torch.ops.vllm.*.
    if _LIB is None:
        _LIB = Library(LIB_NAMESPACE, "FRAGMENT")

    try:
        direct_register_custom_op(
            op_name=CAPTURE_QK_OP_NAME,
            op_func=_capture_qk_impl,
            mutates_args=["q_buf", "k_buf"],
            fake_impl=_capture_qk_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        # Already registered in this process — fine.
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    try:
        direct_register_custom_op(
            op_name=QK_PROBE_OP_NAME,
            op_func=_qk_probe_impl,
            mutates_args=["sink"],
            fake_impl=_qk_probe_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # Registered alongside the QK ops so one call serves either worker.
    try:
        direct_register_custom_op(
            op_name=HS_PROBE_OP_NAME,
            op_func=_hs_probe_impl,
            mutates_args=["sink"],
            fake_impl=_hs_probe_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # Mark registered only once the handles resolve, so a partial failure surfaces
    # here rather than as a confusing AttributeError at capture time.
    ns = getattr(torch.ops, LIB_NAMESPACE, None)
    if ns is None or not hasattr(ns, CAPTURE_QK_OP_NAME) or not hasattr(
        ns, QK_PROBE_OP_NAME
    ) or not hasattr(ns, HS_PROBE_OP_NAME):
        raise RuntimeError(
            f"register_graph_ops: ops under torch.ops.{LIB_NAMESPACE} "
            "did not resolve after registration"
        )

    _OPS_REGISTERED = True


def qk_probe(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
             layer_idx: int) -> None:
    """Thin wrapper over ``torch.ops.vllm_hook.qk_probe`` (resolved at call time)."""
    return torch.ops.vllm_hook.qk_probe(q, k, sink, layer_idx)


# bound op handle (lazy: torch.ops.* is only valid after registration)


def capture_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Thin wrapper over ``torch.ops.vllm_hook.capture_qk`` (resolved at call time),
    so callers can import and call it directly; traced as the opaque graph node."""
    return torch.ops.vllm_hook.capture_qk(q, k, q_buf, k_buf, index, active)
