"""Custom ops for CUDA-graph QK capture.

Why a custom op rather than a forward hook: under ``enforce_eager=False`` vLLM
runs the model under ``torch.compile``. Dynamo traces each ``nn.Module.forward``
once and replaces it with compiled code, so the eager ``register_forward_hook``
callbacks (which fire on ``nn.Module.__call__``) are bypassed — zero capture. A
custom op is instead recorded as an opaque graph node that replays every step: a
class-level forward wrap that calls such an op lands in the captured graph and
fires every step (counter == layers x steps). This module registers the real
capture op that the QK wrap (``install.py``) calls.

The library namespace matters: the op is invoked as
``torch.ops.vllm_hook.capture_qk``, so it MUST live under a library named
``vllm_hook`` — NOT vLLM's default ``vllm`` namespace, which is what
``direct_register_custom_op`` targets unless given a ``target_lib``. A namespace
mismatch makes Dynamo raise ``AttributeError`` when it traces the call. The
``Library`` object must stay alive for the op's lifetime, so we hold it in a
module global (``_LIB``).

Scatter semantics of ``capture_qk(q, k, q_buf, k_buf, index, active)``
(``mutates_args=("q_buf", "k_buf")``):

- ``index`` is the per-layer routing view of length CAP (the registry slab
  column count). Only its leading ``num_tokens`` entries (one per row of q/k)
  are meaningful this step; the op slices ``idx = index[:q.shape[0]]`` so the
  index length matches the source row count, which ``index_copy_`` requires.
  ``q.shape[0]`` is the shape-specialised, constant-per-graph token width, so
  the slice length is a graph constant, not a data-dependent symbol — graph-legal
  under piecewise capture.
- For each token row t with idx[t] != 0:  q_buf[idx[t]] = q[t], k_buf[idx[t]] =
  k[t]. idx[t] == 0 means discard: row 0 of the buffers is a reserved,
  write-only sentinel that every discarded token scatters into, and egress NEVER
  reads it. Multiple discarded tokens all write row 0; ``index_copy_`` with
  duplicate destination indices is non-deterministic in PyTorch, but that is
  harmless here precisely because row 0 is write-only.
- ``active`` is a device int/bool gate kept in the signature for fake/meta op
  stability and as an explicit "any token captured this layer this step" marker.
  It is NOT multiplied into q/k: the sentinel-row routing already makes inactive
  tokens a no-op, and a multiply would (a) allocate a full batch-sized temporary
  every layer every step inside the captured region and (b) risk zeroing real
  data if active were ever miscomputed while an index pointed at a real row.
- Fixed shapes, opaque node: a CUDA real impl plus a fake/meta impl returning
  None. q shape (num_tokens, num_q_heads*head_dim); k shape (num_tokens,
  num_kv_heads*head_dim); q_buf (CAP+1, num_q_heads*head_dim); k_buf (CAP+1,
  num_kv_heads*head_dim).

The implementation is fully vectorized — no host sync, no ``.item()``, no Python
loop over tokens — so it is safe to record inside a CUDA graph. It uses
``index_copy_``, which scatters every source row into ``buf[idx[row]]``;
discarded tokens (index 0) all land on the sentinel row and are ignored.
"""
from __future__ import annotations

import os

import torch

_DBG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"
_DBG_NONCAP = [0]  # count of non-capturing (real-forward) impl fires logged

# ---------------------------------------------------------------------------
# module globals (kept alive for the op lifetime)
# ---------------------------------------------------------------------------
# The dedicated vllm_hook library; see module docstring for why this must not be
# vLLM's vllm namespace. Populated by register_graph_ops.
_LIB = None
_OPS_REGISTERED = False

# Counts real executions of the capture op impl (prefill plus graph-capture
# passes, but NOT cudagraph replays, where the python impl does not run). Read
# via get_fire_count from the execute_model wrapper to confirm the op is actually
# in the executed graph. Cheap; no effect on correctness.
_FIRE_COUNT = [0]


def get_fire_count() -> int:
    return _FIRE_COUNT[0]


# ---------------------------------------------------------------------------
# qk_probe: impl-runs-at-runtime capture (the prefill path)
# ---------------------------------------------------------------------------
# Why this exists alongside capture_qk: the device-buffer scatter (capture_qk)
# needs the per-step routing index built before the forward, but vLLM v1 builds
# the current step's input_batch inside execute_model, so pre-forward routing
# can't see the current step. A custom op's impl, however, runs at runtime during
# the forward (for non-cudagraph-replayed passes, i.e. prefill under piecewise),
# exactly where the eager register_forward_hook used to run — with the live
# forward context and the current input_batch. So qk_probe relocates the proven
# eager capture body into an op that survives torch.compile. Decode under
# cudagraph replay does not run the impl; that path keeps the static-buffer
# machinery, a later milestone.
#
# The op stays alive (not DCE'd) by mutating a per-layer sink counter. The heavy
# lifting is delegated to a callback the install layer registers (set_capture_fn),
# so this module stays free of worker/vLLM imports and avoids a circular import
# with install.py.

_CAPTURE_FN = None  # set by install.set_capture_fn; signature (q, k, layer_idx)


def set_capture_fn(fn) -> None:
    """Register the runtime capture callback ``qk_probe``'s impl invokes.

    ``fn(q, k, layer_idx)`` runs mid-forward with the live forward context and
    does the eager per-request slicing, gating, and prefix-K into the worker
    buckets.
    """
    global _CAPTURE_FN
    _CAPTURE_FN = fn


def _qk_probe_impl(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
                   layer_idx: int) -> None:
    """Run the eager capture body, keeping the op alive via ``sink``.

    Bails during vLLM's CUDA-graph capture pass (``is_current_stream_capturing``),
    the same bail the eager hook had, so warmup and capture forwards don't pollute
    the buckets. Real prefill forwards (capturing is False) do the capture.
    """
    sink.add_(1)
    _FIRE_COUNT[0] += 1
    fn = _CAPTURE_FN
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        pass
    # Log non-capturing fires for real (small) token batches only — warmup and
    # profiling use a huge shape (e.g. 16384) and would flood the log. If
    # small-batch fires appear during generation, the impl runs on real forwards
    # and is not replay-skipped; if they never appear, generation forwards are
    # cudagraph replays that skip the impl.
    if _DBG and not capturing and q.shape[0] <= 2048 and _DBG_NONCAP[0] < 30:
        _DBG_NONCAP[0] += 1
        print(f"[graph/dbg] qk_probe impl REAL fire #{_DBG_NONCAP[0]} "
              f"layer={int(layer_idx)} capturing={capturing} "
              f"q={tuple(q.shape)}", flush=True)
    if fn is None:
        return
    if capturing:
        return
    # Capture is best-effort: never let a capture bug crash the model forward.
    try:
        fn(q, k, int(layer_idx))
    except Exception as e:  # noqa: BLE001
        if _DBG and _DBG_NONCAP[0] < 12:
            print(f"[graph/dbg] qk_probe capture body raised: {e!r}", flush=True)


def _qk_probe_fake(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
                   layer_idx: int) -> None:
    return None


# ---------------------------------------------------------------------------
# hs_probe: impl-runs-at-runtime hidden-state capture (mirrors qk_probe)
# ---------------------------------------------------------------------------
# Same mechanism as qk_probe, but for the hidden-states worker: capture the
# decoder layer output (the residual stream) rather than attention q/k. The
# decoder-layer class wrap is the exact point that lands in the captured graph.
# vLLM decoder layers use a fused-residual pattern and return (hidden_states,
# residual); the residual stream is their sum. We pass both tensors to the op
# plus a has_residual flag (1 for the tuple-of-two case, 0 when the layer
# returned a single tensor — then residual is just a copy of hidden and is
# ignored), and the capture body forms hidden + residual exactly like the eager
# hs_hook.
_HS_CAPTURE_FN = None  # set by install_hs.set_hs_capture_fn; (hidden, residual, layer_idx, has_residual)


def set_hs_capture_fn(fn) -> None:
    """Register the runtime hidden-state capture callback ``hs_probe``'s impl calls."""
    global _HS_CAPTURE_FN
    _HS_CAPTURE_FN = fn


def _hs_probe_impl(hidden: torch.Tensor, residual: torch.Tensor,
                   sink: torch.Tensor, layer_idx: int,
                   has_residual: int) -> None:
    """Run the eager hidden-state capture body, keeping the op alive via ``sink``.

    Bails during vLLM's CUDA-graph capture pass, the same bail as the eager hook,
    so warmup and capture forwards don't pollute the buckets. Real forwards
    capture.
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

# Public op name and qualified path, so downstream files import constants rather
# than hard-coding strings.
LIB_NAMESPACE = "vllm_hook"
CAPTURE_QK_OP_NAME = "capture_qk"


def _import_direct_register_custom_op():
    """Locate vLLM's ``direct_register_custom_op`` across vLLM versions.

    The symbol has moved between vLLM releases, so probe each known location in
    turn and return the first that resolves.
    """
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

    Vectorized and graph-safe: no host sync, no ``.item()``, no per-token Python
    loop. ``index_copy_(0, idx, src)`` writes ``buf[idx[t]] = src[t]`` for every
    token t; tokens routed to the sentinel row (index 0) are overwritten there
    and never read by egress.

    On index/source alignment: ``index`` is the per-layer routing view of length
    CAP (``max_num_batched_tokens``), while q/k have only ``num_tokens`` rows this
    step (num_tokens <= CAP, and generally != CAP). ``index_copy_`` requires
    ``idx.numel() == src.shape[0]``, so we slice the index to the source length,
    ``idx = index[:n]``. The routing layer writes the destination row for the
    token at flat position ``pos`` into ``index[pos]`` (pos in 0..num_tokens-1),
    so the leading ``n`` entries align row-for-row with q/k. ``n = q.shape[0]`` is
    the shape-specialised graph-constant token width, so the slice is graph-legal.

    No ``active`` multiply: the sentinel-row routing already makes inactive tokens
    a no-op (they overwrite the write-only row 0). Multiplying would allocate a
    batch-sized temporary every layer every step inside the captured region and
    could zero real data on a routing miscompute. ``active`` is left in the
    signature for op-schema stability but unused.

    No dtype cast when q already matches q_buf (it should: buf dtype is the model
    compute dtype and q/k are post-RoPE in that same dtype). A cast is applied
    only as a defensive fallback, since a dtype mismatch would otherwise raise
    inside ``index_copy_``; in the common case ``.to()`` returns q/k unchanged
    with no allocation.

    All operations run on the capture rank's device; q_buf/k_buf are the static
    per-layer buffers whose data_ptr stays fixed across graph replays.
    """
    del active  # unused: sentinel-row routing supersedes the multiplicative gate
    _FIRE_COUNT[0] += 1  # diagnostic only
    # index_copy_ requires a 1-D LongTensor whose length matches src.shape[0].
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
    """Meta/fake impl: the op mutates ``q_buf``/``k_buf`` in place and returns
    nothing, so under tracing it produces no new tensor."""
    return None


# ---------------------------------------------------------------------------
# registration entry point
# ---------------------------------------------------------------------------


def register_graph_ops() -> None:
    """Register all graph-capture custom ops under the ``vllm_hook`` library.

    Idempotent: safe to call from a re-entrant ``load_model`` or from multiple
    workers in the same process; an "already registered" error is swallowed. The
    ``Library`` handle is held in a module global so the ops stay alive.
    """
    global _LIB, _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    from torch.library import Library

    direct_register_custom_op = _import_direct_register_custom_op()

    # Dedicated library so ops live at torch.ops.vllm_hook.* (matching the call
    # sites in install.py), not torch.ops.vllm.*. Held globally so it stays alive.
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
        # Already registered in this process (e.g. re-entrant load_model) — fine.
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # qk_probe: the impl-runs-at-runtime prefill capture op (see module notes).
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

    # hs_probe: the impl-runs-at-runtime hidden-state capture op (decoder-layer
    # output). Registered alongside the QK ops so a single register_graph_ops
    # call serves either worker; an unused op is simply never called.
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

    # Only mark registered once the op handle actually resolves, so a partial
    # failure (registration raised an already-looking error but the op is not
    # actually present) does not leave callers to hit a confusing AttributeError
    # at capture time. If the handle is missing, surface the real error.
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
    """Thin wrapper over ``torch.ops.vllm_hook.qk_probe`` (the prefill path).

    Resolves the op handle at call time, after ``register_graph_ops`` has run.
    """
    return torch.ops.vllm_hook.qk_probe(q, k, sink, layer_idx)


# ---------------------------------------------------------------------------
# bound op handle (lazy: torch.ops.* is only valid after registration)
# ---------------------------------------------------------------------------


def capture_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    index: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Thin wrapper over ``torch.ops.vllm_hook.capture_qk``.

    Provided so callers (the class-level ``Attention.forward`` wrap in install.py)
    can ``from vllm_hook_plugins.graph import capture_qk`` and call it directly;
    this resolves the op handle at call time, after ``register_graph_ops`` has
    run. The call is traced as the opaque ``vllm_hook::capture_qk`` graph node.
    """
    return torch.ops.vllm_hook.capture_qk(q, k, q_buf, k_buf, index, active)
