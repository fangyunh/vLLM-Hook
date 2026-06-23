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


def _bump_sink(sink) -> None:
    """Increment the DCE keep-alive sink, tolerating ``sink is None``.

    The sink only exists to survive dead-code elimination at COMPILE time (the
    ``mutates_args=["sink"]`` contract marks the op side-effecting so the piecewise
    compiler keeps it). Under ``VLLM_USE_AOT_COMPILE=1`` the compiled graph is
    serialized and RELOADED; the sink — a Python-global tensor baked as a graph
    constant — comes back as ``None`` across that boundary (torch's aot_compile does
    not round-trip eager-seam tensor consts that originate from module globals). By
    reload time DCE is already done and the op IS in the executed graph (we're in
    its impl), so the sink is vestigial: skipping the bump is correct and avoids the
    ``'NoneType' object has no attribute 'add_'`` crash.
    """
    if sink is not None:
        sink.add_(1)


def _qk_probe_impl(q: torch.Tensor, k: torch.Tensor, sink: torch.Tensor,
                   layer_idx: int) -> None:
    """Run the eager capture body, keeping the op alive via ``sink``.

    Bails while ``is_current_stream_capturing`` (capture/warmup forwards) so they
    don't pollute the buckets; real prefill forwards capture.
    """
    _bump_sink(sink)
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
    _bump_sink(sink)  # None-tolerant: sink is vestigial on AOT reload (see _bump_sink)
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


# steer_residual: the MUTATING analogue of qk_probe/hs_probe. The capture probes
# are read-only (return None, mutate only `sink`); steering must CHANGE the
# residual stream so the modification propagates into the downstream layers.
#
# It mutates the residual IN PLACE and declares ``mutates_args=["residual"]`` —
# the same in-graph data-plane primitive ``capture_qk`` uses to write its static
# buffers. This is the load-bearing contract: it tells the piecewise compiler the
# op writes ``residual``, so downstream reads are re-threaded to the mutated value
# and the static-buffer copy into the next cudagraph segment picks up the change.
# (A functional variant that RETURNED a new residual was proven on GPU to fire but
# have its return value DROPPED across the segment seam — the mutation never
# reached downstream layers. In-place + mutates_args is the path that propagates.)
#
# The eager steering math runs in the impl on real (non-replayed) prefill forwards;
# on cudagraph replay (decode) the impl is skipped, so decode-phase steering is out
# of scope for this milestone (mirrors the capture path's prefill-only proof).
_STEER_FN = None  # set by install_steer.set_steer_fn; (hidden, residual, layer_idx, has_residual) -> None


def set_steer_fn(fn) -> None:
    """Register the runtime steering callback ``steer_residual``'s impl invokes.

    ``fn(hidden, residual, layer_idx, has_residual)`` mutates ``residual`` IN PLACE
    (e.g. ``residual[start:end].add_(delta)``) and returns None.
    """
    global _STEER_FN
    _STEER_FN = fn


def _steer_residual_impl(hidden: torch.Tensor, residual: torch.Tensor,
                         sink: torch.Tensor, layer_idx: int,
                         has_residual: int) -> None:
    """Apply activation steering to ``residual`` IN PLACE mid-forward.

    Bumps ``sink`` (keep-alive). Bails while ``is_current_stream_capturing``
    (warmup/capture forwards) and when no callback is set, leaving ``residual``
    untouched so a steering-off step is a true no-op. On a real forward it
    delegates to the steering body, which mutates ``residual`` in place.
    """
    _bump_sink(sink)  # None-tolerant: sink is vestigial on AOT reload (see _bump_sink)
    _FIRE_COUNT[0] += 1
    fn = _STEER_FN
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        pass
    if _DBG and not capturing and residual.shape[0] <= 2048 and _DBG_NONCAP[0] < 30:
        _DBG_NONCAP[0] += 1
        print(f"[graph/dbg] steer_residual impl REAL fire #{_DBG_NONCAP[0]} "
              f"layer={int(layer_idx)} capturing={capturing} "
              f"r={tuple(residual.shape)} has_resid={int(has_residual)}", flush=True)
    if fn is None or capturing:
        return None
    try:
        fn(hidden, residual, int(layer_idx), int(has_residual))
    except Exception as e:  # noqa: BLE001
        if _DBG and _DBG_NONCAP[0] < 12:
            print(f"[graph/dbg] steer_residual body raised: {e!r}", flush=True)
    return None


def _steer_residual_fake(hidden: torch.Tensor, residual: torch.Tensor,
                         sink: torch.Tensor, layer_idx: int,
                         has_residual: int) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


# steer_buffer: the FULL-mode (buffer) analogue of steer_residual — the MUTATING
# counterpart of capture_qk/capture_hs. A real CUDA kernel (NOT a splitting op) that
# reads static routing buffers and does a masked, in-place `residual += coeff * vec`
# (the `add_vector` method). It is absorbed into the decode cudagraph and replays
# every step; the in-place mutation under the `mutates_args=["residual"]` contract is
# what threads the steered residual into the downstream layers (same primitive the
# op-mode steer_residual relies on, but with no Python on the replay path).
#
# This is the LoRA-slot model: a fixed `vec_table[V_max, hidden]` library, a per-row
# `token → vec_id` map, and a per-row `coeff` that gates no-steer rows to 0. Different
# requests pick different (layer, vector, coefficient) purely as buffer contents.


def _steer_buffer_impl(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Masked in-place steering add — unified ``add_vector`` + ``adjust_rs``.

    Per token ``t`` (``unit = vec_table[vec_id[t]]``):
      * ``steer_mode[t] == 0`` (add_vector / no-op): ``c = coeff[t]`` (host-supplied;
        0 = no-op).
      * ``steer_mode[t] == 1`` (adjust_rs): ``c = avg_proj[vec_id[t]] - (residual[t]·unit)``
        — the per-token coefficient is computed IN-KERNEL from the live residual (the
        projection ``residual·unit`` only exists at replay), so adjust_rs cannot be a
        static host coefficient like add_vector.
    Then ``residual[t] += c · unit``. Vectorised + graph-safe (gather + reduction +
    ``where`` + scaled add, no host sync); the routing tensors are length CAP and the
    leading ``n = residual.shape[0]`` rows align with this step's tokens.

    (The arg is ``steer_mode``, not ``mode``: ``mode`` collides with a reserved
    parameter of torch's ``auto_functionalized`` HOP — the wrapper for mutating ops
    under torch.compile — and raises ``auto_functionalized_fake() got multiple values
    for argument 'mode'`` at compile time.)
    """
    # Buffer-mode static buffers are Python-global tensors baked as graph consts; like
    # the probe sinks they can deserialize to None on an AOT-compile reload. Skip the
    # add rather than crash (buffer mode is not AOT-validated; defensive only).
    if coeff is None or vec_id is None or vec_table is None:
        return None
    _FIRE_COUNT[0] += 1  # diagnostic only
    n = residual.shape[0]
    vids = vec_id[:n].to(torch.long)                            # (n,)
    units = vec_table.index_select(0, vids).to(residual.dtype)  # (n, hidden)
    c_add = coeff[:n].to(residual.dtype)                        # (n,) host coeff
    # adjust_rs coefficient, computed for every row but selected only where mode==1.
    proj = (residual[:n] * units).sum(dim=-1)                   # (n,) residual·unit
    avg = avg_proj.index_select(0, vids).to(residual.dtype)     # (n,) per-vector target
    c_adj = avg - proj                                         # (n,)
    is_adj = steer_mode[:n].to(torch.bool)                     # (n,)
    c = torch.where(is_adj, c_adj, c_add)                      # (n,)
    residual[:n].add_(c.unsqueeze(-1) * units)                 # in-place — mutates contract
    return None


def _steer_buffer_fake(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Meta/fake impl: in-place mutation, no new tensor under tracing."""
    return None


QK_PROBE_OP_NAME = "qk_probe"
HS_PROBE_OP_NAME = "hs_probe"
STEER_RESIDUAL_OP_NAME = "steer_residual"

# Constants so downstream files import rather than hard-code op names.
LIB_NAMESPACE = "vllm_hook"
CAPTURE_QK_OP_NAME = "capture_qk"
CAPTURE_HS_OP_NAME = "capture_hs"
STEER_BUFFER_OP_NAME = "steer_buffer"


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
    # Buffer-mode static buffers are Python-global tensors baked as graph consts;
    # like the probe sinks they can deserialize to None on an AOT-compile reload.
    # Skip the scatter rather than crash (buffer mode is not AOT-validated; this is
    # defensive — op mode is the default capture path).
    if q_buf is None or k_buf is None or index is None:
        return None
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
# capture_hs: the HS analogue of capture_qk — scatter the residual stream into
# a static per-layer sink, in place, as a graph-recorded kernel. The HS buffer
# path (VLLM_HOOK_HS_CAPTURE=buffer) is the FULL-mode counterpart to op-mode
# hs_probe: NOT a splitting op, so it is absorbed into the decode cudagraph and
# replays every step (no Python on replay). Unlike QK there is no prefix-K — the
# residual stream is per-token, so the scattered rows ARE the artifact.
# ---------------------------------------------------------------------------


def _capture_hs_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    """Form the residual stream and scatter its rows into ``hs_buf``, in place.

    ``combined = hidden + residual`` for fused-residual layers (``has_residual``
    truthy), else ``hidden``. ``index`` is length CAP but ``hidden`` has only
    ``n = hidden.shape[0]`` rows this step; the leading ``n`` routing entries
    align row-for-row, and row 0 is the discard sentinel for unrequested tokens.
    Vectorised, no host sync — graph-legal. ``has_residual`` is a per-layer
    constant baked at the call site, so this adds no data-dependent control flow.
    """
    _FIRE_COUNT[0] += 1  # diagnostic only
    # Static buffers are Python-global tensors baked as graph consts; like the
    # probe sinks they can deserialize to None on an AOT-compile reload. Skip the
    # scatter rather than crash (mirrors _capture_qk_impl's defensive guard).
    if hs_buf is None or index is None:
        return None
    n = hidden.shape[0]
    idx = index[:n].to(torch.long)
    combined = hidden + residual if has_residual else hidden
    src = combined if combined.dtype == hs_buf.dtype else combined.to(hs_buf.dtype)
    hs_buf.index_copy_(0, idx, src)
    return None


def _capture_hs_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
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
            op_name=CAPTURE_HS_OP_NAME,
            op_func=_capture_hs_impl,
            mutates_args=["hs_buf"],
            fake_impl=_capture_hs_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
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

    # The steering op RETURNS a Tensor (the steered residual) — unlike the
    # read-only probes above. mutates_args=["sink"] keeps it side-effecting; the
    # returned value carries the actual steering into the next graph segment.
    try:
        direct_register_custom_op(
            op_name=STEER_RESIDUAL_OP_NAME,
            op_func=_steer_residual_impl,
            mutates_args=["sink", "residual"],
            fake_impl=_steer_residual_fake,
            target_lib=_LIB,
            dispatch_key="CUDA",
        )
    except Exception as e:  # noqa: BLE001
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise

    # The FULL-mode (buffer) steering op: masked in-place residual add, NOT a
    # splitting op — absorbed into the decode cudagraph and replayed every step.
    try:
        direct_register_custom_op(
            op_name=STEER_BUFFER_OP_NAME,
            op_func=_steer_buffer_impl,
            mutates_args=["residual"],
            fake_impl=_steer_buffer_fake,
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
        ns, CAPTURE_HS_OP_NAME
    ) or not hasattr(ns, QK_PROBE_OP_NAME) or not hasattr(
        ns, HS_PROBE_OP_NAME
    ) or not hasattr(ns, STEER_RESIDUAL_OP_NAME) or not hasattr(
        ns, STEER_BUFFER_OP_NAME
    ):
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


def capture_hs(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    hs_buf: torch.Tensor,
    index: torch.Tensor,
    has_residual: int,
) -> None:
    """Thin wrapper over ``torch.ops.vllm_hook.capture_hs`` (resolved at call time)."""
    return torch.ops.vllm_hook.capture_hs(hidden, residual, hs_buf, index, has_residual)


def steer_buffer(
    residual: torch.Tensor,
    coeff: torch.Tensor,
    vec_id: torch.Tensor,
    vec_table: torch.Tensor,
    avg_proj: torch.Tensor,
    steer_mode: torch.Tensor,
) -> None:
    """Thin wrapper over ``torch.ops.vllm_hook.steer_buffer`` (resolved at call time)."""
    return torch.ops.vllm_hook.steer_buffer(
        residual, coeff, vec_id, vec_table, avg_proj, steer_mode)
