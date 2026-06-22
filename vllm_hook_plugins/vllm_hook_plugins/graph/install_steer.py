"""CUDA-graph activation-steering install (Hybrid mode).

Steering is the MUTATING counterpart of the read-only QK/HS capture paths: it
adds a direction to the residual stream at one decoder layer so the change
propagates through every downstream layer and shifts the model's output. Under
PIECEWISE CUDA graphs this is expressed as the ``vllm_hook::steer_residual``
splitting op — registered as an eager seam exactly like ``qk_probe`` / ``hs_probe``
— except the op RETURNS the steered residual (instead of returning None and
mutating only a sink), so the modification flows into the next graph segment.

The per-request steering math is the eager ``SteerHookActWorker.steering_hook``
body, relocated into the op impl where the request state / forward context are
live. As with the capture paths, the op impl runs on real (non-replayed) prefill
forwards; decode under cudagraph replay is out of scope for this milestone.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.install import _cpu_1d  # shared CPU-tensor coercion
from vllm_hook_plugins.workers._common import (
    get_query_metadata,
    iter_matched_modules,
)
from vllm_hook_plugins.workers.probe_hidden_states_worker import match_layer
from vllm_hook_plugins.workers.steer_activation_worker import (
    _load_steering_vector,
    _resolve_steer_config,
)

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
# Per-instance 0-based layer index (matches the EAGER steer worker's `this_layer`,
# which is compared against config `optimal_layer`); -1 = don't steer this layer.
_STEER_LAYER_ATTR = "_vllm_hook_steer_layer"
_GLOBAL_SINK_STEER = None                 # int64 (1,) keep-alive against DCE

# Set at install so the op impl can reach the worker (vector cache, env config).
_ACTIVE_WORKER_STEER = None
_steer_dbg = {"n": 0, "fire": 0}

_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"


def _steer_body(hidden: torch.Tensor, residual: torch.Tensor,
                layer_idx: int, has_residual: int) -> None:
    """Eager ``steering_hook`` body, run from inside ``steer_residual`` mid-forward.

    Mutates ``residual`` IN PLACE for every request whose resolved steer config
    targets ``layer_idx`` (0-based, matching the eager worker). In-place is the
    contract the ``steer_residual`` op declares (``mutates_args=["residual"]``) so
    the change propagates downstream under the piecewise compiler — a clone-and-
    return variant was proven to have its return value dropped across the seam.
    Requests that don't steer this layer leave the tensor untouched.

    Unlike the eager hook — which adds to the WHOLE batch residual when any request
    targets the layer — this slices per request (``[start:end]``), which is both
    more correct for multi-request batches and identical to eager for the
    single-request case.
    """
    worker = _ACTIVE_WORKER_STEER
    if worker is None:
        return
    mr = worker.model_runner

    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)

    # query_start_loc from attn metadata, else the runner's persistent CPU buffer
    # (attn_metadata is None under torch.compile) — same fallback as the HS path.
    query_start_loc = None
    if metadata is not None:
        query_start_loc, _ = get_query_metadata(metadata)
    if query_start_loc is None:
        query_start_loc = _cpu_1d(getattr(mr, "query_start_loc", None))

    try:
        req_ids = mr.input_batch.req_ids
    except Exception:
        return
    if not req_ids or query_start_loc is None:
        return

    bs = len(req_ids)
    last_indices = query_start_loc
    env_path = getattr(worker, "_env_config_path", None)
    vector_cache = worker._vector_cache

    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = mr.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        steer_arg = (req_state.sampling_params.extra_args or {}).get("steer")
        cfg = _resolve_steer_config(steer_arg, env_path)
        if cfg is None:
            continue
        if int(cfg.get("optimal_layer", -1)) != int(layer_idx):
            continue

        method = cfg.get("method", "adjust_rs")
        coefficient = float(cfg.get("coefficient", 0))
        apply_at_all_positions = bool(cfg.get("apply_at_all_positions", True))
        vector_path = cfg["vector_path"]

        # Cache the loaded vector per path (per-request disk load would be slow).
        data = vector_cache.get(vector_path)
        if data is None:
            raw = _load_steering_vector(vector_path)
            data = {"dir": torch.tensor(raw["dir"])}
            if method == "adjust_rs":
                data["avg_proj"] = raw["avg_proj"]
            vector_cache[vector_path] = data

        start = int(last_indices[i].item())
        end = int(last_indices[i + 1].item())
        if end <= start:
            continue

        seg = residual[start:end]  # view into residual; add_ mutates it IN PLACE

        steering_vec = data["dir"].to(residual.device, dtype=residual.dtype)

        with PROF.timed("steer.apply", tier=2):
            if method == "add_vector":
                if not apply_at_all_positions:
                    raise NotImplementedError(
                        "Only supports apply_at_all_positions=True for now.")
                seg.add_(coefficient * steering_vec.view(1, -1))
            elif method == "adjust_rs":
                unit_vec = steering_vec
                avg_proj = data["avg_proj"].to(residual.device, dtype=residual.dtype)
                current_projections = torch.matmul(seg, unit_vec)
                coeff = (avg_proj - current_projections).unsqueeze(-1)
                seg.add_(coeff * unit_vec.view(1, -1))
            else:
                raise ValueError(f"Unknown steering method: {method}")

        PROF.incr("steer.fire")
        if _DEBUG and _steer_dbg["fire"] < 8:
            _steer_dbg["fire"] += 1
            print(f"[graph/dbg] steer APPLIED req={str(req_id)[:8]} "
                  f"layer={layer_idx} method={method} coeff={coefficient} "
                  f"rows={start}:{end} r={tuple(residual.shape)}", flush=True)


def _wrap_layer_class(cls: type) -> None:
    """Class-wrap ``cls.forward`` to steer the layer's residual output.

    Idempotent per class. Runs AFTER the original forward (we steer its output).
    The op mutates the residual IN PLACE, so the output object is returned
    unchanged — the next layer reads the same (now-steered) residual tensor. For
    the fused-residual pattern the layer returns ``(hidden, residual)`` and we pass
    index 1 (the residual) as the mutated arg, matching the eager hook which
    modifies ``output[1]`` and leaves ``hidden`` untouched. For a single-tensor
    output the whole tensor is the residual stream.
    """
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
            layer_idx = getattr(self, _STEER_LAYER_ATTR, -1)
            if layer_idx < 0 or _GLOBAL_SINK_STEER is None:
                return out
            if (isinstance(out, tuple) and len(out) >= 2
                    and isinstance(out[0], torch.Tensor)
                    and isinstance(out[1], torch.Tensor)):
                torch.ops.vllm_hook.steer_residual(
                    out[0], out[1], _GLOBAL_SINK_STEER, layer_idx, 1)
            else:
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor):
                    torch.ops.vllm_hook.steer_residual(
                        h, h, _GLOBAL_SINK_STEER, layer_idx, 0)
            return out
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_steer_hosts(worker) -> None:
    """Install the Hybrid-mode steering path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. Tags each layer
    with its 0-based index (matching the eager worker's ``this_layer``), makes the
    keep-alive sink, class-wraps the layer ``forward``, and registers
    ``steer_residual`` as a splitting op. Steering itself happens in the op impl
    (``_steer_body``).

    No ``should_capture`` rank gating: steering MUST be applied identically on
    every TP rank (the residual is replicated across ranks; steering only rank 0
    would desync the ranks), unlike capture which dedups to rank 0.
    """
    global _ACTIVE_WORKER_STEER, _GLOBAL_SINK_STEER

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_steer] no model on model_runner; skip steer install")
        return

    register_graph_ops()
    from vllm_hook_plugins.graph.ops import set_steer_fn
    _ACTIVE_WORKER_STEER = worker
    set_steer_fn(_steer_body)

    # State the steering body reads (the eager worker builds these in
    # _install_hooks, which graph mode skips). Keep per-worker so a config-less
    # request resolves nothing rather than crashing.
    if not getattr(worker, "_vector_cache", None):
        worker._vector_cache = {}
    if not hasattr(worker, "_env_config_path"):
        worker._env_config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_steer] no decoder layers matched LAYER_PATTERNS; "
              "steering inactive")
        return

    device_t = next(model.parameters()).device
    _GLOBAL_SINK_STEER = torch.zeros(1, dtype=torch.int64, device=device_t)
    n_wired = 0
    for name, module, layer_num0 in matched:
        setattr(module, _STEER_LAYER_ATTR, layer_num0)  # 0-based (eager convention)
        n_wired += 1
    for _, module, _ in matched:
        _wrap_layer_class(type(module))

    # Register steer_residual as a splitting op (load-bearing eager seam; see
    # install.py). Set on the WORKER's resolved config — the compiler reads that.
    try:
        cc = worker.vllm_config.compilation_config
        sops = list(getattr(cc, "splitting_ops", None) or [])
        if "vllm_hook::steer_residual" not in sops:
            sops.append("vllm_hook::steer_residual")
            cc.splitting_ops = sops
        print(f"[graph/install_steer] splitting_ops has steer_residual="
              f"{'vllm_hook::steer_residual' in sops} (n={len(sops)})")
    except Exception as e:  # noqa: BLE001
        print(f"[graph/install_steer] could not set splitting_ops: {e}")

    print(f"[graph/install_steer] op-mode steering wired: {n_wired} layer(s)")


__all__ = ["install_steer_hosts"]
