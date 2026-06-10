"""CUDA-graph hidden-state capture install (Hybrid mode)."""
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
    init_async_save_thread,
    iter_matched_modules,
)
from vllm_hook_plugins.workers.probe_hidden_states_worker import match_layer

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_HS_LAYER_ATTR = "_vllm_hook_hs_layer"   # per-instance 1-based layer num; -1 = skip
_GLOBAL_SINK_HS = None                    # int64 (1,) keep-alive against DCE

# Set at install so the op impl can reach the worker and egress keys.
_ACTIVE_WORKER_HS = None
_LAYER_TO_NAME_HS: Dict[int, str] = {}    # 1-based layer num -> egress bucket key
_capture_dbg = {"n": 0, "cap": 0}

_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"


def _capture_body_hs(hidden: torch.Tensor, residual: torch.Tensor,
                     layer_idx: int, has_residual: int) -> None:
    """Eager ``hs_hook`` body, run from inside the ``hs_probe`` op mid-forward.

    Forms the residual stream and writes the same worker buckets as the eager
    path. ``layer_idx`` is 1-based, matching the config's layer list.
    """
    worker = _ACTIVE_WORKER_HS
    if worker is None:
        return
    module_name = _LAYER_TO_NAME_HS.get(int(layer_idx))
    if module_name is None:
        return
    mr = worker.model_runner

    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)

    # Prefer query_start_loc from attn metadata; fall back to the model_runner's
    # persistent CPU buffers since attn_metadata can be None under torch.compile.
    query_start_loc = None
    if metadata is not None:
        query_start_loc, _ = get_query_metadata(metadata)
    if query_start_loc is None:
        query_start_loc = _cpu_1d(getattr(mr, "query_start_loc", None))

    try:
        req_ids = mr.input_batch.req_ids
    except Exception:
        return
    if not req_ids:
        return
    if query_start_loc is None:
        return

    # Residual stream: hidden + residual for fused-residual layers, else hidden.
    combined = hidden + residual if has_residual else hidden

    bs = len(req_ids)
    last_indices = query_start_loc
    default_mode = getattr(worker, "hs_mode", "last_token")
    default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    if _DEBUG and _capture_dbg["n"] < 6 and combined.shape[0] <= 2048:
        _capture_dbg["n"] += 1
        print(f"[graph/dbg] _capture_body_hs layer={layer_idx} "
              f"h={tuple(combined.shape)} bs={bs}", flush=True)

    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = mr.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_hidden_states") is None:
            continue

        # output_hidden_states: True for all layers, or a 1-based filter list.
        output_layers = extra.get("output_hidden_states")
        if isinstance(output_layers, list) and layer_idx not in output_layers:
            continue

        # Gate on prefill vs decode; mid-forward, output_token_ids reflects this pass.
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hs_mode", default_mode)
        start = int(last_indices[i].item())
        end = int(last_indices[i + 1].item())
        if end <= start:
            continue

        if req_mode == "last_token":
            activation = combined[end - 1].detach().clone()
        else:
            activation = combined[start:end].detach().clone()

        PROF.incr("hook.fire.hs")
        bucket = worker._disk_states if extra.get("save_to_disk") \
            else worker._captured_states
        if req_id not in bucket:
            bucket[req_id] = {}
        layer_states = bucket[req_id]
        if module_name not in layer_states:
            layer_states[module_name] = {
                "hidden_states": [], "layer_num": layer_idx, "hs_mode": req_mode,
            }
        layer_states[module_name]["hidden_states"].append(activation)

        if _DEBUG and _capture_dbg["cap"] < 8:
            _capture_dbg["cap"] += 1
            print(f"[graph/dbg] hs_probe CAPTURED req={str(req_id)[:8]} "
                  f"layer={layer_idx} mode={req_mode} hs={tuple(activation.shape)} "
                  f"bucket={'disk' if extra.get('save_to_disk') else 'rpc'}",
                  flush=True)


def _wrap_layer_class(cls: type) -> None:
    """Class-wrap ``cls.forward`` to call ``hs_probe`` on the layer output.

    Idempotent per class. The probe runs AFTER the original forward (we need its
    output). The layer num is read as a plain install-time int attr (-1 = skip),
    not a per-instance object, to stay trace-safe.
    """
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
            layer_idx = getattr(self, _HS_LAYER_ATTR, -1)
            if layer_idx >= 0 and _GLOBAL_SINK_HS is not None:
                if (isinstance(out, tuple) and len(out) >= 2
                        and isinstance(out[0], torch.Tensor)
                        and isinstance(out[1], torch.Tensor)):
                    torch.ops.vllm_hook.hs_probe(
                        out[0], out[1], _GLOBAL_SINK_HS, layer_idx, 1)
                else:
                    h = out[0] if isinstance(out, tuple) else out
                    if isinstance(h, torch.Tensor):
                        torch.ops.vllm_hook.hs_probe(
                            h, h, _GLOBAL_SINK_HS, layer_idx, 0)
            return out
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_hs_hosts(worker) -> None:
    """Install the Hybrid-mode HS capture path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. On the capture
    rank: tag each layer with its 1-based num, map layer->name, make the sink,
    class-wrap, and register ``hs_probe`` as a splitting op. Capture itself
    happens in the op impl (``_capture_body_hs``).
    """
    global _ACTIVE_WORKER_HS, _GLOBAL_SINK_HS

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_hs] no model on model_runner; skip HS host install")
        return

    register_graph_ops()
    from vllm_hook_plugins.graph.ops import set_hs_capture_fn
    _ACTIVE_WORKER_HS = worker
    set_hs_capture_fn(_capture_body_hs)

    tp_size = worker.parallel_config.tensor_parallel_size
    should_capture = tp_size <= 1 or worker.rank % tp_size == 0

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden_size = int(getattr(text_cfg, "hidden_size"))
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
    # _conf must match what the eager worker writes (get_captured_states/flush_disk).
    worker._conf = {"hidden_size": hidden_size, "num_layers": num_layers}
    worker._should_capture = should_capture
    if not hasattr(worker, "hs_mode"):
        worker.hs_mode = "last_token"

    if not getattr(worker, "_captured_states", None):
        worker._captured_states = {}
    if not getattr(worker, "_disk_states", None):
        worker._disk_states = {}

    if hasattr(worker, "_background_save_loop"):
        init_async_save_thread(worker, worker._background_save_loop, "vllm-hook-hs-io")

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_hs] no decoder layers matched LAYER_PATTERNS; "
              "HS graph capture inactive")
        return

    device_t = next(model.parameters()).device
    n_wired = 0
    if should_capture:
        _GLOBAL_SINK_HS = torch.zeros(1, dtype=torch.int64, device=device_t)
        for name, module, layer_num0 in matched:
            ln = layer_num0 + 1  # 1-based, matching HuggingFace/Eagle
            setattr(module, _HS_LAYER_ATTR, ln)
            _LAYER_TO_NAME_HS[ln] = name
            n_wired += 1
    for _, module, _ in matched:
        _wrap_layer_class(type(module))

    # Register hs_probe as a splitting op (load-bearing eager seam; see install.py).
    try:
        cc = worker.vllm_config.compilation_config
        sops = list(getattr(cc, "splitting_ops", None) or [])
        if "vllm_hook::hs_probe" not in sops:
            sops.append("vllm_hook::hs_probe")
            cc.splitting_ops = sops
        print(f"[graph/install_hs] splitting_ops has hs_probe="
              f"{'vllm_hook::hs_probe' in sops} (n={len(sops)})")
    except Exception as e:  # noqa: BLE001
        print(f"[graph/install_hs] could not set splitting_ops: {e}")

    print(f"[graph/install_hs] op-mode HS capture wired: {n_wired} layer(s); "
          f"should_capture={should_capture}; hidden_size={hidden_size}; "
          f"hooks_on={getattr(worker, '_default_hooks_on', 'prefill')}")


__all__ = ["install_hs_hosts"]
