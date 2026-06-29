"""CUDA-graph hidden-state capture install (Hybrid mode)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.hosts import HSHookHost
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.install import (  # shared helpers (no op-mode coupling)
    _cpu_1d,
    _resolve_max_num_batched_tokens,
    install_prepare_inputs_routing,
)
from vllm_hook_plugins.workers._common import (
    get_query_metadata,
    init_async_save_thread,
    iter_matched_modules,
)
from vllm_hook_plugins.workers.probe_hidden_states_worker import match_layer

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_HS_LAYER_ATTR = "_vllm_hook_hs_layer"   # per-instance 1-based layer num; -1 = skip
_HS_HOST_ATTR = "_vllm_hook_hs_host"     # per-instance HSHookHost (buffer mode)
_GLOBAL_SINK_HS = None                    # int64 (1,) keep-alive against DCE

# Set at install so the op impl can reach the worker and egress keys.
_ACTIVE_WORKER_HS = None
_LAYER_TO_NAME_HS: Dict[int, str] = {}    # 1-based layer num -> egress bucket key
_capture_dbg = {"n": 0, "cap": 0}

_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"

# Capture path, mirroring VLLM_HOOK_QK_CAPTURE:
#   "op"     (default): hs_probe splitting op + impl-runs-mid-forward (PIECEWISE).
#   "buffer": static-buffer + device-routing scatter (capture_hs), NO splitting op,
#             absorbed into the decode cudagraph — the FULL_DECODE_ONLY path.
_CAPTURE_MODE_HS = os.environ.get("VLLM_HOOK_HS_CAPTURE", "op")

# Batched egress (prototype, shared knob with QK): one clone per layer + per-request
# views instead of one clone per (layer, request). See graph/install.py::_egress.
_BATCHED_EGRESS = os.environ.get("VLLM_HOOK_BATCHED_EGRESS") == "1"

# Per-step debug-print budget for the routing/egress wrapper.
_dbg_hs_call_n = [0]
_DBG_HS_MAX_CALLS = 6


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

            # ---- buffer mode: scatter the residual stream into the static host
            # buffer via the capture_hs op (NO splitting op → absorbed into the
            # decode cudagraph). do_capture is an install-time constant, so this
            # adds no data-dependent control flow to the traced region. ----
            if _CAPTURE_MODE_HS == "buffer":
                host = getattr(self, _HS_HOST_ATTR, None)
                if host is not None and host.do_capture:
                    if (isinstance(out, tuple) and len(out) >= 2
                            and isinstance(out[0], torch.Tensor)
                            and isinstance(out[1], torch.Tensor)):
                        torch.ops.vllm_hook.capture_hs(
                            out[0], out[1], host.hs_buf, host.capture_index, 1)
                    else:
                        h = out[0] if isinstance(out, tuple) else out
                        if isinstance(h, torch.Tensor):
                            torch.ops.vllm_hook.capture_hs(
                                h, h, host.hs_buf, host.capture_index, 0)
                return out

            # ---- op mode (default): emit one hs_probe splitting op. ----
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


def install_hs_hosts(worker) -> Optional[HostRegistry]:
    """Install the CUDA-graph HS capture path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. Dispatches by
    ``VLLM_HOOK_HS_CAPTURE``:

    * **op** (default): tag each layer, make the sink, class-wrap, register
      ``hs_probe`` as a splitting op — capture runs in the op impl
      (``_capture_body_hs``) as an eager seam (PIECEWISE). Returns None.
    * **buffer**: build a per-layer ``HSHookHost`` (static ``hs_buf``) + a
      ``HostRegistry`` for routing, class-wrap to emit ``capture_hs`` (NO
      splitting op — absorbed into the decode cudagraph under FULL_DECODE_ONLY).
      Returns the registry; the worker installs the execute_model wrapper.
    """
    global _ACTIVE_WORKER_HS, _GLOBAL_SINK_HS

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_hs] no model on model_runner; skip HS host install")
        return None

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
        worker._graph_registry = None
        return None

    device_t = next(model.parameters()).device

    # ---- buffer mode (FULL-decode milestone): static buffers + routing ----
    if _CAPTURE_MODE_HS == "buffer":
        registry = _install_hs_buffer(
            worker, model, matched, num_layers, hidden_size,
            should_capture, device_t)
        worker._graph_registry = registry
        return registry

    # ---- op mode (default): tag layers, sink, wrap, hs_probe splitting op ----
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

    worker._graph_registry = None
    print(f"[graph/install_hs] op-mode HS capture wired: {n_wired} layer(s); "
          f"should_capture={should_capture}; hidden_size={hidden_size}; "
          f"hooks_on={getattr(worker, '_default_hooks_on', 'prefill')}")
    return None


def _install_hs_buffer(worker, model, matched, num_layers, hidden_size,
                       should_capture, device) -> Optional[HostRegistry]:
    """Build the buffer-mode HS hosts + routing registry (no splitting op).

    One ``HSHookHost`` per matched decoder layer holds a static ``hs_buf``
    ``(cap+1, hidden)``; a shared ``HostRegistry`` owns the per-(layer, token)
    routing slabs the ``capture_hs`` scatter reads on replay. Buffers are built
    here — at load_model, before the cudagraph pool — so their data_ptrs stay
    fixed across replays.
    """
    cap = _resolve_max_num_batched_tokens(worker)
    buf_dtype = model.dtype if hasattr(model, "dtype") \
        else next(model.parameters()).dtype

    registry: Optional[HostRegistry] = None
    if should_capture:
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )

    n_hosts = 0
    for name, module, layer_num0 in matched:
        if should_capture and 0 <= layer_num0 < num_layers:
            host = HSHookHost(
                module_name=name,
                layer_num=layer_num0,            # 0-based registry-slab row
                egress_layer_num=layer_num0 + 1,  # 1-based bucket layer_num
                cap=cap,
                hidden=hidden_size,
                dtype=buf_dtype,
                device=device,
                has_residual=1,                   # refined per-call in the wrap
                do_capture=True,
            )
            setattr(module, _HS_HOST_ATTR, host)
            registry.register_host(host)
            _LAYER_TO_NAME_HS[layer_num0 + 1] = name
            n_hosts += 1
        _wrap_layer_class(type(module))

    if registry is not None:
        registry.assign_views()
        buf_bytes = sum(
            h.hs_buf.numel() * h.hs_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        print(f"[graph/install_hs] HS static buffers: {buf_bytes / (1024**2):.1f} "
              f"MiB on {device} (cap={cap})")

    print(f"[graph/install_hs] buffer-mode HS capture wired: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}; "
          f"hidden_size={hidden_size}; NO splitting op (rides decode cudagraph)")
    return registry


# ---------------------------------------------------------------------------
# Buffer-mode routing + egress + execute_model wrapper (HS analogue of the QK
# path in graph/install.py). All per-request Python lives in the wrapper, OUTSIDE
# the compiled region; the graph only READS the routing buffers we upload here.
# ---------------------------------------------------------------------------


def _any_hs_requested(model_runner) -> bool:
    """True if any request in the batch set output_hidden_states (host-side, no
    device sync) — lets the wrapper short-circuit when nothing wants HS."""
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return False
    requests = getattr(model_runner, "requests", None)
    if not req_ids or requests is None:
        return False
    for req_id in req_ids:
        req_state = requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if extra and extra.get("output_hidden_states") is not None:
            return True
    return False


def _build_routing_hs(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """Fill the registry's pinned mirrors with a destination ROW per captured token,
    mirroring the eager hs_hook gating (output_hidden_states filter, hooks_on).

    Row r = flat_position + 1 (row 0 = discard sentinel); the same r serves every
    requested layer. Layer filter is 1-based (config layers); registry rows are
    0-based, so a 1-based layer ``ln`` maps to row ``ln-1``. Returns one plan per
    active request for egress to reuse.
    """
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1
    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
    any_active_pinned = registry.any_active_pinned          # (num_layers,)
    cap = registry.cap

    dbg_rows: list = []
    plans: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_hidden_states") is None:
            dbg_rows.append(f"r{i}:no_output_hs")
            continue

        # output_hidden_states: True (all layers) | [1-based layer list].
        output_spec = extra.get("output_hidden_states")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on",
                             getattr(model_runner, "_default_hooks_on", "prefill"))
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                dbg_rows.append(f"r{i}:SKIP_prefill_gate")
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hs_mode",
                            getattr(model_runner, "_worker_hs_mode", "last_token"))
        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end <= start:
            continue
        end = min(end, cap)  # over-cap tokens stay in the sentinel
        if end <= start:
            continue

        # Which 0-based registry rows this request wants.
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers))
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
        if not rows_layers:
            continue

        rows = torch.arange(start + 1, end + 1, dtype=torch.int64)
        layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
        capture_index_pinned[layer_idx_t[:, None], start:end] = rows[None, :]
        any_active_pinned[layer_idx_t] = 1

        bucket_key = "_disk_states" if extra.get("save_to_disk") else "_captured_states"
        plans.append({
            "req_idx": i,
            "req_id": req_id,
            "start": start,
            "end": end,
            "layers": set(rows_layers),
            "hs_mode": req_mode,
            "bucket_key": bucket_key,
            "req_state": req_state,
        })
        dbg_rows.append(f"r{i}:qlen={end - start},mode={req_mode},nlayers={len(rows_layers)}")

    if _DEBUG and _dbg_hs_call_n[0] <= _DBG_HS_MAX_CALLS:
        print(f"[graph/dbg-hs]   routing bs={bs} plans={len(plans)} :: "
              + " | ".join(dbg_rows), flush=True)
    return plans


def _egress_hs(worker, registry: HostRegistry, plans: list) -> None:
    """Drain the scattered hs_buf rows into the worker buckets in the exact shape
    the eager hs_hook produces (the reuse contract): bucket[req_id][module_name] =
    {"hidden_states": [...], "layer_num": <1-based>, "hs_mode": mode}.

    last_token: the request's final row; all_tokens: the [start+1, end+1) slice.
    No prefix-K — the residual stream is per-token, so the scattered rows ARE the
    artifact (contrast QK, where k_all reconstructs the cached key history).

    With ``VLLM_HOOK_BATCHED_EGRESS=1`` (Lever A: reduce-then-free) the per-(layer,
    request) clones become ONE ``index_select`` per layer that gathers only the rows
    being saved into a compact own-storage tensor; requests take VIEWS into it —
    O(num_layers) launches instead of O(num_layers * num_requests), same data. Unlike a
    wide [1:W+1] snapshot, the gather's size equals the saved bytes, so a sparse
    last_token view pins only its own row (no W-row snapshot OOM) and the drain guard's
    per-request bytes are exact.
    """
    if not plans:
        return
    dm = getattr(worker, "_capture_drain", None)
    for layer_num, host in registry.iter_hosts():
        hs_buf = host.hs_buf          # (cap+1, hidden)
        module_name = host.module_name
        ln = host.egress_layer_num    # 1-based, matching the eager artifact
        cap = host.cap

        layer_plans = [p for p in plans if layer_num in p["layers"]]
        if not layer_plans:
            continue

        # ---- batched (Lever A: reduce-then-free) — ONE index_select of ONLY the rows
        # we will save, into a compact own-storage tensor per layer, instead of cloning
        # the wide [1:W+1] snapshot and handing out views. A last_token request
        # contributes its single row; all_tokens its [start+1, end+1) span. The gathered
        # tensor's size == the saved bytes, so a sparse last_token view no longer pins a
        # W-row snapshot (the OOM blind spot) and the drain guard's per-request bytes are
        # exact. Still ONE launch per layer (index_select copies into fresh storage, so
        # the views survive buffer reuse, exactly like the old .clone()). The throttle
        # gate stays inline below; under active throttling this gathers a few extra rows
        # (never referenced, freed at end of layer) — bounded by W, no worse than before.
        reduced = None
        red_off = {}  # req_idx -> (lo, hi) into `reduced`; last_token uses reduced[lo]
        if _BATCHED_EGRESS:
            row_idx = []
            for p in layer_plans:
                lo = len(row_idx)
                if p["hs_mode"] == "last_token":
                    row_idx.append(min(p["end"], cap))
                    red_off[p["req_idx"]] = (lo, lo + 1)
                else:
                    row_idx.extend(range(p["start"] + 1, p["end"] + 1))
                    red_off[p["req_idx"]] = (lo, len(row_idx))
            rows_t = torch.tensor(row_idx, dtype=torch.int64, device=hs_buf.device)
            reduced = hs_buf.index_select(0, rows_t)

        for plan in layer_plans:
            start = plan["start"]
            end = plan["end"]
            req_id = plan["req_id"]
            req_mode = plan["hs_mode"]

            # Stage 3 admission throttle (placed ABOVE the clone/note/gauge — HS accounts
            # bytes before its bucket gate, so throttling only at the gate would still
            # clone + double-count). Skip a throttled request uniformly; refuse a NEW
            # request only over the resident ceiling; never throttle an admitted one.
            bucket = getattr(worker, plan["bucket_key"])
            if dm is not None and dm.throttle_enabled:
                if req_id in dm._throttled:
                    continue
                if req_id not in bucket and not dm.admit(req_id, False):
                    PROF.incr("capture.throttled")
                    if _DEBUG and _capture_dbg.get("thr", 0) < 8:
                        _capture_dbg["thr"] = _capture_dbg.get("thr", 0) + 1
                        print(f"[graph/dbg-hs] capture throttled (resident ceiling) "
                              f"req={str(req_id)[:8]} resident={dm.resident_bytes}B "
                              f"ceiling={dm.ceiling_bytes}B", flush=True)
                    continue

            # Rows are start+1 .. end (sentinel-offset). Per-request path clones so we own
            # the data; batched path takes a view into the compact reduced gather.
            if _BATCHED_EGRESS:
                lo, hi = red_off[plan["req_idx"]]
                hs_tok = reduced[lo] if req_mode == "last_token" else reduced[lo:hi]
            elif req_mode == "last_token":
                last_row = min(end, cap)  # row (end-1)+1 == end, bounded by cap
                hs_tok = hs_buf[last_row, :].detach().clone()
            else:
                hs_tok = hs_buf[start + 1:end + 1, :].detach().clone()

            PROF.incr("hook.fire.hs")
            # SAVED bytes = this reduced view/clone. .numel() on a view returns its logical
            # (reduced) element count, so this is byte-identical to the per-request and
            # eager paths regardless of batching.
            nbytes = hs_tok.numel() * hs_tok.element_size()
            PROF.gauge("captured.bytes.hs", nbytes)
            # Both the throttle ceiling (resident_bytes) and the drain trigger (gpu_bytes)
            # accrue the per-request REDUCED bytes (batching-invariant; matches the
            # decrement-on-pop at flush), making gpu_bytes a CURRENT-residency gauge.
            if dm is not None:
                dm.note_resident(nbytes)
                dm.note(nbytes)

            if req_id not in bucket:
                bucket[req_id] = {}
            layer_states = bucket[req_id]
            if module_name not in layer_states:
                layer_states[module_name] = {
                    "hidden_states": [], "layer_num": ln, "hs_mode": req_mode,
                }
            layer_states[module_name]["hidden_states"].append(hs_tok)


def install_execute_model_wrapper_hs(model_runner, worker) -> None:
    """Install buffer-mode HS routing (``_prepare_inputs`` wrapper) + egress
    (``execute_model`` wrapper). Idempotent. Op mode captures inside the forward
    (hs_probe eager seam), so it skips both wrappers.

    Routing runs after vLLM's input prep (so a NEW request's prefill routes
    correctly); egress drains the static hs_buf post-forward using the plans the
    routing wrapper stashed on the registry.
    """
    if getattr(model_runner, "_vllm_hook_hs_wrapped", False):
        return
    if _CAPTURE_MODE_HS != "buffer":
        print("[graph/install_hs] capture mode=op — execute_model wrapper skipped "
              "(capture runs inside the forward)")
        return

    model_runner._vllm_hook_hs_wrapped = True
    model_runner._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
    model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    # Routing — runs after _update_states + input prep, so prefill routes correctly.
    install_prepare_inputs_routing(model_runner, worker, _build_routing_hs, label="hs")

    # W4/W6c: budget-driven GPU->pinned drain manager (no-op unless a budget env is set).
    from vllm_hook_plugins.graph.drain import CaptureDrainManager
    _model = getattr(model_runner, "model", None)
    try:
        _dev = next(_model.parameters()).device if _model is not None else "cpu"
    except Exception:  # noqa: BLE001
        _dev = "cpu"
    worker._capture_drain = CaptureDrainManager.from_env(_dev)
    if worker._capture_drain.enabled:
        print(f"[graph/install_hs] capture drain enabled: budget="
              f"{worker._capture_drain.budget_bytes} bytes")
    if _BATCHED_EGRESS:
        print("[graph/install_hs] batched egress ON (1 clone/layer + per-request views)")

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        _dbg_hs_call_n[0] += 1
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        model_runner._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

        from vllm_hook_plugins.graph.ops import get_fire_count
        fire_before = get_fire_count()

        result = orig_execute_model(scheduler_output, *args, **kwargs)

        fire_after = get_fire_count()

        plans = getattr(registry, "_pending_plans", None) or []
        if plans:
            with PROF.timed("graph.egress"):
                _egress_hs(worker, registry, plans)
            # W4: drain GPU-resident clones to pinned host if over the byte budget.
            dm = getattr(worker, "_capture_drain", None)
            if dm is not None:
                dm.maybe_drain(worker)
        registry._pending_plans = []  # consume

        if _DEBUG and (plans or _dbg_hs_call_n[0] <= _DBG_HS_MAX_CALLS):
            n_disk = sum(len(v) for v in getattr(worker, "_disk_states", {}).values())
            n_rpc = sum(len(v) for v in getattr(worker, "_captured_states", {}).values())
            print(f"[graph/dbg-hs] call#{_dbg_hs_call_n[0]}: plans={len(plans)} "
                  f"op_fired={fire_after - fire_before} (total={fire_after}) "
                  f"disk={n_disk} rpc={n_rpc}", flush=True)

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install_hs] execute_model wrapper installed (HS egress)")


__all__ = ["install_hs_hosts", "install_execute_model_wrapper_hs"]
