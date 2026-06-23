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
from typing import Any, Dict, Optional

import torch

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.hosts import SteerHost
from vllm_hook_plugins.graph.install import (  # shared helpers
    _cpu_1d,
    _resolve_max_num_batched_tokens,
    install_prepare_inputs_routing,
)
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
_STEER_HOST_ATTR = "_vllm_hook_steer_host"   # per-instance SteerHost (buffer mode)
_GLOBAL_SINK_STEER = None                 # int64 (1,) keep-alive against DCE

# Set at install so the op impl can reach the worker (vector cache, env config).
_ACTIVE_WORKER_STEER = None
_steer_dbg = {"n": 0, "fire": 0}

_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"

# Steering path, mirroring VLLM_HOOK_QK_CAPTURE / VLLM_HOOK_HS_CAPTURE:
#   "op"     (default): steer_residual splitting op (PIECEWISE eager seam).
#   "buffer": steer_buffer masked in-place add, NO splitting op, absorbed into the
#             decode cudagraph — the FULL_DECODE_ONLY path (add_vector only).
_STEER_MODE = os.environ.get("VLLM_HOOK_STEER_MODE", "op")

_dbg_steer_call_n = [0]
_DBG_STEER_MAX_CALLS = 8


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

            # ---- buffer mode: masked in-place steer_buffer add (NO splitting op →
            # absorbed into the decode cudagraph). do_steer is an install-time
            # constant, so this adds no data-dependent control flow. ----
            if _STEER_MODE == "buffer":
                host = getattr(self, _STEER_HOST_ATTR, None)
                if host is not None and host.do_steer:
                    if (isinstance(out, tuple) and len(out) >= 2
                            and isinstance(out[1], torch.Tensor)):
                        host.steer(out[1])      # mutate the residual in place
                    elif isinstance(out, torch.Tensor):
                        host.steer(out)
                return out

            # ---- op mode (default): steer_residual splitting op. ----
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

    # ---- buffer mode (FULL-decode milestone): steer_buffer masked add ----
    if _STEER_MODE == "buffer":
        _install_steer_buffer(worker, model, matched, device_t)
        return

    # ---- op mode (default): steer_residual splitting op ----
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


# ---------------------------------------------------------------------------
# Buffer mode (FULL decode): SteerRegistry + masked steer_buffer add
# ---------------------------------------------------------------------------


class SteerRegistry:
    """Per-worker owner of the steering routing slabs + the resident vector table.

    Mirrors ``HostRegistry`` (so it plugs into ``install_prepare_inputs_routing``),
    but its slabs are the steering data plane: per-(layer, token) ``coeff_all`` float,
    ``vec_id_all`` int (which vector), and ``mode_all`` int (0=add_vector, 1=adjust_rs).
    ``vec_table (V_max, hidden)`` is the LoRA-style vector library and ``avg_proj_table
    (V_max,)`` holds each adjust_rs vector's target projection; both are pre-allocated so
    their ``data_ptr`` is fixed across replays (rows written in place). All ranks steer,
    so ``should_capture`` is always True.
    """

    def __init__(self, num_layers, cap, hidden, v_max, device, dtype):
        self.num_layers = int(num_layers)
        self.cap = int(cap)
        self.hidden = int(hidden)
        self.v_max = int(v_max)
        self.device = torch.device(device)
        self.should_capture = True  # steering is applied on EVERY TP rank

        self.coeff_all = torch.zeros(num_layers, cap, dtype=torch.float32, device=device)
        self.vec_id_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        self.mode_all = torch.zeros(num_layers, cap, dtype=torch.int64, device=device)
        # The resident vector library + per-vector adjust_rs target (fixed data_ptr).
        self.vec_table = torch.zeros(v_max, hidden, dtype=dtype, device=device)
        self.avg_proj_table = torch.zeros(v_max, dtype=torch.float32, device=device)

        pin = self.device.type == "cuda"
        self.coeff_pinned = torch.zeros(num_layers, cap, dtype=torch.float32, pin_memory=pin)
        self.vec_id_pinned = torch.zeros(num_layers, cap, dtype=torch.int64, pin_memory=pin)
        self.mode_pinned = torch.zeros(num_layers, cap, dtype=torch.int64, pin_memory=pin)
        self._upload_event = torch.cuda.Event() if self.device.type == "cuda" else None

        self.hosts: Dict[int, SteerHost] = {}
        self.vec_paths: Dict[str, int] = {}  # vector_path -> row in vec_table
        self.vec_has_avgproj: set = set()    # vids whose avg_proj_table row is loaded
        self._next_vec_id = 0
        self._pending_plans: list = []

    def register_host(self, host: SteerHost) -> None:
        self.hosts[host.layer_num] = host

    def assign_views(self) -> None:
        for layer_num, host in self.hosts.items():
            host.bind_views(self.coeff_all[layer_num], self.vec_id_all[layer_num],
                            self.mode_all[layer_num], self.vec_table,
                            self.avg_proj_table)

    def begin_step(self) -> None:
        pass  # steering keeps no forward-context stash

    def reset_pinned(self, width: Optional[int] = None) -> None:
        if self._upload_event is not None:
            self._upload_event.synchronize()
        if width is None:
            self.coeff_pinned.zero_()
            self.vec_id_pinned.zero_()
            self.mode_pinned.zero_()
        else:
            w = max(1, min(int(width), self.cap))
            self.coeff_pinned[:, :w].zero_()
            self.vec_id_pinned[:, :w].zero_()
            self.mode_pinned[:, :w].zero_()

    def upload(self, width: Optional[int] = None) -> None:
        # Only the [:, :width] column prefix the cudagraph-padded forward reads (see
        # install._upload_width); the stale tail beyond width is never read by the op.
        if width is None:
            self.coeff_all.copy_(self.coeff_pinned, non_blocking=True)
            self.vec_id_all.copy_(self.vec_id_pinned, non_blocking=True)
            self.mode_all.copy_(self.mode_pinned, non_blocking=True)
        else:
            w = max(1, min(int(width), self.cap))
            self.coeff_all[:, :w].copy_(self.coeff_pinned[:, :w], non_blocking=True)
            self.vec_id_all[:, :w].copy_(self.vec_id_pinned[:, :w], non_blocking=True)
            self.mode_all[:, :w].copy_(self.mode_pinned[:, :w], non_blocking=True)
        if self._upload_event is not None:
            self._upload_event.record()

    def vec_id_for_path(self, path: str, worker) -> Optional[int]:
        """Resolve ``vector_path`` to a vec_table row, loading + caching on first use.

        The vector's ``dir`` is written into a PRE-ALLOCATED ``vec_table`` row in place
        (data_ptr fixed), and its ``avg_proj`` (adjust_rs vectors carry it) into the
        matching ``avg_proj_table`` row, so a never-before-seen vector is admitted
        without moving either table — graph safe. Returns None if the path can't be
        loaded or the table is full.
        """
        vid = self.vec_paths.get(path)
        if vid is not None:
            return vid
        cache = getattr(worker, "_vector_cache", None) if worker is not None else None
        if cache is None and worker is not None:
            worker._vector_cache = cache = {}
        data = cache.get(path) if cache is not None else None
        if data is None:
            try:
                raw = _load_steering_vector(path)
            except Exception:  # noqa: BLE001
                return None
            d = raw["dir"]
            data = {"dir": d if torch.is_tensor(d) else torch.tensor(d)}
            if "avg_proj" in raw:  # adjust_rs vectors carry the target projection
                ap = raw["avg_proj"]
                data["avg_proj"] = float(ap.item()) if torch.is_tensor(ap) else float(ap)
            if cache is not None:
                cache[path] = data
        if self._next_vec_id >= self.v_max:
            return None  # table full
        vid = self._next_vec_id
        self._next_vec_id += 1
        self.vec_table[vid].copy_(
            data["dir"].to(self.vec_table.device, dtype=self.vec_table.dtype).view(-1))
        if "avg_proj" in data:
            self.avg_proj_table[vid] = float(data["avg_proj"])
            self.vec_has_avgproj.add(vid)
        self.vec_paths[path] = vid
        return vid


def _install_steer_buffer(worker, model, matched, device) -> SteerRegistry:
    """Build the buffer-mode steering hosts + routing registry (no splitting op).

    One ``SteerHost`` per matched decoder layer; a ``SteerRegistry`` owns the
    per-(layer, token) coeff/vec_id routing slabs and the resident vector table.
    Routing runs in the ``_prepare_inputs`` wrapper (per-request config → coeff/vec
    buffers); the layer wrap applies ``steer_buffer`` (masked in-place add). There is
    NO egress — steering produces no artifacts, only the residual mutation.
    """
    global _GLOBAL_SINK_STEER
    _GLOBAL_SINK_STEER = None  # buffer mode doesn't use the op-mode sink

    cap = _resolve_max_num_batched_tokens(worker)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = int(getattr(text_cfg, "hidden_size"))
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0)) or (
        max(ln for _, _, ln in matched) + 1)
    v_max = int(os.environ.get("VLLM_HOOK_STEER_VMAX", "16"))
    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    registry = SteerRegistry(num_layers, cap, hidden, v_max, device, buf_dtype)

    n_wired = 0
    for name, module, layer_num0 in matched:
        if 0 <= layer_num0 < num_layers:
            host = SteerHost(module_name=name, layer_num=layer_num0, cap=cap,
                             do_steer=True)
            setattr(module, _STEER_HOST_ATTR, host)
            registry.register_host(host)
            n_wired += 1
        _wrap_layer_class(type(module))
    registry.assign_views()

    worker._graph_registry = registry
    # Routing only — steering has no egress, so no execute_model wrapper is needed.
    install_prepare_inputs_routing(worker.model_runner, worker, _build_routing_steer,
                                   label="steer")
    print(f"[graph/install_steer] buffer-mode steering wired: {n_wired} layer(s) over "
          f"{num_layers} slots; cap={cap} hidden={hidden} V_max={v_max}; "
          f"NO splitting op (rides decode cudagraph)")
    return registry


def _build_routing_steer(model_runner, registry: SteerRegistry, qsl_cpu: list) -> list:
    """Fill the pinned coeff/vec_id/mode mirrors from each request's steer config.

    For every request whose resolved config targets layer L, write the vector's
    ``vec_id`` and the per-token ``mode`` across the request's token span at row L:
      * ``add_vector`` → ``mode=0`` + ``coeff=coefficient`` (host-supplied delta).
      * ``adjust_rs``  → ``mode=1`` + ``coeff=0`` (the op computes the coefficient in
        kernel from the live residual; the vector must carry ``avg_proj``).
    Unsteered (layer, token) entries stay 0 (an exact no-op). Uses flat positions
    (steering modifies the current step's residual; no accumulation).
    """
    if qsl_cpu is None:
        return []
    worker = _ACTIVE_WORKER_STEER
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1
    coeff_pinned = registry.coeff_pinned
    vecid_pinned = registry.vec_id_pinned
    mode_pinned = registry.mode_pinned
    cap = registry.cap
    env_path = getattr(worker, "_env_config_path", None) if worker else None

    plans: list = []
    dbg: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        steer_arg = (req_state.sampling_params.extra_args or {}).get("steer")
        cfg = _resolve_steer_config(steer_arg, env_path)
        if cfg is None:
            continue
        layer = int(cfg.get("optimal_layer", -1))
        if not (0 <= layer < registry.num_layers):
            continue
        method = cfg.get("method", "adjust_rs")
        if method not in ("add_vector", "adjust_rs"):
            dbg.append(f"r{i}:method={method}_unsupported")
            continue
        vector_path = cfg.get("vector_path")
        if not vector_path:
            continue
        vid = registry.vec_id_for_path(vector_path, worker)
        if vid is None:
            dbg.append(f"r{i}:vec_load_failed")
            continue
        if method == "adjust_rs" and vid not in registry.vec_has_avgproj:
            # adjust_rs needs the vector's avg_proj target; without it the in-kernel
            # coefficient would be wrong, so skip rather than steer incorrectly.
            dbg.append(f"r{i}:adjust_rs_no_avg_proj")
            continue
        mode_val = 1 if method == "adjust_rs" else 0
        coefficient = 0.0 if method == "adjust_rs" else float(cfg.get("coefficient", 0.0))

        start = int(qsl_cpu[i])
        end = min(int(qsl_cpu[i + 1]), cap)
        if end <= start:
            continue
        coeff_pinned[layer, start:end] = coefficient
        vecid_pinned[layer, start:end] = vid
        mode_pinned[layer, start:end] = mode_val
        plans.append({"req_idx": i, "layer": layer, "method": method, "vid": vid})
        dbg.append(f"r{i}:L{layer},{method},coeff={coefficient},vid={vid},rows={start}:{end}")

    if _DEBUG and _dbg_steer_call_n[0] <= _DBG_STEER_MAX_CALLS:
        _dbg_steer_call_n[0] += 1
        print(f"[graph/dbg-steer] routing bs={bs} steered={len(plans)} :: "
              + " | ".join(dbg), flush=True)
    return plans


__all__ = ["install_steer_hosts"]
