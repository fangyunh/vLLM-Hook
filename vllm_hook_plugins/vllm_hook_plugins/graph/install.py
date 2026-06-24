"""CUDA-graph QK capture — install, per-step routing, and egress."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.hosts import QKHookHost
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.workers._common import (
    get_query_metadata,
    init_async_save_thread,
    iter_matched_modules,
)
from vllm_hook_plugins.workers.probe_hookqk_worker import (
    _read_cached_keys,
    key_cache_from_layer_kv,
    match_attn,
)


# ---------------------------------------------------------------------------
# Process-wide graph-mode flag
# ---------------------------------------------------------------------------
# Mirrored in an env var because create_engine_config runs in the driver but
# load_model runs in the (possibly spawned) worker, where the module global would
# not survive — the env var, set before workers launch, does.
_GRAPH_MODE_ENV = "VLLM_HOOK_GRAPH_MODE"
_graph_mode_enabled = False


def set_graph_mode(enabled: bool) -> None:
    """Arm/disarm the graph path. Sets both the module global and the env var so
    spawned workers (where load_model runs) inherit the decision."""
    global _graph_mode_enabled
    _graph_mode_enabled = bool(enabled)
    os.environ[_GRAPH_MODE_ENV] = "1" if enabled else "0"


def graph_mode_enabled() -> bool:
    """True if the graph path should install: module global (driver) or inherited
    env var (worker subprocess)."""
    return _graph_mode_enabled or os.environ.get(_GRAPH_MODE_ENV) == "1"


# ---------------------------------------------------------------------------
# Class-level Attention.forward wrap (survives the Dynamo instance-hook bypass)
# ---------------------------------------------------------------------------
# Per-class originals → wrap is idempotent and reversible. One wrapped class
# serves every layer; an instance with no host/layer attr falls straight through.
_WRAPPED_ATTN_CLASSES: Dict[type, Any] = {}
# Seam mode: backend-impl classes whose .forward we wrapped (e.g. FlashAttentionImpl).
# Wrapping the impl (not the Attention module) puts capture INSIDE the opaque
# unified_attention[_with_output] op — the eager seam vLLM already pays for — so no
# extra splitting op / FX split point is added. Per-class originals → idempotent.
_WRAPPED_IMPL_CLASSES: Dict[type, Any] = {}
_HOST_ATTR = "_vllm_hook_qk_host"          # per-instance QKHookHost (buffer mode)
_REG_ATTR = "_vllm_hook_qk_registry"       # per-instance HostRegistry back-ref
# Op mode: a plain int layer-index attr + a module-global keep-alive sink. This
# shape (closed-over global + op call) survives compilation and avoids reading a
# per-instance object inside the traced forward (a graph-break risk).
_LAYER_ATTR = "_vllm_hook_qk_layer"
_GLOBAL_SINK = None  # int64 (1,) keep-alive for qk_probe; set in install

# Prefix-K forward-context stash inside the traced forward. Off by default: the
# get_forward_context() call there is novel and may graph-break, and it's only
# needed for prefix-K (num_cached > 0). Enable once core capture is confirmed.
_PREFIXK_STASH = os.environ.get("VLLM_HOOK_QK_PREFIXK_STASH") == "1"

# Capture mechanism. "op" (default): qk_probe op runs the eager capture body
# mid-forward — works for prefill under PIECEWISE (forward not replayed) and
# sidesteps the input_batch staleness that made the buffer path capture nothing.
# "buffer": static-buffer + device-routing, for the decode-under-cudagraph milestone.
# "seam" (prototype): capture inside the attention backend impl's eager seam, sharing
# the unified_attention split — NO extra splitting op. See docs/qk_seam_capture_prototype.md.
_CAPTURE_MODE = os.environ.get("VLLM_HOOK_QK_CAPTURE", "op")

# Set at install for the op-mode capture body.
_ACTIVE_WORKER = None
_LAYER_TO_NAME: Dict[int, str] = {}
_capture_dbg = {"n": 0}  # one-shot capture confirmation counter
_ctx_dumped = [False]    # one-shot forward-context exploration dump

# Diagnostic logging for the hot path. ON during bring-up; VLLM_HOOK_QK_DEBUG=0
# to mute. Compact, flushed, capped to the first few calls.
_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"
_DBG_MAX_CALLS = int(os.environ.get("VLLM_HOOK_QK_DEBUG_MAX", "8"))
_dbg_call_n = [0]

# _NO_PREFIXK skips prefix-K (bisection / no prefix caching); _PREFIXK_SYNC forces
# a CUDA sync after the gather so an async device assert surfaces at our line
# instead of a later replay (debug localizer only).
_NO_PREFIXK = os.environ.get("VLLM_HOOK_QK_NO_PREFIXK") == "1"
_PREFIXK_SYNC = os.environ.get("VLLM_HOOK_QK_PREFIXK_SYNC") == "1"


def _dbg_prefixk(meta_kv, module_name, req_idx, num_cached, total_len, layer_idx):
    """Log the prefix-K gather inputs (block_table, computed block_ids min/max,
    key_cache geometry) so an OOB block index — the device-assert suspect — is
    diagnosable."""
    try:
        bt = getattr(meta_kv, "block_table", None)
        bt_shape = tuple(bt.shape) if isinstance(bt, torch.Tensor) else type(bt).__name__
        bt_dtype = str(bt.dtype) if isinstance(bt, torch.Tensor) else "?"
        info = (f"[graph/dbg] prefixK layer={layer_idx} req_idx={req_idx} "
                f"num_cached={num_cached} total_len={total_len} "
                f"block_table={bt_shape}/{bt_dtype}")
        if isinstance(bt, torch.Tensor):
            import math as _m
            ctx = get_forward_context()
            kc = key_cache_from_layer_kv(ctx.no_compile_layers[module_name].kv_cache)
            block_size = kc.shape[1]
            nbn = _m.ceil(total_len / block_size)
            row = bt[req_idx, :nbn] if bt.dim() == 2 else bt[:nbn]
            info += (f" key_cache={tuple(kc.shape)} block_size={block_size} "
                     f"num_blocks_needed={nbn} num_blocks_total={kc.shape[0]} "
                     f"block_ids={row.tolist()} "
                     f"min={int(row.min())} max={int(row.max())}")
        print(info, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[graph/dbg] prefixK dbg failed: {e!r}", flush=True)


def _dbg(msg: str) -> None:
    if _DEBUG and _dbg_call_n[0] <= _DBG_MAX_CALLS:
        print(f"[graph/dbg] {msg}", flush=True)


def _cpu_1d(x):
    """Coerce a query_start_loc/seq_lens carrier (CpuGpuBuffer-like or plain
    tensor) to a 1-D CPU tensor, preferring the CPU mirror to avoid a device sync.
    None if nothing usable."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    # CpuGpuBuffer: .cpu is a CPU tensor attribute, NOT the Tensor.cpu method.
    for attr in ("cpu", "np", "gpu"):
        v = getattr(x, attr, None)
        if v is None or callable(v):
            continue
        if isinstance(v, torch.Tensor):
            return v.detach().to("cpu")
        try:
            return torch.as_tensor(v)
        except Exception:  # noqa: BLE001
            continue
    return None


def _capture_body(q_in: torch.Tensor, k_in: torch.Tensor, layer_idx: int) -> None:
    """Run the eager qkv_hook capture body inside the qk_probe op (op mode).

    Invoked by ``graph/ops._qk_probe_impl`` mid-forward, where get_forward_context()
    and input_batch are current — the same point the eager hook ran. Does the
    identical per-request slicing / hooks_on / hookq_mode / prefix-K as the eager
    ``qkv_hook`` and writes the SAME worker buckets. q_in/k_in are post-RoPE q/k
    for the whole batch (num_tokens, dim); layer_idx is the host's layer_num.
    """
    worker = _ACTIVE_WORKER
    if worker is None:
        return
    module_name = _LAYER_TO_NAME.get(int(layer_idx))
    if module_name is None:
        return
    mr = worker.model_runner

    # Gate diagnostics on real (small) batches — warmup uses a huge shape with no
    # metadata and would flood the log budget.
    _small = q_in.shape[0] <= 2048

    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)

    # Under PIECEWISE ctx.attn_metadata is None, so fall back to the model_runner's
    # persistent buffers, which ARE current mid-forward (built in _prepare_inputs).
    query_start_loc = None
    seq_lens = None
    if metadata is not None:
        query_start_loc, seq_lens = get_query_metadata(metadata)
    if query_start_loc is None:
        query_start_loc = _cpu_1d(getattr(mr, "query_start_loc", None))
    if seq_lens is None:
        seq_lens = _cpu_1d(getattr(mr, "seq_lens", None))

    try:
        req_ids = mr.input_batch.req_ids
    except Exception:
        return

    # Skip dummy/profiling forwards (no real requests) — vLLM runs many at
    # warmup/capture and they would exhaust the debug budget.
    if not req_ids:
        return

    if query_start_loc is None:
        if _DEBUG and _capture_dbg["n"] < 12:
            _capture_dbg["n"] += 1
            print(f"[graph/dbg] _capture_body layer={layer_idx} q={tuple(q_in.shape)}"
                  f" reqs={len(req_ids)} EXIT: query_start_loc is None", flush=True)
        return

    # bs = request count. query_start_loc may be a padded persistent buffer, so use
    # len(req_ids), not len(query_start_loc)-1; valid offsets are qsl[:bs+1].
    bs = len(req_ids)
    if _DEBUG and _small and _capture_dbg["n"] < 10:
        _capture_dbg["n"] += 1
        try:
            _qsl_head = [int(query_start_loc[j]) for j in range(min(bs + 1, len(query_start_loc)))]
        except Exception:  # noqa: BLE001
            _qsl_head = "?"
        print(f"[graph/dbg] _capture_body layer={layer_idx} q={tuple(q_in.shape)} "
              f"RUNS: bs={bs} qsl[:bs+1]={_qsl_head} meta={'ctx' if metadata is not None else 'mr'}",
              flush=True)
    last_indices = query_start_loc
    default_mode = getattr(worker, "hookq_mode", "all_tokens")
    default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = mr.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_qk") is None:
            continue

        # output_qk layer filter: True (all) | [layers] | {layer: heads}.
        output_spec = extra.get("output_qk")
        if isinstance(output_spec, dict):
            if layer_idx not in {int(k) for k in output_spec.keys()}:
                continue
        elif isinstance(output_spec, list):
            if layer_idx not in output_spec:
                continue

        # hooks_on gating; output_token_ids==[] detects prefill. Correct here
        # (mid-forward) — the request state reflects the CURRENT step.
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hookq_mode", default_mode)
        start = int(last_indices[i].item())
        end = int(last_indices[i + 1].item())

        if req_mode == "all_tokens":
            q_tok = q_in[start:end, :].detach().clone()
        else:
            q_tok = q_in[end - 1, :].detach().clone()
        k_tok = k_in[start:end, :].detach().clone()

        # Prefix-K reconstruction (identical to eager qkv_hook); only fires when
        # prefix caching trimmed the prompt (num_cached > 0).
        if seq_lens is not None and metadata is not None and not _NO_PREFIXK:
            try:
                sv = seq_lens[i]
                total_len = int(sv.item()) if hasattr(sv, "item") else int(sv)
                query_len = end - start
                num_cached = total_len - query_len
                if num_cached > 0:
                    PROF.incr("kv.prefix_recon")
                    meta_kv = metadata if not isinstance(metadata, dict) \
                        else next(iter(metadata.values()))
                    if _DEBUG and _capture_dbg.get("pk", 0) < 16:
                        _capture_dbg["pk"] = _capture_dbg.get("pk", 0) + 1
                        _dbg_prefixk(meta_kv, module_name, i, num_cached,
                                     total_len, layer_idx)
                    with PROF.timed("kv.prefix_recon"):
                        prefix_k = _read_cached_keys(
                            module_name, meta_kv, i, num_cached, total_len)
                    # Debug localizer: surface an async device assert from the
                    # gather HERE. Legal — the op is an eager seam, not a replay.
                    if _PREFIXK_SYNC:
                        torch.cuda.synchronize()
                    if prefix_k is not None:
                        k_tok = torch.cat(
                            [prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok],
                            dim=0)
            except Exception as _pk_e:  # noqa: BLE001
                PROF.incr("kv.prefix_recon.errors")
                if _DEBUG and _capture_dbg.get("pke", 0) < 8:
                    _capture_dbg["pke"] = _capture_dbg.get("pke", 0) + 1
                    print(f"[graph/dbg] prefix_recon EXC layer={layer_idx} "
                          f"i={i}: {_pk_e!r}", flush=True)

        PROF.incr("hook.fire.qk")
        bucket = worker._disk_states if extra.get("save_to_disk") \
            else worker._captured_states
        if req_id not in bucket:
            bucket[req_id] = {}
        layer_states = bucket[req_id]
        if module_name not in layer_states:
            layer_states[module_name] = {
                "q": [], "k_all": [], "layer_num": layer_idx,
                "hookq_mode": req_mode,
            }
        layer_states[module_name]["q"].append(q_tok)
        layer_states[module_name]["k_all"].append(k_tok)

        if _DEBUG and _capture_dbg.get("cap", 0) < 8:
            _capture_dbg["cap"] = _capture_dbg.get("cap", 0) + 1
            print(f"[graph/dbg] qk_probe CAPTURED req={str(req_id)[:8]} "
                  f"layer={layer_idx} mode={req_mode} q={tuple(q_tok.shape)} "
                  f"k={tuple(k_tok.shape)} bucket={'disk' if extra.get('save_to_disk') else 'rpc'}",
                  flush=True)


def _wrap_attn_class(cls: type) -> None:
    """Class-wrap ``cls.forward``, idempotent per class. Op mode emits one qk_probe
    op; buffer mode calls ``host.capture(q,k)``. args[0]=post-RoPE q, args[1]=k
    (the eager hook's input[0]/input[1]); capture runs before the original forward.

    Buffer mode also snapshots the live forward context onto the registry on its
    first fire each step — the only place it's provably live — so egress can read
    kv_cache/attn_metadata for prefix-K post-forward.
    """
    if cls in _WRAPPED_ATTN_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_ATTN_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            # Op mode (default): read the int layer attr (-1 = don't capture) and
            # emit one qk_probe op against the global sink. No per-instance object
            # touched inside the traced forward.
            if _CAPTURE_MODE == "op":
                layer_idx = getattr(self, _LAYER_ATTR, -1)
                if layer_idx >= 0 and len(args) >= 2:
                    torch.ops.vllm_hook.qk_probe(
                        args[0], args[1], _GLOBAL_SINK, layer_idx)
                return orig_fwd(self, *args, **kwargs)

            # ---- buffer mode (decode milestone): static-buffer scatter ----
            # do_capture is an install-time constant, so this branch adds no
            # data-dependent control flow to the traced region.
            host = getattr(self, _HOST_ATTR, None)
            if host is not None and host.do_capture:
                # Stash the live forward context for post-forward prefix-K egress
                # (only the first attn layer of the step takes effect; bare-except
                # so a torn-down context never breaks the forward). Gated off:
                # get_forward_context() in traced Python may graph-break, and it's
                # only needed for prefix-K. If it breaks compile, grab kv_cache refs
                # at install time instead.
                if _PREFIXK_STASH:
                    reg = getattr(self, _REG_ATTR, None)
                    if reg is not None and reg.fwd_ctx is None:
                        try:
                            ctx = get_forward_context()
                            reg.stash_forward_context(
                                ctx, getattr(ctx, "attn_metadata", None)
                            )
                        except Exception:  # noqa: BLE001
                            pass
                # args[0]=post-RoPE q, args[1]=k (positional in every Attention
                # signature; guard length defensively).
                if len(args) >= 2:
                    host.capture(args[0], args[1])
            return orig_fwd(self, *args, **kwargs)

        return wrapped

    cls.forward = make_wrapped(orig_forward)


# ---------------------------------------------------------------------------
# Seam mode (prototype): capture inside the attention backend impl's eager seam
# ---------------------------------------------------------------------------


def _seam_capture(q: torch.Tensor, k: torch.Tensor, layer_idx: int) -> None:
    """Run the shared capture body from inside the attention impl's eager seam.

    Called by the ``type(module.impl).forward`` wrap (``_wrap_impl_class``), which
    executes inside the opaque ``unified_attention[_with_output]`` op — untraced and
    re-run eagerly every step (prefill AND decode replay). q/k arrive in kernel layout
    ``[num_tokens, heads, head_dim]``; reshape to the flat ``[num_tokens, heads*head_dim]``
    the eager artifact / ``_capture_body`` expects (a pure view-flatten, identical data).

    Bails while ``is_current_stream_capturing`` (warmup/capture forwards), mirroring the
    qk_probe op. Best-effort: a capture bug must never crash attention.
    """
    capturing = False
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        pass
    if capturing:
        return
    try:
        q2d = q.reshape(q.shape[0], -1) if q.dim() > 2 else q
        k2d = k.reshape(k.shape[0], -1) if k.dim() > 2 else k
        _capture_body(q2d, k2d, int(layer_idx))
    except Exception as e:  # noqa: BLE001
        n = _capture_dbg.get("seam_err", 0)
        if _DEBUG and n < 8:
            _capture_dbg["seam_err"] = n + 1
            print(f"[graph/dbg] seam capture raised: {e!r}", flush=True)


def _wrap_impl_class(cls: type) -> None:
    """Class-wrap an attention backend impl's ``forward`` (e.g. FlashAttentionImpl).

    Idempotent per class. The impl's forward is invoked as
    ``impl.forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)`` from
    inside the opaque attention op, so in the wrap: ``args[0]=layer`` (carries _LAYER_ATTR),
    ``args[1]=query``, ``args[2]=key``. Capture runs BEFORE the real attention reads q/k;
    it only clones, never mutates. An impl with no tagged layer (layer_idx<0) falls straight
    through, so wrapping the class is harmless on non-capture ranks.
    """
    if cls in _WRAPPED_IMPL_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_IMPL_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            # args[0]=layer, args[1]=query, args[2]=key (positional in every v1 impl
            # forward; guard length defensively for signature drift across backends).
            if len(args) >= 3:
                layer = args[0]
                layer_idx = getattr(layer, _LAYER_ATTR, -1)
                if layer_idx >= 0:
                    _seam_capture(args[1], args[2], layer_idx)
            return orig_fwd(self, *args, **kwargs)

        return wrapped

    cls.forward = make_wrapped(orig_forward)


def _resolve_max_num_batched_tokens(worker) -> int:
    """Return max_num_batched_tokens for the buffer cap, version-robustly: probe
    model_runner.scheduler_config then worker.vllm_config.scheduler_config. Raises
    if neither carrier has it."""
    for owner in (worker.model_runner, worker):
        sched = getattr(owner, "scheduler_config", None)
        if sched is not None:
            mnbt = getattr(sched, "max_num_batched_tokens", None)
            if mnbt is not None:
                return int(mnbt)
    vc = getattr(worker, "vllm_config", None)
    if vc is not None:
        sched = getattr(vc, "scheduler_config", None)
        if sched is not None:
            mnbt = getattr(sched, "max_num_batched_tokens", None)
            if mnbt is not None:
                return int(mnbt)
    raise RuntimeError(
        "could not resolve max_num_batched_tokens for QK buffer CAP"
    )


# ---------------------------------------------------------------------------
# Host install (capture rank only) — runs at load_model, before compile/capture
# ---------------------------------------------------------------------------


def install_qk_hosts(worker) -> Optional[HostRegistry]:
    """Class-wrap Attention.forward (and, in buffer mode, build per-layer hosts).

    Runs at load_model, after the model is built but BEFORE compile/capture, so
    buffers land before the cudagraph pool and their data_ptrs stay fixed across
    replays. On a non-capture rank no buffers/hosts are built; the class-wrap still
    installs (harmless — the per-instance check short-circuits). Returns the
    HostRegistry (buffer mode) or None.
    """
    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install] no model on model_runner; skip QK host install")
        return None

    # Register the capture op(s) once per process, BEFORE any wrap can fire.
    register_graph_ops()

    # Op + seam modes: wire the SAME capture body + the worker it writes into.
    # (Seam mode calls _capture_body directly via the impl wrap; set_capture_fn is
    # harmless there — the qk_probe op it feeds is never emitted in seam mode.)
    if _CAPTURE_MODE in ("op", "seam"):
        from vllm_hook_plugins.graph.ops import set_capture_fn
        global _ACTIVE_WORKER
        _ACTIVE_WORKER = worker
        set_capture_fn(_capture_body)

    # Only TP rank 0 captures — residual streams are replicated across ranks.
    tp_size = worker.parallel_config.tensor_parallel_size
    should_capture = tp_size <= 1 or worker.rank % tp_size == 0

    # Model dims pulled EXACTLY like the eager worker so buffer widths and the
    # analyzer config match.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    num_h = int(getattr(text_cfg, "num_attention_heads"))
    num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
    hidden = int(getattr(text_cfg, "hidden_size"))
    # _conf.head_dim must match the eager worker (analyzer reads _conf): hidden//num_h.
    # Buffer sizing honours an explicit config.head_dim when present, because the real
    # post-RoPE q width is num_h * actual_head_dim; a mismatch would mis-size buffers.
    head_dim = hidden // num_h                       # for _conf (eager parity)
    buf_head_dim = int(getattr(text_cfg, "head_dim", None) or head_dim)  # for buffers
    attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
    q_dim = num_h * buf_head_dim
    k_dim = num_kv * buf_head_dim

    # _conf feeds get_captured_states / flush_disk payload["config"] — populate it
    # identically to the eager worker.
    worker._conf = dict(
        num_attention_heads=num_h,
        num_key_value_heads=num_kv,
        hidden_size=hidden,
        head_dim=head_dim,
        attention_multiplier=attn_mult,
    )
    worker._should_capture = should_capture
    # Worker-wide fallback for hookq_mode when a request omits it (matches eager).
    if not hasattr(worker, "hookq_mode"):
        worker.hookq_mode = "all_tokens"

    # Egress buckets — same dicts the eager path / RPC retrieval consume.
    if not hasattr(worker, "_captured_states") or worker._captured_states is None:
        worker._captured_states = {}
    if not hasattr(worker, "_disk_states") or worker._disk_states is None:
        worker._disk_states = {}

    # Async disk-save thread (no-op unless VLLM_HOOK_ASYNC_SAVE=1). The bound
    # method is mixed in via worker_extension_cls.
    if hasattr(worker, "_background_save_loop"):
        init_async_save_thread(worker, worker._background_save_loop, "vllm-hook-qk-io")

    # CAP = max_num_batched_tokens: every token in the largest possible all-prefill
    # batch can land in a distinct buffer row.
    cap = _resolve_max_num_batched_tokens(worker)

    device = next(model.parameters()).device

    # Enumerate matched attn modules first so we know num_layers for the registry.
    matched = list(iter_matched_modules(model, match_attn))
    if not matched:
        print("[graph/install] no attention modules matched ATTN_PATTERNS; "
              "QK graph capture inactive")
        worker._graph_registry = None
        return None
    num_layers = max(layer_num for _, _, layer_num in matched) + 1

    # ---- op mode (default): no static buffers/registry. Tag each attn instance
    # with its layer index, map layer->module name, create the sink, class-wrap.
    # Capture runs inside the qk_probe op mid-forward. ----
    if _CAPTURE_MODE == "op":
        global _GLOBAL_SINK
        device_t = next(model.parameters()).device
        n_wired = 0
        if should_capture:
            _GLOBAL_SINK = torch.zeros(1, dtype=torch.int64, device=device_t)
            for name, module, layer_num in matched:
                setattr(module, _LAYER_ATTR, layer_num)
                _LAYER_TO_NAME[layer_num] = name
                n_wired += 1
        for _, module, _ in matched:
            _wrap_attn_class(type(module))
        worker._graph_registry = None
        # The load-bearing fix: register qk_probe as a splitting op so the piecewise
        # compiler keeps it as an eager seam between cudagraph segments. Without it
        # the op is absorbed into a replayed segment and its Python impl is skipped
        # → captures nothing. Set on the WORKER's resolved config (what its compiler
        # reads) since the driver-side append may be lost on re-resolution.
        try:
            cc = worker.vllm_config.compilation_config
            sops = list(getattr(cc, "splitting_ops", None) or [])
            if "vllm_hook::qk_probe" not in sops:
                sops.append("vllm_hook::qk_probe")
                cc.splitting_ops = sops
            print(f"[graph/install] splitting_ops has qk_probe="
                  f"{'vllm_hook::qk_probe' in sops} (n={len(sops)})")
        except Exception as e:  # noqa: BLE001
            print(f"[graph/install] could not set splitting_ops: {e}")
        print(f"[graph/install] op-mode QK capture wired: {n_wired} layer(s); "
              f"should_capture={should_capture}; q_dim={q_dim} k_dim={k_dim}; "
              f"hooks_on={getattr(worker, '_default_hooks_on', 'prefill')}")
        return None

    # ---- seam mode (prototype): capture inside the attention impl's eager seam.
    # Tag each attn instance with its layer index and class-wrap its BACKEND impl
    # (type(module.impl)) so the shared _capture_body runs inside the opaque
    # unified_attention op — the seam vLLM already pays for. NO splitting op is added
    # (_hook_plugin's create-engine-config declaration is gated on capture=="op", and
    # we skip the qk_probe append here), so the attention break count is unchanged.
    # See docs/qk_seam_capture_prototype.md. ----
    if _CAPTURE_MODE == "seam":
        n_wired = 0
        wrapped_impls: set = set()
        if should_capture:
            for name, module, layer_num in matched:
                setattr(module, _LAYER_ATTR, layer_num)
                _LAYER_TO_NAME[layer_num] = name
                impl = getattr(module, "impl", None)
                if impl is not None:
                    _wrap_impl_class(type(impl))
                    wrapped_impls.add(type(impl).__name__)
                else:
                    print(f"[graph/install] seam: no .impl on {name}; "
                          f"layer {layer_num} will not capture")
                n_wired += 1
        worker._graph_registry = None
        print(f"[graph/install] seam-mode QK capture wired: {n_wired} layer(s); "
              f"should_capture={should_capture}; impls={sorted(wrapped_impls)}; "
              f"q_dim={q_dim} k_dim={k_dim}; NO splitting op added "
              f"(rides unified_attention seam)")
        return None

    # ---- buffer mode (decode-under-cudagraph milestone): static buffers ----
    registry: Optional[HostRegistry] = None
    if should_capture:
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )

    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    # Build a host per matched module, attach to the instance, wrap its class.
    # Non-capture rank: no buffers/host attached; the wrap still installs but its
    # per-instance check short-circuits.
    n_hosts = 0
    for name, module, layer_num in matched:
        if should_capture:
            host = QKHookHost(
                module_name=name,
                layer_num=layer_num,
                cap=cap,
                q_dim=q_dim,
                k_dim=k_dim,
                dtype=buf_dtype,
                device=device,
                do_capture=True,
            )
            # Host + registry back-ref on the instance, so the class-wrap finds them
            # (host for capture, registry to stash the forward context for prefix-K).
            setattr(module, _HOST_ATTR, host)
            setattr(module, _REG_ATTR, registry)
            registry.register_host(host)
            # layer_num -> module name, for egress bucket / prefix-K keys.
            _LAYER_TO_NAME[layer_num] = name
            n_hosts += 1
        _wrap_attn_class(type(module))

    if registry is not None:
        registry.assign_views()
        # Buffers are allocated before KV-cache profiling and can be multiple GB
        # (cap scales with max_num_batched_tokens), perturbing memory accounting —
        # log the size so an OOM here is diagnosable.
        buf_bytes = sum(
            h.q_buf.numel() * h.q_buf.element_size()
            + h.k_buf.numel() * h.k_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        print(f"[graph/install] QK static buffers: {buf_bytes / (1024**2):.1f} MiB "
              f"on {device} (cap={cap})")

    worker._graph_registry = registry
    print(f"[graph/install] QK hosts installed: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}; "
          f"cap={cap}; q_dim={q_dim} k_dim={k_dim}")
    return registry


# ---------------------------------------------------------------------------
# Routing: build the pinned per-(layer, token) destination index for one step
# ---------------------------------------------------------------------------


def _build_routing(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """Fill the registry's pinned mirrors with a destination ROW per captured token
    for the current batch (the scatter runs in-graph against these). Mirrors the
    qkv_hook gating (output_qk filter, hooks_on, hookq_mode).

    Must run pre-forward — the compiled graph only READS the indices at replay.
    ``qsl_cpu`` is the host-side cumulative query_start_loc (len bs+1) from
    pre-forward CPU state, so no per-step device sync. Returns one ``plan`` per
    active request, which egress reuses so it doesn't re-derive the gating.

    Row r = flat_position + 1 (row 0 is the discard sentinel); the same r serves
    every requested layer. Unrequested tokens stay at 0 (exact no-ops). Written as
    vectorised slices, so host cost is O(num_requests), not O(tokens * layers).
    """
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1

    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
    any_active_pinned = registry.any_active_pinned          # (num_layers,)
    cap = registry.cap

    dbg_rows: list = []  # per-request decision trace (diagnostic only)
    plans: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            dbg_rows.append(f"r{i}:no_state")
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_qk") is None:
            dbg_rows.append(f"r{i}:no_output_qk")
            continue

        # output_qk: True (all) | [layer_ids] | {layer: [heads]} — only keys filter
        # layers; None = all layers.
        output_spec = extra.get("output_qk")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, dict):
            layer_filter = {int(k) for k in output_spec.keys()}
        elif isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        # hooks_on: "prefill" (default) | "decode" | "both"; output_token_ids==[]
        # detects prefill robustly under prefix caching.
        hooks_on = extra.get("hooks_on", getattr(model_runner, "_default_hooks_on", "prefill"))
        n_out = len(req_state.output_token_ids)
        n_computed = getattr(req_state, "num_computed_tokens", None)
        n_prompt = len(getattr(req_state, "prompt_token_ids", []) or [])
        _qlen = int(qsl_cpu[i + 1]) - int(qsl_cpu[i])
        dbg_rows.append(
            f"r{i}:qlen={_qlen},out_toks={n_out},computed={n_computed},"
            f"prompt={n_prompt},hooks_on={hooks_on}"
        )
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                dbg_rows.append(f"r{i}:SKIP_prefill_gate(out_toks={n_out})")
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hookq_mode", getattr(model_runner, "_worker_hookq_mode", "all_tokens"))

        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end <= start:
            dbg_rows.append(f"r{i}:empty_range")
            continue
        qlen = end - start  # this step's scheduled tokens for req i

        # ---- ABSOLUTE-position routing (k_buf accumulation) ----
        # Each token scatters to row = its ABSOLUTE sequence position + 1, so q_buf /
        # k_buf accumulate across decode steps: by step t the buffer holds rows for
        # positions 0..total-1. This is the buffer-mode analogue of the eager path's
        # prefix-K reconstruction — egress reads the full key history from the buffer
        # (k_buf[1 : abs_end+1]) instead of re-reading the paged KV cache. With flat
        # (per-step) rows a decode token would overwrite row 1 (the first prompt key),
        # so accumulation REQUIRES absolute rows. num_computed_tokens_cpu[i] is the
        # host-side count of tokens already processed = the absolute position of this
        # step's first token (pre-step value; the scheduler advances it post-forward).
        try:
            abs_start = int(model_runner.input_batch.num_computed_tokens_cpu[i])
        except Exception:  # noqa: BLE001
            abs_start = start  # fallback: flat == absolute for a single fresh prefill
        abs_end = abs_start + qlen
        # Clamp the routed span to buffer capacity (rows > cap → sentinel/discard).
        abs_end_c = min(abs_end, cap)
        n_route = abs_end_c - abs_start
        if n_route <= 0:
            dbg_rows.append(f"r{i}:over_cap(abs_start={abs_start})")
            continue

        # Determine which layers this request wants.
        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        # Route EVERY new token, for BOTH modes: q and k scatter together (q reduced
        # at egress in last_token mode). Flat columns [start, start+n_route) carry this
        # step's tokens; their destination rows are the absolute positions +1.
        rows = torch.arange(abs_start + 1, abs_end_c + 1, dtype=torch.int64)
        layer_idx = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx[:, None], start:start + n_route] = rows[None, :]
        any_active_pinned[layer_idx] = 1

        # Stash everything egress needs so it does not re-derive the gating. seq_start=0
        # assumes capture began at the sequence start (fresh prefill, no prefix-cache
        # hit) — true for the FULL milestones; a cached-prefix prefill is out of scope.
        bucket_key = "_disk_states" if extra.get("save_to_disk") else "_captured_states"
        plans.append({
            "req_idx": i,
            "req_id": req_id,
            "abs_start": abs_start,
            "abs_end": abs_end_c,
            "seq_start": 0,
            "layers": req_layers,
            "hookq_mode": req_mode,
            "bucket_key": bucket_key,
            "req_state": req_state,
        })
        dbg_rows.append(f"r{i}:qlen={qlen},abs=[{abs_start},{abs_end_c}),mode={req_mode}")

    if _DEBUG and _dbg_call_n[0] <= _DBG_MAX_CALLS:
        print(f"[graph/dbg]   routing bs={bs} plans={len(plans)} :: "
              + " | ".join(dbg_rows), flush=True)
    return plans


# ---------------------------------------------------------------------------
# Egress: drain the static buffers into the reuse-contract worker buckets
# ---------------------------------------------------------------------------


def _read_cached_keys_from_ctx(
    ctx, module_name, attn_metadata, req_idx: int, num_cached: int, total_len: int
):
    """Like ``_read_cached_keys`` but reads the kv_cache from the supplied ``ctx``
    instead of get_forward_context() — graph egress runs post-forward, after
    set_forward_context has exited, so it uses the context the wrap stashed.
    """
    try:
        # kv_cache lives on the Attention wrapper in the forward context, not the
        # PyTorch module.
        kv_cache = ctx.no_compile_layers[module_name].kv_cache
        # Layout trap: the extractor takes kv_cache[:, 0]; kv_cache[0] would gather
        # the size-2 key/value axis → OOB device assert.
        key_cache = key_cache_from_layer_kv(kv_cache)

        num_blocks = key_cache.shape[0]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]

        if block_ids.numel() == 0 or int(block_ids.max()) >= num_blocks \
                or int(block_ids.min()) < 0:
            return None

        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)
        return prefix_keys[:num_cached].detach()
    except Exception:
        return None


def _egress(worker, registry: HostRegistry, plans: list) -> None:
    """Drain the scattered q_buf/k_buf rows into the worker buckets in the exact
    shape eager ``qkv_hook`` produces (the reuse contract), so egress/analyzers run
    unchanged: bucket[req_id][module_name] = {"q":[...], "k_all":[...],
    "layer_num": L, "hookq_mode": mode}.

    Buffers are scattered by ABSOLUTE sequence position (see _build_routing), so they
    accumulate the full history. Per (request, layer): q is this step's slice
    ``q_buf[abs_start+1 : abs_end+1]`` (all_tokens) or its final row (last_token);
    k_all is the ACCUMULATED full key history ``k_buf[seq_start+1 : abs_end+1]`` — the
    buffer-mode analogue of the eager path's prefix-K reconstruction, no paged-cache
    read needed.
    """
    if not plans:
        return

    for layer_num, host in registry.iter_hosts():
        q_buf = host.q_buf  # (cap+1, q_dim)
        k_buf = host.k_buf   # (cap+1, k_dim)
        module_name = host.module_name
        cap = host.cap

        for plan in plans:
            if layer_num not in plan["layers"]:
                continue

            abs_start = plan["abs_start"]
            abs_end = plan["abs_end"]
            seq_start = plan["seq_start"]
            req_id = plan["req_id"]
            req_mode = plan["hookq_mode"]

            # Clone so we own the data (buffers are overwritten / extended next step);
            # .cpu() is deferred to flush. Rows are absolute-position + 1 (row 0 is the
            # sentinel). q = this step's queries (already incremental).
            if req_mode == "all_tokens":
                q_tok = q_buf[abs_start + 1:abs_end + 1, :].detach().clone()
            else:
                last_row = min(abs_end, cap)  # last token at abs_end-1 → row abs_end
                q_tok = q_buf[last_row, :].detach().clone()

            # W3 incremental k_all: clone only THIS step's NEW keys
            # (k_buf[abs_start+1:abs_end+1]) instead of the whole accumulated prefix
            # (k_buf[seq_start+1:abs_end+1]). On decode that is O(1) rows/step instead of
            # O(seq_len) — removing the per-step O(N^2) GPU clone (the QK decode hot spot,
            # ~10x HS). The growing-prefix artifact eager produces is reconstructed at
            # flush from these rows + the per-step prefix end (abs_end), so the captured
            # k_all is byte-identical; ``k_prefix_ends`` marks the entry as incremental.
            k_new = k_buf[abs_start + 1:abs_end + 1, :].detach().clone()

            nbytes = (q_tok.numel() * q_tok.element_size()
                      + k_new.numel() * k_new.element_size())
            PROF.incr("hook.fire.qk")
            PROF.gauge("captured.bytes.qk", nbytes)
            # W4: accrue GPU-resident clone bytes for the budget-driven drain.
            dm = getattr(worker, "_capture_drain", None)
            if dm is not None:
                dm.note(nbytes)

            bucket = getattr(worker, plan["bucket_key"])
            if req_id not in bucket:
                bucket[req_id] = {}
            layer_states = bucket[req_id]
            if module_name not in layer_states:
                layer_states[module_name] = {
                    "q": [], "k_all": [], "k_prefix_ends": [],
                    "layer_num": layer_num, "hookq_mode": req_mode,
                }
            layer_states[module_name]["q"].append(q_tok)
            layer_states[module_name]["k_all"].append(k_new)
            layer_states[module_name]["k_prefix_ends"].append(int(abs_end - seq_start))


# ---------------------------------------------------------------------------
# Pre-forward query bounds + cheap "any QK requested" check
# ---------------------------------------------------------------------------


def _any_qk_requested(model_runner) -> bool:
    """True if any request in the batch set output_qk. Cheap, host-side, no device
    sync — lets the wrapper short-circuit the whole routing/egress block when no
    request wants QK, so an unrelated capture-rank worker pays nothing per step.
    """
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
        if extra and extra.get("output_qk") is not None:
            return True
    return False


def _preforward_qsl(model_runner, scheduler_output) -> Optional[list]:
    """Build the cumulative query_start_loc as a host-side list, pre-forward.

    Routing must run before the forward, but the forward context (the eager path's
    qsl source) is only established inside execute_model — so build qsl from
    scheduler_output / input_batch, which the runner populates pre-forward, no
    device sync. qsl is the exclusive cumsum of per-request scheduled token counts
    in req_ids order, matching attn_metadata.query_start_loc. Returns (bs+1,) or
    None (warmup / counts unavailable) → caller skips routing.
    """
    try:
        req_ids = list(model_runner.input_batch.req_ids)
    except Exception:
        return None
    if not req_ids:
        return None

    # Primary: scheduler_output.num_scheduled_tokens is {req_id: count}.
    counts = None
    nst = getattr(scheduler_output, "num_scheduled_tokens", None)
    if isinstance(nst, dict):
        try:
            counts = [int(nst[r]) for r in req_ids]
        except Exception:
            counts = None
    # Fallback: a per-row array on the input_batch (some versions stage it there).
    if counts is None:
        arr = getattr(model_runner.input_batch, "num_scheduled_tokens", None)
        if arr is not None:
            try:
                counts = [int(arr[i]) for i in range(len(req_ids))]
            except Exception:
                counts = None
    if counts is None:
        return None

    qsl = [0]
    for n in counts:
        qsl.append(qsl[-1] + max(0, n))
    return qsl


# ---------------------------------------------------------------------------
# _prepare_inputs routing — the load-bearing integration point (plan §5)
# ---------------------------------------------------------------------------


def _runner_qsl(model_runner) -> Optional[list]:
    """Host-side cumulative query_start_loc for the CURRENT step, read from the
    model runner's OWN buffer AFTER ``_prepare_inputs`` populated it.

    This is the robust qsl source: it is the same query_start_loc attention uses,
    and it is correct for a NEW request's prefill. Pre-forward reconstruction from
    scheduler_output (``_preforward_qsl``) is NOT — a new request is added to
    ``input_batch`` only inside ``_update_states``, which runs after the
    execute_model wrapper's pre-forward section, so the prefill step routed nothing
    (observed on GPU: qsl_bs=None, plans=0 on every prefill).
    """
    try:
        req_ids = model_runner.input_batch.req_ids
        num_reqs = len(req_ids)
        if num_reqs == 0:
            return None
        qsl_buf = model_runner.query_start_loc
        np_arr = getattr(qsl_buf, "np", None)
        if np_arr is not None:
            return [int(x) for x in np_arr[:num_reqs + 1]]
        cpu_t = getattr(qsl_buf, "cpu", None)
        if cpu_t is not None:
            return [int(x) for x in cpu_t[:num_reqs + 1].tolist()]
    except Exception:  # noqa: BLE001
        return None
    return None


def _upload_width(model_runner, qsl_cpu, cap: int) -> int:
    """Smallest contiguous column prefix the routing upload must cover this step.

    The capture/steer ops read ``index[:n]`` where ``n = q.shape[0]`` is the PADDED
    token count — a cudagraph decode batch is padded up to a captured size — so the
    upload must cover ``[0, padded_n)`` with ``[real_tokens, padded_n)`` left at the
    sentinel (0). We bound ``padded_n`` above by ``max(cudagraph_batch_sizes)`` (the
    largest captured decode graph) and the real token count (eager prefill is not
    padded). Uploading only ``[:, :real_tokens]`` would be WRONG: the cudagraph
    padding rows would read stale routing and scatter dummy tokens into real buffer
    rows. ``max(real_n, max_graph_batch)`` is a safe upper bound on ``padded_n`` in
    every case: decode pads up to a captured size (≤ max_graph_batch); a prefill that
    fits a captured size pads up to it (≤ max_graph_batch); a prefill too big to
    cudagraph runs eager (``padded_n == real_n``). Returns a width in ``[1, cap]``;
    ``cap`` = no shrink (always safe), used as the fallback when ``qsl_cpu`` is absent.
    """
    if not qsl_cpu:
        return int(cap)  # can't bound padded_n this step → full width (always safe)
    real_n = int(qsl_cpu[-1])
    try:
        sizes = getattr(model_runner, "cudagraph_batch_sizes", None)
        maxbs = max(sizes) if sizes else cap
    except Exception:  # noqa: BLE001
        maxbs = cap
    return max(1, min(int(cap), max(real_n, int(maxbs))))


def install_prepare_inputs_routing(model_runner, worker, build_routing_fn,
                                   label: str = "qk") -> None:
    """Wrap ``model_runner._prepare_inputs`` to build + upload the capture routing
    right after vLLM finishes input prep — when ``input_batch`` and
    ``query_start_loc`` reflect THIS step (post ``_update_states``). The routing
    must land before the (possibly cudagraph-replayed) forward, and _prepare_inputs
    is the legal off-graph home for it; the resulting plans are stashed on the
    registry for the execute_model wrapper's post-forward egress.

    Idempotent. Defensive: an internal vLLM surface, so any failure degrades to
    "no capture this step" rather than crashing the forward.
    """
    flag = f"_vllm_hook_{label}_prep_wrapped"
    if getattr(model_runner, flag, False):
        return
    orig_prepare = getattr(model_runner, "_prepare_inputs", None)
    if orig_prepare is None or not callable(orig_prepare):
        print(f"[graph/install] no _prepare_inputs to wrap ({label}); routing disabled")
        return
    setattr(model_runner, flag, True)

    def wrapped_prepare_inputs(scheduler_output, *args, **kwargs):
        result = orig_prepare(scheduler_output, *args, **kwargs)
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
        if registry is None or not registry.should_capture:
            return result
        # Skip during vLLM's cudagraph capture pass — a host sync / pinned write
        # while capturing is illegal. Buffers are populated at replay, not capture.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return result
        except Exception:  # noqa: BLE001
            pass
        try:
            registry.begin_step()
            qsl_cpu = _runner_qsl(model_runner)
            width = _upload_width(model_runner, qsl_cpu, registry.cap)

            # W1 invalidation: when the registry reports a stable routing signature
            # (same key + upload width as last step), the device slabs already hold an
            # identical routing — skip reset/build/upload so the step is a pure graph
            # replay with the resident buffer. Steering returns a real key on a stable
            # batch (its routing is bit-identical every decode step); capture returns
            # None (its absolute-position rows advance each step) so it never skips
            # here. A composition change (finish/admit) or prefill<->decode flip moves
            # the key and forces a re-route. (W3 adds capture's own incremental path.)
            key = registry.routing_key(model_runner, qsl_cpu) \
                if qsl_cpu is not None else None
            if (key is not None
                    and key == getattr(registry, "_last_route_key", None)
                    and width == getattr(registry, "_last_route_width", None)):
                registry._pending_plans = getattr(registry, "_last_plans", [])
                PROF.incr("graph.route.skip")
                return result

            with PROF.timed("graph.route"):
                # Only reset+upload the column prefix the (cudagraph-padded) forward
                # actually reads — for decode that is ~max_graph_batch, not the full
                # max_num_batched_tokens slab (a much smaller H2D copy per step).
                registry.reset_pinned(width)
                plans = build_routing_fn(model_runner, registry, qsl_cpu) \
                    if qsl_cpu is not None else []
                registry.upload(width)
            registry._pending_plans = plans
            registry._last_route_key = key
            registry._last_route_width = width
            registry._last_plans = plans
            PROF.incr("graph.route.build")
        except Exception as e:  # noqa: BLE001
            registry._pending_plans = []
            registry._last_route_key = None  # force a re-route after an error
            if _DEBUG:
                print(f"[graph/dbg] _prepare_inputs routing ({label}) raised: {e!r}",
                      flush=True)
        return result

    model_runner._prepare_inputs = wrapped_prepare_inputs
    print(f"[graph/install] _prepare_inputs routing wrapper installed ({label})")


# ---------------------------------------------------------------------------
# execute_model wrapper (never compiled — all per-request Python lives here)
# ---------------------------------------------------------------------------


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Install the QK buffer-mode routing (``_prepare_inputs`` wrapper) + egress
    (``execute_model`` wrapper). Idempotent.

    Routing runs in the ``_prepare_inputs`` wrapper (where ``input_batch`` /
    ``query_start_loc`` are current for this step — including a NEW request's
    prefill); egress drains the static buffers post-forward here, reading the plans
    the routing wrapper stashed on the registry.
    """
    if getattr(model_runner, "_vllm_hook_qk_wrapped", False):
        return

    # Op and seam modes capture inside the forward (qk_probe op / attention impl
    # seam respectively), so neither needs the routing+egress wrappers — skip them.
    if _CAPTURE_MODE in ("op", "seam"):
        print(f"[graph/install] capture mode={_CAPTURE_MODE} — execute_model wrapper "
              f"skipped (capture runs inside the forward)")
        return

    model_runner._vllm_hook_qk_wrapped = True

    # Surface worker fallback mode + default phase onto the runner so the routing
    # helper reads them without reaching through the worker.
    model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
    model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    # Routing — runs after _update_states + input prep, so prefill routes correctly.
    install_prepare_inputs_routing(model_runner, worker, _build_routing, label="qk")

    # W4/W6c: budget-driven GPU->pinned drain manager (no-op unless a budget env is set).
    from vllm_hook_plugins.graph.drain import CaptureDrainManager
    _model = getattr(model_runner, "model", None)
    try:
        _dev = next(_model.parameters()).device if _model is not None else "cpu"
    except Exception:  # noqa: BLE001
        _dev = "cpu"
    worker._capture_drain = CaptureDrainManager.from_env(_dev)
    if worker._capture_drain.enabled:
        print(f"[graph/install] capture drain enabled: budget="
              f"{worker._capture_drain.budget_bytes} bytes")

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        _dbg_call_n[0] += 1
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # Refresh mode/phase each step so a later worker mutation can't desync routing.
        model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

        from vllm_hook_plugins.graph.ops import get_fire_count
        fire_before = get_fire_count()

        # Forward: _prepare_inputs (wrapped) builds+uploads routing, then the graph
        # replays and capture_qk scatters into the static buffers.
        result = orig_execute_model(scheduler_output, *args, **kwargs)

        fire_after = get_fire_count()

        # Post-forward: drain the buffers into the worker buckets using the plans the
        # routing wrapper stashed this step.
        plans = getattr(registry, "_pending_plans", None) or []
        if plans:
            with PROF.timed("graph.egress"):
                _egress(worker, registry, plans)
            # W4: drain GPU-resident clones to pinned host if over the byte budget.
            dm = getattr(worker, "_capture_drain", None)
            if dm is not None:
                dm.maybe_drain(worker)
        registry._pending_plans = []  # consume

        if _DEBUG and (plans or _dbg_call_n[0] <= _DBG_MAX_CALLS):
            n_disk = sum(len(v) for v in getattr(worker, "_disk_states", {}).values())
            n_rpc = sum(len(v) for v in getattr(worker, "_captured_states", {}).values())
            print(
                f"[graph/dbg] call#{_dbg_call_n[0]}: plans={len(plans)} "
                f"op_fired={fire_after - fire_before} (total={fire_after}) "
                f"disk_entries={n_disk} rpc_entries={n_rpc}",
                flush=True,
            )
        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install] execute_model wrapper installed (routing + egress)")


# ---------------------------------------------------------------------------
# load_model monkey-patch — the single install entry point for register()
# ---------------------------------------------------------------------------

_LOAD_MODEL_PATCHED = False


def patch_worker_load_model() -> None:
    """Monkey-patch ``Worker.load_model`` to install the graph path after the model
    is built (before compile/capture). Idempotent. Runs the worker's own
    ``graph_install`` only when graph mode is armed AND the worker defines one —
    so the steer worker (no graph path) is untouched and graph-off is the original
    behaviour byte-for-byte.
    """
    global _LOAD_MODEL_PATCHED
    if _LOAD_MODEL_PATCHED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    orig_load_model = Worker.load_model

    def patched_load_model(self, *args, **kwargs):
        result = orig_load_model(self, *args, **kwargs)

        # Gate 1: graph mode armed, else leave the eager path untouched.
        if not graph_mode_enabled():
            return result

        # Gate 2: dispatch by the worker's own graph_install (QK/HS define it; steer
        # doesn't, so it stays eager).
        graph_install = getattr(self, "graph_install", None)
        if not callable(graph_install):
            return result

        try:
            graph_install()
        except Exception as e:  # noqa: BLE001
            # Never let an install failure take down model loading — fall back to
            # no capture rather than crashing.
            print(f"[graph/install] graph install FAILED ({e}); continuing "
                  f"without graph capture.")
            PROF.incr("graph.install.errors")

        return result

    Worker.load_model = patched_load_model
    _LOAD_MODEL_PATCHED = True


# ---------------------------------------------------------------------------
# Public entry points (for _hook_plugin.register())
# ---------------------------------------------------------------------------

__all__ = [
    "set_graph_mode",
    "graph_mode_enabled",
    "patch_worker_load_model",
    "install_qk_hosts",
    "install_execute_model_wrapper",
    "install_prepare_inputs_routing",
    "_runner_qsl",
]
