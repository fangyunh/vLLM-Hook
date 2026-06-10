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

    # Op mode: wire the capture body + the worker it writes into.
    if _CAPTURE_MODE == "op":
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
        # Clamp to buffer capacity (highest routable position is cap-1 → row cap);
        # over-cap tokens stay in the sentinel and are discarded.
        end = min(end, cap)
        if end <= start:
            continue

        # Determine which layers this request wants.
        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        # Route EVERY new token, for BOTH modes: q and k scatter together, and even
        # in last_token mode k_all keeps the full window (q is reduced at egress).
        # Vectorised: assign the [start, end) column span for all requested layers
        # in one indexed write (O(num_requests), not a per-token double loop).
        rows = torch.arange(start + 1, end + 1, dtype=torch.int64)
        layer_idx = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx[:, None], start:end] = rows[None, :]
        any_active_pinned[layer_idx] = 1

        # Stash everything egress needs so it does not re-derive the gating.
        bucket_key = "_disk_states" if extra.get("save_to_disk") else "_captured_states"
        plans.append({
            "req_idx": i,
            "req_id": req_id,
            "start": start,
            "end": end,
            "layers": req_layers,
            "hookq_mode": req_mode,
            "bucket_key": bucket_key,
            "req_state": req_state,
        })

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

    Per (request, layer): q is the (query_len, q_dim) slice (all_tokens) or final
    row (last_token); k_all is the key slice with prefix keys prepended. metadata /
    seq_lens / kv-cache come from the wrap's stashed forward context (the live one
    is gone post-forward); absent stash → no prefix-K, matching eager.
    """
    if not plans:
        return

    ctx = registry.fwd_ctx
    metadata = registry.fwd_attn_metadata

    # seq_lens (prefix-K total length) from the stashed current-step metadata.
    seq_lens = None
    if metadata is not None:
        _, seq_lens = get_query_metadata(metadata)

    # Normalise per-layer metadata (hybrid models key by layer) to one object for
    # block_table reads.
    meta_for_kv = metadata
    if isinstance(metadata, dict):
        try:
            meta_for_kv = next(iter(metadata.values()))
        except StopIteration:
            meta_for_kv = None

    for layer_num, host in registry.iter_hosts():
        q_buf = host.q_buf  # (cap+1, q_dim)
        k_buf = host.k_buf   # (cap+1, k_dim)
        module_name = host.module_name

        for plan in plans:
            if layer_num not in plan["layers"]:
                continue

            start = plan["start"]
            end = plan["end"]
            query_len = end - start
            req_id = plan["req_id"]
            req_mode = plan["hookq_mode"]
            req_idx = plan["req_idx"]

            # Rows are start+1 .. end (sentinel-offset). Clone so we own the data
            # (buffers are overwritten next step); .cpu() deferred to flush. The
            # last_token row end is populated because routing always covers the full
            # range; clamp to cap defensively.
            cap = host.cap
            if req_mode == "all_tokens":
                q_tok = q_buf[start + 1:end + 1, :].detach().clone()
            else:
                last_row = min(end, cap)  # row (end-1)+1 == end, bounded by cap
                q_tok = q_buf[last_row, :].detach().clone()
            k_tok = k_buf[start + 1:end + 1, :].detach().clone()

            # Prefix-K: when num_cached > 0 (total seq_len minus new query_len),
            # read the missing prefix keys from the paged KV cache and prepend.
            # Needs the stashed ctx (live context gone post-forward).
            if seq_lens is not None and meta_for_kv is not None and ctx is not None:
                try:
                    sv = seq_lens[req_idx]
                    total_len = int(sv.item()) if hasattr(sv, "item") else int(sv)
                    num_cached = total_len - query_len
                    if num_cached > 0:
                        PROF.incr("kv.prefix_recon")
                        with PROF.timed("kv.prefix_recon"):
                            prefix_k = _read_cached_keys_from_ctx(
                                ctx, module_name, meta_for_kv, req_idx,
                                num_cached, total_len,
                            )
                        if prefix_k is not None:
                            k_tok = torch.cat(
                                [prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok],
                                dim=0,
                            )
                except Exception:
                    PROF.incr("kv.prefix_recon.errors")

            PROF.incr("hook.fire.qk")
            PROF.gauge(
                "captured.bytes.qk",
                q_tok.numel() * q_tok.element_size()
                + k_tok.numel() * k_tok.element_size(),
            )

            bucket = getattr(worker, plan["bucket_key"])
            if req_id not in bucket:
                bucket[req_id] = {}
            layer_states = bucket[req_id]
            if module_name not in layer_states:
                layer_states[module_name] = {
                    "q": [], "k_all": [], "layer_num": layer_num,
                    "hookq_mode": req_mode,
                }
            layer_states[module_name]["q"].append(q_tok)
            layer_states[module_name]["k_all"].append(k_tok)


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
# execute_model wrapper (never compiled — all per-request Python lives here)
# ---------------------------------------------------------------------------


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Wrap ``model_runner.execute_model`` for routing + egress. Idempotent. An
    instance-method replacement, outside the compiled region — the legal home for
    per-request Python (the graph only READS the buffers we upload here).
    """
    if getattr(model_runner, "_vllm_hook_qk_wrapped", False):
        return

    # Op mode captures inside the op, so it needs no wrapper — skip it.
    if _CAPTURE_MODE == "op":
        print("[graph/install] capture mode=op — execute_model wrapper skipped "
              "(capture runs inside qk_probe)")
        return

    model_runner._vllm_hook_qk_wrapped = True

    # Surface worker fallback mode + default phase onto the runner so the routing
    # helper reads them without reaching through the worker.
    model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
    model_runner._default_hooks_on = getattr(
        worker, "_default_hooks_on", "prefill"
    )

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        _dbg_call_n[0] += 1
        if _DEBUG and _dbg_call_n[0] == 1:
            print("[graph/dbg] execute_model WRAPPER INVOKED (1st call) — the "
                  "wrapper is on the live forward path", flush=True)
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)

        # Non-capture rank or no hosts: nothing to route/drain — straight through.
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # Don't route during vLLM's graph capture pass — a host sync / pinned write
        # while capturing is illegal. Buffers are populated at replay, not capture,
        # so skipping is correct. Mirrors the eager hook's is_current_stream_capturing bail.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return orig_execute_model(scheduler_output, *args, **kwargs)
        except Exception:  # noqa: BLE001
            pass

        # Cheap host-side gate: skip the whole routing/egress block when no request
        # wants QK.
        any_qk = _any_qk_requested(model_runner)
        if not any_qk:
            _dbg(f"call#{_dbg_call_n[0]}: wrapper live, any_qk=False -> skip")
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # ---- pre-forward: build routing on the pinned mirror + upload ----
        # qsl comes from pre-forward CPU state, not get_forward_context() (the
        # context is only live inside execute_model). The wrap stashes the live
        # context during the forward for egress's prefix-K.
        plans: list = []
        registry.begin_step()  # clear the stash so the wrap re-snapshots this step
        # Refresh the mode/phase snapshot each step so a later worker mutation can't
        # desync the routing helper.
        model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        qsl_cpu = _preforward_qsl(model_runner, scheduler_output)

        with PROF.timed("graph.route"):
            registry.reset_pinned()
            if qsl_cpu is not None:
                plans = _build_routing(model_runner, registry, qsl_cpu)
            registry.upload()

        from vllm_hook_plugins.graph.ops import get_fire_count
        fire_before = get_fire_count()

        # ---- forward (graph replays; capture_qk scatters inline; the wrap stashes
        # the forward context) ----
        result = orig_execute_model(scheduler_output, *args, **kwargs)

        fire_after = get_fire_count()

        # ---- post-forward: drain buffers into the worker buckets (the stashed
        # context supplies prefix-K metadata; the live one is gone). ----
        if plans:
            with PROF.timed("graph.egress"):
                _egress(worker, registry, plans)

        # Print on every QK step (plans>0) and the first few calls, so the decisive
        # prefill rows show without flooding on decode.
        if _DEBUG and (plans or _dbg_call_n[0] <= _DBG_MAX_CALLS):
            qsl_bs = (len(qsl_cpu) - 1) if qsl_cpu is not None else None
            n_disk = sum(len(v) for v in getattr(worker, "_disk_states", {}).values())
            n_rpc = sum(len(v) for v in getattr(worker, "_captured_states", {}).values())
            print(
                f"[graph/dbg] call#{_dbg_call_n[0]}: any_qk=True qsl_bs={qsl_bs} "
                f"plans={len(plans)} op_fired={fire_after - fire_before} "
                f"(total={fire_after}) disk_entries={n_disk} rpc_entries={n_rpc}",
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
]
