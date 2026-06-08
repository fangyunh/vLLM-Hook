"""Integration core for CUDA-graph QK capture (file 3/5).

This module wires the static-buffer hosts (graph/hosts.py) + routing registry
(graph/registry.py) + capture op (graph/ops.py) into a live vLLM worker so that
post-RoPE Q/K is captured inside the CUDA graph and replays every step — the
thing the eager `register_forward_hook` path (workers/probe_hookqk_worker.py)
cannot do under `torch.compile` (Dynamo traces each `nn.Module.forward` once and
replaces it, bypassing instance hooks; see graph/ops.py for the full lesson).

THREE PIECES (all installed from a single monkey-patch of Worker.load_model):

  1. install_qk_hosts(worker)
        On the capture rank, build one QKHookHost per matched attention module
        (buffers allocated HERE = load_model timing = BEFORE vLLM's cudagraph
        pool, so data_ptr is fixed across replays — §8.4 / C3), register them in
        a HostRegistry stored on the worker, assign the row-views, and CLASS-WRAP
        each `type(attn_module).forward` so it calls `host.capture(q, k)` on the
        post-RoPE q,k (args[0], args[1]) BEFORE the original forward. Class-level
        only — instance hooks are bypassed by Dynamo (step0_8). Idempotent per
        class.

  2. install_execute_model_wrapper(model_runner, worker)
        Wrap `model_runner.execute_model` (an instance method, NEVER compiled),
        so ALL per-request Python logic lives here — legal because the wrapper is
        outside the traced region. Per step:
          PRE-forward (routing, §8.5): from scheduler_output + input_batch.req_ids
            + model_runner.requests, build the per-(layer, token) destination row
            on the registry's PINNED mirrors honoring output_qk layer filter,
            hooks_on prefill/decode gating, and hookq_mode; one upload() fans it
            out to the device slabs. Unrequested tokens/layers stay at row 0 (the
            discard sentinel) — an exact no-op via the op's active-gate.
          forward: call the original execute_model.
          POST-forward (egress, §8.6): for each active request/layer, slice the
            scattered q_buf/k_buf rows, run prefix-K reconstruction (relocated
            _read_cached_keys), and append into the SAME worker buckets the eager
            path uses (_captured_states / _disk_states) — REUSE CONTRACT, so
            get_captured_states / flush_disk / the analyzer work unchanged.

  3. patch_worker_load_model()
        Monkey-patch vllm.v1.worker.gpu_worker.Worker.load_model so the above two
        installers run after super().load_model(), guarded by the graph-mode flag
        and by the worker actually being the QK worker.

GATING: the graph path is only armed when the plugin sets the process flag (see
`set_graph_mode` / `graph_mode_enabled`), which _hook_plugin only does when
VLLM_HOOK_ALLOW_CUDAGRAPH == "1". Otherwise the legacy v0.2.0 eager path runs
untouched.
"""
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
    match_attn,
)


# ---------------------------------------------------------------------------
# Process-wide graph-mode flag.
# ---------------------------------------------------------------------------
# The plugin's _patched_create_engine_config sets this (to True) only when
# VLLM_HOOK_ALLOW_CUDAGRAPH == "1"; otherwise it stays False and the
# load_model patch is a no-op, preserving the v0.2.0 eager path EXACTLY.
# We mirror it in an env var too, because create_engine_config runs in the
# driver process while load_model runs in the (possibly forked/spawned) worker
# process — the module global would not survive a spawn, but the env var (set
# before workers are launched) does.
_GRAPH_MODE_ENV = "VLLM_HOOK_GRAPH_MODE"
_graph_mode_enabled = False


def set_graph_mode(enabled: bool) -> None:
    """Arm/disarm the CUDA-graph QK capture path for this process tree.

    Called by the plugin's _patched_create_engine_config. Sets both the module
    global (fast path, same process) and an env var (so spawned/forked worker
    processes inherit the decision — load_model runs there, not in the driver).
    """
    global _graph_mode_enabled
    _graph_mode_enabled = bool(enabled)
    os.environ[_GRAPH_MODE_ENV] = "1" if enabled else "0"


def graph_mode_enabled() -> bool:
    """True when the CUDA-graph QK path should install at load_model.

    Checks the module global first (same-process driver) then the inherited env
    var (worker subprocess). Either being set arms the path.
    """
    return _graph_mode_enabled or os.environ.get(_GRAPH_MODE_ENV) == "1"


# ---------------------------------------------------------------------------
# Class-level Attention.forward wrap (Dynamo-bypass-proof; see step0_8).
# ---------------------------------------------------------------------------
# Per-class originals so the wrap is idempotent and reversible. The wrap reads a
# host off the *instance* (set in install_qk_hosts) so one wrapped class serves
# every layer; if an instance has no host (non-attn instance of the same class,
# or non-capture rank) it falls straight through to the original forward.
_WRAPPED_ATTN_CLASSES: Dict[type, Any] = {}
# Attribute name under which each attn module instance carries its QKHookHost.
_HOST_ATTR = "_vllm_hook_qk_host"
# Attribute name under which each attn module instance carries a back-reference
# to the worker's HostRegistry, so the wrap can stash the live forward context
# for POST-forward egress (prefix-K) reads without re-deriving it.
_REG_ATTR = "_vllm_hook_qk_registry"
# op mode: a plain int layer-index attribute on each attn instance + a single
# module-global keep-alive sink tensor. This mirrors step0_8's PROVEN shape
# (closed-over global tensor + op call) and avoids reading a per-instance host
# OBJECT inside the traced forward (which Dynamo might graph-break on).
_LAYER_ATTR = "_vllm_hook_qk_layer"
_GLOBAL_SINK = None  # int64 (1,) keep-alive for qk_probe; set in install (op mode)

# Staging gate for the in-forward forward-context stash used by prefix-K
# reconstruction. Default OFF so the first GPU run validates core QK capture
# without the (novel, possibly graph-breaking) get_forward_context() call inside
# the traced Attention.forward. Enable with VLLM_HOOK_QK_PREFIXK_STASH=1 once
# core capture is confirmed. See _wrap_attn_class for the full rationale.
_PREFIXK_STASH = os.environ.get("VLLM_HOOK_QK_PREFIXK_STASH") == "1"

# Capture mechanism:
#   "op"     (default) — impl-runs-at-runtime: the qk_probe op runs the eager
#                        capture body mid-forward (works for PREFILL under
#                        PIECEWISE, where the forward is NOT cudagraph-replayed).
#                        No pre-forward routing — sidesteps the input_batch
#                        staleness that made the buffer path capture nothing.
#   "buffer"           — the §8 static-buffer + device-routing path (needed for
#                        DECODE-under-cudagraph; kept for that later milestone).
_CAPTURE_MODE = os.environ.get("VLLM_HOOK_QK_CAPTURE", "op")

# Set at install for the impl-runs capture body (op mode).
_ACTIVE_WORKER = None
_LAYER_TO_NAME: Dict[int, str] = {}
_capture_dbg = {"n": 0}  # one-shot capture confirmation counter (op mode)
_ctx_dumped = [False]    # one-shot forward-context exploration dump

# Diagnostic logging for the execute_model wrapper hot path. Default ON during
# bring-up so the first GPU runs are observable; set VLLM_HOOK_QK_DEBUG=0 to mute.
# Prints are compact, flushed, and capped to the first few calls.
_DEBUG = os.environ.get("VLLM_HOOK_QK_DEBUG", "1") == "1"
_DBG_MAX_CALLS = int(os.environ.get("VLLM_HOOK_QK_DEBUG_MAX", "8"))
_dbg_call_n = [0]


def _dbg(msg: str) -> None:
    if _DEBUG and _dbg_call_n[0] <= _DBG_MAX_CALLS:
        print(f"[graph/dbg] {msg}", flush=True)


def _cpu_1d(x):
    """Coerce a query_start_loc/seq_lens carrier to a 1-D CPU tensor, or None.

    vLLM v1 stores these as CpuGpuBuffer-like objects (``.cpu``/``.gpu``/``.np``
    tensor attributes) on the model_runner, or sometimes as plain tensors. We
    prefer the CPU mirror to avoid a device sync. Returns None if nothing usable.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    # CpuGpuBuffer: .cpu is a CPU tensor attribute (NOT the Tensor.cpu method).
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
    """The eager qkv_hook capture body, run INSIDE the qk_probe op (op mode).

    This is invoked by graph/ops._qk_probe_impl at runtime, mid-forward, where
    get_forward_context() and model_runner.input_batch are CURRENT (the same
    point the eager register_forward_hook ran). It performs the identical
    per-request slicing / hooks_on gating / hookq_mode / prefix-K as the eager
    ``ProbeHookQKWorker.qkv_hook`` and writes into the SAME worker buckets, so
    get_captured_states / flush_disk / the analyzer consume it unchanged.

    ``q_in``/``k_in`` are the post-RoPE q/k passed to Attention.forward (whole
    batch, (num_tokens, dim)); ``layer_idx`` is the host's layer_num. No static
    buffers or pre-forward routing are involved — capture is the op's side effect.
    """
    worker = _ACTIVE_WORKER
    if worker is None:
        return
    module_name = _LAYER_TO_NAME.get(int(layer_idx))
    if module_name is None:
        return
    mr = worker.model_runner

    # Diagnostics gated on REAL (small) batches — warmup uses a huge shape
    # (e.g. 16384) with no metadata and would flood/consume the log budget.
    _small = q_in.shape[0] <= 2048

    ctx = get_forward_context()
    metadata = getattr(ctx, "attn_metadata", None)

    # Resolve query_start_loc / seq_lens. The eager path read these from
    # ctx.attn_metadata, but under torch.compile/PIECEWISE that field is None
    # (confirmed by the CTX dump). Fall back to the model_runner's persistent
    # buffers (mr.query_start_loc / mr.seq_lens), which ARE current mid-forward
    # (built in _prepare_inputs before the model runs). These are CpuGpuBuffer-
    # like, so _cpu_1d unwraps them to CPU tensors.
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

    # Silently skip dummy/profiling forwards (no real requests) — vLLM runs many
    # of these at warmup/capture (e.g. q=512, reqs=0) and they would flood the
    # log and exhaust the debug budget before the real prompt forward appears.
    if not req_ids:
        return

    if query_start_loc is None:
        if _DEBUG and _capture_dbg["n"] < 12:
            _capture_dbg["n"] += 1
            print(f"[graph/dbg] _capture_body layer={layer_idx} q={tuple(q_in.shape)}"
                  f" reqs={len(req_ids)} EXIT: query_start_loc is None", flush=True)
        return

    # bs = actual request count. mr.query_start_loc may be a PADDED persistent
    # buffer (length max_reqs+1), so DO NOT use len(query_start_loc)-1; the valid
    # cumulative offsets are query_start_loc[:bs+1], indexed by request.
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

        # output_qk layer filter (True=all | [layers] | {layer: heads}).
        output_spec = extra.get("output_qk")
        if isinstance(output_spec, dict):
            if layer_idx not in {int(k) for k in output_spec.keys()}:
                continue
        elif isinstance(output_spec, list):
            if layer_idx not in output_spec:
                continue

        # hooks_on prefill/decode gating — output_token_ids==[] detects prefill.
        # CORRECT here (unlike the pre-forward path): we are mid-forward, so the
        # request state reflects the CURRENT step.
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

        # Prefix-K reconstruction (identical to eager qkv_hook). Needs the attn
        # metadata's block_table, which is only available when ctx.attn_metadata
        # is present (eager path). Under compile it is None, so prefix-K is
        # deferred — harmless for fresh prompts (num_cached==0). TODO: reconstruct
        # block_table from ctx.slot_mapping / no_compile_layers for prefix caching.
        if seq_lens is not None and metadata is not None:
            try:
                sv = seq_lens[i]
                total_len = int(sv.item()) if hasattr(sv, "item") else int(sv)
                query_len = end - start
                num_cached = total_len - query_len
                if num_cached > 0:
                    PROF.incr("kv.prefix_recon")
                    with PROF.timed("kv.prefix_recon"):
                        meta_kv = metadata if not isinstance(metadata, dict) \
                            else next(iter(metadata.values()))
                        prefix_k = _read_cached_keys(
                            module_name, meta_kv, i, num_cached, total_len)
                    if prefix_k is not None:
                        k_tok = torch.cat(
                            [prefix_k.to(k_tok.device, dtype=k_tok.dtype), k_tok],
                            dim=0)
            except Exception:
                PROF.incr("kv.prefix_recon.errors")

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
    """Class-wrap ``cls.forward`` to call ``host.capture(q, k)`` pre-forward.

    Idempotent per class. The capture fires on the post-RoPE q,k passed to the
    vLLM Attention module — ``args[0]`` is q, ``args[1]`` is k (the same inputs
    the eager ``qkv_hook`` reads as ``input[0]``/``input[1]``). Capture happens
    BEFORE the original forward so it is recorded as a graph node upstream of the
    attention compute, exactly like step0_8's counter wrap.

    The wrap ALSO snapshots the live forward context onto the registry on its
    first fire of each step (a pure-Python side effect on install-time-constant
    objects; under graph replay it does not run, which is fine because egress
    only needs the snapshot during a non-captured / eager-recorded step and the
    stash persists for the matching egress drain). This is the ONLY place the
    forward context is provably live (the eager qkv_hook relied on exactly this),
    so it is where the kv_cache handles + attn_metadata for prefix-K are read.

    NOTE (GPU-validate): the snapshot uses getattr-on-self for the host/registry.
    step0_8 proved a closure-captured counter lands in the compiled graph; the
    getattr-on-self pattern here must be confirmed to (a) still record the
    capture_qk node in the compiled graph and (b) not graph-break on the host
    attribute. The branch is on install-time constants only, so it should be
    Dynamo-constant-folded, but verify the counter-equality check on the attn
    class on the first GPU run.
    """
    if cls in _WRAPPED_ATTN_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_ATTN_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            # op mode (default): lean path matching step0_8 — read a plain int
            # layer attribute (install-time constant; -1 => don't capture) and
            # emit ONE qk_probe op against the closed-over global sink. No
            # per-instance host object is touched inside the traced forward.
            if _CAPTURE_MODE == "op":
                layer_idx = getattr(self, _LAYER_ATTR, -1)
                if layer_idx >= 0 and len(args) >= 2:
                    torch.ops.vllm_hook.qk_probe(
                        args[0], args[1], _GLOBAL_SINK, layer_idx)
                return orig_fwd(self, *args, **kwargs)

            # ---- buffer mode (decode milestone): static-buffer scatter. ----
            host = getattr(self, _HOST_ATTR, None)
            # do_capture is an INSTALL-TIME constant (True only on capture rank)
            # — this branch is on a python-level constant, never on per-request
            # graph data, so it does not introduce data-dependent control flow
            # into the traced region. When a host is present we emit the single
            # opaque capture_qk node; otherwise we fall straight through.
            if host is not None and host.do_capture:
                # Stash the live forward context for POST-forward egress (prefix-K
                # reconstruction). Only the first attn layer of the step takes
                # effect (registry clears the stash each step); its metadata
                # carries the batch-wide query_start_loc / block_table prefix-K
                # needs. Wrapped in a bare except because a missing/torn-down
                # context must never break the forward — egress simply falls back
                # to no prefix-K then.
                #
                # STAGING GATE (default OFF): calling get_forward_context() inside
                # the traced Attention.forward is NOVEL — vLLM normally reads the
                # forward context inside the opaque attention op, not in traced
                # Python — so it may provoke a Dynamo graph break under
                # torch.compile. It is ONLY needed for prefix-K (num_cached > 0),
                # which fresh-prompt capture never triggers. So the first GPU run
                # validates the core capture path WITHOUT this risk; set
                # VLLM_HOOK_QK_PREFIXK_STASH=1 to enable prefix-K once core capture
                # is confirmed. (If enabling it breaks compile, the fix is to grab
                # kv_cache refs at install time instead of stashing here.)
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
                # args[0] = post-RoPE q, args[1] = post-RoPE k. Positional in
                # every vLLM Attention.forward signature; guard length defensively.
                if len(args) >= 2:
                    host.capture(args[0], args[1])
            return orig_fwd(self, *args, **kwargs)

        return wrapped

    cls.forward = make_wrapped(orig_forward)


def _resolve_max_num_batched_tokens(worker) -> int:
    """Return ``max_num_batched_tokens`` for the buffer CAP, version-robustly.

    The scheduler config is reachable via the worker's ``vllm_config`` on every
    vLLM v1 release; the model_runner may also expose ``scheduler_config``
    directly on some versions. Probe both, preferring the model_runner attribute
    when present.
    """
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
# Host install (capture rank only) — runs at load_model, before compile/capture.
# ---------------------------------------------------------------------------


def install_qk_hosts(worker) -> Optional[HostRegistry]:
    """Build + register the per-layer QK hosts and class-wrap Attention.forward.

    Runs inside the worker process from the load_model patch, AFTER the model is
    built but BEFORE warm-up/compile/capture (so buffers are allocated before the
    cudagraph pool — fixed data_ptr across replays, §8.4 / C3).

    On a non-capture rank (TP rank != 0) we build NO registry and allocate NO
    buffers; the class-wrap still installs (harmless — its per-instance host
    check short-circuits because no host is attached), so the model's forward
    behaviour is identical to the capture rank's, only without the scatter.

    Returns the HostRegistry (also stored on ``worker._graph_registry``), or None
    on the non-capture rank / when no attention modules match.
    """
    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install] no model on model_runner; skip QK host install")
        return None

    # Register the capture op(s) once per process, BEFORE any wrap can fire.
    register_graph_ops()

    # op mode: wire the impl-runs capture body + the worker it writes into.
    if _CAPTURE_MODE == "op":
        from vllm_hook_plugins.graph.ops import set_capture_fn
        global _ACTIVE_WORKER
        _ACTIVE_WORKER = worker
        set_capture_fn(_capture_body)

    # Capture-rank gating (mirrors probe_hookqk_worker.install_hooks): only TP
    # rank 0 captures — residual streams are replicated across TP ranks after
    # all-reduce, so the data is identical (§9).
    tp_size = worker.parallel_config.tensor_parallel_size
    should_capture = tp_size <= 1 or worker.rank % tp_size == 0

    # Model dims — pulled EXACTLY like probe_hookqk_worker.install_hooks so the
    # buffer widths match the eager path and the analyzer config is identical.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    num_h = int(getattr(text_cfg, "num_attention_heads"))
    num_kv = int(getattr(text_cfg, "num_key_value_heads", num_h))
    hidden = int(getattr(text_cfg, "hidden_size"))
    # _conf.head_dim MUST match the eager worker EXACTLY (the analyzer reads
    # _conf), so derive it the same way (hidden // num_h). For BUFFER sizing we
    # additionally honour an explicit config.head_dim when present, because the
    # real post-RoPE q width is num_h * (model's actual head_dim) — if that
    # differs from hidden//num_h, the eager-derived width would mis-size the
    # buffers and index_copy_ would raise on a width mismatch. A first-capture
    # assertion (q_dim == q.shape[1]) below verifies the chosen width on GPU.
    head_dim = hidden // num_h                       # for _conf (eager parity)
    buf_head_dim = int(getattr(text_cfg, "head_dim", None) or head_dim)  # for buffers
    attn_mult = float(getattr(text_cfg, "attention_multiplier", 1 / math.sqrt(head_dim)))
    q_dim = num_h * buf_head_dim
    k_dim = num_kv * buf_head_dim

    # _conf is consumed by get_captured_states / flush_disk (payload["config"]),
    # so the egress path MUST populate it identically to the eager worker.
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

    # Background I/O thread for async disk saves (no-op unless VLLM_HOOK_ASYNC_SAVE=1).
    # _background_save_loop is defined on ProbeHookQKWorker, which the Worker
    # mixes in via worker_extension_cls, so the bound method is present.
    if hasattr(worker, "_background_save_loop"):
        init_async_save_thread(worker, worker._background_save_loop, "vllm-hook-qk-io")

    # CAP = max_num_batched_tokens (covers an all-prefill batch — every token in
    # the largest possible batch can land in a distinct buffer row). The
    # scheduler config lives on the worker's vllm_config across vLLM versions;
    # probe a few known carriers so the lookup is version-robust.
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

    # ---- op mode (default): NO static buffers / registry. Tag each attn
    # instance with its layer index, map layer->module name (for egress keys /
    # prefix-K), create the global keep-alive sink, and class-wrap. Capture runs
    # inside the qk_probe op mid-forward — sidesteps the input_batch staleness
    # that made the pre-forward buffer routing capture nothing. ----
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
        # Ensure our op is a splitting op on the WORKER's resolved config too —
        # this is the config the worker's compiler actually reads, and it runs at
        # load_model (BEFORE compile/capture). The driver-side append
        # (_hook_plugin) may be lost if vLLM re-resolves splitting_ops, so set it
        # here as well. If absent, our op is absorbed into a replayed cudagraph
        # segment and captures nothing — this is the load-bearing fix.
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

    # ---- buffer mode (decode-under-cudagraph milestone): static buffers. ----
    registry: Optional[HostRegistry] = None
    if should_capture:
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )

    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    # Build a host per matched module, attach to the instance, wrap its class.
    # On a NON-capture rank we allocate no buffers and attach no host: the class
    # wrap still installs (so forward behaviour matches the capture rank) but its
    # per-instance host check short-circuits, so the scatter never runs and no
    # memory is wasted (§9 / TP — only the capture rank materialises buffers).
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
            # Attach the host to the *instance* so the class-level wrap finds it
            # via getattr(self, _HOST_ATTR). One wrapped class serves every layer.
            setattr(module, _HOST_ATTR, host)
            # Back-ref to the registry so the wrap can stash the live forward
            # context (for POST-forward prefix-K) from inside the forward.
            setattr(module, _REG_ATTR, registry)
            registry.register_host(host)
            # op mode: map layer_num -> module name so the capture body can key
            # the egress buckets / prefix-K kv_cache lookup by module name.
            _LAYER_TO_NAME[layer_num] = name
            n_hosts += 1
        _wrap_attn_class(type(module))

    if registry is not None:
        registry.assign_views()
        # Buffer memory accounting: CAP scales with max_num_batched_tokens, and
        # the buffers are allocated BEFORE the KV-cache profiling run, so for
        # large models / large CAP this can be multiple GB on the capture rank
        # and perturb gpu_memory_utilization accounting. Log it so an OOM here is
        # diagnosable. (GPU-validate: consider capping CAP at min(mnbt,
        # max_model_len) for the paper's larger-model runs — TODO, not wired.)
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
# Routing: build the pinned per-(layer, token) destination index for one step.
# ---------------------------------------------------------------------------


def _build_routing(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """Fill the registry's pinned mirrors for the current batch.

    Mirrors the per-request gating in ``qkv_hook`` (output_qk layer filter,
    hooks_on prefill/decode gating, hookq_mode) but writes a DESTINATION ROW per
    captured token instead of capturing inline. The scatter then runs inside the
    graph against these uploaded indices.

    ``qsl_cpu`` is the cumulative query-start-loc as a HOST-SIDE python list
    (length bs+1); the caller derives it from pre-forward CPU state (see
    ``_preforward_qsl``) so routing carries the CURRENT step's bounds without a
    per-step device sync. Routing MUST run pre-forward because the compiled graph
    only READS the uploaded indices at replay time.

    Returns a list of ``plan`` dicts (one per active request) that egress reuses
    so it does not recompute the gating — each holds the per-request slice bounds
    and the bucket routing the egress drain needs.

    Row assignment: capture-requested token t for layer L gets a UNIQUE row
    ``r = start + 1`` (where ``start`` is its position in the flat token stream),
    so ``q_buf[r]`` / ``k_buf[r]`` hold exactly that token after the scatter.
    Row 0 is reserved as the discard sentinel, hence the ``+ 1`` offset. The same
    row r is used for every requested layer (each layer has its own buffers).
    Unrequested tokens stay at 0 (the caller zeroes the mirrors via
    reset_pinned before calling this) — exact no-ops.

    The pinned mirror is filled with VECTORISED tensor slices (one assignment per
    request, not per (token, layer) scalar), so per-step host cost is ~O(num
    requests) tensor ops, not O(num_tokens * num_layers) python scalar writes.
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

        # output_qk: True (all layers) | [layer_ids] | {layer: [heads]} — only
        # the keys are used for layer filtering (heads forwarded to the analyzer
        # by the caller). None => no filter (all layers).
        output_spec = extra.get("output_qk")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, dict):
            layer_filter = {int(k) for k in output_spec.keys()}
        elif isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        # hooks_on: "prefill" (default) | "decode" | "both". output_token_ids==[]
        # detects the first (prefill) pass robustly under prefix caching.
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
        # Clamp the routed range to buffer capacity. Row = flat position + 1, so
        # the highest routable position is cap-1 (row cap). Tokens beyond that
        # (cannot happen when cap == max_num_batched_tokens, but stay safe) are
        # simply not routed -> they land in the sentinel and are discarded.
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

        # Route EVERY new token of this request, for BOTH modes. The capture op
        # uses ONE index per token to scatter q and k together, and even in
        # last_token mode k_all must hold every new token's key (only q is
        # reduced to the last token, done at egress) — matching the eager
        # qkv_hook, which slices q but keeps the full k window.
        #
        # VECTORISED write (was a per-(token, layer) python double loop): the
        # destination row for flat position `pos` is `pos + 1` for EVERY requested
        # layer, so build the row vector once and assign the [start, end) column
        # span for all requested layers in one indexed assignment. Cost is
        # O(num_requests) tensor ops, independent of token count x layer count.
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
# Egress: drain the static buffers into the REUSE-CONTRACT worker buckets.
# ---------------------------------------------------------------------------


def _read_cached_keys_from_ctx(
    ctx, module_name, attn_metadata, req_idx: int, num_cached: int, total_len: int
):
    """Prefix-K reader that uses an EXPLICITLY-PASSED forward context.

    Identical algorithm to probe_hookqk_worker._read_cached_keys, but reads the
    kv_cache from the supplied ``ctx`` instead of calling get_forward_context().
    The eager path could call get_forward_context() because it ran mid-forward;
    the graph egress runs POST-forward (set_forward_context has exited), so it
    must use the context the attention wrap stashed DURING the forward. This
    reproduces the eager reader's output byte-for-byte given the same ctx +
    attn_metadata + req_idx.
    """
    try:
        # kv_cache is bound to the vLLM Attention wrapper in the forward context,
        # not to the PyTorch module: ctx.no_compile_layers[layer_name].kv_cache.
        kv_cache = ctx.no_compile_layers[module_name].kv_cache
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]

        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        block_table = attn_metadata.block_table
        num_blocks_needed = math.ceil(total_len / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]

        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)
        return prefix_keys[:num_cached].detach()
    except Exception:
        return None


def _egress(worker, registry: HostRegistry, plans: list) -> None:
    """Read the scattered q_buf/k_buf rows back into the worker buckets.

    Populates ``worker._captured_states`` / ``worker._disk_states`` with the
    EXACT shape the eager ``qkv_hook`` produces, so get_captured_states /
    flush_disk / _save_safetensors / the analyzer all work UNCHANGED:

        bucket[req_id][module_name] = {"q":[...], "k_all":[...],
                                       "layer_num": L, "hookq_mode": mode}

    For each active (request, layer):
      * all_tokens: q is the (query_len, q_dim) slice of this request's scattered
        rows; for last_token q is the (q_dim,) final row.
      * k_all is the (query_len, k_dim) key slice with cached prefix keys
        prepended (prefix-K reconstruction), matching eager.

    metadata / seq_lens / the kv-cache handle for prefix-K all come from the
    forward context the attention wrap STASHED on the registry during the forward
    (registry.fwd_ctx / registry.fwd_attn_metadata). They are read here, POST-
    forward, because get_forward_context() is no longer valid once execute_model
    has returned (set_forward_context has exited). Falling back to no prefix-K
    when the stash is absent matches the eager path's own except-fallback.
    """
    if not plans:
        return

    ctx = registry.fwd_ctx
    metadata = registry.fwd_attn_metadata

    # seq_lens for the prefix-K total length come from the stashed (current-step)
    # metadata — same source the eager path read mid-forward.
    seq_lens = None
    if metadata is not None:
        _, seq_lens = get_query_metadata(metadata)

    # Per-layer attn-metadata view (hybrid models key metadata by layer; the
    # eager path passes the first dict entry to the prefix-K reader). Normalise to
    # a single metadata object for block_table reads.
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

            # Rows for this request's tokens are start+1 .. end (sentinel-offset).
            # Slice once; clone so we own the data (the buffers are overwritten
            # next step). .cpu() is deferred to retrieval/flush, matching eager.
            #
            # NOTE: q_buf[end] for last_token depends on row `end` (= (end-1)+1)
            # having been scattered this step. _build_routing always routes the
            # full [start, end) range for BOTH modes (see its docstring), so row
            # `end` is populated. The clamp `end = min(end, cap)` in _build_routing
            # also bounds the row at cap; guard the read defensively here too.
            cap = host.cap
            if req_mode == "all_tokens":
                q_tok = q_buf[start + 1:end + 1, :].detach().clone()
            else:
                last_row = min(end, cap)  # row (end-1)+1 == end, bounded by cap
                q_tok = q_buf[last_row, :].detach().clone()
            k_tok = k_buf[start + 1:end + 1, :].detach().clone()

            # Prefix-K reconstruction (relocated from qkv_hook). seq_lens[i] is
            # the total length (cached + new); query_len is new tokens only. When
            # num_cached > 0, read the missing prefix keys from the paged KV cache
            # and prepend, so k_all is the FULL key history the analyzer expects.
            # Reads the kv_cache via the STASHED forward context (post-forward the
            # live context is gone), so ctx must be present.
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
# Pre-forward query bounds + cheap "any QK requested" check.
# ---------------------------------------------------------------------------


def _any_qk_requested(model_runner) -> bool:
    """True if ANY request in the current batch set ``extra_args["output_qk"]``.

    Cheap, host-side, no device sync — lets the wrapper short-circuit the entire
    routing/upload/egress block (including the qsl D2H read) on the common case
    where no request wants QK, so an unrelated capture-rank worker does not pay
    the per-step cost for the whole serving lifetime.
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
    """Derive the cumulative query_start_loc as a HOST-SIDE list, pre-forward.

    Routing must run BEFORE the forward (the compiled graph only READS the
    uploaded indices at replay), but in vLLM v1 the forward context / attn
    metadata (the eager path's query_start_loc source) is only established INSIDE
    execute_model. So we instead build qsl from ``scheduler_output`` /
    ``input_batch``, which the runner populates pre-forward — entirely on the
    host, with no device sync.

    qsl is the exclusive cumulative sum of per-request scheduled (new) token
    counts, in ``input_batch.req_ids`` order: qsl[0]=0, qsl[i+1]=qsl[i]+n_i. This
    matches attn_metadata.query_start_loc the eager path used (the runner builds
    that same cumsum from the same per-request counts).

    Returns the (bs+1,) python list, or None when the counts are unavailable
    (warmup / shape mismatch) so the caller skips routing this step.

    GPU-VALIDATE: ``scheduler_output.num_scheduled_tokens`` is a dict
    {req_id: int} on current vLLM v1; confirm the attribute name/shape on the
    target version. Fallbacks probe ``input_batch.num_scheduled_tokens`` /
    ``num_computed_tokens`` if the scheduler-output form differs.
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
# execute_model wrapper (NEVER compiled — all per-request Python lives here).
# ---------------------------------------------------------------------------


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Wrap ``model_runner.execute_model`` for routing + egress.

    Idempotent: a second call is a no-op (guarded by a sentinel attribute). The
    wrapper is an instance-method replacement on the model_runner, OUTSIDE the
    compiled region, so it is the legal home for every per-request Python branch
    (the compiled graph only READS the routing buffers we upload here).
    """
    if getattr(model_runner, "_vllm_hook_qk_wrapped", False):
        return

    # op mode captures inside the qk_probe op (mid-forward); the pre-forward
    # routing + post-forward egress wrapper is only needed for the buffer
    # (decode-under-cudagraph) path. Skip it so op mode pays nothing per step.
    if _CAPTURE_MODE == "op":
        print("[graph/install] capture mode=op — execute_model wrapper skipped "
              "(capture runs inside qk_probe)")
        return

    model_runner._vllm_hook_qk_wrapped = True

    # Surface the worker fallback mode + default phase onto the runner so the
    # routing helper can read them without reaching back through the worker.
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

        # Do NOT touch routing during vLLM's CUDA-graph CAPTURE pass: a host
        # sync / pinned write while the stream is capturing is illegal and can
        # corrupt the graph. The buffers are populated at REPLAY time, not capture
        # time, so skipping routing here is correct. Mirrors the eager hook's
        # is_current_stream_capturing() bail.
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return orig_execute_model(scheduler_output, *args, **kwargs)
        except Exception:  # noqa: BLE001
            pass

        # Cheap host-side gate: if no request wants QK, skip the whole routing /
        # upload / egress block (and its costs) entirely. An unrelated capture
        # worker then pays nothing per step.
        any_qk = _any_qk_requested(model_runner)
        if not any_qk:
            _dbg(f"call#{_dbg_call_n[0]}: wrapper live, any_qk=False -> skip")
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # ---- PRE-forward: build routing on the pinned mirror + upload. ----
        # query_start_loc comes from PRE-forward CPU state (scheduler_output /
        # input_batch), NOT from get_forward_context(): in vLLM v1 the forward
        # context is only established INSIDE execute_model, so reading it here
        # would see None/stale metadata. The attention wrap stashes the LIVE
        # context onto the registry during the forward for egress's prefix-K.
        plans: list = []
        registry.begin_step()  # clear the stash so the wrap re-snapshots this step
        # Refresh the mode/phase snapshot from the worker each step so a later
        # mutation of worker.hookq_mode / _default_hooks_on cannot desync the
        # routing helper (cheap getattrs; current code never mutates them).
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

        # ---- forward (compiled graph replays; capture_qk scatters inline; the
        # attn wrap stashes the live forward context on the registry). ----
        result = orig_execute_model(scheduler_output, *args, **kwargs)

        fire_after = get_fire_count()

        # ---- POST-forward: drain static buffers into the worker buckets. The
        # stashed forward context supplies metadata/seq_lens/kv_cache for prefix-K
        # (the live context is gone now). ----
        if plans:
            with PROF.timed("graph.egress"):
                _egress(worker, registry, plans)

        # Print on every prefill/QK step (plans>0) and for the first few calls,
        # so we always see the decisive prefill rows without flooding on decode.
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
# load_model monkey-patch — the single install entry point for register().
# ---------------------------------------------------------------------------

_LOAD_MODEL_PATCHED = False


def patch_worker_load_model() -> None:
    """Monkey-patch ``Worker.load_model`` to install the QK graph path.

    Called once from the plugin's ``register()``. After the original
    ``load_model`` builds the model (and BEFORE warm-up/compile/capture), the
    patched method installs the QK hosts + execute_model wrapper — but ONLY when:
      * graph mode is armed (graph_mode_enabled(), set by the plugin when
        VLLM_HOOK_ALLOW_CUDAGRAPH == "1"), AND
      * this worker is actually the QK worker (it carries the RPC methods the QK
        extension defines, e.g. ``get_captured_states``) — a cheap hasattr guard
        so the HS / steer workers are never touched.

    Idempotent: re-patching is a no-op. When graph mode is off, the patched
    method is byte-for-byte the original behaviour, preserving v0.2.0 EXACTLY.
    """
    global _LOAD_MODEL_PATCHED
    if _LOAD_MODEL_PATCHED:
        return

    from vllm.v1.worker.gpu_worker import Worker

    orig_load_model = Worker.load_model

    def patched_load_model(self, *args, **kwargs):
        result = orig_load_model(self, *args, **kwargs)

        # Gate 1: graph mode must be armed (else legacy eager path, untouched).
        if not graph_mode_enabled():
            return result

        # Gate 2: only the QK worker. The QK extension defines get_captured_states;
        # the HS / steer workers do not, so this hasattr check selects QK only.
        if not hasattr(self, "get_captured_states"):
            return result

        try:
            # Prefer the worker's thin graph-install entry (it seeds the egress
            # buckets as instance dicts, then delegates straight back here). Fall
            # back to the installers directly if a worker build predates it.
            graph_install = getattr(self, "graph_install", None)
            if callable(graph_install):
                graph_install()
            else:
                install_qk_hosts(self)
                install_execute_model_wrapper(self.model_runner, self)
        except Exception as e:  # noqa: BLE001
            # Never let an install failure take down model loading — fall back to
            # the eager path's behaviour (no capture) rather than crashing.
            print(f"[graph/install] QK graph install FAILED ({e}); continuing "
                  f"without graph capture.")
            PROF.incr("graph.install.errors")

        return result

    Worker.load_model = patched_load_model
    _LOAD_MODEL_PATCHED = True


# ---------------------------------------------------------------------------
# Public entry points (for _hook_plugin.register()).
# ---------------------------------------------------------------------------

__all__ = [
    "set_graph_mode",
    "graph_mode_enabled",
    "patch_worker_load_model",
    "install_qk_hosts",
    "install_execute_model_wrapper",
]
