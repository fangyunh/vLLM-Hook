"""CUDA-graph QK capture — install, per-step routing, and egress."""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.hosts import QKHookHost
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import QKReqCaptureRecord
from vllm_hook_plugins.graph.ring_sizing import resolve_ring_bytes_auto
from vllm_hook_plugins.workers._common import iter_matched_modules
from vllm_hook_plugins.workers.probe_hookqk_worker import (
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
_HOST_ATTR = "_vllm_hook_qk_host"          # per-instance QKHookHost
_REG_ATTR = "_vllm_hook_qk_registry"       # per-instance HostRegistry back-ref

# Prefix-K forward-context stash inside the traced forward. Off by default: the
# get_forward_context() call there is novel and may graph-break, and it's only
# needed for prefix-K (num_cached > 0). Enable once core capture is confirmed.
_PREFIXK_STASH = os.environ.get("VLLM_HOOK_QK_PREFIXK_STASH") == "1"


def _require_buffer_mode() -> None:
    """Buffer mode is the only FULL-cudagraph capture path.

    The PIECEWISE op/seam capture mechanism (v0.3.0) was removed; ``VLLM_HOOK_QK_CAPTURE``
    is retained only so an explicit ``op``/``seam`` request fails loud instead of silently
    running buffer. Unset (or ``=buffer``) is the normal case.
    """
    mode = os.environ.get("VLLM_HOOK_QK_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise RuntimeError(
            f"VLLM_HOOK_QK_CAPTURE={mode!r} is no longer supported: the PIECEWISE op/seam "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph capture path; "
            "unset VLLM_HOOK_QK_CAPTURE or set it to 'buffer'.")

# Batched egress (VLLM_HOOK_BATCHED_EGRESS, default ON; =0 is the per-request fallback): ONE
# index_select per layer gathers only the rows being saved into a compact own-storage tensor;
# each request takes a VIEW into it, cutting launches from O(layers x requests) to O(layers).
# Byte-identical: index_select copies into fresh storage, so views survive buffer reuse exactly
# like the old per-request .clone() did.
_BATCHED_EGRESS = os.environ.get("VLLM_HOOK_BATCHED_EGRESS", "1") == "1"

_capture_dbg = {"n": 0}  # one-shot capture confirmation counter

# _NO_PREFIXK skips prefix-K reconstruction (bisection / no prefix caching).
_NO_PREFIXK = os.environ.get("VLLM_HOOK_QK_NO_PREFIXK") == "1"

# ---------------------------------------------------------------------------
# Capture-ring helpers. The QK path scatters q + k into TWO parallel per-layer rings
# sharing ONE GpuCaptureRing cursor, then drains them off-loop to durable per-layer
# raw files (graph/ring_drain_qk.py). Mirrors the HS ring (graph/install_hs.py) —
# duplicated here (not imported) so install_hs's import of this module stays acyclic.
# ---------------------------------------------------------------------------

def _ring_reserve_or_block(ring: GpuCaptureRing, n: int, consumer=None) -> int:
    """Reserve ``n`` contiguous ring rows for this step's q/k capture, BLOCKING (polling) on
    ring-full rather than dropping (the never-drop contract). Runs on the ENGINE thread inside the
    ``_prepare_inputs`` routing wrapper.

    With the OFF-LOOP consumer drain the block is genuine: ``time.sleep`` releases the GIL, so the
    consumer thread can drain earlier (already-forwarded) steps and ``advance_drain`` to free rows —
    no deadlock, since the consumer only drains PAST steps whose forwards already completed. With
    the SYNCHRONOUS drain (``consumer is None``) the previous step already fully drained, so the
    reserve succeeds at once and the block only engages on a mis-sized ring. Fails loud
    (``RingBackpressureError``, re-raised by the routing wrapper) on a DEAD consumer or a mis-sized
    ring. QK twin of ``install_hs._ring_reserve_or_block``."""
    start = ring.reserve(n)
    if start is not None:
        return start
    timeout = float(os.environ.get("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", "10") or "10")
    poll = float(os.environ.get("VLLM_HOOK_RING_BACKPRESSURE_POLL_S", "0.001") or "0.001")
    PROF.incr("qk.ring.backpressure")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if consumer is not None and not consumer.is_alive():
            err = getattr(consumer, "error", None)
            raise RingBackpressureError(
                f"QK capture ring full and the off-loop drain consumer is DEAD: need {n} rows, "
                f"free={ring.free_rows()} of {ring.n_slots} rows/layer. Consumer error: {err!r}")
        time.sleep(poll)
        start = ring.reserve(n)
        if start is not None:
            return start
    raise RingBackpressureError(
        f"QK capture ring full: need {n} rows, free={ring.free_rows()} of {ring.n_slots} "
        f"rows/layer; reserve blocked past {timeout}s. Raise VLLM_HOOK_RING_GPU_BYTES or "
        f"the off-loop drain is not keeping up.")


def _resolve_qk_ring_rows(worker, num_layers, q_dim, k_dim, buf_dtype, device) -> tuple:
    """Size the shared QK capture ring, resolved LAZILY at install (after vLLM carves KV).

    ``ring_bytes`` comes from ``resolve_ring_bytes_auto`` — a FIXED 4 GiB ring by default
    (``VLLM_HOOK_RING_GPU_BYTES``), or the legacy ``0.9 x reserve_frac x total_gpu`` ratio when
    ``VLLM_HOOK_CAPTURE_GPU_RESERVE_FRAC`` is set; each path applies its own fit/overcommit gate.
    q and k share ONE cursor (the ``capture_qk`` op scatters both at the SAME index), so a slot costs
    ``(q_dim + k_dim)`` elements; with ``num_layers`` parallel per-layer ring pairs, ``R`` = per-layer
    ring rows = ``ring_bytes // (num_layers * (q_dim + k_dim) * dtype_size)``. Both ``q_buf`` and
    ``k_buf`` get ``R`` usable rows (+1 sentinel), so each satisfies ``R >= cap`` by construction of
    the budget. Fails loud if ``R < 1`` (spec §8: no silent degrade). Returns ``(R, ring_bytes)``."""
    elem = torch.empty(0, dtype=buf_dtype).element_size()
    if str(device).startswith("cuda"):
        total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_gpu = 1 << 30
    try:
        gpu_util = float(getattr(worker.vllm_config.cache_config,
                                 "gpu_memory_utilization", 0.9))
    except Exception:  # noqa: BLE001
        gpu_util = 0.9
    ring_bytes = resolve_ring_bytes_auto(total_gpu, gpu_util)   # fixed 4 GiB default; legacy ratio opt-in
    per_slot_bytes = (int(q_dim) + int(k_dim)) * elem
    R = int(ring_bytes // (num_layers * per_slot_bytes))
    if R < 1:
        raise RuntimeError(
            f"QK capture ring too small: ring_bytes={ring_bytes} num_layers={num_layers} "
            f"q_dim={q_dim} k_dim={k_dim} dtype={buf_dtype} -> R={R} rows/layer (<1). Raise "
            f"VLLM_HOOK_RING_GPU_BYTES or reduce the model.")
    return R, ring_bytes


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


def _wrap_attn_class(cls: type) -> None:
    """Class-wrap ``cls.forward``, idempotent per class. Calls ``host.capture(q,k)``
    (the buffer-mode static scatter). args[0]=post-RoPE q, args[1]=k (the eager hook's
    input[0]/input[1]); capture runs before the original forward.

    Also snapshots the live forward context onto the registry on its first fire each
    step — the only place it's provably live — so egress can read kv_cache/attn_metadata
    for prefix-K post-forward.
    """
    if cls in _WRAPPED_ATTN_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_ATTN_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            # Static-buffer scatter. do_capture is an install-time constant, so this
            # branch adds no data-dependent control flow to the traced region.
            host = getattr(self, _HOST_ATTR, None)
            if host is not None and host.do_capture:
                # Stash the live forward context for post-forward prefix-K egress (only the first
                # attn layer of the step takes effect; bare-except so a torn-down context never
                # breaks the forward). Gated off by default since get_forward_context() in traced
                # Python may graph-break.
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

    _require_buffer_mode()  # op/seam removed; fail loud on an explicit request

    # Register the capture op(s) once per process, BEFORE any wrap can fire.
    register_graph_ops()

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
    # Under TP each rank's Attention op produces only its SHARD of heads (vLLM shards attention
    # heads by tp_size, replicating KV heads when there are fewer than tp_size); head_dim is not
    # sharded, only the head COUNT is. The static capture buffers must be sized to THIS rank's
    # sharded width, or the capture_qk scatter's index_copy_ mismatches q_src and crashes at
    # cudagraph warmup. Byte-identical at tp_size=1. _conf above deliberately keeps the FULL
    # counts, matching the eager worker's config.
    sh_num_h = num_h // tp_size
    sh_num_kv = max(1, num_kv // tp_size)
    q_dim = sh_num_h * buf_head_dim
    k_dim = sh_num_kv * buf_head_dim

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
    # v0.6.0 score-capture defaults (graph mode skips install_hooks, so set them here too).
    # SCORE mode is OUT OF SCOPE for the QK capture-ring path (v1 = raw q/k only): the score is an
    # O(S^2) [S_q,S_k] matrix recomputed at retrieval from staged Q/K, which the ring's per-step
    # scatter → durable q/k files does not stage. A worker-wide score default is a fail-loud install
    # error (a per-request qk_capture="score" is refused in _build_routing).
    worker._score_mode_default = os.environ.get("VLLM_HOOK_QK_SCORE", "0") == "1"
    worker._score_head_default = int(os.environ.get("VLLM_HOOK_QK_SCORE_HEAD", "0"))
    worker._score_dtype = torch.float16
    if worker._score_mode_default:
        raise RuntimeError(
            "VLLM_HOOK_QK_SCORE=1 (attention-score capture) is not supported on the QK capture-ring "
            "path (v1 = raw q/k only). Unset it, or use the eager/bank path for score capture.")

    # Egress buckets — same dicts the eager path / RPC retrieval consume.
    if not hasattr(worker, "_captured_states") or worker._captured_states is None:
        worker._captured_states = {}
    if not hasattr(worker, "_disk_states") or worker._disk_states is None:
        worker._disk_states = {}

    # Rank-1c: writer PROCESS for off-GIL serialize+write (no-op unless VLLM_HOOK_WRITER_PROCESS=1).
    from vllm_hook_plugins.graph.writer_process import init_writer_process
    init_writer_process(worker)

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

    buf_dtype = model.dtype if hasattr(model, "dtype") else next(model.parameters()).dtype

    # ---- Capture-ring geometry. Each captured layer holds TWO persistent ring buffers — q_buf
    # (R+1, q_dim) and k_buf (R+1, k_dim) — with row R the shared SENTINEL (pad / no-capture
    # discard). ALL layers advance in lockstep off ONE shared GpuCaptureRing cursor (q + k scatter
    # to the SAME index via capture_qk), so one reserve serves every layer; the HostRegistry's
    # per-(layer, token) routing slabs now carry ADVANCING ring slots, not batch positions. Buffers
    # are built here (load_model, before the cudagraph pool) so their data_ptrs stay fixed across
    # replays. Only rank 0 captures — QK reads PRE-attention q/k (before the TP collective), so
    # rank-0-only needs no TP-symmetric bake.
    registry: Optional[HostRegistry] = None
    ring: Optional[GpuCaptureRing] = None
    n_hosts = 0
    if should_capture:
        R, ring_bytes = _resolve_qk_ring_rows(worker, num_layers, q_dim, k_dim, buf_dtype, device)
        if R < cap:
            print(f"[graph/install] WARNING: QK ring rows/layer R={R} < token cap={cap}; a single "
                  f"max-token step may exceed the ring -> backpressure. Steady decode still fits.")
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )
        # ONE shared logical cursor across the parallel per-layer q/k rings. ring.buf is NOT
        # allocated (storage is the per-layer q_buf/k_buf); we use only
        # reserve/physical_slots/segments/advance_drain/free_rows/SENTINEL. row_bytes/row_shape
        # describe a K row for the mmap default sizing (the drain sizes q + k files independently).
        elem = torch.empty(0, dtype=buf_dtype).element_size()
        ring = GpuCaptureRing(row_bytes=k_dim * elem, n_slots=R, device=device,
                              dtype=buf_dtype, row_shape=(k_dim,))
        registry._qk_ring = ring
        registry._qk_step_entries = []
        # Pad / no-capture lanes route to the ring SENTINEL row (R), NOT 0 — ring slot 0 is a REAL
        # storage row now. reset_pinned fills this; the incremental / GPU routers use column-diff
        # (not advancing positions), so disable them → the wrapper takes the legacy
        # reset -> build -> upload branch (identical to HS).
        registry.sentinel_row = ring.SENTINEL
        registry.incremental_enabled = False
        registry.gpu_routing = False
        registry.capture_index_all.fill_(ring.SENTINEL)
        for _slot in registry._ring.slots:
            _slot["capture_index"].fill_(ring.SENTINEL)
        worker._capture_ring = ring

    # Build a host per matched module (with its (R+1, dim) ring buffers), attach, wrap its class.
    # Non-capture rank: no buffers/host attached; the wrap still installs but short-circuits.
    for name, module, layer_num in matched:
        if should_capture:
            q_buf = torch.zeros(R + 1, q_dim, dtype=buf_dtype, device=device)
            k_buf = torch.zeros(R + 1, k_dim, dtype=buf_dtype, device=device)
            host = QKHookHost(
                module_name=name,
                layer_num=layer_num,
                cap=cap,                 # token cap: register_host check + routing slab width
                q_dim=q_dim,
                k_dim=k_dim,
                dtype=buf_dtype,
                device=device,
                do_capture=True,
                q_buf=q_buf,             # ring storage: (R+1, dim), NOT (cap+1, dim)
                k_buf=k_buf,
            )
            setattr(module, _HOST_ATTR, host)
            setattr(module, _REG_ATTR, registry)
            registry.register_host(host)
            n_hosts += 1
        _wrap_attn_class(type(module))

    if registry is not None:
        registry.assign_views()
        buf_bytes = sum(
            h.q_buf.numel() * h.q_buf.element_size()
            + h.k_buf.numel() * h.k_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        print(f"[graph/install] QK capture ring: {buf_bytes / (1024**2):.1f} MiB on {device} "
              f"(R={ring.n_slots} rows/layer, {num_layers} layers, "
              f"ring_bytes={ring_bytes / (1024**2):.0f} MiB budget, token cap={cap}, "
              f"sentinel_row={registry.sentinel_row}; q_dim={q_dim} k_dim={k_dim})")

    worker._graph_registry = registry
    print(f"[graph/install] QK ring hosts installed: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}; cap={cap}; "
          f"NO splitting op (rides decode cudagraph)")
    return registry


# ---------------------------------------------------------------------------
# Routing: build the pinned per-(layer, token) destination index for one step
# ---------------------------------------------------------------------------


def _build_routing(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """Capture-ring routing: map each captured token's batch column to an ADVANCING shared-ring
    slot (persists until the drain reads it), NOT the old batch-position row ``p+1`` overwritten
    each step.

    q and k scatter to the SAME index (the ``capture_qk`` op writes q_buf[idx] AND k_buf[idx]), so
    ONE ``ring.reserve(qlen)`` per request advances both rings and serves every requested layer. K is
    kept EVERY step (the per-step k rows concatenate to the full ``k_full``), so the reserve is the
    WHOLE span ``qlen = end - start`` every step (unlike the HS ring's last_token 1-row reserve). q is
    kept only on ``emit_q`` (all_tokens: every step; last_token: the final prefill chunk + each decode
    step) — recorded in the METADATA, not the routing: the op scatters q into all the reserved slots
    regardless, and a non-emit step's q rows are simply never referenced by the sidecar (dead,
    harmless, like the HS sentinel lanes).

    LayerEntry COLLAPSE: stashes ONE :class:`QKReqCaptureRecord` per capturing request on
    ``registry._qk_step_entries`` — carrying that request's OWN 0-based layer list — instead of
    fanning out ``num_layers`` ``QKStepEntry`` objects on the engine loop (the O(reqs x layers)
    per-fire allocation; ``QKStepEntry`` has 8 fields, all shared across a request's layers except
    ``layer``). The drain expands each record into the identical flat ``QKStepEntry`` list OFF the
    loop, so the sidecar / demux are byte-for-byte unchanged. Records this step's shared-ring start
    slot + total reserved rows (``_qk_step_start`` / ``_qk_step_rows``) for the off-loop consumer.
    Gating (output_qk filter, hooks_on, hookq_mode, chunked-last_token emit_q) mirrors the eager
    qkv_hook. Runs pre-forward on the ENGINE thread; the reserve BLOCKS on a full ring (never drops),
    and RingBackpressureError propagates (the wrapper re-raises it).
    """
    registry._qk_step_entries = []
    registry._qk_step_start = None
    registry._qk_step_rows = 0
    if not registry.should_capture:
        return []
    ring: Optional[GpuCaptureRing] = getattr(registry, "_qk_ring", None)
    if ring is None:
        return []
    consumer = getattr(registry, "_qk_consumer", None)
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1
    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
    cap = registry.cap

    plans: list = []
    records: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_qk") is None:
            continue

        # output_qk: True (all) | [layer_ids] | {layer: [heads]} — only keys filter layers.
        output_spec = extra.get("output_qk")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, dict):
            layer_filter = {int(k) for k in output_spec.keys()}
        elif isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on", getattr(model_runner, "_default_hooks_on", "prefill"))
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hookq_mode", getattr(model_runner, "_worker_hookq_mode", "all_tokens"))
        # SCORE mode is out of scope on the ring path (v1 raw q/k only). A per-request request is
        # refused (skip + a capped warn) rather than silently captured as q/k; the worker-wide
        # default already fails loud at install.
        cap_mode = extra.get("qk_capture",
                             "score" if getattr(model_runner, "_worker_score_mode", False) else "qk")
        if cap_mode == "score":
            PROF.incr("qk.ring.score_unsupported")
            if _capture_dbg.get("score_warn", 0) < 1:
                _capture_dbg["score_warn"] = 1
                print("[graph/install] WARNING: per-request qk_capture='score' is not supported on "
                      "the QK capture-ring path (v1 raw q/k only); this request is NOT captured.",
                      flush=True)
            continue

        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end > cap:  # invariant: a step's tokens <= max_num_batched_tokens == cap
            end = cap
        if end <= start:
            continue
        qlen = end - start  # this step's scheduled tokens for req i

        # abs_end = absolute key count through this step; num_computed_tokens_cpu[i] is the pre-step
        # processed count (the cached-prefix length on the request's FIRST capture step — 0 for a
        # fresh prefill; the reader FAILS LOUD on a >0 first step, since v1 does not prepend the
        # trimmed prefix keys from paged KV).
        try:
            num_computed = int(model_runner.input_batch.num_computed_tokens_cpu[i])
        except Exception:  # noqa: BLE001
            num_computed = start
        abs_end = num_computed + qlen

        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        # last_token + chunked prefill: K still ACCUMULATES on every prefill chunk (so k_full is
        # complete), but Q + prefix marker are emitted only on the FINAL chunk. all_tokens emits Q
        # every step; decode is always "final" (emit Q).
        emit_q = True
        if req_mode == "last_token" and len(req_state.output_token_ids) == 0:
            try:
                num_prompt = int(model_runner.input_batch.num_prompt_tokens[i])
            except Exception:  # noqa: BLE001
                num_prompt = abs_end
            emit_q = abs_end >= num_prompt

        # ---- ring reserve (shared cursor) + advancing slots ----
        # Reserve the WHOLE span (K needs every token every step) and route ALL qlen columns to the
        # reserved slots. q rides the same slots; emit_q only gates the METADATA q record.
        n = qlen
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._qk_step_start is None:
            registry._qk_step_start = start_slot
        registry._qk_step_rows += n
        phys = ring.physical_slots(start_slot, n)                 # n ints in [0, R)
        phys_t = torch.tensor(phys, dtype=torch.int64)
        layer_idx_t = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        # q metadata (LOGICAL slots, wrap-agnostic — the drain writes files in logical order):
        #   all_tokens : the whole span -> q_start = start_slot, q_rows = qlen
        #   last_token : only the span's last token -> q_start = start_slot + qlen - 1, q_rows = 1
        #   non-emit   : q_start = -1, q_rows = 0 (dead q rows, unreferenced)
        if emit_q:
            if req_mode == "all_tokens":
                q_start, q_rows = start_slot, n
            else:
                q_start, q_rows = start_slot + n - 1, 1
            prefix_end = int(abs_end)
        else:
            q_start, q_rows, prefix_end = -1, 0, -1

        # LayerEntry COLLAPSE: ONE per-request record (this request's OWN 0-based layers in req_layers
        # order) instead of num_layers QKStepEntry objects here; the drain expands it off-loop into the
        # identical flat per-(req, layer) QKStepEntry list. Every field but `layer` is shared across
        # this request's layers, so the record is exact.
        records.append(QKReqCaptureRecord(
            req_id=str(req_id),
            k_start=int(start_slot), k_rows=int(n),
            q_start=int(q_start), q_rows=int(q_rows),
            prefix_end=int(prefix_end), num_computed=int(num_computed),
            layers=[int(L) for L in req_layers]))

        plans.append({"req_id": req_id, "n_rows": n, "layers": req_layers,
                      "hookq_mode": req_mode, "emit_q": emit_q})

    registry._qk_step_entries = records
    return plans


# ---------------------------------------------------------------------------
# Prefix-cache key reader (RESERVED for the deferred QK-ring prefix path). The
# per-step egress that used this was deleted with the QK ring port; the v1 ring
# path FAILS LOUD on a cached prefix instead (reader raises when first-step
# num_computed > 0). Kept as the reference impl for prepending [0, num_cached)
# cached keys into the k ring at drain time.
# ---------------------------------------------------------------------------


def _read_cached_keys_buffer(model_runner, module_name, req_idx: int,
                             num_cached: int, bounds_cache: dict | None = None):
    """Read the ``[0, num_cached)`` cached prefix keys for request ``req_idx``
    directly from the persistent paged KV cache + this step's block table — the
    OFF-GRAPH, post-forward analogue of the eager ``_read_cached_keys``.

    When prefix caching is active, the scheduler trims a prompt's shared prefix
    (``num_computed_tokens`` starts at the cached length C) so those C keys are
    NEVER forwarded → the static-buffer scatter never sees them. The eager path
    reads them back from paged KV mid-forward; buffer-mode egress runs post-forward
    (``get_forward_context()`` is gone), so we read the SAME data from persistent
    sources instead of an in-forward stash — which would graph-break the FULL decode
    graph (the reason ``_PREFIXK_STASH`` stayed off):

      * kv_cache    — ``compilation_config.static_forward_context[name]`` is the same
                      persistent Attention layer ``get_forward_context().no_compile_layers``
                      exposes; its ``.kv_cache`` blocks outlive the forward.
      * block table — ``input_batch.block_table`` is committed (H2D) in ``_prepare_inputs``
                      this step and indexed by the SAME input-batch slot ``req_idx`` the
                      routing uses, so row ``req_idx`` is current at egress.

    Returns a CLONE ``(num_cached, num_kv_heads*head_size)`` on the cache device, or
    None on any error (caller degrades to forwarded-only keys — never crashes). The
    bounds guard is load-bearing: an out-of-range block id makes the gather an
    UNRECOVERABLE device-side assert.
    """
    try:
        sfc = model_runner.vllm_config.compilation_config.static_forward_context
        layer = sfc.get(module_name)
        if layer is None:
            return None
        key_cache = key_cache_from_layer_kv(layer.kv_cache)
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape

        mgbt = model_runner.input_batch.block_table
        num_reqs = model_runner.input_batch.num_reqs
        block_table = mgbt.block_tables[0].get_device_tensor(num_reqs)
        num_blocks_needed = math.ceil(num_cached / block_size)
        block_ids = block_table[req_idx, :num_blocks_needed]
        # The OOB bounds guard is load-bearing (an out-of-range block id makes the gather an
        # UNRECOVERABLE device-side assert) but its `int(...)` costs TWO .item() device syncs.
        # block_ids depends only on (req_idx, num_blocks_needed), not the layer, so the check is
        # cached per (req_idx, num_blocks_needed): the first layer pays the sync, later layers reuse
        # it. Byte-identical — the bounds result is genuinely layer-invariant.
        ok = None
        ck = (req_idx, num_blocks_needed) if bounds_cache is not None else None
        if ck is not None:
            ok = bounds_cache.get(ck)
        if ok is None:
            ok = not (block_ids.numel() == 0 or int(block_ids.max()) >= num_blocks
                      or int(block_ids.min()) < 0)
            if ck is not None:
                bounds_cache[ck] = ok
        if not ok:
            return None

        prefix_keys = key_cache[block_ids].reshape(-1, num_kv_heads * head_size)
        return prefix_keys[:num_cached].detach().clone()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Pre-forward query bounds + cheap "any QK requested" check
# ---------------------------------------------------------------------------


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


# Sentinel routing key meaning "no request captures anything this step" (W1′
# idle-skip). A constant tuple, so it is identical across consecutive idle decode
# steps → the routing wrapper skips reset/build/upload after ONE transition
# zero-upload, and the resident (zeroed) slab replays as a no-op scatter. Distinct
# from any real per-request signature and from None (None forces a rebuild).
_IDLE_ROUTE_KEY = ("__idle__",)


def _capture_idle_key(model_runner, qsl_cpu, output_attr: str) -> Optional[tuple]:
    """W1′ idle-skip key for buffer-mode capture (QK + HS).

    Returns ``_IDLE_ROUTE_KEY`` only when NO request in the batch would emit a
    capture plan this step — a pure non-capturing step, e.g. every decode step of a
    ``hooks_on=prefill`` workload (the profiled qk-lasttok / hs-lasttok case). The
    routing wrapper then skips ``reset_pinned`` + ``_build_routing`` + ``upload``;
    the device slabs (zeroed on the active→idle transition) make the baked-in
    scatter a no-op, so the whole per-step routing tax disappears on idle steps.

    Returns ``None`` (force a rebuild — today's behaviour, byte-identical) the
    moment any request WOULD capture, or when the batch is unreadable. Cheap:
    scalar gating only (no tensor writes, no f-strings), early-exits to ``None`` on
    the first capturing request. Mirrors the active-request gate of
    ``_build_routing`` / ``_build_routing_hs`` up to the plan decision; a request
    that passes this gate but ultimately produces no plan (empty layer set / empty
    range) only costs a missed skip, never wrong data.
    """
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:  # noqa: BLE001
        return None
    if qsl_cpu is None or not req_ids:
        return None
    bs = len(qsl_cpu) - 1
    default_hooks_on = getattr(model_runner, "_default_hooks_on", "prefill")
    requests = model_runner.requests
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_state = requests.get(req_ids[i])
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get(output_attr) is None:
            continue
        hooks_on = extra.get("hooks_on", default_hooks_on)
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue
        # This request would capture this step → not idle → force a rebuild.
        return None
    return _IDLE_ROUTE_KEY


def install_prepare_inputs_routing(model_runner, worker, build_routing_fn,
                                   label: str = "qk", routing_key_fn=None) -> None:
    """Wrap ``model_runner._prepare_inputs`` to build + upload the capture routing
    right after vLLM finishes input prep — when ``input_batch`` and
    ``query_start_loc`` reflect THIS step (post ``_update_states``). The routing
    must land before the (possibly cudagraph-replayed) forward, and _prepare_inputs
    is the legal off-graph home for it; the resulting plans are stashed on the
    registry for the execute_model wrapper's post-forward egress.

    ``routing_key_fn(model_runner, registry, qsl_cpu)`` is the W1 invalidation-key
    source. When given (capture passes a ``_capture_idle_key``-based function), it
    overrides ``registry.routing_key`` so capture gets the idle-skip without the
    registry needing the worker's gating. Defaults to ``registry.routing_key``
    (what steer uses).

    Idempotent. Defensive: an internal vLLM surface, so most failures degrade to
    "no capture this step" rather than crashing the forward -- except ``RingBackpressureError``,
    which propagates by design (never-drop contract; see below).
    """
    flag = f"_vllm_hook_{label}_prep_wrapped"
    if getattr(model_runner, flag, False):
        return
    orig_prepare = getattr(model_runner, "_prepare_inputs", None)
    if orig_prepare is None or not callable(orig_prepare):
        print(f"[graph/install] no _prepare_inputs to wrap ({label}); routing disabled")
        return
    setattr(model_runner, flag, True)
    # DIAGNOSTIC (default off, byte-identical): force every step to re-route (disable the W1
    # idle-skip) so a profiling run gets a clean per-step routing sample at a fixed batch.
    # Re-uploading an identical routing changes no captured value — it only re-does redundant work.
    _no_skip = os.environ.get("VLLM_HOOK_ROUTE_NO_SKIP") == "1"

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
            # qsl_cpu is None when `_runner_qsl` cannot read input_batch / query_start_loc (any
            # read failure, including the relocated V2 model-runner surfaces); all routing
            # branches below degrade to `plans = []` for that step.
            width = _upload_width(model_runner, qsl_cpu, registry.cap)

            # W1 invalidation: when the key + upload width match last step, the device slabs
            # already hold an identical routing, so reset/build/upload is skipped and the step is
            # a pure graph replay. Steering's key is real on a stable batch (bit-identical every
            # decode step). Capture's W1' key returns _IDLE_ROUTE_KEY on a fully non-capturing step
            # (consecutive idle steps skip after one transition zero-upload) and None the moment
            # any request captures. A composition change or prefill<->decode flip moves the key and
            # forces a re-route.
            key = (routing_key_fn(model_runner, registry, qsl_cpu)
                   if routing_key_fn is not None
                   else registry.routing_key(model_runner, qsl_cpu)) \
                if qsl_cpu is not None else None
            if (not _no_skip
                    and key is not None
                    and key == getattr(registry, "_last_route_key", None)
                    and width == getattr(registry, "_last_route_width", None)):
                registry._pending_plans = getattr(registry, "_last_plans", [])
                PROF.incr("graph.route.skip")
                return result

            with PROF.timed("graph.route"):
                if getattr(registry, "gpu_routing", False):
                    # GPU-side routing: O(reqs) host work + a GPU scatter fills the slabs,
                    # replacing the O(num_layers x cap) host build. Byte-identical device slabs.
                    # STEER (VLLM_HOOK_STEER_GPU_ROUTING): SteerRegistry resolves configs itself
                    # (ignores build_routing_fn). CAPTURE (VLLM_HOOK_CAPTURE_GPU_ROUTING):
                    # HostRegistry runs build_routing_fn for the gating/plans, then scatters the
                    # capture-index. Each registry sets its own gpu_routing, so they don't cross.
                    registry._pending_assignments = []
                    plans = registry.build_and_upload_gpu(model_runner, qsl_cpu, width,
                                                          build_routing_fn)
                    PROF.incr("graph.route.gpu")
                elif getattr(registry, "incremental_enabled", False):
                    # Build plans every step (cheap O(reqs) scalar work) but write+upload ONLY the
                    # capture-index columns that changed since last step. Steady-state decode /
                    # idle steps change 0 columns -> 0 upload (a pure replay); a prefill / finish /
                    # condense touches O(changed).
                    registry._pending_assignments = []
                    plans = build_routing_fn(model_runner, registry, qsl_cpu) \
                        if qsl_cpu is not None else []
                    uploaded = registry.apply_incremental_routing(
                        registry._pending_assignments, width)
                    PROF.incr("graph.route.upload" if uploaded
                              else "graph.route.noupload")
                else:
                    # Legacy: reset+upload the full column prefix the (cudagraph-padded)
                    # forward reads — for decode ~max_graph_batch, not the full cap slab.
                    # tier-2 sub-timers (FINE only) decompose the per-fire routing cost into
                    # reset / host-build / H2D-upload so a profiling run can attribute the
                    # O(reqs x layers) scaling; strict no-op unless VLLM_HOOK_PROFILE_FINE=1.
                    with PROF.timed("graph.route.reset", tier=2):
                        registry.reset_pinned(width)
                    with PROF.timed("graph.route.buildfn", tier=2):
                        plans = build_routing_fn(model_runner, registry, qsl_cpu) \
                            if qsl_cpu is not None else []
                    with PROF.timed("graph.route.upload_h2d", tier=2):
                        registry.upload(width)
            registry._pending_plans = plans
            registry._last_route_key = key
            registry._last_route_width = width
            registry._last_plans = plans
            PROF.incr("graph.route.build")
        except RingBackpressureError:
            # NEVER-DROP: the HS capture ring is full and could not be relieved within the block
            # timeout (a mis-sized ring or a dead off-loop drain consumer). Do NOT clear the plans
            # and continue — that would silently drop a capturing request. PROPAGATE so the engine
            # step fails LOUD. The reserve genuinely BLOCKED on the engine thread inside
            # build_routing_fn (the off-loop consumer frees rows there); only an unrelievable full
            # ring reaches here. This branch is HS-only (QK has no ring, never raises it).
            raise
        except Exception as e:  # noqa: BLE001
            registry._pending_plans = []
            registry._last_route_key = None  # force a re-route after an error
            if hasattr(registry, "force_full_routing"):
                registry.force_full_routing()  # re-establish the whole slab next step
            # This handler serves QK, HS and steer alike, and a routing failure degrades
            # the step to no capture / no steering — silent degradation is undiagnosable.
            # _prepare_inputs runs EVERY step, so the PROF.incr call below is unconditional —
            # but PROF.incr itself is a no-op unless VLLM_HOOK_PROFILE=1 (see _profiler.py),
            # so the counter only records the true rate when profiling is on. The
            # human-readable warning below is one-shot regardless (or a persistent failure
            # would flood the log at step rate), so under the default (profiling off) it is
            # the entire signal for this failure.
            PROF.incr("graph.route.errors")
            if _capture_dbg.get("route_err_warn", 0) < 1:
                _capture_dbg["route_err_warn"] = 1
                print(f"[graph/install] WARNING: _prepare_inputs routing FAILED "
                      f"({label}: {e}); this step captures nothing. Warned once — "
                      f"PROF counter graph.route.errors carries the full count under "
                      f"VLLM_HOOK_PROFILE=1.",
                      flush=True)
        return result

    model_runner._prepare_inputs = wrapped_prepare_inputs
    print(f"[graph/install] _prepare_inputs routing wrapper installed ({label})")


# ---------------------------------------------------------------------------
# execute_model wrapper (never compiled — all per-request Python lives here)
# ---------------------------------------------------------------------------


def install_execute_model_wrapper(model_runner, worker) -> None:
    """Install the QK capture-RING routing (``_prepare_inputs`` wrapper) + the per-step drain
    (``execute_model`` wrapper). Idempotent.

    Routing runs after vLLM's input prep (so a NEW request's prefill routes correctly), reserving
    advancing shared-ring slots and scattering q + k into the per-layer q/k rings; the drain reads
    THIS step's newly-scattered ring region post-forward and writes it durably to disk (per-layer q +
    k raw files + shared sidecar) — no RPC/bank/egress copy-out (storage-only). Two modes: OFF-LOOP
    consumer thread (default) or SYNCHRONOUS on-loop drain (``VLLM_HOOK_RING_SYNC_DRAIN=1``, the
    fallback used for validation).
    """
    if getattr(model_runner, "_vllm_hook_qk_wrapped", False):
        return

    model_runner._vllm_hook_qk_wrapped = True

    # Surface worker fallback mode + default phase onto the runner so the routing helper reads them
    # without reaching through the worker.
    model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
    model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
    # Score is unsupported on the ring path; keep these so the routing gate reads them (a request
    # asking for score is refused there).
    model_runner._worker_score_mode = getattr(worker, "_score_mode_default", False)
    model_runner._worker_score_head = getattr(worker, "_score_head_default", 0)

    # Routing — runs after _update_states + input prep, so prefill routes correctly.
    # W1′: idle-skip key gates on output_qk (skips the per-step tax on non-capturing steps).
    def _qk_routing_key(_runner, _registry, _qsl):
        return _capture_idle_key(_runner, _qsl, "output_qk")

    install_prepare_inputs_routing(model_runner, worker, _build_routing, label="qk",
                                   routing_key_fn=_qk_routing_key)

    # --- Build the QK ring drain (capture-ring path, no bank/RPC): scatter -> ring -> drain -> disk.
    #   * OFF-LOOP (default): a dedicated CONSUMER THREAD owns the drain; the execute_model wrapper
    #     does an O(1) enqueue (this step's entries + a CUDA event recorded AFTER the scatter) and the
    #     thread does the D2H + write off the engine loop. advance_drain frees ring rows -> genuine
    #     reserve backpressure (never-drop).
    #   * SYNCHRONOUS (VLLM_HOOK_RING_SYNC_DRAIN=1): the per-step on-loop drain (fallback path);
    #     the .cpu() D2H is stream-ordered after the in-graph capture_qk scatter.
    registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
    ring = getattr(registry, "_qk_ring", None) if registry is not None else None
    drain = None
    _sync_drain = os.environ.get("VLLM_HOOK_RING_SYNC_DRAIN", "0") == "1"
    if registry is not None and ring is not None:
        from vllm_hook_plugins.graph.ring_drain_hs import _torch_dtype_name
        from vllm_hook_plugins.graph.ring_drain_qk import (
            MultiLayerQKRingDrain, OffLoopQKRingDrain)
        # layers = [(layer_num, q_buf, k_buf), ...]; layer_num is 0-based (== eager match_attn).
        layers = [(layer_num, host.q_buf, host.k_buf)
                  for layer_num, host in registry.iter_hosts()]
        buf_dtype = layers[0][1].dtype if layers else torch.float32
        q_dim = int(layers[0][1].shape[1]) if layers else 0
        k_dim = int(layers[0][2].shape[1]) if layers else 0
        base = os.environ.get("VLLM_HOOK_RING_DIR", "./qk_ring_dump")
        run_dir = os.path.join(base, f"tp_rank_{int(getattr(worker, 'rank', 0))}")
        header = {"dtype": _torch_dtype_name(buf_dtype),
                  "q_row_shape": [q_dim], "k_row_shape": [k_dim],
                  "q_dim": q_dim, "k_dim": k_dim,
                  "hookq_mode": getattr(worker, "hookq_mode", "all_tokens")}
        if _sync_drain:
            drain = MultiLayerQKRingDrain(ring, layers, run_dir, header)
            registry._qk_consumer = None
            _mode = "sync per-step"
        else:
            # Per-request delivery (GATED, default OFF): the consumer demuxes each step's q + k rows by
            # req_id into a PerRequestIndex + a FINISH signal drives assemble_qk delivery, INSTEAD of
            # writing the shared per-layer files. Default OFF = the shared-file QK drain, unchanged.
            _per_request = os.environ.get("VLLM_HOOK_RING_PER_REQUEST", "0") == "1"
            drain = OffLoopQKRingDrain(ring, layers, run_dir, header, per_request=_per_request)
            drain.start()                        # spin up the consumer BEFORE the first enqueue
            registry._qk_consumer = drain         # reserve-block reads is_alive() for the dead backstop
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_ring={drain._ring_depth}){_pr}"
        worker._qk_drain = drain
        worker._qk_run_dir = run_dir
        import atexit
        atexit.register(lambda d=drain: d.close())   # best-effort backstop; flush_ring is the contract
        print(f"[graph/install] QK ring drain ON -> {run_dir} "
              f"(R={ring.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install] no capture ring; QK drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        # Refresh mode/phase each step so a later worker mutation can't desync routing.
        model_runner._worker_hookq_mode = getattr(worker, "hookq_mode", "all_tokens")
        model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")
        model_runner._worker_score_mode = getattr(worker, "_score_mode_default", False)
        model_runner._worker_score_head = getattr(worker, "_score_head_default", 0)

        # Forward: _prepare_inputs (wrapped) builds+uploads routing (reserving ring slots), then the
        # graph replays and capture_qk scatters q + k into the per-layer rings.
        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        # Post-forward: hand THIS step's newly-scattered ring region to the drain. plans is non-empty
        # iff this step reserved ring rows (active step); idle steps skip via the W1' routing key.
        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_qk_drain", None)

        # Component-1 capture evidence (VHP prof_harvest): the off-loop ring path never runs the
        # eager register_forward_hook, so hook.fire.qk / captured.bytes.qk (emitted only there)
        # read 0 even though the ring captured + persisted this step. captured.bytes.qk is the
        # q + k_full bytes the drain writes to NVMe this step, sampled once (never per-append).
        if ring is not None and layers:
            _cap_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
            if _cap_rows > 0:
                _elt = layers[0][1].element_size()
                PROF.gauge("captured.bytes.qk",
                           float(_cap_rows) * len(layers) * (q_dim + k_dim) * _elt)
        if plans and drain is not None:
            # LayerEntry COLLAPSE: `_qk_step_entries` holds ONE QKReqCaptureRecord per request now; the
            # drain expands them into the flat QKStepEntry list off-loop (record_entries / _drain_item /
            # _demux_into_index).
            entries = getattr(registry, "_qk_step_entries", None) or []
            if _sync_drain:
                with PROF.timed("graph.drain"):
                    drain.record_entries(entries)
                    drain.drain_once()
            else:
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()               # current (forward) stream, after the scatter ops
                start_logical = getattr(registry, "_qk_step_start", None)
                n_rows = int(getattr(registry, "_qk_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(entries, start_logical, n_rows, event)

        # Per-request delivery: enqueue a FINISH for each request finished since the previous step
        # (finished_req_ids lists requests dropped from input_batch BEFORE this step, so their last
        # rows were already enqueued -> FIFO holds). OUTSIDE the `if plans` gate so a finish is never
        # lost on an idle step. No-op unless the drain is per-request.
        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        # hook.fire.qk: once per captured layer per FINISHED request -> harvest recovers the
        # capturing-request count (hook_fire_count / n_layers) as the per-request-KB denominator.
        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.qk", len(_fin_evidence) * len(layers))

        registry._pending_plans = []        # consume
        registry._qk_step_entries = []      # consume (the list is now owned by the queue item)
        registry._qk_step_start = None
        registry._qk_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install] execute_model wrapper installed (QK ring drain)")


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
    "_capture_idle_key",
]
