"""CUDA-graph hidden-state capture install — capture-ring path.

The in-graph ``capture_hs`` scatter writes each captured token's residual straight into a
persistent, fixed-size GPU RING (one parallel ring per layer, sharing ONE logical cursor) at an
advancing slot; a per-step drain reads the newly-written contiguous region and writes it durably
to local disk. No per-step egress copy-out, no clone bank, no RPC — the path is
scatter -> ring -> drain -> disk.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.graph import register_graph_ops
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.hosts import HSHookHost
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import ReqCaptureRecord
from vllm_hook_plugins.graph.ring_sizing import resolve_ring_bytes_auto
from vllm_hook_plugins.graph.install import (  # shared helpers (no op-mode coupling)
    _capture_idle_key,
    _resolve_max_num_batched_tokens,
    install_prepare_inputs_routing,
)
from vllm_hook_plugins.workers._common import iter_matched_modules
from vllm_hook_plugins.workers.probe_hidden_states_worker import match_layer

logger = logging.getLogger(__name__)


# RingBackpressureError is defined in gpu_capture_ring (imported above) so graph/install.py can
# re-raise it out of the shared routing wrapper without a circular import; re-exported here for the
# existing tests + the never-drop contract. See its docstring.

# Class-wrap bookkeeping, idempotent and reversible.
_WRAPPED_LAYER_CLASSES: Dict[type, Any] = {}
_HS_HOST_ATTR = "_vllm_hook_hs_host"     # per-instance HSHookHost
_capture_dbg = {"n": 0, "cap": 0}


def _require_buffer_mode_hs() -> None:
    """Buffer mode is the only FULL-cudagraph HS capture path. The PIECEWISE hs_probe
    op mechanism was removed; ``VLLM_HOOK_HS_CAPTURE`` is retained only so an explicit
    ``op`` request fails loud instead of silently running buffer."""
    mode = os.environ.get("VLLM_HOOK_HS_CAPTURE", "buffer").strip().lower()
    if mode not in ("", "buffer"):
        raise RuntimeError(
            f"VLLM_HOOK_HS_CAPTURE={mode!r} is no longer supported: the PIECEWISE hs_probe "
            "capture mode was removed. Buffer mode is the only FULL-cudagraph HS path; "
            "unset VLLM_HOOK_HS_CAPTURE or set it to 'buffer'.")

# Ring sizing is resolved by graph/ring_sizing.py::resolve_ring_bytes_auto — a FIXED 4 GiB ring by
# default (VLLM_HOOK_RING_GPU_BYTES), with the legacy VLLM_HOOK_CAPTURE_GPU_RESERVE_FRAC ratio kept
# for back-compat. See that module for the precedence + fit-check gate.


# Selective-drain census. The counts belong to the drain OBJECT (one per worker process, created
# at install), not a module-level int, so reading them from live state needs no global that
# survives a re-install or leaks between tests. Observation channel: a plain getter here plus a
# `collective_rpc("get_drain_row_counts")` on the worker.
def get_drain_row_counts(worker) -> dict:
    """Rows the HS ring drain ACTUALLY copied out of the ring / did not copy, this worker process.

    `hs.drain.rows_copied` is accumulated at the copy site inside `_read_segments`, never derived
    from what the requests asked for — without it, a selective-drain validation leg cannot tell
    "copied only the wanted tiles" from "silently copied everything" (both reconstruct
    byte-identically). `hs.drain.rows_skipped` is what an unconditional full drain of the same
    steps would have copied, minus that; it reads 0 whenever the full drain runs instead (flag off,
    per-request delivery, synchronous drain) — `selective` / `selective_disabled_reason` say which.
    `hs.drain.degenerate_steps` counts steps where an ARMED selective drain found every installed
    layer wanted over the whole span and took the flag-off fast path — the only witness separating
    "fast path fired every step" from "never fired" on an all-layers workload (both report
    `rows_skipped == 0`). All zeros when no ring drain is installed.
    """
    drain = getattr(worker, "_hs_drain", None)
    counts = getattr(drain, "row_counts", None)
    if drain is None or not callable(counts):
        return {"hs.drain.rows_copied": 0, "hs.drain.rows_skipped": 0,
                "hs.drain.degenerate_steps": 0,
                "selective": False, "selective_disabled_reason": None}
    return counts()



def _route_vectorized_enabled() -> bool:
    """Whether ``_build_routing_hs`` builds the routing plane in ONE shot (vectorized) rather than
    the legacy per-request torch scatter. Read ONCE at install and captured in the routing-wrapper
    closure, so it never costs a per-step env read. Byte-identical either way: only HOW the plane /
    entries / plans are built changes, never the values.

    ``VLLM_HOOK_ROUTE_VECTORIZED`` — DEFAULT OFF, kept opt-in: an A/B against the legacy loop found
    no collapse in the linear-in-N routing-build cost, because constructing the per-request
    ``LayerEntry`` dataclasses — not the plane-fill scatter this flag amortizes — is the dominant
    O(N) term, and it is shared by both paths. Do NOT flip the default without a winning A/B."""
    return os.environ.get("VLLM_HOOK_ROUTE_VECTORIZED", "0") == "1"


def _route_decode_cache_enabled() -> bool:
    """Whether the HS routing build reuses a cached per-request gating across steady-decode steps
    (VLLM_HOOK_ROUTE_DECODE_CACHE, default ON; kill switch =0). Read ONCE at install and threaded
    into ``_build_routing_hs`` via the ``decode_cache`` kwarg (mirrors ``_route_vectorized_enabled``),
    so the per-step routing build never re-reads the env. Byte-identical: the cache only skips
    RE-DERIVING routing that is constant for a request's lifetime — every emitted value (plane,
    records, plans, ring cursor) matches the slow build, so this is a pure host-cost reduction.
    """
    return os.environ.get("VLLM_HOOK_ROUTE_DECODE_CACHE", "1") != "0"



@dataclass
class _DecodeEntry:
    """Config-INVARIANT routing fields for one capturing request (constant for its lifetime). The
    COLUMN and the ring SLOT are deliberately NOT stored — both are recomputed each step (column live
    from qsl, slot fresh from the reserve), because condensation moves the column and the ring cursor
    advances every step."""
    __slots__ = ("layer_rows", "mode", "layers")
    layer_rows: object   # np.ndarray int64, 0-based registry rows (slow path's rows_layers order)
    mode: str            # "last_token" | "all_tokens"
    layers: list         # [L+1 for L in layer_rows] — the ReqCaptureRecord.layers template


def _wrap_layer_class(cls: type) -> None:
    """Class-wrap ``cls.forward`` to scatter the layer's residual stream into the
    static host buffer via the ``capture_hs`` op (NO splitting op → absorbed into the
    decode cudagraph).

    Idempotent per class. The scatter runs AFTER the original forward (we need its
    output). ``do_capture`` is an install-time constant, so this adds no
    data-dependent control flow to the traced region.
    """
    if cls in _WRAPPED_LAYER_CLASSES:
        return
    orig_forward = cls.forward
    _WRAPPED_LAYER_CLASSES[cls] = orig_forward

    def make_wrapped(orig_fwd):
        def wrapped(self, *args, **kwargs):
            out = orig_fwd(self, *args, **kwargs)
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
        return wrapped

    cls.forward = make_wrapped(orig_forward)


def install_hs_hosts(worker) -> Optional[HostRegistry]:
    """Install the CUDA-graph HS capture path via decoder-layer wrap.

    Runs from the ``load_model`` patch, BEFORE compile/capture. Builds a per-layer
    ``HSHookHost`` (static ``hs_buf``) + a ``HostRegistry`` for routing and class-wraps
    the layer to emit ``capture_hs`` (NO splitting op — absorbed into the decode
    cudagraph under FULL). Returns the registry; the worker installs the execute_model
    wrapper.
    """
    _require_buffer_mode_hs()  # op mode removed; fail loud on an explicit request

    model = getattr(worker.model_runner, "model", None)
    if model is None:
        print("[graph/install_hs] no model on model_runner; skip HS host install")
        return None

    register_graph_ops()

    tp_size = worker.parallel_config.tensor_parallel_size
    should_capture = tp_size <= 1 or worker.rank % tp_size == 0
    # Under TP the FULL decode cudagraph must be IDENTICAL across ranks. The capture_hs op reads
    # the decoder layer's POST-all_reduce residual and is baked INSIDE the compiled layer that
    # holds the TP collective, so baking it on rank 0 only breaks the NCCL graph-capture lockstep
    # -> engine-init hang at torch.cuda.synchronize() in the cudagraph-capture __enter__ (QK is
    # fine — it reads PRE-attention q/k, before the collective). Fix: BAKE the op on ALL ranks
    # (symmetric graph); only rank 0 produces real data (the residual is REPLICATED), so
    # non-capture ranks keep should_capture=False and their capture_index stays the zero sentinel
    # (a no-op discard). Kill switch VLLM_HOOK_HS_TP_SYMMETRIC=0. Diagnostic
    # VLLM_HOOK_HS_CAPTURE_ALL_RANKS=1 forces real capture on all ranks (the validation path).
    bake_op = should_capture
    if tp_size > 1 and os.environ.get("VLLM_HOOK_HS_TP_SYMMETRIC", "1") != "0":
        bake_op = True
    if os.environ.get("VLLM_HOOK_HS_CAPTURE_ALL_RANKS") == "1":
        should_capture = True
        bake_op = True

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

    # Rank-1c: writer PROCESS for off-GIL serialize+write (no-op unless VLLM_HOOK_WRITER_PROCESS=1).
    from vllm_hook_plugins.graph.writer_process import init_writer_process
    init_writer_process(worker)

    matched = list(iter_matched_modules(model, match_layer))
    if not matched:
        print("[graph/install_hs] no decoder layers matched LAYER_PATTERNS; "
              "HS graph capture inactive")
        worker._graph_registry = None
        return None

    device_t = next(model.parameters()).device

    # Static buffers + routing; the capture_hs scatter rides the decode cudagraph.
    registry = _install_hs_buffer(
        worker, model, matched, num_layers, hidden_size,
        should_capture, device_t, bake_op)
    worker._graph_registry = registry
    return registry


def _resolve_ring_rows(worker, num_layers, hidden_size, buf_dtype, device) -> tuple:
    """Size the shared GPU capture ring, resolved LAZILY at install (after vLLM carves KV).

    ``ring_bytes`` comes from ``resolve_ring_bytes_auto`` — a FIXED 4 GiB ring by default
    (``VLLM_HOOK_RING_GPU_BYTES``), or the legacy ``0.9 x reserve_frac x total_gpu`` ratio when
    ``VLLM_HOOK_CAPTURE_GPU_RESERVE_FRAC`` is set; each path applies its own fit/overcommit gate.
    ``R`` = per-layer ring rows = ``ring_bytes // (num_layers x hidden x dtype_size)`` — because the
    ``num_layers`` parallel per-layer rings share the budget and one shared cursor. Fails loud if
    ``R < 1`` (spec §8: no silent degrade). Returns ``(R, ring_bytes)``.
    """
    elem_size = torch.empty(0, dtype=buf_dtype).element_size()
    if str(device).startswith("cuda"):
        total_gpu = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        total_gpu = 1 << 30  # CPU (tests): a nominal 1 GiB budget
    try:
        gpu_util = float(getattr(worker.vllm_config.cache_config,
                                 "gpu_memory_utilization", 0.9))
    except Exception:  # noqa: BLE001
        gpu_util = 0.9
    ring_bytes = resolve_ring_bytes_auto(total_gpu, gpu_util)   # fixed 4 GiB default; legacy ratio opt-in
    per_layer_row_bytes = hidden_size * elem_size
    R = int(ring_bytes // (num_layers * per_layer_row_bytes))
    if R < 1:
        raise RuntimeError(
            f"HS capture ring too small: ring_bytes={ring_bytes} num_layers={num_layers} "
            f"hidden={hidden_size} dtype={buf_dtype} -> R={R} rows/layer (<1). Raise "
            f"VLLM_HOOK_RING_GPU_BYTES or reduce the model.")
    return R, ring_bytes


def _install_hs_buffer(worker, model, matched, num_layers, hidden_size,
                       should_capture, device, bake_op=None) -> Optional[HostRegistry]:
    """Build the HS capture-ring hosts + routing registry (no splitting op).

    Each matched decoder layer holds a persistent ``hs_buf`` ``(R+1, hidden)``: rows ``[0, R)``
    are the layer's ring slots and row ``R`` is the shared SENTINEL (pad / no-capture discard).
    ALL layers' rings advance in lockstep off ONE shared ``GpuCaptureRing`` logical cursor
    (``registry._hs_ring``) — every captured layer scatters the SAME tokens each step, so one
    reserve serves every layer. The ``HostRegistry`` still owns the per-(layer, token) routing
    slabs (``capture_index`` width = ``cap`` tokens) the ``capture_hs`` scatter reads on replay;
    those now carry ADVANCING ring slots, not batch positions. Buffers are built here — at
    load_model, before the cudagraph pool — so their data_ptrs stay fixed across replays.

    ``bake_op`` (default = ``should_capture``) installs the wrap+host+op so the scatter is
    baked into the graph. Under TP it is True on ALL ranks (symmetric graph — see
    ``install_hs_hosts``) while ``should_capture`` stays rank-0-only; a non-capture rank bakes the
    op against a sentinel-filled capture_index (no-op discard) and never drains.
    """
    if bake_op is None:
        bake_op = should_capture
    cap = _resolve_max_num_batched_tokens(worker)
    buf_dtype = model.dtype if hasattr(model, "dtype") \
        else next(model.parameters()).dtype

    # Shared capture-ring geometry: R rows/layer, sentinel row == R (== GpuCaptureRing.SENTINEL).
    R, ring_bytes = _resolve_ring_rows(worker, num_layers, hidden_size, buf_dtype, device)
    if R < cap:
        print(f"[graph/install_hs] WARNING: ring rows/layer R={R} < token cap={cap}; a single "
              f"max-token step may exceed the ring -> backpressure. Steady decode still fits.")

    registry: Optional[HostRegistry] = None
    ring: Optional[GpuCaptureRing] = None
    if bake_op:
        registry = HostRegistry(
            num_layers=num_layers, cap=cap, device=device,
            should_capture=should_capture,
        )
        # ONE shared logical cursor across the parallel per-layer rings. row_bytes/dtype/row_shape
        # describe a layer's row; ring.buf is NOT allocated (storage is the per-layer hs_bufs) —
        # we use only reserve/physical_slots/drained_segments/advance_drain/free_rows/SENTINEL.
        ring = GpuCaptureRing(row_bytes=hidden_size * torch.empty(0, dtype=buf_dtype).element_size(),
                              n_slots=R, device=device, dtype=buf_dtype, row_shape=(hidden_size,))
        registry._hs_ring = ring
        registry._hs_step_entries = []
        # Pad / no-capture lanes must route to the ring SENTINEL row (R), NOT 0: ring slot 0 is a
        # REAL storage row now, so a zero-filled lane would corrupt it. reset_pinned fills this.
        registry.sentinel_row = ring.SENTINEL
        # The capture-ring uses advancing positions, not column-diff, so the incremental / GPU
        # routers are disabled: the wrapper takes the legacy reset -> build -> upload branch.
        registry.incremental_enabled = False
        registry.gpu_routing = False
        # Prime the device slab + every pinned mirror to the sentinel so any pre-first-upload read
        # (e.g. the cudagraph capture pass, which skips routing) scatters to the discard row, never
        # a real slot 0.
        registry.capture_index_all.fill_(ring.SENTINEL)
        for _slot in registry._ring.slots:
            _slot["capture_index"].fill_(ring.SENTINEL)
        worker._capture_ring = ring

    n_hosts = 0
    for name, module, layer_num0 in matched:
        if bake_op and 0 <= layer_num0 < num_layers:
            # Pre-build the (R+1, hidden) ring buffer and hand it to the host (host.cap stays the
            # token cap so register_host's cap check + the routing slab width still match).
            hs_buf = torch.zeros(R + 1, hidden_size, dtype=buf_dtype, device=device)
            host = HSHookHost(
                module_name=name,
                layer_num=layer_num0,            # 0-based registry-slab row
                egress_layer_num=layer_num0 + 1,  # 1-based artifact layer_num
                cap=cap,
                hidden=hidden_size,
                dtype=buf_dtype,
                device=device,
                has_residual=1,                   # refined per-call in the wrap
                do_capture=True,
                hs_buf=hs_buf,
            )
            setattr(module, _HS_HOST_ATTR, host)
            registry.register_host(host)
            n_hosts += 1
        _wrap_layer_class(type(module))

    if registry is not None:
        registry.assign_views()
        buf_bytes = sum(
            h.hs_buf.numel() * h.hs_buf.element_size()
            for _, h in registry.iter_hosts()
        )
        print(f"[graph/install_hs] HS capture ring: {buf_bytes / (1024**2):.1f} MiB on {device} "
              f"(R={R} rows/layer, {num_layers} layers, ring_bytes={ring_bytes / (1024**2):.0f} "
              f"MiB budget, token cap={cap}, sentinel_row={registry.sentinel_row})")

    _tpsym = " (TP-symmetric op baked; DISCARD, no real capture on this rank)" \
        if (bake_op and not should_capture) else ""
    print(f"[graph/install_hs] buffer-mode HS capture wired: {n_hosts} host(s) over "
          f"{num_layers} layer slot(s); should_capture={should_capture}{_tpsym}; "
          f"hidden_size={hidden_size}; NO splitting op (rides decode cudagraph)")
    return registry


# ---------------------------------------------------------------------------
# Buffer-mode routing + egress + execute_model wrapper (HS analogue of the QK
# path in graph/install.py). All per-request Python lives in the wrapper, OUTSIDE
# the compiled region; the graph only READS the routing buffers we upload here.
# ---------------------------------------------------------------------------


def _ring_reserve_or_block(ring: GpuCaptureRing, n: int, consumer=None) -> int:
    """Reserve ``n`` contiguous ring rows for this step's capture, BLOCKING (polling) on ring-full
    rather than dropping (the never-drop contract). This runs on the ENGINE thread inside the
    ``_prepare_inputs`` routing wrapper.

    With the OFF-LOOP consumer drain, the block is GENUINE: while this polls, ``time.sleep`` releases
    the GIL, the consumer thread drains earlier (already-forwarded) steps and calls ``advance_drain``,
    which frees rows and lets the reserve succeed. No deadlock — the consumer only drains PAST steps
    whose forwards already completed, independent of this blocked engine step. With the SYNCHRONOUS
    drain (``consumer is None``) the previous step fully drains, so ``free_rows == R`` and a reserve
    of ``n <= cap <= R`` succeeds immediately; the block then only engages on a mis-sized ring.

    Fails loud (``RingBackpressureError``, which the routing wrapper RE-RAISES — never swallows) when
    the block cannot be relieved: a DEAD consumer (``is_alive()`` False → fail fast, don't wait the
    whole timeout) or a mis-sized ring (a single step needs more than the whole ring holds → after
    the bounded poll). ``VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S`` (default 10) / ``..._POLL_S`` (0.001)."""
    start = ring.reserve(n)
    if start is not None:
        return start
    timeout = float(os.environ.get("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", "10") or "10")
    poll = float(os.environ.get("VLLM_HOOK_RING_BACKPRESSURE_POLL_S", "0.001") or "0.001")
    PROF.incr("hs.ring.backpressure")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Dead-consumer backstop: if the off-loop drain died, no one will ever free rows — fail
        # fast rather than block the whole timeout. (Surfaces the consumer's own error too.)
        if consumer is not None and not consumer.is_alive():
            err = getattr(consumer, "error", None)
            raise RingBackpressureError(
                f"HS capture ring full and the off-loop drain consumer is DEAD: need {n} rows, "
                f"free={ring.free_rows()} of {ring.n_slots} rows/layer. Consumer error: {err!r}")
        time.sleep(poll)
        start = ring.reserve(n)
        if start is not None:
            return start
    raise RingBackpressureError(
        f"HS capture ring full: need {n} rows, free={ring.free_rows()} of {ring.n_slots} "
        f"rows/layer; reserve blocked past {timeout}s. The ring cannot hold this step — raise "
        f"VLLM_HOOK_RING_GPU_BYTES or the off-loop drain is not keeping up.")


def _build_routing_hs(model_runner, registry: HostRegistry, qsl_cpu: list,
                      vectorized: Optional[bool] = None,
                      decode_cache: Optional[bool] = None) -> list:
    """Capture-ring routing: map each captured token's batch column to an ADVANCING shared-ring
    slot (persists until the drain reads it), NOT the old batch-position row ``p+1``.

    All requested layers of a request scatter the SAME tokens to the SAME ring slots (parallel
    per-layer rings + ONE shared logical cursor), so ONE ``ring.reserve(n)`` per request serves
    every layer. ``all_tokens`` reserves the whole span (``n = end - start``); ``last_token``
    reserves ``n = 1`` and routes ONLY the span's last column to that slot (the rest stay at the
    SENTINEL, so no ring space is spent on tokens we won't keep).

    LayerEntry COLLAPSE: stashes ONE ``ReqCaptureRecord`` per capturing request on
    ``registry._hs_step_entries`` — carrying that request's OWN 1-based layer list — instead of
    fanning out ``num_layers`` ``LayerEntry`` objects on the engine loop (the O(reqs x layers)
    per-fire allocation a profile flagged as a co-dominant routing binder). The drain expands each
    record into the identical flat ``LayerEntry`` list OFF the loop, so the sidecar / demux are
    byte-for-byte unchanged. Returns a lightweight plan per active request (the wrapper's active/idle
    gate + W1' cache read it as truthy/empty).

    Gating (output_hidden_states filter, hooks_on prefill/decode/both) mirrors the eager hs_hook.
    Layer filter is 1-based (config layers) -> 0-based registry rows via ``ln-1``; the record's
    ``layers`` store the 1-based artifact numbers (``L+1``), matching the eager path's ``layer_num``.
    """
    registry._hs_step_entries = []
    # The OFF-LOOP consumer needs this step's shared-ring start slot + total reserved rows to read
    # exactly [start, start+rows) (the engine may reserve later steps ahead of the drain cursor, so
    # the whole pending region would over-read). Reset here; accumulated over the reserving requests.
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    # TP symmetry: a non-capture rank bakes the op but never routes — leave capture_index at the
    # sentinel so the scatter is a pure no-op discard (residual is replicated; rank 0 holds the
    # real data). The registry's upload is should_capture-gated too; this is explicit.
    if not registry.should_capture:
        return []
    ring: Optional[GpuCaptureRing] = getattr(registry, "_hs_ring", None)
    if ring is None:
        return []
    # The off-loop drain consumer (None on the synchronous path) — passed to the reserve-block so a
    # dead consumer fails fast/loud instead of blocking the whole timeout.
    consumer = getattr(registry, "_hs_consumer", None)
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1
    capture_index_pinned = registry.capture_index_pinned  # (num_layers, cap)
    cap = registry.cap

    # VLLM_HOOK_ROUTE_DECODE_CACHE: cached-gating decode fast-path wins even when
    # VLLM_HOOK_ROUTE_VECTORIZED is also set (checked first, before the vectorized dispatch).
    if decode_cache is None:
        decode_cache = _route_decode_cache_enabled()
    if decode_cache:
        return _build_routing_hs_decode_cache(model_runner, registry, qsl_cpu)

    # VLLM_HOOK_ROUTE_VECTORIZED: build the whole step's plane / entries / plans in ONE shot,
    # killing the O(reqs) per-request torch.tensor + scatter the legacy loop below pays. Same
    # gating, same reserve order (shared cursor), same values -> byte-identical device slab,
    # _hs_step_entries, _hs_step_start/_rows, and plans (test_route_vectorized_parity).
    if vectorized is None:
        vectorized = _route_vectorized_enabled()
    if vectorized:
        return _build_routing_hs_vectorized(
            model_runner, registry, qsl_cpu, ring, consumer, req_ids, bs,
            capture_index_pinned, cap)

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
        if not extra or extra.get("output_hidden_states") is None:
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

        # ---- ring reserve (shared cursor) + advancing slots ----
        # last_token keeps only the span's last token, so it needs a single ring row; all_tokens
        # keeps the whole span. Reserve ONCE (all layers share these slots) and BLOCK on full.
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot            # step's rows begin at the first reserve
        registry._hs_step_rows += n                          # total rows this step (all reserves)
        phys = ring.physical_slots(start_slot, n)           # n ints in [0, R)
        layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
        if req_mode == "last_token":
            # Only the span's last column -> the reserved slot; [start, end-1) stay SENTINEL.
            capture_index_pinned[layer_idx_t, end - 1] = int(phys[0])
        else:
            phys_t = torch.tensor(phys, dtype=torch.int64)
            capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        # LayerEntry COLLAPSE: ONE per-request record (this request's OWN 1-based layers in fan-out
        # order) instead of num_layers LayerEntry objects here; the drain expands it off-loop.
        records.append(ReqCaptureRecord(
            req_id=str(req_id), logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=[L + 1 for L in rows_layers]))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    registry._hs_step_entries = records
    return plans


def _build_routing_hs_vectorized(model_runner, registry: HostRegistry, qsl_cpu: list,
                                 ring: GpuCaptureRing, consumer, req_ids, bs: int,
                                 capture_index_pinned, cap: int) -> list:
    """Vectorized twin of ``_build_routing_hs``'s legacy loop — byte-identical, cheaper build.

    The legacy loop pays, PER capturing request, a ``torch.tensor(rows_layers)`` (+ a
    ``torch.tensor(phys)`` for all_tokens) and a torch advanced-index scatter into the pinned
    plane — an O(reqs) torch-dispatch cost that dominates routing, linear in concurrent capturing
    requests. Here a cheap Python gating pass (dict/attr reads only) reserves ring slots in batch
    order (IDENTICAL cursor / start_slot / ``_hs_step_start`` / ``_hs_step_rows`` to the legacy
    path — the never-drop reserve is unchanged)
    and accumulates flat ``(layer_row, col, slot)`` numpy triples; the plane is then written with
    ONE advanced-index assign. Requests occupy DISJOINT columns (qsl is cumulative) and a request's
    layer rows are distinct, so the flat triples carry NO duplicate ``(row, col)`` pairs — the
    single assign writes exactly the cells the per-request scatters would, to the same int64 values,
    leaving every other cell at the sentinel ``reset_pinned`` set. The per-request ``ReqCaptureRecord``
    order (request order) and each record's ``layers`` order (``[L+1 for L in rows_layers]``) — and
    ``plans`` — match the legacy path exactly, so the drain's off-loop expansion is byte-identical.
    """
    plans: list = []
    records: list = []
    # Flat plane triples across ALL requests -> ONE torch advanced-index assign at the end.
    rows_acc: list = []
    cols_acc: list = []
    slots_acc: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_hidden_states") is None:
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

        # Which 0-based registry rows this request wants (SAME expression as the legacy path, so
        # rows_layers order — and thus the LayerEntry order — is identical, set-iteration and all).
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers))
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
        if not rows_layers:
            continue

        # ---- ring reserve (shared cursor) + advancing slots — IDENTICAL to the legacy path ----
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = ring.physical_slots(start_slot, n)            # n ints in [0, R)

        # ---- collect flat plane triples (numpy, no per-request torch dispatch) ----
        rows_np = np.asarray(rows_layers, dtype=np.int64)
        nl = int(rows_np.shape[0])
        if req_mode == "last_token":
            # Legacy: capture_index_pinned[layer_idx_t, end-1] = int(phys[0]).
            rows_acc.append(rows_np)
            cols_acc.append(np.full(nl, end - 1, dtype=np.int64))
            slots_acc.append(np.full(nl, int(phys[0]), dtype=np.int64))
        else:
            # Legacy: capture_index_pinned[layer_idx_t[:,None], start:end] = phys_t[None,:], i.e.
            # cell (L, start+j) = phys[j] for every L in rows_layers, j in [0, n).
            phys_np = np.asarray(phys, dtype=np.int64)       # length n
            cols_span = np.arange(start, end, dtype=np.int64)  # length n
            rows_acc.append(np.repeat(rows_np, n))
            cols_acc.append(np.tile(cols_span, nl))
            slots_acc.append(np.tile(phys_np, nl))

        # LayerEntry COLLAPSE: ONE per-request record, request order then rows_layers order (matches
        # legacy); the drain expands it into the flat LayerEntry list off-loop.
        records.append(ReqCaptureRecord(
            req_id=str(req_id), logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=[L + 1 for L in rows_layers]))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    # ONE advanced-index assign fills the whole step's plane (no cross-request cell collisions).
    if rows_acc:
        rows_t = torch.from_numpy(np.concatenate(rows_acc))
        cols_t = torch.from_numpy(np.concatenate(cols_acc))
        slots_t = torch.from_numpy(np.concatenate(slots_acc))
        capture_index_pinned[rows_t, cols_t] = slots_t

    registry._hs_step_entries = records
    return plans


def _build_routing_hs_decode_cache(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """Cached-gating decode fast-path (VLLM_HOOK_ROUTE_DECODE_CACHE). Byte-identical to
    ``_build_routing_hs`` (vectorized=False): a STABLE-DECODE request (cache hit, this step contributes
    exactly one new token, not a prefill) skips the gating/layer-build and contributes its CACHED
    ``layer_rows`` + the LIVE column + a FRESH ring slot to one batched plane write; a NEW/CHANGED
    request runs the full slow body and (re)populates the cache. Requests are walked in ``input_batch``
    order and the ring is reserved in that order, so the shared cursor assigns the identical slots the
    slow path would — the byte-identity invariant.
    """
    registry._hs_step_entries = []
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    if not registry.should_capture:
        return []
    ring: Optional[GpuCaptureRing] = getattr(registry, "_hs_ring", None)
    if ring is None:
        return []
    consumer = getattr(registry, "_hs_consumer", None)
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []
    bs = len(qsl_cpu) - 1
    cap = registry.cap
    capture_index_pinned = registry.capture_index_pinned
    if not hasattr(registry, "_dc_entries"):
        registry._dc_entries = {}
    cache = registry._dc_entries

    live: set = set()
    plans: list = []
    records: list = []
    rows_acc: list = []
    cols_acc: list = []
    slots_acc: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        key = str(req_id)
        live.add(key)
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end <= start:
            continue
        end = min(end, cap)
        if end <= start:
            continue
        n_tokens = end - start
        is_prefill = len(req_state.output_token_ids) == 0
        entry = cache.get(key)

        # ---- FAST PATH: cached config + one new token + decoding -> only the slot is new. ----
        if entry is not None and n_tokens == 1 and not is_prefill:
            n = 1
            start_slot = _ring_reserve_or_block(ring, n, consumer)
            if registry._hs_step_start is None:
                registry._hs_step_start = start_slot
            registry._hs_step_rows += n
            col = end - 1                       # n_tokens==1 -> start == end-1 (both modes)
            phys0 = start_slot % ring.n_slots
            nl = int(entry.layer_rows.shape[0])
            rows_acc.append(entry.layer_rows)
            cols_acc.append(np.full(nl, col, dtype=np.int64))
            slots_acc.append(np.full(nl, phys0, dtype=np.int64))
            records.append(ReqCaptureRecord(
                req_id=key, logical_start=start_slot, n_rows=n, hs_mode=entry.mode,
                layers=entry.layers))
            plans.append({"req_id": req_id, "n_rows": n,
                          "layers": [L - 1 for L in entry.layers], "hs_mode": entry.mode})
            continue

        # ---- SLOW PATH: full gating build (identical to the legacy/vectorized body) + cache it. ----
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_hidden_states") is None:
            cache.pop(key, None)
            continue
        output_spec = extra.get("output_hidden_states")
        layer_filter = ({int(x) for x in output_spec}
                        if isinstance(output_spec, list) else None)
        hooks_on = extra.get("hooks_on",
                             getattr(model_runner, "_default_hooks_on", "prefill"))
        if hooks_on != "both":
            if hooks_on == "prefill" and not is_prefill:
                cache.pop(key, None)
                continue
            if hooks_on == "decode" and is_prefill:
                continue
        req_mode = extra.get("hs_mode",
                             getattr(model_runner, "_worker_hs_mode", "last_token"))
        if layer_filter is None:
            rows_layers = list(range(registry.num_layers))
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
        if not rows_layers:
            continue
        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = ring.physical_slots(start_slot, n)
        rows_np = np.asarray(rows_layers, dtype=np.int64)
        nl = int(rows_np.shape[0])
        if req_mode == "last_token":
            rows_acc.append(rows_np)
            cols_acc.append(np.full(nl, end - 1, dtype=np.int64))
            slots_acc.append(np.full(nl, int(phys[0]), dtype=np.int64))
        else:
            phys_np = np.asarray(phys, dtype=np.int64)
            cols_span = np.arange(start, end, dtype=np.int64)
            rows_acc.append(np.repeat(rows_np, n))
            cols_acc.append(np.tile(cols_span, nl))
            slots_acc.append(np.tile(phys_np, nl))
        layers_tmpl = [L + 1 for L in rows_layers]
        records.append(ReqCaptureRecord(
            req_id=key, logical_start=start_slot, n_rows=n, hs_mode=req_mode,
            layers=layers_tmpl))
        plans.append({"req_id": req_id, "n_rows": n, "layers": rows_layers,
                      "hs_mode": req_mode})
        # Cache the config-invariant fields for the request's next (decode) steps — but ONLY for
        # requests that capture on decode. hooks_on="prefill" (the capture DEFAULT) captures on prefill
        # and must capture NOTHING on decode; caching it would make the fast-path fire on decode. Leave
        # it uncached so its decode steps fall through to this slow body, which correctly skips via the
        # `hooks_on=="prefill" and not is_prefill` gate above. hooks_on in {both, decode} always caches.
        if hooks_on != "prefill":
            cache[key] = _DecodeEntry(layer_rows=rows_np, mode=req_mode, layers=layers_tmpl)
        else:
            cache.pop(key, None)

    # Evict entries whose request left input_batch (finish/abort) — no cross-step leak.
    if len(cache) > len(live):
        for k in list(cache):
            if k not in live:
                del cache[k]

    if rows_acc:
        rows_t = torch.from_numpy(np.concatenate(rows_acc))
        cols_t = torch.from_numpy(np.concatenate(cols_acc))
        slots_t = torch.from_numpy(np.concatenate(slots_acc))
        capture_index_pinned[rows_t, cols_t] = slots_t

    registry._hs_step_entries = records
    return plans


def install_execute_model_wrapper_hs(model_runner, worker) -> None:
    """Install the HS capture-ring routing (``_prepare_inputs`` wrapper) + the per-step drain
    (``execute_model`` wrapper). Idempotent.

    Routing runs after vLLM's input prep (so a NEW request's prefill routes correctly), reserving
    advancing ring slots; the drain reads THIS step's newly-scattered ring region post-forward and
    writes it durably to disk (per-layer raw files + shared sidecar) — no RPC/bank/egress copy-out.
    """
    if getattr(model_runner, "_vllm_hook_hs_wrapped", False):
        return

    model_runner._vllm_hook_hs_wrapped = True
    model_runner._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
    model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

    # Read VLLM_HOOK_ROUTE_VECTORIZED / VLLM_HOOK_ROUTE_DECODE_CACHE ONCE at install and pin them
    # in the closures below (mirrors ROUTE_NO_SKIP), so the per-step routing build never re-reads
    # the env.
    _route_vec = _route_vectorized_enabled()
    _route_dc = _route_decode_cache_enabled()

    # Routing — runs after _update_states + input prep, so prefill routes correctly.
    # W1′: idle-skip key gates on output_hidden_states (skips the per-step tax on
    # non-capturing steps; rebuilds the moment any request captures).
    def _hs_routing_key(_runner, _registry, _qsl):
        return _capture_idle_key(_runner, _qsl, "output_hidden_states")

    if _route_decode_cache_enabled():
        print("[graph/install_hs] HS routing decode-cache ENABLED "
              "(default ON; VLLM_HOOK_ROUTE_DECODE_CACHE=0 to disable)", flush=True)

    def _hs_build_routing(_runner, _registry, _qsl):
        return _build_routing_hs(_runner, _registry, _qsl,
                                  vectorized=_route_vec, decode_cache=_route_dc)

    install_prepare_inputs_routing(model_runner, worker, _hs_build_routing, label="hs",
                                   routing_key_fn=_hs_routing_key)

    # --- Build the multi-layer host drain (capture-ring path, no bank/RPC): scatter -> ring ->
    # drain -> disk. Two modes:
    #   * OFF-LOOP (default): a dedicated CONSUMER THREAD owns the drain; the execute_model wrapper
    #     does an O(1) enqueue (this step's entries + a CUDA event recorded AFTER the scatter) and the
    #     thread does the D2H + write off the engine loop, overlapping decode. advance_drain frees
    #     ring rows -> genuine reserve backpressure (never-drop).
    #   * SYNCHRONOUS (VLLM_HOOK_RING_SYNC_DRAIN=1): the per-step on-loop drain (fallback path). The
    #     .cpu() D2H is stream-ordered after the in-graph capture_hs scatter.
    registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
    ring = getattr(registry, "_hs_ring", None) if registry is not None else None
    drain = None
    _sync_drain = os.environ.get("VLLM_HOOK_RING_SYNC_DRAIN", "0") == "1"
    if registry is not None and ring is not None:
        from vllm_hook_plugins.graph.ring_drain_hs import (
            MultiLayerRingDrain, OffLoopRingDrain, _torch_dtype_name, record_captured_cells)
        hidden = int(worker._conf["hidden_size"])
        layers = [(host.egress_layer_num, host.hs_buf) for _, host in registry.iter_hosts()]
        buf_dtype = layers[0][1].dtype if layers else torch.float32
        base = os.environ.get("VLLM_HOOK_RING_DIR", "./hs_ring_dump")
        run_dir = os.path.join(base, f"tp_rank_{int(getattr(worker, 'rank', 0))}")
        header = {"dtype": _torch_dtype_name(buf_dtype),
                  "row_shape": [hidden], "hidden": hidden}
        if _sync_drain:
            drain = MultiLayerRingDrain(ring, layers, run_dir, header)
            registry._hs_consumer = None
            _mode = "sync per-step"
        else:
            # Per-request delivery (GATED, default OFF): the consumer demuxes each step's rows by
            # req_id into a PerRequestIndex + a FINISH signal drives assembly, INSTEAD of writing the
            # shared per-layer files. Default OFF = the shared-file drain, unchanged.
            _per_request = os.environ.get("VLLM_HOOK_RING_PER_REQUEST", "0") == "1"
            drain = OffLoopRingDrain(ring, layers, run_dir, header, per_request=_per_request)
            drain.start()                       # spin up the consumer thread BEFORE the first enqueue
            registry._hs_consumer = drain        # reserve-block reads is_alive() for the dead backstop
            _pr = " + per-request delivery" if _per_request else ""
            _mode = f"OFF-LOOP consumer thread (drain_ring={drain._ring_depth}){_pr}"
        # Say what the drain DECIDED, not what was asked for: it resolves selectivity at
        # construction (graph/ring_drain_hs._resolve_selective), and two configurations legitimately
        # refuse an armed flag and FULL-drain instead (per-request delivery, the synchronous drain).
        if getattr(drain, "selective", False):
            _mode += " + SELECTIVE drain (default ON; VLLM_HOOK_DRAIN_SELECTIVE=0 to disable)"
        elif getattr(drain, "selective_disabled_reason", None):
            _mode += " + selective drain has no effect here"
            # The flag defaults ON, so most drains that hit this branch were never "armed" by
            # anyone -- they simply run a config (per-request delivery, the sync drain) selective
            # drain does not reach, and there is nothing for an operator to act on. Only escalate to
            # a warning when the env was set EXPLICITLY; the default-driven case gets an info line.
            _explicit = os.environ.get("VLLM_HOOK_DRAIN_SELECTIVE") is not None
            if _explicit:
                logger.warning(
                    "selective drain (VLLM_HOOK_DRAIN_SELECTIVE=%s, set explicitly) has no effect "
                    "for this drain: %s. The full drain runs instead (every installed layer, every "
                    "row) -- byte-identical, but the Lever C saving is NOT in effect.",
                    os.environ.get("VLLM_HOOK_DRAIN_SELECTIVE"), drain.selective_disabled_reason)
                print("[graph/install_hs] *** selective drain requested but IGNORED: "
                      f"{drain.selective_disabled_reason} -> FULL drain ***", flush=True)
            else:
                logger.info(
                    "selective drain (VLLM_HOOK_DRAIN_SELECTIVE, default ON) has no effect for this "
                    "drain: %s. The full drain runs -- byte-identical, no action needed.",
                    drain.selective_disabled_reason)
                print("[graph/install_hs] selective drain (default) has no effect here: "
                      f"{drain.selective_disabled_reason} -> FULL drain", flush=True)
        else:
            _mode += " + selective drain OFF (VLLM_HOOK_DRAIN_SELECTIVE=0)"
        worker._hs_drain = drain
        worker._hs_run_dir = run_dir
        # atexit is a best-effort backstop only (the worker process is often killed, not joined —
        # so the parity oracle / caller MUST call flush_ring() to persist the sidecar).
        import atexit
        atexit.register(lambda d=drain: d.close())
        print(f"[graph/install_hs] HS ring drain ON -> {run_dir} "
              f"(R={ring.n_slots} rows/layer, {len(layers)} layers, {_mode})", flush=True)
    else:
        print("[graph/install_hs] no capture ring; HS drain NOT wired", flush=True)

    orig_execute_model = model_runner.execute_model

    def wrapped_execute_model(scheduler_output, *args, **kwargs):
        registry: Optional[HostRegistry] = getattr(worker, "_graph_registry", None)
        if registry is None or not registry.should_capture:
            return orig_execute_model(scheduler_output, *args, **kwargs)

        model_runner._worker_hs_mode = getattr(worker, "hs_mode", "last_token")
        model_runner._default_hooks_on = getattr(worker, "_default_hooks_on", "prefill")

        # Forward: _prepare_inputs (wrapped) builds+uploads routing (reserving ring slots), then
        # the graph replays and capture_hs scatters into the per-layer rings. Timed
        # (measurement-only): graph.forward includes routing (graph.route, nested) + the replay.
        with PROF.timed("graph.forward"):
            result = orig_execute_model(scheduler_output, *args, **kwargs)

        # Post-forward: hand THIS step's newly-scattered ring region to the drain. plans is non-empty
        # iff this step reserved ring rows (active step); idle steps skip via the W1' routing key.
        plans = getattr(registry, "_pending_plans", None) or []
        drain = getattr(worker, "_hs_drain", None)
        # LayerEntry COLLAPSE: `_hs_step_entries` holds ONE ReqCaptureRecord per request now; the
        # drain expands them into the flat LayerEntry list off-loop (record_entries / _drain_item).
        # Fetched here (ahead of the `if plans` block below), not only inside it, because the
        # captured-bytes gauge right below also needs it under selective drain.
        records = getattr(registry, "_hs_step_entries", None) or []

        # Component-1 capture evidence (VHP prof_harvest): the off-loop ring path never runs the
        # eager register_forward_hook, so hook.fire.hs / captured.bytes.hs (emitted only there)
        # read 0 even though the ring captured + persisted this step; report it here instead.
        # captured.bytes.hs is the bytes THIS drain actually writes to NVMe this step, which is NOT
        # `_cap_rows * len(layers)` (every installed layer) once selective drain is active: a subset
        # request only costs its OWN `n_rows * len(rec.layers)`, and `record_captured_cells` sums
        # exactly that in O(records), not the O(records x layers) `build_copy_plans` pays. The
        # degenerate all-layers case collapses to the old formula (every record tiles the whole
        # span), and when selective drain is not active the drain copies every layer regardless of
        # what any record names, so the old formula is correct there too. Sampled once per step.
        if ring is not None:
            _cap_rows = int(getattr(registry, "_hs_step_rows", 0) or 0)
            if _cap_rows > 0:
                if drain is not None and drain._selective_active():
                    _cells = record_captured_cells(records)
                else:
                    _cells = _cap_rows * len(layers)
                PROF.gauge("captured.bytes.hs", float(_cells) * float(ring.row_bytes))
        if plans and drain is not None:
            if _sync_drain:
                # SYNCHRONOUS: read + write on the engine loop (the fallback path). The .cpu() D2H is
                # stream-ordered after the in-graph scatter (same/default stream), so no event needed.
                with PROF.timed("graph.drain"):
                    drain.record_entries(records)
                    drain.drain_once()
            else:
                # OFF-LOOP: record a CUDA event on the forward stream AFTER the scatter, then O(1)
                # enqueue (records + this step's start slot / row count + event). The consumer thread
                # waits the event, D2Hs, expands + writes, and advance_drains — all off the engine loop.
                event = None
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record()             # current (forward) stream, after the scatter ops
                start_logical = getattr(registry, "_hs_step_start", None)
                n_rows = int(getattr(registry, "_hs_step_rows", 0) or 0)
                if n_rows > 0 and start_logical is not None:
                    drain.enqueue(records, start_logical, n_rows, event)

        # Per-request delivery: enqueue a FINISH for each request finished since the previous step.
        # In vLLM v1 `scheduler_output.finished_req_ids` lists requests finished BETWEEN the prior and
        # current step (`_update_states` drops them from input_batch BEFORE this step's forward), so a
        # request here finished at the PREVIOUS step — its last rows were already enqueued then, so
        # the FIFO invariant (rows before finish) holds. Runs OUTSIDE the `if plans` gate so a finish
        # is never lost on an idle (non-capturing) step. No-op unless the drain is per-request.
        if drain is not None and getattr(drain, "per_request", False):
            finished = getattr(scheduler_output, "finished_req_ids", None)
            if finished:
                for _rid in finished:
                    drain.enqueue_finish(_rid)

        # hook.fire.hs: once per captured layer per FINISHED request, so the harvest's
        # hook_fire_count / n_layers recovers the capturing-request count (its per-request-KB
        # denominator, warm-up-inclusive to match the cumulative captured.bytes numerator). Runs
        # every step, independent of per_request mode (a finish can land on an idle step).
        _fin_evidence = getattr(scheduler_output, "finished_req_ids", None)
        if _fin_evidence:
            PROF.incr("hook.fire.hs", len(_fin_evidence) * len(layers))

        registry._pending_plans = []       # consume
        registry._hs_step_entries = []     # consume (the list is now owned by the queue item)
        registry._hs_step_start = None
        registry._hs_step_rows = 0

        return result

    model_runner.execute_model = wrapped_execute_model
    print("[graph/install_hs] execute_model wrapper installed (HS ring drain)")


__all__ = ["install_hs_hosts", "install_execute_model_wrapper_hs"]
