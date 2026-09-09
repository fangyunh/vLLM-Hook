"""Multi-layer host drain for the QK capture ring (QK port of ring_drain_hs).

QK captures TWO tensors per token: post-RoPE ``q`` and ``k``, scattered by ``capture_qk`` into
per-layer static buffers at the SAME routed index — TWO parallel per-layer rings (``q_buf``,
``k_buf``) sharing ONE logical cursor. Each step reserves ``qlen`` rows (K needs every key); this
drain reads the new ``[drain, write)`` region from BOTH buffers per layer and appends to that
layer's two raw files plus one shared sidecar (``qk_ring_meta.jsonl``). The ``q`` slots on a
non-emit (``last_token`` mid-prefill) step are written but never referenced by the sidecar — dead,
harmless; only q_start/q_rows/prefix_end distinguish an emit_q step from a keep-K-only step.
Mirrors ``ring_drain_hs`` (sync + off-loop drains) and reuses its byte-sink helpers and off-loop
cross-stream discipline verbatim; only the per-layer buffer count (2, not 1) differs.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Dict, List, Optional, Tuple

import torch

from vllm_hook_plugins._profiler import PROF
from .gpu_capture_ring import GpuCaptureRing
from .per_request_delivery import PerRequestIndex
from .ring_drain_hs import (
    _MmapLayerWriter,
    _STOP,
    _DrainItem,
    _Finish,
    _dbg,
    _match_disk_route,
    _raw_bytes,
    _ring_debug,
    _sanitize_req_id,
    _torch_dtype_name,
)
from .ring_metadata import QKStepEntry, StepMeta, expand_qk_records, write_qk_sidecar

logger = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _resolve_qk_mmap_capacity_bytes(n_slots: int, row_bytes: int) -> int:
    """Per-layer raw-file mmap pre-size for a q OR k file (see ring_drain_hs._resolve_mmap_capacity_bytes):
    ``VLLM_HOOK_RING_MMAP_BYTES`` overrides outright, else ``max(2 GiB, n_slots * row_bytes)``. A
    STARTING size, not a hard cap — ``_MmapLayerWriter`` falls back to a plain append past it."""
    override = os.environ.get("VLLM_HOOK_RING_MMAP_BYTES")
    if override:
        return int(override)
    return max(2 * _GIB, int(n_slots) * int(row_bytes))


class _PerRequestQKDiskStaging:
    """QK DISK route's per-request staging (QK port of ``_PerRequestDiskStaging``): stream ONE
    request's demuxed q + k rows to its OWN per-request run_dir laid out exactly like the shared QK
    run — per-layer ``qk_q_layer_<L>.raw`` + ``qk_k_layer_<L>.raw`` (via reused
    :class:`_MmapLayerWriter`s) + a per-request ``qk_ring_meta.jsonl`` sidecar — so
    ``ring_reader.load_multilayer_qk_ring_artifact(run_dir)`` reconstructs that single request
    byte-identically.

    RELABEL INVARIANT: because only THIS request writes these files, each ``QKStepEntry``'s
    ``k_start`` / ``q_start`` is the RUNNING per-``(req, layer)`` row count into its own file (0, then
    n_rows, ...) — the exact offset the QK reader keys on, now scoped to one request. (The shared-file
    drain instead uses the global ring cursor as the slot; here we RELABEL to a per-request-local
    offset.) ``prefix_end`` (the request's cumulative key count) and ``num_computed`` (its cached-prefix
    length) are ALREADY per-request, so they pass through UNCHANGED — and the reader's first-step
    ``num_computed > 0`` deferral guard fires identically.

    Q is written COMPACTLY (only the emitted q rows, matching the host ``assemble_qk`` path), so a
    ``last_token`` non-emit step appends k only; its entry carries ``q_start=-1, q_rows=0``. K is
    appended EVERY step (its rows concatenate to ``k_full``).

    Written entirely on the drain's CONSUMER thread (one writer per request), so the row appends need
    no lock; the drain guards only the ``_disk_staging`` dict membership it lives in."""

    def __init__(self, req_id: str, run_dir: str, header: dict, capacity_bytes: int,
                 use_mmap: bool):
        self.req_id = str(req_id)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "qk_ring_meta.jsonl")
        self._capacity = int(capacity_bytes)
        self._use_mmap = bool(use_mmap)
        self._q_writers: Dict[int, _MmapLayerWriter] = {}   # layer -> q mmap writer
        self._k_writers: Dict[int, _MmapLayerWriter] = {}   # layer -> k mmap writer
        self._q_plain: Dict[int, str] = {}                  # layer -> q raw path (plain path)
        self._k_plain: Dict[int, str] = {}                  # layer -> k raw path (plain path)
        self._q_rows: Dict[int, int] = {}                   # layer -> cumulative q rows == next q_start
        self._k_rows: Dict[int, int] = {}                   # layer -> cumulative k rows == next k_start
        self._entries: List[QKStepEntry] = []
        self._closed = False

    def _q_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_q_layer_{layer}.raw")

    def _k_path(self, layer: int) -> str:
        return os.path.join(self.run_dir, f"qk_k_layer_{layer}.raw")

    def _append_one(self, writers: dict, plain: dict, path_fn, layer: int,
                    rows_cpu: torch.Tensor) -> None:
        data = _raw_bytes(rows_cpu)
        if self._use_mmap:
            w = writers.get(layer)
            if w is None:
                w = _MmapLayerWriter(path_fn(layer), self._capacity)
                writers[layer] = w
            w.append(data)
        else:
            p = plain.get(layer)
            if p is None:
                p = path_fn(layer)
                open(p, "wb").close()   # truncate up front: never append onto stale bytes
                plain[layer] = p
            with open(p, "ab") as f:
                f.write(data)

    def append(self, layer: int, q_rows_cpu, k_rows_cpu: torch.Tensor,
               prefix_end: int, num_computed: int) -> None:
        """Append ONE step's already-on-host rows for this ``(req, layer)`` to its per-request q/k
        files at the per-request-local relabeled offset, recording the matching ``QKStepEntry``. The
        bytes are materialized here (``_raw_bytes`` copies), so the caller's source view is fully
        consumed before the ring frees it — no clone needed. ``q_rows_cpu`` is None on a non-emit
        (``last_token`` mid-prefill) step: append k only, record ``q_start=-1, q_rows=0``."""
        k_start = self._k_rows.get(layer, 0)
        k_n = int(k_rows_cpu.shape[0])
        self._append_one(self._k_writers, self._k_plain, self._k_path, layer, k_rows_cpu)
        self._k_rows[layer] = k_start + k_n
        if q_rows_cpu is not None and int(q_rows_cpu.shape[0]) > 0:
            q_start = self._q_rows.get(layer, 0)
            q_n = int(q_rows_cpu.shape[0])
            self._append_one(self._q_writers, self._q_plain, self._q_path, layer, q_rows_cpu)
            self._q_rows[layer] = q_start + q_n
        else:
            q_start, q_n = -1, 0
        self._entries.append(QKStepEntry(
            req_id=self.req_id, layer=int(layer),
            k_start=int(k_start), k_rows=int(k_n),
            q_start=int(q_start), q_rows=int(q_n),
            prefix_end=int(prefix_end), num_computed=int(num_computed)))

    def close(self) -> None:
        """Finalize on the request's FINISH: msync+truncate every per-layer q/k mmap writer, then
        write this request's QK sidecar. Idempotent. TOLERATES A PARTIAL / ABORTED STAGING: the
        sidecar references ONLY the layers actually appended, and each writer
        close AND the sidecar write are BEST-EFFORT (a vanished run_dir / partial mmap must never raise
        out of ``_handle_finish`` and wedge the off-loop consumer)."""
        if self._closed:
            return
        self._closed = True
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            try:
                w.close()
            except Exception:            # noqa: BLE001 -- best-effort msync of a partial/aborted writer
                logger.exception(
                    "qk per-request staging: writer close failed for req %r (partial staging); "
                    "continuing", self.req_id)
        try:
            if os.path.isdir(self.run_dir):
                write_qk_sidecar(self.meta_path, [StepMeta(list(self._entries))], self.header)
        except Exception:                # noqa: BLE001 -- a partial/vanished dir must never wedge finish
            logger.exception(
                "qk per-request staging: sidecar write failed for req %r under %r (partial/aborted "
                "staging); delivery skipped", self.req_id, self.run_dir)

    def discard(self) -> None:
        """ABORT cleanup: release this request's open q/k writers WITHOUT writing a sidecar (an aborted
        request is never delivered/read), then remove its staging dir. Idempotent; best-effort."""
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            try:
                w.close()
            except Exception:  # noqa: BLE001 -- best-effort release of a partial mmap
                pass
        self._q_writers = {}
        self._k_writers = {}
        self._closed = True
        import shutil
        shutil.rmtree(self.run_dir, ignore_errors=True)


class MultiLayerQKRingDrain:
    """Drains a shared-cursor ``GpuCaptureRing`` across N per-layer ``(q_buf, k_buf)`` pairs.

    ``layers`` is ``[(layer_num, q_buf, k_buf), ...]`` in layer order (``layer_num`` is 0-based, ==
    the eager qkv_hook's ``match_attn`` layer number). Each drain appends the SAME ``[drain, write)``
    rows from every layer's q_buf and k_buf to that layer's two raw files (the shared logical row
    offset is the row offset into EVERY per-layer file — the invariant the reader keys on).
    ``record_entries`` queues this step's ``QKStepEntry`` records; ``drain_once`` copies the pending
    region, appends per layer (q + k), advances the shared drain cursor, and returns rows moved.
    ``close`` writes the accumulated shared sidecar.
    """

    def __init__(self, ring: GpuCaptureRing, layers: List[Tuple[int, torch.Tensor, torch.Tensor]],
                 run_dir: str, header: dict, setup_sink: bool = True):
        self.ring = ring
        self.layers = list(layers)
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.header = dict(header)
        self.meta_path = os.path.join(run_dir, "qk_ring_meta.jsonl")
        self.q_raw_paths = {ln: os.path.join(run_dir, f"qk_q_layer_{ln}.raw")
                            for ln, _, _ in self.layers}
        self.k_raw_paths = {ln: os.path.join(run_dir, f"qk_k_layer_{ln}.raw")
                            for ln, _, _ in self.layers}
        # mmap-NVMe raw sink, default OFF (VLLM_HOOK_RING_MMAP=1 opts in). Same semantics + fallback
        # as the HS drain (plain GIL-releasing write() is the default); one writer per q AND k file.
        # ``setup_sink=False`` (the per-request delivery mode, OffLoopQKRingDrain(per_request=True))
        # skips the shared per-layer raw files entirely — those rows are demuxed by req_id into a
        # PerRequestIndex instead, so opening/pre-sizing the shared sink would be pure waste. The
        # default (True) path is byte-for-byte identical to before this param existed.
        self._mmap_enabled = False
        self._q_writers: Dict[int, _MmapLayerWriter] = {}
        self._k_writers: Dict[int, _MmapLayerWriter] = {}
        if setup_sink:
            self._mmap_enabled = os.environ.get("VLLM_HOOK_RING_MMAP", "0") != "0"
            if self._mmap_enabled:
                try:
                    for ln, q_buf, k_buf in self.layers:
                        q_cap = _resolve_qk_mmap_capacity_bytes(
                            ring.n_slots, q_buf.shape[1] * q_buf.element_size())
                        k_cap = _resolve_qk_mmap_capacity_bytes(
                            ring.n_slots, k_buf.shape[1] * k_buf.element_size())
                        self._q_writers[ln] = _MmapLayerWriter(self.q_raw_paths[ln], q_cap)
                        self._k_writers[ln] = _MmapLayerWriter(self.k_raw_paths[ln], k_cap)
                except OSError as e:
                    logger.warning(
                        "qk ring mmap sink: failed to mmap raw file(s) under %s (%s); falling back to "
                        "the plain append path for the whole run (unset VLLM_HOOK_RING_MMAP to silence)",
                        run_dir, e)
                    for w in (*self._q_writers.values(), *self._k_writers.values()):
                        try:
                            w.close()
                        except Exception:  # noqa: BLE001
                            pass
                    self._q_writers = {}
                    self._k_writers = {}
                    self._mmap_enabled = False
            if not self._mmap_enabled:
                for p in (*self.q_raw_paths.values(), *self.k_raw_paths.values()):
                    open(p, "wb").close()
        self._steps: List[StepMeta] = []
        self._pending_entries: List[QKStepEntry] = []
        self._closed = False

    def record_entries(self, entries: List) -> None:
        # `entries` is per-request QKReqCaptureRecord (or already-flat QKStepEntry for a direct-drain
        # caller); expand_qk_records fans each record into the flat per-(req, layer) QKStepEntry
        # list. Runs on-loop for the sync drain; the off-loop path expands in the consumer thread.
        self._pending_entries.extend(expand_qk_records(entries))

    def _append(self, writers: dict, raw_paths: dict, ln: int, rows_cpu: torch.Tensor) -> None:
        data = _raw_bytes(rows_cpu)
        writer = writers.get(ln) if self._mmap_enabled else None
        if writer is not None:
            writer.append(data)
        else:
            with open(raw_paths[ln], "ab") as f:
                f.write(data)

    def drain_once(self) -> int:
        """Copy the ring's pending ``[drain, write)`` rows out of every per-layer q_buf AND k_buf,
        append them per layer, queue this step's sidecar entries, and advance the shared drain
        cursor. Returns rows moved (0 if nothing pending)."""
        moved = self.ring.pending_rows()
        if moved == 0:
            return 0
        segments = self.ring.drained_segments()
        for ln, q_buf, k_buf in self.layers:
            qp = [q_buf[s:e].detach().to("cpu") for s, e in segments]
            kp = [k_buf[s:e].detach().to("cpu") for s, e in segments]
            self._append(self._q_writers, self.q_raw_paths, ln,
                         qp[0] if len(qp) == 1 else torch.cat(qp, dim=0))
            self._append(self._k_writers, self.k_raw_paths, ln,
                         kp[0] if len(kp) == 1 else torch.cat(kp, dim=0))
        if self._pending_entries:
            self._steps.append(StepMeta(list(self._pending_entries)))
            self._pending_entries = []
        self.ring.advance_drain(moved)
        return moved

    def close(self) -> None:
        """Flush+truncate+release every mmap writer (q + k), then write the shared QK sidecar
        (idempotent; safe to call from ``flush_ring`` + atexit)."""
        if self._closed:
            return
        for w in (*self._q_writers.values(), *self._k_writers.values()):
            w.close()
        write_qk_sidecar(self.meta_path, self._steps, self.header)
        self._closed = True


class OffLoopQKRingDrain(MultiLayerQKRingDrain):
    """Off-loop (consumer-thread) sibling of ``MultiLayerQKRingDrain`` — the QK analogue of
    ``OffLoopRingDrain``.

    The engine loop, per active step, does an O(1) ``enqueue(entries, start_logical, n_rows, event)``
    and does NOT drain. A dedicated CONSUMER THREAD waits the step's scatter event, reads each
    per-layer q_buf AND k_buf ``[start_logical, start_logical+n_rows)`` region on a DEDICATED COPY
    STREAM (``record_stream`` guards the source rows), writes per-layer q + k raw + sidecar, then
    ``advance_drain(n_rows)`` — which FREES ring rows and so RELEASES the engine's reserve
    backpressure. Byte-identical to the sync path (reads only committed ``[drain, write)`` rows,
    fenced by the event; FIFO invariant keeps every file's row offset == the logical cursor).

    TWO consumer modes (``per_request``, default OFF — additive, the default path is unchanged; the
    QK port of ``OffLoopRingDrain``'s per-request delivery):
      * shared-file (default): appends each layer's drained q + k rows to its two raw files + sidecar.
      * per-request (``per_request=True``): demuxes each step's q + k rows BY req_id into a
        ``PerRequestIndex`` via the ``("q", layer)`` / ``("k", layer)`` staging convention
        (``_demux_into_index``) and consumes ``_Finish`` items (``enqueue_finish`` -> ``_handle_finish``
        -> ``mark_finished``) to drive per-request ``assemble_qk`` delivery, writing NO shared file.

    DISK SUB-ROUTE (WITHIN per-request mode; INACTIVE unless ``route_to_disk`` is called): a request the
    router marks streams its demuxed q + k rows to its OWN per-request run_dir
    (``_PerRequestQKDiskStaging``: per-layer ``qk_q_layer_<L>.raw`` + ``qk_k_layer_<L>.raw`` + a
    per-request QK sidecar) instead of the host-buffer index, and on ``_Finish`` the file is msync'd and
    handed to an ``OffloadProcess`` for transfer to ``dest``, then freed. All the concurrency fixes from
    ``OffLoopRingDrain`` are ported: id-divergence match (``_match_disk_route``), per-request finalize
    isolation, partial-staging tolerance, single-owner staging-dir lifecycle, host-residency abort marks.
    A per-request run with NO disk routes is byte-identical to the host-buffer path.
    """

    def __init__(self, ring, layers, run_dir: str, header: dict,
                 per_request: bool = False, index: Optional[PerRequestIndex] = None,
                 offload=None, disk_base: Optional[str] = None):
        # per_request (GATED, default OFF): when ON the consumer demuxes each step's q + k rows BY
        # req_id into a PerRequestIndex (the ("q", layer) / ("k", layer) staging convention) and
        # enqueue_finish() drives QK assembly, INSTEAD of writing the shared per-layer files. Default
        # OFF keeps the shared-file drain byte-for-byte unchanged (MultiLayerQKRingDrain's mmap sink
        # follows per_request via setup_sink=not per_request below).
        super().__init__(ring, layers, run_dir, header, setup_sink=not per_request)
        self.per_request = bool(per_request)
        self.index: Optional[PerRequestIndex] = (
            index if index is not None
            else (PerRequestIndex() if self.per_request else None))
        # DISK ROUTE (a per-request SUB-mode, default INACTIVE): identical maps + single-owner
        # staging-dir lifecycle as OffLoopRingDrain (HS). Both maps stay empty until route_to_disk()
        # is called, so a per_request run with NO disk routes is byte-identical to the host path.
        self._offload = offload
        self._disk_base = disk_base or os.path.join(run_dir, "perreq")
        self._disk_routed: Dict[str, str] = {}
        self._disk_staging: Dict[str, _PerRequestQKDiskStaging] = {}
        self._disk_delivered_src: Dict[str, str] = {}
        # DEFERRED settled-reclaim (rmtree-vs-offload-read race fix, mirror of OffLoopRingDrain):
        # delivered-source dirs clear_request_disk found the offload STILL READING (copytree in
        # flight) when a confirm TIMEOUT dropped the confirm-path unlink; reclaimed once the offload
        # SETTLES (_reclaim_settled_pending). EMPTY on the happy path -> a strict no-op. Guarded by
        # _index_lock.
        self._disk_reclaim_pending: Dict[str, str] = {}
        self._disk_aborted: set = set()
        self._host_aborted: set = set()
        # Per-(req_id, layer) RUNNING cumulative prefix_ends list, used to build the LAST-WRITE-WINS
        # kmeta the ("k", layer) note_rows stores (assemble_qk reads it at finish). Consumer-thread-
        # owned like the disk maps; guarded by _index_lock for uniformity, cleared per-req on finish.
        self._qk_kmeta: Dict[str, Dict[int, list]] = {}
        self._perreq_cap = int(os.environ.get(
            "VLLM_HOOK_RING_PERREQ_MMAP_BYTES", str(64 * 1024 * 1024)) or (64 * 1024 * 1024))
        self._perreq_mmap = os.environ.get("VLLM_HOOK_RING_MMAP", "0") != "0"
        # Drain-OWNED lock serializing EVERY access to the shared PerRequestIndex + disk/abort maps
        # (same discipline + rationale as OffLoopRingDrain: plain Lock, never nested, never held
        # across close/rmtree/copytree/D2H).
        self._index_lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        dev = self.layers[0][1].device if self.layers else torch.device("cpu")
        self._is_cuda = (dev.type == "cuda") and torch.cuda.is_available()
        self._stream = torch.cuda.Stream(dev) if self._is_cuda else None
        self._ring_depth = max(1, int(os.environ.get("VLLM_HOOK_CAPTURE_DRAIN_RING", "3") or "3"))
        self._copy_events = ([torch.cuda.Event() for _ in range(self._ring_depth)]
                             if self._stream is not None else [])
        self._ring_idx = 0
        # Per-layer PERSISTENT pinned host buffers (q + k), reused each step (grown on demand).
        self._q_pinned: dict = {ln: None for ln, _, _ in self.layers}
        self._k_pinned: dict = {ln: None for ln, _, _ in self.layers}
        self._thread = threading.Thread(
            target=self._run, name="vllm-hook-qk-ring-drain", daemon=True)
        self._started = False
        self._error: Optional[BaseException] = None

    # ---- engine side (O(1)) ----
    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def is_alive(self) -> bool:
        return bool(self._started and self._thread.is_alive())

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    def enqueue(self, entries: list, start_logical: int, n_rows: int, event=None) -> None:
        """O(1) hand-off. ``entries`` ownership TRANSFERS to the queue item (the caller reassigns
        ``registry._qk_step_entries = []``, so the old list is owned solely here)."""
        with PROF.timed("graph.enqueue"):
            self._q.put(_DrainItem(entries, int(start_logical), int(n_rows), event))

    def enqueue_finish(self, req_id) -> None:
        """O(1) hand-off of a per-request FINISH (reuses the HS ``_Finish`` item). NO-OP unless
        per_request mode is on. Must be enqueued AFTER the request's last row-entries so the consumer
        marks it finished only once every row is drained/noted (FIFO invariant)."""
        if not self.per_request:
            return
        self._q.put(_Finish(str(req_id)))

    def route_to_disk(self, req_id, dest, offload=None) -> None:
        """SEAM for the router: mark ``req_id`` for the per-request DISK route — its q + k rows stream
        to its own NVMe run_dir and, on finish, the file is offloaded to ``dest`` — INSTEAD of the
        host-buffer PerRequestIndex. NO-OP unless per_request mode is on. MUST be called BEFORE the
        request's rows reach the consumer (the router runs at request-start). Registered under
        ``_index_lock``; lazily starts an OffloadProcess on first use unless one is injected.

        The lazy OffloadProcess is CONSTRUCTED OFF ``_index_lock`` (the process backend spawns a
        child + threads; building under the lock would stall the consumer) then adopted under the
        lock only if still absent -- a racing route finds one set and DISCARDS its loser (closes it),
        so exactly one is ever adopted (no double-construct leak)."""
        if not self.per_request:
            return
        req_id = str(req_id)
        # Build any lazily-created OffloadProcess OFF the lock (peek is racy-but-safe: a lost race
        # just builds a spare that is closed below). An injected `offload` never needs a build.
        new_offload = None
        if offload is None and self._offload is None:
            from vllm_hook_plugins.graph.offload_process import OffloadProcess
            use_proc = os.environ.get("VLLM_HOOK_OFFLOAD_PROCESS", "0") == "1"
            new_offload = OffloadProcess(use_process=use_proc)
        with self._index_lock:
            if offload is not None:
                self._offload = offload
            elif self._offload is None and new_offload is not None:
                self._offload = new_offload
                new_offload = None   # adopted -> don't close it below
                import atexit
                atexit.register(self._offload.close)
            self._disk_routed[req_id] = str(dest)
        if new_offload is not None:
            new_offload.close()      # lost the construct race (another route adopted one) -> discard
        if _ring_debug():
            _dbg(f"qk route_to_disk: req_id={req_id!r} (EXTERNAL) dest={dest!r} "
                 f"offload={type(self._offload).__name__}")

    def disk_residency(self) -> int:
        """Number of disk-routed requests still holding per-request staging state. Drops to 0 once
        every routed request has finished (close+offload+free)."""
        with self._index_lock:
            return len(self._disk_staging)

    def unlink_delivered_source(self, req_id) -> bool:
        """Remove the SERVER-side per-request staging SOURCE dir for a DELIVERED disk-routed request,
        reclaiming live NVMe. The durable CLIENT dest copy is untouched. Idempotent -> False when
        there is nothing recorded."""
        req_id = str(req_id)
        with self._index_lock:
            src = self._disk_delivered_src.pop(req_id, None)
        if src is None:
            return False
        import shutil
        shutil.rmtree(src, ignore_errors=True)
        return True

    def _reclaim_settled_pending(self) -> None:
        """DEFERRED settled-reclaim (rmtree-vs-offload-read race fix; mirror of OffLoopRingDrain):
        rmtree each parked delivered-source whose offload has now SETTLED. Populated ONLY by
        ``clear_request_disk`` when it found the offload in flight; EMPTY on the happy path -> a
        strict no-op. Called from the consumer loop (per item) and ``finalize_all`` (shutdown).
        Snapshot under ``_index_lock``, ``settled()`` + rmtree OFF the lock. A never-settling offload's
        source is LEFT rather than rmtree'd mid-copy."""
        off = self._offload
        if off is None:
            return
        with self._index_lock:
            pending = list(self._disk_reclaim_pending.items())
        if not pending:
            return
        settled_ids = [ext for ext, _ in pending if off.settled(ext)]
        if not settled_ids:
            return
        to_rm = []
        with self._index_lock:
            for ext in settled_ids:
                src = self._disk_reclaim_pending.pop(ext, None)
                if src is not None:
                    to_rm.append((ext, src))
        import shutil
        for ext, src in to_rm:
            shutil.rmtree(src, ignore_errors=True)
            if _ring_debug():
                _dbg(f"qk reclaim settled delivered-src: req={ext!r} src={src!r}")

    def mark_host_aborted(self, req_id) -> None:
        """ABORT cleanup for the HOST (RPC) route: mark ``req_id`` so the consumer never (re-)stages
        its drained rows into the ``PerRequestIndex`` after ``clear_ring_request`` freed its entry.
        MARK ONLY A REQUEST WITH LIVE HOST STATE (else a completed request would leave a stale mark no
        ``_Finish`` prunes). The index is keyed by the INTERNAL id, this abort id is EXTERNAL -> match
        with the exact-or-``{ext}-`` rule. NO-OP unless per_request mode is on. The index tuple layer
        keys (``("q"/"k", layer)``) do not affect the req_id match."""
        if not self.per_request or self.index is None:
            return
        req_id = str(req_id)
        with self._index_lock:
            live = any(_match_disk_route(rid, (req_id,)) is not None
                       for rid in self.index.live_req_ids())
            if live:
                self._host_aborted.add(req_id)
        if _ring_debug():
            _dbg(f"qk mark_host_aborted: req={req_id!r} live={live} "
                 f"host_aborted={list(self._host_aborted)}")

    def clear_request_disk(self, req_id) -> None:
        """ABORT cleanup for the DISK route (single-owner staging-dir lifecycle): MARK the request
        aborted; do NOT destroy its live staging. Only the CONSUMER thread ever creates, writes, or
        deletes a per-request staging dir, so the engine-thread abort must never rmtree a dir the
        consumer might still be demuxing this request's remaining rows into. Under ``_index_lock``:
        pop ``_disk_routed`` and, iff there is live staging or a still-live route, add the EXTERNAL id
        to ``_disk_aborted`` — the consumer DISCARDs its staging on this request's ``_Finish`` (or at
        ``finalize_all``). A recorded ``_disk_delivered_src`` means a finish already finalized +
        SUBMITTED this dir (FIFO: consumer done WRITING) -- but the OFFLOAD thread may still be READING
        it (copytree) if a confirm TIMEOUT skipped the confirm-path unlink, so it is rmtree'd here ONLY
        once the offload has SETTLED; if still in flight it is parked in ``_disk_reclaim_pending`` and
        reclaimed by ``_reclaim_settled_pending`` once the offload settles -- never mid-copy, never
        leaked. Strict no-op when per_request is off."""
        if not self.per_request:
            return
        req_id = str(req_id)
        marked = False
        with self._index_lock:
            routed = self._disk_routed.pop(req_id, None)
            src = self._disk_delivered_src.pop(req_id, None)
            if req_id in self._disk_staging or routed is not None:
                self._disk_aborted.add(req_id)
                marked = True
        # Only rmtree the delivered-source once the offload has SETTLED (no longer reading it);
        # otherwise defer to _reclaim_settled_pending (never rmtree mid-copytree, never leak). Decided
        # OFF the lock: settled() takes the offload's OWN lock -- keep _index_lock unheld across it and
        # rmtree. Happy path: src is None here (confirm success already unlinked) -> inert.
        reclaimed = False
        deferred = False
        if src is not None:
            if self._offload is None or self._offload.settled(req_id):
                import shutil
                shutil.rmtree(src, ignore_errors=True)   # settled offload -> safe to reclaim now
                reclaimed = True
            else:
                with self._index_lock:
                    self._disk_reclaim_pending[req_id] = src   # rmtree once the offload settles
                deferred = True
        if _ring_debug():
            _dbg(f"qk clear_request_disk MARK-abort: req={req_id!r} marked={marked} "
                 f"delivered_src_reclaimed={reclaimed} deferred_reclaim={deferred} "
                 f"(consumer owns the staging-dir discard)")

    # ---- consumer thread ----
    def _finalize_finish_isolated(self, req_id) -> None:
        """Run ``_handle_finish`` under PER-REQUEST FINALIZE ISOLATION: a finalize error (a partially
        staged aborted disk request whose ``close()``/offload raises, a double-submit, a marshal error)
        must fail for THAT request ONLY — caught, logged LOUD with the req_id, the consumer CONTINUES.
        CONTRAST the DRAIN path (``_drain_item``): a drain error never advances the ring cursor, so the
        never-drop guarantee must fail LOUD and stay FATAL to ``_run``'s outer handler."""
        try:
            self._handle_finish(req_id)
        except Exception:  # noqa: BLE001 -- isolate ONE request's finalize; never wedge the consumer
            logger.exception(
                "qk off-loop ring drain: per-request FINALIZE failed for req_id=%r; that request's "
                "delivery is dropped, the consumer continues", req_id)
            if _ring_debug():
                _dbg(f"qk finish FAILED (isolated, consumer continues): req_id={req_id!r}")

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is _STOP:
                    self._q.task_done()
                    break
                try:
                    if isinstance(item, _Finish):
                        # PER-REQUEST FINALIZE ISOLATION (never wedges the consumer).
                        self._finalize_finish_isolated(item.req_id)
                    else:
                        # DRAIN stays fatal: a failure here never advances the ring cursor.
                        self._drain_item(item)
                    # Deferred settled-reclaim of any delivered-source parked by a confirm-timeout
                    # abort. No-op when nothing is pending (the happy path); never raises.
                    self._reclaim_settled_pending()
                finally:
                    self._q.task_done()
        except BaseException as e:  # noqa: BLE001 — surface + let backpressure fail loud
            self._error = e
            logger.exception("qk off-loop ring drain consumer thread died")

    def _pinned_buf(self, cache: dict, ln: int, n_rows: int, width: int, dtype) -> torch.Tensor:
        buf = cache[ln]
        if buf is None or buf.shape[0] < n_rows:
            buf = torch.empty(n_rows, width, dtype=dtype, pin_memory=self._is_cuda)
            cache[ln] = buf
        return buf[:n_rows]

    def _read_segments(self, segments: List[Tuple[int, int]], event):
        """D2H each per-layer q_buf AND k_buf ``[segments]`` into contiguous (LOGICAL-order) host
        buffers. cuda: on the dedicated copy stream (wait the scatter event, ``record_stream`` the
        source, K-deep event ring). cpu (tests): plain ``.to('cpu')``. Returns
        ``[(ln, q_rows_cpu, k_rows_cpu), ...]``."""
        total = sum(e - s for s, e in segments)
        if self._stream is not None:
            slot = self._ring_idx
            stale = self._copy_events[slot] if slot < len(self._copy_events) else None
            if stale is not None:
                stale.synchronize()
            if event is not None:
                self._stream.wait_event(event)
            pieces = []
            with torch.cuda.stream(self._stream):
                for ln, q_buf, k_buf in self.layers:
                    qbuf = self._pinned_buf(self._q_pinned, ln, total, q_buf.shape[1], q_buf.dtype)
                    kbuf = self._pinned_buf(self._k_pinned, ln, total, k_buf.shape[1], k_buf.dtype)
                    off = 0
                    for s, e in segments:
                        n = e - s
                        qsrc, ksrc = q_buf[s:e], k_buf[s:e]
                        qbuf[off:off + n].copy_(qsrc, non_blocking=True)
                        kbuf[off:off + n].copy_(ksrc, non_blocking=True)
                        qsrc.record_stream(self._stream)
                        ksrc.record_stream(self._stream)
                        off += n
                    pieces.append((ln, qbuf, kbuf))
            done = self._copy_events[slot] if slot < len(self._copy_events) else None
            if done is not None:
                done.record(self._stream)
                done.synchronize()
            else:
                self._stream.synchronize()
            self._ring_idx = (slot + 1) % self._ring_depth
            return pieces
        # CPU path (tests)
        if event is not None:
            try:
                event.synchronize()
            except Exception:  # noqa: BLE001 — CPU stub events
                pass
        out = []
        for ln, q_buf, k_buf in self.layers:
            qp = [q_buf[s:e].detach().to("cpu") for s, e in segments]
            kp = [k_buf[s:e].detach().to("cpu") for s, e in segments]
            out.append((ln,
                        qp[0] if len(qp) == 1 else torch.cat(qp, dim=0),
                        kp[0] if len(kp) == 1 else torch.cat(kp, dim=0)))
        return out

    def _drain_item(self, item: _DrainItem) -> None:
        ring = self.ring
        assert item.start_logical == ring._drain, (
            f"FIFO drain violation: item.start_logical={item.start_logical} != "
            f"ring._drain={ring._drain}")
        segments = ring.segments_at(item.start_logical, item.n_rows)
        with PROF.timed("bank.consumer.d2h"):
            pieces = self._read_segments(segments, item.event)
        if self.per_request:
            # Per-request delivery: split this step's q + k rows by req_id into the PerRequestIndex.
            # (_demux_into_index expands item.entries -> flat QKStepEntry itself, off-loop.)
            self._demux_into_index(item, pieces)
        else:
            # Shared-file path (default): append every layer's q + k rows to its two raw files.
            for ln, q_rows, k_rows in pieces:
                self._append(self._q_writers, self.q_raw_paths, ln, q_rows)
                self._append(self._k_writers, self.k_raw_paths, ln, k_rows)
            # LayerEntry COLLAPSE: expand this step's per-request records into the flat per-(req,
            # layer) QKStepEntry list OFF the engine loop (here, on the consumer thread) — same fields,
            # same order the on-loop fan-out produced. StepMeta / sidecar bytes stay byte-identical.
            entries = expand_qk_records(item.entries)
            if entries:
                self._steps.append(StepMeta(entries))
        # Free the rows LAST — only after the D2H landed AND the rows were consumed, so the engine can
        # never scatter into a physical slot the consumer is still reading (never-drop + no torn read).
        ring.advance_drain(item.n_rows)

    def _demux_into_index(self, item: _DrainItem, pieces) -> None:
        """Slice each layer's contiguous drained q + k host rows by each ``QKStepEntry``'s req_id range
        and stage them in the ``PerRequestIndex`` under the two-stream convention: k rows under
        ``("k", layer)`` EVERY step (with a LAST-WRITE-WINS cumulative ``prefix_ends`` kmeta), q rows
        under ``("q", layer)`` only on emit steps. ``assemble_qk`` rebuilds ``k_all`` from those.

        ``pieces`` are ``(layer, q_rows, k_rows)`` holding this step's ``[start_logical,
        start_logical+n_rows)`` region in LOGICAL order, so an entry for ``(req, layer)`` occupies host
        offset ``entry.k_start - item.start_logical`` in the k buffer (and ``entry.q_start -
        item.start_logical`` in the q buffer for the emitted rows). Host-buffer slices are CLONED (the
        source is a reused pinned buffer / ring view the engine may overwrite). DISK-routed slices are
        written to the request's per-request q/k files synchronously here (``_raw_bytes`` copies),
        consuming the view before ``advance_drain`` — no clone, never entering the host index.

        When no request is disk-routed (the default per_request path), every entry takes the
        clone+note branch, byte-identical to the shared-file reconstruction."""
        by_layer = {ln: (q, k) for ln, q, k in pieces}
        base = int(item.start_logical)
        # LayerEntry COLLAPSE: expand this step's per-request records into the flat per-(req, layer)
        # QKStepEntry list OFF the engine loop (here, on the consumer thread) — same fields + order the
        # on-loop fan-out produced, so the demux slices exactly the rows it did before. Heterogeneous
        # per-request layer sets are preserved: each record carries its own `layers`, so each entry's
        # (req_id, layer) range is that request's own.
        entries = expand_qk_records(item.entries)
        with self._index_lock:
            routed_keys = tuple(self._disk_routed) if self._disk_routed else ()
        any_disk = bool(routed_keys)
        # (req_id, layer, q_clone_or_None, k_clone, prefix_end) — cloned OFF the lock; noted UNDER it.
        staged = []
        for e in entries:
            qk = by_layer.get(e.layer)
            if qk is None:
                continue          # entry's layer not among the drained layers (should not happen)
            q_layer_rows, k_layer_rows = qk
            k_off = int(e.k_start) - base
            k_slice = k_layer_rows[k_off:k_off + int(e.k_rows)]
            if int(e.q_rows) > 0 and int(e.q_start) >= 0:
                q_off = int(e.q_start) - base
                q_slice = q_layer_rows[q_off:q_off + int(e.q_rows)]
            else:
                q_slice = None
            # e.req_id is the INTERNAL '{external}-{rand}' under serve; the disk routes are keyed by
            # the EXTERNAL id -> match with the exact-or-'{ext}-' rule, then STAGE keyed by the resolved
            # external id so finish/confirm/abort/unlink all agree.
            ext = _match_disk_route(e.req_id, routed_keys) if any_disk else None
            if ext is not None:
                if _ring_debug():
                    _dbg(f"qk demux DISK hit: entry.req_id={e.req_id!r} -> route={ext!r} "
                         f"layer={e.layer} k_rows={int(e.k_rows)} q_rows={int(e.q_rows)}")
                # PER-ENTRY DISK ISOLATION (defense-in-depth): a single disk request's staging write
                # must NEVER wedge the whole consumer -- catch it here, log LOUD, mark the request
                # aborted (its remaining rows skipped + dir reclaimed), and CONTINUE. Host rows keep
                # their never-drop guarantee (cloned/noted below regardless; cursor advances anyway).
                try:
                    self._disk_write(ext, e.layer, q_slice, k_slice,
                                     int(e.prefix_end), int(e.num_computed))
                except Exception:  # noqa: BLE001 -- isolate ONE disk request; never wedge the consumer
                    logger.exception(
                        "qk off-loop ring drain: per-request DISK demux write failed for req=%r "
                        "layer=%s; that request's disk delivery is dropped + its staging reclaimed, "
                        "the consumer continues", ext, e.layer)
                    with self._index_lock:
                        if ext in self._disk_staging or ext in self._disk_routed:
                            self._disk_aborted.add(ext)
                    if _ring_debug():
                        _dbg(f"qk demux DISK write FAILED (isolated): req={ext!r} layer={e.layer}")
            else:
                if any_disk and _ring_debug():
                    _dbg(f"qk demux DISK miss: entry.req_id={e.req_id!r} not in routes "
                         f"{list(routed_keys)} -> host index")
                staged.append((e.req_id, int(e.layer),
                               None if q_slice is None else q_slice.clone(),
                               k_slice.clone(), int(e.prefix_end)))
        with self._index_lock:
            # ABORT SKIP re-checked HERE so it is atomic with the note: a request aborted after its
            # rows were drained (HOST -> mark_host_aborted, or DISK whose _disk_routed was popped ->
            # _disk_aborted so its post-pop rows fell through to `staged`) must NOT (re-)create a host
            # slot nothing frees.
            ab_host = tuple(self._host_aborted) if self._host_aborted else ()
            ab_disk = tuple(self._disk_aborted) if self._disk_aborted else ()
            for req_id, layer, q_clone, k_clone, prefix_end in staged:
                if ((ab_host and _match_disk_route(req_id, ab_host) is not None)
                        or (ab_disk and _match_disk_route(req_id, ab_disk) is not None)):
                    if _ring_debug():
                        _dbg(f"qk demux HOST-SKIP aborted: req={req_id!r} layer={layer}")
                    continue
                # k stream EVERY step, carrying the cumulative prefix_ends (LAST-WRITE-WINS on the
                # ("k", layer) key -- assemble_qk reads the final list at finish). prefix_end < 0 is a
                # non-emit (last_token mid-prefill) step -> no new boundary -> kmeta=None.
                kmeta = None
                if prefix_end >= 0:
                    lst = self._qk_kmeta.setdefault(req_id, {}).setdefault(layer, [])
                    lst.append(prefix_end)
                    kmeta = {"prefix_ends": list(lst)}
                self.index.note_rows(req_id, ("k", layer), k_clone, kmeta=kmeta)
                if q_clone is not None:
                    self.index.note_rows(req_id, ("q", layer), q_clone)

    def _disk_write(self, req_id, layer, q_slice, k_slice, prefix_end: int, num_computed: int) -> None:
        """Append a disk-routed request's step q + k rows to its per-request files (creating its
        staging on the first row). The dict membership is guarded by ``_index_lock``; the file write
        runs lock-free on the single consumer-thread writer.

        SKIP GUARD (single-owner dir lifecycle): re-check under the lock, BEFORE creating/appending,
        that this request is neither ABORTED nor un-routed — ``_demux_into_index`` matched it against a
        ``routed_keys`` snapshot taken BEFORE the lock, so a concurrent abort could land in that window.
        Skipping here means the consumer NEVER opens a layer file inside a dir the abort slated for
        discard (closing the ``rmtree``-vs-``open`` race)."""
        with self._index_lock:
            if req_id in self._disk_aborted or req_id not in self._disk_routed:
                if _ring_debug():
                    _dbg(f"qk disk_write SKIP (aborted/unrouted): req={req_id!r} layer={layer}")
                return
            stg = self._disk_staging.get(req_id)
            if stg is None:
                stg = _PerRequestQKDiskStaging(
                    req_id, os.path.join(self._disk_base, _sanitize_req_id(req_id)),
                    self.header, self._perreq_cap, self._perreq_mmap)
                self._disk_staging[req_id] = stg
        stg.append(layer, q_slice, k_slice, prefix_end, num_computed)

    def _handle_finish(self, req_id) -> None:
        """Finish a request: for a DISK-routed request finalize its per-request q/k files (msync +
        sidecar) and hand it to the OffloadProcess for transfer to the client dest, then free its
        staging (residency -> 0); for a host-buffer request mark it finished in the PerRequestIndex.
        FIFO: this ``_Finish`` trails all of the request's ``_DrainItem``s. SINGLE-OWNER ABORT RECLAIM:
        the CONSUMER thread owns the discard of an aborted disk request's staging dir."""
        req_id = str(req_id)
        with self._index_lock:
            aborted_ext = (_match_disk_route(req_id, tuple(self._disk_aborted))
                           if self._disk_aborted else None)
            if aborted_ext is not None:
                self._disk_aborted.discard(aborted_ext)
                self._host_aborted.discard(aborted_ext)
                self._disk_routed.pop(aborted_ext, None)
                self._disk_delivered_src.pop(aborted_ext, None)
                self._qk_kmeta.pop(aborted_ext, None)
                stg_abort = self._disk_staging.pop(aborted_ext, None)
                ext = None
                disk_dest = None
                stg = None
            else:
                stg_abort = None
                ext = _match_disk_route(req_id, tuple(self._disk_routed))
                disk_dest = self._disk_routed.pop(ext, None) if ext is not None else None
                stg = self._disk_staging.pop(ext, None) if disk_dest is not None else None
                if stg is not None and self._offload is not None:
                    self._disk_delivered_src[ext] = stg.run_dir
        if aborted_ext is not None:
            if stg_abort is not None:
                stg_abort.discard()          # close fds + rmtree the source (no offload, no sidecar)
            if _ring_debug():
                _dbg(f"qk finish ABORT-reclaim: id={req_id!r} route={aborted_ext!r} "
                     f"discarded={stg_abort is not None} (single-owner consumer discard)")
            return
        if disk_dest is not None:
            if _ring_debug():
                _dbg(f"qk finish DISK: id={req_id!r} route={ext!r} "
                     f"run_dir={(stg.run_dir if stg else None)!r} dest={disk_dest!r} "
                     f"submit={stg is not None and self._offload is not None}")
            if stg is not None:
                stg.close()                     # msync + per-request QK sidecar (single writer, off-lock)
                if self._offload is not None:
                    self._offload.submit(ext, stg.run_dir, disk_dest)  # non-blocking, never-drop
            return
        if self.index is None:
            return
        with self._index_lock:
            host_ab = (_match_disk_route(req_id, tuple(self._host_aborted))
                       if self._host_aborted else None)
            if host_ab is not None:
                self.index.free(req_id)
                self._host_aborted.discard(host_ab)
                self._qk_kmeta.pop(req_id, None)
                if _ring_debug():
                    _dbg(f"qk finish HOST-abort drop: id={req_id!r} mark={host_ab!r}")
                return
            if req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)
                self._qk_kmeta.pop(req_id, None)   # cumulative list is stored on the entry now

    # ---- shutdown / flush ----
    def finalize_all(self) -> None:
        """END-OF-RUN ONLY: mark every still-live per-request request finished so it becomes
        deliverable via ``pop_deliverable_qk``. Closes the last-step straggler gap (a request finishing
        on the FINAL executed step never gets its ``_Finish``). MUST run only at genuine end-of-run.
        STRICT NO-OP when per_request is off (index is None + empty disk maps) -> the shared-file
        default path is byte-identical. Mirrors ``OffLoopRingDrain.finalize_all``."""
        with self._index_lock:
            disk_pending = list(self._disk_staging.keys())
        for req_id in disk_pending:
            self._finalize_finish_isolated(req_id)
        with self._index_lock:
            aborted_pending = list(self._disk_aborted)
        for ab_id in aborted_pending:
            self._finalize_finish_isolated(ab_id)
        # End-of-run settled-reclaim of any confirm-timeout-parked delivered-source (no-op when none).
        self._reclaim_settled_pending()
        if self.index is None:
            return
        with self._index_lock:
            if self._host_aborted:
                ab_host = tuple(self._host_aborted)
                for rid in [r for r in self.index.live_req_ids()
                            if _match_disk_route(r, ab_host) is not None]:
                    self.index.free(rid)
                self._host_aborted.clear()
            for req_id in self.index.live_req_ids():
                self.index.mark_finished(req_id)
            self._qk_kmeta.clear()

    def ring_residency(self) -> "Tuple[int, int]":
        """NON-DESTRUCTIVE ``(host_live_count, disk_residency)`` — number of requests still holding a
        host-buffer ``PerRequestIndex`` entry and the number still holding per-request DISK staging.
        Read WITHOUT stopping the drain / popping / freeing (the residency gate polls it mid-serving).
        ``disk_residency`` takes ``_index_lock`` itself (non-reentrant), so it runs OUTSIDE the host
        read's hold."""
        with self._index_lock:
            host_live = len(self.index.live_req_ids()) if self.index is not None else 0
        disk = int(self.disk_residency())
        return (int(host_live), disk)

    def stop(self) -> None:
        """Drain the queue, join the consumer, finalize end-of-run stragglers, surface a consumer-thread
        error. Idempotent. ``_STOP`` is enqueued AFTER every row/finish item, so the joined consumer has
        noted every row into the index; only THEN does ``finalize_all()`` mark still-live stragglers.
        The finalize runs on this (collector) thread once the consumer is provably not running."""
        if self._started and self._thread.is_alive():
            self._q.put(_STOP)
            join_s = float(os.environ.get("VLLM_HOOK_RING_DRAIN_JOIN_S", "60") or "60")
            self._thread.join(timeout=join_s)
        self._started = False
        if not self._thread.is_alive():
            self.finalize_all()
        if self._error is not None:
            raise RuntimeError(
                "qk off-loop ring drain consumer thread failed; captured QK may be incomplete"
            ) from self._error
