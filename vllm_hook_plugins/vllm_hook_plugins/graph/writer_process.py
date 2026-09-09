"""A separate PROCESS that serializes+writes captured artifacts off the engine GIL.

On the disk path, serializing inline (pad + safetensors-encode / pickle) holds the CPython GIL
and starves the decode loop -> the disk path's ~200 ms flat on-loop contention floor. A thread
cannot escape the GIL either; this moves the serialize into a spawned child *process* (own
interpreter, own GIL). The engine loop only pays the ~0.3 ms handoff: ``torch.multiprocessing.Queue``
moves each CPU tensor's storage to shared memory (a GIL-light memcpy + a small handle pickle)
rather than pickling the bytes on the loop -- the "shared_memory HANDLE, never Queue.put(dict)"
the plan requires, done transparently by torch.multiprocessing. The child runs the SAME pure
``write_artifact`` the inline path calls, so the on-disk artifact is byte-identical.

Default-ON for the disk path; ``VLLM_HOOK_WRITER_PROCESS=0`` forces the inline serialize path
(on the engine loop). Byte-identical either way (same pure ``write_artifact``).
NO torch import at module level: the child imports this module during spawn, and must set its
BLAS/OMP thread caps BEFORE torch is first imported (an uncapped torch import spawns dozens of
BLAS threads and can exhaust the process rlimit). torch is imported lazily -- in the parent's
__init__ and, in the child, only after the caps are set.

REUSE-AFTER-FREE: ``submit()`` carries the flushed request's ``req_ids``; the feeder thread
(``_feed``) pushes each onto ``self.pages_done`` the moment it has COPIED that request's bytes out
(after ``pack_tensor_tree`` succeeds, or after the inline handoff-failure fallback writes the
artifact) -- never before. A ring-page consumer polling ``poll_pages_done()`` must not return the
request's ring pages to the freelist before that signal; releasing earlier would let the capture
path overwrite a page the feeder is still reading. This requires ``self._pack`` (default ON): the
RAW (pack=0) handoff hands the child the ORIGINAL tensors, which it only copies when it
serializes them itself, so there is no early copy-out point to signal from -- pack=0 combined
with the capture ring is therefore a documented limitation (pages held by a raw-handoff request
are never early-released), not something this module silently corrects.
"""
from __future__ import annotations

import os
import queue as _queue

# The child does pure CPU serialize (no BLAS, no CUDA); cap its thread pools so importing torch
# in a spawned child of the (already forked) worker never trips the process/thread rlimit.
_CHILD_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def _writer_child(q) -> None:
    """Child entry: drain the queue, serialize+write each item with our own GIL.

    The thread caps + CUDA_VISIBLE_DEVICES="" are set in the PARENT before spawn (see
    ``_child_env`` below) so they are already in the child's inherited environment when the
    spawn bootstrap imports the plugin package (which pulls torch, whose OMP/BLAS pools are
    sized at import). These lines are belt-and-braces for a direct/odd invocation only; by the
    time they run torch is already imported, so they cannot resize the pools -- the parent set
    is the load-bearing one."""
    for k, v in _CHILD_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from vllm_hook_plugins.graph.artifact_writer import write_artifact
    from vllm_hook_plugins.graph.tensor_pack import unpack_tensor_tree
    while True:
        item = q.get()
        if item is None:
            break
        try:
            tag = item[0]
            if tag == "P":
                # packed: ("P", worker_kind, buffer, manifest, run_dir, mode, tp,
                #          use_safetensors, force_pt, pt_filename) -- ONE shm mapping crossed.
                (_, wk, buffer, manifest, run_dir, mode, tp, use_st, fpt, ptn) = item
                cpu_cache = unpack_tensor_tree(buffer, manifest)
                write_artifact(wk, cpu_cache, run_dir, mode, tp, use_st, fpt, ptn)
            else:
                # "R" raw legacy handoff: ("R",) + write_artifact's exact arg tuple.
                write_artifact(*item[1:])
        except Exception as e:  # noqa: BLE001 — never crash the writer on one bad item
            print(f"[writer-process] save failed: {e!r}", flush=True)


class WriterProcess:
    """Lazily-started spawned daemon + a bounded torch.multiprocessing.Queue. One process, one
    consumer -> same-run_id artifacts serialize in submit order (matches the thread path)."""

    def __init__(self, maxsize: int = 4, n_writers: int = 1) -> None:
        import torch.multiprocessing as tmp  # parent already has torch
        self._n_writers = max(1, int(n_writers))  # N children consume the shared queue (throughput)
        # Handoff sharing strategy. With packing (default) each flush shares ONE storage, so the
        # strategy no longer gates the mapping COUNT; 'file_system' is kept as belt-and-braces
        # (shm_open by NAME + close the fd after mmap -> flat open-fd count) and as the safe
        # default for the pack-OFF legacy path. Escape hatch: VLLM_HOOK_WRITER_SHARING=file_descriptor.
        _want = os.environ.get("VLLM_HOOK_WRITER_SHARING", "file_system")
        try:
            if _want in tmp.get_all_sharing_strategies():
                tmp.set_sharing_strategy(_want)
        except Exception:  # noqa: BLE001 — never fail writer init on a strategy tweak
            pass
        self._ctx = tmp.get_context("spawn")
        self._q = self._ctx.Queue(maxsize=max(1, int(maxsize)))
        # cancel_join_thread: on interpreter exit don't block joining the queue's feeder thread
        # (the daemon child may be terminated first, leaving a full OS pipe with no reader ->
        # the classic "join a queue with no consumer" shutdown hang). We accept dropping any
        # not-yet-flushed items on abrupt exit (same best-effort as the daemon thread path).
        self._q.cancel_join_thread()
        # Load-bearing fix: put the thread caps + empty CUDA_VISIBLE_DEVICES into os.environ
        # BEFORE start(), so spawn snapshots them into the child's environment and they apply to
        # the child's very first torch import (env vars are read at torch/libgomp init). Setting
        # them only inside _writer_child is too late: unpickling the target by reference imports
        # the plugin package (-> torch) first. Restore the parent's values right after start()
        # (the parent already has torch/CUDA initialized, so its live state is unaffected).
        saved = {k: os.environ.get(k) for k in (*_CHILD_THREAD_ENV, "CUDA_VISIBLE_DEVICES")}
        try:
            for k, v in _CHILD_THREAD_ENV.items():
                os.environ.setdefault(k, v)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self._procs = []
            for _i in range(self._n_writers):
                _p = self._ctx.Process(target=_writer_child, args=(self._q,),
                                       daemon=True, name=f"vllm-hook-writer-{_i}")
                _p.start()
                self._procs.append(_p)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # ---- producer feeder: pack OFF the engine loop, hand ONE buffer to the child ----
        # submit() (called on the serial EngineCore loop) only enqueues the raw cpu_cache here;
        # THIS thread does the pack memcpy (GIL-released) + the mp handoff, so the loop never
        # pays either. Bounded like the mp queue -> a full internal queue makes submit() return
        # False and the caller falls through to the inline/thread ladder (no loop stall).
        import threading
        from vllm_hook_plugins.graph.tensor_pack import pack_tensor_tree
        self._pack = os.environ.get("VLLM_HOOK_WRITER_PACK", "1") != "0"
        # pack=0 (VLLM_HOOK_WRITER_PACK=0) is an unsupported combination with the capture ring --
        # the RAW handoff below hands the child the ORIGINAL tensors (which may be page-backed
        # views), so _feed has no early copy-out point to signal pages_done from; a request's
        # pages would never be released via the writer-completion signal. Pack stays the shipped
        # default specifically so the ring's early-release path is always available; this is a
        # documented limitation, not force-corrected, since forcing pack=1 here would silently
        # override the escape hatch this env var is FOR.
        self._pack_fn = pack_tensor_tree
        self._put_timeout = float(os.environ.get("VLLM_HOOK_WRITER_PUT_TIMEOUT", "30") or "30")
        _inq_max = int(os.environ.get("VLLM_HOOK_WRITER_INQ_MAX", "64") or "64")
        self._inq = _queue.Queue(maxsize=max(1, _inq_max))
        # req_ids the feeder has copied bytes out for (pack_tensor_tree, or the inline
        # handoff-failure fallback) -- drained by a ring-page consumer to release the
        # corresponding ring pages. A page must not be released before this signal, else a later
        # stream() could overwrite bytes the feeder is still reading (reuse-after-free).
        self.pages_done = _queue.Queue()
        self._feeder = threading.Thread(target=self._feed, name="vllm-hook-writer-feeder",
                                        daemon=True)
        self._feeder.start()

    def _feed(self) -> None:
        """Drain the internal queue; pack each cpu_cache into ONE buffer + manifest (or pass it
        raw when VLLM_HOOK_WRITER_PACK=0) and hand it to the writer child. Runs on a daemon
        thread so the pack memcpy stays OFF the engine loop.

        DURABILITY (the 'never lose an artifact' contract): submit() already returned True for
        every item here, so flush_disk did NOT run its own inline fallback. If we cannot deliver
        to the child -- pack raises (e.g. the coalesced buffer OOMs), the mp queue stays full past
        the timeout, or the child died -- WE write the artifact in-process on THIS feeder thread
        (off the engine loop) rather than drop it. A blocking self._q.put would otherwise hang the
        feeder forever on a dead child with a full queue; the timeout converts that into a fallback.

        REUSE-AFTER-FREE SIGNAL: ``submit()`` carries the flushed request's ``req_ids`` alongside
        its item. The moment this thread has COPIED a request's bytes out of whatever the caller
        handed it -- i.e. right after ``pack_tensor_tree`` succeeds (a fresh coalesced buffer) or
        right after the inline-fallback ``write_artifact`` succeeds (a serialized file) -- those
        req_ids are pushed onto ``self.pages_done``. Only then is it safe for a ring-page consumer
        to return the request's ring pages to the freelist: before that point the caller's
        page-backed views may be the ONLY copy of the data, and reusing the page would corrupt it
        out from under this thread. The RAW (``self._pack`` False) handoff has no such point --
        the child holds the ORIGINAL tensors until it serializes them itself -- so it deliberately
        does NOT signal (documented limitation on ``self._pack`` above)."""
        from vllm_hook_plugins.graph.artifact_writer import write_artifact
        while True:
            raw = self._inq.get()
            if raw is None:
                break
            (wk, cpu_cache, run_dir, mode, tp, use_st, fpt, ptn, req_ids) = raw
            try:
                if not any(p.is_alive() for p in self._procs):
                    raise RuntimeError("no writer child is alive")
                if self._pack:
                    buffer, manifest = self._pack_fn(cpu_cache)
                    # Pre-share the ONE buffer HERE so a share/mmap failure is CATCHABLE on this
                    # thread. Otherwise mp.Queue does the share inside its own internal _feed
                    # thread (torch reduce_storage -> _share_filename_cpu_), out of reach of this
                    # try -> an uncatchable crash + a lost artifact (the exact mmap-ENOMEM the pack
                    # fixes, but for the buffer itself under extreme pressure). Idempotent: mp's
                    # later pickle reuses the already-shared storage, so NO second mapping is made.
                    buffer.share_memory_()
                    self._q.put(("P", wk, buffer, manifest, run_dir, mode, tp,
                                 bool(use_st), bool(fpt), ptn), timeout=self._put_timeout)
                    # The pack just copied every tensor into the ONE fresh buffer above -> the
                    # caller's original (possibly page-backed) tensors are no longer referenced
                    # by anything downstream. Safe to release their ring pages now.
                    for _rid in req_ids:
                        self.pages_done.put(_rid)
                else:
                    # RAW path: the child holds the ORIGINAL tensors (incl. any page-backed
                    # views) until it serializes them itself -> no early copy-out here, so we do
                    # NOT put req_ids on pages_done (see the pack=0 note in __init__).
                    self._q.put(("R", wk, cpu_cache, run_dir, mode, tp,
                                 bool(use_st), bool(fpt), ptn), timeout=self._put_timeout)
            except Exception as e:  # noqa: BLE001 — deliver failed -> write it ourselves, never drop
                try:
                    write_artifact(wk, cpu_cache, run_dir, mode, int(tp),
                                   bool(use_st), bool(fpt), ptn)
                    # The inline write above just serialized (copied) every tensor -> safe to
                    # release their ring pages too.
                    for _rid in req_ids:
                        self.pages_done.put(_rid)
                    print(f"[writer-process] feeder handoff failed ({e!r}); wrote {run_dir} "
                          f"inline on the feeder thread", flush=True)
                except Exception as e2:  # noqa: BLE001 — only now is the artifact truly lost
                    print(f"[writer-process] feeder FALLBACK write FAILED for {run_dir}: {e2!r} "
                          f"(after handoff error {e!r})", flush=True)

    def poll_pages_done(self) -> list:
        """Drain ``self.pages_done`` non-blocking into a list: req_ids whose disk-path bytes the
        feeder has just copied out (via ``pack_tensor_tree`` or the inline handoff-failure
        fallback). Called by the ring-page consumer to recycle ring pages."""
        out = []
        while True:
            try:
                out.append(self.pages_done.get_nowait())
            except _queue.Empty:
                break
        return out

    @classmethod
    def from_env(cls):
        """Return a started WriterProcess UNLESS VLLM_HOOK_WRITER_PROCESS=0 forces it off.

        Default-ON for the disk path (it is only fed by flush_disk); byte-identical either way.
        Set VLLM_HOOK_WRITER_PROCESS=0 to force the legacy in-process thread/inline serialize
        path."""
        if os.environ.get("VLLM_HOOK_WRITER_PROCESS", "1") == "0":
            return None
        maxsize = int(os.environ.get("VLLM_HOOK_WRITER_PROCESS_QSIZE", "4") or "4")
        n_writers = int(os.environ.get("VLLM_HOOK_WRITER_PROCESS_N", "1") or "1")
        return cls(maxsize, n_writers)

    def alive(self) -> bool:
        return bool(getattr(self, "_procs", None)) and any(p.is_alive() for p in self._procs)

    def submit(self, worker_kind: str, cpu_cache: dict, run_dir: str, default_mode: str,
               tp_rank: int, use_safetensors: bool, force_pt: bool, pt_filename: str,
               *, req_ids: list = None, block: bool = False) -> bool:
        """Hand off to the off-engine feeder. block=False -> put_nowait (legacy: caller inlines
        on full). block=True -> bounded _inq.put(timeout) so the ENGINE waits off-serialize for
        feeder space instead of serializing 190 MB inline; returns False only if the child is
        dead or the bounded wait times out (data-safety fall-through).

        ``req_ids``: the internal request ids whose data lives in this ``cpu_cache`` -- cpu_cache
        is keyed by module name, not req_id, so the caller (``flush_disk``) passes them
        explicitly. ``_feed`` pushes each onto ``self.pages_done`` once it has copied the bytes
        out (pack or the inline fallback), letting the ring-page consumer release the
        corresponding ring pages without a reuse-after-free."""
        if not self.alive():
            return False
        item = (worker_kind, cpu_cache, run_dir, default_mode, int(tp_rank),
                bool(use_safetensors), bool(force_pt), pt_filename, list(req_ids or []))
        try:
            if block:
                self._inq.put(item, timeout=self._put_timeout)
            else:
                self._inq.put_nowait(item)
            return True
        except Exception:  # noqa: BLE001 — Full/timeout/torn-down -> caller decides
            return False

    def close(self) -> None:
        # Timed/guarded puts so a dead or stuck child can never hang shutdown (a bare put on a
        # full mp queue with no consumer blocks forever, and the try/except can't interrupt it).
        try:
            self._inq.put(None, timeout=10)   # stop the feeder AFTER it drains its backlog
            self._feeder.join(timeout=15)
        except Exception:  # noqa: BLE001
            pass
        try:
            for _p in self._procs:                 # one sentinel per child (shared queue)
                if _p.is_alive():
                    self._q.put(None, timeout=10)  # stop each child AFTER it drains
            for _p in self._procs:
                _p.join(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def _tree_bytes(obj) -> int:
    """Sum the byte size of every tensor leaf in a cpu_cache. Duck-typed (``element_size*numel``)
    so this module needs no torch import (the writer child imports it before torch is set up)."""
    total = 0
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        else:
            try:
                total += int(x.element_size()) * int(x.numel())   # torch tensor
            except Exception:  # noqa: BLE001 — non-tensor leaf
                pass
    return total


def _mem_limit_bytes() -> int:
    """The process's RAM ceiling: the cgroup memory limit (v2 then v1) if set, else host RAM. Using
    the cgroup limit keeps the budget honest under a container/job-scheduler memory cap (host RAM
    alone would be far too generous)."""
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                n = int(v)
                if 0 < n < (1 << 62):        # skip "max" / absurd sentinels
                    return n
        except Exception:  # noqa: BLE001
            pass
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return 8 * (1 << 30)


def _resolve_flush_budget() -> int:
    """RAM budget for the write backlog. Absolute ``VLLM_HOOK_FLUSH_RAM_BUDGET_BYTES`` wins; else a
    fraction (``VLLM_HOOK_FLUSH_RAM_BUDGET_FRAC``, default 0.25) of the process RAM limit. This is
    how much backlog the engine absorbs into RAM at FULL SPEED before the lossless backpressure
    engages -- RAM buys throughput; the engine only slows near OOM."""
    b = os.environ.get("VLLM_HOOK_FLUSH_RAM_BUDGET_BYTES")
    if b:
        return max(1 << 20, int(b))
    frac = float(os.environ.get("VLLM_HOOK_FLUSH_RAM_BUDGET_FRAC", "0.25") or "0.25")
    return max(1 << 20, int(frac * _mem_limit_bytes()))


def init_writer_process(worker) -> None:
    """Start ``worker._writer_process`` once, UNLESS VLLM_HOOK_WRITER_PROCESS=0 (then set None).
    Default-ON for the disk path. Idempotent; called from both graph install and the eager
    install_hooks. Cheap idle child when no disk artifacts are produced (only ``flush_disk``
    feeds it)."""
    if hasattr(worker, "_writer_process"):
        return
    try:
        worker._writer_process = WriterProcess.from_env()
        if worker._writer_process is not None:
            import atexit
            atexit.register(worker._writer_process.close)  # teardown drain: flush pending writes
            print("[writer-process] async serialize+write process ON", flush=True)
    except Exception as e:  # noqa: BLE001 — never fail worker init on the writer
        worker._writer_process = None
        print(f"[writer-process] failed to start, falling back to in-process save: {e!r}",
              flush=True)
