"""No-GPU THREADED interleaving tests for the two abort-vs-finish races in the off-loop HS
capture-ring per-request path (Task 12 review): C1 (host-route free strands ``_deliverable`` ->
permanent KeyError wedge) and I1 (disk abort races finish -> ``_disk_delivered_src`` + source-dir
leak).

Both existing abort tests (in ``test_serve_ring_delivery_lifecycle.py``) are SINGLE-THREADED: the
abort runs strictly before finish, or strictly after a fully-drained finish. The dangerous case --
the consumer's ``mark_finished`` / delivered-source record landing IN THE GAP of the abort path's
critical sections -- was untested. These tests drive the REAL worker/drain methods with a gated lock
(``_GapLock``) that deterministically parks the running thread mid-gap so the other thread's
``_handle_finish`` interleaves exactly there.

RED-then-GREEN: each test drives the SHIPPED (fixed) methods and asserts the post-fix invariant. Run
against the pre-fix code (revert the two atomic-section fixes) and:
  * C1 test -> the final ``pop_deliverable`` raises ``KeyError`` (``_deliverable`` still references the
    freed R) -- a wedge.
  * I1 test -> ``_disk_delivered_src`` still holds R (the recorded source that no confirm reclaims).
Confirmed RED against the reverted code, GREEN against the fix, and re-run for flake-freedom.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_abort_finish_race.py -q
"""
import os
import shutil
import tempfile
import threading

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker

_HIDDEN = 4


class _GapLock:
    """A ``threading.Lock`` wrapper that, once ARMED, parks the releasing thread on its NEXT
    ``__exit__`` (AFTER the underlying lock is released, so the OTHER thread can acquire it) until a
    partner thread signals ``resume``. This lands a concurrent critical section EXACTLY in the gap
    between the abort path's two locked sections (pre-fix) or right after the single fused section
    (post-fix). Fires once. Also supports the plain acquire/release protocol for any direct users."""

    def __init__(self):
        self._l = threading.Lock()
        self._armed = False
        self._fired = False
        self.handoff = threading.Event()   # set when the parked thread hands control over
        self.resume = threading.Event()    # set by the partner to release the parked thread

    def arm(self):
        self._armed = True

    def __enter__(self):
        self._l.acquire()
        return self

    def __exit__(self, *exc):
        self._l.release()                  # release FIRST so the partner thread can take the lock
        if self._armed and not self._fired:
            self._fired = True
            self._armed = False
            self.handoff.set()
            self.resume.wait(timeout=5.0)

    def acquire(self, *a, **k):
        return self._l.acquire(*a, **k)

    def release(self):
        return self._l.release()


class _GapIndex(PerRequestIndex):
    """Arms the shared ``_GapLock`` when ``pop_deliverable`` runs, so the lock's following release
    (the drain section's) parks the driver mid-gap. Base pop_deliverable/free/mark_finished are
    inherited UNCHANGED -- this only decides WHEN to arm, never what the index does."""

    def __init__(self, lock):
        super().__init__()
        self._gap_lock = lock

    def pop_deliverable(self):
        self._gap_lock.arm()
        return super().pop_deliverable()


def _make_drain(tmp, index=None, offload=None):
    """A per_request OffLoopRingDrain whose consumer thread is NOT started -- the test drives the
    guarded methods directly from two threads so it exercises the LOCK discipline, not the ring D2H."""
    dtype = torch.float32
    ring = GpuCaptureRing(row_bytes=_HIDDEN * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=64, device="cpu", dtype=dtype, row_shape=(_HIDDEN,))
    hs_buf = torch.zeros(65, _HIDDEN, dtype=dtype)
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [_HIDDEN], "hidden": _HIDDEN}
    run_dir = tempfile.mkdtemp(prefix="abort_race_", dir=tmp)
    kw = {"per_request": True, "disk_base": os.path.join(run_dir, "staging")}
    if index is not None:
        kw["index"] = index
    if offload is not None:
        kw["offload"] = offload
    return OffLoopRingDrain(ring, [(1, hs_buf)], run_dir, header, **kw)


def _worker(drain):
    w = ProbeHiddenStatesWorker()
    w._hs_drain = drain
    w._conf = {"hidden_size": _HIDDEN, "num_layers": 1}
    return w


def _fake_offload():
    """OffloadProcess (thread backend) whose transfer just records the call (best-effort copytree so
    a surviving dest is real)."""
    calls = []

    def fake_transfer(src, dest):
        calls.append((src, dest))
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
        except Exception:  # noqa: BLE001 -- src may be mid-rmtree (documented I1 residual)
            pass

    return OffloadProcess(transfer_fn=fake_transfer), calls


# =============================================================== C1: host-route abort/finish race ===
def test_c1_host_abort_gap_finish_no_deliverable_wedge():
    """R is host-routed, live but NOT finished. The driver runs the REAL ``clear_ring_request(R)``;
    the gap seam parks it right after the drain section's lock releases; the consumer then runs the
    REAL ``_handle_finish(R)`` (mark_finished) IN THAT GAP.

    Pre-fix (target free in a separate section): mark_finished appends R to ``_deliverable`` and the
    later bare free(R) drops R from ``_entries`` -> the final ``pop_deliverable`` KeyErrors and stays
    wedged. Post-fix (target free fused into the drain section): R is already freed when the consumer
    runs, so ``_handle_finish`` no-ops on the ``live_req_ids()`` guard -> ``_deliverable`` stays
    consistent, R freed exactly once, and a later ``pop_deliverable`` succeeds."""
    tmp = tempfile.mkdtemp(prefix="c1_")
    try:
        lock = _GapLock()
        idx = _GapIndex(lock)
        drain = _make_drain(tmp, index=idx)
        drain._index_lock = lock
        w = _worker(drain)
        with lock:
            idx.note_rows("R", 1, torch.zeros(1, _HIDDEN))   # R live, not finished
        assert idx.live_req_ids() == {"R"}

        def consumer():
            lock.handoff.wait(timeout=5.0)
            drain._handle_finish("R")     # mark_finished lands in the post-drain-section gap
            lock.resume.set()

        ct = threading.Thread(target=consumer, name="c1-consumer")
        ct.start()
        w.clear_ring_request("R")         # the REAL abort cleanup (driver thread)
        ct.join(timeout=10)

        assert lock._fired, "gap seam never fired -- the interleaving was not exercised (vacuous)"
        assert not ct.is_alive(), "consumer thread wedged"
        # Post-fix invariants: R gone from _entries, _deliverable consistent, pop succeeds, freed once.
        assert idx.live_req_ids() == set(), "R must be freed from _entries exactly once"
        assert idx._deliverable == [], "_deliverable must not reference a freed R (C1 wedge)"
        assert idx.pop_deliverable() == [], "pop_deliverable must succeed (no KeyError wedge)"
        assert not (getattr(w, "_ring_perreq_stash", None) or {}), "no stash entry should remain"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================== I1: disk abort/finish race ===
def test_i1_disk_abort_in_finish_window_no_delivered_src_leak():
    """R is disk-routed with live staging. The consumer runs the REAL ``_handle_finish(R)``; the gap
    seam parks it right after its first locked section releases; the driver then runs the REAL
    ``clear_request_disk(R)`` IN THAT WINDOW.

    Pre-fix (delivered_src recorded in a SECOND section): the abort sees all three maps empty and does
    nothing; ``_handle_finish`` then records ``_disk_delivered_src[R]`` that no confirm ever reclaims
    -> permanent source-dir + dict leak. Post-fix (delivered_src recorded in the SAME section that
    pops the staging): the abort sees the recorded source and reclaims it (pop + rmtree) -> no dict
    leak. (``close()``/``submit()`` racing the abort's rmtree is the documented best-effort residual;
    it never re-leaks the dict/source.)"""
    tmp = tempfile.mkdtemp(prefix="i1_")
    op, _calls = _fake_offload()
    try:
        lock = _GapLock()
        drain = _make_drain(tmp, offload=op)
        drain._index_lock = lock
        dest = os.path.join(tmp, "delivered", "R")
        drain.route_to_disk("R", dest)
        drain._disk_write("R", 1, torch.zeros(2, _HIDDEN), 2, "all_tokens")   # create live staging
        assert drain.disk_residency() == 1

        cerr = []

        def consumer():
            lock.arm()                        # fire on _handle_finish's FIRST locked-section release
            try:
                drain._handle_finish("R")     # close()/submit() may race the abort rmtree (residual)
            except BaseException as e:        # noqa: BLE001 -- documented residual, not the leak
                cerr.append(repr(e))
            lock.resume.set()                 # belt-and-suspenders (driver already resumed it)

        def driver():
            lock.handoff.wait(timeout=5.0)
            drain.clear_request_disk("R")     # the REAL abort cleanup, IN the finish window
            lock.resume.set()

        ct = threading.Thread(target=consumer, name="i1-consumer")
        dt = threading.Thread(target=driver, name="i1-driver")
        ct.start(); dt.start()
        ct.join(timeout=10); dt.join(timeout=10)

        assert lock._fired, "gap seam never fired -- the interleaving was not exercised (vacuous)"
        assert not ct.is_alive() and not dt.is_alive(), "a thread wedged"
        # Post-fix invariant (the leak signal): the recorded server-side source is reclaimed, not
        # stranded. Pre-fix this dict still holds "R".
        assert "R" not in drain._disk_delivered_src, (
            "delivered-source leaked: clear_request_disk could not see the recorded source (I1)")
        assert drain.disk_residency() == 0, "disk staging not freed on abort"
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_i1_disk_abort_after_finish_reclaims_delivered_source():
    """CLEAN (single-threaded) guard for the I1 reclaim contract: a disk-routed request that fully
    FINISHES (delivered_src recorded, files closed) and is THEN aborted -- ``clear_request_disk`` must
    pop the recorded source + rmtree it, so nothing leaks. No concurrent close()/submit() residual."""
    tmp = tempfile.mkdtemp(prefix="i1b_")
    op, calls = _fake_offload()
    try:
        drain = _make_drain(tmp, offload=op)
        dest = os.path.join(tmp, "delivered", "R")
        drain.route_to_disk("R", dest)
        drain._disk_write("R", 1, torch.zeros(2, _HIDDEN), 2, "all_tokens")
        drain._handle_finish("R")                 # full finish: record delivered_src, close, submit
        assert op.wait("R", timeout=5.0) is True, "R never delivered"
        src = drain._disk_delivered_src.get("R")
        assert src and os.path.isdir(src), "server source should exist until reclaimed"

        drain.clear_request_disk("R")             # abort after finish -> reclaim the source
        assert "R" not in drain._disk_delivered_src, "delivered-source dict entry leaked"
        assert not os.path.exists(src), "server-side source dir not reclaimed on abort-after-finish"
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for t in (test_c1_host_abort_gap_finish_no_deliverable_wedge,
              test_i1_disk_abort_in_finish_window_no_delivered_src_leak,
              test_i1_disk_abort_after_finish_reclaims_delivered_source):
        t()
        print(f"PASS  {t.__name__}")
    print("VERDICT: PASS (3/3)")
