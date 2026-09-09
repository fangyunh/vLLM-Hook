"""No-GPU CONCURRENCY test for the retrieval-vs-consumer race on the shared PerRequestIndex
(Task 7 review — CRITICAL).

The off-loop capture-ring runs a CONSUMER THREAD that mutates a PerRequestIndex
(note_rows / mark_finished) while the engine/retrieval thread pops+frees the SAME index MID-serving
(get_ring_per_request). PerRequestIndex has no lock (by contract we may not add one), so
OffLoopRingDrain owns a lock (`_index_lock`) and both sides serialize every index access through it.

The decisive race (documented in the fix): a consumer `mark_finished(rB)` -> `_deliverable.append(rB)`
that lands AFTER `pop_deliverable` scanned past rB but BEFORE its `self._deliverable = []` rebind is
discarded with the old list; `mark_finished`'s `if not e["finished"]` guard blocks the only re-append,
so rB is finished in `_entries` yet never re-enters `_deliverable` -> lost forever (the serve client
hangs).

Two tests drive the REAL guarded paths (producer: the drain's `_handle_finish`; retriever: the exact
pop+free critical section `get_ring_per_request` uses):
  * `test_injected_rebind_race_...` is DETERMINISTIC: a test-only `PerRequestIndex` SUBCLASS gates the
    pop's `_deliverable = []` rebind so a concurrent `mark_finished` append lands in the discard window
    on demand. WITHOUT the lock the request is lost; WITH the lock the producer cannot append until
    pop releases the lock, so it is delivered on the next pop. The base pop_deliverable/mark_finished
    are 100% the real methods (a subclass is not a modification of PerRequestIndex).
  * `test_concurrent_delivery_is_lossless_with_lock` / `..._worker_...` are stochastic hammering
    regression guards: with the lock every finished request is delivered exactly once, no error.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_per_request_concurrency.py -q
"""
import contextlib
import tempfile
import threading

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker

_HIDDEN = 4
_N_REQS = 4000          # hammering-test size


def _make_drain(index=None):
    """A per_request OffLoopRingDrain whose consumer thread is NOT started — the test drives the
    guarded index methods directly from two threads, so we exercise the LOCK, not the ring D2H.
    `index` lets a test pass an instrumented PerRequestIndex subclass."""
    dtype = torch.float32
    ring = GpuCaptureRing(row_bytes=_HIDDEN * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=64, device="cpu", dtype=dtype, row_shape=(_HIDDEN,))
    hs_buf = torch.zeros(65, _HIDDEN, dtype=dtype)
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [_HIDDEN], "hidden": _HIDDEN}
    tmp = tempfile.mkdtemp(prefix="pr_race_")
    kw = {"per_request": True}
    if index is not None:
        kw["index"] = index
    return OffLoopRingDrain(ring, [(1, hs_buf)], tmp, header, **kw)


def _guarded_pop_free(drain):
    """EXACTLY the atomic critical section get_ring_per_request runs: pop the finished set + free
    those ids under the drain's lock, then return the delivered ids (marshaling happens off-lock)."""
    with drain._index_lock:
        popped = drain.index.pop_deliverable()
        for rid, _ in popped:
            drain.index.free(rid)
    return [rid for rid, _ in popped]


# ------------------------- DETERMINISTIC race (RED without the lock) -------------------------
class _GatedIndex(PerRequestIndex):
    """Test-only PerRequestIndex subclass that deterministically drives the lost-`_deliverable`
    append race. When ARMED, the pop's rebind (`self._deliverable = []`) blocks on a gate so a
    concurrent `mark_finished` append can be timed to land in the discard window. The base
    pop_deliverable / mark_finished are inherited UNCHANGED — this only instruments the rebind."""

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_gate_armed", False)
        object.__setattr__(self, "_rebind_reached", threading.Event())
        object.__setattr__(self, "_may_rebind", threading.Event())

    def __setattr__(self, name, value):
        if name == "_deliverable" and getattr(self, "_gate_armed", False) and value == []:
            # pop is at the discard point; announce it and wait so a concurrent mark_finished append
            # can land on the OLD list before this rebind drops it (bounded so the WITH-lock case,
            # where the producer is blocked on the lock and never appends, just times out).
            self._rebind_reached.set()
            self._may_rebind.wait(timeout=1.0)
        object.__setattr__(self, name, value)


def _injected_scenario(guard: bool) -> bool:
    """Run ONE deterministic interleaving; return True iff request 'rB' is delivered (not lost).

    Retriever: a single guarded pop over an EMPTY _deliverable -> loop exits at once -> the rebind
    gate blocks. Producer: once pop is at the rebind, finish rB (guarded mark_finished -> append).
    WITHOUT the lock the append lands on the soon-discarded old list -> lost. WITH the lock the
    producer is blocked acquiring _index_lock until pop returns, so no append is lost -> rB is
    delivered by the final pop."""
    idx = _GatedIndex()
    drain = _make_drain(index=idx)
    if not guard:
        drain._index_lock = contextlib.nullcontext()   # bypass the fix -> reproduce the race
    idx.note_rows("rB", 1, torch.zeros(1, _HIDDEN))     # rB live but not finished
    idx._gate_armed = True                              # next rebind blocks on the gate
    delivered: list = []

    def retriever():
        delivered.extend(_guarded_pop_free(drain))      # empty pop -> blocks at the rebind gate

    def producer():
        idx._rebind_reached.wait(timeout=2.0)           # wait until pop is exactly at the rebind
        drain._handle_finish("rB")                      # guarded mark_finished -> _deliverable.append
        idx._may_rebind.set()                           # release pop's rebind

    tr = threading.Thread(target=retriever); tp = threading.Thread(target=producer)
    tr.start(); tp.start(); tr.join(timeout=10); tp.join(timeout=10)
    idx._may_rebind.set()                               # unstick any WITH-lock timeout wait
    idx._gate_armed = False                             # final pop must not block
    delivered.extend(_guarded_pop_free(drain))          # collect rB if the lock kept it
    return "rB" in delivered


def test_injected_rebind_race_lost_without_lock_safe_with_lock():
    """RED-then-GREEN, DETERMINISTIC. Without the drain lock the injected append is discarded by
    pop's rebind -> rB LOST. With the lock the producer cannot append mid-pop -> rB delivered."""
    assert _injected_scenario(guard=False) is False, (
        "expected the UNLOCKED index to lose rB (append discarded by pop's rebind) — race did not fire")
    assert _injected_scenario(guard=True) is True, (
        "with the drain lock rB must be delivered, never lost")


# ------------------------- stochastic hammering (GREEN regression guard) -------------------------
def _run_hammer(guard: bool, n_reqs: int = _N_REQS):
    """Producer marks n_reqs requests finished via the guarded `_handle_finish`; a retriever spins
    the guarded pop+free. Returns (delivered_list, errors)."""
    drain = _make_drain()
    if not guard:
        drain._index_lock = contextlib.nullcontext()
    index = drain.index
    ids = [f"r{i}" for i in range(n_reqs)]
    for rid in ids:
        index.note_rows(rid, 1, torch.zeros(1, _HIDDEN))
    delivered: list = []
    errors: list = []
    producer_done = threading.Event()
    barrier = threading.Barrier(2)

    def producer():
        try:
            barrier.wait()
            for rid in ids:
                drain._handle_finish(rid)
        except BaseException as e:              # noqa: BLE001
            errors.append(("producer", repr(e)))
        finally:
            producer_done.set()

    def retriever():
        try:
            barrier.wait()
            while True:
                got = _guarded_pop_free(drain)
                delivered.extend(got)
                if producer_done.is_set() and not got and not _guarded_pop_free(drain):
                    break
        except BaseException as e:              # noqa: BLE001
            errors.append(("retriever", repr(e)))

    tp = threading.Thread(target=producer); tr = threading.Thread(target=retriever)
    tp.start(); tr.start(); tp.join(timeout=120); tr.join(timeout=120)
    if guard:
        delivered.extend(_guarded_pop_free(drain))
    return delivered, errors


def test_concurrent_delivery_is_lossless_with_lock():
    """WITH the drain-owned lock: every finished request delivered EXACTLY once (none lost, none
    double), no dict-mutation error."""
    delivered, errors = _run_hammer(guard=True)
    assert errors == [], f"guarded run raised: {errors}"
    assert len(delivered) == len(set(delivered)), "a request was delivered more than once"
    assert set(delivered) == {f"r{i}" for i in range(_N_REQS)}, "a finished request was lost under the lock"


def test_worker_get_ring_per_request_is_lock_guarded():
    """Drive the REAL worker `get_ring_per_request` (marshal + stash + free) on the retriever side
    concurrently with the consumer's guarded finishes. Every request served exactly once, no error
    — the worker's retrieval critical section uses the same drain lock end-to-end."""
    drain = _make_drain()
    w = ProbeHiddenStatesWorker()
    w._conf = {"hidden_size": _HIDDEN, "num_layers": 1}
    w._hs_drain = drain
    n = 800
    served: set = set()
    errors: list = []
    producer_done = threading.Event()
    barrier = threading.Barrier(2)

    def producer():
        try:
            barrier.wait()
            for i in range(n):
                rid = f"w{i}"
                with drain._index_lock:
                    drain.index.note_rows(rid, 1, torch.zeros(1, _HIDDEN))
                drain._handle_finish(rid)
        except BaseException as e:              # noqa: BLE001
            errors.append(("producer", repr(e)))
        finally:
            producer_done.set()

    def retriever():
        try:
            barrier.wait()
            i = 0
            # Each call bulk-pops EVERY currently-finished request into the stash (and frees it),
            # returning the asked one. Keep asking until the producer is done and the index is fully
            # drained into the stash.
            while not errors:
                blob = w.get_ring_per_request(f"w{i % n}")
                if blob is not None:
                    served.add(f"w{i % n}")
                i += 1
                if producer_done.is_set() and not drain.index.live_req_ids():
                    break
        except BaseException as e:              # noqa: BLE001
            errors.append(("retriever", repr(e)))

    tp = threading.Thread(target=producer); tr = threading.Thread(target=retriever)
    tp.start(); tr.start(); tp.join(timeout=120); tr.join(timeout=120)
    # Final flush: bulk-pop anything a late finish left deliverable in the index into the stash,
    # then account served (popped+asked) UNION stash (popped, not yet asked).
    w.get_ring_per_request("__flush__")
    served |= set(getattr(w, "_ring_perreq_stash", {}).keys())
    assert errors == [], f"worker retrieval raced: {errors}"
    assert not drain.index.live_req_ids(), "requests left undelivered in the index"
    assert served == {f"w{i}" for i in range(n)}, f"lost {n - len(served)} worker-path request(s)"


if __name__ == "__main__":
    for t in (test_injected_rebind_race_lost_without_lock_safe_with_lock,
              test_concurrent_delivery_is_lossless_with_lock,
              test_worker_get_ring_per_request_is_lock_guarded):
        t()
        print(f"PASS  {t.__name__}")
    print("VERDICT: PASS (3/3)")
