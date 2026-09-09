"""Unit tests for vllm_hook_plugins.graph.offload_process (Task 4: OffloadProcess protocol).

No GPU, no engine -- pure filesystem + threading. Run:
  pytest tests/unit/test_offload_process.py -vv

Covers the brief's Step 1 list:
  (a) submit a temp file with a fake transfer_fn that records its calls -> wait() returns True
      after the transfer runs.
  (b) a transfer_fn that fails twice then succeeds is retried and eventually confirmed (never
      dropped).
  (c) the real shutil.copy default actually copies -- dest file bytes match src bytes.
Plus poll_done()'s "since last poll" semantics and close() drain+join.
"""
import os
import threading
import time

import pytest

from vllm_hook_plugins.graph.offload_process import OffloadProcess


# ---------------------------------------------------------------------------
# (a) fake transfer_fn, wait() confirms completion
# ---------------------------------------------------------------------------


def test_submit_then_wait_returns_true_and_records_call(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello world")
    dest = tmp_path / "dest.bin"

    calls = []

    def fake_transfer(src_path, dest_path):
        calls.append((src_path, dest_path))

    op = OffloadProcess(transfer_fn=fake_transfer)
    try:
        op.submit("req-1", str(src), str(dest))
        assert op.wait("req-1", timeout=5.0) is True
        assert calls == [(str(src), str(dest))]
    finally:
        op.close()


def test_wait_times_out_for_unknown_req():
    op = OffloadProcess(transfer_fn=lambda s, d: None)
    try:
        assert op.wait("never-submitted", timeout=0.1) is False
    finally:
        op.close()


def test_submit_is_non_blocking(tmp_path):
    """submit() must return immediately even if the transfer_fn is slow."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest = tmp_path / "dest.bin"
    release = threading.Event()

    def slow_transfer(src_path, dest_path):
        release.wait(timeout=5.0)

    op = OffloadProcess(transfer_fn=slow_transfer)
    try:
        op.submit("req-slow", str(src), str(dest))
        # submit() returned -- the transfer is still blocked on `release`. wait() with a tiny
        # timeout must NOT be satisfied yet (proves submit didn't block until completion).
        assert op.wait("req-slow", timeout=0.2) is False
        release.set()
        assert op.wait("req-slow", timeout=5.0) is True
    finally:
        op.close()


def test_submit_is_non_blocking_under_backlog(tmp_path):
    """Critical fix: submit() must stay non-blocking even with a deep backlog queued behind a
    stalled worker -- the old bounded (maxsize=64) queue would block a caller past that depth."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    release = threading.Event()

    def blocking_transfer(src_path, dest_path):
        release.wait(timeout=10.0)

    op = OffloadProcess(transfer_fn=blocking_transfer)
    try:
        n = 200  # well past the old maxsize=64 bound
        start = time.time()
        for i in range(n):
            op.submit(f"req-backlog-{i}", str(src), str(tmp_path / f"out-{i}.bin"))
        elapsed = time.time() - start
        assert elapsed < 2.0, f"submit() blocked under backlog: {elapsed:.2f}s for {n} jobs"
    finally:
        release.set()
        op.close()


# ---------------------------------------------------------------------------
# (b) never-drop: fails twice then succeeds -> held + retried, eventually confirmed
# ---------------------------------------------------------------------------


def test_failing_transfer_is_retried_and_eventually_confirmed(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"retry me")
    dest = tmp_path / "dest.bin"

    attempts = []

    def flaky_transfer(src_path, dest_path):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient failure")
        # third attempt succeeds

    op = OffloadProcess(transfer_fn=flaky_transfer)
    try:
        op.submit("req-flaky", str(src), str(dest))
        assert op.wait("req-flaky", timeout=5.0) is True
        assert len(attempts) == 3
    finally:
        op.close()


def test_always_failing_transfer_never_drops_the_job(tmp_path):
    """A transfer_fn that never succeeds must be held (retried up to the bound) and must NOT be
    silently confirmed done -- wait() stays False, and poll_done() never reports the req."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"never succeeds")
    dest = tmp_path / "dest.bin"

    def always_fails(src_path, dest_path):
        raise RuntimeError("permanent failure")

    op = OffloadProcess(transfer_fn=always_fails, retry_delay=0.01)
    try:
        op.submit("req-doomed", str(src), str(dest))
        assert op.wait("req-doomed", timeout=1.0) is False
        assert "req-doomed" not in op.poll_done()
    finally:
        op.close()


# ---------------------------------------------------------------------------
# poll_failed() -- the machine-readable give-up signal
# ---------------------------------------------------------------------------


def test_poll_failed_reports_giveup_wait_wakes_promptly_but_false(tmp_path):
    """A job that exhausts retries: shows up in poll_failed(), never poll_done(), and wait()
    wakes PROMPTLY (via the give-up Event.set(), not by timing out) but still reports False."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest = tmp_path / "dest.bin"

    def always_fails(src_path, dest_path):
        raise RuntimeError("permanent failure")

    op = OffloadProcess(transfer_fn=always_fails, retry_delay=0.01, max_retries=3)
    try:
        op.submit("req-giveup", str(src), str(dest))
        start = time.time()
        # Large timeout on purpose -- the give-up should wake this well before it elapses,
        # proving the wakeup is a real signal and not merely a timeout.
        result = op.wait("req-giveup", timeout=10.0)
        elapsed = time.time() - start
        assert result is False
        assert elapsed < 2.0, f"wait() did not wake promptly on give-up: {elapsed:.2f}s"
        failed = op.poll_failed()
        assert "req-giveup" in failed
        assert "req-giveup" not in op.poll_done()
        assert op.poll_failed() == []  # drained -- "since last poll" semantics
    finally:
        op.close()


# ---------------------------------------------------------------------------
# (c) real shutil.copy default
# ---------------------------------------------------------------------------


def test_default_transfer_fn_copies_bytes_identically(tmp_path):
    src = tmp_path / "src.bin"
    payload = os.urandom(4096)
    src.write_bytes(payload)
    dest = tmp_path / "sub" / "dest.bin"
    os.makedirs(dest.parent, exist_ok=True)

    op = OffloadProcess()  # default transfer_fn
    try:
        op.submit("req-real", str(src), str(dest))
        assert op.wait("req-real", timeout=5.0) is True
        assert dest.read_bytes() == payload
    finally:
        op.close()


# ---------------------------------------------------------------------------
# poll_done() -- "since last poll" semantics
# ---------------------------------------------------------------------------


def test_poll_done_returns_each_finished_req_exactly_once(tmp_path):
    src_a = tmp_path / "a.bin"
    src_b = tmp_path / "b.bin"
    src_a.write_bytes(b"a")
    src_b.write_bytes(b"b")

    op = OffloadProcess()
    try:
        op.submit("req-a", str(src_a), str(tmp_path / "a_out.bin"))
        assert op.wait("req-a", timeout=5.0) is True
        first = op.poll_done()
        assert "req-a" in first

        # Nothing new finished since the first poll -> empty.
        second = op.poll_done()
        assert second == []

        op.submit("req-b", str(src_b), str(tmp_path / "b_out.bin"))
        assert op.wait("req-b", timeout=5.0) is True
        third = op.poll_done()
        assert third == ["req-b"]
    finally:
        op.close()


# ---------------------------------------------------------------------------
# close() -- drains the queue and joins the worker
# ---------------------------------------------------------------------------


def test_close_drains_pending_jobs_before_returning(tmp_path):
    n = 5
    srcs = []
    dests = []
    for i in range(n):
        s = tmp_path / f"src_{i}.bin"
        s.write_bytes(f"payload-{i}".encode())
        srcs.append(s)
        dests.append(tmp_path / f"dest_{i}.bin")

    op = OffloadProcess()
    for i in range(n):
        op.submit(f"req-{i}", str(srcs[i]), str(dests[i]))
    op.close()

    for i in range(n):
        assert dests[i].read_bytes() == srcs[i].read_bytes()


def test_close_returns_promptly_when_worker_is_stuck(tmp_path):
    """Important-1 fix: close() must be TIME-BOUNDED. A transfer_fn that blocks forever (never
    raises, so the retry loop never gets a chance to give up either) must not hang shutdown --
    close() returns within its timeout regardless of the stuck worker."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest = tmp_path / "dest.bin"
    never = threading.Event()  # deliberately never set -- the transfer blocks forever

    def stuck_transfer(src_path, dest_path):
        never.wait()

    op = OffloadProcess(transfer_fn=stuck_transfer)
    op.submit("req-stuck", str(src), str(dest))
    start = time.time()
    op.close(timeout=1.0)
    elapsed = time.time() - start
    assert elapsed < 3.0, f"close() did not return promptly: {elapsed:.2f}s"
    # (the worker thread itself is left running, blocked forever on `never` -- documented
    # residual limitation; it's a daemon thread so it doesn't block process/test-suite exit)


# ---------------------------------------------------------------------------
# resubmit-while-in-flight guard
# ---------------------------------------------------------------------------


def test_resubmit_while_in_flight_raises(tmp_path):
    """Important-3 fix: submit() must reject a second submission for a req_id that is still in
    flight (queued or mid-retry) -- a shared Event would otherwise let a stale completion satisfy
    the new caller's wait(), and poll_done() could double-report."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest = tmp_path / "dest.bin"
    release = threading.Event()

    def slow_transfer(src_path, dest_path):
        release.wait(timeout=5.0)

    op = OffloadProcess(transfer_fn=slow_transfer)
    try:
        op.submit("req-dup", str(src), str(dest))
        with pytest.raises(ValueError):
            op.submit("req-dup", str(src), str(dest))
    finally:
        release.set()
        op.close()


def test_poll_based_resubmit_does_not_leak_stale_completion(tmp_path):
    """Review-round-2 regression: the settle transition in _mark_done (inflight.discard ->
    result write -> done.append -> ev.set()) must be atomic with submit()'s resubmit check.

    Before the fix, poll_done() could observe job-1's completion (the append happened) BEFORE its
    ev.set() actually ran. A caller that detects completion via poll_done() (a pattern the module
    docstring explicitly endorses) and immediately resubmits the same req_id could have submit()'s
    ev.clear() run BEFORE the old job's still-pending ev.set() -- which then overwrites it, letting
    job-1's stale completion satisfy job-2's wait() even though job-2 has not run at all.

    Reproduced deterministically (no sleeps) by intercepting the shared Event's .set() method to
    pause the worker thread at exactly that gap, using a background thread for the poll+resubmit so
    a correct (lock-serialized) fix -- which blocks poll_done()/submit() until the old job's set()
    actually finishes -- cannot deadlock this test (a FIXED _mark_done holds the shared lock across
    its own ev.set(), so poll_done()/submit() called on the MAIN thread here would hang until the
    interceptor's own internal timeout, not a real deadlock, but not clean either -- the background
    thread avoids that entirely).
    """
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest1 = tmp_path / "dest1.bin"
    dest2 = tmp_path / "dest2.bin"

    job1_release = threading.Event()
    job2_release = threading.Event()  # deliberately never set -- job-2 must not complete here
    reached_set_point = threading.Event()
    release_set = threading.Event()

    def transfer_fn(src_path, dest_path):
        if dest_path == str(dest1):
            job1_release.wait(timeout=10.0)
        elif dest_path == str(dest2):
            job2_release.wait(timeout=10.0)

    op = OffloadProcess(transfer_fn=transfer_fn)
    try:
        op.submit("req-race", str(src), str(dest1))

        # Grab the Event submit() created for this req_id and intercept .set() on that exact
        # instance so we can pause the worker deterministically right after the "settle" bookkeeping
        # but before the completion signal actually fires -- reproducing the race window without
        # relying on timing luck.
        ev = op._events["req-race"]
        real_set = ev.set

        def intercepted_set():
            reached_set_point.set()
            release_set.wait(timeout=10.0)
            real_set()

        ev.set = intercepted_set

        job1_release.set()  # now let job-1's transfer_fn (and _mark_done) actually run

        assert reached_set_point.wait(timeout=5.0), "worker never reached the ev.set() point"
        # job-1's completion is already bookkept (append happened) even though ev.set() has NOT run
        # yet -- the exact race precondition. Peek the list directly instead of calling poll_done()
        # here: post-fix, poll_done() shares the lock _mark_done is still holding at this point, so
        # calling it from THIS thread would legitimately block until release_set fires below (that
        # is the fix working) -- only the background thread below is allowed to observe that block.
        assert "req-race" in op._done

        # Poll-based resubmit, from a background thread: a correct fix serializes poll_done() +
        # submit() behind the same lock _mark_done is still holding, so this must not run on the
        # main thread here.
        polled = []
        submit_done = threading.Event()

        def do_poll_and_resubmit():
            polled.extend(op.poll_done())
            op.submit("req-race", str(src), str(dest2))
            submit_done.set()

        t = threading.Thread(target=do_poll_and_resubmit, daemon=True)
        t.start()
        # Give the background thread a brief, bounded chance to run to completion BEFORE releasing
        # the old job's set() -- on BUGGY code there is no lock contention so this reliably finishes
        # well within the bound (pure in-memory dict/set/list ops, no I/O); on FIXED code it is
        # deterministically still blocked on the shared lock regardless of this bound, so the wait
        # simply elapses. Either way the outcome below is not timing-dependent.
        submit_done.wait(timeout=0.5)

        # Now let the OLD job's (job-1's) delayed ev.set() actually run.
        release_set.set()
        t.join(timeout=5.0)
        assert submit_done.is_set(), "resubmit never completed"
        assert polled == ["req-race"]

        # job-2 is still blocked on job2_release (never set) -- wait() must report False, not leak
        # job-1's stale completion into job-2's signal.
        assert op.wait("req-race", timeout=1.0) is False
    finally:
        job1_release.set()
        job2_release.set()
        op.close()


def test_resubmit_after_completion_is_allowed(tmp_path):
    """The guard is scoped to IN-FLIGHT duplicates only -- once a req_id has settled (completed),
    submitting it again (a fresh delivery reusing an old id) must not raise."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    dest1 = tmp_path / "dest1.bin"
    dest2 = tmp_path / "dest2.bin"

    op = OffloadProcess()
    try:
        op.submit("req-reuse", str(src), str(dest1))
        assert op.wait("req-reuse", timeout=5.0) is True
        op.submit("req-reuse", str(src), str(dest2))  # must not raise
        assert op.wait("req-reuse", timeout=5.0) is True
    finally:
        op.close()
