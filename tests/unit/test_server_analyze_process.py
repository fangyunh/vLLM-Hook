"""Unit tests for vllm_hook_plugins.graph.server_analyze_process (Task 8: ServerAnalyzeProcess).

No GPU, no engine. Mirrors test_offload_process.py's style. Run:
  pytest tests/unit/test_server_analyze_process.py -vv

Covers the brief's Step 1 list:
  (a) a fake reducible analyzer run over a synthetic per-request artifact -- from a HOST BUFFER
      (source_kind="inflight") AND from a WRITTEN FILE (source_kind="from_disk") -- returns the
      SAME small result as calling that analyzer directly, and the two sources agree with each
      other for equivalent data.
  (b) the process/child does not import or initialize CUDA: structural env-var contract (the mp
      backend blanks CUDA_VISIBLE_DEVICES in the PARENT before Process.start(), and restores it
      right after -- the load-bearing timing writer_process.py established).
  (c) failure isolation: one request's analyze error is caught, reported via poll_failed() with
      its error message, and does NOT kill the worker -- other requests still complete, including
      one submitted AFTER the failure.
Plus the injectable analyze_fn seam, resubmit-in-flight guard, poll_done() since-last-poll
semantics, and a bounded close().

NOTE on scope: no test here spawns a REAL torch.multiprocessing child running the DEFAULT
registry-based analyze_fn end-to-end. A spawned child is a fresh interpreter that must re-import
`vllm_hook_plugins` from scratch (~tens of seconds -- the same cost writer_process.py's child
already pays once at worker install, amortized there behind model loading); paying it inside a
no-GPU unit test would make the suite unacceptably slow to run "a few times for flake-freedom".
test_multi_writer.py established the same discipline for WriterProcess ("Does NOT spawn real
children ... it checks the env plumbing ... with fakes"). The mp env-var contract is instead
verified structurally (below) by swapping in a fake torch.multiprocessing context that records
what the real one would have received; the DEFAULT analyze_fn's correctness (registry lookup +
probes=/run_id= dispatch) is exercised directly on the THREAD backend, which runs the identical
`_consume_loop` core -- see the module docstring's PROCESS-VS-THREAD section.
"""
import json
import os
import threading
import time
import types

import pytest

from vllm_hook_plugins.graph import server_analyze_process as sap
from vllm_hook_plugins.graph.server_analyze_process import ServerAnalyzeProcess
from vllm_hook_plugins.registry import PluginRegistry


# ---------------------------------------------------------------------------
# A fake reducible analyzer -- same __init__(hook_dir, layer_to_heads) / .analyze(analyzer_spec=,
# run_id=, probes=) contract as the real analyzers (HiddenStatesAnalyzer etc.), needs no torch/
# model. probes= reads probes["hs_cache"]; run_id= reads hook_dir/run_id/artifact.json -- the
# "written file" source. reduce="mean" mirrors HiddenStatesAnalyzer's reducible case.
# ---------------------------------------------------------------------------


class _FakeReducibleAnalyzer:
    ACCEPTS = "hs"  # capability-declaration convention the real analyzers use (attn_tracker etc.)

    def __init__(self, hook_dir, layer_to_heads=None):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads or {}

    def analyze(self, analyzer_spec=None, run_id=None, probes=None):
        if probes is not None:
            hs_cache = probes["hs_cache"]
        elif run_id is not None:
            path = os.path.join(self.hook_dir, run_id, "artifact.json")
            with open(path) as f:
                hs_cache = json.load(f)
        else:
            raise ValueError("_FakeReducibleAnalyzer.analyze: pass probes= or run_id=")
        reduce = (analyzer_spec or {}).get("reduce", "mean")
        result = {}
        for layer, data in hs_cache.items():
            values = data["hidden_states"]
            if reduce == "mean":
                result[layer] = sum(values) / len(values)
            elif reduce == "boom":
                raise RuntimeError(f"forced failure for layer {layer}")
            else:
                raise NotImplementedError(reduce)
        return {"hidden_states": result}


_FAKE_ANALYZER_NAME = "__test_server_analyze_fake_reducible__"
PluginRegistry.register_analyzer(_FAKE_ANALYZER_NAME, _FakeReducibleAnalyzer)


def _write_artifact(hook_dir, run_id, hs_cache):
    run_dir = os.path.join(hook_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "artifact.json"), "w") as f:
        json.dump(hs_cache, f)


# ---------------------------------------------------------------------------
# (a) host-buffer == file == direct-call, via the DEFAULT registry-based analyze_fn on the
#     thread backend (use_process=False keeps the test fast; it runs the identical _consume_loop
#     core the mp backend uses -- see the module docstring).
# ---------------------------------------------------------------------------


def test_inflight_source_matches_calling_the_analyzer_directly(tmp_path):
    hook_dir = str(tmp_path)
    probes = {"hs_cache": {"0": {"hidden_states": [1.0, 2.0, 3.0]},
                           "1": {"hidden_states": [10.0, 20.0]}}}
    spec = {"reduce": "mean"}
    expected = _FakeReducibleAnalyzer(hook_dir, {}).analyze(analyzer_spec=spec, probes=probes)

    proc = ServerAnalyzeProcess(use_process=False)
    try:
        source = {"hook_dir": hook_dir, "probes": probes}
        assert proc.submit("r-inflight", "inflight", source, _FAKE_ANALYZER_NAME, spec) is True
        result = proc.wait("r-inflight", timeout=5.0)
        assert result == expected == {"hidden_states": {"0": 2.0, "1": 15.0}}
    finally:
        proc.close()


def test_from_disk_source_matches_calling_the_analyzer_directly(tmp_path):
    hook_dir = str(tmp_path)
    run_id = "run-abc"
    hs_cache = {"0": {"hidden_states": [4.0, 6.0]}}
    _write_artifact(hook_dir, run_id, hs_cache)
    spec = {"reduce": "mean"}
    expected = _FakeReducibleAnalyzer(hook_dir, {}).analyze(analyzer_spec=spec, run_id=run_id)

    proc = ServerAnalyzeProcess(use_process=False)
    try:
        source = {"hook_dir": hook_dir, "run_id": run_id}
        assert proc.submit("r-disk", "from_disk", source, _FAKE_ANALYZER_NAME, spec) is True
        result = proc.wait("r-disk", timeout=5.0)
        assert result == expected == {"hidden_states": {"0": 5.0}}
    finally:
        proc.close()


def test_inflight_and_from_disk_agree_for_equivalent_data(tmp_path):
    """The SAME logical artifact, delivered through the two different sources, must reduce to the
    identical small result -- the load-bearing claim for the router's inflight/from_disk split."""
    hook_dir = str(tmp_path)
    hs_cache = {"0": {"hidden_states": [2.0, 4.0, 6.0]}}
    spec = {"reduce": "mean"}
    run_id = "run-same-data"
    _write_artifact(hook_dir, run_id, hs_cache)

    proc = ServerAnalyzeProcess(use_process=False)
    try:
        probes = {"hs_cache": hs_cache}
        proc.submit("r-a", "inflight", {"hook_dir": hook_dir, "probes": probes},
                    _FAKE_ANALYZER_NAME, spec)
        proc.submit("r-b", "from_disk", {"hook_dir": hook_dir, "run_id": run_id},
                    _FAKE_ANALYZER_NAME, spec)
        r_inflight = proc.wait("r-a", timeout=5.0)
        r_disk = proc.wait("r-b", timeout=5.0)
        assert r_inflight == r_disk == {"hidden_states": {"0": 4.0}}
    finally:
        proc.close()


def test_unknown_analyzer_name_surfaces_as_a_failure_not_a_crash(tmp_path):
    """The default analyze_fn's registry lookup failure is a normal analyze-error -- caught and
    isolated like any other, not a special case."""
    proc = ServerAnalyzeProcess(use_process=False)
    try:
        proc.submit("r-bad-name", "inflight", {"hook_dir": str(tmp_path), "probes": {"hs_cache": {}}},
                    "__no_such_analyzer__", None)
        assert proc.wait("r-bad-name", timeout=5.0) is None
        failed = proc.poll_failed()
        assert len(failed) == 1 and failed[0][0] == "r-bad-name"
        assert "__no_such_analyzer__" in failed[0][1]
    finally:
        proc.close()


# ---------------------------------------------------------------------------
# injectable analyze_fn seam (mirrors OffloadProcess's transfer_fn)
# ---------------------------------------------------------------------------


def test_injected_analyze_fn_forces_thread_backend_and_is_used():
    calls = []

    def fake_fn(source_kind, source, analyzer_name, analyzer_spec):
        calls.append((source_kind, analyzer_name, analyzer_spec))
        return {"echo": source_kind}

    # use_process=True is passed deliberately -- an injected analyze_fn must override it (a
    # closure is not guaranteed picklable across an mp spawn boundary).
    proc = ServerAnalyzeProcess(analyze_fn=fake_fn, use_process=True)
    try:
        assert proc._use_process is False
        assert proc.submit("r1", "inflight", {"probes": {}}, "anything", {"reduce": "mean"}) is True
        assert proc.wait("r1", timeout=5.0) == {"echo": "inflight"}
        assert calls == [("inflight", "anything", {"reduce": "mean"})]
    finally:
        proc.close()


# ---------------------------------------------------------------------------
# (c) failure isolation
# ---------------------------------------------------------------------------


def test_failure_is_isolated_other_requests_still_complete_and_worker_survives():
    def flaky_fn(source_kind, source, analyzer_name, analyzer_spec):
        if source.get("boom"):
            raise RuntimeError("synthetic failure")
        return {"ok": True}

    proc = ServerAnalyzeProcess(analyze_fn=flaky_fn)
    try:
        proc.submit("good-1", "inflight", {"probes": {}}, "x", None)
        proc.submit("bad-1", "inflight", {"probes": {}, "boom": True}, "x", None)
        proc.submit("good-2", "inflight", {"probes": {}}, "x", None)

        assert proc.wait("good-1", timeout=5.0) == {"ok": True}
        assert proc.wait("good-2", timeout=5.0) == {"ok": True}
        assert proc.wait("bad-1", timeout=5.0) is None  # failure -> None, never raises to the caller

        failed = proc.poll_failed()
        assert len(failed) == 1
        assert failed[0][0] == "bad-1"
        assert "synthetic failure" in failed[0][1]
        assert "bad-1" not in proc.poll_done()

        # Prove the worker genuinely survived the failure (not just that it hadn't crashed yet).
        proc.submit("good-3", "inflight", {"probes": {}}, "x", None)
        assert proc.wait("good-3", timeout=5.0) == {"ok": True}
    finally:
        proc.close()


# ---------------------------------------------------------------------------
# poll_done() since-last-poll semantics; resubmit-in-flight guard; bounded close()
# ---------------------------------------------------------------------------


def test_poll_done_returns_each_finished_req_exactly_once():
    def fn(source_kind, source, analyzer_name, analyzer_spec):
        return {"v": source["n"]}

    proc = ServerAnalyzeProcess(analyze_fn=fn)
    try:
        proc.submit("a", "inflight", {"probes": {}, "n": 1}, "x", None)
        assert proc.wait("a", timeout=5.0) == {"v": 1}
        first = proc.poll_done()
        assert "a" in first
        assert proc.poll_done() == []  # nothing new since the first poll

        proc.submit("b", "inflight", {"probes": {}, "n": 2}, "x", None)
        assert proc.wait("b", timeout=5.0) == {"v": 2}
        assert proc.poll_done() == ["b"]
    finally:
        proc.close()


def test_resubmit_while_in_flight_raises():
    release = threading.Event()

    def slow_fn(source_kind, source, analyzer_name, analyzer_spec):
        release.wait(timeout=5.0)
        return {"ok": True}

    proc = ServerAnalyzeProcess(analyze_fn=slow_fn)
    try:
        proc.submit("dup", "inflight", {"probes": {}}, "x", None)
        with pytest.raises(ValueError):
            proc.submit("dup", "inflight", {"probes": {}}, "x", None)
    finally:
        release.set()
        proc.close()


def test_wait_times_out_for_unknown_req():
    proc = ServerAnalyzeProcess(analyze_fn=lambda *a: {"ok": True})
    try:
        assert proc.wait("never-submitted", timeout=0.1) is None
    finally:
        proc.close()


def test_close_returns_promptly_when_worker_is_stuck():
    never = threading.Event()  # deliberately never set -- the analyze blocks forever

    def stuck_fn(source_kind, source, analyzer_name, analyzer_spec):
        never.wait()

    proc = ServerAnalyzeProcess(analyze_fn=stuck_fn)
    proc.submit("stuck", "inflight", {"probes": {}}, "x", None)
    start = time.time()
    proc.close(timeout=1.0)
    elapsed = time.time() - start
    assert elapsed < 3.0, f"close() did not return promptly: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# (b) CUDA-free contract on the mp backend -- structural (see the module docstring on why no test
# here spawns a real production child).
# ---------------------------------------------------------------------------


class _RecordingMpQueue:
    """Stand-in for a torch.multiprocessing.Queue: a real (in-process) queue.Queue underneath, so
    the collector thread's blocking get()/put() still behave correctly -- only Process itself is
    faked (no real child is ever spawned)."""

    def __init__(self, maxsize=0):
        import queue as _q
        self._q = _q.Queue(maxsize=maxsize)

    def cancel_join_thread(self):
        pass

    def put(self, item, timeout=None, block=True):
        self._q.put(item, block=block, timeout=timeout)

    def get(self, timeout=None):
        return self._q.get(timeout=timeout)


class _RecordingProc:
    def __init__(self, target, args, daemon, name, recorder):
        self._recorder = recorder

    def start(self):
        # This is the LOAD-BEARING moment: what the real child would inherit is exactly what is
        # in os.environ right now (spawn snapshots it at Process.start()).
        self._recorder["cuda_visible_at_start"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._recorder["omp_at_start"] = os.environ.get("OMP_NUM_THREADS")

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class _RecordingCtx:
    def __init__(self, recorder):
        self._recorder = recorder

    def Queue(self, maxsize=0):
        return _RecordingMpQueue(maxsize)

    def Process(self, target, args=(), daemon=False, name=None):
        return _RecordingProc(target, args, daemon, name, self._recorder)


def test_process_backend_blanks_cuda_before_start_and_restores_after(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    recorder = {}

    import torch.multiprocessing as tmp
    monkeypatch.setattr(tmp, "get_context", lambda kind: _RecordingCtx(recorder))

    proc = ServerAnalyzeProcess(use_process=True)
    try:
        assert proc._use_process is True
        # What the real child's environment would have snapshotted at spawn time.
        assert recorder["cuda_visible_at_start"] == "", (
            "CUDA_VISIBLE_DEVICES must be blanked in the parent BEFORE Process.start()")
        assert recorder["omp_at_start"] == "1"
        # The parent's own environment is restored immediately after (its CUDA context, if any,
        # is untouched -- only the child's inherited snapshot was ever blanked).
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "3"
        assert os.environ.get("OMP_NUM_THREADS") is None
    finally:
        proc.close()


# ---------------------------------------------------------------------------
# init_server_analyze_process(worker) -- the lazy-start hook (mirrors init_writer_process).
# No real spawn here either, same discipline as the block above / test_multi_writer.py.
# ---------------------------------------------------------------------------


def test_init_default_off_sets_none(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_SERVER_ANALYZE_PROCESS", raising=False)
    w = types.SimpleNamespace()
    sap.init_server_analyze_process(w)
    assert w._server_analyze_process is None


def test_init_is_idempotent(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_SERVER_ANALYZE_PROCESS", raising=False)
    w = types.SimpleNamespace()
    sap.init_server_analyze_process(w)
    w._server_analyze_process = "sentinel"  # simulate an already-started process
    sap.init_server_analyze_process(w)  # must be a no-op (hasattr guard)
    assert w._server_analyze_process == "sentinel"


def test_init_env_on_starts_a_process(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_SERVER_ANALYZE_PROCESS", "1")

    class _Fake:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(sap, "ServerAnalyzeProcess", _Fake)
    w = types.SimpleNamespace()
    sap.init_server_analyze_process(w)
    assert isinstance(w._server_analyze_process, _Fake)


# ---------------------------------------------------------------------------
# submit() returns False when child is not alive (mirrors WriterProcess.submit guard)
# ---------------------------------------------------------------------------


def test_submit_returns_false_and_does_not_enqueue_when_child_not_alive(monkeypatch):
    """Verify that submit() refuses to enqueue and returns False when the mp backend child is
    dead, matching WriterProcess.submit()'s guard (see the finding in code review)."""
    # Simulate a dead process backend
    recorder = {}

    import torch.multiprocessing as tmp
    monkeypatch.setattr(tmp, "get_context", lambda kind: _RecordingCtx(recorder))

    proc = ServerAnalyzeProcess(use_process=True)
    try:
        assert proc._use_process is True
        assert proc.alive() is False  # _RecordingProc.is_alive() returns False

        # Attempt to submit should return False without enqueuing
        result = proc.submit("r-dead", "inflight", {"probes": {}}, "x", None)
        assert result is False, "submit() must return False when child is not alive"

        # Verify the req_id was never added to _inflight (guard cleaned it up)
        assert "r-dead" not in proc._inflight

        # Verify no event was created for this req_id (since we returned early)
        # (it was created inside the lock, but then cleaned up; we can't directly assert
        # on the event dict, but we can verify wait() times out immediately without blocking)
        start = time.time()
        result_wait = proc.wait("r-dead", timeout=0.1)
        elapsed = time.time() - start
        assert result_wait is None
        assert elapsed < 0.5, "wait() should return immediately on a never-submitted req_id"
    finally:
        proc.close()
