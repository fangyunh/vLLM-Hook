import queue
import types
from vllm_hook_plugins.graph import writer_process as wp


class _FakeWriter:
    """Exercises submit()'s block path without spawning a child."""
    def __init__(self, maxsize):
        self._inq = queue.Queue(maxsize=maxsize)
        self._put_timeout = 0.2
        self._procs = [types.SimpleNamespace(is_alive=lambda: True)]  # multi-writer: N children
    alive = wp.WriterProcess.alive
    submit = wp.WriterProcess.submit


def test_block_true_enqueues_when_space():
    w = _FakeWriter(maxsize=2)
    ok = w.submit("hs", {}, "/d", "m", 0, False, False, "x.pt", block=True)
    assert ok is True
    assert w._inq.qsize() == 1


def test_block_false_returns_false_when_full():
    w = _FakeWriter(maxsize=1)
    w._inq.put_nowait("filler")
    ok = w.submit("hs", {}, "/d", "m", 0, False, False, "x.pt", block=False)
    assert ok is False  # non-blocking path unchanged


def test_block_true_times_out_to_false_when_full_and_undrained():
    w = _FakeWriter(maxsize=1)
    w._inq.put_nowait("filler")  # never drained -> bounded block must time out, not hang
    ok = w.submit("hs", {}, "/d", "m", 0, False, False, "x.pt", block=True)
    assert ok is False
