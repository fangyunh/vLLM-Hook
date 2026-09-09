# tests/unit/test_writer_pages_done.py
"""AR3: the writer feeder signals req_ids on pages_done only AFTER it has copied a request's
bytes out (pack_tensor_tree succeeding, or the inline handoff-failure fallback writing the
artifact) -- never before, and never on the RAW (pack=0) handoff, which hands the child the
original tensors with no early copy-out point. No GPU, no real spawned child: these drive
WriterProcess._feed()/.poll_pages_done() directly on bare instances built via types.SimpleNamespace
(mirrors the existing _FakeWriter pattern in test_writer_no_engine_inline.py)."""
import queue
import types

from vllm_hook_plugins.graph import writer_process as wp


class _FakeBuffer:
    """Stand-in for pack_tensor_tree's returned buffer -- avoids a real torch tensor + the mp
    shared_memory_() IPC call, which needs a real torch.multiprocessing.Queue consumer."""

    def share_memory_(self):
        pass


def _new_writer(pack: bool, pack_fn):
    w = types.SimpleNamespace()
    w._procs = [types.SimpleNamespace(is_alive=lambda: True)]
    w._pack = pack
    w._pack_fn = pack_fn
    w._q = queue.Queue()
    w._put_timeout = 5.0
    w._inq = queue.Queue()
    w.pages_done = queue.Queue()
    return w


def test_poll_pages_done_drains_nonblocking_in_order():
    w = types.SimpleNamespace(pages_done=queue.Queue())
    w.pages_done.put("a")
    w.pages_done.put("b")
    assert wp.WriterProcess.poll_pages_done(w) == ["a", "b"]
    assert wp.WriterProcess.poll_pages_done(w) == []          # drained -> empty, no blocking


def test_submit_carries_req_ids_and_poll_returns_them_after_feed():
    """Drive _feed once on the PACKED path with a stub pack fn; assert the req_ids submit()
    carried land on pages_done only after the pack 'copies out' the data."""
    w = _new_writer(pack=True, pack_fn=lambda cpu_cache: (_FakeBuffer(), {"manifest": True}))

    w._inq.put(("hs", {"hs_cache": {}}, "/tmp/run", "all_tokens", 0, False, False,
                "hidden_states.pt", ["req-a", "req-b"]))
    w._inq.put(None)                                          # stop _feed after one item

    wp.WriterProcess._feed(w)

    assert wp.WriterProcess.poll_pages_done(w) == ["req-a", "req-b"]
    tag = w._q.get_nowait()[0]                                # packed item reached the mp-queue
    assert tag == "P"                                         # stand-in, tag unchanged


def test_submit_signature_accepts_req_ids_default_none():
    """submit() gains req_ids as an optional keyword; omitting it (every pre-AR3 caller) must
    still enqueue (stored as an empty list), not raise."""
    w = types.SimpleNamespace()
    w._inq = queue.Queue(maxsize=2)
    w._put_timeout = 0.2
    w._procs = [types.SimpleNamespace(is_alive=lambda: True)]
    w.alive = wp.WriterProcess.alive.__get__(w)
    ok = wp.WriterProcess.submit(w, "hs", {}, "/d", "m", 0, False, False, "x.pt", block=True)
    assert ok is True
    item = w._inq.get_nowait()
    assert item[-1] == []                                     # req_ids defaulted to []


def test_submit_req_ids_flow_into_the_queued_item():
    w = types.SimpleNamespace()
    w._inq = queue.Queue(maxsize=2)
    w._put_timeout = 0.2
    w._procs = [types.SimpleNamespace(is_alive=lambda: True)]
    w.alive = wp.WriterProcess.alive.__get__(w)
    ok = wp.WriterProcess.submit(w, "hs", {}, "/d", "m", 0, False, False, "x.pt",
                                 req_ids=["r1", "r2"], block=True)
    assert ok is True
    item = w._inq.get_nowait()
    assert item[-1] == ["r1", "r2"]


def test_inline_fallback_signals_pages_done_after_write_succeeds(monkeypatch):
    """If pack raises, _feed falls back to an inline write_artifact -- that ALSO copies the data
    out, so pages_done must still be signaled (and only once that write has actually run)."""
    written = []
    monkeypatch.setattr(
        "vllm_hook_plugins.graph.artifact_writer.write_artifact",
        lambda *a, **k: written.append(a),
    )

    def _boom(_):
        raise RuntimeError("simulated pack failure")

    w = _new_writer(pack=True, pack_fn=_boom)
    w._inq.put(("hs", {"hs_cache": {}}, "/tmp/run2", "all_tokens", 0, False, False,
                "hidden_states.pt", ["req-c"]))
    w._inq.put(None)

    wp.WriterProcess._feed(w)

    assert written, "inline fallback write_artifact must have run"
    assert wp.WriterProcess.poll_pages_done(w) == ["req-c"]


def test_raw_pack_off_path_does_not_signal_pages_done():
    """Documented limitation (AR3): pack=0 hands the child the ORIGINAL tensors, which it only
    copies when it serializes them itself -- _feed has no early copy-out point there, so it must
    NOT claim the bytes were copied out yet."""
    w = _new_writer(pack=False, pack_fn=None)
    w._inq.put(("hs", {"hs_cache": {}}, "/tmp/run3", "all_tokens", 0, False, False,
                "hidden_states.pt", ["req-d"]))
    w._inq.put(None)

    wp.WriterProcess._feed(w)

    assert wp.WriterProcess.poll_pages_done(w) == []          # NOT signaled on the raw path
    tag = w._q.get_nowait()[0]
    assert tag == "R"
