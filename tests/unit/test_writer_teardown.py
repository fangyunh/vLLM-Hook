import atexit
from unittest import mock
from vllm_hook_plugins.graph import writer_process as wp


def test_init_registers_atexit_close(monkeypatch):
    started = mock.Mock()
    started.close = mock.Mock()
    monkeypatch.setattr(wp.WriterProcess, "from_env", classmethod(lambda cls: started))
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: registered.append(fn))

    class W:  # a bare stand-in worker
        pass
    w = W()
    wp.init_writer_process(w)
    assert started.close in registered  # teardown drain wired
