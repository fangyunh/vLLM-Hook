import os
import threading
import time
from vllm_hook_plugins.hook_client import HookClient


def _make_client(hook_dir):
    c = HookClient.__new__(HookClient)      # bypass __init__ (no server needed)
    c._hook_dir = hook_dir                  # the real attribute set by __init__ (not `hook_dir`)
    return c


def test_wait_returns_true_when_file_appears_late(tmp_path):
    c = _make_client(str(tmp_path))
    run_id = "r1"
    run_dir = tmp_path / run_id / "tp_rank_0"
    run_dir.mkdir(parents=True)

    def _writer():
        time.sleep(0.3)
        (run_dir / "hidden_states.pt").write_bytes(b"x")   # atomic-complete file appears late

    threading.Thread(target=_writer).start()
    assert c._wait_artifact_dir(run_id, timeout_s=3.0, poll_s=0.02) is True


def test_wait_returns_false_on_timeout(tmp_path):
    c = _make_client(str(tmp_path))
    assert c._wait_artifact_dir("never", timeout_s=0.2, poll_s=0.02) is False
