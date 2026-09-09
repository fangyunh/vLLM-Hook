"""No-GPU tests for multi-writer plumbing (Phase 2b): VLLM_HOOK_WRITER_PROCESS_N + N-child
alive/close logic. Does NOT spawn real children (that needs a GPU worker) -- it checks the
env plumbing and the any-child-alive semantics with fakes.
"""
import types
from unittest import mock

from vllm_hook_plugins.graph import writer_process as wp


def test_from_env_passes_writer_process_n(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_WRITER_PROCESS", "1")
    monkeypatch.setenv("VLLM_HOOK_WRITER_PROCESS_N", "4")
    seen = {}

    def fake_init(self, maxsize=4, n_writers=1):
        seen["maxsize"], seen["n_writers"] = maxsize, n_writers

    monkeypatch.setattr(wp.WriterProcess, "__init__", fake_init)
    inst = wp.WriterProcess.from_env()
    assert inst is not None and seen["n_writers"] == 4


def test_from_env_off_returns_none(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_WRITER_PROCESS", "0")
    assert wp.WriterProcess.from_env() is None


def test_from_env_default_n_is_one(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_WRITER_PROCESS", "1")
    monkeypatch.delenv("VLLM_HOOK_WRITER_PROCESS_N", raising=False)
    seen = {}
    monkeypatch.setattr(wp.WriterProcess, "__init__",
                        lambda self, maxsize=4, n_writers=1: seen.update(n_writers=n_writers))
    wp.WriterProcess.from_env()
    assert seen["n_writers"] == 1


def _fake_wp(alive_flags):
    w = wp.WriterProcess.__new__(wp.WriterProcess)
    w._procs = [mock.Mock(is_alive=mock.Mock(return_value=a)) for a in alive_flags]
    return w


def test_alive_true_if_any_child_alive():
    assert _fake_wp([False, True, False]).alive() is True


def test_alive_false_if_all_dead():
    assert _fake_wp([False, False]).alive() is False


def test_alive_false_before_spawn():
    w = wp.WriterProcess.__new__(wp.WriterProcess)  # no _procs attr yet
    assert w.alive() is False
