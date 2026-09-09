import os
import importlib
import pytest

hp = importlib.import_module("vllm_hook_plugins._hook_plugin")


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_SINK", raising=False)


def test_drop_is_global_override(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_SINK", "drop")
    # drop wins even over an explicit save_to_disk=True
    assert hp._resolve_sink({"save_to_disk": True}) == "drop"
    assert hp._resolve_sink({}) == "drop"


def test_explicit_save_to_disk_wins_over_env_default(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_SINK", "rpc")
    assert hp._resolve_sink({"save_to_disk": True}) == "disk"   # explicit requirement honored
    monkeypatch.setenv("VLLM_HOOK_SINK", "disk")
    assert hp._resolve_sink({"save_to_disk": False}) == "rpc"


def test_env_default_when_request_is_silent(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_SINK", "disk")
    assert hp._resolve_sink({}) == "disk"
    monkeypatch.setenv("VLLM_HOOK_SINK", "rpc")
    assert hp._resolve_sink({}) == "rpc"


def test_unset_falls_back_to_save_to_disk(monkeypatch):
    # VLLM_HOOK_SINK unset: today's behaviour — truthy save_to_disk -> disk else rpc
    assert hp._resolve_sink({"save_to_disk": True}) == "disk"
    assert hp._resolve_sink({"save_to_disk": False}) == "rpc"
    assert hp._resolve_sink({}) == "rpc"
