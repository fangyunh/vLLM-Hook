"""No-GPU unit gates for the decode-cache flag reader + the _DecodeEntry container."""
import os
import numpy as np
import pytest
from vllm_hook_plugins.graph.install_hs import _route_decode_cache_enabled, _DecodeEntry


def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_ROUTE_DECODE_CACHE", raising=False)
    assert _route_decode_cache_enabled() is True


def test_flag_off_kill_switch(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_ROUTE_DECODE_CACHE", "0")
    assert _route_decode_cache_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_ROUTE_DECODE_CACHE", "1")
    assert _route_decode_cache_enabled() is True


def test_decode_entry_holds_fields():
    e = _DecodeEntry(layer_rows=np.array([0, 1, 2], dtype=np.int64),
                     mode="all_tokens", layers=[1, 2, 3])
    assert e.mode == "all_tokens"
    assert e.layers == [1, 2, 3]
    assert list(e.layer_rows) == [0, 1, 2]
