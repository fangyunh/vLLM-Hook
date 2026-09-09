import pytest
from vllm_hook_plugins.graph.ring_sizing import (
    resolve_ring_bytes_fixed,
    resolve_ring_bytes_auto,
    DEFAULT_RING_GPU_BYTES,
)

GiB = 1 << 30


# --- Change 1: fixed ring (default 4 GiB) -------------------------------------
def test_default_ring_bytes_is_4gib():
    assert DEFAULT_RING_GPU_BYTES == 4 * GiB


def test_fixed_ring_returns_exact_bytes_when_it_fits():
    # 4 GiB ring at gpu_util=0.9 on 80 GiB: free margin 8 GiB -> fits, returned verbatim.
    assert resolve_ring_bytes_fixed(80 * GiB, 4 * GiB, 0.9) == 4 * GiB


def test_fixed_ring_fit_check_rejects_when_no_room():
    # 5 GiB ring at gpu_util=0.95 on 80 GiB: free margin 4 GiB < 5 GiB -> raise.
    with pytest.raises(ValueError):
        resolve_ring_bytes_fixed(80 * GiB, 5 * GiB, 0.95)


def test_auto_defaults_to_fixed_4gib(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_RING_GPU_BYTES", raising=False)
    assert resolve_ring_bytes_auto(80 * GiB, 0.9) == 4 * GiB


def test_auto_honors_explicit_ring_gpu_bytes(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_RING_GPU_BYTES", str(2 * GiB))
    assert resolve_ring_bytes_auto(80 * GiB, 0.9) == 2 * GiB


def test_auto_fit_check_still_enforced(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_RING_GPU_BYTES", str(5 * GiB))
    with pytest.raises(ValueError):
        resolve_ring_bytes_auto(80 * GiB, 0.95)   # free margin 4 GiB < 5 GiB ring
