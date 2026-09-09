"""cpu_list_batched: byte-identity, owned storage, and every fallback — GPU-free.

The cat/split/clone logic is device-agnostic; pin_memory/non_blocking are cuda plumbing the
LSF parity oracle covers. These tests pin the value contract and the fallback matrix.
"""
import torch

from vllm_hook_plugins.workers._common import cpu_list_batched


def test_value_identity_matches_per_tensor_cpu():
    ts = [torch.randn(3, 5) for _ in range(4)]
    out = cpu_list_batched(ts)
    assert len(out) == len(ts)
    for a, b in zip(out, ts):
        assert torch.equal(a, b.cpu())


def test_single_row_tensors_the_drain_shape():
    ts = [torch.randn(1, 8) for _ in range(6)]  # one row per decode step
    out = cpu_list_batched(ts)
    for a, b in zip(out, ts):
        assert torch.equal(a, b)
    assert all(o.shape == (1, 8) for o in out)


def test_varying_first_dim_is_cat_able():
    ts = [torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)]  # prefill chunks
    out = cpu_list_batched(ts)
    for a, b in zip(out, ts):
        assert torch.equal(a, b)


def test_owned_storage_not_a_shared_view():
    ts = [torch.randn(2, 4) for _ in range(3)]
    out = cpu_list_batched(ts)
    ptrs = {o.untyped_storage().data_ptr() for o in out}
    assert len(ptrs) == len(out), "each result must own independent storage"
    for o in out:
        assert o.untyped_storage().nbytes() == o.numel() * o.element_size(), \
            "each result's storage is its own bytes, not a slice of shared staging"


def test_fallback_mixed_trailing_shape():
    ts = [torch.randn(2, 4), torch.randn(2, 5)]  # not cat-able
    out = cpu_list_batched(ts)
    for a, b in zip(out, ts):
        assert torch.equal(a, b.cpu())


def test_fallback_mixed_dtype():
    ts = [torch.randn(2, 4), torch.randn(2, 4).to(torch.float16)]
    out = cpu_list_batched(ts)
    for a, b in zip(out, ts):
        assert torch.equal(a, b.cpu())


def test_empty_list():
    assert cpu_list_batched([]) == []


def test_non_tensor_element_falls_back():
    ts = [torch.randn(2, 4), 7]
    out = cpu_list_batched(ts)
    assert torch.equal(out[0], ts[0].cpu())
    assert out[1] == 7


def test_cpu_list_seam_dispatches_to_batched_only_when_gate_on_hs(monkeypatch):
    """_cpu_list (HS worker) must CALL cpu_list_batched when the gate is on, and NOT when off."""
    import vllm_hook_plugins.workers._common as common
    import vllm_hook_plugins.workers.probe_hidden_states_worker as hs

    calls = {"n": 0}
    real = common.cpu_list_batched

    def spy(tensors):
        calls["n"] += 1
        return real(tensors)

    monkeypatch.setattr(common, "cpu_list_batched", spy)
    ts = [torch.randn(2, 4) for _ in range(3)]

    # gate OFF -> cpu_list_batched is NOT called; result is the verbatim comprehension
    monkeypatch.setattr(hs, "_BATCHED_FLUSH", False)   # raising=True: the attr MUST already exist
    out_off = hs._cpu_list(ts, None)
    assert calls["n"] == 0
    for a, b in zip(out_off, ts):
        assert torch.equal(a, b.cpu())

    # gate ON -> cpu_list_batched IS called; result still byte-identical (fallback on CPU)
    monkeypatch.setattr(hs, "_BATCHED_FLUSH", True)
    out_on = hs._cpu_list(ts, None)
    assert calls["n"] == 1
    for a, b in zip(out_on, ts):
        assert torch.equal(a, b.cpu())


def test_cpu_list_seam_dispatches_to_batched_only_when_gate_on_qk(monkeypatch):
    """_cpu_list (QK worker) must CALL cpu_list_batched when the gate is on, and NOT when off."""
    import vllm_hook_plugins.workers._common as common
    import vllm_hook_plugins.workers.probe_hookqk_worker as qk

    calls = {"n": 0}
    real = common.cpu_list_batched

    def spy(tensors):
        calls["n"] += 1
        return real(tensors)

    monkeypatch.setattr(common, "cpu_list_batched", spy)
    ts = [torch.randn(2, 4) for _ in range(3)]

    # gate OFF -> cpu_list_batched is NOT called; result is the verbatim comprehension
    monkeypatch.setattr(qk, "_BATCHED_FLUSH", False)   # raising=True: the attr MUST already exist
    out_off = qk._cpu_list(ts, None)
    assert calls["n"] == 0
    for a, b in zip(out_off, ts):
        assert torch.equal(a, b.cpu())

    # gate ON -> cpu_list_batched IS called; result still byte-identical (fallback on CPU)
    monkeypatch.setattr(qk, "_BATCHED_FLUSH", True)
    out_on = qk._cpu_list(ts, None)
    assert calls["n"] == 1
    for a, b in zip(out_on, ts):
        assert torch.equal(a, b.cpu())
