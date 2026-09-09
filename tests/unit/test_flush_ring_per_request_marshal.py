"""No-GPU regression guard for the flush_ring_per_request RPC serialization contract.

Job 574989 PROVED the per-request ring mechanism ("2 delivered request(s), residency_after=0") but
crashed before the VERDICT with `AttributeError: 'list' object has no attribute 'detach'` in the
oracle's `_perreq_layer_store`. Root cause: the worker's `flush_ring_per_request` returned RAW torch
tensors over `collective_rpc`, and vLLM's `collective_rpc` does NOT round-trip torch tensors across
the worker->driver process boundary -- they arrive on the driver as plain Python LISTS. Every other
tensor-returning method in the worker (`get_captured_states`) serializes to zstd-pickle BYTES for
exactly this reason; `flush_ring_per_request` was the anomaly.

This test exercises the SERIALIZE (worker) -> DESERIALIZE (oracle) round-trip with NO GPU / no engine
boot: it feeds the REAL oracle deserialization path (`_flush_ring_per_request` fed a fake handle whose
`collective_rpc` returns the serialized blob) and the REAL crash site (`_perreq_layer_store`), then
asserts every tensor `torch.equal`-round-trips and residency matches.

RED/GREEN: against the raw-tensor version the payload arrives as lists and `_perreq_layer_store`
raises (see test_raw_tensor_payload_crashes_perreq_store, which reproduces the exact job-574989
crash); against the bytes fix it round-trips (test_bytes_roundtrip_delivers_equal_tensors PASSES).
"""
import importlib.util
import os
import pickle
import sys

import pytest
import torch
import zstandard as zstd

# Import the parity oracle by file path -- it does NOT import vllm at module top level (the vllm /
# HookLLM imports live inside capture()), so this stays a pure no-GPU unit test with no engine boot.
_ORACLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "cuda_graph", "tests", "ring", "hs_ring_perreq_parity.py",
)


def _load_oracle():
    spec = importlib.util.spec_from_file_location("hs_ring_perreq_parity_undertest", _ORACLE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ORACLE = _load_oracle()


def _worker_serialize(deliverables, residency_after):
    """Replicates ProbeHiddenStatesWorker.flush_ring_per_request's exact bytes path:
    `_ZSTD_COMPRESSOR.compress(pickle.dumps((deliverables, residency_after)))` with
    `zstd.ZstdCompressor(level=1)` (== the module-level `_ZSTD_COMPRESSOR`)."""
    raw = pickle.dumps((deliverables, residency_after))
    return zstd.ZstdCompressor(level=1).compress(raw)


class _FakeInner:
    """Stands in for llm.llm / llm.llm_engine: its collective_rpc returns the rank-0 blob list, the
    exact shape the real vLLM collective_rpc hands back (one entry per rank; TP=1 -> one)."""

    def __init__(self, blob):
        self._blob = blob

    def collective_rpc(self, name):
        assert name == "flush_ring_per_request"
        return [self._blob]


class _FakeLLM:
    def __init__(self, blob):
        self.llm = _FakeInner(blob)
        self.llm_engine = None


def _fake_deliverables():
    """{req_id: {layer_num(1-based): cpu_f32_tensor}} -- the flush_ring_per_request payload shape."""
    return {
        "r0": {1: torch.randn(3, 4), 2: torch.randn(2, 4)},
        "r1": {1: torch.randn(1, 4)},
    }


def test_bytes_roundtrip_delivers_equal_tensors():
    """GREEN after the fix: worker-serialize -> real oracle deserialize (_flush_ring_per_request) ->
    real crash site (_perreq_layer_store) yields tensors torch.equal to the originals, residency
    preserved."""
    deliverables = _fake_deliverables()
    residency_after = 0
    blob = _worker_serialize(deliverables, residency_after)
    assert isinstance(blob, (bytes, bytearray)), "worker must return bytes, not raw tensors"

    # Real oracle deserialization path (decompress + unpickle) end-to-end.
    got = _ORACLE._flush_ring_per_request(_FakeLLM(blob))
    assert got is not None
    got_deliverables, got_residency = got
    assert got_residency == residency_after

    for req_id, per_layer in deliverables.items():
        # Real crash site: _perreq_layer_store does t.detach().to(f32).cpu() on each tensor.
        store = _ORACLE._perreq_layer_store(got_deliverables, req_id)
        assert set(store.keys()) == set(per_layer.keys())
        for layer, t in per_layer.items():
            assert torch.equal(store[layer], t.to(torch.float32))


def test_residency_nonzero_roundtrips():
    """Residency is an int in the same tuple -> it must survive the bytes round-trip too (the gate
    reads it to decide PASS/FAIL)."""
    blob = _worker_serialize({}, 5)
    got_deliverables, got_residency = _ORACLE._flush_ring_per_request(_FakeLLM(blob))
    assert got_deliverables == {} and got_residency == 5


def test_second_call_empty_is_idempotent():
    """A second call after everything is popped+freed serializes ({}, 0) (Task 7's idempotent
    contract) -- it round-trips to an empty delivery with residency 0."""
    blob = _worker_serialize({}, 0)
    got_deliverables, got_residency = _ORACLE._flush_ring_per_request(_FakeLLM(blob))
    assert got_deliverables == {} and got_residency == 0


def test_raw_tensor_payload_crashes_perreq_store():
    """RED reproduction of job 574989: when flush_ring_per_request returns raw tensors,
    collective_rpc coerces each tensor to a plain Python list on the driver -> _perreq_layer_store's
    `t.detach()` raises AttributeError. This is the exact failure the bytes fix removes; keeping it
    here documents WHY the serialization is load-bearing."""
    deliverables = _fake_deliverables()
    # Model collective_rpc's tensor->list coercion.
    listified = {rid: {layer: t.tolist() for layer, t in per_layer.items()}
                 for rid, per_layer in deliverables.items()}
    with pytest.raises(AttributeError):
        _ORACLE._perreq_layer_store(listified, "r0")
