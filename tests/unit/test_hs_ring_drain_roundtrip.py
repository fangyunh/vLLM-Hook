"""No-GPU round-trip for the multi-layer HS capture-ring drain (plan Task 9).

MultiLayerRingDrain writes one raw file PER LAYER + a SHARED sidecar; load_multilayer_ring_artifact
reconstructs {req_id: {layer: tensor}} byte-identically. All per-layer rings share one logical
cursor, so a LayerEntry.logical_start indexes every layer's file. Covers: two-request/mixed-layer,
a genuine physical wrap, per-request layer subsets, and the bfloat16 write/read path (numpy has no
bf16, so the writer reinterprets as uint16 and the reader views back).

Run:  conda activate vllm_hook_env && python tests/unit/test_hs_ring_drain_roundtrip.py
"""
import os
import sys

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.ring_drain_hs import MultiLayerRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact


def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)  # +1 sentinel row (row R), never drained


def _write_slots(hs_buf, ring, start_slot, rows):
    for j, p in enumerate(ring.physical_slots(start_slot, rows.shape[0])):
        hs_buf[p] = rows[j]


def _run(tmp, dtype=torch.float32):
    hidden, R = 4, 64
    num_layers = 3
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in (1, 2, 3)}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = MultiLayerRingDrain(ring, [(L, hs_bufs[L]) for L in (1, 2, 3)], tmp, header)

    def _data(n, base):
        return (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)

    # --- Step 1: A all_tokens on ALL layers (3 rows) + B last_token on layers {2} (1 row) ---
    startA = ring.reserve(3)
    dataA = {L: _data(3, 100 * L) for L in (1, 2, 3)}
    for L in (1, 2, 3):
        _write_slots(hs_bufs[L], ring, startA, dataA[L])
    startB = ring.reserve(1)
    dataB2 = _data(1, 999)
    _write_slots(hs_bufs[2], ring, startB, dataB2)   # only layer 2 gets B's row
    drain.record_entries(
        [LayerEntry("A", L, startA, 3, "all_tokens") for L in (1, 2, 3)]
        + [LayerEntry("B", 2, startB, 1, "last_token")])
    assert drain.drain_once() == 4

    drain.close()
    out = load_multilayer_ring_artifact(tmp)
    for L in (1, 2, 3):
        assert torch.equal(out["A"][L], dataA[L]), f"A layer {L} mismatch"
    assert torch.equal(out["B"][2], dataB2), "B layer 2 mismatch"
    assert 1 not in out["B"] and 3 not in out["B"], "B leaked into non-requested layers"


def test_roundtrip_float32(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("f32")
    _run(tmp, torch.float32)


def test_roundtrip_bfloat16(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("bf16")
    _run(tmp, torch.bfloat16)


def test_wrapped_drain_reconstructs(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("wrap")
    hidden, R, dtype = 4, 4, torch.float32   # tiny ring -> force a physical wrap
    ring = GpuCaptureRing(row_bytes=hidden * 4, n_slots=R, device="cpu",
                          dtype=dtype, row_shape=(hidden,))
    buf = _hs_buf(R, hidden, dtype)
    header = {"dtype": "float32", "row_shape": [hidden], "hidden": hidden}
    drain = MultiLayerRingDrain(ring, [(1, buf)], tmp, header)

    dataA = torch.arange(3 * hidden, dtype=torch.float32).reshape(3, hidden)
    sA = ring.reserve(3)
    _write_slots(buf, ring, sA, dataA)
    drain.record_entries([LayerEntry("A", 1, sA, 3, "all_tokens")])
    assert drain.drain_once() == 3

    dataB = (torch.arange(4 * hidden, dtype=torch.float32).reshape(4, hidden) + 100)
    sB = ring.reserve(4)                       # logical [3,7) -> physical [3,4) then wraps [0,3)
    assert sB == 3
    _write_slots(buf, ring, sB, dataB)
    assert len(ring.drained_segments()) == 2, "expected a genuine physical wrap"
    drain.record_entries([LayerEntry("B", 1, sB, 4, "all_tokens")])
    assert drain.drain_once() == 4

    drain.close()
    out = load_multilayer_ring_artifact(tmp)
    assert torch.equal(out["A"][1], dataA)
    assert torch.equal(out["B"][1], dataB), "wrapped block not reconstructed in logical order"


# --- standalone-runner scaffolding (mkdtemp when not under pytest) ---
def _mkdtmp(tag):
    import tempfile
    return tempfile.mkdtemp(prefix=f"hsring_{tag}_")


def main():
    tests = [test_roundtrip_float32, test_roundtrip_bfloat16, test_wrapped_drain_reconstructs]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'} ({len(tests) - failures}/{len(tests)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
