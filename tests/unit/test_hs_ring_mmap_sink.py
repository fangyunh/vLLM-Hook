"""No-GPU tests for the mmap-NVMe raw sink (VLLM_HOOK_RING_MMAP) in the HS capture-ring drain.

`MultiLayerRingDrain._append_layer_rows` writes each per-layer raw file either via a pre-sized
`mmap.mmap(..., access=mmap.ACCESS_WRITE)` (opt-in, `VLLM_HOOK_RING_MMAP=1`) or the original
`open(path, "ab")` append (the DEFAULT since 2026-08-14). Both paths must:
  1. reconstruct byte-identically through `load_multilayer_ring_artifact` (incl. across a genuine
     ring wrap);
  2. produce IDENTICAL raw-file bytes for the same input sequence (the correctness crux — the
     mmap writer must not reorder/pad/duplicate a single byte vs the plain-append reference);
  3. fall back safely (never lose/corrupt data) when a request would overflow a too-small
     `VLLM_HOOK_RING_MMAP_BYTES` pre-size, and still reconstruct correctly.

Run:  conda activate vllm_hook_env && python tests/unit/test_hs_ring_mmap_sink.py
      (or under pytest: pytest tests/unit/test_hs_ring_mmap_sink.py)
"""
import os
import sys
import tempfile

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.ring_drain_hs import (
    MultiLayerRingDrain, _resolve_mmap_capacity_bytes, _torch_dtype_name)
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact


def _mkdtmp(tag):
    return tempfile.mkdtemp(prefix=f"hsmmap_{tag}_")


def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)  # +1 sentinel row (row R), never drained


def _write_slots(hs_buf, ring, start_slot, rows):
    for j, p in enumerate(ring.physical_slots(start_slot, rows.shape[0])):
        hs_buf[p] = rows[j]


def _set_mmap_env(enabled, mmap_bytes=None):
    if enabled is None:
        os.environ.pop("VLLM_HOOK_RING_MMAP", None)
    else:
        os.environ["VLLM_HOOK_RING_MMAP"] = "1" if enabled else "0"
    if mmap_bytes is None:
        os.environ.pop("VLLM_HOOK_RING_MMAP_BYTES", None)
    else:
        os.environ["VLLM_HOOK_RING_MMAP_BYTES"] = str(mmap_bytes)


def _clear_mmap_env():
    os.environ.pop("VLLM_HOOK_RING_MMAP", None)
    os.environ.pop("VLLM_HOOK_RING_MMAP_BYTES", None)


# --- shared step sequence (incl. a genuine physical wrap) driven against a drain instance ---

def _drive_sequence(drain, ring, hs_bufs, layer_ids, dtype):
    """3 steps against an R=4 ring: step 1 fills 3/4 rows (no wrap), step 2 reserves 4 more rows ->
    forces a genuine physical wrap (drained_segments() == 2 segments), step 3 is a 1-row last_token
    write on a single-layer subset. Returns {(req_id, layer): tensor} of what was written, for the
    reconstruction assert."""
    hidden = hs_bufs[layer_ids[0]].shape[1]
    expected = {}

    def _data(n, base):
        return (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)

    # Step 1: request A, all_tokens, ALL layers, 3 rows.
    sA = ring.reserve(3)
    dataA = _data(3, 100)
    for L in layer_ids:
        _write_slots(hs_bufs[L], ring, sA, dataA)
        expected[("A", L)] = dataA
    drain.record_entries([LayerEntry("A", L, sA, 3, "all_tokens") for L in layer_ids])
    assert drain.drain_once() == 3

    # Step 2: request B, 4 rows -> logical [3,7) on an n_slots=4 ring -> physical [3,4) then wraps
    # [0,3): a GENUINE physical wrap while the drain cursor is still at 3.
    sB = ring.reserve(4)
    assert sB == 3
    dataB = _data(4, 1000)
    for L in layer_ids:
        _write_slots(hs_bufs[L], ring, sB, dataB)
        expected[("B", L)] = dataB
    assert len(ring.drained_segments()) == 2, "expected a genuine physical wrap"
    drain.record_entries([LayerEntry("B", L, sB, 4, "all_tokens") for L in layer_ids])
    assert drain.drain_once() == 4

    # Step 3: request C, last_token, only the FIRST layer.
    sC = ring.reserve(1)
    dataC = _data(1, 9999)
    L0 = layer_ids[0]
    _write_slots(hs_bufs[L0], ring, sC, dataC)
    expected[("C", L0)] = dataC
    drain.record_entries([LayerEntry("C", L0, sC, 1, "last_token")])
    assert drain.drain_once() == 1

    return expected


def _build(tmp, dtype=torch.float32, hidden=4, R=4, layer_ids=(1, 2)):
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = MultiLayerRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header)
    return ring, hs_bufs, drain


def _assert_reconstruction(tmp, expected):
    out = load_multilayer_ring_artifact(tmp)
    for (req_id, layer), tensor in expected.items():
        assert torch.equal(out[req_id][layer], tensor), f"{req_id} layer {layer} mismatch"
    assert 2 not in out.get("C", {}), "C (single-layer request) leaked into layer 2"


# --- Test 1: mmap path round-trips through a genuine wrap (float32 + bfloat16) ---

def _run_mmap_roundtrip(tmp, dtype):
    _set_mmap_env(True)
    try:
        ring, hs_bufs, drain = _build(tmp, dtype)
        assert drain._mmap_enabled, "VLLM_HOOK_RING_MMAP=1 should arm the mmap writers"
        expected = _drive_sequence(drain, ring, hs_bufs, [1, 2], dtype)
        drain.close()
        _assert_reconstruction(tmp, expected)
    finally:
        _clear_mmap_env()


def test_mmap_roundtrip_float32(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("f32")
    _run_mmap_roundtrip(tmp, torch.float32)


def test_mmap_roundtrip_bfloat16(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("bf16")
    _run_mmap_roundtrip(tmp, torch.bfloat16)


def test_mmap_default_off(tmp_path=None):
    """VLLM_HOOK_RING_MMAP unset -> the plain GIL-releasing write() path, NOT mmap.

    Flipped 2026-08-14. mmap is opt-in because its memcpy holds the GIL on the drain consumer
    thread; removing it recovered ~98% of the phase=both serve gap (knee 4->8, saturation 16->32,
    K=3 across three nodes). Guards the default itself, so re-flipping it silently fails here.
    """
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("default")
    _clear_mmap_env()
    try:
        ring, hs_bufs, drain = _build(tmp, torch.float32)
        assert not drain._mmap_enabled, "default (env unset) must be the plain append path"
        assert drain._mmap_writers == {}, "no mmap writer may be opened on the default path"
    finally:
        _clear_mmap_env()


# --- Test 2: byte-identity vs the plain open(ab)+write control, same input sequence ---

def test_mmap_byte_identical_to_plain_append(tmp_path=None):
    base = str(tmp_path) if tmp_path is not None else _mkdtmp("identity")
    dir_mmap = os.path.join(base, "mmap")
    dir_plain = os.path.join(base, "plain")
    os.makedirs(dir_mmap, exist_ok=True)
    os.makedirs(dir_plain, exist_ok=True)

    torch.manual_seed(0)
    dtype = torch.float32
    layer_ids = [1, 2]

    _set_mmap_env(True)
    try:
        ring_m, bufs_m, drain_m = _build(dir_mmap, dtype, layer_ids=layer_ids)
        expected_m = _drive_sequence(drain_m, ring_m, bufs_m, layer_ids, dtype)
        drain_m.close()
    finally:
        _clear_mmap_env()

    _set_mmap_env(False)
    try:
        ring_p, bufs_p, drain_p = _build(dir_plain, dtype, layer_ids=layer_ids)
        expected_p = _drive_sequence(drain_p, ring_p, bufs_p, layer_ids, dtype)
        drain_p.close()
    finally:
        _clear_mmap_env()

    assert expected_m.keys() == expected_p.keys()
    for k in expected_m:
        assert torch.equal(expected_m[k], expected_p[k])  # sanity: same input sequence

    for L in layer_ids:
        raw_m = os.path.join(dir_mmap, f"hs_layer_{L}.raw")
        raw_p = os.path.join(dir_plain, f"hs_layer_{L}.raw")
        with open(raw_m, "rb") as f:
            bytes_m = f.read()
        with open(raw_p, "rb") as f:
            bytes_p = f.read()
        assert len(bytes_m) == len(bytes_p), (
            f"layer {L}: mmap file {len(bytes_m)}B != plain file {len(bytes_p)}B "
            "(the ftruncate-down must drop the zero-padded pre-sized tail)")
        assert bytes_m == bytes_p, f"layer {L}: mmap raw bytes differ from the plain-append reference"

    # Both must also reconstruct identically via the reader.
    _assert_reconstruction(dir_mmap, expected_m)
    _assert_reconstruction(dir_plain, expected_p)


# --- Test 3: a too-small VLLM_HOOK_RING_MMAP_BYTES forces the overflow fallback mid-run ---

def test_mmap_overflow_fallback_reconstructs(tmp_path=None):
    tmp = str(tmp_path) if tmp_path is not None else _mkdtmp("overflow")
    hidden, dtype = 4, torch.float32
    row_bytes = hidden * torch.empty(0, dtype=dtype).element_size()  # 16 bytes/row
    # Big enough for step 1 (3 rows = 48B) but NOT step 2 (4 more rows = 64B) -> forces the
    # overflow fallback mid-sequence (partial mmap write + plain-append tail), on EVERY layer.
    tiny_cap = 3 * row_bytes
    _set_mmap_env(True, mmap_bytes=tiny_cap)
    try:
        layer_ids = [1, 2]
        ring, hs_bufs, drain = _build(tmp, dtype, hidden=hidden, layer_ids=layer_ids)
        for L in layer_ids:
            w = drain._mmap_writers[L]
            assert w.capacity == tiny_cap
        expected = _drive_sequence(drain, ring, hs_bufs, layer_ids, dtype)
        # Every layer's writer must have tripped the overflow fallback by now.
        for L in layer_ids:
            assert drain._mmap_writers[L]._overflowed, f"layer {L} should have overflowed"
        drain.close()
        _assert_reconstruction(tmp, expected)
        # File length must equal exactly the real written bytes (8 rows total across the run:
        # 3 + 4 + 1), never the (already-shrunk) mmap capacity and never short.
        total_rows = 3 + 4 + 1
        for L in layer_ids:
            size = os.path.getsize(os.path.join(tmp, f"hs_layer_{L}.raw"))
            assert size == total_rows * row_bytes, f"layer {L}: unexpected raw file size {size}"
    finally:
        _clear_mmap_env()


def test_resolve_mmap_capacity_default_and_override():
    hidden, dtype, R = 4, torch.float32, 64
    ring = GpuCaptureRing(row_bytes=hidden * 4, n_slots=R, device="cpu",
                          dtype=dtype, row_shape=(hidden,))
    _clear_mmap_env()
    default_cap = _resolve_mmap_capacity_bytes(ring)
    assert default_cap == 2 * 1024 ** 3, "small ring should floor at the 2 GiB default"
    os.environ["VLLM_HOOK_RING_MMAP_BYTES"] = "12345"
    try:
        assert _resolve_mmap_capacity_bytes(ring) == 12345
    finally:
        _clear_mmap_env()


# --- standalone-runner scaffolding (mkdtemp when not under pytest) ---
def main():
    tests = [
        test_mmap_roundtrip_float32,
        test_mmap_roundtrip_bfloat16,
        test_mmap_default_off,
        test_mmap_byte_identical_to_plain_append,
        test_mmap_overflow_fallback_reconstructs,
        test_resolve_mmap_capacity_default_and_override,
    ]
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
