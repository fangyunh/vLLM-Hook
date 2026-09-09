"""No-GPU round-trip for the QK capture-ring drain + reader (plan Task 15).

MultiLayerQKRingDrain writes TWO raw files per layer (q + k) + a shared QK sidecar;
load_multilayer_qk_ring_artifact reconstructs, per (req, layer), the flat ``q`` and the growing-prefix
``k_all = [k_full[:L] for L in k_prefix_ends]`` byte-identically. Both per-layer files share ONE ring
cursor, so a ``k_start`` / ``q_start`` indexes each file. Covers: an all_tokens growing prefix across
≥3 steps (float32 + bfloat16), the last_token flat q (one row per emit + full k history), a genuine
physical wrap, and the deferred prefix-cache guard (first-step num_computed > 0 -> reader raises).

Run:  conda activate vllm_hook_env && python tests/unit/test_qk_ring_drain_roundtrip.py
"""
import os
import sys
import tempfile

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.ring_drain_hs import _torch_dtype_name
from vllm_hook_plugins.graph.ring_drain_qk import MultiLayerQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import QKStepEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_qk_ring_artifact


def _scatter(buf, ring, start_slot, rows):
    for j, p in enumerate(ring.physical_slots(start_slot, rows.shape[0])):
        buf[p] = rows[j]


def _build(tmp, R, q_dim, k_dim, layers, dtype):
    ring = GpuCaptureRing(row_bytes=k_dim * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(k_dim,))
    q_bufs = {L: torch.zeros(R + 1, q_dim, dtype=dtype) for L in layers}
    k_bufs = {L: torch.zeros(R + 1, k_dim, dtype=dtype) for L in layers}
    header = {"dtype": _torch_dtype_name(dtype),
              "q_row_shape": [q_dim], "k_row_shape": [k_dim],
              "q_dim": q_dim, "k_dim": k_dim, "hookq_mode": "all_tokens"}
    drain = MultiLayerQKRingDrain(
        ring, [(L, q_bufs[L], k_bufs[L]) for L in layers], tmp, header)
    return ring, q_bufs, k_bufs, drain


def _data(n, width, base, dtype):
    return (torch.arange(n * width, dtype=torch.float32).reshape(n, width) + base).to(dtype)


# --------------------------- (a) all_tokens growing prefix ----------------------------
def _run_growing_prefix(tmp, dtype):
    R, q_dim, k_dim = 64, 8, 4
    layers = (0, 1)
    ring, q_bufs, k_bufs, drain = _build(tmp, R, q_dim, k_dim, layers, dtype)

    # Full per-layer histories: 5 keys / 5 queries (all_tokens emits every step).
    q_master = {L: _data(5, q_dim, 900 + 100 * L, dtype) for L in layers}
    k_master = {L: _data(5, k_dim, 100 * L, dtype) for L in layers}

    off = 0
    for nrows, abs_end in ((3, 3), (1, 4), (1, 5)):   # prefill(3) + 2 decode(1) steps
        s = ring.reserve(nrows)
        entries = []
        for L in layers:
            _scatter(q_bufs[L], ring, s, q_master[L][off:off + nrows])
            _scatter(k_bufs[L], ring, s, k_master[L][off:off + nrows])
            entries.append(QKStepEntry(req_id="A", layer=L, k_start=s, k_rows=nrows,
                                       q_start=s, q_rows=nrows, prefix_end=abs_end, num_computed=0))
        drain.record_entries(entries)
        assert drain.drain_once() == nrows
        off += nrows

    drain.close()
    out = load_multilayer_qk_ring_artifact(tmp)
    for L in layers:
        rec = out["A"][L]
        assert torch.equal(rec["q"], q_master[L]), f"layer {L} q mismatch"
        assert torch.equal(rec["k_full"], k_master[L]), f"layer {L} k_full mismatch"
        assert rec["k_prefix_ends"] == [3, 4, 5], f"layer {L} prefix_ends {rec['k_prefix_ends']}"
        expect_kall = [k_master[L][:3], k_master[L][:4], k_master[L][:5]]
        assert len(rec["k_all"]) == 3
        for got, exp in zip(rec["k_all"], expect_kall):
            assert torch.equal(got, exp), f"layer {L} k_all growing-prefix mismatch"


def test_growing_prefix_float32():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["VLLM_HOOK_RING_MMAP_BYTES"] = "1048576"   # small mmap so no 2 GiB ftruncate
        try:
            _run_growing_prefix(tmp, torch.float32)
        finally:
            os.environ.pop("VLLM_HOOK_RING_MMAP_BYTES", None)


def test_growing_prefix_bfloat16():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["VLLM_HOOK_RING_MMAP_BYTES"] = "1048576"
        try:
            _run_growing_prefix(tmp, torch.bfloat16)
        finally:
            os.environ.pop("VLLM_HOOK_RING_MMAP_BYTES", None)


# --------------------------------- (b) last_token ---------------------------------
def test_last_token_flat_q_and_full_k():
    with tempfile.TemporaryDirectory() as tmp:
        R, q_dim, k_dim = 64, 8, 4
        ring, q_bufs, k_bufs, drain = _build(tmp, R, q_dim, k_dim, (0,), torch.float32)
        # prefill(final chunk, P=3, emit only the LAST q) + 2 decode(1) steps.
        # K keeps the whole span every step -> k_full is 5 rows; q is 3 rows (last-prefill + 2 dec).
        k_master = _data(5, k_dim, 0, torch.float32)
        # q written per step (prefill writes 3 rows, only row 2 is referenced).
        q_prefill = _data(3, q_dim, 900, torch.float32)
        q_dec1 = _data(1, q_dim, 800, torch.float32)
        q_dec2 = _data(1, q_dim, 700, torch.float32)

        # step 1: prefill, reserve 3, q_start = last slot (s+2), q_rows 1
        s = ring.reserve(3)
        _scatter(q_bufs[0], ring, s, q_prefill)
        _scatter(k_bufs[0], ring, s, k_master[0:3])
        drain.record_entries([QKStepEntry("A", 0, s, 3, s + 2, 1, 3, 0)])
        assert drain.drain_once() == 3
        # step 2: decode
        s = ring.reserve(1)
        _scatter(q_bufs[0], ring, s, q_dec1)
        _scatter(k_bufs[0], ring, s, k_master[3:4])
        drain.record_entries([QKStepEntry("A", 0, s, 1, s, 1, 4, 0)])
        assert drain.drain_once() == 1
        # step 3: decode
        s = ring.reserve(1)
        _scatter(q_bufs[0], ring, s, q_dec2)
        _scatter(k_bufs[0], ring, s, k_master[4:5])
        drain.record_entries([QKStepEntry("A", 0, s, 1, s, 1, 5, 0)])
        assert drain.drain_once() == 1

        drain.close()
        rec = load_multilayer_qk_ring_artifact(tmp)["A"][0]
        expect_q = torch.cat([q_prefill[2:3], q_dec1, q_dec2], dim=0)     # last-prefill + 2 decode
        assert torch.equal(rec["q"], expect_q), "last_token q not the emitted rows"
        assert torch.equal(rec["k_full"], k_master), "last_token k_full not the full history"
        assert rec["k_prefix_ends"] == [3, 4, 5]
        for got, L in zip(rec["k_all"], (3, 4, 5)):
            assert torch.equal(got, k_master[:L]), "last_token k_all growing-prefix mismatch"


# ----------------------------------- (c) wrap -------------------------------------
def test_physical_wrap_plain_append():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["VLLM_HOOK_RING_MMAP"] = "0"   # exercise the plain open(ab)+write sink too
        try:
            R, q_dim, k_dim = 4, 2, 2      # tiny ring -> force a physical wrap
            ring, q_bufs, k_bufs, drain = _build(tmp, R, q_dim, k_dim, (0,), torch.float32)
            qA, kA = _data(3, q_dim, 900, torch.float32), _data(3, k_dim, 100, torch.float32)
            s = ring.reserve(3)
            _scatter(q_bufs[0], ring, s, qA)
            _scatter(k_bufs[0], ring, s, kA)
            drain.record_entries([QKStepEntry("A", 0, s, 3, s, 3, 3, 0)])
            assert drain.drain_once() == 3
            # step 2 wraps: logical [3,7) -> physical [3,4) then [0,3).
            qB, kB = _data(4, q_dim, 500, torch.float32), _data(4, k_dim, 300, torch.float32)
            s = ring.reserve(4)
            assert s == 3
            _scatter(q_bufs[0], ring, s, qB)
            _scatter(k_bufs[0], ring, s, kB)
            assert len(ring.drained_segments()) == 2, "expected a genuine physical wrap"
            drain.record_entries([QKStepEntry("B", 0, s, 4, s, 4, 4, 0)])
            assert drain.drain_once() == 4

            drain.close()
            out = load_multilayer_qk_ring_artifact(tmp)
            assert torch.equal(out["A"][0]["q"], qA) and torch.equal(out["A"][0]["k_full"], kA)
            assert torch.equal(out["B"][0]["q"], qB), "wrapped q not reconstructed in logical order"
            assert torch.equal(out["B"][0]["k_full"], kB), "wrapped k not in logical order"
        finally:
            os.environ.pop("VLLM_HOOK_RING_MMAP", None)


# ----------------------------- (d) prefix-cache guard -----------------------------
def test_first_step_num_computed_raises():
    with tempfile.TemporaryDirectory() as tmp:
        R, q_dim, k_dim = 64, 8, 4
        ring, q_bufs, k_bufs, drain = _build(tmp, R, q_dim, k_dim, (0,), torch.float32)
        s = ring.reserve(2)
        _scatter(q_bufs[0], ring, s, _data(2, q_dim, 0, torch.float32))
        _scatter(k_bufs[0], ring, s, _data(2, k_dim, 0, torch.float32))
        # First (and only) step for this request has num_computed=5 (a cached prefix) -> the reader
        # must FAIL LOUD rather than return a k_full short by the cached prefix.
        drain.record_entries([QKStepEntry("A", 0, s, 2, s, 2, 7, 5)])
        drain.drain_once()
        drain.close()
        raised = False
        try:
            load_multilayer_qk_ring_artifact(tmp)
        except NotImplementedError:
            raised = True
        assert raised, "reader must raise on a first-step num_computed > 0 (deferred prefix)"


def main():
    tests = [
        test_growing_prefix_float32,
        test_growing_prefix_bfloat16,
        test_last_token_flat_q_and_full_k,
        test_physical_wrap_plain_append,
        test_first_step_num_computed_raises,
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
