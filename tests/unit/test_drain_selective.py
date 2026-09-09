"""No-GPU tests for Tasks 14 + 15 (capture-overhead-reduction plan).

TASK 14 (first half of this file) split the two roles of `LayerEntry.logical_start` into
`logical_start` (the ring-wide reservation position) and `file_row` (the row offset into THAT
LAYER's own raw file). It changed NO copy behaviour -- every installed layer still received every
row of every step -- so those tests prove the SPLIT is real (the two fields are independently
carried end-to-end through the sidecar) while the VALUE stays the same as before (`file_row ==
logical_start` on every entry an all-layers drain produces).

TASK 15 (second half, from "SELECTIVE DRAIN" below) is what makes them diverge: behind
`VLLM_HOOK_DRAIN_SELECTIVE` (DEFAULT ON since Task 19; `=0` is the kill switch and the full-drain
control every unarmed arm below pins explicitly) the off-loop consumer copies only the `(layer,
row-range)` tiles some request actually asked for, so a layer nobody wanted this step is not copied
at all and a layer only ONE request wanted receives only THAT request's rows. Reconstruction stays
byte-identical (`torch.equal`); only the raw FILES get smaller.

Covers (Task 14):
  * `LayerEntry.file_row` default (unstamped -> `logical_start`) and explicit override;
  * cursor arithmetic: the per-layer running file cursor (`MultiLayerRingDrain._file_rows`) tracks
    "rows appended so far to this layer" across multiple steps and layers;
  * sidecar round-trip carries `file_row` as a field DISTINCT from `logical_start` (not conflated);
  * compat, both directions: a NEW reader on an OLD (pre-"fr"-key) sidecar falls back to
    `logical_start`; an OLD reader (5-key access only) on a NEW sidecar ignores the unknown "fr" key;
  * the task's own regression invariant -- on an all-layers drain, `file_row == logical_start` for
    EVERY entry -- as a property over several step/batch/layer-count shapes, through BOTH the
    synchronous (`MultiLayerRingDrain`) and off-loop (`OffLoopRingDrain`) drains, including a genuine
    physical ring wrap;
  * end-to-end reconstruction (`load_multilayer_ring_artifact`) is unaffected -- still byte-identical
    now that it keys on `file_row` instead of `logical_start`.

Covers (Task 15):
  * the copy list as a PURE function (`build_copy_plans` / `LayerCopyPlan`) over uniform / subset /
    heterogeneous / wrapping / merging cases, and its equivalence whether fed per-request records or
    already-expanded `LayerEntry`s;
  * THE DEGENERATE-CASE CONTRACT: every request wanting every installed layer -> each layer's copy
    list is EXACTLY `ring.segments_at(step_start, step_rows)`, i.e. what the code did before this
    task -- which is what makes "flag off is a no-op" and "all-layers is a no-op" one statement;
  * `VLLM_HOOK_DRAIN_SELECTIVE` parsing (DEFAULT ON since Task 19, "1" is the only ON value);
  * the out-of-scope consumers: `per_request=True` and the SYNCHRONOUS drain both FULL-drain even
    when the flag is armed, enforced at the use site as well as at construction;
  * `ring.advance_drain` is still called with the FULL step span in every case (never-drop /
    backpressure semantics unchanged);
  * end-to-end: a selective drain's reconstruction is byte-identical to the full drain's while its
    raw files are genuinely smaller, including across a physical ring wrap;
  * the `hs.drain.rows_{copied,skipped}` counters + their worker RPC surface;
  * the copy-stream ordering discipline in `_read_segments` (source-order pin -- the real ordering
    oracle is the GPU leg; this box has no CUDA).

Run:  conda activate vllm_hook_env && pytest tests/unit/test_drain_selective.py -q
      (or standalone: python tests/unit/test_drain_selective.py)
"""
import inspect
import json
import os
import sys
import tempfile
import time

import pytest
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph import ring_drain_hs as rdh
from vllm_hook_plugins.graph.ring_drain_hs import (
    LayerCopyPlan, MultiLayerRingDrain, OffLoopRingDrain, _merge_ranges, _stamp_file_row,
    _torch_dtype_name, build_copy_plans, record_captured_cells)
from vllm_hook_plugins.graph.ring_metadata import (
    LayerEntry, ReqCaptureRecord, StepMeta, expand_records, read_sidecar, write_sidecar)
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact


def _mkdtmp(tag):
    return tempfile.mkdtemp(prefix=f"drainsel_{tag}_")


# --------------------------------------------------------------------------- #
# LayerEntry.file_row default / override
# --------------------------------------------------------------------------- #

def test_file_row_defaults_to_logical_start():
    e = LayerEntry("r0", 1, logical_start=42, n_rows=3, hs_mode="all_tokens")
    assert e.file_row == 42
    assert e.file_row == e.logical_start


def test_file_row_explicit_override_is_kept():
    e = LayerEntry("r0", 1, logical_start=42, n_rows=3, hs_mode="all_tokens", file_row=7)
    assert e.file_row == 7
    assert e.file_row != e.logical_start


def test_file_row_default_is_independent_per_instance():
    # A regression against a mutable-default-argument-style bug: two entries with different
    # logical_start must each get their OWN defaulted file_row, not share state.
    a = LayerEntry("A", 0, logical_start=5, n_rows=1, hs_mode="last_token")
    b = LayerEntry("B", 0, logical_start=99, n_rows=1, hs_mode="last_token")
    assert a.file_row == 5
    assert b.file_row == 99


# --------------------------------------------------------------------------- #
# _stamp_file_row: DIRECT arithmetic test with a synthetic cursor_before that DIVERGES from
# step_start_logical (fix per review: the property tests above cannot distinguish a genuinely
# wired cursor from a dropped/no-op _stamp_file_row call, because every production call site
# today produces cursor_before[ln] == step_start_logical for every layer -- an
# all-layers-in-lockstep coincidence that ALSO happens to match LayerEntry.__post_init__'s
# file_row = logical_start default. A synthetic cursor_before that does NOT equal
# step_start_logical is impossible to satisfy via that default, so this fails loud if stamping
# is ever silently skipped.
# --------------------------------------------------------------------------- #

def test_stamp_file_row_arithmetic_direct_diverges_from_logical_start():
    """Two layers, DIFFERENT cursor_before values in the SAME call (so a single shared counter
    substituted for the per-layer dict would also fail this -- per-layer divergence is
    untestable through the real drain today, since every layer moves in lockstep until Task 15's
    selective drain; a direct call to _stamp_file_row can test it now)."""
    e1 = LayerEntry("A", 1, logical_start=100, n_rows=3, hs_mode="all_tokens")   # layer 1
    e2 = LayerEntry("A", 2, logical_start=100, n_rows=3, hs_mode="all_tokens")   # layer 2, same step
    e3 = LayerEntry("B", 1, logical_start=105, n_rows=2, hs_mode="all_tokens")   # layer 1, later slice
    cursor_before = {1: 50, 2: 900}   # deliberately DIFFERENT per layer; neither == step_start_logical
    step_start_logical = 100

    _stamp_file_row([e1, e2, e3], cursor_before, step_start_logical)

    # base[layer] + (entry.logical_start - step_start_logical)
    assert e1.file_row == 50 + (100 - 100) == 50
    assert e2.file_row == 900 + (100 - 100) == 900
    assert e3.file_row == 50 + (105 - 100) == 55

    # Every result DIVERGES from logical_start -- none of these values could be produced by
    # LayerEntry.__post_init__'s default (file_row = logical_start). If _stamp_file_row were
    # ever accidentally skipped (a no-op / dropped call site), file_row would stay at 100, 100,
    # 105 respectively and every assertion above would already have failed; these are restated
    # explicitly so the "diverges from the default" property is visible even if the exact
    # arithmetic above is refactored.
    assert e1.file_row != e1.logical_start
    assert e2.file_row != e2.logical_start
    assert e3.file_row != e3.logical_start
    # Layer-distinctness: e1 and e2 share a step and a logical_start but differ by LAYER only.
    assert e1.file_row != e2.file_row


def test_stamp_file_row_missing_layer_in_cursor_before_defaults_to_zero():
    # An entry whose layer has no cursor_before entry (never appended this run) starts at 0 --
    # matches _append_layer_rows' self._file_rows.get(ln, 0) convention for a fresh layer.
    e = LayerEntry("A", 7, logical_start=20, n_rows=1, hs_mode="last_token")
    _stamp_file_row([e], cursor_before={}, step_start_logical=20)
    assert e.file_row == 0


# --------------------------------------------------------------------------- #
# Sidecar round-trip: file_row survives as a field DISTINCT from logical_start
# --------------------------------------------------------------------------- #

def test_sidecar_roundtrip_carries_file_row_distinct_from_logical_start(tmp_path):
    # Construct entries where file_row DELIBERATELY differs from logical_start, to prove the sidecar
    # carries both independently rather than silently re-deriving one from the other.
    steps = [StepMeta([
        LayerEntry("r0", 0, logical_start=100, n_rows=5, hs_mode="all_tokens", file_row=0),
        LayerEntry("r0", 1, logical_start=105, n_rows=5, hs_mode="all_tokens", file_row=5),
    ])]
    header = {"dtype": "float32", "row_shape": [4], "hidden": 4}
    p = tmp_path / "meta.jsonl"
    write_sidecar(str(p), steps, header)

    # The raw JSON line must carry a SEPARATE "fr" key alongside "o" (logical_start) -- assert on the
    # literal on-disk shape, not just the round-tripped Python object.
    lines = p.read_text().splitlines()
    assert len(lines) == 3  # header + 2 entries
    row0 = json.loads(lines[1])
    assert row0["o"] == 100 and row0["fr"] == 0
    row1 = json.loads(lines[2])
    assert row1["o"] == 105 and row1["fr"] == 5

    _, got = read_sidecar(str(p))
    flat = [e for s in got for e in s.entries]
    assert [(e.logical_start, e.file_row) for e in flat] == [(100, 0), (105, 5)]


# --------------------------------------------------------------------------- #
# Compatibility, both directions (brief Step 3)
# --------------------------------------------------------------------------- #

def test_new_reader_old_sidecar_falls_back_to_logical_start(tmp_path):
    """A sidecar written BEFORE this change has no "fr" key on any entry line. The current
    (post-Task-14) `read_sidecar` must fall back to "o" (logical_start) for `file_row` -- exactly
    correct, since every such artifact was written with every layer receiving every row."""
    header = {"dtype": "float32", "row_shape": [4], "hidden": 4}
    p = tmp_path / "old_meta.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"__header__": header}) + "\n")
        # Pre-Task-14 row shape: r/l/o/n/m only, no "fr".
        f.write(json.dumps({"s": 0, "r": "r0", "l": 0, "o": 0, "n": 5, "m": "all_tokens"}) + "\n")
        f.write(json.dumps({"s": 0, "r": "r0", "l": 1, "o": 5, "n": 5, "m": "all_tokens"}) + "\n")

    h, got = read_sidecar(str(p))
    assert h == header
    flat = [e for s in got for e in s.entries]
    assert len(flat) == 2
    for e in flat:
        assert e.file_row == e.logical_start, (
            "old sidecar (no 'fr' key) must fall back file_row -> logical_start")
    assert [e.logical_start for e in flat] == [0, 5]


def _old_read_sidecar(path: str):
    """A frozen copy of the PRE-Task-14 `read_sidecar` body -- 5-key access only (r/l/o/n/m), no
    knowledge of "fr". Used to prove an OLD reader tolerates a NEW sidecar (ignores the unknown key)
    rather than merely asserting it "should" by inspection."""
    header = None
    steps_by_idx = {}
    order = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0:
                header = obj["__header__"]
                continue
            s = obj["s"]
            if s not in steps_by_idx:
                steps_by_idx[s] = StepMeta([])
                order.append(s)
            steps_by_idx[s].entries.append(
                LayerEntry(obj["r"], obj["l"], obj["o"], obj["n"], obj["m"])
            )
    steps = [steps_by_idx[s] for s in order]
    return header, steps


def test_old_reader_ignores_unknown_fr_key_on_new_sidecar(tmp_path):
    """An OLD reader (pre-Task-14 read_sidecar, 5-key access) must not choke on a NEW sidecar that
    carries the extra "fr" key -- confirmed by literally running the old parsing code against a
    sidecar written by the CURRENT write_sidecar (which does emit "fr")."""
    steps = [StepMeta([LayerEntry("r0", 0, logical_start=0, n_rows=5, hs_mode="all_tokens"),
                       LayerEntry("r0", 1, logical_start=5, n_rows=5, hs_mode="all_tokens")])]
    header = {"dtype": "float32", "row_shape": [4], "hidden": 4}
    p = tmp_path / "new_meta.jsonl"
    write_sidecar(str(p), steps, header)

    # Sanity: the file really does carry the new key (otherwise this test would be vacuous).
    lines = p.read_text().splitlines()
    assert all("fr" in json.loads(line) for line in lines[1:])

    h, got = _old_read_sidecar(str(p))   # must not raise
    assert h == header
    flat = [(e.req_id, e.layer, e.logical_start, e.n_rows, e.hs_mode)
            for s in got for e in s.entries]
    assert flat == [("r0", 0, 0, 5, "all_tokens"), ("r0", 1, 5, 5, "all_tokens")]


# --------------------------------------------------------------------------- #
# Per-layer running cursor: arithmetic across multiple steps and layers
# --------------------------------------------------------------------------- #

def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row, never drained


def _write_slots(hs_buf, ring, start_slot, rows):
    for j, p in enumerate(ring.physical_slots(start_slot, rows.shape[0])):
        hs_buf[p] = rows[j]


def _build_sync(tmp, layer_ids, hidden=4, R=64, dtype=torch.float32):
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = MultiLayerRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header)
    return ring, hs_bufs, drain


def test_file_rows_cursor_starts_at_zero_per_layer(tmp_path):
    _, _, drain = _build_sync(str(tmp_path), layer_ids=(1, 2, 3))
    assert drain._file_rows == {1: 0, 2: 0, 3: 0}


def test_file_rows_cursor_advances_by_rows_appended_multi_step(tmp_path):
    """Drive several steps of varying row counts across 3 layers (every layer appended every step,
    the only path this task builds) and assert the per-layer running cursor equals the CUMULATIVE
    row count appended so far -- identical across all layers at every point, since every layer gets
    every row."""
    hidden = 4
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain = _build_sync(str(tmp_path), layer_ids, hidden=hidden, R=256)

    running_total = 0
    for n in (3, 1, 5, 1, 2):   # mixed all_tokens / last_token-shaped step sizes
        s = ring.reserve(n)
        data = torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden)
        for L in layer_ids:
            _write_slots(hs_bufs[L], ring, s, data.to(hs_bufs[L].dtype))
        drain.record_entries([LayerEntry("A", L, s, n, "all_tokens") for L in layer_ids])
        assert drain.drain_once() == n
        running_total += n
        assert drain._file_rows == {L: running_total for L in layer_ids}, (
            f"cursor mismatch after step n={n}: {drain._file_rows} != "
            f"{{L: {running_total} for L in layer_ids}}")
    drain.close()


# --------------------------------------------------------------------------- #
# THE regression invariant: on an all-layers drain, file_row == logical_start for
# every entry -- as a property over several step/batch/layer-count shapes.
# --------------------------------------------------------------------------- #

# Each shape: (num_layers, ring_slots, step_specs). step_specs is a list of steps; each step is a
# list of (req_id, n_rows, mode) tuples -- every request in a step is written to EVERY installed
# layer (matches the shared-file drain's unconditional per-layer append this task preserves).
_SHAPES = [
    pytest.param(1, 64, [[("A", 1, "last_token")]], id="1layer_1step_1req"),
    pytest.param(3, 64,
                 [[("A", 3, "all_tokens"), ("B", 1, "last_token")], [("C", 5, "all_tokens")]],
                 id="3layer_2step_mixed_multireq"),
    pytest.param(5, 4,   # tiny ring -> forces a genuine physical wrap mid-sweep
                 [[("A", 3, "all_tokens")], [("B", 4, "all_tokens")], [("C", 1, "last_token")]],
                 id="5layer_wrap"),
    pytest.param(2, 128,
                 [[("A", 1, "last_token")] for _ in range(10)],
                 id="2layer_10steps_singlereq"),
    pytest.param(4, 32,
                 [[("A", 2, "all_tokens"), ("B", 3, "all_tokens"), ("C", 1, "last_token")]],
                 id="4layer_1step_3req_heterogeneous_rowcount"),
]


def _drive_sync(tmp, num_layers, ring_slots, step_specs, hidden=4, dtype=torch.float32):
    layer_ids = tuple(range(1, num_layers + 1))
    ring, hs_bufs, drain = _build_sync(tmp, layer_ids, hidden=hidden, R=ring_slots, dtype=dtype)
    for step in step_specs:
        entries = []
        for (rid, n, mode) in step:
            s = ring.reserve(n)
            assert s is not None, "shape under-sized its ring -- fix the test shape, not the code"
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden)
                    + (hash(rid) % 500)).to(dtype)
            for L in layer_ids:
                _write_slots(hs_bufs[L], ring, s, data)
                entries.append(LayerEntry(str(rid), L, s, n, mode))
        drain.record_entries(entries)
        assert drain.drain_once() == sum(n for _, n, _ in step)
    drain.close()
    return os.path.join(tmp, "hs_ring_meta.jsonl")


def _drive_offloop(tmp, num_layers, ring_slots, step_specs, hidden=4, dtype=torch.float32):
    layer_ids = tuple(range(1, num_layers + 1))
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=ring_slots, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(ring_slots, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = OffLoopRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header)
    drain.start()
    for step in step_specs:
        entries = []
        start_logical = None
        total = 0
        for (rid, n, mode) in step:
            s = ring.reserve(n)
            assert s is not None, "shape under-sized its ring -- fix the test shape, not the code"
            if start_logical is None:
                start_logical = s
            total += n
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden)
                    + (hash(rid) % 500)).to(dtype)
            for L in layer_ids:
                _write_slots(hs_bufs[L], ring, s, data)
                entries.append(LayerEntry(str(rid), L, s, n, mode))
        drain.enqueue(entries, start_logical, total, None)
        deadline = time.monotonic() + 10.0
        while ring.pending_rows() > 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert ring.pending_rows() == 0, "off-loop consumer did not drain within timeout"
    drain.stop()    # join the consumer thread
    drain.close()   # write the sidecar
    return os.path.join(tmp, "hs_ring_meta.jsonl")


def _assert_file_row_matches_logical_start(meta_path):
    _, steps = read_sidecar(meta_path)
    entries = [e for s in steps for e in s.entries]
    assert entries, "no entries recorded -- vacuous check"
    for e in entries:
        assert e.file_row == e.logical_start, (
            f"file_row {e.file_row} != logical_start {e.logical_start} "
            f"for req={e.req_id} layer={e.layer}")
    return entries


@pytest.mark.parametrize("num_layers,ring_slots,step_specs", _SHAPES)
def test_sync_drain_file_row_equals_logical_start_property(
        tmp_path, num_layers, ring_slots, step_specs):
    meta_path = _drive_sync(str(tmp_path), num_layers, ring_slots, step_specs)
    entries = _assert_file_row_matches_logical_start(meta_path)
    # Non-vacuity: every (req, layer) pair from the step spec is actually represented.
    expected_pairs = {(str(rid), L) for step in step_specs for (rid, n, mode) in step
                       for L in range(1, num_layers + 1)}
    got_pairs = {(e.req_id, e.layer) for e in entries}
    assert got_pairs == expected_pairs


@pytest.mark.parametrize("num_layers,ring_slots,step_specs", _SHAPES)
def test_offloop_drain_file_row_equals_logical_start_property(
        tmp_path, num_layers, ring_slots, step_specs):
    meta_path = _drive_offloop(str(tmp_path), num_layers, ring_slots, step_specs)
    entries = _assert_file_row_matches_logical_start(meta_path)
    expected_pairs = {(str(rid), L) for step in step_specs for (rid, n, mode) in step
                       for L in range(1, num_layers + 1)}
    got_pairs = {(e.req_id, e.layer) for e in entries}
    assert got_pairs == expected_pairs


# --------------------------------------------------------------------------- #
# End-to-end reconstruction is unaffected (now keyed on file_row, not logical_start)
# --------------------------------------------------------------------------- #

def test_reconstruction_still_byte_identical_via_file_row(tmp_path):
    tmp = str(tmp_path)
    hidden = 4
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain = _build_sync(tmp, layer_ids, hidden=hidden, R=64)

    def _data(n, base):
        return (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base)

    startA = ring.reserve(3)
    dataA = {L: _data(3, 100 * L) for L in layer_ids}
    for L in layer_ids:
        _write_slots(hs_bufs[L], ring, startA, dataA[L])
    startB = ring.reserve(1)
    dataB2 = _data(1, 999)
    _write_slots(hs_bufs[2], ring, startB, dataB2)
    drain.record_entries(
        [LayerEntry("A", L, startA, 3, "all_tokens") for L in layer_ids]
        + [LayerEntry("B", 2, startB, 1, "last_token")])
    assert drain.drain_once() == 4
    drain.close()

    out = load_multilayer_ring_artifact(tmp)
    for L in layer_ids:
        assert torch.equal(out["A"][L], dataA[L]), f"A layer {L} mismatch"
    assert torch.equal(out["B"][2], dataB2), "B layer 2 mismatch"
    assert 1 not in out["B"] and 3 not in out["B"]


def test_reconstruction_across_physical_wrap_via_file_row(tmp_path):
    tmp = str(tmp_path)
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
    sB = ring.reserve(4)
    assert sB == 3
    _write_slots(buf, ring, sB, dataB)
    assert len(ring.drained_segments()) == 2, "expected a genuine physical wrap"
    drain.record_entries([LayerEntry("B", 1, sB, 4, "all_tokens")])
    assert drain.drain_once() == 4

    drain.close()
    out = load_multilayer_ring_artifact(tmp)
    assert torch.equal(out["A"][1], dataA)
    assert torch.equal(out["B"][1], dataB), "wrapped block not reconstructed in file_row order"

    _assert_file_row_matches_logical_start(os.path.join(tmp, "hs_ring_meta.jsonl"))


# =========================================================================== #
# SELECTIVE DRAIN (Task 15) -- VLLM_HOOK_DRAIN_SELECTIVE, DEFAULT ON since Task 19
# =========================================================================== #

def _mk_ring(R, hidden, dtype=torch.float32):
    return GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))


def _sig(layer, logical_row):
    """A value unique to (layer, logical ring row), so a copy that lands on the wrong layer or the
    wrong row is caught by VALUE, not merely by shape."""
    return float(int(layer) * 10000 + int(logical_row))


def _fill_step(hs_bufs, ring, start, n):
    """Write the (layer, logical-row) signature into EVERY installed layer's ring slots for
    [start, start+n). Every layer is filled deliberately -- so a selective drain that wrongly copies
    an unwanted layer picks up plausible-looking data rather than zeros, and only the file
    size / file_row bookkeeping can catch it."""
    for L, buf in hs_bufs.items():
        for j, p in enumerate(ring.physical_slots(start, n)):
            buf[p] = _sig(L, start + j)


def _expect_block(layer, start, n, hidden, dtype=torch.float32):
    return torch.tensor([[_sig(layer, start + j)] * hidden for j in range(n)], dtype=dtype)


def _mk_offloop_drain(tmp, layer_ids, R=64, hidden=4, dtype=torch.float32, per_request=False):
    """An OffLoopRingDrain whose consumer thread is NOT started -- for tests that call the
    consumer-side methods (`_read_segments`) directly on the calling thread."""
    ring = _mk_ring(R, hidden, dtype)
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = OffLoopRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header,
                             per_request=per_request)
    return ring, hs_bufs, drain


class _OffLoopHarness:
    """Drives real steps through a real ``OffLoopRingDrain`` on CPU (no CUDA on this box, so
    ``_read_segments`` takes its CPU branch). ``step()`` reserves ring rows per request, fills every
    layer's slots, and enqueues ONE ``_DrainItem`` worth of per-request records -- the exact shape
    ``_build_routing_hs`` produces."""

    def __init__(self, tmp, layer_ids, R=64, hidden=4, dtype=torch.float32, per_request=False):
        self.layer_ids = tuple(layer_ids)
        self.hidden = hidden
        self.dtype = dtype
        self.tmp = tmp
        self.ring = _mk_ring(R, hidden, dtype)
        self.hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in self.layer_ids}
        header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
        self.drain = OffLoopRingDrain(
            self.ring, [(L, self.hs_bufs[L]) for L in self.layer_ids], tmp, header,
            per_request=per_request)
        self.drain.start()
        self.placed = []      # (req_id, layers, logical_start, n_rows)

    def step(self, specs):
        """specs: list of ``(req_id, n_rows, mode, layers)`` -- ``layers`` is THAT request's own
        wanted layer list (heterogeneous sets across requests are the point)."""
        records = []
        start_logical = None
        total = 0
        for rid, n, mode, layers in specs:
            s = self.ring.reserve(n)
            assert s is not None, "test shape under-sized its ring"
            if start_logical is None:
                start_logical = s
            total += n
            _fill_step(self.hs_bufs, self.ring, s, n)
            records.append(ReqCaptureRecord(req_id=str(rid), logical_start=s, n_rows=n,
                                            hs_mode=mode, layers=list(layers)))
            self.placed.append((str(rid), tuple(layers), s, n))
        self.drain.enqueue(records, start_logical, total, None)
        self.wait()
        return start_logical, total

    def wait(self):
        deadline = time.monotonic() + 10.0
        while self.ring.pending_rows() > 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert self.drain.error is None, f"consumer died: {self.drain.error!r}"
        assert self.ring.pending_rows() == 0, "off-loop consumer did not drain within timeout"

    def finish(self):
        self.drain.stop()
        self.drain.close()

    def layer_file_rows(self, layer):
        path = os.path.join(self.tmp, f"hs_layer_{layer}.raw")
        if not os.path.exists(path):
            return 0
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        return os.path.getsize(path) // (self.hidden * itemsize)

    def assert_reconstruction(self):
        """Every placed (request, layer) block reconstructs BYTE-IDENTICALLY, and no request carries
        a layer it never asked for."""
        out = load_multilayer_ring_artifact(self.tmp)
        want_layers = {}
        for rid, layers, s, n in self.placed:
            for L in layers:
                want_layers.setdefault(rid, set()).add(L)
                assert torch.equal(out[rid][L], _expect_block(L, s, n, self.hidden, self.dtype)), (
                    f"req={rid} layer={L} rows[{s}:{s + n}) mismatch:\n"
                    f"got {out[rid][L]}\nwant {_expect_block(L, s, n, self.hidden, self.dtype)}")
        for rid, layers in want_layers.items():
            assert set(out[rid]) == layers, (
                f"req={rid} reconstructed layers {sorted(out[rid])} != wanted {sorted(layers)}")
        return out


@pytest.fixture
def selective_on(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
    # Keep the per-layer mmap pre-size small: these tests assert on real file SIZES, and the
    # production default pre-sizes every layer file to >= 2 GiB (sparse) before truncating on close.
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    return True


@pytest.fixture
def selective_off(monkeypatch):
    # PINNED "0", not deleted: since Task 19 the flag DEFAULTS ON, so an unset env is the ARMED
    # state. Every "unarmed" arm in this file pins it for the same reason.
    monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    return False


# --------------------------------------------------------------------------- #
# Step 1: the copy list, as a pure function
# --------------------------------------------------------------------------- #

def test_merge_ranges_sorts_and_merges_touching_and_overlapping():
    # touching (5+3 == 8), overlapping (10..14 vs 12..16), disjoint (20..22), out of order input.
    assert _merge_ranges([(20, 2), (5, 3), (8, 1), (12, 4), (10, 2)]) == [(5, 4), (10, 6), (20, 2)]


def test_merge_ranges_drops_empty_and_handles_empty_input():
    assert _merge_ranges([]) == []
    assert _merge_ranges([(4, 0), (7, 0)]) == []
    assert _merge_ranges([(4, 0), (7, 2)]) == [(7, 2)]


def test_copy_plan_degenerate_all_layers_equals_todays_segments():
    """THE CONTRACT. Every request wants every installed layer -> each layer's copy list is exactly
    `ring.segments_at(step_start, step_rows)`, the single list today's `_drain_item` passes for ALL
    layers. This is what makes "flag off is a no-op" and "all-layers is a no-op" one statement."""
    ring = _mk_ring(64, 4)
    layers = (1, 2, 3, 4)
    ring._write = ring._drain = 10                       # step starts at logical 10
    records = [ReqCaptureRecord("A", 10, 3, "all_tokens", list(layers)),
               ReqCaptureRecord("B", 13, 2, "all_tokens", list(layers))]
    plans = build_copy_plans(records, layers, ring, 10, 5, selective=True)
    today = tuple(ring.segments_at(10, 5))
    assert set(plans) == set(layers)
    for L in layers:
        assert plans[L].segments == today, f"layer {L}: {plans[L].segments} != {today}"
        assert plans[L].total_rows == 5


def test_copy_plan_degenerate_all_layers_equals_todays_segments_across_a_wrap():
    """Same contract, but the step straddles the physical wrap (2 segments)."""
    ring = _mk_ring(8, 4)
    layers = (1, 2)
    records = [ReqCaptureRecord("A", 6, 4, "all_tokens", list(layers))]
    plans = build_copy_plans(records, layers, ring, 6, 4, selective=True)
    today = tuple(ring.segments_at(6, 4))
    assert len(today) == 2, "test shape must actually wrap"
    for L in layers:
        assert plans[L].segments == today
        assert plans[L].total_rows == 4


def test_copy_plan_skips_a_layer_nobody_asked_for():
    ring = _mk_ring(64, 4)
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 3])]
    plans = build_copy_plans(records, (1, 2, 3, 4), ring, 0, 3, selective=True)
    assert set(plans) == {1, 3}, "layers 2 and 4 were wanted by nobody -> no copy at all"
    for L in (1, 3):
        assert plans[L].segments == ((0, 3),)


def test_copy_plan_heterogeneous_layer_sets_give_different_row_ranges_per_layer():
    """A wants layers [1,2] over rows [0,3); B wants [2,3] over rows [3,5). Layer 2 is the only one
    that must copy the whole span; layer 1 copies A's rows only and layer 3 B's rows only."""
    ring = _mk_ring(64, 4)
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 2]),
               ReqCaptureRecord("B", 3, 2, "all_tokens", [2, 3])]
    plans = build_copy_plans(records, (1, 2, 3), ring, 0, 5, selective=True)
    assert plans[1].ranges == ((0, 3),) and plans[1].segments == ((0, 3),)
    assert plans[2].ranges == ((0, 5),), "A and B are TOUCHING -> one merged range"
    assert plans[2].segments == ((0, 5),)
    assert plans[3].ranges == ((3, 2),) and plans[3].segments == ((3, 5),)
    assert (plans[1].total_rows, plans[2].total_rows, plans[3].total_rows) == (3, 5, 2)


def test_copy_plan_row_offset_compacts_across_a_gap():
    """Layer 2 is wanted by A (rows 0..2) and C (rows 7..8) but not by the B rows in between, so its
    copy is COMPACTED: C's rows land at compacted offset 3, not at 7."""
    ring = _mk_ring(64, 4)
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [2]),
               ReqCaptureRecord("B", 3, 4, "all_tokens", [5]),
               ReqCaptureRecord("C", 7, 2, "all_tokens", [2])]
    plans = build_copy_plans(records, (2, 5), ring, 0, 9, selective=True)
    p = plans[2]
    assert p.ranges == ((0, 3), (7, 2))
    assert p.segments == ((0, 3), (7, 9))
    assert p.total_rows == 5
    assert p.row_offset(0) == 0
    assert p.row_offset(1) == 1
    assert p.row_offset(7) == 3, "C's first row is the 4th COPIED row, not the 8th ring row"
    assert p.row_offset(8) == 4


def test_copy_plan_row_offset_rejects_a_row_it_never_copied():
    ring = _mk_ring(64, 4)
    plans = build_copy_plans([ReqCaptureRecord("A", 0, 3, "all_tokens", [1])],
                             (1,), ring, 0, 3, selective=True)
    with pytest.raises(KeyError):
        plans[1].row_offset(5)       # never copied -> must fail loud, never mis-place


def test_copy_plan_from_records_equals_from_expanded_entries():
    """The brief specifies the union over `expand_records(item.entries)`; the implementation reads
    the records directly (same (layer, range) pairs, no O(reqs x layers) dataclass allocation on the
    consumer thread). Pin the equivalence rather than assume it."""
    ring = _mk_ring(32, 4)
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 2]),
               ReqCaptureRecord("B", 3, 1, "last_token", [2, 3]),
               ReqCaptureRecord("C", 4, 2, "all_tokens", [1, 3])]
    from_records = build_copy_plans(records, (1, 2, 3), ring, 0, 6, selective=True)
    from_entries = build_copy_plans(expand_records(records), (1, 2, 3), ring, 0, 6, selective=True)
    assert from_records == from_entries


def test_copy_plan_ignores_a_layer_that_is_not_installed():
    ring = _mk_ring(32, 4)
    records = [ReqCaptureRecord("A", 0, 2, "all_tokens", [1, 99])]
    plans = build_copy_plans(records, (1, 2), ring, 0, 2, selective=True)
    assert set(plans) == {1}


def test_copy_plan_not_selective_is_the_full_span_for_every_installed_layer():
    """selective=False must reproduce today's behaviour exactly: EVERY installed layer copies the
    whole step span, including layers no request asked for."""
    ring = _mk_ring(64, 4)
    records = [ReqCaptureRecord("A", 10, 3, "all_tokens", [1])]
    plans = build_copy_plans(records, (1, 2, 3), ring, 10, 3, selective=False)
    today = tuple(ring.segments_at(10, 3))
    assert set(plans) == {1, 2, 3}
    for L in (1, 2, 3):
        assert plans[L].segments == today and plans[L].total_rows == 3


def test_copy_plan_empty_entries_selective_copies_nothing():
    ring = _mk_ring(64, 4)
    assert build_copy_plans([], (1, 2), ring, 0, 4, selective=True) == {}


# --------------------------------------------------------------------------- #
# The copy list is BOUNDED BY THE STEP SPAN (review finding 1). Before this lever the copy was
# structurally bounded -- `segments_at(item.start_logical, item.n_rows)` could not reach outside the
# step no matter what the records said. Building the list from the records converts that structural
# invariant into an unchecked one: a record claiming rows the step does not own would have the
# consumer copy rows `item.event` does not fence and the engine may be actively scattering into --
# right shapes, right layer sets, stale-or-torn contents. Unreachable from all four routing builders
# today; that is exactly the argument that justified the zero-row guard, and this failure is worse.
# --------------------------------------------------------------------------- #

def test_copy_plan_rejects_a_record_reaching_past_the_step_span():
    ring = _mk_ring(64, 4)
    # The step owns [10, 15); the record claims 20 rows from 10 -> 15 rows the step does not own.
    with pytest.raises(ValueError, match="outside this step"):
        build_copy_plans([ReqCaptureRecord("A", 10, 20, "all_tokens", [1])],
                         (1,), ring, 10, 5, selective=True)


def test_copy_plan_rejects_a_record_starting_before_the_step_span():
    ring = _mk_ring(64, 4)
    with pytest.raises(ValueError, match="outside this step"):
        build_copy_plans([ReqCaptureRecord("A", 8, 2, "all_tokens", [1])],
                         (1,), ring, 10, 5, selective=True)


def test_copy_plan_rejects_an_out_of_span_record_even_when_another_layer_is_fine():
    """The check is per layer, so a good layer must not mask a bad one."""
    ring = _mk_ring(64, 4)
    with pytest.raises(ValueError, match="layer 2"):
        build_copy_plans([ReqCaptureRecord("A", 10, 5, "all_tokens", [1]),
                          ReqCaptureRecord("B", 14, 4, "all_tokens", [2])],
                         (1, 2), ring, 10, 5, selective=True)


@pytest.mark.parametrize("ranges,lo,hi", [
    ([(10, 5)], 10, 5),                 # exactly the whole span
    ([(10, 1)], 10, 5),                 # first row only
    ([(14, 1)], 10, 5),                 # last row only
    ([(10, 2), (13, 2)], 10, 5),        # two sub-ranges with a gap, both inside
    ([(6, 4)], 6, 4),                   # a span that wraps physically (ring has 8 slots below)
])
def test_copy_plan_span_check_does_not_fire_on_in_span_records(ranges, lo, hi):
    """The bound must never fire on anything the routing builders actually produce -- including at
    both boundaries and across a physical wrap (the check is on LOGICAL rows, so wrapping is
    irrelevant to it)."""
    ring = _mk_ring(8, 4)
    records = [ReqCaptureRecord(f"r{i}", s, n, "all_tokens", [1])
               for i, (s, n) in enumerate(ranges)]
    plans = build_copy_plans(records, (1,), ring, lo, hi, selective=True)
    assert plans[1].total_rows == sum(n for _, n in ranges)


# --------------------------------------------------------------------------- #
# Step 3 / flag parsing
# --------------------------------------------------------------------------- #

def test_flag_defaults_on(monkeypatch):
    """DEFAULT ON since Task 19 (the degenerate all-layers path became free, so the lever costs
    nothing where it has nothing to skip). Inverted from `test_flag_defaults_off`; the kill-switch
    pin below is what keeps the OFF state covered, mirroring the CAPTURE_FUSED flip."""
    monkeypatch.delenv("VLLM_HOOK_DRAIN_SELECTIVE", raising=False)
    assert rdh._drain_selective_enabled() is True


def test_flag_one_is_on(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
    assert rdh._drain_selective_enabled() is True


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "on", "2"])
def test_flag_only_the_literal_one_is_on(monkeypatch, value):
    """`== "1"` stays the rule after the default-ON flip: "0" is the kill switch (the full-drain
    control every regression leg pins), and a stray "true"/"yes" reads as OFF rather than silently
    meaning the opposite of what it says."""
    monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", value)
    assert rdh._drain_selective_enabled() is False


def test_offloop_shared_file_drain_arms_selective(tmp_path, selective_on):
    h = _OffLoopHarness(str(tmp_path), (1, 2))
    try:
        assert h.drain.selective is True
        assert h.drain.selective_disabled_reason is None
        assert h.drain._selective_active() is True
    finally:
        h.finish()


def test_offloop_drain_default_is_full_drain(tmp_path, selective_off):
    h = _OffLoopHarness(str(tmp_path), (1, 2))
    try:
        assert h.drain.selective is False
        assert h.drain.selective_disabled_reason is None   # not armed -> nothing to explain
        assert h.drain._selective_active() is False
    finally:
        h.finish()


def test_per_request_mode_refuses_selective_and_says_why(tmp_path, selective_on):
    """Step 3 DECISION: FULL-DRAIN FALLBACK (not a raise). `per_request=True` + the flag armed
    drains every layer, and the drain carries a machine-readable reason for the install log."""
    h = _OffLoopHarness(str(tmp_path), (1, 2), per_request=True)
    try:
        assert h.drain.selective is False
        assert h.drain.selective_disabled_reason is not None
        assert "per-request" in h.drain.selective_disabled_reason.lower()
        assert "VLLM_HOOK_RING_PER_REQUEST" in h.drain.selective_disabled_reason
    finally:
        h.finish()


def test_sync_drain_refuses_selective_and_says_why(tmp_path, selective_on):
    _, _, drain = _build_sync(str(tmp_path), layer_ids=(1, 2))
    assert drain.selective is False
    assert drain.selective_disabled_reason is not None
    assert "sync" in drain.selective_disabled_reason.lower()
    assert "VLLM_HOOK_RING_SYNC_DRAIN" in drain.selective_disabled_reason
    assert drain._selective_active() is False


def test_per_request_full_drain_is_enforced_at_the_use_site(tmp_path, selective_on):
    """Defense in depth: even if something force-sets `.selective` True on a per-request drain, the
    use site (`_selective_active`) still full-drains -- the two features must never meet, and the
    guard must not live only in the constructor."""
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3), per_request=True)
    try:
        h.drain.selective = True                       # force the disallowed combination
        assert h.drain._selective_active() is False
        h.step([("A", 3, "all_tokens", [1])])          # A wants ONE of three layers
        counts = h.drain.row_counts()
        assert counts["hs.drain.rows_copied"] == 3 * 3, "per-request mode must copy every layer"
        assert counts["hs.drain.rows_skipped"] == 0
        # The demuxed rows are still correct (its `off` is a dense step-image offset).
        h.drain.enqueue_finish("A")
        deadline = time.monotonic() + 10.0
        while not h.drain.index._deliverable and time.monotonic() < deadline:
            time.sleep(0.001)
        delivered = dict(h.drain.index.pop_deliverable())
        assert set(delivered) == {"A"}
        assert torch.equal(delivered["A"][1], _expect_block(1, 0, 3, h.hidden))
    finally:
        h.finish()


# --------------------------------------------------------------------------- #
# Step 2 invariants: advance_drain, ordering
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("armed", [False, True])
def test_advance_drain_is_called_with_the_full_step_span_in_every_case(
        tmp_path, monkeypatch, armed):
    """INVARIANT 1: the ring frees the WHOLE step span whether or not a row was copied -- never-drop
    and backpressure are untouched by this lever."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    if armed:
        monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
    else:
        monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3, 4))
    calls = []
    orig = h.ring.advance_drain

    def _spy(n):
        calls.append(int(n))
        return orig(n)

    h.ring.advance_drain = _spy
    try:
        # Step 1: nobody wants layers 3/4. Step 2: only ONE row of the span is wanted at all.
        h.step([("A", 3, "all_tokens", [1]), ("B", 2, "all_tokens", [2])])
        h.step([("C", 1, "last_token", [1]), ("D", 4, "all_tokens", [])])
        assert calls == [5, 5], f"advance_drain must free the full span, got {calls}"
        assert h.ring.pending_rows() == 0
    finally:
        h.finish()


def test_read_segments_keeps_the_copy_stream_ordering(tmp_path):
    """INVARIANT 2 (source-order pin). This box has no CUDA driver, so the CUDA branch of
    `_read_segments` cannot be executed here -- the real oracle is the GPU ring-parity leg. What a
    no-GPU test CAN do is fail if the ordering calls are deleted or reordered: stale-event wait ->
    wait_event(this step's scatter) -> copy_ + record_stream -> record + wait completion. "Fewer
    copies must not become unordered copies" is the subtlest way to get selective drain wrong."""
    src = inspect.getsource(OffLoopRingDrain._read_segments)
    order = ["stale.synchronize()", "wait_event(event)", ".copy_(", "record_stream(",
             ".record(self._stream)", "done.synchronize()"]
    pos = -1
    for token in order:
        i = src.find(token, pos + 1)
        assert i > pos, (
            f"copy-stream ordering token {token!r} missing or out of order in _read_segments; "
            f"the ordering discipline is load-bearing (see the docstring)")
        pos = i
    # The copies must be issued INSIDE the dedicated stream context, after the event wait.
    assert src.index("with torch.cuda.stream(self._stream)") < src.index(".copy_(")


# --------------------------------------------------------------------------- #
# Step 2 end-to-end: fewer bytes, identical values
# --------------------------------------------------------------------------- #

_HETERO_STEPS = [
    [("A", 3, "all_tokens", [1, 2]), ("B", 2, "all_tokens", [2, 3])],
    [("C", 1, "last_token", [4])],
    [("D", 2, "all_tokens", [1, 2, 3, 4])],
]


def _expected_layer_rows(steps, layer_ids):
    rows = {L: 0 for L in layer_ids}
    for step in steps:
        for _rid, n, _mode, layers in step:
            for L in layers:
                rows[L] += n
    return rows


def test_selective_drain_copies_only_the_wanted_tiles_and_reconstructs_identically(
        tmp_path, selective_on):
    layer_ids = (1, 2, 3, 4)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=64)
    total_rows = 0
    for step in _HETERO_STEPS:
        _, n = h.step(step)
        total_rows += n
    h.finish()

    h.assert_reconstruction()                      # byte-identical values (torch.equal)

    want = _expected_layer_rows(_HETERO_STEPS, layer_ids)
    got = {L: h.layer_file_rows(L) for L in layer_ids}
    assert got == want, f"per-layer file rows {got} != wanted-tile rows {want}"
    # Non-vacuity: a FULL drain would have written total_rows to every layer.
    assert all(got[L] < total_rows for L in layer_ids), (
        f"no layer was actually compacted (full drain would write {total_rows} rows each): {got}")

    counts = h.drain.row_counts()
    assert counts["hs.drain.rows_copied"] == sum(want.values())
    assert counts["hs.drain.rows_skipped"] == len(layer_ids) * total_rows - sum(want.values())
    assert counts["hs.drain.rows_skipped"] > 0
    assert counts["selective"] is True


def test_full_drain_reconstruction_is_identical_to_selective(tmp_path, monkeypatch):
    """The acceptance bar: same workload, flag OFF vs flag ON -> `torch.equal` on every
    reconstructed block. Only the FILES differ (that is the lever)."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    layer_ids = (1, 2, 3, 4)
    outs = {}
    sizes = {}
    for armed in (False, True):
        if armed:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
        else:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
        d = str(tmp_path / f"armed_{int(armed)}")
        os.makedirs(d, exist_ok=True)
        h = _OffLoopHarness(d, layer_ids, R=64)
        for step in _HETERO_STEPS:
            h.step(step)
        h.finish()
        outs[armed] = h.assert_reconstruction()
        sizes[armed] = {L: h.layer_file_rows(L) for L in layer_ids}

    assert set(outs[False]) == set(outs[True])
    for rid in outs[False]:
        assert set(outs[False][rid]) == set(outs[True][rid])
        for L in outs[False][rid]:
            assert torch.equal(outs[False][rid][L], outs[True][rid][L]), (
                f"selective drain changed VALUES for req={rid} layer={L}")
    assert sizes[True] != sizes[False], "selective must actually write fewer rows here"
    for L in layer_ids:
        assert sizes[True][L] <= sizes[False][L]


def test_all_layers_wanted_is_a_no_op_end_to_end(tmp_path, monkeypatch):
    """The degenerate case, driven for real: when every request wants every installed layer the
    armed drain must produce the IDENTICAL sidecar (`file_row` included) and identical file sizes as
    the unarmed one -- "flag off" and "all-layers" really are the same statement."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    layer_ids = (1, 2, 3)
    steps = [[("A", 3, "all_tokens", list(layer_ids)), ("B", 1, "last_token", list(layer_ids))],
             [("C", 2, "all_tokens", list(layer_ids))]]
    sidecars = {}
    sizes = {}
    for armed in (False, True):
        if armed:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
        else:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
        d = str(tmp_path / f"deg_{int(armed)}")
        os.makedirs(d, exist_ok=True)
        h = _OffLoopHarness(d, layer_ids, R=64)
        for step in steps:
            h.step(step)
        h.finish()
        h.assert_reconstruction()
        _, got = read_sidecar(os.path.join(d, "hs_ring_meta.jsonl"))
        sidecars[armed] = [(e.req_id, e.layer, e.logical_start, e.n_rows, e.hs_mode, e.file_row)
                           for s in got for e in s.entries]
        sizes[armed] = {L: h.layer_file_rows(L) for L in layer_ids}
    assert sidecars[True] == sidecars[False]
    assert sizes[True] == sizes[False]
    assert all(fr == ls for (_r, _l, ls, _n, _m, fr) in sidecars[True]), (
        "all-layers drain must still stamp file_row == logical_start")


def test_selective_drain_across_a_physical_wrap(tmp_path, selective_on):
    """A wrapping step under selective drain: the wanted range maps through `segments_at` to TWO
    physical segments, and reconstruction stays byte-identical."""
    layer_ids = (1, 2)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=8)
    h.step([("A", 5, "all_tokens", [1])])            # fills [0,5)
    start, _ = h.step([("B", 4, "all_tokens", [2])])  # [5,9) -> wraps 5..8 + 0..1
    assert len(h.ring.segments_at(start, 4)) == 2, "test shape must actually wrap"
    h.finish()
    h.assert_reconstruction()
    assert h.layer_file_rows(1) == 5
    assert h.layer_file_rows(2) == 4


def test_selective_drain_plain_append_sink(tmp_path, monkeypatch):
    """The non-mmap sink (VLLM_HOOK_RING_MMAP=0) takes the same append path -- cover it too, since
    the file-size assertions above only exercised the mmap writer's truncate-on-close."""
    monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP", "0")
    layer_ids = (1, 2, 3)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=64)
    h.step([("A", 2, "all_tokens", [1]), ("B", 3, "all_tokens", [3])])
    h.finish()
    h.assert_reconstruction()
    assert {L: h.layer_file_rows(L) for L in layer_ids} == {1: 2, 2: 0, 3: 3}


def test_selective_drain_multi_step_file_row_tracks_the_compacted_cursor(tmp_path, selective_on):
    """The Task-14 split earning its keep: layer 2 skips step 1 entirely, so on step 2 its
    `file_row` is 0 while its `logical_start` is 3 -- the two fields genuinely diverge, and the
    reader keys on the right one."""
    h = _OffLoopHarness(str(tmp_path), (1, 2), R=64)
    h.step([("A", 3, "all_tokens", [1])])
    h.step([("B", 2, "all_tokens", [1, 2])])
    h.finish()
    _, got = read_sidecar(os.path.join(str(tmp_path), "hs_ring_meta.jsonl"))
    by_key = {(e.req_id, e.layer): e for s in got for e in s.entries}
    assert by_key[("A", 1)].file_row == 0 and by_key[("A", 1)].logical_start == 0
    assert by_key[("B", 1)].file_row == 3 and by_key[("B", 1)].logical_start == 3
    assert by_key[("B", 2)].logical_start == 3
    assert by_key[("B", 2)].file_row == 0, (
        "layer 2 was never appended on step 1, so B's rows are the FIRST rows of its file")
    h.assert_reconstruction()


def test_selective_drain_leaves_an_unwanted_layer_file_empty_and_readable(tmp_path, selective_on):
    """A layer NO request ever wants is never appended, so its raw file closes at 0 bytes (the mmap
    writer pre-sizes then truncates to the real length). The reader must never open it -- an empty
    file cannot be `np.memmap`ed -- which holds because only ENTRIES name layers, and no entry names
    this one. Guards the whole selective drain against a future reader that eagerly opens every
    layer file."""
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3), R=64)
    h.step([("A", 3, "all_tokens", [1])])
    h.finish()
    for L in (2, 3):
        p = os.path.join(str(tmp_path), f"hs_layer_{L}.raw")
        assert os.path.exists(p) and os.path.getsize(p) == 0, f"layer {L} should be empty"
    h.assert_reconstruction()      # must not raise on the empty files


def test_stamp_file_row_zero_row_entry_never_crashes_the_consumer(tmp_path, selective_on):
    """An entry referencing NO rows addresses nothing, so it keeps the dense arithmetic instead of
    asking `row_offset` for a row the plan (which drops empty ranges) never covered. Unreachable
    from today's routing builders -- both size `n >= 1` -- but a fatal consumer-thread crash is a
    disproportionate response to an empty entry, and 'unreachable' is how the last defect got in."""
    ring = _mk_ring(64, 4)
    plans = build_copy_plans([ReqCaptureRecord("A", 0, 3, "all_tokens", [1])],
                             (1,), ring, 0, 3, selective=True)
    e = LayerEntry("B", 1, logical_start=90, n_rows=0, hs_mode="all_tokens")
    _stamp_file_row([e], {1: 7}, 0, plans=plans)     # 90 is nowhere near the plan's (0,3)
    assert e.file_row == 7 + 90


def test_selective_drain_end_to_end_gap_within_one_step(tmp_path, selective_on):
    """A layer wanted by two NON-ADJACENT requests in ONE step -- the case `_HETERO_STEPS` never
    produces (it only ever yields one contiguous merged range per layer per step). Layer 1 must copy
    A's rows and C's rows but NOT B's four rows in between, and both blocks must reconstruct
    byte-identically from the compacted file: C's rows sit at file row 3, not 7."""
    h = _OffLoopHarness(str(tmp_path), (1, 2), R=64)
    h.step([("A", 3, "all_tokens", [1]),
            ("B", 4, "all_tokens", [2]),      # the gap in layer 1's copy
            ("C", 2, "all_tokens", [1])])
    h.finish()
    h.assert_reconstruction()
    assert {L: h.layer_file_rows(L) for L in (1, 2)} == {1: 5, 2: 4}
    _, got = read_sidecar(os.path.join(str(tmp_path), "hs_ring_meta.jsonl"))
    by_key = {(e.req_id, e.layer): e for s in got for e in s.entries}
    assert by_key[("C", 1)].logical_start == 7
    assert by_key[("C", 1)].file_row == 3, "C's rows are the 4th..5th COPIED rows of layer 1"


def test_selective_drain_end_to_end_gap_across_a_wrap(tmp_path, selective_on):
    """The same gap case where the SECOND range also straddles the physical wrap, so layer 1's copy
    list is three physical segments for two logical ranges. Also re-checks that step 1's drained
    bytes were consumed before the engine reused those physical slots (A's block is written at
    physical 0..4 and physical 0 is overwritten by step 2's last row)."""
    h = _OffLoopHarness(str(tmp_path), (1, 2), R=8)
    h.step([("A", 4, "all_tokens", [1])])                 # logical [0,4)
    start, total = h.step([("B", 2, "all_tokens", [1]),   # logical [4,6)
                           ("C", 1, "all_tokens", [2]),   # logical [6,7)  -> layer 1's gap
                           ("D", 2, "all_tokens", [1])])  # logical [7,9)  -> wraps
    assert (start, total) == (4, 5)
    plan = build_copy_plans(
        [ReqCaptureRecord("B", 4, 2, "all_tokens", [1]), ReqCaptureRecord("D", 7, 2, "all_tokens", [1])],
        (1,), h.ring, 4, 5, selective=True)[1]
    assert plan.segments == ((4, 6), (7, 8), (0, 1)), "shape must be gap + wrap, not one of them"
    h.finish()
    h.assert_reconstruction()
    assert {L: h.layer_file_rows(L) for L in (1, 2)} == {1: 8, 2: 1}


def test_rows_copied_is_counted_at_the_copy_site_not_derived_from_the_plan(tmp_path, selective_on):
    """The brief requires the counters be REAL counts of what was copied, "not derived from the
    request's layer list". A plan-derived counter (`sum(p.total_rows for p in plans.values())`)
    agrees with the copy-site count on every workload where the copies follow the plan -- so pinning
    this needs a case where they CANNOT agree: a plan naming a layer that is not installed. There is
    no buffer for it, so the copy loop skips it; only a counter that lives at the copy site notices."""
    ring, hs_bufs, drain = _mk_offloop_drain(str(tmp_path), (1,), R=64)
    try:
        s = ring.reserve(3)
        _fill_step(hs_bufs, ring, s, 3)
        plans = {1: LayerCopyPlan.from_ranges([(s, 3)], ring),
                 99: LayerCopyPlan.from_ranges([(s, 3)], ring)}     # phantom: never installed
        assert sum(p.total_rows for p in plans.values()) == 6       # what a derived counter reads
        before = drain._rows_copied
        pieces = drain._read_segments(plans, None)
        assert [ln for ln, _ in pieces] == [1], "only installed layers can be copied"
        assert drain._rows_copied - before == 3, (
            "rows_copied must count the rows the copy loop ACTUALLY issued (3), not the plan's 6")
    finally:
        drain.close()


@pytest.mark.parametrize("armed", [False, True])
def test_rows_copied_plus_skipped_equals_the_full_drain_volume(tmp_path, monkeypatch, armed):
    """The accounting identity, in place of the `max(0, ...)` clamp that used to hide its violation:
    copied + skipped == what an unconditional full drain of the same steps would have copied. Holds
    in both arms; under a full drain `skipped` is 0."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    if armed:
        monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
    else:
        monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
    layer_ids = (1, 2, 3, 4)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=64)
    total_rows = 0
    for step in _HETERO_STEPS:
        _, n = h.step(step)
        total_rows += n
    h.finish()
    counts = h.drain.row_counts()
    full = len(layer_ids) * total_rows
    assert counts["hs.drain.rows_copied"] + counts["hs.drain.rows_skipped"] == full
    assert counts["hs.drain.rows_copied"] <= full
    if armed:
        assert counts["hs.drain.rows_skipped"] > 0
    else:
        assert counts["hs.drain.rows_skipped"] == 0
        assert counts["hs.drain.rows_copied"] == full


def test_selective_counters_are_zero_before_any_drain(tmp_path, selective_on):
    h = _OffLoopHarness(str(tmp_path), (1, 2))
    try:
        assert h.drain.row_counts()["hs.drain.rows_copied"] == 0
        assert h.drain.row_counts()["hs.drain.rows_skipped"] == 0
    finally:
        h.finish()


def test_sync_drain_counts_every_layer_as_copied(tmp_path, selective_on):
    """The sync drain full-drains (Step 3), so its counters must say so: nothing skipped."""
    ring, hs_bufs, drain = _build_sync(str(tmp_path), layer_ids=(1, 2, 3))
    s = ring.reserve(2)
    for L in (1, 2, 3):
        _write_slots(hs_bufs[L], ring, s, torch.zeros(2, 4))
    drain.record_entries([LayerEntry("A", 1, s, 2, "all_tokens")])   # only layer 1 wanted
    assert drain.drain_once() == 2
    drain.close()
    counts = drain.row_counts()
    assert counts["hs.drain.rows_copied"] == 2 * 3
    assert counts["hs.drain.rows_skipped"] == 0
    assert counts["selective"] is False


# --------------------------------------------------------------------------- #
# Step 4: the counters are reachable by RPC
# --------------------------------------------------------------------------- #

def test_get_drain_row_counts_rpc_surface(tmp_path, selective_on):
    from vllm_hook_plugins.graph.install_hs import get_drain_row_counts
    from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker

    class _FakeWorker:
        pass

    w = _FakeWorker()
    assert get_drain_row_counts(w)["hs.drain.rows_copied"] == 0   # no drain installed -> zeros

    h = _OffLoopHarness(str(tmp_path), (1, 2, 3))
    try:
        h.step([("A", 2, "all_tokens", [1])])
        w._hs_drain = h.drain
        counts = get_drain_row_counts(w)
        assert counts["hs.drain.rows_copied"] == 2
        assert counts["hs.drain.rows_skipped"] == 4
        assert counts["selective"] is True
        # The worker method must delegate to it (that is the collective_rpc channel Task 16 reads).
        assert callable(getattr(ProbeHiddenStatesWorker, "get_drain_row_counts", None))
        assert ProbeHiddenStatesWorker.get_drain_row_counts(w) == counts
    finally:
        h.finish()


# =========================================================================== #
# TASK 19 -- THE DEGENERATE ALL-LAYERS FAST PATH
#
# When every request wants every installed layer over the whole step span, the selective branch
# builds -- at O(records x layers) interpreter cost plus one sort/merge per layer, on the GIL-holding
# consumer thread -- a plan PROVABLY IDENTICAL to the `selective=False` one-liner. Measured on this
# box at bs128 x 32 layers: 1.40 ms/step for the plan build plus 0.55 ms/step for the `row_offset`
# stamping it forces, ~1.95 ms/step of pure bookkeeping that buys nothing. Task 19 detects that case
# in ONE O(records) walk and falls through to the flag-off code path.
#
# The tests below pin BOTH halves of the claim, because either alone is worthless:
#   * EQUALITY -- the fast path's plans equal the slow path's (that was already true and stays true);
#   * THAT IT ACTUALLY FIRED -- witnessed structurally by the SHARED plan object (the slow path
#     builds one LayerCopyPlan per layer and can never return the same object for two layers) and by
#     the drain's `hs.drain.degenerate_steps` counter. Without this half an equality test passes
#     just as happily on a fast path that never runs.
#   * THE SPAN BOUND -- detection walks a cursor from `start_logical` and demands each record start
#     exactly at it, ending exactly at `start_logical + n_rows`, so an out-of-span record can never
#     reach the fast path; it falls through to the slow path, which raises. The bound is therefore
#     enforced twice (refused at detection, and structural in the emitted whole-span plan).
# =========================================================================== #

def _deg(records, layers, lo, n_rows):
    return rdh.is_degenerate_full_step(records, {int(L) for L in layers}, lo, n_rows)


def _all_share_one_plan(plans):
    """The observable signature of the fast path: ONE plan object handed to every layer. The
    selective slow path constructs a fresh `LayerCopyPlan` per layer, so it can never do this."""
    vals = list(plans.values())
    return bool(vals) and all(p is vals[0] for p in vals)


def test_degenerate_all_layers_takes_the_shared_whole_span_plan():
    """RED before Task 19: the selective branch built one plan per layer even here."""
    ring = _mk_ring(64, 4)
    layers = (1, 2, 3, 4)
    records = [ReqCaptureRecord("A", 10, 3, "all_tokens", list(layers)),
               ReqCaptureRecord("B", 13, 2, "all_tokens", list(layers))]
    plans = build_copy_plans(records, layers, ring, 10, 5, selective=True)
    assert _all_share_one_plan(plans), "degenerate step must reuse the flag-off whole-span plan"
    assert plans == build_copy_plans(records, layers, ring, 10, 5, selective=False)


@pytest.mark.parametrize("R,layers,steps", [
    # (ring slots, installed layers, [ (start, [n_rows per request]) ])
    (64, (1,), [(0, [4])]),                               # one layer, one request
    (64, (1, 2, 3), [(0, [1, 1, 1, 1])]),                 # decode shape: 4 reqs x 1 row
    (64, (1, 2, 3, 4), [(10, [3, 2])]),                   # mixed prefill widths
    (64, tuple(range(1, 33)), [(7, [1] * 16)]),           # 32 layers x 16 requests
    (8, (1, 2), [(6, [4])]),                              # straddles the physical wrap
    (8, (1, 2, 3), [(5, [2, 2])]),                        # two requests across the wrap
    (64, (5, 9, 30), [(0, [2, 3, 1])]),                   # non-contiguous layer numbering
])
def test_degenerate_fast_path_plans_equal_the_slow_path_and_actually_fire(R, layers, steps):
    """THE PROPERTY. Over many shapes the fast path must (a) fire and (b) produce a dict EQUAL to
    what the record-driven slow path produces. (b) alone cannot fail on a fast path that never runs,
    which is why (a) is asserted in the same test."""
    for lo, ns in steps:
        ring = _mk_ring(R, 4)
        recs = []
        s = lo
        for i, n in enumerate(ns):
            recs.append(ReqCaptureRecord(f"r{i}", s, n, "all_tokens", list(layers)))
            s += n
        total = s - lo
        fast = build_copy_plans(recs, layers, ring, lo, total, selective=True)
        assert _all_share_one_plan(fast), f"fast path did not fire for {layers} {ns}"
        assert _deg(recs, layers, lo, total) is True
        # The slow path, forced: same records, but one layer removed from the installed set means
        # the degenerate test is false, so this is the record-driven build for the layers that
        # remain. Compare it cell by cell against the fast plan.
        slow_like = {ln: LayerCopyPlan.from_ranges([(r.logical_start, r.n_rows) for r in recs], ring)
                     for ln in layers}
        assert fast == slow_like
        assert fast == build_copy_plans(recs, layers, ring, lo, total, selective=False)


@pytest.mark.parametrize("desc,records,layers,lo,n_rows", [
    ("subset layer set", [ReqCaptureRecord("A", 0, 4, "all_tokens", [1, 2])], (1, 2, 3), 0, 4),
    ("heterogeneous sets", [ReqCaptureRecord("A", 0, 2, "all_tokens", [1, 2]),
                            ReqCaptureRecord("B", 2, 2, "all_tokens", [2, 3])], (1, 2, 3), 0, 4),
    ("gap between records", [ReqCaptureRecord("A", 0, 2, "all_tokens", [1]),
                             ReqCaptureRecord("B", 3, 1, "all_tokens", [1])], (1,), 0, 4),
    ("records out of order", [ReqCaptureRecord("A", 2, 2, "all_tokens", [1]),
                              ReqCaptureRecord("B", 0, 2, "all_tokens", [1])], (1,), 0, 4),
    ("does not reach the end", [ReqCaptureRecord("A", 0, 2, "all_tokens", [1])], (1,), 0, 4),
    ("starts before the span", [ReqCaptureRecord("A", 8, 2, "all_tokens", [1])], (1,), 10, 5),
    ("reaches past the span", [ReqCaptureRecord("A", 10, 20, "all_tokens", [1])], (1,), 10, 5),
    ("zero-row record", [ReqCaptureRecord("A", 0, 0, "all_tokens", [1]),
                         ReqCaptureRecord("B", 0, 4, "all_tokens", [1])], (1,), 0, 4),
    ("right count, wrong layers", [ReqCaptureRecord("A", 0, 4, "all_tokens", [1, 9])], (1, 2), 0, 4),
    ("right count, duplicated layer", [ReqCaptureRecord("A", 0, 4, "all_tokens", [1, 1])],
     (1, 2), 0, 4),
    ("no records at all", [], (1, 2), 0, 4),
    ("flat LayerEntry input", [LayerEntry("A", 1, 0, 4, "all_tokens")], (1,), 0, 4),
    ("no layers installed", [ReqCaptureRecord("A", 0, 4, "all_tokens", [])], (), 0, 4),
])
def test_degenerate_detection_is_false_on_everything_else(desc, records, layers, lo, n_rows):
    """A false POSITIVE would silently full-drain a workload that could have skipped rows (the
    lever's whole value); the detection is exact, and conservatively false whenever it cannot cheaply
    prove the degenerate shape."""
    assert _deg(records, layers, lo, n_rows) is False, desc


def test_degenerate_detection_accepts_only_records_inside_the_span():
    """The span bound, as a property of the DETECTION rather than of the plan: anything the fast path
    accepts is provably contained in [lo, lo+n_rows), because the walk starts at lo, requires each
    record to start exactly at the running cursor, and requires the cursor to end exactly at hi."""
    lo, n_rows = 10, 6
    for ns in ([6], [1, 5], [2, 2, 2], [1, 1, 1, 1, 1, 1]):
        s = lo
        recs = []
        for i, n in enumerate(ns):
            recs.append(ReqCaptureRecord(f"r{i}", s, n, "all_tokens", [1]))
            s += n
        assert _deg(recs, (1,), lo, n_rows) is True
        for r in recs:
            assert lo <= r.logical_start and r.logical_start + r.n_rows <= lo + n_rows


def test_fast_path_cannot_bypass_the_out_of_span_raise():
    """A record reaching past the step must still RAISE, not slip through a cheaper path."""
    ring = _mk_ring(64, 4)
    bad = [ReqCaptureRecord("A", 10, 3, "all_tokens", [1]),
           ReqCaptureRecord("B", 13, 4, "all_tokens", [1])]      # ends at 17, span ends at 15
    assert _deg(bad, (1,), 10, 5) is False
    with pytest.raises(ValueError, match="outside this step"):
        build_copy_plans(bad, (1,), ring, 10, 5, selective=True)


def test_detection_is_pure_and_leaves_the_records_alone():
    ring = _mk_ring(64, 4)
    layers = (1, 2)
    recs = [ReqCaptureRecord("A", 0, 3, "all_tokens", list(layers))]
    before = [(r.req_id, r.logical_start, r.n_rows, list(r.layers)) for r in recs]
    build_copy_plans(recs, layers, ring, 0, 3, selective=True)
    assert [(r.req_id, r.logical_start, r.n_rows, list(r.layers)) for r in recs] == before


# --- the counter (the non-vacuity witness the GPU A/B reads over the RPC) ------------------- #

def test_degenerate_steps_counter_counts_the_all_layers_steps(tmp_path, selective_on):
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3), R=64)
    try:
        h.step([("A", 3, "all_tokens", [1, 2, 3]), ("B", 1, "last_token", [1, 2, 3])])
        h.step([("C", 2, "all_tokens", [1, 2, 3])])
        counts = h.drain.row_counts()
        assert counts["hs.drain.degenerate_steps"] == 2
        assert counts["hs.drain.rows_skipped"] == 0
        assert counts["selective"] is True
    finally:
        h.finish()
    h.assert_reconstruction()


def test_degenerate_steps_counter_stays_zero_on_a_subset_workload(tmp_path, selective_on):
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3), R=64)
    try:
        h.step([("A", 3, "all_tokens", [1])])
        counts = h.drain.row_counts()
        assert counts["hs.drain.degenerate_steps"] == 0
        assert counts["hs.drain.rows_skipped"] > 0
    finally:
        h.finish()
    h.assert_reconstruction()


def test_degenerate_steps_counter_stays_zero_when_the_flag_is_off(tmp_path, selective_off):
    """Flag OFF is already the whole-span path; the counter means "the ARMED lever found nothing to
    skip", so it must not tick for a drain that was never selective in the first place."""
    h = _OffLoopHarness(str(tmp_path), (1, 2), R=64)
    try:
        h.step([("A", 3, "all_tokens", [1, 2])])
        assert h.drain.row_counts()["hs.drain.degenerate_steps"] == 0
    finally:
        h.finish()


def test_degenerate_steps_is_in_the_rpc_surface_when_no_drain_is_installed():
    from vllm_hook_plugins.graph.install_hs import get_drain_row_counts

    class _FakeWorker:
        pass

    assert get_drain_row_counts(_FakeWorker())["hs.drain.degenerate_steps"] == 0


# --- the stamping half: a degenerate step must take the DENSE arithmetic ---------------------- #

def test_degenerate_step_stamps_with_the_dense_arithmetic_not_row_offset(tmp_path, selective_on,
                                                                         monkeypatch):
    """`_stamp_file_row(plans=...)` costs a `plans.get` + a `bisect` per (request x layer) entry --
    0.55 ms/step at bs128 x 32L on this box -- to compute exactly `logical_start - step_start`, which
    is what `plans=None` computes directly. On a degenerate step the consumer must pass None. Pinned
    by spying on the call, because the RESULT is identical either way (that is the point) and no
    value assertion could ever tell the two apart."""
    seen = []
    real = rdh._stamp_file_row

    def _spy(entries, cursor_before, step_start_logical, plans=None):
        seen.append(plans)
        return real(entries, cursor_before, step_start_logical, plans=plans)

    monkeypatch.setattr(rdh, "_stamp_file_row", _spy)
    h = _OffLoopHarness(str(tmp_path), (1, 2, 3), R=64)
    try:
        h.step([("A", 2, "all_tokens", [1, 2, 3])])          # degenerate  -> plans must be None
        h.step([("B", 2, "all_tokens", [1])])                # subset      -> plans must be passed
    finally:
        h.finish()
    assert len(seen) == 2, seen
    assert seen[0] is None, "degenerate step must stamp with the dense arithmetic"
    assert seen[1] is not None, "a compacted step still needs the plan's row_offset"
    h.assert_reconstruction()


def test_mixed_compacted_then_degenerate_steps_reconstruct_byte_identically(tmp_path, selective_on):
    """The interaction the fast path could plausibly break: after a COMPACTED step the per-layer file
    cursors have diverged, and the next (degenerate) step stamps with `cursor_before[ln] +
    (logical_start - step_start)`. That is still each layer's true file position -- but only because
    the cursor is per layer. Drive both step kinds in one run and read the artifact back."""
    layer_ids = (1, 2, 3)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=64)
    try:
        h.step([("A", 3, "all_tokens", [2])])                       # compacted: only layer 2
        h.step([("B", 2, "all_tokens", list(layer_ids)),            # degenerate
                ("C", 1, "last_token", list(layer_ids))])
        h.step([("D", 2, "all_tokens", [1, 3])])                    # compacted again
        h.step([("E", 1, "all_tokens", list(layer_ids))])           # degenerate again
        counts = h.drain.row_counts()
        assert counts["hs.drain.degenerate_steps"] == 2
        assert counts["hs.drain.rows_skipped"] > 0
    finally:
        h.finish()
    h.assert_reconstruction()
    # layer 2: 3 (step 1) + 3 (degenerate span B+C) + 0 + 1 = 7; layers 1/3: 0+3+2+1 = 6.
    assert {L: h.layer_file_rows(L) for L in layer_ids} == {1: 6, 2: 7, 3: 6}


def test_degenerate_fast_path_end_to_end_matches_the_unarmed_drain(tmp_path, monkeypatch):
    """The Task 15 no-op contract, re-run now that the degenerate case takes a DIFFERENT code path:
    identical sidecar (file_row included) and identical file sizes, armed vs unarmed."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP_BYTES", str(1 << 20))
    layer_ids = (1, 2, 3, 4)
    steps = [[("A", 3, "all_tokens", list(layer_ids)), ("B", 1, "last_token", list(layer_ids))],
             [("C", 2, "all_tokens", list(layer_ids))],
             [("D", 1, "all_tokens", list(layer_ids)), ("E", 1, "all_tokens", list(layer_ids))]]
    sidecars, sizes, degen = {}, {}, {}
    for armed in (False, True):
        if armed:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "1")
        else:
            monkeypatch.setenv("VLLM_HOOK_DRAIN_SELECTIVE", "0")
        d = str(tmp_path / f"deg19_{int(armed)}")
        os.makedirs(d, exist_ok=True)
        h = _OffLoopHarness(d, layer_ids, R=64)
        for step in steps:
            h.step(step)
        degen[armed] = h.drain.row_counts()["hs.drain.degenerate_steps"]
        h.finish()
        h.assert_reconstruction()
        _, got = read_sidecar(os.path.join(d, "hs_ring_meta.jsonl"))
        sidecars[armed] = [(e.req_id, e.layer, e.logical_start, e.n_rows, e.hs_mode, e.file_row)
                           for s in got for e in s.entries]
        sizes[armed] = {L: h.layer_file_rows(L) for L in layer_ids}
    assert degen == {False: 0, True: 3}, "every step here is degenerate under the armed drain"
    assert sidecars[True] == sidecars[False]
    assert sizes[True] == sizes[False]


# =========================================================================== #
# FINAL-REVIEW FIX I1 -- record_captured_cells: the `captured.bytes.hs` PROF gauge multiplier
# (install_hs.py). The gauge used to read `_cap_rows * len(layers)` (installed layer count)
# unconditionally, which was the actual bytes the drain wrote to NVMe only while every drain copied
# every installed layer -- true before Task 19, false on any subset workload once selective drain
# defaulted ON. `record_captured_cells` sums each per-request record's own
# `n_rows * len(rec.layers)` in O(records), the same quantity a selective drain actually copies,
# without repaying `build_copy_plans`' O(records x layers) merge on the caller's (engine) loop.
# =========================================================================== #

def test_record_captured_cells_sums_per_record_rows_times_layers():
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 2]),
               ReqCaptureRecord("B", 3, 2, "last_token", [3])]
    assert record_captured_cells(records) == 3 * 2 + 2 * 1


def test_record_captured_cells_skips_zero_row_records():
    records = [ReqCaptureRecord("A", 0, 0, "all_tokens", [1, 2, 3]),
               ReqCaptureRecord("B", 0, 4, "all_tokens", [1])]
    assert record_captured_cells(records) == 4


def test_record_captured_cells_empty_input_is_zero():
    assert record_captured_cells([]) == 0


def test_record_captured_cells_flat_layerentry_counts_one_cell_each():
    """A caller that hands already-flat `LayerEntry`s (no `.layers` attribute) gets one cell per
    entry -- mirrors `expand_records`' fan-out (one LayerEntry IS one (req, layer) cell)."""
    entries = [LayerEntry("A", 1, 0, 3, "all_tokens"), LayerEntry("A", 2, 0, 3, "all_tokens")]
    assert record_captured_cells(entries) == 3 + 3


@pytest.mark.parametrize("desc,records,layers,lo,n_rows", [
    ("heterogeneous touching",
     [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 2]),
      ReqCaptureRecord("B", 3, 2, "all_tokens", [2, 3])], (1, 2, 3), 0, 5),
    ("gap within one layer",
     [ReqCaptureRecord("A", 0, 2, "all_tokens", [1]),
      ReqCaptureRecord("B", 3, 4, "all_tokens", [5]),
      ReqCaptureRecord("C", 7, 2, "all_tokens", [1])], (1, 5), 0, 9),
    ("degenerate all-layers",
     [ReqCaptureRecord("A", 10, 3, "all_tokens", [1, 2, 3, 4]),
      ReqCaptureRecord("B", 13, 2, "all_tokens", [1, 2, 3, 4])], (1, 2, 3, 4), 10, 5),
    ("single request subset",
     [ReqCaptureRecord("A", 0, 3, "all_tokens", [1, 3])], (1, 2, 3, 4), 0, 3),
])
def test_record_captured_cells_equals_the_copy_plans_total(desc, records, layers, lo, n_rows):
    """The equivalence the gauge fix relies on: summing per-record cells must agree with the ACTUAL
    per-layer union `build_copy_plans` computes -- on subset/heterogeneous/gapped shapes AND on the
    degenerate all-layers one, where the fast path takes a completely different code path to arrive
    at (provably) the same total."""
    ring = _mk_ring(64, 4)
    plans = build_copy_plans(records, layers, ring, lo, n_rows, selective=True)
    assert record_captured_cells(records) == sum(p.total_rows for p in plans.values()), desc


def test_record_captured_cells_matches_the_drains_own_rows_copied_counter(tmp_path, selective_on):
    """End-to-end: sum `record_captured_cells` per step and compare it against what the REAL
    off-loop drain reports it actually copied (`hs.drain.rows_copied`), on the same heterogeneous
    multi-step workload `test_selective_drain_copies_only_the_wanted_tiles_and_reconstructs_
    identically` already uses -- so this is checked against BOTH the drain's own counter and (via
    that other test) an independently-computed per-layer total."""
    layer_ids = (1, 2, 3, 4)
    h = _OffLoopHarness(str(tmp_path), layer_ids, R=64)
    predicted = 0
    for step in _HETERO_STEPS:
        for (_rid, n, _mode, req_layers) in step:
            predicted += n * len(req_layers)
        h.step(step)
    h.finish()
    assert predicted > 0, "vacuous predicted total"
    assert h.drain.row_counts()["hs.drain.rows_copied"] == predicted


def test_record_captured_cells_ignores_the_selective_off_or_refused_case_by_contract():
    """Documents the caller's obligation (install_hs.py gates on `_selective_active()` before
    calling this): when the drain is NOT running selectively, it copies every installed layer
    regardless of what any record names, so `record_captured_cells` -- which only sums what the
    records THEMSELVES claim -- reads LOWER than the real copied volume on a subset workload. This
    is not a bug in the function; it is why the gauge fix in install_hs.py never calls it unguarded."""
    records = [ReqCaptureRecord("A", 0, 3, "all_tokens", [1])]      # wants 1 of 4 installed layers
    installed = (1, 2, 3, 4)
    real_full_drain_cells = 3 * len(installed)                      # what selective=False copies
    assert record_captured_cells(records) < real_full_drain_cells


# --- standalone-runner scaffolding (mirrors the sibling ring test files' pattern) ---
def main():
    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
