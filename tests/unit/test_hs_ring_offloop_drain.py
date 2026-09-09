"""No-GPU tests for the OFF-LOOP (consumer-thread) HS capture-ring drain (Task A).

The engine loop enqueues each step's (entries, start_logical, n_rows, event) O(1); a dedicated
consumer THREAD waits the scatter event, D2Hs the step's ring rows, writes per-layer raw + sidecar,
and advance_drain()s (frees ring rows -> releases reserve backpressure). On CPU (device="cpu")
streams/events are no-ops, so this drives the FULL consumer/queue/backpressure machinery without a
GPU.

Covers:
  * byte-identical round-trip through the off-loop consumer (float32 + bfloat16), reconstructed via
    load_multilayer_ring_artifact — the correctness crux (the consumer reads only committed rows);
  * a GENUINE physical wrap driven through the reserve-BLOCK backpressure with a live consumer
    (block-then-succeed, never drop);
  * the reserve-block plumbing: blocks until a background drain frees rows (not a drop), and RAISES
    RingBackpressureError fast on a DEAD consumer;
  * flush joins the thread + writes the sidecar (never loses a row); a consumer error surfaces LOUD;
  * the shared install_prepare_inputs_routing wrapper RE-RAISES RingBackpressureError (does not
    swallow it into a silent drop) — the assumption-6 fix.

Run:  conda activate vllm_hook_env && python tests/unit/test_hs_ring_offloop_drain.py
      (or under pytest: pytest tests/unit/test_hs_ring_offloop_drain.py)
"""
import os
import sys
import tempfile
import threading
import time
import types

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_drain_hs import (
    OffLoopRingDrain, _torch_dtype_name)
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block, _build_routing_hs


def _mkdtmp(tag):
    return tempfile.mkdtemp(prefix=f"hsoffloop_{tag}_")


def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


def _build(R, hidden, layer_ids, dtype, tmp):
    # Pin the drain to FULL (non-selective) mode. This file exercises the OFF-LOOP CONSUMER
    # THREAD machinery (backpressure, physical wrap, byte-identical reconstruction) that predates
    # Lever C, and `_engine_step` below feeds flat `LayerEntry`s (no per-request `.layers`), so
    # under the post-Task-19 default (VLLM_HOOK_DRAIN_SELECTIVE unset -> ON) `is_degenerate_full_
    # step` never fires for them and every drain here would silently take the record-driven
    # (selective) branch instead of the whole-span one this file was written to cover -- the output
    # stays byte-identical either way (Task 15's contract), so nothing here would fail, but the
    # code path under test would have quietly changed. `setdefault` only takes effect when nothing
    # else in this process already set the var. Lever C's OWN coverage (selective on vs off,
    # subset/heterogeneous/degenerate shapes) lives in test_drain_selective.py.
    os.environ.setdefault("VLLM_HOOK_DRAIN_SELECTIVE", "0")
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    drain = OffLoopRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header)
    return ring, hs_bufs, drain


def _engine_step(ring, hs_bufs, drain, reqs, expected, dtype, consumer=None):
    """One engine step. reqs = [(req_id, n_rows, [layers], mode)]. Reserves (blocking when
    `consumer` given), writes deterministic data into each layer's ring slots, records LayerEntrys,
    and O(1)-enqueues. Records expected data per (req, layer) for the reconstruction assert."""
    entries = []
    start_logical = None
    total = 0
    for (rid, n, layers, mode) in reqs:
        s = _ring_reserve_or_block(ring, n, consumer)
        if start_logical is None:
            start_logical = s
        total += n
        phys = ring.physical_slots(s, n)
        for L in layers:
            hidden = hs_bufs[L].shape[1]
            # unique per (rid, L, row) so a cross-layer / cross-request leak is caught
            base = (hash((rid, L)) % 997) * 1000
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)
            for j, p in enumerate(phys):
                hs_bufs[L][p] = data[j]
            entries.append(LayerEntry(str(rid), L, s, n, mode))
            expected.setdefault(str(rid), {})[L] = data
    drain.enqueue(entries, start_logical, total, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0, f"consumer did not drain: pending={ring.pending_rows()}"


def _assert_reconstructs(tmp, expected):
    out = load_multilayer_ring_artifact(tmp)
    for rid, layers in expected.items():
        for L, data in layers.items():
            assert rid in out and L in out[rid], f"missing {rid}/{L} in reconstruction"
            assert torch.equal(out[rid][L], data), f"{rid}/{L} mismatch"
    # no leak into non-requested layers
    for rid, layers in out.items():
        for L in layers:
            assert L in expected.get(rid, {}), f"leaked {rid}/{L}"


# --------------------------- (1) byte-identical round-trip ---------------------------
def _roundtrip(dtype, tag):
    tmp = _mkdtmp(tag)
    hidden, R, layer_ids = 4, 512, (1, 2, 3)   # ring big enough that no wrap/block occurs
    ring, hs_bufs, drain = _build(R, hidden, layer_ids, dtype, tmp)
    drain.start()
    expected = {}
    # step 1: A all_tokens (3 rows) on all layers + B last_token (1 row) on layer 2 only
    _engine_step(ring, hs_bufs, drain,
                 [("A", 3, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], expected, dtype)
    # step 2: A decode all_tokens (1 row) on all layers
    _engine_step(ring, hs_bufs, drain,
                 [("A2", 1, [1, 2, 3], "all_tokens")], expected, dtype)
    # step 3: C all_tokens (5 rows) on layers {1,3}
    _engine_step(ring, hs_bufs, drain,
                 [("C", 5, [1, 3], "all_tokens")], expected, dtype)
    _wait_drained(ring)
    drain.stop()          # join the consumer thread
    drain.close()         # write the sidecar
    assert not drain.is_alive()
    _assert_reconstructs(tmp, expected)


def test_offloop_roundtrip_float32():
    _roundtrip(torch.float32, "f32")


def test_offloop_roundtrip_bfloat16():
    _roundtrip(torch.bfloat16, "bf16")


# --------------------------- (2) wrap through backpressure ---------------------------
def test_offloop_wrap_with_backpressure():
    """Tiny ring forces PHYSICAL WRAP across steps; the reserve blocks on a full ring and the live
    consumer frees rows (advance_drain) so it succeeds — never a drop — and every step reconstructs
    byte-identically. This is the concurrency crux: read-behind-write-cursor + reserve-block."""
    tmp = _mkdtmp("wrap")
    hidden, R, dtype = 4, 4, torch.float32     # R=4 rows/layer -> repeated wraps
    ring, hs_bufs, drain = _build(R, hidden, (1,), dtype, tmp)
    drain.start()
    expected = {}
    # 8 steps of 2-3 rows each on a 4-row ring => many wraps + real blocking (consumer must free).
    plan = [3, 2, 3, 2, 3, 2, 3, 2]
    for i, n in enumerate(plan):
        _engine_step(ring, hs_bufs, drain, [(f"R{i}", n, [1], "all_tokens")], expected, dtype,
                     consumer=drain)
    _wait_drained(ring)
    drain.stop()
    drain.close()
    _assert_reconstructs(tmp, expected)


# --------------------------- (3) reserve-block plumbing ---------------------------
def test_reserve_block_blocks_then_succeeds_not_drop():
    """A full ring makes reserve BLOCK (poll) until a background advance_drain frees rows — it
    returns the freed slot, never None/drop. Proves the never-drop block directly on the helper."""
    ring = GpuCaptureRing(row_bytes=8, n_slots=4, device="cpu", dtype=torch.float32, row_shape=(4,))
    assert ring.reserve(4) == 0                 # fill it
    assert ring.reserve(1) is None              # genuinely full

    def _free_later():
        time.sleep(0.05)
        ring.advance_drain(2)                   # a "consumer" drains 2 rows

    t = threading.Thread(target=_free_later)
    t.start()
    t0 = time.monotonic()
    slot = _ring_reserve_or_block(ring, 1)      # must BLOCK ~0.05s, then succeed
    dt = time.monotonic() - t0
    t.join()
    assert slot == 4, f"blocked reserve returned wrong slot {slot}"
    assert dt >= 0.03, f"reserve did not actually block (dt={dt:.3f}s)"


def test_reserve_block_raises_fast_on_dead_consumer():
    """A full ring + a DEAD off-loop consumer must raise RingBackpressureError FAST (is_alive()
    backstop), not hang the whole timeout — never a silent drop."""
    tmp = _mkdtmp("dead")
    hidden, R, dtype = 4, 4, torch.float32
    ring, hs_bufs, drain = _build(R, hidden, (1,), dtype, tmp)
    # drain NOT started -> is_alive() False == a dead consumer.
    assert not drain.is_alive()
    assert ring.reserve(4) == 0                 # fill the ring
    os.environ["VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S"] = "5"   # long, to prove we DON'T wait it
    try:
        t0 = time.monotonic()
        raised = False
        try:
            _ring_reserve_or_block(ring, 1, drain)
        except RingBackpressureError:
            raised = True
        dt = time.monotonic() - t0
        assert raised, "dead consumer + full ring must raise RingBackpressureError, not drop/hang"
        assert dt < 1.0, f"did not fail fast on dead consumer (waited {dt:.2f}s)"
        assert ring._write == 4, "cursor moved despite refusal"
    finally:
        os.environ.pop("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", None)


# --------------------------- (4) flush + error surfacing ---------------------------
def test_flush_joins_thread_and_writes_sidecar():
    tmp = _mkdtmp("flush")
    hidden, R, dtype = 4, 128, torch.float32
    ring, hs_bufs, drain = _build(R, hidden, (1, 2), dtype, tmp)
    drain.start()
    expected = {}
    _engine_step(ring, hs_bufs, drain, [("A", 2, [1, 2], "all_tokens")], expected, dtype)
    _wait_drained(ring)
    drain.stop()
    drain.close()
    assert not drain.is_alive(), "consumer thread not joined by stop()"
    assert os.path.exists(os.path.join(tmp, "hs_ring_meta.jsonl")), "sidecar not written"
    _assert_reconstructs(tmp, expected)
    drain.stop()   # idempotent second flush must not raise


def test_consumer_error_surfaces_loud_at_flush():
    """A failure inside the consumer must not be swallowed: the thread records it, dies, and stop()
    (flush_ring) re-raises it — never a silent partial capture."""
    tmp = _mkdtmp("err")
    hidden, R, dtype = 4, 128, torch.float32
    ring, hs_bufs, drain = _build(R, hidden, (1,), dtype, tmp)

    def _boom(*a, **k):
        raise RuntimeError("injected drain failure")

    drain._append_layer_rows = _boom      # sabotage the file write on the consumer thread
    drain.start()
    expected = {}
    _engine_step(ring, hs_bufs, drain, [("A", 2, [1], "all_tokens")], expected, dtype)
    # the consumer thread should die on the injected error
    deadline = time.monotonic() + 5.0
    while drain.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not drain.is_alive(), "consumer thread should have died on the injected error"
    raised = False
    try:
        drain.stop()
    except RuntimeError:
        raised = True
    assert raised, "stop()/flush_ring must RE-RAISE a consumer-thread error (never silent)"
    assert drain.error is not None


# --------------------------- (5) shared wrapper re-raises (assumption-6 fix) ---------------------------
def _make_ring_registry(num_layers, cap, R, device="cpu"):
    reg = HostRegistry(num_layers=num_layers, cap=cap, device=device, should_capture=True)
    ring = GpuCaptureRing(row_bytes=8, n_slots=R, device=device,
                          dtype=torch.float32, row_shape=(4,))
    reg._hs_ring = ring
    reg._hs_step_entries = []
    reg.sentinel_row = ring.SENTINEL
    reg.incremental_enabled = False
    reg.gpu_routing = False
    reg.capture_index_all.fill_(ring.SENTINEL)
    for slot in reg._ring.slots:
        slot["capture_index"].fill_(ring.SENTINEL)
    return reg, ring


def _fake_runner(req_id, ntok, extra):
    class _QSL:
        def __init__(self, n):
            import numpy as np
            self.np = np.array([0, n], dtype=np.int64)

    class _SP:
        def __init__(self, e):
            self.extra_args = e

    class _RS:
        def __init__(self, e):
            self.sampling_params = _SP(e)
            self.output_token_ids = ()

    mr = types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=[req_id]),
        query_start_loc=_QSL(ntok),
        requests={req_id: _RS(extra)},
        _default_hooks_on="both",
        _worker_hs_mode="all_tokens",
    )
    mr._prepare_inputs = lambda scheduler_output, *a, **k: None   # orig prep
    return mr


def test_shared_routing_wrapper_reraises_backpressure():
    """install_prepare_inputs_routing must PROPAGATE RingBackpressureError (assumption-6 fix), not
    catch-and-clear it into `_pending_plans = []` (a silent drop of a capturing request)."""
    from vllm_hook_plugins.graph.install import install_prepare_inputs_routing
    extra = {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"}

    os.environ["VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S"] = "0"
    try:
        # --- full ring -> the reserve inside _build_routing_hs raises; wrapper must re-raise ---
        reg, ring = _make_ring_registry(num_layers=2, cap=16, R=4)
        assert ring.reserve(4) == 0                       # fill the ring
        mr = _fake_runner("A", 2, extra)
        worker = types.SimpleNamespace(_graph_registry=reg)
        install_prepare_inputs_routing(mr, worker, _build_routing_hs, label="hs",
                                       routing_key_fn=lambda r, g, q: None)  # never skip -> always build
        raised = False
        try:
            mr._prepare_inputs("sched")
        except RingBackpressureError:
            raised = True
        assert raised, "shared wrapper SWALLOWED RingBackpressureError (never-drop violated)"

        # --- control: room in the ring -> no raise, plans built (re-raise is specific) ---
        reg2, ring2 = _make_ring_registry(num_layers=2, cap=16, R=16)
        mr2 = _fake_runner("A", 2, extra)
        worker2 = types.SimpleNamespace(_graph_registry=reg2)
        install_prepare_inputs_routing(mr2, worker2, _build_routing_hs, label="hs2",
                                       routing_key_fn=lambda r, g, q: None)
        mr2._prepare_inputs("sched")                      # must NOT raise
        assert reg2._pending_plans, "expected plans when the ring has room"
        assert ring2._write == 2, "control reserve did not advance the cursor"
    finally:
        os.environ.pop("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", None)


# --------------------------- (6) per-request demux (Task 5) ---------------------------
def _pr_build(R, hidden, layer_ids, dtype, tmp):
    """Like `_build` but arms the OFF-LOOP drain's per_request mode: the consumer demuxes each
    step's contiguous drained rows by req_id into a PerRequestIndex instead of shared per-layer
    files. Returns (ring, hs_bufs, drain, index)."""
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    index = PerRequestIndex()
    drain = OffLoopRingDrain(ring, [(L, hs_bufs[L]) for L in layer_ids], tmp, header,
                             per_request=True, index=index)
    return ring, hs_bufs, drain, index


def _pr_step(ring, hs_bufs, drain, reqs, expected_blocks, dtype, step_tag, consumer=None):
    """One engine step for the per-request demux path. reqs = [(req_id, n_rows, [layers], mode)].
    Writes STEP-UNIQUE deterministic data into each layer's reserved ring slots (so a mis-ordered or
    cross-request/cross-layer concat is caught), records LayerEntrys, O(1)-enqueues, and appends each
    (req, layer)'s block to expected_blocks[req][layer] in APPEND (step) order — the order the index
    reconstructs in."""
    entries = []
    start_logical = None
    total = 0
    for (rid, n, layers, mode) in reqs:
        s = _ring_reserve_or_block(ring, n, consumer)
        if start_logical is None:
            start_logical = s
        total += n
        phys = ring.physical_slots(s, n)
        for L in layers:
            hidden = hs_bufs[L].shape[1]
            base = (hash((rid, L, step_tag)) % 997) * 1000 + 1   # step-unique, non-zero
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)
            for j, p in enumerate(phys):
                hs_bufs[L][p] = data[j]
            entries.append(LayerEntry(str(rid), L, s, n, mode))
            expected_blocks.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, start_logical, total, None)


def _assert_pr_reconstructs(popped, expected_blocks):
    """popped = list[(req_id, {layer: assembled})] from pop_deliverable; assert byte-identical to a
    hand-built cat of each (req, layer)'s per-step blocks in append order, and that exactly the
    expected req/layer set is present (no leak, no drop)."""
    got = {rid: layers for rid, layers in popped}
    assert set(got) == set(expected_blocks), (
        f"delivered req set {set(got)} != expected {set(expected_blocks)}")
    for rid, layers in expected_blocks.items():
        assert set(got[rid]) == set(layers), (
            f"{rid}: delivered layers {set(got[rid])} != expected {set(layers)}")
        for L, blocks in layers.items():
            want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
            assert torch.equal(got[rid][L], want), f"{rid}/{L} byte mismatch"


def test_offloop_per_request_two_interleaved():
    """Two requests A and B captured with their rows INTERLEAVED across steps; enqueue each one's
    FINISH after its last rows. pop_deliverable must reconstruct each byte-identically, and a
    finished+delivered+freed request must leave live_req_ids() correct."""
    tmp = _mkdtmp("pr_interleave")
    hidden, R, dtype = 4, 512, torch.float32      # roomy ring: no wrap, isolate the demux logic
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _pr_build(R, hidden, layer_ids, dtype, tmp)
    drain.start()
    exp = {}
    # A: all_tokens over {1,2,3}; B: last_token over {2} only — interleaved every step.
    _pr_step(ring, hs_bufs, drain,
             [("A", 3, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain,
             [("A", 1, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s2")
    _pr_step(ring, hs_bufs, drain,
             [("A", 1, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s3")
    # A finishes here (its rows above are all enqueued -> FIFO holds); B keeps going one more step.
    drain.enqueue_finish("A")
    _pr_step(ring, hs_bufs, drain, [("B", 1, [2], "last_token")], exp, dtype, "s4")
    drain.enqueue_finish("B")
    _wait_drained(ring)
    drain.stop()                                   # joins the consumer -> all finishes processed
    assert not drain.is_alive()

    popped = index.pop_deliverable()
    _assert_pr_reconstructs(popped, exp)           # both A and B, byte-identical
    assert index.pop_deliverable() == [], "a delivered request must not be re-delivered"

    # free A only -> B stays live, A gone.
    index.free("A")
    assert index.live_req_ids() == {"B"}, f"live={index.live_req_ids()} after freeing A"
    index.free("B")
    assert index.live_req_ids() == set()


def test_offloop_per_request_survives_wrap():
    """A tiny ring forces PHYSICAL WRAP + reserve-BLOCK across steps; the demuxed slices must SURVIVE
    ring-slot reuse (the consumer clones each slice). Without the clone the stored rows would be
    overwritten on the next wrap and the reconstruction would differ."""
    tmp = _mkdtmp("pr_wrap")
    hidden, R, dtype = 4, 4, torch.float32         # R=4 rows/layer -> repeated wraps
    ring, hs_bufs, drain, index = _pr_build(R, hidden, (1,), dtype, tmp)
    drain.start()
    exp = {}
    # Two requests reserving 2 rows each, alternating, on a 4-row ring -> wraps + real blocking.
    plan = [("X", "a"), ("Y", "b"), ("X", "c"), ("Y", "d"), ("X", "e"), ("Y", "f")]
    for rid, tag in plan:
        _pr_step(ring, hs_bufs, drain, [(rid, 2, [1], "all_tokens")], exp, dtype, tag, consumer=drain)
    drain.enqueue_finish("X")
    drain.enqueue_finish("Y")
    _wait_drained(ring)
    drain.stop()
    _assert_pr_reconstructs(index.pop_deliverable(), exp)


def test_offloop_per_request_ignores_uncaptured_finish():
    """A FINISH for a req_id the drain never captured must NOT create a spurious empty delivery
    (finished_req_ids includes non-capturing requests)."""
    tmp = _mkdtmp("pr_ghost")
    hidden, R, dtype = 4, 64, torch.float32
    ring, hs_bufs, drain, index = _pr_build(R, hidden, (1,), dtype, tmp)
    drain.start()
    exp = {}
    _pr_step(ring, hs_bufs, drain, [("real", 2, [1], "all_tokens")], exp, dtype, "s1")
    drain.enqueue_finish("ghost")                  # never captured
    drain.enqueue_finish("real")
    _wait_drained(ring)
    drain.stop()
    popped = index.pop_deliverable()
    assert [rid for rid, _ in popped] == ["real"], f"ghost leaked into delivery: {popped}"
    _assert_pr_reconstructs(popped, exp)


def test_offloop_per_request_end_of_run_straggler_finalize():
    """A request that finishes on the FINAL executed step never gets its _Finish enqueued
    (`finished_req_ids` is reported the step AFTER the last rows — a step vLLM may never run when
    the last request, or a whole batch, finishes on the same step). Its rows are all drained/noted
    but it is never marked finished, so the streaming path silently drops it. The end-of-run
    finalize in stop() must mark it finished so pop_deliverable delivers it byte-identically — the
    Task 6 multi-request GPU oracle would otherwise lose its last request(s)."""
    tmp = _mkdtmp("pr_straggler")
    hidden, R, dtype = 4, 512, torch.float32       # roomy ring: isolate the finalize logic
    layer_ids = (1, 2)
    ring, hs_bufs, drain, index = _pr_build(R, hidden, layer_ids, dtype, tmp)
    drain.start()
    exp = {}
    # S captures over two steps and finishes on the last one — NO enqueue_finish("S") is ever made.
    _pr_step(ring, hs_bufs, drain, [("S", 3, [1, 2], "all_tokens")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain, [("S", 1, [1, 2], "all_tokens")], exp, dtype, "s2")
    _wait_drained(ring)
    # BEFORE finalize: every row is noted but S was never marked finished -> NOT deliverable.
    assert index.pop_deliverable() == [], "straggler delivered before finalize (finish never enqueued)"
    assert index.live_req_ids() == {"S"}, f"straggler must stay live pre-finalize: {index.live_req_ids()}"
    # stop() at GENUINE end-of-run finalizes every still-live request -> S becomes deliverable.
    drain.stop()
    assert not drain.is_alive()
    _assert_pr_reconstructs(index.pop_deliverable(), exp)   # S, byte-identical
    assert index.pop_deliverable() == [], "finalized straggler must not be re-delivered"


def test_offloop_finalize_noop_when_mode_off():
    """finalize_all() (invoked by stop() at end-of-run) is a STRICT NO-OP when per_request is OFF:
    the index is None, so there is nothing to finalize and the shared-file default path stays
    byte-identical — the whole point of Task 5's gating."""
    tmp = _mkdtmp("finalize_off")
    hidden, R, dtype = 4, 128, torch.float32
    ring, hs_bufs, drain = _build(R, hidden, (1, 2), dtype, tmp)   # per_request OFF (default)
    assert drain.index is None, "default (shared-file) drain must have no PerRequestIndex"
    drain.start()
    expected = {}
    _engine_step(ring, hs_bufs, drain, [("A", 2, [1, 2], "all_tokens")], expected, dtype)
    _wait_drained(ring)
    drain.finalize_all()          # explicit call: harmless (no-op) with index None
    drain.stop()                  # stop() also calls finalize_all() internally -> still a no-op
    drain.close()
    assert not drain.is_alive()
    _assert_reconstructs(tmp, expected)   # shared-file reconstruction unaffected


def main():
    tests = [
        test_offloop_roundtrip_float32,
        test_offloop_roundtrip_bfloat16,
        test_offloop_wrap_with_backpressure,
        test_reserve_block_blocks_then_succeeds_not_drop,
        test_reserve_block_raises_fast_on_dead_consumer,
        test_flush_joins_thread_and_writes_sidecar,
        test_consumer_error_surfaces_loud_at_flush,
        test_shared_routing_wrapper_reraises_backpressure,
        test_offloop_per_request_two_interleaved,
        test_offloop_per_request_survives_wrap,
        test_offloop_per_request_ignores_uncaptured_finish,
        test_offloop_per_request_end_of_run_straggler_finalize,
        test_offloop_finalize_noop_when_mode_off,
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
