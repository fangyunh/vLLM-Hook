"""No-GPU THREADED test for the DISK per-request STAGING-DIR LIFECYCLE race (Task 12 dir-race fix).

The observed serve failure (LSF 583700): the engine-thread abort ``clear_request_disk`` ``rmtree``'d an
aborted disk request's per-request staging dir WHILE the off-loop CONSUMER thread was still demuxing
that same request's REMAINING layers. The consumer's next ``_disk_write`` then created a fresh
``_MmapLayerWriter`` (``open(hs_layer_<L>.raw)``) inside the just-deleted dir ->
``FileNotFoundError``. That raise is in ``_drain_item`` (FATAL, not the isolated finish path), so the
CONSUMER THREAD DIED -> every subsequent per-request finalize stopped -> disk AND host/RPC deliveries
all went empty, and the aborted request's staging leaked.

THE FIX = single-owner staging-dir lifecycle: only the CONSUMER thread creates, writes, AND deletes a
per-request staging dir. The engine-thread abort only MARKS (``_disk_aborted``); the consumer skips a
marked request's writes (``_disk_write`` guard) and DISCARDS its dir on the request's ``_Finish``.

This test drives a REAL ``OffLoopRingDrain(per_request=True)`` with a STARTED consumer thread on CPU
(device="cpu" -> the ring's streams/events are no-ops, so the full consumer/queue/demux machinery
runs without a GPU). A gated ``_PerRequestDiskStaging`` subclass parks the consumer mid-demux of the
aborted request R -- holding R's live staging, its dir present, about to open the NEXT layer's file --
so the driver's ``clear_request_disk(R)`` lands EXACTLY in that window (the traceback interleave).

Interleaved alongside R: a HEALTHY disk request (must still deliver byte-identical) and a HEALTHY host
request (must still deliver byte-identical). Assertions: no ``FileNotFoundError``, the consumer
SURVIVES, both healthy requests deliver, and after the aborted request's ``_Finish`` its
``disk_residency()`` -> 0 with the aborted dir removed.

RED-then-GREEN: reverting the fix (``graph/ring_drain_hs.py`` to the pre-fix ``clear_request_disk``
that pops + ``rmtree``s the live staging) makes the parked ``open`` raise ``FileNotFoundError``, the
consumer dies, and neither healthy request delivers. Confirmed RED against the reverted source, GREEN
against the fix; re-run for flake-freedom.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_dir_lifecycle_race.py -q
"""
import os
import shutil
import threading
import time

import torch

import vllm_hook_plugins.graph.ring_drain_hs as rd
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact

_HIDDEN = 4
_GATE_LAYER = 2                     # park on the aborted request's SECOND layer (dir + layer-1 present)


# --------------------------------------------------------------- gated staging ---
# Module-level gate state (the drain constructs staging internally, so the subclass reads these).
_gate_req = None
_gate_reached = threading.Event()
_gate_release = threading.Event()
_gate_fired = [False]


class _GatedStaging(rd._PerRequestDiskStaging):
    """Real per-request staging, except the FIRST ``append`` for ``(_gate_req, _GATE_LAYER)`` parks the
    consumer -- lock already released, live staging held, dir present, about to ``open`` the next
    layer's file -- until the driver's abort has run. This lands ``clear_request_disk`` in the exact
    rmtree-vs-open window of the traceback."""

    def append(self, layer, rows_cpu, n_rows, mode):
        if (self.req_id == _gate_req and int(layer) == _GATE_LAYER and not _gate_fired[0]):
            _gate_fired[0] = True
            _gate_reached.set()
            _gate_release.wait(timeout=5.0)
        return super().append(layer, rows_cpu, n_rows, mode)


# ------------------------------------------------------------------ helpers ---
def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


def _fake_offload():
    calls = []

    def fake_transfer(src, dest):
        calls.append((src, dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    return OffloadProcess(transfer_fn=fake_transfer), calls


def _build(R, hidden, layer_ids, dtype, tmp, offload):
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    index = PerRequestIndex()
    drain = OffLoopRingDrain(
        ring, [(L, hs_bufs[L]) for L in layer_ids], os.path.join(tmp, "run"), header,
        per_request=True, index=index, offload=offload,
        disk_base=os.path.join(tmp, "staging"))
    return ring, hs_bufs, drain, index


def _pr_step(ring, hs_bufs, drain, reqs, exp, dtype, step_tag):
    """One engine step. reqs = [(req_id, n_rows, [layers], mode)]. Writes STEP-UNIQUE deterministic
    data into each layer's reserved ring slots, records LayerEntrys, O(1)-enqueues, and records the
    expected per-(req,layer) block in APPEND order."""
    entries = []
    start_logical = None
    total = 0
    for (rid, n, layers, mode) in reqs:
        s = _ring_reserve_or_block(ring, n, None)
        if start_logical is None:
            start_logical = s
        total += n
        phys = ring.physical_slots(s, n)
        for L in layers:
            hidden = hs_bufs[L].shape[1]
            base = (hash((rid, L, step_tag)) % 997) * 1000 + 1
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)
            for j, p in enumerate(phys):
                hs_bufs[L][p] = data[j]
            entries.append(LayerEntry(str(rid), L, s, n, mode))
            exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, start_logical, total, None)


def _wait_queue(drain, timeout=10.0):
    """Poll until the consumer processed every queued item (rows AND finishes). Poll (not
    ``_q.join()``) so a WEDGED consumer -- the RED case -- times out instead of hanging forever."""
    deadline = time.monotonic() + timeout
    while drain._q.unfinished_tasks > 0 and time.monotonic() < deadline:
        time.sleep(0.002)


def _assert_disk_reconstructs(dest, rid, exp_for_rid):
    out = load_multilayer_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    assert set(out[rid]) == set(exp_for_rid), (
        f"{rid}: delivered layers {set(out[rid])} != expected {set(exp_for_rid)}")
    for L, blocks in exp_for_rid.items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
        assert torch.equal(out[rid][L], want), f"{rid}/{L} byte mismatch"


def _reset_gate(req):
    global _gate_req
    _gate_req = req
    _gate_reached.clear()
    _gate_release.clear()
    _gate_fired[0] = False


def test_abort_rmtree_vs_consumer_disk_write_single_owner(tmp_path, monkeypatch):
    """The traceback interleave, made deterministic. R (disk) is parked mid-demux on its 2nd layer
    while the driver aborts it; a healthy disk request D and a healthy host request H are captured in
    the same step. Post-fix: the consumer never opens a file in a discarded dir (it SURVIVES), D + H
    deliver byte-identical, and R is reclaimed (residency 0, dir gone). Pre-fix (reverted
    clear_request_disk): the parked open raises FileNotFoundError, the consumer dies, and D + H never
    deliver."""
    monkeypatch.setattr(rd, "_PerRequestDiskStaging", _GatedStaging)   # drain builds the gated staging
    _reset_gate("R")

    op, calls = _fake_offload()
    hidden, R, dtype = _HIDDEN, 512, torch.float32     # roomy ring: no wrap, isolate the dir race
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destR = str(tmp_path / "delivered" / "R")          # (never delivered — R is aborted)
    destD = str(tmp_path / "delivered" / "D")
    drain.route_to_disk("R", destR)                    # R + D -> disk; H -> host (default)
    drain.route_to_disk("D", destD)

    exp = {}
    # ONE step: R, D disk-routed, H host. Entry order R,D,H -> the consumer demuxes R first (parks on
    # R layer 2), then D, then H. R's remaining layer (3) is the "remaining rows" the abort races.
    _pr_step(ring, hs_bufs, drain,
             [("R", 2, [1, 2, 3], "all_tokens"),
              ("D", 2, [1, 2, 3], "all_tokens"),
              ("H", 2, [1, 2, 3], "all_tokens")], exp, dtype, "s1")
    # Enqueue the finishes now (FIFO: they trail the step's rows). finished_req_ids includes the abort.
    drain.enqueue_finish("R")
    drain.enqueue_finish("D")
    drain.enqueue_finish("H")

    # Wait until the consumer is parked mid-demux of R (holding R's live staging, dir present), then
    # run the REAL abort in that window, then release the consumer.
    assert _gate_reached.wait(timeout=5.0), "consumer never reached the mid-demux gate (vacuous)"
    src_R = drain._disk_staging["R"].run_dir
    assert os.path.isdir(src_R), "R's staging dir should be present at the gate"
    drain.clear_request_disk("R")                      # abort R IN the demux window (the race)
    _gate_release.set()

    _wait_queue(drain)

    # Post-fix: the consumer SURVIVED the abort-vs-write race (no FileNotFoundError, no _error).
    assert drain.is_alive(), "consumer thread died (dir-lifecycle race not closed)"
    assert drain.error is None, f"consumer error must be None: {drain.error!r}"

    # The healthy DISK request delivered end-to-end, byte-identical; R (aborted) was never offloaded.
    assert op.wait("D", timeout=5.0) is True, "healthy disk request never delivered (consumer wedged)"
    assert {os.path.basename(s) for s, _ in calls} == {"D"}, (
        f"only D should offload (R aborted, H is host): {calls}")
    _assert_disk_reconstructs(destD, "D", exp["D"])

    # The healthy HOST request delivered via the PerRequestIndex, byte-identical.
    popped = {rid: layers for rid, layers in index.pop_deliverable()}
    assert set(popped) == {"H"}, f"host index delivered {set(popped)} != {{H}}"
    for L, blocks in exp["H"].items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
        assert torch.equal(popped["H"][L], want), f"H/{L} host-buffer byte mismatch"

    # R (aborted) is reclaimed by the single-owner consumer discard: residency 0, dir gone, no leak.
    deadline = time.monotonic() + 5.0
    while drain.disk_residency() > 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    assert drain.disk_residency() == 0, "aborted staging leaked (residency should reach 0)"
    assert not os.path.exists(src_R), "aborted staging dir not removed (NVMe leak)"
    assert "R" not in drain._disk_aborted, "aborted mark not cleared after reclaim"
    assert not os.path.exists(destR), "an aborted request must never be delivered to its dest"

    drain.stop()
    op.close()


if __name__ == "__main__":
    import tempfile
    import pathlib

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    d = tempfile.mkdtemp(prefix="dirrace_")
    try:
        test_abort_rmtree_vs_consumer_disk_write_single_owner(pathlib.Path(d), _MP())
        print("PASS  test_abort_rmtree_vs_consumer_disk_write_single_owner")
        print("VERDICT: PASS (1/1)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
