"""No-GPU regression tests for the serve consumer WEDGE (Task 12): a per-request DISK finalize that
raises must fail for THAT request only and must NEVER kill the off-loop drain consumer thread.

The observed serve failure (LSF 583610): an ABORTED disk-routed request has PARTIAL staging and still
appears in vLLM's ``finished_req_ids``, so the consumer's ``_handle_finish`` runs its DISK branch and
``close()``/``offload.submit`` raise (``FileNotFoundError`` on a never-created ``hs_layer_<L>.raw``).
The raise propagated out of ``_run`` and WEDGED the consumer, so every SUBSEQUENT per-request finalize
stopped -> all deliveries (disk AND host/RPC) went empty.

The fix has two parts, both exercised here on CPU (device="cpu" -> the ring's streams/events are
no-ops, so the FULL consumer/queue/demux/finalize machinery runs without a GPU):
  * Part 1 -- per-request finalize isolation: ``_finalize_finish_isolated`` catches a finalize raise,
    logs it, and the consumer CONTINUES. RED (revert Part 1 -> ``_run`` calls ``_handle_finish``
    directly): the bad request's exception propagates, the consumer dies, and the healthy disk + host
    requests never deliver. GREEN: the bad request is dropped, the healthy ones still deliver.
  * Part 2 -- ``_PerRequestDiskStaging.close()`` tolerates a partial / vanished dir (it references
    only the layers actually appended, and guards the msync + sidecar write), so a partial staging
    closes cleanly instead of raising.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_consumer_wedge_isolation.py -q
"""
import os
import shutil
import time

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import (
    OffLoopRingDrain,
    _PerRequestDiskStaging,
    _torch_dtype_name,
)
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact


# ------------------------------------------------------------------ helpers ---
def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


class _FlakyOffload(OffloadProcess):
    """An OffloadProcess (thread backend) whose ``submit`` RAISES for one chosen req_id -- standing
    in for the real finalize error (partial-aborted-staging FileNotFoundError / a double-submit
    ValueError) that must be ISOLATED. Every OTHER req_id transfers normally via a real copytree so
    the healthy delivered dest is reconstructable."""

    def __init__(self, bad_id, calls):
        def transfer(src, dest):
            calls.append((src, dest))
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy(src, dest)

        super().__init__(transfer_fn=transfer)
        self._bad_id = str(bad_id)

    def submit(self, req_id, src_path, dest):
        if str(req_id) == self._bad_id:
            raise RuntimeError(
                f"simulated finalize failure for {req_id!r} (partial aborted disk staging)")
        return super().submit(req_id, src_path, dest)


def _recording_offload(calls):
    def transfer(src, dest):
        calls.append((src, dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    return OffloadProcess(transfer_fn=transfer)


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
    data into each layer's reserved ring slots, records LayerEntrys, O(1)-enqueues, and appends each
    (req, layer)'s block to exp[req][layer] in APPEND (step) order."""
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
            base = (hash((rid, L, step_tag)) % 997) * 1000 + 1   # step-unique, non-zero
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype)
            for j, p in enumerate(phys):
                hs_bufs[L][p] = data[j]
            entries.append(LayerEntry(str(rid), L, s, n, mode))
            exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, start_logical, total, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0, f"consumer did not drain: pending={ring.pending_rows()}"


def _wait_queue(drain, timeout=10.0):
    """Poll until the consumer has processed every queued item (rows AND finishes). Poll (not
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


# ================================================= Part 1: finalize error does not wedge ===
def test_partial_finalize_error_does_not_wedge_consumer(tmp_path):
    """A DISK-routed request whose finalize RAISES (partial aborted staging), INTERLEAVED with a
    healthy disk request AND a healthy host(RPC) request. The bad request's finalize must be isolated
    -- the consumer SURVIVES and BOTH healthy requests still deliver. Reverting Part 1 (``_run``
    calls ``_handle_finish`` directly) reproduces the wedge: the RuntimeError propagates, the consumer
    dies, and neither healthy request delivers."""
    calls = []
    op = _FlakyOffload("BAD", calls)
    hidden, R, dtype = 4, 512, torch.float32       # roomy ring: no wrap, isolate the finalize logic
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destBAD = str(tmp_path / "delivered" / "BAD")
    destGD = str(tmp_path / "delivered" / "GOODDISK")
    drain.route_to_disk("BAD", destBAD)            # BAD + GOODDISK -> disk; GOODHOST -> host (default)
    drain.route_to_disk("GOODDISK", destGD)

    exp = {}
    # BAD stages only a PARTIAL layer set ({1}) -- like an aborted request that never captured all
    # layers -- and its finalize will raise (the flaky offload). The two healthy requests capture the
    # full set and interleave in the same steps so their rows are enqueued around BAD's.
    _pr_step(ring, hs_bufs, drain,
             [("BAD", 2, [1], "all_tokens"),
              ("GOODDISK", 2, [1, 2, 3], "all_tokens"),
              ("GOODHOST", 2, [1, 2, 3], "all_tokens")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain,
             [("BAD", 1, [1], "all_tokens"),
              ("GOODDISK", 1, [1, 2, 3], "all_tokens"),
              ("GOODHOST", 1, [1, 2, 3], "all_tokens")], exp, dtype, "s2")
    _wait_drained(ring)

    # BAD finishes FIRST (its finalize raises). If Part 1 is reverted, the consumer dies HERE and the
    # two finishes below are never processed -> the healthy requests never deliver.
    drain.enqueue_finish("BAD")
    drain.enqueue_finish("GOODDISK")
    drain.enqueue_finish("GOODHOST")
    _wait_queue(drain)

    # Part 1: the consumer SURVIVED the isolated finalize error (no propagation, no _error set).
    assert drain.is_alive(), "consumer thread wedged by an isolated finalize error"
    assert drain.error is None, f"finalize error must not set drain.error: {drain.error!r}"

    # The bad request's delivery failed for ITSELF (its submit raised -> never confirmed).
    assert op.wait("BAD", timeout=0.3) is False, "BAD must not have delivered (its finalize raised)"

    # The healthy DISK request still delivered end-to-end, byte-identical.
    assert op.wait("GOODDISK", timeout=5.0) is True, "healthy disk request never delivered (wedge)"
    assert {os.path.basename(s) for s, _ in calls} == {"GOODDISK"}, (
        f"only GOODDISK should offload (BAD's submit raised, GOODHOST is host): {calls}")
    _assert_disk_reconstructs(destGD, "GOODDISK", exp["GOODDISK"])

    # The healthy HOST(RPC) request still delivered via the PerRequestIndex, byte-identical.
    popped = {rid: layers for rid, layers in index.pop_deliverable()}
    assert set(popped) == {"GOODHOST"}, f"host index delivered {set(popped)} != {{GOODHOST}}"
    for L, blocks in exp["GOODHOST"].items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
        assert torch.equal(popped["GOODHOST"][L], want), f"GOODHOST/{L} host-buffer byte mismatch"

    drain.stop()
    op.close()


# ================================================= Part 2: partial staging close() is tolerant ===
def test_partial_staging_close_tolerates_partial_layers_and_vanished_dir(tmp_path):
    """``_PerRequestDiskStaging.close()`` must NOT raise on a partial staging: (a) a present dir with
    only SOME layers appended closes cleanly and reconstructs exactly those layers; (b) a run_dir that
    the abort-discard path rmtree'd out from under the finish closes WITHOUT raising (pre-fix:
    ``FileNotFoundError`` writing the sidecar)."""
    dtype, hidden = torch.float32, 4
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}

    # (a) partial-but-present dir: append 3 of 8 possible layers, close, reconstruct only those 3.
    run_dir = str(tmp_path / "present")
    stg = _PerRequestDiskStaging("P", run_dir, header, 64 * 1024 * 1024, use_mmap=True)
    exp = {}
    for L in (1, 3, 5):                             # 3 of {0..7}
        data = (torch.arange(2 * hidden, dtype=dtype).reshape(2, hidden) + (L * 10 + 1))
        stg.append(L, data, 2, "all_tokens")
        exp[L] = data
    stg.close()                                    # must NOT raise
    out = load_multilayer_ring_artifact(run_dir)
    assert set(out["P"]) == {1, 3, 5}, f"partial dir reconstructed layers {set(out['P'])} != {{1,3,5}}"
    for L, want in exp.items():
        assert torch.equal(out["P"][L], want), f"P/{L} byte mismatch"

    # (b) vanished dir (an abort rmtree raced this finish): close() must NOT raise.
    run_dir2 = str(tmp_path / "vanished")
    stg2 = _PerRequestDiskStaging("V", run_dir2, header, 64 * 1024 * 1024, use_mmap=True)
    stg2.append(1, torch.zeros(1, hidden, dtype=dtype), 1, "all_tokens")
    shutil.rmtree(run_dir2)                         # simulate the abort-discard removing it under us
    stg2.close()                                   # must NOT raise (pre-fix: FileNotFoundError)
    assert not os.path.exists(os.path.join(run_dir2, "hs_ring_meta.jsonl")), (
        "close() must not resurrect a vanished aborted run_dir into a sidecar-only delivery")


# ============================================ Part 2: aborted disk request — single-owner discard ===
def test_aborted_disk_request_discards_partial_staging_no_offload(tmp_path):
    """SINGLE-OWNER staging-dir lifecycle (Task 12 dir-race fix): the engine-thread abort
    (``clear_request_disk``) only MARKS -- it drops the route + marks ``_disk_aborted`` but must NOT
    rmtree the live staging dir (that is exactly the race that killed the consumer). The CONSUMER
    thread owns the DISCARD: on the aborted request's ``_Finish`` (or at ``finalize_all``),
    ``_handle_finish`` closes the fds + rmtrees the source, NO offload / NO sidecar -> residency 0, the
    source dir removed (no NVMe leak), the aborted request never delivered."""
    calls = []
    op = _recording_offload(calls)
    hidden, R, dtype = 4, 256, torch.float32
    ring, hs_bufs, drain, index = _build(R, hidden, (1, 2), dtype, str(tmp_path), op)
    # Drive the guarded methods directly (no consumer thread) -- this exercises the abort mark +
    # single-owner discard discipline, not the ring D2H.
    dest = str(tmp_path / "delivered" / "X")
    drain.route_to_disk("X", dest)
    drain._disk_write("X", 1, torch.zeros(3, hidden, dtype=dtype), 3, "all_tokens")  # partial: layer 1
    src = drain._disk_staging["X"].run_dir
    assert drain.disk_residency() == 1 and os.path.isdir(src)

    drain.clear_request_disk("X")                  # abort MARK (does NOT destroy the live dir)

    # Marked, not destroyed: route dropped + marked, but staging + dir survive until the consumer reclaims.
    assert "X" not in drain._disk_routed, "aborted route not dropped"
    assert "X" in drain._disk_aborted, "aborted request not marked for consumer reclaim"
    assert drain.disk_residency() == 1 and os.path.isdir(src), (
        "abort must NOT rmtree the live staging dir (single-owner: consumer reclaims)")
    assert calls == [], "abort must not offload"

    # The consumer processes the aborted request's _Finish -> single-owner DISCARD.
    drain._handle_finish("X")

    assert drain.disk_residency() == 0, "aborted staging not freed after consumer reclaim"
    assert not os.path.exists(src), "aborted staging source dir not removed after reclaim (NVMe leak)"
    assert "X" not in drain._disk_aborted, "aborted mark not cleared after reclaim"
    time.sleep(0.05)
    assert calls == [], f"an aborted request must not be offloaded, got {calls}"
    op.close()


if __name__ == "__main__":
    import tempfile
    for t in (test_partial_finalize_error_does_not_wedge_consumer,
              test_partial_staging_close_tolerates_partial_layers_and_vanished_dir,
              test_aborted_disk_request_discards_partial_staging_no_offload):
        d = tempfile.mkdtemp(prefix="wedge_")
        try:
            import pathlib
            t(pathlib.Path(d))
            print(f"PASS  {t.__name__}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("VERDICT: PASS (3/3)")
