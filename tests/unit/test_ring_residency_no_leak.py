"""No-GPU RESIDENCY tests for the off-loop HS capture-ring per-request path (Task 12 item 1).

The serve gate (tests/cuda_graph/tests/ring/serve_per_request.py) reported the worker's
``ring_residency()`` host count NOT returning to 0 after a batch delivered (after_main=(1,0)) and
after an abort (after_abort=(2,0)): a HOST (RPC-route) ``PerRequestIndex`` entry leaked even though
the disk side was clean. Root cause -- the consumer STILL stages a request's drained rows into the
host index AFTER that request was aborted:
  * HOST abort (leak A): ``clear_ring_request`` frees the entry, but a backlogged / still-in-flight
    ``_DrainItem`` (rows enqueued before the abort, consumed after the free) re-``note_rows``es it,
    re-creating the freed ``_entries`` slot -- which nothing frees again (the aborted request is never
    retrieved), and its ``_Finish`` then ``mark_finished``es the phantom into ``_deliverable``.
  * DISK abort (leak B): ``clear_request_disk`` POPS ``_disk_routed``, so the request's remaining
    demuxed rows no longer match a disk route and fall THROUGH to the host index via ``note_rows`` --
    a permanent host entry whose ``_Finish`` takes the disk abort-reclaim branch and never frees it.

These drive the REAL ``OffLoopRingDrain(per_request=True)`` + ``ProbeHiddenStatesWorker`` methods
(consumer thread NOT started, so the interleave is deterministic): deliver a batch of host requests
via ``get_ring_per_request`` (bulk-drain-into-stash), abort host+disk requests, and assert
``len(index.live_req_ids()) == 0`` + ``ring_residency() == (0, 0)`` once every request is
delivered/aborted.

RED-then-GREEN: against the pre-fix code the abort tests FAIL (the re-noted / route-popped rows leak
a host ``_entries`` slot); against the fix they pass (the consumer skips staging an aborted request,
and the request's ``_Finish`` drops the abort mark). The healthy-batch test passes either way (a
regression guard that the fix does not perturb normal delivery).

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_residency_no_leak.py -q
"""
import os
import shutil
import tempfile

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import (
    OffLoopRingDrain, _DrainItem, _torch_dtype_name)
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker

_HIDDEN = 4
_LAYERS = (1, 2)
_DTYPE = torch.float32


def _build(tmp, offload=None):
    """A per_request OffLoopRingDrain (consumer NOT started) + its worker. The test drives
    _demux_into_index / _handle_finish / get_ring_per_request / clear_ring_request directly, so the
    interleave is deterministic and no GPU/stream is touched."""
    ring = GpuCaptureRing(row_bytes=_HIDDEN * torch.empty(0, dtype=_DTYPE).element_size(),
                          n_slots=512, device="cpu", dtype=_DTYPE, row_shape=(_HIDDEN,))
    hs_bufs = {L: torch.zeros(513, _HIDDEN, dtype=_DTYPE) for L in _LAYERS}
    header = {"dtype": _torch_dtype_name(_DTYPE), "row_shape": [_HIDDEN], "hidden": _HIDDEN}
    run_dir = tempfile.mkdtemp(prefix="residency_", dir=tmp)
    index = PerRequestIndex()
    kw = {"per_request": True, "index": index, "disk_base": os.path.join(run_dir, "staging")}
    if offload is not None:
        kw["offload"] = offload
    drain = OffLoopRingDrain(ring, [(L, hs_bufs[L]) for L in _LAYERS], run_dir, header, **kw)
    w = ProbeHiddenStatesWorker()
    w._hs_drain = drain
    w._conf = {"hidden_size": _HIDDEN, "num_layers": len(_LAYERS)}
    return drain, index, w


def _demux_step(drain, reqs, start_logical, tag):
    """Simulate ONE drained step: demux ``reqs`` = [(req_id, n_rows, mode)] (each capturing every
    _LAYERS layer) straight into the drain via the REAL ``_demux_into_index``. Returns the next
    logical cursor. Deterministic per-(req,layer,tag) data so a stray cross-request stage would be
    visible; here we only assert residency, so the values just need to be present."""
    total = sum(n for _, n, _ in reqs)
    by_layer = {L: torch.zeros(total, _HIDDEN, dtype=_DTYPE) for L in _LAYERS}
    entries = []
    off = 0
    for (rid, n, mode) in reqs:
        for L in _LAYERS:
            base = (hash((rid, L, tag)) % 997) * 1000 + 1
            data = (torch.arange(n * _HIDDEN, dtype=torch.float32).reshape(n, _HIDDEN) + base
                    ).to(_DTYPE)
            by_layer[L][off:off + n] = data
            entries.append(LayerEntry(str(rid), L, start_logical + off, n, mode))
        off += n
    item = _DrainItem(entries, start_logical, total, None)
    drain._demux_into_index(item, [(L, by_layer[L]) for L in _LAYERS])
    return start_logical + total


def _fake_offload():
    calls = []

    def fake_transfer(src, dest):
        calls.append((src, dest))
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    return OffloadProcess(transfer_fn=fake_transfer), calls


# ---------------------------------------------------------------- healthy batch (regression) ---
def test_host_batch_delivery_residency_returns_to_zero():
    """MAIN-batch analog: several host RPC requests captured over two steps, all finishing together,
    delivered via the worker's ``get_ring_per_request`` (bulk-drain-into-stash). Residency must return
    to 0. Passes pre- and post-fix -- guards that the abort fix does not perturb healthy delivery."""
    tmp = tempfile.mkdtemp(prefix="res_ok_")
    try:
        drain, index, w = _build(tmp)
        ids = ["r0", "r1", "r2", "r3"]
        sl = 0
        sl = _demux_step(drain, [(r, 3, "all_tokens") for r in ids], sl, "s1")
        sl = _demux_step(drain, [(r, 1, "all_tokens") for r in ids], sl, "s2")
        assert index.live_req_ids() == set(ids)
        for r in ids:                       # finished together -> all in _deliverable
            drain._handle_finish(r)
        for r in ids:                       # bulk-drain-into-stash frees every popped id
            assert w.get_ring_per_request(r) is not None, f"{r} not delivered"
        assert len(index.live_req_ids()) == 0, "delivered host requests leaked _entries slots"
        assert w.ring_residency() == (0, 0)
        assert not (getattr(w, "_ring_perreq_stash", None) or {}), "stash should be drained"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- leak A: host abort re-note ---
def test_aborted_host_request_backlog_renote_leaves_no_entry():
    """HOST abort: A is generating (live host entry). ``clear_ring_request(A)`` frees it. A BACKLOGGED
    DrainItem (rows enqueued before the abort) is consumed AFTER the free and re-``note_rows``es A;
    then A's ``_Finish`` (finished_req_ids includes aborts) is processed. Post-fix the consumer skips
    staging the aborted A and the finish drops the abort mark, so nothing leaks."""
    tmp = tempfile.mkdtemp(prefix="res_hostabort_")
    try:
        drain, index, w = _build(tmp)
        sl = 0
        sl = _demux_step(drain, [("A", 3, "all_tokens")], sl, "s1")
        assert index.live_req_ids() == {"A"}, "A should be live before the abort"

        w.clear_ring_request("A")                        # client aborts mid-flight (host route)
        assert "A" not in index.live_req_ids(), "clear_ring_request must free the live entry"

        _demux_step(drain, [("A", 1, "all_tokens")], sl, "s2")   # backlogged rows, consumed post-free
        drain._handle_finish("A")                        # the abort's _Finish

        assert len(index.live_req_ids()) == 0, (
            "aborted host request re-created a leaked _entries slot (backlog re-note after free)")
        assert w.ring_residency() == (0, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- leak B: disk abort -> host ---
def test_aborted_disk_request_route_popped_no_host_leak():
    """DISK abort: D is disk-routed (its rows stage to NVMe, never the host index).
    ``clear_ring_request(D)`` -> ``clear_request_disk`` POPS ``_disk_routed``, so D's remaining
    demuxed rows no longer match a disk route. Pre-fix they fall THROUGH to the host index
    (``note_rows``) -> a permanent host entry; post-fix the consumer skips an aborted request's rows,
    so no host entry appears and disk staging is reclaimed on finish."""
    tmp = tempfile.mkdtemp(prefix="res_diskabort_")
    op, _calls = _fake_offload()
    try:
        drain, index, w = _build(tmp, offload=op)
        drain.route_to_disk("D", os.path.join(tmp, "delivered", "D"))
        sl = 0
        sl = _demux_step(drain, [("D", 3, "all_tokens")], sl, "s1")
        assert drain.disk_residency() == 1, "D should hold disk staging"
        assert index.live_req_ids() == set(), "a disk-routed request must not touch the host index"

        w.clear_ring_request("D")                        # abort -> clear_request_disk pops the route
        _demux_step(drain, [("D", 1, "all_tokens")], sl, "s2")   # post-pop rows (would fall to host)

        assert len(index.live_req_ids()) == 0, (
            "disk-aborted request's post-pop rows leaked into the host index")
        drain._handle_finish("D")                        # abort-reclaim discards staging
        assert drain.disk_residency() == 0, "disk staging not freed on abort"
        assert len(index.live_req_ids()) == 0
        assert w.ring_residency() == (0, 0)
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- serve-shape: abort then batch ---
def test_serve_shape_abort_then_main_batch_residency_zero():
    """Faithful reproduction of the serve gate's checkpoints: an abort phase (one HOST + one DISK
    request, aborted mid-flight with backlog rows) then a MAIN batch of host requests. Assert
    ``ring_residency() == (0, 0)`` after the abort (the after_abort=(2,0) leak) AND after the main
    batch (the after_main=(1,0) leak)."""
    tmp = tempfile.mkdtemp(prefix="res_serve_")
    op, _calls = _fake_offload()
    try:
        drain, index, w = _build(tmp, offload=op)
        # ---- abort phase: host + disk aborted mid-flight ----
        drain.route_to_disk("aD", os.path.join(tmp, "delivered", "aD"))
        sl = 0
        sl = _demux_step(drain, [("aR", 3, "all_tokens"), ("aD", 3, "all_tokens")], sl, "a1")
        assert index.live_req_ids() == {"aR"} and drain.disk_residency() == 1
        w.clear_ring_request("aR")
        w.clear_ring_request("aD")
        sl = _demux_step(drain, [("aR", 1, "all_tokens"), ("aD", 1, "all_tokens")], sl, "a2")
        drain._handle_finish("aR")
        drain._handle_finish("aD")
        assert w.ring_residency() == (0, 0), "residency_after_abort must be (0,0)"

        # ---- main batch: host requests, finish together, delivered via get_ring_per_request ----
        ids = ["m0", "m1", "m2"]
        sl = _demux_step(drain, [(r, 3, "all_tokens") for r in ids], sl, "m1")
        for r in ids:
            drain._handle_finish(r)
        for r in ids:
            assert w.get_ring_per_request(r) is not None, f"{r} not delivered"
        assert w.ring_residency() == (0, 0), "residency_after_main must be (0,0)"
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for t in (test_host_batch_delivery_residency_returns_to_zero,
              test_aborted_host_request_backlog_renote_leaves_no_entry,
              test_aborted_disk_request_route_popped_no_host_leak,
              test_serve_shape_abort_then_main_batch_residency_zero):
        t()
        print(f"PASS  {t.__name__}")
    print("VERDICT: PASS (4/4)")
