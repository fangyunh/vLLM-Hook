"""No-GPU RESIDENCY / ABORT / ID-DIVERGENCE tests for the off-loop QK capture-ring per-request path
(Task 13 — the QK port of test_ring_residency_no_leak).

Drives the REAL ``OffLoopQKRingDrain(per_request=True)`` + ``ProbeHookQKWorker`` methods (consumer
thread NOT started, so the interleave is deterministic and no GPU/stream is touched): the two ported
abort leaks and the serve id-divergence match.

  * leak A (HOST abort re-note): ``clear_ring_request`` frees a live host entry; a BACKLOGGED demux
    (rows enqueued before the abort, consumed after) re-``note_rows``es it; ``mark_host_aborted`` +
    the demux abort-skip make the finish drop it -> no leaked ``_entries`` slot.
  * leak B (DISK abort -> host): ``clear_request_disk`` POPS ``_disk_routed`` so the request's
    remaining demuxed rows fall THROUGH to the host index; the ``_disk_aborted`` skip suppresses them.
  * id-divergence: serve rewrites the external id to ``{external}-{rand}``; every hop (demux/finish/
    abort/retrieval) keys on the EXTERNAL id via ``_match_disk_route`` / ``iter_matching_req_ids``.

RED-then-GREEN: against the pre-fix code (mark/skip removed) the abort tests FAIL (the re-noted /
route-popped rows leak a host ``_entries`` slot); against the fix they pass. The healthy-batch test
passes either way (a regression guard).

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_qk_ring_residency_no_leak.py -q
"""
import os
import shutil
import tempfile

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import _DrainItem, _torch_dtype_name
from vllm_hook_plugins.graph.ring_drain_qk import OffLoopQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import QKStepEntry
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker

_QDIM, _KDIM = 8, 4
_LAYERS = (0, 1)
_DTYPE = torch.float32


def _build(tmp, offload=None):
    ring = GpuCaptureRing(row_bytes=_KDIM * torch.empty(0, dtype=_DTYPE).element_size(),
                          n_slots=512, device="cpu", dtype=_DTYPE, row_shape=(_KDIM,))
    q_bufs = {L: torch.zeros(513, _QDIM, dtype=_DTYPE) for L in _LAYERS}
    k_bufs = {L: torch.zeros(513, _KDIM, dtype=_DTYPE) for L in _LAYERS}
    header = {"dtype": _torch_dtype_name(_DTYPE), "q_row_shape": [_QDIM], "k_row_shape": [_KDIM],
              "q_dim": _QDIM, "k_dim": _KDIM, "hookq_mode": "all_tokens"}
    run_dir = tempfile.mkdtemp(prefix="qk_res_", dir=tmp)
    index = PerRequestIndex()
    kw = {"per_request": True, "index": index, "disk_base": os.path.join(run_dir, "staging")}
    if offload is not None:
        kw["offload"] = offload
    drain = OffLoopQKRingDrain(
        ring, [(L, q_bufs[L], k_bufs[L]) for L in _LAYERS], run_dir, header, **kw)
    w = ProbeHookQKWorker()
    w._qk_drain = drain
    w._conf = {}
    return drain, index, w


def _demux_step(drain, reqs, start_logical, tag):
    """Demux ``reqs`` = [(req_id, n)] (all_tokens over _LAYERS) straight into the drain via the REAL
    ``_demux_into_index``. Returns the next logical cursor."""
    total = sum(n for _, n in reqs)
    q_by = {L: torch.zeros(total, _QDIM, dtype=_DTYPE) for L in _LAYERS}
    k_by = {L: torch.zeros(total, _KDIM, dtype=_DTYPE) for L in _LAYERS}
    entries = []
    off = 0
    for (rid, n) in reqs:
        for L in _LAYERS:
            base = (hash((rid, L, tag)) % 991) * 1000 + 1
            q_by[L][off:off + n] = torch.arange(n * _QDIM, dtype=torch.float32).reshape(n, _QDIM) + base
            k_by[L][off:off + n] = torch.arange(n * _KDIM, dtype=torch.float32).reshape(n, _KDIM) + base
            entries.append(QKStepEntry(str(rid), int(L), start_logical + off, n,
                                       start_logical + off, n, start_logical + off + n, 0))
        off += n
    item = _DrainItem(entries, start_logical, total, None)
    drain._demux_into_index(item, [(L, q_by[L], k_by[L]) for L in _LAYERS])
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
    tmp = tempfile.mkdtemp(prefix="qk_res_ok_")
    try:
        drain, index, w = _build(tmp)
        ids = ["r0", "r1", "r2", "r3"]
        sl = 0
        sl = _demux_step(drain, [(r, 3) for r in ids], sl, "s1")
        sl = _demux_step(drain, [(r, 1) for r in ids], sl, "s2")
        assert index.live_req_ids() == set(ids)
        for r in ids:
            drain._handle_finish(r)
        for r in ids:
            assert w.get_ring_per_request(r) is not None, f"{r} not delivered"
        assert len(index.live_req_ids()) == 0, "delivered host requests leaked _entries slots"
        assert w.ring_residency() == (0, 0)
        assert not (getattr(w, "_ring_perreq_stash", None) or {}), "stash should be drained"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- leak A: host abort re-note ---
def test_aborted_host_request_backlog_renote_leaves_no_entry():
    tmp = tempfile.mkdtemp(prefix="qk_res_hostabort_")
    try:
        drain, index, w = _build(tmp)
        sl = 0
        sl = _demux_step(drain, [("A", 3)], sl, "s1")
        assert index.live_req_ids() == {"A"}, "A should be live before the abort"
        w.clear_ring_request("A")                        # client aborts mid-flight (host route)
        assert "A" not in index.live_req_ids(), "clear_ring_request must free the live entry"
        _demux_step(drain, [("A", 1)], sl, "s2")         # backlogged rows, consumed post-free
        drain._handle_finish("A")                        # the abort's _Finish
        assert len(index.live_req_ids()) == 0, (
            "aborted host request re-created a leaked _entries slot (backlog re-note after free)")
        assert w.ring_residency() == (0, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- leak B: disk abort -> host ---
def test_aborted_disk_request_route_popped_no_host_leak():
    tmp = tempfile.mkdtemp(prefix="qk_res_diskabort_")
    op, _calls = _fake_offload()
    try:
        drain, index, w = _build(tmp, offload=op)
        drain.route_to_disk("D", os.path.join(tmp, "delivered", "D"))
        sl = 0
        sl = _demux_step(drain, [("D", 3)], sl, "s1")
        assert drain.disk_residency() == 1, "D should hold disk staging"
        assert index.live_req_ids() == set(), "a disk-routed request must not touch the host index"
        w.clear_ring_request("D")                        # abort -> clear_request_disk pops the route
        _demux_step(drain, [("D", 1)], sl, "s2")         # post-pop rows (would fall to host)
        assert len(index.live_req_ids()) == 0, (
            "disk-aborted request's post-pop rows leaked into the host index")
        drain._handle_finish("D")                        # abort-reclaim discards staging
        assert drain.disk_residency() == 0, "disk staging not freed on abort"
        assert w.ring_residency() == (0, 0)
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- id-divergence: external match ---
def test_id_divergence_external_abort_matches_internal():
    """Serve rewrites the external 'req9' -> internal 'req9-ab12cd34' on the rows. clear_ring_request
    keyed on the EXTERNAL id must free the internal host entry (mark_host_aborted + free_external both
    use the exact-or-'{ext}-' match), so residency returns to 0."""
    tmp = tempfile.mkdtemp(prefix="qk_res_iddiv_")
    try:
        drain, index, w = _build(tmp)
        sl = 0
        _demux_step(drain, [("req9-ab12cd34", 3)], sl, "s1")   # internal id on the rows
        assert index.live_req_ids() == {"req9-ab12cd34"}
        w.clear_ring_request("req9")                           # abort by the EXTERNAL id
        assert len(index.live_req_ids()) == 0, "external-id abort did not free the internal entry"
        assert w.ring_residency() == (0, 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_id_divergence_disk_route_and_confirm_key_external():
    """Disk route registered under the EXTERNAL id; the drain-seen rows carry the INTERNAL id ->
    _match_disk_route stages them under the external key, so residency is a disk (not host) hold and
    the abort by external id reclaims it."""
    tmp = tempfile.mkdtemp(prefix="qk_res_iddiv_disk_")
    op, _calls = _fake_offload()
    try:
        drain, index, w = _build(tmp, offload=op)
        drain.route_to_disk("job7", os.path.join(tmp, "delivered", "job7"))   # EXTERNAL id
        sl = 0
        _demux_step(drain, [("job7-ffff0000", 3)], sl, "s1")   # INTERNAL id on the rows
        assert drain.disk_residency() == 1, "internal-id rows did not match the external disk route"
        assert index.live_req_ids() == set(), "disk-routed rows leaked to the host index"
        w.clear_ring_request("job7")                           # abort by EXTERNAL id
        drain._handle_finish("job7-ffff0000")                  # abort's _Finish (internal id)
        assert drain.disk_residency() == 0, "external-id abort did not reclaim the disk staging"
        assert w.ring_residency() == (0, 0)
    finally:
        op.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- serve-shape: abort then batch ---
def test_serve_shape_abort_then_main_batch_residency_zero():
    tmp = tempfile.mkdtemp(prefix="qk_res_serve_")
    op, _calls = _fake_offload()
    try:
        drain, index, w = _build(tmp, offload=op)
        drain.route_to_disk("aD", os.path.join(tmp, "delivered", "aD"))
        sl = 0
        sl = _demux_step(drain, [("aR", 3), ("aD", 3)], sl, "a1")
        assert index.live_req_ids() == {"aR"} and drain.disk_residency() == 1
        w.clear_ring_request("aR")
        w.clear_ring_request("aD")
        sl = _demux_step(drain, [("aR", 1), ("aD", 1)], sl, "a2")
        drain._handle_finish("aR")
        drain._handle_finish("aD")
        assert w.ring_residency() == (0, 0), "residency_after_abort must be (0,0)"

        ids = ["m0", "m1", "m2"]
        sl = _demux_step(drain, [(r, 3) for r in ids], sl, "m1")
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
              test_id_divergence_external_abort_matches_internal,
              test_id_divergence_disk_route_and_confirm_key_external,
              test_serve_shape_abort_then_main_batch_residency_zero):
        t()
        print(f"PASS  {t.__name__}")
    print("VERDICT: PASS")
