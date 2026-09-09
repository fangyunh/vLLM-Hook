"""No-GPU THREADED test for the QK DISK per-request STAGING-DIR LIFECYCLE race (Task 13 — the QK port
of test_ring_dir_lifecycle_race).

The race (proven on the HS side): the engine-thread abort ``clear_request_disk`` must NOT ``rmtree`` an
aborted disk request's per-request staging dir WHILE the off-loop CONSUMER thread is still demuxing
that same request's REMAINING layers -- else the consumer's next ``_disk_write`` opens a fresh
``_MmapLayerWriter`` inside the just-deleted dir -> ``FileNotFoundError`` in ``_drain_item`` (FATAL) ->
the CONSUMER THREAD DIES -> every subsequent per-request delivery stops.

THE FIX (ported to ``OffLoopQKRingDrain``) = single-owner staging-dir lifecycle: only the CONSUMER
thread creates, writes, AND deletes a per-request staging dir. The engine-thread abort only MARKS
(``_disk_aborted``); the consumer skips a marked request's writes (``_disk_write`` guard) and DISCARDs
its dir on the request's ``_Finish``.

A gated ``_PerRequestQKDiskStaging`` subclass parks the consumer mid-demux of the aborted request R
(holding R's live staging, its dir present, about to open the NEXT layer's q/k files) so the driver's
``clear_request_disk(R)`` lands EXACTLY in that window. Interleaved: a HEALTHY disk request D and a
HEALTHY host request H (both must still deliver byte-identical). Assertions: no ``FileNotFoundError``,
the consumer SURVIVES, both healthy requests deliver, R is reclaimed (residency 0, dir gone).

RED-then-GREEN: reverting the fix (``clear_request_disk`` to pop + ``rmtree`` the live staging) makes
the parked ``open`` raise, the consumer dies, and neither healthy request delivers.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_qk_ring_dir_lifecycle_race.py -q
"""
import os
import shutil
import threading
import time

import torch

import vllm_hook_plugins.graph.ring_drain_qk as rd_qk
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import _torch_dtype_name
from vllm_hook_plugins.graph.ring_drain_qk import OffLoopQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import QKStepEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_qk_ring_artifact

_QDIM, _KDIM = 8, 4
_GATE_LAYER = 1                     # park on the aborted request's SECOND layer (dir + layer-0 present)

_gate_req = None
_gate_reached = threading.Event()
_gate_release = threading.Event()
_gate_fired = [False]


class _GatedStaging(rd_qk._PerRequestQKDiskStaging):
    """Real per-request QK staging, except the FIRST ``append`` for ``(_gate_req, _GATE_LAYER)`` parks
    the consumer -- lock released, live staging held, dir present, about to open the next layer's q/k
    files -- until the driver's abort has run."""

    def append(self, layer, q_rows_cpu, k_rows_cpu, prefix_end, num_computed):
        if (self.req_id == _gate_req and int(layer) == _GATE_LAYER and not _gate_fired[0]):
            _gate_fired[0] = True
            _gate_reached.set()
            _gate_release.wait(timeout=5.0)
        return super().append(layer, q_rows_cpu, k_rows_cpu, prefix_end, num_computed)


def _fake_offload():
    calls = []

    def fake_transfer(src, dest):
        calls.append((src, dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    return OffloadProcess(transfer_fn=fake_transfer), calls


def _build(R, layer_ids, dtype, tmp, offload):
    ring = GpuCaptureRing(row_bytes=_KDIM * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(_KDIM,))
    q_bufs = {L: torch.zeros(R + 1, _QDIM, dtype=dtype) for L in layer_ids}
    k_bufs = {L: torch.zeros(R + 1, _KDIM, dtype=dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "q_row_shape": [_QDIM], "k_row_shape": [_KDIM],
              "q_dim": _QDIM, "k_dim": _KDIM, "hookq_mode": "all_tokens"}
    index = PerRequestIndex()
    drain = OffLoopQKRingDrain(
        ring, [(L, q_bufs[L], k_bufs[L]) for L in layer_ids], os.path.join(tmp, "run"), header,
        per_request=True, index=index, offload=offload, disk_base=os.path.join(tmp, "staging"))
    return ring, q_bufs, k_bufs, drain, index


def _pr_qk_step(ring, q_bufs, k_bufs, drain, reqs, state, exp, tag):
    entries = []
    start_logical = None
    total = 0
    for (rid, n, layers) in reqs:
        s = _ring_reserve_or_block(ring, n, None)
        if start_logical is None:
            start_logical = s
        total += n
        abs_end = state.get(rid, 0) + n
        state[rid] = abs_end
        phys = ring.physical_slots(s, n)
        for L in layers:
            qbase = (hash((rid, L, tag, "q")) % 991) * 1000 + 1
            kbase = (hash((rid, L, tag, "k")) % 991) * 1000 + 1
            q_step = (torch.arange(n * _QDIM, dtype=torch.float32).reshape(n, _QDIM) + qbase)
            k_step = (torch.arange(n * _KDIM, dtype=torch.float32).reshape(n, _KDIM) + kbase)
            for j, p in enumerate(phys):
                q_bufs[L][p] = q_step[j]
                k_bufs[L][p] = k_step[j]
            entries.append(QKStepEntry(str(rid), int(L), s, n, s, n, abs_end, 0))
            e = exp.setdefault(str(rid), {}).setdefault(int(L), {"q": [], "k": [], "ends": []})
            e["k"].append(k_step)
            e["q"].append(q_step)
            e["ends"].append(abs_end)
    drain.enqueue(entries, start_logical, total, None)


def _wait_queue(drain, timeout=10.0):
    deadline = time.monotonic() + timeout
    while drain._q.unfinished_tasks > 0 and time.monotonic() < deadline:
        time.sleep(0.002)


def _assert_disk_qk_reconstructs(dest, rid, exp_for_rid):
    out = load_multilayer_qk_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    for L, e in exp_for_rid.items():
        k_full = e["k"][0] if len(e["k"]) == 1 else torch.cat(e["k"], 0)
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        assert torch.equal(out[rid][L]["q"], q), f"{rid}/{L} q byte mismatch"
        assert torch.equal(out[rid][L]["k_full"], k_full), f"{rid}/{L} k_full byte mismatch"


def _reset_gate(req):
    global _gate_req
    _gate_req = req
    _gate_reached.clear()
    _gate_release.clear()
    _gate_fired[0] = False


def test_abort_rmtree_vs_consumer_disk_write_single_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(rd_qk, "_PerRequestQKDiskStaging", _GatedStaging)
    _reset_gate("R")

    op, calls = _fake_offload()
    R, dtype = 512, torch.float32
    layer_ids = (0, 1, 2)
    ring, q_bufs, k_bufs, drain, index = _build(R, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destR = str(tmp_path / "delivered" / "R")
    destD = str(tmp_path / "delivered" / "D")
    drain.route_to_disk("R", destR)                    # R + D -> disk; H -> host
    drain.route_to_disk("D", destD)

    exp, state = {}, {}
    # ONE step, entry order R,D,H -> the consumer demuxes R first (parks on R layer 1), then D, then H.
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("R", 2, [0, 1, 2]), ("D", 2, [0, 1, 2]), ("H", 2, [0, 1, 2])], state, exp, "s1")
    drain.enqueue_finish("R")
    drain.enqueue_finish("D")
    drain.enqueue_finish("H")

    assert _gate_reached.wait(timeout=5.0), "consumer never reached the mid-demux gate (vacuous)"
    src_R = drain._disk_staging["R"].run_dir
    assert os.path.isdir(src_R), "R's staging dir should be present at the gate"
    drain.clear_request_disk("R")                      # abort R IN the demux window (the race)
    _gate_release.set()

    _wait_queue(drain)

    assert drain.is_alive(), "consumer thread died (dir-lifecycle race not closed)"
    assert drain.error is None, f"consumer error must be None: {drain.error!r}"

    assert op.wait("D", timeout=5.0) is True, "healthy disk request never delivered (consumer wedged)"
    assert {os.path.basename(s) for s, _ in calls} == {"D"}, (
        f"only D should offload (R aborted, H is host): {calls}")
    _assert_disk_qk_reconstructs(destD, "D", exp["D"])

    popped = dict(index.pop_deliverable_qk())
    assert set(popped) == {"H"}, f"host index delivered {set(popped)} != {{H}}"
    for L, e in exp["H"].items():
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        assert torch.equal(popped["H"][L]["q"], q), f"H/{L} host q byte mismatch"

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

    d = tempfile.mkdtemp(prefix="qk_dirrace_")
    try:
        test_abort_rmtree_vs_consumer_disk_write_single_owner(pathlib.Path(d), _MP())
        print("PASS  test_abort_rmtree_vs_consumer_disk_write_single_owner")
        print("VERDICT: PASS (1/1)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
