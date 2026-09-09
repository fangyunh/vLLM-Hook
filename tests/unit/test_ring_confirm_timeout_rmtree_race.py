"""No-GPU THREADED test for the CONFIRM-TIMEOUT rmtree-vs-offload-read race (Task 13 final-review I1).

The race (distinct from the ABORT dir-lifecycle race in test_ring_dir_lifecycle_race.py -- that one
is a MID-DEMUX abort; this one is a NON-aborted request whose OFFLOAD outruns the confirm budget):

  1. A disk-routed request finishes normally. ``_handle_finish`` finalizes its per-request staging dir,
     records it in ``_disk_delivered_src``, and SUBMITS it to the OffloadProcess (which begins
     ``copytree``-READING the source dir on its worker thread).
  2. The driver's ``_await_ring_disk_confirm`` polls ``offload.wait(..., 0.0)`` up to a ~30 s deadline.
     If the offload copy is slow, confirm TIMES OUT -- and so it NEVER calls ``unlink_delivered_source``,
     leaving ``_disk_delivered_src`` populated.
  3. The request's ``finally`` then runs ``clear_ring_request`` -> ``clear_request_disk``, which pops
     ``_disk_delivered_src`` and (pre-fix) ``rmtree``s the source dir -- WHILE the offload thread is
     still reading it. Result: the client ``dest`` lands a PARTIAL tree and the offload's retries fail
     (source gone).

THE FIX: ``clear_request_disk`` gates the delivered-source rmtree on the offload having SETTLED
(``OffloadProcess.settled(req_id)``: done OR gave-up -> no longer reading). If still in flight it parks
the source in ``_disk_reclaim_pending`` and ``_reclaim_settled_pending`` (run per consumer item + at
``finalize_all``) removes it once the offload settles -- never rmtree'd mid-copy, never leaked.

Covered here:
  * HS drain GREEN: the source SURVIVES the mid-flight clear, the offload then completes to a COMPLETE
    byte-identical dest, and the source is reclaimed once settled (deferred sweep).
  * HS drain RED (seam): forcing ``settled()`` True -- i.e. the pre-fix "always rmtree" behavior --
    removes the source WHILE the transfer is still in flight (the corruption trigger the gate prevents).
  * QK drain GREEN: the twin (``OffLoopQKRingDrain`` shares the code via near-verbatim mirror).

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_confirm_timeout_rmtree_race.py -q
"""
import os
import shutil
import threading
import time

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import (
    OffLoopRingDrain, _torch_dtype_name)
from vllm_hook_plugins.graph.ring_drain_qk import OffLoopQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, QKStepEntry
from vllm_hook_plugins.graph.ring_reader import (
    load_multilayer_ring_artifact, load_multilayer_qk_ring_artifact)
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block


_HIDDEN = 4
_QDIM, _KDIM = 8, 4


# ------------------------------------------------------------------ offload ---
def _slow_offload(max_retries=4, retry_delay=0.02):
    """A real OffloadProcess whose transfer signals ``started`` then blocks on ``release`` BEFORE doing
    the real copy -- so the "offload is mid-flight reading the source" window is deterministic. Once
    released it performs the true copytree/copy, landing a COMPLETE dest (proving the source survived)."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def transfer(src, dest):
        calls.append((src, dest))
        started.set()
        release.wait(timeout=10.0)
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy(src, dest)

    op = OffloadProcess(transfer_fn=transfer, max_retries=max_retries, retry_delay=retry_delay)
    return op, started, release, calls


def _wait_queue(drain, timeout=10.0):
    deadline = time.monotonic() + timeout
    while drain._q.unfinished_tasks > 0 and time.monotonic() < deadline:
        time.sleep(0.002)


# =============================================================== HS drain =====
def _hs_build(R, hidden, layer_ids, dtype, tmp, offload):
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: torch.zeros(R + 1, hidden, dtype=dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    index = PerRequestIndex()
    drain = OffLoopRingDrain(
        ring, [(L, hs_bufs[L]) for L in layer_ids], os.path.join(tmp, "run"), header,
        per_request=True, index=index, offload=offload, disk_base=os.path.join(tmp, "staging"))
    return ring, hs_bufs, drain, index


def _hs_step(ring, hs_bufs, drain, reqs, exp, tag):
    entries, start_logical, total = [], None, 0
    for (rid, n, layers, mode) in reqs:
        s = _ring_reserve_or_block(ring, n, None)
        if start_logical is None:
            start_logical = s
        total += n
        phys = ring.physical_slots(s, n)
        for L in layers:
            hidden = hs_bufs[L].shape[1]
            base = (hash((rid, L, tag)) % 997) * 1000 + 1
            data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(dtype=hs_bufs[L].dtype)
            for j, p in enumerate(phys):
                hs_bufs[L][p] = data[j]
            entries.append(LayerEntry(str(rid), L, s, n, mode))
            exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, start_logical, total, None)


def _assert_hs_reconstructs(dest, rid, exp_for_rid):
    out = load_multilayer_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    assert set(out[rid]) == set(exp_for_rid)
    for L, blocks in exp_for_rid.items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
        assert torch.equal(out[rid][L], want), f"{rid}/{L} byte mismatch"


def test_hs_clear_request_disk_defers_rmtree_while_offload_in_flight(tmp_path):
    """GREEN: a normally-finished disk request R whose offload is mid-flight when a confirm-TIMEOUT's
    clear_request_disk runs. The source dir must NOT be rmtree'd (deferred), the offload then completes
    to a COMPLETE dest, and the deferred sweep reclaims the source once the offload settles."""
    op, started, release, calls = _slow_offload()
    dtype, R, layer_ids = torch.float32, 512, (1, 2, 3)
    ring, hs_bufs, drain, index = _hs_build(R, _HIDDEN, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destR = str(tmp_path / "delivered" / "R")
    drain.route_to_disk("R", destR)

    exp = {}
    _hs_step(ring, hs_bufs, drain, [("R", 3, [1, 2, 3], "all_tokens")], exp, "s1")
    drain.enqueue_finish("R")

    # Offload is in flight (transfer_fn entered, blocked on release) -> finish already recorded
    # _disk_delivered_src + submitted. This is the confirm-timeout window.
    assert started.wait(timeout=5.0), "offload transfer never started (vacuous)"
    src_R = os.path.join(str(tmp_path), "staging", "R")
    assert os.path.isdir(src_R), "delivered source dir should still be present while offload reads it"
    assert op.settled("R") is False, "offload must be in flight (not settled) at this point"

    # The confirm-timeout finally: clear_request_disk lands WHILE the offload is copytree-reading src.
    drain.clear_request_disk("R")

    # THE FIX: the source is NOT rmtree'd (deferred), so the in-flight copytree still has its source.
    assert os.path.isdir(src_R), "clear_request_disk rmtree'd the source out from under the offload!"
    assert "R" in drain._disk_reclaim_pending, "the source should be parked for settled-reclaim"

    # Let the offload finish -> a COMPLETE (not partial) client dest, byte-identical.
    release.set()
    assert op.wait("R", timeout=5.0) is True, "offload never confirmed delivery"
    _assert_hs_reconstructs(destR, "R", exp["R"])

    # Deferred settled-reclaim: drive one more consumer-loop iteration (a benign host request) so the
    # per-item sweep runs; the now-settled source is reclaimed (no leak).
    _hs_step(ring, hs_bufs, drain, [("H", 1, [1, 2, 3], "all_tokens")], {}, "s2")
    _wait_queue(drain)
    deadline = time.monotonic() + 5.0
    while os.path.exists(src_R) and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not os.path.exists(src_R), "deferred settled-reclaim never removed the source (NVMe leak)"
    assert "R" not in drain._disk_reclaim_pending, "pending reclaim entry not cleared after settle"

    drain.stop()
    op.close()


def test_hs_pre_fix_would_rmtree_source_mid_offload_RED(tmp_path):
    """RED seam: forcing OffloadProcess.settled -> True reproduces the PRE-FIX behavior (no gate ->
    always rmtree). clear_request_disk then removes the source dir WHILE the offload is still reading
    it -- exactly the corruption the settled-gate prevents (asserted GREEN in the test above)."""
    op, started, release, calls = _slow_offload()
    dtype, R, layer_ids = torch.float32, 512, (1, 2, 3)
    ring, hs_bufs, drain, index = _hs_build(R, _HIDDEN, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destR = str(tmp_path / "delivered" / "R")
    drain.route_to_disk("R", destR)

    _hs_step(ring, hs_bufs, drain, [("R", 3, [1, 2, 3], "all_tokens")], {}, "s1")
    drain.enqueue_finish("R")
    assert started.wait(timeout=5.0), "offload transfer never started (vacuous)"
    src_R = os.path.join(str(tmp_path), "staging", "R")
    assert os.path.isdir(src_R)

    # Pre-fix behavior: settled() always True -> clear_request_disk rmtrees unconditionally.
    op.settled = lambda rid: True
    drain.clear_request_disk("R")

    # The bug: the source is gone WHILE the offload is still mid-copytree (transfer blocked on release).
    assert not os.path.exists(src_R), (
        "expected the pre-fix path to rmtree the source mid-offload (RED demonstration)")
    assert "R" not in drain._disk_reclaim_pending, "pre-fix path does not defer"

    release.set()   # let the (now source-less) transfer unblock so the worker can drain + shut down
    time.sleep(0.05)
    drain.stop()
    op.close()


# =============================================================== QK drain =====
def _qk_build(R, layer_ids, dtype, tmp, offload):
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


def _qk_step(ring, q_bufs, k_bufs, drain, reqs, state, exp, tag):
    entries, start_logical, total = [], None, 0
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
            e = exp.setdefault(str(rid), {}).setdefault(int(L), {"q": [], "k": []})
            e["k"].append(k_step)
            e["q"].append(q_step)
    drain.enqueue(entries, start_logical, total, None)


def _assert_qk_reconstructs(dest, rid, exp_for_rid):
    out = load_multilayer_qk_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    for L, e in exp_for_rid.items():
        k_full = e["k"][0] if len(e["k"]) == 1 else torch.cat(e["k"], 0)
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        assert torch.equal(out[rid][L]["q"], q), f"{rid}/{L} q byte mismatch"
        assert torch.equal(out[rid][L]["k_full"], k_full), f"{rid}/{L} k_full byte mismatch"


def test_qk_clear_request_disk_defers_rmtree_while_offload_in_flight(tmp_path):
    """QK twin of the HS GREEN test: the near-verbatim OffLoopQKRingDrain shares the settled-gate +
    deferred-reclaim code, so the same race is closed for QK."""
    op, started, release, calls = _slow_offload()
    dtype, R, layer_ids = torch.float32, 512, (0, 1, 2)
    ring, q_bufs, k_bufs, drain, index = _qk_build(R, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destR = str(tmp_path / "delivered" / "R")
    drain.route_to_disk("R", destR)

    exp, state = {}, {}
    _qk_step(ring, q_bufs, k_bufs, drain, [("R", 3, [0, 1, 2])], state, exp, "s1")
    drain.enqueue_finish("R")

    assert started.wait(timeout=5.0), "qk offload transfer never started (vacuous)"
    src_R = os.path.join(str(tmp_path), "staging", "R")
    assert os.path.isdir(src_R)
    assert op.settled("R") is False

    drain.clear_request_disk("R")
    assert os.path.isdir(src_R), "qk clear_request_disk rmtree'd the source out from under the offload!"
    assert "R" in drain._disk_reclaim_pending

    release.set()
    assert op.wait("R", timeout=5.0) is True
    _assert_qk_reconstructs(destR, "R", exp["R"])

    _qk_step(ring, q_bufs, k_bufs, drain, [("H", 1, [0, 1, 2])], state, {}, "s2")
    _wait_queue(drain)
    deadline = time.monotonic() + 5.0
    while os.path.exists(src_R) and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not os.path.exists(src_R), "qk deferred settled-reclaim never removed the source (NVMe leak)"
    assert "R" not in drain._disk_reclaim_pending

    drain.stop()
    op.close()


if __name__ == "__main__":
    import tempfile
    import pathlib
    passed = 0
    for fn in (test_hs_clear_request_disk_defers_rmtree_while_offload_in_flight,
               test_hs_pre_fix_would_rmtree_source_mid_offload_RED,
               test_qk_clear_request_disk_defers_rmtree_while_offload_in_flight):
        d = tempfile.mkdtemp(prefix="confirmrace_")
        try:
            fn(pathlib.Path(d))
            print("PASS ", fn.__name__)
            passed += 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"VERDICT: PASS ({passed}/3)")
