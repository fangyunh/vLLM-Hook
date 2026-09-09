"""No-GPU tests for the per-request DISK route (Task 9): a request the router marks via
``OffLoopRingDrain.route_to_disk`` streams its demuxed rows to its OWN per-request NVMe run_dir
(per-layer ``hs_layer_<L>.raw`` via a reused ``_MmapLayerWriter`` + a per-request
``hs_ring_meta.jsonl``); on finish the file is msync'd, its sidecar written, and it is handed to the
``OffloadProcess`` for transfer to the client dest. The delivered run_dir must reconstruct
BYTE-IDENTICAL via ``ring_reader.load_multilayer_ring_artifact`` (scoped to that one request), and
the request's staging state must free (``disk_residency() -> 0``) once it is finished + offloaded.

On CPU (device="cpu") the ring's streams/events are no-ops, so this drives the FULL
consumer/queue/demux/offload machinery without a GPU. The OffloadProcess uses a FAKE, injected
``transfer_fn`` (thread backend) that records its calls AND does a real ``copytree`` so the delivered
dest is readable by the reconstruction assert.

Covers:
  * two interleaved disk-routed requests -> each delivered dest reconstructs byte-identical +
    residency -> 0 after the offload confirm;
  * a MIX of one disk-routed + one host-buffer request in the SAME batch -> disk req reconstructs
    from its file, host req from the PerRequestIndex, no cross-contamination;
  * a disk-route straggler (finish never enqueued) finalized + delivered by stop()/finalize_all;
  * the plain-append (VLLM_HOOK_RING_MMAP=0) staging path is byte-identical to the mmap path;
  * the mp OffloadProcess backend is AVAILABLE (opt-in), delivering a real file end-to-end.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_ring_per_request_disk_route.py -q
"""
import os
import shutil
import time

import pytest
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact


# ------------------------------------------------------------------ helpers ---
def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


def _fake_offload():
    """An OffloadProcess (thread backend, injected fn forces thread) whose transfer records each
    call AND performs a real copytree so the delivered dest is reconstructable."""
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


def _pr_step(ring, hs_bufs, drain, reqs, exp, dtype, step_tag, consumer=None):
    """One engine step. reqs = [(req_id, n_rows, [layers], mode)]. Writes STEP-UNIQUE deterministic
    data into each layer's reserved ring slots (so a mis-ordered / cross-request / cross-layer concat
    is caught), records LayerEntrys, O(1)-enqueues, and appends each (req, layer)'s block to
    exp[req][layer] in APPEND (step) order — the order both the reader and the index reconstruct in."""
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
            exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, start_logical, total, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0, f"consumer did not drain: pending={ring.pending_rows()}"


def _wait_disk_residency_zero(drain, timeout=5.0):
    """The consumer pops a request's staging right after the (non-blocking) offload submit, on its
    own thread — poll for it to reach 0 rather than assume the exact interleave with the offload
    thread's transfer completion."""
    deadline = time.monotonic() + timeout
    while drain.disk_residency() > 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    return drain.disk_residency()


def _assert_disk_reconstructs(dest, rid, exp_for_rid):
    """load_multilayer_ring_artifact(dest) reconstructs EXACTLY this one request, byte-identical to a
    hand-built cat of its per-step blocks per layer (in append/step order)."""
    out = load_multilayer_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    assert set(out[rid]) == set(exp_for_rid), (
        f"{rid}: delivered layers {set(out[rid])} != expected {set(exp_for_rid)}")
    for L, blocks in exp_for_rid.items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
        assert torch.equal(out[rid][L], want), f"{rid}/{L} byte mismatch"


# ------------------------------------------------- (1) two interleaved disk reqs ---
def test_disk_route_two_interleaved_reconstructs_and_residency_zero(tmp_path):
    op, calls = _fake_offload()
    hidden, R, dtype = 4, 512, torch.float32      # roomy ring: no wrap, isolate the disk-route logic
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destA = str(tmp_path / "delivered" / "A")
    destB = str(tmp_path / "delivered" / "B")
    drain.route_to_disk("A", destA)               # BOTH routed to disk at "request-start"
    drain.route_to_disk("B", destB)

    exp = {}
    # A: all_tokens over {1,2,3}; B: last_token over {2} only — interleaved every step.
    _pr_step(ring, hs_bufs, drain,
             [("A", 3, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain,
             [("A", 1, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s2")
    _pr_step(ring, hs_bufs, drain,
             [("A", 1, [1, 2, 3], "all_tokens"), ("B", 1, [2], "last_token")], exp, dtype, "s3")
    drain.enqueue_finish("A")                      # A finishes; B keeps going one more step
    _pr_step(ring, hs_bufs, drain, [("B", 1, [2], "last_token")], exp, dtype, "s4")
    drain.enqueue_finish("B")
    _wait_drained(ring)

    # The offload confirm is the real barrier: it fires only after _handle_finish -> submit -> copy.
    assert op.wait("A", timeout=5.0) is True, "A never confirmed delivered"
    assert op.wait("B", timeout=5.0) is True, "B never confirmed delivered"
    assert {os.path.basename(src) for src, _ in calls} == {"A", "B"}, f"offload transfers = {calls}"

    # residency -> 0 after finish + offload; a disk-routed request NEVER touches the host index.
    assert _wait_disk_residency_zero(drain) == 0, "disk staging not freed after offload confirm"
    assert index.live_req_ids() == set(), "a disk-routed request leaked into the host index"
    assert index.pop_deliverable() == []

    _assert_disk_reconstructs(destA, "A", exp["A"])
    _assert_disk_reconstructs(destB, "B", exp["B"])

    drain.stop()
    op.close()


# ------------------------------------------- (2) mix disk-routed + host-buffer ---
def test_disk_and_host_routes_split_in_one_batch(tmp_path):
    """One request routed to DISK, one left on the host-buffer RPC path (Task 5-7), captured in the
    SAME interleaved batch. The disk req reconstructs from its delivered file; the host req from the
    PerRequestIndex — no cross-contamination, and the host path stays byte-identical."""
    op, calls = _fake_offload()
    hidden, R, dtype = 4, 512, torch.float32
    layer_ids = (1, 2)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destD = str(tmp_path / "delivered" / "D")
    drain.route_to_disk("D", destD)               # D -> disk; H -> host (default)

    exp = {}
    _pr_step(ring, hs_bufs, drain,
             [("D", 2, [1, 2], "all_tokens"), ("H", 2, [1, 2], "all_tokens")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain,
             [("D", 1, [1, 2], "all_tokens"), ("H", 1, [1, 2], "all_tokens")], exp, dtype, "s2")
    drain.enqueue_finish("D")
    drain.enqueue_finish("H")
    _wait_drained(ring)
    assert op.wait("D", timeout=5.0) is True
    drain.stop()                                   # joins consumer -> H's finish processed too

    # H never went to disk (no offload call for it); D never went to the host index.
    assert {os.path.basename(src) for src, _ in calls} == {"D"}, f"only D offloaded, got {calls}"
    assert _wait_disk_residency_zero(drain) == 0

    _assert_disk_reconstructs(destD, "D", exp["D"])       # disk req from its file

    popped = {rid: layers for rid, layers in index.pop_deliverable()}
    assert set(popped) == {"H"}, f"host index delivered {set(popped)} != {{H}}"
    for L, blocks in exp["H"].items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
        assert torch.equal(popped["H"][L], want), f"H/{L} host-buffer byte mismatch"
    op.close()


# ------------------------------------------------- (3) disk straggler finalize ---
def test_disk_route_straggler_finalized_on_stop(tmp_path):
    """A disk-routed request that finishes on the FINAL step never gets its _Finish enqueued. Its
    rows are all drained/staged but the file is not finalized/offloaded until stop()/finalize_all
    closes it — which must deliver it byte-identically and free its staging."""
    op, calls = _fake_offload()
    hidden, R, dtype = 4, 512, torch.float32
    ring, hs_bufs, drain, index = _build(R, hidden, (1, 2), dtype, str(tmp_path), op)
    drain.start()
    destS = str(tmp_path / "delivered" / "S")
    drain.route_to_disk("S", destS)

    exp = {}
    _pr_step(ring, hs_bufs, drain, [("S", 3, [1, 2], "all_tokens")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain, [("S", 1, [1, 2], "all_tokens")], exp, dtype, "s2")
    _wait_drained(ring)
    # BEFORE finalize: S is staged (residency 1) but NOT yet finalized/offloaded (no _Finish).
    assert drain.disk_residency() == 1, "straggler should hold staging pre-finalize"
    assert calls == [], "straggler must not be offloaded before finalize"

    drain.stop()                                   # finalize_all -> _handle_finish(S) -> close+submit+free
    assert op.wait("S", timeout=5.0) is True, "straggler never delivered by finalize"
    assert drain.disk_residency() == 0
    _assert_disk_reconstructs(destS, "S", exp["S"])
    op.close()


# --------------------------------------- (3b) the per-request staging default ---
def test_perreq_mmap_default_is_off(tmp_path, monkeypatch):
    """VLLM_HOOK_RING_MMAP unset -> per-request staging uses plain write(), NOT mmap.

    Sibling of test_mmap_default_off (test_hs_ring_mmap_sink.py) for the OTHER of the two sites
    that read this env var. The shared-sink default and the per-request-staging default are set
    independently in the code, so flipping one and forgetting the other has to fail somewhere.
    """
    monkeypatch.delenv("VLLM_HOOK_RING_MMAP", raising=False)
    op, _ = _fake_offload()
    ring, hs_bufs, drain, index = _build(256, 4, (1, 2, 3), torch.bfloat16, str(tmp_path), op)
    assert drain._perreq_mmap is False, "default (env unset) must be the plain write() path"


# --------------------------------------- (4) plain-append staging == mmap staging ---
def test_disk_route_plain_append_matches_mmap(tmp_path, monkeypatch):
    """VLLM_HOOK_RING_MMAP=0 makes the per-request staging use plain open(ab) writes instead of the
    mmap writer; the delivered artifact must reconstruct byte-identically either way."""
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP", "0")
    op, calls = _fake_offload()
    hidden, R, dtype = 4, 256, torch.bfloat16     # bf16 too (prod dtype, exercises the uint16 path)
    ring, hs_bufs, drain, index = _build(R, hidden, (1, 2, 3), dtype, str(tmp_path), op)
    assert drain._perreq_mmap is False, "VLLM_HOOK_RING_MMAP=0 should disable the per-req mmap"
    drain.start()
    destP = str(tmp_path / "delivered" / "P")
    drain.route_to_disk("P", destP)
    exp = {}
    _pr_step(ring, hs_bufs, drain, [("P", 4, [1, 2, 3], "all_tokens")], exp, dtype, "s1")
    _pr_step(ring, hs_bufs, drain, [("P", 1, [1, 2, 3], "all_tokens")], exp, dtype, "s2")
    drain.enqueue_finish("P")
    _wait_drained(ring)
    assert op.wait("P", timeout=5.0) is True
    assert _wait_disk_residency_zero(drain) == 0
    _assert_disk_reconstructs(destP, "P", exp["P"])
    drain.stop()
    op.close()


# --------------------------------------------- (5) mp OffloadProcess is AVAILABLE ---
def test_offload_mp_backend_available(tmp_path):
    """The mp (child-process) OffloadProcess backend must at least be AVAILABLE (opt-in). Deliver a
    real single file through a spawned child end-to-end; skip only if spawn is unavailable here."""
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(4096))
    dest = tmp_path / "out" / "dest.bin"
    try:
        op = OffloadProcess(use_process=True)     # default transfer (picklable) -> real mp child
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"mp offload backend unavailable in this env: {e!r}")
    try:
        assert op.alive(), "mp child not alive after start"
        op.submit("mp-req", str(src), str(dest))
        assert op.wait("mp-req", timeout=60.0) is True, "mp child never confirmed the transfer"
        assert dest.read_bytes() == src.read_bytes()
    finally:
        op.close()
