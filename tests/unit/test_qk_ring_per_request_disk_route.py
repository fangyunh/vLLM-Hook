"""No-GPU tests for the QK per-request DISK route (Task 13 — the QK port of
test_ring_per_request_disk_route). A request the router marks via ``OffLoopQKRingDrain.route_to_disk``
streams its demuxed q + k rows to its OWN per-request NVMe run_dir (per-layer ``qk_q_layer_<L>.raw`` +
``qk_k_layer_<L>.raw`` via reused ``_MmapLayerWriter``s + a per-request ``qk_ring_meta.jsonl``); on
finish the files are msync'd, the sidecar written, and it is handed to the ``OffloadProcess`` for
transfer to the client dest. The delivered run_dir must reconstruct BYTE-IDENTICAL via
``ring_reader.load_multilayer_qk_ring_artifact`` (scoped to that one request), and the request's
staging must free (``disk_residency() -> 0``) once finished + offloaded.

On CPU (device="cpu") the ring's streams/events are no-ops, so this drives the FULL
consumer/queue/demux/offload machinery without a GPU. The OffloadProcess uses a FAKE injected
``transfer_fn`` (thread backend) that records its calls AND does a real ``copytree`` so the delivered
dest is readable by the reconstruction assert.

Covers: two interleaved disk-routed requests (all_tokens + last_token) -> each delivered dest
reconstructs byte-identical + residency -> 0; a MIX of one disk + one host request -> no
cross-contamination; a disk straggler finalized by stop()/finalize_all; the plain-append
(VLLM_HOOK_RING_MMAP=0) staging path is byte-identical to mmap.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_qk_ring_per_request_disk_route.py -q
"""
import os
import shutil
import time

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import _torch_dtype_name
from vllm_hook_plugins.graph.ring_drain_qk import OffLoopQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import QKStepEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_qk_ring_artifact

_QDIM, _KDIM = 8, 4


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
    """One engine step. reqs = [(rid, n, [layers], mode, emit_q)]. Reserves the shared ring, writes
    STEP-/req-/layer-UNIQUE q + k into the reserved physical slots, records QKStepEntrys (K every step;
    q whole span / last row / none), O(1)-enqueues, and records the expected q/k_full/prefix_ends."""
    entries = []
    start_logical = None
    total = 0
    for (rid, n, layers, mode, emit_q) in reqs:
        s = _ring_reserve_or_block(ring, n, drain)
        if start_logical is None:
            start_logical = s
        total += n
        abs_end = state.get(rid, 0) + n
        state[rid] = abs_end
        phys = ring.physical_slots(s, n)
        for L in layers:
            qbase = (hash((rid, L, tag, "q")) % 991) * 1000 + 1
            kbase = (hash((rid, L, tag, "k")) % 991) * 1000 + 1
            q_step = (torch.arange(n * _QDIM, dtype=torch.float32).reshape(n, _QDIM) + qbase
                      ).to(q_bufs[L].dtype)
            k_step = (torch.arange(n * _KDIM, dtype=torch.float32).reshape(n, _KDIM) + kbase
                      ).to(k_bufs[L].dtype)
            for j, p in enumerate(phys):
                q_bufs[L][p] = q_step[j]
                k_bufs[L][p] = k_step[j]
            if mode == "all_tokens":
                q_start, q_rows, q_emit = s, n, q_step
            elif emit_q:
                q_start, q_rows, q_emit = s + n - 1, 1, q_step[-1:]
            else:
                q_start, q_rows, q_emit = -1, 0, None
            prefix_end = abs_end if emit_q else -1
            entries.append(QKStepEntry(str(rid), int(L), s, n, q_start, q_rows, prefix_end, 0))
            e = exp.setdefault(str(rid), {}).setdefault(int(L), {"q": [], "k": [], "ends": []})
            e["k"].append(k_step)
            if q_emit is not None:
                e["q"].append(q_emit)
            if prefix_end >= 0:
                e["ends"].append(prefix_end)
    drain.enqueue(entries, start_logical, total, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0, f"consumer did not drain: pending={ring.pending_rows()}"


def _wait_disk_residency_zero(drain, timeout=5.0):
    deadline = time.monotonic() + timeout
    while drain.disk_residency() > 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    return drain.disk_residency()


def _assert_disk_qk_reconstructs(dest, rid, exp_for_rid):
    """load_multilayer_qk_ring_artifact(dest) reconstructs EXACTLY this one request, byte-identical to
    the hand-built q / growing-prefix k_all per layer."""
    out = load_multilayer_qk_ring_artifact(dest)
    assert set(out) == {rid}, f"delivered {dest} reconstructed reqs {set(out)} != {{{rid}}}"
    assert set(out[rid]) == set(exp_for_rid), (
        f"{rid}: delivered layers {set(out[rid])} != {set(exp_for_rid)}")
    for L, e in exp_for_rid.items():
        k_full = e["k"][0] if len(e["k"]) == 1 else torch.cat(e["k"], 0)
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        rec = out[rid][L]
        assert torch.equal(rec["q"], q), f"{rid}/{L} q byte mismatch"
        assert rec["k_prefix_ends"] == e["ends"], f"{rid}/{L} prefix_ends {rec['k_prefix_ends']}"
        assert torch.equal(rec["k_full"], k_full), f"{rid}/{L} k_full byte mismatch"
        for i, L2 in enumerate(e["ends"]):
            assert torch.equal(rec["k_all"][i], k_full[:L2]), f"{rid}/{L} k_all[{i}] mismatch"


# ------------------------------------------------- (1) two interleaved disk reqs ---
def test_disk_route_two_interleaved_reconstructs_and_residency_zero(tmp_path):
    op, calls = _fake_offload()
    R, dtype = 512, torch.float32
    layer_ids = (0, 1, 2)
    ring, q_bufs, k_bufs, drain, index = _build(R, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destA = str(tmp_path / "delivered" / "A")
    destB = str(tmp_path / "delivered" / "B")
    drain.route_to_disk("A", destA)
    drain.route_to_disk("B", destB)

    exp, state = {}, {}
    # A: all_tokens over {0,1,2}; B: last_token over {1} -- interleaved.
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("A", 3, [0, 1, 2], "all_tokens", True), ("B", 2, [1], "last_token", True)],
                state, exp, "s1")
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("A", 1, [0, 1, 2], "all_tokens", True), ("B", 1, [1], "last_token", True)],
                state, exp, "s2")
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("A", 1, [0, 1, 2], "all_tokens", True), ("B", 1, [1], "last_token", True)],
                state, exp, "s3")
    drain.enqueue_finish("A")
    _pr_qk_step(ring, q_bufs, k_bufs, drain, [("B", 1, [1], "last_token", True)], state, exp, "s4")
    drain.enqueue_finish("B")
    _wait_drained(ring)

    assert op.wait("A", timeout=5.0) is True, "A never confirmed delivered"
    assert op.wait("B", timeout=5.0) is True, "B never confirmed delivered"
    assert {os.path.basename(src) for src, _ in calls} == {"A", "B"}, f"offloads = {calls}"
    assert _wait_disk_residency_zero(drain) == 0, "disk staging not freed after offload confirm"
    assert index.live_req_ids() == set(), "a disk-routed request leaked into the host index"
    assert index.pop_deliverable_qk() == []

    _assert_disk_qk_reconstructs(destA, "A", exp["A"])
    _assert_disk_qk_reconstructs(destB, "B", exp["B"])
    drain.stop()
    op.close()


# ------------------------------------------- (2) mix disk-routed + host-buffer ---
def test_disk_and_host_routes_split_in_one_batch(tmp_path):
    op, calls = _fake_offload()
    R, dtype = 512, torch.float32
    layer_ids = (0, 1)
    ring, q_bufs, k_bufs, drain, index = _build(R, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    destD = str(tmp_path / "delivered" / "D")
    drain.route_to_disk("D", destD)               # D -> disk; H -> host (default)

    exp, state = {}, {}
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("D", 2, [0, 1], "all_tokens", True), ("H", 2, [0, 1], "all_tokens", True)],
                state, exp, "s1")
    _pr_qk_step(ring, q_bufs, k_bufs, drain,
                [("D", 1, [0, 1], "all_tokens", True), ("H", 1, [0, 1], "all_tokens", True)],
                state, exp, "s2")
    drain.enqueue_finish("D")
    drain.enqueue_finish("H")
    _wait_drained(ring)
    assert op.wait("D", timeout=5.0) is True
    drain.stop()                                   # joins consumer -> H's finish processed too

    assert {os.path.basename(src) for src, _ in calls} == {"D"}, f"only D offloaded, got {calls}"
    assert _wait_disk_residency_zero(drain) == 0
    _assert_disk_qk_reconstructs(destD, "D", exp["D"])

    popped = dict(index.pop_deliverable_qk())
    assert set(popped) == {"H"}, f"host index delivered {set(popped)} != {{H}}"
    for L, e in exp["H"].items():
        k_full = e["k"][0] if len(e["k"]) == 1 else torch.cat(e["k"], 0)
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        assert torch.equal(popped["H"][L]["q"], q), f"H/{L} host q mismatch"
        for i, L2 in enumerate(e["ends"]):
            assert torch.equal(popped["H"][L]["k_all"][i], k_full[:L2]), f"H/{L} host k_all mismatch"
    op.close()


# ------------------------------------------------- (3) disk straggler finalize ---
def test_disk_route_straggler_finalized_on_stop(tmp_path):
    op, calls = _fake_offload()
    R, dtype = 512, torch.float32
    ring, q_bufs, k_bufs, drain, index = _build(R, (0, 1), dtype, str(tmp_path), op)
    drain.start()
    destS = str(tmp_path / "delivered" / "S")
    drain.route_to_disk("S", destS)

    exp, state = {}, {}
    _pr_qk_step(ring, q_bufs, k_bufs, drain, [("S", 3, [0, 1], "all_tokens", True)], state, exp, "s1")
    _pr_qk_step(ring, q_bufs, k_bufs, drain, [("S", 1, [0, 1], "all_tokens", True)], state, exp, "s2")
    _wait_drained(ring)
    assert drain.disk_residency() == 1, "straggler should hold staging pre-finalize"
    assert calls == [], "straggler must not be offloaded before finalize"

    drain.stop()                                   # finalize_all -> _handle_finish(S) -> close+submit+free
    assert op.wait("S", timeout=5.0) is True, "straggler never delivered by finalize"
    assert drain.disk_residency() == 0
    _assert_disk_qk_reconstructs(destS, "S", exp["S"])
    op.close()


# --------------------------------------- (3b) the per-request staging default ---
def test_perreq_mmap_default_is_off(tmp_path, monkeypatch):
    """VLLM_HOOK_RING_MMAP unset -> QK per-request staging uses plain write(), NOT mmap.

    The QK half of the pair in test_ring_per_request_disk_route.py. Four sites read this env var
    (shared sink + per-request staging, in each of ring_drain_hs.py and ring_drain_qk.py) and each
    sets its default independently, so each needs its own pin.
    """
    monkeypatch.delenv("VLLM_HOOK_RING_MMAP", raising=False)
    op, _ = _fake_offload()
    ring, q_bufs, k_bufs, drain, index = _build(256, (0, 1, 2), torch.bfloat16, str(tmp_path), op)
    assert drain._perreq_mmap is False, "default (env unset) must be the plain write() path"


# --------------------------------------- (4) plain-append staging == mmap staging ---
def test_disk_route_plain_append_matches_mmap(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_RING_MMAP", "0")
    op, calls = _fake_offload()
    R, dtype = 256, torch.bfloat16                 # bf16 too (prod dtype, exercises the uint16 path)
    ring, q_bufs, k_bufs, drain, index = _build(R, (0, 1, 2), dtype, str(tmp_path), op)
    assert drain._perreq_mmap is False, "VLLM_HOOK_RING_MMAP=0 should disable the per-req mmap"
    drain.start()
    destP = str(tmp_path / "delivered" / "P")
    drain.route_to_disk("P", destP)
    exp, state = {}, {}
    _pr_qk_step(ring, q_bufs, k_bufs, drain, [("P", 4, [0, 1, 2], "all_tokens", True)], state, exp, "s1")
    _pr_qk_step(ring, q_bufs, k_bufs, drain, [("P", 1, [0, 1, 2], "all_tokens", True)], state, exp, "s2")
    drain.enqueue_finish("P")
    _wait_drained(ring)
    assert op.wait("P", timeout=5.0) is True
    assert _wait_disk_residency_zero(drain) == 0
    _assert_disk_qk_reconstructs(destP, "P", exp["P"])
    drain.stop()
    op.close()


if __name__ == "__main__":
    import tempfile
    import pathlib
    for t in (test_disk_route_two_interleaved_reconstructs_and_residency_zero,
              test_disk_and_host_routes_split_in_one_batch,
              test_disk_route_straggler_finalized_on_stop):
        d = tempfile.mkdtemp(prefix="qk_disk_")
        try:
            t(pathlib.Path(d))
            print(f"PASS  {t.__name__}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("VERDICT: PASS")
