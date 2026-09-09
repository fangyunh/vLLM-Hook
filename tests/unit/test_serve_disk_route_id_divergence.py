"""No-GPU regression test for the SERVE disk-route id-divergence failure (Task 12, fix ``888baba``).

Under ``vllm serve`` the external ``request_id`` is rewritten to an INTERNAL ``{external}-{random8}``
(vLLM ``InputProcessor.assign_request_id``, on by default). The per-request DISK route registers the
route under the EXTERNAL id (``route_to_disk`` at request-start), but the drain sees the INTERNAL id on
the rows (``LayerEntry.req_id``) and on the finish signal. Before the fix the disk demux / finish /
offload keyed on an EXACT req_id match, so the internal id never matched the external route: the rows
fell to the HOST index, NOTHING staged to disk, the offload was never submitted, and the client's
disk-confirm timed out (job 577305 — RPC route + abort + mid-serving all PASSED; only serve DISK hung).

The fix adds ``_match_disk_route`` (exact OR ``{ext}-`` prefix, mirroring
``workers/_common.iter_matching_req_ids``) and keys every disk hop on the resolved EXTERNAL id.

This test drives a REAL ``OffLoopRingDrain(per_request=True)`` on a CPU ring (streams/events are
no-ops, so the full consumer/queue/demux/offload machinery runs with no GPU) with a REAL
``OffloadProcess`` (default ``copytree`` transfer to a tmp dest — not a fake fn), and a DIVERGENT id:
route under the EXTERNAL ``"r0"``, demux rows whose ``LayerEntry.req_id`` is the INTERNAL
``"r0-a1b2c3d4"``, finish under the internal id.

  * GREEN (``test_serve_disk_route_divergent_id_delivers``): with the committed fix the request STAGES
    to disk (never the host index), the offload delivers, ``load_multilayer_ring_artifact(dest)``
    reconstructs it byte-identically, and residency drops to 0.
  * RED  (``test_serve_disk_route_divergent_id_strands_under_exact_match``): monkeypatching
    ``_match_disk_route`` back to EXACT-only reproduces the pre-fix STRAND — the divergent id never
    matches the external route, so the request lands in the host index, nothing stages to disk, no
    offload fires, and no client file is delivered. This proves the GREEN assertions discriminate the
    fix from the bug — i.e. this test would have caught the serve failure with no GPU.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_serve_disk_route_id_divergence.py -q
"""
import os
import time

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact

# The DIVERGENT pair the serve failure produced: the router registered the route under the EXTERNAL
# id, while the drain saw the INTERNAL '{external}-{random8}' on the rows and the finish signal.
_EXT = "r0"
_INT = "r0-a1b2c3d4"


def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


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


def _step(ring, hs_bufs, drain, internal_id, n, layers, mode, route_key, exp, step_tag):
    """One engine step for ONE request whose ring rows carry ``LayerEntry.req_id=internal_id`` (what
    the drain sees under serve), while ``exp`` accumulates under ``route_key`` (the EXTERNAL/route id
    the delivered sidecar ends up keyed by after the fix stages via the resolved external id). Writes
    step-unique deterministic data so a mis-routed / torn / cross-step read would be caught."""
    s = _ring_reserve_or_block(ring, n, None)
    phys = ring.physical_slots(s, n)
    entries = []
    for L in layers:
        hidden = hs_bufs[L].shape[1]
        base = (hash((internal_id, L, step_tag)) % 997) * 1000 + 1   # step-unique, non-zero
        data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(
            hs_bufs[L].dtype)
        for j, p in enumerate(phys):
            hs_bufs[L][p] = data[j]
        entries.append(LayerEntry(str(internal_id), L, s, n, mode))
        exp.setdefault(route_key, {}).setdefault(L, []).append(data)
    drain.enqueue(entries, s, n, None)


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


def _wait_disk_residency_at_least_one(drain, timeout=5.0):
    deadline = time.monotonic() + timeout
    while drain.disk_residency() < 1 and time.monotonic() < deadline:
        time.sleep(0.002)
    return drain.disk_residency()


def _assert_disk_reconstructs(dest, route_key, exp_for_key):
    """``load_multilayer_ring_artifact(dest)`` reconstructs EXACTLY the one request keyed by the
    EXTERNAL/route id (the id the fixed disk path staged under), byte-identical to a hand-built cat of
    its per-step blocks per layer (append/step order)."""
    out = load_multilayer_ring_artifact(dest)
    assert set(out) == {route_key}, f"delivered {dest} reconstructed reqs {set(out)} != {{{route_key}}}"
    assert set(out[route_key]) == set(exp_for_key), (
        f"{route_key}: delivered layers {set(out[route_key])} != expected {set(exp_for_key)}")
    for L, blocks in exp_for_key.items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
        assert torch.equal(out[route_key][L], want), f"{route_key}/{L} byte mismatch"


# ------------------------------------------------------------------ GREEN ---
def test_serve_disk_route_divergent_id_delivers(tmp_path):
    """With the committed ``_match_disk_route`` fix, a disk route registered under the EXTERNAL id
    correctly claims rows/finish arriving under the INTERNAL ``{ext}-{rand}`` id: the request stages
    to disk (never the host index), the REAL OffloadProcess delivers it, it reconstructs
    byte-identically, and residency drops to 0."""
    op = OffloadProcess()             # REAL offload: thread backend, default copytree transfer
    hidden, R, dtype = 4, 512, torch.float32     # roomy ring: no wrap, isolate the id-match logic
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    dest = str(tmp_path / "delivered" / "r0")
    drain.route_to_disk(_EXT, dest)   # router registers the route under the EXTERNAL id (request-start)

    exp = {}
    # Rows + finish arrive under the INTERNAL id (serve's InputProcessor rewrite) -- the divergence.
    _step(ring, hs_bufs, drain, _INT, 3, [1, 2, 3], "all_tokens", _EXT, exp, "s1")
    _step(ring, hs_bufs, drain, _INT, 1, [1, 2, 3], "all_tokens", _EXT, exp, "s2")
    _step(ring, hs_bufs, drain, _INT, 1, [1, 2, 3], "all_tokens", _EXT, exp, "s3")
    _wait_drained(ring)

    # STAGES TO DISK (pre-finish): the divergent internal id matched the external route, so its rows
    # went to per-request disk staging (residency>=1) and NOTHING fell to the host index.
    assert _wait_disk_residency_at_least_one(drain) >= 1, (
        "divergent-id request did not stage to disk (rows fell to the host index)")
    assert index.live_req_ids() == set(), (
        "divergent-id request leaked into the host index instead of staging to disk")

    drain.enqueue_finish(_INT)        # finish signal is the INTERNAL id too
    _wait_drained(ring)

    # DELIVERS: the offload was submitted + confirmed under the EXTERNAL id (submit only happens for
    # the disk route in _handle_finish), residency freed, host index still clean.
    assert op.wait(_EXT, timeout=5.0) is True, "divergent-id disk request never confirmed delivered"
    assert _wait_disk_residency_zero(drain) == 0, "disk staging not freed after delivery"
    assert index.live_req_ids() == set(), "a disk-routed request leaked into the host index"
    _assert_disk_reconstructs(dest, _EXT, exp[_EXT])

    drain.stop()
    op.close()


# -------------------------------------------------------------------- RED ---
def test_serve_disk_route_divergent_id_strands_under_exact_match(tmp_path, monkeypatch):
    """RED discriminator: monkeypatch ``_match_disk_route`` back to EXACT-only (the pre-fix rule).
    The divergent internal id then never matches the external route, so the request STRANDS in the
    host index -- no disk staging, no offload submit, no delivered file. This is exactly the serve
    failure (job 577305) the fix closed; the GREEN test above proves the fix delivers, and this proves
    the regression test discriminates the fix from the bug."""
    import vllm_hook_plugins.graph.ring_drain_hs as rd
    monkeypatch.setattr(rd, "_match_disk_route",
                        lambda rid, keys: str(rid) if str(rid) in keys else None)

    op = OffloadProcess()
    hidden, R, dtype = 4, 512, torch.float32
    layer_ids = (1, 2, 3)
    ring, hs_bufs, drain, index = _build(R, hidden, layer_ids, dtype, str(tmp_path), op)
    drain.start()
    dest = str(tmp_path / "delivered" / "r0")
    drain.route_to_disk(_EXT, dest)

    exp = {}
    _step(ring, hs_bufs, drain, _INT, 3, [1, 2, 3], "all_tokens", _EXT, exp, "s1")
    _step(ring, hs_bufs, drain, _INT, 1, [1, 2, 3], "all_tokens", _EXT, exp, "s2")
    _wait_drained(ring)

    # PRE-FIX STRAND (pre-finish): the exact match failed, so nothing staged to disk and the rows fell
    # to the host index under the INTERNAL id -- the bug.
    assert drain.disk_residency() == 0, "exact-match must NOT stage the divergent id to disk"
    assert _INT in index.live_req_ids(), "stranded rows should sit in the host index under the internal id"
    assert _EXT not in index.live_req_ids(), "nothing should be keyed under the external route id"

    drain.enqueue_finish(_INT)
    _wait_drained(ring)

    # No offload ever fired (nothing was disk-finalized) and no client file was delivered -> the
    # serve disk-confirm would time out (the observed hang).
    assert op.wait(_EXT, timeout=0.5) is False, "exact-match must not deliver a disk file"
    assert op.poll_done() == [], "no offload transfer should have been submitted under exact match"
    assert not os.path.exists(dest), "no delivered dest dir under the pre-fix strand"
    # The stranded request is (mis)handled by the HOST-index finish path, keyed by the internal id.
    popped = dict(index.pop_deliverable())
    assert _INT in popped, "stranded request should surface in the host-index deliverables (the strand)"
    assert _EXT not in popped

    drain.stop()
    op.close()
