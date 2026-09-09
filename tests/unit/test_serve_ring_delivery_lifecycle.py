"""No-GPU tests for the serve-delivery LIFECYCLE of the off-loop HS capture-ring per-request path
(Task 12 part A): block-until-held (RPC + disk-confirm), abort/disconnect cleanup (residency -> 0),
the request-start sink-gate + PROFILE_MODE/RING_PER_REQUEST guard, and the offload source unlink.

On CPU (device="cpu") the ring's streams/events are no-ops, so the FULL consumer/queue/demux/offload
machinery runs without a GPU. The block-until-held / confirm poll loops are driven against a FAKE
async engine whose collective_rpc returns canned per-method responses; the abort cleanup drives a REAL
OffLoopRingDrain + a real ProbeHiddenStatesWorker mixin instance.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_serve_ring_delivery_lifecycle.py -q
"""
import asyncio
import os
import pickle
import time
from types import SimpleNamespace

import pytest
import torch
import zstandard as zstd

from vllm_hook_plugins import _hook_plugin as hp
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker


# ============================================================ shared CPU-drain harness ===
def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


def _fake_offload():
    """OffloadProcess (thread backend; an injected fn forces thread) whose transfer records each call
    AND does a real copytree so a delivered dest is reconstructable."""
    calls = []

    def fake_transfer(src, dest):
        import shutil
        calls.append((src, dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    return OffloadProcess(transfer_fn=fake_transfer), calls


def _build(tmp, offload, layer_ids=(1, 2), hidden=4, R=512, dtype=torch.float32):
    ring = GpuCaptureRing(row_bytes=hidden * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=R, device="cpu", dtype=dtype, row_shape=(hidden,))
    hs_bufs = {L: _hs_buf(R, hidden, dtype) for L in layer_ids}
    header = {"dtype": _torch_dtype_name(dtype), "row_shape": [hidden], "hidden": hidden}
    index = PerRequestIndex()
    drain = OffLoopRingDrain(
        ring, [(L, hs_bufs[L]) for L in layer_ids], os.path.join(tmp, "run"), header,
        per_request=True, index=index, offload=offload, disk_base=os.path.join(tmp, "staging"))
    return ring, hs_bufs, drain, index


def _pr_step(ring, hs_bufs, drain, rid, n, layers, mode, exp, tag):
    s = _ring_reserve_or_block(ring, n, None)
    phys = ring.physical_slots(s, n)
    entries = []
    for L in layers:
        hidden = hs_bufs[L].shape[1]
        base = (hash((rid, L, tag)) % 997) * 1000 + 1
        data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base).to(hs_bufs[L].dtype)
        for j, p in enumerate(phys):
            hs_bufs[L][p] = data[j]
        entries.append(LayerEntry(str(rid), L, s, n, mode))
        exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, s, n, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0, f"consumer did not drain: pending={ring.pending_rows()}"


def _worker(drain):
    """A real ProbeHiddenStatesWorker mixin instance wired to `drain` (the collective_rpc entry
    points are plain methods, callable directly)."""
    class _W(ProbeHiddenStatesWorker):
        pass
    w = _W()
    w._hs_drain = drain
    w._conf = {"hidden_size": 4, "num_layers": 2}
    return w


# ========================================================= (a) ABORT cleanup -> residency 0 ===
def test_abort_frees_host_index_before_finish(tmp_path):
    """A request whose rows are staged but which is ABORTED before finish leaves a live host-index
    entry (never finished, never retrieved). clear_ring_request frees it -> host index back to 0."""
    op, _ = _fake_offload()
    ring, hs_bufs, drain, index = _build(str(tmp_path), op)
    drain.start()
    exp = {}
    _pr_step(ring, hs_bufs, drain, "R", 2, [1, 2], "all_tokens", exp, "s1")
    _pr_step(ring, hs_bufs, drain, "R", 1, [1, 2], "all_tokens", exp, "s2")
    _wait_drained(ring)
    assert index.live_req_ids() == {"R"}, "request should be live in the index pre-abort"

    w = _worker(drain)
    w.clear_ring_request("R")                      # <-- abort/disconnect cleanup
    assert index.live_req_ids() == set(), "aborted request leaked in the host index"
    assert not (getattr(w, "_ring_perreq_stash", None) or {}), "no stash entry should remain"
    drain.stop()
    op.close()


def test_abort_after_finish_no_deliverable_keyerror(tmp_path):
    """The dangerous window: the consumer marked the request finished (it is in BOTH _entries and
    _deliverable) but the client aborted before rpc_raw popped it. clear_ring_request must remove it
    from BOTH -- a bare index.free() would strand it in _deliverable and a later pop_deliverable would
    KeyError. A CONCURRENT finished request S must survive (re-stashed, delivered to its own caller)."""
    op, _ = _fake_offload()
    ring, hs_bufs, drain, index = _build(str(tmp_path), op)
    drain.start()
    exp = {}
    for tag in ("s1", "s2"):
        _pr_step(ring, hs_bufs, drain, "R", 1, [1, 2], "all_tokens", exp, tag)
        _pr_step(ring, hs_bufs, drain, "S", 1, [1, 2], "all_tokens", exp, tag)
    drain.enqueue_finish("R")
    drain.enqueue_finish("S")
    _wait_drained(ring)
    # Both finished -> both in _entries and _deliverable.
    assert index.live_req_ids() == {"R", "S"}

    w = _worker(drain)
    w.clear_ring_request("R")                      # abort R
    # R gone everywhere; S survived, re-stashed as bytes for its own delivery.
    assert index.live_req_ids() == set(), "R (and the drained S) must be freed from _entries"
    stash = w._ring_perreq_stash
    assert "R" not in stash and "S" in stash and isinstance(stash["S"], (bytes, bytearray))
    # No KeyError: _deliverable was drained by the bulk-pop, not left pointing at a freed entry.
    assert index.pop_deliverable() == []
    drain.stop()
    op.close()


def test_abort_frees_disk_staging_residency_zero(tmp_path):
    """A DISK-routed request aborted before finish holds open per-request staging (disk_residency 1).
    SINGLE-OWNER dir lifecycle (Task 12 dir-race fix): ``clear_ring_request`` only MARKS it aborted
    (the engine thread must NOT rmtree a dir the consumer might still be demuxing into); the CONSUMER
    reclaims it on the request's ``_Finish`` (real serve: ``finished_req_ids`` includes the abort) ->
    residency 0, source dir gone, WITHOUT offloading (an aborted request is not delivered)."""
    op, calls = _fake_offload()
    ring, hs_bufs, drain, index = _build(str(tmp_path), op)
    drain.start()
    drain.route_to_disk("D", str(tmp_path / "delivered" / "D"))
    exp = {}
    _pr_step(ring, hs_bufs, drain, "D", 2, [1, 2], "all_tokens", exp, "s1")
    _pr_step(ring, hs_bufs, drain, "D", 1, [1, 2], "all_tokens", exp, "s2")
    _wait_drained(ring)
    assert drain.disk_residency() == 1, "disk-routed request should hold staging pre-abort"
    src_dir = drain._disk_staging["D"].run_dir
    assert os.path.isdir(src_dir)

    w = _worker(drain)
    w.clear_ring_request("D")                      # abort MARK (single-owner: consumer reclaims)
    assert "D" not in drain._disk_routed and "D" in drain._disk_aborted, "abort must MARK, not route"
    assert drain.disk_residency() == 1 and os.path.isdir(src_dir), (
        "abort must NOT rmtree the live staging dir on the engine thread")

    drain.enqueue_finish("D")                      # consumer reclaims on the abort's _Finish
    deadline = time.monotonic() + 5.0
    while drain.disk_residency() > 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    assert drain.disk_residency() == 0, "disk staging not freed by consumer reclaim"
    assert not os.path.exists(src_dir), "aborted staging source dir not reclaimed"
    assert calls == [], "an aborted disk request must NOT be offloaded/delivered"
    assert index.live_req_ids() == set()
    drain.stop()
    op.close()


# ========================================================= (d) offload SOURCE unlink after confirm ===
def test_confirm_unlinks_server_source_keeps_client_dest(tmp_path):
    """After a disk-routed request is delivered, confirm_ring_delivery returns True AND unlinks the
    SERVER-side staging source (bounded live NVMe) while keeping the durable CLIENT dest. Idempotent."""
    op, calls = _fake_offload()
    ring, hs_bufs, drain, index = _build(str(tmp_path), op)
    drain.start()
    dest = str(tmp_path / "delivered" / "D")
    drain.route_to_disk("D", dest)
    exp = {}
    _pr_step(ring, hs_bufs, drain, "D", 3, [1, 2], "all_tokens", exp, "s1")
    _pr_step(ring, hs_bufs, drain, "D", 1, [1, 2], "all_tokens", exp, "s2")
    drain.enqueue_finish("D")
    _wait_drained(ring)
    assert op.wait("D", timeout=5.0) is True, "D never delivered"
    src_dir = drain._disk_delivered_src.get("D")
    assert src_dir and os.path.isdir(src_dir), "server source should exist until confirm"

    w = _worker(drain)
    assert w.confirm_ring_delivery("D", 0.0) is True         # confirm + unlink
    assert not os.path.exists(src_dir), "server-side source not unlinked after confirm"
    # Client dest survives + reconstructs byte-identical.
    out = load_multilayer_ring_artifact(dest)
    assert set(out) == {"D"}
    for L, blocks in exp["D"].items():
        want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
        assert torch.equal(out["D"][L], want)
    assert w.confirm_ring_delivery("D", 0.0) is True          # idempotent (source already gone)
    drain.stop()
    op.close()


# ================================================= (b) BLOCK-UNTIL-HELD poll (RPC + disk confirm) ===
def _probes_bytes(payload):
    return zstd.ZstdCompressor(level=1).compress(pickle.dumps(payload))


class _PollEngine:
    """Fake async engine: collective_rpc returns [None]*none_first then [ready] for `method`."""
    def __init__(self, method, ready, none_first):
        self.method = method
        self.ready = ready
        self.none_first = none_first
        self.n = 0

    async def collective_rpc(self, method, args=()):
        if method != self.method:
            return [None]
        self.n += 1
        return [self.ready] if self.n > self.none_first else [None]


def test_block_until_held_rpc_returns_once_available():
    payload = {"hs_cache": {1: {"hidden_states": torch.zeros(2, 4), "layer_num": 1}}, "config": {}}
    eng = _PollEngine("get_ring_per_request", _probes_bytes(payload), none_first=3)
    got = asyncio.run(hp._await_ring_per_request(eng, "r1"))
    assert eng.n >= 4, "should have polled past the None replies"
    assert isinstance(got, dict) and set(got["hs_cache"]) == {1}


def test_block_until_held_rpc_times_out_loud(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_HOOK_RING_DELIVER_TIMEOUT_S", "0.05")
    eng = _PollEngine("get_ring_per_request", ready=_probes_bytes({}), none_first=10**9)  # never ready
    t0 = time.monotonic()
    got = asyncio.run(hp._await_ring_per_request(eng, "rZ"))
    assert got is None, "must give up (not hang) on timeout"
    assert time.monotonic() - t0 < 5.0, "timeout budget must be honored, not ignored"
    assert "BLOCK-UNTIL-HELD TIMEOUT" in capsys.readouterr().out


def test_disk_confirm_returns_true_once_landed():
    eng = _PollEngine("confirm_ring_delivery", ready=True, none_first=0)
    # none_first counts the not-True replies; emulate False-then-True by returning False first.
    eng.ready = True

    class _CE(_PollEngine):
        async def collective_rpc(self, method, args=()):
            if method != "confirm_ring_delivery":
                return [None]
            self.n += 1
            return [True] if self.n > 3 else [False]
    ce = _CE("confirm_ring_delivery", True, 3)
    assert asyncio.run(hp._await_ring_disk_confirm(ce, "d1")) is True
    assert ce.n >= 4


def test_disk_confirm_times_out_loud(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_HOOK_RING_DELIVER_TIMEOUT_S", "0.05")

    class _NeverEngine:
        async def collective_rpc(self, method, args=()):
            return [False]
    got = asyncio.run(hp._await_ring_disk_confirm(_NeverEngine(), "dZ"))
    assert got is False
    assert "DISK-CONFIRM TIMEOUT" in capsys.readouterr().out


# ============================================================= (c) GUARDS ===
def test_profile_ring_conflict_warns_once(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_HOOK_PROFILE_MODE", "1")
    monkeypatch.setenv("VLLM_HOOK_RING_PER_REQUEST", "1")
    if hasattr(hp._warn_profile_ring_conflict, "_warned"):
        del hp._warn_profile_ring_conflict._warned
    hp._warn_profile_ring_conflict()
    out1 = capsys.readouterr().out
    assert "PROFILE_MODE=1 AND VLLM_HOOK_RING_PER_REQUEST=1" in out1
    hp._warn_profile_ring_conflict()                  # latched: warns at most once
    assert "misconfiguration" not in capsys.readouterr().out


def test_profile_ring_conflict_silent_when_not_both(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_HOOK_PROFILE_MODE", "1")
    monkeypatch.delenv("VLLM_HOOK_RING_PER_REQUEST", raising=False)
    if hasattr(hp._warn_profile_ring_conflict, "_warned"):
        del hp._warn_profile_ring_conflict._warned
    hp._warn_profile_ring_conflict()
    assert capsys.readouterr().out == "", "must not warn unless BOTH envs are set"


# ----- (c) sink-gate: the request-start ring router only arms on a non-drop/non-disk sink -----
class _RecEngine:
    """Fake async engine recording every collective_rpc method; canned per-method replies."""
    def __init__(self, hidden=1536, heads=12, kv=12, n_layers=28):
        tc = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv,
                             hidden_size=hidden, num_hidden_layers=n_layers)
        self.model_config = SimpleNamespace(hf_text_config=tc)
        self.methods = []

    async def collective_rpc(self, method, args=()):
        self.methods.append(method)
        if method in ("route_ring_to_disk", "confirm_ring_delivery"):
            return [True]
        if method == "get_ring_per_request":
            return [_probes_bytes({"hs_cache": {}, "config": {}})]
        return [None]


def _drive_generate(monkeypatch, eng, extra, P=2048):
    """Drive _patched_generate to completion for ONE finished request; return the recorded methods."""
    monkeypatch.setattr(hp, "_graph_mode", lambda: True)   # skip the eager install_hooks RPC

    async def _fake_orig(self, prompt, sampling_params, request_id, **kw):
        yield SimpleNamespace(finished=True, probes=None,
                              prompt_token_ids=list(prompt.prompt_token_ids),
                              outputs=[SimpleNamespace(token_ids=[1, 2])])
    monkeypatch.setattr(hp, "_original_generate", _fake_orig)

    prompt = SimpleNamespace(prompt_token_ids=list(range(P)))
    sp = SimpleNamespace(extra_args=dict(extra), max_tokens=256)

    async def _run():
        async for _ in hp._patched_generate(eng, prompt, sp, "req-1"):
            pass
    asyncio.run(_run())
    return eng.methods


@pytest.fixture
def _ring_env(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_RING_PER_REQUEST", "1")
    monkeypatch.setenv("VLLM_HOOK_ALLOW_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_HOOK_STORAGE_ROUTER", "0")     # isolate the ring gate
    monkeypatch.delenv("VLLM_HOOK_PROFILE_MODE", raising=False)
    monkeypatch.delenv("VLLM_HOOK_SINK", raising=False)
    monkeypatch.delenv("VLLM_HOOK_ROUTER_T_RPC", raising=False)
    monkeypatch.delenv("VLLM_HOOK_ROUTER_T_ANALYZE", raising=False)


def test_sink_gate_drop_suppresses_ring_route(monkeypatch, _ring_env):
    """VLLM_HOOK_SINK=drop -> the request-start router must NOT arm route_ring_to_disk (the finalize
    takes the drop path, so routing+offload would be wasted work + orphaned NVMe staging)."""
    monkeypatch.setenv("VLLM_HOOK_SINK", "drop")
    eng = _RecEngine()
    methods = _drive_generate(monkeypatch, eng,
                              {"output_hidden_states": [], "hs_mode": "all_tokens",
                               "hooks_on": "both", "analyzer": "core_reranker"})
    assert "route_ring_to_disk" not in methods, f"drop sink armed the ring route: {methods}"


def test_sink_gate_explicit_disk_suppresses_ring_route(monkeypatch, _ring_env):
    """An explicit save_to_disk=True (a durable-file requirement) -> ring route suppressed; the
    request takes the disk flush path instead (flush_disk), never route_ring_to_disk."""
    eng = _RecEngine()
    methods = _drive_generate(monkeypatch, eng,
                              {"output_hidden_states": [], "hs_mode": "all_tokens",
                               "hooks_on": "both", "analyzer": "core_reranker",
                               "save_to_disk": True})
    assert "route_ring_to_disk" not in methods, f"explicit disk sink armed the ring route: {methods}"
    assert "flush_disk" in methods, "explicit save_to_disk must still write via flush_disk"


def test_sink_gate_control_rpc_sink_arms_ring_route(monkeypatch, _ring_env):
    """CONTROL (non-vacuity): a plain rpc-sink HS-only request large enough to route DISK transport
    DOES arm route_ring_to_disk -- so the two suppress tests above are real gates, not always-off."""
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_RPC", str(256 * 1024))   # force "large" -> disk transport
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_ANALYZE", str(256 * 1024))
    eng = _RecEngine()
    methods = _drive_generate(monkeypatch, eng,
                              {"output_hidden_states": [], "hs_mode": "all_tokens",
                               "hooks_on": "both", "analyzer": "core_reranker"})
    assert "route_ring_to_disk" in methods, f"rpc-sink large HS should arm the ring route: {methods}"
