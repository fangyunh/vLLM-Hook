"""No-GPU tests for the request-start delivery ROUTER + profile mode (Task 10).

Ties the per-request delivery pieces together on the DRIVER side:
  * reducibility determination from the chosen analyzer's capability (``_analyzer_reducible``):
    ``attn_tracker`` (ACCEPTS="score") reduces, ``core_reranker`` (ACCEPTS="qk") needs raw,
    ``hidden_states`` reduces only when ``analyzer_spec["reduce"]`` in {mean, norm};
  * the router core (``_decide_ring_route``) -> a ``RouteDecision`` per the §4 route matrix;
  * the drain STAGING each matrix row lands on (host-buffer vs per-request NVMe file) — driving a
    REAL ``OffLoopRingDrain`` on CPU (streams/events are no-ops there), so the router's transport
    choice is proven end-to-end without a GPU;
  * env-configurable thresholds (``VLLM_HOOK_ROUTER_T_RPC`` / ``_T_ANALYZE``, Task 11 calibrates);
  * profile mode (``VLLM_HOOK_PROFILE_MODE=1``) disables Component 2 — the finalize action is
    ``profile_stamp`` (stamp request_done at data-prepared; no analyzer/delivery), regardless of route.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_request_start_router.py -q
"""
import os
import time
from types import SimpleNamespace

import torch

from vllm_hook_plugins import register_plugins
from vllm_hook_plugins import _hook_plugin as hp
from vllm_hook_plugins.graph.delivery_router import decide_route, RouteDecision
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.install_hs import _ring_reserve_or_block
from vllm_hook_plugins.graph.offload_process import OffloadProcess
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.graph.ring_drain_hs import OffLoopRingDrain, _torch_dtype_name
from vllm_hook_plugins.graph.ring_metadata import LayerEntry
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact

register_plugins()   # populate PluginRegistry so ACCEPTS lookups resolve (mirrors hook_client)


# =========================================================================================
# (A) reducibility determination from the analyzer's declared capability
# =========================================================================================
def test_analyzer_reducible_from_capability():
    assert hp._analyzer_reducible("attn_tracker", None) is True         # ACCEPTS="score"
    assert hp._analyzer_reducible("attn_tracker", {"reduce": "none"}) is True
    assert hp._analyzer_reducible("core_reranker", None) is False        # ACCEPTS="qk" -> raw
    assert hp._analyzer_reducible("core_reranker", {"reduce": "mean"}) is False  # capability wins
    assert hp._analyzer_reducible("hidden_states", {"reduce": "none"}) is False
    assert hp._analyzer_reducible("hidden_states", {"reduce": "mean"}) is True
    assert hp._analyzer_reducible("hidden_states", {"reduce": "norm"}) is True
    assert hp._analyzer_reducible("hidden_states", None) is False        # absent reduce -> raw
    # No analyzer named -> reduce-driven (bare hidden_states-style spec).
    assert hp._analyzer_reducible(None, {"reduce": "mean"}) is True
    assert hp._analyzer_reducible(None, None) is False
    # Unknown analyzer -> safe default RAW (never lose data).
    assert hp._analyzer_reducible("does_not_exist", None) is False


# =========================================================================================
# (B) thresholds are ENV-configurable (Task 11 calibrates) with documented defaults
# =========================================================================================
def test_thresholds_env_configurable(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_ROUTER_T_RPC", raising=False)
    monkeypatch.delenv("VLLM_HOOK_ROUTER_T_ANALYZE", raising=False)
    assert hp._ring_route_thresholds() == (hp._DEFAULT_RING_T_RPC, hp._DEFAULT_RING_T_ANALYZE)
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_RPC", "1000")
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_ANALYZE", "2000")
    assert hp._ring_route_thresholds() == (1000, 2000)


# =========================================================================================
# (C) the router core -> a RouteDecision per the §4 matrix (with a fake engine)
# =========================================================================================
def _fake_engine(hidden=1536, heads=12, kv=12, n_layers=28):
    tc = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv,
                         hidden_size=hidden, num_hidden_layers=n_layers)
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=tc))


def _fake_prompt(P):
    return SimpleNamespace(prompt_token_ids=list(range(P)))


def test_decide_ring_route_hs_only_gate():
    eng = _fake_engine()
    # QK-also -> not the ring path (returns None).
    extra = {"output_hidden_states": [], "output_qk": {0: [0]}}
    assert hp._decide_ring_route(eng, _fake_prompt(64), extra, 8) is None
    # Not HS at all -> None.
    assert hp._decide_ring_route(eng, _fake_prompt(64), {"output_qk": {0: [0]}}, 8) is None


def test_decide_ring_route_matrix(monkeypatch):
    # Force a low T so a realistic all_tokens prompt lands "large"; a last_token prompt lands "small".
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_RPC", str(256 * 1024))
    monkeypatch.setenv("VLLM_HOOK_ROUTER_T_ANALYZE", str(256 * 1024))
    eng = _fake_engine()
    small = _fake_prompt(8)      # last_token -> tiny
    large = _fake_prompt(2048)   # all_tokens -> MB

    # reducible (attn_tracker) small -> rpc/inflight ; large -> disk/from_disk
    r = hp._decide_ring_route(
        eng, small, {"output_hidden_states": [], "hs_mode": "last_token",
                     "hooks_on": "prefill", "analyzer": "attn_tracker"}, 0)
    assert r == RouteDecision("rpc", "inflight"), r
    r = hp._decide_ring_route(
        eng, large, {"output_hidden_states": [], "hs_mode": "all_tokens",
                     "hooks_on": "both", "analyzer": "attn_tracker"}, 256)
    assert r == RouteDecision("disk", "from_disk"), r

    # needs-raw (core_reranker) small -> rpc/none ; large -> disk/none
    r = hp._decide_ring_route(
        eng, small, {"output_hidden_states": [], "hs_mode": "last_token",
                     "hooks_on": "prefill", "analyzer": "core_reranker"}, 0)
    assert r == RouteDecision("rpc", "none"), r
    r = hp._decide_ring_route(
        eng, large, {"output_hidden_states": [], "hs_mode": "all_tokens",
                     "hooks_on": "both", "analyzer": "core_reranker"}, 256)
    assert r == RouteDecision("disk", "none"), r


# =========================================================================================
# (D) THE ROUTE MATRIX -> the drain STAGES each row on the expected kind (host-buffer vs file)
#     Real OffLoopRingDrain on CPU: this proves transport -> staging end-to-end, no GPU.
# =========================================================================================
def _hs_buf(R, hidden, dtype):
    return torch.zeros(R + 1, hidden, dtype=dtype)   # +1 sentinel row (never drained)


def _fake_offload():
    calls = []

    def fake_transfer(src, dest):
        import shutil
        calls.append((src, dest))
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy(src, dest)

    return OffloadProcess(transfer_fn=fake_transfer), calls


def _build_drain(tmp, offload, layer_ids=(1, 2), hidden=4, R=512, dtype=torch.float32):
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


def _pr_step(ring, hs_bufs, drain, rid, n, layers, mode, exp, tag):
    s = _ring_reserve_or_block(ring, n, None)
    phys = ring.physical_slots(s, n)
    entries = []
    for L in layers:
        hidden = hs_bufs[L].shape[1]
        base = (hash((rid, L, tag)) % 997) * 1000 + 1
        data = (torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden) + base)
        for j, p in enumerate(phys):
            hs_bufs[L][p] = data[j]
        entries.append(LayerEntry(str(rid), L, s, n, mode))
        exp.setdefault(str(rid), {}).setdefault(L, []).append(data)
    drain.enqueue(entries, s, n, None)


def _wait_drained(ring, timeout=10.0):
    deadline = time.monotonic() + timeout
    while ring.pending_rows() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert ring.pending_rows() == 0


def _apply_route(drain, rid, route, dest):
    """Cross the router's TRANSPORT choice to the drain exactly as _patched_generate does at
    request-start: transport='disk' -> route_to_disk (per-request NVMe file staging); 'rpc' ->
    leave it on the default host-buffer path."""
    if route.transport == "disk":
        drain.route_to_disk(rid, dest)


def test_route_matrix_stages_expected_kind(tmp_path):
    """For every §4 route-matrix row, the router's transport choice makes the drain STAGE the
    request on the expected kind: rpc -> host-buffer (PerRequestIndex); disk -> per-request NVMe
    file. All four routed into ONE interleaved batch, so a mis-staged row is caught."""
    T = 1_000_000
    op, calls = _fake_offload()
    ring, hs_bufs, drain, index = _build_drain(str(tmp_path), op)
    drain.start()

    # (analyzer, reduce, predicted_bytes) -> expected (transport, analyze_where, staging)
    rows = [
        ("attn_tracker",  None,               500_000, "rpc",  "inflight",  "host"),   # reducible small
        ("attn_tracker",  None,             5_000_000, "disk", "from_disk", "file"),   # reducible large
        ("core_reranker", None,               500_000, "rpc",  "none",      "host"),   # rawneeded small
        ("core_reranker", None,             5_000_000, "disk", "none",      "file"),   # rawneeded large
    ]
    rids = ["rA", "rB", "rC", "rD"]
    dests = {}
    exp = {}
    for rid, (name, spec, nbytes, exp_t, exp_aw, exp_stage) in zip(rids, rows):
        reducible = hp._analyzer_reducible(name, spec)
        route = decide_route(nbytes, reducible, T, T)
        assert (route.transport, route.analyze_where) == (exp_t, exp_aw), (rid, route)
        dest = str(tmp_path / "delivered" / rid)
        dests[rid] = (dest, exp_stage)
        _apply_route(drain, rid, route, dest)

    # Interleave two steps across all four, then finish each.
    for tag in ("s1", "s2"):
        for rid in rids:
            _pr_step(ring, hs_bufs, drain, rid, 2, [1, 2], "all_tokens", exp, tag)
    for rid in rids:
        drain.enqueue_finish(rid)
    _wait_drained(ring)
    drain.stop()   # joins consumer + finalizes host stragglers

    disk_rids = [r for r, (_, s) in dests.items() if s == "file"]
    host_rids = [r for r, (_, s) in dests.items() if s == "host"]

    # DISK-routed rows staged to a per-request FILE: offloaded, reconstruct byte-identical, and
    # NEVER entered the host index.
    for rid in disk_rids:
        assert op.wait(rid, timeout=5.0) is True, f"{rid} disk route never delivered"
        dest = dests[rid][0]
        out = load_multilayer_ring_artifact(dest)
        assert set(out) == {rid}, f"{rid} file reconstructed {set(out)}"
        for L, blocks in exp[rid].items():
            want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
            assert torch.equal(out[rid][L], want), f"{rid}/{L} disk byte mismatch"
    assert {os.path.basename(s) for s, _ in calls} == set(disk_rids), \
        f"offload transfers {calls} != disk rows {disk_rids}"

    # RPC-routed rows staged to the HOST buffer (PerRequestIndex): delivered from the index, and
    # NEVER offloaded to disk.
    popped = {rid: layers for rid, layers in index.pop_deliverable()}
    assert set(popped) == set(host_rids), f"host index has {set(popped)} != {set(host_rids)}"
    for rid in host_rids:
        for L, blocks in exp[rid].items():
            want = blocks[0] if len(blocks) == 1 else torch.cat(blocks, 0)
            assert torch.equal(popped[rid][L], want), f"{rid}/{L} host byte mismatch"
    op.close()


# =========================================================================================
# (E) profile mode disables Component 2; the finalize action drives the delivery dispatch
# =========================================================================================
def test_finalize_action_dispatch():
    # Non-profile: the RouteDecision drives delivery.
    assert hp._ring_finalize_action(None, False) == "rpc_raw"                       # back-compat
    assert hp._ring_finalize_action(RouteDecision("rpc", "none"), False) == "rpc_raw"
    assert hp._ring_finalize_action(RouteDecision("disk", "none"), False) == "disk_raw"
    assert hp._ring_finalize_action(RouteDecision("rpc", "inflight"), False) == "analyze_inflight"
    assert hp._ring_finalize_action(RouteDecision("disk", "from_disk"), False) == "analyze_from_disk"


def test_profile_mode_disables_component2():
    # PROFILE MODE: Component 2 is DISABLED for EVERY route -> finalize stamps request_done at the
    # data-prepared boundary and runs no analyzer/delivery.
    for route in (None, RouteDecision("rpc", "none"), RouteDecision("disk", "none"),
                  RouteDecision("rpc", "inflight"), RouteDecision("disk", "from_disk")):
        assert hp._ring_finalize_action(route, True) == "profile_stamp", route


def test_profile_mode_env_gate(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_PROFILE_MODE", raising=False)
    assert hp._profile_mode() is False
    monkeypatch.setenv("VLLM_HOOK_PROFILE_MODE", "1")
    assert hp._profile_mode() is True
    monkeypatch.setenv("VLLM_HOOK_PROFILE_MODE", "0")
    assert hp._profile_mode() is False
