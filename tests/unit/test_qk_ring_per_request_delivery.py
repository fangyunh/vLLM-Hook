"""No-GPU per-request QK demux + assembly byte-identity (Task 13 — the QK port of
test_ring_per_request_delivery + the QK reader roundtrip).

Drives the REAL ``OffLoopQKRingDrain(per_request=True)._demux_into_index`` + ``_handle_finish`` and the
REAL ``ProbeHookQKWorker`` marshal (``get_ring_per_request`` -> ``_marshal_perreq_qk``) + driver
deserialize (``_hook_plugin._decompress``) end-to-end with a CPU ring -- NO engine boot, NO GPU. The
consumer thread is NOT started, so the interleave is deterministic and the demux/finish paths run
directly.

Covers, over TWO interleaved requests captured in the SAME steps:
  * per-request QK demux into the ``("q", layer)`` / ``("k", layer)`` staging convention, and
    ``assemble_qk`` rebuilding ``{layer: {"q": <cat q>, "k_all": [k_full[:L] for L in prefix_ends]}}``
    byte-identical to a hand-built reference (all_tokens growing-prefix AND last_token flat-q);
  * the worker marshal round-trip: bytes -> _decompress -> ``qk_cache`` torch.equal to the reference,
    with ``k_all`` delivered as the growing-prefix LIST (not a padded tensor), and residency -> 0;
  * the id-divergence match ("{external}-{rand}" internal id) on retrieval;
  * a not-yet-finished request returns None + stays resident;
  * strict no-op -> None when the ring path is absent / per_request is off.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_qk_ring_per_request_delivery.py -q
"""
import tempfile

import torch

from vllm_hook_plugins._hook_plugin import _decompress
from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex, assemble_qk
from vllm_hook_plugins.graph.ring_drain_hs import _DrainItem, _torch_dtype_name
from vllm_hook_plugins.graph.ring_drain_qk import OffLoopQKRingDrain
from vllm_hook_plugins.graph.ring_metadata import QKStepEntry
from vllm_hook_plugins.workers.probe_hookqk_worker import ProbeHookQKWorker

_QDIM, _KDIM = 8, 4


def _build(layers=(0, 1), dtype=torch.float32, index=None, per_request=True):
    ring = GpuCaptureRing(row_bytes=_KDIM * torch.empty(0, dtype=dtype).element_size(),
                          n_slots=512, device="cpu", dtype=dtype, row_shape=(_KDIM,))
    q_bufs = {L: torch.zeros(513, _QDIM, dtype=dtype) for L in layers}
    k_bufs = {L: torch.zeros(513, _KDIM, dtype=dtype) for L in layers}
    header = {"dtype": _torch_dtype_name(dtype), "q_row_shape": [_QDIM], "k_row_shape": [_KDIM],
              "q_dim": _QDIM, "k_dim": _KDIM, "hookq_mode": "all_tokens"}
    tmp = tempfile.mkdtemp(prefix="qk_pr_")
    kw = {"per_request": per_request}
    if index is not None:
        kw["index"] = index
    drain = OffLoopQKRingDrain(
        ring, [(L, q_bufs[L], k_bufs[L]) for L in layers], tmp, header, **kw)
    return drain, layers


def _demux_qk_step(drain, all_layers, reqs, start_logical, state, exp, tag):
    """Simulate ONE drained step through the REAL ``_demux_into_index``.

    ``reqs`` = [(rid, n, [layers], mode, emit_q)]. K keeps the whole span every step; q is the whole
    span (all_tokens) / the last row (last_token emit) / nothing (non-emit). ``state[rid]`` tracks the
    request's cumulative key count so ``prefix_end == abs_end``. Fills per-layer q/k arrays for the
    step's ``[start_logical, +total)`` region with STEP-/req-/layer-UNIQUE data (so a mis-ordered or
    cross-request/-layer stage is caught) and records the expected q/k_full/prefix_ends into ``exp``.
    Returns the next logical cursor."""
    total = sum(n for _, n, _, _, _ in reqs)
    q_by = {L: torch.zeros(total, _QDIM) for L in all_layers}
    k_by = {L: torch.zeros(total, _KDIM) for L in all_layers}
    entries = []
    off = 0
    for (rid, n, layers, mode, emit_q) in reqs:
        abs_end = state.get(rid, 0) + n
        state[rid] = abs_end
        for L in layers:
            qbase = (hash((rid, L, tag, "q")) % 991) * 1000 + 1
            kbase = (hash((rid, L, tag, "k")) % 991) * 1000 + 1
            q_step = (torch.arange(n * _QDIM, dtype=torch.float32).reshape(n, _QDIM) + qbase)
            k_step = (torch.arange(n * _KDIM, dtype=torch.float32).reshape(n, _KDIM) + kbase)
            q_by[L][off:off + n] = q_step
            k_by[L][off:off + n] = k_step
            if mode == "all_tokens":
                q_start, q_rows, q_emit = start_logical + off, n, q_step
            elif emit_q:
                q_start, q_rows, q_emit = start_logical + off + n - 1, 1, q_step[-1:]
            else:
                q_start, q_rows, q_emit = -1, 0, None
            prefix_end = abs_end if emit_q else -1
            entries.append(QKStepEntry(
                req_id=str(rid), layer=int(L),
                k_start=start_logical + off, k_rows=n,
                q_start=q_start, q_rows=q_rows,
                prefix_end=prefix_end, num_computed=0))
            e = exp.setdefault(str(rid), {}).setdefault(int(L), {"q": [], "k": [], "ends": []})
            e["k"].append(k_step)
            if q_emit is not None:
                e["q"].append(q_emit)
            if prefix_end >= 0:
                e["ends"].append(prefix_end)
        off += n
    item = _DrainItem(entries, start_logical, total, None)
    drain._demux_into_index(item, [(L, q_by[L], k_by[L]) for L in all_layers])
    return start_logical + total


def _ref(exp_for_rid):
    """Hand-built {layer: {"q": cat, "k_all": [k_full[:L] for L in ends]}} for one request."""
    out = {}
    for L, e in exp_for_rid.items():
        k_full = e["k"][0] if len(e["k"]) == 1 else torch.cat(e["k"], 0)
        q = e["q"][0] if len(e["q"]) == 1 else torch.cat(e["q"], 0)
        out[L] = {"q": q, "k_all": [k_full[:L2] for L2 in e["ends"]]}
    return out


def _assert_qk_equal(got, ref, who):
    assert set(got) == set(ref), f"{who}: layers {set(got)} != {set(ref)}"
    for L in ref:
        assert torch.equal(got[L]["q"], ref[L]["q"]), f"{who}/{L} q mismatch"
        gk, rk = got[L]["k_all"], ref[L]["k_all"]
        assert len(gk) == len(rk), f"{who}/{L} k_all length {len(gk)} != {len(rk)}"
        for i, (g, r) in enumerate(zip(gk, rk)):
            assert torch.equal(g, r), f"{who}/{L} k_all[{i}] mismatch"


# ------------------------------------------------- (1) demux + assemble_qk (index) ---
def test_two_interleaved_reqs_demux_and_assemble_qk():
    """A: all_tokens over {0,1}; B: last_token over {1} only -- interleaved every step. Each request's
    ``assemble_qk`` output (q + growing-prefix k_all) is byte-identical to a hand-built reference, and
    the two requests never cross-contaminate."""
    idx = PerRequestIndex()
    drain, layers = _build(layers=(0, 1), index=idx)
    exp, state, sl = {}, {}, 0
    # step 1: A prefill(3) all_tokens, B prefill(2) last_token FINAL chunk (emit).
    sl = _demux_qk_step(drain, layers,
                        [("A", 3, [0, 1], "all_tokens", True),
                         ("B", 2, [1], "last_token", True)], sl, state, exp, "s1")
    # step 2 + 3: both decode (emit).
    sl = _demux_qk_step(drain, layers,
                        [("A", 1, [0, 1], "all_tokens", True),
                         ("B", 1, [1], "last_token", True)], sl, state, exp, "s2")
    sl = _demux_qk_step(drain, layers,
                        [("A", 1, [0, 1], "all_tokens", True),
                         ("B", 1, [1], "last_token", True)], sl, state, exp, "s3")
    assert idx.live_req_ids() == {"A", "B"}
    for rid in ("A", "B"):
        drain._handle_finish(rid)
    popped = dict(idx.pop_deliverable_qk())
    assert set(popped) == {"A", "B"}
    _assert_qk_equal(popped["A"], _ref(exp["A"]), "A")
    _assert_qk_equal(popped["B"], _ref(exp["B"]), "B")
    # B (last_token over 1 layer) delivered q = one row per emit step (3 rows), k_full = 4 rows.
    assert popped["B"][1]["q"].shape[0] == 3
    assert popped["B"][1]["k_all"][-1].shape[0] == 4


def test_last_token_non_emit_prefill_chunk_keeps_k_only():
    """last_token + chunked prefill: K accumulates on the non-emit chunk (no q, no prefix boundary),
    then the FINAL chunk + decode emit q. assemble_qk's k_all/q match the reference exactly."""
    idx = PerRequestIndex()
    drain, layers = _build(layers=(0,), index=idx)
    exp, state, sl = {}, {}, 0
    sl = _demux_qk_step(drain, layers, [("C", 2, [0], "last_token", False)], sl, state, exp, "c1")
    sl = _demux_qk_step(drain, layers, [("C", 2, [0], "last_token", True)], sl, state, exp, "c2")
    sl = _demux_qk_step(drain, layers, [("C", 1, [0], "last_token", True)], sl, state, exp, "c3")
    drain._handle_finish("C")
    got = dict(idx.pop_deliverable_qk())["C"]
    _assert_qk_equal(got, _ref(exp["C"]), "C")
    assert got[0]["q"].shape[0] == 2                 # emit only on chunk-2 (last prompt tok) + 1 decode
    assert [k.shape[0] for k in got[0]["k_all"]] == [4, 5]   # prefix ends after emit steps


# ---------------------------------------- (2) worker marshal bytes round-trip ---
def test_worker_get_ring_per_request_bytes_roundtrip():
    """The REAL worker marshal (get_ring_per_request -> _marshal_perreq_qk compress+pickle) and driver
    _decompress: bytes (not raw tensors), qk_cache torch.equal to the reference, k_all a LIST, and
    residency -> 0 after each delivered request is freed."""
    idx = PerRequestIndex()
    drain, layers = _build(layers=(0, 1), index=idx)
    w = ProbeHookQKWorker()
    w._conf = {"hidden_size": 16}
    w._qk_drain = drain
    exp, state, sl = {}, {}, 0
    sl = _demux_qk_step(drain, layers,
                        [("A", 3, [0, 1], "all_tokens", True)], sl, state, exp, "s1")
    sl = _demux_qk_step(drain, layers,
                        [("A", 1, [0, 1], "all_tokens", True)], sl, state, exp, "s2")
    drain._handle_finish("A")

    blob = w.get_ring_per_request("A")
    assert isinstance(blob, (bytes, bytearray)), "worker must return bytes, not raw tensors"
    payload = _decompress(blob)
    assert set(payload.keys()) == {"qk_cache", "config"}
    assert payload["config"] == {"hidden_size": 16}
    qk = payload["qk_cache"]
    ref = _ref(exp["A"])
    assert set(qk.keys()) == set(ref.keys())
    for L in ref:
        assert qk[L]["layer_num"] == L
        assert isinstance(qk[L]["k_all"], list), "k_all must be delivered as the growing-prefix LIST"
        assert torch.equal(qk[L]["q"], ref[L]["q"])
        for g, r in zip(qk[L]["k_all"], ref[L]["k_all"]):
            assert torch.equal(g, r)
    assert idx.live_req_ids() == set(), "delivered request left resident"
    assert w.ring_residency() == (0, 0)


def test_suffixed_internal_req_id_matches_on_retrieval():
    """vLLM rewrites the external id to '{external}-{rand}'; get_ring_per_request matches it (exact or
    '{external}-' prefix), the same id rule the disk/host hops key on."""
    idx = PerRequestIndex()
    drain, layers = _build(layers=(0,), index=idx)
    w = ProbeHookQKWorker()
    w._conf = {}
    w._qk_drain = drain
    exp, state, sl = {}, {}, 0
    _demux_qk_step(drain, layers, [("ext-abc123", 2, [0], "all_tokens", True)], sl, state, exp, "s1")
    drain._handle_finish("ext-abc123")
    blob = w.get_ring_per_request("ext")               # external id -> matches the suffixed internal
    assert blob is not None
    qk = _decompress(blob)["qk_cache"]
    _assert_qk_equal({L: {"q": v["q"], "k_all": v["k_all"]} for L, v in qk.items()},
                     _ref(exp["ext-abc123"]), "ext")


def test_not_yet_finished_returns_none_and_keeps_residency():
    idx = PerRequestIndex()
    drain, layers = _build(layers=(0,), index=idx)
    w = ProbeHookQKWorker()
    w._conf = {}
    w._qk_drain = drain
    exp, state, sl = {}, {}, 0
    _demux_qk_step(drain, layers, [("R", 2, [0], "all_tokens", True)], sl, state, exp, "s1")
    assert w.get_ring_per_request("R") is None         # noted, never finished
    assert idx.live_req_ids() == {"R"}


def test_noop_when_ring_path_not_installed_or_per_request_off():
    w = ProbeHookQKWorker()
    assert w.get_ring_per_request("R") is None          # no _qk_drain
    drain, _ = _build(layers=(0,), per_request=False)
    w._qk_drain = drain
    assert w.get_ring_per_request("R") is None          # per_request off
    assert w.ring_residency() is None
    assert w.flush_ring_per_request() is None


if __name__ == "__main__":
    for t in [test_two_interleaved_reqs_demux_and_assemble_qk,
              test_last_token_non_emit_prefill_chunk_keeps_k_only,
              test_worker_get_ring_per_request_bytes_roundtrip,
              test_suffixed_internal_req_id_matches_on_retrieval,
              test_not_yet_finished_returns_none_and_keeps_residency,
              test_noop_when_ring_path_not_installed_or_per_request_off]:
        t()
        print(f"PASS  {t.__name__}")
    print("VERDICT: PASS")
