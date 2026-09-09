"""No-GPU routing smoke test for the QK capture-ring GPU-integration (plan Task 15).

Drives ``graph.install._build_routing`` on ``device="cpu"`` with a FAKE model_runner and a synthetic
``qsl_cpu``, asserting the QK routing-correctness properties the GPU parity oracle then confirms:

  (a) captured token columns hold ADVANCING ring slots (from GpuCaptureRing.physical_slots), NOT the
      old batch-position row ``p+1``; the SAME slots serve every requested layer (q and k share the
      index); pad columns hold the ring SENTINEL.
  (b) the shared cursor advances by ``qlen`` EVERY step (K needs every key) — for last_token TOO,
      unlike the HS ring's 1-row last_token reserve.
  (c) the collapse's per-request records expand (via ``expand_qk_records``) to one QKStepEntry per
      (req, layer): k_start/k_rows recorded every step; q gated by emit_q (all_tokens: q_start==k_start,
      q_rows==qlen; last_token final chunk: q_start==k_start+qlen-1, q_rows==1); prefix_end==abs_end on
      emit; num_computed carried for the reader's prefix guard.
  (d) last_token MID-prefill chunk (emit_q False): NO q recorded (q_rows==0, prefix_end==-1) but K IS
      (k_rows==qlen) and the reserve is the WHOLE span.
  (e) a full ring REFUSES (RingBackpressureError), never a silent drop.

Run:  conda activate vllm_hook_env && python tests/unit/test_qk_ring_routing_smoke.py
"""
import os
import sys

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import (
    QKStepEntry, QKReqCaptureRecord, expand_qk_records)
from vllm_hook_plugins.graph.install import _build_routing


# ------------------------- minimal vLLM-surface fakes -------------------------
class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _RS:
    def __init__(self, extra, output_token_ids=()):
        self.sampling_params = _SP(extra)
        self.output_token_ids = output_token_ids


class _IB:
    def __init__(self, req_ids, num_computed=None, num_prompt=None):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed if num_computed is not None else [0] * len(req_ids)
        self.num_prompt_tokens = num_prompt if num_prompt is not None else [0] * len(req_ids)


class _MR:
    def __init__(self, req_ids, requests, num_computed=None, num_prompt=None,
                 default_hooks_on="both", worker_hookq_mode="all_tokens"):
        self.input_batch = _IB(req_ids, num_computed, num_prompt)
        self.requests = requests
        self._default_hooks_on = default_hooks_on
        self._worker_hookq_mode = worker_hookq_mode
        self._worker_score_mode = False
        self._worker_score_head = 0


def _mr(specs, **kw):
    """specs = [(req_id, extra_args, output_token_ids), ...]."""
    return _MR([s[0] for s in specs],
               {s[0]: _RS(s[1], s[2] if len(s) > 2 else ()) for s in specs}, **kw)


def _make_registry(num_layers, cap, R, q_dim=8, k_dim=4, device="cpu"):
    """A HostRegistry wired for the QK capture-ring path exactly as install_qk_hosts does (minus a
    GPU model): a shared GpuCaptureRing cursor, sentinel_row = ring.SENTINEL, inc/gpu-routing off,
    and the pinned mirrors primed to the sentinel."""
    reg = HostRegistry(num_layers=num_layers, cap=cap, device=device, should_capture=True)
    ring = GpuCaptureRing(row_bytes=k_dim * 4, n_slots=R, device=device,
                          dtype=torch.float32, row_shape=(k_dim,))
    reg._qk_ring = ring
    reg._qk_consumer = None
    reg._qk_step_entries = []
    reg.sentinel_row = ring.SENTINEL
    reg.incremental_enabled = False
    reg.gpu_routing = False
    reg.capture_index_all.fill_(ring.SENTINEL)
    for slot in reg._ring.slots:
        slot["capture_index"].fill_(ring.SENTINEL)
    return reg, ring


def _route(reg, mr, qsl, width=None):
    """Mirror the routing wrapper's legacy branch: reset (fills sentinel) then build."""
    reg.reset_pinned(width if width is not None else reg.cap)
    return _build_routing(mr, reg, qsl)


def _by(entries, req_id, layer=None):
    out = [e for e in entries if e.req_id == req_id and (layer is None or e.layer == layer)]
    return out


# --------------------------------- (a) + (b) ----------------------------------
def test_advancing_slots_shared_across_layers_and_pad_is_sentinel():
    num_layers, cap, R = 4, 16, 100
    reg, ring = _make_registry(num_layers, cap, R)
    SENT = ring.SENTINEL

    # Warm the cursor so start_slot != 0 (0 would coincide with batch position p).
    warm = _mr([("warm", {"output_qk": True, "hookq_mode": "all_tokens", "hooks_on": "both"})])
    _route(reg, warm, [0, 7])
    assert ring._write == 7, f"warm-up should reserve 7 rows, got {ring._write}"

    # One all_tokens request over columns [0,5) -> advancing slots [7..11] on EVERY requested layer.
    mr = _mr([("A", {"output_qk": True, "hookq_mode": "all_tokens", "hooks_on": "both"})])
    plans = _route(reg, mr, [0, 5])
    ci = reg.capture_index_pinned
    for L in range(num_layers):
        got = [int(x) for x in ci[L, 0:5]]
        assert got == [7, 8, 9, 10, 11], f"layer {L} slots {got} != advancing [7..11]"
        assert all(int(x) == SENT for x in ci[L, 5:cap]), f"layer {L} pad not sentinel"
    assert ring._write == 12, f"cursor should be 7+5=12, got {ring._write}"
    assert len(plans) == 1


def test_last_token_reserves_whole_span_and_routes_all_columns():
    # QK-specific: last_token still keeps EVERY key, so it reserves qlen (NOT 1 like HS) and routes
    # ALL columns to advancing slots; only the q METADATA narrows to the last token.
    num_layers, cap, R = 2, 32, 100
    reg, ring = _make_registry(num_layers, cap, R)
    mr = _mr([("A", {"output_qk": True, "hookq_mode": "last_token", "hooks_on": "both"})])
    _route(reg, mr, [0, 20])
    assert ring._write == 20, f"last_token QK must reserve the whole span (20), got {ring._write}"
    ci = reg.capture_index_pinned
    assert [int(x) for x in ci[0, 0:20]] == list(range(0, 20)), "all 20 cols not routed to slots"
    assert all(int(ci[0, c]) == ring.SENTINEL for c in range(20, cap)), "pad not sentinel"


# ----------------------------------- (c) --------------------------------------
def test_qk_entries_all_tokens_and_last_token():
    num_layers, cap, R = 6, 16, 100
    reg, ring = _make_registry(num_layers, cap, R)
    # A: all layers, all_tokens, cols [0,3) -> k_start 0, k_rows 3, q_start 0, q_rows 3, prefix_end 3.
    # B: layer_filter {2,3}, last_token, cols [3,7) (span 4). B's prompt is 4 tokens, so this single
    #    chunk IS the final one (abs_end 4 >= num_prompt 4) -> emit_q: k_start 3, k_rows 4, q_start 6
    #    (3+4-1), q_rows 1, prefix_end 4 (== abs_end = num_computed 0 + qlen 4).
    mr = _mr([
        ("A", {"output_qk": True, "hookq_mode": "all_tokens", "hooks_on": "both"}),
        ("B", {"output_qk": [2, 3], "hookq_mode": "last_token", "hooks_on": "both"}),
    ], num_prompt=[3, 4])
    _route(reg, mr, [0, 3, 7])
    # LayerEntry COLLAPSE: `_qk_step_entries` holds per-request QKReqCaptureRecord; expand_qk_records
    # fans them into the flat per-(req, layer) QKStepEntry list the drain/sidecar consume.
    records = reg._qk_step_entries
    assert all(isinstance(r, QKReqCaptureRecord) for r in records)
    entries = expand_qk_records(records)
    assert all(isinstance(e, QKStepEntry) for e in entries)

    a = _by(entries, "A")
    assert sorted(e.layer for e in a) == [0, 1, 2, 3, 4, 5], f"A layers {[e.layer for e in a]}"
    for e in a:
        assert (e.k_start, e.k_rows) == (0, 3)
        assert (e.q_start, e.q_rows) == (0, 3), "all_tokens q should span the whole step"
        assert e.prefix_end == 3 and e.num_computed == 0

    b = _by(entries, "B")
    assert sorted(e.layer for e in b) == [2, 3], f"B layers {[e.layer for e in b]}"
    for e in b:
        assert (e.k_start, e.k_rows) == (3, 4), "K keeps the whole span even for last_token"
        assert (e.q_start, e.q_rows) == (6, 1), "last_token q is the span's LAST slot only"
        assert e.prefix_end == 4 and e.num_computed == 0
    # last_token reserved the whole span (4), so cursor = 3 + 4 = 7.
    assert ring._write == 7, f"cursor should be 7, got {ring._write}"


# ----------------------------------- (d) --------------------------------------
def test_last_token_midchunk_records_k_but_not_q():
    # A last_token request whose FIRST prefill chunk is not the final one (num_prompt > abs_end):
    # K accumulates (k_rows==qlen) but no q is emitted (q_rows==0, prefix_end==-1), and the reserve
    # is still the WHOLE span.
    num_layers, cap, R = 2, 32, 100
    reg, ring = _make_registry(num_layers, cap, R)
    mr = _mr([("A", {"output_qk": True, "hookq_mode": "last_token", "hooks_on": "both"})],
             num_prompt=[20])   # prompt is 20 tokens; this chunk is only cols [0,8)
    _route(reg, mr, [0, 8])
    assert ring._write == 8, f"mid-chunk must reserve the whole span (8), got {ring._write}"
    assert all(isinstance(r, QKReqCaptureRecord) for r in reg._qk_step_entries)
    for e in _by(expand_qk_records(reg._qk_step_entries), "A"):
        assert (e.k_start, e.k_rows) == (0, 8), "K must accumulate on a non-final chunk"
        assert e.q_rows == 0 and e.q_start == -1, "mid-chunk must NOT emit q"
        assert e.prefix_end == -1, "mid-chunk must NOT record a prefix_end"


# ----------------------------------- (e) --------------------------------------
def test_backpressure_refuses_on_full_ring():
    os.environ["VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S"] = "0"
    try:
        num_layers, cap, R = 2, 16, 4
        reg, ring = _make_registry(num_layers, cap, R)
        assert ring.reserve(3) == 0                       # occupy 3 of 4 rows
        assert ring.free_rows() == 1
        mr = _mr([("A", {"output_qk": True, "hookq_mode": "all_tokens", "hooks_on": "both"})])
        raised = False
        try:
            _route(reg, mr, [0, 2])                        # needs 2 rows, only 1 free
        except RingBackpressureError:
            raised = True
        assert raised, "reserve must SIGNAL backpressure (raise), not drop the capture"
        assert ring._write == 3, f"cursor moved despite refusal: {ring._write}"
    finally:
        os.environ.pop("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", None)


def main():
    tests = [
        test_advancing_slots_shared_across_layers_and_pad_is_sentinel,
        test_last_token_reserves_whole_span_and_routes_all_columns,
        test_qk_entries_all_tokens_and_last_token,
        test_last_token_midchunk_records_k_but_not_q,
        test_backpressure_refuses_on_full_ring,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'} "
          f"({len(tests) - failures}/{len(tests)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
