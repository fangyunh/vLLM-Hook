"""No-GPU routing smoke test for the HS capture-ring GPU-integration (plan Tasks 7-9).

Drives ``graph.install_hs._build_routing_hs`` on ``device="cpu"`` with a FAKE model_runner and a
synthetic ``qsl_cpu``, and asserts the four routing-correctness properties the GPU parity oracle
(Task 10) then confirms end-to-end:

  (a) captured token columns hold ADVANCING ring slots (from GpuCaptureRing.physical_slots), NOT
      the old batch-position row ``p+1``; pad / non-captured columns hold the ring SENTINEL.
  (b) the shared GpuCaptureRing write cursor advances by exactly the reserved row count.
  (c) the collapse's per-request records expand (via ``expand_records``) to one LayerEntry per
      (req, layer) with the right logical_start / n_rows / hs_mode / 1-based layer number
      (last_token reserves 1 row; all_tokens reserves the span).
  (d) on a (nearly) full ring, reserve REFUSES (backpressure signalled: RingBackpressureError)
      rather than silently dropping the capture.

Run:  conda activate vllm_hook_env && python tests/unit/test_hs_ring_routing_smoke.py
      (or under pytest: pytest tests/unit/test_hs_ring_routing_smoke.py)
"""
import os
import sys

import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, ReqCaptureRecord, expand_records
import vllm_hook_plugins.graph.install_hs as ihs
from vllm_hook_plugins.graph.install_hs import _build_routing_hs, RingBackpressureError


# ------------------------- minimal vLLM-surface fakes -------------------------
class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _RS:
    def __init__(self, extra, output_token_ids=()):
        self.sampling_params = _SP(extra)
        self.output_token_ids = output_token_ids


class _IB:
    def __init__(self, req_ids):
        self.req_ids = req_ids


class _MR:
    def __init__(self, req_ids, requests, default_hooks_on="both", worker_hs_mode="all_tokens"):
        self.input_batch = _IB(req_ids)
        self.requests = requests
        self._default_hooks_on = default_hooks_on
        self._worker_hs_mode = worker_hs_mode


def _mr(specs, **kw):
    """specs = [(req_id, extra_args, output_token_ids), ...]."""
    return _MR([s[0] for s in specs],
               {s[0]: _RS(s[1], s[2] if len(s) > 2 else ()) for s in specs}, **kw)


def _make_registry(num_layers, cap, R, device="cpu"):
    """A HostRegistry wired for the capture-ring path exactly as _install_hs_buffer does (minus a
    GPU model): a shared GpuCaptureRing cursor, sentinel_row = ring.SENTINEL, inc disabled, and the
    device slab / pinned mirrors primed to the sentinel."""
    reg = HostRegistry(num_layers=num_layers, cap=cap, device=device, should_capture=True)
    ring = GpuCaptureRing(row_bytes=8, n_slots=R, device=device,
                          dtype=torch.float32, row_shape=(4,))
    reg._hs_ring = ring
    reg._hs_step_entries = []
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
    return _build_routing_hs(mr, reg, qsl)


# --------------------------------- (a) + (b) ----------------------------------
def test_advancing_slots_not_p_plus_1_and_pad_is_sentinel():
    num_layers, cap, R = 4, 16, 100
    reg, ring = _make_registry(num_layers, cap, R)
    SENT = ring.SENTINEL

    # Step 1: advance the cursor with a throwaway request so the next start_slot is NOT 0 (0 would
    # coincide with the first batch position and make "advancing slot" indistinguishable from p).
    warm = _mr([("warm", {"output_hidden_states": True, "hs_mode": "all_tokens",
                          "hooks_on": "both"})])
    _route(reg, warm, [0, 7])
    assert ring._write == 7, f"warm-up should reserve 7 rows, got {ring._write}"

    # Step 2: one all_tokens request over columns [0, 5). It must route to the ADVANCING slots
    # [7, 8, 9, 10, 11] (start_slot == prior write cursor 7), NOT p+1 == [1,2,3,4,5].
    mr = _mr([("A", {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"})])
    plans = _route(reg, mr, [0, 5])
    ci = reg.capture_index_pinned
    got = [int(x) for x in ci[0, 0:5]]
    assert got == [7, 8, 9, 10, 11], f"expected advancing ring slots [7..11], got {got}"
    assert got != [1, 2, 3, 4, 5], "routing still uses the old p+1 batch-position rows"
    # Same slots on every requested layer (parallel per-layer rings, shared cursor).
    for L in range(num_layers):
        assert [int(x) for x in ci[L, 0:5]] == [7, 8, 9, 10, 11], f"layer {L} slots differ"
    # Pad columns [5, cap) hold the SENTINEL (== R), never a real slot 0.
    for L in range(num_layers):
        assert all(int(x) == SENT for x in ci[L, 5:cap]), f"layer {L} pad not sentinel"
    # (b) shared cursor advanced by exactly the 5 reserved rows.
    assert ring._write == 12, f"cursor should be 7+5=12, got {ring._write}"
    assert len(plans) == 1


# ------------------------------- (b) multi-req --------------------------------
def test_shared_cursor_advance_two_requests_mixed_modes():
    num_layers, cap, R = 4, 16, 100
    reg, ring = _make_registry(num_layers, cap, R)
    SENT = ring.SENTINEL
    # A: all_tokens cols [0,3) -> reserve 3 slots [0,1,2]. B: last_token cols [3,7) -> reserve 1
    # slot [3], routed ONLY to the last column (6); [3,6) stay SENTINEL.
    mr = _mr([
        ("A", {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"}),
        ("B", {"output_hidden_states": True, "hs_mode": "last_token", "hooks_on": "both"}),
    ])
    plans = _route(reg, mr, [0, 3, 7])
    ci = reg.capture_index_pinned
    assert [int(x) for x in ci[0, 0:3]] == [0, 1, 2], "A did not take slots [0,1,2]"
    assert all(int(ci[0, c]) == SENT for c in (3, 4, 5)), "B non-last cols not sentinel"
    assert int(ci[0, 6]) == 3, "B last col did not take slot 3"
    assert ring._write == 4, f"cursor should be 3+1=4, got {ring._write}"
    assert len(plans) == 2


# ----------------------------------- (c) --------------------------------------
def test_layer_entries_recorded_per_req_layer():
    num_layers, cap, R = 6, 16, 100
    reg, ring = _make_registry(num_layers, cap, R)
    # A: all layers, all_tokens, cols [0,3) -> logical_start 0, n_rows 3, 6 entries (layers 1..6).
    # B: layer_filter [2,3] (1-based), last_token, cols [3,5) -> logical_start 3, n_rows 1, layers 2,3.
    mr = _mr([
        ("A", {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": "both"}),
        ("B", {"output_hidden_states": [2, 3], "hs_mode": "last_token", "hooks_on": "both"}),
    ])
    _route(reg, mr, [0, 3, 5])
    # LayerEntry COLLAPSE: `_hs_step_entries` holds per-request ReqCaptureRecord; expand_records fans
    # them into the flat per-(req, layer) LayerEntry list the drain/sidecar consume.
    records = reg._hs_step_entries
    assert all(isinstance(r, ReqCaptureRecord) for r in records)
    entries = expand_records(records)
    a = sorted([e for e in entries if e.req_id == "A"], key=lambda e: e.layer)
    b = sorted([e for e in entries if e.req_id == "B"], key=lambda e: e.layer)
    assert [e.layer for e in a] == [1, 2, 3, 4, 5, 6], f"A layers wrong: {[e.layer for e in a]}"
    assert all(e.logical_start == 0 and e.n_rows == 3 and e.hs_mode == "all_tokens" for e in a)
    assert [e.layer for e in b] == [2, 3], f"B layers wrong: {[e.layer for e in b]}"
    assert all(e.logical_start == 3 and e.n_rows == 1 and e.hs_mode == "last_token" for e in b)
    assert all(isinstance(e, LayerEntry) for e in entries)
    # last_token reserves ONE row; all_tokens reserves the span -> total 3 + 1 = 4 rows.
    assert ring._write == 4, f"cursor should be 4, got {ring._write}"


def test_last_token_reserves_single_row_not_whole_span():
    num_layers, cap, R = 2, 32, 100
    reg, ring = _make_registry(num_layers, cap, R)
    # A long last_token prompt (span 20) must consume only ONE ring row (we keep just the last).
    mr = _mr([("A", {"output_hidden_states": True, "hs_mode": "last_token", "hooks_on": "both"})])
    _route(reg, mr, [0, 20])
    assert ring._write == 1, f"last_token must reserve 1 row, reserved {ring._write}"
    ci = reg.capture_index_pinned
    assert int(ci[0, 19]) == 0, "last column did not take the reserved slot 0"
    assert all(int(ci[0, c]) == ring.SENTINEL for c in range(0, 19)), "non-last cols not sentinel"


# ----------------------------------- (d) --------------------------------------
def test_backpressure_refuses_on_full_ring_instead_of_dropping():
    # Ring holds only 4 rows/layer. Pre-fill 3 (free=1), then a request needing 2 rows must be
    # REFUSED (RingBackpressureError) rather than silently dropped. Timeout 0 -> raise at once.
    os.environ["VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S"] = "0"
    try:
        num_layers, cap, R = 2, 16, 4
        reg, ring = _make_registry(num_layers, cap, R)
        assert ring.reserve(3) == 0                       # occupy 3 of 4 rows
        assert ring.free_rows() == 1
        mr = _mr([("A", {"output_hidden_states": True, "hs_mode": "all_tokens",
                        "hooks_on": "both"})])
        raised = False
        try:
            _route(reg, mr, [0, 2])                        # needs 2 rows, only 1 free
        except RingBackpressureError:
            raised = True
        assert raised, "reserve must SIGNAL backpressure (raise), not drop the capture"
        # The refused request left the write cursor unmoved (never partially reserved).
        assert ring._write == 3, f"cursor moved despite refusal: {ring._write}"
    finally:
        os.environ.pop("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", None)


def test_reserve_succeeds_when_space_frees():
    # Sanity companion to (d): once the drain advances, the same reserve succeeds (never-drop is a
    # BLOCK, not a permanent refuse) — proven directly on the ring cursor.
    ring = GpuCaptureRing(row_bytes=8, n_slots=4, device="cpu", dtype=torch.float32, row_shape=(4,))
    assert ring.reserve(4) == 0
    assert ring.reserve(1) is None
    ring.advance_drain(4)
    assert ring.reserve(1) == 4


def main():
    tests = [
        test_advancing_slots_not_p_plus_1_and_pad_is_sentinel,
        test_shared_cursor_advance_two_requests_mixed_modes,
        test_layer_entries_recorded_per_req_layer,
        test_last_token_reserves_single_row_not_whole_span,
        test_backpressure_refuses_on_full_ring_instead_of_dropping,
        test_reserve_succeeds_when_space_frees,
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
