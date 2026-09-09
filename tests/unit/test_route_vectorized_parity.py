"""No-GPU BIT-IDENTITY gate for the vectorized HS capture-routing build (VLLM_HOOK_ROUTE_VECTORIZED).

The vectorized ``_build_routing_hs`` (graph/install_hs.py) must be a pure how-it's-built optimization:
the routing plane and every derived value stay BIT-IDENTICAL, only the per-request torch scatters are
amortized into one advanced-index assign. This drives BOTH the legacy (``vectorized=False``) and the
vectorized (``vectorized=True``) path on the SAME synthetic inputs, from an IDENTICAL fresh registry +
ring, and asserts EXACT equality of:

  * the full ``capture_index_pinned`` plane  (``torch.equal``)
  * ``_hs_step_entries``                      (list of ReqCaptureRecord — same order + fields; and
                                               the flat LayerEntry they ``expand_records`` to)
  * ``_hs_step_start`` / ``_hs_step_rows``    (the off-loop drain's window)
  * the returned ``plans``
  * the shared ring write cursor              (reserve order / never-drop unchanged)

Coverage: N in {1,4,16}; homogeneous AND heterogeneous per-request layer sets; output_hidden_states
True and list; last_token and all_tokens; hooks_on prefill/decode/both against prefill-vs-decode
requests; over-cap spans (end>cap); a non-capturing request mixed in; empty batch; should_capture
False; out-of-range / all-filtered layer lists; config-default hooks_on/hs_mode. Plus a never-drop
gate: a full ring must RAISE (RingBackpressureError) in the vectorized path exactly as in the legacy.

Run:  conda activate vllm_hook_env && pytest tests/unit/test_route_vectorized_parity.py -q
"""
import os
import sys

import pytest
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, ReqCaptureRecord, expand_records
from vllm_hook_plugins.graph.install_hs import _build_routing_hs


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


def _make_registry(num_layers, cap, R, should_capture=True, device="cpu"):
    """A HostRegistry wired for the capture-ring path exactly as _install_hs_buffer does (minus a
    GPU model): a shared GpuCaptureRing cursor, sentinel_row = ring.SENTINEL, inc disabled, and the
    device slab / pinned mirrors primed to the sentinel."""
    reg = HostRegistry(num_layers=num_layers, cap=cap, device=device,
                       should_capture=should_capture)
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


# ------------------------------- scenario model -------------------------------
class Scenario:
    def __init__(self, name, num_layers, cap, R, reqs, **mr_kwargs):
        self.name = name
        self.num_layers = num_layers
        self.cap = cap
        self.R = R
        self.reqs = reqs            # [{"id":.., "extra":.., "span":.., "otids":()}, ...]
        self.mr_kwargs = mr_kwargs

    def qsl(self):
        q = [0]
        for r in self.reqs:
            q.append(q[-1] + r["span"])
        return q

    def specs(self):
        return [(r["id"], r["extra"], r.get("otids", ())) for r in self.reqs]


def _run(scn: Scenario, vectorized: bool):
    """Fresh registry + ring, reset the plane, run ONE routing build; snapshot every output.
    (should_capture=True for every parametrized scenario; the False early-return is covered by its
    own test, so it never appears in mr_kwargs here.)"""
    reg, ring = _make_registry(scn.num_layers, scn.cap, scn.R)
    mr = _mr(scn.specs(), **scn.mr_kwargs)
    reg.reset_pinned(scn.cap)                    # sentinel everywhere, as the routing wrapper does
    plans = _build_routing_hs(mr, reg, scn.qsl(), vectorized=vectorized)
    return {
        "plane": reg.capture_index_pinned.clone(),
        "entries": list(reg._hs_step_entries),
        "step_start": reg._hs_step_start,
        "step_rows": reg._hs_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _assert_identical(scn: Scenario):
    a = _run(scn, vectorized=False)
    b = _run(scn, vectorized=True)
    assert torch.equal(a["plane"], b["plane"]), (
        f"{scn.name}: capture_index_pinned differs\nlegacy=\n{a['plane']}\nvec=\n{b['plane']}")
    assert a["entries"] == b["entries"], (
        f"{scn.name}: _hs_step_entries differ\nlegacy={a['entries']}\nvec={b['entries']}")
    assert a["step_start"] == b["step_start"], f"{scn.name}: _hs_step_start differs"
    assert a["step_rows"] == b["step_rows"], f"{scn.name}: _hs_step_rows differs"
    assert a["plans"] == b["plans"], f"{scn.name}: plans differ\nlegacy={a['plans']}\nvec={b['plans']}"
    assert a["ring_write"] == b["ring_write"], f"{scn.name}: ring write cursor differs"
    # After the LayerEntry collapse `_hs_step_entries` holds per-request ReqCaptureRecord; both paths
    # must produce the same records AND expand (off-loop) to the same flat LayerEntry list.
    assert all(isinstance(e, ReqCaptureRecord) for e in b["entries"])
    assert expand_records(a["entries"]) == expand_records(b["entries"]), (
        f"{scn.name}: expanded LayerEntry list differs")
    assert all(isinstance(e, LayerEntry) for e in expand_records(b["entries"]))
    return a, b


# --------------------------------- extras ------------------------------------
def _all_alltok(hooks="both"):
    return {"output_hidden_states": True, "hs_mode": "all_tokens", "hooks_on": hooks}


def _all_lasttok(hooks="both"):
    return {"output_hidden_states": True, "hs_mode": "last_token", "hooks_on": hooks}


def _list_alltok(layers, hooks="both"):
    return {"output_hidden_states": list(layers), "hs_mode": "all_tokens", "hooks_on": hooks}


def _list_lasttok(layers, hooks="both"):
    return {"output_hidden_states": list(layers), "hs_mode": "last_token", "hooks_on": hooks}


# --------------------------------- scenarios ---------------------------------
def _scenarios():
    S = []

    # N=1 — all-layers, both modes, prefill span.
    S.append(Scenario("N1_alltok_alllayers", 6, 32, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 5}]))
    S.append(Scenario("N1_lasttok_alllayers", 6, 32, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 5}]))

    # N=4 — homogeneous all-layers, all_tokens (the common cb shape).
    S.append(Scenario("N4_homogeneous_alltok", 8, 64, 400,
                      [{"id": f"r{i}", "extra": _all_alltok(), "span": 3 + i} for i in range(4)]))

    # N=4 — HETEROGENEOUS per-request layer sets (True + several lists), mixed modes.
    S.append(Scenario("N4_heterogeneous_layers", 8, 64, 400, [
        {"id": "a", "extra": _all_alltok(), "span": 4},
        {"id": "b", "extra": _list_alltok([2, 3]), "span": 2},
        {"id": "c", "extra": _list_lasttok([1, 4, 5]), "span": 6},
        {"id": "d", "extra": _list_alltok([8, 7, 6]), "span": 3},   # unsorted list -> set order
    ]))

    # N=16 — decode-shaped (every span 1, all_tokens, all-layers) = the route_cost_probe regime.
    S.append(Scenario("N16_decode_alltok", 12, 64, 800,
                      [{"id": f"d{i}", "extra": _all_alltok(), "span": 1, "otids": (7, 7)}
                       for i in range(16)]))

    # N=16 — mixed modes + mixed layer sets + mixed spans.
    reqs16 = []
    for i in range(16):
        if i % 3 == 0:
            reqs16.append({"id": f"m{i}", "extra": _all_alltok(), "span": 1 + (i % 4)})
        elif i % 3 == 1:
            reqs16.append({"id": f"m{i}", "extra": _list_lasttok([1, 3, 5, 7]), "span": 2 + (i % 3)})
        else:
            reqs16.append({"id": f"m{i}", "extra": _list_alltok([2, 4]), "span": 1 + (i % 2)})
    S.append(Scenario("N16_mixed", 8, 96, 800, reqs16))

    # hooks_on=prefill: a DECODE request (otids non-empty) is skipped; a PREFILL request routes.
    S.append(Scenario("hooks_prefill_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("prefill"), "span": 4, "otids": ()},        # routes
        {"id": "dec", "extra": _all_alltok("prefill"), "span": 1, "otids": (1, 2, 3)}, # skipped
        {"id": "pre2", "extra": _all_lasttok("prefill"), "span": 3, "otids": ()},      # routes
    ]))

    # hooks_on=decode: a PREFILL request is skipped; a DECODE request routes.
    S.append(Scenario("hooks_decode_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("decode"), "span": 4, "otids": ()},         # skipped
        {"id": "dec", "extra": _all_alltok("decode"), "span": 1, "otids": (9,)},       # routes
        {"id": "dec2", "extra": _list_lasttok([2, 3], "decode"), "span": 5, "otids": (9,)},  # routes
    ]))

    # Config-DEFAULT hooks_on / hs_mode (extra omits both keys -> mr defaults supply them).
    S.append(Scenario("config_default_hooks_mode", 6, 32, 200,
                      [{"id": "A", "extra": {"output_hidden_states": True}, "span": 4},
                       {"id": "B", "extra": {"output_hidden_states": [2, 4]}, "span": 3}],
                      default_hooks_on="both", worker_hs_mode="all_tokens"))
    S.append(Scenario("config_default_lasttok", 6, 32, 200,
                      [{"id": "A", "extra": {"output_hidden_states": True}, "span": 7}],
                      default_hooks_on="both", worker_hs_mode="last_token"))

    # Over-cap: A fills [0,5); B's span crosses cap -> clamped to [5,8) (n = 3), all_tokens.
    S.append(Scenario("overcap_alltok_partial", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 5},
        {"id": "B", "extra": _all_alltok(), "span": 10},     # end 15 -> clamp 8
    ]))
    # Over-cap single request, all_tokens (whole clamp).
    S.append(Scenario("overcap_alltok_single", 4, 8, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 12}]))
    # Over-cap last_token: last kept column is cap-1 after clamp.
    S.append(Scenario("overcap_lasttok", 4, 8, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 12}]))
    # A request whose start is already >= cap -> clamped end <= start -> skipped entirely.
    S.append(Scenario("overcap_second_fully_skipped", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 8},      # fills [0,8) == cap
        {"id": "B", "extra": _all_alltok(), "span": 4},      # start 8 >= cap -> skip
    ]))

    # Non-capturing requests mixed in (empty extra + output_hidden_states None).
    S.append(Scenario("noncapturing_mixed", 6, 32, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 3},
        {"id": "B", "extra": {}, "span": 4},                             # not extra -> skip
        {"id": "C", "extra": {"output_hidden_states": None}, "span": 2}, # explicit None -> skip
        {"id": "D", "extra": _list_lasttok([1, 6]), "span": 5},
    ]))

    # Layer list with OUT-OF-RANGE entries (0 and > num_layers) -> filtered to valid rows.
    S.append(Scenario("layer_list_out_of_range", 4, 32, 200,
                      [{"id": "A", "extra": _list_alltok([0, 1, 4, 99]), "span": 3}]))
    # Layer list that filters to EMPTY (all out of range) -> request skipped, nothing routed.
    S.append(Scenario("layer_list_all_filtered", 4, 32, 200, [
        {"id": "A", "extra": _list_alltok([50, 99]), "span": 3},   # empty rows_layers -> skip
        {"id": "B", "extra": _all_alltok(), "span": 2},            # routes
    ]))

    # Empty batch.
    S.append(Scenario("empty_batch", 6, 32, 200, []))

    # A zero-span request (qsl has a repeat) mixed in -> skipped by end<=start.
    S.append(Scenario("zero_span_mixed", 6, 32, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 0},   # end==start -> skip
        {"id": "B", "extra": _all_alltok(), "span": 4},
    ]))

    return S


_SCN = _scenarios()


@pytest.mark.parametrize("scn", _SCN, ids=[s.name for s in _SCN])
def test_vectorized_bit_identical(scn):
    _assert_identical(scn)


def test_a_capturing_scenario_is_non_vacuous():
    """Guard: at least one scenario writes real (non-sentinel) slots, so bit-identity isn't a
    vacuous both-paths-no-op. Uses the homogeneous N4 case."""
    scn = next(s for s in _SCN if s.name == "N4_homogeneous_alltok")
    a, b = _assert_identical(scn)
    ring_sent = scn.R
    wrote = (b["plane"] != ring_sent).sum().item()
    assert wrote > 0, "expected real routed slots, plane is all sentinel (vacuous)"
    assert len(b["entries"]) > 0 and len(b["plans"]) == 4


def test_should_capture_false_both_return_empty():
    reg_a, _ = _make_registry(4, 16, 100, should_capture=False)
    mr = _mr([("A", _all_alltok())])
    assert _build_routing_hs(mr, reg_a, [0, 3], vectorized=False) == []
    reg_b, _ = _make_registry(4, 16, 100, should_capture=False)
    assert _build_routing_hs(mr, reg_b, [0, 3], vectorized=True) == []


def _reserve_refused(vectorized):
    """Pre-fill the ring to free=1, then a request needing 2 rows; timeout 0 -> raise at once.
    Returns (raised, ring_write_after, plane_clone)."""
    os.environ["VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S"] = "0"
    try:
        reg, ring = _make_registry(2, 16, 4)
        assert ring.reserve(3) == 0        # occupy 3 of 4 rows -> free 1
        reg.reset_pinned(reg.cap)
        mr = _mr([("A", _all_alltok())])
        raised = False
        try:
            _build_routing_hs(mr, reg, [0, 2], vectorized=vectorized)   # needs 2, only 1 free
        except RingBackpressureError:
            raised = True
        return raised, ring._write, reg.capture_index_pinned.clone()
    finally:
        os.environ.pop("VLLM_HOOK_RING_BACKPRESSURE_TIMEOUT_S", None)


def test_vectorized_never_drop_raises_like_legacy():
    """NEVER-DROP: a full ring must RAISE in BOTH paths (not silently drop the capture), leaving the
    write cursor unmoved and the plane untouched — bit-identical on the refuse path too."""
    raised_l, wr_l, plane_l = _reserve_refused(vectorized=False)
    raised_v, wr_v, plane_v = _reserve_refused(vectorized=True)
    assert raised_l and raised_v, "reserve must SIGNAL backpressure (raise) in both paths"
    assert wr_l == wr_v == 3, "the refused request must not advance the cursor in either path"
    assert torch.equal(plane_l, plane_v), "plane differs on the refuse path"


def main():
    import traceback
    tests = ([lambda s=s: test_vectorized_bit_identical(s) for s in _SCN]
             + [test_a_capturing_scenario_is_non_vacuous,
                test_should_capture_false_both_return_empty,
                test_vectorized_never_drop_raises_like_legacy])
    names = ([s.name for s in _SCN]
             + ["non_vacuous", "should_capture_false", "never_drop"])
    failures = 0
    for name, t in zip(names, tests):
        try:
            t()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            traceback.print_exc()
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'} "
          f"({len(tests) - failures}/{len(tests)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
