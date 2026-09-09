"""No-GPU BIT-IDENTITY gate for the HS "LayerEntry collapse" (graph/install_hs.py).

The collapse moves the per-(request x captured layer) ``LayerEntry`` fan-out OFF the engine loop:
``_build_routing_hs`` now builds ONE ``ReqCaptureRecord`` per capturing request (carrying that
request's OWN 1-based ``layers`` list), and the drain expands each record into the flat
``LayerEntry`` list — same fields, SAME order — off-loop. This test proves the expanded list is
byte-for-byte the SAME list the PRE-CHANGE ``_build_routing_hs`` produced inline.

Oracle: ``_legacy_build_routing_hs`` below is a FROZEN copy of the pre-change legacy build (the flat
``LayerEntry`` fan-out + the plane fill + the ring reserve + the plans). Running it on a fresh
registry/ring gives the golden {plane, flat LayerEntry list, ``_hs_step_start``, ``_hs_step_rows``,
plans, ring cursor}. The production side runs the NEW ``_build_routing_hs`` (records) and expands via
``expand_records``. The frozen oracle is self-validating: it must match production even BEFORE the
collapse landed (both then produce flat ``LayerEntry`` — the pass-through), which is what proves the
copy is faithful; it must STILL match after (records -> expand), which is byte-identity.

Covers: N in {1,4,16}; HETEROGENEOUS per-request layer sets (A=[1,2,3], B=[5,10], C=all — the owner
requirement); output_hidden_states True and list; last_token and all_tokens; hooks_on
prefill/decode/both; over-cap; non-capturing / empty. Asserts the routing PLANE,
``_hs_step_start``/``_hs_step_rows``, and ``plans`` are unchanged too, and that each request's
delivered layers are exactly its OWN.

Run:  conda activate vllm_hook_env && pytest tests/unit/test_layerentry_collapse_parity.py -q
"""
import os
import sys
from typing import Optional

import pytest
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, ReqCaptureRecord, expand_records
from vllm_hook_plugins.graph.install_hs import _build_routing_hs, _ring_reserve_or_block


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


# ---------------- FROZEN pre-change legacy build (the golden oracle) ----------------
def _legacy_build_routing_hs(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """VERBATIM copy of the PRE-collapse ``_build_routing_hs`` legacy branch: build one flat
    ``LayerEntry`` per (req, layer) inline on the loop, fill the plane, reserve the ring, return
    plans. Debug prints stripped (they never affect the outputs). This is the golden the collapse
    must reproduce off-loop."""
    registry._hs_step_entries = []
    registry._hs_step_start = None
    registry._hs_step_rows = 0
    if not registry.should_capture:
        return []
    ring = getattr(registry, "_hs_ring", None)
    if ring is None:
        return []
    consumer = getattr(registry, "_hs_consumer", None)
    try:
        req_ids = model_runner.input_batch.req_ids
    except Exception:
        return []

    bs = len(qsl_cpu) - 1
    capture_index_pinned = registry.capture_index_pinned
    cap = registry.cap

    plans: list = []
    entries: list = []
    for i in range(bs):
        if i >= len(req_ids):
            break
        req_id = req_ids[i]
        req_state = model_runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_hidden_states") is None:
            continue

        output_spec = extra.get("output_hidden_states")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on",
                             getattr(model_runner, "_default_hooks_on", "prefill"))
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hs_mode",
                             getattr(model_runner, "_worker_hs_mode", "last_token"))
        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end <= start:
            continue
        end = min(end, cap)
        if end <= start:
            continue

        if layer_filter is None:
            rows_layers = list(range(registry.num_layers))
        else:
            rows_layers = [ln - 1 for ln in layer_filter
                           if 1 <= ln <= registry.num_layers]
        if not rows_layers:
            continue

        n = 1 if req_mode == "last_token" else (end - start)
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._hs_step_start is None:
            registry._hs_step_start = start_slot
        registry._hs_step_rows += n
        phys = ring.physical_slots(start_slot, n)
        layer_idx_t = torch.tensor(rows_layers, dtype=torch.long)
        if req_mode == "last_token":
            capture_index_pinned[layer_idx_t, end - 1] = int(phys[0])
        else:
            phys_t = torch.tensor(phys, dtype=torch.int64)
            capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        for L in rows_layers:
            entries.append(LayerEntry(req_id=str(req_id), layer=L + 1,
                                      logical_start=start_slot, n_rows=n, hs_mode=req_mode))
        plans.append({
            "req_id": req_id,
            "n_rows": n,
            "layers": rows_layers,
            "hs_mode": req_mode,
        })

    registry._hs_step_entries = entries
    return plans


# ------------------------------- scenario model -------------------------------
class Scenario:
    def __init__(self, name, num_layers, cap, R, reqs, **mr_kwargs):
        self.name = name
        self.num_layers = num_layers
        self.cap = cap
        self.R = R
        self.reqs = reqs
        self.mr_kwargs = mr_kwargs

    def qsl(self):
        q = [0]
        for r in self.reqs:
            q.append(q[-1] + r["span"])
        return q

    def specs(self):
        return [(r["id"], r["extra"], r.get("otids", ())) for r in self.reqs]


def _snapshot_legacy(scn: Scenario):
    reg, ring = _make_registry(scn.num_layers, scn.cap, scn.R)
    mr = _mr(scn.specs(), **scn.mr_kwargs)
    reg.reset_pinned(scn.cap)
    plans = _legacy_build_routing_hs(mr, reg, scn.qsl())
    return {
        "plane": reg.capture_index_pinned.clone(),
        "flat_entries": list(reg._hs_step_entries),   # already flat LayerEntry
        "step_start": reg._hs_step_start,
        "step_rows": reg._hs_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _snapshot_production(scn: Scenario, vectorized: bool):
    reg, ring = _make_registry(scn.num_layers, scn.cap, scn.R)
    mr = _mr(scn.specs(), **scn.mr_kwargs)
    reg.reset_pinned(scn.cap)
    plans = _build_routing_hs(mr, reg, scn.qsl(), vectorized=vectorized)
    records = list(reg._hs_step_entries)
    return {
        "plane": reg.capture_index_pinned.clone(),
        "records": records,
        "flat_entries": expand_records(records),      # OFF-LOOP expansion, the byte-identity target
        "step_start": reg._hs_step_start,
        "step_rows": reg._hs_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _assert_collapse_identical(scn: Scenario, vectorized: bool):
    gold = _snapshot_legacy(scn)
    prod = _snapshot_production(scn, vectorized=vectorized)
    tag = f"{scn.name} (vec={vectorized})"
    assert torch.equal(gold["plane"], prod["plane"]), (
        f"{tag}: routing plane differs\nlegacy=\n{gold['plane']}\nprod=\n{prod['plane']}")
    assert gold["flat_entries"] == prod["flat_entries"], (
        f"{tag}: expanded LayerEntry list differs\nlegacy={gold['flat_entries']}\n"
        f"prod={prod['flat_entries']}")
    assert gold["step_start"] == prod["step_start"], f"{tag}: _hs_step_start differs"
    assert gold["step_rows"] == prod["step_rows"], f"{tag}: _hs_step_rows differs"
    assert gold["plans"] == prod["plans"], (
        f"{tag}: plans differ\nlegacy={gold['plans']}\nprod={prod['plans']}")
    assert gold["ring_write"] == prod["ring_write"], f"{tag}: ring write cursor differs"
    # The collapse's shape: production stashes RECORDS, expansion yields LayerEntry.
    assert all(isinstance(r, ReqCaptureRecord) for r in prod["records"]), \
        f"{tag}: _hs_step_entries must be ReqCaptureRecord after the collapse"
    assert all(isinstance(e, LayerEntry) for e in prod["flat_entries"]), \
        f"{tag}: expand_records must yield LayerEntry"
    return gold, prod


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

    # N=1 — all-layers, both modes.
    S.append(Scenario("N1_alltok_alllayers", 6, 32, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 5}]))
    S.append(Scenario("N1_lasttok_alllayers", 6, 32, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 5}]))

    # N=4 — homogeneous all-layers, all_tokens.
    S.append(Scenario("N4_homogeneous_alltok", 8, 64, 400,
                      [{"id": f"r{i}", "extra": _all_alltok(), "span": 3 + i} for i in range(4)]))

    # N=4 — the OWNER REQUIREMENT: heterogeneous per-request layer sets in one batch.
    #   A=[1,2,3]  B=[5,10]  C=all  D=[8,7,6] (unsorted -> set order).
    S.append(Scenario("N4_owner_heterogeneous", 12, 64, 400, [
        {"id": "A", "extra": _list_alltok([1, 2, 3]), "span": 4},
        {"id": "B", "extra": _list_lasttok([5, 10]), "span": 6},
        {"id": "C", "extra": _all_alltok(), "span": 3},
        {"id": "D", "extra": _list_alltok([8, 7, 6]), "span": 2},
    ]))

    # N=16 — decode-shaped (span 1, all_tokens, all layers) = the route_cost_probe regime.
    S.append(Scenario("N16_decode_alltok", 12, 64, 800,
                      [{"id": f"d{i}", "extra": _all_alltok(), "span": 1, "otids": (7, 7)}
                       for i in range(16)]))

    # N=16 — mixed modes + mixed heterogeneous layer sets + mixed spans.
    reqs16 = []
    for i in range(16):
        if i % 3 == 0:
            reqs16.append({"id": f"m{i}", "extra": _all_alltok(), "span": 1 + (i % 4)})
        elif i % 3 == 1:
            reqs16.append({"id": f"m{i}", "extra": _list_lasttok([1, 3, 5, 7]), "span": 2 + (i % 3)})
        else:
            reqs16.append({"id": f"m{i}", "extra": _list_alltok([2, 4]), "span": 1 + (i % 2)})
    S.append(Scenario("N16_mixed", 8, 96, 800, reqs16))

    # hooks_on gates: prefill / decode.
    S.append(Scenario("hooks_prefill_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("prefill"), "span": 4, "otids": ()},        # routes
        {"id": "dec", "extra": _all_alltok("prefill"), "span": 1, "otids": (1, 2, 3)},  # skipped
        {"id": "pre2", "extra": _all_lasttok("prefill"), "span": 3, "otids": ()},       # routes
    ]))
    S.append(Scenario("hooks_decode_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("decode"), "span": 4, "otids": ()},          # skipped
        {"id": "dec", "extra": _all_alltok("decode"), "span": 1, "otids": (9,)},        # routes
        {"id": "dec2", "extra": _list_lasttok([2, 3], "decode"), "span": 5, "otids": (9,)},  # routes
    ]))

    # Config-DEFAULT hooks_on / hs_mode.
    S.append(Scenario("config_default_hooks_mode", 6, 32, 200,
                      [{"id": "A", "extra": {"output_hidden_states": True}, "span": 4},
                       {"id": "B", "extra": {"output_hidden_states": [2, 4]}, "span": 3}],
                      default_hooks_on="both", worker_hs_mode="all_tokens"))
    S.append(Scenario("config_default_lasttok", 6, 32, 200,
                      [{"id": "A", "extra": {"output_hidden_states": True}, "span": 7}],
                      default_hooks_on="both", worker_hs_mode="last_token"))

    # Over-cap.
    S.append(Scenario("overcap_alltok_partial", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 5},
        {"id": "B", "extra": _all_alltok(), "span": 10},
    ]))
    S.append(Scenario("overcap_alltok_single", 4, 8, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 12}]))
    S.append(Scenario("overcap_lasttok", 4, 8, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 12}]))
    S.append(Scenario("overcap_second_fully_skipped", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 8},
        {"id": "B", "extra": _all_alltok(), "span": 4},
    ]))

    # Non-capturing requests mixed in.
    S.append(Scenario("noncapturing_mixed", 6, 32, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 3},
        {"id": "B", "extra": {}, "span": 4},
        {"id": "C", "extra": {"output_hidden_states": None}, "span": 2},
        {"id": "D", "extra": _list_lasttok([1, 6]), "span": 5},
    ]))

    # Out-of-range layers + all-filtered.
    S.append(Scenario("layer_list_out_of_range", 4, 32, 200,
                      [{"id": "A", "extra": _list_alltok([0, 1, 4, 99]), "span": 3}]))
    S.append(Scenario("layer_list_all_filtered", 4, 32, 200, [
        {"id": "A", "extra": _list_alltok([50, 99]), "span": 3},
        {"id": "B", "extra": _all_alltok(), "span": 2},
    ]))

    # Empty batch + a zero-span request.
    S.append(Scenario("empty_batch", 6, 32, 200, []))
    S.append(Scenario("zero_span_mixed", 6, 32, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 0},
        {"id": "B", "extra": _all_alltok(), "span": 4},
    ]))

    return S


_SCN = _scenarios()


@pytest.mark.parametrize("scn", _SCN, ids=[s.name for s in _SCN])
def test_collapse_bit_identical_legacy_route(scn):
    _assert_collapse_identical(scn, vectorized=False)


@pytest.mark.parametrize("scn", _SCN, ids=[s.name for s in _SCN])
def test_collapse_bit_identical_vectorized_route(scn):
    """The plane-fill-vectorized build must ALSO collapse to the same golden flat LayerEntry list."""
    _assert_collapse_identical(scn, vectorized=True)


def test_a_capturing_scenario_is_non_vacuous():
    """Guard: at least one scenario writes real (non-sentinel) slots + multiple layers, so the
    equality isn't a vacuous both-empty no-op."""
    scn = next(s for s in _SCN if s.name == "N4_homogeneous_alltok")
    _, prod = _assert_collapse_identical(scn, vectorized=False)
    wrote = (prod["plane"] != scn.R).sum().item()
    assert wrote > 0, "expected real routed slots, plane is all sentinel (vacuous)"
    assert len(prod["records"]) == 4
    assert len(prod["flat_entries"]) == 4 * scn.num_layers   # every req captures all layers


def test_owner_heterogeneous_each_request_gets_its_own_layers():
    """OWNER REQUIREMENT: with heterogeneous per-request layer sets, each request's delivered layers
    are EXACTLY its own — never a global/merged set."""
    scn = next(s for s in _SCN if s.name == "N4_owner_heterogeneous")
    _, prod = _assert_collapse_identical(scn, vectorized=False)
    # Per-request records carry that request's OWN 1-based layer set (fan-out / set-iteration order,
    # which the byte-identity test pins vs the legacy oracle; here we assert the SET is exactly its
    # own — never a global/merged set).
    by_id = {r.req_id: r for r in prod["records"]}
    assert set(by_id["A"].layers) == {1, 2, 3}
    assert set(by_id["B"].layers) == {5, 10}
    assert set(by_id["C"].layers) == set(range(1, scn.num_layers + 1))
    assert set(by_id["D"].layers) == {6, 7, 8}      # unsorted list -> set-iteration order
    # And the expanded flat entries deliver exactly those layers per request (no cross-bleed).
    delivered: dict = {}
    for e in prod["flat_entries"]:
        delivered.setdefault(e.req_id, []).append(e.layer)
    assert sorted(delivered["A"]) == [1, 2, 3]
    assert sorted(delivered["B"]) == [5, 10]
    assert sorted(delivered["C"]) == list(range(1, scn.num_layers + 1))
    assert sorted(delivered["D"]) == [6, 7, 8]


def test_should_capture_false_returns_empty():
    reg, _ = _make_registry(4, 16, 100, should_capture=False)
    mr = _mr([("A", _all_alltok())])
    assert _build_routing_hs(mr, reg, [0, 3], vectorized=False) == []
    assert reg._hs_step_entries == []


def main():
    import traceback
    tests = ([lambda s=s: test_collapse_bit_identical_legacy_route(s) for s in _SCN]
             + [lambda s=s: test_collapse_bit_identical_vectorized_route(s) for s in _SCN]
             + [test_a_capturing_scenario_is_non_vacuous,
                test_owner_heterogeneous_each_request_gets_its_own_layers,
                test_should_capture_false_returns_empty])
    names = ([f"legacy:{s.name}" for s in _SCN]
             + [f"vec:{s.name}" for s in _SCN]
             + ["non_vacuous", "owner_heterogeneous", "should_capture_false"])
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
