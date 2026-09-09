"""No-GPU BIT-IDENTITY gate for the QK "LayerEntry collapse" (graph/install.py::_build_routing).

The collapse moves the per-(request x captured layer) ``QKStepEntry`` fan-out OFF the engine loop:
``_build_routing`` now builds ONE ``QKReqCaptureRecord`` per capturing request (carrying that request's
OWN 0-based ``layers`` list plus the seven fields shared across its layers), and the drain expands each
record into the flat ``QKStepEntry`` list — same fields, SAME order — off-loop. This test proves the
expanded list is byte-for-byte the SAME list the PRE-CHANGE ``_build_routing`` produced inline.

Oracle: ``_legacy_build_routing`` below is a FROZEN copy of the pre-change build (the flat
``QKStepEntry`` fan-out + the plane fill + the ring reserve + the plans). Running it on a fresh
registry/ring gives the golden {plane, flat QKStepEntry list, ``_qk_step_start``, ``_qk_step_rows``,
plans, ring cursor}. The production side runs the NEW ``_build_routing`` (records) and expands via
``expand_qk_records``. The frozen oracle is self-validating: it produces the flat ``QKStepEntry`` the
collapse must reproduce off-loop.

Covers: N in {1,4,16}; HETEROGENEOUS per-request layer sets (A=[0,1,2], B={5,10}, C=all, D=[8,7,6] — the
owner requirement); output_qk True, list, and dict; last_token and all_tokens; hooks_on
prefill/decode/both (emit_q true/false); chunked-prefill MID-chunk non-emit; over-cap; score-mode skip;
non-capturing / empty. Asserts the routing PLANE, ``_qk_step_start``/``_qk_step_rows``, and ``plans`` are
unchanged too, and that each request's delivered layers are exactly its OWN.

Run:  conda activate vllm_hook_env && pytest tests/unit/test_qk_layerentry_collapse_parity.py -q
"""
import sys
from typing import Optional

import pytest
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing, RingBackpressureError
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import (
    QKStepEntry, QKReqCaptureRecord, expand_qk_records)
from vllm_hook_plugins.graph.install import _build_routing, _ring_reserve_or_block


# ------------------------- minimal vLLM-surface fakes -------------------------
class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _RS:
    def __init__(self, extra, output_token_ids=()):
        self.sampling_params = _SP(extra)
        self.output_token_ids = output_token_ids


class _IB:
    def __init__(self, req_ids, num_computed, num_prompt):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed
        self.num_prompt_tokens = num_prompt


class _MR:
    def __init__(self, req_ids, requests, num_computed, num_prompt,
                 default_hooks_on="both", worker_hookq_mode="all_tokens"):
        self.input_batch = _IB(req_ids, num_computed, num_prompt)
        self.requests = requests
        self._default_hooks_on = default_hooks_on
        self._worker_hookq_mode = worker_hookq_mode
        self._worker_score_mode = False
        self._worker_score_head = 0


def _make_registry(num_layers, cap, R, q_dim=8, k_dim=4, should_capture=True, device="cpu"):
    """A HostRegistry wired for the QK capture-ring path exactly as the smoke test / install_qk_hosts
    does: a shared GpuCaptureRing cursor, sentinel_row = ring.SENTINEL, inc/gpu-routing off, pinned
    mirrors primed to the sentinel."""
    reg = HostRegistry(num_layers=num_layers, cap=cap, device=device,
                       should_capture=should_capture)
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


# ---------------- FROZEN pre-change legacy build (the golden oracle) ----------------
def _legacy_build_routing(model_runner, registry: HostRegistry, qsl_cpu: list) -> list:
    """VERBATIM copy of the PRE-collapse ``_build_routing``: build one flat ``QKStepEntry`` per
    (req, layer) inline on the loop, fill the plane, reserve the ring, return plans. Debug prints +
    PROF counters stripped (they never affect the outputs). This is the golden the collapse must
    reproduce off-loop."""
    registry._qk_step_entries = []
    registry._qk_step_start = None
    registry._qk_step_rows = 0
    if not registry.should_capture:
        return []
    ring = getattr(registry, "_qk_ring", None)
    if ring is None:
        return []
    consumer = getattr(registry, "_qk_consumer", None)
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
        if not extra or extra.get("output_qk") is None:
            continue

        output_spec = extra.get("output_qk")
        layer_filter: Optional[set] = None
        if isinstance(output_spec, dict):
            layer_filter = {int(k) for k in output_spec.keys()}
        elif isinstance(output_spec, list):
            layer_filter = {int(x) for x in output_spec}

        hooks_on = extra.get("hooks_on", getattr(model_runner, "_default_hooks_on", "prefill"))
        if hooks_on != "both":
            is_prefill = len(req_state.output_token_ids) == 0
            if hooks_on == "prefill" and not is_prefill:
                continue
            if hooks_on == "decode" and is_prefill:
                continue

        req_mode = extra.get("hookq_mode",
                             getattr(model_runner, "_worker_hookq_mode", "all_tokens"))
        cap_mode = extra.get("qk_capture",
                             "score" if getattr(model_runner, "_worker_score_mode", False) else "qk")
        if cap_mode == "score":
            continue

        start = int(qsl_cpu[i])
        end = int(qsl_cpu[i + 1])
        if end > cap:
            end = cap
        if end <= start:
            continue
        qlen = end - start

        try:
            num_computed = int(model_runner.input_batch.num_computed_tokens_cpu[i])
        except Exception:  # noqa: BLE001
            num_computed = start
        abs_end = num_computed + qlen

        if layer_filter is None:
            req_layers = list(range(registry.num_layers))
        else:
            req_layers = [L for L in layer_filter if 0 <= L < registry.num_layers]
        if not req_layers:
            continue

        emit_q = True
        if req_mode == "last_token" and len(req_state.output_token_ids) == 0:
            try:
                num_prompt = int(model_runner.input_batch.num_prompt_tokens[i])
            except Exception:  # noqa: BLE001
                num_prompt = abs_end
            emit_q = abs_end >= num_prompt

        n = qlen
        start_slot = _ring_reserve_or_block(ring, n, consumer)
        if registry._qk_step_start is None:
            registry._qk_step_start = start_slot
        registry._qk_step_rows += n
        phys = ring.physical_slots(start_slot, n)
        phys_t = torch.tensor(phys, dtype=torch.int64)
        layer_idx_t = torch.tensor(req_layers, dtype=torch.long)
        capture_index_pinned[layer_idx_t[:, None], start:end] = phys_t[None, :]

        if emit_q:
            if req_mode == "all_tokens":
                q_start, q_rows = start_slot, n
            else:
                q_start, q_rows = start_slot + n - 1, 1
            prefix_end = int(abs_end)
        else:
            q_start, q_rows, prefix_end = -1, 0, -1

        for L in req_layers:
            entries.append(QKStepEntry(
                req_id=str(req_id), layer=int(L),
                k_start=int(start_slot), k_rows=int(n),
                q_start=int(q_start), q_rows=int(q_rows),
                prefix_end=int(prefix_end), num_computed=int(num_computed)))

        plans.append({"req_id": req_id, "n_rows": n, "layers": req_layers,
                      "hookq_mode": req_mode, "emit_q": emit_q})

    registry._qk_step_entries = entries
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

    def _mr(self):
        req_ids = [r["id"] for r in self.reqs]
        requests = {r["id"]: _RS(r["extra"], r.get("otids", ())) for r in self.reqs}
        num_computed = [r.get("num_computed", 0) for r in self.reqs]
        num_prompt = [r.get("num_prompt", 0) for r in self.reqs]
        return _MR(req_ids, requests, num_computed, num_prompt, **self.mr_kwargs)


def _snapshot_legacy(scn: Scenario):
    reg, ring = _make_registry(scn.num_layers, scn.cap, scn.R)
    mr = scn._mr()
    reg.reset_pinned(scn.cap)
    plans = _legacy_build_routing(mr, reg, scn.qsl())
    return {
        "plane": reg.capture_index_pinned.clone(),
        "flat_entries": list(reg._qk_step_entries),   # already flat QKStepEntry
        "step_start": reg._qk_step_start,
        "step_rows": reg._qk_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _snapshot_production(scn: Scenario):
    reg, ring = _make_registry(scn.num_layers, scn.cap, scn.R)
    mr = scn._mr()
    reg.reset_pinned(scn.cap)
    plans = _build_routing(mr, reg, scn.qsl())
    records = list(reg._qk_step_entries)
    return {
        "plane": reg.capture_index_pinned.clone(),
        "records": records,
        "flat_entries": expand_qk_records(records),   # OFF-LOOP expansion, the byte-identity target
        "step_start": reg._qk_step_start,
        "step_rows": reg._qk_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _assert_collapse_identical(scn: Scenario):
    gold = _snapshot_legacy(scn)
    prod = _snapshot_production(scn)
    tag = scn.name
    assert torch.equal(gold["plane"], prod["plane"]), (
        f"{tag}: routing plane differs\nlegacy=\n{gold['plane']}\nprod=\n{prod['plane']}")
    assert gold["flat_entries"] == prod["flat_entries"], (
        f"{tag}: expanded QKStepEntry list differs\nlegacy={gold['flat_entries']}\n"
        f"prod={prod['flat_entries']}")
    assert gold["step_start"] == prod["step_start"], f"{tag}: _qk_step_start differs"
    assert gold["step_rows"] == prod["step_rows"], f"{tag}: _qk_step_rows differs"
    assert gold["plans"] == prod["plans"], (
        f"{tag}: plans differ\nlegacy={gold['plans']}\nprod={prod['plans']}")
    assert gold["ring_write"] == prod["ring_write"], f"{tag}: ring write cursor differs"
    # The collapse's shape: production stashes RECORDS, expansion yields QKStepEntry.
    assert all(isinstance(r, QKReqCaptureRecord) for r in prod["records"]), \
        f"{tag}: _qk_step_entries must be QKReqCaptureRecord after the collapse"
    assert all(isinstance(e, QKStepEntry) for e in prod["flat_entries"]), \
        f"{tag}: expand_qk_records must yield QKStepEntry"
    return gold, prod


# --------------------------------- extras ------------------------------------
def _all_alltok(hooks="both"):
    return {"output_qk": True, "hookq_mode": "all_tokens", "hooks_on": hooks}


def _all_lasttok(hooks="both"):
    return {"output_qk": True, "hookq_mode": "last_token", "hooks_on": hooks}


def _list_alltok(layers, hooks="both"):
    return {"output_qk": list(layers), "hookq_mode": "all_tokens", "hooks_on": hooks}


def _list_lasttok(layers, hooks="both"):
    return {"output_qk": list(layers), "hookq_mode": "last_token", "hooks_on": hooks}


def _dict_alltok(layer_to_heads, hooks="both"):
    return {"output_qk": dict(layer_to_heads), "hookq_mode": "all_tokens", "hooks_on": hooks}


def _dict_lasttok(layer_to_heads, hooks="both"):
    return {"output_qk": dict(layer_to_heads), "hookq_mode": "last_token", "hooks_on": hooks}


# --------------------------------- scenarios ---------------------------------
def _scenarios():
    S = []

    # N=1 — all-layers, both modes. last_token emit (single chunk == whole prompt).
    S.append(Scenario("N1_alltok_alllayers", 6, 32, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 5}]))
    S.append(Scenario("N1_lasttok_alllayers_emit", 6, 32, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 5, "num_prompt": 5}]))

    # N=1 — last_token MID-prefill chunk (num_prompt > abs_end -> emit_q False, K-only, prefix_end=-1).
    S.append(Scenario("N1_lasttok_midchunk_noemit", 6, 32, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 8, "num_prompt": 20}]))

    # N=4 — homogeneous all-layers, all_tokens.
    S.append(Scenario("N4_homogeneous_alltok", 8, 64, 400,
                      [{"id": f"r{i}", "extra": _all_alltok(), "span": 3 + i} for i in range(4)]))

    # N=4 — the OWNER REQUIREMENT: heterogeneous per-request layer sets in one batch.
    #   A=[0,1,2] (list)  B={5,10} (dict, last_token emit)  C=all (True)  D=[8,7,6] (unsorted list).
    S.append(Scenario("N4_owner_heterogeneous", 12, 64, 400, [
        {"id": "A", "extra": _list_alltok([0, 1, 2]), "span": 4},
        {"id": "B", "extra": _dict_lasttok({5: [0], 10: [1]}), "span": 6, "num_prompt": 6},
        {"id": "C", "extra": _all_alltok(), "span": 3},
        {"id": "D", "extra": _list_alltok([8, 7, 6]), "span": 2},
    ]))

    # N=16 — decode-shaped (span 1, all_tokens, all layers) = the route_cost_probe regime.
    S.append(Scenario("N16_decode_alltok", 12, 64, 800,
                      [{"id": f"d{i}", "extra": _all_alltok(), "span": 1,
                        "num_computed": 20 + i, "otids": (7, 7)}
                       for i in range(16)]))

    # N=16 — mixed modes + mixed heterogeneous layer sets (list + dict) + mixed spans.
    reqs16 = []
    for i in range(16):
        if i % 3 == 0:
            reqs16.append({"id": f"m{i}", "extra": _all_alltok(), "span": 1 + (i % 4)})
        elif i % 3 == 1:
            reqs16.append({"id": f"m{i}", "extra": _list_lasttok([0, 3, 5, 7]),
                           "span": 2 + (i % 3), "num_prompt": 2 + (i % 3)})
        else:
            reqs16.append({"id": f"m{i}", "extra": _dict_alltok({2: [0], 4: [1, 2]}),
                           "span": 1 + (i % 2)})
    S.append(Scenario("N16_mixed", 8, 128, 800, reqs16))

    # hooks_on gates: prefill / decode. is_prefill = (len(output_token_ids) == 0).
    S.append(Scenario("hooks_prefill_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("prefill"), "span": 4, "otids": ()},        # routes
        {"id": "dec", "extra": _all_alltok("prefill"), "span": 1, "otids": (1, 2, 3)},  # skipped
        {"id": "pre2", "extra": _all_lasttok("prefill"), "span": 3, "otids": (),
         "num_prompt": 3},                                                              # routes
    ]))
    S.append(Scenario("hooks_decode_gate", 6, 32, 200, [
        {"id": "pre", "extra": _all_alltok("decode"), "span": 4, "otids": ()},          # skipped
        {"id": "dec", "extra": _all_alltok("decode"), "span": 1,
         "num_computed": 9, "otids": (9,)},                                             # routes
        {"id": "dec2", "extra": _list_lasttok([2, 3], "decode"), "span": 1,
         "num_computed": 5, "otids": (9,)},                                             # routes (decode)
    ]))

    # Config-DEFAULT hooks_on / hookq_mode (extra omits both -> model_runner defaults).
    S.append(Scenario("config_default_hooks_mode", 6, 32, 200,
                      [{"id": "A", "extra": {"output_qk": True}, "span": 4},
                       {"id": "B", "extra": {"output_qk": [1, 3]}, "span": 3}],
                      default_hooks_on="both", worker_hookq_mode="all_tokens"))
    S.append(Scenario("config_default_lasttok", 6, 32, 200,
                      [{"id": "A", "extra": {"output_qk": True}, "span": 7, "num_prompt": 7}],
                      default_hooks_on="both", worker_hookq_mode="last_token"))

    # Over-cap (end > cap clamps to cap; K reserve == clamped span).
    S.append(Scenario("overcap_alltok_partial", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 5},
        {"id": "B", "extra": _all_alltok(), "span": 10},
    ]))
    S.append(Scenario("overcap_alltok_single", 4, 8, 200,
                      [{"id": "A", "extra": _all_alltok(), "span": 12}]))
    S.append(Scenario("overcap_lasttok", 4, 8, 200,
                      [{"id": "A", "extra": _all_lasttok(), "span": 12, "num_prompt": 8}]))
    S.append(Scenario("overcap_second_fully_skipped", 4, 8, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 8},
        {"id": "B", "extra": _all_alltok(), "span": 4},
    ]))

    # Score-mode request skipped (QK-specific: v1 ring path is raw q/k only).
    S.append(Scenario("score_mode_skipped", 6, 32, 200, [
        {"id": "A", "extra": {"output_qk": True, "hookq_mode": "all_tokens",
                              "hooks_on": "both", "qk_capture": "score"}, "span": 4},   # skipped
        {"id": "B", "extra": _all_alltok(), "span": 3},                                 # routes
    ]))

    # Non-capturing requests mixed in.
    S.append(Scenario("noncapturing_mixed", 6, 32, 200, [
        {"id": "A", "extra": _all_alltok(), "span": 3},
        {"id": "B", "extra": {}, "span": 4},
        {"id": "C", "extra": {"output_qk": None}, "span": 2},
        {"id": "D", "extra": _list_lasttok([0, 5]), "span": 5, "num_prompt": 5},
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
def test_collapse_bit_identical(scn):
    _assert_collapse_identical(scn)


def test_a_capturing_scenario_is_non_vacuous():
    """Guard: at least one scenario writes real (non-sentinel) slots + multiple layers, so the
    equality isn't a vacuous both-empty no-op."""
    scn = next(s for s in _SCN if s.name == "N4_homogeneous_alltok")
    _, prod = _assert_collapse_identical(scn)
    wrote = (prod["plane"] != scn.R).sum().item()
    assert wrote > 0, "expected real routed slots, plane is all sentinel (vacuous)"
    assert len(prod["records"]) == 4
    assert len(prod["flat_entries"]) == 4 * scn.num_layers   # every req captures all layers


def test_owner_heterogeneous_each_request_gets_its_own_layers():
    """OWNER REQUIREMENT: with heterogeneous per-request layer sets, each request's delivered layers
    are EXACTLY its own — never a global/merged set. (QK layers are 0-based.)"""
    scn = next(s for s in _SCN if s.name == "N4_owner_heterogeneous")
    _, prod = _assert_collapse_identical(scn)
    by_id = {r.req_id: r for r in prod["records"]}
    assert set(by_id["A"].layers) == {0, 1, 2}
    assert set(by_id["B"].layers) == {5, 10}
    assert set(by_id["C"].layers) == set(range(scn.num_layers))
    assert set(by_id["D"].layers) == {6, 7, 8}      # unsorted list -> set-iteration order
    # And the expanded flat entries deliver exactly those layers per request (no cross-bleed).
    delivered: dict = {}
    for e in prod["flat_entries"]:
        delivered.setdefault(e.req_id, []).append(e.layer)
    assert sorted(delivered["A"]) == [0, 1, 2]
    assert sorted(delivered["B"]) == [5, 10]
    assert sorted(delivered["C"]) == list(range(scn.num_layers))
    assert sorted(delivered["D"]) == [6, 7, 8]


def test_should_capture_false_returns_empty():
    reg, _ = _make_registry(4, 16, 100, should_capture=False)
    mr = _MR(["A"], {"A": _RS(_all_alltok())}, [0], [0])
    assert _build_routing(mr, reg, [0, 3]) == []
    assert reg._qk_step_entries == []


def main():
    import traceback
    tests = ([lambda s=s: test_collapse_bit_identical(s) for s in _SCN]
             + [test_a_capturing_scenario_is_non_vacuous,
                test_owner_heterogeneous_each_request_gets_its_own_layers,
                test_should_capture_false_returns_empty])
    names = ([s.name for s in _SCN]
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
