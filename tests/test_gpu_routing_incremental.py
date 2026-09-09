"""No-GPU unit test for the refresh_slot_config optimizations (Lever A host half).

Two changes are under test, both required to be BYTE-IDENTICAL to the scalar
full-rebuild they replace:

  fix 1  vectorize the O(reqs x layers) per-element layer-mask write
  fix 2  re-resolve ONLY the batch positions whose req_id changed (O(delta), not
         O(batch)) -- the term that made the host cost grow with the backlog

`test_only_changed_slots_are_reresolved` is the RED test: it fails on the
pre-optimization code, which re-resolves every slot on any req_ids change.
Everything else is a characterization/equivalence oracle that must stay green --
those are what make the byte-identity claim real, and each one was verified to
FAIL against a deliberately-broken implementation before being trusted.

    conda activate vllm_hook_env && python tests/test_gpu_routing_incremental.py
"""
import os
os.environ["VLLM_HOOK_STEER_GPU_ROUTING"] = "1"   # build slot arrays on CPU for the test

import random
import sys

import torch

import vllm_hook_plugins.graph.install_steer as ist
from vllm_hook_plugins.graph.install_steer import SteerRegistry


# ---- minimal fakes for the model_runner surface refresh_slot_config reads ----
class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _RS:
    output_token_ids = ()          # read by _resolve_step_cols to tell prefill from decode

    def __init__(self, extra):
        self.sampling_params = _SP(extra)


class _IB:
    def __init__(self, req_ids):
        self.req_ids = req_ids


class _MR:
    def __init__(self, req_ids, requests):
        self.input_batch = _IB(req_ids)
        self.requests = requests


class _Worker:
    _vector_cache = {}
    _env_config_path = None


NUM_LAYERS, CAP = 8, 64


def _fresh_registry():
    reg = SteerRegistry(NUM_LAYERS, CAP, hidden=4, v_max=8, device="cpu",
                        dtype=torch.float32)
    # Pre-seed vectors so vec_id_for_path resolves WITHOUT touching disk.
    reg.vec_paths = {"vecAdd": 0, "vecAdj": 1}
    reg.vec_has_avgproj = {1}          # only the adjust_rs vector carries avg_proj
    return reg


def _steer(method, layer, coeff=0.0, vec="vecAdd"):
    return {"method": method, "optimal_layer": layer, "vector_path": vec,
            "coefficient": coeff}


def _mr(specs):
    """specs = list of (req_id, extra_args_or_None)."""
    return _MR([s[0] for s in specs], {rid: _RS(extra) for rid, extra in specs})


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _table(reg, bs):
    """The slot table's LIVE region. Positions >= bs are stale by design (the
    scatter only indexes [0, bs)), so equivalence is defined on [0, bs)."""
    return (reg.slot_vid[:bs].clone(), reg.slot_mode[:bs].clone(),
            reg.slot_coeff[:bs].clone(), reg.slot_layer_mask[:bs].clone())


def _assert_table_equal(got, want, ctx):
    names = ("slot_vid", "slot_mode", "slot_coeff", "slot_layer_mask")
    for n, g, w in zip(names, got, want):
        _assert(torch.equal(g, w), f"{ctx}: {n} differs from a full rebuild")


def _rebuild_from_scratch(specs):
    """Ground truth: a virgin registry that has only ever seen this one batch."""
    reg = _fresh_registry()
    reg.refresh_slot_config(_mr(specs))
    return reg


# --------------------------------------------------------------------------
# RED: this is the behaviour fix 2 adds. Pre-optimization it re-resolves ALL
# slots on any req_ids change, so the counter reads 8 instead of 1.
# --------------------------------------------------------------------------
def test_only_changed_slots_are_reresolved():
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    base = [(f"R{i}", {"steer": _steer("add_vector", i % NUM_LAYERS, coeff=float(i))})
            for i in range(8)]
    reg.refresh_slot_config(_mr(base))

    calls = {"n": 0}
    real = ist._resolve_steer_config

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    ist._resolve_steer_config = counting
    try:
        # ONE request departs and is replaced at the same position; the other 7 are
        # untouched and their configs are immutable, so 7 re-resolutions are waste.
        churned = list(base)
        churned[3] = ("NEW", {"steer": _steer("adjust_rs", 5, vec="vecAdj")})
        reg.refresh_slot_config(_mr(churned))
    finally:
        ist._resolve_steer_config = real

    _assert(calls["n"] <= 2,
            f"expected to re-resolve only the changed slot, re-resolved {calls['n']} "
            f"of 8 (cost scales with batch, not with churn)")


# --------------------------------------------------------------------------
# Equivalence oracles: the byte-identity contract.
# --------------------------------------------------------------------------
def test_incremental_equals_full_rebuild_under_churn():
    """A randomized CB churn sequence -- arrivals, departures, slot reuse, reorder,
    shrink and grow -- must leave a table bit-identical to a from-scratch rebuild
    at EVERY step. This is the test that would catch a stale slot."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    rng = random.Random(20260722)
    reg = _fresh_registry()

    def cfg_for(rid):
        """Config is a deterministic function of req_id -- mirrors the invariant
        that a request's steer config is immutable for its lifetime."""
        h = sum(ord(c) for c in rid)
        if h % 4 == 0:
            return None                                   # unsteered
        if h % 4 == 1:
            return {"steer": _steer("adjust_rs",
                                    sorted(random.Random(h).sample(
                                        range(NUM_LAYERS), k=1 + h % NUM_LAYERS)),
                                    vec="vecAdj")}
        if h % 4 == 2:
            return {"steer": _steer("add_vector", h % NUM_LAYERS,
                                    coeff=float(h % 7), vec="vecAdd")}
        return {"steer": _steer("add_vector", "all", coeff=1.5, vec="vecAdd")}

    live, nxt = [], 0
    for step in range(60):
        for _ in range(rng.randint(0, 3)):                # departures
            if live:
                live.pop(rng.randrange(len(live)))
        for _ in range(rng.randint(0, 4)):                # arrivals
            live.append(f"Q{nxt}")
            nxt += 1
        if len(live) > 1 and rng.random() < 0.3:          # compaction / reorder
            rng.shuffle(live)
        if not live:
            continue
        specs = [(rid, cfg_for(rid)) for rid in live]
        reg.refresh_slot_config(_mr(specs))
        _assert_table_equal(_table(reg, len(live)),
                            _table(_rebuild_from_scratch(specs), len(live)),
                            f"churn step {step} (bs={len(live)})")


def test_batch_grows_past_previous_high_water():
    """Shrink to 1, then grow to 6. Positions 1..5 hold values from the FIRST batch;
    a diff that only walks the previous (short) req_ids would leave them stale."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    big = [(f"A{i}", {"steer": _steer("add_vector", i % NUM_LAYERS, coeff=float(i + 1))})
           for i in range(6)]
    reg.refresh_slot_config(_mr(big))
    reg.refresh_slot_config(_mr([("solo", {"steer": _steer("adjust_rs", 2, vec="vecAdj")})]))
    grown = [(f"B{i}", {"steer": _steer("add_vector", (i + 3) % NUM_LAYERS,
                                        coeff=float(100 + i))}) for i in range(6)]
    reg.refresh_slot_config(_mr(grown))
    _assert_table_equal(_table(reg, 6), _table(_rebuild_from_scratch(grown), 6),
                        "grow past previous high-water mark")


def test_heterogeneous_layer_sets_preserved():
    """Every request a DIFFERENT method/vector/coefficient/layer set in one batch --
    the case a shared-layer-set fast path would silently flatten."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    specs = [
        ("h0", {"steer": _steer("add_vector", [0], coeff=1.0, vec="vecAdd")}),
        ("h1", {"steer": _steer("adjust_rs", [1, 2, 3], vec="vecAdj")}),
        ("h2", {"steer": _steer("add_vector", "all", coeff=-2.5, vec="vecAdd")}),
        ("h3", None),
        ("h4", {"steer": _steer("adjust_rs", [7], vec="vecAdj")}),
        ("h5", {"steer": _steer("add_vector", [0, 4, 6], coeff=0.25, vec="vecAdd")}),
    ]
    reg.refresh_slot_config(_mr(specs))

    # Independent expectation, built by hand -- not by re-running the code under test.
    want = torch.zeros(6, NUM_LAYERS, dtype=torch.bool)
    for i, ls in ((0, [0]), (1, [1, 2, 3]), (2, list(range(NUM_LAYERS))),
                  (4, [7]), (5, [0, 4, 6])):
        for L in ls:
            want[i, L] = True
    _assert(torch.equal(reg.slot_layer_mask[:6], want),
            "heterogeneous per-request layer sets not preserved")
    _assert([int(v) for v in reg.slot_mode[:6]] == [0, 1, 0, 0, 1, 0], "mode column wrong")
    _assert([int(v) for v in reg.slot_vid[:6]] == [0, 1, 0, 0, 1, 0], "vid column wrong")
    _assert([float(v) for v in reg.slot_coeff[:6]] == [1.0, 0.0, -2.5, 0.0, 0.0, 0.25],
            "coeff column wrong")


def test_unsteered_slot_is_fully_cleared_on_reuse():
    """A steered request's position taken by an UNSTEERED one must go fully inert --
    the failure mode where a skipped slot keeps the previous occupant's vector."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    reg.refresh_slot_config(_mr([
        ("x", {"steer": _steer("add_vector", "all", coeff=9.0, vec="vecAdd")}),
        ("y", {"steer": _steer("adjust_rs", [1, 2], vec="vecAdj")}),
    ]))
    reg.refresh_slot_config(_mr([("x", {"steer": _steer("add_vector", "all",
                                                        coeff=9.0, vec="vecAdd")}),
                                 ("z", None)]))
    _assert(not bool(reg.slot_layer_mask[1].any()), "unsteered reused slot kept layers")
    _assert(int(reg.slot_mode[1]) == 0 and int(reg.slot_vid[1]) == 0
            and float(reg.slot_coeff[1]) == 0.0, "unsteered reused slot kept config")


def test_skip_still_holds_on_unchanged_req_ids():
    """The existing whole-rebuild skip must survive the change."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    mr = _mr([("A", {"steer": _steer("add_vector", 2, coeff=1.0)})])
    _assert(reg.refresh_slot_config(mr) is True, "first refresh builds")
    _assert(reg.refresh_slot_config(mr) is False, "unchanged req_ids must skip")


# --------------------------------------------------------------------------
# The pre-latch probe. `build_and_upload_gpu` falls back to `_resolve_step_cols`
# whenever `_step_cols is None`, which is precisely the UNGATED (default) case --
# so before the fix it re-resolved every slot on every non-skipped step, re-adding
# the O(batch) host term fix 2 above removes. RED at 5 calls instead of 1.
# --------------------------------------------------------------------------
def _run_steps(reg, specs, qsl, n_steps):
    """Drive n_steps of the wrapper's routing_key -> build_and_upload_gpu sequence.

    Returns (resolve_call_count, slot_col_args_seen). The scatter is stubbed: this is
    about which host work runs, not about the scatter's output (covered elsewhere).
    """
    import vllm_hook_plugins.graph.steer_routing_gpu as srg

    mr = _mr(specs)
    calls = {"n": 0}
    seen = []
    real_resolve = SteerRegistry._resolve_step_cols
    real_scatter = srg.scatter_routing

    def counting(self, model_runner, qsl_cpu):
        calls["n"] += 1
        return real_resolve(self, model_runner, qsl_cpu)

    def stub_scatter(*a, slot_col=None, **kw):
        seen.append(slot_col)

    SteerRegistry._resolve_step_cols = counting
    srg.scatter_routing = stub_scatter
    try:
        for _ in range(n_steps):
            reg.routing_key(mr, qsl)          # what the routing wrapper asks first
            reg.build_and_upload_gpu(mr, qsl, width=len(qsl) - 1)
    finally:
        SteerRegistry._resolve_step_cols = real_resolve
        srg.scatter_routing = real_scatter
    return calls["n"], seen


def test_prelatch_probe_runs_once_not_every_step():
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    specs = [(f"r{i}", {"steer": _steer("add_vector", 2, coeff=1.0)}) for i in range(4)]
    n, seen = _run_steps(reg, specs, list(range(5)), n_steps=5)
    _assert(n == 1, f"ungated pre-latch probe ran {n} times across 5 steady steps (want 1)")
    _assert(reg._any_gated is False, "nothing gated -> _any_gated must stay False")
    _assert(all(s is None for s in seen),
            "ungated steps must pass slot_col=None (keeps the kernel byte-identical)")


def test_gated_request_latches_on_the_step_it_enters():
    """The fix must not cost a step: a gated arrival changes the composition, so the
    probe runs on that very step and slot_col is uploaded immediately."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    plain = [("a", {"steer": _steer("add_vector", 2, coeff=1.0)})]
    n1, seen1 = _run_steps(reg, plain, [0, 1], n_steps=3)
    _assert(reg._any_gated is False, "plain batch must not latch")

    gated_cfg = dict(_steer("add_vector", 2, coeff=1.0), phase="prefill",
                     positions="last_token")
    grown = plain + [("b", {"steer": gated_cfg})]
    n2, seen2 = _run_steps(reg, grown, [0, 1, 2], n_steps=1)
    _assert(n2 == 1, "a composition change must run the probe")
    _assert(reg._any_gated is True, "a gated request must latch on the step it enters")
    _assert(seen2 and seen2[-1] is not None,
            "slot_col must be uploaded on the latching step itself, not one step later")


def main():
    tests = [
        test_only_changed_slots_are_reresolved,
        test_prelatch_probe_runs_once_not_every_step,
        test_gated_request_latches_on_the_step_it_enters,
        test_incremental_equals_full_rebuild_under_churn,
        test_batch_grows_past_previous_high_water,
        test_heterogeneous_layer_sets_preserved,
        test_unsteered_slot_is_fully_cleared_on_reuse,
        test_skip_still_holds_on_unchanged_req_ids,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'} "
          f"({len(tests) - failures}/{len(tests)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
