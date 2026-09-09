"""No-GPU unit test for Lever A Task A1: SteerRegistry.refresh_slot_config.

GPU routing (VLLM_HOOK_STEER_GPU_ROUTING) replaces the per-step host build with a
per-slot config table (each request's (vid, mode, coeff, layer_mask) resolved once,
indexed by batch position, refreshed only on a req_ids change) + a GPU scatter. This
test covers the HOST half — refresh_slot_config — with no GPU and no engine:

  * resolution: fake req_states -> the slot table, then expanded by the Phase-0 CPU
    scatter (gpu_routing_spike.scatter_slabs), equals what apply_incremental_routing
    produces from the equivalent assignments (byte-identical);
  * the req_ids-change skip (rebuild only when composition/order changes);
  * slot reuse (a finished request's position taken by a newcomer updates the table);
  * adjust_rs without avg_proj is skipped (matches _build_routing_steer).

The env is set to "1" so __init__ builds the slot arrays even on CPU (production also
requires cuda; the wrapper gate `gpu_routing` still checks device).

    conda activate vllm_hook_env && python tests/test_gpu_routing_config.py
"""
import os
os.environ["VLLM_HOOK_STEER_GPU_ROUTING"] = "1"   # build slot arrays on CPU for the test

import sys

import torch

# make the Phase-0 spike importable (scatter_slabs, slot_config_of, Req, qsl_of, ...)
_SPIKE_DIR = os.path.join(os.path.dirname(__file__),
                          "cuda_graph", "tests", "steer_graph")
sys.path.insert(0, _SPIKE_DIR)

import vllm_hook_plugins.graph.install_steer as ist
from vllm_hook_plugins.graph.install_steer import SteerRegistry
from gpu_routing_spike import scatter_slabs, Req, qsl_of, assignments_of  # noqa: E402


# ---- minimal fakes for the model_runner surface refresh_slot_config reads ----
class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _RS:
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


NUM_LAYERS, CAP = 8, 32


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
    """specs = list of (req_id, extra_args_or_None). extra like {'steer': {...}}."""
    req_ids = [s[0] for s in specs]
    requests = {rid: _RS(extra) for rid, extra in specs}
    return _MR(req_ids, requests)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _expand(reg, qsl, width):
    qsl_dev = torch.tensor(qsl, dtype=torch.int64)
    return scatter_slabs(qsl_dev, reg.slot_vid, reg.slot_mode, reg.slot_coeff,
                         reg.slot_layer_mask, NUM_LAYERS, CAP, width)


def test_resolution_matches_host_slabs():
    """Three requests (add_vector single-layer, adjust_rs multi-layer, unsteered) ->
    refresh -> expand -> equals apply_incremental_routing on the equivalent assignments."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    mr = _mr([
        ("A", {"steer": _steer("add_vector", 2, coeff=2.5, vec="vecAdd")}),
        ("B", {"steer": _steer("adjust_rs", [1, 3], vec="vecAdj")}),
        ("C", None),                       # unsteered
    ])
    changed = reg.refresh_slot_config(mr)
    _assert(changed is True, "first refresh must (re)build + return True")

    # equivalent Req list for the host reference (1 token each -> decode-like columns)
    reqs = [Req(1, [2], 0, 0, 2.5), Req(1, [1, 3], 1, 1, 0.0), Req(1, [], 0, 0, 0.0)]
    qsl = qsl_of(reqs)
    width = max(qsl[-1], 8)
    ref = _fresh_registry()
    ref.apply_incremental_routing(assignments_of(reqs, qsl, CAP), width)

    oc, ov, om = _expand(reg, qsl, width)
    _assert(torch.equal(oc[:, :width], ref.coeff_all[:, :width]), "coeff slab mismatch")
    _assert(torch.equal(ov[:, :width], ref.vec_id_all[:, :width]), "vec_id slab mismatch")
    _assert(torch.equal(om[:, :width], ref.mode_all[:, :width]), "mode slab mismatch")


def test_skip_on_unchanged_req_ids():
    """Identical req_ids -> the second refresh is a no-op skip (returns False)."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    mr = _mr([("A", {"steer": _steer("add_vector", 2, coeff=1.0)})])
    _assert(reg.refresh_slot_config(mr) is True, "first refresh builds")
    _assert(reg.refresh_slot_config(mr) is False, "unchanged req_ids must skip")


def test_slot_reuse_updates_table():
    """B finishes, C takes position 1 with a DIFFERENT config -> refresh rebuilds and
    position 1 reflects C, not stale B (the load-bearing churn correctness case)."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    reg.refresh_slot_config(_mr([
        ("A", {"steer": _steer("add_vector", 2, coeff=1.0, vec="vecAdd")}),
        ("B", {"steer": _steer("add_vector", 5, coeff=3.0, vec="vecAdd")}),
    ]))
    changed = reg.refresh_slot_config(_mr([
        ("A", {"steer": _steer("add_vector", 2, coeff=1.0, vec="vecAdd")}),
        ("C", {"steer": _steer("adjust_rs", 4, vec="vecAdj")}),   # reuses position 1
    ]))
    _assert(changed is True, "changed req_ids must rebuild")
    # position 1 now C: adjust_rs -> mode 1, vid 1, layer 4 only; B's layer-5/mode-0 gone
    _assert(int(reg.slot_mode[1]) == 1, "reused slot mode not updated to adjust_rs")
    _assert(int(reg.slot_vid[1]) == 1, "reused slot vid not updated")
    _assert(bool(reg.slot_layer_mask[1, 4]) and not bool(reg.slot_layer_mask[1, 5]),
            "reused slot layer mask carries stale B layers")


def test_adjust_rs_without_avgproj_skipped():
    """adjust_rs whose vector lacks avg_proj must contribute nothing (matches
    _build_routing_steer's `vid not in vec_has_avgproj -> continue`)."""
    ist._ACTIVE_WORKER_STEER = _Worker()
    reg = _fresh_registry()
    reg.vec_has_avgproj = set()          # neither vector has avg_proj now
    reg.refresh_slot_config(_mr([("A", {"steer": _steer("adjust_rs", 3, vec="vecAdj")})]))
    _assert(not bool(reg.slot_layer_mask[0].any()), "adjust_rs w/o avg_proj must be inert")


def main():
    tests = [
        test_resolution_matches_host_slabs,
        test_skip_on_unchanged_req_ids,
        test_slot_reuse_updates_table,
        test_adjust_rs_without_avgproj_skipped,
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
