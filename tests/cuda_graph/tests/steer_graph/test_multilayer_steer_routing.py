"""No-GPU unit test for SteerRegistry.apply_incremental_routing multi-layer support.

The buffer-mode steer routing writes a per-(layer, token) coeff/vec_id/mode plane that
the baked steer_buffer op reads. The incremental router (VLLM_HOOK_INCREMENTAL_ROUTING=1,
the default) diffs the desired plane vs a shadow and uploads only changed cells.

The bug this guards: the shadow was 1-D (one interned state per token COLUMN), so a
request steering N layers — N assignments over the SAME columns — collapsed to the LAST
layer only (the earlier layers were silently dropped, so graph-vs-eager diverged even for
linear add_vector). The shadow must be 2-D (num_layers, cap) so every (layer, column) is
tracked independently.

Runs on CPU (no cudagraph): apply_incremental_routing writes plain CPU tensors when the
registry device is 'cpu'. Call it directly with hand-built assignments and assert the
device slabs (coeff_all/vec_id_all/mode_all) hold the right per-layer values.

    conda activate vllm_hook_env && python tests/cuda_graph/tests/steer_graph/test_multilayer_steer_routing.py
"""
import torch

from vllm_hook_plugins.graph.install_steer import SteerRegistry


def _fresh(num_layers=8, cap=16, hidden=4, v_max=4):
    return SteerRegistry(num_layers, cap, hidden, v_max, device="cpu", dtype=torch.float32)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_multilayer_one_request_writes_all_target_layers():
    """One request steering layers [1,3,5] over columns [0,5) must write ALL three rows."""
    reg = _fresh()
    coeff, vid, mode = 2.5, 1, 0  # add_vector-style
    assignments = [(0, 5, L, coeff, vid, mode) for L in (1, 3, 5)]
    uploaded = reg.apply_incremental_routing(assignments, width=8)
    _assert(uploaded is True, "first non-empty apply must upload")

    for L in (1, 3, 5):
        _assert(torch.all(reg.coeff_all[L, 0:5] == coeff),
                f"layer {L} coeff not written (multi-layer collapse bug): "
                f"{reg.coeff_all[L, 0:5].tolist()}")
        _assert(torch.all(reg.vec_id_all[L, 0:5] == vid), f"layer {L} vec_id not written")
        _assert(torch.all(reg.mode_all[L, 0:5] == mode), f"layer {L} mode not written")
    # Every other layer stays a clean zero (exact no-op for the op).
    for L in range(reg.num_layers):
        if L in (1, 3, 5):
            continue
        _assert(torch.all(reg.coeff_all[L] == 0), f"layer {L} unexpectedly nonzero")
    # Columns past the request span stay zero on the target rows.
    for L in (1, 3, 5):
        _assert(torch.all(reg.coeff_all[L, 5:] == 0), f"layer {L} tail not zero")


def test_idempotent_second_call_no_upload():
    """A stable batch (identical assignments) must skip the upload — the CB replay path."""
    reg = _fresh()
    assignments = [(0, 5, L, 2.5, 1, 0) for L in (1, 3, 5)]
    reg.apply_incremental_routing(assignments, width=8)
    uploaded = reg.apply_incremental_routing(assignments, width=8)
    _assert(uploaded is False, "identical routing must be a no-upload replay")
    for L in (1, 3, 5):
        _assert(torch.all(reg.coeff_all[L, 0:5] == 2.5), f"layer {L} value lost on replay")


def test_deactivation_zeroes_vacated_layers():
    """When the request finishes (no assignments), all previously-steered rows must clear."""
    reg = _fresh()
    assignments = [(0, 5, L, 2.5, 1, 0) for L in (1, 3, 5)]
    reg.apply_incremental_routing(assignments, width=8)
    uploaded = reg.apply_incremental_routing([], width=8)
    _assert(uploaded is True, "deactivation must upload the zeroing")
    for L in (1, 3, 5):
        _assert(torch.all(reg.coeff_all[L] == 0),
                f"layer {L} stale steer leaked after finish: {reg.coeff_all[L].tolist()}")


def test_single_layer_unchanged_regression():
    """Single-layer steer (the validated common case) must be byte-identical: exactly one
    row written, all others zero — the 2-D shadow must reduce to the 1-D behaviour."""
    reg = _fresh()
    uploaded = reg.apply_incremental_routing([(0, 4, 3, 1.0, 2, 1)], width=6)  # adjust_rs-style
    _assert(uploaded is True, "single-layer apply must upload")
    _assert(torch.all(reg.coeff_all[3, 0:4] == 1.0), "single-layer coeff wrong")
    _assert(torch.all(reg.vec_id_all[3, 0:4] == 2), "single-layer vec_id wrong")
    _assert(torch.all(reg.mode_all[3, 0:4] == 1), "single-layer mode wrong")
    for L in range(reg.num_layers):
        if L == 3:
            continue
        _assert(torch.all(reg.coeff_all[L] == 0), f"layer {L} nonzero for single-layer steer")


def test_two_requests_disjoint_columns_multi_and_single_layer():
    """Req A (cols 0:3, layers [1,2]) + Req B (cols 3:6, layer [4]) — disjoint columns, one
    multi-layer + one single-layer, all must land in the right (layer, column) cells."""
    reg = _fresh()
    assignments = [
        (0, 3, 1, 2.0, 1, 0), (0, 3, 2, 2.0, 1, 0),   # req A -> layers 1,2
        (3, 6, 4, 3.0, 1, 0),                          # req B -> layer 4
    ]
    reg.apply_incremental_routing(assignments, width=8)
    _assert(torch.all(reg.coeff_all[1, 0:3] == 2.0), "reqA layer1 wrong")
    _assert(torch.all(reg.coeff_all[2, 0:3] == 2.0), "reqA layer2 wrong")
    _assert(torch.all(reg.coeff_all[4, 3:6] == 3.0), "reqB layer4 wrong")
    # No cross-contamination: reqA cols must be zero on layer 4, reqB cols zero on 1,2.
    _assert(torch.all(reg.coeff_all[4, 0:3] == 0), "reqB layer leaked into reqA cols")
    _assert(torch.all(reg.coeff_all[1, 3:6] == 0), "reqA layer leaked into reqB cols")
    _assert(torch.all(reg.coeff_all[2, 3:6] == 0), "reqA layer leaked into reqB cols")


def test_last_token_writes_only_the_final_column():
    """positions=last_token must narrow the assignment to a single column, and the
    shadow must zero the columns a previous all_tokens step had written."""
    reg = _fresh()
    # Step 1: all_tokens over [0,5) at layer 2.
    reg.apply_incremental_routing([(0, 5, 2, 1.0, 0, 0)], width=8)
    _assert(torch.all(reg.coeff_all[2, 0:5] == 1.0), "setup: all_tokens not written")
    # Step 2: same request now narrowed to its last column only.
    reg.apply_incremental_routing([(4, 5, 2, 1.0, 0, 0)], width=8)
    _assert(reg.coeff_all[2, 4] == 1.0, "last column not written")
    _assert(torch.all(reg.coeff_all[2, 0:4] == 0.0),
            f"vacated columns not zeroed: {reg.coeff_all[2, 0:4].tolist()}")


def test_gated_off_step_zeroes_every_steer_column():
    """A request that steers nothing this step (non-final chunk / phase gate) must leave
    a fully zeroed plane — no stale steering may replay."""
    reg = _fresh()
    reg.apply_incremental_routing([(0, 5, 3, 2.0, 1, 1)], width=8)
    _assert(torch.any(reg.coeff_all[3, 0:5] != 0.0), "setup: nothing written")
    reg.apply_incremental_routing([], width=8)      # gated off -> no assignments
    _assert(torch.all(reg.coeff_all[3, 0:5] == 0.0), "stale steer survived")
    _assert(torch.all(reg.vec_id_all[3, 0:5] == 0), "stale vec_id survived")
    _assert(torch.all(reg.mode_all[3, 0:5] == 0), "stale mode survived")


def main():
    tests = [
        test_multilayer_one_request_writes_all_target_layers,
        test_idempotent_second_call_no_upload,
        test_deactivation_zeroes_vacated_layers,
        test_single_layer_unchanged_regression,
        test_two_requests_disjoint_columns_multi_and_single_layer,
        test_last_token_writes_only_the_final_column,
        test_gated_off_step_zeroes_every_steer_column,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print("=" * 60)
    if failures:
        print(f"VERDICT: FAIL ({failures}/{len(tests)} failed)")
        return 1
    print(f"VERDICT: PASS ({len(tests)}/{len(tests)})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
