"""Storage-router decision verification (pure CPU, no GPU/engine).

Asserts VLLM_HOOK_STORAGE_ROUTER's per-request save_to_disk decision reproduces the
EMPIRICALLY-PROVEN fastest-path table (docs/storage_router_plan.md), across model configs
and a prompt-length sweep:

    (HS, last_token)  -> RPC  (save_to_disk=False)   # ~104 KB tiny; knee 12->32 on RPC
    (HS, all_tokens)  -> disk (save_to_disk=True)     # MB+; disk defers, RPC would block
    (QK, last_token)  -> disk (save_to_disk=True)     # MB+; RPC knee ~1 vs disk 8
    (QK, all_tokens)  -> disk (save_to_disk=True)

route_to_disk(worker, kb) returns True == save_to_disk=True == disk. The decision compares RPC's
on-loop BLOCKING ship against disk's on-loop HANDOFF (~20 ms), NOT the 200 ms async-saver
contention floor (which overlaps decode). Run standalone (no conftest engine machinery):

    python tests/test_storage_router_decisions.py
"""
from vllm_hook_plugins.run_utils import predict_artifact_kb, route_to_disk

# (name, hidden, head_dim, hs_layers, qk_head_layers)
MODELS = [
    ("granite-3.1-8b", 4096, 128, 13, 41),   # the cb-validated config
    ("qwen2-1.5b",     1536, 128, 28, 12),
]
# Realistic prompt lengths: the serverbench applies a chat template (~59 tok floor), so real
# requests are P >= ~64. Includes the old harm case (P=123). The tiny end (P<64) is exercised
# separately (test_tiny_artifacts_go_rpc) since sub-crossover artifacts correctly route to RPC.
REALISTIC = [64, 123, 256, 512, 1024, 2048]


def _kb(model, worker, gran, P):
    _n, hidden, head_dim, hs_L, qk_hl = model
    if worker == "hs":
        return predict_artifact_kb("hs", gran, P, hs_L, 1, head_dim, hidden, 2, 0, "prefill")
    return predict_artifact_kb("qk", gran, P, qk_hl, 1, head_dim, hidden, 2, 0, "prefill")


def test_predictor_exact_anchors():
    # HS last_token prefill-only == L*hidden*b, independent of prompt length
    kb = predict_artifact_kb("hs", "last_token", 999, 13, 1, 128, 4096, 2, 0, "prefill")
    assert abs(kb - 104.0) < 0.1, kb
    # QK last_token @P=2165, 13 head-layers (1 head/layer) ~= 7035.2 KB measured
    kb = predict_artifact_kb("qk", "last_token", 2165, 13, 1, 128, 4096, 2, 0, "prefill")
    assert abs(kb - 7035.2) / 7035.2 < 0.01, kb


def test_proven_optimum_table_at_realistic_sizes():
    # The empirically-proven fastest-path table holds for every realistic prompt + both models:
    #   HS-last -> RPC (tiny)  |  HS-all / QK-last / QK-all -> disk (large).
    want = {("hs", "last_token"): False, ("hs", "all_tokens"): True,
            ("qk", "last_token"): True, ("qk", "all_tokens"): True}
    for model in MODELS:
        for (worker, gran), expect_disk in want.items():
            for P in REALISTIC:
                kb = _kb(model, worker, gran, P)
                assert route_to_disk(worker, kb) is expect_disk, \
                    (model[0], worker, gran, P, round(kb, 1), "want disk" if expect_disk else "want RPC")


def test_regression_p123_qk_is_disk():
    # The exact request (granite QK-last, P=123, ~1.2 MB) that COLLAPSED the loop at rate 12 when
    # the old cost model mis-routed it to RPC. Must route to DISK now.
    kb = _kb(MODELS[0], "qk", "last_token", 123)
    assert kb > 1000.0 and route_to_disk("qk", kb) is True, kb


def test_size_monotonic_single_crossover():
    # The decision is a single size threshold per worker: below it -> RPC, above -> disk, no
    # flip-flop. Sweep kb and assert at most one RPC->disk transition, in the right direction.
    for worker in ("hs", "qk"):
        decisions = [route_to_disk(worker, kb) for kb in range(1, 20001, 5)]
        transitions = [(a, b) for a, b in zip(decisions, decisions[1:]) if a != b]
        assert len(transitions) <= 1, (worker, len(transitions))
        assert all(a is False and b is True for a, b in transitions), worker  # RPC(False)->disk(True)


def test_tiny_artifacts_go_rpc():
    # Below the crossover, ANY worker's artifact routes to RPC -- shipping a tiny blob on-loop
    # beats the disk handoff. This is the design intent (tiny -> RPC), not a bug: it is safe
    # (a ~50 KB QK block is ~13 ms, unlike the 1.2 MB harm case) and rarely hit post-template.
    assert route_to_disk("qk", 50.0) is False   # ~50 KB QK -> RPC
    assert route_to_disk("hs", 104.0) is False  # HS-last -> RPC


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fails += 1; print(f"FAIL  {t.__name__}: {e}")
        except Exception:  # noqa: BLE001
            fails += 1; print(f"ERROR {t.__name__}"); traceback.print_exc()
    # Print the decision table for the eye.
    print("\nDecision table (disk / RPC), granite-3.1-8b:")
    print(f"{'P':>6} | {'HS-last':>8} {'HS-all':>8} {'QK-last':>8} {'QK-all':>8}")
    for P in [8, 16, 32] + REALISTIC:
        row = []
        for w, g in (("hs", "last_token"), ("hs", "all_tokens"),
                     ("qk", "last_token"), ("qk", "all_tokens")):
            kb = _kb(MODELS[0], w, g, P)
            row.append("disk" if route_to_disk(w, kb) else "RPC")
        print(f"{P:>6} | {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)
