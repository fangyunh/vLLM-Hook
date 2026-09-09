"""No-GPU unit tests for the steering phase x positions gate.

These are the truth table every steer path must agree on: the eager hook, the graph
host router, and the graph GPU router all bottom out in steer_span(). A divergence
here is a graph-vs-eager parity failure that the LSF oracle would catch far later.
"""
import glob
import json
import os

import pytest
import torch

from vllm_hook_plugins.workers.steer_activation_worker import (
    STEER_COL_ALL,
    STEER_COL_NONE,
    is_default_steer_modes,
    resolve_steer_modes,
    steer_col_for,
    steer_span,
)


# ---- resolve_steer_modes ------------------------------------------------------

def test_defaults_reproduce_todays_behaviour():
    assert resolve_steer_modes({}) == ("both", "all_tokens")
    assert resolve_steer_modes(None) == ("both", "all_tokens")
    assert resolve_steer_modes({"method": "add_vector"}) == ("both", "all_tokens")


def test_explicit_values_are_honoured():
    cfg = {"phase": "prefill", "positions": "last_token"}
    assert resolve_steer_modes(cfg) == ("prefill", "last_token")


def test_apply_at_all_positions_false_maps_to_last_token():
    # Un-breaks the shipped korean/chinese language-steering configs.
    assert resolve_steer_modes({"apply_at_all_positions": False}) == ("both", "last_token")
    assert resolve_steer_modes({"apply_at_all_positions": True}) == ("both", "all_tokens")


def test_explicit_positions_wins_over_legacy_key():
    cfg = {"apply_at_all_positions": False, "positions": "all_tokens"}
    assert resolve_steer_modes(cfg) == ("both", "all_tokens")


@pytest.mark.parametrize("bad", ["prefil", "PREFILL", "none", True, 0])
def test_invalid_phase_raises(bad):
    with pytest.raises(ValueError, match="phase"):
        resolve_steer_modes({"phase": bad})


@pytest.mark.parametrize("bad", ["last_tokens", "lasttoken", "all", 1])
def test_invalid_positions_raises(bad):
    with pytest.raises(ValueError, match="positions"):
        resolve_steer_modes({"positions": bad})


def test_is_default_steer_modes():
    assert is_default_steer_modes("both", "all_tokens") is True
    assert is_default_steer_modes("prefill", "all_tokens") is False
    assert is_default_steer_modes("both", "last_token") is False


# ---- steer_span: the truth table ---------------------------------------------
# Span is [start, end) = [10, 15): a 5-token pass.
_S, _E = 10, 15


@pytest.mark.parametrize(
    "phase,positions,is_prefill,is_final,expected",
    [
        # all_tokens x both -> today's behaviour: whole span, always.
        ("both", "all_tokens", True, True, (_S, _E)),
        ("both", "all_tokens", True, False, (_S, _E)),
        ("both", "all_tokens", False, True, (_S, _E)),
        # phase gates.
        ("prefill", "all_tokens", True, True, (_S, _E)),
        ("prefill", "all_tokens", False, True, None),
        ("decode", "all_tokens", True, True, None),
        ("decode", "all_tokens", False, True, (_S, _E)),
        # last_token: final prefill chunk -> last row; non-final chunk -> nothing.
        ("both", "last_token", True, True, (_E - 1, _E)),
        ("both", "last_token", True, False, None),
        ("both", "last_token", False, True, (_E - 1, _E)),
        # the composition the user asked for: exactly one token, ever.
        ("prefill", "last_token", True, True, (_E - 1, _E)),
        ("prefill", "last_token", True, False, None),
        ("prefill", "last_token", False, True, None),
        # decode-only last_token == decode-only all_tokens for a 1-token pass (below).
        ("decode", "last_token", False, True, (_E - 1, _E)),
        ("decode", "last_token", True, True, None),
    ],
)
def test_steer_span_truth_table(phase, positions, is_prefill, is_final, expected):
    assert steer_span(phase, positions, is_prefill, is_final, _S, _E) == expected


def test_decode_pass_last_token_equals_all_tokens():
    """A decode pass has exactly 1 query token, so the two positions modes coincide."""
    for positions in ("all_tokens", "last_token"):
        assert steer_span("both", positions, False, True, 7, 8) == (7, 8)


def test_empty_span_is_never_steered():
    assert steer_span("both", "all_tokens", True, True, 4, 4) is None
    assert steer_span("both", "last_token", False, True, 9, 9) is None


# ---- steer_col_for: the graph routing sentinel --------------------------------

def test_steer_col_all_for_whole_span():
    assert steer_col_for("both", "all_tokens", True, True, _S, _E) == STEER_COL_ALL


def test_steer_col_none_when_gated_off():
    assert steer_col_for("prefill", "all_tokens", False, True, _S, _E) == STEER_COL_NONE
    assert steer_col_for("both", "last_token", True, False, _S, _E) == STEER_COL_NONE


def test_steer_col_is_the_absolute_last_column():
    assert steer_col_for("both", "last_token", True, True, _S, _E) == _E - 1


def test_single_token_span_all_tokens_is_still_col_all():
    """A 1-token decode pass under all_tokens must stay STEER_COL_ALL, not the column
    index -- otherwise the default path would start emitting a per-column sentinel and
    lose the 'no H2D needed' property."""
    assert steer_col_for("both", "all_tokens", False, True, 3, 4) == STEER_COL_ALL


# ---- SteerRegistry gate wiring (no GPU: device="cpu") -------------------------

from vllm_hook_plugins.graph.install_steer import SteerRegistry  # noqa: E402


class _FakeSamplingParams:
    def __init__(self, extra_args):
        self.extra_args = extra_args


class _FakeReqState:
    def __init__(self, steer_cfg, n_out=0, n_prompt=0):
        self.sampling_params = _FakeSamplingParams({"steer": steer_cfg} if steer_cfg else {})
        self.output_token_ids = [0] * n_out
        self.num_prompt = n_prompt


class _FakeInputBatch:
    def __init__(self, req_ids, num_computed, num_prompt):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = num_computed
        self.num_prompt_tokens = num_prompt


class _FakeRunner:
    def __init__(self, states, num_computed, num_prompt):
        self.requests = states
        self.input_batch = _FakeInputBatch(list(states.keys()), num_computed, num_prompt)


def _registry(num_layers=4, cap=32):
    return SteerRegistry(num_layers, cap, hidden=4, v_max=4, device="cpu",
                         dtype=torch.float32)


def _qsl(qlens):
    q = [0]
    for n in qlens:
        q.append(q[-1] + n)
    return q


def _runner(cfgs, qlens, n_outs, n_prompts, computed):
    states = {f"r{i}": _FakeReqState(c, n_outs[i], n_prompts[i])
              for i, c in enumerate(cfgs)}
    return _FakeRunner(states, computed, n_prompts), _qsl(qlens)


def test_default_config_never_latches_any_gated():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt"}
    runner, qsl = _runner([cfg], [5], [0], [5], [0])
    cols = reg._resolve_step_cols(runner, qsl)
    assert cols == [STEER_COL_ALL]
    assert reg._any_gated is False


def test_gated_config_latches_and_stays_latched():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    runner, qsl = _runner([cfg], [5], [0], [5], [0])
    assert reg._resolve_step_cols(runner, qsl) == [4]      # end-1 of [0,5)
    assert reg._any_gated is True
    # Latching: a later all-default step must NOT clear it.
    runner2, qsl2 = _runner([{"method": "add_vector", "vector_path": "x.pt"}],
                            [1], [1], [5], [5])
    reg._resolve_step_cols(runner2, qsl2)
    assert reg._any_gated is True


def test_non_final_prefill_chunk_steers_nothing():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    # prompt=10, this chunk covers [0,4) -> 0+4 < 10 -> not final.
    runner, qsl = _runner([cfg], [4], [0], [10], [0])
    assert reg._resolve_step_cols(runner, qsl) == [STEER_COL_NONE]
    # final chunk: computed=6, qlen=4 -> 10 >= 10 -> emit at column 3.
    runner2, qsl2 = _runner([cfg], [4], [0], [10], [6])
    assert reg._resolve_step_cols(runner2, qsl2) == [3]


def test_phase_prefill_is_inert_during_decode():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "phase": "prefill"}
    runner, qsl = _runner([cfg], [1], [3], [5], [5])   # n_out=3 -> decoding
    assert reg._resolve_step_cols(runner, qsl) == [STEER_COL_NONE]


def test_heterogeneous_batch_resolves_per_slot():
    reg = _registry()
    default = {"method": "add_vector", "vector_path": "x.pt"}
    gated = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    runner, qsl = _runner([default, gated, None], [3, 4, 2], [0, 0, 0],
                          [3, 4, 2], [0, 0, 0])
    # slot0 whole span; slot1 last col of [3,7) == 6; slot2 has no steer config.
    assert reg._resolve_step_cols(runner, qsl) == [STEER_COL_ALL, 6, STEER_COL_NONE]


def test_routing_key_is_unchanged_shape_when_not_gated():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt"}
    runner, qsl = _runner([cfg], [5], [0], [5], [0])
    key = reg.routing_key(runner, qsl)
    assert key == (("r0",), tuple(qsl))


def test_routing_key_distinguishes_identical_chunks_when_gated():
    """THE HAZARD: two consecutive equal-length prefill chunks give identical
    (req_ids, qsl). Without the gate term the wrapper would SKIP the final chunk and the
    request would steer nothing at all.

    Models the PRODUCTION sequence: `_any_gated` is latched by the BUILD (which resolves
    every config anyway), not by routing_key -- so step 1's key is the plain 2-tuple and
    step 2's is the extended 3-tuple. They differ, which is what forces the rebuild.
    Step 1 can never be skipped (`_last_route_key` starts None), so its build always runs
    and always latches before step 2's key is computed.
    """
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    # chunk 1 of 2: computed=0, qlen=8, prompt=16 -> not final.
    r1, q1 = _runner([cfg], [8], [0], [16], [0])
    k1 = reg.routing_key(r1, q1)
    assert reg._any_gated is False, "routing_key must not pay the O(bs) resolve pre-latch"
    reg._resolve_step_cols(r1, q1)          # the build; this is what latches
    assert reg._any_gated is True
    # chunk 2 of 2: computed=8, qlen=8, prompt=16 -> final. Same req_ids, same qsl.
    r2, q2 = _runner([cfg], [8], [0], [16], [8])
    k2 = reg.routing_key(r2, q2)
    assert q1 == q2 and r1.input_batch.req_ids == r2.input_batch.req_ids
    assert k1 != k2, "routing_key must distinguish the final chunk from a mid chunk"


def test_routing_key_distinguishes_chunks_once_latched():
    """The same hazard with the latch already set (steady gated deployment): both keys
    are 3-tuples and must still differ on the mid->final chunk transition."""
    reg = _registry()
    reg._any_gated = True
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    r1, q1 = _runner([cfg], [8], [0], [16], [0])    # mid chunk  -> STEER_COL_NONE
    r2, q2 = _runner([cfg], [8], [0], [16], [8])    # final chunk -> column 7
    k1, k2 = reg.routing_key(r1, q1), reg.routing_key(r2, q2)
    assert k1 == (("r0",), tuple(q1), (STEER_COL_NONE,))
    assert k2 == (("r0",), tuple(q2), (7,))
    assert k1 != k2


def test_routing_key_is_stable_through_steady_decode():
    """The gate term must NOT move every decode step, or the W1 skip never fires."""
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    reg._any_gated = True
    keys = []
    for step in range(3):
        runner, qsl = _runner([cfg], [1], [1 + step], [5], [5 + step])
        keys.append(reg.routing_key(runner, qsl))
    assert keys[0] == keys[1] == keys[2]


def test_routing_key_stashes_step_cols_for_the_build():
    reg = _registry()
    cfg = {"method": "add_vector", "vector_path": "x.pt", "positions": "last_token"}
    runner, qsl = _runner([cfg], [5], [0], [5], [0])
    reg._resolve_step_cols(runner, qsl)   # the build latches _any_gated
    reg.routing_key(runner, qsl)          # now routing_key takes the gated branch
    assert reg._step_cols == [4]


# ---- GPU scatter predicate (CPU via the torch backend) -----------------------

from vllm_hook_plugins.graph.steer_routing_gpu import scatter_routing  # noqa: E402


def _slabs(num_layers, cap):
    return (torch.zeros(num_layers, cap, dtype=torch.float32),
            torch.zeros(num_layers, cap, dtype=torch.int64),
            torch.zeros(num_layers, cap, dtype=torch.int64))


def _scatter(qsl, vids, modes, coeffs, masks, slot_col, num_layers=3, cap=16):
    c, v, m = _slabs(num_layers, cap)
    qsl_t = torch.tensor(qsl, dtype=torch.int64)
    sv = torch.tensor(vids, dtype=torch.int64)
    sm = torch.tensor(modes, dtype=torch.int64)
    sc = torch.tensor(coeffs, dtype=torch.float32)
    mk = torch.tensor(masks, dtype=torch.bool)
    col = None if slot_col is None else torch.tensor(slot_col, dtype=torch.int64)
    scatter_routing(qsl_t, sv, sm, sc, mk, c, v, m, qsl[-1], qsl[-1],
                    slot_col=col, backend="torch")
    return c, v, m


def test_slot_col_none_is_byte_identical_to_all_sentinel():
    args = ([0, 5], [2], [1], [3.0], [[True, False, True]])
    a = _scatter(*args, slot_col=None)
    b = _scatter(*args, slot_col=[STEER_COL_ALL])
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_slot_col_all_steers_the_whole_span():
    c, v, m = _scatter([0, 5], [2], [1], [3.0], [[True, False, True]],
                       slot_col=[STEER_COL_ALL])
    assert torch.all(c[0, 0:5] == 3.0) and torch.all(c[2, 0:5] == 3.0)
    assert torch.all(c[1, 0:5] == 0.0)          # layer not in the mask
    assert torch.all(v[0, 0:5] == 2) and torch.all(m[0, 0:5] == 1)


def test_slot_col_selects_exactly_one_column():
    c, v, m = _scatter([0, 5], [2], [1], [3.0], [[True, False, True]], slot_col=[4])
    assert c[0, 4] == 3.0 and v[0, 4] == 2 and m[0, 4] == 1
    assert torch.all(c[0, 0:4] == 0.0), f"non-target columns steered: {c[0, 0:5].tolist()}"
    assert torch.all(c[2, 0:4] == 0.0)


def test_slot_col_none_sentinel_steers_nothing():
    c, v, m = _scatter([0, 5], [2], [1], [3.0], [[True, True, True]],
                       slot_col=[STEER_COL_NONE])
    assert torch.all(c == 0.0) and torch.all(v == 0) and torch.all(m == 0)


def test_slot_col_is_per_slot_in_a_mixed_batch():
    """slot0 whole span [0,3), slot1 single column 5, slot2 inert."""
    c, v, m = _scatter([0, 3, 7, 9], [1, 2, 3], [0, 0, 0], [1.0, 2.0, 3.0],
                       [[True, False, False]] * 3,
                       slot_col=[STEER_COL_ALL, 5, STEER_COL_NONE])
    assert torch.all(c[0, 0:3] == 1.0)
    assert c[0, 5] == 2.0
    assert torch.all(c[0, 3:5] == 0.0) and torch.all(c[0, 6:9] == 0.0)


def test_padding_columns_are_never_steered_even_with_slot_col():
    c, v, m = _slabs(3, 16)
    qsl = torch.tensor([0, 4], dtype=torch.int64)
    scatter_routing(qsl, torch.tensor([1]), torch.tensor([0]), torch.tensor([9.0]),
                    torch.tensor([[True, False, False]]), c, v, m,
                    real_n=4, width=12, slot_col=torch.tensor([STEER_COL_ALL]),
                    backend="torch")
    assert torch.all(c[0, 0:4] == 9.0)
    assert torch.all(c[0, 4:12] == 0.0), "cudagraph padding columns were steered"


# ---- shipped configs ---------------------------------------------------------

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_shipped_steer_config_resolves():
    """A shipped config that raises at resolve time is a broken deployment. This is the
    regression guard for the korean/chinese configs, which set apply_at_all_positions
    false and used to hit NotImplementedError on the eager path."""
    paths = glob.glob(os.path.join(_REPO, "model_configs", "activation_steer", "*.json"))
    assert paths, "no activation_steer configs found"
    for p in paths:
        with open(p) as f:
            cfg = json.load(f).get("steering", {})
        phase, positions = resolve_steer_modes(cfg)   # must not raise
        assert phase in ("prefill", "decode", "both")
        assert positions in ("all_tokens", "last_token")


def test_language_configs_map_to_last_token():
    for name in ("Phi-3-mini-4k-instruct-korean", "Phi-3-mini-4k-instruct-chinese"):
        p = os.path.join(_REPO, "model_configs", "activation_steer", f"{name}.json")
        with open(p) as f:
            cfg = json.load(f)["steering"]
        assert resolve_steer_modes(cfg) == ("both", "last_token")


def test_hook_llm_load_config_rejects_a_bad_mode(tmp_path):
    """Offline fails LOUD at boot: a typo would otherwise steer every token while the
    caller believes they are steering one. (The serve path has no config file, so its
    workers warn-and-skip instead — see SteerHookActWorker._bad_cfg_warned.)"""
    from vllm_hook_plugins.hook_llm import HookLLM

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"steering": {"method": "add_vector",
                                             "positions": "last_token"}}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"steering": {"method": "add_vector",
                                            "positions": "last_tokens"}}))

    obj = HookLLM.__new__(HookLLM)          # bypass __init__ (it boots an engine)
    obj._steering_config = None
    obj._hookq_mode = "all_tokens"
    obj._qk_capture = "qk"
    obj._score_head = 0
    obj._hs_mode = "last_token"
    obj._output_layers = None
    HookLLM.load_config(obj, str(good))     # must not raise
    assert obj._steering_config["positions"] == "last_token"
    with pytest.raises(ValueError, match="positions"):
        HookLLM.load_config(obj, str(bad))


def test_new_parity_fixtures_have_the_intended_modes():
    expected = {
        "Qwen2-1.5B-Instruct_lastprefill": ("prefill", "last_token"),
        "Qwen2-1.5B-Instruct_lasttok": ("both", "last_token"),
        "Qwen2-1.5B-Instruct_decodeonly": ("decode", "all_tokens"),
        "Qwen2-1.5B-Instruct_prefillonly": ("prefill", "all_tokens"),
    }
    for name, want in expected.items():
        p = os.path.join(_REPO, "model_configs", "activation_steer", f"{name}.json")
        with open(p) as f:
            cfg = json.load(f)["steering"]
        assert resolve_steer_modes(cfg) == want, name
