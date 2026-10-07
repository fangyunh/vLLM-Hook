"""Engine-config policy and the compile-cache key stamp.

MIA requires vLLM's V2 runner and accepts vLLM 0.29's default FULL_AND_PIECEWISE; PIECEWISE alone
is refused. The stamp keeps engines that bake different capture ops or buffers apart in vLLM's
compile cache.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("vllm")  # `import mia` pulls in vLLM; skip, never error the whole collection

import mia.core._plugin as plugin
from mia.core._plugin import (
    UnsupportedGraphModeError,
    _mia_source_id,
    mia_graph_layout,
    stamp_compile_cache_key,
    validate_graph_mode,
)
from mia.errors import MiaConfigurationError


def test_piecewise_is_rejected_not_silently_downgraded():
    with pytest.raises(UnsupportedGraphModeError, match="PIECEWISE"):
        validate_graph_mode("PIECEWISE")


@pytest.mark.parametrize("mode", ["NONE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
def test_supported_modes_pass(mode):
    validate_graph_mode(mode)


def test_mia_never_forces_the_v1_runner():
    """MIA never sets VLLM_USE_V2_MODEL_RUNNER=0: it targets V2 only."""
    source = open(plugin.__file__).read()
    assert 'VLLM_USE_V2_MODEL_RUNNER"] = "0"' not in source
    assert '"VLLM_USE_V2_MODEL_RUNNER"] = "0"' not in source


def test_piecewise_alone_is_rejected_with_actionable_message():
    """The error names the supported modes and enforce_eager for the eager path."""
    with pytest.raises(UnsupportedGraphModeError) as excinfo:
        validate_graph_mode("PIECEWISE")
    message = str(excinfo.value)
    assert "FULL_AND_PIECEWISE" in message
    assert "enforce_eager" in message


# --- The worker kind goes into additional_config: vLLM's hash ignores worker_extension_cls. ---


class _FakeEngineArgs:
    """The engine-args fields the stamp reads: additional_config and the TP degree."""

    def __init__(self, additional_config=None, tensor_parallel_size=1):
        self.additional_config = additional_config
        self.tensor_parallel_size = tensor_parallel_size


def test_compile_cache_stamp_distinguishes_worker_kinds():
    stamps = {}
    for kind in ("hidden_states", "qk", "steer"):
        args = _FakeEngineArgs()
        stamp_compile_cache_key(args, kind)
        stamps[kind] = json.dumps(args.additional_config, sort_keys=True)

    assert len(set(stamps.values())) == 3, (
        f"every MIA worker kind must hash to a DIFFERENT compile-cache key, got {stamps}")


def test_compile_cache_stamp_is_json_serializable_and_idempotent():
    args = _FakeEngineArgs()
    stamp_compile_cache_key(args, "qk")
    once = json.dumps(args.additional_config, sort_keys=True)  # the hash factor's own encoding
    stamp_compile_cache_key(args, "qk")
    assert json.dumps(args.additional_config, sort_keys=True) == once


def test_compile_cache_stamp_preserves_a_callers_additional_config():
    original = {"user_key": {"a": 1}}
    args = _FakeEngineArgs(original)
    stamp_compile_cache_key(args, "steer")
    assert args.additional_config["user_key"] == {"a": 1}
    assert "mia_graph_capture" in args.additional_config
    # The caller may reuse its dict for another engine, so it must not gain MIA's key.
    assert original == {"user_key": {"a": 1}}, "the caller's dict was mutated in place"
    assert args.additional_config is not original


def test_compile_cache_stamp_keys_on_mia_source_not_only_worker_kind():
    """Editing MIA moves nothing vLLM hashes, so the stamp names MIA's source."""
    args = _FakeEngineArgs()
    stamp_compile_cache_key(args, "qk")
    stamp = args.additional_config["mia_graph_capture"]
    assert stamp["mia"], "the stamp carries no MIA source identifier"
    assert stamp["mia"] == _mia_source_id()
    # It must reach the hash factor's encoding.
    assert stamp["mia"] in json.dumps(args.additional_config, sort_keys=True)


def test_compile_cache_stamp_warns_rather_than_mutating_a_non_dict(capsys):
    """A caller's SupportsHash additional_config is left alone, with a printed warning."""
    sentinel = object()
    args = _FakeEngineArgs(sentinel)
    stamp_compile_cache_key(args, "qk")
    assert args.additional_config is sentinel
    assert "cannot stamp the compile-cache key" in capsys.readouterr().out


# --- The capture-buffer layout goes in too: MIA env sets shapes baked into the graph. ---

_HS_LAYOUT_ENV = ("MIA_HS_TP_SHARD", "MIA_HS_CAPTURE_ALL_RANKS", "MIA_HS_TP_SYMMETRIC",
                  "MIA_HS_CAPTURE", "MIA_APERTURE_GPU_BYTES", "MIA_STEER_VMAX",
                  "MIA_STEER_MODE", "MIA_QK_CAPTURE")


@pytest.fixture
def clean_layout_env(monkeypatch):
    """No MIA layout variable inherited from the caller's shell."""
    for name in _HS_LAYOUT_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _stamp(tp=1, kind="hidden_states"):
    """The JSON-encoded additional_config the stamp writes for this TP degree and worker kind."""
    args = _FakeEngineArgs(tensor_parallel_size=tp)
    stamp_compile_cache_key(args, kind)
    return json.dumps(args.additional_config, sort_keys=True)


def test_compile_cache_stamp_separates_the_hs_capture_layouts(clean_layout_env):
    """Four HS layouts at one model, TP degree and worker kind bake four buffer geometries."""
    layouts = {}
    layouts["round_robin"] = _stamp(tp=4)                           # 8 of 32 layers per rank
    clean_layout_env.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    layouts["all_ranks"] = _stamp(tp=4)                             # 32 of 32 on every rank
    clean_layout_env.delenv("MIA_HS_CAPTURE_ALL_RANKS")
    clean_layout_env.setenv("MIA_HS_TP_SHARD", "0")
    layouts["rank0"] = _stamp(tp=4)                                 # 32 on rank 0, 0 elsewhere
    clean_layout_env.delenv("MIA_HS_TP_SHARD")
    clean_layout_env.setenv("MIA_HS_TP_SYMMETRIC", "0")
    layouts["no_sinks"] = _stamp(tp=4)                              # unowned layers bake NO op

    assert len(set(layouts.values())) == 4, (
        "HS capture layouts that bake different buffers must not share a compile-cache key; "
        f"collided: {layouts}")


def test_compile_cache_stamp_moves_with_the_aperture_budget(clean_layout_env):
    """The aperture budget sets every capture buffer's row count, in both capture paths."""
    for kind in ("hidden_states", "qk"):
        auto = _stamp(tp=4, kind=kind)
        clean_layout_env.setenv("MIA_APERTURE_GPU_BYTES", str(4 * (1 << 30)))
        four_gib = _stamp(tp=4, kind=kind)
        clean_layout_env.setenv("MIA_APERTURE_GPU_BYTES", str(8 * (1 << 30)))
        eight_gib = _stamp(tp=4, kind=kind)
        clean_layout_env.delenv("MIA_APERTURE_GPU_BYTES")
        assert len({auto, four_gib, eight_gib}) == 3, (
            f"{kind}: apertures of different sizes give different R, so different buffers")


def test_compile_cache_stamp_moves_with_steer_v_max(clean_layout_env):
    """V_max is a dimension of the steer op's vector and avg_proj tables."""
    default = _stamp(kind="steer")
    clean_layout_env.setenv("MIA_STEER_VMAX", "64")
    assert _stamp(kind="steer") != default


def test_compile_cache_stamp_is_stable_for_an_unchanged_layout(clean_layout_env):
    """Settings that do not shape the graph leave the key alone (a moved key means a recompile)."""
    before = _stamp(tp=4)
    assert _stamp(tp=4) == before, "the same configuration stamped twice must agree"
    for name, value in (("MIA_APERTURE_DIR", "/tmp/somewhere-else"),
                        ("MIA_CAPTURE_FUSED", "0"),
                        ("MIA_STEER_FUSED", "0"),
                        ("MIA_APERTURE_BACKPRESSURE_TIMEOUT_S", "70"),
                        ("MIA_WRITER_PROCESS", "0"),
                        ("MIA_APERTURE_WRITE_MODE", "auto"),
                        ("MIA_ROUTE_VECTORIZED", "1")):
        clean_layout_env.setenv(name, value)
        assert _stamp(tp=4) == before, f"{name} does not shape the graph; it must not rekey it"
        clean_layout_env.delenv(name)


def test_graph_layout_refuses_a_near_miss_hs_flag(clean_layout_env):
    """A flag spelling MIA refuses is refused when the layout is stamped, too."""
    clean_layout_env.setenv("MIA_HS_CAPTURE_ALL_RANKS", "true")
    with pytest.raises(MiaConfigurationError, match="MIA_HS_CAPTURE_ALL_RANKS"):
        mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), "hidden_states")


def test_graph_layout_names_the_owned_layer_rule_and_the_sinks(clean_layout_env):
    """The layout names the sharding rule, the owned layers, the sinks and the aperture budget."""
    layout = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), "hidden_states")
    assert layout["hs_layout"] == "round_robin"
    assert layout["hs_layer_shard_rule"] == "round_robin"
    assert layout["hs_owned_layers"] == "layers i where i % 4 == tp_rank"
    assert layout["hs_sinks"] is True
    assert layout["aperture_gpu_bytes"] == "auto"
    assert layout["tp_size"] == 4
    # TP = 1 owns every layer and bakes no sink at all.
    single = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=1), "hidden_states")
    assert single["hs_layout"] == "single" and single["hs_sinks"] is False


def test_graph_layout_is_json_native(clean_layout_env):
    """vLLM hashes additional_config with json.dumps, so every layout value must be JSON."""
    for kind in ("hidden_states", "qk", "steer"):
        layout = mia_graph_layout(_FakeEngineArgs(tensor_parallel_size=4), kind)
        assert json.loads(json.dumps(layout, sort_keys=True)) == layout
