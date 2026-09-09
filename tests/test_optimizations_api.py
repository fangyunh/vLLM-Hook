"""Unit tests for the public optimization surface (vllm_hook_plugins/optimizations.py).

No GPU, no engine — pure env/config mapping. Run: pytest tests/test_optimizations_api.py -vv

The load-bearing test here is ``test_advertised_defaults_match_the_code``: PUBLIC_LEVERS
documents a shipped default per lever, and a table that lies about defaults is worse than no
table (it is what sent the batched-egress lever two months in the wrong direction). If you flip
a default, that test fails until the table is updated too.
"""

import os

import pytest

from vllm_hook_plugins.optimizations import (
    PUBLIC_LEVERS,
    apply_optimizations,
    describe,
)

ALL_ENV = [spec[0] for spec in PUBLIC_LEVERS.values()]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts with all public levers unset -> the shipped defaults."""
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)


# --- mapping -------------------------------------------------------------------

def test_bools_map_to_1_and_0():
    apply_optimizations({"optimizations": {"batched_egress": False, "steer_fused": True}})
    assert os.environ["VLLM_HOOK_BATCHED_EGRESS"] == "0"
    assert os.environ["VLLM_HOOK_STEER_FUSED"] == "1"


@pytest.mark.parametrize("value", ["on", "true", "yes", "1", True])
def test_truthy_spellings(value):
    apply_optimizations({"optimizations": {"batched_egress": value}})
    assert os.environ["VLLM_HOOK_BATCHED_EGRESS"] == "1"


@pytest.mark.parametrize("value", ["off", "false", "no", "0", False])
def test_falsy_spellings(value):
    apply_optimizations({"optimizations": {"batched_egress": value}})
    assert os.environ["VLLM_HOOK_BATCHED_EGRESS"] == "0"


@pytest.mark.parametrize("value", ["auto", "default"])
def test_auto_defers_to_the_builtin_default(value):
    """compact_kall is TRI-state: unset != "0". 'auto' must leave the env unset."""
    apply_optimizations({"optimizations": {"compact_kall": value}})
    assert "VLLM_HOOK_QK_COMPACT_KALL" not in os.environ


# --- precedence ----------------------------------------------------------------

def test_env_beats_config(monkeypatch):
    """A run script's env always wins, so existing harnesses are unaffected by a config."""
    monkeypatch.setenv("VLLM_HOOK_BATCHED_EGRESS", "0")
    applied = apply_optimizations({"optimizations": {"batched_egress": True}})
    assert os.environ["VLLM_HOOK_BATCHED_EGRESS"] == "0"
    assert "batched_egress" not in applied


def test_config_applies_when_env_absent():
    applied = apply_optimizations({"optimizations": {"batched_egress": False}})
    assert applied == {"batched_egress": "0"}


# --- artifact_dtype (the one lossy lever) --------------------------------------

def test_artifact_dtype_passes_dtype_through():
    apply_optimizations({"optimizations": {"artifact_dtype": "int8"}})
    assert os.environ["VLLM_HOOK_ARTIFACT_DTYPE"] == "int8"


@pytest.mark.parametrize("value", ["off", "native", False])
def test_artifact_dtype_off_means_unset_not_zero(value):
    """The quant module treats ANY value as a dtype name; "0" would be a bogus dtype."""
    apply_optimizations({"optimizations": {"artifact_dtype": value}})
    assert "VLLM_HOOK_ARTIFACT_DTYPE" not in os.environ


# --- ring_max_batched_tokens (tri-state: auto | <int> | off) -------------------

def test_ring_mbt_auto_enables():
    apply_optimizations({"optimizations": {"ring_max_batched_tokens": "auto"}})
    assert os.environ["VLLM_HOOK_RING_MAX_BATCHED_TOKENS"] == "auto"


def test_ring_mbt_true_enables_auto():
    apply_optimizations({"optimizations": {"ring_max_batched_tokens": True}})
    assert os.environ["VLLM_HOOK_RING_MAX_BATCHED_TOKENS"] == "auto"


def test_ring_mbt_int_passes_through():
    apply_optimizations({"optimizations": {"ring_max_batched_tokens": 2048}})
    assert os.environ["VLLM_HOOK_RING_MAX_BATCHED_TOKENS"] == "2048"


@pytest.mark.parametrize("value", ["off", "false", "no", False])
def test_ring_mbt_off_means_unset(value):
    """Off/unset is the shipped default (opt-in); the env stays absent, not '0'."""
    apply_optimizations({"optimizations": {"ring_max_batched_tokens": value}})
    assert "VLLM_HOOK_RING_MAX_BATCHED_TOKENS" not in os.environ


# --- fail loud -----------------------------------------------------------------

def test_unknown_key_raises():
    with pytest.raises(ValueError, match="bached_egress"):
        apply_optimizations({"optimizations": {"bached_egress": True}})


def test_internal_knob_is_not_public():
    """Advanced/diagnostic knobs stay env-only; naming one in a config is an error, not a no-op."""
    with pytest.raises(ValueError):
        apply_optimizations({"optimizations": {"vec_egress": True}})


def test_non_object_block_raises():
    with pytest.raises(ValueError):
        apply_optimizations({"optimizations": ["batched_egress"]})


# --- back-compat ---------------------------------------------------------------

@pytest.mark.parametrize("cfg", [
    {},
    {"model_info": {"name": "x"}, "hidden_states": {"layers": [], "mode": "last_token"}},
    {"optimizations": {}},
])
def test_configs_without_the_block_are_noops(cfg):
    """Every v0.2.0-era config file must keep loading with no env side effects."""
    before = dict(os.environ)
    assert apply_optimizations(cfg) == {}
    assert dict(os.environ) == before


# --- the table must not lie ----------------------------------------------------

def test_advertised_defaults_match_the_code():
    """PUBLIC_LEVERS' documented default == what the package actually resolves when unset.

    Imports are inside the test: these modules read their env at import time, so they must be
    imported AFTER the fixture clears the environment.
    """
    from vllm_hook_plugins.graph import install, ops
    from vllm_hook_plugins.optimizations import env_is_on
    from vllm_hook_plugins.workers import probe_hookqk_worker as qkw

    resolved = {
        # Module-level constants: the real thing the graph path reads. batched_egress is a QK-only
        # lever now — the HS capture-ring path deleted its per-step egress (no gather to batch), so
        # only install._BATCHED_EGRESS (QK) remains.
        "batched_egress": "on" if install._BATCHED_EGRESS else "off",
        "steer_fused": "on" if ops._STEER_FUSED else "off",
        "compact_kall": "auto" if qkw._COMPACT_KALL_ENV is None else qkw._COMPACT_KALL_ENV,
        # Call-time knobs: resolve through env_is_on, the same helper the code calls, so this
        # cannot drift the way a hand-copied `os.environ.get(NAME, "1")` mirror would.
        "storage_router": "on" if env_is_on("storage_router") else "off",
        "writer_process": "on" if os.environ.get("VLLM_HOOK_WRITER_PROCESS", "1") == "1" else "off",
        "artifact_dtype": "native" if os.environ.get("VLLM_HOOK_ARTIFACT_DTYPE") is None else "set",
        # Opt-in (default off): unset -> OFF. Tri-state, so mirror presence, not a boolean.
        "ring_max_batched_tokens": "off" if os.environ.get("VLLM_HOOK_RING_MAX_BATCHED_TOKENS") is None else "set",
        # Opt-in since 2026-08-14. Hand mirror, like writer_process: the drain reads this inline in
        # __init__ (no module constant to import). FOUR sites read it identically -- shared sink +
        # per-request staging, in each of ring_drain_hs.py and ring_drain_qk.py. The default itself
        # is pinned at the code by test_mmap_default_off / test_perreq_mmap_default_is_off; this row
        # only keeps the advertised table honest.
        "ring_mmap": "on" if os.environ.get("VLLM_HOOK_RING_MMAP", "0") != "0" else "off",
    }
    for key, (_env, advertised, _doc) in PUBLIC_LEVERS.items():
        assert resolved[key] == advertised, (
            f"{key}: table advertises {advertised!r} but the code resolves {resolved[key]!r}. "
            "Flip the table or the default — do not let them disagree.")


def test_no_lever_is_lossy_or_behavior_changing_by_default():
    """Values must stay byte-identical to eager out of the box; only artifact_dtype is lossy."""
    assert PUBLIC_LEVERS["artifact_dtype"][1] == "native"


def test_describe_lists_every_lever():
    text = describe()
    for key in PUBLIC_LEVERS:
        assert key in text


# --- storage_router: the flip is only safe because it cannot touch offline ---------

def test_storage_router_defaults_on(monkeypatch):
    from vllm_hook_plugins.optimizations import env_is_on
    monkeypatch.delenv("VLLM_HOOK_STORAGE_ROUTER", raising=False)
    assert env_is_on("storage_router") is True


def test_storage_router_can_be_disabled(monkeypatch):
    from vllm_hook_plugins.optimizations import env_is_on
    monkeypatch.setenv("VLLM_HOOK_STORAGE_ROUTER", "0")
    assert env_is_on("storage_router") is False


def test_storage_router_is_serve_only():
    """The router must never fire offline: the OFFLINE LLM.generate patch must not call it.

    This is the whole basis for defaulting it ON -- if it ever gets wired into the offline
    path, that decision has to be revisited, so pin it.
    """
    import inspect
    from vllm_hook_plugins import _hook_plugin

    offline = inspect.getsource(_hook_plugin._patched_llm_generate)
    assert "_maybe_storage_route" not in offline, (
        "storage_router reached the OFFLINE path; it is defaulted ON only because it is "
        "serve-only and strictly inert offline.")

    serve = inspect.getsource(_hook_plugin._patched_generate)
    assert "_maybe_storage_route" in serve, "router vanished from the serve path"


# --- the router must never override an EXPLICIT storage choice ---------------------
# save_to_disk is not just a perf knob: True is how you require a durable artifact FILE.
# The router prices HS last_token as RPC, so an override would silently write nothing.

def _route_gate(extra: dict) -> bool:
    """Mirror of the guard in _patched_generate: does the router get to decide?"""
    return "save_to_disk" not in extra


def test_explicit_disk_is_never_overridden():
    """HS last_token + save_to_disk=True must still produce a file (the reported case)."""
    from vllm_hook_plugins.run_utils import predict_artifact_kb, route_to_disk
    kb = predict_artifact_kb("hs", "last_token", 1024, 32, 1, 96, 3072, 2, 300, "both")
    assert route_to_disk("hs", kb) is False, "router still prices HS last_token as RPC"
    # ... and precisely because it would say RPC, the gate must keep it out.
    assert _route_gate({"save_to_disk": True, "output_hidden_states": []}) is False


def test_explicit_rpc_is_never_overridden():
    assert _route_gate({"save_to_disk": False, "output_hidden_states": []}) is False


def test_router_decides_when_no_preference():
    assert _route_gate({"output_hidden_states": []}) is True


def test_serve_gate_is_wired_in_the_source():
    """Pin the guard itself: the router call must sit behind the 'no preference' check."""
    import inspect
    from vllm_hook_plugins import _hook_plugin
    src = inspect.getsource(_hook_plugin._patched_generate)
    assert '"save_to_disk" not in extra' in src, (
        "the storage router must only fire when the caller expressed no preference")
