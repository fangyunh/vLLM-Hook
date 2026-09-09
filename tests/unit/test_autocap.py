"""No-GPU unit tests for the auto-derived max_num_batched_tokens OOM fix (Change 2).

The formula + the min-only decision + the tri-state knob parse live as PURE functions in
graph/ring_sizing.py so they are testable with no vLLM import and no GPU. The thin _hook_plugin
glue (NVML total-mem read + post-build config mutation) is exercised by the GPU validation run.
"""
import pytest
from vllm_hook_plugins.graph.ring_sizing import (
    compute_safe_max_batched_tokens,
    apply_min_only,
    per_layer_token_bytes_hs,
    per_layer_token_bytes_qk,
    parse_autocap_setting,
)

GiB = 1 << 30
KiB = 1 << 10


# --- per-layer token bytes -----------------------------------------------------
def test_per_layer_token_bytes_hs():
    # HS = hidden_size * dtype_size (Llama-3.1-8B bf16 = 4096*2 = 8 KiB)
    assert per_layer_token_bytes_hs(4096, 2) == 8 * KiB


def test_per_layer_token_bytes_qk():
    # QK = (n_q_heads + n_kv_heads) * head_dim * dtype_size (buffers hold ALL heads)
    assert per_layer_token_bytes_qk(32, 8, 128, 2) == (32 + 8) * 128 * 2


# --- the formula (the plan's validated worked-check) ---------------------------
def test_hs_worked_check_yields_4096():
    # Llama-3.1-8B, HS all-32L bf16, 80 GiB, gpu_util 0.9, 4 GiB ring:
    # free_margin = 0.1*80 - 4 - 1 = 3 GiB; bytes/token = 32*8 KiB = 256 KiB;
    # safe_cap = 3 GiB / (256 KiB * 3) = 4096.
    plt = per_layer_token_bytes_hs(4096, 2)
    cap = compute_safe_max_batched_tokens(
        total_gpu_bytes=80 * GiB, gpu_mem_util=0.9, ring_gpu_bytes=4 * GiB,
        n_layers_captured=32, per_layer_token_bytes=plt, safety=3, headroom_bytes=1 * GiB)
    assert cap == 4096


def test_safety_scales_cap_inversely():
    plt = per_layer_token_bytes_hs(4096, 2)
    c3 = compute_safe_max_batched_tokens(80 * GiB, 0.9, 4 * GiB, 32, plt, safety=3, headroom_bytes=1 * GiB)
    c1 = compute_safe_max_batched_tokens(80 * GiB, 0.9, 4 * GiB, 32, plt, safety=1, headroom_bytes=1 * GiB)
    assert c3 == 4096
    assert c1 == 3 * c3   # safety=1 -> 12288


def test_no_margin_returns_none():
    # ring (8 GiB) + headroom (1 GiB) exceed the free margin (0.1*80 = 8 GiB) -> None.
    plt = per_layer_token_bytes_hs(4096, 2)
    assert compute_safe_max_batched_tokens(
        80 * GiB, 0.9, 8 * GiB, 32, plt, safety=3, headroom_bytes=1 * GiB) is None


def test_degenerate_bytes_per_token_returns_none():
    assert compute_safe_max_batched_tokens(80 * GiB, 0.9, 4 * GiB, 0, 8192, safety=3) is None


# --- min-only decision ---------------------------------------------------------
def test_min_only_no_op_when_safe_ge_current():
    # safe >= vLLM's resolved value -> leave it (byte-identical, the no-regression path).
    assert apply_min_only(current=2048, safe=4096) is None


def test_min_only_lowers_when_safe_lt_current():
    assert apply_min_only(current=8192, safe=4096) == 4096


def test_min_only_none_safe_is_no_op():
    assert apply_min_only(current=8192, safe=None) is None


def test_min_only_uses_safe_when_current_unresolved():
    assert apply_min_only(current=None, safe=4096) == 4096


# --- tri-state knob parse (VLLM_HOOK_RING_MAX_BATCHED_TOKENS) ------------------
@pytest.mark.parametrize("raw", [None, "", "0", "off", "false", "no"])
def test_autocap_off_by_default(raw):
    assert parse_autocap_setting(raw) == ("off", None)


@pytest.mark.parametrize("raw", ["auto", "on", "true", "yes", "1", "AUTO", " auto "])
def test_autocap_auto_spellings(raw):
    assert parse_autocap_setting(raw) == ("auto", None)


def test_autocap_explicit_int():
    assert parse_autocap_setting("2048") == ("explicit", 2048)


def test_autocap_garbage_is_off():
    assert parse_autocap_setting("banana") == ("off", None)


# --- _hook_plugin glue (min-only gating; the auto path needs NVML/GPU -> validation run) ------
class _SchedCfg:
    def __init__(self, mnbt):
        self.max_num_batched_tokens = mnbt


class _FakeConfig:
    def __init__(self, mnbt):
        self.scheduler_config = _SchedCfg(mnbt)


@pytest.fixture(autouse=True)
def _clear_autocap_env(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_RING_MAX_BATCHED_TOKENS", raising=False)
    monkeypatch.delenv("VLLM_HOOK_WORKER", raising=False)


def test_glue_explicit_lowers(monkeypatch):
    from vllm_hook_plugins import _hook_plugin
    monkeypatch.setenv("VLLM_HOOK_RING_MAX_BATCHED_TOKENS", "2048")
    cfg = _FakeConfig(8192)
    _hook_plugin._maybe_autocap_max_batched_tokens(cfg, "hidden_states")
    assert cfg.scheduler_config.max_num_batched_tokens == 2048


def test_glue_explicit_no_op_when_ge(monkeypatch):
    from vllm_hook_plugins import _hook_plugin
    monkeypatch.setenv("VLLM_HOOK_RING_MAX_BATCHED_TOKENS", "8192")
    cfg = _FakeConfig(2048)
    _hook_plugin._maybe_autocap_max_batched_tokens(cfg, "qk")
    assert cfg.scheduler_config.max_num_batched_tokens == 2048   # min-only: leave the lower value


def test_glue_off_is_no_op(monkeypatch):
    from vllm_hook_plugins import _hook_plugin
    cfg = _FakeConfig(8192)
    _hook_plugin._maybe_autocap_max_batched_tokens(cfg, "hidden_states")
    assert cfg.scheduler_config.max_num_batched_tokens == 8192


def test_glue_steer_is_no_op(monkeypatch):
    from vllm_hook_plugins import _hook_plugin
    monkeypatch.setenv("VLLM_HOOK_RING_MAX_BATCHED_TOKENS", "2048")
    cfg = _FakeConfig(8192)
    _hook_plugin._maybe_autocap_max_batched_tokens(cfg, "steer")
    assert cfg.scheduler_config.max_num_batched_tokens == 8192   # no capture ring -> untouched


def test_glue_worker_kind_resolution():
    from vllm_hook_plugins import _hook_plugin
    assert _hook_plugin._worker_kind(_hook_plugin._WORKER_EXT_QK) == "qk"
    assert _hook_plugin._worker_kind(_hook_plugin._WORKER_EXT_HS) == "hidden_states"
    assert _hook_plugin._worker_kind(_hook_plugin._WORKER_EXT_STEER) == "steer"


def test_glue_worker_kind_env_wins_over_wrapper(monkeypatch):
    """The VHP profiler (and vllm serve) replace worker_extension_cls with a wrapper/leave it unset
    and select the real worker via VLLM_HOOK_WORKER; the env must win so autocap sizes the right
    worker. Regression: this returned 'hidden_states' for a QK run wrapped by VHPProbeWorker."""
    from vllm_hook_plugins import _hook_plugin
    wrapper = "vllm_hook_profiling.probe_worker.VHPProbeWorker"
    monkeypatch.setenv("VLLM_HOOK_WORKER", "qk")
    assert _hook_plugin._worker_kind(wrapper) == "qk"
    monkeypatch.setenv("VLLM_HOOK_WORKER", "steer")
    assert _hook_plugin._worker_kind(wrapper) == "steer"
    monkeypatch.setenv("VLLM_HOOK_WORKER", "hidden_states")
    assert _hook_plugin._worker_kind(wrapper) == "hidden_states"
