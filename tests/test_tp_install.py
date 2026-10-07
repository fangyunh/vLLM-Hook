"""What each TP rank installs, and what MIA refuses, at tensor_parallel_size > 1.

The real install functions run on CPU against a fake V2 worker: HS rank gating and its symmetric
sink op, Q/K on every rank with its own shard, pipeline parallelism refused, and the aperture sized
from the model.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")  # `import mia` pulls in vLLM; skip, never error the whole collection

import mia.core._plugin as plugin
import mia.core.delivery.writer_process as wp
from mia.core._plugin import _derive_safe_max_batched_tokens, _model_dims
from mia.errors import MiaConfigurationError, MiaRefusal, MiaSizingError
from mia.core.aperture.aperture_metadata import ReqCaptureRecord
from mia.core.aperture.aperture_reader import read_sidecar_header
from mia.core.aperture.aperture_sizing import (
    DEFAULT_APERTURE_GPU_BYTES,
    DEFAULT_AUTOCAP_HEADROOM_BYTES,
    DEFAULT_AUTOCAP_SAFETY,
    compute_safe_max_batched_tokens,
    per_layer_token_bytes_qk,
)
from mia.core.hooks.install import (
    _resolve_qk_aperture_rows,
    install_execute_model_wrapper,
    install_qk_hosts,
)
from mia.core.hooks.install_hs import (
    _resolve_aperture_rows,
    install_execute_model_wrapper_hs,
    install_hs_hosts,
)
from mia.core.hooks.install_steer import install_steer_hosts
from mia.core.hooks.ops import _capture_hs_impl
from mia.core.hooks.registry import get_registry
from mia.core.runtime.tp_shard import qk_shard, resolve_tp_coords
from mia.workers.hs_capture_worker import HSCaptureWorker

GIB = 1 << 30
HIDDEN, LAYERS, H_Q, H_KV, CAP = 16, 3, 4, 2, 16


class _Runner:
    """V2-shaped fake runner (module name is what require_v2_runner checks)."""

    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self, model):
        self.model = model
        self.block_tables = types.SimpleNamespace(
            input_block_tables=(torch.zeros((4, 3), dtype=torch.int32),))

    def add_requests(self, so):
        pass

    def finish_requests(self, so):
        pass

    def prepare_inputs(self, *a, **k):
        return None

    def execute_model(self, *a, **k):
        return "executed"


def _model(tp, *, attn_heads=None):
    """A model with fresh classes (MIA wraps classes), named like vLLM's Llama for the matchers."""
    local_q = H_Q // tp
    local_kv = max(1, H_KV // tp)
    q_heads = local_q if attn_heads is None else attn_heads

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads, self.num_kv_heads, self.head_size = q_heads, local_kv, HIDDEN // H_Q

        def forward(self, q, k, v):
            return q

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.attn = Attn()

        def forward(self, h, r):
            return h, r

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                                          num_attention_heads=H_Q, num_key_value_heads=H_KV)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(LAYERS)])
            self.w = nn.Parameter(torch.zeros(1))

    return Model()


def _worker(tp, rank, pp=1, model=None):
    model = model if model is not None else _model(tp)
    return SimpleNamespace(
        model_runner=_Runner(model),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        rank=rank,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=CAP)),
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A tmp aperture dir, a small aperture, the synchronous drain and a counting writer stub."""
    monkeypatch.setenv("MIA_APERTURE_DIR", str(tmp_path / "aperture"))
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(1 << 20))
    monkeypatch.setenv("MIA_APERTURE_SYNC_DRAIN", "1")
    monkeypatch.delenv("MIA_HS_CAPTURE_ALL_RANKS", raising=False)
    monkeypatch.delenv("MIA_HS_TP_SYMMETRIC", raising=False)
    monkeypatch.delenv("MIA_HS_TP_SHARD", raising=False)
    started = []
    monkeypatch.setattr(wp.WriterProcess, "from_env",
                        classmethod(lambda cls: started.append(1) or object()))
    return SimpleNamespace(aperture=tmp_path / "aperture", writers=started)


def _install_hs(worker):
    install_hs_hosts(worker)
    install_execute_model_wrapper_hs(worker.model_runner, worker)


def _install_qk(worker):
    install_qk_hosts(worker)
    install_execute_model_wrapper(worker.model_runner, worker)


def _hosts(worker, attr):
    return {n: getattr(m, attr) for n, m in worker.model_runner.model.named_modules()
            if getattr(m, attr, None) is not None}


# --- HS ---

def test_hs_non_capture_rank_allocates_and_starts_nothing(env, monkeypatch):
    monkeypatch.setenv("MIA_HS_TP_SHARD", "0")                # the rank-0-only A/B layout
    w = _worker(tp=2, rank=1)
    _install_hs(w)
    assert w._should_capture is False
    assert get_registry(w, "hs") is None                     # no routing slabs / mirrors
    assert getattr(w, "_capture_aperture", None) is None     # no aperture
    assert getattr(w, "_hs_drain", None) is None             # no drain (thread or sync)
    assert w._writer_process is None and env.writers == []   # no writer process started
    assert w._writer_mode == "none (HS sink rank: captures nothing)"   # ...and says so
    assert not (env.aperture / "tp_rank_1").exists()         # no empty rank dir
    assert not env.aperture.exists() or not any(env.aperture.iterdir())
    assert HSCaptureWorker.flush_aperture(w) is None


def test_hs_non_capture_rank_still_bakes_the_op_symmetrically(env, monkeypatch):
    """A sink rank bakes rank 0's graph; only the buffer's row count differs (1 vs R+1)."""
    monkeypatch.setenv("MIA_HS_TP_SHARD", "0")
    w0, w1 = _worker(tp=2, rank=0), _worker(tp=2, rank=1)
    _install_hs(w0)
    _install_hs(w1)
    h0, h1 = _hosts(w0, "_mia_hs_host"), _hosts(w1, "_mia_hs_host")
    assert set(h0) == set(h1) == {f"model.layers.{i}" for i in range(LAYERS)}
    for name in h0:
        a, b = h0[name], h1[name]
        assert a.do_capture and b.do_capture
        assert (a.layer_num, a.egress_layer_num, a.has_residual) == \
            (b.layer_num, b.egress_layer_num, b.has_residual)
        assert a.capture_index.shape == b.capture_index.shape == (CAP,)
        assert a.hs_buf.shape[1] == b.hs_buf.shape[1] == HIDDEN
        assert a.hs_buf.shape[0] > CAP and b.hs_buf.shape == (1, HIDDEN)
        assert int(b.capture_index.abs().sum()) == 0
    # One sink buffer per layer, like rank 0's per-layer apertures.
    assert len({h.hs_buf.data_ptr() for h in h1.values()}) == LAYERS
    # A replay of the sink op lands every token on the sink row, in bounds.
    b = h1["model.layers.0"]
    x = torch.randn(CAP, HIDDEN)
    _capture_hs_impl(x, x, b.hs_buf, b.capture_index, 1)
    assert b.hs_buf.shape == (1, HIDDEN)
    assert any(torch.equal(b.hs_buf[0], row) for row in (x + x))


def test_hs_capture_rank_gets_the_aperture_a_header_and_a_rank0_dir(env):
    w = _worker(tp=2, rank=0)
    _install_hs(w)
    assert w._should_capture is True and get_registry(w, "hs") is not None
    assert w._capture_aperture is not None and w._capture_aperture.n_slots >= 1
    assert env.writers == [1]
    run_dir = env.aperture / "tp_rank_0"
    assert run_dir.is_dir()
    # Nothing captured yet, so no dir to return.
    assert HSCaptureWorker.flush_aperture(w) is None
    hdr = read_sidecar_header(str(run_dir / "hs_aperture_meta.jsonl"))
    assert (hdr["tp_rank"], hdr["tp_size"], hdr["num_layers"], hdr["capture_all_ranks"]) \
        == (0, 2, LAYERS, False)
    assert hdr["row_shape"] == [HIDDEN]


def test_hs_flush_returns_the_dir_once_it_holds_rows(env):
    w = _worker(tp=1, rank=0)
    _install_hs(w)
    drain, ap = w._hs_drain, w._capture_aperture
    start = ap.reserve(2)
    drain.record_entries([ReqCaptureRecord(req_id="r0", logical_start=start, n_rows=2,
                                           hs_mode="all_tokens", layers=[1, 2, 3])])
    drain.drain_once()
    assert HSCaptureWorker.flush_aperture(w) == str(env.aperture / "tp_rank_0")


def test_hs_capture_all_ranks_diagnostic_still_captures_everywhere(env, monkeypatch):
    monkeypatch.setenv("MIA_HS_CAPTURE_ALL_RANKS", "1")
    w = _worker(tp=2, rank=1)
    _install_hs(w)
    assert w._should_capture and w._capture_aperture is not None
    assert (env.aperture / "tp_rank_1").is_dir()


def test_hs_eager_non_capture_rank_starts_no_writer(env, monkeypatch):
    """The eager path's install_hooks: rank 1 starts no writer process either."""
    monkeypatch.setattr("mia.workers.hs_capture_worker.require_v2_runner", lambda r: None)
    w = _worker(tp=2, rank=1)
    w._hooks_installed = False
    w.parallel_config = SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1)
    HSCaptureWorker.install_hooks(w)
    assert w._should_capture is False and w._writer_process is None and env.writers == []
    assert w._writer_mode == "none (HS sink rank: captures nothing)"
    w0 = _worker(tp=2, rank=0)
    w0._hooks_installed = False
    HSCaptureWorker.install_hooks(w0)
    assert w0._should_capture is True and env.writers == [1]


# --- QK ---

@pytest.mark.parametrize("rank", [0, 1])
def test_qk_every_rank_captures_its_own_shard(env, rank):
    """Each rank gets hosts sized to its shard, its own tp_rank dir and its header geometry."""
    w = _worker(tp=2, rank=rank)
    _install_qk(w)
    assert w._should_capture is True and get_registry(w, "qk") is not None
    head_dim = HIDDEN // H_Q
    hosts = _hosts(w, "_mia_qk_host")
    assert len(hosts) == LAYERS
    for h in hosts.values():
        assert h.q_buf.shape[1] == (H_Q // 2) * head_dim
        assert h.k_buf.shape[1] == (H_KV // 2) * head_dim
    run_dir = env.aperture / f"tp_rank_{rank}"
    assert w._qk_run_dir == str(run_dir) and run_dir.is_dir()
    w._qk_drain.close()
    hdr = read_sidecar_header(str(run_dir / "qk_aperture_meta.jsonl"))
    assert hdr["tp_rank"] == rank and hdr["tp_size"] == 2
    assert hdr["q_head_start"] == rank * (H_Q // 2)
    assert hdr["kv_head_start"] == rank * (H_KV // 2)
    assert (hdr["num_attention_heads"], hdr["num_key_value_heads"], hdr["head_dim"]) \
        == (H_Q, H_KV, head_dim)
    assert hdr["q_row_shape"] == [hdr["num_local_q_heads"] * head_dim]
    assert w._conf["num_attention_heads"] == H_Q          # _conf = the MERGED layout


def test_qk_refuses_a_model_whose_attention_modules_shard_differently(env):
    w = _worker(tp=2, rank=0, model=_model(2, attn_heads=3))
    with pytest.raises(MiaConfigurationError, match="shard geometry mismatch"):
        install_qk_hosts(w)


def test_tp_coords_fall_back_to_rank_mod_tp():
    assert resolve_tp_coords(_worker(tp=4, rank=6)) == (2, 4)
    assert resolve_tp_coords(SimpleNamespace(rank=0)) == (0, 1)


# --- Pipeline parallelism: refused loudly ---

def _drive_seam(monkeypatch, engine_pp=1, resolved_pp=1):
    """Run the engine-config seam on fake args; return (config, calls to vLLM's original)."""
    called = []
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(pipeline_parallel_size=resolved_pp,
                                        tensor_parallel_size=1),
        use_v2_model_runner=True,
    )
    monkeypatch.setattr(plugin, "_original_create_engine_config",
                        lambda self, *a, **k: called.append(1) or config, raising=False)
    monkeypatch.delenv("MIA_WORKER", raising=False)
    monkeypatch.delenv("MIA_ALLOW_CUDAGRAPH", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    args = SimpleNamespace(worker_extension_cls=None, enforce_eager=False,
                           compilation_config=None, pipeline_parallel_size=engine_pp)
    return plugin._patched_create_engine_config(args), called


def test_pp_is_refused_at_engine_construction_before_any_model_is_built(monkeypatch):
    with pytest.raises(MiaConfigurationError, match="pipeline parallelism"):
        _drive_seam(monkeypatch, engine_pp=2)


def test_pp_in_the_resolved_config_is_refused_too(monkeypatch):
    with pytest.raises(MiaConfigurationError, match="pipeline_parallel_size=4"):
        _drive_seam(monkeypatch, engine_pp=1, resolved_pp=4)


def test_pp1_passes_the_seam(monkeypatch):
    config, called = _drive_seam(monkeypatch)
    assert called == [1] and config.parallel_config.pipeline_parallel_size == 1


@pytest.mark.parametrize("which", ["hs", "qk", "steer"])
def test_pp_is_refused_at_install_and_is_a_refusal(env, which):
    """Refused at install too, as a MiaRefusal so load_model re-raises it."""
    w = _worker(tp=1, rank=0, pp=2)
    fn = {"hs": install_hs_hosts, "qk": install_qk_hosts, "steer": install_steer_hosts}[which]
    with pytest.raises(MiaConfigurationError, match="pipeline") as e:
        fn(w)
    assert isinstance(e.value, MiaRefusal)


# --- Aperture sizing at 8B and 70B shapes ---

H100 = 80 * GIB
ROW_70B_HS = 80 * 8192 * 2                   # one token, every layer, bf16: 1.25 MiB
ROW_8B_HS = 32 * 4096 * 2


def _rows(monkeypatch, util, num_layers, width, rows_needed, total=H100):
    """Drive install_hs._resolve_aperture_rows with a fake CUDA device of ``total`` bytes."""
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda d: SimpleNamespace(total_memory=total))
    w = SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=util)))
    return _resolve_aperture_rows(w, num_layers, width, torch.bfloat16, "cuda:0",
                                  rows_needed=rows_needed)


def test_the_old_fixed_default_could_not_hold_one_70b_step(monkeypatch):
    """4 GiB holds 3276 of a 70B step's rows: a fixed default could not admit an 8192-token step."""
    assert DEFAULT_APERTURE_GPU_BYTES // ROW_70B_HS == 3276 < 8192


def test_70b_all_layer_hs_default_now_holds_a_full_step(monkeypatch):
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    R, nbytes = _rows(monkeypatch, 0.85, 80, 8192, 8192)
    assert R >= 8192 and nbytes == 8192 * ROW_70B_HS == 10 * GIB


def test_70b_default_that_cannot_fit_refuses_at_install_naming_the_knob(monkeypatch):
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    with pytest.raises(MiaSizingError, match="MIA_APERTURE_GPU_BYTES") as e:
        _rows(monkeypatch, 0.90, 80, 8192, 8192)
    assert "10.00 GiB" in str(e.value)


def test_an_explicit_aperture_always_wins(monkeypatch):
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(4 * GIB))
    R, nbytes = _rows(monkeypatch, 0.85, 80, 8192, 8192)
    assert nbytes == 4 * GIB and R == 3276          # smaller than a step: the operator's call
    monkeypatch.setenv("MIA_APERTURE_GPU_BYTES", str(11 * GIB))
    assert _rows(monkeypatch, 0.85, 80, 8192, 8192)[1] == 11 * GIB


def test_8b_tp1_defaults_are_unchanged(monkeypatch):
    """At TP=1 and an 8192-token step, both 8B capture kinds get exactly 4 GiB."""
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    R, nbytes = _rows(monkeypatch, 0.9, 32, 4096, 8192)
    assert (R, nbytes) == (16384, 4 * GIB)
    w = SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.9)))
    R, nbytes = _resolve_qk_aperture_rows(w, 32, 4096, 1024, torch.bfloat16, "cuda:0",
                                          rows_needed=8192)
    assert nbytes == 4 * GIB and R == (4 * GIB) // (32 * 5120 * 2)


def test_qk_aperture_uses_sharded_widths_at_70b(monkeypatch):
    """A 70B Q/K row is sized per rank's shard, so 4 GiB holds a full 8192-token step at TP4/TP8."""
    monkeypatch.delenv("MIA_APERTURE_GPU_BYTES", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda d: SimpleNamespace(total_memory=H100))
    w = SimpleNamespace(vllm_config=SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.85)))
    for tp in (4, 8):
        s = qk_shard(0, tp, 64, 8, 128)
        R, nbytes = _resolve_qk_aperture_rows(w, 80, s.q_width, s.k_width, torch.bfloat16,
                                              "cuda:0", rows_needed=8192)
        assert nbytes == 4 * GIB and R >= 8192, (tp, R)


# --- Autocap (opt-in OOM guard): sharded Q/K widths and the model-sized aperture ---

def _cfg70b(tp, util=0.85, mnbt=8192):
    text = SimpleNamespace(num_hidden_layers=80, hidden_size=8192, num_attention_heads=64,
                           num_key_value_heads=8, head_dim=128)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text, dtype="torch.bfloat16"),
        cache_config=SimpleNamespace(gpu_memory_utilization=util),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=mnbt),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


@pytest.fixture
def nvml(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform",
                        SimpleNamespace(get_device_total_memory=lambda i: H100), raising=False)
    for k in ("MIA_APERTURE_GPU_BYTES", "MIA_APERTURE_AUTOCAP_SAFETY",
              "MIA_APERTURE_AUTOCAP_HEADROOM_BYTES"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("tp,local_q,local_kv", [(1, 64, 8), (4, 16, 2), (8, 8, 1)])
def test_autocap_dims_are_per_rank_shards(nvml, tp, local_q, local_kv):
    dims = _model_dims(_cfg70b(tp))
    assert (dims["n_q_heads"], dims["n_kv_heads"], dims["head_dim"]) == (local_q, local_kv, 128)


def test_autocap_qk_cap_is_computed_on_the_shard(nvml):
    got = _derive_safe_max_batched_tokens(_cfg70b(4), ["qk"])
    want = compute_safe_max_batched_tokens(H100, 0.85, DEFAULT_APERTURE_GPU_BYTES, 80,
                                           per_layer_token_bytes_qk(16, 2, 128, 2))
    full = compute_safe_max_batched_tokens(H100, 0.85, DEFAULT_APERTURE_GPU_BYTES, 80,
                                           per_layer_token_bytes_qk(64, 8, 128, 2))
    assert got == want and got > 3 * full


@pytest.mark.parametrize("shard,layers_per_rank", [("0", 80), (None, 20)])
def test_autocap_accounts_for_a_default_aperture_grown_to_the_step(nvml, monkeypatch, shard,
                                                                    layers_per_rank):
    """The cap fits the aperture grown to one step plus the transient (rank 0's per-rank row)."""
    if shard is None:
        monkeypatch.delenv("MIA_HS_TP_SHARD", raising=False)
    else:
        monkeypatch.setenv("MIA_HS_TP_SHARD", shard)
    monkeypatch.delenv("MIA_HS_CAPTURE_ALL_RANKS", raising=False)
    row = layers_per_rank * 8192 * 2
    cfg = _cfg70b(4, util=0.5, mnbt=1 << 20)
    naive = compute_safe_max_batched_tokens(H100, 0.5, DEFAULT_APERTURE_GPU_BYTES,
                                            layers_per_rank, 8192 * 2)
    got = _derive_safe_max_batched_tokens(cfg, ["hidden_states"])
    margin = round(0.5 * H100) - DEFAULT_AUTOCAP_HEADROOM_BYTES

    def fits(cap):
        return cap * row * DEFAULT_AUTOCAP_SAFETY + max(
            DEFAULT_APERTURE_GPU_BYTES, cap * row) <= margin

    assert not fits(naive)          # a fixed-aperture cap would overcommit
    assert fits(got) and not fits(got + 1)
