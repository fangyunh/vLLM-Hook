"""Graph-mode routing and drain wrappers around V2's `prepare_inputs` and `execute_model`.

The routing wrapper hands each builder a `StepView` of the batch `prepare_inputs` returns and
returns that batch untouched; failures propagate. Fakes carry a sentinel tail past `num_reqs`, as
in test_runner_adapter.py. No GPU: the device is reported absent to every test.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM; skip, never error the whole collection

from mia.graph.capture_aperture import ApertureBackpressureError
from mia.graph.install import (
    install_execute_model_wrapper,
    install_prepare_inputs_routing,
    prefix_block_ids,
)
from mia.graph.install_hs import install_execute_model_wrapper_hs
from mia.graph.registry import get_registry, set_registry
from mia.runner import StepView, UnsupportedRunnerError

_STALE = 999999

#: The per-step drain wrapper installer of each capture subsystem.
_INSTALL_EXECUTE = {"hs": install_execute_model_wrapper_hs, "qk": install_execute_model_wrapper}


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """Report no CUDA device: these CPU tests must not open a context on a GPU node."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class _Runner:
    """V2-shaped fake runner: request add/finish hooks and padded block tables, sentinel tail."""

    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self, num_reqs=1, pad=3):
        max_num_reqs = num_reqs + pad
        self.block_tables = types.SimpleNamespace(
            input_block_tables=(
                torch.full((max_num_reqs, 3), _STALE, dtype=torch.int32),
            )
        )
        self.block_tables.input_block_tables[0][:num_reqs] = torch.arange(
            num_reqs * 3, dtype=torch.int32).reshape(num_reqs, 3)
        self.added, self.finished = [], []

    def add_requests(self, so):
        self.added.append(so)

    def finish_requests(self, so):
        self.finished.append(so)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        return types.SimpleNamespace(
            req_ids=["r0"], num_reqs=1,
            num_scheduled_tokens=np.array([4], dtype=np.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            query_start_loc_np=np.array([0, 4], dtype=np.int32),
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            prefill_len_np=np.array([4], dtype=np.int32),
            is_prefilling_np=np.array([True]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.intp),
        )


class _V1Runner:
    __module__ = "vllm.v1.worker.gpu_model_runner"


def _fake_registry(**overrides):
    """A registry double complete enough for the wrapper's reset -> build -> upload path."""
    base = dict(
        should_capture=True,
        cap=8,
        incremental_enabled=False,
        gpu_routing=False,
        begin_step=lambda: None,
        routing_key=lambda step: None,
        reset_pinned=lambda width=None: None,
        upload=lambda width=None: None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _runner_with_req_states(num_reqs=1):
    """A `_Runner` with the req_states.prompt_len that step_view() reads through idx_mapping."""
    runner = _Runner(num_reqs=num_reqs)
    runner.req_states = types.SimpleNamespace(
        prompt_len=types.SimpleNamespace(np=np.array([4] * (num_reqs + 3), dtype=np.int32))
    )
    return runner


def test_builder_receives_a_step_view_and_the_batch_is_passed_through():
    """No registry armed: the builder never runs and vLLM's InputBatch passes through as is."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()  # no registry for label "test" at all
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    result = runner.prepare_inputs(object(), object(), object())
    assert result.req_ids == ["r0"], "the wrapper must return vLLM's InputBatch untouched"
    assert seen == []


def test_builder_runs_when_capture_is_armed():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert len(seen) == 1 and isinstance(seen[0], StepView)
    assert seen[0].is_prefilling_np.tolist() == [True]
    assert seen[0].req_ids == ["r0"]


def test_registry_absent_for_this_label_skips_even_if_another_label_is_armed():
    """Routing installed under label "hs" never sees the "qk" registry."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "qk", _fake_registry())
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="hs")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_should_capture_false_skips_the_builder():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry(should_capture=False))
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_never_drop_aperture_backpressure_propagates():
    """Never drop: ApertureBackpressureError reaches the engine."""
    runner = _runner_with_req_states()
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())

    def _boom(step, registry):
        raise ApertureBackpressureError("aperture full; consumer dead")

    install_prepare_inputs_routing(runner, worker, _boom, label="test")
    with pytest.raises(ApertureBackpressureError):
        runner.prepare_inputs(object(), object(), object())


def test_other_exceptions_also_propagate_rather_than_capturing_nothing(capsys):
    """Any other routing failure propagates too: skipping capture would look like a complete run."""
    runner = _runner_with_req_states()
    worker = types.SimpleNamespace()
    registry = _fake_registry()
    set_registry(worker, "test", registry)

    def _boom(step, registry):
        raise RuntimeError("routing build blew up")

    install_prepare_inputs_routing(runner, worker, _boom, label="test")
    with pytest.raises(RuntimeError, match="routing build blew up"):
        runner.prepare_inputs(object(), object(), object())
    # The original exception, so the traceback points at the real failure.
    out = capsys.readouterr().out
    assert "FATAL" in out and "routing failed" in out.lower()
    # Registry state is reset on the way out: a caller that catches it inherits no half-built plan.
    assert registry._pending_plans == []
    assert registry._last_route_key is None


def test_skips_during_cudagraph_capture_pass(monkeypatch):
    """No routing while vLLM captures a graph: the aperture fills at replay."""
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True,
                        raising=False)
    install_prepare_inputs_routing(runner, worker,
                                   lambda step, registry: seen.append(step), label="test")
    runner.prepare_inputs(object(), object(), object())
    assert seen == []


def test_install_is_idempotent_per_label():
    runner, seen = _runner_with_req_states(), []
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    build_fn = lambda step, registry: seen.append(step)
    install_prepare_inputs_routing(runner, worker, build_fn, label="test")
    wrapped_once = runner.prepare_inputs
    install_prepare_inputs_routing(runner, worker, build_fn, label="test")
    assert runner.prepare_inputs is wrapped_once, "a second install must be a no-op"


def test_v1_runner_is_refused_at_install_not_silently_skipped():
    worker = types.SimpleNamespace()
    set_registry(worker, "test", _fake_registry())
    with pytest.raises(UnsupportedRunnerError):
        install_prepare_inputs_routing(_V1Runner(), worker, lambda step, registry: None,
                                       label="test")


def test_two_subsystems_can_register_without_clobbering():
    worker = types.SimpleNamespace()
    set_registry(worker, "hs", "HS-REG")
    set_registry(worker, "qk", "QK-REG")
    assert get_registry(worker, "hs") == "HS-REG"   # not overwritten by the later set
    assert get_registry(worker, "qk") == "QK-REG"
    assert get_registry(worker, "steer") is None


def test_set_registry_overwrite_replaces_only_its_own_subsystem():
    worker = types.SimpleNamespace()
    set_registry(worker, "hs", "HS-REG-1")
    set_registry(worker, "qk", "QK-REG")
    set_registry(worker, "hs", "HS-REG-2")
    assert get_registry(worker, "hs") == "HS-REG-2"
    assert get_registry(worker, "qk") == "QK-REG"


# --- prefix_block_ids: V2 block tables are in batch order, so row i is request i's row. ---


def _step_with_block_tables(*group_rows, req_index_live_rows=2):
    """A StepView whose `block_tables` is `group_rows`, other fields minimal placeholders."""
    n = req_index_live_rows
    return StepView(
        req_ids=[f"r{i}" for i in range(n)],
        num_reqs=n,
        num_scheduled_tokens=np.array([1] * n, dtype=np.int32),
        query_start_loc=torch.arange(n + 1, dtype=torch.int32),
        query_start_loc_np=np.arange(n + 1, dtype=np.int32),
        num_computed_tokens_np=np.zeros(n, dtype=np.int32),
        prefill_len_np=np.array([1] * n, dtype=np.int32),
        prompt_len_np=np.array([1] * n, dtype=np.int32),
        is_prefilling_np=np.array([True] * n),
        seq_lens=torch.ones(n, dtype=torch.int32),
        block_tables=group_rows,
        extra_args={},
    )


def test_prefix_block_ids_reads_batch_ordered_v2_tables():
    """A plain row/column slice: distinct rows catch a swap, the sentinel tail a missing slice."""
    group0 = torch.tensor(
        [[7, 8, 9], [1, 2, 3], [_STALE, _STALE, _STALE], [_STALE, _STALE, _STALE]],
        dtype=torch.int32,
    )
    step = _step_with_block_tables(group0)

    assert prefix_block_ids(step, 0, num_blocks=2).tolist() == [7, 8]
    assert prefix_block_ids(step, 1, num_blocks=3).tolist() == [1, 2, 3]


def test_prefix_block_ids_selects_the_right_kv_cache_group():
    """`group` picks the KV-cache group's table; disjoint values catch the wrong group."""
    group0 = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    group1 = torch.tensor([[70, 80, 90]], dtype=torch.int32)
    step = _step_with_block_tables(group0, group1, req_index_live_rows=1)

    assert prefix_block_ids(step, 0, num_blocks=2, group=0).tolist() == [7, 8]
    assert prefix_block_ids(step, 0, num_blocks=2, group=1).tolist() == [70, 80]


def test_prefix_block_ids_returns_none_without_block_tables():
    """No KV-cache group (an attention-free model) gives None, not an IndexError."""
    step = _step_with_block_tables(req_index_live_rows=1)
    assert prefix_block_ids(step, 0, num_blocks=2) is None


# --- The drain wrapper skips dummy/profile passes and blinds the stale routing table meanwhile. ---


def _fake_capture_registry(**overrides):
    """A registry double whose non-zero `capture_index_all` is the routing table the op reads."""
    base = dict(
        should_capture=True,
        capture_index_all=torch.tensor([[5, 6, 0, 0], [1, 2, 3, 0]], dtype=torch.int64),
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class _ExecRunner:
    """V2-shaped fake with what the execute_model wrapper's install touches."""

    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self):
        self.added, self.finished = [], []

    def add_requests(self, so):
        self.added.append(so)

    def finish_requests(self, so):
        self.finished.append(so)

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        return types.SimpleNamespace(
            req_ids=["r0"], num_reqs=1,
            num_scheduled_tokens=np.array([4], dtype=np.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            query_start_loc_np=np.array([0, 4], dtype=np.int32),
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            prefill_len_np=np.array([4], dtype=np.int32),
            is_prefilling_np=np.array([True]),
            seq_lens=torch.tensor([4], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.intp),
        )


@pytest.mark.parametrize("subsystem,install_name,plans_attr,entries_attr,rows_attr,start_attr", [
    ("hs", "install_execute_model_wrapper_hs", "_pending_plans",
     "_hs_step_entries", "_hs_step_rows", "_hs_step_start"),
    ("qk", "install_execute_model_wrapper", "_pending_plans",
     "_qk_step_entries", "_qk_step_rows", "_qk_step_start"),
])
def test_drain_wrapper_skips_dummy_runs_and_forwards_v2_kwargs(
        subsystem, install_name, plans_attr, entries_attr, rows_attr, start_attr):
    """A dummy/profile pass never reaches the drain; V2's kwargs reach execute_model as is."""
    install_fn = _INSTALL_EXECUTE[subsystem]
    runner = _ExecRunner()
    calls = []

    def fake_orig(scheduler_output, **kw):
        calls.append(kw)
        return "out"

    runner.execute_model = fake_orig

    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    registry = _fake_capture_registry()
    # State left by a previous real step: a dummy pass must not drain it.
    setattr(registry, plans_attr, ["STALE_PLAN"])
    setattr(registry, entries_attr, ["STALE_RECORD"])
    setattr(registry, rows_attr, _STALE)
    setattr(registry, start_attr, _STALE)
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    drained = []
    drain_attr = "_hs_drain" if subsystem == "hs" else "_qk_drain"
    setattr(worker, drain_attr,
            types.SimpleNamespace(enqueue=lambda *a, **k: drained.append(a),
                                  per_request=False))

    # dummy_run: no drain, kwargs forwarded as is.
    result = runner.execute_model(object(), dummy_run=True)
    assert result == "out"
    assert drained == [], "a dummy run must never reach the drain"
    assert calls[-1] == {"dummy_run": True}
    assert getattr(registry, plans_attr) == ["STALE_PLAN"], (
        "stale state from before the dummy pass must survive it unchanged")

    # is_profile, with the other dummy-pass kwargs.
    result = runner.execute_model(object(), dummy_run=True, is_profile=True,
                                  skip_attn_for_dummy_run=True, context_len=7)
    assert result == "out"
    assert drained == []
    assert calls[-1] == {"dummy_run": True, "is_profile": True,
                          "skip_attn_for_dummy_run": True, "context_len": 7}

    # A real step with the same leftover plan drains: the guard keys on dummy_run/is_profile.
    result = runner.execute_model(object(), dummy_run=False, is_profile=False)
    assert result == "out"
    assert drained, "a real step with pending plans must reach the drain"
    assert calls[-1] == {"dummy_run": False, "is_profile": False}


@pytest.mark.parametrize("subsystem,install_name", [
    ("hs", "install_execute_model_wrapper_hs"),
    ("qk", "install_execute_model_wrapper"),
])
def test_drain_wrapper_blinds_stale_routing_during_dummy_pass_and_restores(subsystem, install_name):
    """A dummy pass replays the graph, so the routing table reads all-zero (discard) during it."""
    install_fn = _INSTALL_EXECUTE[subsystem]
    runner = _ExecRunner()
    registry = _fake_capture_registry()
    stale = registry.capture_index_all.clone()
    seen_during_call = {}

    def fake_orig(scheduler_output, **kw):
        # What the baked op would read at replay.
        seen_during_call["ci"] = registry.capture_index_all.clone()
        return "out"

    runner.execute_model = fake_orig
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    runner.execute_model(object(), dummy_run=True)

    assert seen_during_call["ci"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]], (
        "capture_index_all must read all-inactive DURING a dummy pass")
    assert registry.capture_index_all.tolist() == stale.tolist(), (
        "the real routing table must be restored byte-for-byte after the dummy pass")


@pytest.mark.parametrize("subsystem,install_name", [
    ("hs", "install_execute_model_wrapper_hs"),
    ("qk", "install_execute_model_wrapper"),
])
def test_drain_wrapper_restores_stale_routing_on_exception(subsystem, install_name):
    """A dummy pass that raises still restores the real routing table."""
    install_fn = _INSTALL_EXECUTE[subsystem]
    runner = _ExecRunner()
    registry = _fake_capture_registry()
    stale = registry.capture_index_all.clone()

    def fake_orig(scheduler_output, **kw):
        raise RuntimeError("boom mid dummy forward")

    runner.execute_model = fake_orig
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, registry)

    install_fn(runner, worker)
    with pytest.raises(RuntimeError, match="boom mid dummy forward"):
        runner.execute_model(object(), dummy_run=True)

    assert registry.capture_index_all.tolist() == stale.tolist(), (
        "an exception mid dummy-forward must still restore the real routing table")


@pytest.mark.parametrize("subsystem,install_name,wrapped_attr", [
    ("hs", "install_execute_model_wrapper_hs", "_mia_hs_wrapped"),
    ("qk", "install_execute_model_wrapper", "_mia_qk_wrapped"),
])
def test_execute_model_wrapper_is_idempotent(subsystem, install_name, wrapped_attr):
    install_fn = _INSTALL_EXECUTE[subsystem]
    runner = _ExecRunner()
    runner.execute_model = lambda so, **kw: "out"
    worker = types.SimpleNamespace(hs_mode="last_token", hookq_mode="all_tokens",
                                    _default_hooks_on="prefill")
    set_registry(worker, subsystem, _fake_capture_registry())

    install_fn(runner, worker)
    wrapped_once = runner.execute_model
    install_fn(runner, worker)
    assert runner.execute_model is wrapped_once, "a second install must be a no-op"
    assert getattr(runner, wrapped_attr) is True
