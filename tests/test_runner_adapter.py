"""Adapter contract, proven against fakes so it runs anywhere in milliseconds.

The fakes mimic vLLM 0.29's V2 shapes: a transient InputBatch dataclass with numpy
fields, req_states with req_id_to_index, and block_tables.input_block_tables.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")  # `import mia` pulls in vLLM; skip, never error the whole collection

import mia.workers.hs_capture_worker as hs
import mia.workers.qk_capture_worker as qk
import mia.workers.steer_worker as steer
from mia.runner import (
    StepView,
    UnsupportedRunnerError,
    install_request_arg_stash,
    is_v2_runner,
    require_v2_runner,
    step_view,
)

# A value no live row holds, so a dropped `[:n]` slice shows up in an assertion.
_STALE = 999999


class _FakeV2Runner:
    __module__ = "vllm.v1.worker.gpu.model_runner"

    def __init__(self, num_reqs=2, pad=3):
        # Like V2's block tables: max_num_reqs rows; live rows [[0, 1, 2], [3, 4, 5]], stale tail.
        max_num_reqs = num_reqs + pad
        self.block_tables = types.SimpleNamespace(
            input_block_tables=(
                torch.arange(max_num_reqs * 3, dtype=torch.int32).reshape(
                    max_num_reqs, 3
                ),
            )
        )
        # Indexed by req_state slot, not batch row: distinct values expose a skipped idx_mapping.
        self.req_states = types.SimpleNamespace(
            prompt_len=types.SimpleNamespace(
                np=np.arange(1000, 1000 + max_num_reqs, dtype=np.int32)
            )
        )
        self.added = []
        self.finished = []

    def add_requests(self, scheduler_output):
        self.added.append(scheduler_output)

    def finish_requests(self, scheduler_output):
        self.finished.append(scheduler_output)


class _FakeV1Runner:
    __module__ = "vllm.v1.worker.gpu_model_runner"


def _fake_input_batch(num_reqs=2, pad=3):
    """A V2 InputBatch, every field padded past num_reqs with a sentinel tail (as under graphs)."""
    max_num_reqs = num_reqs + pad

    num_scheduled_tokens = np.full(max_num_reqs, _STALE, dtype=np.int32)
    num_scheduled_tokens[:num_reqs] = [5, 1]

    query_start_loc_np = np.full(max_num_reqs + 1, _STALE, dtype=np.int32)
    query_start_loc_np[: num_reqs + 1] = [0, 5, 6]
    query_start_loc = torch.from_numpy(query_start_loc_np.copy())

    num_computed_tokens_np = np.full(max_num_reqs, _STALE, dtype=np.int32)
    num_computed_tokens_np[:num_reqs] = [0, 9]

    prefill_len_np = np.full(max_num_reqs, _STALE, dtype=np.int32)
    prefill_len_np[:num_reqs] = [5, 9]

    # A True tail: a leaked tail element differs from row 1's live False.
    is_prefilling_np = np.full(max_num_reqs, True)
    is_prefilling_np[:num_reqs] = [True, False]

    seq_lens = torch.full((max_num_reqs,), _STALE, dtype=torch.int32)
    seq_lens[:num_reqs] = torch.tensor([5, 10], dtype=torch.int32)

    # Batch row -> req_state slot, reversed for live rows; tail rows map to unused slots.
    idx_mapping_np = np.arange(max_num_reqs, dtype=np.intp)
    idx_mapping_np[:num_reqs] = np.arange(num_reqs - 1, -1, -1, dtype=np.intp)

    return types.SimpleNamespace(
        req_ids=[f"r{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc_np,
        num_computed_tokens_np=num_computed_tokens_np,
        prefill_len_np=prefill_len_np,
        idx_mapping_np=idx_mapping_np,
        is_prefilling_np=is_prefilling_np,
        seq_lens=seq_lens,
    )


def _sched_output(new_reqs=(), finished=()):
    return types.SimpleNamespace(
        scheduled_new_reqs=[
            types.SimpleNamespace(
                req_id=rid,
                sampling_params=types.SimpleNamespace(extra_args=extra),
            )
            for rid, extra in new_reqs
        ],
        finished_req_ids=list(finished),
    )


def test_detects_v2_by_module_path():
    assert is_v2_runner(_FakeV2Runner()) is True
    assert is_v2_runner(_FakeV1Runner()) is False


def test_v1_runner_fails_loud_rather_than_capturing_nothing():
    with pytest.raises(UnsupportedRunnerError, match="V2"):
        require_v2_runner(_FakeV1Runner())


def test_stash_keeps_extra_args_that_v2_drops():
    runner = _FakeV2Runner()
    stash = install_request_arg_stash(runner)
    runner.add_requests(_sched_output(new_reqs=[("r0", {"steer": {"layer": 15}})]))
    assert stash["r0"] == {"steer": {"layer": 15}}


def test_stash_prunes_on_finish_so_it_cannot_grow_without_bound():
    runner = _FakeV2Runner()
    stash = install_request_arg_stash(runner)
    runner.add_requests(_sched_output(new_reqs=[("r0", {"a": 1})]))
    runner.finish_requests(_sched_output(finished=["r0"]))
    assert "r0" not in stash


def test_stash_install_is_idempotent():
    runner = _FakeV2Runner()
    first = install_request_arg_stash(runner)
    second = install_request_arg_stash(runner)
    assert first is second
    runner.add_requests(_sched_output(new_reqs=[("r0", {"a": 1})]))
    assert len(first) == 1  # not double-wrapped


def test_step_view_maps_every_field_and_slices_to_num_reqs():
    """Exact contents: a dropped `[:n]` anywhere in `step_view()` leaks a sentinel."""
    runner = _FakeV2Runner()
    stash = install_request_arg_stash(runner)
    view = step_view(runner, _fake_input_batch(), stash)

    assert isinstance(view, StepView)
    assert view.req_ids == ["r0", "r1"]
    assert view.num_reqs == 2
    assert view.num_scheduled_tokens.tolist() == [5, 1]
    assert view.query_start_loc.tolist() == [0, 5, 6]
    assert view.query_start_loc_np.tolist() == [0, 5, 6]
    assert view.num_computed_tokens_np.tolist() == [0, 9]
    assert view.prefill_len_np.tolist() == [5, 9]
    # Reversed by idx_mapping: identity indexing would read [1000, 1001].
    assert view.prompt_len_np.tolist() == [1001, 1000]
    assert view.is_prefilling_np.tolist() == [True, False]
    assert view.seq_lens.tolist() == [5, 10]
    assert len(view.block_tables) == 1
    assert view.block_tables[0].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert view.extra_args is stash


def test_step_view_is_immutable():
    view = step_view(_FakeV2Runner(), _fake_input_batch(), {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.num_reqs = 99


# --- Each eager worker's install_hooks() refuses a V1 runner before touching anything else. ---


def test_workers_refuse_a_v1_runner():
    class _Worker(hs.HSCaptureWorker):
        def __init__(self):  # bypass vLLM's Worker.__init__
            self.model_runner = _FakeV1Runner()

    with pytest.raises(UnsupportedRunnerError):
        _Worker().install_hooks()


def test_qk_worker_refuses_a_v1_runner():
    class _Worker(qk.QKCaptureWorker):
        def __init__(self):
            self.model_runner = _FakeV1Runner()

    with pytest.raises(UnsupportedRunnerError):
        _Worker().install_hooks()


def test_steer_worker_refuses_a_v1_runner():
    """The refusal runs outside install_hooks' broad except, so it is raised, not printed."""
    class _Worker(steer.SteerWorker):
        def __init__(self):
            self.model_runner = _FakeV1Runner()

    with pytest.raises(UnsupportedRunnerError):
        _Worker().install_hooks()
