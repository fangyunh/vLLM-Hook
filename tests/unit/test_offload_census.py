"""Unit tests for vllm_hook_plugins.graph.census (offload cost-attribution measurement).

No GPU, no engine -- pure CPU tensors. Run:
  pytest tests/unit/test_offload_census.py -vv

Covers the three success criteria from the measurement spec
(docs/superpowers/specs/2026-07-29-offload-cost-attribution-measurement.md):
  (1) cpu_list_measured returns values equal to [t.cpu() for t in tensors] elementwise;
  (2) the accumulator totals are populated and self-consistent (n/bytes match the input,
      alloc_s/copy_s are non-negative and their sum is the alloc_frac denominator);
  (3) census_bucket counts tensors and bytes correctly on a hand-built request bucket.
"""
import json
import os

import torch

from vllm_hook_plugins.graph import census


# ---------------------------------------------------------------------------
# cpu_list_measured
# ---------------------------------------------------------------------------


def test_cpu_list_measured_matches_reference_values():
    torch.manual_seed(0)
    tensors = [torch.randn(4, 8, dtype=torch.float16) for _ in range(5)]
    expected = [t.cpu() for t in tensors]

    acc = census.new_accumulator()
    out = census.cpu_list_measured(tensors, acc)

    assert len(out) == len(expected)
    for got, want in zip(out, expected):
        assert torch.equal(got, want)
        assert got.dtype == want.dtype
        assert got.device.type == "cpu"


def test_cpu_list_measured_empty_list_is_a_noop():
    acc = census.new_accumulator()
    out = census.cpu_list_measured([], acc)
    assert out == []
    assert acc == {"n": 0, "bytes": 0, "alloc_s": 0.0, "copy_s": 0.0}


def test_cpu_list_measured_accumulator_is_self_consistent():
    torch.manual_seed(1)
    tensors = [torch.randn(3, 6, dtype=torch.float32) for _ in range(7)]
    expected_bytes = sum(t.numel() * t.element_size() for t in tensors)

    acc = census.new_accumulator()
    census.cpu_list_measured(tensors, acc)

    assert acc["n"] == len(tensors)
    assert acc["bytes"] == expected_bytes
    assert acc["alloc_s"] >= 0.0
    assert acc["copy_s"] >= 0.0
    # perf_counter is monotonic and each half was timed separately -> the sum is a
    # non-negative real duration (may be ~0 on a very fast/optimized run, never negative).
    assert acc["alloc_s"] + acc["copy_s"] >= 0.0


def test_cpu_list_measured_accumulates_across_calls():
    torch.manual_seed(2)
    batch_a = [torch.randn(2, 2) for _ in range(3)]
    batch_b = [torch.randn(2, 2) for _ in range(4)]

    acc = census.new_accumulator()
    census.cpu_list_measured(batch_a, acc)
    census.cpu_list_measured(batch_b, acc)

    assert acc["n"] == 7
    expected_bytes = sum(t.numel() * t.element_size() for t in batch_a + batch_b)
    assert acc["bytes"] == expected_bytes


# ---------------------------------------------------------------------------
# census_bucket
# ---------------------------------------------------------------------------


def _hand_built_bucket():
    """A two-module QK-shaped bucket: module 0 has 3 q + 3 k_all appends (both=3 chunks),
    module 1 has 2 appends. Sizes are deliberately distinct dtypes/shapes."""
    q0 = [torch.randn(4, 16, dtype=torch.float16) for _ in range(3)]
    k0 = [torch.randn(4, 16, dtype=torch.float16) for _ in range(3)]
    q1 = [torch.randn(2, 16, dtype=torch.float32) for _ in range(2)]
    k1 = [torch.randn(2, 16, dtype=torch.float32) for _ in range(2)]
    return {
        "layer.0.self_attn": {"q": q0, "k_all": k0, "layer_num": 0, "hookq_mode": "all_tokens"},
        "layer.1.self_attn": {"q": q1, "k_all": k1, "layer_num": 1, "hookq_mode": "all_tokens"},
    }


def test_census_bucket_counts_tensors_and_bytes():
    bucket = _hand_built_bucket()
    rec = census.census_bucket(bucket)

    all_tensors = (bucket["layer.0.self_attn"]["q"] + bucket["layer.0.self_attn"]["k_all"]
                   + bucket["layer.1.self_attn"]["q"] + bucket["layer.1.self_attn"]["k_all"])
    expected_total_bytes = sum(t.numel() * t.element_size() for t in all_tensors)

    assert rec["total_tensors"] == len(all_tensors) == 10
    assert rec["total_bytes"] == expected_total_bytes
    assert rec["num_layers"] == 2

    # per-key breakdown: q and k_all each contribute 3+2=5 tensors here.
    assert rec["per_key_count"]["q"] == 5
    assert rec["per_key_count"]["k_all"] == 5
    assert "hidden_states" not in rec["per_key_count"]
    assert "scores" not in rec["per_key_count"]

    q_bytes = sum(t.numel() * t.element_size()
                  for t in bucket["layer.0.self_attn"]["q"] + bucket["layer.1.self_attn"]["q"])
    assert rec["per_key_bytes"]["q"] == q_bytes

    # appends-per-module reflects the number of forward passes that touched each module.
    assert rec["appends_per_module"]["layer.0.self_attn"] == 3
    assert rec["appends_per_module"]["layer.1.self_attn"] == 2

    # size histogram sums back to the total tensor count.
    assert sum(rec["size_histogram"].values()) == rec["total_tensors"]

    # dtype set covers both dtypes used above.
    assert set(rec["dtypes"]) == {"torch.float16", "torch.float32"}

    # all tensors here are CPU (no GPU in this unit test).
    assert rec["device_counts"]["cpu"] == rec["total_tensors"]
    assert rec["device_counts"]["cuda"] == 0


def test_census_bucket_empty_bucket():
    rec = census.census_bucket({})
    assert rec["total_tensors"] == 0
    assert rec["total_bytes"] == 0
    assert rec["num_layers"] == 0
    assert rec["appends_per_module"] == {}


def test_census_bucket_ignores_non_tensor_and_unknown_keys():
    bucket = {
        "mod": {
            "hidden_states": [torch.randn(2, 4)],
            "layer_num": 3,           # not a tensor -- must be ignored, not crash
            "hs_mode": "last_token",  # string, ignored
            "k_prefix_ends": [4],     # list of ints, not tensors -- ignored (not a known key
                                       # anyway, but also guards the hasattr(t, "numel") check)
        }
    }
    rec = census.census_bucket(bucket)
    assert rec["total_tensors"] == 1
    assert rec["per_key_count"] == {"hidden_states": 1}


# ---------------------------------------------------------------------------
# census_record -- the alloc_frac / bandwidth derivation
# ---------------------------------------------------------------------------


def test_census_record_computes_alloc_frac_and_bandwidth():
    bucket = {"total_tensors": 10, "total_bytes": 1000, "per_key_count": {}, "per_key_bytes": {},
              "size_histogram": {}, "dtypes": [], "device_counts": {"cuda": 0, "cpu": 10},
              "num_layers": 1, "appends_per_module": {}}
    acc = {"n": 10, "bytes": 1000, "alloc_s": 0.75, "copy_s": 0.25}

    rec = census.census_record(worker="qk", sink="rpc", req_id="req-1", bucket=bucket, acc=acc)

    assert rec["worker"] == "qk"
    assert rec["sink"] == "rpc"
    assert rec["req_id"] == "req-1"
    assert rec["n_tensors"] == 10
    assert rec["total_bytes"] == 1000
    assert rec["measured_n_tensors"] == 10
    assert rec["measured_bytes"] == 1000
    assert rec["alloc_frac"] == 0.75
    assert rec["bandwidth_gbps"] == 1000 / 0.25 / 1e9


def test_census_record_handles_zero_duration():
    bucket = {"total_tensors": 0, "total_bytes": 0}
    acc = {"n": 0, "bytes": 0, "alloc_s": 0.0, "copy_s": 0.0}
    rec = census.census_record(worker="hs", sink="disk", req_id="req-2", bucket=bucket, acc=acc)
    assert rec["alloc_frac"] is None
    assert rec["bandwidth_gbps"] is None


# ---------------------------------------------------------------------------
# census_emit -- robustness (never raises), format, and default-path routing
# ---------------------------------------------------------------------------


def test_census_emit_writes_one_json_line_per_call(tmp_path, monkeypatch):
    out_path = tmp_path / "census.jsonl"
    monkeypatch.setenv("VLLM_HOOK_CAPTURE_CENSUS_OUT", str(out_path))

    census.census_emit({"a": 1})
    census.census_emit({"a": 2})

    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}


def test_census_emit_falls_back_to_profile_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_CAPTURE_CENSUS_OUT", raising=False)
    monkeypatch.setenv("VLLM_HOOK_PROFILE_DIR", str(tmp_path))

    census.census_emit({"b": 1})

    out_path = tmp_path / "census.jsonl"
    assert out_path.exists()
    assert json.loads(out_path.read_text().splitlines()[0]) == {"b": 1}


def test_census_emit_swallows_failures(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_PROFILE_DIR", raising=False)
    # A path under a file (not a directory) can never be opened for append -> guaranteed
    # failure, exercising the swallow-and-count branch without raising into the caller.
    bogus_dir = "/dev/null/not-a-real-subdir"
    monkeypatch.setenv("VLLM_HOOK_CAPTURE_CENSUS_OUT", os.path.join(bogus_dir, "census.jsonl"))

    before = census.census_emit_failures()
    census.census_emit({"c": 1})  # must not raise
    after = census.census_emit_failures()

    assert after == before + 1


# ---------------------------------------------------------------------------
# census_enabled -- gate read once at import
# ---------------------------------------------------------------------------


def test_census_enabled_reflects_import_time_env():
    # The test process was started without VLLM_HOOK_CAPTURE_CENSUS=1, so the module-level
    # gate (read once at import, matching the rest of the package's env-gated levers) is off.
    assert census.census_enabled() is False
