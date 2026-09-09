"""Unit tests for the EAGER QK compact k_all layout.

No GPU, no engine — pure CPU accumulation + serialize round-trip. Run:
  pytest tests/unit/test_qk_eager_compact.py -vv

The eager forward hook sees each pass's FULL key prefix (prefill [P,Kd], then [P+1,Kd],
[P+2,Kd], ... one per decode step). Appending those verbatim makes the capture bucket —
and every artifact downstream of it — O(seq^2). The buffer/graph path has always stored
only each pass's NEW rows plus the per-pass prefix length (``k_prefix_ends``), which is
O(seq) and which ``_k_all_cpu_list`` rebuilds into the exact growing prefixes.

``append_k_prefix`` gives the eager path that same layout, so the compact machinery it
was previously excluded from (the RPC ``_use_compact_kall`` ship and the native-compact
safetensors layout) now fires for eager captures too. Invariants:
  (A) byte-identity — the rebuilt growing prefixes equal what eager used to store;
  (B) O(seq) residency — the bucket holds the unique keys, not the growing prefixes;
  (C) the quantized path and a non-monotonic (preempted+recomputed) request keep the
      legacy full-prefix append.
"""
import glob
import math
import os
import tempfile

import torch

from vllm_hook_plugins.graph import artifact_writer as aw
from vllm_hook_plugins import run_utils as ru
from vllm_hook_plugins.workers.probe_hookqk_worker import (
    _k_all_compact,
    _k_all_cpu_list,
    _use_compact_kall,
    append_k_prefix,
    new_qk_entry,
)

Kd = 6  # kv_hidden (small, arbitrary)
H = 8   # hidden


def _prefixes(lengths, seed=0):
    """The per-pass FULL key prefixes the eager hook sees: each is a prefix of the last."""
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(max(lengths), Kd, generator=g)
    return [full[:n].clone() for n in lengths]


def _new_entry(quantized=False):
    """A fresh eager QK entry, built by the hook's own initializer."""
    qmeta = {"tag": "int8"} if quantized else None
    return new_qk_entry(0, "all_tokens", q_qmeta=qmeta, k_qmeta=qmeta)


def test_native_entry_starts_in_delta_mode():
    """The hook's initializer is what arms compact accumulation for eager captures."""
    entry = new_qk_entry(3, "all_tokens")

    assert entry["k_prefix_ends"] == []
    assert entry["k_all"] == [] and entry["q"] == []
    assert entry["layer_num"] == 3 and entry["hookq_mode"] == "all_tokens"


def test_quantized_entry_starts_on_the_legacy_path():
    """Quantized captures keep the scale/qmeta channels and no delta accumulation."""
    entry = new_qk_entry(0, "all_tokens", q_qmeta={"tag": "int8"}, k_qmeta={"tag": "int8"})

    assert "k_prefix_ends" not in entry
    assert entry["_q_scale"] == [] and entry["_k_all_scale"] == []
    assert entry["_q_qmeta"] == {"tag": "int8"} and entry["_k_all_qmeta"] == {"tag": "int8"}


def _capture(lengths, seed=0, quantized=False):
    entry = _new_entry(quantized)
    passes = _prefixes(lengths, seed)
    for k_tok in passes:
        append_k_prefix(entry, k_tok)
    return entry, passes


def test_growing_prefixes_rebuild_byte_identically():
    """(A) What the eager path used to store verbatim is rebuilt exactly."""
    entry, passes = _capture([4, 5, 6, 7], seed=1)

    rebuilt = _k_all_cpu_list(entry)

    assert len(rebuilt) == len(passes)
    for got, want in zip(rebuilt, passes):
        assert torch.equal(got, want)


def test_capture_residency_is_O_seq_not_O_seq2():
    """(B) The bucket holds the unique keys, not the sum of the growing prefixes."""
    lengths = [4, 5, 6, 7]
    entry, _ = _capture(lengths, seed=1)

    stored = sum(t.shape[0] for t in entry["k_all"])
    assert stored == max(lengths)          # O(seq): the unique key rows, once
    assert stored < sum(lengths)           # what the verbatim append cost


def test_stored_deltas_do_not_pin_the_full_prefix_storage():
    """A delta kept as a VIEW would hold its whole pass's prefix alive — no win."""
    entry, _ = _capture([4, 5, 6, 7], seed=1)

    for t in entry["k_all"][1:]:
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()


def test_prefill_then_decode_records_every_pass():
    """The common trajectory shape: one prefill chunk then one row per decode step."""
    entry, passes = _capture([16, 17, 18, 19, 20], seed=2)

    assert entry["k_prefix_ends"] == [16, 17, 18, 19, 20]
    assert [t.shape[0] for t in entry["k_all"]] == [16, 1, 1, 1, 1]
    for got, want in zip(_k_all_cpu_list(entry), passes):
        assert torch.equal(got, want)


def test_chunked_prefill_passes_rebuild_byte_identically():
    """all_tokens captures every prefill chunk — those grow by the chunk size, not by 1."""
    entry, passes = _capture([8, 16, 24, 25, 26], seed=3)

    assert [t.shape[0] for t in entry["k_all"]] == [8, 8, 8, 1, 1]
    for got, want in zip(_k_all_cpu_list(entry), passes):
        assert torch.equal(got, want)


def test_single_snapshot_stays_off_the_compact_wire():
    """(C) last_token+prefill captures once; the trajectory default must not compact it."""
    entry, passes = _capture([9], seed=4)

    assert _use_compact_kall(entry) is False
    rebuilt = _k_all_cpu_list(entry)
    assert len(rebuilt) == 1 and torch.equal(rebuilt[0], passes[0])


def test_trajectory_entry_arms_the_compact_wire():
    """Two or more passes is the O(seq^2) case the compact ship exists for."""
    entry, _ = _capture([4, 5], seed=5)

    assert _use_compact_kall(entry) is True
    full, ends = _k_all_compact(entry)
    assert full.shape[0] == 5 and ends == [4, 5]


def test_non_monotonic_prefix_falls_back_to_full_prefixes():
    """(C) A preempted+recomputed request re-prefills, so the prefix length DROPS.
    That cannot be delta-encoded; the entry must fall back to storing full prefixes
    and still rebuild every pass byte-identically."""
    entry = _new_entry()
    grow = _prefixes([4, 5, 6], seed=6)
    for k_tok in grow:
        append_k_prefix(entry, k_tok)
    restart = _prefixes([3, 4], seed=7)      # recompute: back down to 3
    for k_tok in restart:
        append_k_prefix(entry, k_tok)

    assert "k_prefix_ends" not in entry      # delta mode dropped for this entry
    rebuilt = _k_all_cpu_list(entry)
    assert len(rebuilt) == 5
    for got, want in zip(rebuilt, grow + restart):
        assert torch.equal(got, want)


def test_quantized_entry_keeps_the_legacy_full_prefix_append():
    """(C) Quantized captures pack rows; they stay on the untouched verbatim path."""
    entry, passes = _capture([4, 5, 6], seed=8, quantized=True)

    assert "k_prefix_ends" not in entry
    assert [t.shape[0] for t in entry["k_all"]] == [4, 5, 6]
    for got, want in zip(_k_all_cpu_list(entry), passes):
        assert torch.equal(got, want)


def test_compact_rpc_wire_rebuilds_the_padded_k_all_on_the_driver():
    """The other ingest my change newly feeds: the RPC ship. The worker sends
    k_full+k_prefix_ends and the driver rebuilds the padded k_all that both the serve
    and offline paths used to receive directly."""
    from torch.nn.utils.rnn import pad_sequence
    from vllm_hook_plugins._hook_plugin import _reconstruct_compact_qk

    entry, passes = _capture([4, 5, 6, 7], seed=13)
    legacy = pad_sequence(passes, batch_first=True)      # what eager used to ship
    full, ends = _k_all_compact(entry)
    probes = {"qk_cache": {"model.layers.0.self_attn": {
        "q": torch.zeros(4, H), "k_full": full, "k_prefix_ends": ends,
        "layer_num": 0, "hookq_mode": "all_tokens"}}}

    _reconstruct_compact_qk(probes)

    got = probes["qk_cache"]["model.layers.0.self_attn"]
    assert "k_full" not in got and "k_prefix_ends" not in got
    assert torch.equal(got["k_all"], legacy)


def test_last_token_both_phase_entry_round_trips_on_disk():
    """last_token + hooks_on=both captures the last q row per pass but the FULL key
    prefix, so it now goes compact too — a shape the eager path never produced before.
    The writer's last_token branch pads k_all, so the compact entry must expand for it."""
    entry, passes = _capture([9, 10, 11], seed=11)
    q = [torch.randn(H, generator=torch.Generator().manual_seed(12)) for _ in passes]
    full, ends = _k_all_compact(entry)
    cpu_entry = {"q": q, "k_full": [full], "k_prefix_ends": [ends],
                 "layer_num": 0, "hookq_mode": "last_token"}
    cache = {"config": {"model": "x"},
             "qk_cache": {"model.layers.0.self_attn": cpu_entry}}

    with tempfile.TemporaryDirectory() as d:
        run_dir = os.path.join(d, "tp_rank_0")
        os.makedirs(run_dir)
        aw.save_qk_cache_safetensors(cache, run_dir, "last_token", 0)
        st = glob.glob(os.path.join(run_dir, "qk.safetensors"))
        assert st, "no qk.safetensors written (unexpected .pt fallback)"
        loaded = ru._load_and_merge_qk_safetensors("/x", "/y", st)

    got = loaded["qk_cache"]["model.layers.0.self_attn"]
    assert len(got["k_all"]) == len(passes)
    for a, b in zip(got["k_all"], passes):
        assert torch.equal(a, b)


def test_eager_entry_serializes_compact_to_disk_and_round_trips():
    """End to end: an eager-captured entry now takes the native-compact safetensors
    layout (O(seq) on disk) and the analyzer's own loader rebuilds it byte-identically."""
    entry, passes = _capture([4, 5, 6, 7], seed=9)
    q = [torch.randn(n, H, generator=torch.Generator().manual_seed(10 + n))
         for n in (4, 1, 1, 1)]
    full, ends = _k_all_compact(entry)
    # what flush_disk builds for a compact entry
    cpu_entry = {"q": q, "k_full": [full], "k_prefix_ends": [ends],
                 "layer_num": 0, "hookq_mode": "all_tokens"}
    cache = {"config": {"model": "x"},
             "qk_cache": {"model.layers.0.self_attn": cpu_entry}}

    with tempfile.TemporaryDirectory() as d:
        run_dir = os.path.join(d, "tp_rank_0")
        os.makedirs(run_dir)
        aw.save_qk_cache_safetensors(cache, run_dir, "all_tokens", 0)
        st = glob.glob(os.path.join(run_dir, "qk.safetensors"))
        assert st, "no qk.safetensors written (unexpected .pt fallback)"
        from safetensors import safe_open
        with safe_open(st[0], framework="pt") as sf:
            k_shape = next(tuple(sf.get_slice(k).get_shape())
                           for k in sf.keys() if k.endswith("__k"))
        loaded = ru._load_and_merge_qk_safetensors("/x", "/y", st)

    assert math.prod(k_shape) == max(len(p) for p in passes) * Kd   # O(seq) on disk
    got = loaded["qk_cache"]["model.layers.0.self_attn"]
    assert len(got["k_all"]) == len(passes)
    for a, b in zip(got["k_all"], passes):
        assert torch.equal(a, b)
    for a, b in zip(got["q"], q):
        assert torch.equal(a, b)
