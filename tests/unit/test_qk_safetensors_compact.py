"""Unit tests for the QK safetensors NATIVE-COMPACT all_tokens layout.

No GPU, no engine — pure CPU serialize -> deserialize round-trip. Run:
  pytest tests/unit/test_qk_safetensors_compact.py -vv

Guards the fix that stopped the QK disk artifact from growing O(seq^2): the worker writes
k in the compact (k_full + k_prefix_ends) form, and the safetensors writer now stores it
CONCATENATED (O(seq)) instead of expanding to the padded growing-prefix tensor (O(seq^2),
which blew a ~900-token request to 100+ GB). Two invariants:
  (A) byte-identity — the reader rebuilds exactly the growing-prefix k_all and per-step q;
  (B) O(seq) on disk — the stored k tensor is the unique keys, not the padded prefixes.
A last_token case guards the legacy padded path (must be untouched).
"""
import glob
import math
import os
import tempfile

import torch

from vllm_hook_plugins.graph import artifact_writer as aw
from vllm_hook_plugins import run_utils as ru

H, Kd = 8, 6  # hidden, kv_hidden (small, arbitrary)


def _compact_entry(layer_num, seed):
    """An all_tokens entry as flush_disk writes it: per-step q chunks + compact k."""
    g = torch.Generator().manual_seed(seed)
    # request 0: prefill 4 + 3 decode; request 1: prefill 3 + 2 decode
    kf0 = torch.randn(7, Kd, generator=g)
    kf1 = torch.randn(5, Kd, generator=g)
    q = [torch.randn(n, H, generator=g) for n in (4, 1, 1, 1, 3, 1, 1)]
    return {
        "q": q,
        "k_full": [kf0, kf1],
        "k_prefix_ends": [[4, 5, 6, 7], [3, 4, 5]],
        "layer_num": layer_num,
        "hookq_mode": "all_tokens",
    }


def _expected(entry):
    q = [t.clone() for t in entry["q"]]
    k_all = []
    for full, ends in zip(entry["k_full"], entry["k_prefix_ends"]):
        k_all.extend(full[:L].clone() for L in ends)
    return q, k_all


def _save_load(cpu_cache, default_mode):
    with tempfile.TemporaryDirectory() as d:
        run_dir = os.path.join(d, "tp_rank_0")
        os.makedirs(run_dir)
        aw.save_qk_cache_safetensors(cpu_cache, run_dir, default_mode, 0)
        st = glob.glob(os.path.join(run_dir, "qk.safetensors"))
        assert st, "no qk.safetensors written (unexpected .pt fallback)"
        from safetensors import safe_open
        with safe_open(st[0], framework="pt") as sf:
            k_shapes = {k: tuple(sf.get_slice(k).get_shape())
                        for k in sf.keys() if k.endswith("__k")}
        loaded = ru._load_and_merge_qk_safetensors("/x", "/y", st)
        return loaded, k_shapes


def test_compact_all_tokens_byte_identical():
    cache = {"config": {"model": "x"}, "qk_cache": {
        "model.layers.0.self_attn": _compact_entry(0, 1),
        "model.layers.3.self_attn": _compact_entry(3, 2),
    }}
    expected = {m: _expected(e) for m, e in cache["qk_cache"].items()}
    loaded, _ = _save_load(cache, "all_tokens")

    qk = loaded["qk_cache"]
    for mod, (eq, ek) in expected.items():
        got = qk[mod]
        assert len(got["q"]) == len(eq)
        assert len(got["k_all"]) == len(ek)
        for a, b in zip(got["q"], eq):
            assert torch.equal(a, b)
        for a, b in zip(got["k_all"], ek):
            assert torch.equal(a, b)


def test_compact_all_tokens_is_O_seq_on_disk():
    """The stored k tensor holds the unique keys (Sum k_full rows), not the padded
    growing prefixes (Sum prefix lengths, padded to max) — i.e. NOT O(seq^2)."""
    entry = _compact_entry(0, 1)
    cache = {"config": {"model": "x"}, "qk_cache": {"model.layers.0.self_attn": entry}}
    _, k_shapes = _save_load(cache, "all_tokens")

    stored_numel = math.prod(next(iter(k_shapes.values())))
    compact_numel = sum(t.shape[0] for t in entry["k_full"]) * Kd
    padded_numel = sum(L for ends in entry["k_prefix_ends"] for L in ends) * Kd
    assert stored_numel == compact_numel
    assert stored_numel < padded_numel  # strictly smaller than the growing-prefix content


def test_last_token_legacy_path_still_round_trips():
    g = torch.Generator().manual_seed(7)
    entry = {
        "q": [torch.randn(H, generator=g), torch.randn(H, generator=g)],  # 1D per request
        "k_all": [torch.randn(6, Kd, generator=g), torch.randn(4, Kd, generator=g)],
        "layer_num": 0,
        "hookq_mode": "last_token",
    }
    cache = {"config": {"model": "x"}, "qk_cache": {"model.layers.0.self_attn": entry}}
    eq = [t.clone() for t in entry["q"]]
    ek = [t.clone() for t in entry["k_all"]]
    loaded, _ = _save_load(cache, "last_token")

    got = loaded["qk_cache"]["model.layers.0.self_attn"]
    assert len(got["q"]) == 2 and len(got["k_all"]) == 2
    for a, b in zip(got["q"], eq):
        assert torch.equal(a, b)
    for a, b in zip(got["k_all"], ek):
        assert torch.equal(a, b)
