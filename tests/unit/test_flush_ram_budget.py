"""No-GPU tests for the surviving RAM-budget helpers (`_tree_bytes`, `_mem_limit_bytes`,
`_resolve_flush_budget`) in `graph/writer_process.py`.

The off-engine flush-dispatch BYTE-BUDGET QUEUE these helpers used to feed
(`_ByteBudgetQueue`/`init_flush_dispatch`/`_flush_dispatch_loop`/`_drain_flush_dispatch`) was
removed when Task A6 replaced it with the off-loop ring-drain consumer's built-in backpressure --
the RAM bound is now enforced by that consumer (`graph/gpu_capture_ring.py`,
`graph/ring_drain_hs.py`), not a queue in front of the writer process.
The three tree/size helpers below are unrelated to that queue (pure byte-accounting utilities)
and are kept; their dedicated tests survive here.
"""
from vllm_hook_plugins.graph import writer_process as wp


class _FakeT:
    """Duck-typed tensor: _tree_bytes sums element_size()*numel()."""
    def __init__(self, nbytes):
        self._n = int(nbytes)

    def element_size(self):
        return 1

    def numel(self):
        return self._n


def test_tree_bytes_sums_tensor_leaves_only():
    assert wp._tree_bytes({"a": [_FakeT(100), _FakeT(50)], "cfg": "not-a-tensor", "n": 7}) == 150


def test_budget_resolution_absolute_env(monkeypatch):
    monkeypatch.setenv("VLLM_HOOK_FLUSH_RAM_BUDGET_BYTES", str(3 << 30))
    assert wp._resolve_flush_budget() == (3 << 30)


def test_budget_resolution_fraction(monkeypatch):
    monkeypatch.delenv("VLLM_HOOK_FLUSH_RAM_BUDGET_BYTES", raising=False)
    monkeypatch.setenv("VLLM_HOOK_FLUSH_RAM_BUDGET_FRAC", "0.1")
    monkeypatch.setattr(wp, "_mem_limit_bytes", lambda: 100 << 30)
    assert wp._resolve_flush_budget() == int(0.1 * (100 << 30))
