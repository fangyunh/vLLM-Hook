"""No-GPU test for POISON-ENTRY ISOLATION in PerRequestIndex.pop_deliverable_qk (Task 13 final-review I2).

The wedge (pre-fix): ``assemble_qk`` fail-loud-raises ``ValueError`` for a layer with k rows but zero q
rows -- a genuine anomaly reachable when a ``last_token`` request is marked finished mid-prefill (e.g.
via ``finalize_all`` at shutdown). ``pop_deliverable_qk`` set ``self._deliverable = []`` only AFTER its
assembly loop, so one poison entry raised OUT of the pop BEFORE the clear ran. Every later pop then
re-processed the same poison entry and re-raised == a PERMANENT WEDGE of QK RPC delivery, and the poison
entry was never freed so residency never dropped.

THE FIX: ``pop_deliverable_qk`` isolates per entry -- a failing ``assemble_qk`` is caught, logged loud,
and that ONE entry is DROPPED + FREED (residency falls, it can never re-poison a later pop); the good
entries still deliver, and ``_deliverable`` is ALWAYS cleared. ``assemble_qk``'s validation is unchanged
(it still raises); the raise just no longer escapes the pop.

RED-then-GREEN: this file replicates the exact PRE-FIX loop inline (``_prefix_pop``) and shows it raises
+ wedges (``_deliverable`` left populated, poison entry never freed), then drives the REAL (fixed)
``pop_deliverable_qk`` on an identical index and shows it isolates.

Run:  conda activate vllm_hook_env && python -m pytest tests/unit/test_qk_poison_entry_isolation.py -q
"""
import pytest
import torch

from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex, assemble_qk


_LAYER = 5


def _stage_good(idx, rid, base):
    """Stage a well-formed single-step QK request (k + q on one layer, one prefix boundary)."""
    k = (torch.arange(0, 4).reshape(2, 2).float() + base)
    q = (torch.full((1, 2), 100.0) + base)
    idx.note_rows(rid, ("k", _LAYER), k, kmeta={"prefix_ends": [2]})
    idx.note_rows(rid, ("q", _LAYER), q)
    idx.mark_finished(rid)
    return q, [torch.cat([k], 0)[:2]]   # (expected q, expected k_all)


def _stage_poison(idx, rid):
    """Stage a POISON request: k rows but NEVER any q rows (the last_token-finished-mid-prefill anomaly)."""
    idx.note_rows(rid, ("k", _LAYER), torch.arange(0, 4).reshape(2, 2).float(),
                  kmeta={"prefix_ends": [2]})
    idx.mark_finished(rid)


def _build_index_with_poison_among_good():
    """One index: good g0, POISON p, good g1 -- all finished, in _deliverable order [g0, p, g1]."""
    idx = PerRequestIndex()
    exp = {}
    exp["g0"] = _stage_good(idx, "g0", base=0.0)
    _stage_poison(idx, "p")
    exp["g1"] = _stage_good(idx, "g1", base=1000.0)
    assert idx._deliverable == ["g0", "p", "g1"]
    return idx, exp


def _prefix_pop(idx):
    """The EXACT PRE-FIX pop_deliverable_qk body: assemble each, clear _deliverable AFTER the loop.
    A poison entry raises out of the loop BEFORE the clear -> _deliverable stays populated (the wedge)."""
    out = []
    for req_id in idx._deliverable:
        out.append((req_id, assemble_qk(idx._entries[req_id])))
    idx._deliverable = []
    return out


def test_pre_fix_pop_raises_and_wedges_RED():
    """RED: the pre-fix loop raises on the poison entry, delivers NONE of the good ones, and leaves
    _deliverable populated (so a retry re-raises forever) with the poison entry never freed."""
    idx, _ = _build_index_with_poison_among_good()
    with pytest.raises(ValueError):
        _prefix_pop(idx)
    # The wedge signature: _deliverable never cleared -> a second call re-raises on the same poison.
    assert idx._deliverable == ["g0", "p", "g1"], "pre-fix leaves _deliverable populated (permanent wedge)"
    with pytest.raises(ValueError):
        _prefix_pop(idx)
    assert "p" in idx.live_req_ids(), "pre-fix never frees the poison entry (residency never drops)"


def test_pop_deliverable_qk_isolates_poison_delivers_the_rest_GREEN():
    """GREEN: the real (fixed) pop_deliverable_qk isolates the poison entry -- drops + frees it, delivers
    both good requests, and clears _deliverable (a second pop is empty, residency dropped)."""
    idx, exp = _build_index_with_poison_among_good()

    got = dict(idx.pop_deliverable_qk())        # must NOT raise
    assert set(got) == {"g0", "g1"}, f"good requests must still deliver: got {set(got)}"
    for rid, (exp_q, exp_k_all) in exp.items():
        assert torch.equal(got[rid][_LAYER]["q"], exp_q), f"{rid} q mismatch"
        assert len(got[rid][_LAYER]["k_all"]) == len(exp_k_all)
        for a, b in zip(got[rid][_LAYER]["k_all"], exp_k_all):
            assert torch.equal(a, b), f"{rid} k_all mismatch"

    # The poison entry was dropped + FREED (residency falls); the good ones were freed by the drain
    # caller normally, but pop itself only frees the poison -- so g0/g1 remain live until freed here.
    assert "p" not in idx.live_req_ids(), "poison entry must be dropped + freed (residency drops)"

    # _deliverable was cleared even though an entry raised -> no re-poison on a later pop.
    assert idx._deliverable == [], "_deliverable must be cleared"
    assert idx.pop_deliverable_qk() == [], "a second pop must be empty (never re-processes the poison)"


def test_all_poison_pop_returns_empty_and_frees_all():
    """Degenerate: every finished entry is poison -> pop returns [], all are freed, _deliverable cleared."""
    idx = PerRequestIndex()
    for rid in ("a", "b", "c"):
        _stage_poison(idx, rid)
    assert idx.pop_deliverable_qk() == []
    assert idx.live_req_ids() == set(), "all poison entries must be freed"
    assert idx._deliverable == []


if __name__ == "__main__":
    test_pre_fix_pop_raises_and_wedges_RED()
    print("PASS  test_pre_fix_pop_raises_and_wedges_RED")
    test_pop_deliverable_qk_isolates_poison_delivers_the_rest_GREEN()
    print("PASS  test_pop_deliverable_qk_isolates_poison_delivers_the_rest_GREEN")
    test_all_poison_pop_returns_empty_and_frees_all()
    print("PASS  test_all_poison_pop_returns_empty_and_frees_all")
    print("VERDICT: PASS (3/3)")
