import pytest
import torch
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex, assemble_qk

def test_demux_assemble_and_free():
    idx = PerRequestIndex()
    a0 = torch.arange(6).reshape(3,2).float(); a1 = torch.arange(6,10).reshape(2,2).float()
    idx.note_rows("r0", 0, a0);  idx.note_rows("r0", 0, a1)   # two steps, same (req,layer) -> concat in order
    idx.note_rows("r1", 0, torch.zeros(1,2))
    assert idx.pop_deliverable() == []                        # nothing finished yet
    idx.mark_finished("r0")
    got = idx.pop_deliverable()
    assert len(got) == 1 and got[0][0] == "r0"
    assert torch.equal(got[0][1][0], torch.cat([a0, a1], 0))  # layer 0 assembled in step order
    idx.free("r0")
    assert "r0" not in idx.live_req_ids() and "r1" in idx.live_req_ids()  # r1 still live, not delivered

def test_two_requests_no_cross_leak():
    idx = PerRequestIndex()
    idx.note_rows("a", 0, torch.ones(1,2)); idx.note_rows("b", 0, torch.full((1,2), 2.0))
    idx.mark_finished("a"); idx.mark_finished("b")
    by = {r: d for r, d in idx.pop_deliverable()}
    assert torch.equal(by["a"][0], torch.ones(1,2)) and torch.equal(by["b"][0], torch.full((1,2),2.0))


# --- Task 3: QK q + k_all assembly ------------------------------------------------------------
#
# Contract under test (see per_request_delivery.py::assemble_qk docstring for the full writeup):
#   1. q/k STAGING: a (req, layer) stages two independent row streams as DISTINCT layer keys on the
#      same entry -- ("q", layer) and ("k", layer) -- via the UNCHANGED note_rows. k is noted every
#      step (growing key history); q only on emit_q steps.
#   2. prefix_ends ACCUMULATION: the caller passes kmeta={"prefix_ends": [...]} as the FULL
#      CUMULATIVE list on every k-stream note_rows call that has a new prefix boundary (kmeta=None on
#      steps with none, e.g. a non-final prefill chunk). note_rows's existing LAST-WRITE-WINS kmeta
#      store means the final call's list is what survives -- no accumulation logic needed downstream.
#
# Scenario below mirrors a real QK capture: 4 steps on one (req, layer) --
#   step0: prefill chunk 1 (not final) -- k only, 2 rows, kmeta=None (no boundary yet)
#   step1: prefill chunk 2 (final)     -- k 1 row (running total 3) + q 1 row, prefix_ends=[3]
#   step2: decode                      -- k 1 row (running total 4) + q 1 row, prefix_ends=[3,4]
#   step3: decode                      -- k 1 row (running total 5) + q 1 row, prefix_ends=[3,4,5]
# k_full = cat of all 4 k blocks (5 rows); q = cat of the 3 q blocks (3 rows);
# k_all = [k_full[:3], k_full[:4], k_full[:5]].

def _build_qk_steps():
    k0 = torch.arange(0, 4).reshape(2, 2).float()     # rows 0,1
    k1 = torch.arange(4, 6).reshape(1, 2).float()      # row 2
    k2 = torch.arange(6, 8).reshape(1, 2).float()      # row 3
    k3 = torch.arange(8, 10).reshape(1, 2).float()     # row 4
    q1 = torch.full((1, 2), 100.0)
    q2 = torch.full((1, 2), 200.0)
    q3 = torch.full((1, 2), 300.0)
    return k0, k1, k2, k3, q1, q2, q3


def _expected_qk():
    k0, k1, k2, k3, q1, q2, q3 = _build_qk_steps()
    k_full = torch.cat([k0, k1, k2, k3], 0)
    q = torch.cat([q1, q2, q3], 0)
    k_all = [k_full[:3], k_full[:4], k_full[:5]]
    return q, k_all


def test_assemble_qk_direct_multistep():
    k0, k1, k2, k3, q1, q2, q3 = _build_qk_steps()
    layer = 5
    entry = {
        "layers": {
            ("k", layer): [k0, k1, k2, k3],
            ("q", layer): [q1, q2, q3],
        },
        "finished": True,
        "kmeta": {("k", layer): {"prefix_ends": [3, 4, 5]}},
    }
    out = assemble_qk(entry)
    exp_q, exp_k_all = _expected_qk()
    assert set(out.keys()) == {layer}
    assert torch.equal(out[layer]["q"], exp_q)
    assert len(out[layer]["k_all"]) == len(exp_k_all)
    for got, exp in zip(out[layer]["k_all"], exp_k_all):
        assert torch.equal(got, exp)


def test_pop_deliverable_qk_via_index_multistep():
    k0, k1, k2, k3, q1, q2, q3 = _build_qk_steps()
    layer = 5
    idx = PerRequestIndex()
    # step0: prefill chunk 1, k only, no prefix boundary yet
    idx.note_rows("r0", ("k", layer), k0)
    # step1: prefill final chunk -- k + q + first prefix boundary (cumulative list so far)
    idx.note_rows("r0", ("k", layer), k1, kmeta={"prefix_ends": [3]})
    idx.note_rows("r0", ("q", layer), q1)
    # step2: decode -- k + q + cumulative prefix_ends (last-write-wins overwrites step1's kmeta)
    idx.note_rows("r0", ("k", layer), k2, kmeta={"prefix_ends": [3, 4]})
    idx.note_rows("r0", ("q", layer), q2)
    # step3: decode -- k + q + full cumulative prefix_ends
    idx.note_rows("r0", ("k", layer), k3, kmeta={"prefix_ends": [3, 4, 5]})
    idx.note_rows("r0", ("q", layer), q3)

    assert idx.pop_deliverable_qk() == []          # nothing finished yet
    idx.mark_finished("r0")
    got = idx.pop_deliverable_qk()
    assert len(got) == 1 and got[0][0] == "r0"
    assembled = got[0][1]
    exp_q, exp_k_all = _expected_qk()
    assert torch.equal(assembled[layer]["q"], exp_q)
    for got_row, exp_row in zip(assembled[layer]["k_all"], exp_k_all):
        assert torch.equal(got_row, exp_row)
    idx.free("r0")
    assert "r0" not in idx.live_req_ids()


# --- Code-review fix: GQA fail-loud on k-rows-but-zero-q-rows ---------------------------------
#
# assemble_qk used to fabricate a 0-row q tensor shaped like a k row when a layer had k rows but
# no q rows at all. Under GQA (q_dim != k_dim, true for every target model here) that width is
# simply wrong, and the case is a genuine anomaly anyway -- a captured layer always emits at least
# one q row (all_tokens: every step; last_token: final prefill chunk + every decode step). This
# must now raise ValueError, mirroring the existing k-required guard in the same function.

def test_assemble_qk_raises_on_zero_q_rows():
    layer = 5
    entry = {
        "layers": {
            ("k", layer): [torch.arange(0, 4).reshape(2, 2).float()],
            # no ("q", layer) key at all -- zero q rows staged
        },
        "finished": True,
        "kmeta": {("k", layer): {"prefix_ends": [2]}},
    }
    with pytest.raises(ValueError):
        assemble_qk(entry)


def test_pop_deliverable_qk_isolates_zero_q_poison_entry():
    """assemble_qk still fail-loud-raises on a poison (k-rows, zero-q) entry (above), but that raise
    must NOT escape pop_deliverable_qk -- else it leaves _deliverable uncleared and re-raises on every
    later pop == a permanent wedge, and the poison entry is never freed. pop_deliverable_qk now
    ISOLATES the poison entry: it is DROPPED + FREED (residency falls), _deliverable is cleared, and a
    later pop returns empty rather than re-raising. (Updated from the old raises-behavior test.)"""
    layer = 5
    idx = PerRequestIndex()
    idx.note_rows("r0", ("k", layer), torch.arange(0, 4).reshape(2, 2).float(),
                  kmeta={"prefix_ends": [2]})
    # deliberately never call note_rows("r0", ("q", layer), ...)
    idx.mark_finished("r0")
    got = idx.pop_deliverable_qk()            # must NOT raise now
    assert got == []                          # sole poison entry -> nothing delivered
    assert "r0" not in idx.live_req_ids()     # dropped + FREED so residency falls
    assert idx.pop_deliverable_qk() == []     # _deliverable cleared -> never re-poisons a later pop
