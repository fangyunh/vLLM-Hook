"""No-GPU BIT-IDENTITY gate for the decode-cache HS routing build (VLLM_HOOK_ROUTE_DECODE_CACHE).

Drives a MULTI-STEP request lifecycle (admit -> prefill -> decode x k -> finish + condense -> admit)
through the DEFAULT path (_build_routing_hs, vectorized=False) and the CACHE path
(_build_routing_hs_decode_cache), each on its OWN persistent registry+ring, and asserts EXACT equality
at EVERY step of the plane, _hs_step_entries (+ their expand_records), _hs_step_start/_rows, plans, and
the ring write cursor. The cache path must never diverge — same reserve order => same slots => byte-id.

Run: conda activate vllm_hook_env && pytest tests/unit/test_route_decode_cache_parity.py -q
"""
import numpy as np
import torch

from vllm_hook_plugins.graph.gpu_capture_ring import GpuCaptureRing
from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.ring_metadata import ReqCaptureRecord, expand_records
from vllm_hook_plugins.graph.install_hs import (
    _build_routing_hs, _build_routing_hs_decode_cache)


class _SP:
    def __init__(self, extra): self.extra_args = extra


class _RS:
    def __init__(self, extra, output_token_ids=()):
        self.sampling_params = _SP(extra); self.output_token_ids = output_token_ids


class _IB:
    def __init__(self, req_ids): self.req_ids = req_ids


class _MR:
    def __init__(self, req_ids, requests, default_hooks_on="both", worker_hs_mode="all_tokens"):
        self.input_batch = _IB(req_ids); self.requests = requests
        self._default_hooks_on = default_hooks_on; self._worker_hs_mode = worker_hs_mode


def _make_registry(num_layers, cap, R):
    reg = HostRegistry(num_layers=num_layers, cap=cap, device="cpu", should_capture=True)
    ring = GpuCaptureRing(row_bytes=8, n_slots=R, device="cpu",
                          dtype=torch.float32, row_shape=(4,))
    reg._hs_ring = ring
    reg._hs_step_entries = []
    reg.sentinel_row = ring.SENTINEL
    reg.incremental_enabled = False
    reg.gpu_routing = False
    reg.capture_index_all.fill_(ring.SENTINEL)
    for slot in reg._ring.slots:
        slot["capture_index"].fill_(ring.SENTINEL)
    return reg, ring


def _qsl(spans):
    q = [0]
    for s in spans:
        q.append(q[-1] + s)
    return q


def _snapshot(reg, ring, plans):
    return {
        "plane": reg.capture_index_pinned.clone(),
        "entries": list(reg._hs_step_entries),
        "step_start": reg._hs_step_start,
        "step_rows": reg._hs_step_rows,
        "plans": plans,
        "ring_write": ring._write,
    }


def _run_step(reg, ring, batch, build_fn):
    """batch = {'ids':[...], 'extra':{id:extra}, 'otids':{id:tuple}, 'spans':[...]}."""
    reqs = {i: _RS(batch["extra"][i], batch["otids"].get(i, ())) for i in batch["ids"]}
    mr = _MR(batch["ids"], reqs)
    reg.reset_pinned(reg.cap)                       # what the routing wrapper does each step
    plans = build_fn(mr, reg, _qsl(batch["spans"]))
    return _snapshot(reg, ring, plans)


def _assert_step(sa, sb, tag):
    assert torch.equal(sa["plane"], sb["plane"]), f"{tag}: plane differs\n{sa['plane']}\n{sb['plane']}"
    assert sa["entries"] == sb["entries"], f"{tag}: entries differ\n{sa['entries']}\n{sb['entries']}"
    assert sa["step_start"] == sb["step_start"], f"{tag}: step_start differs"
    assert sa["step_rows"] == sb["step_rows"], f"{tag}: step_rows differs"
    assert sa["plans"] == sb["plans"], f"{tag}: plans differ\n{sa['plans']}\n{sb['plans']}"
    assert sa["ring_write"] == sb["ring_write"], f"{tag}: ring cursor differs"
    assert expand_records(sa["entries"]) == expand_records(sb["entries"]), f"{tag}: expand differs"


# all-32-of-4-layers, all_tokens, hooks_on=both -> both prefill & decode capture (the §7.1 shape)
_EX_ALL = {"output_hidden_states": [1, 2, 3, 4], "hs_mode": "all_tokens", "hooks_on": "both"}
_EX_LST = {"output_hidden_states": [2, 3], "hs_mode": "last_token", "hooks_on": "decode"}
_EX_PRE = {"output_hidden_states": [1, 2, 3, 4], "hs_mode": "all_tokens", "hooks_on": "prefill"}


def _lifecycle_steps():
    """admit r0,r1 (prefill) -> both decode x2 -> r1 finishes (condense, r0 column shifts is n/a here
    since r0 stays index 0) -> admit r2 (prefill) while r0 decodes -> r0,r2 decode."""
    return [
        {"ids": ["r0", "r1"], "extra": {"r0": _EX_ALL, "r1": _EX_ALL},
         "otids": {}, "spans": [6, 4]},                                   # prefill both
        {"ids": ["r0", "r1"], "extra": {"r0": _EX_ALL, "r1": _EX_ALL},
         "otids": {"r0": (1,), "r1": (1,)}, "spans": [1, 1]},             # decode both
        {"ids": ["r0", "r1"], "extra": {"r0": _EX_ALL, "r1": _EX_ALL},
         "otids": {"r0": (1, 2), "r1": (1, 2)}, "spans": [1, 1]},         # decode both
        {"ids": ["r0"], "extra": {"r0": _EX_ALL},
         "otids": {"r0": (1, 2, 3)}, "spans": [1]},                       # r1 finished
        {"ids": ["r0", "r2"], "extra": {"r0": _EX_ALL, "r2": _EX_ALL},
         "otids": {"r0": (1, 2, 3, 4)}, "spans": [1, 5]},                 # admit r2 (prefill) + r0 decode
        {"ids": ["r0", "r2"], "extra": {"r0": _EX_ALL, "r2": _EX_ALL},
         "otids": {"r0": (1, 2, 3, 4, 5), "r2": (1,)}, "spans": [1, 1]},  # both decode
    ]


def test_decode_cache_lifecycle_byte_identical():
    rega, ringa = _make_registry(num_layers=4, cap=16, R=4096)
    regb, ringb = _make_registry(num_layers=4, cap=16, R=4096)
    for k, batch in enumerate(_lifecycle_steps()):
        sa = _run_step(rega, ringa, batch, lambda m, r, q: _build_routing_hs(m, r, q, vectorized=False))
        sb = _run_step(regb, ringb, batch, _build_routing_hs_decode_cache)
        _assert_step(sa, sb, f"step{k}")


def test_decode_cache_last_token_and_condensation():
    """last_token + a batch that condenses (r0 finishes, r1 moves to column 0)."""
    rega, ringa = _make_registry(num_layers=4, cap=16, R=4096)
    regb, ringb = _make_registry(num_layers=4, cap=16, R=4096)
    steps = [
        {"ids": ["r0", "r1"], "extra": {"r0": _EX_LST, "r1": _EX_LST},
         "otids": {"r0": (1,), "r1": (1,)}, "spans": [1, 1]},
        {"ids": ["r1"], "extra": {"r1": _EX_LST},
         "otids": {"r1": (1, 2)}, "spans": [1]},          # r0 finished -> r1 at column 0
    ]
    for k, batch in enumerate(steps):
        sa = _run_step(rega, ringa, batch, lambda m, r, q: _build_routing_hs(m, r, q, vectorized=False))
        sb = _run_step(regb, ringb, batch, _build_routing_hs_decode_cache)
        _assert_step(sa, sb, f"cond{k}")


def test_decode_cache_prefill_only_lifecycle_byte_identical():
    """hooks_on='prefill' (the capture DEFAULT): the request captures on PREFILL and must capture
    NOTHING on its decode steps. The cache path must NOT fast-fire on decode -> byte-identical to the
    default path (which skips via `if hooks_on=='prefill' and not is_prefill: continue`). This is the
    RED guard for the fold-in fix: WITHOUT the Task-3 caching guard the cache path fast-fires on the
    decode steps (plane/plans/cursor all diverge) so this test FAILS; WITH the guard it PASSES."""
    rega, ringa = _make_registry(num_layers=4, cap=16, R=4096)
    regb, ringb = _make_registry(num_layers=4, cap=16, R=4096)
    steps = [
        {"ids": ["r0"], "extra": {"r0": _EX_PRE},
         "otids": {}, "spans": [5]},                    # prefill -> captures (non-vacuous)
        {"ids": ["r0"], "extra": {"r0": _EX_PRE},
         "otids": {"r0": (1,)}, "spans": [1]},           # decode  -> must capture NOTHING
        {"ids": ["r0"], "extra": {"r0": _EX_PRE},
         "otids": {"r0": (1, 2)}, "spans": [1]},         # decode  -> must capture NOTHING
    ]
    for k, batch in enumerate(steps):
        sa = _run_step(rega, ringa, batch, lambda m, r, q: _build_routing_hs(m, r, q, vectorized=False))
        sb = _run_step(regb, ringb, batch, _build_routing_hs_decode_cache)
        _assert_step(sa, sb, f"pre{k}")
