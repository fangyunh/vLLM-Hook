"""No-GPU marshal-shape + delivery-contract test for Task 7 (production RPC-raw per-request HS
delivery on finish).

Exercises the REAL worker marshal (``ProbeHiddenStatesWorker.get_ring_per_request`` ->
``_marshal_perreq_hs`` compress+pickle) and the REAL driver deserialize
(``_hook_plugin._decompress``) end-to-end with a fake ``PerRequestIndex`` -- NO engine boot, NO GPU.

Load-bearing lesson under test (Task 6): ``collective_rpc`` does NOT round-trip raw torch tensors
(they arrive on the driver as plain lists), so the worker MUST serialize to bytes. Here the worker
returns bytes, ``_decompress`` unpickles them, and the reconstructed ``{layer_num: tensor}`` is
``torch.equal`` to the input; residency drops to 0 as each delivered request is freed.
"""
import torch

from vllm_hook_plugins._hook_plugin import _decompress
from vllm_hook_plugins.graph.per_request_delivery import PerRequestIndex
from vllm_hook_plugins.workers.probe_hidden_states_worker import ProbeHiddenStatesWorker


class _FakeDrain:
    """Stands in for the OffLoopRingDrain the worker reads via ``_hs_drain`` -- only the two
    attributes ``get_ring_per_request`` touches (``per_request`` gate + the ``PerRequestIndex``)."""

    def __init__(self, index, per_request=True):
        self.per_request = per_request
        self.index = index


def _worker(index=None, per_request=True):
    w = ProbeHiddenStatesWorker()
    w._conf = {"hidden_size": 4, "num_layers": 3}
    if index is not None:
        w._hs_drain = _FakeDrain(index, per_request)
    return w


def _finished_index(reqs):
    """A PerRequestIndex with each req_id noted (per-step rows per layer) and marked finished.
    ``reqs`` = ``{req_id: {layer_num: [row_block, ...]}}``."""
    idx = PerRequestIndex()
    for rid, layers in reqs.items():
        for layer, blocks in layers.items():
            for b in blocks:
                idx.note_rows(rid, layer, b)
        idx.mark_finished(rid)
    return idx


def test_marshal_shape_roundtrip_matches_eager_hs_cache():
    """GREEN: worker get_ring_per_request -> bytes -> driver _decompress yields the eager-shaped
    ``{"hs_cache": {layer_num: {"hidden_states": tensor, "layer_num": int}}, "config": ...}`` with
    tensors torch.equal to the assembled (step-concatenated) rows, and index residency -> 0."""
    l1a = torch.arange(8).reshape(2, 4).float()
    l1b = torch.arange(8, 12).reshape(1, 4).float()   # layer 1: two steps -> cat = 3 rows
    l2 = torch.arange(20, 28).reshape(2, 4).float()    # layer 2: one step
    idx = _finished_index({"r0": {1: [l1a, l1b], 2: [l2]}})
    w = _worker(idx)

    blob = w.get_ring_per_request("r0")
    assert isinstance(blob, (bytes, bytearray)), "worker must return bytes, not raw tensors"

    payload = _decompress(blob)
    assert set(payload.keys()) == {"hs_cache", "config"}
    assert payload["config"] == {"hidden_size": 4, "num_layers": 3}
    hs = payload["hs_cache"]
    assert set(hs.keys()) == {1, 2}
    # value shape mirrors the eager path: {"hidden_states": tensor, "layer_num": int}
    assert hs[1]["layer_num"] == 1 and hs[2]["layer_num"] == 2
    assert torch.equal(hs[1]["hidden_states"], torch.cat([l1a, l1b], 0))
    assert torch.equal(hs[2]["hidden_states"], l2)

    # residency gate: r0 delivered + freed -> nothing left in the index.
    assert idx.live_req_ids() == set()


def test_bulk_pop_into_stash_serves_each_request_once():
    """pop_deliverable is BULK: asking for one finished request drains BOTH into the stash and
    frees them from the index (residency 0 immediately); the un-asked one is served from the stash
    on its own later call, exactly once."""
    a = torch.ones(1, 4)
    b = torch.full((1, 4), 2.0)
    idx = _finished_index({"ra": {1: [a]}, "rb": {1: [b]}})
    w = _worker(idx)

    blob_a = w.get_ring_per_request("ra")
    assert idx.live_req_ids() == set()                       # both drained+freed on the first call
    assert torch.equal(_decompress(blob_a)["hs_cache"][1]["hidden_states"], a)

    blob_b = w.get_ring_per_request("rb")                    # served from the stash (index empty)
    assert torch.equal(_decompress(blob_b)["hs_cache"][1]["hidden_states"], b)

    assert w.get_ring_per_request("ra") is None              # delivered once -> gone
    assert w.get_ring_per_request("rb") is None


def test_not_yet_delivered_returns_none_and_keeps_residency():
    """A request still in flight (noted but NOT finished) is not deliverable: get_ring_per_request
    returns None and the request stays resident in the index (blocking is Task 12, not here)."""
    idx = PerRequestIndex()
    idx.note_rows("r0", 1, torch.ones(1, 4))                 # noted, never mark_finished
    w = _worker(idx)
    assert w.get_ring_per_request("r0") is None
    assert idx.live_req_ids() == {"r0"}                      # still resident, not freed


def test_suffixed_internal_req_id_matches():
    """vLLM may append '-<suffix>' to the external id; the stash lookup matches it (same rule as
    get_captured_states / iter_matching_req_ids)."""
    t = torch.arange(4).reshape(1, 4).float()
    idx = _finished_index({"ext-abc123": {2: [t]}})
    w = _worker(idx)
    blob = w.get_ring_per_request("ext")
    assert blob is not None
    assert torch.equal(_decompress(blob)["hs_cache"][2]["hidden_states"], t)


def test_noop_when_ring_path_not_installed():
    """Strict no-op -> None (never perturbs bank/shared/QK/eager) when the drain is absent or
    per_request mode is off."""
    assert _worker(index=None).get_ring_per_request("r0") is None            # no _hs_drain
    idx = _finished_index({"r0": {1: [torch.ones(1, 4)]}})
    assert _worker(idx, per_request=False).get_ring_per_request("r0") is None  # per_request off
