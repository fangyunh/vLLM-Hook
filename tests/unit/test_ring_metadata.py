# tests/unit/test_ring_metadata.py
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, StepMeta, write_sidecar, read_sidecar

def test_sidecar_roundtrip(tmp_path):
    steps = [StepMeta([LayerEntry("r0", 0, 0, 5, "all_tokens"),
                       LayerEntry("r0", 1, 5, 5, "all_tokens")]),
             StepMeta([LayerEntry("r1", 0, 10, 1, "last_token")])]
    header = {"dtype": "bfloat16", "row_shape": [4096], "hidden": 4096}
    p = tmp_path / "meta.jsonl"
    write_sidecar(str(p), steps, header)
    h, got = read_sidecar(str(p))
    assert h == header
    flat = [(e.req_id, e.layer, e.logical_start, e.n_rows, e.hs_mode)
            for s in got for e in s.entries]
    assert flat == [("r0",0,0,5,"all_tokens"),("r0",1,5,5,"all_tokens"),("r1",0,10,1,"last_token")]


def test_empty_middle_step_elided_flat_order_preserved(tmp_path):
    # Middle StepMeta has zero entries — it must be elided from `read_sidecar`'s returned steps,
    # not preserved as an empty placeholder; the flat, logical_start-ordered entry list must survive
    # intact regardless.
    steps = [StepMeta([LayerEntry("r0", 0, 0, 5, "all_tokens")]),
              StepMeta([]),
              StepMeta([LayerEntry("r1", 0, 5, 1, "last_token")])]
    header = {"dtype": "bfloat16", "row_shape": [4096], "hidden": 4096}
    p = tmp_path / "meta_empty_step.jsonl"
    write_sidecar(str(p), steps, header)
    h, got = read_sidecar(str(p))
    assert h == header
    flat = [(e.req_id, e.layer, e.logical_start, e.n_rows, e.hs_mode)
            for s in got for e in s.entries]
    assert flat == [("r0", 0, 0, 5, "all_tokens"), ("r1", 0, 5, 1, "last_token")]
    # The empty middle step is elided entirely — only the two non-empty steps come back.
    assert len(got) == 2


def test_header_tuple_value_normalized_to_list(tmp_path):
    # A caller may pass a tuple (e.g. row_shape=(4096,)); JSON has no tuple type, so write_sidecar
    # must normalize it to a list so the round-trip is deterministic.
    header = {"row_shape": (4096,), "hidden": 4096}
    p = tmp_path / "meta_tuple_header.jsonl"
    write_sidecar(str(p), [], header)
    h, got = read_sidecar(str(p))
    assert h["row_shape"] == [4096]
    assert isinstance(h["row_shape"], list)
    assert h["hidden"] == 4096
    assert got == []
