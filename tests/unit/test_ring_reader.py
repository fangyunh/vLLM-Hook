import torch
from vllm_hook_plugins.graph.ring_metadata import LayerEntry, StepMeta, write_sidecar
from vllm_hook_plugins.graph.ring_reader import load_ring_artifact

def test_reconstruction_byte_identical(tmp_path):
    hidden = 4
    # 3 rows for (r0,layer0) all_tokens, then 1 row for (r0,layer1) last_token
    rows = torch.arange(4*4, dtype=torch.float32).reshape(4, hidden)  # 4 rows total
    raw = tmp_path / "raw.bin"; raw.write_bytes(rows.contiguous().numpy().tobytes())
    steps = [StepMeta([LayerEntry("r0", 0, 0, 3, "all_tokens"),
                       LayerEntry("r0", 1, 3, 1, "last_token")])]
    write_sidecar(str(tmp_path/"m.jsonl"), steps, {"dtype":"float32","row_shape":[hidden],"hidden":hidden})
    out = load_ring_artifact(str(raw), str(tmp_path/"m.jsonl"))
    assert torch.equal(out["r0"][0], rows[0:3])
    assert torch.equal(out["r0"][1], rows[3:4])


def test_multi_block_per_key_concatenated_in_logical_order(tmp_path):
    hidden = 4
    # 6 rows total, all belonging to the SAME (req_id, layer) key, split into three blocks at
    # ascending logical_starts: [0:2], [2:5], [5:6]. The sidecar lists them OUT of logical order
    # (last block first) to prove the reader sorts by logical_start rather than relying on the
    # upstream drain's write order / encounter order.
    rows = torch.arange(6 * hidden, dtype=torch.float32).reshape(6, hidden)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(rows.contiguous().numpy().tobytes())
    steps = [StepMeta([
        LayerEntry("r0", 0, 5, 1, "all_tokens"),   # rows[5:6] — written LAST logically, listed FIRST
        LayerEntry("r0", 0, 0, 2, "all_tokens"),   # rows[0:2] — written FIRST logically, listed SECOND
        LayerEntry("r0", 0, 2, 3, "all_tokens"),   # rows[2:5] — middle block, listed LAST
    ])]
    write_sidecar(str(tmp_path / "m.jsonl"), steps,
                  {"dtype": "float32", "row_shape": [hidden], "hidden": hidden})
    out = load_ring_artifact(str(raw), str(tmp_path / "m.jsonl"))
    assert torch.equal(out["r0"][0], rows[0:6])


def test_bf16_roundtrip_byte_identical(tmp_path):
    hidden = 4
    # Wide-range random-ish values (not all zeros — zeros would pass even a broken byte mapping).
    torch.manual_seed(0)
    t = (torch.randn(4, hidden) * 1000).to(torch.bfloat16)
    raw = tmp_path / "raw.bin"
    # bf16_tensor.numpy() raises TypeError — must go through the uint16 view to get raw bytes.
    raw.write_bytes(t.contiguous().view(torch.uint16).numpy().tobytes())
    steps = [StepMeta([LayerEntry("r0", 0, 0, 3, "all_tokens"),
                       LayerEntry("r0", 1, 3, 1, "last_token")])]
    write_sidecar(str(tmp_path / "m.jsonl"), steps,
                  {"dtype": "bfloat16", "row_shape": [hidden], "hidden": hidden})
    out = load_ring_artifact(str(raw), str(tmp_path / "m.jsonl"))
    assert out["r0"][0].dtype == torch.bfloat16
    assert out["r0"][1].dtype == torch.bfloat16
    assert torch.equal(out["r0"][0], t[0:3])
    assert torch.equal(out["r0"][1], t[3:4])
