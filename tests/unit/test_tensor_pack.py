import pickle

import torch

from vllm_hook_plugins.graph.tensor_pack import pack_tensor_tree, unpack_tensor_tree


def _assert_tree_equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: type"
        assert a.shape == b.shape and a.dtype == b.dtype, f"{path}: shape/dtype"
        assert torch.equal(a, b), f"{path}: values"
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys"
        for k in a:
            _assert_tree_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: seq"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_tree_equal(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: raw {a!r} != {b!r}"


def test_roundtrip_mixed_dtypes_and_structure():
    obj = {
        "config": {"num_attention_heads": 16, "name": "m"},
        "qk_cache": {
            "layer.0": {
                "q": [torch.randn(7, 128, dtype=torch.float16),
                      torch.randn(3, 128, dtype=torch.float16)],
                "k_all": [torch.randn(5, 1024, dtype=torch.bfloat16)],
                "layer_num": 0,
                "hookq_mode": "all_tokens",
                "q_scale": [torch.randn(7, 1, dtype=torch.float32), None],
            }
        },
        "packed_u8": torch.randint(0, 255, (64,), dtype=torch.uint8),
    }
    buf, man = pack_tensor_tree(obj)
    assert buf.dtype == torch.uint8 and buf.dim() == 1
    out = unpack_tensor_tree(buf, man)
    _assert_tree_equal(obj, out)


def test_shared_storage_is_packed_once():
    # k_all growing-prefix case: [full[:1], full[:3], full[:5]] all share `full`.
    full = torch.randn(5, 1024, dtype=torch.bfloat16)
    k_all = [full[:1], full[:3], full[:5]]
    obj = {"k_all": k_all}
    buf, man = pack_tensor_tree(obj)
    # buffer holds `full` ONCE (5*1024*2 bytes, +<=7 alignment pad), NOT 1+3+5 copies.
    unique_bytes = full.untyped_storage().nbytes()
    assert buf.numel() <= unique_bytes + 8, (
        f"expected ~{unique_bytes} deduped bytes, got {buf.numel()}")
    out = unpack_tensor_tree(buf, man)
    _assert_tree_equal(obj, out)
    # And the reconstructed prefixes still share ONE storage (dedup preserved).
    outs = out["k_all"]
    assert (outs[0].untyped_storage().data_ptr()
            == outs[2].untyped_storage().data_ptr())


def test_non_contiguous_tensor():
    t = torch.randn(8, 8, dtype=torch.float32)[:, ::2]  # non-contiguous view
    obj = {"t": t}
    buf, man = pack_tensor_tree(obj)
    out = unpack_tensor_tree(buf, man)
    _assert_tree_equal(obj, out)


def test_empty_and_none_and_scalar_leaves():
    obj = {"a": torch.empty(0, dtype=torch.float16), "b": None,
           "c": 3, "d": [torch.zeros(2, 2), None, "x"]}
    buf, man = pack_tensor_tree(obj)
    out = unpack_tensor_tree(buf, man)
    _assert_tree_equal(obj, out)


def test_manifest_has_no_tensors():
    obj = {"q": [torch.randn(4, 4)], "n": 2}
    _, man = pack_tensor_tree(obj)
    # manifest must pickle WITHOUT invoking any tensor reduction (pure python).
    blob = pickle.dumps(man)

    def walk(m):
        assert not isinstance(m, torch.Tensor), "manifest contains a tensor"
        if isinstance(m, dict):
            for v in m.values():
                walk(v)
        elif isinstance(m, (list, tuple)):
            for v in m:
                walk(v)
    walk(man)
    assert isinstance(blob, (bytes, bytearray))
