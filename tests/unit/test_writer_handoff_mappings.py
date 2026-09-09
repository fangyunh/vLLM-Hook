import os
import time

import torch


def _maps_count():
    with open("/proc/self/maps") as f:
        return sum(1 for _ in f)


def _synthetic_cpu_cache(n_layers=8, batch=16):
    # Mirrors an HS all_tokens flush: batch separate host tensors per layer -> many storages.
    hs_cache = {}
    for L in range(n_layers):
        hs_cache[f"model.layers.{L}"] = {
            "hidden_states": [torch.randn(5, 64, dtype=torch.float16)
                              for _ in range(batch)],
            "layer_num": L, "hs_mode": "all_tokens",
        }
    return {"config": {"name": "m"}, "hs_cache": hs_cache}


def _walk_equal(a, b, p="cc"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b), f"{p}: values differ"
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{p}: keys"
        for k in a:
            _walk_equal(a[k], b[k], f"{p}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{p}: len"
        for i, (x, y) in enumerate(zip(a, b)):
            _walk_equal(x, y, f"{p}[{i}]")
    else:
        assert a == b, f"{p}: {a!r} != {b!r}"


def test_handoff_resource_count_collapses():
    """Core proof: sharing N tensors through torch mp creates ~1 shm segment (fd + VMA) PER
    TENSOR; sharing the packed buffer creates O(1). Measured under the 'file_descriptor'
    strategy (torch's default), where the per-tensor growth is directly countable via the
    open-fd count AND the /proc/self/maps VMA count -- the exact mechanism that exhausts
    vm.max_map_count. Packing fixes even this worst-case strategy. Synchronous; no GPU/child."""
    import resource

    import torch.multiprocessing as tmp  # importing registers torch's tensor reducers
    from multiprocessing.reduction import ForkingPickler

    from vllm_hook_plugins.graph.tensor_pack import pack_tensor_tree

    # Headroom so holding ~2N shm fds can't trip EMFILE inside the test itself.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 4096), _hard))
    except Exception:
        pass
    tmp.set_sharing_strategy("file_descriptor")

    def _fds():
        return len(os.listdir("/proc/self/fd"))

    N = 200
    ts = [torch.randn(5, 64, dtype=torch.float16) for _ in range(N)]

    f0, m0 = _fds(), _maps_count()
    blob_raw = ForkingPickler.dumps({"x": ts})        # one shm fd + VMA per tensor
    raw_fd, raw_maps = _fds() - f0, _maps_count() - m0

    buf, man = pack_tensor_tree({"x": ts})
    f1 = _fds()
    blob_packed = ForkingPickler.dumps((buf, man))    # ONE shared storage
    packed_fd = _fds() - f1

    del blob_raw, blob_packed
    assert raw_fd >= N, f"raw handoff opened only {raw_fd} shm fds (want >= {N})"
    assert raw_maps >= N // 2, f"raw handoff added only {raw_maps} VMAs (want >= {N // 2})"
    assert packed_fd <= 8, f"packed handoff opened {packed_fd} shm fds (want <= 8)"


def test_writer_process_roundtrip_byte_identical(tmp_path):
    """A real spawned WriterProcess: submit a synthetic HS cache with pack ON, read the .pt
    back, compare torch.equal to a direct in-process write_artifact of the same cache."""
    from vllm_hook_plugins.graph.artifact_writer import write_artifact
    from vllm_hook_plugins.graph.writer_process import WriterProcess

    cc = _synthetic_cpu_cache(n_layers=4, batch=8)

    ref_dir = str(tmp_path / "ref")
    os.makedirs(ref_dir, exist_ok=True)
    write_artifact("hs", cc, ref_dir, "all_tokens", 0, False, False, "hidden_states.pt")
    ref = torch.load(os.path.join(ref_dir, "hidden_states.pt"),
                     map_location="cpu", weights_only=False)

    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"
    wp = WriterProcess(maxsize=4)
    out_dir = str(tmp_path / "wp")
    os.makedirs(out_dir, exist_ok=True)
    assert wp.submit("hs", cc, out_dir, "all_tokens", 0, False, False, "hidden_states.pt")
    wp.close()

    got_fp = os.path.join(out_dir, "hidden_states.pt")
    deadline = time.time() + 30
    while not os.path.exists(got_fp) and time.time() < deadline:
        time.sleep(0.2)
    assert os.path.exists(got_fp), "writer child did not write the artifact"
    got = torch.load(got_fp, map_location="cpu", weights_only=False)
    _walk_equal(cc, got)
    _walk_equal(ref, got)


def test_feeder_falls_back_to_inline_write_on_pack_failure(tmp_path):
    """Durability contract: if the feeder's pack/handoff raises, the artifact is still written
    (in-process on the feeder thread), never silently dropped. submit() returned True, so the
    flush_disk caller does NOT retry -- the feeder is the safety net."""
    from vllm_hook_plugins.graph.writer_process import WriterProcess
    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"
    cc = _synthetic_cpu_cache(n_layers=3, batch=5)
    wp = WriterProcess(maxsize=4)

    def _boom(_):
        raise RuntimeError("simulated pack OOM")

    wp._pack_fn = _boom  # feeder reads self._pack_fn at call time -> next item raises in pack
    out_dir = str(tmp_path / "fb")
    os.makedirs(out_dir, exist_ok=True)
    assert wp.submit("hs", cc, out_dir, "all_tokens", 0, False, False, "hidden_states.pt")
    wp.close()

    fp = os.path.join(out_dir, "hidden_states.pt")
    dl = time.time() + 30
    while not os.path.exists(fp) and time.time() < dl:
        time.sleep(0.2)
    assert os.path.exists(fp), "feeder must write inline when pack fails (no silent artifact loss)"
    got = torch.load(fp, map_location="cpu", weights_only=False)
    _walk_equal(cc, got)
    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"


def test_feeder_fallback_on_share_failure(tmp_path):
    """If the buffer's shm share raises (the mmap-ENOMEM failure mode, now caught on the feeder
    thread because we pre-share there), the artifact is still written inline -- never dropped."""
    from vllm_hook_plugins.graph.tensor_pack import pack_tensor_tree
    from vllm_hook_plugins.graph.writer_process import WriterProcess
    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"
    cc = _synthetic_cpu_cache(n_layers=3, batch=5)
    wp = WriterProcess(maxsize=4)

    class _NoShareTensor:
        def __init__(self, t):
            self._t = t

        def share_memory_(self):
            raise RuntimeError("simulated mmap ENOMEM on share")

    def _pack_unshareable(obj):
        buf, man = pack_tensor_tree(obj)
        return _NoShareTensor(buf), man  # .share_memory_() will raise in the feeder's try

    wp._pack_fn = _pack_unshareable
    out_dir = str(tmp_path / "sh")
    os.makedirs(out_dir, exist_ok=True)
    assert wp.submit("hs", cc, out_dir, "all_tokens", 0, False, False, "hidden_states.pt")
    wp.close()

    fp = os.path.join(out_dir, "hidden_states.pt")
    dl = time.time() + 30
    while not os.path.exists(fp) and time.time() < dl:
        time.sleep(0.2)
    assert os.path.exists(fp), "share failure must fall back to an inline write (no silent loss)"
    _walk_equal(cc, torch.load(fp, map_location="cpu", weights_only=False))
    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"


def test_writer_process_pack_off_roundtrip(tmp_path):
    """The pack-OFF legacy handoff still round-trips (escape hatch stays valid)."""
    os.environ["VLLM_HOOK_WRITER_PACK"] = "0"
    from vllm_hook_plugins.graph.writer_process import WriterProcess
    cc = _synthetic_cpu_cache(n_layers=2, batch=4)
    wp = WriterProcess(maxsize=4)
    out_dir = str(tmp_path / "wp0")
    os.makedirs(out_dir, exist_ok=True)
    assert wp.submit("hs", cc, out_dir, "all_tokens", 0, False, False, "hidden_states.pt")
    wp.close()
    fp = os.path.join(out_dir, "hidden_states.pt")
    dl = time.time() + 30
    while not os.path.exists(fp) and time.time() < dl:
        time.sleep(0.2)
    assert os.path.exists(fp)
    got = torch.load(fp, map_location="cpu", weights_only=False)
    _walk_equal(cc, got)
    os.environ["VLLM_HOOK_WRITER_PACK"] = "1"  # restore default for other tests
