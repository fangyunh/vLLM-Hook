"""VLLM_HOOK_CAPTURE_FUSED gating — default ON dispatches to the fused kernel; =0 pins the aten
fallback (kept genuinely exercised — this is the kill switch, not dead code)."""
import importlib

import torch


def _reload_ops(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_HOOK_CAPTURE_FUSED", raising=False)
    else:
        monkeypatch.setenv("VLLM_HOOK_CAPTURE_FUSED", value)
    import vllm_hook_plugins.graph.ops as ops
    return importlib.reload(ops)


class _FakeCuda(torch.Tensor):
    """CPU tensor that reports is_cuda=True, so the dispatch guard can be exercised without a GPU.

    Do NOT instead monkeypatch `torch.Tensor.is_cuda` globally — it succeeds on this torch build
    but mutates a type every other test shares. `as_subclass` is local and free.
    """
    @property
    def is_cuda(self):
        return True


def test_flag_defaults_on(monkeypatch):
    ops = _reload_ops(monkeypatch, None)
    assert ops._CAPTURE_FUSED is True


def test_flag_reads_one(monkeypatch):
    ops = _reload_ops(monkeypatch, "1")
    assert ops._CAPTURE_FUSED is True


def test_flag_zero_is_off(monkeypatch):
    """The kill switch: =0 must still disable fusion and run the aten body, unset default or not."""
    ops = _reload_ops(monkeypatch, "0")
    assert ops._CAPTURE_FUSED is False


def test_cpu_tensors_never_take_the_fused_path(monkeypatch):
    """Guard: the dispatch is `is_cuda`-gated, so a CPU call must reach aten even flag-on."""
    ops = _reload_ops(monkeypatch, "1")
    called = []
    monkeypatch.setattr(
        "vllm_hook_plugins.graph.capture_triton.capture_hs_fused",
        lambda *a, **k: called.append(1),
    )
    h = torch.ones(2, 4)
    r = torch.ones(2, 4)
    buf = torch.zeros(3, 4)
    idx = torch.tensor([1, 2], dtype=torch.int64)
    ops._capture_hs_impl(h, r, buf, idx, 1)
    assert called == []
    assert torch.equal(buf[1], torch.full((4,), 2.0))


def test_fused_failure_falls_back_to_aten(monkeypatch):
    """A Triton error must never be load-bearing — the aten body still runs."""
    ops = _reload_ops(monkeypatch, "1")

    def _boom(*a, **k):
        raise RuntimeError("triton exploded")

    monkeypatch.setattr(
        "vllm_hook_plugins.graph.capture_triton.capture_hs_fused", _boom)
    h = torch.ones(2, 4).as_subclass(_FakeCuda)
    r = torch.ones(2, 4)
    buf = torch.zeros(3, 4)
    idx = torch.tensor([1, 2], dtype=torch.int64)
    ops._capture_hs_impl(h, r, buf, idx, 1)
    assert torch.equal(buf[1], torch.full((4,), 2.0))


# --------------------------------------------------------------------------- #
# Final-review fix I3: the byte-identity dispatch gate must also require
# hs_buf.dtype == hidden.dtype.
#
# WHY. The two paths round DIFFERENTLY when the dtypes differ. aten computes
# `hidden + residual` in the HIDDEN dtype (one rounding) and then casts to hs_buf's
# dtype (a second). The fused kernel widens to fp32, adds, and rounds ONCE on store.
# bf16 hidden -> fp32 buf therefore makes the fused result strictly MORE precise --
# not `torch.equal`, which is the contract this dispatch promises. Unreachable in
# production today (`buf_dtype = model.dtype`), which is exactly why nothing caught
# it: the 840-cell GPU oracle allocates its buffer with the same dtype as `hidden`
# in every single cell, so its own `combined.dtype != buf.dtype` reference branch
# was dead code implying coverage that did not exist.
# --------------------------------------------------------------------------- #
def test_buf_dtype_mismatch_declines_fusion_and_takes_aten(monkeypatch):
    ops = _reload_ops(monkeypatch, "1")
    called = []
    monkeypatch.setattr(
        "vllm_hook_plugins.graph.capture_triton.capture_hs_fused",
        lambda *a, **k: called.append(1),
    )
    h = torch.ones(2, 4, dtype=torch.bfloat16).as_subclass(_FakeCuda)
    r = torch.ones(2, 4, dtype=torch.bfloat16)
    buf = torch.zeros(3, 4, dtype=torch.float32)   # fp32 sink, bf16 source
    idx = torch.tensor([1, 2], dtype=torch.int64)
    ops._capture_hs_impl(h, r, buf, idx, 1)
    assert called == [], (
        "hs_buf.dtype != hidden.dtype must DECLINE the fused path: the kernel's single "
        "rounding on store is not bit-equal to aten's add-then-cast double rounding")
    assert torch.equal(buf[1], torch.full((4,), 2.0, dtype=torch.float32))


def test_matching_dtypes_still_fuse(monkeypatch):
    """Control for the guard above — it must reject ONLY the mismatch, not fusion itself."""
    ops = _reload_ops(monkeypatch, "1")
    called = []
    monkeypatch.setattr(
        "vllm_hook_plugins.graph.capture_triton.capture_hs_fused",
        lambda *a, **k: called.append(1),
    )
    h = torch.ones(2, 4, dtype=torch.bfloat16).as_subclass(_FakeCuda)
    r = torch.ones(2, 4, dtype=torch.bfloat16)
    buf = torch.zeros(3, 4, dtype=torch.bfloat16)
    idx = torch.tensor([1, 2], dtype=torch.int64)
    ops._capture_hs_impl(h, r, buf, idx, 1)
    assert called == [1]


# --------------------------------------------------------------------------- #
# Final-review fix I1: the fused kernel must clamp its DESTINATION row.
#
# `tl.store(..., mask=mask)` masks only the HIDDEN dimension. The aten body it
# replaced (`hs_buf.index_copy_`) bounds-checks every index -- IndexError on CPU,
# CUDA_KERNEL_ASSERT on GPU -- so before the clamp an out-of-range `dst` was a
# SILENT write into whatever allocation follows hs_buf (KV cache, weights) on the
# now-DEFAULT path.
#
# The behavioural proof is the GPU oracle (capture_fused_unit.py::run_oob_cells,
# which drives genuinely out-of-range destinations and asserts they land on
# SENTINEL with in-range rows untouched). This test is the no-GPU regression pin
# so the clamp cannot be deleted without CI noticing.
# --------------------------------------------------------------------------- #
def test_fused_kernel_clamps_destination_into_the_legal_row_range():
    import inspect

    from vllm_hook_plugins.graph import capture_triton

    # NOTE: read the MODULE source, not the kernel's. `capture_hs_fused_kernel` is a
    # `triton.jit` JITFunction, and `inspect.getsource` on one raises TypeError ("module,
    # class, method, function... expected") -- it is not a plain function object.
    src = inspect.getsource(capture_triton)
    assert "dst = tl.where((dst >= 0) & (dst <= SENTINEL), dst, SENTINEL)" in src, (
        "the destination bounds clamp is gone. Without it the fused kernel -- which is the "
        "DEFAULT capture path -- stores at an unchecked `dst`, so a routing bug corrupts a "
        "neighbouring allocation instead of crashing the way aten's index_copy_ did.")
    # ...and it must come AFTER both destination branches, or one of them escapes unclamped.
    assert src.index("dst = tl.load(index_ptr + row)") < src.index(
        "dst = tl.where((dst >= 0) & (dst <= SENTINEL)"), \
        "the clamp must come after the indexed destination load"
