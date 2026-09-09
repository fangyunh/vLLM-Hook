"""Bit-exactness oracle: Triton-fused capture_hs vs the aten reference body.

No engine, no cudagraph — a direct kernel-vs-kernel compare across the shape/dtype/routing space
the real path produces (indexed destination: dst = index[row]).
torch.equal, not allclose: the fused kernel widens to fp32 and rounds once on store, exactly as
aten's fp32-opmath add does, so any difference is a bug.

The SENTINEL row is excluded: several source rows route there concurrently, so its
content is undefined by construction (see capture_triton.py's module docstring).

THREE cell families:
  1. run_indexed_cells      -- dst = index[row], today's routing plane.
  2. run_oob_cells          -- OUT-OF-RANGE destinations are contained on SENTINEL and never
                               touch memory outside hs_buf (final-review fix I1).
  3. run_dispatch_dtype_cells -- ops._capture_hs_impl DECLINES to fuse on a buffer/hidden dtype
                               mismatch (final-review fix I3); drives the DISPATCH, not the
                               kernel, because that is where the gate lives.
"""
import itertools
import sys

import torch

from vllm_hook_plugins.graph.capture_triton import capture_hs_fused


def aten_reference(hidden, residual, buf, index, has_residual):
    n = hidden.shape[0]
    idx = index[:n].to(torch.long)
    combined = hidden + residual if has_residual else hidden
    src = combined if combined.dtype == buf.dtype else combined.to(buf.dtype)
    buf.index_copy_(0, idx, src)


def make_index(n, rows, pattern, device):
    """Destination rows. `rows` usable + 1 sentinel at index `rows`."""
    if pattern == "contiguous":          # steady-state decode: one advancing run
        return (torch.arange(n, device=device, dtype=torch.int64) % rows)
    if pattern == "all_sentinel":        # no request capturing this layer
        return torch.full((n,), rows, device=device, dtype=torch.int64)
    if pattern == "half_sentinel":       # mixed batch: some requests capture, some don't
        idx = torch.arange(n, device=device, dtype=torch.int64) % rows
        idx[1::2] = rows
        return idx
    if pattern == "scattered":           # multi-request churn: disjoint runs, out of order
        g = torch.Generator(device="cpu").manual_seed(1234)
        return torch.randperm(rows, generator=g)[:n].to(device).to(torch.int64)
    raise ValueError(pattern)


def run_indexed_cells():
    """Mode 1: today's routing plane. dst = index[row]."""
    dev = "cuda"
    fails, total = 0, 0
    for n, hid, dt, pat, has_res in itertools.product(
            [1, 8, 64, 256, 512, 1024, 4096], [1536, 4096],
            [torch.bfloat16, torch.float16, torch.float32],
            ["contiguous", "all_sentinel", "half_sentinel", "scattered"], [0, 1]):
        rows = max(n, 64)
        g = torch.Generator(device="cpu").manual_seed(n * 7 + hid + int(dt is torch.float32))
        h = torch.randn(n, hid, generator=g).to(dev).to(dt)
        r = torch.randn(n, hid, generator=g).to(dev).to(dt)
        if not has_res:
            r = h                        # the aten body passes hidden twice when has_residual==0
        index = make_index(n, rows, pat, dev)

        buf_a = torch.zeros(rows + 1, hid, dtype=dt, device=dev)
        buf_f = torch.zeros(rows + 1, hid, dtype=dt, device=dev)
        aten_reference(h, r, buf_a, index, has_res)
        capture_hs_fused(h, r, buf_f, index, has_res)
        torch.cuda.synchronize()

        total += 1
        if not torch.equal(buf_a[:rows], buf_f[:rows]):   # exclude the sentinel row
            fails += 1
            d = (buf_a[:rows].float() - buf_f[:rows].float()).abs().max().item()
            print(f"[capture-fused] INDEXED MISMATCH n={n} hid={hid} dt={dt} pat={pat} "
                  f"has_res={has_res} max|d|={d:.3e}", flush=True)
    return fails, total


def run_oob_cells():
    """Mode 3 (final-review fix I1): OUT-OF-RANGE destinations must land on SENTINEL, and must
    not touch memory outside ``hs_buf``.

    WHY THIS CELL EXISTS. ``tl.store``'s mask covers only the HIDDEN dimension, so before the
    destination clamp an out-of-range ``dst`` was a silent write into whatever allocation follows
    (or precedes) ``hs_buf`` -- KV cache, weights. The aten body this kernel replaced bounds-
    checked every index (``IndexError`` / ``CUDA_KERNEL_ASSERT``), so the default flip removed
    the last line of defence turning a routing bug into a crash rather than cross-allocation
    corruption.

    HOW IT DETECTS THE OLD BEHAVIOUR. ``hs_buf`` is carved out of the MIDDLE of a larger zeroed
    ``arena``, so there is a canary band on each side. An unclamped store at ``dst < 0`` or
    ``dst > SENTINEL`` lands in a canary; the clamp sends it to SENTINEL instead. Asserting the
    canaries stay zero is therefore a direct test for the corruption, not a proxy for it.

    The reference is aten fed the SAME index with the out-of-range entries replaced by SENTINEL
    -- i.e. exactly the contract the clamp promises. (aten cannot be fed the raw index: it would
    raise, which is the whole point.) The clamp is the IDENTITY on in-range indices, so these
    cells also re-prove that the in-range rows are untouched by it.
    """
    dev = "cuda"
    guard = 8
    fails, total = 0, 0
    for n, hid, dt, kind in itertools.product(
            [1, 8, 64, 512], [1536, 4096],
            [torch.bfloat16, torch.float16, torch.float32],
            ["negative", "past_end", "far_past_end", "mixed"]):
        rows = max(n, 64)
        sentinel = rows
        g = torch.Generator(device="cpu").manual_seed(n * 13 + hid + len(kind))
        h = torch.randn(n, hid, generator=g).to(dev).to(dt)
        r = torch.randn(n, hid, generator=g).to(dev).to(dt)

        # hs_buf lives in the MIDDLE of the arena so both directions have a canary.
        arena_f = torch.zeros(guard + rows + 1 + guard, hid, dtype=dt, device=dev)
        arena_a = torch.zeros_like(arena_f)
        buf_f = arena_f[guard:guard + rows + 1]
        buf_a = arena_a[guard:guard + rows + 1]

        index = torch.arange(n, device=dev, dtype=torch.int64) % rows
        if kind == "negative":
            index[::2] = -3
        elif kind == "past_end":
            index[::2] = sentinel + 1
        elif kind == "far_past_end":
            index[::2] = sentinel + guard + 3
        else:                                     # mixed: both directions, same launch
            index[::3] = -1
            index[1::3] = sentinel + guard + 1

        # Reference: the same destinations with everything out of [0, SENTINEL] sent to SENTINEL.
        ref_index = torch.where((index >= 0) & (index <= sentinel), index,
                                torch.full_like(index, sentinel))
        if not ((ref_index != index).any()):
            continue                                  # not an OOB cell after all; skip
        aten_reference(h, r, buf_a, ref_index, 1)
        capture_hs_fused(h, r, buf_f, index, 1)
        torch.cuda.synchronize()

        total += 1
        ok = torch.equal(buf_a[:rows], buf_f[:rows])
        lead_clean = bool(torch.all(arena_f[:guard] == 0).item())
        tail_clean = bool(torch.all(arena_f[guard + rows + 1:] == 0).item())
        if not (ok and lead_clean and tail_clean):
            fails += 1
            print(f"[capture-fused] OOB MISMATCH n={n} hid={hid} dt={dt} kind={kind} "
                  f"rows_ok={ok} lead_canary_clean={lead_clean} "
                  f"tail_canary_clean={tail_clean}", flush=True)
    return fails, total


def run_dispatch_dtype_cells():
    """Mode 4 (final-review fix I3): ``ops._capture_hs_impl`` must DECLINE to fuse when
    ``hs_buf.dtype != hidden.dtype``.

    The 840 cells above drive ``capture_hs_fused`` directly and allocate ``buf`` with the same
    dtype as ``hidden`` in every single one -- so ``aten_reference``'s ``combined.dtype !=
    buf.dtype`` branch was DEAD, implying a coverage the sweeps did not have. These cells drive
    the DISPATCH instead, across matched AND mismatched buffer dtypes, and assert bit-equality
    with the aten reference in both. That passes only if the mismatch is routed to aten: the
    kernel widens to fp32, adds and rounds ONCE on store, while aten rounds at ``hidden +
    residual`` in the hidden dtype and AGAIN on the cast -- fused would be strictly more precise,
    not equal. The dead reference branch becomes live here, so it is kept rather than deleted.
    """
    from vllm_hook_plugins.graph import ops

    dev = "cuda"
    fails, total = 0, 0
    for n, hid, hdt, bdt, has_res in itertools.product(
            [1, 8, 512], [1536, 4096],
            [torch.bfloat16, torch.float16, torch.float32],
            [torch.bfloat16, torch.float16, torch.float32], [0, 1]):
        rows = max(n, 64)
        g = torch.Generator(device="cpu").manual_seed(n * 17 + hid)
        h = torch.randn(n, hid, generator=g).to(dev).to(hdt)
        r = torch.randn(n, hid, generator=g).to(dev).to(hdt)
        if not has_res:
            r = h
        index = torch.arange(n, device=dev, dtype=torch.int64) % rows

        buf_a = torch.zeros(rows + 1, hid, dtype=bdt, device=dev)
        buf_d = torch.zeros(rows + 1, hid, dtype=bdt, device=dev)
        aten_reference(h, r, buf_a, index, has_res)
        ops._capture_hs_impl(h, r, buf_d, index, has_res)
        torch.cuda.synchronize()

        total += 1
        if not torch.equal(buf_a[:rows], buf_d[:rows]):
            fails += 1
            d = (buf_a[:rows].float() - buf_d[:rows].float()).abs().max().item()
            print(f"[capture-fused] DISPATCH-DTYPE MISMATCH n={n} hid={hid} hidden={hdt} "
                  f"buf={bdt} has_res={has_res} max|d|={d:.3e}", flush=True)
    return fails, total


def main():
    if not torch.cuda.is_available():
        print("[capture-fused] VERDICT: FAIL (no CUDA)", flush=True)
        return 1
    f1, t1 = run_indexed_cells()
    f3, t3 = run_oob_cells()
    f4, t4 = run_dispatch_dtype_cells()
    print(f"[capture-fused] indexed {t1 - f1}/{t1} exact", flush=True)
    print(f"[capture-fused] oob-clamp {t3 - f3}/{t3} contained", flush=True)
    print(f"[capture-fused] dispatch-dtype {t4 - f4}/{t4} exact", flush=True)
    if t3 == 0 or t4 == 0:
        print("[capture-fused] VERDICT: FAIL (a new cell family ran ZERO cells -- vacuous)",
              flush=True)
        return 1
    fails = f1 + f3 + f4
    print(f"[capture-fused] VERDICT: {'PASS' if fails == 0 else 'FAIL'}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
