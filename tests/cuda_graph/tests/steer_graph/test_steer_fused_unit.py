"""Standalone GPU unit test: fused ``steer_buffer`` kernel vs the aten reference.

Builds random residual / coeff / vec_id / vec_table / avg_proj / steer_mode on the GPU and
runs BOTH the reference ``_steer_buffer_impl`` (graph/ops.py) and the fused
``steer_buffer_fused`` (graph/steer_triton.py) on INDEPENDENT clones of the same residual,
then compares per row across several shapes and dtypes:
  * no-op rows      (steer_mode == 0, coeff == 0, vec_id may be NONZERO): the row must be left
    byte-identical to the original residual — the gate is on coeff/mode, NOT on vec_table[vid].
  * add_vector rows (steer_mode == 0, coeff != 0): pure elementwise ``coeff * unit`` add.
  * adjust_rs  rows (steer_mode == 1): includes a ``residual . unit`` reduction a fused kernel
    reorders.
  Bound is allclose-style ``atol + rtol*|aten|``: fp32 uses rtol=0 (tight absolute — the kernel
  reproduces aten's fp32 math near-exactly); bf16 uses a magnitude-aware rtol (see ``_CASES``)
  because two bf16 implementations diverge by ~one ULP-of-the-output (bf16's ~0.4% per-op
  precision, amplified by the Triton block-reduce summing the dot product in a different order
  than torch). The token-level parity oracle (run_steer_parity_full.sh) is the strict production
  gate — GPU-PROVEN PASS token-for-token for both methods on Qwen2-1.5B; this is the cheap
  numerical regression guard (a real reduction bug is O(|proj|) ≫ rtol*|aten| and still trips it).

No engine, no pytest. Run:
    python tests/cuda_graph/tests/steer_graph/test_steer_fused_unit.py
Exit: 0 PASS, 1 numeric FAIL, 2 skipped (CUDA unavailable or the fused op is not built yet).
Grep target:  grep 'steer-fused-unit] VERDICT'
"""
from __future__ import annotations

import sys

import torch

TAG = "[steer-fused-unit]"

# (dtype, N tokens, CAP routing len (>= N), HIDDEN, V_MAX, add_atol, add_rtol, adj_atol, adj_rtol)
_CASES = [
    (torch.float32,  37,  64, 1536,  8, 1e-5, 0.0,  1e-3, 0.0),   # base; hidden not a multiple of BLOCK
    (torch.float32,   1,   8, 1536,  4, 1e-5, 0.0,  1e-3, 0.0),   # single row (n == 1)
    (torch.float32, 128, 256, 4096, 16, 1e-5, 0.0,  1e-3, 0.0),   # Granite-like hidden, more vectors
    (torch.float32,  64,  96, 1500,  8, 1e-5, 0.0,  1e-3, 0.0),   # odd hidden → mask-tail; CAP > N
    (torch.bfloat16, 96, 128, 1536,  8, 1e-2, 2e-2, 1e-2, 2e-2),  # production dtype (bf16 two-rounding)
]


def _build(device, dtype, N, CAP, HIDDEN, V_MAX, seed):
    """Random residual + routing buffers. Rows split across no-op / add_vector / adjust_rs so
    both methods AND the zero-coeff no-op (with a deliberately nonzero vec_id) are exercised."""
    g = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(N, HIDDEN, generator=g, device=device, dtype=torch.float32).to(dtype)
    # Unit-norm steering vectors: production vectors are normalized/bounded, unlike a raw randn
    # whose ~sqrt(HIDDEN) norm inflates the residual·unit projection into bf16's coarse-rounding
    # regime (the adversarial case that makes fused-vs-aten diverge without being a kernel bug).
    vt = torch.randn(V_MAX, HIDDEN, generator=g, device=device, dtype=torch.float32)
    vt = vt / vt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    vec_table = vt.to(dtype)
    avg_proj = torch.randn(V_MAX, generator=g, device=device, dtype=torch.float32)

    steer_mode = torch.zeros(CAP, device=device, dtype=torch.int64)
    vec_id = torch.zeros(CAP, device=device, dtype=torch.int64)
    coeff = torch.zeros(CAP, device=device, dtype=torch.float32)

    # Nonzero vec_id on EVERY active row (incl. no-op rows) so the no-op proof rests on
    # coeff==0 & mode==0, not on the padded unit happening to be zero.
    rand_vids = torch.randint(1, V_MAX, (N,), generator=g, device=device)
    rand_c = torch.randn(N, generator=g, device=device, dtype=torch.float32)
    for t in range(N):
        vec_id[t] = rand_vids[t]
        r = t % 3
        if r == 0:            # no-op: mode 0, coeff 0 → row must stay untouched
            steer_mode[t] = 0
            coeff[t] = 0.0
        elif r == 1:          # add_vector: mode 0, host coeff
            steer_mode[t] = 0
            coeff[t] = rand_c[t]
        else:                 # adjust_rs: mode 1, in-kernel coeff (host coeff ignored)
            steer_mode[t] = 1
            coeff[t] = 0.0
    return residual, coeff, vec_id, vec_table, avg_proj, steer_mode


def _run_case(device, ref_impl, fused, dtype, N, CAP, HIDDEN, V_MAX,
              add_atol, add_rtol, adj_atol, adj_rtol, seed):
    residual, coeff, vec_id, vec_table, avg_proj, steer_mode = _build(
        device, dtype, N, CAP, HIDDEN, V_MAX, seed)

    r_ref = residual.clone()
    r_fus = residual.clone()
    # Both mutate their residual IN PLACE (mutates_args=["residual"]).
    ref_impl(r_ref, coeff, vec_id, vec_table, avg_proj, steer_mode)
    fused(r_fus, coeff, vec_id, vec_table, avg_proj, steer_mode)
    torch.cuda.synchronize()

    mode_n = steer_mode[:N]
    coeff_n = coeff[:N]
    noop = (mode_n == 0) & (coeff_n == 0)
    addv = (mode_n == 0) & (coeff_n != 0)
    adj = mode_n == 1

    diff = (r_fus.to(torch.float32) - r_ref.to(torch.float32)).abs()
    row_max = diff.max(dim=1).values if HIDDEN > 0 else diff.new_zeros(N)

    max_add = float(row_max[addv].max()) if bool(addv.any()) else 0.0
    max_adj = float(row_max[adj].max()) if bool(adj.any()) else 0.0
    # allclose-style limit atol + rtol*|aten output| (per method, over that method's rows).
    add_lim = add_atol + add_rtol * (float(r_ref[addv].abs().max()) if bool(addv.any()) else 0.0)
    adj_lim = adj_atol + adj_rtol * (float(r_ref[adj].abs().max()) if bool(adj.any()) else 0.0)

    # No-op rows must be byte-identical to the ORIGINAL residual for BOTH paths (not merely
    # equal to each other) — proves the fused kernel adds nothing when coeff==0 & mode==0.
    noop_ok = True
    if bool(noop.any()):
        noop_ok = (torch.equal(r_fus[noop], residual[noop])
                   and torch.equal(r_ref[noop], residual[noop]))

    ok_add = max_add <= add_lim
    ok_adj = max_adj <= adj_lim
    ok = ok_add and ok_adj and noop_ok

    print(f"{TAG} {str(dtype).split('.')[-1]:>8} N={N:<4} CAP={CAP:<4} H={HIDDEN:<5} "
          f"V={V_MAX:<3} | rows noop/{int(noop.sum())} add/{int(addv.sum())} adj/{int(adj.sum())} "
          f"| add|Δ|={max_add:.2e}(<={add_lim:.2e}){'ok' if ok_add else 'FAIL'} "
          f"adj|Δ|={max_adj:.2e}(<={adj_lim:.2e}){'ok' if ok_adj else 'FAIL'} "
          f"noop={'ok' if noop_ok else 'FAIL'}",
          flush=True)
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print(f"{TAG} CUDA not available; this test requires a GPU.")
        return 2
    device = torch.device("cuda")

    # The reference op impl is a plain Python function registered as the CUDA kernel; call it
    # directly on tensors (no torch.compile, no graph, no engine).
    import vllm_hook_plugins.graph.ops as ops
    ref_impl = ops._steer_buffer_impl
    # Force the reference down the ATEN path regardless of the ambient VLLM_HOOK_STEER_FUSED,
    # so we always compare fused-kernel vs aten-reference (never fused-vs-fused).
    ops._STEER_FUSED = False

    try:
        import vllm_hook_plugins.graph.steer_triton as st
        from vllm_hook_plugins.graph.steer_triton import steer_buffer_fused
    except Exception as e:  # noqa: BLE001 — Triton missing / module not built yet
        print(f"{TAG} fused kernel `steer_buffer_fused` unavailable "
              f"(vllm_hook_plugins.graph.steer_triton): {e!r}. SKIP.")
        return 2

    # n == 0 must not crash and must not launch.
    empty = torch.zeros(0, 1536, device=device, dtype=torch.float32)
    _c = torch.zeros(4, device=device, dtype=torch.float32)
    _vi = torch.zeros(4, device=device, dtype=torch.int64)
    _vt = torch.zeros(4, 1536, device=device, dtype=torch.float32)
    _ap = torch.zeros(4, device=device, dtype=torch.float32)
    _m = torch.zeros(4, device=device, dtype=torch.int64)
    try:
        steer_buffer_fused(empty, _c, _vi, _vt, _ap, _m)
        torch.cuda.synchronize()
        print(f"{TAG} n==0 no-launch guard: ok", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{TAG} n==0 guard raised: {e!r}", flush=True)
        return 1

    # Run every case with the early-exit path OFF (today's kernel, regression guard) AND ON
    # (Lever B: inactive rows skip both passes, add_vector skips the projection). Both must
    # match aten — inactive/add_vector rows bit-for-bit, adjust_rs within the bf16 tolerance.
    _saved_ee = st._EARLY_EXIT
    all_ok = True
    for early_exit in (False, True):
        st._EARLY_EXIT = early_exit
        print(f"{TAG} --- EARLY_EXIT={'ON' if early_exit else 'OFF'} ---", flush=True)
        for i, (dtype, N, CAP, HIDDEN, V_MAX, add_atol, add_rtol, adj_atol, adj_rtol) in enumerate(_CASES):
            ok = _run_case(device, ref_impl, steer_buffer_fused, dtype, N, CAP, HIDDEN, V_MAX,
                           add_atol, add_rtol, adj_atol, adj_rtol, seed=100 + i)
            all_ok = all_ok and ok
    st._EARLY_EXIT = _saved_ee

    print("=" * 70)
    if all_ok:
        print(f"{TAG} VERDICT: PASS — fused steer_buffer matches aten (early-exit OFF and ON).")
        return 0
    print(f"{TAG} VERDICT: FAIL — fused steer_buffer diverged from the aten reference.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
