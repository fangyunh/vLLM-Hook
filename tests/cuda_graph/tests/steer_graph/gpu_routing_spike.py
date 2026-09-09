"""Phase-0 feasibility spike: GPU-side steer routing scatter == host apply_incremental_routing.

The GPU-routing plan (Path 2) replaces the per-step host build
(``_build_routing_steer`` -> ``SteerRegistry.apply_incremental_routing``, an
O(reqs x layers) Python/numpy loop that rewrites the ``(num_layers, cap)`` routing
plane every batch-change step) with:

  (a) a per-REQUEST-SLOT config table -- each request's single ``(vid, mode, coeff,
      layer_mask)`` resolved once, kept in small GPU arrays, refreshed only on a
      composition change; and
  (b) a per-step GPU SCATTER that expands that config + the on-GPU query_start_loc
      into the SAME ``coeff_all / vec_id_all / mode_all`` device slabs the baked
      ``steer_buffer`` op reads.

This spike is the GO/NO-GO gate (plan Task 0.2 Steps 1-3): it proves the scatter
reproduces the host-built slabs **byte-identically** (``torch.equal``), across a
matrix of batches -- single-layer, multi-layer, multi-request-heterogeneous-layers,
padding, and a multi-step churn sequence (admit/finish/reuse). It is device-agnostic
(both the host build and the scatter run on plain tensors), so it runs on CPU with no
GPU -- the algorithm is what's under test here, not the kernel.

    conda activate vllm_hook_env && \
      python tests/cuda_graph/tests/steer_graph/gpu_routing_spike.py

Step 4 (can the scatter run inside / off a captured cudagraph reading query_start_loc)
is a separate GPU-only probe, not covered by this CPU-runnable spike.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from vllm_hook_plugins.graph.install_steer import SteerRegistry


# ---------------------------------------------------------------------------
# The prototype GPU scatter (torch first, per plan 0.2). Stateless full rebuild
# over [:, :width]; the same tensor code runs on cuda in production.
# ---------------------------------------------------------------------------
def scatter_slabs(
    qsl_dev: torch.Tensor,          # (bs+1,) int64 cumulative token counts (query_start_loc)
    slot_vid: torch.Tensor,         # (bs,)   int64  per-request vector id
    slot_mode: torch.Tensor,        # (bs,)   int64  per-request mode (0 add_vector, 1 adjust_rs)
    slot_coeff: torch.Tensor,       # (bs,)   float32 per-request coefficient
    slot_layer_mask: torch.Tensor,  # (bs, num_layers) bool  which layers this request steers
    num_layers: int,
    cap: int,
    width: int,
):
    """Expand the per-slot config into (coeff_all, vec_id_all, mode_all) over [:, :width].

    column c belongs to request i where qsl[i] <= c < qsl[i+1]  (found by searchsorted);
    a cell (layer, c) is active iff request i steers `layer` and c is a real (non-padding)
    column. Returns full ``(num_layers, cap)`` slabs with only [:, :width] filled and the
    tail left zero (the op never reads past the padded token count <= width).
    """
    dev = qsl_dev.device
    bs = int(qsl_dev.numel()) - 1
    w = max(1, min(int(width), cap))
    real_n = int(qsl_dev[-1].item())

    out_coeff = torch.zeros(num_layers, cap, dtype=torch.float32, device=dev)
    out_vecid = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    out_mode = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    if bs <= 0:
        return out_coeff, out_vecid, out_mode

    col = torch.arange(w, device=dev)                                  # [w]
    # request owning each column; clamp keeps the gather in-range for padding cols
    req_of_col = torch.searchsorted(qsl_dev, col, right=True) - 1      # [w]
    req_of_col = req_of_col.clamp_(0, bs - 1)
    valid = col < real_n                                               # [w] real vs padding

    vid_c = torch.where(valid, slot_vid[req_of_col], torch.zeros_like(req_of_col))    # [w]
    mode_c = torch.where(valid, slot_mode[req_of_col], torch.zeros_like(req_of_col))  # [w]
    coef_c = torch.where(valid, slot_coeff[req_of_col],
                         torch.zeros(w, dtype=torch.float32, device=dev))             # [w]

    # (num_layers, w) active mask: request steers `layer` AND column is real
    mask = slot_layer_mask[req_of_col].transpose(0, 1) & valid.unsqueeze(0)           # [nL, w]

    nl = num_layers
    out_coeff[:, :w] = torch.where(mask, coef_c.unsqueeze(0).expand(nl, -1),
                                   torch.zeros(1, dtype=torch.float32, device=dev))
    out_vecid[:, :w] = torch.where(mask, vid_c.unsqueeze(0).expand(nl, -1),
                                   torch.zeros(1, dtype=torch.int64, device=dev))
    out_mode[:, :w] = torch.where(mask, mode_c.unsqueeze(0).expand(nl, -1),
                                  torch.zeros(1, dtype=torch.int64, device=dev))
    return out_coeff, out_vecid, out_mode


# ---------------------------------------------------------------------------
# A "batch" = qsl + per-request steer config. Helpers to derive both the
# assignments the host router consumes AND the slot config the scatter consumes,
# from the SAME source, so the two are truly compared on equal inputs.
# ---------------------------------------------------------------------------
class Req:
    """One request's steer config: token span + a single (vid, mode, coeff) over a layer set."""
    def __init__(self, ntok: int, layers: List[int], vid: int, mode: int, coeff: float):
        self.ntok = ntok
        self.layers = layers
        self.vid = vid
        self.mode = mode
        self.coeff = coeff


def qsl_of(reqs: List[Req]) -> List[int]:
    q = [0]
    for r in reqs:
        q.append(q[-1] + r.ntok)
    return q


def assignments_of(reqs: List[Req], qsl: List[int], cap: int):
    """Exactly what _build_routing_steer emits: one (start,end,layer,coeff,vid,mode) per
    (request, target layer), sharing the request's [start,end) column span."""
    out = []
    for i, r in enumerate(reqs):
        start = int(qsl[i])
        end = min(int(qsl[i + 1]), cap)
        if end <= start:
            continue
        for L in r.layers:
            out.append((start, end, L, r.coeff, r.vid, r.mode))
    return out


def slot_config_of(reqs: List[Req], num_layers: int, device="cpu"):
    bs = len(reqs)
    slot_vid = torch.zeros(bs, dtype=torch.int64, device=device)
    slot_mode = torch.zeros(bs, dtype=torch.int64, device=device)
    slot_coeff = torch.zeros(bs, dtype=torch.float32, device=device)
    slot_mask = torch.zeros(bs, num_layers, dtype=torch.bool, device=device)
    for i, r in enumerate(reqs):
        slot_vid[i] = r.vid
        slot_mode[i] = r.mode
        slot_coeff[i] = r.coeff
        for L in r.layers:
            slot_mask[i, L] = True
    return slot_vid, slot_mode, slot_coeff, slot_mask


# ---------------------------------------------------------------------------
# The comparison: drive the SAME registry through a sequence of batches with the
# host router, and independently scatter each step; assert [:, :width] equality.
# ---------------------------------------------------------------------------
def _fresh_registry(num_layers, cap, device="cpu"):
    return SteerRegistry(num_layers, cap, hidden=4, v_max=8, device=device,
                         dtype=torch.float32)


def compare_sequence(name: str, steps: List[List[Req]], num_layers: int, cap: int,
                     width_fn=None, device="cpu") -> List[str]:
    """steps = list of batches (each a list of Req). Byte-identity every step vs the
    host router driven statefully through the SAME sequence (so incremental diffing,
    slot reuse, and deactivation are all exercised)."""
    fails: List[str] = []
    reg = _fresh_registry(num_layers, cap, device)
    for si, reqs in enumerate(steps):
        qsl = qsl_of(reqs)
        real_n = qsl[-1]
        width = int(width_fn(real_n, cap)) if width_fn else real_n
        width = max(1, min(width, cap))

        # host reference: stateful incremental router
        assignments = assignments_of(reqs, qsl, cap)
        reg.apply_incremental_routing(assignments, width)
        ref_coeff = reg.coeff_all[:, :width].clone()
        ref_vid = reg.vec_id_all[:, :width].clone()
        ref_mode = reg.mode_all[:, :width].clone()

        # scatter (stateless full rebuild over [:, :width])
        qsl_dev = torch.tensor(qsl, dtype=torch.int64, device=device)
        sv, sm, sc, smask = slot_config_of(reqs, num_layers, device)
        oc, ov, om = scatter_slabs(qsl_dev, sv, sm, sc, smask, num_layers, cap, width)
        oc, ov, om = oc[:, :width], ov[:, :width], om[:, :width]

        if not torch.equal(oc, ref_coeff):
            fails.append(f"{name} step{si}: coeff mismatch "
                         f"(max|d|={(oc - ref_coeff).abs().max().item()})")
        if not torch.equal(ov, ref_vid):
            fails.append(f"{name} step{si}: vec_id mismatch")
        if not torch.equal(om, ref_mode):
            fails.append(f"{name} step{si}: mode mismatch")
    return fails


def build_matrix():
    """(name, steps, num_layers, cap, width_fn) cases covering the plan's Step-3 matrix."""
    nL, cap = 12, 64
    pad = lambda real_n, cap: min(cap, max(real_n, 16))   # emulate _upload_width padding

    cases = []
    # single-layer, single request (the validated common case)
    cases.append(("single_layer", [[Req(5, [3], 2, 1, 0.0)]], nL, cap, None))
    # multi-layer single request
    cases.append(("multi_layer", [[Req(5, [1, 3, 5], 1, 0, 2.5)]], nL, cap, None))
    # multi-request heterogeneous layer sets, disjoint columns
    cases.append(("multireq_hetero",
                  [[Req(3, [1, 2], 1, 0, 2.0), Req(4, [4], 3, 1, 0.0),
                    Req(2, [0, 7, 11], 5, 0, 1.25)]], nL, cap, None))
    # padding: real_n < width (the cudagraph-padded decode step)
    cases.append(("padded_width", [[Req(4, [2, 6], 1, 0, 3.0)]], nL, cap, pad))
    # decode-like: all reqs 1 token, mixed steer/non-steer
    cases.append(("decode_mixed",
                  [[Req(1, [4], 1, 0, 1.0), Req(1, [], 0, 0, 0.0),
                    Req(1, [4, 9], 2, 1, 0.0), Req(1, [4], 1, 0, 1.0)]], nL, cap, pad))
    # CHURN sequence: admit -> stable -> finish-one -> reuse-slot -> all-idle.
    churn = [
        [Req(3, [5], 1, 0, 2.0), Req(3, [5], 1, 0, 2.0)],            # admit 2
        [Req(1, [5], 1, 0, 2.0), Req(1, [5], 1, 0, 2.0)],            # stable decode
        [Req(1, [5], 1, 0, 2.0)],                                   # req1 finishes
        [Req(1, [5], 1, 0, 2.0), Req(2, [3, 8], 4, 1, 0.0)],        # new req reuses slot, diff layers
        [Req(1, [], 0, 0, 0.0)],                                    # steer-less step (deactivate)
        [Req(1, [5], 1, 0, 2.0)],                                   # re-activate
    ]
    cases.append(("churn_slot_reuse", churn, nL, cap, pad))
    # width-shrink churn: a wide prefill then a narrow decode (stale-tail zeroing)
    shrink = [
        [Req(20, [7], 2, 1, 0.0)],       # wide prefill (width padded to >=20)
        [Req(1, [7], 2, 1, 0.0)],        # narrow decode (columns 1..19 must clear)
    ]
    cases.append(("width_shrink", shrink, nL, cap, pad))
    return cases


def main():
    torch.manual_seed(0)
    all_fails: List[str] = []
    ncases = 0
    for name, steps, nL, cap, wf in build_matrix():
        ncases += 1
        fails = compare_sequence(name, steps, nL, cap, wf)
        if fails:
            all_fails.extend(fails)
            for f in fails:
                print(f"FAIL  {f}")
        else:
            print(f"PASS  {name} ({len(steps)} step(s))")
    print("=" * 66)
    if all_fails:
        print(f"VERDICT: FAIL ({len(all_fails)} mismatch(es) across {ncases} cases)")
        return 1
    print(f"VERDICT: PASS (scatter byte-identical to host router, {ncases}/{ncases} cases)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
