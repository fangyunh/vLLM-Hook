"""Lever A Task A2 GPU test: graph/steer_routing_gpu.scatter_routing (Triton + torch)
is byte-identical to the host SteerRegistry.apply_incremental_routing, and cheap.

Drives the SAME matrix + churn sequences as the Phase-0 spike (single/multi-layer,
heterogeneous multi-request, padding, decode-mixed, churn/slot-reuse, width-shrink),
but expands the slot config on the GPU via the production scatter (both backends) into
resident slabs, and asserts torch.equal vs a stateful host router at every step. Then
times the Triton scatter (into resident slabs) at B x N — expect a small ~constant that
does NOT scale with N or B (vs the host build's 0.08..4.3 ms/step).

    (LSF) python test_scatter_routing_gpu.py     # run from the steer_graph dir
"""
from __future__ import annotations

import statistics
import sys

import torch

from vllm_hook_plugins.graph.install_steer import SteerRegistry
from vllm_hook_plugins.graph.steer_routing_gpu import scatter_routing
from gpu_routing_spike import (Req, assignments_of, build_matrix, qsl_of,
                               slot_config_of)


def _host_slabs(reg, reqs, qsl, cap, width):
    reg.apply_incremental_routing(assignments_of(reqs, qsl, cap), width)
    return (reg.coeff_all[:, :width].clone(), reg.vec_id_all[:, :width].clone(),
            reg.mode_all[:, :width].clone())


def compare_sequence_gpu(name, steps, num_layers, cap, width_fn, dev, backend):
    """Stateful host router vs stateless GPU scatter (`backend`) each step; byte-identity."""
    fails = []
    host = SteerRegistry(num_layers, cap, 4, 8, device="cpu", dtype=torch.float32)
    oc = torch.zeros(num_layers, cap, dtype=torch.float32, device=dev)
    ov = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    om = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    for si, reqs in enumerate(steps):
        qsl = qsl_of(reqs)
        real_n = qsl[-1]
        width = int(width_fn(real_n, cap)) if width_fn else real_n
        width = max(1, min(width, cap))
        hc, hv, hm = _host_slabs(host, reqs, qsl, cap, width)

        qsl_dev = torch.tensor(qsl, dtype=torch.int64, device=dev)
        sv, sm, sc, smask = slot_config_of(reqs, num_layers, dev)
        oc.zero_(); ov.zero_(); om.zero_()
        scatter_routing(qsl_dev, sv, sm, sc, smask, oc, ov, om, real_n, width, backend=backend)
        torch.cuda.synchronize()
        if not torch.equal(oc[:, :width].cpu(), hc):
            fails.append(f"{name}[{backend}] step{si}: coeff mismatch")
        if not torch.equal(ov[:, :width].cpu(), hv):
            fails.append(f"{name}[{backend}] step{si}: vec_id mismatch")
        if not torch.equal(om[:, :width].cpu(), hm):
            fails.append(f"{name}[{backend}] step{si}: mode mismatch")
    return fails


def time_triton(dev, num_layers=32, cap=2048):
    print("\n  [launch cost] Triton scatter into resident slabs (ms/step, median 200):")
    print(f"  {'B':>4} {'N':>4} {'ms':>9}")
    oc = torch.zeros(num_layers, cap, dtype=torch.float32, device=dev)
    ov = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    om = torch.zeros(num_layers, cap, dtype=torch.int64, device=dev)
    for B in (1, 8, 32, 64):
        for N in (1, 8, 32):
            reqs = [Req(1, list(range(N)), 1 + (i % 4), 0, 2.0) for i in range(B)]
            qsl = qsl_of(reqs)
            width = min(cap, max(qsl[-1], 16))
            qsl_dev = torch.tensor(qsl, dtype=torch.int64, device=dev)
            sv, sm, sc, smask = slot_config_of(reqs, num_layers, dev)
            for _ in range(20):
                scatter_routing(qsl_dev, sv, sm, sc, smask, oc, ov, om, qsl[-1], width,
                                backend="triton")
            torch.cuda.synchronize()
            ts = []
            for _ in range(200):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                scatter_routing(qsl_dev, sv, sm, sc, smask, oc, ov, om, qsl[-1], width,
                                backend="triton")
                e1.record(); torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            print(f"  {B:>4} {N:>4} {statistics.median(ts):>9.4f}")


def main():
    if not torch.cuda.is_available():
        print("VERDICT: SKIP (no cuda)"); return 2
    dev = torch.device("cuda")
    print(f"scatter_routing GPU test on {torch.cuda.get_device_name(0)}")
    all_fails = []
    for backend in ("triton", "torch"):
        nfail = 0
        for name, steps, nL, cap, wf in build_matrix():
            fails = compare_sequence_gpu(name, steps, nL, cap, wf, dev, backend)
            all_fails += fails
            nfail += len(fails)
            print(f"  {'PASS' if not fails else 'FAIL'}  {backend:>6}  {name} "
                  f"({len(steps)} step(s))")
        print(f"  -> {backend}: {'PASS' if not nfail else f'FAIL ({nfail})'}")
    time_triton(dev)
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not all_fails else f'FAIL ({len(all_fails)})'}")
    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
