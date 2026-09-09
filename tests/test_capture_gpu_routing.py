"""No-GPU unit test for Phase C: capture (QK/HS) GPU routing byte-identity.

The capture routing slab ``capture_index_all[L, c]`` is the position-deterministic ``c+1``
iff layer L is active at column c (the request owning c captures L). Phase C
(VLLM_HOOK_CAPTURE_GPU_ROUTING) replaces the O(num_layers x cap) host build
(HostRegistry.apply_incremental_routing) with an O(reqs) per-slot capture layer-mask + a GPU
scatter (scatter_capture_routing). This test drives BOTH the host router and the scatter
(torch backend, CPU) from the same synthetic assignments and asserts byte-identity across
single/all-layer, multi-layer, heterogeneous multi-request, and padding cases.

    conda activate vllm_hook_env && python tests/test_capture_gpu_routing.py
"""
import os
os.environ["VLLM_HOOK_CAPTURE_GPU_ROUTING"] = "1"   # build the slot mask on CPU for the test

import sys

import numpy as np
import torch

from vllm_hook_plugins.graph.registry import HostRegistry
from vllm_hook_plugins.graph.steer_routing_gpu import scatter_capture_routing

NL, CAP = 12, 64


def _pad(real_n, cap):
    return min(cap, max(real_n, 16))


def _assign(reqs):
    """reqs = [(ntok, layers_or_None), ...] -> (qsl, assignments, real_n)."""
    qsl = [0]
    for n, _ in reqs:
        qsl.append(qsl[-1] + n)
    assigns = [(qsl[i], qsl[i + 1], reqs[i][1]) for i in range(len(reqs))]
    return qsl, assigns, qsl[-1]


def _scatter_slab(reqs, qsl, assigns, real_n, width):
    reg = HostRegistry(NL, CAP, device="cpu")
    bs = len(reqs)
    reg._slot_mask_h[:bs].zero_()
    qn = np.asarray(qsl, dtype=np.int64)
    for (s, e, layers) in assigns:
        if e <= s:
            continue
        i = int(np.searchsorted(qn, s))
        if layers is None:
            reg._slot_mask_h[i, :] = True
        else:
            reg._slot_mask_h[i, list(layers)] = True
    reg.slot_layer_mask[:bs].copy_(reg._slot_mask_h[:bs])
    scatter_capture_routing(torch.tensor(qsl, dtype=torch.int64), reg.slot_layer_mask,
                            reg.capture_index_all, real_n, width, backend="torch")
    return reg.capture_index_all[:, :width]


def _host_slab(assigns, width):
    ref = HostRegistry(NL, CAP, device="cpu")
    ref.apply_incremental_routing(assigns, width)
    return ref.capture_index_all[:, :width]


CASES = {
    "single_all": [(5, None)],
    "single_layer": [(5, (3,))],
    "multi_layer": [(5, (1, 3, 5))],
    "hetero": [(3, (1, 2)), (4, None), (2, (0, 7, 11))],
    "decode_mixed": [(1, (4,)), (1, None), (1, (4, 9)), (1, (4,))],
    "padded": [(4, (2, 6))],
}


def main():
    fails = []
    for name, reqs in CASES.items():
        qsl, assigns, real_n = _assign(reqs)
        width = _pad(real_n, CAP)
        got = _scatter_slab(reqs, qsl, assigns, real_n, width)
        ref = _host_slab(assigns, width)
        if torch.equal(got, ref):
            print(f"PASS  {name}")
        else:
            fails.append(name)
            print(f"FAIL  {name} (max|d|={(got - ref).abs().max().item()})")
    print("=" * 50)
    ok = not fails
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} ({len(CASES) - len(fails)}/{len(CASES)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
