"""No-GPU unit test: an adjust_rs vector's ``avg_proj`` may be a tensor OR a float.

The graph loader coerces either (``float(ap.item()) if torch.is_tensor(ap) else
float(ap)``); the eager use site does ``data["avg_proj"].to(device, dtype)``, which needs
a tensor. Before the fix a vector storing a plain float therefore ran fine under FULL
CUDA graphs — the production path — and raised inside the forward on eager, the path you
reach when you go to VALIDATE it. Every shipped vector stores a 0-d tensor, so this is
about the containers a user can hand us, not about anything in the tree.

    conda activate vllm_hook_env && python tests/test_steer_avgproj_container.py
"""
import sys

import numpy as np
import torch

from vllm_hook_plugins.workers import steer_activation_worker as saw


CFG = {"method": "adjust_rs", "optimal_layer": 0, "coefficient": 1.0,
       "vector_path": "unit-test-vector"}
DIR = np.arange(4, dtype=np.float32)


class _Worker:
    """Just the two attributes _vector_cache_for touches."""

    def __init__(self):
        self._vector_cache = {}


def _load_with(avg_proj):
    """Run the real _vector_cache_for against a vector carrying this avg_proj."""
    w = _Worker()
    real = saw._load_steering_vector
    saw._load_steering_vector = lambda path: {"dir": DIR, "avg_proj": avg_proj}
    try:
        return saw.SteerHookActWorker._vector_cache_for(w, CFG)
    finally:
        saw._load_steering_vector = real


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_float_avg_proj_survives_the_eager_use_site():
    data = _load_with(20.0)
    _assert(torch.is_tensor(data["avg_proj"]),
            "a float avg_proj must be normalized to a tensor at load")
    # The exact call the eager path makes; this is what used to raise AttributeError.
    got = data["avg_proj"].to("cpu", dtype=torch.float32)
    _assert(float(got) == 20.0, f"value changed through normalization: {float(got)}")


def test_tensor_avg_proj_is_unchanged():
    src = torch.tensor(20.0)
    data = _load_with(src)
    _assert(data["avg_proj"] is src,
            "as_tensor must be a no-op on a tensor (shipped vectors must not be copied)")


def test_both_containers_agree():
    a = _load_with(20.0)["avg_proj"].to("cpu", dtype=torch.float32)
    b = _load_with(torch.tensor(20.0))["avg_proj"].to("cpu", dtype=torch.float32)
    _assert(torch.equal(a, b), "float and tensor containers must resolve identically")


def test_add_vector_still_carries_no_avg_proj():
    """Only adjust_rs reads avg_proj; add_vector must not start requiring one."""
    w = _Worker()
    real = saw._load_steering_vector
    saw._load_steering_vector = lambda path: {"dir": DIR}
    try:
        data = saw.SteerHookActWorker._vector_cache_for(
            w, dict(CFG, method="add_vector"))
    finally:
        saw._load_steering_vector = real
    _assert("avg_proj" not in data, "add_vector vectors must not gain an avg_proj key")


def main():
    tests = [
        test_float_avg_proj_survives_the_eager_use_site,
        test_tensor_avg_proj_is_unchanged,
        test_both_containers_agree,
        test_add_vector_still_carries_no_avg_proj,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'} "
          f"({len(tests) - failures}/{len(tests)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
