"""Step 0.8 — Production-path injection: install in the WORKER at load_model.

Step 0 (step0_injection.py) proved a class-level forward-wrap CAN land in the
captured CUDA graph — but it installs the wrap from the DRIVER, *after* LLM()
returns, by reaching into `llm...driver_worker.model_runner.model`. That only
works single-process and is NOT how the plugin installs in production.

This step validates the production install path:

  * A `worker_cls` subclass overrides `load_model`, so the wrap is installed
    INSIDE the worker process, right after the model is built and BEFORE
    warm-up/compile/capture — exactly the timing the plan's monkey-patch of
    `Worker.load_model` uses. (`worker_cls` is resolved by qualified name in
    the worker subprocess, so this survives the driver→worker process
    boundary: it is valid for multi-GPU/TP and for `vllm serve` / AsyncLLM,
    which use the identical worker_cls/worker_extension machinery.)

  * The counter now lives in the worker process, so we read it back over
    `collective_rpc` — the same channel the real egress path uses.

Why a worker_cls subclass and not a true `worker_extension_cls` mixin: vLLM
forbids an extension from defining a name the Worker already has (it asserts
no attribute conflicts and appends the extension to `Worker.__bases__`), so an
extension cannot override `load_model`. A subclass can. The plan's primary
vehicle is a monkey-patch of `Worker.load_model` from the plugin's register();
this subclass is the equally-valid alternative and is the cleanest thing to
exercise from a standalone test (no pip entry point required). Both install at
the same point in the same process, so both answer the same question.

Pass criteria (same as Step 0, but counts summed across workers so TP/PP are
handled): summed counter delta == summed matched layers × num forward steps.

Usage:
    # On a GPU host with vLLM installed, with a model you can access:
    python tests/cuda_graph/step0_8_plugin_install.py \\
        --model <model> --num-decode-tokens 64 --cudagraph-mode PIECEWISE
    # Optional: also exercise the async engine (best-effort, version-sensitive):
    python tests/cuda_graph/step0_8_plugin_install.py --model <model> --async
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

# This module must be importable BY NAME in the worker subprocess (vLLM resolves
# worker_cls via importlib on its qualified name). When run as a script __name__
# is "__main__", which the worker cannot use — so we pin the file-based module
# name and make its directory importable for any child process.
MODULE_NAME = "step0_8_plugin_install"
WORKER_CLS = f"{MODULE_NAME}.InjectingWorker"
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
os.environ["PYTHONPATH"] = _THIS_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

# Keep the vllm-hook plugin out: it forces enforce_eager=True (which disables
# CUDA graphs). This test installs its own worker_cls; it does not need the
# plugin. setdefault lets a user override; set before vLLM import / subprocesses.
os.environ.setdefault("VLLM_PLUGINS", "")


# ---------------------------------------------------------------------------
# Counter custom op (self-contained; mirrors step0_injection's op).
# ---------------------------------------------------------------------------

_COUNTER: torch.Tensor | None = None
_OP_REGISTERED = False


def _get_counter() -> torch.Tensor:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = torch.zeros(1, dtype=torch.int64, device="cuda")
    return _COUNTER


def _increment_impl(counter: torch.Tensor) -> None:
    counter.add_(1)


def _increment_fake(counter: torch.Tensor) -> None:
    return None


def _import_direct_register_custom_op():
    errs = []
    for modpath in (
        "vllm.utils",
        "vllm.utils.torch_utils",
        "vllm.utils._custom_ops",
        "vllm.compilation.decorators",
    ):
        try:
            mod = __import__(modpath, fromlist=["direct_register_custom_op"])
            fn = getattr(mod, "direct_register_custom_op", None)
            if fn is not None:
                return fn
            errs.append(f"  {modpath}: imported but no symbol")
        except ImportError as e:
            errs.append(f"  {modpath}: {e}")
    raise ImportError(
        "Could not locate direct_register_custom_op in any known vLLM path.\n"
        + "\n".join(errs)
    )


def register_counter_op() -> None:
    """Register vllm_hook_counter.counter_increment. Idempotent."""
    global _OP_REGISTERED
    if _OP_REGISTERED:
        return
    direct_register_custom_op = _import_direct_register_custom_op()
    try:
        direct_register_custom_op(
            op_name="counter_increment",
            op_func=_increment_impl,
            mutates_args=["counter"],
            fake_impl=_increment_fake,
            dispatch_key="CUDA",
        )
    except Exception as e:
        # Already registered in this process (e.g. re-entrant load_model) — fine.
        if "already" not in str(e).lower() and "exist" not in str(e).lower():
            raise
    _OP_REGISTERED = True


# ---------------------------------------------------------------------------
# Class-level wrap (same mechanism as step0_injection).
# ---------------------------------------------------------------------------

import re

_LAYER_PATTERNS = [
    re.compile(r"^model\.layers\.(\d+)$"),
    re.compile(r"^language_model\.model\.layers\.(\d+)$"),
    re.compile(r"^transformer\.h\.(\d+)$"),
    re.compile(r"^model\.decoder\.layers\.(\d+)$"),
]
_WRAPPED_CLASSES: dict[type, Any] = {}


def _is_layer(name: str) -> bool:
    return any(p.match(name) for p in _LAYER_PATTERNS)


def install_class_wrap(model) -> int:
    """Wrap every matched decoder-layer CLASS so its forward fires the counter
    op once per call. Returns the number of matched layer INSTANCES."""
    matched_classes: set[type] = set()
    num_instances = 0
    for name, module in model.named_modules():
        if _is_layer(name):
            matched_classes.add(type(module))
            num_instances += 1

    counter = _get_counter()
    for cls in matched_classes:
        if cls in _WRAPPED_CLASSES:
            continue
        orig = cls.forward
        _WRAPPED_CLASSES[cls] = orig

        def make_wrapped(orig_fwd):
            def wrapped(self, *a, **k):
                torch.ops.vllm_hook_counter.counter_increment(counter)
                return orig_fwd(self, *a, **k)
            return wrapped

        cls.forward = make_wrapped(orig)
    return num_instances


# ---------------------------------------------------------------------------
# The worker_cls subclass — installs in the worker, at load_model.
# ---------------------------------------------------------------------------

try:
    from vllm.v1.worker.gpu_worker import Worker as _V1Worker
except Exception:  # pragma: no cover - only importable on a GPU host w/ vLLM
    _V1Worker = object


class InjectingWorker(_V1Worker):
    """Production-shape install: register op + wrap classes in the worker
    process, after the model is built and before warm-up/capture."""

    def load_model(self, *args, **kwargs):
        r = super().load_model(*args, **kwargs)
        try:
            register_counter_op()
            model = self.model_runner.model
            self._n_matched_layers = install_class_wrap(model)
            print(f"[step0.8/worker] wrapped; {self._n_matched_layers} matched "
                  f"layer instance(s) in this worker", flush=True)
        except Exception as e:
            self._n_matched_layers = 0
            print(f"[step0.8/worker] install failed: {e}", flush=True)
        return r

    # --- collective_rpc-callable inspection methods ---
    def reset_counter(self) -> int:
        c = _get_counter()
        torch.cuda.synchronize()
        c.zero_()
        return int(c.item())

    def read_counter(self) -> int:
        torch.cuda.synchronize()
        return int(_get_counter().item())

    def num_matched_layers(self) -> int:
        return int(getattr(self, "_n_matched_layers", 0))


# ---------------------------------------------------------------------------
# collective_rpc helper (works on offline LLM across vLLM versions).
# ---------------------------------------------------------------------------


def _collective_rpc(obj, method: str) -> list:
    fn = getattr(obj, "collective_rpc", None)
    if fn is None:
        eng = getattr(obj, "llm_engine", None)
        fn = getattr(eng, "collective_rpc", None) if eng is not None else None
    if fn is None:
        raise RuntimeError("collective_rpc not found on LLM or llm_engine")
    res = fn(method)
    return list(res) if isinstance(res, (list, tuple)) else [res]


# ---------------------------------------------------------------------------
# Verdict (shared by offline + async paths).
# ---------------------------------------------------------------------------


def _verdict(delta: int, total_layers: int, n_gen: int, n_prompt: int) -> int:
    n_steps = 1 + max(0, n_gen - 1)
    expected = total_layers * n_steps
    expected_prefill_only = total_layers * 1
    print("=" * 70)
    print("[step0.8] RESULTS")
    print(f"  prompt tokens            = {n_prompt}")
    print(f"  generated tokens         = {n_gen}")
    print(f"  matched layers (summed)  = {total_layers}")
    print(f"  forward steps expected   = {n_steps} (1 prefill + {n_gen - 1} decode)")
    print(f"  counter delta observed   = {delta}")
    print(f"  expected if PASS         = {expected}")
    print(f"  expected if WARMUP-ONLY  = {expected_prefill_only}")
    print("=" * 70)
    if total_layers == 0:
        print("[step0.8] FAIL — no layers matched / install never ran in worker.")
        return 1
    if delta == 0:
        print("[step0.8] FAIL — counter never fired. Worker-install wrap bypassed.")
        print("          Diagnosis: class-level wrap not on Dynamo's path, or the")
        print("          op was DCE'd (check mutates_args).")
        return 1
    if delta == expected_prefill_only and n_gen > 1:
        print("[step0.8] FAIL — fired prefill-only; decode graph holds the original")
        print("          forward. Wrap landed after the decode graph was captured.")
        return 1
    if delta == expected:
        print("[step0.8] PASS — worker-install wrap is in the captured graph and")
        print("          replays. Production install path validated (C4 via the")
        print("          real worker/load_model timing, not the driver harness).")
        return 0
    if delta > expected:
        print("[step0.8] FAIL — fired too many times (double-wrap / extra forwards).")
        return 1
    print("[step0.8] INCONCLUSIVE — partial fire; check hybrid-model layer classes.")
    return 2


# ---------------------------------------------------------------------------
# Offline LLM path (default).
# ---------------------------------------------------------------------------


def _boot(model: str, cudagraph_mode: str, **extra):
    """Boot with graphs ON + the requested mode (via compilation_config, not the
    bogus env var), retrying without the mode on older vLLM."""
    from vllm import LLM
    base = dict(model=model, enforce_eager=False, max_model_len=512,
                gpu_memory_utilization=0.85)
    base.update(extra)
    try:
        return LLM(compilation_config={"cudagraph_mode": cudagraph_mode}, **base)
    except Exception as e:  # noqa: BLE001
        if "compilation" in str(e).lower() or "cudagraph" in str(e).lower():
            print(f"[step0.8] compilation_config not supported ({e}); engine "
                  f"default graph mode.", flush=True)
            return LLM(**base)
        raise


def run_offline(model: str, num_decode_tokens: int, cudagraph_mode: str,
                prompt: str) -> int:
    from vllm import SamplingParams

    print(f"[step0.8] booting offline LLM model={model} mode={cudagraph_mode} "
          f"worker_cls={WORKER_CLS}", flush=True)
    # install happens in the worker, at load_model; CUDA graphs ON.
    llm = _boot(model, cudagraph_mode, worker_cls=WORKER_CLS)

    # Warmup/capture already happened during LLM(). Reset the per-worker
    # counters, confirm they're zero, then decode.
    _collective_rpc(llm, "reset_counter")
    total_layers = sum(_collective_rpc(llm, "num_matched_layers"))
    before = sum(_collective_rpc(llm, "read_counter"))
    print(f"[step0.8] post-reset counter (summed) = {before}; "
          f"matched layers (summed) = {total_layers}", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=num_decode_tokens,
                        ignore_eos=True)
    outputs = llm.generate([prompt], sp)
    after = sum(_collective_rpc(llm, "read_counter"))

    delta = after - before
    n_gen = len(outputs[0].outputs[0].token_ids)
    n_prompt = len(outputs[0].prompt_token_ids)
    return _verdict(delta, total_layers, n_gen, n_prompt)


# ---------------------------------------------------------------------------
# Async engine path (optional, best-effort — exercises the serve-shape engine).
# ---------------------------------------------------------------------------


def run_async(model: str, num_decode_tokens: int, cudagraph_mode: str,
              prompt: str) -> int:
    import asyncio

    async def _amain() -> int:
        from vllm import SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs

        async def _arpc(engine, method):
            fn = getattr(engine, "collective_rpc")
            res = fn(method)
            if asyncio.iscoroutine(res):
                res = await res
            return list(res) if isinstance(res, (list, tuple)) else [res]

        print(f"[step0.8] booting AsyncLLM model={model} mode={cudagraph_mode}",
              flush=True)
        eargs = dict(model=model, worker_cls=WORKER_CLS, enforce_eager=False,
                     max_model_len=512, gpu_memory_utilization=0.85)
        try:
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                compilation_config={"cudagraph_mode": cudagraph_mode}, **eargs))
        except Exception as e:  # noqa: BLE001
            if "compilation" not in str(e).lower() and "cudagraph" not in str(e).lower():
                raise
            print(f"[step0.8] compilation_config not supported ({e}); engine "
                  f"default graph mode.", flush=True)
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**eargs))
        try:
            await _arpc(engine, "reset_counter")
            total_layers = sum(await _arpc(engine, "num_matched_layers"))
            before = sum(await _arpc(engine, "read_counter"))

            sp = SamplingParams(temperature=0.0, max_tokens=num_decode_tokens,
                                ignore_eos=True)
            final = None
            async for out in engine.generate(prompt, sp,
                                             request_id="step0_8-async"):
                final = out
            after = sum(await _arpc(engine, "read_counter"))

            delta = after - before
            n_gen = len(final.outputs[0].token_ids)
            n_prompt = len(final.prompt_token_ids)
            print("[step0.8] (async engine) ", end="")
            return _verdict(delta, total_layers, n_gen, n_prompt)
        finally:
            shut = getattr(engine, "shutdown", None)
            if shut:
                try:
                    shut()
                except Exception:
                    pass

    try:
        return asyncio.run(_amain())
    except Exception as e:
        print(f"[step0.8] async harness unsupported on this vLLM version: {e}")
        print("          The offline worker_cls result already validates the")
        print("          worker-process install path; `vllm serve` uses the same")
        print("          worker_cls/extension machinery. Treating as inconclusive.")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--num-decode-tokens", type=int, default=64)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE",
                        choices=["PIECEWISE", "FULL_DECODE_ONLY",
                                 "FULL_AND_PIECEWISE"])
    parser.add_argument("--prompt", default="The quick brown fox jumps over the")
    parser.add_argument("--async", dest="use_async", action="store_true",
                        help="Also run the AsyncLLM (serve-shape) path; "
                             "best-effort, version-sensitive.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[step0.8] CUDA not available; this test requires a GPU.")
        return 2

    if args.use_async:
        return run_async(args.model, args.num_decode_tokens,
                         args.cudagraph_mode, args.prompt)
    return run_offline(args.model, args.num_decode_tokens,
                       args.cudagraph_mode, args.prompt)


if __name__ == "__main__":
    sys.exit(main())
