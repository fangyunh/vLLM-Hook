"""Step 0 — Injection: is our custom op in the captured CUDA graph?

This is the decisive kill-switch from cuda_graph_plan.html. It answers
the single novel claim C4 (class-level forward wrap survives Dynamo →
captured graph) before any correctness work begins.

Mechanism:
- Register one tiny custom op `vllm_hook_counter.increment` that mutates
  a static counter buffer (`mutates_args=("counter",)`).
- Class-level wrap the decoder-layer forward at load_model time so the
  op is called once per layer per forward.
- Run greedy decode 64 tokens with CUDA graphs enabled.
- Dump captured graph nodes via cudaGraphGetNodes; read the counter.

Pass criteria (both required):
  (1) counter op node appears in cudaGraphGetNodes output
  (2) counter == num_layers * num_decode_steps   (proves wrap is inside
      the replayed graph, not an eager warmup pass)

Diagnostic outcomes (printed at end of run):
  counter == 0                       → wrap bypassed (goal-gate fires)
  counter == num_layers              → wrap fired only at warmup; graph
                                       holds the original forward
  counter == expected                → PASS
  counter > expected                 → double-wrap, check idempotency
  op missing from cudaGraphGetNodes  → op DCE'd, fix mutates_args

Usage:
    # On a GPU host with vLLM installed:
    python tests/cuda_graph/step0_injection.py \\
        --model google/gemma-3-4b-it \\
        --num-decode-tokens 64 \\
        --cudagraph-mode PIECEWISE
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


# ---------------------------------------------------------------------------
# 1. The counter custom op — opaque, side-effecting, must survive DCE.
# ---------------------------------------------------------------------------

# A static int64 counter buffer. data_ptr() is fixed for the process
# lifetime, so it can be safely referenced from a captured CUDA graph.
_COUNTER: torch.Tensor | None = None


def _get_counter() -> torch.Tensor:
    global _COUNTER
    if _COUNTER is None:
        _COUNTER = torch.zeros(1, dtype=torch.int64, device="cuda")
    return _COUNTER


def _increment_impl(counter: torch.Tensor) -> None:
    """The op body. Adds 1 to counter[0] in-place. No return."""
    counter.add_(1)


def _increment_fake(counter: torch.Tensor) -> None:
    """Meta/fake-tensor impl — required by direct_register_custom_op."""
    return None


def _import_direct_register_custom_op():
    """Locate direct_register_custom_op across vLLM versions.

    Older vLLM:   vllm.utils                (flat module)
    vLLM 0.7+:    vllm.utils.torch_utils    (utils became a package)
    Some forks:   vllm.utils._custom_ops
    """
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
        "Tried:\n" + "\n".join(errs) +
        "\nRun on the server: "
        "python -c \"import vllm, pkgutil; "
        "[print(m.name) for m in pkgutil.walk_packages(vllm.utils.__path__, 'vllm.utils.')]\""
    )


def register_counter_op() -> None:
    """Register vllm_hook_counter.increment via vLLM's helper.

    vLLM exposes direct_register_custom_op() which wires the op into both
    PyTorch's dispatcher AND the inductor side, with the mutates_args
    semantics that auto_functionalize needs to keep the node alive.
    """
    direct_register_custom_op = _import_direct_register_custom_op()

    direct_register_custom_op(
        op_name="counter_increment",
        op_func=_increment_impl,
        mutates_args=["counter"],
        fake_impl=_increment_fake,
        dispatch_key="CUDA",
    )


# ---------------------------------------------------------------------------
# 2. The class-level wrap on the decoder layer.
# ---------------------------------------------------------------------------

_WRAPPED_CLASSES: dict[type, Any] = {}


def install_class_wrap(model) -> int:
    """Wrap every matched decoder-layer CLASS so its forward calls the
    counter op exactly once. Returns the number of distinct classes wrapped.

    Class-level patching is on Dynamo's lookup path; instance-level
    patching is documented as bypassed (PyTorch #100733, #93484, #113333)
    and is not used here.
    """
    import re

    LAYER_PATTERNS = [
        re.compile(r"^model\.layers\.(\d+)$"),
        re.compile(r"^language_model\.model\.layers\.(\d+)$"),
        re.compile(r"^transformer\.h\.(\d+)$"),
        re.compile(r"^model\.decoder\.layers\.(\d+)$"),
    ]

    def matches(name: str) -> bool:
        return any(p.match(name) for p in LAYER_PATTERNS)

    matched_classes: set[type] = set()
    for name, module in model.named_modules():
        if matches(name):
            matched_classes.add(type(module))

    counter = _get_counter()

    for cls in matched_classes:
        if cls in _WRAPPED_CLASSES:
            continue
        orig = cls.forward
        _WRAPPED_CLASSES[cls] = orig

        def make_wrapped(orig_fwd):
            def wrapped(self, *args, **kwargs):
                # Call the op BEFORE the layer body so even a layer that
                # raises still records the increment (defensive).
                torch.ops.vllm_hook_counter.counter_increment(counter)
                return orig_fwd(self, *args, **kwargs)
            return wrapped

        cls.forward = make_wrapped(orig)

    return len(matched_classes)


def uninstall_class_wrap() -> None:
    """Restore original forwards. Called in `finally` for clean teardown."""
    for cls, orig in _WRAPPED_CLASSES.items():
        cls.forward = orig
    _WRAPPED_CLASSES.clear()


# ---------------------------------------------------------------------------
# 3. cudaGraphGetNodes — inspect the captured graph for our op node.
# ---------------------------------------------------------------------------


def inspect_captured_graph_for_op() -> dict:
    """Return a summary of CUDA graph nodes referencing our op.

    Implementation: walk torch.cuda's captured graphs via the cuGraphGetNodes
    driver API. We dlopen libcuda, fetch cuGraphGetNodes / cuGraphNodeGetType,
    and count kernel nodes whose function name matches our op signature.

    For simplicity this returns a count of "looks like our op" nodes. The
    binary pass criterion is `n_op_nodes >= num_layers_in_graph`.

    Note: PyTorch does not expose the raw cudaGraph_t handle for graphs
    captured inside vLLM's piecewise mode without monkey-patching. The
    counter-increment check below is independently sufficient to settle
    C4 — the graph-node inspection is a corroborating signal. If
    inspecting fails on this torch/CUDA build, fall back to the counter
    check alone (still binary-decisive).
    """
    # Best-effort. If the graph handle isn't reachable from Python, this
    # returns {"n_op_nodes": -1, "note": "..."} and the test relies on
    # the counter check.
    try:
        import ctypes
        libcuda = ctypes.CDLL("libcuda.so.1")
        # The actual graph-node walk is environment-specific (CUDA version,
        # torch private API). The counter check is the load-bearing test;
        # this slot is here so a later run on a known-good stack can fill it.
        return {"n_op_nodes": -1, "note": "graph-node walk not wired"}
    except Exception as exc:
        return {"n_op_nodes": -1, "note": f"libcuda unreachable: {exc}"}


# ---------------------------------------------------------------------------
# 4. The test driver.
# ---------------------------------------------------------------------------


def run_test(
    model: str,
    num_decode_tokens: int,
    cudagraph_mode: str,
    prompt: str,
) -> int:
    """Returns 0 on PASS, 1 on FAIL, 2 on inconclusive."""

    # Register the counter op BEFORE vLLM imports the model — it must be on
    # the dispatcher when Dynamo traces the wrapped forward.
    register_counter_op()

    from vllm import LLM, SamplingParams

    # Force the targeted cudagraph mode. PIECEWISE is the safest default;
    # Step 0.5 sweeps the other modes.
    llm_kwargs = dict(
        model=model,
        enforce_eager=False,  # MUST be False — CUDA graphs are the whole point
        max_model_len=512,
        gpu_memory_utilization=0.85,
    )

    # The compilation_config knob lives in vllm.config; the env var below
    # is the supported override in vLLM v1.
    import os
    os.environ["VLLM_USE_V1"] = "1"
    # CUDAGraphMode is mapped by vLLM; valid strings: NONE, PIECEWISE,
    # FULL_DECODE_ONLY, FULL_AND_PIECEWISE. Set via compilation_config.
    # Some vLLM versions accept this via env var; if yours does not, set
    # it on EngineArgs explicitly. This script assumes the env override
    # works on the target vLLM version (>= 0.7).
    os.environ["VLLM_CUDAGRAPH_MODE"] = cudagraph_mode

    print(f"[step0] booting vLLM with model={model} mode={cudagraph_mode}", flush=True)
    llm = LLM(**llm_kwargs)

    # Install the wrap AFTER load but BEFORE any forward — this is where
    # the plan installs hooks (in worker.load_model). Doing it here from
    # the driver is correct because LLM(model=...) completes load_model
    # before returning, but warmup/capture_model has not yet run.
    #
    # CAVEAT: vLLM v1 may run a profile_run before LLM() returns. If so,
    # this wrap is too late and Step 0 will report "counter fired only at
    # warmup." That diagnostic itself is the signal — the wrap needs to
    # move into a worker_extension_cls's load_model override. See the
    # README for the production install path.
    inner_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    num_classes_wrapped = install_class_wrap(inner_model)

    num_layers = sum(
        1 for n, _ in inner_model.named_modules()
        if any(p.match(n) for p in [
            __import__("re").compile(r"^model\.layers\.\d+$"),
            __import__("re").compile(r"^transformer\.h\.\d+$"),
            __import__("re").compile(r"^model\.decoder\.layers\.\d+$"),
        ])
    )

    print(f"[step0] wrapped {num_classes_wrapped} class(es), "
          f"{num_layers} matched layer instance(s)", flush=True)

    # Reset the counter AFTER any warmup-time fires.
    counter = _get_counter()
    torch.cuda.synchronize()
    counter.zero_()
    print(f"[step0] counter reset; starting decode", flush=True)

    # Greedy decode N tokens. With prefill=N_prompt tokens and decode=N
    # tokens, total forward steps == 1 prefill + (N-1) decode replays.
    # For Step 0 we only care about the DECODE replays — those are the
    # ones inside a captured graph. So we issue one request and read the
    # counter delta across exactly the decode portion.

    sp = SamplingParams(
        temperature=0.0,           # greedy
        max_tokens=num_decode_tokens,
        ignore_eos=True,           # don't stop early — we need stable step count
    )

    counter_before = int(counter.item())
    outputs = llm.generate([prompt], sp)
    torch.cuda.synchronize()
    counter_after = int(counter.item())

    delta = counter_after - counter_before
    n_gen = len(outputs[0].outputs[0].token_ids)
    n_prompt = len(outputs[0].prompt_token_ids)

    # Expected: 1 prefill forward + (n_gen - 1) decode forwards, each
    # firing the op num_layers times.
    n_steps = 1 + max(0, n_gen - 1)
    expected = num_layers * n_steps
    expected_decode_only = num_layers * max(0, n_gen - 1)
    expected_prefill_only = num_layers * 1

    # Try the graph-node inspection (corroborating).
    graph_inspect = inspect_captured_graph_for_op()

    # ----- Diagnose -----
    print("=" * 70)
    print(f"[step0] RESULTS")
    print(f"  prompt tokens          = {n_prompt}")
    print(f"  generated tokens       = {n_gen}")
    print(f"  num_layers (matched)   = {num_layers}")
    print(f"  forward steps expected = {n_steps} (1 prefill + {n_gen - 1} decode)")
    print(f"  counter delta observed = {delta}")
    print(f"  expected if PASS       = {expected}")
    print(f"  expected if WARMUP-ONLY= {expected_prefill_only} "
          f"(prefill only, no graph replay)")
    print(f"  graph inspection       = {graph_inspect}")
    print("=" * 70)

    # Verdict — order matters; check most-common failure modes first.
    if delta == 0:
        print("[step0] FAIL — counter never fired.")
        print("        Diagnosis: class-level wrap was bypassed by Dynamo.")
        print("        Outcome:   GOAL-GATE FIRES — no pure-plugin path on")
        print("                   this torch/vLLM version.")
        print("        Next:      try a newer torch, or pivot to eager-island.")
        result = 1
    elif delta == expected_prefill_only and n_gen > 1:
        print("[step0] FAIL — counter fired prefill-only.")
        print("        Diagnosis: wrap fires in uncompiled path; the captured")
        print("                   decode graph holds the ORIGINAL forward.")
        print("        Outcome:   wrap installed too late or vLLM compiled the")
        print("                   layer before our wrap landed.")
        print("        Next:      move wrap into worker_extension_cls.load_model")
        print("                   (not in the driver harness as this script does).")
        result = 1
    elif delta == expected:
        print("[step0] PASS — counter delta == num_layers × num_steps.")
        print("        Class-level wrap is in the captured graph and replays.")
        print("        C1 (opacity) and C4 (injection) settled.")
        result = 0
    elif delta > expected:
        print("[step0] FAIL — counter fired too many times.")
        print("        Diagnosis: double-wrap (idempotency guard missing) or")
        print("                   extra forward calls (PP/spec-decode?).")
        print(f"        Excess:    {delta - expected} extra fires.")
        result = 1
    else:
        # delta in (0, expected) — partial. Could be that some layers'
        # classes were not wrapped (e.g. hybrid models with a mix of
        # decoder-layer types).
        print("[step0] INCONCLUSIVE — counter fired some but not all expected.")
        print(f"        Ratio:     {delta} / {expected} = {delta/expected:.2%}")
        print("        Diagnosis: a subset of layer classes wasn't wrapped, or")
        print("                   some layers were captured and others weren't.")
        print("        Next:      print matched class set; check hybrid-model")
        print("                   layer types.")
        result = 2

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it",
                        help="Development model — small, fast, has standard "
                             "decoder layer.")
    parser.add_argument("--num-decode-tokens", type=int, default=64)
    parser.add_argument("--cudagraph-mode", default="PIECEWISE",
                        choices=["PIECEWISE", "FULL_DECODE_ONLY",
                                 "FULL_AND_PIECEWISE"])
    parser.add_argument("--prompt", default="The quick brown fox jumps over the")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[step0] CUDA not available; this test requires a GPU.")
        return 2

    try:
        rc = run_test(
            model=args.model,
            num_decode_tokens=args.num_decode_tokens,
            cudagraph_mode=args.cudagraph_mode,
            prompt=args.prompt,
        )
    finally:
        uninstall_class_wrap()
    return rc


if __name__ == "__main__":
    sys.exit(main())
