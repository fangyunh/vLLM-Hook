"""v0.3.0 hidden-states CUDA-graph capture — developer verification harness.

Boots HookLLM with the hidden-states probe worker under enforce_eager=False and
compilation_config cudagraph_mode=PIECEWISE, then asserts the new graph capture
path (hs_probe custom op on the decoder-layer output, registered as a splitting
op) actually produced real, request-specific hidden states.

This is the HS analogue of test_qk_graph_capture.py. It is RED on a path where
the op is folded into a replayed cudagraph segment (capture skipped -> empty ->
FileNotFoundError in analyze) and GREEN once the splitting-op seam lands.

GUARDS:
  * TORCHDYNAMO_DISABLE=0 forced first (beats hook_llm's setdefault to "1").
  * VLLM_HOOK_ALLOW_CUDAGRAPH=1 REQUIRED (else the plugin forces enforce_eager).

WHAT IS ASSERTED:
  1. analyze() returns a non-empty hidden_states dict per case (capture non-empty).
  2. Per-layer norms are finite and non-zero.
  3. Two different prompts yield DIFFERENT hidden states (request-specific, not a
     constant/empty artifact).

Exit code: 0 on PASS, 1 on FAIL.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # MUST precede plugin import

import getpass
import multiprocessing as mp
import sys
import time

import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


def _fail(msg):
    print(f"\n[hs-graph] FAIL: {msg}")
    print("[hs-graph] VERDICT: FAIL")
    sys.exit(1)


def main():
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{model.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()
    hook_dir = os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}")
    dtype = torch.float if "Qwen2-1.5B" in model else torch.float16

    if os.environ.get("VLLM_HOOK_ALLOW_CUDAGRAPH") != "1":
        _fail("VLLM_HOOK_ALLOW_CUDAGRAPH=1 is NOT set — the plugin would force "
              "enforce_eager=True and disable CUDA graphs, making this test "
              "meaningless.")

    print("=" * 72)
    print(f"[hs-graph] model                     = {model}")
    print(f"[hs-graph] enforce_eager             = False  (CUDA graph ON)")
    print(f"[hs-graph] cudagraph_mode            = {cudagraph_mode}")
    print(f"[hs-graph] config_file               = {config_file}")
    print(f"[hs-graph] VLLM_HOOK_ALLOW_CUDAGRAPH = {os.environ.get('VLLM_HOOK_ALLOW_CUDAGRAPH')}")
    print("=" * 72)

    extra = {}
    if cudagraph_mode and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    try:
        llm = HookLLM(
            model=model,
            worker_name="probe_hidden_states",
            analyzer_name="hidden_states",
            config_file=config_file,
            download_dir="./cache/",
            hook_dir=hook_dir,
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=False,
            enable_prefix_caching=True,
            enable_hook=True,
            tensor_parallel_size=1,
            **extra,
        )
    except Exception as e:
        _fail(f"ENGINE BOOT FAILED under graph mode {cudagraph_mode}: "
              f"{type(e).__name__}: {e}")

    print("\n[hs-graph] engine booted under CUDA-graph mode. Running HS capture...\n")

    cases = [
        "The capital of France is",
        "Quantum computing leverages superposition to",
    ]

    summaries = []
    for case in cases:
        print("=" * 56)
        print(f"[case] prompt: '{case}'")
        t0 = time.time()
        out = llm.generate(case, SamplingParams(temperature=0.0, max_tokens=8),
                           save_to_disk=True)
        t1 = time.time()
        print(f"[case] generation runtime: {(t1 - t0):.3f}s")

        stats = llm.analyze(analyzer_spec={"reduce": "norm"})
        if not stats or "hidden_states" not in stats or not stats["hidden_states"]:
            _fail(f"analyze() returned empty hidden_states for '{case}' — capture "
                  f"was empty (the Dynamo-bypass failure this test guards against).")

        hs = stats["hidden_states"]
        # Sum the per-(layer, request[0]) norms into one scalar fingerprint.
        norms = {}
        total = 0.0
        for layer, vals in sorted(hs.items()):
            v = float(vals[0]) if isinstance(vals, (list, tuple)) else float(vals)
            norms[layer] = v
            total += v
            if not (v == v) or v in (float("inf"), float("-inf")):
                _fail(f"non-finite norm {v} for layer {layer} — corrupt capture.")
        print(f"[case] captured {len(hs)} layers; per-layer norms: "
              + ", ".join(f"{k.split('.')[-1]}={x:.3f}" for k, x in norms.items()))
        print(f"[case] norm-sum fingerprint: {total:.6f}")
        summaries.append(total)
        llm.llm_engine.reset_prefix_cache()

    print("=" * 56)
    print(f"[hs-graph] case0 fingerprint: {summaries[0]:.6f}")
    print(f"[hs-graph] case1 fingerprint: {summaries[1]:.6f}")
    diff = abs(summaries[0] - summaries[1])
    print(f"[hs-graph] |difference|     : {diff:.6f}")

    ZERO_TOL = 1e-9
    DIFF_TOL = 1e-4
    if abs(summaries[0]) < ZERO_TOL and abs(summaries[1]) < ZERO_TOL:
        _fail("both fingerprints ~0 — analyzer ran on empty/zero hidden states.")
    if diff < DIFF_TOL:
        _fail(f"the two prompts produced identical hidden states within {DIFF_TOL} "
              f"(diff={diff:.2e}) — capture is not request-specific.")

    print("\n" + "=" * 72)
    print("[hs-graph] VERDICT: PASS")
    print("[hs-graph] CUDA-graph hidden-state capture is non-empty, finite, and "
          "request-specific.")
    print("=" * 72)
    sys.exit(0)


if __name__ == "__main__":
    main()
