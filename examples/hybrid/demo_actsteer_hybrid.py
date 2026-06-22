"""Activation-steering demo under HYBRID (CUDA-graph) mode — Granite-3.1-8B.

Hybrid counterpart of examples/demo_actsteer.py. Identical workload and settings,
the ONLY differences being the three switches that move steering off the legacy
eager path onto the v0.3.0 CUDA-graph (Hybrid) path:

  1. ``VLLM_HOOK_ALLOW_CUDAGRAPH=1`` (set below, before the plugin imports) — arms
     the graph install path and stops the plugin forcing ``enforce_eager``.
  2. ``enforce_eager=False`` — let vLLM compile + capture CUDA graphs.
  3. ``compilation_config={"cudagraph_mode": "PIECEWISE"}`` — the mode the
     steer_residual splitting op is proven under.

Steering itself is unchanged: the eager ``steering_hook`` math runs inside the
``vllm_hook::steer_residual`` op (an eager seam between graph segments) and mutates
the residual IN PLACE, so the edit propagates through the cudagraph-replayed
downstream layers — on prefill AND every decode step.

Granite's config uses ``method="adjust_rs"`` at ``optimal_layer=15`` (vs the Qwen2
demo's add_vector), so this also exercises the adjust_rs path under Hybrid mode.

Self-check: PASS iff the run completes under cudagraph AND steering changed the
output (steered text != unsteered text) for at least one case.

Env knobs (same as the stock demo, plus the model default):
  VLLM_HOOK_DEMO_MODEL       HF id           (default ibm-granite/granite-3.1-8b-instruct)
  VLLM_HOOK_CONFIG_FILE      steer config    (default activation_steer/<model>.json)
  VLLM_HOOK_DEMO_MAX_TOKENS  decode length   (default 2048, like the stock demo)
"""
import os
import multiprocessing as mp

import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# (1) ARM HYBRID MODE — must be set before vllm_hook_plugins imports.
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


if __name__ == "__main__":
    cache_dir = "./cache/"
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f'model_configs/activation_steer/{model.split("/")[-1]}.json')
    max_tokens = int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "2048"))
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")

    dtype_map = {
        'microsoft/Phi-3-mini-4k-instruct': 'auto',
        'mistralai/Mistral-7B-Instruct-v0.3': torch.float16,
        'ibm-granite/granite-3.1-8b-instruct': torch.float16,
        'Qwen/Qwen2-1.5B-Instruct': torch.float,
    }

    print(f"[hybrid-steer] model={model} config={config_file} "
          f"max_tokens={max_tokens} cudagraph_mode={cudagraph_mode}")

    llm = HookLLM(
        model=model,
        worker_name="steer_hook_act",
        config_file=config_file,
        download_dir=cache_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map.get(model, torch.float16),
        # (2) Hybrid mode: do NOT force eager. (3) PIECEWISE cudagraphs.
        enforce_eager=False,
        compilation_config={"cudagraph_mode": cudagraph_mode},
        # Steering mutates the residual at layer 15 -> the KV cached for layers >15
        # would be the STEERED KV; disable prefix caching so a later unsteered call
        # can't reuse steered KV (the eager demo resets the cache for the same reason).
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
        "What is the difference between the 13 colonies and the other British colonies in North America? Your answer must contain exactly 6 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n...\n* Bullet point fix.",
    ]

    any_changed = False
    for idx, case in enumerate(test_cases):
        print("=" * 60)
        messages = [{"role": "user", "content": case}]
        example = llm.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
        )

        steered = llm.generate(example, sampling_params)[0].outputs[0].text
        print("With activation steering (Hybrid/cudagraph):")
        print(steered[:400])

        unsteered = llm.generate(example, sampling_params, use_hook=False)[0].outputs[0].text
        print("\nWithout activation steering (Hybrid/cudagraph):")
        print(unsteered[:400])

        changed = steered.strip() != unsteered.strip()
        any_changed = any_changed or changed
        print(f"\n[hybrid-steer] case {idx}: steered_changed_output={changed}")

    print("=" * 60)
    if any_changed:
        print("[hybrid-steer] VERDICT: PASS — steering runs under CUDA graphs and "
              "changes the output.")
    else:
        print("[hybrid-steer] VERDICT: FAIL — steering had no effect under Hybrid "
              "mode (check the steer op fired / raise the dummy avg_proj).")
