"""Long-decode Hidden-States capture demo for PROFILING — Granite-8B.

Why this exists (vLLM-Hook-Profiling #2):
  The stock examples/demo_hiddenstate.py captures hidden states with the worker's
  DEFAULT ``hooks_on="prefill"`` and a very short decode (max_tokens=10). At ~270ms
  per rep the few-ms capture cost is buried in GPU jitter, so the inference
  hook/payload tax can't be resolved (it reads negative). This variant:

    1. **Enables capture in BOTH phases** — ``extra_args={"hooks_on": "both"}`` —
       so the residual-stream hook fires on every DECODE step too. The capture cost
       then ACCUMULATES over the decode instead of being a one-shot prefill cost,
       making it large enough to measure cleanly.
    2. **Uses a long decode** — ``max_tokens`` defaults to 128 (env-overridable) —
       so the run is multi-second and steady-clock, lifting the signal above noise.

  last_token vs all_tokens is selected by VLLM_HOOK_CONFIG_FILE exactly as the stock
  demo (defaults to the last_token granite config). The profiler traces the first
  ``save_to_disk=True`` generate below and bakes max_tokens + hooks_on + hs_mode into
  the campaign run.yaml.

Env knobs:
  VLLM_HOOK_DEMO_MODEL      HF id            (default ibm-granite/granite-3.1-8b-instruct)
  VLLM_HOOK_CONFIG_FILE     config json      (default hidden_states/<model>.json = last_token;
                                              point at *_alltok.json for all_tokens)
  VLLM_HOOK_DEMO_MAX_TOKENS decode length    (default 128)
  VLLM_HOOK_DEMO_HOOKS_ON   prefill|decode|both (default both)
"""
import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


if __name__ == "__main__":
    cache_dir = "./cache/"
    hook_dir = "/dev/shm/vllm_hook"
    # Default to Granite-8B (the profiling target), unlike the stock demo's Qwen default.
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{model.split('/')[-1]}.json")
    # Long decode + both-phase capture are the whole point of this profiling variant.
    max_tokens = int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("VLLM_HOOK_DEMO_HOOKS_ON", "both")

    print(f"[longdec-hs] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on}")

    llm = HookLLM(
        model=model,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=config_file,
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        "The capital of France is",
    ]

    print("=" * 50)
    for case in test_cases:
        # THE TRACED CALL: long decode + capture in prefill AND decode. HookLLM
        # preserves hooks_on (it only adds output_hidden_states/hs_mode on top).
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        sp.extra_args = {"hooks_on": hooks_on}

        output = llm.generate(case, sp, save_to_disk=True)
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{case}'  (decode={max_tokens}, hooks_on={hooks_on})")
        print(f"Generated: '{output[0].outputs[0].text.strip()[:160]}'")
        for layer_name, tensors in sorted(stats["hidden_states"].items()):
            t = tensors[0]
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
        llm.llm_engine.reset_prefix_cache()
