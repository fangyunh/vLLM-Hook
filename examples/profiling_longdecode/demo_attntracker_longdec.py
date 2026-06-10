"""Long-decode Attention-Tracker (Q/K capture) demo for PROFILING — Granite-8B.

Why this exists (vLLM-Hook-Profiling #2):
  The stock examples/demo_attntracker.py captures Q/K with the worker's DEFAULT
  ``hooks_on="prefill"`` and a short decode (max_tokens=50). On the short-decode
  cells the per-rep GPU jitter swamps the few-ms capture cost, so the inference
  hook/payload tax can't be resolved (it even reads negative). This variant:

    1. **Enables capture in BOTH phases** — ``extra_args={"hooks_on": "both"}`` —
       so the hook fires on every DECODE step too, not just prefill. The capture
       cost then ACCUMULATES over the decode (like the steer workload) instead of
       being a one-shot prefill cost, making it large enough to measure.
    2. **Uses a long decode** — ``max_tokens`` defaults to 128 (env-overridable) —
       so the run is multi-second and steady-clock, lifting the signal above the
       short-cell noise.

  last_token vs all_tokens is selected by VLLM_HOOK_CONFIG_FILE exactly as the
  stock demo (defaults to the last_token granite config). The profiler traces the
  first ``save_to_disk=True`` generate below and bakes max_tokens + hooks_on +
  hookq_mode into the campaign run.yaml.

Env knobs:
  VLLM_HOOK_DEMO_MODEL      HF id            (default ibm-granite/granite-3.1-8b-instruct)
  VLLM_HOOK_CONFIG_FILE     config json      (default attention_tracker/<model>.json = last_token;
                                              point at *_alltok.json for all_tokens)
  VLLM_HOOK_DEMO_MAX_TOKENS decode length    (default 128)
  VLLM_HOOK_DEMO_HOOKS_ON   prefill|decode|both (default both)
"""
import os
import multiprocessing as mp
import torch
import time

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


def apply_chat_template_and_get_ranges(tokenizer, model_name: str, instruction: str, data: str):
    """Following https://github.com/khhung-906/Attention-Tracker/blob/main/models/attn_model.py"""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": "Data: " + data},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    instruction_len = len(tokenizer.encode(instruction))
    data_len = len(tokenizer.encode(data))
    if "granite-3.1" in model_name:
        data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
    elif "Mistral-7B" in model_name:
        data_range = ((3, 3 + instruction_len), (-1 - data_len, -1))
    elif "Qwen2-1.5B" in model_name:
        data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
    else:
        raise NotImplementedError
    return text, data_range


if __name__ == "__main__":
    cache_dir = "./cache/"
    hook_dir = "/dev/shm/vllm_hook"
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f'model_configs/attention_tracker/{model.split("/")[-1]}.json')
    # Long decode + both-phase capture are the whole point of this profiling variant.
    max_tokens = int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("VLLM_HOOK_DEMO_HOOKS_ON", "both")

    dtype_map = {
        "mistralai/Mistral-7B-Instruct-v0.3": torch.float16,
        "ibm-granite/granite-3.1-8b-instruct": torch.float16,
        "Qwen/Qwen2-1.5B-Instruct": torch.float,
    }

    print(f"[longdec-attn] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on}")

    llm = HookLLM(
        model=model,
        worker_name="probe_hook_qk",
        analyzer_name="attn_tracker",
        config_file=config_file,
        download_dir=cache_dir,
        hook_dir=hook_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map.get(model, torch.float16),
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        {
            "instruction": "Analyze and output the sentence attitude:",
            "data": "The weather is nice today, with a soft breeze and sunshine.",
        },
    ]

    for case in test_cases:
        instruction, data = case["instruction"], case["data"]
        print("=" * 50)
        print(f"Instruction: '{instruction}'\nData: '{data}'")
        text, input_range = apply_chat_template_and_get_ranges(llm.tokenizer, model, instruction, data)

        # THE TRACED CALL: long decode + capture in prefill AND decode. The hooks_on
        # flag is preserved verbatim into sampling_params.extra_args by HookLLM
        # (_build_extra_args adds output_qk/hookq_mode but never overwrites hooks_on).
        sp = SamplingParams(temperature=0.1, max_tokens=max_tokens)
        sp.extra_args = {"hooks_on": hooks_on}

        t0 = time.time()
        output = llm.generate(text, sp, save_to_disk=True)
        t1 = time.time()
        print(f"hook llm generation runtime: {(t1 - t0):.3f}s  (decode={max_tokens}, hooks_on={hooks_on})")

        stats = llm.analyze(
            probes=getattr(output[0], "probes", None),
            analyzer_spec={"input_range": input_range, "attn_func": "sum_normalize"})
        print(f"hook llm analysis runtime: {(time.time() - t1):.3f}s")
        print(f"Generated: {output[0].outputs[0].text[:160]!r}")
        print(f"Attention tracker score: {stats['score'][0]:.3f}")
        llm.llm_engine.reset_prefix_cache()
