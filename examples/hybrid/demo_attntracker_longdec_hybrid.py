"""Long-decode Attention-Tracker (Q/K capture) under HYBRID (CUDA-graph) mode — Granite-8B.

Hybrid counterpart of examples/profiling_longdecode/demo_attntracker_longdec.py.
Same workload (long decode=128, capture in BOTH prefill+decode, same instruction/
data and data_range), the only changes being the three Hybrid switches:

  1. ``VLLM_HOOK_ALLOW_CUDAGRAPH=1`` (before plugin import).
  2. ``enforce_eager=False``.
  3. ``compilation_config={"cudagraph_mode": "PIECEWISE"}``.

Capture is unchanged: the ``vllm_hook::qk_probe`` splitting op runs as an eager seam
and writes the SAME worker buckets the eager path uses, so the attention-tracker
analyzer is identical. Proves the QK Hybrid path on a real 8B model.

Self-check: PASS iff the attention-tracker score is finite (capture produced
usable Q/K for the analyzer).

Env knobs (identical to the eager long-decode demo):
  VLLM_HOOK_DEMO_MODEL      HF id            (default ibm-granite/granite-3.1-8b-instruct)
  VLLM_HOOK_CONFIG_FILE     config json      (default attention_tracker/<model>.json = last_token)
  VLLM_HOOK_DEMO_MAX_TOKENS decode length    (default 128)
  VLLM_HOOK_DEMO_HOOKS_ON   prefill|decode|both (default both)
"""
import os
import multiprocessing as mp
import time

import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")
# (1) ARM HYBRID MODE — before the plugin imports.
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"

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
    max_tokens = int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("VLLM_HOOK_DEMO_HOOKS_ON", "both")
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")

    dtype_map = {
        "mistralai/Mistral-7B-Instruct-v0.3": torch.float16,
        "ibm-granite/granite-3.1-8b-instruct": torch.float16,
        "Qwen/Qwen2-1.5B-Instruct": torch.float,
    }

    print(f"[hybrid-attn] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on} cudagraph_mode={cudagraph_mode}")

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
        # (2) Hybrid: no forced eager. (3) PIECEWISE cudagraphs.
        enforce_eager=False,
        compilation_config={"cudagraph_mode": cudagraph_mode},
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        {
            "instruction": "Analyze and output the sentence attitude:",
            "data": "The weather is nice today, with a soft breeze and sunshine.",
        },
    ]

    ok = True
    for case in test_cases:
        instruction, data = case["instruction"], case["data"]
        print("=" * 50)
        print(f"Instruction: '{instruction}'\nData: '{data}'")
        text, input_range = apply_chat_template_and_get_ranges(llm.tokenizer, model, instruction, data)

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

        score = stats.get("score", [None])[0] if stats else None
        finite = score is not None and bool(torch.isfinite(torch.tensor(float(score))))
        print(f"Attention tracker score: {score}")
        if not finite:
            ok = False
        llm.llm_engine.reset_prefix_cache()

    print("=" * 50)
    print(f"[hybrid-attn] VERDICT: {'PASS' if ok else 'FAIL'} — QK capture under "
          f"CUDA graphs (prefill+decode) {'produced a usable score.' if ok else 'FAILED.'}")
