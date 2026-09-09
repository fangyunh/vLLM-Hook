import os
import json
import multiprocessing as mp
import time
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_hook_plugins import HookLLM
from vllm import SamplingParams


def _print_evidence(elapsed_s: float, n_tokens: int) -> None:
    """Compact end-of-run evidence: wall-clock/decode-step timing, the profiler's
    counters (meaningful only with VLLM_HOOK_PROFILE=1), and the active optimization
    lever state -- so a reader can tell from the log whether anything actually ran
    differently, not just that the script printed text."""
    per_step = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"[evidence] generate: {elapsed_s * 1000:.1f} ms total, "
          f"{per_step:.2f} ms/decode-step over {n_tokens} tokens")

    from vllm_hook_plugins._profiler import PROF
    snap = PROF.summary_only()
    if snap["enabled"]:
        print(f"[evidence] profiler counters: {snap['counters']}")
    else:
        print("[evidence] profiler disabled -- set VLLM_HOOK_PROFILE=1 to see hook/ring counters")

    from vllm_hook_plugins.optimizations import describe
    print("[evidence] active optimization levers:")
    print(describe())


if __name__ == "__main__":

    cache_dir = "./cache/"
    # Profiler-friendly: model + steer config are overridable via env so the same
    # demo can be traced for different models (set VLLM_HOOK_DEMO_MODEL /
    # VLLM_HOOK_CONFIG_FILE).
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", 'microsoft/Phi-3-mini-4k-instruct')
    # Resolved once and reused below for both the engine config (HookLLM's
    # config_file=) and the per-request steering override base -- previously the
    # latter recomputed the default path from `model` alone, so it silently
    # diverged from the engine's actual config whenever VLLM_HOOK_CONFIG_FILE
    # was set.
    config_path = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f'model_configs/activation_steer/{model.split("/")[-1]}.json')

    # Graph mode is strictly opt-in: VLLM_HOOK_ALLOW_CUDAGRAPH=1 arms the FULL
    # CUDA-graph capture ring and lets enforce_eager below go False; unset/anything
    # else keeps today's eager default unchanged. Its companion knob,
    # VLLM_HOOK_RING_MAX_BATCHED_TOKENS, only ever LOWERS the scheduler's token
    # budget (byte-identical capture either way) and is left at its "off" default here.
    GRAPH_MODE = os.environ.get("VLLM_HOOK_ALLOW_CUDAGRAPH") == "1"
    print(f"[demo_actsteer] mode={'FULL CUDA-graph capture' if GRAPH_MODE else 'eager'} "
          f"(VLLM_HOOK_ALLOW_CUDAGRAPH={'1' if GRAPH_MODE else '0'})")

    dtype_map = {
        'microsoft/Phi-3-mini-4k-instruct': 'auto',
        'mistralai/Mistral-7B-Instruct-v0.3': torch.float16,
        'ibm-granite/granite-3.1-8b-instruct': torch.float16,
        'Qwen/Qwen2-1.5B-Instruct': torch.float
    }

    llm = HookLLM(
        model=model,
        worker_name="steer_hook_act",
        config_file=config_path,
        download_dir=cache_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map.get(model, torch.float16),
        enforce_eager=not GRAPH_MODE,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1  # the number of gpus
    )

    test_cases = [
        "Write a dialogue between two people, one is dressed up in a ball gown and the other is dressed down in sweats. The two are going to a nightly event. Your answer must contain exactly 3 bullet points in the markdown format (use \"* \" to indicate each bullet) such as:\n* This is the first point.\n* This is the second point.",
        "What is the difference between the 13 colonies and the other British colonies in North America? Your answer must contain exactly 6 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n...\n* Bullet point fix."
    ]

    # Per-request steering: uses the JSON config as-is vs. overrides method+coefficient.
    # Reuses the same config_path resolved above for the engine config.
    with open(config_path) as f:
        config = json.load(f)
    default_config   = config["steering"]
    sampling_params_list = [
        SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
        ),
        SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            stop_token_ids=[llm.tokenizer.eos_token_id, 32007],
            extra_args={"steer": {**default_config  , "method": "add_vector", "coefficient": 10}},
        ),
    ]

    for case, sampling_params in zip(test_cases, sampling_params_list):
        print("=" * 50)
        prompt = case
        messages = [{"role": "user", "content": prompt}]
        example = llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        t0 = time.time()
        output = llm.generate(example, sampling_params)
        elapsed = time.time() - t0
        print("With activation steering:")
        print(output[0].outputs[0].text)
        _print_evidence(elapsed, len(output[0].outputs[0].token_ids))

        llm.llm_engine.reset_prefix_cache()
        output = llm.generate(example, sampling_params, use_hook=False)
        print("Without activation steering:")
        print(output[0].outputs[0].text)
        llm.llm_engine.reset_prefix_cache()


    ### batch processing 
    print("=" * 50)
    print("Batch processing examples...")
    examples = [
        llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": case}], add_generation_prompt=True, tokenize=False
        )
        for case in test_cases
    ]

    t0 = time.time()
    outputs = llm.generate(examples, sampling_params_list)
    elapsed = time.time() - t0
    llm.llm_engine.reset_prefix_cache()
    outputs_original = llm.generate(examples, sampling_params_list, use_hook=False)
    llm.llm_engine.reset_prefix_cache()

    for steered, original in zip(outputs, outputs_original):
        print("=" * 50)
        print("With activation steering:")
        print(steered.outputs[0].text)
        print("Without activation steering:")
        print(original.outputs[0].text)

    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    _print_evidence(elapsed, n_tokens)

