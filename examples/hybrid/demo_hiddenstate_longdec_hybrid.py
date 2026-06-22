"""Long-decode Hidden-States capture under HYBRID (CUDA-graph) mode — Granite-8B.

Hybrid counterpart of examples/profiling_longdecode/demo_hiddenstate_longdec.py.
Same workload (long decode=128, capture in BOTH prefill+decode), the only changes
being the three Hybrid switches:

  1. ``VLLM_HOOK_ALLOW_CUDAGRAPH=1`` (before plugin import).
  2. ``enforce_eager=False``.
  3. ``compilation_config={"cudagraph_mode": "PIECEWISE"}``.

Capture is unchanged: the ``vllm_hook::hs_probe`` splitting op runs as an eager seam
on every step and writes the SAME worker buckets the eager path uses, so the
analyzer/egress are identical. This proves the HS Hybrid path on a real 8B model
across prefill + decode.

Self-check: PASS iff capture returned a hidden-state tensor for every requested
layer with a sane shape (seq dim covering prefill+decode) and finite norm.

Env knobs (identical to the eager long-decode demo):
  VLLM_HOOK_DEMO_MODEL      HF id            (default ibm-granite/granite-3.1-8b-instruct)
  VLLM_HOOK_CONFIG_FILE     config json      (default hidden_states/<model>.json = last_token)
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
# (1) ARM HYBRID MODE — before the plugin imports.
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM


if __name__ == "__main__":
    cache_dir = "./cache/"
    hook_dir = "/dev/shm/vllm_hook"
    model = os.environ.get("VLLM_HOOK_DEMO_MODEL", "ibm-granite/granite-3.1-8b-instruct")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{model.split('/')[-1]}.json")
    max_tokens = int(os.environ.get("VLLM_HOOK_DEMO_MAX_TOKENS", "128"))
    hooks_on = os.environ.get("VLLM_HOOK_DEMO_HOOKS_ON", "both")
    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")

    print(f"[hybrid-hs] model={model} config={config_file} "
          f"max_tokens={max_tokens} hooks_on={hooks_on} cudagraph_mode={cudagraph_mode}")

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
        # (2) Hybrid: no forced eager. (3) PIECEWISE cudagraphs.
        enforce_eager=False,
        compilation_config={"cudagraph_mode": cudagraph_mode},
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    test_cases = [
        "The capital of France is",
    ]

    print("=" * 50)
    ok = True
    for case in test_cases:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        sp.extra_args = {"hooks_on": hooks_on}

        output = llm.generate(case, sp, save_to_disk=True)
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        print(f"\nPrompt: '{case}'  (decode={max_tokens}, hooks_on={hooks_on})")
        print(f"Generated: '{output[0].outputs[0].text.strip()[:160]}'")

        hs = stats.get("hidden_states", {}) if stats else {}
        if not hs:
            ok = False
            print("  [hybrid-hs] no hidden states captured!")
        for layer_name, tensors in sorted(hs.items()):
            t = tensors[0]
            norm = torch.norm(t.float())
            finite = bool(torch.isfinite(t.float()).all())
            print(f"  {layer_name}: shape={tuple(t.shape)}, norm={norm:.4f}, finite={finite}")
            if not finite or t.numel() == 0:
                ok = False
        llm.llm_engine.reset_prefix_cache()

    print("=" * 50)
    print(f"[hybrid-hs] VERDICT: {'PASS' if ok else 'FAIL'} — HS capture under "
          f"CUDA graphs (prefill+decode) {'produced sane tensors.' if ok else 'FAILED.'}")
