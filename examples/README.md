# Building your own demo

A demo is one Python file: configure an engine, generate, read back what was captured.

## 1. Skeleton

Copy this. The four lines before the `vllm` import are mandatory — set them first or the engine
starts with the wrong runtime.

```python
import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM

if __name__ == "__main__":
    llm = HookLLM(
        model="Qwen/Qwen2-1.5B-Instruct",
        worker_name="probe_hidden_states",     # what to capture
        analyzer_name="hidden_states",         # what to do with it
        config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
        download_dir="./cache/",
        dtype=torch.float16,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        enable_hook=True,
        enforce_eager=True,
    )

    out = llm.generate("The capital of France is",
                       SamplingParams(temperature=0.0, max_tokens=10),
                       save_to_disk=True)
    stats = llm.analyze(analyzer_spec={"reduce": "none"})

    print(out[0].outputs[0].text)
    for layer, tensors in sorted(stats["hidden_states"].items()):
        print(layer, tuple(tensors[0].shape))
```

Run from the repo root: `python examples/my_demo.py`.

## 2. Pick a worker and analyzer

`worker_name` decides what is captured. `analyzer_name` decides what happens to it, and is
optional — omit it if you only want the raw tensors.

| `worker_name` | captures | config section | reference demo |
|---|---|---|---|
| `probe_hidden_states` | hidden states | `hidden_states` | `demo_hiddenstate.py` |
| `probe_hook_qk` | attention Q/K | `hookq` | `demo_attntracker.py` |
| `steer_hook_act` | — (steers instead) | `steering` | `demo_actsteer.py` |
| `probe_spotlight` | — (steers attention) | — | `demo_spotlight.py` |
| `token_highlighter` | gradient influence | — | `demo_token_highlighter.py` |

| `analyzer_name` | reference demo |
|---|---|
| `hidden_states` | `demo_hiddenstate.py` |
| `attn_tracker` | `demo_attntracker.py` |
| `core_reranker` | `demo_corer.py` |
| `hnode_hallucination` | `demo_halludetect.py` |
| `science_hallucination` | `demo_scihal.py` |
| `token_highlighter` | `demo_token_highlighter.py` |

## 3. Write the config

One JSON under `model_configs/<use_case>/<model_name>.json`. Only the section matching your
worker is read.

Capture hidden states from layers 1–4, last token only:

```json
{
  "model_info": { "name": "Qwen/Qwen2-1.5B-Instruct" },
  "hidden_states": { "layers": [1, 2, 3, 4], "mode": "last_token" }
}
```

`mode` is `last_token` or `all_tokens`. For QK use `"hookq": {"hookq_mode": "last_token"}`.
For steering use `"steering": {"method": "add_vector", "coefficient": 1.0, "optimal_layer": 14,
"vector_path": "steering_vectors/qwen2_dummy.pt"}`.

Copy the closest existing file in `model_configs/` rather than writing one from scratch.

## 4. Get your data back

Two paths. **Pick one — mixing them silently returns nothing.**

```python
# Disk: artifacts written, analyze() reads them back.
out   = llm.generate(prompt, sp, save_to_disk=True)
stats = llm.analyze(analyzer_spec={...})

# In-memory: captures ride back on the output object.
out   = llm.generate(prompt, sp, save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})
```

Under `save_to_disk=True`, `out[0].probes` is always `None`.

For raw tensors with no analysis, use the `hidden_states` analyzer with `reduce="none"` — it
returns what was captured, unchanged.

## 5. Optional: FULL CUDA-graph mode

Capture and steering run under CUDA graphs instead of eager. Off by default.

```bash
VLLM_HOOK_ALLOW_CUDAGRAPH=1 python examples/my_demo.py
```

Your demo must also pass `enforce_eager=False`; without the env var the plugin forces eager
regardless. See `demo_capture_ring.py`.

## 6. Gotchas

- Set the `mp.set_start_method` / env lines **before** importing `vllm`.
- Run from the repo root — config and vector paths are relative to it.
- `enforce_eager=True` is required unless you enabled graph mode.
- Call `llm.llm_engine.reset_prefix_cache()` between prompts if you capture the same prefix twice.
- Profiler counters need `VLLM_HOOK_PROFILE=1`; without it they are no-ops.
- Performance levers: `from vllm_hook_plugins.optimizations import describe; print(describe())`.

## 7. Notebooks

`notebooks/` has the same demos in notebook form. Register the kernel first:

```bash
pip install ipykernel
python -m ipykernel install --user --name vllm_hook_env
```
