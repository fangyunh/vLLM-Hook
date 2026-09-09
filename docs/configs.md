# vLLM-Hook supported configurations

This document enumerates the supported configs and how to invoke each from user code.

---

## Configuration axes

| Axis | Values | How it's selected |
|---|---|---|
| **Execution path** | `offline` (in-process `HookLLM`) · `serve` (`vllm serve` + `HookClient`) | -|
| **Storage** | `rpc` (in-memory via `collective_rpc`) · `disk` (artifact under `/dev/shm/vllm_hook/<run_id>/`) · `shm` (legacy shared memory, hidden states-only) | per-request `extra_args["save_to_disk"]` (SHM via `VLLM_HOOK_USE_SHM=1`) |
| **Disk format** | `pt` (`torch.save`) · `st` (safetensors ) | `VLLM_HOOK_USE_SAFETENSORS={0,1}` |

> **Async save note:** the old per-request `sync`/`async` save-mode axis (`VLLM_HOOK_ASYNC_SAVE`) has been removed. It is superseded by the **writer process** (`VLLM_HOOK_WRITER_PROCESS`, default **on**) — a persistent child process that serializes and writes disk artifacts off the engine GIL (see the `writer_process` lever in `optimizations.py`). Unlike the old knob, this isn't a per-request axis you opt into: it's a process-wide default that's already on, so it does not appear as a selectable dimension in the coverage matrices below.

---

## Coverage matrix

### Attention tracker 

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `attn-offline-rpc-na`   | offline | rpc  | —  |
| `attn-offline-disk-pt`  | offline | disk | pt |
| `attn-offline-disk-st`  | offline | disk | st |
| `attn-serve-rpc-na`     | serve   | rpc  | —  |
| `attn-serve-disk-pt`    | serve   | disk | pt |
| `attn-serve-disk-st`    | serve   | disk | st |

### Hidden states 

Same 6 axis combinations as above, plus the legacy SHM fast-path:

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `hs-offline-shm-na` | offline | shm  | —  |

SHM is gated by `VLLM_HOOK_USE_SHM=1` and only supports `probe_hidden_states` in `last_token` mode (auto-disabled otherwise; see `shm_utils.py`).

### CoRer

CoRer is intrinsically two-pass and only uses the disk path (the analyzer needs both runs' artifacts on disk to compute the difference). No `rpc` cells.

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `corer-offline-disk-pt` | offline | disk | pt |
| `corer-offline-disk-st` | offline | disk | st |
| `corer-serve-disk-pt`   | serve   | disk | pt |
| `corer-serve-disk-st`   | serve   | disk | st |

### Activation steering 

Steering modifies the residual stream in-place and produces no artifacts, so storage/format/async axes don't apply. Per-request via `extra_args["steer"]`.

| Cell ID | Path |
|---|---|
| `actsteer-offline-na-na` | offline |
| `actsteer-serve-na-na`   | serve   |

---

## Selecting a configuration from user code

All hook activation is **per-request** via `SamplingParams.extra_args` (offline) or `extra_body["vllm_xargs"]` (serve). Different requests in the same batch can use different configs.

The two execution paths are documented below. For each path, the same code shape covers all four use cases — only `worker_name` / `analyzer_name` (offline) or `VLLM_HOOK_WORKER` (serve) varies:

| Use case | `worker_name` / `VLLM_HOOK_WORKER` | `analyzer_name` |
|---|---|---|
| attention tracker | `probe_hook_qk` / `qk` | `attn_tracker` |
| CoRer | `probe_hook_qk` / `qk` | `core_reranker` |
| hidden states | `probe_hidden_states` / `hidden_states` | `hidden_states` |
| activation steering | `steer_hook_act` / `steer` | (none — no artifacts) |

### Offline (`HookLLM`)

```python
from vllm_hook_plugins import HookLLM
from vllm import SamplingParams

llm = HookLLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    worker_name="probe_hook_qk",
    analyzer_name="attn_tracker",
    config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json",
)

# rpc (in-memory) path:
out   = llm.generate(text, SamplingParams(...), save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})

# disk path (artifact under /dev/shm/vllm_hook/<run_id>/):
out   = llm.generate(text, SamplingParams(...), save_to_disk=True, run_id="run-1")
stats = llm.analyze(analyzer_spec={...})  # uses the last run_id

# activation steering (worker_name="steer_hook_act", no analyzer): no save_to_disk, difference is observed by comparing against a use_hook=False baseline.
out_steered = llm.generate(text, SamplingParams(...))
out_plain   = llm.generate(text, SamplingParams(...), use_hook=False)
```

Format/save-mode are env-vars on the offline driver process, set **before** `HookLLM(...)` is constructed (the worker subprocess inherits them at spawn):

```bash
VLLM_HOOK_USE_SAFETENSORS=1   # write .safetensors instead of .pt
VLLM_HOOK_USE_SHM=1           # legacy shared-memory fast path (hidden states + last_token only)
```

### Serve (`vllm serve` + `HookClient` / openai client)

Start the server with `VLLM_HOOK_WORKER` set to the worker that matches your use case:

```bash
# probes (attention tracker / CoRer / hidden states):
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOOK_WORKER=qk \
  vllm serve ibm-granite/granite-3.1-8b-instruct \
    --enforce-eager --max-model-len 2048 --port 8770

# activation steering:
VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOOK_WORKER=steer \
  vllm serve microsoft/Phi-3-mini-4k-instruct \
    --enforce-eager --max-model-len 2048 --port 8770
```

For probe use cases, `HookClient` mirrors the offline `HookLLM` API:

```python
from vllm_hook_plugins import HookClient

hook = HookClient(base_url="http://localhost:8770/v1",
                  analyzer_name="attn_tracker",
                  config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json")

# rpc path:
resp  = hook.generate(model=MODEL, messages=msgs, max_tokens=10)
stats = hook.analyze(analyzer_spec={...})

# disk path:
hook.generate(model=MODEL, messages=msgs, save_to_disk=True, run_id="run-2", max_tokens=1)
stats = hook.analyze(analyzer_spec={...})
```

For activation steering there's no artifact to analyze, so a plain openai client suffices. Each request carries its own steer config as a JSON-encoded string under `vllm_xargs["steer"]` (vllm_xargs only allows scalar values; the plugin decodes the string back to a dict before the worker reads it). Different requests can use different configs:

```python
import openai, json

with open("model_configs/activation_steer/Phi-3-mini-4k-instruct.json") as f:
    base = json.load(f)["steering"]

client = openai.OpenAI(base_url="http://localhost:8770/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct",
    messages=[...], max_tokens=200, temperature=0.0,
    extra_body={"vllm_xargs": {"steer": json.dumps({**base, "coefficient": 5})}},
)
```
See [`examples/demo_actsteer_serve.py`](examples/demo_actsteer_serve.py) for a runnable example with requests using different steer configs.

`VLLM_HOOK_USE_SAFETENSORS` is set when launching `vllm serve` (the server's worker process reads it at hook-fire time).

---

## Preliminary study regarding the storage variant choice

We have done a preliminary test regarding different storage variants using hidden-states extraction as an example. Numbers below are for `last_token` mode at 512-token prompts on Qwen2-1.5B-Instruct, averaged over 4 captured-layer counts {1, 4, 16, 28} (5 timed repetitions per cell after 5 warm-up runs that are discarded).

> **Historical note:** these numbers were measured under the old per-request `VLLM_HOOK_ASYNC_SAVE` background-save-thread path described above, which has since been removed and superseded by the writer process. The `-async` cell labels below are archival measurement IDs only — they are not a configuration you can select today; see the coverage matrices above for what's currently selectable.

| Variant | gen (ms) | total (ms) | analyze overhead (ms) |
|---|---:|---:|---:|
| **disk-st-async** | 40.0 | 41.8 | 1.8 |
| disk-pt-async | 41.0 | 49.8 | 8.7 |
| disk-pt | 47.3 | 53.8 | 6.5 |
| shm | 53.1 | 53.1 | 0.0 |
| rpc | 65.3 | 65.3 | 0.0 |

### Takeaways

- **disk-st (via the now-removed async-save path) was the fastest measured variant.** It minimized generate-side latency (async I/O off the critical path) and produced the smallest safetensors artifact. Today's recommended equivalent is `disk` + `st` with the writer process (on by default) — it wasn't re-benchmarked against this table, but it's the closest currently-selectable configuration to what these numbers measured.
- **rpc is the slowest path across the board** — `collective_rpc` serializes the tensor through Python/IPC, and the cost grows with the captured-layer count. Avoid rpc when artifacts are large; use disk.
- **`shm` is no longer competitive** post-refactor even at `last_token`. The legacy fast-path is kept for back-compat but the disk-st variant (measured under the old async-save path) beat it on every measured cell.
