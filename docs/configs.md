# MIA supported configurations

This document enumerates the supported configs and how to invoke each from user code.

---

## Configuration axes

| Axis | Values | How it's selected |
|---|---|---|
| **Execution path** | `serve` (`vllm serve` + `MiaClient`) | — |
| **Storage** | `rpc` (on the output: `.probes`) · `disk` (artifact under `<hook_dir>/<run_id>/`) · `shm` (legacy shared memory, hidden states-only) | per-request `extra_args["save_to_disk"]` (SHM via `MIA_USE_SHM=1`) |
| **Disk format** | `pt` (`torch.save`) · `st` (safetensors ) | `MIA_USE_SAFETENSORS={0,1}` |

> **Async save note:** the old per-request `sync`/`async` save-mode axis (`MIA_ASYNC_SAVE`) has been removed. It is superseded by the **writer process** (`MIA_WRITER_PROCESS`, default **on**) — a persistent child process that serializes and writes disk artifacts off the engine GIL (see the `writer_process` lever in `optimizations.py`). Unlike the old knob, this isn't a per-request axis you opt into: it's a process-wide default that's already on, so it does not appear as a selectable dimension in the coverage matrices below. It runs on **every TP rank that writes artifacts**: vLLM's daemonic TP workers used to fall back to the in-process save. Each rank logs its mode. The child exits when its worker dies (`MIA_CHILD_PARENT_POLL_S`, default `1.0` s, is how often an idle child checks).

---

## Coverage matrix

### Attention tracker 

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `attn-serve-rpc-na`     | serve   | rpc  | —  |
| `attn-serve-disk-pt`    | serve   | disk | pt |
| `attn-serve-disk-st`    | serve   | disk | st |

### Hidden states 

The same 3 combinations as above.

### CoRer

CoRer is intrinsically two-pass and only uses the disk path (the analyzer needs both runs' artifacts on disk to compute the difference). No `rpc` cells.

| Cell ID | Path | Storage | Format |
|---|---|---|---|
| `corer-serve-disk-pt`   | serve   | disk | pt |
| `corer-serve-disk-st`   | serve   | disk | st |

### Activation steering 

Steering modifies the residual stream in-place and produces no artifacts, so storage/format/async axes don't apply. Per-request via `extra_args["steer"]`.

| Cell ID | Path |
|---|---|
| `actsteer-serve-na-na`   | serve   |

---

## Selecting a configuration from user code

All hook activation is **per-request** via `extra_body["vllm_xargs"]` under `vllm serve`, or
`SamplingParams.extra_args` when driving an engine in-process. Different requests in the same
batch can use different configs.

The same code shape covers all four use cases — only `worker_name` / `analyzer_name`, or
`MIA_WORKER` on the server, varies:

| Use case | `worker_name` / `MIA_WORKER` | `analyzer_name` |
|---|---|---|
| attention tracker | `capture_qk` / `qk` | `attn_tracker` |
| CoRer | `capture_qk` / `qk` | `core_reranker` |
| hidden states | `capture_hs` / `hidden_states` | `hidden_states` |
| activation steering | `steer` / `steer` | (none — no artifacts) |

### Driving an engine in-process (`MiaLLM`)

`MiaLLM` builds a vLLM engine in your own process, under CUDA graphs by default
(`enforce_eager=True` opts out) — the quickest way to run capture on one machine, not a
deployment path. For serving, use `vllm serve` with `MiaClient` below.

```python
import torch
from mia import MiaLLM
from vllm import SamplingParams

llm = MiaLLM(
    model="ibm-granite/granite-3.1-8b-instruct",
    worker_name="capture_qk",
    analyzer_name="attn_tracker",
    config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json",
    hook_dir="/dev/shm/mia",  # where disk artifacts are written
    dtype=torch.float16,      # attn_tracker cannot read bfloat16
)

# rpc (in-memory) path:
out   = llm.generate(text, SamplingParams(...), save_to_disk=False)
stats = llm.analyze(probes=out[0].probes, analyzer_spec={...})

# disk path (artifact under /dev/shm/mia/<run_id>/); reset the prefix cache when re-capturing a prompt:
llm.llm_engine.reset_prefix_cache()
out   = llm.generate(text, SamplingParams(...), save_to_disk=True, run_id="run-1")
stats = llm.analyze(analyzer_spec={...})  # uses the last run_id
```

Steering (`worker_name="steer"`) applies the config's `steering` section; `extra_args["steer"]` overrides it per request:

```python
import json

llm = MiaLLM(model="microsoft/Phi-3-mini-4k-instruct", worker_name="steer",
             config_file="model_configs/activation_steer/Phi-3-mini-4k-instruct.json")
with open("model_configs/activation_steer/Phi-3-mini-4k-instruct.json") as f:
    base = json.load(f)["steering"]

steer = {**base, "method": "add_vector", "coefficient": 10}
out_steered = llm.generate(text, SamplingParams(temperature=0.0, max_tokens=200, extra_args={"steer": steer}))
out_plain   = llm.generate(text, SamplingParams(temperature=0.0, max_tokens=200), use_hook=False)
```

Format/save-mode are env-vars on the driver process, set **before** `MiaLLM(...)` is constructed (the worker subprocess inherits them at spawn):

```bash
MIA_USE_SAFETENSORS=1   # write .safetensors instead of .pt
MIA_USE_SHM=1           # legacy shared-memory fast path (hidden states + last_token only)
```

### Serve (`vllm serve` + `MiaClient` / openai client)

Start the server with `MIA_WORKER` set to the worker that matches your use case. It runs under
CUDA graphs by default; add `--enforce-eager` only when you need bit-exact logprobs:

```bash
# probes (attention tracker / CoRer / hidden states):
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=qk \
  vllm serve ibm-granite/granite-3.1-8b-instruct --max-model-len 2048 --port 8770

# activation steering:
VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=steer \
  vllm serve microsoft/Phi-3-mini-4k-instruct --max-model-len 2048 --port 8770
```

The server exposes captured activations at `/v1/mia/delivered`: protect it with `--api-key KEY`
(and `MiaClient(..., api_key=KEY)`), or turn it off with `MIA_DELIVERY_ROUTE=0` (hidden states
are then read from files only).

For probe use cases, `MiaClient` mirrors the in-process `MiaLLM` API:

```python
from mia import MiaClient

hook = MiaClient(base_url="http://localhost:8770/v1",
                  analyzer_name="attn_tracker",
                  config_file="model_configs/attention_tracker/granite-3.1-8b-instruct.json")

# rpc path:
resp  = hook.generate(model=MODEL, messages=msgs, max_tokens=10)
stats = hook.analyze(analyzer_spec={...})

# disk path:
hook.generate(model=MODEL, messages=msgs, save_to_disk=True, run_id="run-2", max_tokens=1)
stats = hook.analyze(analyzer_spec={...})
```

#### Three ways to send a prompt

`generate` goes to `/v1/chat/completions`, so **the server applies the chat template** and the
token layout is the server's. An analyzer that scores token spans cannot use it — the spans were
computed against a different tokenization. Use the completions entry points for those:

| Call | Endpoint | Template | Use it when |
|---|---|---|---|
| `generate(messages=[...])` | chat | server applies it | ordinary capture; you only need the text back |
| `generate_tokens(ids)` | completions | none | an analyzer scores token spans, or you need a continuation |
| `generate_text(prompt)` | completions | none | you templated the prompt yourself |

Both completions calls take one sequence or a list of them; **a list shares one `run_id`**, the
way a list passed to the in-process `generate` does.

To check that the server prompted on the ids you meant — and to read back the ids it *generated*,
which is what a second pass needs — ask vLLM for them:

```python
resp = hook.generate_tokens(ids, model=MODEL, max_tokens=1, save_to_disk=True,
                            extra_body={"return_token_ids": True})
assert list(resp.choices[0].prompt_token_ids) == list(ids)   # nothing re-tokenized
continuation = list(resp.choices[0].token_ids)               # ids, never detokenized text
```

Rebuild a continuation from those ids, not from `choices[0].text`: detokenizing and
re-tokenizing is not an identity, and the drift is silent.

#### Per-request arguments

| Argument | What it does | In-process equivalent |
|---|---|---|
| `save_to_disk=` | artifact to `hook_dir/<run_id>/` instead of riding back on the response | same |
| `run_id=` | names the artifact directory; a batch shares one | same |
| `steer=` | a steering config for this request alone | `extra_args["steer"]` |
| `extra_xargs=` | any other per-request knob, e.g. `{"hooks_on": "both"}` | `SamplingParams.extra_args` |
| `capture=False` | arm nothing — a plain request | `use_hook=False` |
| `extra_body=` | vLLM's own request extensions, merged with MIA's `vllm_xargs` | — |

`extra_xargs` values must be **scalars** unless the key is one the plugin JSON-decodes
(`output_qk`, `output_hidden_states`, `steer`); a dict under any other key would reach the worker
as a string and do nothing, so the client refuses it instead. On a collision in `extra_body`,
MIA's own keys win — a caller cannot redirect `run_id` or `hook_dir`.

`analyze(probes=...)` analyzes a payload you already hold, and `.tokenizer` gives the served
model's tokenizer for computing spans (pass `tokenizer_for=<model id>` to the constructor).

#### No prefix-cache reset over serve

The in-process path calls `llm.llm_engine.reset_prefix_cache()` between captures of the same
prompt. **vLLM 0.29 exposes no endpoint for that**, so if your run captures the same prefix twice
— CoRer does — start the server with `--no-enable-prefix-caching`. A cached prefix means the
second pass captures nothing for those tokens, and the run still looks like it worked.

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
See [`examples/demo_actsteer_serve.py`](../examples/demo_actsteer_serve.py) for a runnable example with requests using different steer configs.

`MIA_USE_SAFETENSORS` is set when launching `vllm serve` (the server's worker process reads it at hook-fire time).

---

## Sizing the capture aperture, and the GPU memory it costs

The aperture is a **fixed** GPU allocation — fixed because a buffer that grows with traffic
OOMs under load. It is taken in addition to vLLM's own KV-cache budget, so the two have to fit
on the card together, and MIA checks that at engine start rather than letting the allocation
fail later.

| env var | default | effect |
|---|---|---|
| `MIA_APERTURE_GPU_BYTES` | **4 GiB** | the aperture's byte budget. **Per rank** at TP > 1, not per engine |
| `MIA_APERTURE_MAX_BATCHED_TOKENS` | off | derive (`auto`) or pin `max_num_batched_tokens` so a heavy capture's per-step transient cannot OOM at high batch. MIN-ONLY: it never raises the budget, so it is byte-identical whenever the derived cap is the larger one |
| `MIA_APERTURE_BACKPRESSURE_TIMEOUT_S` | `10` (s) | how long a step waits for free capture space before `ApertureBackpressureError`. Capture blocks; it never silently drops rows. Raise it if a slow disk makes a heavy run hit it |

### The one rule: `gpu_memory_utilization` must leave room for the aperture

Yes — explicitly, and it is enforced. `gpu_memory_utilization` is vLLM's flag, not MIA's
(`--gpu-memory-utilization` on the server); it tells vLLM what fraction of the card to claim for
weights and KV cache. **The aperture lives entirely in the fraction vLLM does not claim**, and
engine start fails unless:

```
MIA_APERTURE_GPU_BYTES  ≤  (1 − gpu_memory_utilization) × total GPU bytes
```

On an 80 GiB card, with the 4 GiB default:

| `gpu_memory_utilization` | left for the aperture | 4 GiB default |
|---|---|---|
| 0.80 | 16.0 GiB | fits easily |
| 0.90 | 8.0 GiB | fits |
| 0.95 | 4.0 GiB | exactly at the limit |
| 0.97 | 2.4 GiB | **refused** |

Two things worth knowing about that check. It is computed from the *fraction*, not from a
measurement — so it does not know about anything else sharing the card; and it runs at engine
construction, so you get a named error instead of a CUDA OOM mid-run:

```
aperture 4.00 GiB + gpu_memory_utilization=0.97 leaves no room (free margin 2.40 GiB):
lower gpu_memory_utilization or MIA_APERTURE_GPU_BYTES
```

When you set nothing, MIA also checks the default against what **one max-token step** actually
needs (`max_num_batched_tokens × row bytes × captured layers`). If a step needs more than 4 GiB
it grows the budget to fit — but only within that same free margin. If it cannot, it refuses and
names all four ways out: lower `gpu_memory_utilization`, lower `max_num_batched_tokens`, capture
fewer layers, or set `MIA_APERTURE_GPU_BYTES` yourself.

An explicit `MIA_APERTURE_GPU_BYTES` always wins and skips that growth — including when it is
*smaller* than one step needs, which is legal and will surface as `ApertureBackpressureError`
when a max-token step cannot be admitted. The demos use `gpu_memory_utilization=0.7`, which
leaves 24 GiB on an 80 GiB card and never runs into this.

At TP > 1 the budget is per rank, which is cheaper than it sounds for hidden states: a step costs
`max_num_batched_tokens × ceil(L / tp) × hidden × 2` on each rank, so 2.5 GiB for
Llama-3.1-70B at TP4 rather than 10 GiB.

---

## Reading your data back

Where each call puts its data, how to read it and when it is ready:
[examples/README.md](../examples/README.md#where-the-captured-data-goes). In short:
`out[i].probes` / `llm.analyze()` offline, `response.probes` / `client.analyze()` served,
`<hook_dir>/<run_id>/` with `save_to_disk`, and `load_delivered(<capture dir>)` for the
hidden-state files. A served capture sent without `save_to_disk` may land in
`<hook_dir>/<run_id>/` instead of on the response; `client.analyze()` reads it either way. An
explicit `save_to_disk` is never overridden.

## Limits

- Graph-mode capture can move per-token logprobs slightly; `enforce_eager=True`
  (`--enforce-eager`) is bit-exact.
- Q/K capture turns prefix caching off unless you set it.
- Offline score capture, and Q/K with explicit prefix caching or DP > 1, run eager; a served score
  request on a graph engine is refused (start the server with `--enforce-eager`).
- Graph steering: at most `MIA_STEER_VMAX` (16) distinct vectors per engine; more are refused.

## Tuning

Optional performance levers and their env vars (the defaults need no change):

```python
from mia.optimizations import describe; print(describe())
```
