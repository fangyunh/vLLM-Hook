# The entire workflow — running vLLM-Hook end to end

This is the step-by-step guide. It carries **one** use case — capturing **hidden states** — from a
fresh shell to a GPU job, then re-runs that same task with each setting changed. Every setting is a
delta from something already working, so you never assemble a matrix from scratch.

If you only read one thing: §2 is the working example, §3 puts it on the GPU, and §4 is the
setting most likely to surprise you.

**Scope.** Branch `capture_ring`, vLLM 0.21 + the V1 model runner (env `vllm_hook_env`). GPU work
goes through LSF (`bsub`) because the GPU is on a central cluster in `exclusive_process` mode.

**Related docs** — this one does not duplicate them:

| doc | what it holds |
|---|---|
| `docs/configs.md` | the full storage/config coverage matrix per use case |
| `docs/use_cases/README.md` | the catalogue of use cases |

---

## 0. What you are building toward

By the end of §2 you will have run a prompt through a real vLLM engine, captured the residual-stream
hidden states of the layers you asked for, and printed their shapes and norms:

```
Prompt: 'The capital of France is'
Generated: 'Paris. The capital of France is Paris.'
  layer_1: shape=(1, 1536), norm=41.2812
  layer_2: shape=(1, 1536), norm=58.9375
  layer_3: shape=(1, 1536), norm=71.5000
  layer_4: shape=(1, 1536), norm=88.3125
```

That is the whole loop: **declare what to capture → generate → read it back**. Everything after §2
changes *where* that runs or *how fast*, never what you write.

---

## 1. Setup, once

### 1.1 Environment

```bash
conda activate vllm_hook_env          # the expected env on this system
pip install -r requirement.txt        # vllm, torch, numpy, blake3
pip install -e vllm_hook_plugins      # editable install of the plugin
```

The plugin **auto-loads**. It registers two `vllm.general_plugins` entry points, so merely importing
vLLM patches it — you never call an "enable" function. That also means a *baseline* run that must
not involve the plugin needs `VLLM_PLUGINS=""`, not just "don't import it".

### 1.2 Where models live — read this before your first download

**Every model file must land under `/proj/dmfexp/fangyunh/`. Never in HOME.** HOME has a tight
per-user quota that a single 8B model blows through, and the failure is confusing: `git commit`
starts failing with `unable to write loose object file: Disk quota exceeded`, or an LSF log
truncates mid-run. A "disk full" on this box is **always** the HOME quota, never the device.

Two symlinks do the redirect and you need **both** — they cover different paths:

| symlink | → target | catches |
|---|---|---|
| `cache/` (repo root) | `/proj/…/vLLM-Hook-offload/hf_cache` | anything passing `download_dir="./cache/"` — every example and harness here |
| `~/.cache/huggingface` | `/proj/…/vLLM-Hook-offload/home_hf_cache` | everything else — bare `huggingface_hub`, `transformers`, `vllm serve` with no `download_dir` |

Verify before you download anything, because a broken symlink fails **silently** into the quota:

```bash
readlink -f ~/.cache/huggingface     # MUST print a /proj/... path
du -sh ~/.cache                      # MUST stay ~GB, not tens of GB
```

To repair:

```bash
cd ~/vLLM-Hook && rm -f cache && ln -s /proj/dmfexp/fangyunh/vLLM-Hook-offload/hf_cache cache
rm -f ~/.cache/huggingface && ln -s /proj/dmfexp/fangyunh/vLLM-Hook-offload/home_hf_cache ~/.cache/huggingface
```

`/proj` capacity is **not** a constraint — do not shrink an experiment to save space there. The only
limits that bite are **GPU memory** and **host RAM**. The one obligation is hygiene: delete captured
data after a campaign finishes.

---

## 2. The example: capture hidden states, offline

### 2.1 Declare what to capture

A config file says *what* to capture. It does not say where it goes or how fast it runs.

`model_configs/hidden_states/Qwen2-1.5B-Instruct.json`:

```json
{
  "model_info": {
    "name": "Qwen/Qwen2-1.5B-Instruct"
  },
  "hidden_states": {
    "layers": [1, 2, 3, 4],
    "mode": "last_token"
  }
}
```

Two axes, and both matter for cost:

| field | values | meaning |
|---|---|---|
| `layers` | list of ints, or `[]` | which decoder layers. **`[]` means ALL layers** — that is the expensive setting, not a "default off" |
| `mode` | `last_token` \| `all_tokens` | one vector per request, or one per token. `all_tokens` scales with sequence length |

Config files live at `model_configs/<use_case>/<model_short_name>.json`. Several shapes ship for
this model — `Qwen2-1.5B-Instruct_alltok.json`, `..._alltok_4L.json`, `..._lasttok_all.json` — so
you can switch axis without editing anything.

### 2.2 Run it

```python
import os
import multiprocessing as mp
import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import SamplingParams
from vllm_hook_plugins import HookLLM

llm = HookLLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    worker_name="probe_hidden_states",          # captures the residual stream
    analyzer_name="hidden_states",              # post-processes what was captured
    config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
    download_dir="./cache/",                    # -> /proj, see 1.2
    hook_dir="/dev/shm/vllm_hook",              # where disk artifacts go
    gpu_memory_utilization=0.7,
    max_model_len=2048,
    dtype=torch.float16,
    enable_hook=True,
)

out = llm.generate(
    "The capital of France is",
    SamplingParams(temperature=0.0, max_tokens=10),
    save_to_disk=True,
)
stats = llm.analyze(analyzer_spec={"reduce": "none"})

print(out[0].outputs[0].text.strip())
for layer_name, tensors in sorted(stats["hidden_states"].items()):
    t = tensors[0]
    print(f"  {layer_name}: shape={tuple(t.shape)}, norm={torch.norm(t.float()):.4f}")
```

`examples/demo_hiddenstate.py` is this script, ready to run:

```bash
python examples/demo_hiddenstate.py
```

Three things worth knowing about what you just ran:

- **`worker_name` picks the engine-side capture; `analyzer_name` picks the driver-side
  post-processing.** They are independent. One engine process runs exactly one worker.
- **`analyzer_spec={"reduce": ...}`** takes `none` (raw tensors), `mean`, or `norm`. `norm` is the
  cheap way to eyeball a batch.
- **By default HS captures PREFILL ONLY.** The worker's `_default_hooks_on` is `"prefill"`. To
  capture generated tokens too, ask per request:

  ```python
  SamplingParams(temperature=0.0, max_tokens=10,
                 extra_args={"hooks_on": "both"})     # prefill | decode | both
  ```

  This is the single most common "why did I capture nothing at decode?" — see §9.

### 2.3 Read it back two ways

```python
# (a) in-memory RPC — artifacts ride back on the output object
out = llm.generate(prompt, sp, save_to_disk=False)
hs = out[0].probes["hs_cache"]        # {layer: {"hidden_states": ..., "layer_num": ...}}

# (b) disk — artifact written under hook_dir/<run_id>/, analyzer reads it
out = llm.generate(prompt, sp, save_to_disk=True)
stats = llm.analyze(analyzer_spec={"reduce": "norm"})
```

For a batch, per-request probes are merged onto `outputs[0]`, so `out[0].probes` is always the place
to look regardless of batch size.

---

## 3. Run it on the GPU — the LSF path

The GPU is on a central LSF cluster and is `exclusive_process`: **one Python process cannot boot two
engines**. So each engine gets its own process, and multi-engine tests run them sequentially inside
one job.

### 3.1 A complete job script

There is no standalone demo job script shipped under this name anymore. The skeleton below shows
the job-script shape you'll write for any GPU run — copy it as the starting point for your own.
For a job script that is shipped today and runs as-is, see
`tests/cuda_graph/tests/hs_graph/run_hs_parity_full.sh`, the canonical hidden-states parity runner
(§3.3):

```bash
#!/bin/bash
#BSUB -J hs_demo
#BSUB -G grp_exploratory
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[ngpus=1,mem=32GB]"
#BSUB -o tests/cuda_graph/logs/hs_demo.%J.out
#BSUB -e tests/cuda_graph/logs/hs_demo.%J.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env
cd ~/vLLM-Hook
mkdir -p tests/cuda_graph/logs

# Model resolution pings the HF API even when cached; a burst of jobs gets 429'd and
# surfaces as "server failed to become ready". Offline once the model is cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export VLLM_HOOK_DEMO_MODEL="Qwen/Qwen2-1.5B-Instruct"
export VLLM_HOOK_CONFIG_FILE="model_configs/hidden_states/Qwen2-1.5B-Instruct.json"
export VLLM_LOGGING_LEVEL=WARNING

echo "[job] host=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
python -u examples/demo_hiddenstate.py
```

Stamping branch + SHA into the log is not decoration. An editable install imports the **live** tree
at job start, so a job that begins after you edit a file measures the edit. Print the SHA and check
it before trusting a number.

### 3.2 Submit and watch

```bash
cd ~/vLLM-Hook
bsub -G grp_exploratory < tests/cuda_graph/tests/hs_graph/run_hs_parity_full.sh   # -G is mandatory
bjobs                                                    # queued / running / done
bpeek <JOBID>                                            # live stdout
grep 'VERDICT' tests/cuda_graph/logs/hs_parity_full.<JOBID>.out  # after it lands
bkill <JOBID>                                            # stop it
```

Logs appear under `tests/cuda_graph/logs/` **only once the job starts**, not at submit time.

Two habits that save hours:

- **`bsub` needs a login shell** when you wrap it: `bash -lc 'bsub …'`. LSF propagates the
  submitting environment, and a non-interactive shell never sources `~/.bashrc` — so your `HF_HOME`
  exports silently vanish.
- **A hard-killed job leaks.** vLLM creates a Unix socket named with a UUID in the process CWD per
  engine boot; a killed job leaves it behind. They are harmless and invisible to git (git does not
  track sockets), but they accumulate. Sweep with `find . -maxdepth 1 -type s -delete`.

### 3.3 The parity harnesses are the worked reference

Every capture path has a graph-vs-eager numerical oracle under
`tests/cuda_graph/tests/<topic>/`, driven by a `run_*.sh` wrapper. For hidden states:

```bash
bsub -G grp_exploratory < tests/cuda_graph/tests/ring/run_hs_ring_parity.sh
grep 'VERDICT' tests/cuda_graph/logs/hs_ring_parity.*.out
```

When you are unsure how to wire a setting, read the harness that already pins it. They are
maintained; prose drifts.

---

## 4. Setting: eager → FULL CUDA graphs

### 4.1 Why this exists

By default the plugin **forces `enforce_eager=True`**. It has to: the eager capture path uses
`register_forward_hook`, and `torch.compile` traces each `forward` once and replaces it, so the hook
never fires. Rather than capture nothing, the plugin gives up the compile + cudagraph speedup — which
is the dominant cost of hooking at serving time.

FULL graph mode (eager prefill + full-graph decode) uses a different mechanism — a static-buffer
scatter op baked **into** the decode graph — so capture survives replay. Hooked decode under FULL
beats eager decode rather than taxing it.

### 4.2 How to turn it on

Two env vars before the plugin imports, two `HookLLM` kwargs:

```python
import os
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"   # arm graph mode
os.environ["VLLM_HOOK_HS_CAPTURE"]      = "buffer"   # the only path (default; may omit)

llm = HookLLM(
    model="Qwen/Qwen2-1.5B-Instruct",
    worker_name="probe_hidden_states",
    analyzer_name="hidden_states",
    config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
    enforce_eager=False,                                  # (a) let compile + cudagraph run
    compilation_config={"cudagraph_mode": "FULL"},        # (b) graph prefill + decode
    gpu_memory_utilization=0.7, max_model_len=2048, enable_hook=True,
)
```

**Do not set `TORCHDYNAMO_DISABLE` by hand.** `vllm_hook_plugins/hook_llm.py` gates it on
`VLLM_HOOK_ALLOW_CUDAGRAPH` for you: the eager path needs compile OFF (or Dynamo traces the hooks
away), the graph path needs it ON (or cudagraph replays unfused, ~+2.3 ms/step). Setting it
yourself is how the serve path once silently ran uncompiled for weeks. An explicit caller value
always wins, so only pass one if you mean it.

### 4.3 ⚠️ Retrieval is different under graph mode — `probes` will be EMPTY

On this branch the HS graph path routes captures through a **capture ring**: the baked op scatters
each token's residual into a persistent GPU ring, an off-loop consumer drains it to durable
per-layer raw files, and there is no RPC bank to read. So:

```python
out = llm.generate(prompt, sp)
out[0].probes["hs_cache"]        # EMPTY under graph mode — this is not a bug
```

Read it back through the ring instead:

```python
# 1. final drain + write the metadata sidecar; returns the per-worker run_dir.
#    HookLLM wraps vllm.LLM, so the RPC handle is `.llm` on some paths and
#    `.llm_engine` on others — try both, as the parity harness does.
run_dir = None
for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
    if h is None:
        continue
    rows = h.collective_rpc("flush_ring")
    if rows:
        run_dir = rows[0]      # TP=1 -> a single rank-0 result
        break

# 2. reconstruct
from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact
data = load_multilayer_ring_artifact(run_dir)
```

`flush_ring` is called **by string method name**, so `collective_rpc` ships no callable to the
worker and `VLLM_ALLOW_INSECURE_SERIALIZATION` is not needed. Call it once, after all requests
finish, because the worker process is usually killed rather than joined — anything not flushed is
gone. It returns `None` if the ring path was never installed.

`tests/cuda_graph/tests/ring/hs_ring_parity.py` is the worked reference.

The eager path is unchanged and still returns `probes`. That asymmetry is what the parity oracles
compare.

### 4.4 ⚠️ Set `VLLM_HOOK_RING_DIR` — its default is a relative path

```bash
export VLLM_HOOK_RING_DIR="/dev/shm/vllm_hook_${USER}/ring_${LSB_JOBID:-manual}"
```

Unset, it defaults to `./hs_ring_dump` — **relative to the working directory**, i.e. straight into
the repo and therefore into your HOME quota. An all-layers `all_tokens` run fills tens of GB fast.
Point it at node-local NVMe or `/dev/shm`.

### 4.5 Ring sizing

`VLLM_HOOK_RING_GPU_BYTES` is the only sizing knob (default **4 GiB**), checked at install against
your free GPU budget. A sweep across 0.75–16 GiB found the ring is flat and non-monotonic and
**never blocks** at ≥4 GiB — the ring is not the bottleneck in this workload, the durable sink is.
If you undersize it, you get a loud `RingBackpressureError`, never a silent drop.

---

## 5. Setting: offline → serve

Same capture, same config file, different front door. The worker runs server-side; the analyzer runs
in your client.

### 5.1 Start the server

```bash
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_HOOK_WORKER=hidden_states \
  vllm serve Qwen/Qwen2-1.5B-Instruct \
  --enforce-eager --max-model-len 2048 --port 8770
```

`VLLM_HOOK_WORKER` selects the worker — `{qk, hidden_states, steer}` — because **`vllm serve` never
reads a config file.** Anything a config file would have said must arrive per request instead.

For FULL graph mode, add the same two env vars from §4 plus the compilation flag:

```bash
VLLM_HOOK_ALLOW_CUDAGRAPH=1 VLLM_HOOK_WORKER=hidden_states VLLM_HOOK_HS_CAPTURE=buffer \
VLLM_HOOK_RING_DIR=/dev/shm/vllm_hook_$USER/ring \
  vllm serve Qwen/Qwen2-1.5B-Instruct --compilation-config '{"cudagraph_mode":"FULL"}'
```

### 5.2 Client

```python
from vllm_hook_plugins.hook_client import HookClient

client = HookClient(
    base_url="http://localhost:8770/v1",
    analyzer_name="hidden_states",
    config_file="model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
    hook_dir="/dev/shm/vllm_hook",       # must be a path BOTH sides can see
)

resp = client.generate(
    messages=[{"role": "user", "content": "The capital of France is"}],
    model="Qwen/Qwen2-1.5B-Instruct",
    max_tokens=10, temperature=0.0,
    save_to_disk=None,                   # tri-state — see below
)
stats = client.analyze(analyzer_spec={"reduce": "norm"})
```

**`save_to_disk` is tri-state, and the default is not "off":**

| value | meaning |
|---|---|
| `None` *(default)* | no preference — the server's storage router picks the faster path per request. `analyze()` reads whichever it chose, so this is transparent |
| `True` | **require** a durable artifact file under `hook_dir/run_id`. Honored exactly; the router never overrides it |
| `False` | require the in-memory path; artifacts ride back on the response |

The client always sends a `run_id`, so a request the router sends to disk stays findable. The disk
path requires server and client to **share a filesystem** (same host).

### 5.3 Raw OpenAI client

If you would rather not use `HookClient`, per-request settings go through `vllm_xargs`. It only
accepts scalars, so a dict must be JSON-encoded as a string — the plugin decodes it server-side:

```python
extra_body={"vllm_xargs": {"hooks_on": "both"}}
```

`examples/demo_actsteer_serve.py` shows the pattern end-to-end (for steering, but the mechanism is
identical).

---

## 6. Setting: where artifacts go

Three orthogonal axes, a full Cartesian product:

| axis | values | selected by |
|---|---|---|
| transport | `rpc` \| `disk` \| `shm` | `save_to_disk=` per request; `VLLM_HOOK_USE_SHM=1` |
| format | `pt` \| `safetensors` | `VLLM_HOOK_USE_SAFETENSORS=1` |

Rules of thumb, measured: **small artifacts favor RPC, large favor disk.** HS `last_token` is small
(~100 KB) and belongs on RPC; HS `all_tokens` over many layers is large and belongs on disk. You
usually do not have to choose — leave `save_to_disk=None` on the serve path and let the storage
router decide.

`docs/configs.md` has the full per-use-case coverage matrix and code for each cell.

---

## 7. Setting: a different worker

The workflow above is worker-shaped, not hidden-state-shaped. Swap two arguments and everything else
holds:

| Use case | `worker_name` | `analyzer_name` | graph env switch | result |
|---|---|---|---|---|
| Hidden states | `probe_hidden_states` | `hidden_states` | `VLLM_HOOK_HS_CAPTURE=buffer` | `probes["hs_cache"]` |
| Attention Q/K | `probe_hook_qk` | `attn_tracker` (or `core_reranker`) | `VLLM_HOOK_QK_CAPTURE=buffer` | `probes["qk_cache"]` |
| Steering | `steer_hook_act` | *(none)* | `VLLM_HOOK_STEER_MODE=buffer` | no artifact — the output text changes |

Notes that bite:

- **Steering produces no artifacts.** You verify it by comparing against an unsteered baseline
  (`use_hook=False`), not by reading a tensor. Its config lives under `steering` in the config file,
  with `phase` (`prefill`/`decode`/**`both`**) and `positions` (**`all_tokens`**/`last_token`) axes.
- **QK layers are 0-based; HS layers are 1-based.** This has caught people out.
- **One engine, one worker.** `worker_name` picks it; per-worker levers for the other worker are a
  harmless no-op.

---

## 8. Setting: the optimization levers

`vllm_hook_plugins/optimizations.py::PUBLIC_LEVERS` is the **entire supported optimization API** — 8
levers. The package reads ~70 other env names; those are internal tuning constants and diagnostics,
and they are **rejected** if you name one in a config file, so a typo cannot silently do nothing.

| lever | default | env |
|---|---|---|
| `batched_egress` | on | `VLLM_HOOK_BATCHED_EGRESS` |
| `steer_fused` | on | `VLLM_HOOK_STEER_FUSED` |
| `compact_kall` | auto | `VLLM_HOOK_QK_COMPACT_KALL` |
| `writer_process` | on | `VLLM_HOOK_WRITER_PROCESS` |
| `storage_router` | on | `VLLM_HOOK_STORAGE_ROUTER` |
| `artifact_dtype` | native | `VLLM_HOOK_ARTIFACT_DTYPE` |
| `ring_mmap` | **off** | `VLLM_HOOK_RING_MMAP` |
| `ring_max_batched_tokens` | off | `VLLM_HOOK_RING_MAX_BATCHED_TOKENS` |

**Precedence: env > config file > default.** An explicit env always wins. In a config file
(offline `HookLLM` only — `vllm serve` never parses one):

```json
{
  "model_info":    {"name": "Qwen/Qwen2-1.5B-Instruct"},
  "hidden_states": {"layers": [], "mode": "last_token"},
  "optimizations": {"artifact_dtype": "int8"}
}
```

`HookLLM.__init__` applies these **before** `LLM(...)` spawns workers, so every worker process
inherits the env.

Two you should know by name:

- **`artifact_dtype` is the only lossy lever.** Everything else here is byte-identical to eager
  output. Quantization happens before the host copy, so it is quantized through host, RPC wire and
  disk alike; the driver dequantizes.
- **`ring_mmap` defaults OFF and should stay off** unless your ring dir is genuinely
  networked-GPFS. The mmap sink holds the GIL for its whole memcpy on the drain consumer thread;
  plain `write()` releases it. Turning it on halved serving capacity in measurement.

---

## 9. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Captured nothing at decode; prefill fine | HS defaults to `hooks_on="prefill"` | `extra_args={"hooks_on": "both"}` per request |
| `probes["hs_cache"]` empty under graph mode | Graph HS goes through the ring, not the RPC bank | `flush_ring()` + `load_multilayer_ring_artifact` — §4.3 |
| Captured nothing at all, graph mode | `torch.compile` traced the eager hooks away | Set `VLLM_HOOK_ALLOW_CUDAGRAPH=1`; do not touch `TORCHDYNAMO_DISABLE` |
| HOME fills with tens of GB | `VLLM_HOOK_RING_DIR` unset → `./hs_ring_dump` | Point it at `/dev/shm` or node-local NVMe — §4.4 |
| `git commit`: "unable to write loose object file" | HOME quota, not the device | `du -sh ~/.cache/* \| sort -rh \| head`; `~/.cache/pip` and `~/.cache/ccache` are safe to delete |
| "server failed to become ready", exit 0 | HF API rate-limited (429) during model resolution | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once cached |
| CUDA OOM at high batch under graph capture | Per-step capture transient | `VLLM_HOOK_RING_MAX_BATCHED_TOKENS=auto` (MIN-only, byte-identical) |
| `RingBackpressureError` | Ring undersized for the workload | Raise `VLLM_HOOK_RING_GPU_BYTES` (default 4 GiB) |
| A regression appears/disappears between runs | Stale compile cache; vLLM's cache key ignores `worker_extension_cls` | `VLLM_DISABLE_COMPILE_CACHE=1` — mandatory when A/B-ing native vs plugin in one campaign |
| Baseline run is mysteriously slow | The plugin auto-loaded and forced eager | `VLLM_PLUGINS=""` for a true no-plugin baseline |
| UUID-named zero-byte files in the repo root | Leaked vLLM IPC sockets from killed jobs | `find . -maxdepth 1 -type s -delete` |

### Where the truth lives

When prose and code disagree, trust in this order:

1. **The parity harnesses** (`tests/cuda_graph/tests/*/`) — graph-vs-eager numerical oracles at
   `rtol=atol=1e-2`. They are the source of truth for what works.
2. **`tests/unit/`** — no GPU needed, runs in under a minute:
   ```bash
   python -m pytest tests/unit tests/test_optimizations_api.py -q
   ```
3. **`optimizations.py::PUBLIC_LEVERS`** — the lever table, guarded by a test that fails if a
   default flips without the table being updated.
