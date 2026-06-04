# CUDA-graph feasibility verification

Branch: `graph_enable`. Plan: `claude_docs/cuda_graph_plan.html`.

These scripts run the **gate steps** from the plan's validation plan, in
order. They settle the load-bearing claims (C1, C4, C5, and the
idle-tax floor) **before** any implementation work — so a `FAIL` here
saves weeks of correctness coding on a mechanism that wouldn't have
worked anyway.

```
tests/cuda_graph/
├── step0_injection.py       ← decisive: does the wrap land in the captured graph? (driver install)
├── step0_5_mode_matrix.py   ← which CUDAGraphModes survive the wrap?
├── step0_7_idle_tax.py      ← LATENCY: does the per-layer launch cost fit the budget? (+ VRAM readout)
├── step0_8_plugin_install.py← production path: install in the WORKER at load_model, read back via collective_rpc
├── step0_9_mem_budget.py    ← MEMORY: static-buffer VRAM footprint, KV-cache impact, churn boundedness
├── run_step0_injection.sh        ← LSF batch wrapper (1 GPU)
├── run_step0_5_mode_matrix.sh    ← LSF batch wrapper (1 GPU)
├── run_step0_7_idle_tax.sh       ← LSF batch wrapper (1 GPU)
├── run_step0_8_plugin_install.sh ← LSF batch wrapper (1 GPU)
├── run_step0_9_mem_budget.sh     ← LSF batch wrapper (1 GPU)
└── submit_all_gates.sh           ← login-node helper; chains them with bsub -w
```

All run on a **single GPU host** with vLLM ≥ 0.7 installed. They
do not modify any project source; they import vLLM and the plugin only
when invoked.

**Two injection tests — and on vLLM v1, `step0_8` is the decisive one.**
`step0_injection.py` installs the wrap from the *driver* (after `LLM()`
returns). On **vLLM v1 this is structurally too late**: `torch.compile` +
CUDA-graph capture run *inside* `LLM()` during warmup, before the driver
regains control — so the captured graph holds the original forward and the
late wrap never fires (the probe reports **0**, by design). It is kept as an
**informational** probe that documents exactly this.
`step0_8_plugin_install.py` installs inside the *worker* at `load_model` (via a
`worker_cls` whose `load_model` registers the op + wraps the layer class)
**before** compile/capture — the same timing the plan's monkey-patch of
`Worker.load_model` uses. **This is the gate that can PASS on v1**, and the one
the suite treats as decisive; `step0_5` and `step0_7` drive the same
worker-install mechanism.

---

## Prerequisites

```bash
# On the GPU host:
pip install -r requirement.txt           # vLLM + zstandard + safetensors
pip install -e vllm_hook_plugins         # this project, in editable mode
# No HF login needed for the default model. For a gated model
# (e.g. google/gemma-3-4b-it), `huggingface-cli login` first and pass --model.
```

Recommended dev model: **`Qwen/Qwen2-1.5B-Instruct`** (small, fast, **ungated**,
standard `model.layers.<i>` decoder). All scripts default to it. Override with
`--model <repo>` (Python) or `MODEL=<repo>` (LSF wrappers).

> **Three environment facts these scripts rely on (and handle for you):**
> 1. **The vllm-hook plugin forces `enforce_eager=True`** (`_hook_plugin.py`),
>    which would disable CUDA graphs. The scripts set `VLLM_PLUGINS=""` so the
>    plugin does **not** load — they register their own op / install their own
>    `worker_cls`, so they don't need it. (`setdefault`, so you can override.)
> 2. **`VLLM_CUDAGRAPH_MODE` is not a real vLLM variable** (it logs as "unknown"
>    and is ignored). The graph mode is set via
>    `compilation_config={"cudagraph_mode": ...}` and the **realized** mode is
>    printed at boot (`realized cudagraph_mode = ...`) — check that line to be
>    sure the requested mode actually took.
> 3. **vLLM forks the EngineCore by default** (`VLLM_WORKER_MULTIPROC_METHOD=fork`),
>    which crashes with *"Cannot re-initialize CUDA in forked subprocess"* once
>    the parent has touched CUDA — and a subprocess engine also hides the model
>    from the driver-install tests. The scripts set
>    `VLLM_ENABLE_V1_MULTIPROCESSING=0` (engine in-process). For multi-GPU /
>    true cross-process validation, export `VLLM_ENABLE_V1_MULTIPROCESSING=1`
>    **and** `VLLM_WORKER_MULTIPROC_METHOD=spawn` instead.

---

## Step 0 — Injection (the decisive kill-switch) — ½ day

Run:

```bash
python tests/cuda_graph/step0_injection.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --num-decode-tokens 64 \
    --cudagraph-mode PIECEWISE
```

What it does:

1. Registers a tiny `vllm_hook_counter.counter_increment` custom op via
   vLLM's `direct_register_custom_op`, with `mutates_args=["counter"]`
   so it survives Dynamo / autodfunctionalize / DCE.
2. Boots vLLM with CUDA graphs enabled.
3. Class-level wraps every matched decoder-layer **class** so its
   `forward` calls the counter op once. Instance-level patching is
   documented as bypassed by Dynamo (PyTorch #100733, #93484, #113333)
   and is **not** attempted — class-level is the only mechanism this
   plan ships on.
4. Greedy-decodes 64 tokens.
5. Reads the counter and compares against `num_layers × num_steps`.

How to read the output:

| Counter delta | Diagnosis | What to do |
|---|---|---|
| `0` | Wrap bypassed entirely | **Goal-gate fires** — no pure-plugin path on this torch/vLLM. Either bump versions or pivot to eager-island. |
| `num_layers` | Wrap fired only at prefill | Wrap landed AFTER vLLM compiled the decode graph. Move the wrap into a `worker_extension_cls.load_model` override (the harness here installs from the driver, which races with vLLM's startup on some versions). |
| `num_layers × num_steps` | **PASS** | C1 + C4 settled. Proceed to Step 0.5. |
| `> expected` | Double-wrap | Class registry missing idempotency guard; or spec-decoding / PP issuing extra forward calls. |
| `0 < δ < expected` | Partial wrap | Hybrid model with multiple decoder-layer classes; print the matched class set and re-run. |

The graph-node inspection slot in `inspect_captured_graph_for_op()` is
currently a placeholder — the counter check is independently decisive.
A later run on a known-good CUDA/torch stack can fill in the
`cuGraphGetNodes` walk for corroboration.

---

## Step 0.5 — Mode matrix — ½ day

Only after Step 0 PASSes on PIECEWISE. Run:

```bash
python tests/cuda_graph/step0_5_mode_matrix.py \
    --model Qwen/Qwen2-1.5B-Instruct
```

What it does: spawns three subprocesses, one per `CUDAGraphMode`
(`PIECEWISE`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE`), each running
the Step 0 harness. Reports a 3-row mode matrix.

Decisions:

- **PIECEWISE FAIL** → goal-gate fires (same as Step 0 with default mode).
- **PIECEWISE PASS, FULL_DECODE_ONLY PASS** → ship dual-mode as planned.
- **PIECEWISE PASS, FULL_DECODE_ONLY FAIL** → ship piecewise-only,
  document the throughput ceiling, file FULL as deferred work.

Notes:

- The mode is applied via `compilation_config={"cudagraph_mode": ...}`, and
  each run prints `realized cudagraph_mode = ...` right after boot. **Check that
  line**: if it does not match the requested mode, the modes collapsed to the
  engine default (older vLLM without the `cudagraph_mode` field) and the matrix
  is not meaningful.
- `FULL_DECODE_ONLY` requires that prefill and decode go through
  different paths; some older vLLM versions require an extra flag —
  check `vllm/config/compilation.py` on your version.

---

## Step 0.7 — Idle-tax budget — ½ day

Only after Step 0 PASSes. Step 0.7 does NOT depend on Step 0.5 passing
— it can run in parallel.

```bash
python tests/cuda_graph/step0_7_idle_tax.py \
    --model Qwen/Qwen2-1.5B-Instruct \
    --num-decode-tokens 128 \
    --batches 1 8 64
```

What it does:

1. Boots vLLM unhooked, times decode at three batch sizes.
2. Re-boots vLLM with Step 0's counter wrap on **every** matched
   decoder-layer class.
3. Re-times decode at the same batch sizes.
4. Computes `(wrapped - baseline) / num_layers` — the per-layer
   per-step launch cost.

Pass criteria:

- per-layer overhead ≤ 3 µs
- aggregate idle overhead ≤ 5% at bs=1

The realistic deployed state for the plugin is *loaded but mostly idle*
— the per-layer-per-step launch cost is what every user of the plugin
pays, even when no hook is active in their request. This is the
single most important throughput number in the plan.

Failure pivots:

- **per-layer > 5 µs, aggregate > 10% at bs=1** → the all-layers-wrapped
  design is not competitive with eager-island at small batch.
  Document and propose:
  - (i) `VLLM_HOOK_ACTIVE_LAYERS=[7]` deploy-time list narrowing
    which layers ever activate (does not touch per-request freedom,
    only narrows which layers can be activated).
  - (ii) Wait for CUDA-12.4 conditional-node integration in PyTorch.
- **per-layer ≤ 3 µs but aggregate > 5% at bs=1 only on large models**
  → expected (overhead scales linearly with depth); the plan ships
  with a documented bs=1-on-large-models caveat.

> **Guard added.** `step0_7` now reads the counter after the wrapped run and
> **FAILs if it is 0** — that would mean the op never fired, so the "wrapped"
> timings measured an *absent* op (a false PASS). Confirm injection with
> `step0_injection.py` / `step0_8_plugin_install.py` first.
>
> **VRAM readout added.** `step0_7` also prints device-global VRAM used
> (baseline vs wrapped, via `torch.cuda.mem_get_info()`). This is a *sanity*
> readout only — baseline and wrapped boot sequentially in one process and the
> counter wrap allocates ~nothing, so it is **not** an isolated buffer cost.
> The authoritative memory numbers come from **Step 0.9**.

---

## Step 0.8 — Production-path injection (worker install) — ½ day

Only after Step 0 PASSes. Runs in parallel with Step 0.5 / 0.7.

```bash
python tests/cuda_graph/step0_8_plugin_install.py \
    --model <model-you-can-access> \
    --num-decode-tokens 64 \
    --cudagraph-mode PIECEWISE
    # add --async to also exercise the AsyncLLM (serve-shape) engine
```

What it does:

1. Defines a `worker_cls` **subclass** whose `load_model` registers the counter
   op and installs the class-level wrap — **inside the worker process**, after
   the model is built and before warm-up/capture.
2. Boots vLLM with `worker_cls=step0_8_plugin_install.InjectingWorker`,
   `enforce_eager=False`.
3. Resets the per-worker counter via `collective_rpc("reset_counter")`,
   greedy-decodes, then reads it back via `collective_rpc("read_counter")`
   (counts summed across workers, so TP/PP are handled).
4. Applies the same PASS/FAIL verdict as Step 0.

Why this exists, on top of Step 0: Step 0 installs from the *driver*, which
only works single-process and is **not** the production path. Step 0.8 installs
in the *worker* at `load_model` — the same process and timing the plan's
monkey-patch of `Worker.load_model` uses — and reads results back over
`collective_rpc`, exactly like the real egress path. A `worker_cls` subclass is
used because a `worker_extension_cls` mixin cannot override `load_model` (vLLM
asserts no attribute conflicts and appends the extension to `Worker.__bases__`);
the subclass installs at the identical point, so it answers the same question
the production monkey-patch would. Because `worker_cls` is resolved by qualified
name in the worker subprocess, a PASS here is representative of multi-GPU and of
`vllm serve` / AsyncLLM (which use the same machinery).

Decisions:

- **PASS** → the production install path lands the op in the captured graph;
  the install-mechanism risk is retired (not just the driver-harness one).
- **prefill-only FAIL** → the wrap landed after the decode graph was captured;
  move the install earlier (the monkey-patch must run before
  `compile_or_warm_up_model`).
- **`--async` inconclusive** → the async harness is version-sensitive; the
  offline `worker_cls` result already validates the worker-process path.

---

## Step 0.9 — Memory budget (VRAM + KV-cache impact) — ½ day

Only after Step 0 PASSes. Runs in parallel with Step 0.5 / 0.7 / 0.8.

```bash
python tests/cuda_graph/step0_9_mem_budget.py \
    --model <model-you-can-access> --policy last_token
# stress the worst case (per-step / all-tokens sizing):
python tests/cuda_graph/step0_9_mem_budget.py --model <model> --policy all_tokens
```

Step 0.7 measures the *latency* idle tax. Step 0.9 measures the *other* cost —
**VRAM** — which is usually the bigger throughput risk at scale: static capture
buffers occupy memory that would otherwise be KV cache, so fewer concurrent
requests fit. The step0_7 counter op allocates ~nothing, so it cannot measure
this; Step 0.9 allocates **real-shaped** per-layer buffers instead.

What it does (baseline and buffers boot in **separate processes** so the KV
numbers are apples-to-apples):

1. Allocates per-layer static buffers (capture rank only) sized by policy —
   `last_token`: `(max_num_seqs+1) × hidden`; `all_tokens`:
   `max_num_batched_tokens × hidden` — **before** vLLM profiles the KV cache,
   so their effect on `num_gpu_blocks` is real.
2. Reports **measured vs predicted** static footprint and buffers as a % of
   device VRAM.
3. Reports **KV-block retention** = `num_gpu_blocks(buffers) / num_gpu_blocks(baseline)`.
4. **Churns** many short requests, samples worker `memory_allocated`, and
   reports drift — confirming the footprint stays flat (no growth/leak).

Pass criteria (defaults; override on the CLI):

- KV-block retention ≥ `--min-kv-retention` (default **0.90**)
- churn drift ≤ `--max-drift-pct` (default **2.0%**) over `--churn-requests` (200)

Failure pivots: switch to `last_token` sizing, narrow the activatable set via
`VLLM_HOOK_ACTIVE_LAYERS`, or accept fewer concurrent requests and document it.

**Scope honesty:** this validates the *static-buffer* design (footprint, KV
impact, boundedness). It does **not** validate the not-yet-built per-request
capture egress / row-allocator (free-on-completion, no stale-row reads) — that
is a Step-2 correctness item once the real ops exist.

---

## What passing all gates proves

| Gate | Settles | If green |
|---|---|---|
| Step 0 | C1, C4 (driver install) | Class-level wrap lands in the captured graph and replays |
| Step 0.5 | C5 per-mode | Which `CUDAGraphMode`s the plan ships under |
| Step 0.7 | Idle-tax floor (latency) | Per-layer launch cost is within the throughput budget |
| Step 0.8 | C4 via the production install path | Worker-side install at `load_model` lands in the graph; valid for serve/multi-GPU |
| Step 0.9 | Memory budget | Static-buffer VRAM + KV-cache impact within budget; footprint bounded under churn |

Greens on all of them mean the *mechanism is feasible and competitive*
on the dev model — in **both** time and memory — **and** the real install
path (not just the driver harness) works. The remaining validation (Steps
1-7 in the plan) is correctness and family generality work — substantial,
but no longer load-bearing on a single unproven claim.

A red on any one gate is the cheapest possible signal that the plan
needs to be revised (or that the goal needs to be re-stated as
"piecewise-only" or "eager-island instead"). The point of running
these first is to *find out before building*.

---

## Running on the LSF cluster (`bsub`)

The GPU resources live on a central LSF cluster. Each gate ships with a
small `#BSUB`-headed wrapper that requests 1 GPU + 4 CPUs + 32 GB RAM in
exclusive process mode — the same shape as `claude_docs/run_demo.sh`.

### Submit one gate at a time

```bash
# From the project root on the LSF login node, on the graph_enable branch:
bsub < tests/cuda_graph/run_step0_injection.sh
# → "Job <NNNN> is submitted to default queue <normal>."
bjobs                                         # track
bpeek <NNNN>                                  # live stdout
tail -f tests/cuda_graph/logs/step0.<NNNN>.out  # after it lands on a node
```

The same command form works for `run_step0_5_mode_matrix.sh`,
`run_step0_7_idle_tax.sh`, `run_step0_8_plugin_install.sh` and
`run_step0_9_mem_budget.sh`. Run order: **step0 first**; step0_5, step0_7,
step0_8 and step0_9 can run in any order (or in parallel) once step0 is green.

### Submit all gates with dependencies

```bash
bash tests/cuda_graph/submit_all_gates.sh
```

This is a login-node helper (don't bsub it — it just calls `bsub` itself).
It submits step0 first, captures its job ID, and queues
step0_5, step0_7, step0_8 and step0_9 each with `-w "done(<step0-id>)"` so they only run
if step0 finishes cleanly. If step0 fails, LSF leaves the dependents
pending — `bkill` them and diagnose step0 first.

### Overriding defaults

The wrappers honour env vars so you can tweak without editing the file:

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct bsub < tests/cuda_graph/run_step0_injection.sh
MODE=FULL_DECODE_ONLY                  bsub < tests/cuda_graph/run_step0_injection.sh
BATCHES="1 4 16 64"                    bsub < tests/cuda_graph/run_step0_7_idle_tax.sh
```

### Logs

Each gate writes to `tests/cuda_graph/logs/<gate>.<JOBID>.{out,err}`.
The directory is auto-created by the wrappers. Keep the
`PASS`/`FAIL`/`INCONCLUSIVE` line from each `.out` for the writeup.

## Hand-off checklist for the GPU run

Before the first submission, confirm on the login/compute node:

- [ ] `conda activate vllm_hook_env` works (matches `run_demo.sh`)
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `python -c "import vllm; print(vllm.__version__)"` ≥ `0.7.0`
- [ ] Default model `Qwen/Qwen2-1.5B-Instruct` is **ungated** — no HF login
      needed. Only `huggingface-cli login` / `HF_TOKEN` if you pass a gated
      `--model` (e.g. `google/gemma-3-4b-it`).
- [ ] The dev model fits at `gpu_memory_utilization=0.85`, `max_model_len=512`
      (Qwen2-1.5B ≈ 3-4 GiB; the wrapper's 1-GPU allocation is ample)
- [ ] You're on the `graph_enable` branch

Time budget: all three gates run in **half a day to a day** on one
H100 / A100 once they reach the head of the queue. If they don't,
something is wrong with the harness, not the plan.
