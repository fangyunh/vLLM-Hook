# CUDA-graph feasibility verification

Branch: `graph_enable`. Plan: `claude_docs/cuda_graph_plan.html`.

These scripts run the **gate steps** from the plan's validation plan, in
order. They settle the load-bearing claims (C1, C4, C5, and the
idle-tax floor) **before** any implementation work — so a `FAIL` here
saves weeks of correctness coding on a mechanism that wouldn't have
worked anyway.

```
tests/cuda_graph/
├── step0_injection.py       ← decisive: does the wrap land in the captured graph?
├── step0_5_mode_matrix.py   ← which CUDAGraphModes survive the wrap?
├── step0_7_idle_tax.py      ← does the per-layer launch cost fit the throughput budget?
├── run_step0_injection.sh        ← LSF batch wrapper (1 GPU)
├── run_step0_5_mode_matrix.sh    ← LSF batch wrapper (1 GPU)
├── run_step0_7_idle_tax.sh       ← LSF batch wrapper (1 GPU)
└── submit_all_gates.sh           ← login-node helper; chains the three with bsub -w
```

All three run on a **single GPU host** with vLLM ≥ 0.7 installed. They
do not modify any project source; they import vLLM and the plugin only
when invoked.

---

## Prerequisites

```bash
# On the GPU host:
pip install -r requirement.txt           # vLLM + zstandard + safetensors
pip install -e vllm_hook_plugins         # this project, in editable mode
huggingface-cli login                    # for the gated dev model below
```

Recommended dev model: **`google/gemma-3-4b-it`** (small, fast,
standard decoder-layer class). All scripts default to it.

---

## Step 0 — Injection (the decisive kill-switch) — ½ day

Run:

```bash
python tests/cuda_graph/step0_injection.py \
    --model google/gemma-3-4b-it \
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
    --model google/gemma-3-4b-it
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

- If your vLLM build does not honour `VLLM_CUDAGRAPH_MODE`, the script
  will still run but every mode collapses to whatever the default is.
  Confirm in the boot logs (`Using cudagraph mode = ...`).
- `FULL_DECODE_ONLY` requires that prefill and decode go through
  different paths; some older vLLM versions require an extra flag —
  check `vllm/config/compilation.py` on your version.

---

## Step 0.7 — Idle-tax budget — ½ day

Only after Step 0 PASSes. Step 0.7 does NOT depend on Step 0.5 passing
— it can run in parallel.

```bash
python tests/cuda_graph/step0_7_idle_tax.py \
    --model google/gemma-3-4b-it \
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

---

## What passing all three gates proves

| Gate | Settles | If green |
|---|---|---|
| Step 0 | C1, C4 | Class-level wrap lands in the captured graph and replays |
| Step 0.5 | C5 per-mode | Which `CUDAGraphMode`s the plan ships under |
| Step 0.7 | Idle-tax floor | Per-layer launch cost is within the throughput budget |

Greens on all three means the *mechanism is feasible and competitive*
on the dev model. The remaining validation (Steps 1-7 in the plan) is
correctness and family generality work — substantial, but no longer
load-bearing on a single unproven claim.

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

The same command form works for `run_step0_5_mode_matrix.sh` and
`run_step0_7_idle_tax.sh`. Run order: **step0 first**; step0_5 and
step0_7 can run in either order (or in parallel) once step0 is green.

### Submit all three with dependencies

```bash
bash tests/cuda_graph/submit_all_gates.sh
```

This is a login-node helper (don't bsub it — it just calls `bsub` three
times itself). It submits step0 first, captures its job ID, and queues
step0_5 and step0_7 each with `-w "done(<step0-id>)"` so they only run
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
- [ ] `huggingface-cli whoami` is logged in (for the gated dev model);
      or `HF_TOKEN` set in the env before `bsub`
- [ ] Gemma-3-4B fits at `gpu_memory_utilization=0.85`, `max_model_len=512`
      (~9 GiB GPU mem; the wrapper's 1-GPU allocation is sufficient)
- [ ] You're on the `graph_enable` branch

Time budget: all three gates run in **half a day to a day** on one
H100 / A100 once they reach the head of the queue. If they don't,
something is wrong with the harness, not the plan.
