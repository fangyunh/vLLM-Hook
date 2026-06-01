# profiling/runners — LSF (bsub) job scripts for the GPU cluster

Submits the profiling benchmarks from `profiling/` to the LSF cluster
described in `claude_docs/submit_job.txt`. Style and conventions match
`tests/cuda_graph/run_step0_*.sh` and `claude_docs/run_demo.sh`:

- `#BSUB` headers (job name, GPU, CPU, memory, wall time, logs).
- `source ~/miniconda3/etc/profile.d/conda.sh` + `conda activate vllm_hook_env`.
- `cd ~/vLLM-Hook` — all paths inside the script are repo-relative.
- Override knobs via env vars passed before `bsub`.
- One GPU per job by default; `-R "rusage[ngpus=1,mem=…]"`.

---

## What you get

| Script | Purpose | Wall time | Outputs |
|---|---|---|---|
| `run_smoke.sh` | 5-min profiler sanity check — runs `examples/demo_hiddenstate.py` with profiling on, prints the top timers. **Run this first on a new node.** | ~5 min | console; `/dev/shm/vllm_hook_profile_smoke/profile-*.json` |
| `run_quick.sh` | The R0..R5 deliverable from `plan.html` §7. | ~30 min | `profiling/results/quick-*.csv`, `*.jsonl`, `*-summary.md` |
| `run_storage.sh` | Storage-variant matrix (replaces the prelim table in `configs.md`). | ~2-4 hr | `storage-*.csv`, `*-summary.md`, `*-heatmap.png` |
| `run_idle_tax.sh` | Per-layer plugin-tax verdict (PASS/FAIL vs `cuda_graph_plan.html` budgets). | ~15 min | `idle-*.csv`; non-zero exit on FAIL |
| `run_serve.sh` | Serve-path throughput. Reproduces the `plan.html` §7 asyncio jam. | ~30-60 min | `serve-*.csv` |
| `run_full.sh` | Overnight cartesian sweep (`plan-full`). Resumable. | 4-12 hr | `full-*.csv`, `*-summary.md` |
| `submit_all.sh` | **Login-node** helper: queues `run_smoke` → (`run_quick` ∥ `run_idle_tax`) with `done(...)` dependencies. | — | n/a |

Logs land in `profiling/runners/logs/<name>.<JOBID>.{out,err}`. The CSV /
JSON / markdown outputs land in `profiling/results/` and are suffixed with
the LSF job id so concurrent jobs never clobber each other.

---

## One-time setup (per cluster node / user)

```bash
# Login node:
git clone https://github.com/IBM/vLLM-Hook.git   # if not already cloned
cd vLLM-Hook
git checkout graph_enable

# Confirm conda env exists; create if needed.
conda env list | grep -q vllm_hook_env || conda create -n vllm_hook_env python=3.11
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_hook_env

pip install -r requirement.txt
pip install -e vllm_hook_plugins
pip install pynvml psutil matplotlib    # for memory sampler + plot_grid

# If you'll use a gated model (gemma, granite-private, etc.):
huggingface-cli login
```

Verify the profiler imports without errors on the login node:

```bash
python -c "from vllm_hook_plugins._profiler import PROF, is_enabled; print('ok')"
```

(This does not need a GPU. The smoke job does.)

---

## The standard workflow (recommended order)

```bash
# 1) On the login node, queue the three-step "first deliverable":
bash profiling/runners/submit_all.sh
```

That prints three job IDs and the topology:

```
smoke    : 123456   (runs immediately when a GPU is free)
quick    : 123457   (waits for smoke done)
idle_tax : 123458   (waits for smoke done)
```

```bash
# 2) Track them:
bjobs                                                  # all your jobs
bpeek  123456                                          # live stdout (smoke)
tail -f profiling/runners/logs/smoke.123456.out        # after it lands
```

`smoke` writes a one-shot timer table to its log; if you see
non-empty `hook.fire.hs` and `worker.cpu_transfer.hs` rows it's healthy.
A blank timers section means `VLLM_HOOK_PROFILE=1` didn't reach the worker
subprocess — see "If smoke fails" below.

```bash
# 3) After quick + idle_tax finish, the results are in:
ls profiling/results/

# The advisor table (R0..R5):
cat profiling/results/quick-*-123457-summary.md

# The idle-tax verdict:
cat profiling/runners/logs/idle.123458.out
```

That's everything `plan.html` §7's first deliverable asks for.

---

## Submitting individual jobs

Each script is self-contained — `bsub < <script>` works in isolation:

```bash
# R0..R5 with a bigger model:
MODEL=ibm-granite/granite-3.1-8b-instruct \
    bsub < profiling/runners/run_quick.sh

# Storage matrix at 4 prompt lengths × 2 batch sizes:
PROMPT_LENS="16 64 256 512" BATCH_SIZES="1 8" \
    bsub < profiling/runners/run_storage.sh

# Idle-tax with bigger decode windows:
BATCHES="1 4 16 64" N_DECODE=256 \
    bsub < profiling/runners/run_idle_tax.sh

# Serve-path jam:
WORKERS="baseline qk hidden_states" CONCURRENCY="1 2 4 8 16" DURATION=60 \
    bsub < profiling/runners/run_serve.sh

# Full overnight sweep:
bsub < profiling/runners/run_full.sh
```

Every script honors these overrides (see the script header for the full
list): `MODEL`, `PROMPT_LENS`, `BATCH_SIZES`, `MAX_TOKENS`, `REPS`,
`WARMUP`, `HOOK_DIR`, `PROFILE_DIR`, `GPU_MEM_UTIL`. Specific scripts have
extras: `BATCHES`, `N_DECODE` (idle_tax); `WORKERS`, `CONCURRENCY`,
`DURATION`, `PORT` (serve); `START_FROM`, `OUT_CSV` (full, for resume).

---

## Monitoring

| Need | Command |
|---|---|
| All your queued/running jobs | `bjobs` |
| One job in detail | `bjobs -l <JOBID>` |
| Live stdout of a running job | `bpeek <JOBID>` |
| Tail the .out file after it lands on a node | `tail -f profiling/runners/logs/<name>.<JOBID>.out` |
| Tail the .err file | `tail -f profiling/runners/logs/<name>.<JOBID>.err` |
| Cancel a job | `bkill <JOBID>` |
| Why is my job pending? | `bjobs -l <JOBID>` — look at PENDING REASONS |
| Available queues | `bqueues` |

The grid sweeps write the CSV after **every cell**, so even an in-progress
job has partial results:

```bash
head -5 profiling/results/quick-*-<JOBID>.csv
```

---

## Reading the outputs

After `run_quick.sh` (or any grid sweep) finishes you get four artifacts:

```
profiling/results/quick-<model>-<jobid>.csv         # one row per cell × rep
profiling/results/quick-<model>-<jobid>.jsonl       # per-cell raw timer samples
profiling/results/quick-<model>-<jobid>-summary.md  # human-readable headline tables
profiling/runners/logs/quick.<jobid>.out            # what the job printed
```

The CSV columns:

| Column pattern | Meaning |
|---|---|
| Cell context | `model`, `worker`, `storage_variant`, `mode`, `prompt_len`, `batch_size`, `max_tokens`, … |
| Headline metrics | `gen_lat_mean`, `wait_lat_mean`, `analyze_lat_mean`, `peak_gpu_mb`, `artifact_kb_mean`, `prefill_tok_per_sec`, `decode_tok_per_sec` |
| Per-stage timers | `timer.<name>.mean`, `timer.<name>.p99` (e.g. `timer.worker.cpu_transfer.hs.mean`) — units are **ms** |
| Gauges | `gauge.<name>.sum`, `gauge.<name>.max` (e.g. `gauge.captured.bytes.hs.sum` — bytes) |
| Counters | `counter.<name>` (e.g. `counter.hook.fire.hs` — total fires this cell) |

To re-render the summary or generate the heatmap after the fact:

```bash
python profiling/analyze/summarize.py \
    profiling/results/quick-<model>-<jobid>.csv \
    --output /tmp/quick.md

python profiling/analyze/plot_grid.py \
    profiling/results/storage-<model>-<jobid>.csv \
    --metric gen_lat_mean --group-by storage_variant \
    --output /tmp/heatmap.png

python profiling/analyze/compare_baselines.py \
    profiling/results/quick-<model>-<jobid>.csv
```

---

## If smoke fails

The most common failure: the smoke job runs cleanly but `profile-*.json`
is empty or has no `hook.*` timers. Diagnosis (in order):

1. **Did the env var reach the worker subprocess?**
   `VLLM_WORKER_MULTIPROC_METHOD=spawn` means the worker inherits the
   driver's env *at spawn time*. The bsub script exports `VLLM_HOOK_PROFILE`
   *before* it calls Python — verify with:
   ```bash
   grep -E "VLLM_HOOK_PROFILE" profiling/runners/logs/smoke.<JOBID>.out
   ```
2. **Did the example pass `save_to_disk=True`?**
   The smoke job uses `examples/demo_hiddenstate.py`, which does. If you
   point smoke at a different example, check it actually exercises a
   capture path.
3. **Plugin not editable-installed?** `pip install -e vllm_hook_plugins`
   on the **compute node** sometimes differs from what was installed on the
   login node. Re-run `pip install -e vllm_hook_plugins` inside an
   interactive job:
   ```bash
   bsub -Is -gpu num=1 -R "rusage[ngpus=1,mem=16GB]" /bin/bash
   conda activate vllm_hook_env
   pip install -e vllm_hook_plugins
   exit
   ```

---

## Chaining: how to add more dependents

`submit_all.sh` queues the smoke→{quick,idle_tax} fan-out. To extend
that chain (e.g. add storage after quick, serve after idle_tax), copy
the pattern:

```bash
QUICK_ID=$(bsub -w "done($SMOKE_ID)" < profiling/runners/run_quick.sh \
            | awk '{print $2}' | tr -d '<>')

STORAGE_ID=$(bsub -w "done($QUICK_ID)" < profiling/runners/run_storage.sh \
              | awk '{print $2}' | tr -d '<>')

SERVE_ID=$(bsub -w "done($IDLE_ID)" < profiling/runners/run_serve.sh \
            | awk '{print $2}' | tr -d '<>')
```

LSF's `done(<id>)` means the dependent only starts if the parent exited
**successfully**. If you want it to start regardless of exit status use
`ended(<id>)`.

---

## Notes specific to this cluster

- `~/miniconda3/etc/profile.d/conda.sh` is hard-coded — change it if your
  conda lives elsewhere (e.g. `/opt/conda/etc/profile.d/conda.sh`).
- `/dev/shm/` is the recommended `HOOK_DIR` and `PROFILE_DIR` host — it's
  tmpfs (RAM-backed), so artifact writes don't go to spinning disk.
- The `-W` wall-time on each `#BSUB` header is a soft cap; raise it if
  your model is much larger than Qwen2-1.5B (granite-3.1-8b is ~3× slower
  per cell).
- All scripts use `set -euo pipefail`. A single failing Python step kills
  the whole job — that's intentional (chained jobs should not see partial
  results) but means a crash mid-grid leaves only the rows checkpointed up
  to that cell. Use `START_FROM` on `run_full.sh` to resume.
