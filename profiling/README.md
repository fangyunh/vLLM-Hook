# profiling/ — vLLM-Hook performance harness

Implements the plan in [`claude_docs/profiling_plan.md`](../claude_docs/profiling_plan.md).

What it does:
- A process-local profiler (`vllm_hook_plugins/_profiler.py`) collects
  per-stage timers, counters, and gauges. Activated by `VLLM_HOOK_PROFILE=1`;
  fully no-op when disabled, so production performance is unchanged.
- A grid sweep (`bench_grid.py`) drives the six benchmark rows from
  `plan.html` §7 plus optional full-storage / full-axis sweeps; writes CSV +
  JSONL with every tier-1 metric and the per-cell profile snapshot.
- A serve-path throughput driver (`bench_throughput_serve.py`) reproduces
  the asyncio-loop jam from `plan.html` §7.
- An idle-tax bench (`bench_idle_tax.py`) measures the per-layer launch
  cost — the same number `tests/cuda_graph/step0_7_idle_tax.py` would
  measure with a fully wrapped plugin.
- Microbenches isolate hook fire, storage, serialization, and analyzer
  cost so a surprising grid number can be attributed cleanly.
- `analyze/` turns the CSV into a markdown report (`summarize.py`),
  heatmaps (`plot_grid.py`), and a hook-vs-baseline diff
  (`compare_baselines.py`).

---

## Quick start (single GPU)

```bash
# 1) Run the R0-R5 deliverable (plan.html §7). ~30 min on A100.
python profiling/bench_grid.py --plan plan-quick \
    --model Qwen/Qwen2-1.5B-Instruct \
    --output profiling/results/quick.csv

# 2) Summarize.
python profiling/analyze/summarize.py profiling/results/quick.csv \
    --output profiling/results/quick.md

# 3) Idle-tax verdict (PASS/FAIL vs cuda_graph_plan.html budgets).
python profiling/bench_idle_tax.py \
    --batches 1 8 64 --output profiling/results/idle.csv

# 4) Serve-path jam (needs vllm in PATH).
python profiling/bench_throughput_serve.py \
    --workers baseline,qk,hidden_states --duration 30 \
    --concurrency 1 2 4 8 --output profiling/results/serve.csv
```

---

## CSV / JSONL schema

Every row carries §1.7 context (model, dtype, worker, storage_variant,
mode, prompt_len, batch_size, n_layers_captured, ...), the headline
metrics (`gen_lat_mean`, `analyze_lat_mean`, `peak_gpu_mb`,
`artifact_kb_mean`, `prefill_tok_per_sec`, `decode_tok_per_sec`), and
flattened profiler snapshots:

```
timer.<name>.mean      timer.<name>.p50      timer.<name>.p99
gauge.<name>.mean      counter.<name>
```

So e.g. the per-fire cost of the hidden-states hook on a particular cell
lives in `gauge.captured.bytes.hs.sum` and the disk write time in
`timer.worker.disk_write.safetensors.mean`.

JSONL is one record per cell with the **raw** sample arrays — re-percentile
without re-running.

---

## Env vars (profiler)

| Env var | Default | Effect |
|---|---|---|
| `VLLM_HOOK_PROFILE` | `0` | Master switch. Off → every wrap is a no-op. |
| `VLLM_HOOK_PROFILE_FINE` | `0` | Tier-2 timers (CUDA events, per-fire GPU clone). Implies `_CUDA`. |
| `VLLM_HOOK_PROFILE_CUDA` | `0` | Allow CUDA-event timers (forces a sync — distorts timing). |
| `VLLM_HOOK_PROFILE_MEM` | `0` | Start the NVML+RSS background sampler. |
| `VLLM_HOOK_PROFILE_DIR` | `/tmp/vllm_hook_profile` | Where `PROF.dump()` writes JSON. |

---

## Adding a new metric

1. Identify the call site. Wrap with `with PROF.timed("my.timer"):` or
   `PROF.incr("my.counter")`.
2. The harness automatically flattens it into `timer.my.timer.*` and
   `counter.my.counter` columns.
3. Optionally add a row in `analyze/summarize.py` to surface it in the
   markdown report.

---

## Cluster runners (LSF)

See [`profiling/runners/README.md`](runners/README.md) for the full
scheduling guide. Short version:

```bash
# Login node — queue the smoke check + R0..R5 + idle-tax with dependencies:
bash profiling/runners/submit_all.sh

# Individual jobs:
MODEL=ibm-granite/granite-3.1-8b-instruct \
    bsub < profiling/runners/run_quick.sh

PROMPT_LENS="16 64 256 512" BATCH_SIZES="1 8" \
    bsub < profiling/runners/run_storage.sh

WORKERS="baseline qk hidden_states" CONCURRENCY="1 2 4 8" DURATION=60 \
    bsub < profiling/runners/run_serve.sh

bsub < profiling/runners/run_full.sh        # overnight
```

Each runner sources `~/miniconda3/etc/profile.d/conda.sh`, activates
`vllm_hook_env`, sets the profiler env vars, writes results under
`profiling/results/` (suffixed with the LSF job ID), and — for grid
runners — auto-summarizes into a sibling `*-summary.md`. Logs land in
`profiling/runners/logs/<name>.<JOBID>.{out,err}`.

---

## Layout

```
profiling/
├── _common.py                       # prompt synth, env snapshots, NVML peek, CSV/JSONL io
├── profile_one_run.py               # single-cell driver
├── bench_grid.py                    # plan-quick / plan-storage / plan-full
├── bench_throughput_serve.py        # vllm serve + k concurrent clients
├── bench_idle_tax.py                # baseline vs plugin_loaded vs hooks_idle
├── microbench/
│   ├── microbench_hook_fire.py      # isolate per-fire CPU cost (HS worker)
│   ├── microbench_storage_variants.py # rpc / disk-pt / disk-st, no model
│   └── microbench_serialize.py      # tolist+JSON vs bytes+zstd+base64
├── analyze/
│   ├── summarize.py                 # CSV → markdown headline report
│   ├── plot_grid.py                 # heatmaps over (n_layers × prompt_len)
│   └── compare_baselines.py         # hook rows vs baseline rows
├── runners/
│   ├── run_offline_grid.sh
│   ├── run_serve_throughput.sh
│   └── run_idle_tax.sh
├── workloads/                       # (reserved for future real-prompt datasets)
└── results/                         # CSVs land here; tracked .gitignored if you prefer
```

---

## Things this harness does *not* test (yet)

- Real (non-synthetic) prompt distributions on the serve path. Drop a
  `workloads/sharegpt_sample.jsonl` and extend `bench_throughput_serve.py`
  to consume it.
- TP > 1. The driver builds engines with `tensor_parallel_size=N`, but no
  rank-aware result-merging across multiple sampler workers yet.
- The proposed `bytes → zstd → base64` HTTP wire format. The microbench
  measures it in isolation; landing it as an opt-in flag is the next PR.
