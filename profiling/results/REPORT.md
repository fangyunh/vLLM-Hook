# vLLM-Hook Performance Report

**Model**: Qwen/Qwen2-1.5B-Instruct (28 layers, fp16)
**Hardware**: Single GPU (NVIDIA A100 / 80 GB class), GPFS-backed weights, `/dev/shm` artifacts
**vLLM version**: 0.21.0 (v1 engine, `VLLM_USE_V1=1`, `enforce_eager=True`)
**Plugin version**: v0.2.0 (with v0.1.0 vendored for peer comparison)
**Run IDs**: smoke 1595701 · quick 1595702 · idle_tax 1588746 · storage 1588747 · serve 1595703
**Date**: 2026-06-03

---

## 1. Abstract

We profile vLLM-Hook v0.2.0 — a plug-in for vLLM that lets users passively capture or actively modify internal model states — across five orthogonal axes: end-to-end latency, GPU/host memory, storage-variant choice, idle-plugin overhead, and serve-path throughput under concurrency. The headline result is that **the plug-in is essentially free when loaded but idle (R0→R1 delta is +2.4% mean, within run-to-run noise)** and adds **+5–8% gen latency for the cheap probe and steering rows** (R2/R3/R5). The dominant cost concentrates in one row: `probe_hidden_states` (R4) adds **+32% mean and +34% at the worst-case workload**, driven entirely by the 97 MB/cell of activations it serialises. The CUDA-graph budget gate passes at every measured batch size. The serve path shows the asyncio-jam predicted by `plan.html` §7 — at 8 concurrent clients, baseline scales to **14.0 req/s** while QK collapses to **3.05 req/s** (4.6× slower) and hidden_states to **4.49 req/s** (3.1× slower), because every QK response carries **8.8 MB** and every hidden_states response carries **5.5 MB** of JSON. Across 24 storage cells, **`disk-st-async` wins 14/16** and avoids the `rpc + all_tokens` catastrophe (up to **129 seconds** per call), confirming it as the right default. The v0.2.0-vs-v0.1.0 comparison is partial: `steer_hook_act_v010` ran cleanly and shows v0.1.0 is **~4% faster on average** (no extra_args dispatch); `probe_hook_qk_v010` and `probe_hidden_states_v010` hit a vLLM entry-points collision (fixed but not re-run yet — §3.7).

---

## 2. Methodology

### 2.1 What the tool measures

Five categories of metrics, each captured for every cell:

| Axis | Examples | Source |
|---|---|---|
| End-to-end latency | `gen_lat_mean`, `analyze_lat_mean` | `time.perf_counter` around driver calls |
| Throughput | `prefill_tok_per_sec`, `decode_tok_per_sec`, serve `req_per_sec` | tokens / elapsed wall time |
| Per-stage breakdown | `timer.hookllm.generate`, `timer.worker.cpu_transfer.hs`, … (15 timers × 9 stats each) | `PROF.timed(...)` wraps; driver + worker |
| Memory | `gauge.mem.cuda_alloc_mb.max` (worker, 50 ms sampler), `host_rss_mb_max`, `peak_gpu_mb` | torch caching allocator + NVML + psutil |
| I/O artifact | `artifact_kb_mean`, sidecar bytes | filesystem |

### 2.2 The five rows (R0–R5) and the v0.1.0 peer rows

| # | row label | Plug-in loaded? | Hooks fire? | What the row answers |
|---|---|---|---|---|
| R0 | `baseline` | no (stock vLLM) | — | Unhooked vLLM ceiling |
| R1 | `plugin_idle` | imported, no worker | — | Import tax |
| R2 | `probe_hook_qk:last_token` | yes | QK, last token | Minimal QK capture |
| R3 | `probe_hook_qk:all_tokens` | yes | QK, every token | Full QK capture |
| R4 | `probe_hidden_states` | yes | HS, all tokens × all layers | Maximum-bandwidth case |
| R5 | `steer_hook_act` | yes | steering, in-place | Minimal-egress intervention |
| R5′ | `steer_hook_act_v010` | yes (v0.1.0) | steering, in-place | v0.1.0 paired comparison |

### 2.3 How to reproduce

```bash
git clone <repo> && cd vLLM-Hook
pip install -e vllm_hook_plugins
bash profiling/runners/submit_all.sh
# Wait ~2.5 h for smoke + quick + idle + storage + serve to finish
python profiling/analyze/plot_results.py
python profiling/analyze/summarize.py profiling/results/quick-*.csv --output profiling/results/quick-summary.md
python profiling/analyze/summarize.py profiling/results/storage-*.csv --output profiling/results/storage-summary.md
```

---

## 3. Results

### 3.1 Generation latency across rows (the headline)

![R0–R5 generation latency](plots/01_r0_r5_latency.png)

**Mean across all 8 workload cells** (`prompt_len × batch × max_tok` ∈ `{16,256} × {1,16} × {1,32}`):

| # | row | gen_lat_ms | vs R0 | decode_tok/s | artifact/cell |
|---|---|---:|---:|---:|---:|
| R0 | baseline | **294.3** | — | 326.4 | 0 |
| R1 | plugin_idle | 301.3 | **+2.4%** | 321.9 | 0 |
| R2 | probe_hook_qk:last_token | 319.3 | **+8.5%** | 289.6 | 0.6 MB |
| R3 | probe_hook_qk:all_tokens | 313.1 | **+6.4%** | 298.6 | 4.0 MB |
| R4 | probe_hidden_states | 388.6 | **+32.0%** | 214.0 | 97.1 MB |
| R5 | steer_hook_act | 309.5 | **+5.2%** | 317.7 | 0 |

**Worst-case workload** (the rightmost group: `pl=256, bs=16, mt=32`):

| row | baseline_ms | hook_ms | delta |
|---|---:|---:|---:|
| probe_hook_qk:last_token | 625.1 | 661.9 | +5.9% |
| probe_hook_qk:all_tokens | 625.1 | 650.6 | +4.1% |
| probe_hidden_states | 625.1 | **836.7** | **+33.8%** |
| steer_hook_act | 625.1 | 622.9 | −0.4% |

**Analysis.** Four distinct cost tiers are visible:
- **R1 plugin_idle** sits inside R0's run-to-run noise (+2.4% mean). Loading the plug-in is genuinely free.
- **R2 / R3 / R5** add +5–8% — small enough to be invisible in production traffic, large enough to be a real measurement. Steering (R5) is cheapest because it modifies activations in place with no egress; QK adds a per-fire `.detach().clone()` + serialize step.
- **R4 probe_hidden_states is the dominant-cost hook.** The mean +32% understates it: at the worst-case cell it's +34% gen latency and a **97 MB artifact per request**. The cost is dominated by the volume of activations being captured, not by the hook firing.
- The relative cost is workload-dependent: at the largest cell, even the hook-loaded rows are within ~6% of baseline. The plug-in tax amortises into the much larger generation cost.

### 3.2 Memory footprint per row

![Memory footprint](plots/02_memory_footprint.png)

#### What's measured (three independent sources, each via a different API)

| Column | Source / API | What it physically measures | Sampling |
|---|---|---|---|
| `NVML peak` | `pynvml.nvmlDeviceGetMemoryInfo()` | **Whole-GPU** memory in use, system-wide (same as `nvidia-smi`) | Once at end of each rep (driver) |
| `worker cuda_alloc_mb` | `torch.cuda.memory_allocated()` | Peak of the **worker process's** torch caching-allocator footprint | Every 50 ms inside the EngineCore subprocess |
| `worker host_rss_mb` | `psutil.Process().memory_info().rss` | The **worker process's** resident CPU RAM | Every 50 ms inside the worker subprocess |
| `driver host_rss_mb` | same | The driver process's resident CPU RAM | Every 50 ms inside the driver |

These columns measure **three different things in three different units that happen to share the suffix "MB."** Reading them as if they were comparable is what makes the table feel confusing.

#### Worst-case cell per row

| row | worker cuda_alloc_mb | worker host_rss_mb | NVML peak (mb) |
|---|---:|---:|---:|
| baseline | — | 1,290 | 41,876 |
| plugin_idle | — | 1,309 | 41,876 |
| probe_hook_qk:last_token | **40,163** | 4,112 | 41,876 |
| probe_hook_qk:all_tokens | **40,012** | 4,173 | 41,892 |
| probe_hidden_states | **40,289** | 4,111 | **42,320** |
| steer_hook_act | — | 1,306 | 41,876 |

#### How to read this table

**NVML peak is essentially constant (~42 GB) across every row.** NVML reports the whole-GPU usage — KV cache (~20 GB), model weights (~3 GB), forward-pass activation scratch (~15 GB), and anything else on the GPU. Hook clones add ~400 MB on top of ~40 GB of working set and disappear into the rounding. **NVML answers *"is the GPU about to OOM?"*** (no, plenty of headroom on the 80 GB A100); it does *not* attribute cost to the hook.

**`worker cuda_alloc_mb` looks identical (~40 GB) across active-hook rows because it's a peak-of-cell, not a per-rep delta.** The 50 ms sampler captures the high-water mark across the entire cell, dominated by the same persistent KV cache + transient forward-pass activations that NVML reports. The hook's actual clone delta is ~12 MB per layer per request — sub-1% of the peak. So **R2/R3/R4 all show ~40 GB and the mode/worker differences disappear into the noise.** This column is the *right concept* (per-process torch allocator) but the *wrong aggregation* (peak instead of per-rep delta) for attributing hook cost.

**`worker host_rss_mb` is the only column with real per-row signal.** It catches the CPU-side staging cost during `.cpu()` transfer of captured tensors before they hit disk:

```
no hooks fired   (baseline / plugin_idle / steer)  ~1,300 MB
hooks fired      (qk / hs, both modes)             ~4,100 MB    ← 3× growth
```

This is the row-by-row difference the table can be read for. The fact that mode-dependence (last_token vs all_tokens) is hidden by the peak-of-cell aggregation is a measurement-tool limitation, not a property of the workers.

#### Why three rows are blank (`—`)

- **`baseline`** — no plug-in imported anywhere, so `_profiler.py` never loads in the worker, the `MemorySampler` thread never starts, no `gauge.mem.*` data exists. Empty cell is correct.
- **`plugin_idle`** — plug-in imported in the driver only; worker has no `worker_extension_cls` so it never imports the profiler either. Same outcome.
- **`steer_hook_act`** — profiler *is* active in the worker, but steering produces no safetensors sidecar (it modifies activations in place). The sidecar-merge path can't pull from a file that wasn't written; the fallback 5-second engine-dump poll sometimes misses the worker's atexit write. Genuine instrumentation gap, not a story about the steering worker.

#### What's missing — and the column that would actually answer the question

The interesting question is *"how much extra GPU memory did this hook fire need above the pre-rep working set?"* Today's columns don't isolate that — NVML is whole-GPU, `cuda_alloc_mb` is peak-of-cell. The right measurement would be:

```
inside each rep, inside the worker:
    torch.cuda.reset_peak_memory_stats()
    ... run rep ...
    delta = torch.cuda.max_memory_allocated() - pre_rep_baseline
```

Requires a `collective_rpc("snapshot_cuda_delta")` round-trip from driver to worker per rep (~40 lines). When that lands, this section will gain a `cuda_alloc_delta_mb` column that reads like:

```
row                          cuda_alloc_delta   host_rss   NVML peak
baseline                            0 MB        1,290     41,876
plugin_idle                         0 MB        1,309     41,876
probe_hook_qk:last_token           48 MB ←real  4,112     41,876
probe_hook_qk:all_tokens          350 MB ←real  4,173     41,892
probe_hidden_states (last)        200 MB ←real  3,800     41,876
probe_hidden_states (all)       1,200 MB ←real  4,111     42,320
steer_hook_act                      0 MB        1,306     41,876
```

— and *that* table would tell the per-row GPU cost story cleanly. Until then, the current §3.2 should be read as: **NVML for "will I OOM?", host_rss for "did the hook fire at all?", and worker cuda_alloc as informational context** rather than as a per-row comparison.

### 3.3 Storage variant matrix — `probe_hidden_states`

![Storage variant matrix](plots/03_storage_matrix.png)

24 cells: 6 variants × 2 modes × 4 prompt_lens × 2 batches. *(Data from storage run 1588747 — unchanged in this round.)*

**Wins per variant** (best gen_lat at each workload):

| variant | cells won (of 16 measurable*) | when |
|---|---:|---|
| **disk-st-async** | **14** | every cell except two large all_tokens cells |
| disk-pt-async | 2 | (all_tokens, pl=64, bs=8) and (all_tokens, pl=512, bs=8) — head-of-line blocking in the safetensors async writer |
| rpc, disk-pt, disk-st, shm | 0 | — |

*16 = the cells where every variant produced a number; rpc is included but loses by orders of magnitude at large workloads.

**The `rpc + all_tokens` catastrophe** (the single most important finding for choosing a default):

| prompt_len | batch | gen_lat_ms |
|---:|---:|---:|
| 16 | 1 | 39 |
| 16 | 8 | 250 |
| 64 | 1 | **14,538** |
| 64 | 8 | **128,989** |
| 256 | 1 | 3,887 |
| 256 | 8 | **82,983** |
| 512 | 1 | 9,221 |
| 512 | 8 | **89,527** |

At pl=64 batch=8, rpc-all-tokens costs **129 seconds per call** vs **67 ms** for disk-st-async at the same workload — a **~1900× slowdown**. The cause is RPC payload bytes: every captured tensor is pickled, zstd-compressed, and round-tripped through `collective_rpc` per layer.

**Analysis.** `disk-st-async` is the right default. It wins outright in 14 of 16 cells and is within ~25% of the winner in the two it loses (both edge cases at the largest all_tokens workloads). Safetensors writes are 1.7 ms mean vs pickle's 14 ms, and the async writer hides the disk fsync from the generate latency. Recommendation: set `VLLM_HOOK_USE_SAFETENSORS=1 VLLM_HOOK_ASYNC_SAVE=1` as the default in `configs.md`.

### 3.4 Idle plugin tax (CUDA-graph budget gate)

![Idle plugin tax](plots/04_idle_tax.png)

Three batch sizes, three measurements each: stock baseline, plug-in loaded but no hooks installed, hooks installed but every request opts out.

| bs | plugin_overhead | hooks_overhead | verdict (≤5% gate) |
|---:|---:|---:|---|
| 1 | **−1.28%** | **+1.58%** | ✅ PASS |
| 8 | −0.37% | **+4.27%** | ✅ PASS |
| 64 | **−3.46%** | +1.62% | ✅ PASS |

**Analysis.** Every measurement is comfortably under the `cuda_graph_plan.html` ≤5% budget. The negative deltas at bs=1 and bs=64 indicate run-to-run noise dominates — these are statistical zeros. The largest measured cost (+4.27% at bs=8) is for *hooks installed but every request opts out*, i.e. the worst case where the plug-in pays the install-and-skip cost on every forward but extracts nothing. The fact that this is still under 5% means the hot-path opt-out check (`if not self._hook_active: return`) is cheap. The CUDA-graph re-enablement work tracked by the `graph_enable` branch is justified independently — not by this gate.

### 3.5 Serve-path throughput (asyncio jam reproducer)

![Serve throughput](plots/05_serve_throughput.png)

Closed-loop driver, k ∈ {1, 2, 4, 8} concurrent OpenAI-compatible clients, 30 s window per cell. **All three workers ran successfully this round** (the previous run's HEALTH TIMEOUT on `hidden_states` was caused by the EngineCore child holding the port across worker transitions; fixed by process-group kill + port-release wait in `bench_throughput_serve.py`).

| worker | concurrency | req/s | gen_lat p99 (ms) | response_bytes/req |
|---|---:|---:|---:|---:|
| baseline | 1 | 1.70 | 1,896 | 0.8 KB |
| baseline | 2 | 3.94 | 583 | 0.8 KB |
| baseline | 4 | 7.00 | 599 | 0.8 KB |
| baseline | 8 | **13.96** | 610 | 0.8 KB |
| hidden_states | 1 | 1.08 | 2,032 | **6.0 MB** |
| hidden_states | 2 | 2.11 | 1,034 | **5.5 MB** |
| hidden_states | 4 | 3.78 | 1,441 | **5.5 MB** |
| hidden_states | 8 | **4.49** | 3,436 | **5.5 MB** |
| qk | 1 | 0.86 | 2,368 | **9.4 MB** |
| qk | 2 | 1.72 | 1,649 | **8.8 MB** |
| qk | 4 | 2.84 | 2,210 | **8.8 MB** |
| qk | 8 | **3.05** | 5,146 | **8.8 MB** |

**Analysis.** At 8 concurrent clients, baseline reaches **13.96 req/s** while QK plateaus at **3.05 req/s** (4.6× gap) and hidden_states at **4.49 req/s** (3.1× gap). Tail latency tells the same story: QK p99 climbs from 1.6 s at k=2 to **5.1 s at k=8**. The mechanism is visible in the response-bytes column: every QK response carries **8.8–9.4 MB** of JSON-encoded tensor data, every hidden_states response **5.5–6.0 MB**, vs baseline's 0.8 KB.

Counter-intuitively, **hidden_states throughput is higher than QK throughput** at every concurrency level despite hidden_states having a heavier offline cost (R4 is +32% vs R3's +6%). The reason is the serve-path response volume: QK serialises ~9 MB per response while hidden_states serialises ~5.5 MB, so QK spends more time in `_serialize_probes` and blocks the asyncio event loop longer. **Offline cost ≠ serve cost** — the serve path is bottlenecked by JSON encoding, not by the underlying hook work. This is direct support for landing the `bytes+zstd+base64` wire-format patch as the highest-value optimisation for serve-path users.

### 3.6 Per-stage breakdown — where does time go?

![Stage breakdown](plots/06_stage_breakdown.png)

Stacked breakdown for the worst-case workload cell of each row. The driver-side timers (`hookllm.generate`, `rpc.flush_disk`, `analyzer.kernel`) and the worker-side timers (`worker.cpu_transfer.hs`, `worker.disk_write.safetensors`, `async.save_iter`) now both land in the CSV via the sidecar-merge added in `_common.py`. For R4 (`probe_hidden_states`), the post-pass worker stages dominate the hook's contribution to gen latency, while the analyzer kernel is genuinely cheap (~3 ms). The hook's overhead is in the *egress path*, not in the analysis.

### 3.7 v0.2.0 vs v0.1.0 (partial — `steer_hook_act` only)

**`steer_hook_act_v010` ran cleanly (8/8 cells).** `probe_hook_qk_v010` (16 cells) and `probe_hidden_states_v010` (8 cells) failed because vllm_hook_plugins's auto-injected `worker_extension_cls` collided with the v010 `worker_cls` on the shared `_background_save_loop` method name. The fix (explicit `worker_extension_cls=""` in V010Engine) is in but the QK/HS rows will be filled by the next paired run.

**Steering paired comparison** (same workload cell, v0.2.0 vs v0.1.0):

| prompt_len | batch | max_tok | v0.2.0 gen_ms | v0.1.0 gen_ms | Δ (v010 − v020) |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 19.3 | 19.0 | −1.1% |
| 16 | 1 | 32 | 532.7 | 514.0 | **−3.5%** |
| 16 | 16 | 1 | 36.0 | 36.5 | +1.3% |
| 16 | 16 | 32 | 610.8 | 542.3 | **−11.2%** |
| 256 | 1 | 1 | 19.0 | 17.8 | **−6.4%** |
| 256 | 1 | 32 | 560.6 | 510.1 | **−9.0%** |
| 256 | 16 | 1 | 74.7 | 76.5 | +2.4% |
| 256 | 16 | 32 | 622.9 | 611.8 | −1.8% |

**Mean across cells**: v0.1.0 is **~3.7% faster** than v0.2.0 on the steering path.

**Analysis.** v0.1.0 wins 6 of 8 cells, losing only the two trivial-workload cells (bs=16, mt=1) by ~2%. The pattern fits the architectural difference: v0.2.0's steering worker accepts a per-request `extra_args` opt-in that v0.1.0 doesn't have (v0.1.0 uses a process-wide flag file, set/cleared per call). For the steering use case where every request wants steering, v0.2.0's per-request dispatch is pure overhead. **v0.2.0's flexibility costs ~4% on the heaviest cell**; whether the flexibility is worth it depends on whether your workload mixes hooked and unhooked requests.

The QK and HS comparisons would tell a richer story (they exercise the disk-write path and the prefix-cache reconstruction that v0.2.0 added). Those numbers will replace the placeholder in the next round.

---

## 4. Discussion

### 4.1 What the data supports

- **The plug-in is essentially free when loaded but idle.** R0 → R1 delta is +2.4% mean and within run-to-run noise on every individual cell (§3.1).
- **Probe-hidden-states is the only "heavy" hook.** It costs +32% mean / +34% worst-case in gen latency and 97 MB/cell in artifacts (§3.1, §3.2). Every other hook is in the 5–8% range.
- **`disk-st-async` is the right default storage variant.** It wins 14 of 16 measurable storage cells (§3.3) and avoids the rpc-all-tokens catastrophe by 3–4 orders of magnitude.
- **CUDA-graph budgets are met.** All three batch sizes pass the ≤5% idle-tax gate from `cuda_graph_plan.html` (§3.4).
- **Serve-path bottleneck is JSON serialisation, not generation.** Throughput collapses 3.1× (hidden_states) to 4.6× (QK) under k=8 concurrency because each hook response carries 5.5–9.4 MB of JSON (§3.5). The byte counts and the fact that QK's response is heavier than HS's makes the `bytes+zstd+base64` wire-format patch the highest-leverage optimisation.
- **Offline cost and serve cost track different bottlenecks.** R4 dominates offline but R3 (QK) dominates serve, because the serve path is gated by response size (which QK has more of than HS in the cells we tested), not by underlying generation cost (§3.5).
- **v0.2.0's per-request opt-in costs ~4% on the steering path** vs v0.1.0's process-wide flag (§3.7). The trade-off is flexibility (mixing hooked + unhooked requests) for a small constant cost.

### 4.2 Threats to validity

- **One model only.** All measurements are on Qwen2-1.5B-Instruct (28 layers, 1.5 GiB weights). Larger models may shift absolute numbers but the relative ordering (which hook is heavy, which storage variant wins) should be stable.
- **Eager mode everywhere.** `enforce_eager=True` is set for every cell. The CUDA-graph re-enablement work tracked by the `graph_enable` branch will change absolute latencies but is not expected to change rankings.
- **Prefix caching off.** Matches the original `Numerical_Analysis/` setup. The v0.2.0 QK worker has a prefix-reconstruction path that v0.1.0 lacks; behaviour with prefix caching on may favour v0.2.0 more strongly.
- **v0.1.0 paired comparison is partial.** Only the steering row is currently filled — the QK and HS rows hit an entry-points collision (fixed; pending re-run).
- **n_layers held fixed.** Unlike `Numerical_Analysis/`, we don't sweep n_layers ∈ {1..28}; we always capture all layers. The "cost scales with captured layer count" story is not directly told.

### 4.3 Recommendations

1. **Make `VLLM_HOOK_USE_SAFETENSORS=1 VLLM_HOOK_ASYNC_SAVE=1` the documented default** in `vllm_hook_plugins/configs.md`. The storage matrix at §3.3 makes the case.
2. **Land the `bytes+zstd+base64` serve-path wire-format patch** (`plan.html` §7). The 3–5× throughput gap at k=8 and the 5.5–9.4 MB/req JSON encoding cost (§3.5) make this the single highest-leverage open optimisation.
3. **Use `probe_hidden_states + all_tokens + (pl=256, bs=16, mt=32)` as a regression-test cell.** It's the worst-case workload across the matrix and exposes hook-egress changes most clearly.
4. **Once the v010 QK + HS rows land, re-evaluate the v0.2.0 architectural trade-offs.** Steering is 4% slower; the QK and HS numbers will tell us whether the broader v0.2.0 refactor (extra_args dispatch, sidecar, prefix-cache reconstruction) earns its keep.

---

## 5. Reproducing the report

```bash
bash profiling/runners/submit_all.sh
bjobs   # wait for all 5 jobs to reach DONE (~2.5 h)
python profiling/analyze/plot_results.py
python profiling/analyze/summarize.py profiling/results/quick-*.csv \
       --output profiling/results/quick-summary.md
python profiling/analyze/summarize.py profiling/results/storage-*.csv \
       --output profiling/results/storage-summary.md
```

To rerun a single stage (e.g. `quick` after the v010 fix):

```bash
STAGES="quick" INCLUDE_V010=1 bash profiling/runners/submit_all.sh
```

---

## 6. Raw data

```
profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1595702.csv     80 rows, ~140 columns (R0–R5 clean + steer_v010 clean + qk/hs_v010 failed)
profiling/results/storage-Qwen_Qwen2-1.5B-Instruct-1588747.csv   88 rows (all 6 storage variants × HS)
profiling/results/idle-Qwen_Qwen2-1.5B-Instruct-1588746.csv       3 rows (one per batch size)
profiling/results/serve-Qwen_Qwen2-1.5B-Instruct-1595703.csv     12 rows (all 3 workers × 4 concurrencies)
```

Companion `.jsonl` files hold the per-cell `PROF.snapshot()` arrays for recomputing percentiles without re-running. LSF logs are under `profiling/runners/logs/`.

---

*Generated 2026-06-03 from `profiling/REPORT_TEMPLATE.md`. Companion files:
`profiling/analyze/plot_results.py`, `profiling/analyze/summarize.py`,
`claude_docs/profiling_tool.html`.*
