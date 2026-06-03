# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1598923.csv`

_80 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 24.0 | 671.3 | 42.0 | 41,876 | 0.000 |
| R1 | plugin_idle | 24.8 | 654.2 | 40.9 | 41,876 | 0.000 |
| R2 | probe_hook_qk:last_token | 36.8 | 442.1 | 27.6 | 41,876 | 14.4 |
| R3 | probe_hook_qk:all_tokens | 33.7 | 476.6 | 29.8 | 41,876 | 59.4 |
| R4 | probe_hidden_states | 35.5 | 462.7 | 28.9 | 41,876 | 1,352 |
| R5 | steer_hook_act | 22.6 | 717.5 | 44.8 | 41,876 | 0.000 |

## Memory footprint per row (worst-case cell)

_All values are MB. `worker cuda_alloc_mb` is the high-water mark of `torch.cuda.memory_allocated()` sampled every 50 ms inside the EngineCore process — the cleanest signal of what the hook itself costs in GPU memory._

| row | worker cuda_alloc_mb | worker cuda_reserved_mb | worker host_rss_mb | driver host_rss_mb | nvml peak (mb) |
|---|---|---|---|---|---|
| baseline | — | — | 1,076 | 1,076 | 41,876 |
| plugin_idle | — | — | 1,260 | 1,260 | 41,876 |
| probe_hook_qk:last_token | 40,182 | 40,568 | 4,070 | 1,275 | 41,876 |
| probe_hook_qk:all_tokens | 40,311 | 40,570 | 4,118 | 1,279 | 41,878 |
| probe_hidden_states | 40,288 | 41,012 | 4,004 | 1,287 | 42,320 |
| steer_hook_act | — | — | 1,287 | 1,287 | 41,876 |

## v0.2.0 vs v0.1.0 (paired cells)

_Negative `v0.1.0 − v0.2.0` means v0.1.0 was faster on that cell; positive means the current version wins. v0.1.0 has no prefix-cache prefix-recon path, so its artifact size may differ when prefix caching is on._

### steer_hook_act

| prompt_len | batch | max_tok | mode | v0.2.0 gen_ms | v0.1.0 gen_ms | v0.1.0 − v0.2.0 | v0.2.0 art_kb | v0.1.0 art_kb |
|---|---|---|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | last_token | 22.6 | 19.0 | -15.7% | 0.000 | 0.000 |
| 16.0 | 1.000 | 32.0 | last_token | 524.4 | 532.9 | 1.618% | 0.000 | 0.000 |
| 16.0 | 16.0 | 1.000 | last_token | 43.5 | 40.1 | -7.813% | 0.000 | 0.000 |
| 16.0 | 16.0 | 32.0 | last_token | 572.3 | 548.0 | -4.250% | 0.000 | 0.000 |
| 256.0 | 1.000 | 1.000 | last_token | 22.7 | 27.9 | 22.8% | 0.000 | 0.000 |
| 256.0 | 1.000 | 32.0 | last_token | 511.1 | 499.3 | -2.303% | 0.000 | 0.000 |
| 256.0 | 16.0 | 1.000 | last_token | 83.2 | 82.3 | -1.010% | 0.000 | 0.000 |
| 256.0 | 16.0 | 32.0 | last_token | 655.8 | 643.2 | -1.933% | 0.000 | 0.000 |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 24.0 | 24.8 | 3.545% |
| 16.0 | 1.000 | 32.0 | 505.8 | 479.2 | -5.257% |
| 16.0 | 16.0 | 1.000 | 40.3 | 48.5 | 20.5% |
| 16.0 | 16.0 | 32.0 | 583.3 | 602.2 | 3.233% |
| 256.0 | 1.000 | 1.000 | 29.3 | 25.2 | -14.0% |
| 256.0 | 1.000 | 32.0 | 504.2 | 518.5 | 2.831% |
| 256.0 | 16.0 | 1.000 | 81.4 | 82.0 | 0.740% |
| 256.0 | 16.0 | 32.0 | 669.1 | 606.5 | -9.356% |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.analyze | 3,747 |
| analyzer.kernel | 3,746 |
| hookllm.generate | 339.2 |
| async.save_iter | 288.0 |
| worker.cpu_transfer.hs | 42.6 |
| worker.disk_write.safetensors | 23.5 |
| rpc.flush_disk | 15.8 |
| worker.cpu_transfer.qk | 1.221 |
| io.artifact_load.safetensors | 1.122 |
| worker.queue_put | 0.014 |
| hookllm.build_extra | 0.008 |
