# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1588745.csv`

_80 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 15.5 | 1,033 | 64.5 | 21,243 | 0.000 |
| R1 | plugin_idle | 14.9 | 1,072 | 67.0 | 21,243 | 0.000 |
| R2 | probe_hook_qk:last_token | 18.1 | 890.0 | 55.6 | 21,243 | 14.4 |
| R3 | probe_hook_qk:all_tokens | 17.4 | 917.7 | 57.4 | 21,243 | 59.4 |
| R4 | probe_hidden_states | 20.9 | 765.6 | 47.8 | 21,243 | 1,352 |
| R5 | steer_hook_act | 15.6 | 1,029 | 64.3 | 21,243 | 0.000 |

## Memory footprint per row (worst-case cell)

_All values are MB. `worker cuda_alloc_mb` is the high-water mark of `torch.cuda.memory_allocated()` sampled every 50 ms inside the EngineCore process — the cleanest signal of what the hook itself costs in GPU memory._

| row | worker cuda_alloc_mb | worker cuda_reserved_mb | worker host_rss_mb | driver host_rss_mb | nvml peak (mb) |
|---|---|---|---|---|---|
| baseline | — | — | 1,111 | 1,111 | 21,243 |
| plugin_idle | — | — | 1,296 | 1,296 | 21,243 |
| probe_hook_qk:last_token | 20,131 | 20,184 | 3,882 | 1,311 | 21,243 |
| probe_hook_qk:all_tokens | 19,834 | 20,184 | 3,851 | 1,313 | 21,243 |
| probe_hidden_states | 19,946 | 20,628 | 3,804 | 1,294 | 21,687 |
| steer_hook_act | — | — | 1,287 | 1,287 | 21,243 |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 15.5 | 14.9 | -3.677% |
| 16.0 | 1.000 | 32.0 | 447.5 | 444.4 | -0.687% |
| 16.0 | 16.0 | 1.000 | 31.0 | 30.4 | -1.936% |
| 16.0 | 16.0 | 32.0 | 489.7 | 502.8 | 2.684% |
| 256.0 | 1.000 | 1.000 | 15.6 | 15.3 | -1.871% |
| 256.0 | 1.000 | 32.0 | 445.7 | 451.2 | 1.238% |
| 256.0 | 16.0 | 1.000 | 73.4 | 75.4 | 2.691% |
| 256.0 | 16.0 | 32.0 | 528.2 | 521.7 | -1.219% |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.generate | 283.0 |
| worker.cpu_transfer.hs | 46.8 |
| async.save_iter | 41.0 |
| worker.disk_write.safetensors | 29.7 |
| rpc.flush_disk | 13.4 |
| hookllm.analyze | 3.708 |
| analyzer.kernel | 2.846 |
| worker.cpu_transfer.qk | 1.334 |
| io.artifact_load.safetensors | 1.090 |
| worker.queue_put | 0.011 |
| hookllm.build_extra | 0.006 |
