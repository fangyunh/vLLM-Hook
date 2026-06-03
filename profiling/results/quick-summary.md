# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1595702.csv`

_80 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 17.4 | 919.7 | 57.5 | 41,876 | 0.000 |
| R1 | plugin_idle | 19.2 | 834.8 | 52.2 | 41,876 | 0.000 |
| R2 | probe_hook_qk:last_token | 23.9 | 677.6 | 42.3 | 41,876 | 14.4 |
| R3 | probe_hook_qk:all_tokens | 19.9 | 804.4 | 50.3 | 41,876 | 59.4 |
| R4 | probe_hidden_states | 26.5 | 604.3 | 37.8 | 41,876 | 1,352 |
| R5 | steer_hook_act | 19.3 | 831.4 | 52.0 | 41,876 | 0.000 |

## Memory footprint per row (worst-case cell)

_All values are MB. `worker cuda_alloc_mb` is the high-water mark of `torch.cuda.memory_allocated()` sampled every 50 ms inside the EngineCore process — the cleanest signal of what the hook itself costs in GPU memory._

| row | worker cuda_alloc_mb | worker cuda_reserved_mb | worker host_rss_mb | driver host_rss_mb | nvml peak (mb) |
|---|---|---|---|---|---|
| baseline | — | — | 1,112 | 1,112 | 41,876 |
| plugin_idle | — | — | 1,291 | 1,291 | 41,876 |
| probe_hook_qk:last_token | 40,163 | 40,568 | 4,112 | 1,297 | 41,876 |
| probe_hook_qk:all_tokens | 40,012 | 40,584 | 3,968 | 1,295 | 41,892 |
| probe_hidden_states | 40,289 | 41,012 | 4,048 | 1,296 | 42,320 |
| steer_hook_act | — | — | 1,299 | 1,299 | 41,876 |

## v0.2.0 vs v0.1.0 (paired cells)

_Negative `v0.1.0 − v0.2.0` means v0.1.0 was faster on that cell; positive means the current version wins. v0.1.0 has no prefix-cache prefix-recon path, so its artifact size may differ when prefix caching is on._

### steer_hook_act

| prompt_len | batch | max_tok | mode | v0.2.0 gen_ms | v0.1.0 gen_ms | v0.1.0 − v0.2.0 | v0.2.0 art_kb | v0.1.0 art_kb |
|---|---|---|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | last_token | 19.3 | 19.0 | -1.116% | 0.000 | 0.000 |
| 16.0 | 1.000 | 32.0 | last_token | 532.7 | 514.0 | -3.504% | 0.000 | 0.000 |
| 16.0 | 16.0 | 1.000 | last_token | 36.0 | 36.5 | 1.337% | 0.000 | 0.000 |
| 16.0 | 16.0 | 32.0 | last_token | 610.8 | 542.3 | -11.2% | 0.000 | 0.000 |
| 256.0 | 1.000 | 1.000 | last_token | 19.0 | 17.8 | -6.376% | 0.000 | 0.000 |
| 256.0 | 1.000 | 32.0 | last_token | 560.6 | 510.1 | -9.001% | 0.000 | 0.000 |
| 256.0 | 16.0 | 1.000 | last_token | 74.7 | 76.5 | 2.432% | 0.000 | 0.000 |
| 256.0 | 16.0 | 32.0 | last_token | 622.9 | 611.8 | -1.787% | 0.000 | 0.000 |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 17.4 | 19.2 | 10.2% |
| 16.0 | 1.000 | 32.0 | 509.7 | 534.7 | 4.901% |
| 16.0 | 16.0 | 1.000 | 37.9 | 36.7 | -3.143% |
| 16.0 | 16.0 | 32.0 | 551.7 | 586.5 | 6.301% |
| 256.0 | 1.000 | 1.000 | 19.5 | 18.6 | -4.583% |
| 256.0 | 1.000 | 32.0 | 516.0 | 523.5 | 1.453% |
| 256.0 | 16.0 | 1.000 | 77.1 | 79.4 | 2.955% |
| 256.0 | 16.0 | 32.0 | 625.1 | 612.0 | -2.090% |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.analyze | 1,596 |
| analyzer.kernel | 1,595 |
| hookllm.generate | 332.6 |
| worker.cpu_transfer.hs | 60.3 |
| async.save_iter | 58.5 |
| worker.disk_write.safetensors | 21.7 |
| rpc.flush_disk | 18.8 |
| worker.cpu_transfer.qk | 1.545 |
| io.artifact_load.safetensors | 1.477 |
| worker.queue_put | 0.016 |
| hookllm.build_extra | 0.009 |
