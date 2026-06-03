# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1577716.csv`

_48 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 16.6 | 962.1 | 60.1 | 41,876 | 0.000 |
| R1 | plugin_idle | 16.1 | 992.6 | 62.0 | 41,876 | 0.000 |
| R2 | probe_hook_qk:last_token | 20.1 | 797.1 | 49.8 | 41,876 | 14.4 |
| R3 | probe_hook_qk:all_tokens | 19.5 | 820.0 | 51.3 | 41,876 | 59.4 |
| R4 | probe_hidden_states | 21.8 | 735.2 | 45.9 | 41,876 | 1,352 |
| R5 | steer_hook_act | 16.8 | 952.4 | 59.5 | 41,876 | 0.000 |

## Memory footprint per row (worst-case cell)

_All values are MB. `cuda_alloc_delta_mb` is the per-rep spike above the pre-rep working set — the cleanest signal of what the hook itself costs in GPU memory._

| row | cuda_alloc_mb | cuda_peak_alloc_mb | cuda_alloc_delta_mb | cuda_peak_reserved_mb | host_rss_mb | nvml_used_mb |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 | 0.000 | 1,129 | 41,876 |
| plugin_idle | 0.000 | 0.000 | 0.000 | 0.000 | 1,308 | 41,876 |
| probe_hook_qk:last_token | 0.000 | 0.000 | 0.000 | 0.000 | 1,310 | 41,876 |
| probe_hook_qk:all_tokens | 0.000 | 0.000 | 0.000 | 0.000 | 1,315 | 41,876 |
| probe_hidden_states | 0.000 | 0.000 | 0.000 | 0.000 | 1,303 | 41,876 |
| steer_hook_act | 0.000 | 0.000 | 0.000 | 0.000 | 1,319 | 41,876 |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 16.6 | 16.1 | -3.050% |
| 16.0 | 1.000 | 32.0 | 486.5 | 499.2 | 2.602% |
| 16.0 | 16.0 | 1.000 | 33.4 | 34.4 | 2.892% |
| 16.0 | 16.0 | 32.0 | 541.6 | 592.9 | 9.468% |
| 256.0 | 1.000 | 1.000 | 16.8 | 16.9 | 0.563% |
| 256.0 | 1.000 | 32.0 | 491.5 | 484.0 | -1.519% |
| 256.0 | 16.0 | 1.000 | 74.8 | 75.2 | 0.521% |
| 256.0 | 16.0 | 32.0 | 575.8 | 569.2 | -1.160% |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.analyze | 758.4 |
| analyzer.kernel | 757.3 |
| hookllm.generate | 314.0 |
| rpc.flush_disk | 10.2 |
| io.artifact_load.safetensors | 1.469 |
| hookllm.build_extra | 0.009 |
