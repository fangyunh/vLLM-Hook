# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1575395.csv`

_48 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 16.6 | 965.7 | 60.4 | 41,876 | 0.000 |
| R1 | plugin_idle | 16.9 | 945.8 | 59.1 | 41,876 | 0.000 |
| R2 | probe_hook_qk:last_token | 23.2 | 701.2 | 43.8 | 41,876 | 13.6 |
| R3 | probe_hook_qk:all_tokens | 19.0 | 848.3 | 53.0 | 41,876 | 58.6 |
| R4 | probe_hidden_states | 21.1 | 757.2 | 47.3 | 41,876 | 1,351 |
| R5 | steer_hook_act | 16.5 | 971.3 | 60.7 | 41,876 | 0.000 |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 16.6 | 16.9 | 2.105% |
| 16.0 | 1.000 | 32.0 | 465.9 | 479.2 | 2.872% |
| 16.0 | 16.0 | 1.000 | 31.5 | 33.2 | 5.384% |
| 16.0 | 16.0 | 32.0 | 524.3 | 549.4 | 4.794% |
| 256.0 | 1.000 | 1.000 | 16.3 | 16.3 | -0.073% |
| 256.0 | 1.000 | 32.0 | 474.9 | 468.7 | -1.295% |
| 256.0 | 16.0 | 1.000 | 72.8 | 71.6 | -1.642% |
| 256.0 | 16.0 | 32.0 | 544.5 | 544.5 | -0.001% |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.analyze | 505.3 |
| analyzer.kernel | 504.6 |
| hookllm.generate | 297.4 |
| rpc.flush_disk | 12.2 |
| io.artifact_load.safetensors | 0.979 |
| hookllm.build_extra | 0.007 |
