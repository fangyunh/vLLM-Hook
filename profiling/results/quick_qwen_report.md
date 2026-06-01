# Profile report — `profiling/results/quick-Qwen_Qwen2-1.5B-Instruct-1552095.csv`

_48 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | 16.2 | 989.0 | 61.8 | 41,876 | 0.000 |
| R1 | plugin_idle | 16.4 | 976.3 | 61.0 | 41,876 | 0.000 |
| R2 | probe_hook_qk:last_token | — | — | — | — | — |
| R3 | probe_hook_qk:all_tokens | — | — | — | — | — |
| R4 | probe_hidden_states | 0.000 | — | — | — | — |
| R5 | steer_hook_act | 0.000 | — | — | — | — |

## Idle plugin tax (baseline vs plugin_idle)

| prompt_len | batch | max_tok | base_gen_ms | plugin_gen_ms | delta_pct |
|---|---|---|---|---|---|
| 16.0 | 1.000 | 1.000 | 16.2 | 16.4 | 1.271% |
| 16.0 | 1.000 | 32.0 | 480.3 | 471.8 | -1.756% |
| 16.0 | 16.0 | 1.000 | 31.4 | 32.1 | 2.236% |
| 16.0 | 16.0 | 32.0 | 527.0 | 550.7 | 4.488% |
| 256.0 | 1.000 | 1.000 | 16.2 | 21.9 | 35.5% |
| 256.0 | 1.000 | 32.0 | 480.1 | 488.3 | 1.710% |
| 256.0 | 16.0 | 1.000 | 72.8 | 73.8 | 1.320% |
| 256.0 | 16.0 | 32.0 | 549.0 | 568.4 | 3.546% |
