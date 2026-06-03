# Profile report — `profiling/results/storage-Qwen_Qwen2-1.5B-Instruct-1577718.csv`

_88 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | — | — | — | — | — |
| R1 | plugin_idle | — | — | — | — | — |
| R2 | probe_hook_qk:last_token | — | — | — | — | — |
| R3 | probe_hook_qk:all_tokens | — | — | — | — | — |
| R4 | probe_hidden_states | 17.1 | 933.9 | 58.4 | 33,701 | 0.000 |
| R5 | steer_hook_act | — | — | — | — | — |

## Memory footprint per row (worst-case cell)

_All values are MB. `cuda_alloc_delta_mb` is the per-rep spike above the pre-rep working set — the cleanest signal of what the hook itself costs in GPU memory._

| row | cuda_alloc_mb | cuda_peak_alloc_mb | cuda_alloc_delta_mb | cuda_peak_reserved_mb | host_rss_mb | nvml_used_mb |
|---|---|---|---|---|---|---|
| probe_hidden_states | 0.000 | 0.000 | 0.000 | 0.000 | 1,077 | 33,701 |

## Storage variant matrix (HS worker)

| variant | mode | prompt_len | batch | gen_lat_ms | wait_ms | analyze_ms | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|---|---|
| disk-pt | all_tokens | 16.0 | 1.000 | 16.6 | 0.004 | 2.463 | 33,701 | 1,353 |
| disk-pt | all_tokens | 16.0 | 8.000 | 50.0 | 0.008 | 14.1 | 33,711 | 10,812 |
| disk-pt | all_tokens | 64.0 | 1.000 | 19.4 | 0.004 | 2.815 | 33,707 | 5,385 |
| disk-pt | all_tokens | 64.0 | 8.000 | 73.5 | 0.011 | 20.0 | 33,745 | 43,069 |
| disk-pt | all_tokens | 256.0 | 1.000 | 32.2 | 0.008 | 7.095 | 33,731 | 21,513 |
| disk-pt | all_tokens | 256.0 | 8.000 | 181.5 | 0.007 | 88.0 | 33,923 | 172,093 |
| disk-pt | all_tokens | 512.0 | 1.000 | 47.5 | 0.008 | 9.333 | 33,701 | 43,017 |
| disk-pt | all_tokens | 512.0 | 8.000 | 327.7 | 0.008 | 169.6 | 33,701 | 344,125 |
| disk-pt | last_token | 16.0 | 1.000 | 15.6 | 0.004 | 2.061 | 33,701 | 92.7 |
| disk-pt | last_token | 16.0 | 8.000 | 44.1 | 0.007 | 11.1 | 33,701 | 731.1 |
| disk-pt | last_token | 64.0 | 1.000 | 15.8 | 0.004 | 2.076 | 33,701 | 92.7 |
| disk-pt | last_token | 64.0 | 8.000 | 43.1 | 0.006 | 11.6 | 33,701 | 731.1 |
| disk-pt | last_token | 256.0 | 1.000 | 16.3 | 0.003 | 2.037 | 33,705 | 92.7 |
| disk-pt | last_token | 256.0 | 8.000 | 48.6 | 0.006 | 11.9 | 33,705 | 731.1 |
| disk-pt | last_token | 512.0 | 1.000 | 17.4 | 0.004 | 2.045 | 33,701 | 92.7 |
| disk-pt | last_token | 512.0 | 8.000 | 57.7 | 0.007 | 11.0 | 33,701 | 731.1 |
| disk-pt-async | all_tokens | 16.0 | 1.000 | 15.1 | 0.138 | 4.652 | 33,701 | 1,353 |
| disk-pt-async | all_tokens | 16.0 | 8.000 | 39.8 | 0.173 | 27.7 | 33,711 | 10,812 |
| disk-pt-async | all_tokens | 64.0 | 1.000 | 15.6 | 0.137 | 7.943 | 33,707 | 5,385 |
| disk-pt-async | all_tokens | 64.0 | 8.000 | 44.5 | 0.177 | 52.1 | 33,745 | 43,069 |
| disk-pt-async | all_tokens | 256.0 | 1.000 | 19.3 | 0.148 | 20.3 | 33,731 | 21,513 |
| disk-pt-async | all_tokens | 256.0 | 8.000 | 64.6 | 0.214 | 145.5 | 33,923 | 172,093 |
| disk-pt-async | all_tokens | 512.0 | 1.000 | 22.1 | 0.147 | 35.4 | 33,701 | 43,017 |
| disk-pt-async | all_tokens | 512.0 | 8.000 | 89.2 | 0.197 | 377.0 | 33,701 | 344,125 |
| disk-pt-async | last_token | 16.0 | 1.000 | 14.6 | 0.151 | 3.149 | 33,701 | 92.7 |
| disk-pt-async | last_token | 16.0 | 8.000 | 36.7 | 0.178 | 18.3 | 33,701 | 731.1 |
| disk-pt-async | last_token | 64.0 | 1.000 | 15.0 | 0.133 | 3.169 | 33,701 | 92.7 |
| disk-pt-async | last_token | 64.0 | 8.000 | 37.4 | 0.169 | 18.6 | 33,701 | 731.1 |
| disk-pt-async | last_token | 256.0 | 1.000 | 15.3 | 0.136 | 3.130 | 33,705 | 92.7 |
| disk-pt-async | last_token | 256.0 | 8.000 | 43.0 | 0.167 | 17.7 | 33,705 | 731.1 |
| disk-pt-async | last_token | 512.0 | 1.000 | 16.5 | 0.134 | 3.162 | 33,701 | 92.7 |
| disk-pt-async | last_token | 512.0 | 8.000 | 51.9 | 0.177 | 17.7 | 33,701 | 731.1 |
| disk-st | all_tokens | 16.0 | 1.000 | 17.1 | 0.005 | 1.058 | 33,701 | 1,351 |
| disk-st | all_tokens | 16.0 | 8.000 | 353.6 | 0.018 | 2.065 | 33,711 | 10,759 |
| disk-st | all_tokens | 64.0 | 1.000 | 633.2 | 0.021 | 1.925 | 33,707 | 5,383 |
| disk-st | all_tokens | 64.0 | 8.000 | 2,882 | 0.016 | 1.901 | 33,745 | 43,015 |
| disk-st | all_tokens | 256.0 | 1.000 | 648.1 | 0.015 | 1.521 | 33,731 | 21,511 |
| disk-st | all_tokens | 256.0 | 8.000 | 3,002 | 0.016 | 1.981 | 33,923 | 172,040 |
| disk-st | all_tokens | 512.0 | 1.000 | 670.5 | 0.019 | 1.703 | 33,701 | 43,016 |
| disk-st | all_tokens | 512.0 | 8.000 | 3,154 | 0.016 | 1.988 | 33,701 | 344,072 |
| disk-st | last_token | 16.0 | 1.000 | 16.4 | 0.005 | 1.187 | 33,701 | 91.2 |
| disk-st | last_token | 16.0 | 8.000 | 37.5 | 0.007 | 1.459 | 33,701 | 679.2 |
| disk-st | last_token | 64.0 | 1.000 | 16.4 | 0.005 | 0.898 | 33,701 | 91.2 |
| disk-st | last_token | 64.0 | 8.000 | 39.4 | 0.008 | 1.439 | 33,701 | 679.2 |
| disk-st | last_token | 256.0 | 1.000 | 16.9 | 0.005 | 0.968 | 33,705 | 91.2 |
| disk-st | last_token | 256.0 | 8.000 | 44.7 | 0.008 | 1.549 | 33,705 | 679.2 |
| disk-st | last_token | 512.0 | 1.000 | 18.1 | 0.007 | 0.933 | 33,701 | 91.2 |
| disk-st | last_token | 512.0 | 8.000 | 53.2 | 0.007 | 1.391 | 33,701 | 679.2 |
| disk-st-async | all_tokens | 16.0 | 1.000 | 15.1 | 10.4 | 1.026 | 33,701 | 1,352 |
| disk-st-async | all_tokens | 16.0 | 8.000 | 38.2 | 20.8 | 1.660 | 33,711 | 10,760 |
| disk-st-async | all_tokens | 64.0 | 1.000 | 16.4 | 26.9 | 1.243 | 33,707 | 5,384 |
| disk-st-async | all_tokens | 64.0 | 8.000 | 42.6 | 123.2 | 1.821 | 33,745 | 43,016 |
| disk-st-async | all_tokens | 256.0 | 1.000 | 18.5 | 39.1 | 1.381 | 33,731 | 21,512 |
| disk-st-async | all_tokens | 256.0 | 8.000 | 61.9 | 176.1 | 1.787 | 33,923 | 172,040 |
| disk-st-async | all_tokens | 512.0 | 1.000 | 23.7 | 61.5 | 1.509 | 33,701 | 43,016 |
| disk-st-async | all_tokens | 512.0 | 8.000 | 80.4 | 264.2 | 2.233 | 33,701 | 344,072 |
| disk-st-async | last_token | 16.0 | 1.000 | 15.0 | 10.4 | 1.150 | 33,701 | 92.0 |
| disk-st-async | last_token | 16.0 | 8.000 | 36.5 | 10.5 | 1.531 | 33,701 | 680.0 |
| disk-st-async | last_token | 64.0 | 1.000 | 15.1 | 10.4 | 0.916 | 33,701 | 92.0 |
| disk-st-async | last_token | 64.0 | 8.000 | 37.3 | 10.5 | 1.310 | 33,701 | 680.1 |
| disk-st-async | last_token | 256.0 | 1.000 | 15.9 | 10.4 | 1.027 | 33,705 | 92.0 |
| disk-st-async | last_token | 256.0 | 8.000 | 42.7 | 10.5 | 1.324 | 33,705 | 680.1 |
| disk-st-async | last_token | 512.0 | 1.000 | 16.8 | 10.5 | 0.900 | 33,701 | 92.0 |
| disk-st-async | last_token | 512.0 | 8.000 | 51.3 | 10.5 | 1.348 | 33,701 | 680.1 |
| rpc | all_tokens | 16.0 | 1.000 | 21.5 | 0.001 | 0.013 | 33,701 | 0.000 |
| rpc | all_tokens | 16.0 | 8.000 | 93.2 | 0.002 | 0.015 | 33,711 | 0.000 |
| rpc | all_tokens | 64.0 | 1.000 | 650.0 | 0.002 | 0.020 | 33,707 | 0.000 |
| rpc | all_tokens | 64.0 | 8.000 | 5,084 | 0.002 | 0.016 | 33,745 | 0.000 |
| rpc | all_tokens | 256.0 | 1.000 | 698.4 | 0.002 | 0.022 | 33,731 | 0.000 |
| rpc | all_tokens | 256.0 | 8.000 | 5,489 | 0.003 | 0.019 | 33,923 | 0.000 |
| rpc | all_tokens | 512.0 | 1.000 | 756.5 | 0.003 | 0.025 | 33,701 | 0.000 |
| rpc | all_tokens | 512.0 | 8.000 | 6,100 | 0.003 | 0.020 | 33,701 | 0.000 |
| rpc | last_token | 16.0 | 1.000 | 17.1 | 0.002 | 0.015 | 33,701 | 0.000 |
| rpc | last_token | 16.0 | 8.000 | 59.0 | 0.002 | 0.013 | 33,701 | 0.000 |
| rpc | last_token | 64.0 | 1.000 | 17.5 | 0.001 | 0.014 | 33,701 | 0.000 |
| rpc | last_token | 64.0 | 8.000 | 60.5 | 0.002 | 0.013 | 33,701 | 0.000 |
| rpc | last_token | 256.0 | 1.000 | 17.4 | 0.001 | 0.013 | 33,705 | 0.000 |
| rpc | last_token | 256.0 | 8.000 | 64.9 | 0.002 | 0.013 | 33,705 | 0.000 |
| rpc | last_token | 512.0 | 1.000 | 19.0 | 0.001 | 0.013 | 33,701 | 0.000 |
| rpc | last_token | 512.0 | 8.000 | 74.4 | 0.002 | 0.013 | 33,701 | 0.000 |
| shm | last_token | 16.0 | 1.000 | 17.0 | 0.002 | 0.013 | 33,701 | 0.000 |
| shm | last_token | 16.0 | 8.000 | 60.3 | 0.003 | 0.014 | 33,701 | 0.000 |
| shm | last_token | 64.0 | 1.000 | 17.7 | 0.002 | 0.013 | 33,701 | 0.000 |
| shm | last_token | 64.0 | 8.000 | 60.3 | 0.002 | 0.013 | 33,701 | 0.000 |
| shm | last_token | 256.0 | 1.000 | 17.8 | 0.002 | 0.013 | 33,705 | 0.000 |
| shm | last_token | 256.0 | 8.000 | 65.7 | 0.003 | 0.013 | 33,705 | 0.000 |
| shm | last_token | 512.0 | 1.000 | 19.7 | 0.002 | 0.013 | 33,701 | 0.000 |
| shm | last_token | 512.0 | 8.000 | 75.1 | 0.002 | 0.014 | 33,701 | 0.000 |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.generate | 377.6 |
| rpc.flush_disk | 185.7 |
| rpc.get_states | 167.0 |
| io.artifact_load.pt | 21.4 |
| hookllm.analyze | 13.3 |
| rpc.decompress | 7.526 |
| io.artifact_load.safetensors | 1.209 |
| hookllm.merge_probes | 0.027 |
| hookllm.build_extra | 0.005 |
| analyzer.kernel | 0.004 |
