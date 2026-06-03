# Profile report — `profiling/results/storage-Qwen_Qwen2-1.5B-Instruct-1588747.csv`

_88 rows_

## R0–R5 six-row summary (plan.html §7)

| # | row | gen_lat_ms | prefill_tok/s | decode_tok/s | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|
| R0 | baseline | — | — | — | — | — |
| R1 | plugin_idle | — | — | — | — | — |
| R2 | probe_hook_qk:last_token | — | — | — | — | — |
| R3 | probe_hook_qk:all_tokens | — | — | — | — | — |
| R4 | probe_hidden_states | 45.3 | 358.0 | 22.4 | 17,211 | 0.000 |
| R5 | steer_hook_act | — | — | — | — | — |

## Memory footprint per row (worst-case cell)

_All values are MB. `worker cuda_alloc_mb` is the high-water mark of `torch.cuda.memory_allocated()` sampled every 50 ms inside the EngineCore process — the cleanest signal of what the hook itself costs in GPU memory._

| row | worker cuda_alloc_mb | worker cuda_reserved_mb | worker host_rss_mb | driver host_rss_mb | nvml peak (mb) |
|---|---|---|---|---|---|
| probe_hidden_states | 16,060 | 16,152 | 4,039 | 1,469 | 17,211 |

## Storage variant matrix (HS worker)

| variant | mode | prompt_len | batch | gen_lat_ms | wait_ms | analyze_ms | peak_gpu_mb | artifact_kb |
|---|---|---|---|---|---|---|---|---|
| disk-pt | all_tokens | 16.0 | 1.000 | 52.5 | 0.009 | 7.642 | 17,211 | 1,353 |
| disk-pt | all_tokens | 16.0 | 8.000 | 111.8 | 0.019 | 34.2 | 17,219 | 10,812 |
| disk-pt | all_tokens | 64.0 | 1.000 | 64.2 | 0.016 | 9.719 | 17,213 | 5,385 |
| disk-pt | all_tokens | 64.0 | 8.000 | 254.0 | 0.029 | 139.3 | 17,255 | 43,069 |
| disk-pt | all_tokens | 256.0 | 1.000 | 80.5 | 0.012 | 18.2 | 17,239 | 21,513 |
| disk-pt | all_tokens | 256.0 | 8.000 | 689.0 | 0.019 | 174.2 | 17,431 | 172,093 |
| disk-pt | all_tokens | 512.0 | 1.000 | 247.6 | 0.019 | 35.8 | 17,211 | 43,017 |
| disk-pt | all_tokens | 512.0 | 8.000 | 749.9 | 0.022 | 487.6 | 17,273 | 344,125 |
| disk-pt | last_token | 16.0 | 1.000 | 45.5 | 0.008 | 4.129 | 17,211 | 92.7 |
| disk-pt | last_token | 16.0 | 8.000 | 116.3 | 0.010 | 23.9 | 17,211 | 731.1 |
| disk-pt | last_token | 64.0 | 1.000 | 65.8 | 0.008 | 4.258 | 17,211 | 92.7 |
| disk-pt | last_token | 64.0 | 8.000 | 144.3 | 0.054 | 52.3 | 17,213 | 731.1 |
| disk-pt | last_token | 256.0 | 1.000 | 62.2 | 0.081 | 14.0 | 17,211 | 92.7 |
| disk-pt | last_token | 256.0 | 8.000 | 201.1 | 0.035 | 49.9 | 17,213 | 731.1 |
| disk-pt | last_token | 512.0 | 1.000 | 53.3 | 0.009 | 4.721 | 17,211 | 92.7 |
| disk-pt | last_token | 512.0 | 8.000 | 162.4 | 0.017 | 26.8 | 17,211 | 731.1 |
| disk-pt-async | all_tokens | 16.0 | 1.000 | 22.6 | 0.247 | 7.701 | 17,211 | 1,353 |
| disk-pt-async | all_tokens | 16.0 | 8.000 | 57.1 | 0.206 | 48.2 | 17,219 | 10,812 |
| disk-pt-async | all_tokens | 64.0 | 1.000 | 21.5 | 0.165 | 12.6 | 17,213 | 5,385 |
| disk-pt-async | all_tokens | 64.0 | 8.000 | 67.1 | 0.181 | 89.8 | 17,251 | 43,069 |
| disk-pt-async | all_tokens | 256.0 | 1.000 | 28.0 | 0.184 | 35.5 | 17,239 | 21,513 |
| disk-pt-async | all_tokens | 256.0 | 8.000 | 116.4 | 0.199 | 326.7 | 17,431 | 172,093 |
| disk-pt-async | all_tokens | 512.0 | 1.000 | 41.6 | 0.194 | 66.2 | 17,211 | 43,017 |
| disk-pt-async | all_tokens | 512.0 | 8.000 | 145.9 | 0.237 | 503.2 | 17,273 | 344,125 |
| disk-pt-async | last_token | 16.0 | 1.000 | 53.8 | 0.238 | 6.428 | 17,211 | 92.7 |
| disk-pt-async | last_token | 16.0 | 8.000 | 166.5 | 1.150 | 88.8 | 17,211 | 731.1 |
| disk-pt-async | last_token | 64.0 | 1.000 | 81.5 | 0.618 | 9.798 | 17,211 | 92.7 |
| disk-pt-async | last_token | 64.0 | 8.000 | 120.2 | 0.593 | 52.5 | 17,211 | 731.1 |
| disk-pt-async | last_token | 256.0 | 1.000 | 65.6 | 0.691 | 10.3 | 17,211 | 92.7 |
| disk-pt-async | last_token | 256.0 | 8.000 | 131.5 | 0.420 | 110.3 | 17,211 | 731.1 |
| disk-pt-async | last_token | 512.0 | 1.000 | 59.9 | 0.513 | 11.8 | 17,211 | 92.7 |
| disk-pt-async | last_token | 512.0 | 8.000 | 201.5 | 0.456 | 54.2 | 17,211 | 731.1 |
| disk-st | all_tokens | 16.0 | 1.000 | 24.8 | 0.007 | 1.676 | 17,211 | 1,351 |
| disk-st | all_tokens | 16.0 | 8.000 | 71.2 | 0.006 | 2.222 | 17,219 | 10,759 |
| disk-st | all_tokens | 64.0 | 1.000 | 34.2 | 0.007 | 1.814 | 17,213 | 5,383 |
| disk-st | all_tokens | 64.0 | 8.000 | 117.4 | 0.007 | 2.346 | 17,251 | 43,015 |
| disk-st | all_tokens | 256.0 | 1.000 | 52.7 | 0.006 | 1.411 | 17,239 | 21,511 |
| disk-st | all_tokens | 256.0 | 8.000 | 298.9 | 0.020 | 3.270 | 17,431 | 172,040 |
| disk-st | all_tokens | 512.0 | 1.000 | 82.8 | 0.007 | 1.674 | 17,211 | 43,016 |
| disk-st | all_tokens | 512.0 | 8.000 | 530.4 | 0.019 | 3.254 | 17,273 | 344,072 |
| disk-st | last_token | 16.0 | 1.000 | 22.5 | 0.005 | 1.239 | 17,211 | 91.2 |
| disk-st | last_token | 16.0 | 8.000 | 55.4 | 0.010 | 2.106 | 17,211 | 679.2 |
| disk-st | last_token | 64.0 | 1.000 | 22.4 | 0.006 | 1.339 | 17,211 | 91.2 |
| disk-st | last_token | 64.0 | 8.000 | 70.2 | 0.008 | 2.290 | 17,211 | 679.2 |
| disk-st | last_token | 256.0 | 1.000 | 25.4 | 0.006 | 1.315 | 17,211 | 91.2 |
| disk-st | last_token | 256.0 | 8.000 | 75.2 | 0.005 | 1.803 | 17,211 | 679.2 |
| disk-st | last_token | 512.0 | 1.000 | 35.3 | 0.006 | 1.296 | 17,211 | 91.2 |
| disk-st | last_token | 512.0 | 8.000 | 106.3 | 0.006 | 1.884 | 17,211 | 679.2 |
| disk-st-async | all_tokens | 16.0 | 1.000 | 19.9 | 10.5 | 1.326 | 17,211 | 1,352 |
| disk-st-async | all_tokens | 16.0 | 8.000 | 52.0 | 20.8 | 2.089 | 17,219 | 10,760 |
| disk-st-async | all_tokens | 64.0 | 1.000 | 20.3 | 20.8 | 1.374 | 17,213 | 5,384 |
| disk-st-async | all_tokens | 64.0 | 8.000 | 67.7 | 83.0 | 3.287 | 17,251 | 43,016 |
| disk-st-async | all_tokens | 256.0 | 1.000 | 27.1 | 31.5 | 2.517 | 17,239 | 21,512 |
| disk-st-async | all_tokens | 256.0 | 8.000 | 114.9 | 203.1 | 3.065 | 17,431 | 172,040 |
| disk-st-async | all_tokens | 512.0 | 1.000 | 37.9 | 72.1 | 1.667 | 17,211 | 43,016 |
| disk-st-async | all_tokens | 512.0 | 8.000 | 180.1 | 354.2 | 2.911 | 17,273 | 344,072 |
| disk-st-async | last_token | 16.0 | 1.000 | 19.5 | 10.5 | 1.249 | 17,211 | 92.0 |
| disk-st-async | last_token | 16.0 | 8.000 | 50.9 | 10.5 | 1.653 | 17,211 | 680.1 |
| disk-st-async | last_token | 64.0 | 1.000 | 19.5 | 10.5 | 1.183 | 17,211 | 92.0 |
| disk-st-async | last_token | 64.0 | 8.000 | 60.2 | 10.6 | 2.151 | 17,211 | 680.1 |
| disk-st-async | last_token | 256.0 | 1.000 | 22.6 | 10.5 | 1.161 | 17,211 | 92.0 |
| disk-st-async | last_token | 256.0 | 8.000 | 73.7 | 10.5 | 1.636 | 17,211 | 680.1 |
| disk-st-async | last_token | 512.0 | 1.000 | 32.7 | 8.473 | 1.153 | 17,211 | 92.0 |
| disk-st-async | last_token | 512.0 | 8.000 | 100.5 | 10.6 | 1.837 | 17,211 | 680.1 |
| rpc | all_tokens | 16.0 | 1.000 | 38.7 | 0.003 | 0.021 | 17,211 | 0.000 |
| rpc | all_tokens | 16.0 | 8.000 | 250.0 | 0.003 | 0.022 | 17,219 | 0.000 |
| rpc | all_tokens | 64.0 | 1.000 | 14,538 | 0.003 | 0.032 | 17,213 | 0.000 |
| rpc | all_tokens | 64.0 | 8.000 | 128,989 | 0.004 | 0.044 | 17,251 | 0.000 |
| rpc | all_tokens | 256.0 | 1.000 | 3,887 | 0.003 | 0.030 | 17,239 | 0.000 |
| rpc | all_tokens | 256.0 | 8.000 | 82,983 | 0.004 | 0.034 | 17,431 | 0.000 |
| rpc | all_tokens | 512.0 | 1.000 | 9,221 | 0.004 | 0.033 | 17,211 | 0.000 |
| rpc | all_tokens | 512.0 | 8.000 | 89,527 | 0.005 | 0.053 | 17,273 | 0.000 |
| rpc | last_token | 16.0 | 1.000 | 45.3 | 0.010 | 0.039 | 17,211 | 0.000 |
| rpc | last_token | 16.0 | 8.000 | 205.4 | 0.004 | 0.029 | 17,211 | 0.000 |
| rpc | last_token | 64.0 | 1.000 | 41.0 | 0.003 | 0.025 | 17,211 | 0.000 |
| rpc | last_token | 64.0 | 8.000 | 160.3 | 0.003 | 0.022 | 17,211 | 0.000 |
| rpc | last_token | 256.0 | 1.000 | 39.1 | 0.003 | 0.035 | 17,211 | 0.000 |
| rpc | last_token | 256.0 | 8.000 | 183.6 | 0.004 | 0.023 | 17,211 | 0.000 |
| rpc | last_token | 512.0 | 1.000 | 52.1 | 0.003 | 0.029 | 17,211 | 0.000 |
| rpc | last_token | 512.0 | 8.000 | 196.1 | 0.003 | 0.022 | 17,211 | 0.000 |
| shm | last_token | 16.0 | 1.000 | 26.1 | 0.005 | 0.025 | 17,211 | 0.000 |
| shm | last_token | 16.0 | 8.000 | 90.3 | 0.004 | 0.014 | 17,211 | 0.000 |
| shm | last_token | 64.0 | 1.000 | 25.1 | 0.004 | 0.018 | 17,211 | 0.000 |
| shm | last_token | 64.0 | 8.000 | 101.8 | 0.004 | 0.018 | 17,211 | 0.000 |
| shm | last_token | 256.0 | 1.000 | 27.1 | 0.003 | 0.015 | 17,211 | 0.000 |
| shm | last_token | 256.0 | 8.000 | 116.1 | 0.004 | 0.018 | 17,211 | 0.000 |
| shm | last_token | 512.0 | 1.000 | 36.2 | 0.003 | 0.016 | 17,211 | 0.000 |
| shm | last_token | 512.0 | 8.000 | 137.6 | 0.004 | 0.020 | 17,211 | 0.000 |

## Top per-stage timers (global mean ms)

| timer | mean_ms_global |
|---|---|
| hookllm.generate | 3,840 |
| rpc.get_states | 2,706 |
| io.artifact_load.pt | 55.8 |
| async.save_iter | 53.9 |
| rpc.flush_disk | 44.0 |
| worker.disk_write.safetensors | 43.1 |
| hookllm.analyze | 29.3 |
| rpc.decompress | 14.7 |
| worker.cpu_transfer.hs | 14.2 |
| io.artifact_load.safetensors | 1.671 |
| hookllm.merge_probes | 0.050 |
| hookllm.build_extra | 0.014 |
| worker.queue_put | 0.012 |
| analyzer.kernel | 0.009 |
