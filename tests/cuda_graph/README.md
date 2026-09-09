# CUDA-Graph Oracles

## 1. What these are

These are subprocess-isolated **graph-vs-eager numerical oracles**, compared at
`rtol=atol=1e-2`. Each harness captures a run under `--mode graph`, then the same run under
`--mode eager`, then compares the two capture dumps layer-by-layer / request-by-request.

They run on **LSF** (`bsub`), not `pytest`. The GPU is allocated `mode=exclusive_process`, so
only one process may hold the device at a time — a single Python process cannot boot two vLLM
engines to compare them in-process. Each harness instead does `capture --mode graph`,
`capture --mode eager`, `compare` as three separate `python` invocations chained in one shell
script, so each engine's process exit fully releases the GPU before the next one boots.

## 2. How to run one

```bash
bsub -G grp_exploratory < tests/cuda_graph/tests/qk_graph/run_qk_parity_full.sh
bjobs
bpeek <JOBID>
grep VERDICT tests/cuda_graph/logs/<name>.<JOBID>.out
```

- `-G grp_exploratory` is **mandatory** — `bsub` without a group is rejected.
- Submit from the **repo root** (`~/vLLM-Hook`), not from inside `tests/cuda_graph/`: every
  runner does its own `cd ~/vLLM-Hook` and refers to paths relative to the repo root.
- `tests/cuda_graph/logs/` is **gitignored**. Logs only appear once a job actually starts
  running, not at submit time.

## 3. The oracle table

One row per surviving oracle. "Proves" is a one-line summary of what a passing `VERDICT`
means; `harness` is the `.py` doing the capture/compare; `runner` is the `.sh` you `bsub`.

| Oracle | Proves | Harness | Runner |
|---|---|---|---|
| `qk_parity` | Post-RoPE Q/K captured under buffer-mode FULL cudagraph decode matches the legacy eager `register_forward_hook` capture, per layer/request (incl. a per-request-egress fallback leg). | `tests/cuda_graph/tests/qk_graph/qk_parity.py` | `tests/cuda_graph/tests/qk_graph/run_qk_parity_full.sh` |
| `hs_parity` | Hidden states captured under buffer-mode FULL cudagraph decode match the legacy eager capture, prefill-only and prefill+decode legs, with and without batched egress. | `tests/cuda_graph/tests/hs_graph/hs_parity.py` | `tests/cuda_graph/tests/hs_graph/run_hs_parity_full.sh` |
| `hs_multireq` | The vectorized/batched HS egress correctly demuxes a concurrent multi-request batch: each request's graph capture matches that same request run alone (solo-eager), across staggered prompt lengths and staggered `max_tokens`. | `tests/cuda_graph/tests/hs_graph/hs_multireq.py` | `tests/cuda_graph/tests/hs_graph/run_hs_multireq.sh` |
| `steer_parity` | Graph-vs-eager activation-steering parity (steering analogue of `hs_parity`). | `tests/cuda_graph/tests/steer_graph/steer_parity.py` | `tests/cuda_graph/tests/steer_graph/run_steer_parity.sh` |
| `steer_parity_decode` | Under buffer-mode FULL cudagraph decode, the in-place residual-steering op (`vllm_hook::steer_buffer`) honors per-request coeff/vector routing and its mutation propagates correctly on cudagraph replay. | `tests/cuda_graph/tests/steer_graph/steer_parity_decode.py` | `tests/cuda_graph/tests/steer_graph/run_steer_parity_full.sh` |
| `hs_ring_parity` | Graph-vs-eager hidden-state parity for the **capture-ring** path (persistent per-layer GPU ring, one logical cursor across layers), as opposed to the RPC-bank buffer path above. | `tests/cuda_graph/tests/ring/hs_ring_parity.py` | `tests/cuda_graph/tests/ring/run_hs_ring_parity.sh` |
| `hs_ring_perreq_parity` | Byte-identity gate for the **per-request** HS capture-ring demux (each request's slice of the shared ring matches its solo-eager baseline). | `tests/cuda_graph/tests/ring/hs_ring_perreq_parity.py` | `tests/cuda_graph/tests/ring/run_hs_ring_perreq_parity.sh` |
| `hs_ring_selective_parity` | GPU oracle for selective drain (Lever C, `VLLM_HOOK_DRAIN_SELECTIVE`) — exercises the CUDA branch of the ring reader that the CPU-only unit tests can't reach. | `tests/cuda_graph/tests/ring/hs_ring_selective_parity.py` | `tests/cuda_graph/tests/ring/run_hs_ring_selective_parity.sh` |
| `qk_ring_parity` | Graph-vs-eager QK parity for the capture-ring path (post-RoPE q + k scattered into two per-layer GPU rings sharing one cursor). | `tests/cuda_graph/tests/ring/qk_ring_parity.py` | `tests/cuda_graph/tests/ring/run_qk_ring_parity.sh` |
| `qk_ring_perreq_parity` | Byte-identity gate for the per-request QK capture-ring demux (QK analogue of `hs_ring_perreq_parity`). | `tests/cuda_graph/tests/ring/qk_ring_perreq_parity.py` | `tests/cuda_graph/tests/ring/run_qk_ring_perreq_parity.sh` |
| `serve_per_request` | Online-serve validation of the per-request HS capture-ring delivery path, driving the real `AsyncLLM.generate` code path (not the offline `HookLLM` harness scaffolding). | `tests/cuda_graph/tests/ring/serve_per_request.py` | `tests/cuda_graph/tests/ring/run_serve_per_request.sh` |
| `qk_serve_per_request` | Online-serve validation of the per-request QK capture-ring delivery path (QK analogue of `serve_per_request`). | `tests/cuda_graph/tests/ring/qk_serve_per_request.py` | `tests/cuda_graph/tests/ring/run_qk_serve_per_request.sh` |
| `chunked_prefill_parity` | Buffer-mode QK capture is correct when a single prompt is split across multiple prefill steps (forced by booting with a low `max_num_batched_tokens`). | `tests/cuda_graph/tests/chunked_prefill/chunked_prefill_parity.py` | `tests/cuda_graph/tests/chunked_prefill/run_chunked_parity.sh` |
| `prefix_cache_parity` | Buffer-mode QK capture reconstructs full-context K correctly when prefix caching trims a prompt's shared prefix (continuous-batching regime). | `tests/cuda_graph/tests/prefix_cache/prefix_cache_parity.py` | `tests/cuda_graph/tests/prefix_cache/run_prefix_cache_parity.sh` |
| `quant_parity` | GPU value-parity for artifact quantization: int8/int4/int2 captures match the fp16 reference, for both the `qk` and `hs` workers. | `tests/cuda_graph/tests/quant_parity/quant_parity.py` | `tests/cuda_graph/tests/quant_parity/run_quant_parity.sh` |
| `cb_oom_parity` | The continuous-batching OOM fix (continuous async drain + resident-ceiling admission throttle). **PARKED**: deliberately left untouched by controller ruling — see §1 note below. | `tests/cuda_graph/tests/cb_oom/cb_oom_parity.py` | `tests/cuda_graph/tests/cb_oom/run_cb_oom.sh` |
| `qk_score_parity` | GPU-side attention-score capture matches a score recomputed from a QK capture of the same run, across eager, op (PIECEWISE), and buffer (FULL) paths. | `tests/cuda_graph/tests/qk_score/qk_score_parity.py` | `tests/cuda_graph/tests/qk_score/run_qk_score_graph.sh` |
| `mixed_auto_parity` | Mixed-batch merge + auto-select capture-representation integration under buffer-mode FULL. | `tests/cuda_graph/tests/qk_score/mixed_auto_parity.py` | `tests/cuda_graph/tests/qk_score/run_mixed_auto.sh` |
| `qk_multireq` | Multi-request QK capture correctness under buffer-mode FULL: each request in a concurrent batch matches its own solo-eager baseline (q + k_all + per-request layer set), with idle-skip enabled. | `tests/cuda_graph/tests/qk_multireq/qk_multireq.py` | `tests/cuda_graph/tests/qk_multireq/run_qk_multireq.sh` |
| `func_matrix` | The per-step capture/steer optimizations hold for multi-request batches, graph-vs-eager, across scenarios. | `tests/cuda_graph/tests/func_matrix/func_matrix.py` | `tests/cuda_graph/tests/func_matrix/run_func_matrix.sh` |
| `verify_both_decode` | The eager (main-line) hook path truly captures (QK, HS) and steers at the decode stage under `hooks_on="both"`. | `tests/cuda_graph/tests/decode_verify/verify_both_decode.py` | `tests/cuda_graph/tests/decode_verify/run_decode_verify.sh` |
| `verify_graph_decode` | The FULL cudagraph (buffer) capture/steer path is correct at the decode stage against an independent ground truth (teacher forcing) — not a comparison to the eager path. | `tests/cuda_graph/tests/decode_verify/verify_graph_decode.py` | `tests/cuda_graph/tests/decode_verify/run_graph_decode_verify.sh` |
| `capture_fused_unit` | Bit-exactness of the Triton-fused `capture_hs` scatter kernel vs the aten reference, for both destination modes. | `tests/cuda_graph/tests/capture_perf/capture_fused_unit.py` | `tests/cuda_graph/tests/capture_perf/run_capture_fused_unit.sh` |
| `writer_process_equiv` | The disk-writer thread path and the spawned-writer-process path produce byte-identical on-disk artifacts (`.safetensors`/`.json`/`.pt`), under FULL cudagraph + buffer capture. | `tests/cuda_graph/tests/capture_perf/writer_process_equiv.py` | `tests/cuda_graph/tests/capture_perf/run_writer_process.sh` |

Two small standalone helpers back the harnesses above and are not run directly:
`tests/cuda_graph/tests/_aot_helper.py` (AOT-compile evidence via `collective_rpc`, used by the
parity harnesses) and `tests/cuda_graph/tests/decode_verify/_rpc_helper.py` (the same
cross-process trick for `decode_verify`).

**Note on `cb_oom_parity`:** this file is deliberately left as-is (17 references to since-deleted
diagnostic env vars, actively set by its own probes). It is a GPU-only LSF oracle outside the
no-GPU release gate; cleaning up its probes needs a human decision on what those probes should
check instead, not a docs pass. Do not "helpfully" edit it — see `tests/unit/test_no_dead_levers.py`
for the enforcement that would catch a real regression, and the plan's controller notes for the
parking rationale.

## 4. The three that need no GPU job

Five files in this tree don't go through `bsub` at all:

- `tests/cuda_graph/tests/qk_score/auto_select_unittest.py` and
  `tests/cuda_graph/tests/capture_perf/qk_compact_equiv.py` are plain CPU/no-engine scripts —
  run them as `python <path>` from anywhere.
- `tests/cuda_graph/tests/steer_graph/test_steer_fused_unit.py`,
  `tests/cuda_graph/tests/steer_graph/test_scatter_routing_gpu.py`, and
  `tests/cuda_graph/tests/steer_graph/test_multilayer_steer_routing.py` are `pytest`-collected
  from the repo root (`test_*.py` naming). `test_steer_fused_unit.py` and
  `test_scatter_routing_gpu.py` need a GPU; each also has a convenience LSF wrapper
  (`run_steer_fused_unit.sh`, `run_scatter_routing_gpu.sh`) for batch submission on the cluster,
  but neither needs the multi-process capture/compare dance the oracle table above does — one
  process, one engine (or no engine at all for `test_multilayer_steer_routing.py`), so `pytest`
  collection works fine.

## 5. Load-bearing environment facts

These are still true on this branch and worth knowing before you read or write a harness:

- **Compile-cache poisoning.** vLLM's compile-cache key ignores `worker_extension_cls`, so a
  stale cached graph from a previous config can silently poison a later run. Every runner sets
  `export VLLM_DISABLE_COMPILE_CACHE=1` — mandatory whenever you're comparing configs (native vs
  plugin, or one oracle leg vs another) in the same campaign.
- **fork vs spawn.** Every harness forces `multiprocessing.set_start_method("spawn", force=True)`
  and `VLLM_WORKER_MULTIPROC_METHOD=spawn` before importing vLLM. A forked worker would inherit
  the parent process's (possibly already-initialized) CUDA context; `spawn` guarantees a clean
  process and a clean CUDA init for every engine boot.
- **`cudagraph_mode` plumbing.** Harnesses read `VLLM_HOOK_CUDAGRAPH_MODE` (default `FULL`, or
  `PIECEWISE` for the older `hs_parity.py` default) and, in `--mode graph` with the value not
  `NONE`, pass it through as `compilation_config={"cudagraph_mode": <value>}` when building the
  engine.
- **The plugin forces eager unless armed.** `_hook_plugin.py` forces `enforce_eager=True` by
  default — the legacy `register_forward_hook` capture path needs `torch.compile` off, or Dynamo
  traces the hook away. Setting `VLLM_HOOK_ALLOW_CUDAGRAPH=1` is what lets `enforce_eager=False`
  (and cudagraphs) stand, and arms the buffer-mode capture install path used by the graph leg of
  every oracle above.
