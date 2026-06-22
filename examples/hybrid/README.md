# Hybrid-mode (CUDA-graph) demos — Granite-3.1-8B

Drop-in Hybrid-mode counterparts of `examples/demo_actsteer.py` and the two
`examples/profiling_longdecode/` capture demos. They run the **same workloads and
settings** (Granite-3.1-8B, long decode, capture/steer in both phases) but with
**CUDA graphs ON** (v0.3.0 Hybrid mode) instead of the legacy `enforce_eager=True`
path.

A demo becomes Hybrid by flipping exactly three switches — everything else is
identical to the eager demo it mirrors:

```python
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"          # 1. arm the graph install path
...
HookLLM(..., enforce_eager=False,                       # 2. let vLLM compile + capture graphs
        compilation_config={"cudagraph_mode": "PIECEWISE"})   # 3. the proven cudagraph mode
```

Under the hood the capture/steer logic is unchanged: each worker's hook body runs
inside a custom **splitting op** (`vllm_hook::qk_probe` / `hs_probe` / `steer_residual`)
that the piecewise compiler keeps as an **eager seam** between graph segments, so it
fires on every prefill and decode step and writes the same worker buckets the eager
path uses. Steering additionally mutates the residual **in place**
(`mutates_args=["residual"]`) so the edit propagates through the cudagraph-replayed
downstream layers.

## The three demos

| Demo | Worker | What it proves under Hybrid mode |
|---|---|---|
| `demo_actsteer_hybrid.py` | `steer_hook_act` | Activation steering (`adjust_rs`, layer 15) changes the output under CUDA graphs. |
| `demo_hiddenstate_longdec_hybrid.py` | `probe_hidden_states` | Residual-stream capture across prefill+decode (long decode=128). |
| `demo_attntracker_longdec_hybrid.py` | `probe_hook_qk` | Q/K capture + attention-tracker score across prefill+decode. |

Each prints a `VERDICT: PASS/FAIL` self-check.

## Steering vector note

The granite steer config uses `method: adjust_rs` and points at
`steering_vectors/granite_format.pt`, which is not checked in. Generate a dummy
(unit `dir` + scalar `avg_proj`, sized to granite's 4096-wide residual) once:

```bash
python steering_vectors/_make_granite_dummy.py
```

Like `qwen2_dummy.pt`, the direction is meaningless — it exists to exercise the
`adjust_rs` steering math, not to steer toward a real behaviour.

## Running

```bash
# All three, back to back, on the GPU cluster:
bsub < examples/hybrid/run_hybrid_demos.sh
grep VERDICT examples/hybrid/logs/hybrid_demos.*.out

# Rigorous graph-vs-eager parity on Granite-8B (definitive correctness proof):
bsub < examples/hybrid/run_hybrid_granite_steer_parity.sh
grep 'VERDICT' tests/cuda_graph/logs/steer_parity*.out
```

Env knobs match the eager demos: `VLLM_HOOK_DEMO_MODEL`, `VLLM_HOOK_CONFIG_FILE`,
`VLLM_HOOK_DEMO_MAX_TOKENS`, `VLLM_HOOK_DEMO_HOOKS_ON`. The bsub runner caps decode at
128 tokens for a quick check; the script defaults still match the eager demos
(actsteer defaults to 2048).
