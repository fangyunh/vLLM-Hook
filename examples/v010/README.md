# vLLM-Hook v0.1.0 profiling assets

Everything needed to profile the **v0.1.0** plugin variant with **our** harness
(`vllm-hook-profile`), per `vLLM-Hook-Profiling/docs/PROFILING_WORKFLOW_v010_hybrid.md` §1B.

## What v0.1.0 is (and why it needs a separate driver)

v0.1.0 predates the `worker_extension_cls` mixin path. It attaches to vLLM via
`LLM(worker_cls=<class>)` (a full Worker **replacement**) and gates hooks per
forward on **filesystem flags + env vars**, not on `SamplingParams.extra_args`:

| env | meaning |
|---|---|
| `VLLM_HOOK_DIR` | artifact root |
| `VLLM_HOOK_FLAG` | present → hooks capture/steer this pass; absent → inert |
| `VLLM_RUN_ID` | file of run_ids; the worker reads the **last** line each pass |
| `VLLM_HOOK_LAYERS` / `VLLM_HOOK_HS_MODE` | hidden-states targets |
| `VLLM_HOOK_LAYER_HEADS` / `VLLM_HOOKQ_MODE` | qk targets |
| `VLLM_ACTSTEER_CONFIG` | steer config (JSON with an absolute `vector_path`) |

The v0.1.0 worker classes live in the local fork at
`vLLM-Hook-Profiling/vendor/vLLM-Hook-v010/profiling/peers/v010/workers/`.

## How the harness drives it

The harness ports this paradigm in `vllm_hook_profiling/drivers/offline_v010.py`
(`driver: offline_v010`). It reuses the entire v0.2.x pipeline (warm-up, rep loop,
per-rep latency/throughput/memory, aggregation, TTFT, CSV, report) and overrides
only the engine build (`LLM(worker_cls=...)`) and the tier control:

| tier | v0.1.0 mechanics |
|---|---|
| `baseline` | plain `LLM` (no worker), flag never set |
| `idle_hook` | v0.1.0 worker installed (hooks fire every forward) but flag never set → inert → hook-dispatch tax |
| `active` | worker installed + flag set around each measured `generate` → full capture/steer |

`VLLM_PLUGINS=""` (exported by the runner) disables the fork's `vllm.general_plugins`
entry point for **every** tier — otherwise it injects a v0.2.x `worker_extension_cls`
whenever the field is falsy and forces `enforce_eager=True`, contaminating the
baseline and colliding with the v0.1.0 `worker_cls`.

## Files here

| file | purpose |
|---|---|
| `configs/{hs_lasttok,hs_alltok,qk_lasttok,qk_alltok}_granite.json`, `configs/steer_phi3.json` | capture/steer targets (same layers/heads/vector as v020) the `offline_v010` driver translates into the v0.1.0 env |
| `run_v010.bsub` | LSF runner — parametrized by `VHP_RUN_YAML`; sets `vllm_hook_env_v010`, `PYTHONPATH`, `VLLM_PLUGINS=""` |
| `demo_v010_selfcheck.py` | optional `VERDICT: PASS/FAIL` sanity demo (peer `V010Engine`); **not** the profiling path |
| `logs/` | LSF `.out`/`.err` |

The campaign specs (hand-authored run.yamls) and outputs live in the profiler repo
under `results/v010/vllm-hook_<workload>_<model>/`.

## Run a campaign

```bash
cd ~/vLLM-Hook-Profiling
bsub -G grp_exploratory -gpu "num=1:mode=exclusive_process" \
     -env "all, VHP_RUN_YAML=results/v010/<campaign>/run.yaml, VHP_REPS=16" \
     < ../vLLM-Hook/examples/v010/run_v010.bsub
```

Heavy `all_tokens` campaigns: keep `VHP_REPS ≤ 8` (the 2-hour LSF RUNLIMIT — v0.1.0
re-serializes the growing capture cache every decode step, so all_tokens is much
heavier in-forward than v0.2.x).
