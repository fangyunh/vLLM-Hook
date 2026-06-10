# Long-decode profiling demos (Granite-8B)

Drop-in replacements for `examples/demo_attntracker.py` / `examples/demo_hiddenstate.py`
used **only for profiling** with vLLM-Hook-Profiling. They fix the short-cell
measurement noise (#2) by making the capture cost large enough to measure:

- **Capture in both phases** — `extra_args={"hooks_on": "both"}` (the workers default
  to `"prefill"` only). The hook now fires on every **decode** step, so the capture
  cost accumulates over the decode instead of being a one-shot prefill cost.
- **Long decode** — `max_tokens` defaults to **128** (env `VLLM_HOOK_DEMO_MAX_TOKENS`),
  so the run is multi-second and steady-clock, lifting the signal above GPU jitter.

The profiler traces the first `save_to_disk=True` generate and bakes
`max_tokens` + `hooks_on` + `hookq_mode`/`hs_mode` into the campaign `run.yaml`.

## The 4 tasks → (script, config)

`last_token` vs `all_tokens` is the config file, exactly like the stock demos:

| Task | Script | `VLLM_HOOK_CONFIG_FILE` |
|---|---|---|
| qk · last_token | `demo_attntracker_longdec.py` | *(default)* `model_configs/attention_tracker/granite-3.1-8b-instruct.json` |
| qk · all_tokens | `demo_attntracker_longdec.py` | `model_configs/attention_tracker/granite-3.1-8b-instruct_alltok.json` |
| hs · last_token | `demo_hiddenstate_longdec.py` | *(default)* `model_configs/hidden_states/granite-3.1-8b-instruct.json` |
| hs · all_tokens | `demo_hiddenstate_longdec.py` | `model_configs/hidden_states/granite-3.1-8b-instruct_alltok.json` |

All default to `ibm-granite/granite-3.1-8b-instruct` (override with `VLLM_HOOK_DEMO_MODEL`).

## Env knobs

| Var | Default | Meaning |
|---|---|---|
| `VLLM_HOOK_DEMO_MODEL` | `ibm-granite/granite-3.1-8b-instruct` | HF model id |
| `VLLM_HOOK_CONFIG_FILE` | per-script last_token config | swap to `*_alltok.json` for all_tokens |
| `VLLM_HOOK_DEMO_MAX_TOKENS` | `128` | decode length (the #2 lever) |
| `VLLM_HOOK_DEMO_HOOKS_ON` | `both` | `prefill` \| `decode` \| `both` |

## Running through the profiler

Point an LSF runner's `SCRIPT` at one of these (everything else in
`examples/lsf/run_attn_*/run_hs_*.bsub` stays the same — they already set the model
and `VLLM_HOOK_CONFIG_FILE`). For example, to profile qk·all_tokens with long decode:

```bash
export VLLM_HOOK_DEMO_MODEL="ibm-granite/granite-3.1-8b-instruct"
export VLLM_HOOK_CONFIG_FILE="model_configs/attention_tracker/granite-3.1-8b-instruct_alltok.json"
export VLLM_HOOK_DEMO_MAX_TOKENS=128
SCRIPT="$VLLM_HOOK_SRC/examples/profiling_longdecode/demo_attntracker_longdec.py"
vllm-hook-profile setup --script "$SCRIPT" && vllm-hook-profile run <generated run.yaml>
```

> **Note — same campaign folder name.** The campaign name is derived from
> worker + capture-mode + model (not decode length), so a long-decode run writes to
> the *same* `results/vllm-hook_probe-hook-qk-lasttok_…/` folder and **overwrites** the
> short-decode results. That's intended (long-decode is the corrected methodology). If
> you want to keep both, copy the old folder aside first.

> **Cost.** `hooks_on=both` + decode 128 captures on every step, so the artifact and
> async-flush grow with decode length, and each rep is multi-second. With
> `VHP_REPS=20` (the heat-soak default) expect long jobs; tune `VLLM_HOOK_DEMO_MAX_TOKENS`
> and `VHP_REPS` to taste.
