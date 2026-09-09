# bluevela

LSF job scripts for the `BLUEVELA_LSF` cluster. Submit from a login node, from the repo root.

## Run the hidden-state capture demo

```bash
bash -lc 'bsub -G grp_exploratory < bluevela/run_hs_capture.sh'
bjobs                                              # queued / running / done
bpeek <JOBID>                                      # live stdout
grep -E 'mode=|model.layers' bluevela/logs/hs_capture.<JOBID>.out
bkill <JOBID>                                      # stop it
```

`-G grp_exploratory` is mandatory. Logs appear under `bluevela/logs/` only once the job starts,
not at submit time; the directory is gitignored.

## Knobs

| var | default | effect |
|---|---|---|
| `HS_MODE` | `both` | `eager`, `graph`, or `both` legs |
| `VLLM_HOOK_DEMO_MODEL` | `Qwen/Qwen2-1.5B-Instruct` | model to capture from |
| `VLLM_HOOK_CONFIG_FILE` | `model_configs/hidden_states/Qwen2-1.5B-Instruct.json` | which layers / `mode` |
| `VLLM_HOOK_PROFILE` | `1` | profiler counters (no-ops if `0`) |

Override at submit time:

```bash
bash -lc 'HS_MODE=graph VLLM_HOOK_DEMO_MODEL=Qwen/Qwen2.5-3B-Instruct \
  bsub -G grp_exploratory < bluevela/run_hs_capture.sh'
```

## What the two legs mean

- **eager** — forward-hook capture, the shipped default.
- **graph** — FULL CUDA-graph capture ring (`VLLM_HOOK_ALLOW_CUDAGRAPH=1`). Without that var the
  plugin forces `enforce_eager=True`, so this leg is the only one that exercises the ring.

Both legs use the same prompts and config, so the printed per-layer shapes and norms are directly
comparable. Differing norms between legs mean the graph path is not reproducing the eager capture.

## Gotchas

- Submit under `bash -lc`. A non-interactive shell never sources `~/.bashrc`, so `HF_HOME` and
  friends silently vanish and the job fails at model resolution.
- `HF_HUB_OFFLINE=1` is set in the script. Unset it for a first run with an uncached model.
- A hard-killed job leaves a UUID-named socket in the repo root. Sweep with
  `find . -maxdepth 1 -type s -delete`.
- The GPU is `exclusive_process` — one engine per job. Two legs run sequentially, not in parallel.
