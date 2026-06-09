# Token Highlighter Documentation

Original Paper: [Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for LLMs](https://arxiv.org/pdf/2412.18171).

---

## Goals, Architecture, and Design Intent

### Goal

Identify prompt tokens that most increase the likelihood of a fixed **affirmation target phrase** (affirmation loss), then optionally **mitigate** by scaling those driver-token embeddings by `β < 1` during a later prefill so generation runs against an altered KV cache.

High-level algorithm:

1. **Affirmation loss** (teacher-forced): `L_aff(X) = -Σ_i log P(y_i | X, y_<i)`
2. **Per-token influence scores** via closed-form last-block attribution (`forward_attr`).
3. **Driver selection**: `mean_std` or `top_percentage` (`α`, `k`).
4. **Mitigation**: soft-remove drivers at prefill → decode from mitigated cache.

### Architecture (single engine and worker)

```
┌─────────────────────────────────────────────────────────────┐
│  HookLLM  —  single vLLM LLM + worker_extension_cls         │
│                                                             │
│  HighlighterWorker (token_highlighter)                      │
│    • same class for capture AND mitigate                    │
│    • mode per request: extra_args['highlighter_mode']       │
│                                                             │
│  HighlighterAnalyzer (token_highlighter)                    │
│    • post-hoc scoring / formatting from disk traces         │
└─────────────────────────────────────────────────────────────┘
```


| Component               | Role                                                                                                                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HookLLM**             | One GPU engine; passes `highlighter_mode`, `run_id`, `hook_dir` via `SamplingParams.extra_args`.                                                                                         |
| **HighlighterWorker**   | **Capture:** hooks, traces, `highlighter_activations.pt`. **Mitigate:** load `highlighter.pt`, queue β-scaled prompt embeddings, run scheduler prefill + decode.                         |
| **HighlighterAnalyzer** | Analytic scores from `highlighter_activations.pt` + `weight_bundle` (pure tensor math; no `from_pretrained`). Falls back to reading existing `highlighter.pt` if activations are absent. |


Registry (`vllm_hook_plugins/__init__.py`): worker and analyzer name `token_highlighter`.

There is **no second vLLM engine** and **no separate capture vs mitigate worker class**. Mitigation is a **later** `generate(..., highlighter_mode="mitigate")` on the same `HookLLM` instance, reusing the capture `run_id` so the worker loads the same `highlighter.pt`.

### Summary


| Phase        | Who                       | What                                                                                                    |
| ------------ | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Capture**  | Worker + scheduler        | Teacher prefill (for scoring), hooks, trace I/O; then **decode** (typically unmitigated KV).            |
| **Analyze**  | Analyzer (driver process) | Turn activations into `token_scores` + `soft_indices` in `highlighter.pt`.                              |
| **Mitigate** | Worker + scheduler        | Load scores → **prefill with softened prompt embeddings** → K/V cache reflects mitigation → **decode**. |


Scoring uses **forward_attr** only: vLLM hooks during capture + closed-form attribution in the analyzer.

### Design intent

- **Worker** owns everything that must touch the live vLLM model: hooks, capture I/O, embedding soft-removal at mitigate prefill, KV effects via normal attention after the embedding hook.
- **Analyzer** owns offline math on traces so analyze does not load a second full model when `weight_bundle` is present.
- One capture `run_id` can be analyzed many times (different `analyzer_spec`) without re-running vLLM.
- Implementation lives in `workers/highlighter_worker.py`, `analyzers/highlighter_analyzer.py`, `utils/TokenHighlighter/`*.

---

## vLLM runtime integration

How `HighlighterWorker` fits into vLLM and how that differs from probe workers (HS/QK).

### Worker mixin (not a separate engine)

`HookLLM` passes `worker_extension_cls=HighlighterWorker` into vLLM. vLLM **mixes** that class onto its GPU `Worker` (multiple inheritance). Inside mixin methods, `self` is the real vLLM worker: `self.execute_model`, `self.model_runner`, `self.rank`, etc. The mixin **adds** methods and can **replace** methods on that object; it does not replace the scheduler or engine.

```
Driver (HookLLM)                    GPU worker subprocess
─────────────────                   ─────────────────────
LLM.generate(extra_args=…)   →      scheduler loop:
  patched generate                      execute_model(scheduler_output)
    collective_rpc("install_hooks")        ↑ (wrapped after install)
```

### `install_hooks` — project convention, not a vLLM callback

vLLM’s scheduler calls `execute_model` every scheduling step. It **does not** call `install_hooks`. That name is this repo’s **lazy setup RPC**: `_hook_plugin.py` calls `collective_rpc("install_hooks")` once before the first hooked request so setup runs in the worker **after** the model is loaded.

For highlighter, `install_hooks()` does three things (no forward pass):

1. **Init state** — capture buffers, mitigate queues, config fields.
2. **Register the embedding forward hook** on `input_embeddings` (always attached for the engine lifetime).
3. **Wrap `execute_model`** — save vLLM’s original method as `orig_execute`, replace `self.execute_model` with a closure that calls `_highlighter_execute_model_step(orig_execute, ...)`.

Capture Q/K/V + RoPE probe hooks are **not** installed here. They install on the **first capture step** in `_ensure_capture_hooks()`, after `_sync_highlighter_paths` applies `extra_args["highlighter"]` (so `require_attn_metadata`, `allow_prerope_fallback`, etc. are known).

### `_highlighter_execute_model_step` — one scheduler step

Each call = one vLLM scheduling step (one batch of tokens through the model):

| Phase | Capture | Mitigate |
| ----- | ------- | -------- |
| **Pre** (`super_execute_model` not yet run) | Sync config/paths; `_ensure_capture_hooks()`; track prompts; optionally patch scheduler for extended teacher prefill (`prompt + target[:-1]`); set `cap.active=True` | `_load_scores()`; `_queue_soft_embeddings()` into `_pending_soft` |
| **Forward** | `orig_execute_model(...)` — hooks fill `cap.live` while `cap.active` | Same — embedding hook swaps β-scaled driver rows when token ids match `_pending_soft` |
| **Post** | Restore scheduler; `cap.active=False`; when prefill done → `_finish_capture()` writes `highlighter_activations.pt` (or defer via `_capture_finish_pending`) | Return (no capture I/O) |

The analyzer runs in the **driver** process on disk artifacts; the plugin does **not** call `get_captured_states` / `flush_disk` for highlighter (probe-only post-path).

### Two hook layers

| Hook | When registered | Capture | Mitigate |
| ---- | --------------- | ------- | -------- |
| **Embedding soft hook** | `install_hooks()` | Registered always; **no-op** (`_pending_soft` empty) | Active when mitigate queued soft embeddings |
| **Forward-attr + RoPE probe** | First capture step | Records Q/K/V, `h_input`, `h_L` while `cap.active` | Not used |

Mitigate does not re-run forward_attr scoring; it reads `highlighter.pt` from the capture/analyze pass.

### Plugin (`_hook_plugin.py`) — minimal highlighter surface

- `needs_hooks` includes `highlighter_mode` (with HS/QK/steer) **only** to trigger `collective_rpc("install_hooks")`.
- Post-generate probe RPCs (`get_captured_states`, `flush_disk`, `clear_captured_states`) run only for HS/QK requests (same pattern as excluding steer with `not wants_steer`).
- All highlighter-specific behavior lives in the worker mixin, not in the plugin.

### Configuration delivery

Settings come from `model_configs/token_highlighter/*.json` → `HookLLM._highlighter_config` → `extra_args["highlighter"]` on every request (same pattern as `extra_args["steer"]`). No `VLLM_HIGHLIGHTER_*` env vars. Runtime overrides: mutate `llm._highlighter_config` before `generate()` (e.g. `target_token_ids` after tokenizer encode, `beta` for ablations).

### Prefix caching

`enable_prefix_caching=False` in demos is a safe default: capture needs a full prefill with hooks; mitigate needs a **fresh** prefill with β-scaled embeddings. Prefix-cache hits can skip those forwards. Caching can work if prompts do not share prefixes you care about; call `llm.llm_engine.reset_prefix_cache()` between capture and mitigate on the same prompt when caching is enabled.

---

## End-to-End Pipeline

### Overview

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ capture  │ ──► │ analyze  │ ──► │ mitigate │     │ (optional│
  │ generate │     │          │     │ generate │     │  inspect)│
  └──────────┘     └──────────┘     └──────────┘     └──────────┘
       │                │                 │
       │                │                 └── same prompt, same run_id,
       │                │                     β-scaled embeds at prefill
       └── hooks +      └── highlighter.pt
           activations       (scores, soft_indices)
```

Artifacts per run: `{hook_dir}/{run_id}/tp_rank_0/` → `highlighter_activations.pt`, `highlighter.pt`.

### Capture (`highlighter_mode="capture"`)

**Purpose:** Score the prompt (identify drivers) and complete a normal generation pass for inspection. Capture does **not** apply soft-removal unless you intentionally set `beta` behavior via config; the mitigate pass is where embedding scaling is applied for steering.

**Worker** (same `HighlighterWorker`):

1. **Setup** (once per engine): `collective_rpc("install_hooks")` wraps `execute_model` and registers the mitigate embedding hook. **Capture hooks** (last-block **Q, K, V** post-RoPE, `h_input`, `h_L`, plus a one-shot RoPE probe) install on the first capture step via `_ensure_capture_hooks()`.
2. **Teacher prefill** on `prompt + target[:-1]` (`extended` or `suffix` capture mode) so activations at both prompt and generation positions exist in the trace.
3. **Persist traces:** `highlighter_activations.pt` (per-sequence `capture` tensors + precomputed `g_loss`) + `weight_bundle` (`W_O`, `W_Q`, `W_V`, `W_K`, `final_norm`, `input_norm`, `rope_spec`, head counts) from the **live vLLM weights during capture**. The large unembedding `W_U` is **not** stored: the worker precomputes tiny `g_loss = dL/dh^N` while `W_U` is resident on-device (~10 MB artifact vs hundreds of MB with `W_U`).
4. **Scheduler** continues: after extended teacher prefill, **restore the prompt boundary** (rewind computed-token cursor), then **decode** the user request. KV at prompt positions reflects **unmitigated** embeddings for this pass (no soft removal yet).

**Analyzer (after capture):**

1. Load `highlighter_activations.pt` + `weight_bundle`.
2. Run `compute_grad_influences` (closed-form; default **CPU**).
3. Write `highlighter.pt` with `token_scores`, `soft_indices`, and metadata (`soft_beta`, `applied_mode`, …).

No full-model forward at analyze time when `weight_bundle` is present.

#### `forward_attr` scoring

Approximate `‖∂L_aff/∂x_i‖` by back-propagating through the **last decoder block's attention sub-block only** (see fidelity note below):

1. `g_j = W_U^T(p_j - e_{y_j})` on LM-head input (`norm(h_L)` when final norm exists).
2. Optional **final norm Jacobian** (`rmsnorm` / `layernorm` / `gemma_rms`) if found.
3. **Value + key paths** → per-prompt-token gradient vector. The value path holds α fixed (∂/∂v_i); the key path carries the **full softmax Jacobian** `δ_ji = α_ji(l_ji − l̄_j)` (α is **not** frozen). **Q/K/V captures are already post-RoPE** (the model rotated them), but `W_K` / `W_V` are **pre-RoPE** projections, so the scorer applies **inverse RoPE** (`R_i^T` from `rope_spec`) to each gradient before `@ W_K` / `@ W_V` — gated per stream by the capture-time runtime probe (K rotated, V usually not). Finally an input-norm VJP maps back onto the residual-stream input `h_input`.
4. Gradient score for prompt token `x_i` = **L2 norm** of that vector (which encodes its influence on the affirmation loss).

**Fidelity vs full one-block autograd.** Given `g_j`, the attention-sub-block backprop above is *exact* (value + key paths, input/final-norm VJPs, GQA, `R_iᵀ`). The one systematic approximation is the **last-block MLP/FFN**: the gradient entering the attention output should be `(I + J_MLP_j)·g_j` but we use `g_j`. Because the MLP is position-wise it adds **no cross-position path** — it only rescales the per-generation-position upstream gradient. The boundary token (`prompt_len−1`, also the first generation position) gets its exact residual-skip identity term `+g_0` restored. Out-of-scope caveats (would change recomputed α): custom attention scaling, logit soft-capping, sliding-window masks (e.g. Gemma2).

See `utils/TokenHighlighter/grad_influence.py` for a formal derivation. Offline validation: `examples/compare_token_highlighter_scorers.py` (not part of the deployed pipeline).

#### Validation

`examples/compare_token_highlighter_scorers.py` compares `forward_attr` (vLLM pipeline) against a standalone HF reference. Two references are useful:

1. **Last-block autograd** (`dL/dh^(L-1)`) — the quantity `forward_attr` directly approximates: ρ≈0.93, driver Jaccard≈0.83 (Qwen2-1.5B, 4 prompts).
2. **Full embedding autograd** (`||dL/d(prompt embedding)||`) — the gold-standard influence metric: driver Jaccard≈**0.38**, Spearman ρ≈0.49 (same setup; saved in `examples/results/embedding_vs_forward_attr.json`).

Exact per-token ranking vs full embedding backprop is weaker, but **driver sets still overlap partially** — enough for mitigation in many cases, especially on jailbreak-style prompts where late-layer salience dominates. Regenerate metrics per model/prompt set before production deployment.

**Efficiency.** `forward_attr` reuses the deployment model's prefill forward (hooks) and runs only closed-form matrix ops at analyze time — no second model, no backward pass, ~10 MB activations artifact.

#### Capture modes (`highlighter.capture`)


| Mode                 | Forwards                                              | When                                                                                            |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `extended` (default) | 1 ideal                                               | `len(prompt)+len(target) ≤ max_num_batched_tokens` in one prefill chunk                         |
| `suffix`             | 2+ (`ceil(len(prompt) / max_num_batched_tokens) + 1`) | Chunked prefill or incomplete extended capture; merges real-prompt activations + teacher suffix |


Suffix fallback emits a **banner** + `UserWarning`. Prefer increasing `max_num_batched_tokens` so extended capture fits in one step.

### Mitigate (`highlighter_mode="mitigate"`)

**Purpose:** Regenerate the **same prompt** with driver embeddings scaled by **β < 1**, so prefill writes **mitigated K/V** into the paged cache; decode reads that cache.

**Prerequisite:** `highlighter.pt` from capture + analyze. Pass the same `run_id` as capture (`run_id=` / `scores_run_id` in `extra_args`, or demo `capture_run_id`).

**Worker** (same `HighlighterWorker`):

1. `_load_scores()` — read `highlighter.pt` for the run; map `prompt_ids` → `token_scores`, `soft_indices` (drivers).
2. `_queue_soft_embeddings()` — for each new mitigate request, scale embedding rows for driver indices by **β** (from `highlighter.beta` in config).
3. **Embedding hook** — before layers run on prefill, swapped rows use softened embeddings.
4. `super().execute_model()` — normal vLLM **prefill** (mitigated prompt KV) + **decode**.

Mitigate does **not** re-run forward_attr hooks for scoring; it only consumes stored drivers. A full prefill overwrites prompt-slot KV with the mitigated representation.

```
[Worker]     load highlighter.pt  →  soft_indices + β
[Worker]     queue β · embed(driver tokens)
[Scheduler]  prefill  →  attention stores mitigated K/V in cache
[Scheduler]  decode   →  completion from mitigated cache
```

### Request API

```python
llm.generate(prompt, highlighter_mode="capture", ...)   # trace → highlighter_activations.pt
llm.analyze(analyzer_spec={...})                       # scores → highlighter.pt
llm.generate(prompt, highlighter_mode="mitigate", run_id=capture_run_id, ...)  # soft prefill + decode
```

`HookLLM` passes `highlighter_mode`, `run_id`, and the `highlighter` config block via `SamplingParams.extra_args` (see `examples/demo_token_highlighter.py`).

---

## Configuration

Per-model defaults live in `model_configs/token_highlighter/<model_short>.json`. `HookLLM.load_config` reads the `highlighter` object and passes it on every request as `extra_args["highlighter"]` (same pattern as `steer` for steering).

### Full example config file

Copy to `model_configs/token_highlighter/<your-model>.json` and trim fields you do not need — omitted keys use the defaults in the tables below.

```json
{
    "model_info": {
        "name": "Qwen/Qwen2-1.5B-Instruct"
    },
    "highlighter": {
        "target_phrase": "Sure! I can help with that",
        "target_token_ids": null,
        "capture": "extended",
        "mode": "top_percentage",
        "alpha": 0.25,
        "threshold_k": 2.0,
        "beta": 0.1,
        "require_attn_metadata": false,
        "allow_prerope_fallback": false,
        "reselect_drivers": false
    }
}
```

`temperature` sometimes appears in shipped configs (e.g. paper §4.1 demos) for **documentation only** — decoding temperature is set on `generate(..., temperature=...)`, not read by the worker.

### `highlighter` block (JSON → worker)

Read at `HookLLM` init; forwarded on every `generate()` via `extra_args["highlighter"]`.


| Field                    | Default                        | Explanation                                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `target_phrase`          | `"Sure! I can help with that"` | Affirmation string teacher-forced during capture (`prompt + target[:-1]`). Defines the affirmation loss the scorer attributes.                                                                                                                                     |
| `target_token_ids`       | *(unset)*                      | Integer list of token ids for the affirmation phrase. **Overrides** `target_phrase` when set. Usually computed after tokenizer load: `llm._highlighter_config["target_token_ids"] = tok.encode(phrase, add_special_tokens=False)`.                                 |
| `capture`                | `"extended"`                   | `"extended"`: one prefill on `prompt + target[:-1]` when the full sequence fits `max_num_batched_tokens`. `"suffix"`: multi-pass fallback (chunked prefill or explicit opt-in); worker prints a banner when suffix is required automatically.                      |
| `mode`                   | `"top_percentage"`             | How driver tokens are chosen from per-token influence scores: `"top_percentage"` (top α fraction) or `"mean_std"` (scores above `mean + k·std`). Used by the analyzer when writing `highlighter.pt` and optionally at mitigate time if `reselect_drivers` is true. |
| `alpha`                  | `0.25`                         | With `mode: "top_percentage"`, fraction of highest-scoring prompt tokens flagged as drivers (e.g. `0.25` → top 25%).                                                                                                                                               |
| `threshold_k`            | `2.0`                          | With `mode: "mean_std"`, flag tokens with score **>** `mean + k·std`. Larger `k` → fewer drivers. Also used when `reselect_drivers` recomputes drivers at mitigate time.                                                                                           |
| `beta`                   | `0.4`                          | During **mitigate** prefill, driver prompt embedding rows are scaled by this factor (`embedding *= beta`). `1.0` = no shrinking; smaller values = stronger suppression. Paper-style 7B runs often use `0.3`–`0.5`.                                                 |
| `require_attn_metadata`  | `false`                        | Applied when capture hooks install (first `highlighter_mode="capture"` request, after `extra_args['highlighter']` sync). When `true`, Q/K/V hooks fire only when vLLM attention forward metadata is present (stricter). Shipped model JSONs use `false` for broader vLLM compatibility. |
| `allow_prerope_fallback` | `false`                        | Applied with capture hook install. When `true`, allows pre-RoPE projection hooks if no inner `.attn` core is found. Degrades `forward_attr` fidelity; use only when post-RoPE hooks cannot be installed. |
| `reselect_drivers`       | `false`                        | At **mitigate** time, recompute driver indices from stored `token_scores` using `mode` / `alpha` / `threshold_k` instead of using analyzer-saved `soft_indices` in `highlighter.pt`.                                                                               |


### Per-request parameters (`generate()`)

Not stored in the JSON file; passed each call via `HookLLM.generate(...)` (wired into `SamplingParams.extra_args`).


| Parameter                      | Default                                                             | Explanation                                                                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `highlighter_mode`             | *(required)*                                                        | `"capture"`: install hooks, teacher prefill, write `highlighter_activations.pt`, then decode (unmitigated KV). `"mitigate"`: load `highlighter.pt`, apply β-scaled embeddings at prefill, decode from mitigated KV. |
| `run_id`                       | new UUID per capture; reuses `_last_run_id` for mitigate if omitted | Ties artifacts under `{hook_dir}/{run_id}/`. For mitigate, pass the capture run id so the worker loads the correct `highlighter.pt`.                                                                                |
| `scores_run_id`                | `run_id`                                                            | Run id that owns `highlighter.pt` when it differs from the mitigate request’s `run_id` (advanced; rarely needed).                                                                                                   |
| `temperature`, `max_tokens`, … | vLLM `SamplingParams` defaults                                      | Standard decoding knobs; do not affect scoring. Paper demos use `temperature=0.6`, `top_p=0.9` from config notes, not from the `highlighter` block.                                                                 |


`hook_dir` and `run_id_file` are set automatically by `HookLLM` from its constructor `hook_dir` / `download_dir`.

### `analyzer_spec` (`analyze()`)

Passed to `llm.analyze(analyzer_spec={...})`. `HookLLM` **merges** `mode`, `alpha`, `threshold_k`, and `beta` (as `soft_beta`) from the loaded `highlighter` JSON when those keys are omitted.


| Field                    | Default                               | Explanation                                                                                                                                         |
| ------------------------ | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `top_k`                  | `5`                                   | Number of highest-scoring tokens to include in formatted `top_tokens` output.                                                                       |
| `mode`                   | from JSON `highlighter.mode`          | Driver selection mode when writing / re-deriving `soft_indices` in `highlighter.pt`.                                                                |
| `alpha`                  | from JSON `highlighter.alpha`         | Top-fraction driver selection (see above).                                                                                                          |
| `threshold_k`            | from JSON `highlighter.threshold_k`   | Mean+std threshold (see above).                                                                                                                     |
| `soft_beta`              | from JSON `highlighter.beta`          | Recorded in `highlighter.pt` metadata (`soft_beta` field per sequence).                                                                             |
| `device`                 | `"cpu"`                               | Device for closed-form `forward_attr` math in the analyzer (`"cuda"` optional).                                                                     |
| `run_id`                 | `_last_run_id` from last `generate()` | Which capture run’s `highlighter_activations.pt` to score.                                                                                          |
| `run_id_file`            | `{hook_dir}/RUN_ID.txt`               | Fallback list of run ids when `run_id` is omitted.                                                                                                  |
| `write_scores`           | `true`                                | Write `highlighter.pt` after scoring. Set `false` to only return in-memory results.                                                                 |
| `artifact_wait_seconds`  | `2.0`                                 | Max wait for `highlighter_activations.pt` to appear on disk after capture.                                                                          |
| `artifact_poll_interval` | `0.05`                                | Poll interval while waiting for capture artifacts.                                                                                                  |
| `loss_grad_fn`           | `null`                                | Advanced: custom callable replacing default affirmation-loss gradient on `h_L`. Rarely needed when `g_loss` is precomputed in the capture artifact. |


Minimal analyze call when a JSON config is loaded:

```python
llm.analyze(analyzer_spec={"top_k": 5})
```

---

## Usage examples

### Minimal `HookLLM`

```python
hook_dir = "./cache/_v1_qk_peeks"
llm = HookLLM(
    model=model_path,
    worker_name="token_highlighter",
    analyzer_name="token_highlighter",
    hook_dir=hook_dir,
    enable_hook=True,
)

# 1) Capture — worker hooks + trace; decode is unmitigated
out_cap = llm.generate(
    prompt,
    highlighter_mode="capture",
    temperature=0.0,
    max_tokens=32,
)
capture_run_id = llm._last_run_id

# 2) Analyze — forward_attr: scores → highlighter.pt (no second GPU model if weight_bundle present)
stats = llm.analyze(analyzer_spec={"top_k": 5})

# 3) Mitigate — same worker: soft embeddings at prefill → mitigated KV → decode
out_mit = llm.generate(
    prompt,
    highlighter_mode="mitigate",
    run_id=capture_run_id,
    temperature=0.0,
    max_tokens=32,
)
```

### Demos


| Path                                           | Notes                                                                                     |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `examples/demo_token_highlighter.py`           | Script loop; plain `HookLLM` + JSON config via `extra_args`                               |
| `notebooks/demo_token_highlighter.ipynb`       | Local; smaller models (tested locally with NVIDIA RTX 5070, 8 GB VRAM)                    |
| `notebooks/demo_token_highlighter_colab.ipynb` | 7B / paper-style decoding (tested with NVIDIA RTX 3060, 12 GB VRAM with AWQ quantization) |


**VRAM:** `max_num_batched_tokens` ≈ 512, `gpu_memory_utilization` ≈ 0.8 with 8GB; call `llm_engine.engine_core.shutdown()` between large model swaps.

**New captures after code updates:** Re-run capture so `highlighter_activations.pt` includes `weight_bundle` (and `norm_kind`).

---

## Current support and limitations

### Supported (intended vLLM deployment)

**Decoder-only causal transformer LMs** with a standard pre-norm → blocks → (optional) final norm → `lm_head` layout:

- **Stack:** `model.layers` / `transformer.h` / `layers`
- **Attention:** `self_attn` / `attn` / `attention`; vLLM fused `qkv_proj` or HF-style separate Q/K/V
- **LM head:** `lm_head` or `get_output_embeddings()`
- **GQA:** `num_key_value_heads` in config
- **Final norm:** `locate_final_norm()` — Llama/Qwen/Mistral `model.norm`, GPT-2 `ln_f`, OPT/NeoX paths, etc.
- **Norm math:** `rmsnorm`, `layernorm`, `gemma_rms`

Typical checkpoints: LLaMA, Mistral, Qwen, Vicuna, Phi, GPT-2, OPT-shaped, Gemma-family.

**Operational assumptions:**

- `h_L` hook = last **block** output, yet to be fed through **global** final norm (if applied at all) before `lm_head` (Llama-class).
- `tensor_parallel_size=1` in demos (artifacts on `tp_rank_0`).
- Extended capture fits in `max_num_batched_tokens` or suffix merge succeeds.

### Limitations


| Area                      | Limitation                                                                                                                                                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Coverage**              | Not every model vLLM can load — only standard transformer decoders above                                                                                                                                                                                                             |
| **Architecture**          | No encoder–decoder, Mamba/RWKV-only, MLA/custom attention without compatible Q/K/V                                                                                                                                                                                                   |
| **Discovery**             | Unknown layer/norm/head paths → errors or silent wrong scores (e.g. missing norm → raw `h_L` for `g_j`)                                                                                                                                                                              |
| **Norms**                 | Only three analytic kinds (RMSNorm, LayerNorm, and Gemma-RMS); custom norms need new `norm_kind`                                                                                                                                                                                     |
| **Tensor parallel**       | No merged `weight_bundle` across ranks                                                                                                                                                                                                                                               |
| **Quantization**          | Must expose readable weights on hooked modules; not all quant formats tested                                                                                                                                                                                                         |
| **forward_attr** accuracy | Last-block attribution: exact attention-sub-block VJP (α **not** frozen), but the last block's **MLP/FFN Jacobian is omitted** (`g_j` used in place of `(I+J_MLP_j)·g_j`) and only the last block is traced; relies on the residual identity `dh^{L-1}/dx ≈ I` — validate per family |
| **Mitigation**            | Task-specific (affirmation phrase + β/α); not a general safety filter                                                                                                                                                                                                                |


---

## Roadmap and future directions

- Colab / paper-scale demo hardening (7B, paper `α`, `β`, temperature)
- Multi-prompt deployability report (`examples/compare_token_highlighter_scorers.py` + driver overlap)
- Score at `_finish_capture` in-worker (optional: skip separate analyze pass for `forward_attr`)
- Broader `locate_final_norm` / block discovery for new vLLM model classes
- TP-aware artifact merge for `weight_bundle`
- Optional hook at LM-head input for tighter alignment with `g_j` semantics

