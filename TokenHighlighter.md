# Token Highlighter Documentation

Original Paper: [Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for LLMs](https://arxiv.org/pdf/2412.18171).

---

## Goals, Architecture, and Design Intent

### Goal

Identify prompt tokens that most increase the likelihood of a fixed **affirmation target phrase** (affirmation loss), then optionally **mitigate** by scaling those driver-token embeddings by `β < 1` during a later prefill so generation runs against an altered KV cache.

High-level algorithm:

1. **Affirmation loss** (teacher-forced): `L_aff(X) = -Σ_i log P(y_i | X, y_<i)`
2. **Per-token influence scores** (scorer-dependent with our approximation as the default).
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


| Component               | Role                                                                                                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HookLLM**             | One GPU engine; passes `highlighter_mode`, `run_id`, `hook_dir` via `SamplingParams.extra_args`.                                                                       |
| **HighlighterWorker**   | **Capture:** hooks, traces, optional in-worker autograd scores. **Mitigate:** load `highlighter.pt`, queue β-scaled prompt embeddings, run scheduler prefill + decode. |
| **HighlighterAnalyzer** | **forward_attr:** analytic scores from `highlighter_activations.pt` + `weight_bundle` (no `from_pretrained`). **Autograd traces:** read existing `highlighter.pt`.     |


Registry (`vllm_hook_plugins/__init__.py`): worker and analyzer name `token_highlighter`.

There is **no second vLLM engine** and **no separate capture vs mitigate worker class**. Mitigation is a **later** `generate(..., highlighter_mode="mitigate")` on the same `HookLLM` instance, reusing the capture `run_id` so the worker loads the same `highlighter.pt`.

### Summary


| Phase        | Who                       | What                                                                                                    |
| ------------ | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Capture**  | Worker + scheduler        | Teacher prefill (for scoring), hooks, trace I/O; then **decode** (typically unmitigated KV).            |
| **Analyze**  | Analyzer (driver process) | Turn activations into `token_scores` + `soft_indices` in `highlighter.pt`.                              |
| **Mitigate** | Worker + scheduler        | Load scores → **prefill with softened prompt embeddings** → K/V cache reflects mitigation → **decode**. |


Default production scorer: `forward_attr` (vLLM hooks + analytic math). `autograd` is a reference path (extra HF model, scores in-worker).

### Design intent

- **Worker** owns everything that must touch the live vLLM model: hooks, capture I/O, embedding soft-removal at mitigate prefill, KV effects via normal attention after the embedding hook.
- **Analyzer** owns offline math on traces so analyze does not load a second full model when `weight_bundle` is present.
- One capture `run_id` can be analyzed many times (different `analyzer_spec`) without re-running vLLM.
- Implementation lives in `workers/highlighter_worker.py`, `analyzers/highlighter_analyzer.py`, `utils/TokenHighlighter/`*.

---

## End-to-End Pipeline

### Overview (typical `forward_attr` deployment)

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

1. **Install hooks** (once per engine via `collective_rpc("install_hooks")`) on the last decoder block: **Q, K, V** (post-RoPE, captured at the inner `.attn` core input), the block input `h_input`, and the block output `h_L`. A runtime RoPE probe also compares pre-projection vs post-`.attn` Q/K/V to record which streams are rotated (if any at all).
2. **Teacher prefill** on `prompt + target[:-1]` (`extended` or `suffix` capture mode) so activations at both prompt and generation positions exist in the trace.
3. **Persist traces:**
  - **forward_attr:** `highlighter_activations.pt` (per-sequence `capture` tensors) + `weight_bundle` (`W_U`, `W_O`, `W_Q`, `W_V`, `W_K`, `final_norm`, `input_norm`, `rope_spec`, head counts) from the **live vLLM weights during capture**.
  - **autograd:** `highlighter.pt` written in-worker (HF backward on prompt embeddings **before** the main vLLM forward); no Q/K/V analyze path required.
4. **Scheduler** continues: restore prompt boundary after extended teacher prefill, then **prefill + decode** for the user request. KV at prompt positions reflects **unmitigated** embeddings for this pass (no soft removal yet).

**Analyzer (after capture, `forward_attr` only):**

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

See `utils/TokenHighlighter/grad_influence.py` for a formal derivation. Validate vs `autograd`: `examples/validate_scorer_agreement.py`, `examples/verify_last_attn_grad.py`.

#### Capture modes (`VLLM_HIGHLIGHTER_CAPTURE`)


| Mode                 | Forwards                                              | When                                                                                            |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `extended` (default) | 1 ideal                                               | `len(prompt)+len(target) ≤ max_num_batched_tokens` in one prefill chunk                         |
| `suffix`             | 2+ (`ceil(len(prompt) / max_num_batched_tokens) + 1`) | Chunked prefill or incomplete extended capture; merges real-prompt activations + teacher suffix |


Suffix fallback emits a **banner** + `UserWarning`. Prefer increasing `max_num_batched_tokens` so extended capture fits in one step.

#### `autograd` capture (alternate)

```
[Worker]  HF backward on prompt embeddings → highlighter.pt
[Scheduler]  normal vLLM prefill + decode (no forward_attr activations for scoring)
```

Uses `VLLM_HIGHLIGHTER_GRAD_DEVICE` (default `cpu`) for the HF scorer copy. Heavier; use for reference / validation.

### Mitigate (`highlighter_mode="mitigate"`)

**Purpose:** Regenerate the **same prompt** with driver embeddings scaled by **β < 1**, so prefill writes **mitigated K/V** into the paged cache; decode reads that cache.

**Prerequisite:** `highlighter.pt` from capture + analyze (or autograd capture). Pass the same `run_id` as capture (`run_id=` / `scores_run_id` in `extra_args`, or demo `capture_run_id`).

**Worker** (same `HighlighterWorker`):

1. `_load_scores()` — read `highlighter.pt` for the run; map `prompt_ids` → `token_scores`, `soft_indices` (drivers).
2. `_queue_soft_embeddings()` — for each new mitigate request, scale embedding rows for driver indices by **β** (from config / env).
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
llm.generate(prompt, highlighter_mode="capture", ...)   # trace + (forward_attr) later analyze
llm.analyze(analyzer_spec={...})                       # forward_attr: writes highlighter.pt
llm.generate(prompt, highlighter_mode="mitigate", run_id=capture_run_id, ...)  # soft prefill + decode
```

`HookLLM` / demo wrappers set `extra_args['highlighter_mode']` and `run_id` on `SamplingParams` (see `examples/demo_token_highlighter.py`).

---

## Environment configuration


| Variable                                  | Default          | Meaning                                                                            |
| ----------------------------------------- | ---------------- | ---------------------------------------------------------------------------------- |
| `VLLM_HIGHLIGHTER_TARGET_PHRASE`          | (model JSON)     | Affirmation string to teacher-force (target)                                       |
| `VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS`       | —                | Comma-separated token ids (overrides phrase; set before engine start)              |
| `VLLM_HIGHLIGHTER_SCORER`                 | `forward_attr`   | Scorer: `forward_attr` (analytic) or `autograd` (HF reference)                     |
| `VLLM_HIGHLIGHTER_CAPTURE`                | `extended`       | Capture mode: `extended` (1 forward) or `suffix` (forced 2-pass)                   |
| `VLLM_HIGHLIGHTER_MODE`                   | `top_percentage` | Driver selection: `top_percentage` or `mean_std`                                   |
| `VLLM_HIGHLIGHTER_THRESHOLD_K`            | `2.0`            | `mean_std`: flag scores above `mean + k·std`                                       |
| `VLLM_HIGHLIGHTER_ALPHA`                  | `0.25`           | `top_percentage`: top fraction of tokens flagged                                   |
| `VLLM_HIGHLIGHTER_BETA`                   | `0.1`            | Embedding scale on **driver** rows during **mitigate** prefill (`1.0` = no shrink) |
| `VLLM_HIGHLIGHTER_MITIGATE`               | `1`              | Default mode if `highlighter_mode` omitted in env-only setups                      |
| `VLLM_HIGHLIGHTER_GRAD_DEVICE`            | `cpu`            | Device for the HF model in `autograd` scoring                                      |
| `VLLM_HIGHLIGHTER_ALLOW_PREROPE_FALLBACK` | `0`              | Allow pre-RoPE proj hooks when no inner `.attn` core (approximation degrades)      |


Per-model defaults: `model_configs/token_highlighter/<model_short>.json`.

Analyzer-only (in `analyzer_spec`): `top_k`, `mode`, `alpha`, `threshold_k`, `device` (`cpu` default when using `weight_bundle`).

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


| Path                                           | Notes                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `examples/demo_token_highlighter.py`           | Script loop; `HighlighterDemoLLM` sets env + `extra_args`              |
| `notebooks/demo_token_highlighter.ipynb`       | Local; smaller models (tested locally with NVIDIA RTX 5070, 8 GB VRAM) |
| `notebooks/demo_token_highlighter_colab.ipynb` | 7B / paper-style decoding (tested with NVIDIA RTX 3060, 12 GB VRAM)    |


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
- Multi-prompt deployability report (`validate_scorer_agreement.py` + driver overlap)
- Score at `_finish_capture` in-worker (optional: skip separate analyze pass for `forward_attr`)
- Broader `locate_final_norm` / block discovery for new vLLM model classes
- TP-aware artifact merge for `weight_bundle`
- Optional hook at LM-head input for tighter alignment with `g_j` semantics

