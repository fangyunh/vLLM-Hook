# CUDA-Graph Capture — **Hybrid Mode** (v0.3.0)

*What the `graph/` capture path is, why it works, and how to run its tests.
This is the short intro; the full design + evidence live in
`docs/v0.3.0_qk_cuda_graph_report.md` and `docs/cuda_graph_plan.html`.*

> **Naming.** This method was called **"op mode"** during development. It is now
> **Hybrid mode** — the name reflects what it actually is: a single forward that
> **interleaves replayed CUDA-graph segments with eager seams**, rather than being
> all-graph or all-eager.

---

## 1. The one-paragraph version

vLLM-Hook taps a running model to capture attention **Q/K** and **hidden states**.
In **v0.2.0** this only works with CUDA graphs **off** (`enforce_eager=True`),
because capture rode on Python `register_forward_hook` callbacks — and a CUDA graph
replay skips Python entirely. **Hybrid mode (v0.3.0)** makes capture work with CUDA
graphs **on**: it expresses capture as a **custom operator** that the compiler
records as a node, and registers that op as a vLLM **splitting op** so the piecewise
compiler keeps it as an **eager seam between graph segments** — the same treatment
vLLM already gives attention. Capture then fires on every real forward, writes the
*same* data structures v0.2.0 produced (the **reuse contract** — nothing downstream
changes), and is **numerically parity-proven** against the eager path for both QK
and hidden states.

---

## 2. Why v0.2.0 was eager-only

When `enforce_eager=False`, vLLM runs the model under `torch.compile` and records
the hot path into a CUDA graph it then **replays**. That breaks forward hooks three
independent ways:

| # | Blocker | Consequence |
|---|---------|-------------|
| 1 | **Graph replay skips Python.** The graph is recorded GPU ops; replay never re-enters `forward()`. | The hook **never fires** during fast decode. |
| 2 | **Hooks installed too late.** v0.2.0 installs lazily, per-request, *after* compile/capture. | The captured graph predates the hook. |
| 3 | **Hook bodies are graph-illegal.** `.item()` host syncs, fresh allocations, dict lookups, per-request branching. | None of that is legal inside a captured region. |

So v0.2.0 takes the safe route and forces `enforce_eager=True` whenever hooks are
active — correct, but it forfeits vLLM's fastest decode path.

---

## 3. The core idea — join the graph on attention's terms

Three architectural moves turn "a Python callback the graph skips" into "a
first-class operation the graph knows about":

```
   v0.2.0 (eager)                          Hybrid mode (v0.3.0)
   ──────────────                          ────────────────────
   register_forward_hook    ──────────►    a custom OP (vllm_hook::qk_probe / hs_probe)
   (invisible to compiler)                 (an opaque node the compiler records)

   installed per-request    ──────────►    installed at load_model
   AFTER compile                           BEFORE compile / capture

   runs as Python in forward ─────────►    runs as an EAGER SEAM between
   (skipped on replay)                     graph segments (runs every step)
```

1. **Custom op instead of a hook.** Registered via `direct_register_custom_op` with
   `mutates_args=["sink"]`, the op is *opaque* to Dynamo — recorded as one
   side-effecting node, never dead-code-eliminated. A forward hook fundamentally
   lacks this property.
2. **Install before compile.** Installed at the worker's `load_model` (model built,
   not yet compiled), so the op exists when the compiler traces the model.
3. **Decide per-request as data, not Python branches in the graph.** All per-request
   logic (which layers, prefill vs decode, token range) runs *inside the eager seam*,
   where request state is live — the same vantage point the old hook had.

---

## 4. The decisive mechanism — the eager seam (splitting op)

This is the single fix that flipped the result from "captures nothing" to "captures
correctly," and it is why the method is *hybrid*.

Under **PIECEWISE** mode, vLLM does not capture the whole model as one graph. It
**splits the computation graph at a set of "splitting ops"** — primarily attention,
which must run eagerly (paged attention can't live in a static graph). The result
alternates:

```
   [ cudagraph segment ] → (attention: eager seam) → [ cudagraph segment ] → (attention) → ...
        replayed                 runs every step          replayed              runs every step
```

Our capture op is called from a class-level wrap on the attention/decoder-layer
`forward`. **Unless told otherwise, the compiler folds it into the neighbouring
cudagraph segment** — and then its body is skipped on replay, exactly like the old
hook (observed on GPU: the op fired at warmup, then **never** during real
generation). The fix is one declaration — **register the op as a splitting op**:

```python
# tell the piecewise compiler "treat our op like attention"
compilation_config.splitting_ops.append("vllm_hook::qk_probe")
```

Now the compiler makes the op a **seam**, not segment-interior code:

```
   [ segment ] → (qk_probe: EAGER SEAM, runs every step) → (attention) → [ segment ] → ...
                       ▲
                       └── capture body runs here, with live request state,
                           on every prefill/decode step — never skipped on replay.
```

We don't fight the graph; we **join it on the same terms attention does**. That mix
of replayed segments and eager seams in one forward is the "hybrid."

---

## 5. The reuse contract — why nothing downstream changed

The capture body (the relocated v0.2.0 logic: per-request slicing, layer filtering,
prefill/decode gating, prefix-K reconstruction) writes into the **exact same worker
dictionaries** the eager path used:

```
bucket[req_id][module_name] = {"q": [...], "k_all": [...], "layer_num": L, "hookq_mode": mode}
```

Because egress (`get_captured_states`, `flush_disk`, the safetensors writer) and the
analyzers read those same structures, **they did not change at all.** Hybrid mode is
a new *producer* behind an unchanged interface — the eager path remains untouched as
the fallback. All new code is isolated under `vllm_hook_plugins/vllm_hook_plugins/graph/`:

- `graph/ops.py` — registers `vllm_hook::qk_probe` / `vllm_hook::hs_probe` (dedicated
  `vllm_hook` library, kept alive; `mutates_args=["sink"]`; CUDA dispatch).
- `graph/install.py` (QK) / `graph/install_hs.py` (HS) — monkey-patch
  `Worker.load_model` to register the op, class-wrap the target `forward`, and add the
  op to `splitting_ops`. Class-level wrap because the compiler bypasses instance hooks.
- `_hook_plugin.py` — only arms this path when **`VLLM_HOOK_ALLOW_CUDAGRAPH=1`**;
  otherwise v0.2.0 behaviour is byte-for-byte preserved.

---

## 6. Status & scope

**Proven on GPU** (Qwen2-1.5B, vLLM v0.21.0, single GPU, TP=1), all parity-checked
graph-vs-eager: **QK and hidden states** · PIECEWISE mode · `last_token` and
`all_tokens` · all layers · prefill capture · fresh prompts · prefix caching (QK).
The egress/analyzers are unchanged via the reuse contract.

**Deferred** (not load-bearing for the current prefill-phase use cases): decode-phase
capture (`hooks_on=both`), `FULL_DECODE_ONLY` mode, multi-request batching, multi-GPU
(TP>1), and the **steering** worker under graph mode — steering must *mutate* the
residual inside the graph, which has no eager seam to ride and needs the deferred
"buffer mode" data plane (see the plan, §8/§9).

---

## 7. Running the tests (LSF)

These run on the GPU cluster via `bsub`, **not** pytest (see the plan's §11 runbook
for the full workflow, prerequisites, and the Step-0 feasibility gates).

```bash
# Hybrid-mode parity oracles (graph-vs-eager; QK + HS both PASS):
bsub < tests/cuda_graph/tests/qk_graph/run_qk_parity.sh
bsub < tests/cuda_graph/tests/hs_graph/run_hs_parity.sh
bjobs ; bpeek <JOBID>
grep VERDICT tests/cuda_graph/logs/qk_parity.<JOBID>.out   # PASS = all tensors within tol
```

Each oracle runs `capture --mode graph` → `capture --mode eager` → `compare` as
**three sequential subprocesses in one job** (the cluster GPU is `exclusive_process`,
so one process can't boot two engines). Variants: `_alltok` (all layers / all_tokens)
and, for QK, `_prefix` (prefix-K reconstruction).

---

*Companion docs: `docs/v0.3.0_qk_cuda_graph_report.md` (architecture walkthrough +
evidence), `docs/cuda_graph_plan.html` (full design, claims, and LSF runbook).*
