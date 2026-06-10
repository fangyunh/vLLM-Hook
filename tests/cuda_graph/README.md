# CUDA-Graph Capture — **Hybrid Mode**

## 1. Introduction

vLLM-Hook taps a running model to capture attention **Q/K** and **hidden states**.
In **v0.2.0** this only works with CUDA graphs **off** (`enforce_eager=True`),
because capture on Python `register_forward_hook` callbacks and a CUDA graph
replay skips Python entirely. **Hybrid mode** makes capture work with CUDA
graphs **on**: it expresses capture as a **custom operator** that the compiler
records as a node, and registers that op as a vLLM **splitting op** so the piecewise
compiler keeps it as an **eager section between graph segments**, which is the same treatment
vLLM already gives attention. Capture then fires on every real forward, writes the
*same* data structures v0.2.0 produced.

---

## 2. Motivation

V0.2.0 forcely set `enforce_eager=true` because when `enforce_eager=False`, vLLM runs the model under `torch.compile` and records
the hot path into a CUDA graph it then **replays**. That breaks forward hooks three
independent ways:

1. **Graph replay skips Python.**: The graph is recorded GPU ops; replay never re-enters `forward()`. The hook **never fires** during fast decode.
2. **Hooks installed too late.**: v0.2.0 installs lazily, per-request, *after* compile/capture. The captured graph predates the hook.
3. **Hook bodies are graph-illegal.**: `.item()` host syncs, fresh allocations, dict lookups, per-request branching. None of that is legal inside a captured region.

---

## 3. Hybrid Mode

Three architectural moves turn "a Python callback the graph skips" into "a
first-class operation the graph knows about":

```
   v0.2.0 (eager)                          Hybrid mode (v0.3.0)
   ──────────────                          ────────────────────
   register_forward_hook    ──────────►    a custom OP (vllm_hook::qk_probe / hs_probe)
   (invisible to compiler)                 (an opaque node the compiler records)

   installed per-request    ──────────►    installed at load_model
   AFTER compile                           BEFORE compile / capture,

   runs as Python in forward ─────────►    runs as an EAGER section between
   (skipped on replay)                     graph segments (runs every step)
```

1. **Custom op instead of a hook.** Registered via `direct_register_custom_op` with
   `mutates_args=["sink"]`, the op recorded as one side-effecting node, never dead-code-eliminated. 
   A forward hook fundamentally lacks this property.
2. **Install before compile.** Installed at the worker's `load_model`, so the op exists when the compiler traces the model.
3. **Decide per-request as data, not Python branches in the graph.** All per-request
   logic (which layers, prefill vs decode, token range) runs *inside the eager section*,
   where request state is live.

---

## 4. Mechanism

Under **PIECEWISE** mode, vLLM does not capture the whole model as one graph. It
**splits the computation graph at a set of "splitting ops"** — primarily attention,
which must run eagerly (paged attention can't live in a static graph). The result
alternates:

```
   [ cudagraph segment ] → (attention: eager section) → [ cudagraph segment ] → (attention) → ...
        replayed                 runs every step          replayed              runs every step
```

Our capture op is called from a class-level wrap on the attention/decoder-layer
`forward`.

```python
# tell the piecewise compiler "treat our op like attention"
compilation_config.splitting_ops.append("vllm_hook::qk_probe")
```

Now the compiler makes the op a **eager section**, not segment-interior code:

```
  One layer:

   [ graph segment ] → (qk_probe: EAGER section, runs every step) → (attention) → [ graph segment ] → ...
                       ▲
                       └── capture body runs here, with live request state,
                           on every prefill/decode step, never skipped on replay.
```

---

As same as the v0.2.0, the capture body, including the per-request slicing, layer filtering,
prefill/decode gating, prefix-K reconstruction, writes into the **exact same worker
dictionaries** the eager path used:

```
bucket[req_id][module_name] = {"q": [...], "k_all": [...], "layer_num": L, "hookq_mode": mode}
```

Because egress (`get_captured_states`, `flush_disk`, the safetensors writer) and the
analyzers read those same structures, they did not change at all. Hybrid mode is
a new *producer* behind an unchanged interface: the eager path remains untouched as
the fallback. All new code is isolated under `vllm_hook_plugins/vllm_hook_plugins/graph/`:

- `graph/ops.py` — registers `vllm_hook::qk_probe` / `vllm_hook::hs_probe` (dedicated
  `vllm_hook` library, kept alive; `mutates_args=["sink"]`; CUDA dispatch).
- `graph/install.py` (QK) / `graph/install_hs.py` (HS) — monkey-patch
  `Worker.load_model` to register the op, class-wrap the target `forward`, and add the
  op to `splitting_ops`. Class-level wrap because the compiler bypasses instance hooks.
- `_hook_plugin.py` — only arms this path when **`VLLM_HOOK_ALLOW_CUDAGRAPH=1`**;
  otherwise v0.2.0 behaviour is byte-for-byte preserved.


---


