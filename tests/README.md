# MIA's tests

Model compatibility tests for the `mia` package. They validate that hooks, workers and
analyzers work correctly with vLLM models.

## Layout

```
tests/
├── conftest.py     shared fixtures, the `gpu` marker and `requires_gpu`
├── use_cases/      one test per use case — these boot a real engine
├── test_plugin_config.py     engine-config policy: V2 runner, cudagraph mode, no PP
├── test_runner_adapter.py    the V2 model-runner adapter contract
├── test_graph_routing.py     per-step routing under CUDA graphs
├── test_tp_install.py        what each rank installs at tensor_parallel_size > 1
└── test_client_requests.py   the wire format MiaClient puts on the request
```

`use_cases/` is where a new worker or analyzer belongs — add a test there alongside your
demo. The five modules beside it are deliberately few: they cover the parts of the 0.29 /
V2 port that have no use case of their own, and that would fail silently rather than
loudly if they regressed.

---
## Run Tests

Install pytest first (`pip install pytest`), and run from the project root.

### The hermetic gate (no GPU needed)

Tests that boot a real engine carry the `gpu` marker. To run everything else — the gate
CI and code review use — select on the **marker**:

```bash
pytest tests -q -m "not gpu"      # 104 passed, 11 deselected
```

Use `-m`, **never `-k "not gpu"`**: `-k` filters test names, so it lets the engine tests through
and drops CPU tests whose names merely contain "gpu".

### The GPU tests

```bash
pytest tests/use_cases -m gpu
```

- They boot one engine at a time (each test shuts its engine down) on small models: opt-125m,
  gpt2, Qwen2-1.5B, Phi-3-mini and Mistral-7B (gated: `hf auth login`), downloaded on first use.
- Keep them in their own pytest run, apart from the gate: on a GPU in exclusive-process mode
  (common on clusters) a second process cannot open the device while another holds it, and the
  engine fails with `CUDA-capable device(s) is/are busy or unavailable`.
- They use `gpu_memory_utilization` 0.2–0.5 of the card, and write `hs_aperture_dump/` /
  `qk_aperture_dump/` in the working directory.
- Models without a shipped config get a random test config in pytest's temporary directory.

Run only the attention tracker tests, or one model:

```bash
pytest tests/use_cases/test_attntracker.py -m gpu -vv
pytest "tests/use_cases/test_attntracker.py::test_attention_tracker[gpt2]" -vv
```

---

## Common Failures

- **Nothing captured** (`probes` is `None`, analyzer output empty): the engine log prints
  `no decoder layers matched` / `no attention modules matched` when the model's module names are
  not ones MIA knows ([supported models](../docs/configs.md#supported-models)); in eager mode it
  prints `Installed 0 ... hooks`.
- **`CUDA-capable device(s) is/are busy or unavailable`**: another process holds the GPU; see
  above.
- **`Cannot find any model weights`**: the model was not downloaded (offline, or a gated model
  without `hf auth login`).
