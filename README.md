# 🪝 vLLM-Hook MIA
*A modular plugin library for vLLM.*

📄 [Preprint] [**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](https://arxiv.org/abs/2603.06588v1)

MIA is a plugin library designed to let developers and researchers **inspect**, **analyze**, and **steer** the internal operations of large language models running under the **vLLM** inference engine.  

This includes dynamic analysis of:  
- attention patterns  
- attention heads  
- activations  
- custom intervention behaviors  

---

## 📰 News & Events

- **July 24, 2026** — Featured in IBM Think: [*A new way of debugging open-weight models*](https://www.ibm.com/think/news/new-way-debugging-open-weight-models).

- **July 6, 2026** — Presented at ICML 2026: [*vLLM-Hook: Live Programming of Model Internals on vLLM*](https://icml.cc/virtual/2026/75729). (ICML registration and login are required to view the presentation.)

---

## 🚀 Features

- **Model-agnostic plugin system** for vLLM engines  
- **Extensible worker/analyzer abstraction**  
  - Easy to define new hooks, analyzers, and behaviors  
- **Introspection** of model internals  
- **Interventions** (activation steering, attention control, etc.)  
- **FULL CUDA-graph support** — capture and steering stay graph-safe, no fallback to eager
  ([one measured caveat on `logprobs`](docs/configs.md))  
- **Example applications**:  
  - Safety guardrails  
  - Reranking  
  - Enhanced instruction following  

---

## 📊 Performance

Capture and steering cost a few percent of decode throughput and keep the engine's CUDA graphs.
Measured on vLLM 0.29.0 + the V2 runner, server path, K=3 median, against the native vLLM tier of
the same launch (TPOT overhead):

| req/s | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Hidden states | +4.9 % | +6.1 % | +6.4 % | +5.8 % |
| Q/K | +6.7 % | +8.3 % | +8.4 % | +7.9 % |
| Steering | +3.9 % | +4.1 % | +3.7 % | +5.6 % |

Capture saturates at rate 32; steering holds to 64 (+5.2 %).

---

## 🧩 Supported Configurations

MIA targets the **server path** (`vllm serve` + `MiaClient`) under FULL CUDA graphs. Each use case
(attention tracker, activation steering, hidden-state extraction, …) runs across a Cartesian
product of storage (`rpc` / `disk`) and disk format (`pt` / `safetensors`). See
[`docs/configs.md`](docs/configs.md) for code snippets showing how to select each config.

`MiaLLM` builds an engine in your own process — the quickest way to exercise graph-mode capture
on one machine, and what [`examples/demo_capture_aperture.py`](examples/demo_capture_aperture.py)
uses.

MIA requires vLLM's **V2 model runner** and accepts `cudagraph_mode` `FULL` or `NONE`; it refuses
anything else at engine-config time rather than installing and silently capturing nothing.

Tensor parallelism (TP > 1) is supported for `capture_hs`, `capture_qk` and `steer`: each capturing
rank writes its own `tp_rank_<r>/` directory and MIA's loaders merge them. Pipeline parallelism is
not supported, and Q/K `score` capture requires TP = 1.

---

## 📦 Installation

**Requirements:** Linux, an NVIDIA GPU with a CUDA 13 driver, Python 3.12. Gated models (Llama, Mistral) need `hf auth login`.

### 1. Clone the repository

```bash
git clone https://github.com/IBM/vLLM-Hook.git
cd ./vLLM-Hook
```

### 2. Create an environment and install

MIA is validated on **vLLM 0.29.0 with torch 2.13.0**:

```bash
conda create -n mia_v029 python=3.12 pip
conda activate mia_v029
pip install vllm==0.29.0          # also installs torch 2.13.0
pip uninstall -y torchcodec       # vLLM's audio/video decoder; MIA does not use it
pip install -e . --no-deps        # the plugin itself, from the repo root
pip install zstandard
```

Versions match [`requirement.txt`](requirement.txt). `pip check` flags the removed `torchcodec`; that is expected.

### 3. Check the install

No GPU needed — the hermetic test gate covers engine-config policy, the runner adapter, graph
routing, TP install and the client wire format:

```bash
pytest tests -q -m "not gpu"
```

---

## 👉 Usage

MIA installs into the **server**, so a demo talks to a `vllm serve` you start yourself. One
server serves one worker kind at a time, selected with `MIA_WORKER`.

```bash
MIA_ALLOW_CUDAGRAPH=1 VLLM_WORKER_MULTIPROC_METHOD=spawn MIA_WORKER=hidden_states \
    vllm serve Qwen/Qwen2.5-3B-Instruct \
    --max-model-len 2048 --port 8770 --compilation-config '{"cudagraph_mode": "FULL"}'
```

Then, from the repo root:

```bash
python examples/demo_hiddenstate.py
```

Every demo prints the exact `vllm serve` command it needs if nothing is listening, so you never
have to guess. [`examples/README.md`](examples/README.md) is the walkthrough — getting started,
where captured data lands, how to confirm a capture actually happened, and tensor parallelism.
For the full list of use cases see [`docs/use_cases/`](docs/use_cases/README.md).

### Use cases

| Use case | Demo |
|---|---|
| Attention Tracker (in-model safety guardrail) | `python examples/demo_attntracker.py` |
| Core Reranker (in-model relevance ranking) | `python examples/demo_corer.py` |
| Activation Steering (enhanced instruction following) | `python examples/demo_actsteer.py` |
| Hidden-State Probe | `python examples/demo_hiddenstate.py` |
| Science Hallucination Detector | `python examples/demo_scihal.py` |
| H-Node Hallucination Detector | `python examples/demo_halludetect.py` |
| AttnLink-U (schema linking) | `python examples/demo_attnlink.py` |

You can customize model configurations in the `model_configs/` folder, e.g.:

```
model_configs/<example_name>/<model_name>.json
```
For example `model_configs/attention_tracker/granite-3.1-8b-instruct.json`.

---

## 🏠 Plugin Architecture

The main package is structured as follows:

```
mia/
├── analyzers/
│   ├── attention_tracker_analyzer.py
│   ├── core_reranker_analyzer.py
├── workers/
│   ├── qk_capture_worker.py
│   ├── steer_worker.py
├── graph/
│   ├── install.py
│   ├── capture_aperture.py
├── llm.py
├── runner.py
├── optimizations.py
├── registry.py
```

Each component handles a key stage of the plugin lifecycle:

- **Registry** — manages available hooks and extensions  
- **Workers** — define execution behavior and orchestration  
- **Analyzers** — optionally conduct analysis based on the saved statistics  
- **Graph** — installs the capture/steering ops and the GPU capture aperture under CUDA graphs  
- **Runner** — the one place that touches vLLM's V2 model-runner internals  
- **Optimizations** — the public performance levers (`optimizations.py::PUBLIC_LEVERS`)  


---

## 🤝 Contributing

We welcome contributions from the community!  

### To contribute:
1. **Fork** this repository  
2. **Create a branch** (`git checkout -b feature/amazing-feature`)  
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)  
4. **Push** to your branch (`git push origin feature/amazing-feature`)  
5. **Open a Pull Request**  

### Guidelines:
- Users are encouraged to define new worker/analyzer, but should not touch llm
- Include examples and documentation for new features  
- New use cases must be added to [`docs/use_cases/README.md`](docs/use_cases/README.md) with the contributor's GitHub handle

---

## 🌟 Feeling Inspired
```
@article{ko2026vllm,
  title={vLLM Hook v0: A Plug-in for Programming Model Internals on vLLM},
  author={Ko, Ching-Yun and Chen, Pin-Yu},
  journal={arXiv preprint arXiv:2603.06588},
  year={2026}
}
```
---


## IBM ❤️ Open Source AI

MIA is built on vLLM.hook, which was started by IBM Research.
- Built for the **vLLM** ecosystem  
- Inspired by community efforts to make LLMs more interpretable and controllable
