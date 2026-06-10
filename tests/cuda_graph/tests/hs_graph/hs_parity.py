"""Subprocess-isolated graph-vs-eager HIDDEN-STATE parity oracle (v0.3.0).

HS analogue of tests/cuda_graph/tests/qk_graph/qk_parity.py. Captures the same
prompts twice — once under CUDA-graph PIECEWISE mode (the hs_probe op path), once
under the legacy eager register_forward_hook path (VLLM_HOOK_ALLOW_CUDAGRAPH
unset) — in SEPARATE processes (exclusive-process GPU), then asserts the per-layer
hidden states match within tolerance.

    python hs_parity.py capture --mode graph --out g.pkl
    python hs_parity.py capture --mode eager --out e.pkl
    python hs_parity.py compare --graph g.pkl --eager e.pkl
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import getpass
import pickle
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _to_cpu_f32(t):
    if isinstance(t, (list, tuple)):
        if not t:
            return None
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().to(torch.float32).cpu()


def _capture_store(out):
    probes = getattr(out[0], "probes", None)
    hs = (probes or {}).get("hs_cache", {})
    store = {}
    for layer, entry in hs.items():
        if not isinstance(entry, dict):
            continue
        store[str(layer)] = {
            "hidden_states": _to_cpu_f32(entry.get("hidden_states")),
            "layer_num": entry.get("layer_num"),
        }
    return store


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{_MODEL.split('/')[-1]}.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[hs-parity:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    result = {}
    for case in _CASES:
        out = llm.generate(case["text"], sp, save_to_disk=False)
        store = _capture_store(out)
        result[case["name"]] = store
        print(f"[hs-parity:{mode}] case={case['name']} captured {len(store)} layers",
              flush=True)
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception:  # noqa: BLE001
            pass

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[hs-parity:{mode}] wrote {out_path}", flush=True)
    return 0


def compare(graph_path, eager_path, rtol, atol):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    overall_ok = True
    total = matched = 0
    for case in sorted(set(g) & set(e)):
        common = sorted(set(g[case]) & set(e[case]))
        if not common:
            print(f"[hs-parity] case={case}: NO common layers "
                  f"(graph={len(g[case])}, eager={len(e[case])})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g[case][layer].get("hidden_states"), e[case][layer].get("hidden_states")
            if gv is None or ev is None:
                continue
            total += 1
            if gv.shape != ev.shape:
                print(f"[hs-parity] case={case} layer={layer}: SHAPE MISMATCH "
                      f"graph={tuple(gv.shape)} eager={tuple(ev.shape)}")
                overall_ok = False
                continue
            ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
            md = (gv - ev).abs().max().item() if gv.numel() else 0.0
            mean = (gv - ev).abs().mean().item() if gv.numel() else 0.0
            matched += int(ok)
            overall_ok = overall_ok and ok
            print(f"[hs-parity] case={case} layer={layer}: match={ok} "
                  f"max|Δ|={md:.3e} mean|Δ|={mean:.3e} shape={tuple(gv.shape)}")

    print("=" * 60)
    print(f"[hs-parity] {matched}/{total} tensors within rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print("[hs-parity] VERDICT: PASS — graph HS capture matches eager.")
        return 0
    print("[hs-parity] VERDICT: FAIL — graph HS capture diverged from eager.")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--mode", choices=["graph", "eager"], required=True)
    pc.add_argument("--out", required=True)
    pk = sub.add_parser("compare")
    pk.add_argument("--graph", required=True)
    pk.add_argument("--eager", required=True)
    pk.add_argument("--rtol", type=float, default=1e-2)
    pk.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out))
    else:
        sys.exit(compare(args.graph, args.eager, args.rtol, args.atol))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
    os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")
    main()
