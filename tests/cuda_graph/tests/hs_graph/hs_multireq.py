"""HS multi-request graph-vs-solo-eager parity oracle (vectorized egress, VEC path).

WHY: the vectorized HS egress (the VEC path) lays a whole
concurrent batch out as reduced[li*n+j] and splits it per request with ARITHMETIC offsets. A
wrong offset would bleed one request's rows into another's bucket — invisible to the 1-request
hs_parity oracle. This proves, per request in a CONCURRENT batch, that the VEC capture matches
that request's SOLO-EAGER baseline (run alone, legacy register_forward_hook), with staggered
prompt lengths (distinct slab last-token positions) + staggered max_tokens (condense as requests
finish at different steps).

METHOD (the M5 trick): HookLLM.generate merges per-request probes onto outputs[0] and raises on
heterogeneous layer sets, so we prepend a NON-capturing sentinel (output_hidden_states=None ->
outputs[0].probes is None -> merge SKIPPED) and read pristine per-request outputs[1..N].probes.

Scenarios (VLLM_HOOK_MULTIREQ_SCENARIO):
  both   — N=4, hooks_on=both, last_token, ALL layers, different max_tokens. VEC FIRES for every
           step (homogeneous) -> the core multi-request VEC proof + condense churn.
  subset — N=4, hooks_on=both, last_token, every request the SAME subset [1,2,3,4]. VEC FIRES via
           the v0.5.10 broadening (homogeneous but not all-registered) -> multi-req subset proof.
  hetero — N=4, hooks_on=both, last_token, per-request DIFFERENT layer subsets. VEC returns False
           (heterogeneous) -> falls back to the standard path under concurrency (fallback guard).

PASS = every real request's every captured layer matches its solo-eager baseline (value + shape +
layer set) within rtol=atol=1e-2 — no cross-request bleed, no dropped step.

    python hs_multireq.py capture --mode graph --out g.pkl
    python hs_multireq.py capture --mode eager --out e.pkl
    python hs_multireq.py compare --graph g.pkl --eager e.pkl
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
_SCENARIO = os.environ.get("VLLM_HOOK_MULTIREQ_SCENARIO", "both").lower()

_PROMPTS = [
    "Hello there.",
    "The capital city of France, a country in western Europe, is",
    "Quantum",
    "In the beginning, long before the stars and galaxies first formed across the "
    "vast and silent cosmos, the entire universe was",
]
# Per-request 1-based layer sets (output_hidden_states list) for the heterogeneous scenario.
_HETERO_LAYERS = [[1, 2], [6, 11], [3], [1, 4, 8]]
_SUBSET_LAYERS = [1, 2, 3, 4]   # shared subset for the homogeneous-subset scenario
_BOTH_MAXTOK = [4, 8, 12, 16]   # churn: staggered finishes -> input_batch condense


def _specs(scenario):
    """Per real request: (max_tokens, hooks_on, output_hidden_states). True = all layers."""
    if scenario == "hetero":
        return [(mt, "both", layers) for mt, layers in zip(_BOTH_MAXTOK, _HETERO_LAYERS)]
    if scenario == "subset":
        # Every request captures the SAME subset -> homogeneous but NOT all-registered.
        # VEC must FIRE here (the v0.5.10 broadening) and match solo-eager per request.
        return [(mt, "both", list(_SUBSET_LAYERS)) for mt in _BOTH_MAXTOK]
    return [(mt, "both", True) for mt in _BOTH_MAXTOK]   # both: homogeneous all layers


def _to_cpu_f32(t):
    if isinstance(t, (list, tuple)):
        if not t:
            return None
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().to(torch.float32).cpu()


def _capture_store(out_obj):
    probes = getattr(out_obj, "probes", None)
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


def _make_llm(enforce_eager, cudagraph_mode):
    from vllm_hook_plugins import HookLLM
    _user = os.environ.get("USER") or getpass.getuser()
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        "model_configs/hidden_states/Qwen2-1.5B-Instruct_lasttok_all.json")
    extra = {}
    if not enforce_eager and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}
    return HookLLM(
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
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )


def _sp(max_tokens, hooks_on, output_hs):
    from vllm import SamplingParams
    extra = {"hooks_on": hooks_on, "output_hidden_states": output_hs, "hs_mode": "last_token"}
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra)


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ.setdefault("VLLM_HOOK_HS_CAPTURE", "buffer")
        enforce_eager = False
    else:  # solo-eager ground truth
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    specs = _specs(_SCENARIO)
    print(f"[hs-multireq:{_SCENARIO}:{mode}] booting enforce_eager={enforce_eager} "
          f"cudagraph_mode={cudagraph_mode if mode=='graph' else 'n/a'} "
          f"nreq={len(_PROMPTS)}", flush=True)

    llm = _make_llm(enforce_eager, cudagraph_mode)
    result = {}

    if mode == "graph":
        prompts = ["Sentinel prompt that captures nothing."] + list(_PROMPTS)
        sps = [_sp(4, "both", None)]   # sentinel: output_hidden_states=None -> merge skipped
        for mt, hooks_on, ohs in specs:
            sps.append(_sp(mt, hooks_on, ohs))
        outs = llm.generate(prompts, sps, use_hook=True)
        for i in range(len(_PROMPTS)):
            store = _capture_store(outs[i + 1])
            result[i] = store
            print(f"[hs-multireq:{_SCENARIO}:graph] req{i} layers={sorted(store)} "
                  f"rows={[tuple(v['hidden_states'].shape) for v in store.values() if v['hidden_states'] is not None][:1]}",
                  flush=True)
    else:
        for i, (mt, hooks_on, ohs) in enumerate(specs):
            out = llm.generate([_PROMPTS[i]], [_sp(mt, hooks_on, ohs)], use_hook=True)
            store = _capture_store(out[0])
            result[i] = store
            print(f"[hs-multireq:{_SCENARIO}:eager] req{i} layers={sorted(store)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"scenario": _SCENARIO, "specs": specs, "store": result}, f)
    print(f"[hs-multireq:{_SCENARIO}:{mode}] wrote {out_path}", flush=True)
    return 0


def _expected_layer_set(ohs):
    if isinstance(ohs, list):
        return {int(x) for x in ohs}   # 1-based layer_num
    return None


def compare(graph_path, eager_path, rtol, atol):
    with open(graph_path, "rb") as f:
        gd = pickle.load(f)
    with open(eager_path, "rb") as f:
        ed = pickle.load(f)
    g, e, specs = gd["store"], ed["store"], gd["specs"]

    overall_ok = True
    total = matched = 0
    for i in sorted(set(g) & set(e)):
        gl, el = g[i], e[i]
        ohs = specs[i][2]
        gset = {int(v["layer_num"]) for v in gl.values() if v.get("layer_num") is not None}
        eset = {int(v["layer_num"]) for v in el.values() if v.get("layer_num") is not None}
        exp = _expected_layer_set(ohs)
        set_ok = (gset == eset) and (exp is None or gset == exp)
        overall_ok = overall_ok and set_ok
        print(f"[hs-multireq] req{i} layer_set graph={sorted(gset)} eager={sorted(eset)} "
              f"expect={sorted(exp) if exp else 'all'} -> {'OK' if set_ok else 'MISMATCH'}")

        common = sorted(set(gl) & set(el), key=lambda x: int(gl[x].get("layer_num", -1)))
        if not common:
            print(f"[hs-multireq] req{i}: NO common layers (graph={len(gl)} eager={len(el)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = gl[layer].get("hidden_states"), el[layer].get("hidden_states")
            if gv is None or ev is None:
                continue
            total += 1
            if gv.shape != ev.shape:
                print(f"[hs-multireq] req{i} layer={layer}: SHAPE MISMATCH graph={tuple(gv.shape)} "
                      f"eager={tuple(ev.shape)} (cross-request bleed or dropped step?)")
                overall_ok = False
                continue
            ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
            md = (gv - ev).abs().max().item() if gv.numel() else 0.0
            matched += int(ok)
            overall_ok = overall_ok and ok
            print(f"[hs-multireq] req{i} layer={layer}: match={ok} max|Δ|={md:.3e} "
                  f"shape={tuple(gv.shape)}")

    print("=" * 64)
    print(f"[hs-multireq:{gd['scenario']}] {matched}/{total} tensors within rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print(f"[hs-multireq:{gd['scenario']}] VERDICT: PASS — every request's graph capture "
              f"matches its solo-eager baseline (value + shape + layer set).")
        return 0
    print(f"[hs-multireq:{gd['scenario']}] VERDICT: FAIL — a request diverged from solo-eager.")
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
    main()
