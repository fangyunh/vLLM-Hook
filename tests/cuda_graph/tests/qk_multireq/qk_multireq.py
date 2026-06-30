"""QK multi-request graph-vs-solo-eager parity oracle (v0.5.6 W1′).

WHY: v0.5.6 adds W1′ idle-skip to buffer-mode QK capture — on a step where NO
request captures (every decode step of a hooks_on=prefill workload), the routing
wrapper skips reset/build/upload and the zeroed slab replays as a no-op scatter.
The skip fires ONLY when the WHOLE batch is idle; the moment any request captures
the routing rebuilds (byte-identical to v0.5.5). This oracle proves that under W1′:
  (1) a MULTI-REQUEST batch still captures each request correctly (constraint c:
      different requests, different per-request layer sets, in ONE graph batch);
  (2) idle decode steps that are now SKIPPED corrupt nothing (the prefill scenario
      makes every decode step idle); and
  (3) the always-active decode-capture path (hooks_on=both) is unchanged (the both
      scenario keeps every step active so idle-skip never fires — an M5 re-run that
      guards against an active-path regression).

METHOD (the M5 trick): HookLLM.generate MERGES per-request probes onto outputs[0]
and RAISES on heterogeneous layer sets. We prepend a NON-capturing sentinel request
(extra_args output_qk=None → outputs[0].probes is None → merge SKIPPED), then read
the pristine per-request outputs[1..N].probes. Each real request is compared to its
OWN SOLO-EAGER baseline (the request run alone, eager register_forward_hook path) —
the real ground truth for "did the batch capture this request right".

Scenarios (env VLLM_HOOK_MULTIREQ_SCENARIO):
  prefill — N=4, hooks_on=prefill, HETEROGENEOUS per-request layer sets, different
            prompt lengths. Capture at prefill; EVERY decode step is idle → W1′
            idle-skip fires on all decode steps. (c) + idle-skip correctness.
  both    — N=4, hooks_on=both, all layers, different max_tokens (churn). Every step
            captures → idle-skip never fires; requests finish at different steps →
            input_batch condense. Active-path + composition-churn regression.

PASS = for every real request, every captured layer's q AND k_all match the solo-eager
baseline within rtol=atol=1e-2, the per-request layer SET matches its spec, and the
per-request k_all length matches solo-eager (no cross-request bleed / drop).

    python qk_multireq.py capture --mode graph --out g.pkl
    python qk_multireq.py capture --mode eager --out e.pkl   # solo-eager per request
    python qk_multireq.py compare --graph g.pkl --eager e.pkl
"""
import os
# MUST precede the plugin import (beat hook_llm's setdefault TORCHDYNAMO_DISABLE=1).
os.environ["TORCHDYNAMO_DISABLE"] = "0"

import argparse
import getpass
import pickle
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16
_SCENARIO = os.environ.get("VLLM_HOOK_MULTIREQ_SCENARIO", "prefill").lower()

# Four real requests with clearly DIFFERENT lengths (staggered prefill) + max_tokens.
_PROMPTS = [
    "Hello there.",
    "The capital city of France, a country in western Europe, is",
    "Quantum",
    "In the beginning, long before the stars and galaxies first formed across the "
    "vast and silent cosmos, the entire universe was",
]
# Per-request 0-based layer sets for the heterogeneous (prefill) scenario.
_HETERO_LAYERS = [[0, 1], [5, 10], [2], [0, 3, 7]]
_BOTH_MAXTOK = [4, 8, 12, 16]   # churn: staggered finishes


def _specs(scenario):
    """Per real request: (max_tokens, hooks_on, output_qk override).

    output_qk None here means "use all layers" (override with True at call time);
    a list means a heterogeneous per-request layer set.
    """
    if scenario == "both":
        return [(mt, "both", True) for mt in _BOTH_MAXTOK]
    # prefill (default): hooks_on=prefill, heterogeneous per-request layer sets,
    # max_tokens>1 so there ARE decode steps to idle-skip.
    return [(8, "prefill", layers) for layers in _HETERO_LAYERS]


def _to_cpu_f32(t):
    if isinstance(t, (list, tuple)):
        if not t:
            return None
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().to(torch.float32).cpu()


def _capture_store(out_obj):
    """Extract {layer: {q, k_all, layer_num}} from one request's generate output."""
    probes = getattr(out_obj, "probes", None)
    qk = (probes or {}).get("qk_cache", {})
    store = {}
    for layer, entry in qk.items():
        if not isinstance(entry, dict):
            continue
        store[str(layer)] = {
            "q": _to_cpu_f32(entry.get("q")),
            "k_all": _to_cpu_f32(entry.get("k_all")),
            "layer_num": entry.get("layer_num"),
        }
    return store


def _make_llm(enforce_eager, cudagraph_mode):
    from vllm_hook_plugins import HookLLM
    _user = os.environ.get("USER") or getpass.getuser()
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        "model_configs/attention_tracker/Qwen2-1.5B-Instruct_alltok.json")
    extra = {}
    if not enforce_eager and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}
    return HookLLM(
        model=_MODEL,
        worker_name="probe_hook_qk",
        analyzer_name="attn_tracker",
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{_user}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,   # solo-eager has no batch-mate to share a prefix
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )


def _sp(max_tokens, hooks_on, output_qk):
    from vllm import SamplingParams
    extra = {"hooks_on": hooks_on, "output_qk": output_qk}
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra)


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ.setdefault("VLLM_HOOK_QK_CAPTURE", "buffer")
        enforce_eager = False
    else:  # solo-eager ground truth (legacy register_forward_hook path)
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    specs = _specs(_SCENARIO)
    print(f"[multireq:{_SCENARIO}:{mode}] booting enforce_eager={enforce_eager} "
          f"cudagraph_mode={cudagraph_mode if mode=='graph' else 'n/a'} "
          f"nreq={len(_PROMPTS)}", flush=True)

    llm = _make_llm(enforce_eager, cudagraph_mode)
    result = {}

    if mode == "graph":
        # ONE batch: a non-capturing sentinel (output_qk=None) at index 0 so the
        # HookLLM merge is skipped, then the 4 real requests. Read pristine per-req
        # outputs[1..N].probes.
        prompts = ["Sentinel prompt that captures nothing."] + list(_PROMPTS)
        sps = [_sp(4, "prefill", None)]
        for mt, hooks_on, oqk in specs:
            sps.append(_sp(mt, hooks_on, oqk))
        outs = llm.generate(prompts, sps, use_hook=True)
        for i in range(len(_PROMPTS)):
            store = _capture_store(outs[i + 1])
            result[i] = store
            print(f"[multireq:{_SCENARIO}:graph] req{i} layers={sorted(store)} "
                  f"k_all_rows={[tuple(v['k_all'].shape) for v in store.values() if v['k_all'] is not None][:1]}",
                  flush=True)
    else:
        # SOLO-EAGER baseline: run each real request ALONE (single request → no merge,
        # outputs[0].probes pristine), same per-request spec as the graph batch.
        for i, (mt, hooks_on, oqk) in enumerate(specs):
            out = llm.generate([_PROMPTS[i]], [_sp(mt, hooks_on, oqk)], use_hook=True)
            store = _capture_store(out[0])
            result[i] = store
            print(f"[multireq:{_SCENARIO}:eager] req{i} layers={sorted(store)}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"scenario": _SCENARIO, "specs": specs, "store": result}, f)
    print(f"[multireq:{_SCENARIO}:{mode}] wrote {out_path}", flush=True)
    return 0


def _expected_layer_set(oqk, num_layers_hint=64):
    """The 0-based layer-num set a spec's output_qk should capture (list only;
    True = all, validated structurally not against a fixed count)."""
    if isinstance(oqk, list):
        return {int(x) for x in oqk}
    return None  # True/all — don't pin a count


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
        oqk = specs[i][2]
        # (c) per-request layer-set check: the graph request must capture exactly the
        # layers it asked for (heterogeneous case). Compare the captured layer_num set
        # vs the solo-eager request's captured set (same spec → same set).
        gset = {int(v["layer_num"]) for v in gl.values() if v.get("layer_num") is not None}
        eset = {int(v["layer_num"]) for v in el.values() if v.get("layer_num") is not None}
        exp = _expected_layer_set(oqk)
        set_ok = (gset == eset) and (exp is None or gset == exp)
        if not set_ok:
            overall_ok = False
        print(f"[multireq] req{i} layer_set graph={sorted(gset)} eager={sorted(eset)} "
              f"expect={sorted(exp) if exp else 'all'} -> {'OK' if set_ok else 'MISMATCH'}")

        common = sorted(set(gl) & set(el),
                        key=lambda x: int(gl[x].get("layer_num", -1)))
        if not common:
            print(f"[multireq] req{i}: NO common layers (graph={len(gl)} eager={len(el)})")
            overall_ok = False
            continue
        for layer in common:
            for key in ("q", "k_all"):
                gv, ev = gl[layer].get(key), el[layer].get(key)
                if gv is None or ev is None:
                    continue
                total += 1
                if gv.shape != ev.shape:
                    print(f"[multireq] req{i} layer={layer} {key}: SHAPE MISMATCH "
                          f"graph={tuple(gv.shape)} eager={tuple(ev.shape)} "
                          f"(cross-request bleed or dropped token?)")
                    overall_ok = False
                    continue
                ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
                md = (gv - ev).abs().max().item() if gv.numel() else 0.0
                if ok:
                    matched += 1
                else:
                    overall_ok = False
                print(f"[multireq] req{i} layer={layer} {key}: match={ok} "
                      f"max|Δ|={md:.3e} shape={tuple(gv.shape)}")

    print("=" * 64)
    print(f"[multireq:{gd['scenario']}] {matched}/{total} tensors within "
          f"rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print(f"[multireq:{gd['scenario']}] VERDICT: PASS — every request's graph "
              f"capture matches its solo-eager baseline (q + k_all + layer set).")
        return 0
    print(f"[multireq:{gd['scenario']}] VERDICT: FAIL — a request diverged from its "
          f"solo-eager baseline (value, shape, or layer set).")
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
