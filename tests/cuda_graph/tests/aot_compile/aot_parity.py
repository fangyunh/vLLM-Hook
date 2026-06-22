"""AOT-compile parity oracle for Hybrid-mode capture/steer (v0.3.x).

Proves the AOT-compile fixes on GPU. Three capture modes per worker, each in its
OWN process (exclusive-process GPU):

  * eager   — legacy register_forward_hook ground truth (AOT off, enforce_eager).
  * aot     — graph mode + VLLM_USE_AOT_COMPILE=1, FRESH compile (compiles, saves
              the AOT artifact, runs).
  * reload  — graph mode + VLLM_USE_AOT_COMPILE=1 against the SAME VLLM_CACHE_ROOT,
              so the engine LOADS the artifact 'aot' saved and replays it (Dynamo
              tracing skipped). This is the path that used to crash with
              ``'NoneType' object has no attribute 'add_'`` (the sink baked as a
              graph const deserialized to None) and that cross-loaded another
              worker's graph (shared AOT cache key).

For qk/hs the oracle compares captured tensors (allclose). For steer it compares
the EFFECT on the next-token distribution (argmax + that steering moved it).

PASS = aot AND reload both reproduce eager. A crash in 'reload' (regression) fails
the job loudly.

Usage:
    python aot_parity.py capture --worker qk --mode eager  --out e.pkl
    python aot_parity.py capture --worker qk --mode aot    --out a.pkl
    python aot_parity.py capture --worker qk --mode reload --out r.pkl
    python aot_parity.py compare --worker qk --eager e.pkl --aot a.pkl --reload r.pkl
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "0"  # before plugin import (graph mode)

import argparse
import pickle
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16
_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_AOT_MAX_TOKENS", "8"))
_COEFF = float(os.environ.get("VLLM_STEER_COEFF", "20.0"))
_TOPK = int(os.environ.get("VLLM_STEER_LOGPROBS", "20"))

_WORKERS = {
    "qk": {
        "worker_name": "probe_hook_qk", "analyzer_name": "attn_tracker",
        "config": f"model_configs/attention_tracker/{_MODEL.split('/')[-1]}.json",
        "cache_key": "qk_cache",
    },
    "hs": {
        "worker_name": "probe_hidden_states", "analyzer_name": "hidden_states",
        "config": f"model_configs/hidden_states/{_MODEL.split('/')[-1]}.json",
        "cache_key": "hs_cache",
    },
    "steer": {
        "worker_name": "steer_hook_act", "analyzer_name": None,
        "config": f"model_configs/activation_steer/{_MODEL.split('/')[-1]}.json",
        "cache_key": None,
    },
}

_CASES = [
    {"name": "clean", "text": "The capital of France is"},
    {"name": "other", "text": "Quantum computing leverages superposition to"},
]


def _to_cpu_f32(t):
    if isinstance(t, (list, tuple)):
        t = t[0] if t else None
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().to(torch.float32).cpu()


def _capture_store(out, cache_key):
    probes = getattr(out[0], "probes", None)
    cache = (probes or {}).get(cache_key, {})
    store = {}
    for layer, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        store[str(layer)] = {
            "q": _to_cpu_f32(entry.get("q")),
            "k_all": _to_cpu_f32(entry.get("k_all")),
            "hidden_states": _to_cpu_f32(entry.get("hidden_states")),
            "layer_num": entry.get("layer_num"),
        }
    return store


def _logprob_map(out):
    o = out[0].outputs[0]
    lps = o.logprobs[0] if o.logprobs else {}
    return {
        "logprobs": {int(tid): float(lp.logprob) for tid, lp in lps.items()},
        "token_id": int(o.token_ids[0]) if o.token_ids else None,
        "text": o.text,
    }


def capture(worker, mode, out_path):
    spec = _WORKERS[worker]
    if mode == "eager":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        os.environ["VLLM_USE_AOT_COMPILE"] = "0"
        enforce_eager = True
    else:  # aot (fresh) or reload — identical engine config; the shared cache root
           # (set by the run script) makes 'reload' hit the artifact 'aot' saved.
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        enforce_eager = False

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "PIECEWISE")
    config_file = os.environ.get("VLLM_HOOK_CONFIG_FILE", spec["config"])
    user = os.environ.get("USER", "u")

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode != "eager" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[aot-parity:{worker}:{mode}] booting AOT={os.environ['VLLM_USE_AOT_COMPILE']} "
          f"graph={os.environ['VLLM_HOOK_ALLOW_CUDAGRAPH']} enforce_eager={enforce_eager} "
          f"cache_root={os.environ.get('VLLM_CACHE_ROOT')}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name=spec["worker_name"],
        analyzer_name=spec["analyzer_name"],
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR", f"/dev/shm/vllm_hook_{user}"),
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

    result = {}
    if worker == "steer":
        sp_base = SamplingParams(temperature=0.0, max_tokens=1, logprobs=_TOPK)
        sp_steer = SamplingParams(temperature=0.0, max_tokens=1, logprobs=_TOPK,
                                  extra_args={"steer": {"coefficient": _COEFF}})
        for case in _CASES:
            base = _logprob_map(llm.generate(case["text"], sp_base, use_hook=False))
            steer = _logprob_map(llm.generate(case["text"], sp_steer, use_hook=True))
            result[case["name"]] = {"base": base, "steer": steer}
            print(f"[aot-parity:{worker}:{mode}] case={case['name']} "
                  f"base={base['token_id']}({base['text']!r}) "
                  f"steer={steer['token_id']}({steer['text']!r})", flush=True)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
        for case in _CASES:
            out = llm.generate(case["text"], sp, use_hook=True)
            store = _capture_store(out, spec["cache_key"])
            result[case["name"]] = store
            print(f"[aot-parity:{worker}:{mode}] case={case['name']} "
                  f"captured {len(store)} layers", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[aot-parity:{worker}:{mode}] wrote {out_path}", flush=True)
    return 0


def _align(g, e):
    if g is None or e is None or g.shape != e.shape:
        if g is not None and e is not None and g.dim() == e.dim() \
                and g.shape[1:] == e.shape[1:]:
            n = min(g.shape[0], e.shape[0])
            return g[-n:], e[-n:]
        return None, None
    return g, e


def _compare_capture(label, cand, eager, cache_key, rtol, atol):
    keys = ("q", "k_all") if cache_key == "qk_cache" else ("hidden_states",)
    ok_all, total, matched = True, 0, 0
    for case in sorted(set(cand) & set(eager)):
        cl, el = cand[case], eager[case]
        for layer in sorted(set(cl) & set(el)):
            for key in keys:
                cv, ev = cl[layer].get(key), el[layer].get(key)
                if cv is None or ev is None:
                    continue
                total += 1
                ga, ea = _align(cv, ev)
                if ga is None:
                    print(f"[aot-parity:{label}] case={case} L{layer} {key}: SHAPE "
                          f"MISMATCH cand={tuple(cv.shape)} eager={tuple(ev.shape)}")
                    ok_all = False
                    continue
                ok = torch.allclose(ga, ea, rtol=rtol, atol=atol)
                md = (ga - ea).abs().max().item() if ga.numel() else 0.0
                matched += int(ok)
                ok_all = ok_all and ok
                if not ok:
                    print(f"[aot-parity:{label}] case={case} L{layer} {key}: "
                          f"match=False max|Δ|={md:.3e} shape={tuple(ga.shape)}")
    print(f"[aot-parity:{label}] {matched}/{total} tensors within "
          f"rtol={rtol} atol={atol} -> {'OK' if ok_all and total else 'FAIL'}")
    return ok_all and total > 0


def _compare_steer(label, cand, eager, atol, steer_min, ratio):
    def maxdiff(a, b):
        sh = set(a) & set(b)
        return (max(abs(a[t] - b[t]) for t in sh), len(sh)) if sh else (float("inf"), 0)
    ok_all, any_case = True, False
    for case in sorted(set(cand) & set(eager)):
        any_case = True
        cs = cand[case]["steer"]["logprobs"]
        es, eb = eager[case]["steer"]["logprobs"], eager[case]["base"]["logprobs"]
        d_steer, _ = maxdiff(es, eb)
        d_par, n = maxdiff(cs, es)
        tok_ok = cand[case]["steer"]["token_id"] == eager[case]["steer"]["token_id"]
        vacuous = d_steer < steer_min
        par_ok = d_par <= atol or d_par <= d_steer * ratio
        case_ok = tok_ok and par_ok and not vacuous
        ok_all = ok_all and case_ok
        print(f"[aot-parity:{label}] case={case}: argmax_match={tok_ok} "
              f"D_steer={d_steer:.3e} D_parity={d_par:.3e} (shared={n}) "
              f"-> {'OK' if case_ok else 'FAIL'}"
              + ("  [VACUOUS]" if vacuous else ""))
    return ok_all and any_case


def compare(worker, eager_path, aot_path, reload_path, rtol, atol):
    spec = _WORKERS[worker]
    with open(eager_path, "rb") as f:
        e = pickle.load(f)
    with open(aot_path, "rb") as f:
        a = pickle.load(f)
    with open(reload_path, "rb") as f:
        r = pickle.load(f)

    if worker == "steer":
        aot_ok = _compare_steer("aot", a, e, 0.15, 0.1, 0.25)
        rl_ok = _compare_steer("reload", r, e, 0.15, 0.1, 0.25)
    else:
        aot_ok = _compare_capture("aot", a, e, spec["cache_key"], rtol, atol)
        rl_ok = _compare_capture("reload", r, e, spec["cache_key"], rtol, atol)

    print("=" * 64)
    print(f"[aot-parity:{worker}] aot(fresh-compile) vs eager: {'PASS' if aot_ok else 'FAIL'}")
    print(f"[aot-parity:{worker}] reload(from-cache) vs eager: {'PASS' if rl_ok else 'FAIL'}")
    if aot_ok and rl_ok:
        print(f"[aot-parity:{worker}] VERDICT: PASS — AOT compile + reload reproduce eager.")
        return 0
    print(f"[aot-parity:{worker}] VERDICT: FAIL")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--worker", choices=list(_WORKERS), required=True)
    pc.add_argument("--mode", choices=["eager", "aot", "reload"], required=True)
    pc.add_argument("--out", required=True)
    pk = sub.add_parser("compare")
    pk.add_argument("--worker", choices=list(_WORKERS), required=True)
    pk.add_argument("--eager", required=True)
    pk.add_argument("--aot", required=True)
    pk.add_argument("--reload", required=True)
    pk.add_argument("--rtol", type=float, default=1e-2)
    pk.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.worker, args.mode, args.out))
    else:
        sys.exit(compare(args.worker, args.eager, args.aot, args.reload,
                         args.rtol, args.atol))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
    os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")
    main()
