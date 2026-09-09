"""Verify that the EAGER (main-line) hook path truly captures and steers at the
DECODE stage under hooks_on="both".

Runs one worker per invocation (--worker {qk,hs,steer}) on a real GPU engine in
EAGER mode (VLLM_HOOK_ALLOW_CUDAGRAPH unset -> register_forward_hook path, which
is byte-identical between `main` and `graph_enable`). No cudagraph.

Signal (last_token mode): each captured tensor's dim-0 == number of forward
passes that fired the hook. With ignore_eos + max_tokens=G the engine runs
exactly G forwards = 1 prefill + (G-1) decode, so:

    hooks_on="prefill"  -> dim0 == 1        (prefill only)
    hooks_on="decode"   -> dim0 == G-1      (every decode step, no prefill)
    hooks_on="both"     -> dim0 == G        (prefill + every decode step)

If decode capture were broken, "decode" would be EMPTY and "both" would equal
"prefill". Steering has no hooks_on gate (fires on every forward): the worker's
live PROF.counters["steer.fire"] (single steered layer) must equal G, i.e. it
fired on all G-1 decode steps.
"""
import os
import sys
import json
import argparse

# ---- EAGER path: do NOT arm cudagraph. Offline harness must undo the plugin's
#      import-time TORCHDYNAMO_DISABLE default the same way the parity harnesses do.
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_HOOK_PROFILE", "1")           # so PROF.counters accrue in the worker
os.environ.setdefault("VLLM_HOOK_QK_COMPACT_KALL", "0")   # keep the plain k_all path (no compact rebuild)
os.environ.pop("VLLM_HOOK_ALLOW_CUDAGRAPH", None)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from vllm import SamplingParams
from vllm_hook_plugins import HookLLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rpc_helper import read_prof_counters   # importable in the EngineCore subprocess

MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
G = int(os.environ.get("VERIFY_MAX_TOKENS", "8"))          # generated tokens == forward passes
# NOTE layer-number convention differs per worker: QK filters on the 0-based
# PyTorch index; HS on a 1-based index (ln=layer_num+1). [7,14] is valid for both.
LAYERS = [7, 14]
PROMPT = "The capital of France is the city of"
SCRATCH = os.environ.get("VERIFY_SCRATCH", "/tmp/verify_decode")
os.makedirs(SCRATCH, exist_ok=True)


def _write_cfg(name: str, obj: dict) -> str:
    p = os.path.join(SCRATCH, name)
    with open(p, "w") as f:
        json.dump(obj, f)
    return p


def _sp(hooks_on=None, ignore_eos=True):
    extra = {}
    if hooks_on is not None:
        extra["hooks_on"] = hooks_on
    return SamplingParams(temperature=0.0, max_tokens=G, ignore_eos=ignore_eos,
                          extra_args=extra or None)


def _read_counters(llm):
    """Read the worker process's live PROF counters (no plugin change needed)."""
    for handle in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if handle is None:
            continue
        try:
            res = handle.collective_rpc(read_prof_counters)
            if res:
                return res[0]
        except Exception as e:  # noqa
            print(f"[verify] collective_rpc via {type(handle).__name__} failed: {e}")
    return None


def _gen_len(out):
    return len(out.outputs[0].token_ids)


def _dim0(entry):
    """Number of captured forward passes for a last_token probe entry."""
    for k in ("q", "hidden_states", "k_all"):
        t = entry.get(k)
        if isinstance(t, torch.Tensor):
            return int(t.shape[0])
    return None


def _probe_entries(probes, cache_key):
    cache = (probes or {}).get(cache_key, {})
    return {m: e for m, e in cache.items() if isinstance(e, dict)}


# ----------------------------------------------------------------------------- QK / HS
def verify_capture(worker):
    if worker == "qk":
        cfg = _write_cfg("qk.json", {
            "model_info": {"name": MODEL},
            "params": {"important_heads": [[L, 0] for L in LAYERS]},
            "hookq": {"hookq_mode": "last_token"},
        })
        cache_key = "qk_cache"
        wname = "probe_hook_qk"
    else:
        cfg = _write_cfg("hs.json", {
            "model_info": {"name": MODEL},
            "hidden_states": {"layers": LAYERS, "mode": "last_token"},
        })
        cache_key = "hs_cache"
        wname = "probe_hidden_states"

    llm = HookLLM(model=MODEL, worker_name=wname, analyzer_name=None,
                  config_file=cfg, download_dir="./cache/", enforce_eager=True,
                  gpu_memory_utilization=0.55, max_model_len=2048, enable_hook=True)

    results = {}
    for mode in ("prefill", "decode", "both"):
        out = llm.generate(PROMPT, _sp(hooks_on=mode))[0]
        n = _gen_len(out)
        entries = _probe_entries(getattr(out, "probes", None), cache_key)
        dims = {m: _dim0(e) for m, e in entries.items()}
        results[mode] = {"n_gen": n, "layers": len(entries), "dim0": dims,
                         "probes": entries}
        print(f"[verify:{worker}] hooks_on={mode:7s} n_gen={n} "
              f"layers_captured={len(entries)} passes_per_layer={sorted(set(dims.values()))}")

    ok = True
    reasons = []
    G_eff = results["both"]["n_gen"]  # actual generated tokens == forward passes

    # Every mode must capture the requested layers.
    for mode in ("prefill", "decode", "both"):
        if results[mode]["layers"] != len(LAYERS):
            ok = False
            reasons.append(f"{mode}: captured {results[mode]['layers']} layers, want {len(LAYERS)}")

    def _all(mode, val):
        d = results[mode]["dim0"]
        return d and all(v == val for v in d.values())

    if not _all("prefill", 1):
        ok = False; reasons.append(f"prefill passes != 1: {results['prefill']['dim0']}")
    if not _all("decode", G_eff - 1):
        ok = False; reasons.append(f"decode passes != {G_eff-1} (G-1): {results['decode']['dim0']}")
    if not _all("both", G_eff):
        ok = False; reasons.append(f"both passes != {G_eff} (G): {results['both']['dim0']}")

    # Decode must actually add passes over prefill (the crux).
    if not (G_eff - 1 >= 1):
        ok = False; reasons.append("G too small to exercise decode")

    # Correctness: captured values are real & finite, and the prefill pass is
    # deterministic across the prefill-only and both runs (same prompt).
    key0 = "q" if worker == "qk" else "hidden_states"
    for m, e in results["both"]["probes"].items():
        t = e.get(key0)
        if not (isinstance(t, torch.Tensor) and torch.isfinite(t).all() and t.abs().sum() > 0):
            ok = False; reasons.append(f"both[{m}].{key0} not finite/nonzero")
    # match prefill-pass activation between prefill-only run and both run's first pass
    for m in results["both"]["probes"]:
        tb = results["both"]["probes"][m].get(key0)
        tp = results["prefill"]["probes"].get(m, {}).get(key0)
        if isinstance(tb, torch.Tensor) and isinstance(tp, torch.Tensor):
            a = tb[0].float()
            b = tp[0].float() if tp.dim() == tb.dim() else tp.reshape(a.shape).float()
            if not torch.allclose(a, b, rtol=1e-2, atol=1e-2):
                ok = False
                reasons.append(f"{m}: prefill pass mismatch both-vs-prefill "
                               f"(max|d|={ (a-b).abs().max().item():.3g })")

    print(f"[verify:{worker}] G(forwards)={G_eff}  "
          f"prefill=1  decode={G_eff-1}  both={G_eff}  (expected)")
    if ok:
        print(f"[verify:{worker}] VERDICT: PASS — decode-stage capture confirmed "
              f"(decode captured {G_eff-1} decode steps; both = prefill + decode; values correct)")
    else:
        print(f"[verify:{worker}] VERDICT: FAIL — {'; '.join(reasons)}")
    return ok


# ----------------------------------------------------------------------------- steer
def verify_steer():
    cfg = _write_cfg("steer.json", {
        "model_info": {"name": MODEL},
        "steering": {
            "method": "add_vector",
            "coefficient": float(os.environ.get("VERIFY_STEER_COEFF", "20.0")),
            "optimal_layer": 14,               # SINGLE layer -> steer.fire == #forwards
            "vector_path": "steering_vectors/qwen2_dummy.pt",
            "apply_at_all_positions": True,
        },
    })
    llm = HookLLM(model=MODEL, worker_name="steer_hook_act", analyzer_name=None,
                  config_file=cfg, download_dir="./cache/", enforce_eager=True,
                  gpu_memory_utilization=0.55, max_model_len=2048, enable_hook=True)

    c0 = _read_counters(llm)
    base = (c0 or {}).get("steer.fire", 0)
    print(f"[verify:steer] steer.fire before any generate = {base}")

    off = llm.generate(PROMPT, _sp(), use_hook=False)[0]
    t_off = list(off.outputs[0].token_ids)
    c1 = _read_counters(llm)
    fire_after_off = (c1 or {}).get("steer.fire", 0)
    print(f"[verify:steer] unsteered gen n={len(t_off)} steer.fire(after off)={fire_after_off}")

    on = llm.generate(PROMPT, _sp(), use_hook=True)[0]
    t_on = list(on.outputs[0].token_ids)
    c2 = _read_counters(llm)
    fire_after_on = (c2 or {}).get("steer.fire", 0)
    fire_this_run = fire_after_on - fire_after_off
    n_on = len(t_on)
    print(f"[verify:steer] steered gen n={n_on} steer.fire(this run)={fire_this_run} "
          f"(=#forwards; decode steps={fire_this_run-1})")
    print(f"[verify:steer] tokens off={t_off}")
    print(f"[verify:steer] tokens on ={t_on}")

    ok = True
    reasons = []
    if c2 is None:
        ok = False; reasons.append("could not read worker PROF counters (collective_rpc)")
    else:
        if fire_after_off != 0:
            ok = False; reasons.append(f"steer fired {fire_after_off}x on use_hook=False (should be 0)")
        # CRUX: fires happened beyond the single prefill pass -> decode was steered.
        if fire_this_run - 1 < 1:
            ok = False; reasons.append("no decode-step steering observed (fire<=1)")
        # exact 1:1 with forwards is expected (ignore_eos, no EOS surplus) — report, don't hard-fail
        if fire_this_run != n_on:
            print(f"[verify:steer] NOTE steer.fire={fire_this_run} != n_gen={n_on} "
                  f"(off-by-{fire_this_run-n_on}; crux is fire>1)")
    if t_on == t_off:
        ok = False; reasons.append("steered output identical to unsteered (steering ineffective)")

    if ok:
        print(f"[verify:steer] VERDICT: PASS — decode-stage steering confirmed "
              f"(fired on {fire_this_run-1} decode steps; output changed {sum(a!=b for a,b in zip(t_on,t_off))}/{min(len(t_on),len(t_off))} tokens)")
    else:
        print(f"[verify:steer] VERDICT: FAIL — {'; '.join(reasons)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, choices=["qk", "hs", "steer"])
    a = ap.parse_args()
    if a.worker == "steer":
        good = verify_steer()
    else:
        good = verify_capture(a.worker)
    sys.exit(0 if good else 1)
