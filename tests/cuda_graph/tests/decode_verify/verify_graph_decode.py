"""Verify the FULL CUDA-graph (buffer) path captures/steers CORRECTLY at DECODE.

This is a DIFFERENT mechanism from the eager register_forward_hook path: a
capture_qk/capture_hs scatter op baked into the decode cudagraph, fed by routing
built in a _prepare_inputs wrapper. Whether the eager path is correct says nothing
about this path, so it is verified on its own here.

Correctness is checked against an INDEPENDENT ground truth (NOT "graph == eager"):
TEACHER FORCING. Generate G tokens under FULL cudagraph capturing every decode
step (all_tokens, hooks_on=both). Then feed [prompt + generated] as ONE prefill
and capture every position. The value the decode step captured for the token at
sequence position p MUST equal the value the model computes for position p in the
full-sequence prefill (same weights, same causal context) — i.e. the capture
reflects the model's ACTUAL computation at decode, to numerical tolerance.

    decode-captured[gen token i]  ==  prefill-of-full-seq[position P+i]

Also proves the graph capture op FIRES at decode (last_token dim0 = #passes:
prefill=1, decode=G-1, both=G) and that steering changes decode output in graph
mode. Run one worker per invocation on a real GPU.
"""
import os
import sys
import json
import argparse

# ---- Arm FULL cudagraph + buffer BEFORE importing the plugin ----------------
os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"          # graph mode; leaves torch.compile ON
os.environ.setdefault("VLLM_HOOK_QK_CAPTURE", "buffer")
os.environ.setdefault("VLLM_HOOK_HS_CAPTURE", "buffer")
os.environ.setdefault("VLLM_HOOK_STEER_MODE", "buffer")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_HOOK_QK_COMPACT_KALL", "0")  # plain k_all path
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_HOOK_PROFILE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from vllm import SamplingParams
import vllm_hook_plugins as _vhp
# Optional hard guard: pin which package tree is under test (e.g. run MAIN's
# package via PYTHONPATH while this harness lives in the graph_enable tree).
_EXPECT = os.environ.get("VERIFY_EXPECT_PKG_ROOT")
print(f"[verify] vllm_hook_plugins imported from: {_vhp.__file__}")
if _EXPECT and not os.path.abspath(_vhp.__file__).startswith(os.path.abspath(_EXPECT)):
    raise SystemExit(f"[verify] WRONG PACKAGE — got {_vhp.__file__}, expected under {_EXPECT}")
from vllm_hook_plugins import HookLLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rpc_helper import read_prof_counters

MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
G = int(os.environ.get("VERIFY_MAX_TOKENS", "6"))
LAYERS = [7, 14]
PROMPT = "The capital of France is the city of"
TOL = float(os.environ.get("VERIFY_TF_TOL", "2e-2"))
SCRATCH = os.environ.get("VERIFY_SCRATCH", "/tmp/verify_graph_decode")
os.makedirs(SCRATCH, exist_ok=True)
GRAPH_KW = dict(enforce_eager=False, compilation_config={"cudagraph_mode": "FULL"})


def _cfg(name, obj):
    p = os.path.join(SCRATCH, name)
    with open(p, "w") as f:
        json.dump(obj, f)
    return p


def _sp(hooks_on=None, max_tokens=G, ignore_eos=True):
    extra = {"hooks_on": hooks_on} if hooks_on else {}
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                          ignore_eos=ignore_eos, extra_args=extra or None)


def _read_counters(llm):
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            r = h.collective_rpc(read_prof_counters)
            if r:
                return r[0]
        except Exception as e:  # noqa
            print(f"[verify] collective_rpc via {type(h).__name__} failed: {e}")
    return None


def _entries(probes, key):
    return {m: e for m, e in (probes or {}).get(key, {}).items() if isinstance(e, dict)}


def _dim0(entry):
    for k in ("q", "hidden_states", "k_all"):
        t = entry.get(k)
        if isinstance(t, torch.Tensor):
            return int(t.shape[0])
    return None


# ---------------------------------------------------------------- QK / HS -----
def verify_capture(worker):
    if worker == "qk":
        cfg = _cfg("qk.json", {"model_info": {"name": MODEL},
                               "params": {"important_heads": [[L, 0] for L in LAYERS]},
                               "hookq": {"hookq_mode": "last_token"}})
        cache_key, wname, key0 = "qk_cache", "probe_hook_qk", "q"
    else:
        cfg = _cfg("hs.json", {"model_info": {"name": MODEL},
                               "hidden_states": {"layers": LAYERS, "mode": "last_token"}})
        cache_key, wname, key0 = "hs_cache", "probe_hidden_states", "hidden_states"

    llm = HookLLM(model=MODEL, worker_name=wname, analyzer_name=None, config_file=cfg,
                  download_dir="./cache/", gpu_memory_utilization=0.6, max_model_len=2048,
                  enable_hook=True, enable_prefix_caching=False, **GRAPH_KW)

    # ---- (1) the graph capture op FIRES at decode (last_token dim0 = #passes)
    dim = {}
    for mode in ("prefill", "decode", "both"):
        out = llm.generate(PROMPT, _sp(hooks_on=mode))[0]
        e = _entries(getattr(out, "probes", None), cache_key)
        dim[mode] = {"n": len(out.outputs[0].token_ids), "layers": len(e),
                     "passes": sorted({_dim0(v) for v in e.values()})}
        print(f"[verify:{worker}] FULL last_token hooks_on={mode:7s} "
              f"n_gen={dim[mode]['n']} layers={dim[mode]['layers']} passes={dim[mode]['passes']}")
    Gd = dim["both"]["n"]

    ok, why = True, []
    for m in ("prefill", "decode", "both"):
        if dim[m]["layers"] != len(LAYERS):
            ok = False; why.append(f"{m}: {dim[m]['layers']}/{len(LAYERS)} layers")
    if dim["prefill"]["passes"] != [1]:
        ok = False; why.append(f"prefill passes {dim['prefill']['passes']} != [1]")
    if dim["decode"]["passes"] != [Gd - 1]:
        ok = False; why.append(f"decode passes {dim['decode']['passes']} != [{Gd-1}]")
    if dim["both"]["passes"] != [Gd]:
        ok = False; why.append(f"both passes {dim['both']['passes']} != [{Gd}]")

    # ---- (2) INDEPENDENT ground truth: teacher forcing (all_tokens) ----------
    # gen run: capture every decode step's value (all_tokens, both).
    cfg2 = (_cfg("qk_all.json", {"model_info": {"name": MODEL},
                                 "params": {"important_heads": [[L, 0] for L in LAYERS]},
                                 "hookq": {"hookq_mode": "all_tokens"}}) if worker == "qk"
            else _cfg("hs_all.json", {"model_info": {"name": MODEL},
                                      "hidden_states": {"layers": LAYERS, "mode": "all_tokens"}}))
    llm.load_config(cfg2)  # switch to all_tokens (env/opts already set)

    gout = llm.generate(PROMPT, _sp(hooks_on="both"))[0]
    p_ids = list(gout.prompt_token_ids)
    g_ids = list(gout.outputs[0].token_ids)
    P, Gn = len(p_ids), len(g_ids)
    gen_e = _entries(getattr(gout, "probes", None), cache_key)

    # reference: [prompt + generated[:-1]] as ONE prefill, capture every position.
    ref_prompt = {"prompt_token_ids": p_ids + g_ids[:Gn - 1]}
    rout = llm.generate([ref_prompt], _sp(hooks_on="prefill", max_tokens=1))[0]
    ref_e = _entries(getattr(rout, "probes", None), cache_key)
    print(f"[verify:{worker}] teacher-forcing P={P} G={Gn} tol={TOL} "
          f"gen_layers={len(gen_e)} ref_layers={len(ref_e)}")

    worst = 0.0
    checked = 0
    for m in gen_e:
        gt = gen_e[m].get(key0)          # [passes, maxlen, hidden]  (all_tokens both)
        rt = ref_e.get(m, {}).get(key0)  # [1, P+G-1, hidden]        (all_tokens prefill)
        if not (isinstance(gt, torch.Tensor) and isinstance(rt, torch.Tensor)):
            ok = False; why.append(f"{m}: missing tensor gen/ref"); continue
        ref = rt[0].float()              # position -> vector
        # decode pass j (j=1..Gn-1) captured the token at sequence position P+j-1.
        for j in range(1, Gn):
            pos = P + j - 1
            if j >= gt.shape[0] or pos >= ref.shape[0]:
                ok = False; why.append(f"{m}: shape short (pass {j}, pos {pos})"); break
            got = gt[j, 0, :].float()    # decode pass -> single row
            d = (got - ref[pos]).abs().max().item()
            worst = max(worst, d)
            checked += 1
            if not torch.allclose(got, ref[pos], rtol=TOL, atol=TOL):
                ok = False
                why.append(f"{m}: decode step {j} (pos {pos}) mismatch max|d|={d:.3g}")
    print(f"[verify:{worker}] teacher-forcing checked {checked} decode positions, "
          f"worst max|Δ|={worst:.3g} (tol={TOL})")

    if ok:
        print(f"[verify:{worker}] VERDICT: PASS — FULL-graph decode capture fires at "
              f"decode AND matches the model's own computation at every decode position "
              f"(worst |Δ|={worst:.3g})")
    else:
        print(f"[verify:{worker}] VERDICT: FAIL — {'; '.join(why)}")
    return ok


# ---------------------------------------------------------------- steer -------
def verify_steer():
    cfg = _cfg("steer.json", {"model_info": {"name": MODEL},
                              "steering": {"method": "add_vector",
                                           "coefficient": float(os.environ.get("VERIFY_STEER_COEFF", "20.0")),
                                           "optimal_layer": 14,
                                           "vector_path": "steering_vectors/qwen2_dummy.pt",
                                           "apply_at_all_positions": True}})
    llm = HookLLM(model=MODEL, worker_name="steer_hook_act", analyzer_name=None, config_file=cfg,
                  download_dir="./cache/", gpu_memory_utilization=0.6, max_model_len=2048,
                  enable_hook=True, enable_prefix_caching=False, **GRAPH_KW)

    base = (_read_counters(llm) or {}).get("steer.fire", 0)
    off = llm.generate(PROMPT, _sp(), use_hook=False)[0]
    t_off = list(off.outputs[0].token_ids)
    fire_off = (_read_counters(llm) or {}).get("steer.fire", 0)

    on = llm.generate(PROMPT, _sp(), use_hook=True)[0]
    t_on = list(on.outputs[0].token_ids)
    c2 = _read_counters(llm)
    fire_on = (c2 or {}).get("steer.fire", 0) if c2 else None
    n = len(t_on)
    print(f"[verify:steer] FULL base_fire={base} fire_off={fire_off} "
          f"fire_on_total={fire_on} n_gen={n}")
    print(f"[verify:steer] tokens off={t_off}")
    print(f"[verify:steer] tokens on ={t_on}")
    changed = sum(a != b for a, b in zip(t_on, t_off))

    ok, why = True, []
    # In FULL mode the steer op is baked into the decode graph; PROF.incr is not
    # reachable from inside the replayed graph, so steer.fire is NOT a reliable
    # decode signal here (unlike eager). The observable is the OUTPUT.
    if t_on == t_off:
        ok = False; why.append("steered output identical to unsteered (ineffective)")
    if changed == 0:
        ok = False; why.append("no token changed")
    if ok:
        print(f"[verify:steer] VERDICT: PASS — FULL-graph steering changes decode output "
              f"({changed}/{min(len(t_on),len(t_off))} tokens differ). Token-for-token decode "
              f"parity vs eager is the steer_parity_full oracle.")
    else:
        print(f"[verify:steer] VERDICT: FAIL — {'; '.join(why)}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, choices=["qk", "hs", "steer"])
    a = ap.parse_args()
    good = verify_steer() if a.worker == "steer" else verify_capture(a.worker)
    sys.exit(0 if good else 1)
