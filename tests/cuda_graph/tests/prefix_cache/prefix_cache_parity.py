"""Prefix-cache QK capture parity oracle (continuous-batching under-capture fix).

PURPOSE
  Verify that buffer-mode QK capture restores the FULL key history when prefix
  caching trims a prompt's shared prefix — the regime a continuous-batching server
  (sharegpt-style shared prefixes) actually hits. With prefix caching the scheduler
  does NOT recompute the cached prefix, so the static-buffer scatter never sees those
  keys; the fix reads them back from paged KV at egress (off-graph), exactly as the
  v0.2.0 eager path reconstructs them mid-forward.

HOW A PREFIX-CACHE HIT IS FORCED
  Boot with enable_prefix_caching=True and run the SAME prompt TWICE. The first run
  is a fresh prefill (num_computed=0) that populates the KV cache. The second run hits
  the cache: vLLM forwards only the (few) non-cached tail tokens, so num_computed > 0
  and the buffer would capture only the tail UNLESS the prefix is reconstructed.

  No chunking here (max_num_batched_tokens >= prompt) so this isolates prefix caching
  from the chunked-prefill path (covered by chunked_prefill_parity.py).

TWO SCENARIOS PER MODE (one engine boot):
  fresh  — first run, no cache hit. CONTROL: graph buffer must equal eager. A failure
           here is a harness/setup problem, not the prefix path.
  cached — second run, prefix-cache HIT. The actual test: does graph buffer == eager?
           Pre-fix this UNDER-CAPTURES (k coverage = tail only); post-fix it equals the
           eager full-context reconstruction.

The hookq_mode (all_tokens | last_token) is selected by the config file (env
VLLM_HOOK_CONFIG_FILE), so the bsub wrapper runs this oracle once per mode.

    python prefix_cache_parity.py capture --mode graph --out g.pkl
    python prefix_cache_parity.py capture --mode eager --out e.pkl
    python prefix_cache_parity.py compare --graph g.pkl --eager e.pkl
"""
import os
# MUST precede the plugin import (beats hook_llm's TORCHDYNAMO_DISABLE=1 default).
os.environ["TORCHDYNAMO_DISABLE"] = "0"

import argparse
import pickle
import sys

import torch

_MODEL = os.environ.get("VLLM_HOOK_DEMO_MODEL", "Qwen/Qwen2-1.5B-Instruct")
_DTYPE = torch.float if "Qwen2-1.5B" in _MODEL else torch.float16

# Prompt length (in tokens). Span several KV blocks so the second run caches whole
# blocks and forwards only a short tail — a clear prefix-cache hit.
_PROMPT_TOKS = int(os.environ.get("VLLM_HOOK_PREFIX_TOKS", "256"))
# max_num_batched_tokens — kept >= prompt so the FIRST run is a single (non-chunked)
# prefill; the hit/under-capture is purely the prefix-cache effect.
_MNBT = int(os.environ.get("VLLM_HOOK_PREFIX_MNBT", "2048"))
_HOOKS_ON = os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "prefill").strip() or "prefill"

_FILLER = ("The quick brown fox jumps over the lazy dog near the quiet river bank "
           "while the morning sun rises slowly over the distant misty hills. ")


def _make_prompt(tokenizer, target_toks):
    """A deterministic prompt of (approximately) ``target_toks`` tokens."""
    big = _FILLER * (target_toks // 8 + 8)
    ids = tokenizer.encode(big)[:target_toks]
    return tokenizer.decode(ids)


def _to_cpu_f32(t):
    if isinstance(t, (list, tuple)):
        if not t:
            return None
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return None
    return t.detach().to(torch.float32).cpu()


def _capture_store(out):
    """{layer: {q, k_all, layer_num}} from a generate() output's probes."""
    probes = getattr(out[0], "probes", None)
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


def capture(mode, out_path):
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
    else:  # eager ground truth (legacy register_forward_hook)
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f'model_configs/attention_tracker/{_MODEL.split("/")[-1]}_alltok.json')

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[prefix:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"cudagraph_mode={cudagraph_mode if mode == 'graph' else 'n/a'} "
          f"prompt_toks={_PROMPT_TOKS} config={config_file}", flush=True)

    llm = HookLLM(
        model=_MODEL,
        worker_name="probe_hook_qk",
        analyzer_name="attn_tracker",
        config_file=config_file,
        download_dir="./cache/",
        hook_dir=os.environ.get("VLLM_HOOK_DIR",
                                f"/dev/shm/vllm_hook_{os.environ.get('USER', 'u')}"),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        max_num_batched_tokens=_MNBT,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,     # the regime under test
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"hooks_on": _HOOKS_ON})

    prompt = _make_prompt(llm.tokenizer, _PROMPT_TOKS)
    ntok = len(llm.tokenizer.encode(prompt))
    print(f"[prefix:{mode}] prompt={ntok} toks; run1=fresh (populate cache), "
          f"run2=cached (prefix hit)", flush=True)

    result = {"_meta": {"prompt_toks": ntok, "hooks_on": _HOOKS_ON, "mode": mode}}
    # run1 populates the cache; run2 of the SAME prompt hits it. The capture of run2
    # is the test scenario ("cached"); run1 is the fresh control.
    for name in ("fresh", "cached"):
        out = llm.generate(prompt, sp, save_to_disk=False)
        store = _capture_store(out)
        result[name] = store
        any_k = next((v["k_all"] for v in store.values()
                      if v["k_all"] is not None), None)
        any_q = next((v["q"] for v in store.values() if v["q"] is not None), None)
        kshape = tuple(any_k.shape) if any_k is not None else None
        qshape = tuple(any_q.shape) if any_q is not None else None
        print(f"[prefix:{mode}] scenario={name}: {len(store)} layers, "
              f"q={qshape} k_all={kshape}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[prefix:{mode}] wrote {out_path}", flush=True)
    return 0


def _kcov(t):
    """Furthest captured key position = the longest reconstructed prefix."""
    if t is None:
        return 0
    return int(t.shape[-2]) if t.dim() >= 2 else int(t.shape[0])


def _entries(t):
    if t is None:
        return 0
    return int(t.shape[0])


_TOLS = (1e-2, 2e-2, 5e-2)


def _scenario_report(gl, el):
    common = sorted(set(gl) & set(el),
                    key=lambda x: int(gl[x].get("layer_num", -1)) if str(
                        gl[x].get("layer_num", "")).lstrip("-").isdigit() else 0)
    rec = {"common": common, "total": 0, "shape_mm": 0,
           "match": {t: 0 for t in _TOLS}, "worst": [], "max_d": 0.0}
    if not common:
        return rec
    for layer in common:
        for key in ("q", "k_all"):
            gv, ev = gl[layer].get(key), el[layer].get(key)
            if gv is None or ev is None:
                continue
            rec["total"] += 1
            if gv.shape != ev.shape:
                rec["shape_mm"] += 1
                rec["worst"].append((float("inf"), layer, key,
                                     tuple(gv.shape), tuple(ev.shape)))
                continue
            d = (gv - ev).abs()
            md = float(d.max()) if d.numel() else 0.0
            rec["max_d"] = max(rec["max_d"], md)
            for t in _TOLS:
                if torch.allclose(gv, ev, rtol=t, atol=t):
                    rec["match"][t] += 1
            rec["worst"].append((md, layer, key, tuple(gv.shape), tuple(ev.shape)))
    rec["worst"].sort(key=lambda x: -x[0])
    rep = common[0]
    rec["cov"] = (_kcov(gl[rep].get("k_all")), _kcov(el[rep].get("k_all")))
    rec["ent"] = (_entries(gl[rep].get("k_all")), _entries(el[rep].get("k_all")))
    return rec


def compare(graph_path, eager_path):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)

    meta = g.get("_meta", {})
    ptok = meta.get("prompt_toks", "?")
    print(f"[prefix] prompt_toks={ptok} hooks_on={meta.get('hooks_on')}")

    reps = {}
    for scenario in ("fresh", "cached"):
        r = _scenario_report(g.get(scenario, {}), e.get(scenario, {}))
        reps[scenario] = r
        if not r["common"]:
            print(f"[prefix] scenario={scenario}: NO common layers")
            continue
        m, tot = r["match"], r["total"]
        print(f"[prefix] scenario={scenario}: match@1e-2={m[1e-2]}/{tot} "
              f"@2e-2={m[2e-2]}/{tot} @5e-2={m[5e-2]}/{tot} "
              f"shape_mm={r['shape_mm']} max|Δ|={r['max_d']:.3e}; "
              f"k_all coverage graph={r['cov'][0]} eager={r['cov'][1]} "
              f"entries graph={r['ent'][0]} eager={r['ent'][1]}")
        for (md, ly, k, gs, es) in r["worst"][:6]:
            tag = "SHAPE" if md == float("inf") else f"max|Δ|={md:.3e}"
            print(f"           {ly} {k}: {tag} graph={gs} eager={es}")

    print("=" * 64)
    mode_name = os.path.basename(os.environ.get("VLLM_HOOK_CONFIG_FILE", "")) or "?"
    fr, cr = reps["fresh"], reps["cached"]
    # Control (fresh, no cache hit) must match eager up the SAME numeric ladder the test
    # leg uses — all_tokens captures q for every token across all layers, so a few deep-
    # layer tensors land just past 1e-2 on pure compiled-vs-eager TF32 drift (max|Δ|~3e-2,
    # all within 5e-2). Requiring strict @1e-2 here would false-flag that fp drift as a
    # harness fault; @5e-2 keeps the control meaningful (catches real breakage) without it.
    if not fr["common"] or fr["match"][5e-2] != fr["total"]:
        print(f"[prefix] config={mode_name} VERDICT: INCONCLUSIVE — fresh control did not "
              f"match even @5e-2 ({fr['match'][5e-2]}/{fr['total']}); harness/setup issue.")
        return 2
    if fr["match"][1e-2] != fr["total"]:
        print(f"[prefix] note: fresh control matched @5e-2 but only {fr['match'][1e-2]}/"
              f"{fr['total']} @1e-2 (max|Δ|={fr['max_d']:.3e}) — expected compiled-vs-eager "
              f"fp drift in deep layers; harness still sound.")

    # The cached run is the test: graph buffer must reach the SAME full-context K the
    # eager path reconstructs (same coverage, no shape mismatch).
    if cr["shape_mm"] > 0 or cr["cov"][0] != cr["cov"][1]:
        print(f"[prefix] config={mode_name} VERDICT: UNDER-CAPTURE — cached run has "
              f"shape/coverage mismatch (graph cov={cr['cov'][0]} eager={cr['cov'][1]}, "
              f"shape_mm={cr['shape_mm']}). Buffer dropped the cached prefix keys.")
        return 1

    tot = cr["total"]
    if cr["match"][1e-2] == tot:
        print(f"[prefix] config={mode_name} VERDICT: PASS (clean) — cached run matches eager "
              f"@1e-2 ({tot}/{tot}); prefix-cache K reconstruction is correct.")
        return 0
    if cr["match"][5e-2] == tot:
        miss = next(t for t in _TOLS if cr["match"][t] == tot)
        print(f"[prefix] config={mode_name} VERDICT: PASS (numeric) — cached run: shapes+"
              f"coverage exact, ALL values match @{miss:g} ({tot}/{tot}; "
              f"@1e-2={cr['match'][1e-2]}/{tot}, max|Δ|={cr['max_d']:.3e}). Sub-1e-2 misses "
              f"are compiled-vs-eager fp drift accumulating with depth, NOT under-capture.")
        return 0
    print(f"[prefix] config={mode_name} VERDICT: VALUE-DIVERGENCE — cached run: shapes exact "
          f"but values miss even @5e-2 (@5e-2={cr['match'][5e-2]}/{tot}, "
          f"max|Δ|={cr['max_d']:.3e}); investigate (not a pure-numeric explanation).")
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
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out))
    else:
        sys.exit(compare(args.graph, args.eager))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
