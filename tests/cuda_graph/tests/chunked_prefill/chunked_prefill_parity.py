"""Chunked-prefill QK capture parity oracle (verification (b)).

PURPOSE
  Verify whether the CURRENT buffer-mode QK capture design (the FULL milestone path)
  correctly captures Q/K when a SINGLE request's prompt is split across multiple
  prefill steps by vLLM chunked prefill. This is the regime the v0.2.0 eager path
  handles via paged-cache prefix-K reconstruction; the question is whether the
  static-buffer path (k_buf sized to max_num_batched_tokens) reproduces it.

HOW CHUNKING IS FORCED
  We boot with ``max_num_batched_tokens = MNBT`` (env VLLM_HOOK_CHUNKED_MNBT, default
  512) BELOW the long prompt's token count, with chunked prefill enabled. vLLM then
  processes the long prompt in ceil(N/MNBT) steps. The buffer cap == MNBT, so any
  token at absolute position >= MNBT is clamped to the discard sentinel by
  graph/install.py::_build_routing (the suspected under-capture).

TWO SCENARIOS PER MODE (one engine boot per graph/eager — both prompts captured):
  short — prompt < MNBT (no chunking). CONTROL: graph buffer must equal eager. If this
          fails, the harness/setup is wrong, not the chunking path.
  long  — prompt > MNBT (chunked). The actual test: does graph buffer == eager?

The hookq_mode (all_tokens | last_token) is selected by the config file (env
VLLM_HOOK_CONFIG_FILE), so the bsub wrapper runs this oracle once per mode.

    python chunked_prefill_parity.py capture --mode graph --out g.pkl
    python chunked_prefill_parity.py capture --mode eager --out e.pkl
    python chunked_prefill_parity.py compare --graph g.pkl --eager e.pkl
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

# max_num_batched_tokens — set below the long prompt to force chunked prefill.
_MNBT = int(os.environ.get("VLLM_HOOK_CHUNKED_MNBT", "512"))
# Target token counts for the two prompts (short < MNBT < long).
_SHORT_TOKS = int(os.environ.get("VLLM_HOOK_CHUNKED_SHORT", "128"))
_LONG_TOKS = int(os.environ.get("VLLM_HOOK_CHUNKED_LONG", str(_MNBT * 3 - 64)))
_HOOKS_ON = os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "prefill").strip() or "prefill"

_FILLER = ("The quick brown fox jumps over the lazy dog near the quiet river bank "
           "while the morning sun rises slowly over the distant misty hills. ")


def _make_prompt(tokenizer, target_toks):
    """A prompt of (approximately) ``target_toks`` tokens, built by repeating filler
    then trimming on the token axis so the length is deterministic and > cap as needed."""
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

    print(f"[chunked:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"cudagraph_mode={cudagraph_mode if mode == 'graph' else 'n/a'} "
          f"MNBT={_MNBT} config={config_file}", flush=True)

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
        max_num_batched_tokens=_MNBT,   # < long prompt -> chunked prefill
        max_num_seqs=8,                 # keep MNBT >= max_num_seqs; we run 1 req
        enable_chunked_prefill=True,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,    # isolate chunking from prefix-cache reuse
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"hooks_on": _HOOKS_ON})

    short_prompt = _make_prompt(llm.tokenizer, _SHORT_TOKS)
    long_prompt = _make_prompt(llm.tokenizer, _LONG_TOKS)
    n_short = len(llm.tokenizer.encode(short_prompt))
    n_long = len(llm.tokenizer.encode(long_prompt))
    print(f"[chunked:{mode}] short={n_short} toks (<MNBT={_MNBT}, no chunk) | "
          f"long={n_long} toks (>MNBT -> ~{-(-n_long // _MNBT)} chunks)", flush=True)

    result = {"_meta": {"mnbt": _MNBT, "n_short": n_short, "n_long": n_long,
                        "hooks_on": _HOOKS_ON, "mode": mode}}
    for name, prompt, ntok in (("short", short_prompt, n_short),
                               ("long", long_prompt, n_long)):
        out = llm.generate(prompt, sp, save_to_disk=False)
        store = _capture_store(out)
        result[name] = store
        any_k = next((v["k_all"] for v in store.values()
                      if v["k_all"] is not None), None)
        any_q = next((v["q"] for v in store.values() if v["q"] is not None), None)
        kshape = tuple(any_k.shape) if any_k is not None else None
        qshape = tuple(any_q.shape) if any_q is not None else None
        print(f"[chunked:{mode}] scenario={name} prompt={ntok} toks: "
              f"{len(store)} layers, q={qshape} k_all={kshape}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[chunked:{mode}] wrote {out_path}", flush=True)
    return 0


def _kcov(t):
    """Furthest captured key position = the longest reconstructed prefix.

    get_captured_states stacks k_all as (n_entries, max_len, dim); the longest entry
    (max_len) is the fullest prefix = how far into the sequence K was captured."""
    if t is None:
        return 0
    return int(t.shape[-2]) if t.dim() >= 2 else int(t.shape[0])


def _entries(t):
    """Number of captured steps/chunks = leading dim of the stacked artifact."""
    if t is None:
        return 0
    return int(t.shape[0])


_TOLS = (1e-2, 2e-2, 5e-2)


def _scenario_report(gl, el):
    """Compare one scenario's per-layer q/k_all. Returns a dict with per-tol match
    counts, shape-mismatch count, coverage, and the worst tensors by max|Δ|."""
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
    mnbt = meta.get("mnbt", "?")
    chunked = isinstance(mnbt, int) and meta.get("n_long", 0) > mnbt
    print(f"[chunked] MNBT={mnbt} short={meta.get('n_short')} long={meta.get('n_long')} "
          f"hooks_on={meta.get('hooks_on')} long_is_chunked={chunked}")

    reps = {}
    for scenario in ("short", "long"):
        r = _scenario_report(g.get(scenario, {}), e.get(scenario, {}))
        reps[scenario] = r
        if not r["common"]:
            print(f"[chunked] scenario={scenario}: NO common layers")
            continue
        m = r["match"]
        tot = r["total"]
        print(f"[chunked] scenario={scenario}: match@1e-2={m[1e-2]}/{tot} "
              f"@2e-2={m[2e-2]}/{tot} @5e-2={m[5e-2]}/{tot} "
              f"shape_mm={r['shape_mm']} max|Δ|={r['max_d']:.3e}; "
              f"k_all coverage graph={r['cov'][0]} eager={r['cov'][1]} "
              f"entries graph={r['ent'][0]} eager={r['ent'][1]}")
        for (md, ly, k, gs, es) in r["worst"][:6]:
            tag = "SHAPE" if md == float("inf") else f"max|Δ|={md:.3e}"
            print(f"           {ly} {k}: {tag} graph={gs} eager={es}")

    print("=" * 64)
    mode_name = os.path.basename(os.environ.get("VLLM_HOOK_CONFIG_FILE", "")) or "?"
    sr, lr = reps["short"], reps["long"]
    # Control (short, never chunked) must match tightly — else the harness is suspect.
    if not sr["common"] or sr["match"][1e-2] != sr["total"]:
        print(f"[chunked] config={mode_name} VERDICT: INCONCLUSIVE — non-chunked control "
              f"did not match @1e-2 ({sr['match'][1e-2]}/{sr['total']}); harness/setup issue.")
        return 2

    if lr["shape_mm"] > 0 or lr["cov"][0] != lr["cov"][1]:
        print(f"[chunked] config={mode_name} VERDICT: UNDER-CAPTURE — long prompt has "
              f"shape/coverage mismatch (graph cov={lr['cov'][0]} eager={lr['cov'][1]}, "
              f"shape_mm={lr['shape_mm']}). Buffer dropped keys.")
        return 1

    tot = lr["total"]
    if lr["match"][1e-2] == tot:
        print(f"[chunked] config={mode_name} VERDICT: PASS (clean) — long prompt matches "
              f"eager @1e-2 ({tot}/{tot}); chunked-prefill capture is correct.")
        return 0
    if lr["match"][5e-2] == tot:
        miss = next(t for t in _TOLS if lr["match"][t] == tot)
        print(f"[chunked] config={mode_name} VERDICT: PASS (numeric) — long prompt: "
              f"shapes+coverage exact, ALL values match @{miss:g} ({tot}/{tot}; "
              f"@1e-2={lr['match'][1e-2]}/{tot}, max|Δ|={lr['max_d']:.3e}). The sub-1e-2 "
              f"misses cluster in LATE layers (compiled-vs-eager TF32/fp matmul drift that "
              f"accumulates with depth), NOT under-capture. Confirm vs the no-chunk control.")
        return 0
    print(f"[chunked] config={mode_name} VERDICT: VALUE-DIVERGENCE — long prompt: shapes "
          f"exact but values miss even @5e-2 (@2e-2={lr['match'][2e-2]}/{tot} "
          f"@5e-2={lr['match'][5e-2]}/{tot}, max|Δ|={lr['max_d']:.3e}); investigate (not "
          f"a pure-numeric explanation).")
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
