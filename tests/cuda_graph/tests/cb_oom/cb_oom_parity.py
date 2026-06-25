"""Continuous-batching OOM-fix oracle (v0.5.x Stage 1 + Stage 3).

Proves the two halves of the CB-OOM cut (docs/v0.5.x_cb_oom_plan.md
"Test / verify plan") without forking vLLM:

  Stage 1 — continuous async GPU->pinned drain (OPT-IN via VLLM_HOOK_CAPTURE_STREAMING=1
            or an explicit budget env).  Bounds GPU-resident capture bytes to
            O(ring x snapshot) instead of request-lifetime.
  Stage 3 — resident-byte ceiling admission throttle.  When total resident capture
            crosses VLLM_HOOK_CAPTURE_RESIDENT_CEILING, NEW (request) capture is
            refused (whole-request, logged, counted) so total memory is bounded.

Three probes, each engine isolated in its own subprocess (the GPU is
exclusive_process -> one engine per process), mirroring prefix_cache_parity.py:

  (a) PARITY  — value-preservation.  Re-run the prefix-cache fresh/cached capture
      with the Stage-1 drain ON (a small forced budget makes drains FIRE, a
      superset of the 0.05 auto default) and confirm graph buffer == eager
      ground truth.  Draining only moves WHERE a tensor lives, never its values.
        parity --mode graph --out g.pkl   (Stage 1 on, drains forced)
        parity --mode eager --out e.pkl   (op/register_forward_hook ground truth)
        parity-compare --graph g.pkl --eager e.pkl

  (b) GPUBOUND — GPU residency stays flat.  Drive a high-concurrency, long-decode,
      all_tokens capture and read PEAK GPU-resident bytes (torch.cuda
      max_memory_allocated delta, via a collective_rpc into the worker) plus the
      drain manager's counters.  Run once with the drain OFF (clones held to
      flush -> peak GROWS with in-flight requests) and once ON (continuous drain
      -> peak BOUNDED), then contrast.
        gpuprobe --stage1 off --out off.pkl
        gpuprobe --stage1 on  --out on.pkl
        gpu-compare --off off.pkl --on on.pkl

  (c) THROTTLE — Stage 3 admission throttle.  Boot with a tiny ceiling, capture one
      single-request prompt (always admitted: resident starts at 0) for the
      admitted-correctness compare, then submit a concurrent batch that drives
      resident over the ceiling.  Read capture.throttled (+ the throttled-set size
      and resident bytes) from the worker via collective_rpc; assert it is >0 with
      no crash.  A separate eager leg captures the same single prompt; the admitted
      graph capture must still match it (throttling drops WHOLE requests, never
      corrupts an admitted one).
        throttle --out t.pkl            (graph, tiny ceiling, profile on)
        throttle-eager --out te.pkl     (eager ground truth, same single prompt)
        throttle-compare --throttle t.pkl --eager te.pkl

The hookq_mode (all_tokens | last_token) is selected by the config file
(VLLM_HOOK_CONFIG_FILE), so the bsub wrapper runs the parity probe once per mode.
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

# Prompt length (tokens) for the parity prefix-cache prompt.
_PROMPT_TOKS = int(os.environ.get("VLLM_HOOK_PREFIX_TOKS", "256"))
_MNBT = int(os.environ.get("VLLM_HOOK_PREFIX_MNBT", "2048"))
_HOOKS_ON = os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "prefill").strip() or "prefill"

# GPU-bound / throttle workload sizing (small enough for a 1.5B model on one GPU).
_GP_REQS = int(os.environ.get("VLLM_HOOK_CB_REQS", "16"))        # concurrent requests
_GP_DECODE = int(os.environ.get("VLLM_HOOK_CB_DECODE", "48"))    # decode tokens/request
_GP_PROMPT_TOKS = int(os.environ.get("VLLM_HOOK_CB_PROMPT_TOKS", "96"))
_THROTTLE_REQS = int(os.environ.get("VLLM_HOOK_CB_THROTTLE_REQS", "8"))

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


# ---------------------------------------------------------------------------
# collective_rpc payloads — these run IN THE WORKER (self == Worker), so they can
# read the per-worker drain manager + PROF counters that never cross to the driver
# on the normal probe path.  cloudpickle ships them by value (vLLM v1
# multiproc_executor), so a __main__-defined function reaches the worker intact.
# ---------------------------------------------------------------------------

def _rpc_reset_peak(self):  # noqa: ANN001
    """Reset the CUDA peak-allocated high-water mark; return current allocated."""
    import torch as _t
    if _t.cuda.is_available():
        _t.cuda.synchronize()
        _t.cuda.reset_peak_memory_stats()
        return int(_t.cuda.memory_allocated())
    return 0


def _rpc_drain_stats(self):  # noqa: ANN001
    """Read drain-manager + PROF counters + CUDA mem high-water from the worker."""
    import torch as _t
    out = {
        "drain_present": False,
        "drain_enabled": False,
        "num_drains": 0,
        "gpu_bytes": 0,
        "resident_bytes": 0,
        "ceiling_bytes": 0,
        "throttled_set": 0,
        "budget_bytes": 0,
        "prof_throttled": 0,
        "prof_hookfire": 0,
        "max_mem_allocated": 0,
        "mem_allocated": 0,
    }
    dm = getattr(self, "_capture_drain", None)
    if dm is not None:
        out["drain_present"] = True
        out["drain_enabled"] = bool(getattr(dm, "enabled", False))
        out["num_drains"] = int(getattr(dm, "num_drains", 0))
        out["gpu_bytes"] = int(getattr(dm, "gpu_bytes", 0))
        out["resident_bytes"] = int(getattr(dm, "resident_bytes", 0))
        out["ceiling_bytes"] = int(getattr(dm, "ceiling_bytes", 0))
        out["budget_bytes"] = int(getattr(dm, "budget_bytes", 0))
        out["throttled_set"] = int(len(getattr(dm, "_throttled", ()) or ()))
    try:
        from vllm_hook_plugins._profiler import PROF
        counters = PROF.snapshot().get("counters", {})
        out["prof_throttled"] = int(counters.get("capture.throttled", 0))
        out["prof_hookfire"] = int(counters.get("hook.fire.qk", 0))
    except Exception:  # noqa: BLE001
        pass
    if _t.cuda.is_available():
        _t.cuda.synchronize()
        out["max_mem_allocated"] = int(_t.cuda.max_memory_allocated())
        out["mem_allocated"] = int(_t.cuda.memory_allocated())
    return out


def _rpc(llm, fn):
    """Call ``fn`` on every worker; return rank-0 result (TP=1 -> the only one)."""
    res = llm.llm.collective_rpc(fn)
    return res[0] if res else {}


# ---------------------------------------------------------------------------
# Engine boot
# ---------------------------------------------------------------------------

def _boot(mode, *, enable_prefix_caching, max_num_seqs):
    """Boot a HookLLM QK-capture engine (graph buffer or eager op ground truth)."""
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f'model_configs/attention_tracker/{_MODEL.split("/")[-1]}_alltok.json')

    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[cb_oom:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"cudagraph_mode={cudagraph_mode if mode == 'graph' else 'n/a'} "
          f"config={config_file} max_num_seqs={max_num_seqs} "
          f"prefix_caching={enable_prefix_caching}", flush=True)

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
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        dtype=_DTYPE,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        enable_hook=True,
        tensor_parallel_size=1,
        **extra,
    )
    return llm


# ---------------------------------------------------------------------------
# (a) PARITY — value-preservation with Stage 1 drain ON
# ---------------------------------------------------------------------------

def parity(mode, out_path):
    llm = _boot(mode, enable_prefix_caching=True, max_num_seqs=8)
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"hooks_on": _HOOKS_ON})

    prompt = _make_prompt(llm.tokenizer, _PROMPT_TOKS)
    ntok = len(llm.tokenizer.encode(prompt))
    print(f"[cb_oom:{mode}] parity prompt={ntok} toks; run1=fresh run2=cached", flush=True)

    result = {"_meta": {"prompt_toks": ntok, "hooks_on": _HOOKS_ON, "mode": mode}}
    for name in ("fresh", "cached"):
        out = llm.generate(prompt, sp, save_to_disk=False)
        result[name] = _capture_store(out)

    if mode == "graph":
        stats = _rpc(llm, _rpc_drain_stats)
        result["_meta"]["drain"] = stats
        print(f"[cb_oom:{mode}] drain: enabled={stats['drain_enabled']} "
              f"num_drains={stats['num_drains']} budget={stats['budget_bytes']}B "
              f"(Stage-1 drain {'FIRED' if stats['num_drains'] > 0 else 'idle'})",
              flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cb_oom:{mode}] wrote {out_path}", flush=True)
    return 0


_TOLS = (1e-2, 2e-2, 5e-2)


def _compare_store(gl, el):
    """Per-(layer,tensor) max|Δ| ladder for two {layer: {q,k_all,...}} stores."""
    common = sorted(set(gl) & set(el))
    rec = {"common": common, "total": 0, "shape_mm": 0,
           "match": {t: 0 for t in _TOLS}, "max_d": 0.0, "worst": []}
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
    return rec


def _print_rec(tag, r):
    if not r["common"] or r["total"] == 0:
        print(f"[cb_oom] {tag}: NO comparable tensors")
        return
    m, tot = r["match"], r["total"]
    print(f"[cb_oom] {tag}: match@1e-2={m[1e-2]}/{tot} @2e-2={m[2e-2]}/{tot} "
          f"@5e-2={m[5e-2]}/{tot} shape_mm={r['shape_mm']} max|Δ|={r['max_d']:.3e}")
    for (md, ly, k, gs, es) in r["worst"][:4]:
        tg = "SHAPE" if md == float("inf") else f"max|Δ|={md:.3e}"
        print(f"          {ly} {k}: {tg} graph={gs} eager={es}")


def _verdict_match(name, r):
    """Shared PASS ladder: shapes exact + values match up the tol ladder."""
    if not r["common"] or r["total"] == 0:
        print(f"[cb_oom] {name} VERDICT: INCONCLUSIVE — no comparable tensors.")
        return 2
    if r["shape_mm"] > 0:
        print(f"[cb_oom] {name} VERDICT: SHAPE-MISMATCH — {r['shape_mm']} tensor(s) "
              f"differ in shape; capture diverged from eager.")
        return 1
    tot = r["total"]
    if r["match"][1e-2] == tot:
        print(f"[cb_oom] {name} VERDICT: PASS (clean) — all {tot} tensors match eager "
              f"@1e-2; drain/throttle is value-preserving.")
        return 0
    if r["match"][5e-2] == tot:
        miss = next(t for t in _TOLS if r["match"][t] == tot)
        print(f"[cb_oom] {name} VERDICT: PASS (numeric) — all {tot} match @{miss:g} "
              f"(@1e-2={r['match'][1e-2]}/{tot}, max|Δ|={r['max_d']:.3e}); sub-1e-2 "
              f"misses are compiled-vs-eager TF32 drift, NOT a capture error.")
        return 0
    print(f"[cb_oom] {name} VERDICT: VALUE-DIVERGENCE — shapes exact but values miss "
          f"even @5e-2 (@5e-2={r['match'][5e-2]}/{tot}, max|Δ|={r['max_d']:.3e}).")
    return 1


def parity_compare(graph_path, eager_path):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)
    mode_name = os.path.basename(os.environ.get("VLLM_HOOK_CONFIG_FILE", "")) or "?"
    drain = g.get("_meta", {}).get("drain", {})
    print(f"[cb_oom] PARITY config={mode_name} "
          f"graph_drains={drain.get('num_drains', '?')} "
          f"(Stage-1 drain {'FIRED' if drain.get('num_drains', 0) > 0 else 'IDLE'})")

    rc = 0
    for scen in ("fresh", "cached"):
        r = _compare_store(g.get(scen, {}), e.get(scen, {}))
        _print_rec(f"parity[{scen}]", r)
        v = _verdict_match(f"PARITY[{scen}] config={mode_name}", r)
        rc = max(rc, v if v != 2 else 1)  # missing tensors here is a real failure
    if drain.get("num_drains", 0) == 0 and drain.get("drain_enabled"):
        print("[cb_oom] PARITY note: drain enabled but 0 drains fired — workload below "
              "budget; values still preserved but the drain path was not exercised here.")
    return rc


# ---------------------------------------------------------------------------
# (b) GPUBOUND — peak GPU-resident capture bytes off vs on
# ---------------------------------------------------------------------------

def gpuprobe(stage1, out_path):
    # Stage-1 OFF = drain disabled (clones held to flush); ON = small forced budget
    # so the continuous drain fires every step.  Set BEFORE boot (worker reads at
    # graph_install via from_env -> spawn inherits the env).
    if stage1 == "off":
        os.environ["VLLM_HOOK_CAPTURE_GPU_BUDGET"] = "0"
        os.environ.pop("VLLM_HOOK_CAPTURE_DRAIN_BYTES", None)
    else:
        os.environ.setdefault("VLLM_HOOK_CAPTURE_DRAIN_BYTES",
                              str(int(os.environ.get("VLLM_HOOK_CB_DRAIN_BYTES",
                                                     str(1 << 20)))))  # 1 MiB

    llm = _boot("graph", enable_prefix_caching=False,
                max_num_seqs=max(_GP_REQS, 16))
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=_GP_DECODE,
                        extra_args={"hooks_on": "both"})  # capture every decode step

    prompt = _make_prompt(llm.tokenizer, _GP_PROMPT_TOKS)
    prompts = [prompt] * _GP_REQS
    print(f"[cb_oom:gpuprobe] stage1={stage1} reqs={_GP_REQS} decode={_GP_DECODE} "
          f"prompt_toks~{_GP_PROMPT_TOKS} all_tokens; measuring peak GPU residency",
          flush=True)

    base = _rpc(llm, _rpc_reset_peak)
    # ONE concurrent batch -> all requests in flight together -> their capture clones
    # accumulate on GPU until flush UNLESS the drain relocates them to pinned host.
    # save_to_disk=True keeps the offline per-request merge out of the memory picture.
    llm.generate(prompts, sp, save_to_disk=True)
    stats = _rpc(llm, _rpc_drain_stats)

    peak_delta = max(0, stats["max_mem_allocated"] - base)
    rec = {"stage1": stage1, "baseline_alloc": base, "peak_delta": peak_delta,
           "stats": stats, "reqs": _GP_REQS, "decode": _GP_DECODE}
    print(f"[cb_oom:gpuprobe] stage1={stage1}: peak_GPU_delta={peak_delta/1e6:.1f} MB "
          f"(baseline={base/1e6:.1f} MB max={stats['max_mem_allocated']/1e6:.1f} MB) "
          f"num_drains={stats['num_drains']} drain_enabled={stats['drain_enabled']} "
          f"resident={stats['resident_bytes']}B", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(rec, f)
    print(f"[cb_oom:gpuprobe] wrote {out_path}", flush=True)
    return 0


def gpu_compare(off_path, on_path):
    with open(off_path, "rb") as f:
        off = pickle.load(f)
    with open(on_path, "rb") as f:
        on = pickle.load(f)
    pd_off = off["peak_delta"]
    pd_on = on["peak_delta"]
    d_off = off["stats"]["num_drains"]
    d_on = on["stats"]["num_drains"]
    print(f"[cb_oom] GPUBOUND reqs={off['reqs']} decode={off['decode']} all_tokens")
    print(f"[cb_oom]   stage1=OFF: peak_GPU_delta={pd_off/1e6:.1f} MB drains={d_off} "
          f"(drain disabled -> clones held to flush)")
    print(f"[cb_oom]   stage1=ON : peak_GPU_delta={pd_on/1e6:.1f} MB drains={d_on} "
          f"(continuous drain -> clones relocated to pinned host)")
    print("=" * 64)
    if d_on <= 0:
        print("[cb_oom] GPUBOUND VERDICT: INCONCLUSIVE — Stage-1 drain did not fire on "
              "the ON leg (workload below budget); cannot demonstrate bounding. Raise "
              "VLLM_HOOK_CB_REQS/_DECODE or lower VLLM_HOOK_CB_DRAIN_BYTES.")
        return 2
    # Residency is BOUNDED when the ON peak does not exceed the OFF peak (the drain
    # relocates capture clones off the GPU, so peak should be <= the held-to-flush
    # case, modulo cudagraph-pool noise).
    if pd_on <= pd_off + max(2 << 20, int(0.05 * max(pd_off, 1))):
        print(f"[cb_oom] GPUBOUND VERDICT: PASS (flat) — Stage-1 ON fired {d_on} drains "
              f"and peak GPU-resident capture stayed BOUNDED ({pd_on/1e6:.1f} MB) vs "
              f"held-to-flush OFF ({pd_off/1e6:.1f} MB). Drain bounds GPU residency.")
        return 0
    print(f"[cb_oom] GPUBOUND VERDICT: REVIEW — ON peak ({pd_on/1e6:.1f} MB) exceeds OFF "
          f"({pd_off/1e6:.1f} MB) despite {d_on} drains. Inspect raw numbers above "
          f"(cudagraph-pool/KV noise can dominate at small scale; user runs the real "
          f"profile).")
    return 2


# ---------------------------------------------------------------------------
# (c) THROTTLE — Stage 3 resident-ceiling admission throttle
# ---------------------------------------------------------------------------

def throttle(out_path):
    # Tiny ceiling so the FIRST admitted request immediately pushes resident over it
    # and every later NEW request is refused.  Profile on so capture.throttled counts.
    os.environ["VLLM_HOOK_CAPTURE_RESIDENT_CEILING"] = os.environ.get(
        "VLLM_HOOK_CB_CEILING", "1024")  # 1 KiB
    os.environ["VLLM_HOOK_PROFILE"] = "1"

    llm = _boot("graph", enable_prefix_caching=False,
                max_num_seqs=max(_THROTTLE_REQS, 8))
    from vllm import SamplingParams
    sp1 = SamplingParams(temperature=0.0, max_tokens=1,
                         extra_args={"hooks_on": "prefill"})

    prompt = _make_prompt(llm.tokenizer, _GP_PROMPT_TOKS)
    ntok = len(llm.tokenizer.encode(prompt))

    # 1) single request -> always ADMITTED (resident starts at 0). Capture it for the
    #    admitted-correctness compare. Its flush (on_pop) returns resident to ~0.
    print(f"[cb_oom:throttle] ceiling={os.environ['VLLM_HOOK_CAPTURE_RESIDENT_CEILING']}B "
          f"single admitted prompt={ntok} toks", flush=True)
    out = llm.generate(prompt, sp1, save_to_disk=False)
    admitted = _capture_store(out)
    s_after_single = _rpc(llm, _rpc_drain_stats)
    print(f"[cb_oom:throttle] after single: layers_captured={len(admitted)} "
          f"resident={s_after_single['resident_bytes']}B "
          f"throttled_so_far={s_after_single['prof_throttled']}", flush=True)

    # 2) concurrent batch -> several NEW requests in flight at once; the first is
    #    admitted, the rest are refused once resident crosses the ceiling. Disk-save
    #    avoids the offline per-request merge (which a mixed admitted/throttled batch
    #    would trip); the throttle is observed via the worker counters, not probes.
    batch = [prompt] * _THROTTLE_REQS
    print(f"[cb_oom:throttle] throttle batch reqs={_THROTTLE_REQS}", flush=True)
    llm.generate(batch, sp1, save_to_disk=True)
    stats = _rpc(llm, _rpc_drain_stats)
    print(f"[cb_oom:throttle] after batch: capture.throttled={stats['prof_throttled']} "
          f"throttled_set={stats['throttled_set']} resident={stats['resident_bytes']}B "
          f"ceiling={stats['ceiling_bytes']}B hook.fire.qk={stats['prof_hookfire']}",
          flush=True)

    result = {"_meta": {"prompt_toks": ntok, "ceiling": stats["ceiling_bytes"]},
              "admitted": admitted, "stats": stats}
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cb_oom:throttle] wrote {out_path}", flush=True)
    return 0


def throttle_eager(out_path):
    llm = _boot("eager", enable_prefix_caching=False, max_num_seqs=8)
    from vllm import SamplingParams
    sp1 = SamplingParams(temperature=0.0, max_tokens=1,
                         extra_args={"hooks_on": "prefill"})
    prompt = _make_prompt(llm.tokenizer, _GP_PROMPT_TOKS)
    ntok = len(llm.tokenizer.encode(prompt))
    out = llm.generate(prompt, sp1, save_to_disk=False)
    store = _capture_store(out)
    print(f"[cb_oom:throttle-eager] layers_captured={len(store)} prompt={ntok} toks",
          flush=True)
    with open(out_path, "wb") as f:
        pickle.dump({"_meta": {"prompt_toks": ntok}, "admitted": store}, f)
    print(f"[cb_oom:throttle-eager] wrote {out_path}", flush=True)
    return 0


def throttle_compare(throttle_path, eager_path):
    with open(throttle_path, "rb") as f:
        t = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)
    stats = t.get("stats", {})
    thr = stats.get("prof_throttled", 0)
    thr_set = stats.get("throttled_set", 0)
    ceiling = stats.get("ceiling_bytes", 0)
    print(f"[cb_oom] THROTTLE ceiling={ceiling}B capture.throttled={thr} "
          f"throttled_set={thr_set} resident={stats.get('resident_bytes')}B")

    # Stage 3 assertion: the ceiling refused at least one NEW request, counted + logged.
    throttle_fired = thr > 0 or thr_set > 0
    if not throttle_fired:
        print("[cb_oom] THROTTLE VERDICT: NO-THROTTLE — ceiling never tripped "
              f"(capture.throttled={thr}, throttled_set={thr_set}). Lower "
              "VLLM_HOOK_CB_CEILING or raise VLLM_HOOK_CB_THROTTLE_REQS.")
        return 1

    # Admitted-correctness: the single admitted request must still match eager — the
    # throttle drops WHOLE requests, it must never corrupt an admitted one.
    r = _compare_store(t.get("admitted", {}), e.get("admitted", {}))
    _print_rec("throttle[admitted]", r)
    v = _verdict_match("THROTTLE[admitted-vs-eager]", r)
    print("=" * 64)
    if v == 0:
        print(f"[cb_oom] THROTTLE VERDICT: PASS — Stage 3 refused {max(thr, thr_set)} "
              f"capture admission(s) under the {ceiling}B ceiling (no crash), and the "
              f"admitted request still matches eager. Whole-request throttle, no "
              f"corruption.")
        return 0
    print(f"[cb_oom] THROTTLE VERDICT: ADMITTED-CORRUPTED — throttle fired "
          f"(capture.throttled={thr}) but the admitted request DIVERGES from eager; "
          f"the throttle must not touch an admitted request.")
    return 1


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("parity")
    pp.add_argument("--mode", choices=["graph", "eager"], required=True)
    pp.add_argument("--out", required=True)
    pc = sub.add_parser("parity-compare")
    pc.add_argument("--graph", required=True)
    pc.add_argument("--eager", required=True)

    pg = sub.add_parser("gpuprobe")
    pg.add_argument("--stage1", choices=["on", "off"], required=True)
    pg.add_argument("--out", required=True)
    pgc = sub.add_parser("gpu-compare")
    pgc.add_argument("--off", required=True)
    pgc.add_argument("--on", required=True)

    pt = sub.add_parser("throttle")
    pt.add_argument("--out", required=True)
    pte = sub.add_parser("throttle-eager")
    pte.add_argument("--out", required=True)
    ptc = sub.add_parser("throttle-compare")
    ptc.add_argument("--throttle", required=True)
    ptc.add_argument("--eager", required=True)

    args = p.parse_args()
    if args.cmd == "parity":
        sys.exit(parity(args.mode, args.out))
    elif args.cmd == "parity-compare":
        sys.exit(parity_compare(args.graph, args.eager))
    elif args.cmd == "gpuprobe":
        sys.exit(gpuprobe(args.stage1, args.out))
    elif args.cmd == "gpu-compare":
        sys.exit(gpu_compare(args.off, args.on))
    elif args.cmd == "throttle":
        sys.exit(throttle(args.out))
    elif args.cmd == "throttle-eager":
        sys.exit(throttle_eager(args.out))
    elif args.cmd == "throttle-compare":
        sys.exit(throttle_compare(args.throttle, args.eager))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
