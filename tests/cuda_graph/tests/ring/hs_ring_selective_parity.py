"""GPU oracle for Lever C (selective drain, ``VLLM_HOOK_DRAIN_SELECTIVE``, plan Task 16). Branch
``capture_ring``, base commit ``4fdab35`` (Task 15's fix wave: bound the copy list by the step
span). Answers the one question no CPU unit test can: does the CUDA branch of ``_read_segments``
(``graph/ring_drain_hs.py`` ~1148-1177) copy the RIGHT tiles and nothing else, on a real device?

WHY THIS FILE EXISTS -- read before touching it. Task 15's own reviewer found that every one of its
39 (then 74) unit tests takes the CPU branch of ``_read_segments``; the CUDA branch (stream wait,
the copy loop, ``record_stream``, event record+sync) has ZERO executed coverage on that box. The
copy-stream-ordering unit test is a SOURCE-ORDER PIN (``inspect.getsource`` + token order) -- it
catches a deleted or reordered call, never a wrong-but-present one (waiting the wrong event object,
recording on the wrong stream, a conditional that keeps the token but skips the call). This harness
is the only oracle that actually runs that branch.

READ BACK THE RIGHT WAY. The capture-ring GRAPH path never populates ``out[0].probes`` /
``get_captured_states`` -- both read EMPTY on this path for BOTH arms, which is a false "0 layers"
that reads as a pass if you don't know to avoid it ([[capture-ring-graph-retrieval-oracle]]). Ground
truth for the graph leg is ``collective_rpc("flush_ring")`` (final drain + shared sidecar) +
``graph.ring_reader.load_multilayer_ring_artifact(run_dir)``, exactly like
``tests/cuda_graph/tests/ring/hs_ring_parity.py``.

TOLERANCE. Graph vs eager is compared at ``rtol=atol=1e-2`` (allclose), NOT ``torch.equal`` --
matching every sibling harness in this directory (``hs_ring_parity.py``, ``hs_ring_hetero_
parity.py``). The reason is structural, not laziness: the graph path runs the fused Triton capture
kernel (widen to fp32, round once on store) against the eager path's aten ``hidden + residual`` (add
in hidden dtype, two roundings) -- a real, small, expected numerical difference unrelated to Lever C.
Lever C only changes WHICH rows the drain copies, never their VALUES, so if it broke something the
signature would be a SHAPE mismatch, a LAYER-SET mismatch, or a large ``max|Δ|`` -- all of which this
tolerance still catches. "Byte-identical" language elsewhere in this plan (e.g.
``VLLM_HOOK_CAPTURE_FUSED``'s 840/840 ``torch.equal`` oracle) refers to an ISOLATED op-level unit
test on a fixed input, not this end-to-end graph-vs-eager pipeline.

THE COUNTER IS THE NON-VACUITY WITNESS. A selective run that silently copied everything anyway still
reconstructs byte-identically -- Task 15's own report says so plainly ("Leg C passing while
``rows_skipped == 0`` means the flag never fired and the leg is worthless"). Every leg here therefore
calls ``collective_rpc("get_drain_row_counts")`` after ``flush_ring`` and HARD-ASSERTS its own
non-vacuity condition INSIDE the capture subprocess (raises ``RuntimeError``, nonzero exit) rather
than printing a number for a human to eyeball -- the same fail-loud non-vacuity discipline the
sibling parity oracles use, here extended to three knobs:

    VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE = "1" | "0"   -- counts()["selective"] must equal this
    VLLM_HOOK_SELPAR_REQUIRE_SKIPPED   = "zero" | "positive"  -- rows_skipped must equal / exceed 0
    VLLM_HOOK_SELPAR_REQUIRE_REFUSED   = "1"          -- selective is False AND a disabled_reason
                                                          naming "per-request" is present (leg F)
    VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE = "zero" | "positive"  -- degenerate_steps (Task 19: steps
                                                          where the armed lever found nothing to
                                                          skip and took the flag-off fast path).
                                                          `positive` on the all-layers leg proves
                                                          the fast path RUNS under a real cudagraph
                                                          workload; `zero` on a subset leg proves it
                                                          does not fire where it would silently
                                                          delete the lever's win. rows_skipped
                                                          cannot express either -- it reads 0 on
                                                          both an all-layers fast path and an
                                                          all-layers slow one.

WORKLOAD SHAPE. Up to THREE concurrent requests are submitted in ONE batched
``llm.llm.generate(prompts, sp_list)`` call (bypassing ``HookLLM.generate``'s multi-request probe
merge, same reason ``hs_ring_hetero_parity.py`` does it) -- deliberately NOT the sequential
one-case-at-a-time style ``hs_ring_parity.py`` uses. Submitting concurrently, even for the "uniform"
legs (A/B/C/E), exercises ``build_copy_plans``' range-MERGING across multiple requests' entries in
the SAME step -- the mechanism a sequential single-request-at-a-time harness would never touch, and
the one ``_merge_ranges`` exists for. Each case independently controls its own layers (``True`` = all
/ a 1-based list), ``hooks_on`` (prefill/decode/both), ``hs_mode``, and ``max_tokens`` via env vars
(``VLLM_HOOK_SELPAR_{TEXT,LAYERS,HOOKS,HSMODE,MAXTOK}_{A,B,C}``), so one file drives every leg A-G.

The plugin's own install-config JSON (``VLLM_HOOK_CONFIG_FILE``) does NOT gate which layers get
installed as ring buffers -- ``install_hs_hosts`` sizes that from the MODEL's ``num_hidden_layers``
unconditionally (confirmed by reading ``graph/install_hs.py``; the config's ``hidden_states.layers``
list is only consumed by ``HookLLM._build_extra_args``, the offline config-merge path this harness
bypasses). So every leg here uses the same all-layers/all_tokens config
(``Qwen2-1.5B-Instruct_alltok.json``) -- ALL 28 of Qwen2-1.5B-Instruct's layers install every time,
and it is each case's own ``output_hidden_states`` extra_arg that decides subset vs all. This is load-
bearing for legs C/D/E/G: ``rows_skipped`` is ``len(installed_layers) * item.n_rows - copied``, so it
can only be > 0 when MORE layers are installed than any active record names -- using a layer-limited
install config would make every subset leg vacuously report ``rows_skipped == 0``.

CROSS-ARM GATE (``crossarm``, added after the initial GPU round -- see the fix report in the plan's
Task 16 report). A per-leg ``allclose`` against eager cannot distinguish "matched because the
mechanism is right" from "matched because misaddressed contents happened to land within tolerance" --
selective drain's own failure mode (Task 15's review) is rows copied from the WRONG ring offset,
which is shape-correct and layer-set-correct. Every leg here uses the SAME case-A/case-B prompt
texts, so two legs that both capture the same case's same layer number captured the SAME underlying
hidden state through DIFFERENT mechanisms (full drain vs selective compaction) -- their per-layer
error profiles against eager must therefore agree. ``compare --profile-
out`` writes that profile; ``crossarm --a --b`` asserts two legs' profiles agree on their shared
``case::layer`` cells, using the SAME bands ``hs_ring_hetero_parity.py``'s own crossarm() carries
(see that function's docstring for why bit-identity is deliberately NOT the pass criterion).

    python hs_ring_selective_parity.py capture --mode graph --out g.pkl
    python hs_ring_selective_parity.py capture --mode eager --out e.pkl
    python hs_ring_selective_parity.py compare --graph g.pkl --eager e.pkl --profile-out p.pkl
    python hs_ring_selective_parity.py crossarm --a legA_profile.pkl --b legD_profile.pkl
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

_DEFAULT_HOOKS_ON = (os.environ.get("VLLM_HOOK_PARITY_HOOKS_ON", "both").strip() or "both")
_DEFAULT_HS_MODE = (os.environ.get("VLLM_HOOK_PARITY_HS_MODE", "all_tokens").strip() or "all_tokens")
_DEFAULT_MAX_TOKENS = int(os.environ.get("VLLM_HOOK_PARITY_MAX_TOKENS", "12"))

_NCASES = int(os.environ.get("VLLM_HOOK_SELPAR_NCASES", "2"))
if _NCASES not in (1, 2, 3):
    raise ValueError(f"VLLM_HOOK_SELPAR_NCASES must be 1, 2, or 3, got {_NCASES}")

_CASE_LETTERS = ["A", "B", "C"][:_NCASES]
_CASE_DEFAULT_TEXT = {
    "A": "The capital of France is",
    "B": "Quantum computing leverages superposition to",
    "C": "The history of the internet began with",
}


def _layers_spec(letter: str):
    """``VLLM_HOOK_SELPAR_LAYERS_<letter>`` -- "" (default) => ``True`` (all layers, the sentinel
    the routing builders already treat as "no filter"); a comma list => explicit 1-based layers."""
    raw = os.environ.get(f"VLLM_HOOK_SELPAR_LAYERS_{letter}", "").strip()
    if not raw:
        return True
    return [int(x) for x in raw.split(",") if x.strip()]


def _build_cases():
    cases = []
    for letter in _CASE_LETTERS:
        cases.append({
            "name": letter,
            "text": os.environ.get(f"VLLM_HOOK_SELPAR_TEXT_{letter}", _CASE_DEFAULT_TEXT[letter]),
            "layers": _layers_spec(letter),
            "hooks_on": (os.environ.get(f"VLLM_HOOK_SELPAR_HOOKS_{letter}", _DEFAULT_HOOKS_ON).strip()
                         or _DEFAULT_HOOKS_ON),
            "hs_mode": (os.environ.get(f"VLLM_HOOK_SELPAR_HSMODE_{letter}", _DEFAULT_HS_MODE).strip()
                        or _DEFAULT_HS_MODE),
            "max_tokens": int(os.environ.get(f"VLLM_HOOK_SELPAR_MAXTOK_{letter}",
                                              str(_DEFAULT_MAX_TOKENS))),
        })
    return cases


_CASES = _build_cases()

# Non-vacuity gates -- hard-fail INSIDE the subprocess (raise -> nonzero exit), never a printed
# number a human is trusted to notice. Mirrors hs_ring_parity.py's REQUIRE_WRAP / hs_ring_hetero_
# parity.py's REQUIRE_HETERO.
_REQUIRE_SELECTIVE = os.environ.get("VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE")   # "0" | "1" | unset
_REQUIRE_SKIPPED = os.environ.get("VLLM_HOOK_SELPAR_REQUIRE_SKIPPED")       # "zero"|"positive"|unset
_REQUIRE_REFUSED = os.environ.get("VLLM_HOOK_SELPAR_REQUIRE_REFUSED") == "1"
_REQUIRE_DEGENERATE = os.environ.get("VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE")  # "zero"|"positive"|unset


def _matches_req_id(internal_req_id: str, external_req_id: str) -> bool:
    """Same v1/legacy req_id match rule as workers/_common.py::iter_matching_req_ids."""
    return internal_req_id == external_req_id or internal_req_id.startswith(f"{external_req_id}-")


def _flatten_eager_hs(t):
    """Un-pad a driver-side hs_cache tensor to the FLAT per-token-row layout the ring
    reconstruction produces. Same contract as hs_ring_parity.py::_flatten_eager_hs."""
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.dim() == 2:
        return t
    if t.dim() != 3:
        return None
    num_steps = t.shape[0]
    if num_steps == 0:
        return None
    if num_steps == 1:
        return t[0]
    parts = [t[0]] + [t[i, 0:1] for i in range(1, num_steps)]
    return torch.cat(parts, dim=0)


def _eager_layer_store(output):
    """{layer_num:int -> flat cpu f32 tensor} for ONE RequestOutput's probes."""
    probes = getattr(output, "probes", None)
    hs = (probes or {}).get("hs_cache", {})
    store = {}
    for _mod, entry in hs.items():
        if not isinstance(entry, dict):
            continue
        layer_num = entry.get("layer_num")
        if layer_num is None:
            continue
        flat = _flatten_eager_hs(entry.get("hidden_states"))
        if flat is None:
            continue
        store[int(layer_num)] = flat.detach().to(torch.float32).cpu()
    return store


def _rpc(llm, method, *args):
    """collective_rpc by STRING METHOD NAME (no plain-function payload -> no
    VLLM_ALLOW_INSECURE_SERIALIZATION). Returns rank 0's row, or None."""
    for h in (getattr(llm, "llm", None), getattr(llm, "llm_engine", None)):
        if h is None:
            continue
        try:
            rows = h.collective_rpc(method, args=args) if args else h.collective_rpc(method)
        except Exception as e:  # noqa: BLE001
            print(f"[hs-selpar] collective_rpc({method!r}) via {type(h).__name__} failed: {e}",
                  flush=True)
            continue
        if rows:
            return rows[0]
    return None


def _ring_layer_store(run_dir, req_id):
    """{layer_num:int -> cpu f32 tensor} for ONE request from the durable ring dump."""
    from vllm_hook_plugins.graph.ring_reader import load_multilayer_ring_artifact
    data = load_multilayer_ring_artifact(run_dir)
    matched_key = None
    for internal_id in data:
        if _matches_req_id(internal_id, req_id):
            matched_key = internal_id
            break
    if matched_key is None:
        return {}
    per_layer = data[matched_key]
    return {int(layer): t.detach().to(torch.float32).cpu() for layer, t in per_layer.items()}


def capture(mode, out_path, no_reconstruct=False):
    ring_dir = None
    if mode == "graph":
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "1"
        enforce_eager = False
        ring_dir = os.environ.get("VLLM_HOOK_RING_DIR")
        if not ring_dir:
            raise RuntimeError(
                "VLLM_HOOK_RING_DIR must be set for the graph leg -- an unset ring dir dumps the "
                "durable capture into HOME and corrupts the quota.")
    else:
        os.environ["VLLM_HOOK_ALLOW_CUDAGRAPH"] = "0"
        enforce_eager = True

    cudagraph_mode = os.environ.get("VLLM_HOOK_CUDAGRAPH_MODE", "FULL")
    config_file = os.environ.get(
        "VLLM_HOOK_CONFIG_FILE",
        f"model_configs/hidden_states/{_MODEL.split('/')[-1]}_alltok.json")
    _user = os.environ.get("USER") or getpass.getuser()

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    extra = {}
    if mode == "graph" and cudagraph_mode.upper() != "NONE":
        extra["compilation_config"] = {"cudagraph_mode": cudagraph_mode}

    print(f"[hs-selpar:{mode}] booting model={_MODEL} enforce_eager={enforce_eager} "
          f"config={config_file} ncases={_NCASES} "
          f"selective={os.environ.get('VLLM_HOOK_DRAIN_SELECTIVE', '1(default)')} "
          f"per_request={os.environ.get('VLLM_HOOK_RING_PER_REQUEST', '0')}"
          + (f" ring_dir={ring_dir}" if ring_dir else ""), flush=True)

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
        # OFF in both legs: the two legs must see bit-identical scheduling, and a cached prefix
        # would change which tokens are forwarded (and so which rows the ring reserves). Mirrors
        # hs_ring_hetero_parity.py.
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=int(os.environ.get("VLLM_HOOK_PARITY_TP", "1")),
        **extra,
    )

    prompts = [c["text"] for c in _CASES]
    # ignore_eos: every case stays in the batch for its own max_tokens (shared or per-case), so
    # phase-gated cases (hooks_on=prefill/decode) actually reach both phases in the SAME run.
    sp_list = [
        SamplingParams(temperature=0.0, max_tokens=c["max_tokens"], ignore_eos=True,
                       extra_args={"hooks_on": c["hooks_on"], "hs_mode": c["hs_mode"],
                                   "output_hidden_states": c["layers"]})
        for c in _CASES
    ]
    for c in _CASES:
        layers_desc = "ALL" if c["layers"] is True else c["layers"]
        print(f"[hs-selpar:{mode}] case={c['name']} layers={layers_desc} hooks_on={c['hooks_on']} "
              f"hs_mode={c['hs_mode']} max_tokens={c['max_tokens']}", flush=True)

    # Call the PATCHED vllm.LLM.generate directly, NOT HookLLM.generate: the latter merges
    # multi-request probes onto outputs[0] (per-request tensors -> lists), which would destroy the
    # per-request eager ground truth. extra_args here are already fully specified.
    outputs = llm.llm.generate(prompts, sp_list)

    result = {}
    req_ids_by_case = {}
    for c, out in zip(_CASES, outputs):
        req_ids_by_case[c["name"]] = str(out.request_id)
        token_ids = list(out.outputs[0].token_ids)
        store = _eager_layer_store(out) if mode == "eager" else None
        wanted = None if c["layers"] is True else sorted(c["layers"])
        result[c["name"]] = {"layers": store, "token_ids": token_ids, "wanted_layers": wanted}
        print(f"[hs-selpar:{mode}] case={c['name']} req_id={out.request_id} tokens={len(token_ids)} "
              + (f"eager layers captured={sorted(store)}" if store is not None
                 else "(ring reconstruction deferred to flush)"), flush=True)

    if mode == "graph":
        run_dir = _rpc(llm, "flush_ring")
        if not run_dir and not no_reconstruct:
            raise RuntimeError(
                "flush_ring collective_rpc returned no run_dir -- the capture-ring path is not "
                "installed on the worker (check VLLM_HOOK_ALLOW_CUDAGRAPH / VLLM_HOOK_HS_CAPTURE).")
        print(f"[hs-selpar:{mode}] flush_ring -> run_dir={run_dir}", flush=True)

        # THE non-vacuity witness. Read AFTER flush_ring (drain.stop() joins the consumer thread,
        # so every enqueued step -- including the run's last one -- has already been accounted).
        counts = _rpc(llm, "get_drain_row_counts") or {}
        print(f"[hs-selpar:{mode}] drain row counts (RPC) = {counts}", flush=True)
        result["_drain_counts"] = counts

        sel = counts.get("selective")
        copied = int(counts.get("hs.drain.rows_copied", 0) or 0)
        skipped = int(counts.get("hs.drain.rows_skipped", 0) or 0)
        reason = counts.get("selective_disabled_reason")
        degen = int(counts.get("hs.drain.degenerate_steps", 0) or 0)
        print(f"[hs-selpar:{mode}] SELECTIVE-WITNESS selective={sel} rows_copied={copied} "
              f"rows_skipped={skipped} degenerate_steps={degen} disabled_reason={reason!r}",
              flush=True)

        if _REQUIRE_SELECTIVE is not None:
            want = (_REQUIRE_SELECTIVE == "1")
            if bool(sel) != want:
                raise RuntimeError(
                    f"VLLM_HOOK_SELPAR_REQUIRE_SELECTIVE={_REQUIRE_SELECTIVE!r} but "
                    f"get_drain_row_counts()['selective']={sel!r} -- this leg's whole point rests "
                    f"on that flag, and it did not fire as expected. counts={counts}")
        if _REQUIRE_SKIPPED == "zero" and skipped != 0:
            raise RuntimeError(
                f"VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=zero but rows_skipped={skipped} -- expected "
                f"nothing to skip (every installed layer is wanted by someone). counts={counts}")
        if _REQUIRE_SKIPPED == "positive" and not (skipped > 0):
            raise RuntimeError(
                f"VLLM_HOOK_SELPAR_REQUIRE_SKIPPED=positive but rows_skipped={skipped} -- a subset "
                f"leg with rows_skipped==0 means selective never actually fired; the leg would prove "
                f"nothing. counts={counts}")
        if _REQUIRE_DEGENERATE == "positive" and not (degen > 0):
            raise RuntimeError(
                f"VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE=positive but degenerate_steps={degen} -- on "
                f"this leg every request wants every installed layer, so the Task 19 fast path was "
                f"supposed to fire on every drained step. Zero means it never fired and the "
                f"all-layers no-cost claim is unproven here. counts={counts}")
        if _REQUIRE_DEGENERATE == "zero" and degen != 0:
            raise RuntimeError(
                f"VLLM_HOOK_SELPAR_REQUIRE_DEGENERATE=zero but degenerate_steps={degen} -- a SUBSET "
                f"leg that took the degenerate fast path copied every layer anyway, which silently "
                f"deletes the lever's entire win. counts={counts}")
        if _REQUIRE_REFUSED:
            reason_str = str(reason or "")
            if sel is not False or not reason:
                raise RuntimeError(
                    f"VLLM_HOOK_SELPAR_REQUIRE_REFUSED=1 but selective={sel!r} "
                    f"disabled_reason={reason!r} -- the refusal did not fire the way this negative "
                    f"control expects (selective must read False WITH a reason set, not merely "
                    f"unarmed). counts={counts}")
            if "per-request" not in reason_str and "RING_PER_REQUEST" not in reason_str:
                raise RuntimeError(
                    f"VLLM_HOOK_SELPAR_REQUIRE_REFUSED=1 fired a refusal, but the reason string "
                    f"does not look like the per-request fallback: {reason_str!r} -- wrong refusal "
                    f"branch (e.g. the sync-drain reason) would silently pass a weaker check.")

        if not no_reconstruct and run_dir:
            for c in _CASES:
                store = _ring_layer_store(run_dir, req_ids_by_case[c["name"]])
                result[c["name"]]["layers"] = store
                print(f"[hs-selpar:{mode}] case={c['name']} reconstructed {len(store)} layers "
                      f"{sorted(store)} from ring dump (req_id={req_ids_by_case[c['name']]})",
                      flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[hs-selpar:{mode}] wrote {out_path}", flush=True)
    return 0


def compare(graph_path, eager_path, rtol, atol, profile_out=None):
    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    with open(eager_path, "rb") as f:
        e = pickle.load(f)
    profile = {"cells": {}}

    counts = g.get("_drain_counts")
    if counts is not None:
        print(f"[hs-selpar] graph-leg drain row counts = {counts}")

    overall_ok = True
    total = matched = 0
    for case in sorted(k for k in (set(g) & set(e)) if not k.startswith("_")):
        g_tok, e_tok = g[case].get("token_ids"), e[case].get("token_ids")
        if g_tok != e_tok:
            print(f"[hs-selpar] case={case}: TOKEN MISMATCH graph={g_tok} eager={e_tok}")
            overall_ok = False

        g_layers = g[case].get("layers") or {}
        e_layers = e[case].get("layers") or {}
        wanted = g[case].get("wanted_layers")  # a list (subset) or None ("all" -- no fixed expectation)

        if wanted is not None:
            # Explicit per-request subset: the layer SET itself is part of the contract, not just
            # the tensor values -- a heterogeneous batch must give each request exactly what it
            # asked for, no more, no less (the 746606 failure mode was right shapes/sets, wrong
            # contents; this catches the sibling failure mode of wrong SETS).
            if sorted(e_layers) != wanted:
                print(f"[hs-selpar] case={case}: EAGER LAYER-SET MISMATCH got={sorted(e_layers)} "
                      f"wanted={wanted}")
                overall_ok = False
            if sorted(g_layers) != wanted:
                print(f"[hs-selpar] case={case}: GRAPH LAYER-SET MISMATCH got={sorted(g_layers)} "
                      f"wanted={wanted}")
                overall_ok = False
        else:
            # "all layers" case: no hardcoded expected set (would require knowing the model's
            # layer count here), but graph and eager must still see the SAME set as each other.
            if sorted(g_layers) != sorted(e_layers):
                print(f"[hs-selpar] case={case}: LAYER-SET MISMATCH graph={sorted(g_layers)} "
                      f"eager={sorted(e_layers)}")
                overall_ok = False

        common = sorted(set(g_layers) & set(e_layers))
        if not common:
            print(f"[hs-selpar] case={case}: NO common layers "
                  f"(graph={len(g_layers)}, eager={len(e_layers)})")
            overall_ok = False
            continue
        for layer in common:
            gv, ev = g_layers[layer], e_layers[layer]
            if gv is None or ev is None:
                continue
            total += 1
            if gv.shape != ev.shape:
                print(f"[hs-selpar] case={case} layer={layer}: SHAPE MISMATCH "
                      f"graph={tuple(gv.shape)} eager={tuple(ev.shape)}")
                overall_ok = False
                continue
            ok = torch.allclose(gv, ev, rtol=rtol, atol=atol)
            md = (gv - ev).abs().max().item() if gv.numel() else 0.0
            mean = (gv - ev).abs().mean().item() if gv.numel() else 0.0
            gz = float(gv.abs().max().item()) if gv.numel() else 0.0
            matched += int(ok)
            overall_ok = overall_ok and ok
            print(f"[hs-selpar] case={case} layer={layer}: match={ok} max|Δ|={md:.3e} "
                  f"mean|Δ|={mean:.3e} graph_max|v|={gz:.3e} shape={tuple(gv.shape)}")
            # The per-layer error PROFILE, kept for the cross-arm assertion (see crossarm()).
            # Keyed "case::layer" so two legs using the SAME case name / text (every leg here
            # shares case A's and B's prompts) can be compared cell-for-cell even when their
            # layer SETS differ -- the intersection of keys is what crossarm() actually compares.
            profile["cells"][f"{case}::{layer}"] = {
                "max": md, "mean": mean, "graph_max_abs": gz, "shape": list(gv.shape),
            }

    if profile_out:
        with open(profile_out, "wb") as f:
            pickle.dump(profile, f)
        print(f"[hs-selpar] wrote error profile ({len(profile['cells'])} cells) -> {profile_out}")

    print("=" * 60)
    print(f"[hs-selpar] {matched}/{total} tensors within rtol={rtol} atol={atol}")
    if overall_ok and total > 0:
        print("[hs-selpar] VERDICT: PASS — selective-drain ring HS capture matches eager.")
        return 0
    print("[hs-selpar] VERDICT: FAIL — selective-drain ring HS capture diverged from eager.")
    return 1


# Cross-arm error-profile gate -- ported from hs_ring_hetero_parity.py's crossarm() (Task 9), same
# defect class, same historically-calibrated bands, DIFFERENT lever (selective drain). WHY THIS
# EXISTS, not a nicety: a per-leg allclose against eager only ever
# proves "close to the SAME correct reference" -- it structurally cannot distinguish "matched
# because the mechanism is right" from "matched because stale or misaddressed contents happened to
# land within tolerance of the truth," and selective drain's whole failure mode (Task 15's own
# review) is rows shipped from the WRONG ring offset -- shape-correct, layer-set-correct, contents
# wrong. Two legs that drive the SAME underlying request (same prompt, same layer, so the SAME
# hidden state) through DIFFERENT capture/dispatch mechanisms (full drain vs selective compaction)
# must show the SAME error profile against eager if both are
# computing the right thing -- exactly the property the 746606 defect broke (LSF 746606: shapes and
# layer sets were correct, but the captured VALUES came from an unwritten/stale row, and only a
# human diffing two log sections by hand caught it, twice, before this check existed).
#
# BANDS -- reused verbatim from hs_ring_hetero_parity.py's crossarm(), not re-derived: that
# function's own history is the reason a hard bit-identity gate is NOT used. Its first version
# hard-failed unless max|Δ| agreed to rtol=1e-3; it went RED on a healthy tree (LSF 773415) and
# PASSED bit-exactly on an immediate re-run of the SAME commit (773720) -- cross-boot bit-identity
# usually holds (different capture-kernel specializations baked into each arm's graph can change a
# reduction order) but is not an invariant, so a hard gate on it is intermittently RED on a healthy
# tree, which is worse than no gate. The bands below instead hard-fail on the DEFECT SIGNATURE
# (measured on 746606: graph_max|v| collapsed 88x, error-to-signal ratio moved 1.4e8x) while sitting
# ~2x above the measured BENIGN worst case (773415: graph_max|v| identical to 4 sig figs, error-to-
# signal disagreement up to 4.5x) -- see hs_ring_hetero_parity.py's crossarm() docstring for the
# full calibration. Do NOT tighten these without re-reading that history; this file does not invent
# a stricter gate than the sibling oracle carries.
CROSSARM_SIGNAL_RTOL = 1e-2    # |graph_max|v|| must agree between arms to 1%
CROSSARM_RATIO_BAND = 10.0     # error-to-signal may differ by up to 10x between arms
CROSSARM_BITEXACT_RTOL = 1e-3  # INFORMATIONAL ONLY: "were the arms bit-identical?"


def crossarm(profile_a, profile_b, label_a, label_b,
             signal_rtol=CROSSARM_SIGNAL_RTOL, ratio_band=CROSSARM_RATIO_BAND):
    """Assert two legs' per-layer error profiles (against the SAME eager reference each computed
    independently) AGREE on their shared cells. `profile_a`/`profile_b` are arbitrary legs -- unlike
    the hetero oracle's fixed control/treatment pair, any two of this harness's legs that share a
    case name and at least one layer number can be compared; the shared-key intersection is what
    actually gets checked (see compare()'s profile-out for why the key is "case::layer", not just
    "layer" -- it lets legs with different layer SETS still be compared on their overlap).

    Hard-fail, any of:
      * the shared cell set is EMPTY (vacuous -- nothing was actually compared);
      * a shared cell's SHAPE differs;
      * `graph_max|v|` (the captured tensor's own magnitude) differs by more than `signal_rtol`;
      * the error-to-signal ratio `max|Δ| / graph_max|v|` differs between the two arms by more than
        `ratio_band`-fold.
    Reported but NOT fatal: bit-identity (`bitexact=`), for the same reason
    hs_ring_hetero_parity.py's crossarm() reports it as informational (see the module-level
    CROSSARM_* comment above)."""
    ak, bk = profile_a.get("cells", {}), profile_b.get("cells", {})
    ok = True
    bitexact_cells = 0

    if not ak or not bk:
        print(f"[hs-selpar] CROSSARM({label_a} vs {label_b}): VACUOUS -- {label_a} cells={len(ak)} "
              f"{label_b} cells={len(bk)}; an empty profile cannot witness anything")
        return 1

    shared = sorted(set(ak) & set(bk))
    if not shared:
        print(f"[hs-selpar] CROSSARM({label_a} vs {label_b}): VACUOUS -- no shared case::layer keys "
              f"({label_a} has {sorted(ak)}, {label_b} has {sorted(bk)})")
        return 1

    for key in shared:
        a, b = ak[key], bk[key]
        am, bm = float(a["max"]), float(b["max"])
        as_, bs_ = float(a.get("graph_max_abs", 0.0)), float(b.get("graph_max_abs", 0.0))
        if tuple(a["shape"]) != tuple(b["shape"]):
            print(f"[hs-selpar] CROSSARM {key}: SHAPE MISMATCH {label_a}={tuple(a['shape'])} "
                  f"{label_b}={tuple(b['shape'])}")
            ok = False
            continue

        sig_den = max(abs(as_), abs(bs_))
        sig_rel = abs(as_ - bs_) / sig_den if sig_den > 0 else 0.0
        sig_ok = sig_rel <= signal_rtol

        ar = am / abs(as_) if as_ else float("inf") if am else 0.0
        br = bm / abs(bs_) if bs_ else float("inf") if bm else 0.0
        lo, hi = min(ar, br), max(ar, br)
        ratio = (hi / lo) if lo > 0 else (1.0 if hi == 0 else float("inf"))
        ratio_ok = ratio <= ratio_band

        bitexact = abs(am - bm) <= 1e-9 + CROSSARM_BITEXACT_RTOL * max(abs(am), abs(bm))
        bitexact_cells += int(bitexact)

        print(f"[hs-selpar] CROSSARM {key}: ok={sig_ok and ratio_ok} bitexact={bitexact} "
              f"{label_a}_max|Δ|={am:.6e} {label_b}_max|Δ|={bm:.6e} "
              f"signal={as_:.6e}/{bs_:.6e} (rel={sig_rel:.3e}) "
              f"err_per_signal={ar:.3e}/{br:.3e} (x{ratio:.3g}) shape={tuple(a['shape'])}")
        if not sig_ok:
            print(f"[hs-selpar] CROSSARM {key}: SIGNAL MISMATCH -- the captured tensor's own "
                  f"magnitude differs by {sig_rel:.3e} (> {signal_rtol}). This is the 746606 "
                  f"signature: the rows shipped are not the data.")
        if not ratio_ok:
            print(f"[hs-selpar] CROSSARM {key}: ERROR-PROFILE MISMATCH -- error-to-signal differs "
                  f"by {ratio:.3g}x (> {ratio_band}x).")
        ok = ok and sig_ok and ratio_ok

    print("=" * 60)
    print(f"[hs-selpar] CROSSARM({label_a} vs {label_b}) compared {len(shared)} cells "
          f"(signal_rtol={signal_rtol}, ratio_band={ratio_band}x); "
          f"bit-identical on {bitexact_cells}/{len(shared)}")
    if bitexact_cells < len(shared):
        print("[hs-selpar] CROSSARM NOTE: the arms were not bit-identical on every cell. That is "
              "run-to-run boot variation, NOT a failure on its own (see hs_ring_hetero_parity.py's "
              "crossarm() CALIBRATION note). Informational only.")
    if ok:
        print(f"[hs-selpar] CROSSARM({label_a} vs {label_b}) VERDICT: PASS — the two arms produced "
              f"the SAME per-layer error profile against eager on every shared cell, so agreement "
              f"between them is not a tolerance coincidence over misaddressed contents.")
        return 0
    print(f"[hs-selpar] CROSSARM({label_a} vs {label_b}) VERDICT: FAIL — the two arms' per-layer "
          f"error profiles DIVERGE on a shared cell. Read this as the 746606 signature (structurally "
          f"correct, wrong contents) until proven otherwise; a per-leg allclose PASS does not clear "
          f"it.")
    return 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture")
    pc.add_argument("--mode", choices=["graph", "eager"], required=True)
    pc.add_argument("--out", required=True)
    pc.add_argument("--no-reconstruct", action="store_true",
                     help="skip flush_ring's ring_reader reconstruction (leg F: per-request "
                          "delivery never writes the shared sidecar this reads)")
    pk = sub.add_parser("compare")
    pk.add_argument("--graph", required=True)
    pk.add_argument("--eager", required=True)
    pk.add_argument("--rtol", type=float, default=1e-2)
    pk.add_argument("--atol", type=float, default=1e-2)
    pk.add_argument("--profile-out", default=None,
                    help="write this leg's per-layer error profile here, for `crossarm`")
    px = sub.add_parser("crossarm")
    px.add_argument("--a", required=True, help="profile file from one leg (e.g. legA)")
    px.add_argument("--b", required=True, help="profile file from another leg (e.g. legD)")
    px.add_argument("--label-a", default="a")
    px.add_argument("--label-b", default="b")
    px.add_argument("--signal-rtol", type=float, default=CROSSARM_SIGNAL_RTOL,
                    help="max relative disagreement in graph_max|v| between arms")
    px.add_argument("--ratio-band", type=float, default=CROSSARM_RATIO_BAND,
                    help="max fold-difference in error-to-signal between arms")
    args = p.parse_args()
    if args.cmd == "capture":
        sys.exit(capture(args.mode, args.out, no_reconstruct=args.no_reconstruct))
    elif args.cmd == "crossarm":
        with open(args.a, "rb") as f:
            pa = pickle.load(f)
        with open(args.b, "rb") as f:
            pb = pickle.load(f)
        sys.exit(crossarm(pa, pb, args.label_a, args.label_b, args.signal_rtol, args.ratio_band))
    else:
        sys.exit(compare(args.graph, args.eager, args.rtol, args.atol,
                         profile_out=args.profile_out))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
    main()
