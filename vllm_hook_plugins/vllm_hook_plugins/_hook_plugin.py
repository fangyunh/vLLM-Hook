"""
vLLM general plugin that exposes hidden-state and QK probe outputs via the
OpenAI-compatible API when ``output_hidden_states`` or ``output_qk`` is
passed in ``SamplingParams.extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in setup.py). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate`` and ``LLM.generate`` to retrieve per-request probe
outputs for both online (async) and offline (sync) usage.

For ``vllm serve``, also patches the OpenAI response builders so probe
outputs are included in HTTP responses as ``response.probes``.
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator, Callable
from typing import Any

import zstandard as zstd

from vllm_hook_plugins._profiler import PROF

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

_WORKER_EXT_HS = "vllm_hook_plugins.workers.probe_hidden_states_worker.ProbeHiddenStatesWorker"
_WORKER_EXT_QK = "vllm_hook_plugins.workers.probe_hookqk_worker.ProbeHookQKWorker"
_WORKER_EXT_STEER = "vllm_hook_plugins.workers.steer_activation_worker.SteerHookActWorker"

# Each hook worker emits exactly one custom op into the traced graph; this maps the
# worker extension class to that op so create_engine_config can declare ONLY the
# active worker's splitting op (see _patched_create_engine_config for why this is
# load-bearing under VLLM_USE_AOT_COMPILE). Matched by substring so a fully-qualified
# class path, a bare class name, or a subclass path all resolve.
_WORKER_OP_MARKERS = (
    ("ProbeHookQK", "vllm_hook::qk_probe"),
    ("probe_hookqk", "vllm_hook::qk_probe"),
    ("ProbeHiddenStates", "vllm_hook::hs_probe"),
    ("probe_hidden_states", "vllm_hook::hs_probe"),
    ("SteerHookAct", "vllm_hook::steer_residual"),
    ("steer_activation", "vllm_hook::steer_residual"),
)


def _splitting_op_for_worker(worker_extension_cls) -> str | None:
    """Return the single ``vllm_hook::*`` splitting op the given worker emits, or
    None if the class is not one of our hook workers (then no op is declared)."""
    name = worker_extension_cls if isinstance(worker_extension_cls, str) else \
        getattr(worker_extension_cls, "__name__", "") or str(worker_extension_cls or "")
    for marker, op in _WORKER_OP_MARKERS:
        if marker in name:
            return op
    return None

# Default hook_dir for save_to_disk requests when extra_args["hook_dir"] is not
# set. /dev/shm/vllm_hook is a RAM tmpfs on Linux — fast and ephemeral, matching
# HookClient's default.
_DEFAULT_HOOK_DIR = "/dev/shm/vllm_hook"


def _graph_mode() -> bool:
    """True when the CUDA-graph QK capture path is armed for this process.

    Armed by ``_patched_create_engine_config`` when VLLM_HOOK_ALLOW_CUDAGRAPH==1.
    In graph mode the worker installs capture at load_model, so the generate
    patches must NOT also issue the lazy ``install_hooks`` collective_rpc.
    """
    from vllm_hook_plugins.graph.install import graph_mode_enabled
    return graph_mode_enabled()


def _decompress(data: bytes) -> Any:
    PROF.gauge("rpc.payload_bytes", len(data))
    with PROF.timed("rpc.decompress"):
        if data[:4] == _ZSTD_MAGIC:
            return pickle.loads(_ZSTD_DECOMPRESSOR.decompress(data))
        return pickle.loads(data)


def _trim_probes(probes: dict, key: str, expected_len: int) -> None:
    """Trim probe tensors to expected_len along the sequence dimension.

    The vLLM v1 scheduler may execute one extra forward pass after the EOS
    stop condition is hit. vLLM discards the extra output token, but our
    capture hooks still fire during that pass. This trims the surplus so
    probe shapes are always deterministic.

    Only applies to tensors with a sequence dim (all_tokens mode). last_token
    tensors are shape (bs, hidden) and have no sequence dim to trim.
    """
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            if t is None:
                continue
            # 3D = (bs, seq, hidden) — trim seq (dim 1)
            if t.dim() == 3 and t.shape[1] > expected_len:
                PROF.incr("trim.event")
                entry[tkey] = t[:, :expected_len, :]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _patched_create_engine_config(self, *args, **kwargs):
    """Inject worker extension and (legacy) force eager mode before VllmConfig.

    v0.3.0 gating: when ``VLLM_HOOK_ALLOW_CUDAGRAPH == "1"`` we do NOT force
    eager — the caller's ``enforce_eager`` stands and the CUDA-graph QK capture
    path is armed (graph/install.py). Otherwise we keep the v0.2.0 behaviour
    EXACTLY: force eager so the legacy ``register_forward_hook`` path works.
    """
    import os
    if not self.worker_extension_cls:
        # Default to hidden states worker; users can override via env var.
        worker_type = os.environ.get("VLLM_HOOK_WORKER", "hidden_states")
        if worker_type == "qk":
            self.worker_extension_cls = _WORKER_EXT_QK
        elif worker_type == "steer":
            self.worker_extension_cls = _WORKER_EXT_STEER
        else:
            self.worker_extension_cls = _WORKER_EXT_HS

    graph_mode = os.environ.get("VLLM_HOOK_ALLOW_CUDAGRAPH") == "1"
    if graph_mode:
        # Graph mode: leave enforce_eager as the caller set it, and arm the
        # load_model install path for the worker subprocess(es).
        from vllm_hook_plugins.graph.install import set_graph_mode
        set_graph_mode(True)
    else:
        # Legacy v0.2.0: eager is mandatory for the forward-hook capture path.
        self.enforce_eager = True

    # v0.5.7 Tier 3 (opt-in): densify the FULL decode cudagraph capture sizes past vLLM's
    # default cap of min(max_num_seqs*2, 512). Above ~512 the saturated CB batch (~940) can
    # no longer replay the decode graph and falls back to EAGER (the high-lambda cascade
    # that makes FULL lose to PIECEWISE). vLLM honours an explicit max_cudagraph_capture_size
    # (config/vllm.py: only defaults to 512 when None), so setting it here BEFORE the config
    # is built lets FULL capture decode graphs up to the saturated batch. Cost: more graphs
    # -> more warmup + cudagraph pool memory (the reason it is opt-in). VLLM_HOOK_CUDAGRAPH_SIZES
    # overrides the whole list (coarse spacing -> fewer graphs); _MAX_CAPTURE sets the ceiling
    # and lets vLLM regenerate the fine list. Graph mode only.
    if graph_mode:
        _max_cap = os.environ.get("VLLM_HOOK_CUDAGRAPH_MAX_CAPTURE")
        _sizes = os.environ.get("VLLM_HOOK_CUDAGRAPH_SIZES")
        if _max_cap or _sizes:
            try:
                cc = self.compilation_config
                size_list = ([int(x) for x in _sizes.split(",") if x.strip()]
                             if _sizes else None)
                max_n = int(_max_cap) if _max_cap else (max(size_list) if size_list else None)
                def _apply(setter):
                    if size_list is not None:
                        setter("cudagraph_capture_sizes", sorted(set(size_list)))
                    else:
                        setter("cudagraph_capture_sizes", None)  # regenerate up to max_n
                    if max_n is not None:
                        setter("max_cudagraph_capture_size", max_n)
                if isinstance(cc, dict):
                    _apply(lambda k, v: cc.__setitem__(k, v))
                elif cc is not None:
                    _apply(lambda k, v: setattr(cc, k, v))
                else:
                    self.compilation_config = {
                        **({"cudagraph_capture_sizes": sorted(set(size_list))}
                           if size_list is not None else {}),
                        **({"max_cudagraph_capture_size": max_n} if max_n is not None else {}),
                    }
                print(f"[vllm-hook] Tier 3: cudagraph capture densified "
                      f"(max={max_n}, sizes={size_list or 'auto'})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[vllm-hook] Tier 3 cudagraph densify FAILED, default sizes: {e!r}",
                      flush=True)

    assert _original_create_engine_config is not None
    config = _original_create_engine_config(self, *args, **kwargs)

    # CRITICAL (op-mode capture): register our capture op as a vLLM "splitting
    # op" so the piecewise compiler keeps it in the EAGER seam between cudagraph
    # segments — exactly how `unified_attention` is treated. Otherwise our op is
    # absorbed into a cudagraphed segment and its Python impl is SKIPPED on graph
    # replay (proven on GPU: the op only fired on non-replayed warmup forwards,
    # never during real generation). As a splitting op it runs every step, so the
    # capture body executes mid-forward with the live request state.
    # Only OP-mode capture needs the splitting-op seam. Buffer mode (QK/HS) and
    # seam mode deliberately do NOT declare a splitting op: the capture kernel must
    # be absorbed into the (FULL decode) cudagraph, and declaring a splitting op
    # under FULL would force a fallback to FULL_AND_PIECEWISE (re-introducing an
    # eager seam) — see docs/v0.4.0_full_cuda_graph_plan.md §1.2. The gate is
    # PER-WORKER: each worker checks its own capture-mode env var.
    if graph_mode:
        _op = _splitting_op_for_worker(self.worker_extension_cls)
        _declare = (
            (_op == "vllm_hook::qk_probe"
             and os.environ.get("VLLM_HOOK_QK_CAPTURE", "op") == "op")
            or (_op == "vllm_hook::hs_probe"
                and os.environ.get("VLLM_HOOK_HS_CAPTURE", "op") == "op")
            or (_op == "vllm_hook::steer_residual"
                and os.environ.get("VLLM_HOOK_STEER_MODE", "op") == "op")
        )
        if _declare:
            try:
                cc = config.compilation_config
                ops = list(getattr(cc, "splitting_ops", None) or [])
                # Add ONLY the splitting op for the active worker — not all three.
                #
                # Why per-worker (load-bearing for VLLM_USE_AOT_COMPILE=1): the AOT
                # compile cache key is derived from vllm_config.compute_hash(), which
                # hashes compilation_config.splitting_ops. If every worker advertised
                # the same three ops, all three produce the SAME cache key, so worker
                # B loads worker A's serialized graph (proven on GPU: the HS/steer
                # engines loaded the QK artifact and ran the wrong eager seam → crash
                # / silent miss). One op per worker keeps the keys — and thus the
                # artifacts — distinct.
                #
                # The op the worker actually emits is also (idempotently) re-added by
                # its own graph_install at load_model, so this is just the early,
                # hash-time declaration.
                if _op is not None and _op not in ops:
                    ops.append(_op)
                    cc.splitting_ops = ops
                    print(f"[vllm-hook] added {_op} to "
                          f"compilation_config.splitting_ops ({len(ops)} total) "
                          f"for worker {self.worker_extension_cls}")
            except Exception as e:  # noqa: BLE001
                print(f"[vllm-hook] could not add capture op to splitting_ops: {e}")
        else:
            print(f"[vllm-hook] buffer/seam capture for worker "
                  f"{self.worker_extension_cls}: no splitting op declared "
                  f"(op absorbed into the cudagraph)")

    return config


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach probes to output
# ---------------------------------------------------------------------------


def _prompt_token_len(prompt):
    """Best-effort prompt token length for the D2 serve-path size model."""
    try:
        toks = getattr(prompt, "prompt_token_ids", None)
        if toks is not None:
            return len(toks)
        if isinstance(prompt, dict):
            t = prompt.get("prompt_token_ids")
            if t is not None:
                return len(t)
    except Exception:  # noqa: BLE001
        return None
    return None


def _qk_model_dims(engine):
    """(H_q, H_kv, head_dim) for the serve-path size model, cached on the engine.

    Full/unsharded counts from the engine's text config; head_dim = hidden // H_q to
    match the worker ``_conf`` (and HookLLM's offline dims). Cached so it is read once.
    """
    cached = getattr(engine, "_vllm_hook_qk_dims", "missing")
    if cached != "missing":
        return cached
    dims = None
    try:
        tc = engine.model_config.hf_text_config
        H_q = int(getattr(tc, "num_attention_heads"))
        H_kv = int(getattr(tc, "num_key_value_heads", H_q))
        hidden = int(getattr(tc, "hidden_size"))
        dims = (H_q, H_kv, hidden // H_q)
    except Exception:  # noqa: BLE001
        dims = None
    try:
        engine._vllm_hook_qk_dims = dims
    except Exception:  # noqa: BLE001
        pass
    return dims


async def _patched_generate(
    self,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap AsyncLLM.generate to install hooks and attach probes on finish."""
    # In vLLM v1, the chat endpoint clones SamplingParams into EngineCoreRequest
    # before calling generate(). We must read/modify the clone so our changes take effect.
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = dict(effective_params.extra_args or {})
    # vllm_xargs only allows scalar values, so HookClient JSON-encodes nested
    # structures. Decode them back here before the worker reads extra_args.
    import json as _json
    for _k in ("output_qk", "output_hidden_states", "steer"):
        if isinstance(extra.get(_k), str):
            try:
                _decoded = _json.loads(extra[_k])
            except (ValueError, TypeError):
                # Plain non-JSON strings (e.g. legacy boolean-like) pass through.
                continue
            # output_qk comes back as {str_key: list} — restore int keys
            if _k == "output_qk" and isinstance(_decoded, dict):
                _decoded = {int(k): v for k, v in _decoded.items()}
            extra[_k] = _decoded
    effective_params.extra_args = extra

    wants_hs = extra.get("output_hidden_states") is not None
    wants_qk = extra.get("output_qk") is not None
    wants_steer = isinstance(extra.get("steer"), dict)
    needs_hooks = wants_hs or wants_qk or wants_steer
    save_to_disk = bool(extra.get("save_to_disk"))

    # v0.5.7 D2: serve-path QK auto-select. vllm serve goes through THIS patch, not
    # HookLLM.generate, so the offline admission cannot run here. When opted in
    # (VLLM_HOOK_QK_AUTO_SELECT=1 — the env stands in for the offline analyzer-accepts
    # gate, since the analyzer runs client-side in serve) and the client did NOT pin
    # qk_capture, pick the smaller representation per request from the prompt length +
    # this request's output_qk head set + the model dims. Decided once, immutable.
    if wants_qk and "qk_capture" not in extra:
        import os as _os
        if _os.environ.get("VLLM_HOOK_QK_AUTO_SELECT") == "1":
            try:
                _dims = _qk_model_dims(self)
                _plen = _prompt_token_len(prompt)
                _oqk = extra.get("output_qk")
                if _dims and _plen and isinstance(_oqk, dict):
                    from vllm_hook_plugins.run_utils import qk_score_size_select
                    _pick = qk_score_size_select(
                        _plen, extra.get("hookq_mode", "all_tokens"), _oqk, *_dims)
                    extra["qk_capture"] = _pick
                    if _pick == "score":
                        extra.setdefault("score_head", 0)
                    effective_params.extra_args = extra
                    _n = getattr(_patched_generate, "_d2_log_n", 0)
                    if _n < 8:
                        _patched_generate._d2_log_n = _n + 1
                        print(f"[hookplugin/D2] serve auto-select qk_capture={_pick} "
                              f"(S={_plen} mode={extra.get('hookq_mode','all_tokens')})",
                              flush=True)
            except Exception:  # noqa: BLE001
                pass

    # In graph mode the QK capture path is already installed in the worker at
    # load_model (graph/install.py), so the lazy forward-hook install would only
    # double-capture — skip it. The legacy eager path still installs lazily.
    if (
        needs_hooks
        and not getattr(self, "_vllm_hook_installed", False)
        and not _graph_mode()
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            await self.collective_rpc("install_hooks")
        setattr(self, "_vllm_hook_installed", True)

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished and needs_hooks and not wants_steer:
                if save_to_disk:
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                    with PROF.timed("rpc.flush_disk"):
                        await self.collective_rpc("flush_disk", args=([request_id], run_id, hook_dir))
                    # Leave output.probes unset — caller reads artifacts from disk.
                else:
                    with PROF.timed("rpc.get_states"):
                        states = await self.collective_rpc("get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        probes = parts[0]  # rank 0 result; PP merge not yet needed
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes
            yield output
    finally:
        # Cleanup on abort/disconnect. Runs for BOTH paths: on normal completion the
        # bucket was already popped (get_captured_states / flush_disk), so this clear is a
        # no-op; on an abort BEFORE output.finished it releases the orphan bucket — which
        # for save_to_disk also releases the Stage-3 resident-byte counter (else an aborted
        # disk-mode request leaks it and can wedge the admission ceiling).
        if needs_hooks and not wants_steer:
            await self.collective_rpc("clear_captured_states", args=(request_id,))


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate patch
# ---------------------------------------------------------------------------


def _patched_llm_generate(self, prompts: Any, sampling_params: Any = None, **kwargs) -> list:
    """Wrap LLM.generate to install hooks and dispatch post-generation.

    Each request is dispatched based on its own extra_args:
    - save_to_disk=True -> collective_rpc("flush_disk"); output.probes unset.
    - otherwise        -> collective_rpc("get_captured_states"); attach to output.probes.
    """
    if isinstance(sampling_params, (list, tuple)):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    needs_hooks = any(
        (sp.extra_args or {}).get("output_hidden_states") is not None
        or (sp.extra_args or {}).get("output_qk") is not None
        or bool((sp.extra_args or {}).get("steer"))
        for sp in params_list
    )

    # Graph mode installs the QK path in the worker at load_model; skip the lazy
    # forward-hook install (it would double-capture). Eager path unchanged.
    if (
        needs_hooks
        and not getattr(self, "_vllm_hook_installed", False)
        and not _graph_mode()
    ):
        PROF.incr("rpc.install_hooks")
        with PROF.timed("rpc.install_hooks"):
            self.collective_rpc("install_hooks")
        self._vllm_hook_installed = True

    assert _original_llm_generate is not None
    outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)

    if needs_hooks:
        import os

        # First pass: handle RPC (in-memory) requests immediately, and collect
        # disk-save requests grouped by run_id so all requests sharing the same
        # run_id are flushed together — preventing the second flush from
        # overwriting the first when a batch shares one run_id.
        disk_by_run: dict = {}  # run_id -> [(req_id, hook_dir)]

        for idx, output in enumerate(outputs):
            req_id = output.request_id
            sp = params_list[idx] if idx < len(params_list) else params_list[0] if params_list else None
            extra = (sp.extra_args if sp is not None else None) or {}

            wants_artifacts = extra.get("output_hidden_states") is not None or extra.get("output_qk") is not None
            if extra.get("save_to_disk"):
                run_id = extra.get("run_id") or req_id
                hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                disk_by_run.setdefault(run_id, []).append((req_id, hook_dir))
            elif wants_artifacts:
                with PROF.timed("rpc.get_states"):
                    states = self.collective_rpc("get_captured_states", args=(req_id,))
                parts = [_decompress(s) for s in states if s is not None]
                if parts:
                    probes = parts[0]
                    n_prompt = len(output.prompt_token_ids)
                    n_gen = len(output.outputs[0].token_ids)
                    expected_len = n_prompt + n_gen - 1
                    _trim_probes(probes, "hs_cache", expected_len)
                    _trim_probes(probes, "qk_cache", expected_len)
                    output.probes = probes

        # Second pass: flush disk-save requests. All req_ids sharing a run_id
        # are sent in one collective_rpc call so the worker merges the full
        # batch into a single artifact — matching old execute_model() behavior.
        for run_id, req_list in disk_by_run.items():
            req_ids = [r for r, _ in req_list]
            _, hook_dir = req_list[0]
            with PROF.timed("rpc.flush_disk"):
                self.collective_rpc("flush_disk", args=(req_ids, run_id, hook_dir))

    return outputs


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


def _serialize_probes(probes: dict) -> dict:
    """Serialize probe tensors to lists for JSON transport."""
    import torch
    PROF.incr("serve.serialize_probes.calls")
    with PROF.timed("serve.serialize_probes"):
        result = {}
        n_tensors = 0
        n_elems = 0
        for key, cache in probes.items():
            # config is a flat dict of scalars — pass through as-is.
            if key == "config" and isinstance(cache, dict):
                result[key] = cache
                continue
            if not isinstance(cache, dict):
                continue
            result[key] = {}
            for mod_name, entry in cache.items():
                new_entry = {}
                for k, v in entry.items():
                    if isinstance(v, torch.Tensor):
                        n_tensors += 1
                        n_elems += v.numel()
                        new_entry[k] = v.tolist()
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        # v0.6.0 "scores": a per-pass list of [S_q,S_k] tensors.
                        n_tensors += len(v)
                        n_elems += sum(t.numel() for t in v)
                        new_entry[k] = [t.tolist() for t in v]
                    else:
                        new_entry[k] = v
                result[key][mod_name] = new_entry
        PROF.gauge("serve.serialize_probes.tensors", n_tensors)
        PROF.gauge("serve.serialize_probes.elements", n_elems)
        # Approximate JSON wire size — encoding to bytes here would double the
        # work and is the actual fix discussed in plan.html §7. We use the
        # element count as the proxy and let the harness compute the realized
        # response_bytes from the HTTP response.
    return result


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Inject serialized probes into completion responses."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        probes = getattr(res, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)
            break
    return response


async def _patched_chat_full_generator(self, request, result_generator, *args, **kwargs):
    """Inject serialized probes into chat completion responses."""
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )

    if last_output is not None and hasattr(response, "model_dump"):
        probes = getattr(last_output, "probes", None)
        if probes is not None:
            response.probes = _serialize_probes(probes)

    return response


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches EngineArgs, AsyncLLM.generate, LLM.generate, and the OpenAI
    response builders to enable activation capture via extra_args.

    Usage:
        # Hidden states
        SamplingParams(extra_args={"output_hidden_states": True})
        SamplingParams(extra_args={"output_hidden_states": [layer1, layer2]})

        # QK weights
        SamplingParams(extra_args={"output_qk": True})
        SamplingParams(extra_args={"output_qk": [layer1, layer2]})

    Probe outputs are returned in output.probes and, when using
    vllm serve, injected into the HTTP response body as response.probes.
    """
    global _original_create_engine_config
    global _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    # Arm the v0.3.0 CUDA-graph QK install path: monkey-patch Worker.load_model
    # so it installs the static-buffer hosts + execute_model wrapper after the
    # model is built (graph/install.py). The patch is a strict no-op unless graph
    # mode is enabled AND the worker is the QK worker, so the eager path and the
    # HS / steer workers are untouched.
    #
    # Guarded: importing the graph stack must NEVER be able to disable the eager
    # v0.2.0 plugin. If the graph module fails to import for any reason, log and
    # continue — the eager path (forced when VLLM_HOOK_ALLOW_CUDAGRAPH != "1") is
    # entirely independent of this patch.
    try:
        from vllm_hook_plugins.graph.install import patch_worker_load_model
        patch_worker_load_model()
    except Exception as e:  # noqa: BLE001
        print(f"[vllm-hook] graph load_model patch unavailable ({e}); "
              f"eager path unaffected.")

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # Patch OpenAI-compatible response builders (only available with vllm serve).
    # Module paths differ across vLLM versions; try all known locations.
    for _completion_module in (
        "vllm.entrypoints.openai.completion.serving",   # <0.12
        "vllm.entrypoints.openai.serving_completion",   # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_completion_module)
            _cls = _mod.OpenAIServingCompletion
            _original_completion_response = (
                _cls.request_output_to_completion_response
            )
            _cls.request_output_to_completion_response = _patched_completion_response
            break
        except Exception:
            pass

    for _chat_module in (
        "vllm.entrypoints.openai.chat_completion.serving",  # <0.12
        "vllm.entrypoints.openai.serving_chat",             # ≥0.12
    ):
        try:
            import importlib as _il
            _mod = _il.import_module(_chat_module)
            _cls = _mod.OpenAIServingChat
            _original_chat_full_generator = _cls.chat_completion_full_generator
            _cls.chat_completion_full_generator = _patched_chat_full_generator
            break
        except Exception:
            pass
