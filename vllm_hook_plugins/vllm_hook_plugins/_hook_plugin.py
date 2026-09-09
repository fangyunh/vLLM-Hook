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


def _ring_per_request_mode() -> bool:
    """True when the off-loop HS capture-ring PER-REQUEST delivery path is armed
    (``VLLM_HOOK_RING_PER_REQUEST=1`` under graph mode). Additive gate: when False every existing
    response path (``get_captured_states`` RPC, disk flush, bank) is byte-identical -- the driver
    only reroutes an HS request's RPC retrieval to ``get_ring_per_request`` when this is True."""
    import os
    return (os.environ.get("VLLM_HOOK_RING_PER_REQUEST") == "1"
            and os.environ.get("VLLM_HOOK_ALLOW_CUDAGRAPH") == "1")


def _reconstruct_compact_qk(probes: dict) -> None:
    """Rebuild the padded ``k_all`` that a compact-transfer worker deferred
    (``VLLM_HOOK_QK_COMPACT_KALL``): the worker sends ``k_full`` + ``k_prefix_ends`` to avoid an
    O(seq^2) pad on its engine loop; here in the driver we rebuild
    ``pad_sequence([full[:L] for L in ends])``, byte-identical to the old ``k_stacked``. No-op
    when the worker sent a normal ``k_all``."""
    qk = probes.get("qk_cache") if isinstance(probes, dict) else None
    if not isinstance(qk, dict):
        return
    from torch.nn.utils.rnn import pad_sequence
    for entry in qk.values():
        if not isinstance(entry, dict):
            continue
        full = entry.pop("k_full", None)
        ends = entry.pop("k_prefix_ends", None)
        if full is not None and ends is not None:
            entry["k_all"] = pad_sequence([full[:int(L)] for L in ends], batch_first=True)


def _trim_probes(probes: dict, key: str, expected_len: int) -> None:
    """Trim probe tensors to expected_len along the sequence dimension.

    The vLLM v1 scheduler may run one extra forward pass after EOS is hit; vLLM discards the
    extra output token but capture hooks still fire, so this trims the surplus. Only tensors
    with a sequence dim (all_tokens mode) need trimming -- last_token tensors are (bs, hidden).
    """
    for entry in probes.get(key, {}).values():
        for tkey in ("hidden_states", "q", "k_all"):
            t = entry.get(tkey)
            # A quantized entry hands off a per-pass list (packed + scale + qmeta), already
            # unpadded per pass -- nothing to seq-trim; dequant happens downstream.
            if t is None or isinstance(t, list):
                continue
            # 3D = (bs, seq, hidden) -- trim seq (dim 1)
            if t.dim() == 3 and t.shape[1] > expected_len:
                PROF.incr("trim.event")
                entry[tkey] = t[:, :expected_len, :]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _stable_artifact_files(run_dir: str) -> list:
    """Non-empty, non-``.tmp`` files under ``run_dir`` (recursive), sorted.

    Disk writes are atomic (``.tmp`` + ``os.rename``), so a visible non-``.tmp`` file is
    complete. Empty list when nothing has landed (or on OSError)."""
    import glob
    import os
    try:
        return sorted(
            f for f in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
            if os.path.isfile(f) and not f.endswith(".tmp") and os.path.getsize(f) > 0
        )
    except OSError:
        return []


# save_to_disk=True must return with a durable artifact FILE already on disk. The writer runs
# in a non-blocking child process, so flush_disk can return before the file lands; these
# barriers poll until the artifact's file set is STABLE across one interval (so a multi-file
# artifact -- safetensors + JSON sidecar -- is fully present) before generate() returns. Runs
# in the caller's thread, never the engine loop. Bounded; on timeout the "no artifact" outcome
# stands.
_ARTIFACT_WAIT_S = 10.0
_ARTIFACT_POLL_S = 0.005


def _resolve_sink(extra: dict) -> str:
    """Return this request's sink: 'disk' | 'rpc' | 'drop'.

    Precedence: VLLM_HOOK_SINK=drop is a global override; an explicit per-request
    save_to_disk is honored next; VLLM_HOOK_SINK=disk|rpc is the default when the request
    states no preference; otherwise falls back to save_to_disk semantics."""
    import os
    env = os.environ.get("VLLM_HOOK_SINK", "").lower()
    if env == "drop":
        return "drop"
    if "save_to_disk" in extra:
        return "disk" if bool(extra["save_to_disk"]) else "rpc"
    if env in ("disk", "rpc"):
        return env
    return "disk" if bool(extra.get("save_to_disk")) else "rpc"


async def _await_disk_artifact(run_id: str, hook_dir: str) -> bool:
    import asyncio
    import os
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_ARTIFACT_WAIT_S / _ARTIFACT_POLL_S))):
        files = _stable_artifact_files(run_dir)
        if files and files == prev:
            return True
        prev = files
        await asyncio.sleep(_ARTIFACT_POLL_S)
    return bool(_stable_artifact_files(run_dir))


def _wait_disk_artifact(run_id: str, hook_dir: str) -> bool:
    import os
    import time
    run_dir = os.path.join(hook_dir, run_id)
    prev = None
    for _ in range(max(2, int(_ARTIFACT_WAIT_S / _ARTIFACT_POLL_S))):
        files = _stable_artifact_files(run_dir)
        if files and files == prev:
            return True
        prev = files
        time.sleep(_ARTIFACT_POLL_S)
    return bool(_stable_artifact_files(run_dir))


# ---------------------------------------------------------------------------
# Per-request ring delivery: BLOCK-UNTIL-HELD. The client contract is "the response does not
# return until the artifact/result is HELD". Polled from the async frontend (never the engine
# loop) with asyncio.sleep between polls, so EngineCore keeps decoding other requests while
# each worker-side check stays non-blocking. Bounded + loud on timeout, never an unbounded hang.
# ---------------------------------------------------------------------------
_RING_DELIVER_POLL_S = 0.005


def _ring_deliver_timeout_s() -> float:
    """Total wall-clock budget for a single request's block-until-held / disk-confirm poll, seconds.
    Env-overridable (VLLM_HOOK_RING_DELIVER_TIMEOUT_S); a sane 30 s default (well above a healthy
    off-loop drain's per-request latency, low enough that a wedged consumer fails loud, not forever)."""
    import os
    try:
        return max(0.1, float(os.environ.get("VLLM_HOOK_RING_DELIVER_TIMEOUT_S", "30") or "30"))
    except (TypeError, ValueError):
        return 30.0


async def _await_ring_per_request(engine, request_id):
    """BLOCK-UNTIL-HELD for the host-buffer RPC route: poll get_ring_per_request until it returns this
    request's marshaled probes (its off-loop finish has been processed) or the deliver timeout
    elapses. Returns the decompressed probes dict, or None on timeout (LOUD) -- the caller leaves
    output.probes unset rather than hanging."""
    import asyncio
    import time
    timeout = _ring_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    while True:
        with PROF.timed("rpc.get_ring_per_request"):
            states = await engine.collective_rpc("get_ring_per_request", args=(request_id,))
        parts = [_decompress(s) for s in states if s is not None]
        if parts:
            return parts[0]
        if time.monotonic() >= deadline:
            print(f"[hookplugin/ring] BLOCK-UNTIL-HELD TIMEOUT after {timeout:.1f}s waiting for RPC "
                  f"per-request delivery of {request_id!r}; leaving probes unset (raise "
                  f"VLLM_HOOK_RING_DELIVER_TIMEOUT_S if the off-loop consumer is merely slow, "
                  f"else it is a bug)", flush=True)
            return None
        await asyncio.sleep(_RING_DELIVER_POLL_S)


async def _await_ring_disk_confirm(engine, request_id) -> bool:
    """DISK-route CONFIRM: block until the per-request offload has landed the file at the client dest
    (confirm_ring_delivery -> OffloadProcess.wait). Polled NON-BLOCKING (worker-side timeout 0.0 so
    the RPC never stalls the forward) with asyncio.sleep between polls. Returns True on confirmed
    delivery, False on timeout (LOUD). output.probes stays unset either way (the client reads the
    delivered file, as with save_to_disk)."""
    import asyncio
    import time
    timeout = _ring_deliver_timeout_s()
    deadline = time.monotonic() + timeout
    while True:
        with PROF.timed("rpc.confirm_ring_delivery"):
            res = await engine.collective_rpc("confirm_ring_delivery", args=(request_id, 0.0))
        # collective_rpc returns one result per worker (TP=1 -> one). True on any worker == landed.
        if any(r is True for r in res):
            return True
        if time.monotonic() >= deadline:
            print(f"[hookplugin/ring] DISK-CONFIRM TIMEOUT after {timeout:.1f}s waiting for offload "
                  f"delivery of {request_id!r}; the client dest file may be incomplete (raise "
                  f"VLLM_HOOK_RING_DELIVER_TIMEOUT_S, or check the offload worker)", flush=True)
            return False
        await asyncio.sleep(_RING_DELIVER_POLL_S)


def _warn_profile_ring_conflict() -> None:
    """Once-per-process warning: VLLM_HOOK_PROFILE_MODE=1 with VLLM_HOOK_RING_PER_REQUEST=1 is a
    misconfiguration. Profile mode disables Component 2, but the host-buffer route then demuxes
    rows into an index nothing retrieves (a host-RAM leak) and misreports the capture->NVMe
    boundary. Warns rather than refuses, so a run started this way is never silently wrong."""
    import os
    if getattr(_warn_profile_ring_conflict, "_warned", False):
        return
    if (os.environ.get("VLLM_HOOK_PROFILE_MODE") == "1"
            and os.environ.get("VLLM_HOOK_RING_PER_REQUEST") == "1"):
        _warn_profile_ring_conflict._warned = True
        print("[hookplugin/ring] WARNING: VLLM_HOOK_PROFILE_MODE=1 AND VLLM_HOOK_RING_PER_REQUEST=1 "
              "are BOTH set -- this is a misconfiguration. Profile mode must pair with the "
              "shared-file / disk Component-1 drain, NOT the host-buffer per-request route (which "
              "leaks host RAM -- rows demux into an index nothing retrieves -- and misreports the "
              "capture->NVMe boundary). Unset VLLM_HOOK_RING_PER_REQUEST for profile-mode "
              "Component-1 measurement.", flush=True)


# vLLM version advisory: printed at most once per process (see _note_vllm_version).
_VLLM_VER_NOTED = False


def _note_vllm_version() -> None:
    """Advisory only, never a gate: warn when the running vLLM is off this branch's validated
    version (<0.22, V1 model runner). 0.22-0.24 is compat-proven via the V1 forcing in
    ``_patched_create_engine_config``; 0.25+ is untested on this branch. ``setup.py`` stays
    unpinned so those versions still install -- hence advisory, not a gate. Compares with
    PEP440 ``Version``, not strings (``"0.21.0" > "0.21"`` is true lexically but not
    semantically). Deliberately tolerant of an unparseable/absent version -- must never break
    engine construction over a log line."""
    global _VLLM_VER_NOTED
    if _VLLM_VER_NOTED:
        return
    _VLLM_VER_NOTED = True
    try:
        import vllm
        from packaging.version import Version
        raw = vllm.__version__
        ver = Version(raw)
        if ver < Version("0.22"):
            return                                  # the validated env — say nothing
        if ver < Version("0.25"):
            tier = ("compat-proven (LSF graph parity 6/6), but NOT the env the perf "
                    "numbers were taken on")
        else:
            tier = "UNTESTED on this branch (the v2_hook line covers V2 / 0.25+)"
        print(f"[vllm-hook] NOTE: this branch is developed and GPU-validated on vLLM "
              f"<=0.21 with the V1 model runner; found {raw} — {tier}. The V1 runner is "
              f"forced regardless (VLLM_USE_V2_MODEL_RUNNER=0).", flush=True)
    except Exception:  # noqa: BLE001 — an advisory must never break engine init
        return


def _worker_kind(worker_ext) -> str:
    """Resolve 'hidden_states' | 'qk' | 'steer' for autocap sizing (each worker's per-token
    transient shape differs; 'steer' has no capture ring).

    VLLM_HOOK_WORKER is checked first and is authoritative: ``vllm serve`` selects the worker
    with it, and the profiler replaces ``worker_extension_cls`` with its own mixin while stashing
    the real worker here -- so keying off the extension class alone would mis-size a QK/steer run
    as HS. Falls back to the extension-class string when the env is unset (offline HookLLM sets
    the class directly)."""
    import os
    env_w = (os.environ.get("VLLM_HOOK_WORKER") or "").strip().lower()
    if env_w == "qk":
        return "qk"
    if env_w == "steer":
        return "steer"
    if env_w == "hidden_states":
        return "hidden_states"
    s = worker_ext if isinstance(worker_ext, str) else getattr(worker_ext, "__name__", str(worker_ext))
    sl = s.lower()
    if "hookqk" in sl:
        return "qk"
    if "steer" in sl:
        return "steer"
    return "hidden_states"


_DTYPE_BYTES = {
    "torch.float32": 4, "torch.float": 4, "torch.float16": 2, "torch.half": 2,
    "torch.bfloat16": 2, "torch.float64": 8, "torch.double": 8,
    "torch.int8": 1, "torch.uint8": 1, "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}


def _dtype_element_size(dt) -> int:
    """Bytes per element for a torch dtype, without importing torch (str map + the .itemsize
    attribute torch>=2.1 exposes). Defaults to 2 (bf16) if unknown."""
    itemsize = getattr(dt, "itemsize", None)
    if isinstance(itemsize, int) and itemsize > 0:
        return itemsize
    return _DTYPE_BYTES.get(str(dt), 2)


def _autocap_setting():
    """Resolve the tri-state VLLM_HOOK_RING_MAX_BATCHED_TOKENS knob (off/auto/explicit-int)."""
    import os
    from vllm_hook_plugins.graph.ring_sizing import parse_autocap_setting
    return parse_autocap_setting(os.environ.get("VLLM_HOOK_RING_MAX_BATCHED_TOKENS"))


def _derive_safe_max_batched_tokens(config, worker_kind: str):
    """Safe max_num_batched_tokens for THIS model + GPU + ring (int, or None if inapplicable).

    Reads the GPU total via ``current_platform.get_device_total_memory`` (NVML) so it does NOT
    initialize a CUDA context in the driver process. Worst-case assumption (all_tokens, all layers)
    is deliberate — the min-only rule at the call site keeps that harmless when the real workload is
    lighter."""
    import os
    from vllm.platforms import current_platform
    from vllm_hook_plugins.graph.ring_sizing import (
        resolve_ring_bytes_auto, compute_safe_max_batched_tokens,
        per_layer_token_bytes_hs, per_layer_token_bytes_qk,
        DEFAULT_AUTOCAP_SAFETY, DEFAULT_AUTOCAP_HEADROOM_BYTES,
    )
    mc = config.model_config
    tc = mc.hf_text_config
    n_layers = int(getattr(tc, "num_hidden_layers"))
    dtype_size = _dtype_element_size(getattr(mc, "dtype", None))
    gpu_util = float(config.cache_config.gpu_memory_utilization)
    total_gpu = int(current_platform.get_device_total_memory(0))
    ring_bytes = resolve_ring_bytes_auto(total_gpu, gpu_util)
    hidden = int(getattr(tc, "hidden_size"))
    if worker_kind == "qk":
        h_q = int(getattr(tc, "num_attention_heads"))
        h_kv = int(getattr(tc, "num_key_value_heads", h_q))
        head_dim = int(getattr(tc, "head_dim", 0) or (hidden // h_q))
        plt = per_layer_token_bytes_qk(h_q, h_kv, head_dim, dtype_size)
    else:  # hidden_states
        plt = per_layer_token_bytes_hs(hidden, dtype_size)
    safety = int(os.environ.get("VLLM_HOOK_RING_AUTOCAP_SAFETY") or DEFAULT_AUTOCAP_SAFETY)
    headroom = int(os.environ.get("VLLM_HOOK_RING_AUTOCAP_HEADROOM_BYTES") or DEFAULT_AUTOCAP_HEADROOM_BYTES)
    return compute_safe_max_batched_tokens(
        total_gpu, gpu_util, ring_bytes, n_layers, plt, safety=safety, headroom_bytes=headroom)


def _maybe_autocap_max_batched_tokens(config, worker_kind: str) -> None:
    """Min-only OOM guard: lower ``config.scheduler_config.max_num_batched_tokens`` so a heavy
    full-graph capture's per-step transient fits the free GPU margin.

    Applied after the config is built (against vLLM's resolved budget), so it only ever lowers,
    never raises. Fires only when armed (VLLM_HOOK_RING_MAX_BATCHED_TOKENS) for a capturing
    worker (steer has no ring). Best-effort: any failure leaves the config byte-identical."""
    try:
        mode, explicit = _autocap_setting()
        if mode == "off" or worker_kind not in ("hidden_states", "qk"):
            return
        if mode == "explicit":
            safe = int(explicit)
        else:
            safe = _derive_safe_max_batched_tokens(config, worker_kind)
        if safe is None:
            return
        from vllm_hook_plugins.graph.ring_sizing import apply_min_only
        sc = config.scheduler_config
        current = getattr(sc, "max_num_batched_tokens", None)
        new = apply_min_only(current, safe)
        if new is not None and new != current:
            sc.max_num_batched_tokens = int(new)
            print(f"[vllm-hook] autocap: max_num_batched_tokens {current} -> {new} "
                  f"(worker={worker_kind}, mode={mode}); bounds full-graph capture per-step transient",
                  flush=True)
    except Exception as e:  # noqa: BLE001 -- must never break engine init
        print(f"[vllm-hook] autocap SKIPPED (non-fatal): {e!r}", flush=True)


def _patched_create_engine_config(self, *args, **kwargs):
    """Inject worker extension and (legacy) force eager mode before VllmConfig.

    When ``VLLM_HOOK_ALLOW_CUDAGRAPH=1`` the caller's ``enforce_eager`` stands and the
    CUDA-graph QK capture path is armed (graph/install.py); otherwise eager mode is forced
    so the legacy ``register_forward_hook`` path works."""
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
    # Worker kind (hidden_states | qk | steer), from the extension whether set by the caller
    # (offline HookLLM) or defaulted above (serve). Feeds the autocap guard below.
    _wkind = _worker_kind(self.worker_extension_cls)

    # Newer vLLM can auto-select a V2 GPUModelRunner for some architectures when
    # VLLM_USE_V2_MODEL_RUNNER is unset. That runner does not expose the internals the hook
    # reaches through (_prepare_inputs, input_batch, query_start_loc, requests), so capture would
    # silently return nothing and steering would be a no-op. Force V1 whenever the env is unset
    # (inherited by the spawned worker subprocess). A user who explicitly sets
    # VLLM_USE_V2_MODEL_RUNNER=1 is honoured but warned loudly that capture/steer will be inert.
    _want_v2 = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if _want_v2 == "1":
        print("[vllm-hook] WARNING: VLLM_USE_V2_MODEL_RUNNER=1 with the hook active — "
              "vLLM's V2 model runner is UNSUPPORTED by the hook, so capture/steering "
              "will be a NO-OP. Unset it to let the hook force the V1 runner.", flush=True)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    # Same concern, one level up: the V1 forcing above keeps the RUNNER right, but says
    # nothing about the vLLM VERSION. Note it when we are off the validated 0.21 env.
    _note_vllm_version()

    graph_mode = os.environ.get("VLLM_HOOK_ALLOW_CUDAGRAPH") == "1"
    if graph_mode:
        # Graph mode: leave enforce_eager as the caller set it, and arm the
        # load_model install path for the worker subprocess(es).
        from vllm_hook_plugins.graph.install import set_graph_mode
        set_graph_mode(True)
    else:
        # Eager mode is mandatory for the forward-hook capture path.
        self.enforce_eager = True

    # Opt-in: densify the FULL decode cudagraph capture sizes past vLLM's default cap of
    # min(max_num_seqs*2, 512). Above that the saturated batch can no longer replay the decode
    # graph and falls back to eager. Setting max_cudagraph_capture_size before the config is
    # built lets FULL capture decode graphs up to the saturated batch, at the cost of more
    # warmup + cudagraph pool memory (why this is opt-in). VLLM_HOOK_CUDAGRAPH_SIZES overrides
    # the whole list; _MAX_CAPTURE sets the ceiling and lets vLLM regenerate the fine list.
    # Graph mode only.
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

    # Graph mode only: min-only auto-cap of max_num_batched_tokens so heavy full-graph capture's
    # per-step transient cannot OOM at high batch. Applied post-build against vLLM's resolved
    # budget, so it only ever lowers. Armed by VLLM_HOOK_RING_MAX_BATCHED_TOKENS (default off).
    if graph_mode:
        _maybe_autocap_max_batched_tokens(config, _wkind)

    # Buffer mode declares no splitting op: the capture/steer kernel must be absorbed into the
    # FULL decode cudagraph, and declaring a splitting op under FULL would force a fallback to
    # FULL_AND_PIECEWISE (reintroducing an eager seam). Nothing to declare here.

    return config


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach probes to output
# ---------------------------------------------------------------------------


def _prompt_token_len(prompt):
    """Best-effort prompt token length for the serve-path size model."""
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


def _hs_num_layers(engine) -> int | None:
    """Total decoder layers (for HS 'all layers' capture). Cached on the engine."""
    cached = getattr(engine, "_vllm_hook_hs_nlayers", "missing")
    if cached != "missing":
        return cached
    n = None
    try:
        n = int(getattr(engine.model_config.hf_text_config, "num_hidden_layers"))
    except Exception:  # noqa: BLE001
        n = None
    try:
        engine._vllm_hook_hs_nlayers = n
    except Exception:  # noqa: BLE001
        pass
    return n


def _emit_capture_evidence(engine, output, extra, wants_hs, wants_qk, gen_tokens) -> None:
    """Emit capture evidence (hook.fire.<w> + captured.bytes.<w>) in the driver process.

    Emitted here, not in the worker: under ``vllm serve`` the worker process's profiler dump is
    lost at teardown (SIGTERM kills the child before atexit runs), so evidence emitted in the
    worker never reaches the harvest even though the ring captured and persisted the artifact.
    The driver's dump is collected, and the ring path reaches this finalize per finished
    request, so evidence is reported from this request's actual output.

    Byte-exact for the all_tokens workload: captured bytes = n_layers x tokens x
    per-token-per-layer bytes, matching exactly what the ring drain writes to NVMe (buffers hold
    all heads; head selection happens at analysis). ``hook.fire.<w>`` += n_layers so
    ``hook_fire_count / n_layers`` recovers the capturing-request count. Best-effort: a failure
    here must never perturb the finalize. (The worker-side emission stays for the offline path,
    whose dump is collected via ``dump_profiler``; the two paths never both land, so there is no
    double-count.)
    """
    try:
        n_prompt = len(getattr(output, "prompt_token_ids", None) or [])
        # gen_tokens is accumulated across the generate loop by the caller: the serve path streams
        # in delta mode, so the final output's outputs[].token_ids holds only the last delta --
        # summing per-yield deltas recovers the true generated length.
        n_gen = int(gen_tokens or 0)
        hooks_on = extra.get("hooks_on", "prefill")
        prefill_tok = n_prompt if hooks_on in ("prefill", "both") else 0
        decode_tok = n_gen if hooks_on in ("decode", "both") else 0
        tc = engine.model_config.hf_text_config
        elt = int(getattr(engine.model_config.dtype, "itemsize", 2) or 2)
        if wants_hs:
            oh = extra.get("output_hidden_states")
            n_layers = (len(oh) if isinstance(oh, (list, tuple)) and oh
                        else (_hs_num_layers(engine) or 0))
            mode = extra.get("hs_mode", "all_tokens")
            tokens = (prefill_tok + decode_tok) if mode == "all_tokens" \
                else ((1 if prefill_tok else 0) + decode_tok)
            bpt = int(getattr(tc, "hidden_size")) * elt
            if n_layers > 0:
                PROF.incr("hook.fire.hs", n_layers)
                if tokens > 0:
                    PROF.gauge("captured.bytes.hs", float(n_layers) * tokens * bpt)
        elif wants_qk:
            oq = extra.get("output_qk")
            n_layers = (len(oq) if isinstance(oq, (dict, list, tuple)) and oq
                        else (_hs_num_layers(engine) or 0))
            mode = extra.get("hookq_mode", "all_tokens")
            tokens = (prefill_tok + decode_tok) if mode == "all_tokens" \
                else ((1 if prefill_tok else 0) + decode_tok)
            num_h = int(getattr(tc, "num_attention_heads"))
            num_kv = int(getattr(tc, "num_key_value_heads", num_h) or num_h)
            head_dim = int(getattr(tc, "head_dim", 0) or (int(getattr(tc, "hidden_size")) // num_h))
            bpt = (num_h + num_kv) * head_dim * elt
            if n_layers > 0:
                PROF.incr("hook.fire.qk", n_layers)
                if tokens > 0:
                    PROF.gauge("captured.bytes.qk", float(n_layers) * tokens * bpt)
    except Exception:  # noqa: BLE001 — evidence is best-effort; never perturb the finalize
        pass


def _maybe_storage_route(engine, prompt, extra, max_tokens) -> bool | None:
    """Per-request storage router (VLLM_HOOK_STORAGE_ROUTER=1): predict this
    request's artifact size from its prompt length + captured layers/heads and
    return the min-tax storage choice (True=disk, False=RPC), or None to leave
    the caller's save_to_disk untouched (router off / can't decide / steer).

    Decided at request-START (before capture) because the worker routes each
    request's egress into the disk vs RPC bucket from save_to_disk at admit
    time -- a finish-time flip would arrive after the data already landed.
    """
    import os
    # DEFAULT ON: serve-only (this patch is the AsyncLLM path; offline LLM.generate never calls
    # it), and it reproduces the proven optimum HS-last->RPC / QK+HS-all->disk. `=0` to disable.
    # The default lives in optimizations.PUBLIC_LEVERS -- one source of truth, not a copy here.
    from vllm_hook_plugins.optimizations import env_is_on
    if not env_is_on("storage_router"):
        return None
    _dbg = os.environ.get("VLLM_HOOK_ROUTER_DEBUG") == "1"

    def _log(msg):
        if not _dbg:
            return
        n = getattr(_maybe_storage_route, "_dbg_n", 0)
        if n < 20:
            _maybe_storage_route._dbg_n = n + 1
            print(f"[hookplugin/router] {msg}", flush=True)

    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_qk or wants_hs):
        return None  # steer / nothing to store
    P = _prompt_token_len(prompt)
    if not P:
        _log(f"P=None (prompt type={type(prompt).__name__}) -> no route")
        return None
    hooks_on = extra.get("hooks_on", "both")
    gen_len = int(max_tokens) if max_tokens else 0
    try:
        from vllm_hook_plugins.run_utils import predict_artifact_kb, route_to_disk, predicted_rpc_ms
        if wants_qk:
            dims = _qk_model_dims(engine)
            if not dims:
                _log("qk dims=None -> no route")
                return None
            H_q, _H_kv, head_dim = dims
            oqk = extra.get("output_qk") or {}
            # total (layer,head) pairs captured -> fold into n_layers with H=1
            head_layers = sum(len(v) for v in oqk.values()) if isinstance(oqk, dict) else 0
            if head_layers <= 0:
                return None
            gran = extra.get("hookq_mode", "all_tokens")
            kb = predict_artifact_kb("qk", gran, P, head_layers, 1, head_dim,
                                     H_q * head_dim, 2, gen_len, hooks_on)
            d = route_to_disk("qk", kb)
            _log(f"qk P={P} hl={head_layers} gran={gran} kb={kb:.0f} "
                 f"rpc_ms={predicted_rpc_ms('qk', kb):.0f} -> {'DISK' if d else 'RPC'}")
            return d
        else:  # HS
            dims = _qk_model_dims(engine)
            hidden = dims[0] * dims[2] if dims else None
            if not hidden:
                return None
            layers = extra.get("output_hidden_states")
            n_layers = len(layers) if isinstance(layers, (list, tuple)) and layers \
                else _hs_num_layers(engine)
            if not n_layers:
                return None
            gran = extra.get("hs_mode", "last_token")
            kb = predict_artifact_kb("hs", gran, P, n_layers, 1, dims[2],
                                     hidden, 2, gen_len, hooks_on)
            d = route_to_disk("hs", kb)
            _log(f"hs P={P} L={n_layers} gran={gran} kb={kb:.0f} "
                 f"rpc_ms={predicted_rpc_ms('hs', kb):.0f} -> {'DISK' if d else 'RPC'}")
            return d
    except Exception as e:  # noqa: BLE001
        _log(f"exception {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-request delivery router -- the off-loop HS capture-ring path.
#
# At request-start, predict the request's raw artifact bytes, read the chosen analyzer's
# reducibility, and pick transport (rpc/disk) + analyze_where (none/inflight/from_disk) via
# graph/delivery_router.decide_route. The disk transport is crossed to the worker drain now
# (before the first forward) via the route_ring_to_disk RPC, so the drain stages that request
# to its own NVMe file; analyze_where stays driver-side and drives finalize.
#
# Additive + gated: every function below is inert unless VLLM_HOOK_RING_PER_REQUEST is armed
# (graph mode), so the storage_router + all existing response paths stay byte-identical when
# it is off.
# ---------------------------------------------------------------------------

# RPC-vs-disk crossover thresholds, in bytes. Documented sane defaults, overridable by env --
# never a hardcoded magic constant at the call site. ~512 KiB: below it the RPC on-loop ship
# beats the disk handoff (see run_utils.route_to_disk's proven crossover); above it, stream to
# NVMe.
_DEFAULT_RING_T_RPC = 512 * 1024
_DEFAULT_RING_T_ANALYZE = 512 * 1024


def _profile_mode() -> bool:
    """Component-1-only profiling (VLLM_HOOK_PROFILE_MODE=1): only the capture pipeline (GPU
    forward + in-graph scatter + off-loop drain -> server NVMe) is measured; delivery (analyzer
    + offload/RPC response) is disabled -- the finalize stamps request_done at the data-prepared
    boundary and ships nothing. Default off = full serving runs both stages."""
    import os
    return os.environ.get("VLLM_HOOK_PROFILE_MODE") == "1"


def _ring_route_thresholds() -> "tuple[int, int]":
    """(T_rpc, T_analyze) in bytes for decide_route. Env-configurable (VLLM_HOOK_ROUTER_T_RPC /
    VLLM_HOOK_ROUTER_T_ANALYZE) with the documented defaults above; read every call so the
    crossover can be retuned without a restart."""
    import os
    t_rpc = int(os.environ.get("VLLM_HOOK_ROUTER_T_RPC", _DEFAULT_RING_T_RPC))
    t_analyze = int(os.environ.get("VLLM_HOOK_ROUTER_T_ANALYZE", _DEFAULT_RING_T_ANALYZE))
    return t_rpc, t_analyze


def _analyzer_reducible(analyzer_name, analyzer_spec) -> bool:
    """Is the request's chosen analyzer reducible server-side (ships a small result), or does
    it need the raw tensors delivered whole to the client?

      * ACCEPTS == "qk"    (core_reranker, two-pass)  -> needs raw          -> False
      * ACCEPTS == "score" (attn_tracker)             -> reduces to a score -> True
      * hidden_states (no ACCEPTS): reducible ONLY when a reduce is requested
        (analyzer_spec["reduce"] in {mean, norm}); reduce=none / absent      -> False

    A missing/unknown analyzer name falls back to the reduce-driven rule (a bare hidden_states-style
    spec), and an unresolvable one defaults to NOT reducible = RAW delivery — today's behavior, which
    never loses data. Pure logic over the registry capability + the spec; no engine/GPU."""
    reduce = (analyzer_spec or {}).get("reduce", "none") if isinstance(analyzer_spec, dict) else "none"
    reducible_reduce = reduce in ("mean", "norm")
    if not analyzer_name:
        return reducible_reduce
    entry = None
    try:
        from vllm_hook_plugins.registry import PluginRegistry
        entry = PluginRegistry.get_analyzer(analyzer_name)
        if entry is None:
            from vllm_hook_plugins import register_plugins
            register_plugins()
            entry = PluginRegistry.get_analyzer(analyzer_name)
    except Exception:  # noqa: BLE001
        entry = None
    accepts = getattr(entry.analyzer, "ACCEPTS", None) if entry is not None else None
    if accepts == "qk":
        return False
    if accepts == "score":
        return True
    return reducible_reduce


def _decide_ring_route(engine, prompt, extra, max_tokens):
    """RouteDecision for the off-loop HS capture-ring per-request path, decided at REQUEST-START, or
    None when it does not apply (not HS-only / cannot predict) so the caller falls back to the
    default host-buffer path.

    HS-ONLY: the capture-ring per-request path is HS-only (get_ring_per_request is HS-only), so a
    request that also wants QK is left to the general path. Mirrors _maybe_storage_route's HS
    prediction (predict_artifact_kb) so both routers agree on size."""
    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_hs and not wants_qk):
        return None
    P = _prompt_token_len(prompt)
    if not P:
        return None
    dims = _qk_model_dims(engine)
    hidden = dims[0] * dims[2] if dims else None
    if not hidden:
        return None
    layers = extra.get("output_hidden_states")
    n_layers = len(layers) if isinstance(layers, (list, tuple)) and layers \
        else _hs_num_layers(engine)
    if not n_layers:
        return None
    gran = extra.get("hs_mode", "last_token")
    hooks_on = extra.get("hooks_on", "both")
    gen_len = int(max_tokens) if max_tokens else 0
    try:
        from vllm_hook_plugins.run_utils import predict_artifact_kb
        from vllm_hook_plugins.graph.delivery_router import decide_route
        kb = predict_artifact_kb("hs", gran, P, n_layers, 1, dims[2], hidden, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _ring_route_thresholds()
        return decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
    except Exception:  # noqa: BLE001
        return None


def _decide_ring_route_qk(engine, prompt, extra, max_tokens):
    """RouteDecision for the off-loop QK capture-ring per-request path, decided at REQUEST-START, or
    None when it does not apply (not QK-only / cannot predict) so the caller falls back to the default
    host-buffer path. QK-ONLY (the QK ring path is separate from the HS one). Mirrors
    ``_maybe_storage_route``'s QK prediction so both routers agree on size. Returns None for a non-dict
    ``output_qk`` (whole-model capture) -> the safe host-buffer RPC default."""
    wants_qk = extra.get("output_qk") is not None
    wants_hs = extra.get("output_hidden_states") is not None
    if not (wants_qk and not wants_hs):
        return None
    P = _prompt_token_len(prompt)
    if not P:
        return None
    dims = _qk_model_dims(engine)
    if not dims:
        return None
    H_q, _H_kv, head_dim = dims
    oqk = extra.get("output_qk") or {}
    head_layers = sum(len(v) for v in oqk.values()) if isinstance(oqk, dict) else 0
    if head_layers <= 0:
        return None
    gran = extra.get("hookq_mode", "all_tokens")
    hooks_on = extra.get("hooks_on", "both")
    gen_len = int(max_tokens) if max_tokens else 0
    try:
        from vllm_hook_plugins.run_utils import predict_artifact_kb
        from vllm_hook_plugins.graph.delivery_router import decide_route
        kb = predict_artifact_kb("qk", gran, P, head_layers, 1, head_dim,
                                 H_q * head_dim, 2, gen_len, hooks_on)
        predicted_bytes = int(kb * 1024)
        reducible = _analyzer_reducible(extra.get("analyzer"), extra.get("analyzer_spec"))
        t_rpc, t_analyze = _ring_route_thresholds()
        return decide_route(predicted_bytes, reducible, t_rpc, t_analyze)
    except Exception:  # noqa: BLE001
        return None


def _ring_finalize_action(route, profile_mode: bool) -> str:
    """Finalize action for a ring-per-request HS request. Pure decision (no engine/GPU) so the
    no-GPU test drives it directly:

      * profile_mode           -> 'profile_stamp'      (Component-1-only: stamp request_done, run
                                    NO Component 2 — no analyzer, delivery, offload, or RPC response)
      * route is None          -> 'rpc_raw'            (default host-buffer path — back-compat)
      * analyze_where inflight  -> 'analyze_inflight'   (ServerAnalyzeProcess — see the caller)
      * analyze_where from_disk -> 'analyze_from_disk'  (ServerAnalyzeProcess — see the caller)
      * transport disk, none    -> 'disk_raw'           (delivered via the offloaded per-request file)
      * transport rpc,  none    -> 'rpc_raw'            (host-buffer RPC — get_ring_per_request)
    """
    if profile_mode:
        return "profile_stamp"
    if route is None:
        return "rpc_raw"
    if route.analyze_where == "inflight":
        return "analyze_inflight"
    if route.analyze_where == "from_disk":
        return "analyze_from_disk"
    return "disk_raw" if route.transport == "disk" else "rpc_raw"


def _log_analyze_deferred(action: str) -> None:
    """Log once that server-side CPU analyze (ServerAnalyzeProcess) is deferred, so the
    fall-back to raw delivery is never silent."""
    n = getattr(_log_analyze_deferred, "_n", 0)
    if n < 4:
        _log_analyze_deferred._n = n + 1
        print(f"[hookplugin/ring-router] analyze_where={action!r}: server-side CPU analyze is "
              f"deferred to Task 12; delivering RAW (client analyzes client-side)", flush=True)


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

    # Per-request storage router: pick the min-tax path (disk vs RPC) from this request's
    # predicted artifact size. Decided here at request-start so the worker routes egress into
    # the right bucket.
    #
    # ONLY when the caller expressed NO preference. save_to_disk is not just a perf knob -- it
    # is how you ask for a durable artifact FILE. The router prices HS last_token at ~192 KB and
    # picks RPC, so overriding an explicit save_to_disk=True would silently write nothing for a
    # caller who needs the file on disk, with no way to express the requirement. An explicit
    # value is a requirement; an absent one is "you choose".
    if "save_to_disk" not in extra:
        _routed = _maybe_storage_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _routed is not None and _routed != save_to_disk:
            save_to_disk = _routed
            extra["save_to_disk"] = _routed
            effective_params.extra_args = extra

    # Per-request delivery router: for the off-loop HS capture-ring per-request path, decide
    # this request's transport (rpc/disk) + analyze_where from its predicted artifact size and
    # the chosen analyzer's reducibility, at request-start. Cross the disk transport to the
    # worker drain now -- before the request's first forward -- via route_ring_to_disk, so the
    # drain stages this request to its own NVMe file; analyze_where stays here and drives
    # finalize below. Additive + gated: no-op unless ring per-request mode is armed, so the
    # storage router + every response path stay byte-identical when it is off. Not run in
    # profile mode (delivery is disabled there -- no per-request delivery to route).
    _warn_profile_ring_conflict()  # guard (b): PROFILE_MODE + RING_PER_REQUEST is a misconfig
    _ring_route = None
    # Guard (a): only arm the ring route when the request will actually take the ring finalize
    # branch -- HS-only, not steering, and its effective sink is neither drop nor disk (mirrors
    # the finalize gates below). Otherwise route_ring_to_disk + the offload would fire for a
    # request whose finalize is skipped or goes to drop/disk, orphaning NVMe staging until
    # shutdown.
    if (_ring_per_request_mode() and not _profile_mode()
            and wants_hs and not wants_qk and not wants_steer
            and _resolve_sink(extra) not in ("drop", "disk")):
        _ring_route = _decide_ring_route(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _ring_route is not None and _ring_route.transport == "disk":
            import os as _os_route
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = _os_route.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_ring_to_disk"):
                await self.collective_rpc("route_ring_to_disk", args=(request_id, dest))

    # QK sibling of the HS ring route above: the off-loop QK capture-ring per-request path.
    # Decide this request's transport (rpc/disk) at request-start and cross the disk route to
    # the QK drain now (before the first forward). QK-only (gated on `not wants_hs`), same
    # guards as the HS block. route_ring_to_disk resolves to the QK worker's method.
    _ring_route_qk = None
    if (_ring_per_request_mode() and not _profile_mode()
            and wants_qk and not wants_hs and not wants_steer
            and _resolve_sink(extra) not in ("drop", "disk")):
        _ring_route_qk = _decide_ring_route_qk(
            self, prompt, extra, getattr(effective_params, "max_tokens", 0))
        if _ring_route_qk is not None and _ring_route_qk.transport == "disk":
            import os as _os_route
            run_id = extra.get("run_id") or request_id
            hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
            dest = _os_route.path.join(hook_dir, str(run_id))
            with PROF.timed("rpc.route_ring_to_disk"):
                await self.collective_rpc("route_ring_to_disk", args=(request_id, dest))

    # Serve-path QK auto-select. vllm serve goes through this patch, not HookLLM.generate, so
    # the offline admission cannot run here. When opted in (VLLM_HOOK_QK_AUTO_SELECT=1 -- stands
    # in for the offline analyzer-accepts gate, since the analyzer runs client-side in serve) and
    # the client did not pin qk_capture, pick the smaller representation from the prompt length +
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
    _hook_gen_toks = 0
    _prof_capture = needs_hooks and not wants_steer and _profile_mode()
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if _prof_capture:
                # Accumulate generated tokens across yields: serve streams in DELTA mode, so summing
                # per-yield deltas recovers the full generated length for the evidence byte count.
                for _o in (getattr(output, "outputs", None) or []):
                    _hook_gen_toks += len(getattr(_o, "token_ids", None) or [])
            if output.finished and needs_hooks and not wants_steer and _profile_mode():
                # Profile mode (Component-1-only): the capture pipeline already persisted this
                # request's data to server NVMe (the off-loop drain). Run no delivery -- no
                # analyzer, offload, or RPC response -- and stamp request_done at the
                # data-prepared boundary. output.probes stays unset; the serving product (profile
                # mode off) runs both stages.
                PROF.incr("request_done")
                PROF.event("request_done",
                           {"req_id": str(request_id), "boundary": "data_prepared"})
                # Capture evidence in the driver dump (the worker's is lost at serve teardown)
                # so the harvest sees artifact_kb / hook_fire for this profile.
                _emit_capture_evidence(self, output, extra, wants_hs, wants_qk, _hook_gen_toks)
            elif output.finished and wants_steer:
                # Steer evidence in the driver dump (the worker's steer.fire counter is lost at
                # serve teardown): confirm steering fired. Under FULL cudagraph the baked steer op
                # runs on every forward of a steered request, so a finished steer request was
                # steered. steer.fire feeds hook_fire_count; artifact_kb stays 0 (steer has no
                # artifact).
                PROF.incr("steer.fire")
            elif output.finished and needs_hooks and not wants_steer:
                sink = _resolve_sink(extra)
                if sink == "drop":
                    pass  # drop sink: nothing on the finish path; finally clears the bucket
                elif sink == "disk":
                    run_id = extra.get("run_id") or request_id
                    hook_dir = extra.get("hook_dir") or _DEFAULT_HOOK_DIR
                    with PROF.timed("rpc.flush_disk"):
                        await self.collective_rpc(
                            "flush_disk", args=([request_id], run_id, hook_dir))
                    # Durability is NOT waited for here (fire-and-forget) — the read side
                    # (analyze) and teardown drain guarantee it. Opt-in per-request durable_wait
                    # preserves the read-immediately contract for callers that need it.
                    if extra.get("durable_wait"):
                        with PROF.timed("disk.await_artifact"):
                            await _await_disk_artifact(run_id, hook_dir)
                    # Leave output.probes unset — caller reads artifacts from disk.
                elif _ring_per_request_mode() and wants_hs and not wants_qk:
                    # Ring per-request delivery is HS-only. Gate on `not wants_qk` so a request
                    # that also wants QK is not swallowed by this branch (which returns hs_cache
                    # only, silently dropping qk_cache); it falls through to get_captured_states
                    # instead.
                    #
                    # Dispatch on the request-start RouteDecision. _ring_route is None (no route
                    # decided) => 'rpc_raw', identical to the original default path.
                    action = _ring_finalize_action(_ring_route, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        # Server-side CPU analyze (ServerAnalyzeProcess) is deferred -- it needs
                        # the analyzer name/spec on the server plus a blocking analyze RPC. Falls
                        # back to raw delivery of the same transport, byte-safe (client reduces
                        # client-side); just missing the server-side reduce. Logged once.
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_ring_route is not None
                                                 and _ring_route.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        # Disk transport: the per-request file streamed to NVMe and, on the
                        # worker drain's finish, was handed to the OffloadProcess -> delivered to
                        # the client dest. Block until the offload confirms the file landed, then
                        # leave output.probes unset -- the client reads the delivered file, as
                        # with save_to_disk. On confirm the worker also unlinks the server-side
                        # staging source. Bounded + loud on timeout.
                        with PROF.timed("ring.await_disk_confirm"):
                            await _await_ring_disk_confirm(self, request_id)
                    else:  # rpc_raw — host-buffer path
                        # Off-loop HS capture-ring per-request delivery: the worker demuxes
                        # drained rows by req_id into a PerRequestIndex off-loop and assembles
                        # this request on finish. Block-until-held: poll get_ring_per_request
                        # until it returns this request's marshaled probes or the deliver timeout
                        # elapses (loud; probes left unset, never an unbounded hang).
                        probes = await _await_ring_per_request(self, request_id)
                        if probes is not None:
                            output.probes = probes
                elif _ring_per_request_mode() and wants_qk and not wants_hs:
                    # Off-loop QK capture-ring per-request delivery. QK-only: gate on `not
                    # wants_hs` so a request that also wants HS is not swallowed here (it falls
                    # through to get_captured_states). Dispatch on the request-start
                    # RouteDecision (_ring_route_qk); None => 'rpc_raw' (the default path).
                    action = _ring_finalize_action(_ring_route_qk, False)
                    if action in ("analyze_inflight", "analyze_from_disk"):
                        # Server-side CPU analyze is deferred (as with HS) -> RAW delivery of the SAME
                        # transport, byte-safe (the client reduces client-side). Logged once.
                        _log_analyze_deferred(action)
                        action = ("disk_raw" if (_ring_route_qk is not None
                                                 and _ring_route_qk.transport == "disk")
                                  else "rpc_raw")
                    if action == "disk_raw":
                        # Disk transport: the per-request q/k files streamed to NVMe and, on
                        # finish, were handed to the OffloadProcess -> delivered to the client
                        # dest. Block until the offload confirms, then leave output.probes unset
                        # (the client reads the delivered file, as with save_to_disk).
                        with PROF.timed("ring.await_disk_confirm"):
                            await _await_ring_disk_confirm(self, request_id)
                    else:  # rpc_raw — host-buffer path
                        # BLOCK-UNTIL-HELD: poll get_ring_per_request (QK worker) until it returns this
                        # request's marshaled qk_cache bytes or the deliver timeout elapses (LOUD).
                        probes = await _await_ring_per_request(self, request_id)
                        if probes is not None:
                            output.probes = probes
                else:  # rpc
                    with PROF.timed("rpc.get_states"):
                        states = await self.collective_rpc(
                            "get_captured_states", args=(request_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        probes = parts[0]
                        _reconstruct_compact_qk(probes)
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes
            yield output
    finally:
        # Cleanup on abort/disconnect. Runs for both paths: on normal completion the bucket was
        # already popped (get_captured_states / flush_disk), so this clear is a no-op; on an
        # abort before output.finished it releases the orphan bucket -- which for save_to_disk
        # also releases the resident-byte counter (else an aborted disk-mode request leaks it
        # and can wedge the admission ceiling).
        if needs_hooks and not wants_steer:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
            # clear_captured_states clears only the bank/eager buckets, which the ring path
            # never uses. On an abort before finish, free the request's ring state too -- the
            # host-buffer PerRequestIndex entry/stash and the disk staging -- so
            # disk_residency() and the host index return to 0 (no leak). No-op when the request
            # has no ring state. The QK-only branch resolves clear_ring_request to the QK
            # worker's method (one worker per process).
            if _ring_per_request_mode() and ((wants_hs and not wants_qk)
                                             or (wants_qk and not wants_hs)):
                await self.collective_rpc("clear_ring_request", args=(request_id,))


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

    # The storage router is serve-only (async engine): its decision must precede
    # capture, and the offline path's per-request prompt<->params alignment across
    # every input shape is a separate ergonomics problem. Warn once rather than
    # silently no-op (the flip below would land AFTER the bucket is chosen).
    import os as _os
    if (needs_hooks and _os.environ.get("VLLM_HOOK_STORAGE_ROUTER") == "1"
            and not getattr(_patched_llm_generate, "_router_warned", False)):
        _patched_llm_generate._router_warned = True
        print("[hookplugin] VLLM_HOOK_STORAGE_ROUTER is serve-only; the offline "
              "LLM.generate path honors each request's explicit save_to_disk.",
              flush=True)

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
                if (_ring_per_request_mode()
                        and extra.get("output_hidden_states") is not None
                        and extra.get("output_qk") is None):
                    # HS-only gate (see the serve path): a combined HS+QK request falls through
                    # to get_captured_states rather than being swallowed by this HS-only ring
                    # branch. Off-loop HS capture-ring per-request delivery: retrieve this
                    # request's marshaled probes. Returns None until its off-loop finish has been
                    # processed (this path does not block-until-held; that's serve-only).
                    with PROF.timed("rpc.get_ring_per_request"):
                        states = self.collective_rpc("get_ring_per_request", args=(req_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        output.probes = parts[0]
                else:
                    with PROF.timed("rpc.get_states"):
                        states = self.collective_rpc("get_captured_states", args=(req_id,))
                    parts = [_decompress(s) for s in states if s is not None]
                    if parts:
                        probes = parts[0]
                        _reconstruct_compact_qk(probes)  # rebuild deferred k_all before trim/merge
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        expected_len = n_prompt + n_gen - 1
                        _trim_probes(probes, "hs_cache", expected_len)
                        _trim_probes(probes, "qk_cache", expected_len)
                        output.probes = probes

        # Second pass: finalize disk-save requests.
        #
        # graph+buffer (capture_ring) mode: no per-generate finalize here. Durable capture
        # streams GPU-ring -> off-loop drain -> RING_DIR continuously during generate, so the
        # persist cost is already in gen_lat. Two reasons not to flush here: (1) the legacy
        # flush_disk host buckets (`_disk_states`) are never filled in graph+buffer mode, so
        # flush_disk would write nothing and the durability barrier would stall on a phantom
        # artifact; (2) flush_ring would join+close the consumer thread, breaking the next rep's
        # enqueue in a multi-rep offline run. The ring's atexit backstop (registered at install)
        # writes the reader sidecar at engine teardown; artifact_kb comes from the PROF
        # captured.bytes gauge. Mirrors the server PROFILE_MODE lifecycle (ship nothing at
        # finish; the ring already persisted).
        if disk_by_run and not _graph_mode():
            # Eager path unchanged: register_forward_hook fills _disk_states, so flush_disk writes
            # the artifact and the barrier waits for the writer child to land it. All req_ids sharing
            # a run_id flush together so the second flush doesn't overwrite the first.
            for run_id, req_list in disk_by_run.items():
                req_ids = [r for r, _ in req_list]
                _, hook_dir = req_list[0]
                with PROF.timed("rpc.flush_disk"):
                    self.collective_rpc("flush_disk", args=(req_ids, run_id, hook_dir))
            for run_id, req_list in disk_by_run.items():
                _, hook_dir = req_list[0]
                with PROF.timed("disk.await_artifact"):
                    _wait_disk_artifact(run_id, hook_dir)

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
                        # "scores": a per-pass list of [S_q,S_k] tensors.
                        n_tensors += len(v)
                        n_elems += sum(t.numel() for t in v)
                        new_entry[k] = [t.tolist() for t in v]
                    else:
                        new_entry[k] = v
                result[key][mod_name] = new_entry
        PROF.gauge("serve.serialize_probes.tensors", n_tensors)
        PROF.gauge("serve.serialize_probes.elements", n_elems)
        # Approximate JSON wire size via element count -- encoding to bytes here would double
        # the work. The harness computes the realized response_bytes from the HTTP response.
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

    # Arm the CUDA-graph QK install path: monkey-patch Worker.load_model so it installs the
    # static-buffer hosts + execute_model wrapper after the model is built (graph/install.py).
    # The patch is a strict no-op unless graph mode is enabled and the worker is the QK worker,
    # so the eager path and the HS / steer workers are untouched.
    #
    # Guarded: importing the graph stack must never be able to disable the eager plugin. If the
    # graph module fails to import for any reason, log and continue -- the eager path (forced
    # when VLLM_HOOK_ALLOW_CUDAGRAPH != "1") is entirely independent of this patch.
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
