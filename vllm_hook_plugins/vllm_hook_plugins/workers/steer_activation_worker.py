import os
import json
import torch
from typing import TYPE_CHECKING, Any, Dict, Optional

from vllm.forward_context import get_forward_context

from vllm_hook_plugins._profiler import PROF
from vllm_hook_plugins.workers._common import (get_query_metadata,
                                               iter_matched_modules, match_layer)

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def _load_steering_vector(vector_path: str) -> Dict:
    """Load and parse a steering vector .pt file. Returns the raw dict."""
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found at: {vector_path}")
    return torch.load(vector_path, weights_only=False)


def _resolve_steer_config(steer_arg, env_default_path: Optional[str]) -> Optional[Dict]:
    """Normalize ``extra_args["steer"]`` into a config dict, or None if disabled.

    Accepts:
        True            -> read from env_default_path (legacy)
        {dict}          -> use directly (per-request override)
        False / None    -> steering disabled for this request
    """
    if not steer_arg:
        return None
    if isinstance(steer_arg, dict):
        return steer_arg
    # Legacy True: fall back to the file pointed to by VLLM_ACTSTEER_CONFIG
    if env_default_path and os.path.exists(env_default_path):
        with open(env_default_path) as f:
            return json.load(f).get("steering", {})
    return None


def _parse_steer_layers(optimal_layer, num_layers: int) -> list:
    """``optimal_layer`` -> sorted unique list of valid layer indices.

    Accepts ``int`` (single layer, legacy), ``list[int]`` (multi-layer steer), or ``"all"``.
    Out-of-range / unparseable entries are dropped. Backward-compatible: an int ``L`` in range
    returns ``[L]``, so the single-layer path is unchanged.
    """
    if optimal_layer == "all":
        return list(range(int(num_layers)))
    raw = optimal_layer if isinstance(optimal_layer, (list, tuple)) else [optimal_layer]
    out = set()
    for x in raw:
        try:
            L = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= L < int(num_layers):
            out.add(L)
    return sorted(out)


def _steer_targets_layer(optimal_layer, this_layer: int) -> bool:
    """True if a request whose config has ``optimal_layer`` should steer ``this_layer``.

    Eager-path counterpart of ``_parse_steer_layers`` that needs no ``num_layers`` (``"all"``
    always matches). Accepts int | list[int] | ``"all"``.
    """
    if optimal_layer == "all":
        return True
    if isinstance(optimal_layer, (list, tuple)):
        for x in optimal_layer:
            try:
                if int(x) == this_layer:
                    return True
            except (TypeError, ValueError):
                continue
        return False
    try:
        return int(optimal_layer) == this_layer
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Position + phase modes (the phase x positions gate)
# ---------------------------------------------------------------------------
# Two ORTHOGONAL per-request axes, mirroring the capture path's hooks_on x mode:
#   phase     : "prefill" | "decode" | "both"        (default "both")
#   positions : "all_tokens" | "last_token"          (default "all_tokens")
# Both defaults reproduce the steer-everything behaviour that predates them.
#
# NOTE the deliberate divergence from capture: capture's hooks_on defaults to
# "prefill", steering's phase defaults to "both". Steering today applies everywhere and
# no existing config may change meaning.
_STEER_PHASES = ("prefill", "decode", "both")
_STEER_POSITIONS = ("all_tokens", "last_token")

# Sentinels for the resolved per-step steer column consumed by the graph GPU router.
STEER_COL_ALL = -1    # steer the request's whole span (today's behaviour)
STEER_COL_NONE = -2   # steer nothing this step


def resolve_steer_modes(cfg: Optional[Dict]) -> tuple:
    """``(phase, positions)`` for a resolved steer config dict.

    Legacy ``apply_at_all_positions: false`` maps to ``positions="last_token"`` when no
    explicit ``positions`` key is present; an explicit ``positions`` always wins.

    Raises ``ValueError`` on an unknown value. This is deliberate: a silently-ignored
    ``positions`` typo would steer EVERY token while the caller believes they are
    steering one — the same failure mode ``optimizations.PUBLIC_LEVERS`` fails loud on.
    """
    if not cfg:
        return ("both", "all_tokens")
    phase = cfg.get("phase", "both")
    if phase not in _STEER_PHASES:
        raise ValueError(
            f"steering.phase={phase!r} is invalid; expected one of {_STEER_PHASES}")
    positions = cfg.get("positions")
    if positions is None:
        positions = ("all_tokens" if cfg.get("apply_at_all_positions", True)
                     else "last_token")
    if positions not in _STEER_POSITIONS:
        raise ValueError(
            f"steering.positions={positions!r} is invalid; "
            f"expected one of {_STEER_POSITIONS}")
    return (phase, positions)


def is_default_steer_modes(phase: str, positions: str) -> bool:
    """True iff these modes reproduce the steer-every-token-of-every-pass behaviour."""
    return phase == "both" and positions == "all_tokens"


def steer_span(phase: str, positions: str, is_prefill: bool, is_final_chunk: bool,
               start: int, end: int):
    """The ``(lo, hi)`` half-open column range this request steers this pass, or None.

    THE single source of truth for the phase x positions gate — the eager hook, the graph
    host router and the graph GPU router all bottom out here, so the three paths cannot
    disagree.

    ``is_final_chunk`` is capture's ``emit_q`` gate (``graph/install.py`` ~line 526):
    ``num_computed + qlen >= num_prompt``. It makes ``last_token`` CHUNK-INVARIANT — the
    last *prompt* token, never a scheduler chunk boundary. Without it the steered set
    would depend on ``max_num_batched_tokens`` and stop being reproducible.

    A decode pass has exactly one query token, so ``last_token`` and ``all_tokens``
    coincide there; no special case is needed.
    """
    if end <= start:
        return None
    if phase == "prefill" and not is_prefill:
        return None
    if phase == "decode" and is_prefill:
        return None
    if positions == "last_token":
        if is_prefill and not is_final_chunk:
            return None
        return (end - 1, end)
    return (start, end)


def steer_col_for(phase: str, positions: str, is_prefill: bool, is_final_chunk: bool,
                  start: int, end: int) -> int:
    """``steer_span`` expressed as the graph router's per-slot ``slot_col`` sentinel.

    ``STEER_COL_ALL`` whenever the whole span is steered (so the default path keeps a
    constant, upload-free routing key), ``STEER_COL_NONE`` when nothing is, else the
    absolute flat column to steer.
    """
    span = steer_span(phase, positions, is_prefill, is_final_chunk, start, end)
    if span is None:
        return STEER_COL_NONE
    lo, hi = span
    if lo == start and hi == end:
        return STEER_COL_ALL
    return lo


def _steer_rows(rows: "torch.Tensor", cfg: Dict, data: Dict) -> "torch.Tensor":
    """Apply one steer config to a contiguous block of residual rows; return the result.

    The exact op sequence the eager path has always used — extracted verbatim so the
    default (whole-tensor) branch and the per-request (row-slice) branch cannot drift.
    Both methods are row-wise, so slicing rows changes no value: ``matmul(rows, unit)``
    is a per-row dot product.
    """
    method = cfg.get("method", "adjust_rs")
    steering_vec = data["dir"].to(rows.device, dtype=rows.dtype)
    if method == "add_vector":
        coefficient = float(cfg.get("coefficient", 0))
        return rows + coefficient * steering_vec.view(1, -1)
    if method == "adjust_rs":
        unit_vec = steering_vec  # use dir as unit vector (matches old behavior)
        avg_proj = data["avg_proj"].to(rows.device, dtype=rows.dtype)
        current_projections = torch.matmul(rows, unit_vec)
        coeff = (avg_proj - current_projections).unsqueeze(-1)
        return rows + coeff * unit_vec.view(1, -1)
    raise ValueError(f"Unknown steering method: {method}")


class SteerHookActWorker:
    """Mixin injected into vLLM's GPU Worker via worker_extension_cls.

    Per-request steering: each request can pass its own steering config in
    extra_args["steer"] (dict) — different requests in the same batch can use
    different vectors / methods / coefficients / optimal layers.
    """

    if TYPE_CHECKING:
        model_runner: Any

    _hooks_installed: bool = False
    _bad_cfg_warned: int = 0   # cap the invalid-config warning (hot path)

    def install_hooks(self):
        """Install steering hooks on every transformer layer. Idempotent.

        Each hook checks per-request ``extra_args["steer"]`` and applies
        steering only when the request targets this hook's layer.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        try:
            self._install_hooks()
            print("Hooks installed successfully")
        except Exception as e:
            print(f"Hook installation failed: {e}")

    def dump_profiler(self) -> "str | None":
        """collective_rpc-callable: dump this WORKER process's PROF snapshot to
        VLLM_HOOK_PROFILE_DIR and return the path (None if profiling is off). The
        steer.fire evidence counter lives in the worker (refresh_slot_config), so the
        offline driver reads it only via this dump. Mirrors the HS/QK workers."""
        from vllm_hook_plugins._profiler import PROF
        return PROF.dump(role="worker-rpc")

    def _install_hooks(self):
        model = getattr(self.model_runner, "model", None)
        if model is None:
            print("no model; skip hooks")
            return

        # Cache for steering vectors loaded from disk, keyed by vector_path.
        # Loading per-request would be too slow.
        self._vector_cache: Dict[str, Dict] = {}
        # Legacy fallback: VLLM_ACTSTEER_CONFIG points to a JSON file
        # whose "steering" key has the per-worker default config. Used only when
        # extra_args["steer"] is True (boolean) instead of a dict.
        self._env_config_path = os.environ.get("VLLM_ACTSTEER_CONFIG")

        def steering_hook(input, output, this_layer: int):
            req_ids = getattr(self.model_runner.input_batch, "req_ids", [])
            # Every request whose resolved config targets THIS layer, with its modes.
            entries = []      # (i, req_state, cfg, phase, positions)
            for i, r in enumerate(req_ids):
                req_state = self.model_runner.requests.get(r)
                if not req_state or req_state.sampling_params is None:
                    continue
                steer_arg = (req_state.sampling_params.extra_args or {}).get("steer")
                resolved = _resolve_steer_config(steer_arg, self._env_config_path)
                if resolved is None:
                    continue
                _ol = resolved.get("optimal_layer", -1)
                if not _steer_targets_layer(_ol, this_layer):
                    continue
                try:
                    phase, positions = resolve_steer_modes(resolved)
                except ValueError as exc:
                    # Match the graph routers: an invalid config makes THAT request
                    # inert rather than crashing the forward. Loud but capped — the
                    # offline path already fails hard at HookLLM.load_config, so this
                    # only fires for a serve request that hand-rolled bad extra_args.
                    if self._bad_cfg_warned < 4:
                        self._bad_cfg_warned += 1
                        print(f"[steer] ignoring request with invalid steer config: {exc}",
                              flush=True)
                    continue
                entries.append((i, req_state, resolved, phase, positions))
            if not entries:
                return output

            is_tuple = isinstance(output, tuple)
            if is_tuple:
                hidden_states, residuals = output
            else:
                hidden_states = None
                residuals = output

            PROF.incr("steer.fire")

            all_default = all(is_default_steer_modes(p, q)
                              for (_, _, _, p, q) in entries)
            with PROF.timed("steer.apply", tier=2):
                if all_default:
                    # Default path, unchanged: the FIRST matching config applies to the
                    # whole residual tensor (known limitation: not per-request in a
                    # multi-request batch).
                    cfg = entries[0][2]
                    residuals = _steer_rows(residuals, cfg, self._vector_cache_for(cfg))
                else:
                    residuals = self._steer_per_request(residuals, entries)

            if is_tuple:
                return (hidden_states, residuals)
            else:
                return residuals

        # Hook every transformer layer; the closure decides per-request whether
        # to actually steer based on extra_args["steer"].
        self._hooks = []
        matched = []
        for name, module, layer_num in iter_matched_modules(model, match_layer):
            hook = module.register_forward_hook(
                lambda m, i, o, ln=layer_num: steering_hook(i, o, ln)
            )
            self._hooks.append(hook)
            matched.append(name)

        print(f"Installed {len(self._hooks)} steering hooks on layers: {matched}")

    def _vector_cache_for(self, cfg: Dict) -> Dict:
        """Load + cache a config's steering vector (keyed by ``vector_path``)."""
        vector_path = cfg["vector_path"]
        data = self._vector_cache.get(vector_path)
        if data is None:
            raw = _load_steering_vector(vector_path)
            data = {"dir": torch.tensor(raw["dir"])}
            if cfg.get("method", "adjust_rs") == "adjust_rs":
                # as_tensor, because the use site calls .to(device, dtype): a plain-float
                # avg_proj would run under CUDA graphs (the loader coerces either) but raise
                # here. A no-op for every shipped vector, which already stores a 0-d tensor.
                data["avg_proj"] = torch.as_tensor(raw["avg_proj"])
            self._vector_cache[vector_path] = data
        return data

    def _steer_per_request(self, residuals, entries):
        """Apply each request's config to its OWN rows, honouring phase x positions.

        Only reached when some request in the batch uses non-default modes, which needs
        per-request row spans. ``query_start_loc`` comes from the forward context exactly
        as the HS capture worker reads it. Requests own disjoint flat column ranges within
        a forward, so reading every slice from the ORIGINAL tensor and writing into a
        single clone is safe and order-independent.

        Falls back to the whole-tensor default when the batch metadata is unreadable —
        never crash the forward over a mode selection.
        """
        metadata = getattr(get_forward_context(), "attn_metadata", None)
        qsl, _seq_lens = get_query_metadata(metadata)
        if qsl is None:
            cfg = entries[0][2]
            return _steer_rows(residuals, cfg, self._vector_cache_for(cfg))
        try:
            num_computed = self.model_runner.input_batch.num_computed_tokens_cpu
            num_prompt = self.model_runner.input_batch.num_prompt_tokens
        except Exception:  # noqa: BLE001
            num_computed = num_prompt = None

        out = residuals.clone()
        touched = False
        for (i, req_state, cfg, phase, positions) in entries:
            if i + 1 >= len(qsl):
                continue
            start = int(qsl[i])
            end = int(qsl[i + 1])
            is_prefill = len(req_state.output_token_ids) == 0
            is_final = True
            if (is_prefill and positions == "last_token"
                    and num_computed is not None and num_prompt is not None):
                is_final = (int(num_computed[i]) + (end - start)) >= int(num_prompt[i])
            span = steer_span(phase, positions, is_prefill, is_final, start, end)
            if span is None:
                continue
            lo, hi = span
            out[lo:hi] = _steer_rows(residuals[lo:hi], cfg, self._vector_cache_for(cfg))
            touched = True
        return out if touched else residuals

    # ------------------------------------------------------------------
    # v0.3.0 CUDA-graph steering install (graph mode only)
    # ------------------------------------------------------------------

    def graph_install(self):
        """Install the CUDA-graph steering path (buffer mode).

        Thin delegating entry called by the Worker.load_model monkey-patch
        (graph/install.py:patch_worker_load_model) AFTER the model is built but
        BEFORE warm-up/compile/capture. Only reached when graph mode is armed;
        the eager v0.2.0 register_forward_hook path is untouched.

        Unlike the QK/HS capture workers (which seed egress buckets here), steering
        produces no artifacts — it only mutates the residual — so this just wires
        the steer_buffer op via graph.install_steer.
        """
        from vllm_hook_plugins.graph.install_steer import install_steer_hosts
        install_steer_hosts(self)
