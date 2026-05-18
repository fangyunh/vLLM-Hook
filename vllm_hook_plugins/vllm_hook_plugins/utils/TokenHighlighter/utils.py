"""Stateless helpers for TokenHighlighter worker / analyzer plumbing."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.utils.hooks
from vllm.forward_context import get_forward_context

_CAPTURE_KEYS = ("Q", "K", "V", "h_L")


# --- Model graph ---


def _last_transformer_block(model: torch.nn.Module) -> torch.nn.Module:
    for path in ("model.layers", "transformer.h", "layers"):
        obj: Any = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            if hasattr(obj, "__len__") and len(obj) > 0:
                block = obj[-1]
                if isinstance(block, torch.nn.Module):
                    return block
    raise RuntimeError("Could not locate last transformer block.")


def _attention_on_block(layer: torch.nn.Module) -> torch.nn.Module:
    for name in ("self_attn", "attn", "attention"):
        if (attn := getattr(layer, name, None)) is not None:
            return attn
    raise RuntimeError("Could not locate last self-attention module.")


def locate_last_attention(model: torch.nn.Module) -> torch.nn.Module:
    return _attention_on_block(_last_transformer_block(model))


def get_output_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        out = getter()
        if isinstance(out, torch.nn.Module) and hasattr(out, "weight"):
            return out
    lm_head = getattr(model, "lm_head", None)
    if (
        isinstance(lm_head, torch.nn.Module)
        and type(lm_head).__name__ != "PPMissingLayer"
        and hasattr(lm_head, "weight")
    ):
        return lm_head
    raise RuntimeError(f"Could not resolve output embeddings for {type(model).__name__}.")


def lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    return cast(torch.Tensor, get_output_embeddings(model).weight)


def uses_fused_qkv_proj(last_attn: torch.nn.Module) -> bool:
    """vLLM serving models use fused ``qkv_proj``; HF-style blocks use separate projections."""
    return hasattr(last_attn, "qkv_proj")


def get_attn_key_value_weights(
    last_attn: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (W_O, W_V, W_K) aligned with fused vs split attention layout."""
    attn = cast(Any, last_attn)
    w_o = cast(torch.Tensor, attn.o_proj.weight)
    if uses_fused_qkv_proj(last_attn):
        qkv_w = cast(torch.Tensor, attn.qkv_proj.weight)
        q_size, kv_size = int(attn.q_size), int(attn.kv_size)
        w_k = qkv_w[q_size : q_size + kv_size]
        w_v = qkv_w[q_size + kv_size : q_size + 2 * kv_size]
        return w_o, w_v, w_k
    return (
        w_o,
        cast(torch.Tensor, attn.v_proj.weight),
        cast(torch.Tensor, attn.k_proj.weight),
    )


# --- Autograd scorer weight resolution ---


def _snapshot_from_cache(model_id: str, download_dir: str) -> str | None:
    snapshots_dir = os.path.join(
        os.path.abspath(download_dir), f"models--{model_id.replace('/', '--')}", "snapshots"
    )
    if not os.path.isdir(snapshots_dir):
        return None
    entries = sorted(os.listdir(snapshots_dir))
    return os.path.join(snapshots_dir, entries[-1]) if entries else None


def _has_local_weights(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    return bool(
        names & {"model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"}
    )


def resolve_grad_model_source(model_id: str, model_runner: Any, hook_dir: str | None) -> str:
    if os.path.isdir(model_id):
        if _has_local_weights(model_id):
            return model_id
        snap = _snapshot_from_cache(model_id, model_id)
        if snap and _has_local_weights(snap):
            return snap

    download_dirs: list[str] = []
    vllm_config = getattr(model_runner, "vllm_config", None)
    load_cfg = getattr(vllm_config, "load_config", None) if vllm_config else None
    if load_cfg is not None and getattr(load_cfg, "download_dir", None):
        download_dirs.append(load_cfg.download_dir)
    if hook_dir:
        download_dirs.append(os.path.join(os.path.dirname(os.path.abspath(hook_dir)), "cache"))

    seen: set[str] = set()
    for download_dir in download_dirs:
        resolved_dir = os.path.abspath(download_dir)
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        snapshot = _snapshot_from_cache(model_id, resolved_dir)
        if snapshot and _has_local_weights(snapshot):
            return snapshot

    raise RuntimeError(
        f"Could not resolve a local weight snapshot for '{model_id}'. "
        "Download the model into your vLLM cache (download_dir) or set "
        "VLLM_HIGHLIGHTER_MODEL to the snapshot directory."
    )


# --- Driver selection / soft removal ---


def flag_driver_tokens(
    scores: list[float],
    *,
    threshold_k: float,
    mode: str | None = None,
) -> list[int]:
    if not scores:
        return []
    mode = mode or os.environ.get("VLLM_HIGHLIGHTER_MODE", "mean_std")
    s = torch.tensor(scores, dtype=torch.float32)
    if mode == "top_percentage":
        alpha = float(os.environ.get("VLLM_HIGHLIGHTER_ALPHA", "0.25"))
        k = max(1, int(len(scores) * alpha))
        return sorted(torch.argsort(s, descending=True)[:k].tolist())
    threshold = s.mean() + threshold_k * s.std(unbiased=False)
    return [i for i, v in enumerate(scores) if float(v) > float(threshold)]


def apply_soft_removal(
    prompt_embeds: torch.Tensor,
    score_list: list[float],
    *,
    soft_beta: float,
    threshold_k: float,
    mode: str | None = None,
) -> torch.Tensor:
    drivers = set(flag_driver_tokens(score_list, threshold_k=threshold_k, mode=mode))
    rows = prompt_embeds.detach().squeeze(0).clone()
    return torch.stack(
        [rows[i] * soft_beta if i in drivers else rows[i] for i in range(rows.size(0))],
        dim=0,
    )


# --- Scheduler request / trace I/O ---


def iter_prompt_batches(request: Any) -> list[list[int]]:
    batches: list[list[int]] = []
    for attr in ("seq_group_metadata_list", "scheduled_new_reqs"):
        for obj in getattr(request, attr, []):
            if prompt_ids := getattr(obj, "prompt_token_ids", None):
                batches.append(list(prompt_ids))
    return batches


def request_has_prefill_prompts(request: Any) -> bool:
    return bool(iter_prompt_batches(request))


def prompt_prefill_done(model_runner: Any, prompt_ids: list[int]) -> bool:
    """True when every in-flight request matching ``prompt_ids`` finished prefill."""
    requests = getattr(model_runner, "requests", None)
    input_batch = getattr(model_runner, "input_batch", None)
    if requests is None or input_batch is None:
        return True

    prompt_len = len(prompt_ids)
    matched = False
    for req_id in list(input_batch.req_ids)[: input_batch.num_reqs]:
        req = requests.get(req_id)
        if req is None:
            continue
        ptoks = req.prompt_token_ids
        if ptoks is None:
            continue
        if len(ptoks) < prompt_len or list(ptoks[:prompt_len]) != prompt_ids:
            continue
        matched = True
        if int(req.num_computed_tokens) < prompt_len:
            return False
    return matched


def save_highlighter_sequences(
    hook_dir: str | None,
    run_id_file: str | None,
    sequences: list[dict],
) -> None:
    if not (hook_dir and sequences and run_id_file and os.path.exists(run_id_file)):
        return
    with open(run_id_file, "r") as f:
        ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    if not ids:
        return
    run_id = ids[-1]
    run_dir = os.path.join(hook_dir, run_id, "tp_rank_0")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(
        {"run_id": run_id, "sequences": sequences},
        os.path.join(run_dir, "highlighter.pt"),
    )


def build_highlighter_record(
    prompt_ids: list[int],
    score_list: list[float],
    *,
    tokenizer: Any,
    soft_beta: float,
    threshold_k: float,
    mode: str | None = None,
) -> dict:
    mode = mode or os.environ.get("VLLM_HIGHLIGHTER_MODE", "mean_std")
    record = {
        "token_ids": list(prompt_ids),
        "token_scores": score_list,
        "soft_indices": flag_driver_tokens(score_list, threshold_k=threshold_k, mode=mode),
        "soft_beta": soft_beta,
        "applied_mode": mode,
        "applied_threshold_k": threshold_k,
        "applied_alpha": float(os.environ.get("VLLM_HIGHLIGHTER_ALPHA", "0.25")),
    }
    if tokenizer is not None:
        record["tokens"] = [
            tokenizer.decode([tid], skip_special_tokens=False) for tid in prompt_ids
        ]
    return record


# --- Forward-attr capture ---


@dataclass
class ForwardAttrCapture:
    """Mutable buffers filled by hooks during a real ``execute_model`` prefill."""

    active: bool = False
    live: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def forward_attr_capture_is_ready(cap: ForwardAttrCapture) -> bool:
    """True when hooks populated all last-layer tensors for the current forward."""
    return all(
        key in cap.live and cap.live[key].numel() > 0 for key in _CAPTURE_KEYS
    )


def slice_real_prefill_capture(
    prompt_ids: list[int],
    live: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract per-prompt Q/K/V/h_L from a batched vLLM prefill capture.

    During ``super().execute_model()``, hooks store the *entire* flattened batch in
    ``live`` (all requests, all scheduled query tokens). ``meta`` holds:

    - ``query_start_loc``: length ``num_reqs + 1``; request ``r`` occupies token indices
      ``[query_start_loc[r], query_start_loc[r+1])`` in the flat layout.
    - ``input_ids``: the token id at each flat index (for matching).

    This function finds the batch row whose first ``len(prompt_ids)`` tokens equal
    ``prompt_ids``, then slices each of ``Q``, ``K``, ``V``, ``h_L`` to that row's
    prompt positions only (not decode tokens that may share the same row).
    """
    if not forward_attr_capture_is_ready(
        ForwardAttrCapture(live=live, meta=meta)
    ):
        raise RuntimeError(
            "forward_attr: no prefill capture available; hooks did not run during execute_model."
        )

    if not meta:
        # Single-request prefill with no batch metadata (e.g. attn_metadata unavailable).
        h_len = live["h_L"].size(0)
        if h_len < len(prompt_ids):
            raise RuntimeError(
                f"forward_attr: capture length {h_len} < prompt length {len(prompt_ids)}."
            )
        return {key: live[key][: len(prompt_ids)].detach() for key in _CAPTURE_KEYS}

    query_start_loc = meta["query_start_loc"]
    flat_ids = meta["input_ids"]

    for req_idx in range(int(query_start_loc.numel()) - 1):
        start = int(query_start_loc[req_idx].item())
        seg = flat_ids[start : int(query_start_loc[req_idx + 1].item())].tolist()
        if len(seg) >= len(prompt_ids) and seg[: len(prompt_ids)] == prompt_ids:
            end_idx = start + len(prompt_ids)
            captured = {key: live[key][start:end_idx].detach() for key in _CAPTURE_KEYS}
            if len(captured) != len(_CAPTURE_KEYS):
                raise RuntimeError("forward_attr prefill capture is missing Q, K, V, or h_L.")
            return captured

    raise RuntimeError(
        f"forward_attr: could not match prefill capture to prompt_ids (len={len(prompt_ids)})."
    )


def merge_real_and_teacher_captures(
    prompt_len: int,
    real_capture: dict[str, torch.Tensor],
    teacher_capture: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    merged = {key: teacher_capture[key].clone() for key in _CAPTURE_KEYS}
    for key in merged:
        n = min(prompt_len, real_capture[key].size(0), merged[key].size(0))
        merged[key][:n] = real_capture[key][:n]
    return merged


def teacher_forced_input_ids(
    prompt_ids: list[int],
    target_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    target = target_tensor.to(device).squeeze(0)
    return torch.cat([prompt, target[:-1]])


def _decoder_layer_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        h0 = cast(torch.Tensor, output[0])
        return (
            h0 + cast(torch.Tensor, output[1])
            if len(output) == 2 and isinstance(output[1], torch.Tensor)
            else h0
        )
    return cast(torch.Tensor, output)


def record_batch_meta_from_runner(
    model_runner: Any, meta_dest: dict[str, Any]
) -> bool:
    """Fill ``meta_dest`` from the model runner after a real forward (fallback path)."""
    input_batch = getattr(model_runner, "input_batch", None)
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    if num_reqs <= 0:
        return False

    query_start_loc = None
    qsl_buf = getattr(model_runner, "query_start_loc", None)
    if qsl_buf is not None and hasattr(qsl_buf, "gpu"):
        query_start_loc = qsl_buf.gpu[: num_reqs + 1].detach().clone()
    elif input_batch is not None and hasattr(input_batch, "query_start_loc_np"):
        qsl_np = input_batch.query_start_loc_np[: num_reqs + 1]
        device = model_runner.input_ids.gpu.device
        query_start_loc = torch.tensor(qsl_np, dtype=torch.int32, device=device)

    if query_start_loc is None:
        return False

    num_tokens = int(query_start_loc[-1].item())
    if num_tokens <= 0:
        return False

    meta_dest.clear()
    meta_dest.update(
        {
            "query_start_loc": query_start_loc,
            "input_ids": model_runner.input_ids.gpu[:num_tokens].detach().clone(),
        }
    )
    return True


def _record_batch_meta(model_runner: Any, meta_dest: dict[str, Any]) -> None:
    ctx = get_forward_context()
    attn_md = getattr(ctx, "attn_metadata", None)
    if attn_md is not None and not (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        query_start_loc = getattr(attn_md, "query_start_loc", None)
        if query_start_loc is None and isinstance(attn_md, dict):
            for entry in attn_md.values():
                if (query_start_loc := getattr(entry, "query_start_loc", None)) is not None:
                    break
        if query_start_loc is not None:
            num_tokens = int(getattr(ctx, "num_tokens", 0) or 0) or int(
                query_start_loc[-1].item()
            )
            meta_dest.clear()
            meta_dest.update(
                {
                    "query_start_loc": query_start_loc.detach().clone(),
                    "input_ids": model_runner.input_ids.gpu[:num_tokens]
                    .detach()
                    .clone(),
                }
            )
            return

    record_batch_meta_from_runner(model_runner, meta_dest)


def _register_qkv_capture_hooks(
    last_attn: torch.nn.Module,
    dest: dict[str, torch.Tensor],
    *,
    enabled: Callable[[], bool],
    after_qkv: Callable[[], None] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks that write last-layer Q/K/V into ``dest``.

    vLLM serving (fused): single forward hook on ``qkv_proj`` output, split on ``q_size``/``kv_size``.
    HF-style (split): separate hooks on ``q_proj``, ``k_proj``, ``v_proj`` outputs.
    """
    attn = cast(Any, last_attn)
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _done() -> None:
        if after_qkv is not None:
            after_qkv()

    if uses_fused_qkv_proj(last_attn):
        q_size, kv_size = int(attn.q_size), int(attn.kv_size)

        def qkv_hook(_module, _inputs, output: Any) -> None:
            if not enabled():
                return
            qkv = cast(torch.Tensor, output[0] if isinstance(output, tuple) else output)
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            dest["Q"], dest["K"], dest["V"] = q.detach(), k.detach(), v.detach()
            _done()

        hooks.append(attn.qkv_proj.register_forward_hook(qkv_hook))
        return hooks

    def split_hook(key: str):
        def hook(_m, _i, output: Any) -> None:
            if not enabled():
                return
            dest[key] = cast(torch.Tensor, output).detach()
            if key == "Q":
                _done()

        return hook

    for key, attr in (("Q", "q_proj"), ("K", "k_proj"), ("V", "v_proj")):
        hooks.append(getattr(attn, attr).register_forward_hook(split_hook(key)))
    return hooks


def _register_forward_attr_hooks(
    model: torch.nn.Module,
    dest: dict[str, torch.Tensor],
    *,
    enabled: Callable[[], bool],
    model_runner: Any | None = None,
    meta_dest: dict[str, Any] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register last-layer Q/K/V (+ h_L) hooks writing into ``dest``."""
    last_layer = _last_transformer_block(model)
    last_attn = _attention_on_block(last_layer)

    def after_qkv() -> None:
        if model_runner is not None and meta_dest is not None and enabled():
            _record_batch_meta(model_runner, meta_dest)

    hooks = _register_qkv_capture_hooks(
        last_attn, dest, enabled=enabled, after_qkv=after_qkv
    )

    def layer_hook(_module, _inputs, output: Any) -> None:
        if not enabled():
            return
        dest["h_L"] = _decoder_layer_hidden(output).detach()
        after_qkv()

    hooks.append(last_layer.register_forward_hook(layer_hook))
    return hooks


def install_persistent_forward_attr_hooks(
    model: torch.nn.Module,
    cap: ForwardAttrCapture,
    model_runner: Any,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks that write last-layer Q/K/V (+ h_L) into `cap.live`.
    The hook is persistent and remains installed until `cap.active` is False.
    Only captures prompt activations from real prefill."""
    return _register_forward_attr_hooks(
        model,
        cap.live,
        enabled=lambda: cap.active,
        model_runner=model_runner,
        meta_dest=cap.meta,
    )


def register_ephemeral_forward_attr_hooks(
    model: torch.nn.Module,
    captured: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register hooks that write last-layer Q/K/V (+ h_L) into `captured`
    during teacher-forced forward pass to get activations at required positions for 
    affirmation loss. The hook is removed after each pass."""
    return _register_forward_attr_hooks(model, captured, enabled=lambda: True)
