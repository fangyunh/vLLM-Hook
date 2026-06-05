import os
import glob
from typing import Any, Callable, Dict, Optional

import torch
from transformers import AutoModelForCausalLM

from vllm_hook_plugins.utils.TokenHighlighter.grad_influence import compute_grad_influences
from vllm_hook_plugins.utils.TokenHighlighter.utils import (
    build_highlighter_record,
    flag_driver_tokens,
    load_highlighter_artifact,
    locate_last_attention,
)


def _latest_run_id(run_id_file: str) -> str:
    """Return the most recent non-empty run id from ``VLLM_RUN_ID`` file."""
    with open(run_id_file, "r") as f:
        ids = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    if not ids:
        raise RuntimeError(f"No run ids found in {run_id_file}.")
    return ids[-1]


class HighlighterAnalyzer:
    """Score activations from capture worker; optional custom ``loss_grad_fn``.

    When ``highlighter_activations.pt`` includes ``weight_bundle`` (saved by the vLLM
    worker at capture), scoring uses pure tensor math — no ``from_pretrained``.
    """

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads
        self._model = None
        self._last_attn = None

    def _load_model(self, model_path: str | None) -> torch.nn.Module:
        """Legacy fallback when activations lack ``weight_bundle`` (old traces)."""
        if self._model is not None:
            return self._model
        path = model_path or os.environ.get("VLLM_HIGHLIGHTER_MODEL")
        if not path:
            raise RuntimeError(
                "HighlighterAnalyzer needs weight_bundle in activations, or "
                "analyzer_spec['model_path'] / VLLM_HIGHLIGHTER_MODEL for legacy traces."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=os.path.isdir(path),
            torch_dtype=torch.float16,
        ).to(device)
        self._model.eval()
        self._last_attn = locate_last_attention(self._model)
        return self._model

    def analyze(self, analyzer_spec: Optional[Dict] = None, run_id: str | None = None) -> Optional[Dict]:
        cfg = analyzer_spec or {}
        run_id_file = cfg.get("run_id_file") or os.environ.get("VLLM_RUN_ID")
        if not run_id_file and self.hook_dir:
            run_id_file = os.path.join(self.hook_dir, "RUN_ID.txt")
        if not run_id_file:
            return None

        run_id = run_id or cfg.get("run_id") or _latest_run_id(run_id_file)
        act_trace = load_highlighter_artifact(
            self.hook_dir,
            run_id,
            "highlighter_activations.pt",
            wait_seconds=float(cfg.get("artifact_wait_seconds", 2.0)),
            poll_interval=float(cfg.get("artifact_poll_interval", 0.05)),
        )
        if act_trace is None:
            trace = self._load_scores_trace(run_id)
            if trace is None:
                return {"run_id": run_id, "results": []}
            return self._format_results(trace, cfg)

        loss_grad_fn: Callable[..., torch.Tensor] | None = cfg.get("loss_grad_fn")
        weight_bundle = act_trace.get("weight_bundle")
        device_str = cfg.get("device")
        if device_str:
            device = torch.device(device_str)
        elif weight_bundle is not None:
            device = torch.device("cpu")
        else:
            device = None

        model = None
        last_attn = None
        if weight_bundle is None:
            model = self._load_model(cfg.get("model_path"))
            last_attn = self._last_attn
            if device is None:
                device = next(model.parameters()).device

        top_k = int(cfg.get("top_k", 5))
        sequences_out: list[dict] = []

        for seq in act_trace.get("sequences", []):
            prompt_ids = seq["token_ids"]
            plen = seq.get("prompt_len", len(prompt_ids))
            target_ids = seq.get("target_ids") or []
            if weight_bundle is not None:
                capture = {k: v for k, v in seq["capture"].items()}
            else:
                capture = {k: v.to(model.device) for k, v in seq["capture"].items()}
            # Modern bundles omit the large unembedding W_U and instead ship a tiny
            # precomputed loss gradient g (dL/dh^N at generation positions). Feed it back
            # through loss_grad_fn so scoring needs no W_U.
            seq_loss_grad_fn = loss_grad_fn
            g_loss = capture.pop("g_loss", None)
            if g_loss is not None and seq_loss_grad_fn is None:
                seq_loss_grad_fn = (
                    lambda h_L, prompt_len, W_U, dev, _g=g_loss: _g.to(
                        device=h_L.device, dtype=h_L.dtype
                    )
                )
            scores = compute_grad_influences(
                capture,
                plen,
                last_attn,
                model,
                weights=weight_bundle,
                loss_grad_fn=seq_loss_grad_fn,
                target_ids=target_ids if seq_loss_grad_fn is None else None,
                device=device,
            )
            sequences_out.append(
                build_highlighter_record(
                    prompt_ids,
                    scores,
                    tokenizer=None,
                    soft_beta=float(cfg.get("soft_beta", os.environ.get("VLLM_HIGHLIGHTER_BETA", "0.1"))),
                    threshold_k=float(
                        cfg.get("threshold_k", os.environ.get("VLLM_HIGHLIGHTER_THRESHOLD_K", "2.0"))
                    ),
                    mode=cfg.get("mode"),
                )
            )

        if cfg.get("write_scores", True) and sequences_out:
            run_dir = os.path.join(self.hook_dir, run_id, "tp_rank_0")
            os.makedirs(run_dir, exist_ok=True)
            torch.save(
                {"run_id": run_id, "sequences": sequences_out},
                os.path.join(run_dir, "highlighter.pt"),
            )

        return self._format_results(
            {"run_id": run_id, "sequences": sequences_out}, cfg, top_k=top_k
        )

    def _load_scores_trace(self, run_id: str) -> Optional[Dict]:
        patt = os.path.join(self.hook_dir, run_id, "**", "highlighter.pt")
        paths = glob.glob(patt, recursive=True)
        if not paths:
            return None
        return torch.load(max(paths, key=os.path.getmtime), map_location="cpu")

    def _format_results(
        self, trace: Dict, cfg: Dict, *, top_k: int | None = None
    ) -> Dict:
        top_k = int(top_k if top_k is not None else cfg.get("top_k", 5))
        results = []
        for seq in trace.get("sequences", []):
            scores = seq.get("token_scores", [])
            if not scores:
                continue
            drivers = seq.get("soft_indices", [])
            analysis_drivers = (
                flag_driver_tokens(
                    scores,
                    threshold_k=float(cfg.get("threshold_k", 2.0)),
                    mode=cfg.get("mode"),
                )
                if cfg.get("mode") is not None
                else drivers
            )
            token_ids = seq.get("token_ids", [])
            tokens = seq.get("tokens") or [str(t) for t in token_ids]
            soft_beta = seq.get("soft_beta")
            results.append({
                "token_ids": token_ids,
                "tokens": tokens,
                "token_scores": scores,
                "drivers": drivers,
                "analysis_drivers": analysis_drivers,
                "soft_beta": soft_beta,
                "top_tokens": self._top_tokens(token_ids, tokens, scores, top_k),
            })
        return {"run_id": trace.get("run_id"), "top_k": top_k, "results": results}

    def _top_tokens(
        self,
        token_ids: list[int],
        tokens: list[str],
        scores: list[float],
        top_k: int,
    ) -> list[dict]:
        """Format top-k scored tokens for analyzer output."""
        if top_k <= 0 or not scores:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "idx": i,
                "token": tokens[i] if i < len(tokens) else str(token_ids[i]),
                "score": scores[i],
            }
            for i in order
        ]
