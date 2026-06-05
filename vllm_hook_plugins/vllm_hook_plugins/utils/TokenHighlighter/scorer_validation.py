"""Compare autograd (reference) vs forward_attr (approximation) token influence scores."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from vllm_hook_plugins.utils.TokenHighlighter.utils import flag_driver_tokens


def _as_float_vector(scores: list[float]) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64)


def _driver_set(
    scores: list[float],
    *,
    mode: str,
    threshold_k: float,
    alpha: float,
) -> set[int]:
    return set(flag_driver_tokens(scores, threshold_k=threshold_k, mode=mode, alpha=alpha))


def _topk_set(scores: list[float], k: int) -> set[int]:
    k = max(1, min(k, len(scores)))
    order = np.argsort(-_as_float_vector(scores))[:k]
    return set(int(i) for i in order)


def _ndcg_at_k(relevance: np.ndarray, ranked_desc: np.ndarray, k: int) -> float:
    k = min(k, len(relevance))
    if k <= 0:
        return float("nan")
    gains = relevance[ranked_desc[:k]]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def compare_token_scores(
    scores_ref: list[float],
    scores_approx: list[float],
    *,
    token_ids_ref: list[int] | None = None,
    token_ids_approx: list[int] | None = None,
    driver_mode: str | None = None,
    threshold_k: float = 2.0,
    alpha: float = 0.25,
    top_k_fracs: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> dict[str, Any]:
    """Compare reference (autograd) vs approximation (forward_attr) on one prompt.

    Returns rank correlations, driver-set overlap, top-k overlaps, and a deployability hint.
    """
    if token_ids_ref is not None and token_ids_approx is not None and token_ids_ref != token_ids_approx:
        return {"ok": False, "error": "token_ids mismatch between scorers"}

    n = len(scores_ref)
    if n == 0 or len(scores_approx) != n:
        return {
            "ok": False,
            "error": f"length mismatch: ref={n}, approx={len(scores_approx)}",
        }

    ref = _as_float_vector(scores_ref)
    approx = _as_float_vector(scores_approx)
    mode = driver_mode or __import__("os").environ.get("VLLM_HIGHLIGHTER_MODE", "top_percentage")

    if np.allclose(ref, ref[0]) or np.allclose(approx, approx[0]):
        return {"ok": False, "error": "constant score vector — correlation undefined"}

    rho, p_sp = spearmanr(ref, approx)
    tau, p_kt = kendalltau(ref, approx)
    r_pearson, p_pearson = pearsonr(ref, approx)

    drivers_ref = _driver_set(scores_ref, mode=mode, threshold_k=threshold_k, alpha=alpha)
    drivers_approx = _driver_set(scores_approx, mode=mode, threshold_k=threshold_k, alpha=alpha)
    union = drivers_ref | drivers_approx
    driver_jaccard = len(drivers_ref & drivers_approx) / len(union) if union else 1.0

    topk_overlap: dict[str, float] = {}
    for frac in top_k_fracs:
        k = max(1, int(math.ceil(n * frac)))
        key = f"top_{k}"
        topk_overlap[key] = len(_topk_set(scores_ref, k) & _topk_set(scores_approx, k)) / k

    rank_approx = np.argsort(-approx)
    ndcg = {f"ndcg@{max(1, int(math.ceil(n * f)))}": _ndcg_at_k(ref, rank_approx, max(1, int(math.ceil(n * f))))
            for f in top_k_fracs}

    deployable, reason = assess_deployability(
        spearman_rho=float(rho),
        kendall_tau=float(tau),
        driver_jaccard=float(driver_jaccard),
    )

    return {
        "ok": True,
        "n_tokens": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p_sp),
        "kendall_tau": float(tau),
        "kendall_p": float(p_kt),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "driver_jaccard": float(driver_jaccard),
        "n_drivers_ref": len(drivers_ref),
        "n_drivers_approx": len(drivers_approx),
        "drivers_ref": sorted(drivers_ref),
        "drivers_approx": sorted(drivers_approx),
        "drivers_only_ref": sorted(drivers_ref - drivers_approx),
        "drivers_only_approx": sorted(drivers_approx - drivers_ref),
        "topk_overlap": topk_overlap,
        "ndcg_using_ref_as_relevance": ndcg,
        "deployable": deployable,
        "deployable_reason": reason,
        "driver_mode": mode,
        "threshold_k": threshold_k,
        "alpha": alpha,
    }


def assess_deployability(
    *,
    spearman_rho: float,
    kendall_tau: float,
    driver_jaccard: float,
    min_spearman: float = 0.75,
    min_kendall: float = 0.55,
    min_driver_jaccard: float = 0.5,
) -> tuple[bool, str]:
    """Heuristic: forward_attr is a deployable substitute when all thresholds pass."""
    checks = [
        (spearman_rho >= min_spearman, f"Spearman ρ={spearman_rho:.3f} (need ≥{min_spearman})"),
        (kendall_tau >= min_kendall, f"Kendall τ={kendall_tau:.3f} (need ≥{min_kendall})"),
        (driver_jaccard >= min_driver_jaccard, f"driver Jaccard={driver_jaccard:.3f} (need ≥{min_driver_jaccard})"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        return False, "; ".join(failed)
    return True, "ranking and flagged drivers agree within thresholds"


def print_comparison_report(metrics: dict[str, Any], *, title: str = "") -> None:
    if title:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    if not metrics.get("ok"):
        print(f"Comparison failed: {metrics.get('error', 'unknown')}")
        return
    print(f"Tokens: {metrics['n_tokens']}  |  driver mode: {metrics['driver_mode']}")
    print(
        f"Spearman ρ = {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.4g})  "
        f"|  Kendall τ = {metrics['kendall_tau']:.4f} (p={metrics['kendall_p']:.4g})"
    )
    print(f"Pearson r  = {metrics['pearson_r']:.4f} (scale-sensitive; ranks matter more)")
    print(
        f"Drivers: ref={metrics['n_drivers_ref']} approx={metrics['n_drivers_approx']}  "
        f"Jaccard={metrics['driver_jaccard']:.3f}"
    )
    if metrics.get("drivers_only_ref") or metrics.get("drivers_only_approx"):
        print(f"  only ref:    {metrics['drivers_only_ref']}")
        print(f"  only approx: {metrics['drivers_only_approx']}")
    print("Top-k index overlap (same k on both score vectors):")
    for k, v in metrics["topk_overlap"].items():
        print(f"  {k}: {v:.0%}")
    print("NDCG (approx ranking, autograd scores as relevance):")
    for k, v in metrics["ndcg_using_ref_as_relevance"].items():
        print(f"  {k}: {v:.3f}")
    tag = "YES" if metrics["deployable"] else "NO"
    print(f"Deployable forward_attr substitute? {tag} — {metrics['deployable_reason']}")
