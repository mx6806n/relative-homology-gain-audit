"""Helpers for exp_024: normalization and near-tie sensitivity of the null."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def per_user_ndcg_at_k(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    """One-relevant-item NDCG@k per user from one-based target ranks."""
    ranks = np.asarray(ranks, dtype=np.float64)
    if np.any(ranks < 1):
        raise ValueError("ranks must be >= 1")
    return np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)


def paired_bootstrap_difference(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """User-level paired bootstrap of mean(candidate - baseline)."""
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError("inputs must align")
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        means[i] = diff[rng.integers(0, diff.size, size=diff.size)].mean()
    return {
        "mean_difference": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "users": int(diff.size),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def select_best(results: Sequence[Mapping[str, object]], metric: str = "ndcg@10"):
    """Best metric; ties prefer larger alpha (less reranking), as in exp_012/013."""
    return max(
        results,
        key=lambda r: (float(r["metrics"][metric]), float(r["alpha"])),
    )


def classify_configuration(selected_alpha: float, ci95: Sequence[float]) -> str:
    """Predeclared exp_024 rule (docs/analysis_plan.md, D-080) for one configuration."""
    if selected_alpha >= 1.0:
        return "NULL_SELECTS_SASREC"
    if ci95[0] > 0.0:
        return "IMPROVES_OVER_SASREC"
    return "NO_RELIABLE_IMPROVEMENT"


def overall_classification(labels: Sequence[str]) -> str:
    return (
        "NORMALIZATION_OR_GRID_SENSITIVE"
        if any(label == "IMPROVES_OVER_SASREC" for label in labels)
        else "NULL_ROBUST"
    )
