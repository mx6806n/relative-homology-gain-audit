"""Top-N cache-only reranking for the intended two-stage hybrid."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from relhom.hybrid_cache import (
    candidate_rank_normalize,
    candidate_topology_minmax,
    metrics_from_ranks,
    target_ranks_from_scores,
)


def stage_one_topn_indices(
    logits: np.ndarray,
    seen_mask: np.ndarray,
    *,
    top_n: int,
) -> np.ndarray:
    """Return Stage-1 top-N course indices with lower-index tie breaking."""
    logits = np.asarray(logits, dtype=np.float64)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    if logits.shape != seen_mask.shape:
        raise ValueError("logits and seen_mask must have identical shape")
    if top_n < 1:
        raise ValueError("top_n must be positive")

    candidate_counts = (~seen_mask).sum(axis=1)
    if np.any(candidate_counts < top_n):
        minimum = int(candidate_counts.min())
        raise ValueError(
            f"top_n={top_n} exceeds minimum unseen candidate count={minimum}"
        )

    working = logits.copy()
    working[seen_mask] = -np.inf
    return np.argsort(
        -working,
        axis=1,
        kind="stable",
    )[:, :top_n].astype(np.int16, copy=False)


def topn_rerank_grid(
    logits: np.ndarray,
    deltas: np.ndarray,
    seen_mask: np.ndarray,
    target_indices: np.ndarray,
    *,
    top_n: int,
    alphas: Sequence[float],
    cutoffs: Iterable[int] = (5, 10, 20),
    topology_normalization: str = "minmax",
    return_ranks: bool = False,
) -> dict[str, object]:
    """Rerank only the frozen SASRec top-N shortlist with signed topology.

    ``topology_normalization`` is "minmax" (exp_013 protocol, default)
    or "rank" (shortlist-wise average-tie RankNorm, the exp_024 sensitivity
    analysis). With ``return_ranks=True`` each result also carries the
    per-user target ranks under key "ranks" (not JSON-serialisable).

    Items outside the shortlist remain in their original Stage-1 order. Since
    only target rank is needed for one-relevant-item metrics, targets outside
    top-N retain their original SASRec rank exactly.
    """
    logits = np.asarray(logits, dtype=np.float64)
    deltas = np.asarray(deltas)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    targets = np.asarray(target_indices, dtype=np.int64)

    if logits.shape != deltas.shape or logits.shape != seen_mask.shape:
        raise ValueError("logits, deltas, and seen_mask must align")
    if targets.shape != (logits.shape[0],):
        raise ValueError("target_indices shape mismatch")

    top_indices = stage_one_topn_indices(
        logits,
        seen_mask,
        top_n=top_n,
    ).astype(np.int64, copy=False)

    rows = np.arange(logits.shape[0])
    target_membership = top_indices == targets[:, None]
    target_in_shortlist = target_membership.any(axis=1)
    target_positions = np.argmax(target_membership, axis=1)

    baseline_ranks = target_ranks_from_scores(
        logits,
        seen_mask,
        targets,
    )

    shortlist_mask = np.ones_like(seen_mask, dtype=bool)
    shortlist_mask[rows[:, None], top_indices] = False

    stage_norm_full = candidate_rank_normalize(
        logits,
        shortlist_mask,
    )
    if topology_normalization == "minmax":
        topology_norm_full = candidate_topology_minmax(
            deltas,
            shortlist_mask,
            clamp_negative=False,
        )
    elif topology_normalization == "rank":
        topology_norm_full = candidate_rank_normalize(
            np.asarray(deltas, dtype=np.float64),
            shortlist_mask,
        )
    else:
        raise ValueError("topology_normalization must be 'minmax' or 'rank'")
    stage_norm = np.take_along_axis(
        stage_norm_full,
        top_indices,
        axis=1,
    ).astype(np.float64, copy=False)
    topology_norm = np.take_along_axis(
        topology_norm_full,
        top_indices,
        axis=1,
    ).astype(np.float64, copy=False)

    results: list[dict[str, object]] = []
    for alpha_value in alphas:
        alpha = float(alpha_value)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha values must lie in [0, 1]")

        hybrid = (
            alpha * stage_norm
            + (1.0 - alpha) * topology_norm
        )
        ranks = baseline_ranks.copy()

        if np.any(target_in_shortlist):
            active_rows = np.flatnonzero(target_in_shortlist)
            positions = target_positions[active_rows]
            target_scores = hybrid[active_rows, positions]
            target_item_indices = targets[active_rows]

            active_scores = hybrid[active_rows]
            active_items = top_indices[active_rows]
            better = active_scores > target_scores[:, None]
            tied_before = (
                (active_scores == target_scores[:, None])
                & (active_items < target_item_indices[:, None])
            )
            ranks[active_rows] = (
                1
                + better.sum(axis=1)
                + tied_before.sum(axis=1)
            ).astype(np.int32)

        entry: dict[str, object] = {}
        if return_ranks:
            entry["ranks"] = ranks.copy()
        results.append(
            {
                **entry,
                "top_n": top_n,
                "alpha": alpha,
                "metrics": metrics_from_ranks(
                    ranks,
                    cutoffs=cutoffs,
                ),
                "target_in_shortlist": int(
                    target_in_shortlist.sum()
                ),
                "target_in_shortlist_fraction": float(
                    target_in_shortlist.mean()
                ),
                "rank_movement_vs_sasrec": {
                    "improved": int((ranks < baseline_ranks).sum()),
                    "worsened": int((ranks > baseline_ranks).sum()),
                    "unchanged": int((ranks == baseline_ranks).sum()),
                    "mean_rank_improvement": float(
                        np.mean(baseline_ranks - ranks)
                    ),
                },
            }
        )

    return {
        "top_n": top_n,
        "target_in_shortlist": int(target_in_shortlist.sum()),
        "target_in_shortlist_fraction": float(
            target_in_shortlist.mean()
        ),
        "results": results,
    }


def select_topn_alpha(
    results: Sequence[dict[str, object]],
    *,
    metric: str = "ndcg@10",
) -> dict[str, object]:
    """Select by metric, then larger alpha, then smaller shortlist."""
    if not results:
        raise ValueError("results must not be empty")

    def key(result: dict[str, object]) -> tuple[float, float, float]:
        metrics = result["metrics"]
        if not isinstance(metrics, dict) or metric not in metrics:
            raise ValueError(f"selection metric missing: {metric}")
        return (
            float(metrics[metric]),
            float(result["alpha"]),
            -float(result["top_n"]),
        )

    return dict(max(results, key=key))
