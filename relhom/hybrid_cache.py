"""Score fusion computed from the cached SASRec logits and topology gains."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch


def validate_hybrid_cache_alignment(
    topology_cache: Mapping[str, np.ndarray],
    sasrec_cache: Mapping[str, np.ndarray],
) -> None:
    """Require identical shapes, orderings, targets and hashes across the two caches."""
    deltas = topology_cache["deltas"]
    logits = sasrec_cache["logits"]

    if deltas.shape != logits.shape:
        raise ValueError(
            f"topology shape {deltas.shape} != SASRec shape {logits.shape}"
        )

    for field in ("user_ids", "course_ids"):
        if not np.array_equal(
            topology_cache[field],
            sasrec_cache[field],
        ):
            raise ValueError(f"{field} mismatch between caches")

    if not np.array_equal(
        topology_cache["validation_target_indices"],
        sasrec_cache["validation_target_indices"],
    ):
        raise ValueError("validation target mismatch between caches")

    topology_hashes = sasrec_cache["topology_cache_sha256"]
    if topology_hashes.size != 1:
        raise ValueError("SASRec cache must reference one topology hash")


def candidate_rank_normalize(
    logits: np.ndarray,
    seen_mask: np.ndarray,
) -> np.ndarray:
    """Rank-normalize unseen logits per user to [0, 1].

    The lowest unseen score maps to 0 and the highest maps to 1. Ties receive
    average ranks. Seen entries are stored as NaN and never participate.
    """
    logits = np.asarray(logits, dtype=np.float64)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    if logits.shape != seen_mask.shape:
        raise ValueError("logits and seen_mask must have the same shape")
    if not np.isfinite(logits).all():
        raise ValueError("raw logits must be finite")

    result = np.full(logits.shape, np.nan, dtype=np.float32)

    for row in range(logits.shape[0]):
        candidate_indices = np.flatnonzero(~seen_mask[row])
        n = int(candidate_indices.size)
        if n == 0:
            raise ValueError(f"user row {row} has no unseen candidates")
        if n == 1:
            result[row, candidate_indices[0]] = 1.0
            continue

        values = logits[row, candidate_indices]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(n, dtype=np.float64)

        start = 0
        while start < n:
            stop = start + 1
            while (
                stop < n
                and sorted_values[stop] == sorted_values[start]
            ):
                stop += 1
            average_rank = ((start + 1) + stop) / 2.0
            ranks[order[start:stop]] = average_rank
            start = stop

        result[row, candidate_indices] = (
            (ranks - 1.0) / (n - 1.0)
        ).astype(np.float32)

    return result


def candidate_topology_minmax(
    deltas: np.ndarray,
    seen_mask: np.ndarray,
    *,
    clamp_negative: bool = False,
) -> np.ndarray:
    """Min-max normalize unseen topology scores per user.

    When clamp_negative=True, max(delta, 0) is applied before normalization as
    an explicit mechanism ablation. Constant candidate vectors map to 0.
    Seen entries are NaN.
    """
    deltas = np.asarray(deltas)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    if deltas.shape != seen_mask.shape:
        raise ValueError("deltas and seen_mask must have the same shape")

    result = np.full(deltas.shape, np.nan, dtype=np.float32)

    for row in range(deltas.shape[0]):
        candidate_indices = np.flatnonzero(~seen_mask[row])
        if candidate_indices.size == 0:
            raise ValueError(f"user row {row} has no unseen candidates")

        values = deltas[row, candidate_indices].astype(
            np.float64,
            copy=True,
        )
        if clamp_negative:
            np.maximum(values, 0.0, out=values)

        minimum = float(values.min())
        maximum = float(values.max())
        if maximum == minimum:
            normalized = np.zeros(values.shape, dtype=np.float32)
        else:
            normalized = (
                (values - minimum) / (maximum - minimum)
            ).astype(np.float32)

        result[row, candidate_indices] = normalized

    return result


def target_ranks_from_scores(
    scores: np.ndarray,
    seen_mask: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Return one-based target ranks with course-index tie breaking."""
    scores = np.asarray(scores, dtype=np.float64)
    seen_mask = np.asarray(seen_mask, dtype=bool)
    targets = np.asarray(target_indices, dtype=np.int64)

    if scores.shape != seen_mask.shape:
        raise ValueError("scores and seen_mask must have the same shape")
    if targets.shape != (scores.shape[0],):
        raise ValueError("target_indices shape mismatch")

    rows = np.arange(scores.shape[0])
    if bool(seen_mask[rows, targets].any()):
        raise ValueError("target is masked as seen")

    tensor_scores = torch.from_numpy(scores.copy())
    tensor_mask = torch.from_numpy(seen_mask)
    tensor_targets = torch.from_numpy(targets)
    tensor_rows = torch.arange(scores.shape[0])

    tensor_scores.masked_fill_(tensor_mask, float("-inf"))
    target_scores = tensor_scores[tensor_rows, tensor_targets]
    item_indices = torch.arange(scores.shape[1]).unsqueeze(0)

    better = tensor_scores > target_scores.unsqueeze(1)
    tied_before = (
        (tensor_scores == target_scores.unsqueeze(1))
        & (item_indices < tensor_targets.unsqueeze(1))
    )
    ranks = (
        1
        + better.sum(dim=1)
        + tied_before.sum(dim=1)
    )
    return ranks.cpu().numpy().astype(np.int32, copy=False)


def metrics_from_ranks(
    ranks: np.ndarray,
    *,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Compute one-relevant-item ranking metrics from target ranks."""
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("ranks must be a non-empty one-dimensional array")
    if np.any(ranks < 1):
        raise ValueError("all ranks must be >= 1")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")

    result: dict[str, float | int] = {
        "users": int(ranks.size),
        "missing_targets": 0,
        "mrr": float(np.mean(1.0 / ranks)),
    }
    for k in ks:
        hits = ranks <= k
        result[f"recall@{k}"] = float(np.mean(hits))
        result[f"ndcg@{k}"] = float(
            np.mean(
                np.where(
                    hits,
                    1.0 / np.log2(ranks + 1.0),
                    0.0,
                )
            )
        )
    return result


def evaluate_hybrid_grid(
    rank_norm: np.ndarray,
    topology_norm: np.ndarray,
    seen_mask: np.ndarray,
    target_indices: np.ndarray,
    alphas: Sequence[float],
    *,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> list[dict[str, object]]:
    """Evaluate a fixed alpha grid without retraining or recomputation."""
    if rank_norm.shape != topology_norm.shape:
        raise ValueError("normalized score matrices must have the same shape")
    if rank_norm.shape != seen_mask.shape:
        raise ValueError("normalized scores and mask must have the same shape")

    results: list[dict[str, object]] = []
    for alpha_value in alphas:
        alpha = float(alpha_value)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha values must lie in [0, 1]")

        scores = (
            alpha * rank_norm.astype(np.float64)
            + (1.0 - alpha) * topology_norm.astype(np.float64)
        )
        ranks = target_ranks_from_scores(
            scores,
            seen_mask,
            target_indices,
        )
        results.append(
            {
                "alpha": alpha,
                "metrics": metrics_from_ranks(
                    ranks,
                    cutoffs=cutoffs,
                ),
                "rank_summary": {
                    "mean": float(np.mean(ranks)),
                    "median": float(np.median(ranks)),
                    "p90": float(np.percentile(ranks, 90)),
                },
            }
        )
    return results


def select_alpha(
    results: Sequence[Mapping[str, object]],
    *,
    metric: str = "ndcg@10",
) -> dict[str, object]:
    """Select best validation alpha; ties prefer larger alpha (less reranking)."""
    if not results:
        raise ValueError("results must not be empty")

    def key(result: Mapping[str, object]) -> tuple[float, float]:
        metrics = result["metrics"]
        if not isinstance(metrics, Mapping) or metric not in metrics:
            raise ValueError(f"selection metric missing: {metric}")
        return (
            float(metrics[metric]),
            float(result["alpha"]),
        )

    return dict(max(results, key=key))


def compare_ranks(
    baseline_ranks: np.ndarray,
    hybrid_ranks: np.ndarray,
) -> dict[str, float | int]:
    """Paired target-rank movement diagnostic."""
    baseline = np.asarray(baseline_ranks, dtype=np.int64)
    hybrid = np.asarray(hybrid_ranks, dtype=np.int64)
    if baseline.shape != hybrid.shape:
        raise ValueError("rank arrays must have identical shape")

    improved = hybrid < baseline
    worsened = hybrid > baseline
    unchanged = hybrid == baseline
    delta = baseline - hybrid

    return {
        "users": int(baseline.size),
        "improved": int(improved.sum()),
        "worsened": int(worsened.sum()),
        "unchanged": int(unchanged.sum()),
        "mean_rank_improvement": float(np.mean(delta)),
        "median_rank_improvement": float(np.median(delta)),
    }
