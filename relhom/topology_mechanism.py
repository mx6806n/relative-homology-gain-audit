"""Mechanism diagnostics for the cached relative-homology signal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from relhom.experiment import LeaveTwoOutSplit


@dataclass(frozen=True)
class CorrelationSummary:
    global_spearman: float | None
    per_user_mean: float | None
    per_user_median: float | None
    per_user_defined: int


def _safe_spearman(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size != right.size:
        raise ValueError("correlation arrays must have equal size")
    if left.size < 2:
        return None
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return None if np.isnan(value) else value


def candidate_structural_features(
    splits: Sequence[LeaveTwoOutSplit],
    item_ids: Sequence[str],
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    graph: nx.DiGraph,
    seen_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build candidate feature matrices aligned to the frozen score caches."""
    shape = (len(splits), len(item_ids))
    if seen_mask.shape != shape:
        raise ValueError("seen_mask shape does not match users/courses")

    bundle_size_by_item = np.asarray(
        [len(course_to_concepts.get(item, ())) for item in item_ids],
        dtype=np.int16,
    )
    direct_external_by_item: list[frozenset[str]] = []
    ancestor_external_by_item: list[frozenset[str]] = []

    for item in item_ids:
        bundle = set(course_to_concepts.get(item, ()))
        direct: set[str] = set()
        ancestors: set[str] = set()
        for concept in bundle:
            direct.update(graph.predecessors(concept))
            ancestors.update(nx.ancestors(graph, concept))
        direct.difference_update(bundle)
        ancestors.difference_update(bundle)
        direct_external_by_item.append(frozenset(direct))
        ancestor_external_by_item.append(frozenset(ancestors))

    bundle_size = np.broadcast_to(
        bundle_size_by_item[None, :],
        shape,
    ).copy()
    new_concept_count = np.zeros(shape, dtype=np.int16)
    unmet_direct_count = np.zeros(shape, dtype=np.int16)
    unmet_ancestor_count = np.zeros(shape, dtype=np.int16)
    history_concept_count = np.zeros(len(splits), dtype=np.int16)

    for row, split in enumerate(splits):
        history: set[str] = set()
        for course in split.validation_history:
            history.update(course_to_concepts.get(course, ()))
        history_concept_count[row] = len(history)

        for col, item in enumerate(item_ids):
            if seen_mask[row, col]:
                continue
            bundle = set(course_to_concepts.get(item, ()))
            new_concept_count[row, col] = len(bundle.difference(history))
            unmet_direct_count[row, col] = len(
                direct_external_by_item[col].difference(history)
            )
            unmet_ancestor_count[row, col] = len(
                ancestor_external_by_item[col].difference(history)
            )

    return {
        "bundle_size": bundle_size,
        "new_concept_count": new_concept_count,
        "unmet_direct_count": unmet_direct_count,
        "unmet_ancestor_count": unmet_ancestor_count,
        "history_concept_count": history_concept_count,
    }


def correlation_summary(
    signal: np.ndarray,
    feature: np.ndarray,
    seen_mask: np.ndarray,
) -> CorrelationSummary:
    """Compute global and per-user Spearman correlations on unseen entries."""
    if signal.shape != feature.shape or signal.shape != seen_mask.shape:
        raise ValueError("signal, feature, and mask must align")

    candidate_mask = ~seen_mask
    global_value = _safe_spearman(
        signal[candidate_mask],
        feature[candidate_mask],
    )

    per_user: list[float] = []
    for row in range(signal.shape[0]):
        mask = candidate_mask[row]
        value = _safe_spearman(
            signal[row, mask],
            feature[row, mask],
        )
        if value is not None:
            per_user.append(value)

    return CorrelationSummary(
        global_spearman=global_value,
        per_user_mean=(
            float(np.mean(per_user)) if per_user else None
        ),
        per_user_median=(
            float(np.median(per_user)) if per_user else None
        ),
        per_user_defined=len(per_user),
    )


def target_vs_candidates(
    signal: np.ndarray,
    feature: np.ndarray,
    seen_mask: np.ndarray,
    target_indices: np.ndarray,
) -> dict[str, float | int]:
    """Compare target feature/signal values with each user's unseen pool."""
    targets = np.asarray(target_indices, dtype=np.int64)
    rows = np.arange(signal.shape[0])

    target_signal = signal[rows, targets].astype(np.float64)
    target_feature = feature[rows, targets].astype(np.float64)

    signal_percentiles: list[float] = []
    feature_percentiles: list[float] = []
    for row in range(signal.shape[0]):
        mask = ~seen_mask[row]
        row_signal = signal[row, mask]
        row_feature = feature[row, mask]
        signal_percentiles.append(
            float(np.mean(row_signal <= target_signal[row]))
        )
        feature_percentiles.append(
            float(np.mean(row_feature <= target_feature[row]))
        )

    return {
        "users": int(signal.shape[0]),
        "target_signal_mean": float(np.mean(target_signal)),
        "target_signal_median": float(np.median(target_signal)),
        "target_feature_mean": float(np.mean(target_feature)),
        "target_feature_median": float(np.median(target_feature)),
        "mean_target_signal_weak_percentile": float(
            np.mean(signal_percentiles)
        ),
        "mean_target_feature_weak_percentile": float(
            np.mean(feature_percentiles)
        ),
    }


def linear_structural_r2(
    signal: np.ndarray,
    features: Sequence[np.ndarray],
    seen_mask: np.ndarray,
) -> float:
    """OLS R^2 for signal explained by simple structural count features."""
    candidate_mask = ~seen_mask
    y = signal[candidate_mask].astype(np.float64)
    columns = [
        np.ones(y.shape[0], dtype=np.float64),
    ]
    for feature in features:
        if feature.shape != signal.shape:
            raise ValueError("feature shape mismatch")
        columns.append(feature[candidate_mask].astype(np.float64))

    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    total = y - y.mean()
    ss_res = float(residual @ residual)
    ss_tot = float(total @ total)
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
