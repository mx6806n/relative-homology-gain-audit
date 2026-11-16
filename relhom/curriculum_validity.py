"""Curriculum-validity diagnostics for cached top-N hybrid rankings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit
from relhom.hybrid_cache import candidate_rank_normalize, candidate_topology_minmax
from relhom.topn_hybrid import stage_one_topn_indices


def course_prerequisite_sets(
    graph: nx.DiGraph,
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Return external direct and ancestor prerequisites for each course.

    Concepts already inside the candidate course bundle are excluded from its
    prerequisite requirement set; they are treated as internally covered by the
    course rather than as prior-history requirements.
    """
    direct: dict[str, frozenset[str]] = {}
    ancestors: dict[str, frozenset[str]] = {}

    for course, raw_bundle in course_to_concepts.items():
        bundle = set(raw_bundle)
        direct_required: set[str] = set()
        ancestor_required: set[str] = set()

        for concept in bundle:
            if concept not in graph:
                raise ValueError(
                    f"course {course} contains concept absent from graph: {concept}"
                )
            direct_required.update(graph.predecessors(concept))
            ancestor_required.update(nx.ancestors(graph, concept))

        direct_required.difference_update(bundle)
        ancestor_required.difference_update(bundle)
        direct[course] = frozenset(direct_required)
        ancestors[course] = frozenset(ancestor_required)

    return direct, ancestors


def _history_concepts(
    split: LeaveTwoOutSplit,
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
) -> frozenset[str]:
    concepts: set[str] = set()
    for course in split.validation_history:
        concepts.update(course_to_concepts.get(course, ()))
    return frozenset(concepts)


def prerequisite_status_for_items(
    splits: Sequence[LeaveTwoOutSplit],
    item_ids: Sequence[str],
    item_indices: np.ndarray,
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    requirements: Mapping[str, frozenset[str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return violation and prerequisite-count arrays for selected items."""
    indices = np.asarray(item_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[0] != len(splits):
        raise ValueError("item_indices must be [users, positions]")

    violation = np.zeros(indices.shape, dtype=bool)
    required_count = np.zeros(indices.shape, dtype=np.int16)

    for row, split in enumerate(splits):
        history = _history_concepts(split, course_to_concepts)
        for col, item_index in enumerate(indices[row]):
            course = item_ids[int(item_index)]
            required = requirements.get(course, frozenset())
            required_count[row, col] = len(required)
            violation[row, col] = bool(required.difference(history))

    return violation, required_count


def target_prerequisite_profile(
    splits: Sequence[LeaveTwoOutSplit],
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    requirements: Mapping[str, frozenset[str]],
) -> dict[str, float | int]:
    """Describe whether observed validation targets satisfy prerequisites."""
    violations = 0
    with_requirements = 0

    for split in splits:
        history = _history_concepts(split, course_to_concepts)
        required = requirements.get(
            split.validation_target,
            frozenset(),
        )
        if required:
            with_requirements += 1
        if required.difference(history):
            violations += 1

    return {
        "users": len(splits),
        "targets_with_external_prerequisites": with_requirements,
        "violations": violations,
        "violation_rate": violations / len(splits),
    }


def reranked_shortlist_indices(
    logits: np.ndarray,
    deltas: np.ndarray,
    seen_mask: np.ndarray,
    *,
    top_n: int,
    alpha: float,
) -> np.ndarray:
    """Return item indices in reranked Stage-1 top-N order."""
    top_indices = stage_one_topn_indices(
        logits,
        seen_mask,
        top_n=top_n,
    ).astype(np.int64, copy=False)

    shortlist_mask = np.ones_like(seen_mask, dtype=bool)
    rows = np.arange(logits.shape[0])
    shortlist_mask[rows[:, None], top_indices] = False

    stage_full = candidate_rank_normalize(logits, shortlist_mask)
    topo_full = candidate_topology_minmax(
        deltas,
        shortlist_mask,
        clamp_negative=False,
    )
    stage = np.take_along_axis(stage_full, top_indices, axis=1)
    topo = np.take_along_axis(topo_full, top_indices, axis=1)
    scores = (
        float(alpha) * stage.astype(np.float64)
        + (1.0 - float(alpha)) * topo.astype(np.float64)
    )

    order = np.lexsort((top_indices, -scores), axis=1)
    return np.take_along_axis(top_indices, order, axis=1)


def violation_metrics(
    ranked_violation: np.ndarray,
    *,
    cutoffs: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Compute PVR@K and rank-discounted violation burden."""
    violation = np.asarray(ranked_violation, dtype=bool)
    if violation.ndim != 2:
        raise ValueError("ranked_violation must be two-dimensional")

    result: dict[str, float] = {}
    for cutoff in cutoffs:
        k = min(int(cutoff), violation.shape[1])
        if k < 1:
            raise ValueError("cutoffs must be positive")
        top = violation[:, :k].astype(np.float64)
        result[f"pvr@{cutoff}"] = float(top.mean())

        weights = 1.0 / np.log2(np.arange(2, k + 2))
        per_user = (top * weights).sum(axis=1) / weights.sum()
        result[f"discounted_pvr@{cutoff}"] = float(per_user.mean())

    return result
