"""Tests for curriculum-validity diagnostics."""

from __future__ import annotations

import networkx as nx
import numpy as np

from relhom.curriculum_validity import (
    course_prerequisite_sets,
    reranked_shortlist_indices,
    violation_metrics,
)


def test_course_requirements_exclude_within_course_concepts() -> None:
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    direct, ancestors = course_prerequisite_sets(
        graph,
        {
            "AB": {"A", "B"},
            "C": {"C"},
        },
    )

    assert direct["AB"] == frozenset()
    assert direct["C"] == frozenset({"B"})
    assert ancestors["C"] == frozenset({"A", "B"})


def test_reranked_shortlist_uses_course_index_for_score_ties() -> None:
    logits = np.asarray([[4.0, 3.0, 2.0]], dtype=np.float32)
    deltas = np.asarray([[0, 0, 0]], dtype=np.int16)
    seen = np.zeros_like(deltas, dtype=bool)

    ranked = reranked_shortlist_indices(
        logits,
        deltas,
        seen,
        top_n=3,
        alpha=0.0,
    )
    assert ranked.tolist() == [[0, 1, 2]]


def test_violation_metrics() -> None:
    violations = np.asarray(
        [[True, False], [False, False]],
        dtype=bool,
    )
    metrics = violation_metrics(violations, cutoffs=(1, 2))
    assert metrics["pvr@1"] == 0.5
    assert metrics["pvr@2"] == 0.25
