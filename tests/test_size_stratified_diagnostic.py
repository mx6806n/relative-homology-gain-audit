"""Tests for the exp_022 size-stratified diagnostic."""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest

from relhom.experiment import LeaveTwoOutSplit
from relhom.size_stratified_diagnostic import (
    absorbed_edge_counts,
    bin_sizes,
    bootstrap_mean_ci,
    decide_outcome,
    per_user_target_auc,
    tie_aware_auc,
    topk_size_enrichment,
)
from relhom.topology import GF2RelativeHomologyEvaluator


def test_tie_aware_auc_counts_ties_as_half() -> None:
    assert tie_aware_auc(5.0, np.array([1.0, 5.0, 9.0])) == pytest.approx(0.5)
    assert tie_aware_auc(5.0, np.array([1.0, 2.0])) == 1.0
    assert tie_aware_auc(5.0, np.array([])) is None


def test_per_user_auc_excludes_target_and_respects_strata() -> None:
    signal = np.array([[3.0, 1.0, 5.0, 2.0]])
    seen = np.array([[False, False, False, True]])
    targets = np.array([0])
    assert per_user_target_auc(signal, seen, targets)[0] == pytest.approx(0.5)
    strata = np.array([[7, 7, 9, 7]])
    assert per_user_target_auc(signal, seen, targets, strata)[0] == 1.0
    lonely = np.array([[1, 2, 3, 1]])
    assert np.isnan(per_user_target_auc(signal, seen, targets, lonely)[0])


def test_per_user_auc_rejects_seen_target() -> None:
    with pytest.raises(ValueError):
        per_user_target_auc(
            np.zeros((1, 2)), np.array([[True, False]]), np.array([0])
        )


def test_random_signal_has_chance_auc() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(4000, 30))
    seen = np.zeros_like(signal, dtype=bool)
    targets = rng.integers(0, 30, size=4000)
    values = per_user_target_auc(signal, seen, targets)
    assert np.nanmean(values) == pytest.approx(0.5, abs=0.01)


def test_bin_sizes() -> None:
    edges = [0, 1, 3, 6, 11, 21]
    values = np.array([0, 1, 2, 3, 5, 6, 10, 11, 20, 21, 99])
    assert bin_sizes(values, edges).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_absorbed_edges_match_evaluator_and_bound_holds() -> None:
    """q from the diagnostic must equal the theory's q exactly."""
    rng = random.Random(7)
    concepts = [f"c{i}" for i in range(9)]
    graph = nx.DiGraph()
    graph.add_nodes_from(concepts)
    for i in range(9):
        for j in range(i + 1, 9):
            if rng.random() < 0.35:
                graph.add_edge(concepts[i], concepts[j])
    courses = [f"K{i}" for i in range(6)]
    mapping = {
        course: set(rng.sample(concepts, rng.randint(1, 4))) for course in courses
    }
    splits = [
        LeaveTwoOutSplit(
            user_id=f"U{u}",
            train=tuple(rng.sample(courses[:4], 2)),
            validation_target=courses[4],
            test_target=courses[5],
        )
        for u in range(5)
    ]
    seen = np.array(
        [[course in split.validation_history for course in courses] for split in splits]
    )
    q = absorbed_edge_counts(splits, courses, mapping, graph, seen)
    evaluator = GF2RelativeHomologyEvaluator(graph)
    for row, split in enumerate(splits):
        history: set[str] = set()
        for course in split.validation_history:
            history |= mapping[course]
        for col, course in enumerate(courses):
            if seen[row, col]:
                assert q[row, col] == 0
                continue
            dec = evaluator.relative_h1_gain_decomposition(history, mapping[course])
            assert q[row, col] == dec.absorbed_relative_edges
            assert -dec.new_vertex_count <= dec.delta_beta1 <= q[row, col]


def test_topk_enrichment_detects_size_dominated_ranking() -> None:
    size = np.tile(np.arange(1, 21), (50, 1))
    seen = np.zeros_like(size, dtype=bool)
    result = topk_size_enrichment(size.astype(float), size, seen, k=5)
    assert result.median_size_ratio > 1.5
    assert result.share_users_ratio_above_1 == 1.0


def test_bootstrap_ci_contains_mean() -> None:
    values = np.array([0.4, 0.6, 0.5, np.nan, 0.7])
    out = bootstrap_mean_ci(values, replicates=200, seed=1)
    assert out["n"] == 4
    low, high = out["ci95"]
    assert low <= out["mean"] <= high


@pytest.mark.parametrize(
    ("ci", "diff_low", "expected"),
    [
        ([0.60, 0.62], 0.02, "A_SIZE_MASKS_SIGNAL"),
        ([0.60, 0.62], -0.01, "B_SIZE_INDEPENDENT_SIGNAL"),
        ([0.53, 0.57], 0.05, "C_NO_MEANINGFUL_WITHIN_SIZE_SIGNAL"),
        ([0.45, 0.50], -0.10, "C_NO_MEANINGFUL_WITHIN_SIZE_SIGNAL"),
    ],
)
def test_decision_rule(ci, diff_low, expected) -> None:
    stratified = {"ci95": ci}
    difference = {"ci95": [diff_low, diff_low + 0.02]}
    out = decide_outcome(stratified, difference, min_meaningful_auc=0.55)
    assert out["outcome"] == expected
