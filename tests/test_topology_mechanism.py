"""Tests for topology mechanism diagnostics."""

from __future__ import annotations

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit
from relhom.topology_mechanism import (
    candidate_structural_features,
    correlation_summary,
    target_vs_candidates,
)


def test_structural_features_use_history_concept_coverage() -> None:
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    splits = [
        LeaveTwoOutSplit(
            user_id="U",
            train=("CA", "CB"),
            validation_target="CC",
            test_target="CD",
        )
    ]
    items = ("CA", "CB", "CC", "CD")
    mapping = {
        "CA": {"A"},
        "CB": {"B"},
        "CC": {"C"},
        "CD": {"A", "C"},
    }
    seen = np.asarray([[True, True, False, False]])

    features = candidate_structural_features(
        splits,
        items,
        mapping,
        graph,
        seen,
    )

    assert features["history_concept_count"].tolist() == [2]
    assert features["new_concept_count"][0, 2] == 1
    assert features["new_concept_count"][0, 3] == 1
    assert features["unmet_direct_count"][0, 2] == 0
    assert features["unmet_ancestor_count"][0, 2] == 0


def test_correlation_and_target_percentile() -> None:
    signal = np.asarray([[0, 1, 2]], dtype=np.int16)
    feature = np.asarray([[0, 2, 4]], dtype=np.int16)
    seen = np.zeros_like(signal, dtype=bool)
    targets = np.asarray([1], dtype=np.int16)

    summary = correlation_summary(signal, feature, seen)
    target = target_vs_candidates(
        signal,
        feature,
        seen,
        targets,
    )

    assert summary.global_spearman == 1.0
    assert summary.per_user_median == 1.0
    assert target["mean_target_signal_weak_percentile"] == 2.0 / 3.0
