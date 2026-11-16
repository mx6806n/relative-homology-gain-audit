"""Tests for the full validation topology cache."""

from __future__ import annotations

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit
from relhom.topology_cache import (
    TOPOLOGY_SENTINEL,
    build_validation_topology_cache,
    load_validation_topology_cache,
    validate_topology_cache,
)


def test_validation_cache_scores_unseen_and_masks_history(tmp_path) -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    splits = [
        LeaveTwoOutSplit(
            user_id="U_1",
            train=("CW", "CX", "CY"),
            validation_target="TARGET",
            test_target="OTHER",
        )
    ]
    universe = ("CW", "CX", "CY", "TARGET", "OTHER")
    mapping = {
        "CW": {"W"},
        "CX": {"X"},
        "CY": {"Y"},
        "TARGET": {"C"},
        "OTHER": set(),
    }
    path = tmp_path / "cache.npz"

    stats = build_validation_topology_cache(
        graph,
        splits,
        universe,
        mapping,
        path,
        evaluator_cache_size=64,
    )
    cache = load_validation_topology_cache(path)
    validate_topology_cache(cache, splits, universe)

    assert stats.users == 1
    assert stats.courses == 5
    assert stats.candidate_evaluations == 2
    assert stats.seen_entries == 3
    assert cache["deltas"].dtype == np.int16
    assert cache["deltas"][0, 0] == TOPOLOGY_SENTINEL
    assert cache["deltas"][0, 3] == 1
    assert cache["deltas"][0, 4] == 0
    assert stats.target_positive == 1
    assert stats.users_with_discriminative_signal == 1
