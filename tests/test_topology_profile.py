"""Tests for topology-signal profiling."""

from __future__ import annotations

import networkx as nx

from relhom.experiment import LeaveTwoOutSplit
from relhom.topology_profile import profile_topology_signal


def test_profile_reports_candidate_signal_and_reference_match() -> None:
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

    profile = profile_topology_signal(
        graph,
        splits,
        universe,
        mapping,
        sample_users=1,
        seed=1,
        reference_checks=4,
        cache_size=32,
    )

    assert profile.sampled_users == 1
    assert profile.candidate_evaluations == 2
    assert profile.reference_mismatches == 0
    assert profile.delta_histogram["1"] == 1
    assert profile.delta_histogram["0"] == 1
    assert profile.target_delta_histogram["1"] == 1
    assert profile.targets_with_positive_delta == 1
    assert profile.users_with_discriminative_signal == 1
