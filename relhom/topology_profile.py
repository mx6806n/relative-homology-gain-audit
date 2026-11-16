"""Topology-signal profiling utilities for aligned course sequences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
import random
import time

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit
from relhom.topology import (
    GF2RelativeHomologyEvaluator,
    TopologicalEvaluator,
)


@dataclass(frozen=True)
class TopologyProfile:
    """Summary of candidate-level relative-homology signal."""

    sampled_users: int
    candidate_evaluations: int
    unique_history_concept_states: int
    directed_triangles: int
    raw_edges: int
    transitive_closure_edges: int
    fast_elapsed_seconds: float
    reference_checks: int
    reference_mismatches: int
    reference_elapsed_seconds: float
    cache_info: dict[str, int | float]
    beta_before_histogram: dict[str, int]
    delta_histogram: dict[str, int]
    target_delta_histogram: dict[str, int]
    positive_candidate_fraction: float
    zero_candidate_fraction: float
    negative_candidate_fraction: float
    users_with_discriminative_signal: int
    users_with_positive_candidate: int
    users_with_negative_candidate: int
    targets_with_positive_delta: int
    targets_with_zero_delta: int
    targets_with_negative_delta: int
    targets_at_user_max_delta: int
    targets_above_candidate_median: int
    mean_target_weak_percentile: float
    history_concept_percentiles: dict[str, float]
    unique_delta_count_percentiles: dict[str, float]


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    array = np.asarray(values, dtype=float)
    result = np.percentile(array, [25, 50, 75, 90, 95])
    return {
        "p25": float(result[0]),
        "p50": float(result[1]),
        "p75": float(result[2]),
        "p90": float(result[3]),
        "p95": float(result[4]),
    }


def _string_counter(counter: Counter[int]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter)
    }


def profile_topology_signal(
    graph: nx.DiGraph,
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Sequence[str],
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    *,
    sample_users: int = 250,
    seed: int = 1701,
    reference_checks: int = 25,
    cache_size: int = 131_072,
) -> TopologyProfile:
    """Profile raw-graph Delta beta_1 over all unseen validation candidates.

    User sampling is deterministic. The candidate universe is the complete
    aligned catalogue after removing the training history.

    A limited number of states are cross-checked against the original
    pyflagser mapping-cone implementation.
    """
    if not splits:
        raise ValueError("splits must not be empty")
    if sample_users < 1:
        raise ValueError("sample_users must be positive")
    if reference_checks < 0:
        raise ValueError("reference_checks must be non-negative")

    sample_count = min(sample_users, len(splits))
    rng = random.Random(seed)
    selected_indices = sorted(
        rng.sample(range(len(splits)), sample_count)
    )

    evaluator = GF2RelativeHomologyEvaluator(
        graph,
        cache_size=cache_size,
    )

    closure = nx.transitive_closure_dag(graph)

    beta_hist: Counter[int] = Counter()
    delta_hist: Counter[int] = Counter()
    target_delta_hist: Counter[int] = Counter()
    history_states: set[frozenset[str]] = set()
    history_concept_counts: list[int] = []
    unique_delta_counts: list[int] = []
    target_percentiles: list[float] = []

    candidate_evaluations = 0
    users_with_discriminative = 0
    users_with_positive = 0
    users_with_negative = 0
    target_positive = 0
    target_zero = 0
    target_negative = 0
    target_at_max = 0
    target_above_median = 0

    reference_states: list[frozenset[str]] = []
    reference_state_set: set[frozenset[str]] = set()

    started = time.perf_counter()

    for split_index in selected_indices:
        split = splits[split_index]
        history_courses = set(split.validation_history)

        history_concepts: set[str] = set()
        for course in history_courses:
            history_concepts.update(
                course_to_concepts.get(course, ())
            )
        history_state = frozenset(history_concepts)
        history_states.add(history_state)
        history_concept_counts.append(len(history_state))

        beta_before = evaluator.compute_relative_betti(history_state)
        beta_hist[beta_before] += 1

        if (
            len(reference_states) < reference_checks
            and history_state not in reference_state_set
        ):
            reference_states.append(history_state)
            reference_state_set.add(history_state)

        candidate_deltas: list[int] = []
        target_delta: int | None = None

        for course in item_universe:
            if course in history_courses:
                continue

            candidate_concepts = frozenset(
                course_to_concepts.get(course, ())
            )
            after_state = history_state.union(candidate_concepts)
            beta_after = evaluator.compute_relative_betti(after_state)
            delta = beta_before - beta_after

            candidate_evaluations += 1
            candidate_deltas.append(delta)
            delta_hist[delta] += 1

            if course == split.validation_target:
                target_delta = delta

            if (
                len(reference_states) < reference_checks
                and after_state not in reference_state_set
            ):
                reference_states.append(after_state)
                reference_state_set.add(after_state)

        if target_delta is None:
            raise ValueError(
                f"validation target missing from candidate universe: "
                f"{split.user_id}"
            )

        target_delta_hist[target_delta] += 1
        if target_delta > 0:
            target_positive += 1
        elif target_delta < 0:
            target_negative += 1
        else:
            target_zero += 1

        unique_deltas = set(candidate_deltas)
        unique_delta_counts.append(len(unique_deltas))
        if len(unique_deltas) > 1:
            users_with_discriminative += 1
        if max(candidate_deltas) > 0:
            users_with_positive += 1
        if min(candidate_deltas) < 0:
            users_with_negative += 1
        if target_delta == max(candidate_deltas):
            target_at_max += 1
        if target_delta > median(candidate_deltas):
            target_above_median += 1

        target_percentiles.append(
            sum(
                delta <= target_delta
                for delta in candidate_deltas
            )
            / len(candidate_deltas)
        )

    fast_elapsed = time.perf_counter() - started
    cache_info = evaluator.cache_info()

    reference_mismatches = 0
    reference_elapsed = 0.0
    if reference_states:
        reference = TopologicalEvaluator(graph, coeff=2)
        reference_started = time.perf_counter()
        for state in reference_states:
            fast_beta = evaluator.compute_relative_betti(state)
            reference_beta = reference.compute_relative_betti(state)
            if fast_beta != reference_beta:
                reference_mismatches += 1
        reference_elapsed = time.perf_counter() - reference_started

    total_candidates = sum(delta_hist.values())
    positive_candidates = sum(
        count for delta, count in delta_hist.items() if delta > 0
    )
    zero_candidates = delta_hist[0]
    negative_candidates = sum(
        count for delta, count in delta_hist.items() if delta < 0
    )

    return TopologyProfile(
        sampled_users=sample_count,
        candidate_evaluations=candidate_evaluations,
        unique_history_concept_states=len(history_states),
        directed_triangles=evaluator.directed_triangle_count,
        raw_edges=graph.number_of_edges(),
        transitive_closure_edges=closure.number_of_edges(),
        fast_elapsed_seconds=fast_elapsed,
        reference_checks=len(reference_states),
        reference_mismatches=reference_mismatches,
        reference_elapsed_seconds=reference_elapsed,
        cache_info=cache_info,
        beta_before_histogram=_string_counter(beta_hist),
        delta_histogram=_string_counter(delta_hist),
        target_delta_histogram=_string_counter(target_delta_hist),
        positive_candidate_fraction=(
            positive_candidates / total_candidates
            if total_candidates else 0.0
        ),
        zero_candidate_fraction=(
            zero_candidates / total_candidates
            if total_candidates else 0.0
        ),
        negative_candidate_fraction=(
            negative_candidates / total_candidates
            if total_candidates else 0.0
        ),
        users_with_discriminative_signal=users_with_discriminative,
        users_with_positive_candidate=users_with_positive,
        users_with_negative_candidate=users_with_negative,
        targets_with_positive_delta=target_positive,
        targets_with_zero_delta=target_zero,
        targets_with_negative_delta=target_negative,
        targets_at_user_max_delta=target_at_max,
        targets_above_candidate_median=target_above_median,
        mean_target_weak_percentile=float(
            np.mean(target_percentiles)
        ),
        history_concept_percentiles=_percentiles(
            history_concept_counts
        ),
        unique_delta_count_percentiles=_percentiles(
            unique_delta_counts
        ),
    )


def topology_profile_to_dict(
    profile: TopologyProfile,
) -> dict[str, object]:
    return asdict(profile)
