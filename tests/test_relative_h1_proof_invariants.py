"""Regression tests for the invariants of the comparability theorem."""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from relhom.topology import GF2RelativeHomologyEvaluator


def test_disconnected_component_beta0_exactly_compensates_new_vertex() -> None:
    graph = nx.DiGraph()
    graph.add_edge("W", "X")
    graph.add_node("Z")

    evaluator = GF2RelativeHomologyEvaluator(graph, validate_dag=True)
    decomposition = evaluator.relative_h1_gain_decomposition(
        {"W"},
        {"Z"},
    )

    assert decomposition.new_vertex_count == 1
    assert decomposition.absorbed_relative_edges == 0
    assert decomposition.delta_beta0 == 1
    assert decomposition.delta_rank_d2 == 0
    assert decomposition.delta_beta1 == 0


def test_general_bounds_hold_across_small_dag_candidate_subsets() -> None:
    graph = nx.DiGraph(
        [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
            ("E", "F"),
        ]
    )
    evaluator = GF2RelativeHomologyEvaluator(graph)

    transcript = {"A"}
    candidates = ["B", "C", "D", "E", "F"]

    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            decomposition = evaluator.relative_h1_gain_decomposition(
                transcript,
                set(subset),
            )
            assert 0 <= decomposition.delta_beta0 <= (
                decomposition.new_vertex_count
            )
            assert 0 <= decomposition.delta_rank_d2 <= (
                decomposition.absorbed_relative_edges
            )
            assert (
                -decomposition.new_vertex_count
                <= decomposition.delta_beta1
                <= decomposition.absorbed_relative_edges
            )


def test_connected_positive_gain_requires_more_absorbed_edges_than_vertices() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("C", "L"),
        ]
    )
    evaluator = GF2RelativeHomologyEvaluator(graph)
    transcript = {"W", "X", "Y"}

    for candidate in ({"C"}, {"C", "L"}):
        decomposition = evaluator.relative_h1_gain_decomposition(
            transcript,
            candidate,
        )
        assert decomposition.delta_beta0 == 0
        if decomposition.delta_beta1 > 0:
            assert (
                decomposition.absorbed_relative_edges
                > decomposition.new_vertex_count
                + decomposition.delta_rank_d2
            )


def test_component_interpretation_matches_unhit_weak_components() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
        ]
    )
    evaluator = GF2RelativeHomologyEvaluator(graph)

    before = evaluator.relative_h1_components({"A"})
    after = evaluator.relative_h1_components({"A", "C"})

    assert before.beta0 == 2
    assert after.beta0 == 1
    assert before.beta0 - after.beta0 == 1
