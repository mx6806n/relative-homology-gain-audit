"""Cross-check the fast GF(2) evaluator against the pyflagser reference."""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from relhom.topology import (
    GF2RelativeHomologyEvaluator,
    TopologicalEvaluator,
)


def _all_subsets(nodes: list[str]):
    for size in range(len(nodes) + 1):
        for subset in combinations(nodes, size):
            yield subset


def _assert_all_subsets_match(graph: nx.DiGraph) -> None:
    reference = TopologicalEvaluator(graph, coeff=2)
    fast = GF2RelativeHomologyEvaluator(graph)
    nodes = list(graph.nodes)

    for subset in _all_subsets(nodes):
        assert fast.compute_relative_betti(subset) == (
            reference.compute_relative_betti(subset)
        )


def test_fast_evaluator_matches_hollow_diamond() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    _assert_all_subsets_match(graph)


def test_fast_evaluator_matches_filled_diamond() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("X", "Y"),
        ]
    )
    _assert_all_subsets_match(graph)


def test_fast_evaluator_matches_graph_with_two_triangles() -> None:
    graph = nx.DiGraph(
        [
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("C", "D"),
            ("B", "E"),
            ("D", "E"),
        ]
    )
    _assert_all_subsets_match(graph)


def test_fast_bundle_gain_and_cache() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    fast = GF2RelativeHomologyEvaluator(graph, cache_size=32)

    assert fast.topological_gain(
        {"W", "X", "Y"},
        {"C"},
    ) == 1

    _ = fast.compute_relative_betti({"W", "X", "Y"})
    info = fast.cache_info()
    assert info["hits"] >= 1
    assert info["currsize"] >= 2


def test_relative_h1_components_exactly_decompose_gain() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    fast = GF2RelativeHomologyEvaluator(graph, cache_size=32)

    before = fast.relative_h1_components({"W", "X", "Y"})
    after = fast.relative_h1_components({"W", "X", "Y", "C"})

    assert before.beta1 == (
        before.relative_edge_count - before.rank_d1 - before.rank_d2
    )
    assert after.beta1 == (
        after.relative_edge_count - after.rank_d1 - after.rank_d2
    )

    delta_edges = before.relative_edge_count - after.relative_edge_count
    delta_rank_d1 = before.rank_d1 - after.rank_d1
    delta_rank_d2 = before.rank_d2 - after.rank_d2
    decomposed = delta_edges - delta_rank_d1 - delta_rank_d2

    assert decomposed == fast.topological_gain(
        {"W", "X", "Y"},
        {"C"},
    )
    assert decomposed == 1
