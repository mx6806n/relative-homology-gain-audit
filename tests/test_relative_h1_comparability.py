"""Exact synthetic tests for relative-H1 candidate comparability."""

from __future__ import annotations

import networkx as nx

from relhom.topology import GF2RelativeHomologyEvaluator, TopologicalEvaluator


def _clone_cap_graph(*, attachments: int, clones: int) -> tuple[nx.DiGraph, set[str], set[str]]:
    """Triangle-free graph with k candidate clones sharing d transcript neighbours."""
    graph = nx.DiGraph()
    transcript = {"W"}
    sources = [f"S{i}" for i in range(attachments)]
    clone_nodes = [f"C{j}" for j in range(clones)]

    graph.add_node("W")
    for source in sources:
        graph.add_edge("W", source)
        transcript.add(source)

    for clone in clone_nodes:
        graph.add_node(clone)
        for source in sources:
            graph.add_edge(source, clone)

    return graph, transcript, set(clone_nodes)


def test_clone_family_gain_is_k_times_d_minus_one() -> None:
    for attachments in range(1, 5):
        for clones in range(1, 6):
            graph, transcript, candidate = _clone_cap_graph(
                attachments=attachments,
                clones=clones,
            )
            evaluator = GF2RelativeHomologyEvaluator(graph)
            decomposition = evaluator.relative_h1_gain_decomposition(
                transcript,
                candidate,
            )

            assert evaluator.directed_triangle_count == 0
            assert decomposition.delta_beta0 == 0
            assert decomposition.delta_rank_d2 == 0
            assert decomposition.new_vertex_count == clones
            assert decomposition.absorbed_relative_edges == (
                attachments * clones
            )
            assert decomposition.delta_beta1 == (
                clones * (attachments - 1)
            )


def test_pendant_chain_extension_is_volume_invariant() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("C", "L1"),
            ("L1", "L2"),
            ("L2", "L3"),
            ("L3", "L4"),
            ("L4", "L5"),
        ]
    )
    transcript = {"W", "X", "Y"}
    evaluator = GF2RelativeHomologyEvaluator(graph)

    expected_gain = 1
    for length in range(0, 6):
        candidate = {"C"}
        candidate.update(f"L{i}" for i in range(1, length + 1))
        decomposition = evaluator.relative_h1_gain_decomposition(
            transcript,
            candidate,
        )

        assert decomposition.delta_beta0 == 0
        assert decomposition.delta_rank_d2 == 0
        assert decomposition.absorbed_relative_edges == (
            decomposition.new_vertex_count + expected_gain
        )
        assert decomposition.delta_beta1 == expected_gain


def test_duplicate_cap_inflates_score_while_core_gap_closure_is_retained() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("X", "R"),
            ("Y", "R"),
        ]
    )
    transcript = {"W", "X", "Y"}
    evaluator = GF2RelativeHomologyEvaluator(graph)

    core = evaluator.relative_h1_gain_decomposition(
        transcript,
        {"C"},
    )
    duplicated = evaluator.relative_h1_gain_decomposition(
        transcript,
        {"C", "R"},
    )

    assert core.delta_beta1 == 1
    assert duplicated.delta_beta1 == 2
    assert duplicated.new_vertex_count == core.new_vertex_count + 1
    assert (
        duplicated.absorbed_relative_edges
        == core.absorbed_relative_edges + 2
    )


def test_filled_diamond_d2_term_cancels_one_dimensional_excess() -> None:
    graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    transcript = {"W", "X", "Y"}
    evaluator = GF2RelativeHomologyEvaluator(graph)
    decomposition = evaluator.relative_h1_gain_decomposition(
        transcript,
        {"C"},
    )

    assert evaluator.directed_triangle_count == 2
    assert decomposition.absorbed_relative_edges == 2
    assert decomposition.new_vertex_count == 1
    assert decomposition.delta_beta0 == 0
    assert decomposition.delta_rank_d2 == 1
    assert decomposition.delta_beta1 == 0


def test_four_cycle_motifs_share_one_relative_h1_reduction() -> None:
    motifs = {
        "converging prerequisite diamond": (
            [("W", "X"), ("W", "Y"), ("X", "C"), ("Y", "C")],
            {"W", "X", "Y"},
            "C",
        ),
        "chain with transitive bypass": (
            [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")],
            {"A", "B", "C"},
            "D",
        ),
        "alternating-orientation 4-cycle": (
            [("A", "B"), ("C", "B"), ("C", "D"), ("A", "D")],
            {"A", "B", "C"},
            "D",
        ),
    }
    for name, (edges, transcript, candidate) in motifs.items():
        graph = nx.DiGraph(edges)
        evaluator = GF2RelativeHomologyEvaluator(graph)
        decomposition = evaluator.relative_h1_gain_decomposition(
            transcript,
            candidate,
        )

        assert evaluator.directed_triangle_count == 0, name
        assert decomposition.delta_beta1 == 1, name
        assert decomposition.absorbed_relative_edges == 2, name
        assert decomposition.new_vertex_count == 1, name
        assert decomposition.delta_beta0 == 0, name
        assert decomposition.delta_rank_d2 == 0, name
        assert TopologicalEvaluator(graph).topological_gain(
            transcript,
            candidate,
        ) == 1, name
