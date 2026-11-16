"""Synthetic topology tests for the three toy models of the article."""

import networkx as nx

from relhom.topology import TopologicalEvaluator


def test_hollow_diamond_gap_is_closed_by_candidate() -> None:
    """C unifies two parallel W->X/Y tracks, killing one H_1 class."""
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    evaluator = TopologicalEvaluator(graph)

    transcript = ["W", "X", "Y"]
    beta_before = evaluator.compute_relative_betti(transcript)
    beta_after = evaluator.compute_relative_betti(
        [*transcript, "C"]
    )

    assert beta_before == 1
    assert beta_after == 0
    assert beta_before - beta_after > 0


def test_filled_diamond_has_no_remaining_gap() -> None:
    """X->Y fills the parallel-track cycle before C is added."""
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
            ("X", "Y"),
        ]
    )
    evaluator = TopologicalEvaluator(graph)

    transcript = ["W", "X", "Y"]
    beta_before = evaluator.compute_relative_betti(transcript)
    beta_after = evaluator.compute_relative_betti(
        [*transcript, "C"]
    )

    assert beta_before == 0
    assert beta_after == 0
    assert beta_before - beta_after == 0


def test_disconnected_elective_does_not_close_a_cycle() -> None:
    """D does not close a relative H_1 class."""
    graph = nx.DiGraph()
    graph.add_nodes_from(["X", "Y", "Z", "D"])
    graph.add_edges_from(
        [
            ("X", "D"),
            ("Z", "D"),
        ]
    )
    evaluator = TopologicalEvaluator(graph)

    transcript = ["X", "Y"]
    beta_before = evaluator.compute_relative_betti(transcript)
    beta_after = evaluator.compute_relative_betti(
        [*transcript, "D"]
    )

    assert beta_before == 0
    assert beta_after == 0
    assert beta_before - beta_after == 0
