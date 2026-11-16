"""Tests for top-N two-stage topology reranking."""

from __future__ import annotations

import numpy as np

from relhom.topn_hybrid import (
    select_topn_alpha,
    stage_one_topn_indices,
    topn_rerank_grid,
)


def test_stage_one_topn_uses_logit_order_and_masks_seen() -> None:
    logits = np.asarray([[100.0, 4.0, 3.0, 2.0, 1.0]])
    seen = np.asarray([[True, False, False, False, False]])

    indices = stage_one_topn_indices(
        logits,
        seen,
        top_n=2,
    )

    assert indices.tolist() == [[1, 2]]


def test_target_outside_shortlist_keeps_sasrec_rank() -> None:
    logits = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0]])
    seen = np.zeros_like(logits, dtype=bool)
    deltas = np.asarray([[0, 100, -10, 50, 20]], dtype=np.int16)
    target = np.asarray([4], dtype=np.int16)

    output = topn_rerank_grid(
        logits,
        deltas,
        seen,
        target,
        top_n=2,
        alphas=(0.0, 1.0),
        cutoffs=(5,),
    )

    for result in output["results"]:
        assert result["metrics"]["mrr"] == 0.2


def test_alpha_one_reproduces_stage_one_within_shortlist() -> None:
    logits = np.asarray([[5.0, 4.0, 3.0, 2.0]])
    seen = np.zeros_like(logits, dtype=bool)
    deltas = np.asarray([[0, 10, -5, 7]], dtype=np.int16)
    target = np.asarray([1], dtype=np.int16)

    output = topn_rerank_grid(
        logits,
        deltas,
        seen,
        target,
        top_n=3,
        alphas=(0.0, 1.0),
        cutoffs=(1, 3),
    )
    alpha_one = output["results"][1]

    assert alpha_one["metrics"]["mrr"] == 0.5


def test_topology_can_improve_target_inside_shortlist() -> None:
    logits = np.asarray([[5.0, 4.0, 3.0, 2.0]])
    seen = np.zeros_like(logits, dtype=bool)
    deltas = np.asarray([[0, 1, 100, -1]], dtype=np.int16)
    target = np.asarray([2], dtype=np.int16)

    output = topn_rerank_grid(
        logits,
        deltas,
        seen,
        target,
        top_n=3,
        alphas=(0.0, 1.0),
        cutoffs=(1, 3),
    )

    topology_only = output["results"][0]
    sasrec = output["results"][1]
    assert topology_only["metrics"]["mrr"] == 1.0
    assert sasrec["metrics"]["mrr"] == 1.0 / 3.0


def test_selection_ties_prefer_larger_alpha_then_smaller_n() -> None:
    results = [
        {"top_n": 50, "alpha": 0.8, "metrics": {"ndcg@10": 0.5}},
        {"top_n": 20, "alpha": 0.8, "metrics": {"ndcg@10": 0.5}},
        {"top_n": 10, "alpha": 0.7, "metrics": {"ndcg@10": 0.5}},
    ]

    selected = select_topn_alpha(results, metric="ndcg@10")
    assert selected["top_n"] == 20
    assert selected["alpha"] == 0.8
