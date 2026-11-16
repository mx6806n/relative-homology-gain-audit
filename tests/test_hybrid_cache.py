"""Tests for cache-only hybrid reranking."""

from __future__ import annotations

import numpy as np
import pytest

from relhom.hybrid_cache import (
    candidate_rank_normalize,
    candidate_topology_minmax,
    evaluate_hybrid_grid,
    select_alpha,
    target_ranks_from_scores,
)


def test_candidate_rank_normalize_ignores_seen_and_averages_ties() -> None:
    logits = np.asarray([[9.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    seen = np.asarray([[True, False, False, False]])

    normalized = candidate_rank_normalize(logits, seen)

    assert np.isnan(normalized[0, 0])
    assert normalized[0, 1] == pytest.approx(0.0)
    assert normalized[0, 2] == pytest.approx(0.75)
    assert normalized[0, 3] == pytest.approx(0.75)


def test_topology_minmax_signed_and_clamped_are_distinct() -> None:
    deltas = np.asarray([[-2, -1, 0, 2]], dtype=np.int16)
    seen = np.zeros_like(deltas, dtype=bool)

    signed = candidate_topology_minmax(
        deltas,
        seen,
        clamp_negative=False,
    )
    clamped = candidate_topology_minmax(
        deltas,
        seen,
        clamp_negative=True,
    )

    assert signed[0].tolist() == pytest.approx(
        [0.0, 0.25, 0.5, 1.0]
    )
    assert clamped[0].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 1.0]
    )


def test_alpha_one_preserves_stage_one_order() -> None:
    logits = np.asarray(
        [[100.0, 4.0, 3.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    seen = np.asarray(
        [[True, False, False, False, False]]
    )
    deltas = np.asarray(
        [[-32768, -2, 5, 0, 1]],
        dtype=np.int16,
    )
    target = np.asarray([1], dtype=np.int16)

    rank_norm = candidate_rank_normalize(logits, seen)
    topo_norm = candidate_topology_minmax(deltas, seen)
    baseline_ranks = target_ranks_from_scores(
        logits,
        seen,
        target,
    )
    results = evaluate_hybrid_grid(
        rank_norm,
        topo_norm,
        seen,
        target,
        (0.0, 1.0),
        cutoffs=(1, 3),
    )

    assert baseline_ranks.tolist() == [1]
    assert results[1]["metrics"]["recall@1"] == 1.0


def test_pure_topology_ties_use_item_index_not_sasrec() -> None:
    logits = np.asarray([[1.0, 100.0, 50.0]], dtype=np.float32)
    seen = np.zeros_like(logits, dtype=bool)
    deltas = np.asarray([[0, 0, 0]], dtype=np.int16)
    target = np.asarray([0], dtype=np.int16)

    rank_norm = candidate_rank_normalize(logits, seen)
    topo_norm = candidate_topology_minmax(deltas, seen)
    results = evaluate_hybrid_grid(
        rank_norm,
        topo_norm,
        seen,
        target,
        (0.0,),
        cutoffs=(1,),
    )

    assert results[0]["metrics"]["recall@1"] == 1.0


def test_select_alpha_prefers_more_sasrec_on_exact_metric_tie() -> None:
    results = [
        {"alpha": 0.4, "metrics": {"ndcg@10": 0.5}},
        {"alpha": 0.8, "metrics": {"ndcg@10": 0.5}},
    ]
    selected = select_alpha(results, metric="ndcg@10")
    assert selected["alpha"] == 0.8
