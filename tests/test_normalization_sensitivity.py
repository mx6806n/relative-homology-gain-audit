"""Tests for exp_024 helpers and the rank-normalization option of top-N reranking."""

from __future__ import annotations

import numpy as np
import pytest

from relhom.normalization_sensitivity import (
    classify_configuration,
    overall_classification,
    paired_bootstrap_difference,
    per_user_ndcg_at_k,
    select_best,
)
from relhom.topn_hybrid import topn_rerank_grid


def test_per_user_ndcg() -> None:
    out = per_user_ndcg_at_k(np.array([1, 3, 11]), 10)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == 0.0


def test_paired_bootstrap_zero_for_identical() -> None:
    x = np.random.default_rng(0).random(500)
    out = paired_bootstrap_difference(x, x.copy(), replicates=100, seed=1)
    assert out["mean_difference"] == 0.0
    assert out["ci95"] == [0.0, 0.0]


def test_select_best_prefers_larger_alpha_on_ties() -> None:
    results = [
        {"alpha": 0.9, "metrics": {"ndcg@10": 0.4}},
        {"alpha": 1.0, "metrics": {"ndcg@10": 0.4}},
        {"alpha": 0.5, "metrics": {"ndcg@10": 0.3}},
    ]
    assert select_best(results)["alpha"] == 1.0


def test_classification_rules() -> None:
    assert classify_configuration(1.0, [0.0, 0.0]) == "NULL_SELECTS_SASREC"
    assert classify_configuration(0.95, [0.001, 0.003]) == "IMPROVES_OVER_SASREC"
    assert classify_configuration(0.95, [-0.001, 0.003]) == "NO_RELIABLE_IMPROVEMENT"
    assert overall_classification(["NULL_SELECTS_SASREC", "NO_RELIABLE_IMPROVEMENT"]) == "NULL_ROBUST"
    assert overall_classification(["IMPROVES_OVER_SASREC"]) == "NORMALIZATION_OR_GRID_SENSITIVE"


def _toy():
    rng = np.random.default_rng(3)
    users, items = 60, 15
    logits = rng.normal(size=(users, items))
    deltas = rng.integers(-5, 30, size=(users, items)).astype(np.int16)
    seen = np.zeros((users, items), dtype=bool)
    seen[:, 0] = True
    targets = rng.integers(1, items, size=users)
    return logits, deltas, seen, targets


def test_topn_default_is_unchanged_and_rank_option_works() -> None:
    logits, deltas, seen, targets = _toy()
    alphas = [0.0, 0.5, 1.0]
    default = topn_rerank_grid(logits, deltas, seen, targets, top_n=5, alphas=alphas)
    explicit = topn_rerank_grid(
        logits, deltas, seen, targets, top_n=5, alphas=alphas,
        topology_normalization="minmax",
    )
    for a, b in zip(default["results"], explicit["results"]):
        assert a["metrics"] == b["metrics"]
        assert "ranks" not in a
    ranked = topn_rerank_grid(
        logits, deltas, seen, targets, top_n=5, alphas=alphas,
        topology_normalization="rank", return_ranks=True,
    )
    assert ranked["results"][-1]["metrics"] == default["results"][-1]["metrics"]
    assert ranked["results"][0]["ranks"].shape == (60,)
    with pytest.raises(ValueError):
        topn_rerank_grid(logits, deltas, seen, targets, top_n=5, alphas=alphas,
                         topology_normalization="zscore")
