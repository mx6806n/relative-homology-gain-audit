"""Tests for seed-level metric aggregation."""

from __future__ import annotations

import pytest

from relhom.result_stats import summarize_seed_metrics


def test_summarize_seed_metrics_reports_mean_sd_and_ci() -> None:
    summary = summarize_seed_metrics(
        [
            {"ndcg@10": 0.30},
            {"ndcg@10": 0.40},
            {"ndcg@10": 0.50},
        ],
        ["ndcg@10"],
    )["ndcg@10"]

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["std"] == pytest.approx(0.1)
    assert summary["ci95_low"] < 0.4
    assert summary["ci95_high"] > 0.4
    assert summary["min"] == pytest.approx(0.3)
    assert summary["max"] == pytest.approx(0.5)


def test_summarize_seed_metrics_requires_multiple_runs() -> None:
    with pytest.raises(ValueError, match="at least two seed runs"):
        summarize_seed_metrics(
            [{"ndcg@10": 0.4}],
            ["ndcg@10"],
        )
