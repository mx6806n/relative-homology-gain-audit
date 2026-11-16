"""Tests for the exp_023 no-topology null models."""

from __future__ import annotations

import numpy as np
import pytest

from relhom.null_models import (
    classify_excess,
    cluster_bootstrap_rho_difference,
    fit_vertex_probability,
    independent_vertex_null,
    per_row_spearman,
    safe_spearman,
    uniform_bound_null,
)


def test_uniform_bound_null_respects_bounds_and_covers_range() -> None:
    rng = np.random.default_rng(0)
    m = np.full(20_000, 3)
    q = np.full(20_000, 2)
    draws = uniform_bound_null(m, q, rng)
    assert draws.min() == -3 and draws.max() == 2
    assert set(np.unique(draws)) == {-3, -2, -1, 0, 1, 2}
    m2 = rng.integers(0, 10, 5000)
    q2 = rng.integers(0, 10, 5000)
    d2 = uniform_bound_null(m2, q2, rng)
    assert np.all(d2 >= -m2) and np.all(d2 <= q2)


def test_uniform_bound_null_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        uniform_bound_null(np.array([-1]), np.array([0]), np.random.default_rng(0))


def test_fit_vertex_probability_matches_zero_fraction() -> None:
    m = np.array([1, 2, 3, 2, 1, 4])
    p = fit_vertex_probability(m, 0.4)
    assert np.mean((1 - p) ** m) == pytest.approx(0.4, abs=1e-9)
    draws = independent_vertex_null(m, p, np.random.default_rng(1))
    assert np.all(draws <= 0) and np.all(draws >= -m)


def test_per_row_spearman_and_safe_spearman() -> None:
    signal = np.array([[1, 2, 3, 9], [5, 5, 5, 5]])
    feature = np.array([[1, 2, 3, 0], [1, 2, 3, 4]])
    mask = np.array([[True, True, True, False], [True, True, True, True]])
    out = per_row_spearman(signal, feature, mask)
    assert out[0] == pytest.approx(1.0)
    assert np.isnan(out[1])
    assert safe_spearman(np.array([1]), np.array([2])) is None


def test_cluster_bootstrap_detects_no_difference_for_identical_inputs() -> None:
    rng = np.random.default_rng(2)
    feature = rng.integers(1, 8, 3000)
    observed = feature + rng.integers(0, 3, 3000)
    clusters = np.repeat(np.arange(1000), 3)
    out = cluster_bootstrap_rho_difference(
        observed, observed.copy(), feature, clusters, replicates=200, seed=3
    )
    assert out["difference"] == pytest.approx(0.0)
    assert out["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert out["clusters"] == 1000


def test_cluster_bootstrap_detects_real_excess() -> None:
    rng = np.random.default_rng(4)
    feature = rng.integers(1, 8, 4000)
    observed = feature * 2 + rng.integers(0, 2, 4000)
    null = rng.integers(0, 15, 4000)
    clusters = np.repeat(np.arange(2000), 2)
    out = cluster_bootstrap_rho_difference(
        observed, null, feature, clusters, replicates=200, seed=5
    )
    assert out["ci95"][0] > 0.5


@pytest.mark.parametrize(
    ("ci", "expected"),
    [
        ([0.06, 0.09], "BEYOND_BOUND"),
        ([0.05, 0.09], "BEYOND_BOUND"),
        ([0.03, 0.09], "BOUND_DOMINATED"),
        ([-0.02, 0.01], "BOUND_DOMINATED"),
        ([None, None], "UNDEFINED"),
    ],
)
def test_classify_excess(ci, expected) -> None:
    assert classify_excess(ci, 0.05) == expected
