"""No-topology null models for the size dependence of raw relative-H1 gain.

Used by exp_023. Each null keeps a candidate's size information but removes
topology, so that observed Spearman(|g|, m) can be compared with what the
theorem's bound (-m <= g <= q) or simple cell counting already produce.

Nulls
-----
N1  bound-uniform:      g* ~ Uniform{-m, ..., q}         (bound only)
N2  independent-vertex: g* = -Binomial(m, p), p fitted to the observed zero
                         fraction (only sensible where g <= 0 dominates, i.e.
                         DORIS)
N3  one-dimensional:    g* = q - m  (circuit-rank term; drops h and t)
N4  random bundle:      same-size random bundle in the same graph (DORIS;
                         computed in the run script with the real evaluator)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr


def uniform_bound_null(
    m: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw g* uniformly from the integers in [-m, q] for every entry."""
    m = np.asarray(m, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    if m.shape != q.shape:
        raise ValueError("m and q must align")
    if np.any(m < 0) or np.any(q < 0):
        raise ValueError("m and q must be non-negative")
    width = m + q + 1
    return rng.integers(0, width) - m


def fit_vertex_probability(m: np.ndarray, zero_fraction: float) -> float:
    """p such that mean((1 - p) ** m) equals the observed zero fraction."""
    m = np.asarray(m, dtype=np.float64)
    if not 0.0 < zero_fraction < 1.0:
        raise ValueError("zero_fraction must lie strictly between 0 and 1")
    if np.all(m == 0):
        raise ValueError("m must contain positive values")
    floor = float(np.mean(m == 0))
    if zero_fraction <= floor:
        raise ValueError("zero_fraction is below the share of m == 0 entries")
    return float(
        brentq(
            lambda p: float(np.mean((1.0 - p) ** m)) - zero_fraction,
            1e-9,
            1.0 - 1e-9,
        )
    )


def independent_vertex_null(
    m: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """g* = -Binomial(m, p): each new vertex independently costs 1 or 0."""
    return -rng.binomial(np.asarray(m, dtype=np.int64), p)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.size < 2:
        return None
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(spearmanr(left, right).statistic)
    return None if math.isnan(value) else value


def per_row_spearman(
    signal: np.ndarray,
    feature: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Spearman per row over entries where mask is True; NaN if undefined."""
    if not (signal.shape == feature.shape == mask.shape):
        raise ValueError("signal, feature and mask must align")
    out = np.full(signal.shape[0], np.nan, dtype=np.float64)
    for row in range(signal.shape[0]):
        keep = mask[row]
        value = safe_spearman(signal[row, keep], feature[row, keep])
        if value is not None:
            out[row] = value
    return out


def cluster_bootstrap_rho_difference(
    observed: np.ndarray,
    null: np.ndarray,
    feature: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Cluster bootstrap of Spearman(observed, feature) - Spearman(null, feature).

    Clusters (e.g. students) are resampled with replacement; the null draws
    are held fixed, so the interval reflects sampling of clusters.
    """
    observed = np.asarray(observed)
    null = np.asarray(null)
    feature = np.asarray(feature)
    clusters = np.asarray(clusters)
    if not (observed.shape == null.shape == feature.shape == clusters.shape):
        raise ValueError("inputs must align")

    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    boundaries = np.flatnonzero(sorted_clusters[1:] != sorted_clusters[:-1]) + 1
    groups = np.split(order, boundaries)

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        picks = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in picks])
        obs = safe_spearman(observed[index], feature[index])
        nul = safe_spearman(null[index], feature[index])
        if obs is not None and nul is not None:
            differences.append(obs - nul)

    point_obs = safe_spearman(observed, feature)
    point_null = safe_spearman(null, feature)
    values = np.asarray(differences, dtype=np.float64)
    return {
        "observed_rho": point_obs,
        "null_rho": point_null,
        "difference": (
            None if point_obs is None or point_null is None
            else point_obs - point_null
        ),
        "ci95": (
            [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if values.size else [None, None]
        ),
        "replicates_defined": int(values.size),
        "seed": int(seed),
        "clusters": int(len(groups)),
    }


def classify_excess(ci95: list[float | None], margin: float) -> str:
    """Predeclared exp_023 rule (docs/analysis_plan.md, D-076)."""
    low = ci95[0]
    if low is None:
        return "UNDEFINED"
    return "BEYOND_BOUND" if low >= margin else "BOUND_DOMINATED"
