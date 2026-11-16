"""Size-stratified diagnostics for the cached relative-H1 signal (exp_022).

Question: once candidate size is held fixed, does signed Delta beta_1 still
rank the observed validation target above other candidates?

All AUCs here are tie-aware and exclude the target itself, so the chance level
is exactly 0.5:

    AUC_u = [#(s_c < s_t) + 0.5 * #(s_c == s_t)] / #competitors

where t is the user's validation target and c ranges over the user's other
unseen candidates in the comparison set. Higher signal = ranked higher, which
matches how exp_012/013 used MinMax(Delta beta_1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit


def tie_aware_auc(target_value: float, competitors: np.ndarray) -> float | None:
    """Probability the target outranks a random competitor (ties = 0.5)."""
    competitors = np.asarray(competitors)
    if competitors.size == 0:
        return None
    wins = np.count_nonzero(competitors < target_value)
    ties = np.count_nonzero(competitors == target_value)
    return float((wins + 0.5 * ties) / competitors.size)


def per_user_target_auc(
    signal: np.ndarray,
    seen_mask: np.ndarray,
    targets: np.ndarray,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Per-user target AUC; NaN where no competitor exists.

    If ``strata`` is given, the target is compared only with unseen candidates
    in the same stratum (same stratum key as the target).
    """
    if signal.shape != seen_mask.shape:
        raise ValueError("signal and seen_mask must align")
    if strata is not None and strata.shape != signal.shape:
        raise ValueError("strata must align with signal")

    n_users = signal.shape[0]
    targets = np.asarray(targets, dtype=np.int64)
    out = np.full(n_users, np.nan, dtype=np.float64)

    for row in range(n_users):
        target = int(targets[row])
        if seen_mask[row, target]:
            raise ValueError(f"validation target is marked seen for row {row}")
        mask = ~seen_mask[row]
        mask[target] = False
        if strata is not None:
            mask &= strata[row] == strata[row, target]
        competitors = signal[row, mask]
        value = tie_aware_auc(float(signal[row, target]), competitors)
        if value is not None:
            out[row] = value
    return out


def bin_sizes(values: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    """Map integer sizes to predeclared bins.

    ``edges`` are inclusive lower bounds, e.g. [0, 1, 3, 6, 11, 21] gives bins
    {0}, {1-2}, {3-5}, {6-10}, {11-20}, {21+}.
    """
    edges_arr = np.asarray(edges, dtype=np.int64)
    if edges_arr[0] != 0 or np.any(np.diff(edges_arr) <= 0):
        raise ValueError("bin edges must start at 0 and be strictly increasing")
    return np.searchsorted(edges_arr, values, side="right") - 1


def absorbed_edge_counts(
    splits: Sequence[LeaveTwoOutSplit],
    item_ids: Sequence[str],
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    graph: nx.DiGraph,
    seen_mask: np.ndarray,
) -> np.ndarray:
    """q[u, c]: prerequisite edges newly internal to history ∪ bundle.

    q counts directed edges whose endpoints both lie in H ∪ B but not both in
    H, where H is the user's validation-history concept set and B the
    candidate's concept bundle. Seen entries are left at 0.
    """
    concepts = list(graph.nodes)
    index = {concept: i for i, concept in enumerate(concepts)}
    src = np.asarray([index[u] for u, _ in graph.edges], dtype=np.int64)
    dst = np.asarray([index[v] for _, v in graph.edges], dtype=np.int64)

    bundles = np.zeros((len(item_ids), len(concepts)), dtype=bool)
    for col, item in enumerate(item_ids):
        for concept in course_to_concepts.get(item, ()):
            if concept in index:
                bundles[col, index[concept]] = True

    q = np.zeros(seen_mask.shape, dtype=np.int32)
    for row, split in enumerate(splits):
        history = np.zeros(len(concepts), dtype=bool)
        for course in split.validation_history:
            for concept in course_to_concepts.get(course, ()):
                if concept in index:
                    history[index[concept]] = True
        internal_before = int(np.count_nonzero(history[src] & history[dst]))
        union = bundles | history[None, :]
        internal_after = np.count_nonzero(union[:, src] & union[:, dst], axis=1)
        row_q = internal_after - internal_before
        row_q[seen_mask[row]] = 0
        q[row] = row_q
    return q


@dataclass(frozen=True)
class TopKEnrichment:
    users: int
    k: int
    median_size_ratio: float
    mean_size_ratio: float
    share_users_ratio_above_1: float
    mean_topk_size: float
    mean_pool_size: float


def topk_size_enrichment(
    signal: np.ndarray,
    size: np.ndarray,
    seen_mask: np.ndarray,
    k: int,
) -> TopKEnrichment:
    """Mean size of the top-k by signal relative to the user's unseen pool.

    Ties are broken by course index, matching the exp_012/013 hybrid rule.
    Users whose pool has zero mean size are skipped.
    """
    ratios: list[float] = []
    topk_sizes: list[float] = []
    pool_sizes: list[float] = []
    for row in range(signal.shape[0]):
        cols = np.flatnonzero(~seen_mask[row])
        if cols.size == 0:
            continue
        order = np.lexsort((cols, -signal[row, cols]))
        top = cols[order[: min(k, cols.size)]]
        pool_mean = float(size[row, cols].mean())
        if pool_mean <= 0:
            continue
        top_mean = float(size[row, top].mean())
        ratios.append(top_mean / pool_mean)
        topk_sizes.append(top_mean)
        pool_sizes.append(pool_mean)

    arr = np.asarray(ratios, dtype=np.float64)
    return TopKEnrichment(
        users=int(arr.size),
        k=int(k),
        median_size_ratio=float(np.median(arr)),
        mean_size_ratio=float(np.mean(arr)),
        share_users_ratio_above_1=float(np.mean(arr > 1.0)),
        mean_topk_size=float(np.mean(topk_sizes)),
        mean_pool_size=float(np.mean(pool_sizes)),
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """User-level percentile bootstrap for a mean (users are independent)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[draws].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "replicates": int(replicates),
        "seed": int(seed),
    }


def decide_outcome(
    stratified: dict[str, object],
    difference: dict[str, object],
    *,
    min_meaningful_auc: float,
) -> dict[str, str]:
    """Apply the predeclared exp_022 decision rule (docs/analysis_plan.md, D-074)."""
    low, high = stratified["ci95"]
    diff_low, _ = difference["ci95"]
    if low is None:
        return {"outcome": "UNDEFINED", "meaning": "no defined users"}
    if low >= min_meaningful_auc and diff_low is not None and diff_low > 0:
        return {
            "outcome": "A_SIZE_MASKS_SIGNAL",
            "meaning": (
                "Within equal-size candidates the topology score ranks the "
                "target clearly above chance, and better than when size is "
                "ignored: size was masking a real signal."
            ),
        }
    if low >= min_meaningful_auc:
        return {
            "outcome": "B_SIZE_INDEPENDENT_SIGNAL",
            "meaning": (
                "The topology score carries a size-independent signal, but "
                "removing size does not improve it: the recommender failure "
                "is not explained by size (more likely redundancy with "
                "SASRec)."
            ),
        }
    return {
        "outcome": "C_NO_MEANINGFUL_WITHIN_SIZE_SIGNAL",
        "meaning": (
            "Within equal-size candidates the topology score does not reach "
            "the predeclared minimum meaningful AUC: the score is weak "
            "regardless of size."
        ),
    }
