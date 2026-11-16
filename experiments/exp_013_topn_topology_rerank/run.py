"""Run exp_013: intended top-N SASRec retrieval + topology reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.hybrid_cache import (  # noqa: E402
    metrics_from_ranks,
    target_ranks_from_scores,
    validate_hybrid_cache_alignment,
)
from relhom.sasrec_score_cache import (  # noqa: E402
    load_sasrec_score_cache,
    sha256_file as sasrec_sha256,
)
from relhom.topology_cache import (  # noqa: E402
    load_validation_topology_cache,
    sha256_file as topology_sha256,
)
from relhom.topn_hybrid import (  # noqa: E402
    select_topn_alpha,
    topn_rerank_grid,
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON/YAML object")
    return payload


def _assert_metrics_match(
    observed: dict[str, float | int],
    expected: dict[str, float | int],
    *,
    tolerance: float,
    context: str,
) -> None:
    for metric, expected_value in expected.items():
        if metric in {"users", "missing_targets"}:
            if int(observed[metric]) != int(expected_value):
                raise RuntimeError(
                    f"{context}: integer metric mismatch for {metric}"
                )
        else:
            difference = abs(
                float(observed[metric]) - float(expected_value)
            )
            if difference > tolerance:
                raise RuntimeError(
                    f"{context}: metric mismatch for {metric}: "
                    f"{observed[metric]} vs {expected_value}"
                )


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    topology_path = REPO_ROOT / str(config["topology_cache_path"])
    sasrec_path = REPO_ROOT / str(config["sasrec_cache_path"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    selected_path = output_dir / "selected_topn_hybrid.json"

    for path in (metrics_path, summary_path, selected_path):
        if path.exists():
            raise FileExistsError(
                "exp_013 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    expected_topology_sha = str(
        config["expected_topology_cache_sha256"]
    )
    expected_sasrec_sha = str(
        config["expected_sasrec_cache_sha256"]
    )
    actual_topology_sha = topology_sha256(topology_path)
    actual_sasrec_sha = sasrec_sha256(sasrec_path)
    if actual_topology_sha != expected_topology_sha:
        raise RuntimeError("topology cache SHA-256 mismatch")
    if actual_sasrec_sha != expected_sasrec_sha:
        raise RuntimeError("SASRec cache SHA-256 mismatch")

    topology = load_validation_topology_cache(topology_path)
    sasrec = load_sasrec_score_cache(sasrec_path)
    validate_hybrid_cache_alignment(topology, sasrec)

    linked_hashes = sasrec["topology_cache_sha256"]
    if (
        linked_hashes.size != 1
        or str(linked_hashes.reshape(-1)[0]) != actual_topology_sha
    ):
        raise RuntimeError("SASRec cache references wrong topology cache")

    deltas = topology["deltas"]
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen_mask = deltas == sentinel
    logits = sasrec["logits"]
    targets = topology["validation_target_indices"].astype(
        np.int64,
        copy=False,
    )

    cutoffs = tuple(int(k) for k in config["cutoffs"])
    alphas = tuple(float(a) for a in config["alphas"])
    top_ns = tuple(int(n) for n in config["top_ns"])
    selection_metric = str(config["selection_metric"])

    candidate_counts = (~seen_mask).sum(axis=1)
    minimum_unseen_candidates = int(candidate_counts.min())
    maximum_unseen_candidates = int(candidate_counts.max())
    infeasible_top_ns = [
        top_n for top_n in top_ns
        if top_n > minimum_unseen_candidates
    ]
    if infeasible_top_ns:
        raise ValueError(
            "configured fixed shortlist depths exceed the minimum unseen "
            f"candidate count={minimum_unseen_candidates}: "
            f"{infeasible_top_ns}"
        )

    baseline_ranks = target_ranks_from_scores(
        logits,
        seen_mask,
        targets,
    )
    baseline_metrics = metrics_from_ranks(
        baseline_ranks,
        cutoffs=cutoffs,
    )

    expected_baseline = config["expected_sasrec_validation_metrics"]
    if not isinstance(expected_baseline, dict):
        raise ValueError(
            "expected_sasrec_validation_metrics must be an object"
        )
    _assert_metrics_match(
        baseline_metrics,
        expected_baseline,
        tolerance=float(config["endpoint_tolerance"]),
        context="frozen SASRec baseline",
    )

    by_shortlist: list[dict[str, object]] = []
    flat_results: list[dict[str, object]] = []

    for top_n in top_ns:
        output = topn_rerank_grid(
            logits,
            deltas,
            seen_mask,
            targets,
            top_n=top_n,
            alphas=alphas,
            cutoffs=cutoffs,
        )
        by_shortlist.append(output)
        shortlist_results = output["results"]
        assert isinstance(shortlist_results, list)
        flat_results.extend(shortlist_results)

        alpha_one = next(
            result
            for result in shortlist_results
            if abs(float(result["alpha"]) - 1.0) < 1e-12
        )
        metrics = alpha_one["metrics"]
        assert isinstance(metrics, dict)
        _assert_metrics_match(
            metrics,
            expected_baseline,
            tolerance=float(config["endpoint_tolerance"]),
            context=f"top_n={top_n}, alpha=1",
        )

    selected = select_topn_alpha(
        flat_results,
        metric=selection_metric,
    )
    selected_metrics = selected["metrics"]
    assert isinstance(selected_metrics, dict)

    baseline_ndcg = float(baseline_metrics[selection_metric])
    selected_ndcg = float(selected_metrics[selection_metric])
    absolute_gain = selected_ndcg - baseline_ndcg
    relative_gain = (
        absolute_gain / baseline_ndcg
        if baseline_ndcg != 0.0 else 0.0
    )

    metrics_payload: dict[str, object] = {
        "experiment_id": "exp_013_topn_topology_rerank",
        "status": "completed",
        "phase": "hybrid_validation_tuning",
        "evidence_scope": "validation_only",
        "input_caches": {
            "topology": {
                "path": str(topology_path.relative_to(REPO_ROOT)),
                "sha256": actual_topology_sha,
            },
            "sasrec": {
                "path": str(sasrec_path.relative_to(REPO_ROOT)),
                "sha256": actual_sasrec_sha,
            },
        },
        "protocol": {
            "architecture": "frozen_sasrec_retrieval_then_topology_rerank",
            "top_ns": list(top_ns),
            "alphas": list(alphas),
            "stage1_normalization": (
                "candidate_wise_ranknorm_within_shortlist"
            ),
            "topology_normalization": (
                "signed_delta_beta1_minmax_within_shortlist"
            ),
            "outside_shortlist_policy": (
                "preserve_original_sasrec_order_exactly"
            ),
            "tie_break": "course_index_only",
            "selection_metric": selection_metric,
            "selection_tie_break": (
                "larger_alpha_then_smaller_top_n"
            ),
            "minimum_unseen_candidates": minimum_unseen_candidates,
            "maximum_unseen_candidates": maximum_unseen_candidates,
            "infeasible_original_top_n_50_removed_before_results": True,
        },
        "sasrec_baseline": baseline_metrics,
        "shortlists": by_shortlist,
        "selected": selected,
        "selected_gain_vs_sasrec": {
            "absolute_ndcg@10": absolute_gain,
            "relative_ndcg@10": relative_gain,
        },
        "checks": {
            "cache_hashes": True,
            "cache_alignment": True,
            "sasrec_baseline_reproduced": True,
            "alpha_1_reproduced_for_every_top_n": True,
        },
        "sasrec_retrained": False,
        "topology_recomputed": False,
        "development_test_evaluated": False,
    }

    selected_payload = {
        "source_experiment": "exp_013_topn_topology_rerank",
        "architecture": "sasrec_topn_then_signed_topology_rerank",
        "selection_metric": selection_metric,
        "selected_top_n": int(selected["top_n"]),
        "selected_alpha": float(selected["alpha"]),
        "selected_metrics": selected_metrics,
        "sasrec_baseline_metrics": baseline_metrics,
        "absolute_ndcg@10_gain": absolute_gain,
        "relative_ndcg@10_gain": relative_gain,
        "topology_cache_sha256": actual_topology_sha,
        "sasrec_cache_sha256": actual_sasrec_sha,
        "development_test_evaluated": False,
        "note": (
            "Validation selection only. Review required before multi-seed "
            "or development-test evaluation."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_013 - Top-N SASRec Retrieval + Topology Reranking",
        "",
        "- Evidence scope: validation only",
        f"- Users: {logits.shape[0]:,}",
        f"- Courses: {logits.shape[1]:,}",
        "- SASRec retrained: NO",
        "- Topology recomputed: NO",
        "- Development-test evaluated: NO",
        f"- Minimum unseen candidates/user: {minimum_unseen_candidates}",
        f"- Maximum unseen candidates/user: {maximum_unseen_candidates}",
        "- Original top-N=50 setting: infeasible and removed before results",
        "",
        "## Frozen Stage-1 baseline",
        "",
        f"- MRR: {float(baseline_metrics['mrr']):.6f}",
        f"- Recall@10: {float(baseline_metrics['recall@10']):.6f}",
        f"- NDCG@10: {float(baseline_metrics['ndcg@10']):.6f}",
        "",
        "## Shortlist results",
        "",
    ]

    for shortlist in by_shortlist:
        top_n = int(shortlist["top_n"])
        included = int(shortlist["target_in_shortlist"])
        fraction = float(shortlist["target_in_shortlist_fraction"])
        summary_lines.extend(
            [
                f"### Top-N = {top_n}",
                "",
                (
                    f"- Validation target in shortlist: "
                    f"{included:,}/{logits.shape[0]:,} "
                    f"({fraction:.4f})"
                ),
            ]
        )
        results = shortlist["results"]
        assert isinstance(results, list)
        for result in results:
            result_metrics = result["metrics"]
            movement = result["rank_movement_vs_sasrec"]
            assert isinstance(result_metrics, dict)
            assert isinstance(movement, dict)
            summary_lines.append(
                f"- alpha={float(result['alpha']):.2f}: "
                f"MRR={float(result_metrics['mrr']):.6f}, "
                f"R@10={float(result_metrics['recall@10']):.6f}, "
                f"NDCG@10={float(result_metrics['ndcg@10']):.6f}, "
                f"rank +/−/="
                f"{int(movement['improved']):,}/"
                f"{int(movement['worsened']):,}/"
                f"{int(movement['unchanged']):,}"
            )
        summary_lines.append("")

    selected_movement = selected["rank_movement_vs_sasrec"]
    assert isinstance(selected_movement, dict)
    summary_lines.extend(
        [
            "## Validation-selected top-N hybrid",
            "",
            f"- Selected top-N: {int(selected['top_n'])}",
            f"- Selected alpha: {float(selected['alpha']):.2f}",
            (
                f"- Selected MRR: "
                f"{float(selected_metrics['mrr']):.6f}"
            ),
            (
                f"- Selected Recall@10: "
                f"{float(selected_metrics['recall@10']):.6f}"
            ),
            (
                f"- Selected NDCG@10: "
                f"{selected_ndcg:.6f}"
            ),
            (
                f"- Absolute NDCG@10 gain vs SASRec: "
                f"{absolute_gain:+.6f}"
            ),
            (
                f"- Relative NDCG@10 gain vs SASRec: "
                f"{relative_gain:+.4%}"
            ),
            (
                "- Rank movement vs SASRec: "
                f"improved={int(selected_movement['improved']):,}, "
                f"worsened={int(selected_movement['worsened']):,}, "
                f"unchanged={int(selected_movement['unchanged']):,}"
            ),
            "",
            "## Integrity checks",
            "",
            "- Frozen cache SHA checks: PASS",
            "- Frozen SASRec baseline reproduction: PASS",
            "- alpha=1 reproduction for every shortlist: PASS",
            "",
            (
                "This experiment tests the intended two-stage mechanism. "
                "Only Stage-1 top-N candidates can be reordered by topology; "
                "all courses outside the shortlist keep their SASRec order."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_013_topn_topology_rerank")
    print(
        "Frozen SASRec NDCG@10: "
        f"{float(baseline_metrics['ndcg@10']):.6f}"
    )
    for shortlist in by_shortlist:
        results = shortlist["results"]
        assert isinstance(results, list)
        best = select_topn_alpha(
            results,
            metric=selection_metric,
        )
        best_metrics = best["metrics"]
        assert isinstance(best_metrics, dict)
        print(
            f"Top-N={int(shortlist['top_n'])}: "
            f"target_in={float(shortlist['target_in_shortlist_fraction']):.4f}, "
            f"best_alpha={float(best['alpha']):.2f}, "
            f"NDCG@10={float(best_metrics['ndcg@10']):.6f}"
        )
    print(
        "Selected: "
        f"top_n={int(selected['top_n'])}, "
        f"alpha={float(selected['alpha']):.2f}, "
        f"NDCG@10={selected_ndcg:.6f}, "
        f"gain={absolute_gain:+.6f}"
    )
    print("Development-test evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
