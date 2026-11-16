"""Run exp_012: validation-only SASRec + topology alpha ablation."""

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
    candidate_rank_normalize,
    candidate_topology_minmax,
    compare_ranks,
    evaluate_hybrid_grid,
    select_alpha,
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


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON/YAML object")
    return payload


def _metrics_close(
    observed: dict[str, float | int],
    expected: dict[str, float | int],
    *,
    tolerance: float = 1e-12,
) -> None:
    for metric, expected_value in expected.items():
        if metric in {"users", "missing_targets"}:
            if int(observed[metric]) != int(expected_value):
                raise RuntimeError(
                    f"baseline endpoint mismatch for {metric}"
                )
            continue
        difference = abs(
            float(observed[metric]) - float(expected_value)
        )
        if difference > tolerance:
            raise RuntimeError(
                f"alpha=1 endpoint changed SASRec metric {metric}: "
                f"{observed[metric]} vs {expected_value}"
            )


def _compact_result(result: dict[str, object]) -> str:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    return (
        f"alpha={float(result['alpha']):.2f}: "
        f"MRR={float(metrics['mrr']):.6f}, "
        f"R@10={float(metrics['recall@10']):.6f}, "
        f"NDCG@10={float(metrics['ndcg@10']):.6f}"
    )


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    topology_path = REPO_ROOT / str(config["topology_cache_path"])
    sasrec_path = REPO_ROOT / str(config["sasrec_cache_path"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    selected_path = output_dir / "selected_hybrid.json"

    for path in (metrics_path, summary_path, selected_path):
        if path.exists():
            raise FileExistsError(
                "exp_012 output already exists; refusing to overwrite "
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
        raise RuntimeError(
            "topology cache hash mismatch: "
            f"{actual_topology_sha} != {expected_topology_sha}"
        )
    if actual_sasrec_sha != expected_sasrec_sha:
        raise RuntimeError(
            "SASRec cache hash mismatch: "
            f"{actual_sasrec_sha} != {expected_sasrec_sha}"
        )

    topology = load_validation_topology_cache(topology_path)
    sasrec = load_sasrec_score_cache(sasrec_path)
    validate_hybrid_cache_alignment(topology, sasrec)

    linked_hashes = sasrec["topology_cache_sha256"]
    if (
        linked_hashes.size != 1
        or str(linked_hashes.reshape(-1)[0]) != actual_topology_sha
    ):
        raise RuntimeError(
            "SASRec score cache is not linked to the supplied topology cache"
        )

    deltas = topology["deltas"]
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen_mask = deltas == sentinel
    logits = sasrec["logits"]
    targets = topology["validation_target_indices"].astype(
        np.int64,
        copy=False,
    )

    rank_norm = candidate_rank_normalize(logits, seen_mask)
    signed_topology_norm = candidate_topology_minmax(
        deltas,
        seen_mask,
        clamp_negative=False,
    )
    clamped_topology_norm = candidate_topology_minmax(
        deltas,
        seen_mask,
        clamp_negative=True,
    )

    alphas = tuple(float(alpha) for alpha in config["alphas"])
    cutoffs = tuple(int(k) for k in config["cutoffs"])
    selection_metric = str(config["selection_metric"])

    signed_results = evaluate_hybrid_grid(
        rank_norm,
        signed_topology_norm,
        seen_mask,
        targets,
        alphas,
        cutoffs=cutoffs,
    )
    clamped_results = evaluate_hybrid_grid(
        rank_norm,
        clamped_topology_norm,
        seen_mask,
        targets,
        alphas,
        cutoffs=cutoffs,
    )

    baseline_result = next(
        result
        for result in signed_results
        if abs(float(result["alpha"]) - 1.0) < 1e-12
    )
    baseline_metrics = baseline_result["metrics"]
    assert isinstance(baseline_metrics, dict)

    expected_baseline = config["expected_sasrec_validation_metrics"]
    if not isinstance(expected_baseline, dict):
        raise ValueError(
            "expected_sasrec_validation_metrics must be an object"
        )
    _metrics_close(
        baseline_metrics,
        expected_baseline,
        tolerance=float(config["endpoint_tolerance"]),
    )

    signed_selected = select_alpha(
        signed_results,
        metric=selection_metric,
    )
    clamped_best = select_alpha(
        clamped_results,
        metric=selection_metric,
    )

    baseline_ranks = target_ranks_from_scores(
        rank_norm,
        seen_mask,
        targets,
    )
    selected_signed_scores = (
        float(signed_selected["alpha"]) * rank_norm.astype(np.float64)
        + (
            1.0 - float(signed_selected["alpha"])
        ) * signed_topology_norm.astype(np.float64)
    )
    selected_signed_ranks = target_ranks_from_scores(
        selected_signed_scores,
        seen_mask,
        targets,
    )
    rank_movement = compare_ranks(
        baseline_ranks,
        selected_signed_ranks,
    )

    signed_topology_only = next(
        result
        for result in signed_results
        if abs(float(result["alpha"])) < 1e-12
    )
    clamped_topology_only = next(
        result
        for result in clamped_results
        if abs(float(result["alpha"])) < 1e-12
    )

    candidate_mask = ~seen_mask
    normalization_difference_entries = int(
        np.sum(
            np.not_equal(
                signed_topology_norm[candidate_mask],
                clamped_topology_norm[candidate_mask],
            )
        )
    )
    user_difference_count = int(
        np.sum(
            np.any(
                np.where(
                    candidate_mask,
                    np.nan_to_num(
                        signed_topology_norm,
                        nan=-999.0,
                    )
                    != np.nan_to_num(
                        clamped_topology_norm,
                        nan=-999.0,
                    ),
                    False,
                ),
                axis=1,
            )
        )
    )

    output: dict[str, object] = {
        "experiment_id": "exp_012_hybrid_alpha_ablation",
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
            "sasrec_normalization": (
                "candidate_wise_ranknorm_average_ties"
            ),
            "topology_main": (
                "signed_delta_beta1_candidate_wise_minmax"
            ),
            "topology_ablation": (
                "clamp_negative_to_zero_then_candidate_wise_minmax"
            ),
            "hybrid_formula": (
                "alpha*ranknorm_sasrec + "
                "(1-alpha)*minmax_topology"
            ),
            "tie_break": "course_index_only",
            "alpha_grid": list(alphas),
            "selection_metric": selection_metric,
            "selection_tie_break": (
                "prefer_larger_alpha_on_exact_metric_tie"
            ),
        },
        "alignment": {
            "users": int(logits.shape[0]),
            "courses": int(logits.shape[1]),
            "candidate_entries": int(candidate_mask.sum()),
            "seen_entries": int(seen_mask.sum()),
            "cache_alignment_passed": True,
            "alpha_1_sasrec_reproduction_passed": True,
        },
        "signed": {
            "results": signed_results,
            "selected": signed_selected,
            "topology_only": signed_topology_only,
            "rank_movement_vs_sasrec": rank_movement,
        },
        "clamped_ablation": {
            "results": clamped_results,
            "best": clamped_best,
            "topology_only": clamped_topology_only,
            "normalization_difference_entries": (
                normalization_difference_entries
            ),
            "users_with_normalization_difference": (
                user_difference_count
            ),
        },
        "sasrec_baseline": baseline_result,
        "development_test_evaluated": False,
        "sasrec_retrained": False,
        "topology_recomputed": False,
    }

    selected_payload = {
        "source_experiment": "exp_012_hybrid_alpha_ablation",
        "main_topology": "signed_delta_beta1",
        "selection_metric": selection_metric,
        "selected_alpha": float(signed_selected["alpha"]),
        "selected_metrics": signed_selected["metrics"],
        "sasrec_baseline_metrics": baseline_metrics,
        "topology_cache_sha256": actual_topology_sha,
        "sasrec_cache_sha256": actual_sasrec_sha,
        "development_test_evaluated": False,
        "note": (
            "This freezes the validation-selected signed-topology alpha "
            "subject to post-run review; clamped topology is an ablation."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    signed_selected_metrics = signed_selected["metrics"]
    clamped_best_metrics = clamped_best["metrics"]
    signed_topo_metrics = signed_topology_only["metrics"]
    assert isinstance(signed_selected_metrics, dict)
    assert isinstance(clamped_best_metrics, dict)
    assert isinstance(signed_topo_metrics, dict)

    summary_lines = [
        "# exp_012 - Validation-Only SASRec + Topology Alpha Ablation",
        "",
        f"- Users: {logits.shape[0]:,}",
        f"- Courses: {logits.shape[1]:,}",
        f"- Candidate entries: {int(candidate_mask.sum()):,}",
        f"- Topology cache SHA-256: {actual_topology_sha}",
        f"- SASRec cache SHA-256: {actual_sasrec_sha}",
        "- SASRec retrained: NO",
        "- Topology recomputed: NO",
        "- Development-test evaluated: NO",
        "",
        "## Frozen normalization",
        "",
        "- SASRec: candidate-wise RankNorm with average ties",
        "- Main topology: signed Delta beta_1 candidate-wise MinMaxNorm",
        (
            "- Ablation topology: max(Delta beta_1, 0) before "
            "candidate-wise MinMaxNorm"
        ),
        "- Hybrid tie-break: course index only",
        (
            "- Alpha-selection exact-tie rule: prefer larger alpha "
            "(less reranking)"
        ),
        "",
        "## Signed topology alpha grid",
        "",
    ]
    for result in signed_results:
        summary_lines.append("- " + _compact_result(result))

    summary_lines.extend(
        [
            "",
            "## Main selected hybrid",
            "",
            f"- Selected alpha: {float(signed_selected['alpha']):.2f}",
            (
                f"- Validation MRR: "
                f"{float(signed_selected_metrics['mrr']):.6f}"
            ),
            (
                f"- Validation Recall@10: "
                f"{float(signed_selected_metrics['recall@10']):.6f}"
            ),
            (
                f"- Validation NDCG@10: "
                f"{float(signed_selected_metrics['ndcg@10']):.6f}"
            ),
            (
                f"- SASRec NDCG@10 endpoint: "
                f"{float(baseline_metrics['ndcg@10']):.6f}"
            ),
            (
                f"- Signed topology-only NDCG@10: "
                f"{float(signed_topo_metrics['ndcg@10']):.6f}"
            ),
            (
                "- Rank movement vs SASRec: "
                f"improved={rank_movement['improved']:,}, "
                f"worsened={rank_movement['worsened']:,}, "
                f"unchanged={rank_movement['unchanged']:,}"
            ),
            "",
            "## Clamped-topology mechanism ablation",
            "",
            f"- Best clamped alpha: {float(clamped_best['alpha']):.2f}",
            (
                f"- Best clamped validation NDCG@10: "
                f"{float(clamped_best_metrics['ndcg@10']):.6f}"
            ),
            (
                "- Users whose normalized topology vector changes after "
                f"clamping: {user_difference_count:,}/{logits.shape[0]:,}"
            ),
            "",
            "## Endpoint check",
            "",
            (
                f"- alpha=1.00 validation NDCG@10: "
                f"{float(baseline_metrics['ndcg@10']):.6f}"
            ),
            "- alpha=1.00 reproduces frozen SASRec ranking: PASS",
            "",
            (
                "The signed formulation is the main method. The clamped "
                "variant is reported only as a mechanism ablation."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_012_hybrid_alpha_ablation")
    print(
        "Input cache SHA checks: PASS"
    )
    print(
        "alpha=1 SASRec reproduction: PASS "
        f"(NDCG@10={float(baseline_metrics['ndcg@10']):.6f})"
    )
    print(
        "Signed selected: "
        f"alpha={float(signed_selected['alpha']):.2f}, "
        f"NDCG@10={float(signed_selected_metrics['ndcg@10']):.6f}"
    )
    print(
        "Signed topology-only: "
        f"NDCG@10={float(signed_topo_metrics['ndcg@10']):.6f}"
    )
    print(
        "Clamped best: "
        f"alpha={float(clamped_best['alpha']):.2f}, "
        f"NDCG@10={float(clamped_best_metrics['ndcg@10']):.6f}"
    )
    print(
        "Rank movement signed selected vs SASRec: "
        f"{rank_movement['improved']:,} improved / "
        f"{rank_movement['worsened']:,} worsened / "
        f"{rank_movement['unchanged']:,} unchanged"
    )
    print("Development-test evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
