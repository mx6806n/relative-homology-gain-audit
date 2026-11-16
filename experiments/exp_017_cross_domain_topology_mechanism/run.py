"""Run exp_017: cross-domain replication of topology-volume confounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.data_loader import (  # noqa: E402
    build_course_to_concept_map,
    iter_mooccube2x_user_transcripts,
    load_mooccube2x_prerequisite_graph,
    resolve_mooccube2x_root,
)
from relhom.experiment import LeaveTwoOutSplit  # noqa: E402
from relhom.topology import GF2RelativeHomologyEvaluator  # noqa: E402
from relhom.topology_mechanism import (  # noqa: E402
    candidate_structural_features,
    correlation_summary,
    linear_structural_r2,
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON/YAML object")
    return payload


def _build_domain_splits(
    root: Path,
    *,
    mapping: dict[str, set[str]],
    min_sequence_length: int,
) -> tuple[list[LeaveTwoOutSplit], tuple[str, ...]]:
    """Build deterministic aligned leave-two-out splits without export."""
    aligned_sequences: list[tuple[str, tuple[str, ...]]] = []
    used_courses: set[str] = set()

    for user_id, transcript in iter_mooccube2x_user_transcripts(
        root,
        min_courses=1,
    ):
        aligned = tuple(
            course
            for course in transcript
            if mapping.get(course)
        )
        if len(aligned) < min_sequence_length:
            continue

        aligned_sequences.append((user_id, aligned))
        used_courses.update(aligned)

    if not aligned_sequences:
        raise ValueError("no aligned sequences satisfy minimum length")

    universe = tuple(sorted(used_courses))
    universe_set = set(universe)
    splits: list[LeaveTwoOutSplit] = []
    for user_id, sequence in aligned_sequences:
        unknown = set(sequence).difference(universe_set)
        if unknown:
            raise RuntimeError("internal universe construction mismatch")
        splits.append(
            LeaveTwoOutSplit(
                user_id=user_id,
                train=sequence[:-2],
                validation_target=sequence[-2],
                test_target=sequence[-1],
            )
        )
    return splits, universe


def _score_domain(
    *,
    graph: nx.DiGraph,
    mapping: dict[str, set[str]],
    splits: list[LeaveTwoOutSplit],
    universe: tuple[str, ...],
    cache_size: int,
) -> dict[str, object]:
    evaluator = GF2RelativeHomologyEvaluator(
        graph,
        validate_dag=True,
        cache_size=cache_size,
    )

    users = len(splits)
    courses = len(universe)
    deltas = np.zeros((users, courses), dtype=np.int16)
    seen_mask = np.zeros((users, courses), dtype=bool)
    course_index = {
        course: index
        for index, course in enumerate(universe)
    }

    start = time.perf_counter()
    evaluations = 0

    for row, split in enumerate(splits):
        history_courses = split.validation_history
        history_concepts: set[str] = set()
        for course in history_courses:
            history_concepts.update(mapping.get(course, ()))

        for course in set(history_courses):
            index = course_index.get(course)
            if index is not None:
                seen_mask[row, index] = True

        for col, course in enumerate(universe):
            if seen_mask[row, col]:
                continue
            candidate_concepts = mapping.get(course, set())
            value = evaluator.topological_gain(
                history_concepts,
                candidate_concepts,
            )
            if not np.iinfo(np.int16).min <= value <= np.iinfo(np.int16).max:
                raise OverflowError("Delta beta_1 exceeds int16 range")
            deltas[row, col] = value
            evaluations += 1

    elapsed = time.perf_counter() - start

    features = candidate_structural_features(
        splits,
        universe,
        mapping,
        graph,
        seen_mask,
    )
    abs_delta = np.abs(deltas.astype(np.int32, copy=False))

    correlations = {}
    for name, feature in (
        ("bundle_size", features["bundle_size"]),
        ("new_concept_count", features["new_concept_count"]),
        ("unmet_direct_count", features["unmet_direct_count"]),
        ("unmet_ancestor_count", features["unmet_ancestor_count"]),
    ):
        correlations[f"delta_vs_{name}"] = (
            correlation_summary(
                deltas,
                feature,
                seen_mask,
            ).__dict__
        )
        correlations[f"abs_delta_vs_{name}"] = (
            correlation_summary(
                abs_delta,
                feature,
                seen_mask,
            ).__dict__
        )

    structural_r2 = linear_structural_r2(
        deltas,
        (
            features["bundle_size"],
            features["new_concept_count"],
            features["unmet_direct_count"],
            features["unmet_ancestor_count"],
        ),
        seen_mask,
    )

    candidate_mask = ~seen_mask
    return {
        "users": users,
        "courses": courses,
        "candidate_states": int(candidate_mask.sum()),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "directed_triangles": evaluator.directed_triangle_count,
        "delta_range": [
            int(deltas[candidate_mask].min()),
            int(deltas[candidate_mask].max()),
        ],
        "positive_fraction": float(
            np.mean(deltas[candidate_mask] > 0)
        ),
        "zero_fraction": float(
            np.mean(deltas[candidate_mask] == 0)
        ),
        "negative_fraction": float(
            np.mean(deltas[candidate_mask] < 0)
        ),
        "correlations": correlations,
        "linear_structural_count_r2": structural_r2,
        "elapsed_topology_scoring_seconds": elapsed,
        "evaluator_cache": evaluator.cache_info(),
        "target_performance_inspected": False,
    }


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    root = resolve_mooccube2x_root(None)
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_017 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    domains = tuple(str(x) for x in config["domains"])
    min_sequence_length = int(config["min_sequence_length"])
    cache_size = int(config["evaluator_cache_size"])

    results: dict[str, object] = {}
    for domain in domains:
        graph, prereq_stats = load_mooccube2x_prerequisite_graph(
            root,
            domain=domain,
            validate_dag=True,
        )
        mapping = build_course_to_concept_map(
            root,
            domain,
            allowed_concepts=graph.nodes,
        )
        splits, universe = _build_domain_splits(
            root,
            mapping=mapping,
            min_sequence_length=min_sequence_length,
        )
        domain_result = _score_domain(
            graph=graph,
            mapping=mapping,
            splits=splits,
            universe=universe,
            cache_size=cache_size,
        )
        domain_result["prerequisite_positive_records"] = (
            prereq_stats.positive_records
        )
        results[domain] = domain_result

    cs_reference = config["cs_reference"]
    payload: dict[str, object] = {
        "experiment_id": "exp_017_cross_domain_topology_mechanism",
        "status": "completed",
        "phase": "mechanism_replication",
        "evidence_scope": "target_independent",
        "domains": results,
        "cs_reference": cs_reference,
        "primary_replication_quantity": (
            "Spearman(|Delta beta_1|, new_concept_count)"
        ),
        "interpretation_rule": (
            "No recommendation or target performance is evaluated. "
            "Consistency is assessed descriptively across domains."
        ),
        "selection_performed": False,
        "recommendation_metrics_evaluated": False,
        "topology_target_signal_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# exp_017 - Cross-Domain Topology Mechanism Replication",
        "",
        "- Evidence scope: target-independent mechanism replication",
        "- Selection performed: NO",
        "- Recommendation metrics evaluated: NO",
        "- Topology target-signal performance evaluated: NO",
        "- Development-test evaluated: NO",
        "",
        "## CS reference from exp_015",
        "",
        (
            "- |Delta| vs new-concept Spearman: "
            f"{float(cs_reference['abs_delta_vs_new_concept_spearman']):.6f}"
        ),
        (
            "- |Delta| vs bundle-size Spearman: "
            f"{float(cs_reference['abs_delta_vs_bundle_size_spearman']):.6f}"
        ),
        (
            "- Structural-count OLS R^2: "
            f"{float(cs_reference['structural_count_r2']):.6f}"
        ),
        "",
        "## Untouched-domain mechanism diagnostics",
        "",
    ]

    for domain in domains:
        result = results[domain]
        corr = result["correlations"]
        new_corr = corr["abs_delta_vs_new_concept_count"]
        bundle_corr = corr["abs_delta_vs_bundle_size"]
        lines.extend(
            [
                f"### {domain.upper()}",
                "",
                (
                    f"- Users/courses/candidate states: "
                    f"{int(result['users']):,}/"
                    f"{int(result['courses']):,}/"
                    f"{int(result['candidate_states']):,}"
                ),
                (
                    f"- Graph nodes/edges/triangles: "
                    f"{int(result['graph_nodes']):,}/"
                    f"{int(result['graph_edges']):,}/"
                    f"{int(result['directed_triangles']):,}"
                ),
                (
                    "- |Delta| vs new-concept Spearman: "
                    f"global={new_corr['global_spearman']}, "
                    f"per-user median={new_corr['per_user_median']}"
                ),
                (
                    "- |Delta| vs bundle-size Spearman: "
                    f"global={bundle_corr['global_spearman']}, "
                    f"per-user median={bundle_corr['per_user_median']}"
                ),
                (
                    "- Structural-count OLS R^2: "
                    f"{float(result['linear_structural_count_r2']):.6f}"
                ),
                (
                    "- Delta sign fractions positive/zero/negative: "
                    f"{float(result['positive_fraction']):.4f}/"
                    f"{float(result['zero_fraction']):.4f}/"
                    f"{float(result['negative_fraction']):.4f}"
                ),
                (
                    "- Topology scoring seconds: "
                    f"{float(result['elapsed_topology_scoring_seconds']):.3f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "This experiment tests only whether the structural-volume "
                "relationship diagnosed in CS repeats in Math/Psychology. "
                "It does not evaluate which candidates are true next items "
                "and cannot be used to choose a reranking configuration."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("exp_017_cross_domain_topology_mechanism")
    for domain in domains:
        result = results[domain]
        corr = result["correlations"]["abs_delta_vs_new_concept_count"]
        print(
            f"{domain.upper()}: users={int(result['users']):,}, "
            f"courses={int(result['courses']):,}, "
            f"|Delta|-new global={corr['global_spearman']}, "
            f"per-user-median={corr['per_user_median']}, "
            f"R2={float(result['linear_structural_count_r2']):.6f}"
        )
    print("Recommendation metrics evaluated: False")
    print("Topology target-signal performance evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
