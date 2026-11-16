"""Run exp_018: exact relative-chain decomposition of Delta beta_1."""

from __future__ import annotations

import argparse
import csv
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
from relhom.experiment import (  # noqa: E402
    LeaveTwoOutSplit,
    load_course_universe,
    load_leave_two_out_splits,
)
from relhom.topology import GF2RelativeHomologyEvaluator  # noqa: E402
from relhom.topology_mechanism import correlation_summary  # noqa: E402


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON/YAML object")
    return payload


def _safe_ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    active = mask & (denominator > 0)
    if not np.any(active):
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    values = (
        numerator[active].astype(np.float64)
        / denominator[active].astype(np.float64)
    )
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _load_processed_cs(
    data_dir: Path,
) -> tuple[
    nx.DiGraph,
    dict[str, set[str]],
    list[LeaveTwoOutSplit],
    tuple[str, ...],
]:
    universe = load_course_universe(data_dir / "course_vocab.json")
    splits = load_leave_two_out_splits(
        data_dir / "sequences.jsonl",
        min_sequence_length=5,
        item_universe=universe,
    )

    concept_vocab = json.loads(
        (data_dir / "concept_vocab.json").read_text(encoding="utf-8")
    )
    ordered_concepts = [
        concept
        for concept, _ in sorted(
            concept_vocab.items(),
            key=lambda item: int(item[1]),
        )
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_concepts)
    with (data_dir / "prerequisite_edges.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValueError("unexpected prerequisite edge header")
        for row in reader:
            graph.add_edge(row["source"], row["target"])

    raw_mapping = json.loads(
        (data_dir / "course_to_concepts.json").read_text(encoding="utf-8")
    )
    mapping = {
        str(course): set(map(str, concepts))
        for course, concepts in raw_mapping.items()
    }
    return graph, mapping, splits, universe


def _load_raw_domain(
    root: Path,
    *,
    domain: str,
    min_sequence_length: int,
) -> tuple[
    nx.DiGraph,
    dict[str, set[str]],
    list[LeaveTwoOutSplit],
    tuple[str, ...],
]:
    graph, _ = load_mooccube2x_prerequisite_graph(
        root,
        domain=domain,
        validate_dag=True,
    )
    mapping = build_course_to_concept_map(
        root,
        domain,
        allowed_concepts=graph.nodes,
    )

    aligned_rows: list[tuple[str, tuple[str, ...]]] = []
    used_courses: set[str] = set()
    for user_id, transcript in iter_mooccube2x_user_transcripts(
        root,
        min_courses=1,
    ):
        aligned = tuple(
            course for course in transcript if mapping.get(course)
        )
        if len(aligned) < min_sequence_length:
            continue
        aligned_rows.append((user_id, aligned))
        used_courses.update(aligned)

    if not aligned_rows:
        raise ValueError(f"no aligned users for domain {domain}")

    universe = tuple(sorted(used_courses))
    splits = [
        LeaveTwoOutSplit(
            user_id=user_id,
            train=sequence[:-2],
            validation_target=sequence[-2],
            test_target=sequence[-1],
        )
        for user_id, sequence in aligned_rows
    ]
    return graph, mapping, splits, universe


def _analyze_domain(
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
    shape = (users, courses)

    seen_mask = np.zeros(shape, dtype=bool)
    new_count = np.zeros(shape, dtype=np.int16)
    delta = np.zeros(shape, dtype=np.int16)
    edge_drop = np.zeros(shape, dtype=np.int16)
    d1_rank_drop = np.zeros(shape, dtype=np.int16)
    d2_rank_drop = np.zeros(shape, dtype=np.int16)

    course_index = {
        course: index for index, course in enumerate(universe)
    }
    identity_mismatches = 0
    start = time.perf_counter()

    for row, split in enumerate(splits):
        history_courses = split.validation_history
        history_concepts: set[str] = set()
        for course in history_courses:
            history_concepts.update(mapping.get(course, ()))

        for course in set(history_courses):
            index = course_index.get(course)
            if index is not None:
                seen_mask[row, index] = True

        before = evaluator.relative_h1_components(history_concepts)

        for col, course in enumerate(universe):
            if seen_mask[row, col]:
                continue

            bundle = mapping.get(course, set())
            introduced = bundle.difference(history_concepts)
            after = evaluator.relative_h1_components(
                history_concepts.union(bundle)
            )

            dn1 = before.relative_edge_count - after.relative_edge_count
            dr1 = before.rank_d1 - after.rank_d1
            dr2 = before.rank_d2 - after.rank_d2
            db1 = before.beta1 - after.beta1
            reconstructed = dn1 - dr1 - dr2
            if reconstructed != db1:
                identity_mismatches += 1

            values = (len(introduced), db1, dn1, dr1, dr2)
            if any(
                value < np.iinfo(np.int16).min
                or value > np.iinfo(np.int16).max
                for value in values
            ):
                raise OverflowError("decomposition value exceeds int16 range")

            new_count[row, col] = len(introduced)
            delta[row, col] = db1
            edge_drop[row, col] = dn1
            d1_rank_drop[row, col] = dr1
            d2_rank_drop[row, col] = dr2

    elapsed = time.perf_counter() - start
    candidate_mask = ~seen_mask

    abs_delta = np.abs(delta.astype(np.int32))
    abs_edge = np.abs(edge_drop.astype(np.int32))
    abs_d1 = np.abs(d1_rank_drop.astype(np.int32))
    abs_d2 = np.abs(d2_rank_drop.astype(np.int32))

    contributions = np.stack(
        (
            abs_edge[candidate_mask],
            abs_d1[candidate_mask],
            abs_d2[candidate_mask],
        ),
        axis=1,
    )
    max_contribution = contributions.max(axis=1)
    tie_count = (contributions == max_contribution[:, None]).sum(axis=1)
    unique_dominant = tie_count == 1
    dominant_index = contributions.argmax(axis=1)

    total_abs = contributions.sum(axis=1).astype(np.float64)
    abs_output = abs_delta[candidate_mask].astype(np.float64)
    nonzero_total = total_abs > 0
    cancellation = np.zeros_like(total_abs)
    cancellation[nonzero_total] = (
        1.0 - abs_output[nonzero_total] / total_abs[nonzero_total]
    )

    def corr(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
        return correlation_summary(left, right, seen_mask).__dict__

    candidate_delta = delta[candidate_mask].astype(np.int32)
    result: dict[str, object] = {
        "users": users,
        "courses": courses,
        "candidate_states": int(candidate_mask.sum()),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "directed_triangles": evaluator.directed_triangle_count,
        "identity_mismatches": identity_mismatches,
        "elapsed_seconds": elapsed,
        "correlations": {
            "abs_delta_vs_new_concepts": corr(abs_delta, new_count),
            "edge_drop_vs_new_concepts": corr(edge_drop, new_count),
            "abs_delta_vs_edge_drop": corr(abs_delta, edge_drop),
            "delta_vs_edge_drop": corr(delta, edge_drop),
            "d1_rank_drop_vs_new_concepts": corr(
                d1_rank_drop,
                new_count,
            ),
            "d2_rank_drop_vs_new_concepts": corr(
                d2_rank_drop,
                new_count,
            ),
        },
        "component_summary": {
            "mean_abs_delta": float(abs_output.mean()),
            "median_abs_delta": float(np.median(abs_output)),
            "mean_edge_drop": float(
                edge_drop[candidate_mask].astype(np.float64).mean()
            ),
            "median_edge_drop": float(
                np.median(edge_drop[candidate_mask])
            ),
            "mean_abs_d1_rank_drop": float(
                abs_d1[candidate_mask].astype(np.float64).mean()
            ),
            "mean_abs_d2_rank_drop": float(
                abs_d2[candidate_mask].astype(np.float64).mean()
            ),
            "mean_cancellation_fraction": float(cancellation.mean()),
            "median_cancellation_fraction": float(
                np.median(cancellation)
            ),
            "edge_unique_dominant_fraction": float(
                np.mean(
                    unique_dominant
                    & (dominant_index == 0)
                )
            ),
            "d1_unique_dominant_fraction": float(
                np.mean(
                    unique_dominant
                    & (dominant_index == 1)
                )
            ),
            "d2_unique_dominant_fraction": float(
                np.mean(
                    unique_dominant
                    & (dominant_index == 2)
                )
            ),
            "dominant_tie_fraction": float(
                np.mean(~unique_dominant)
            ),
            "edge_drop_per_new_concept": _safe_ratio_summary(
                edge_drop,
                new_count,
                candidate_mask,
            ),
        },
        "ranges": {
            "delta": [
                int(candidate_delta.min()),
                int(candidate_delta.max()),
            ],
            "edge_drop": [
                int(edge_drop[candidate_mask].min()),
                int(edge_drop[candidate_mask].max()),
            ],
            "d1_rank_drop": [
                int(d1_rank_drop[candidate_mask].min()),
                int(d1_rank_drop[candidate_mask].max()),
            ],
            "d2_rank_drop": [
                int(d2_rank_drop[candidate_mask].min()),
                int(d2_rank_drop[candidate_mask].max()),
            ],
        },
        "evaluator_component_cache_size": len(evaluator._component_cache),
    }
    return result


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_018 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    raw_root = resolve_mooccube2x_root(None)
    processed_cs = REPO_ROOT / str(config["processed_cs_dir"])
    cache_size = int(config["evaluator_cache_size"])
    min_sequence_length = int(config["min_sequence_length"])

    results: dict[str, object] = {}

    graph, mapping, splits, universe = _load_processed_cs(processed_cs)
    results["cs"] = _analyze_domain(
        graph=graph,
        mapping=mapping,
        splits=splits,
        universe=universe,
        cache_size=cache_size,
    )

    for domain in ("math", "psy"):
        graph, mapping, splits, universe = _load_raw_domain(
            raw_root,
            domain=domain,
            min_sequence_length=min_sequence_length,
        )
        results[domain] = _analyze_domain(
            graph=graph,
            mapping=mapping,
            splits=splits,
            universe=universe,
            cache_size=cache_size,
        )

    if any(
        int(result["identity_mismatches"]) != 0
        for result in results.values()
    ):
        raise RuntimeError("relative-chain decomposition identity failed")

    payload: dict[str, object] = {
        "experiment_id": "exp_018_relative_chain_decomposition",
        "status": "completed",
        "phase": "mechanism_explanation",
        "evidence_scope": "target_independent",
        "identity": (
            "Delta beta1 = Delta dim(C1_rel) - Delta rank(d1) "
            "- Delta rank(d2)"
        ),
        "domains": results,
        "selection_performed": False,
        "recommendation_metrics_evaluated": False,
        "target_performance_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# exp_018 - Exact Relative-Chain Decomposition",
        "",
        "- Evidence scope: target-independent mechanism explanation",
        "- Selection performed: NO",
        "- Recommendation metrics evaluated: NO",
        "- Target performance evaluated: NO",
        "- Development-test evaluated: NO",
        "",
        "Exact identity:",
        "",
        "Delta beta_1 = Delta dim(C_1 rel) - Delta rank(d_1) - Delta rank(d_2)",
        "",
    ]

    for domain in ("cs", "math", "psy"):
        result = results[domain]
        corr = result["correlations"]
        comp = result["component_summary"]
        lines.extend(
            [
                f"## {domain.upper()}",
                "",
                (
                    f"- Users/courses/candidate states: "
                    f"{int(result['users']):,}/"
                    f"{int(result['courses']):,}/"
                    f"{int(result['candidate_states']):,}"
                ),
                (
                    "- Identity mismatches: "
                    f"{int(result['identity_mismatches'])}"
                ),
                (
                    "- |Delta| vs new concepts Spearman: "
                    f"{corr['abs_delta_vs_new_concepts']['global_spearman']}"
                ),
                (
                    "- Removed relative edges vs new concepts Spearman: "
                    f"{corr['edge_drop_vs_new_concepts']['global_spearman']}"
                ),
                (
                    "- |Delta| vs removed relative edges Spearman: "
                    f"{corr['abs_delta_vs_edge_drop']['global_spearman']}"
                ),
                (
                    "- d1-rank drop vs new concepts Spearman: "
                    f"{corr['d1_rank_drop_vs_new_concepts']['global_spearman']}"
                ),
                (
                    "- d2-rank drop vs new concepts Spearman: "
                    f"{corr['d2_rank_drop_vs_new_concepts']['global_spearman']}"
                ),
                (
                    "- Mean |Delta| / edge drop / |d1 drop| / |d2 drop|: "
                    f"{float(comp['mean_abs_delta']):.4f} / "
                    f"{float(comp['mean_edge_drop']):.4f} / "
                    f"{float(comp['mean_abs_d1_rank_drop']):.4f} / "
                    f"{float(comp['mean_abs_d2_rank_drop']):.4f}"
                ),
                (
                    "- Mean cancellation fraction: "
                    f"{float(comp['mean_cancellation_fraction']):.4f}"
                ),
                (
                    "- Unique dominant term fractions edge/d1/d2: "
                    f"{float(comp['edge_unique_dominant_fraction']):.4f}/"
                    f"{float(comp['d1_unique_dominant_fraction']):.4f}/"
                    f"{float(comp['d2_unique_dominant_fraction']):.4f}"
                ),
                (
                    "- Dominant-term tie fraction: "
                    f"{float(comp['dominant_tie_fraction']):.4f}"
                ),
                (
                    "- Runtime seconds: "
                    f"{float(result['elapsed_seconds']):.3f}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "This experiment explains the existing raw Delta beta_1 only. "
                "It does not define or evaluate a replacement recommender "
                "score. Any revised score must be specified after reviewing "
                "this decomposition and tested only in a fresh external setting."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("exp_018_relative_chain_decomposition")
    for domain in ("cs", "math", "psy"):
        result = results[domain]
        corr = result["correlations"]
        comp = result["component_summary"]
        print(
            f"{domain.upper()}: "
            f"edge-new={corr['edge_drop_vs_new_concepts']['global_spearman']}, "
            f"|Delta|-edge={corr['abs_delta_vs_edge_drop']['global_spearman']}, "
            f"cancel={float(comp['mean_cancellation_fraction']):.4f}, "
            f"identity_mismatches={int(result['identity_mismatches'])}"
        )
    print("Recommendation metrics evaluated: False")
    print("Target performance evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
