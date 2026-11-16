"""Run exp_016: untouched-domain feasibility audit for Math/Psychology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.data_loader import (  # noqa: E402
    analyze_aligned_cohort,
    build_course_to_concept_map,
    load_mooccube2x_prerequisite_graph,
    resolve_mooccube2x_root,
)
from relhom.domain_feasibility import (  # noqa: E402
    domain_is_feasible,
    select_feasible_domain,
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON/YAML object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    selection_path = output_dir / "selected_domain.json"

    for path in (metrics_path, summary_path, selection_path):
        if path.exists():
            raise FileExistsError(
                "exp_016 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    root = resolve_mooccube2x_root(None)
    domains = tuple(str(x) for x in config["domains"])
    thresholds = config["feasibility_thresholds"]
    if not isinstance(thresholds, dict):
        raise ValueError("feasibility_thresholds must be an object")

    min_users_ge5 = int(thresholds["min_users_ge5"])
    min_unique_courses = int(thresholds["min_unique_courses"])
    min_coverage = float(
        thresholds["min_concept_coverage_percent"]
    )

    rows: list[dict[str, object]] = []
    for domain in domains:
        graph, prereq_stats = load_mooccube2x_prerequisite_graph(
            root,
            domain=domain,
            validate_dag=False,
        )
        mapping = build_course_to_concept_map(
            root,
            domain,
            allowed_concepts=graph.nodes,
        )
        stats = analyze_aligned_cohort(
            root,
            course_to_concept_map=mapping,
            concept_graph=graph,
            export_dir=None,
            min_export_courses=3,
        )

        row: dict[str, object] = {
            "domain": domain,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "graph_is_dag": nx.is_directed_acyclic_graph(graph),
            "prerequisite_total_records": prereq_stats.total_records,
            "prerequisite_positive_records": prereq_stats.positive_records,
            "prerequisite_malformed_records": prereq_stats.malformed_records,
            "users_ge1": int(
                stats["users_by_min_aligned_courses"]["1"]
            ),
            "users_ge2": int(
                stats["users_by_min_aligned_courses"]["2"]
            ),
            "users_ge3": int(
                stats["users_by_min_aligned_courses"]["3"]
            ),
            "users_ge5": int(
                stats["users_by_min_aligned_courses"]["5"]
            ),
            "users_ge10": int(
                stats["users_by_min_aligned_courses"]["10"]
            ),
            "mapped_enrollment_events": int(
                stats["mapped_enrollment_events"]
            ),
            "aligned_next_course_transitions": int(
                stats["aligned_next_course_transitions"]
            ),
            "mapped_courses_available": int(
                stats["mapped_courses_available"]
            ),
            "unique_mapped_courses_used": int(
                stats["unique_mapped_courses_used"]
            ),
            "dag_concepts": int(stats["dag_concepts"]),
            "dag_concepts_mapped_by_courses": int(
                stats["dag_concepts_mapped_by_courses"]
            ),
            "dag_concepts_reached_by_users": int(
                stats["dag_concepts_reached_by_users"]
            ),
            "dag_concept_user_coverage_percent": float(
                stats["dag_concept_user_coverage_percent"]
            ),
            "terminal_target_coverage_percent": float(
                stats["terminal_target_coverage_percent"]
            ),
            "aligned_sequence_length_percentiles": (
                stats["aligned_sequence_length_percentiles"]
            ),
        }
        row["feasible"] = domain_is_feasible(
            row,
            min_users_ge5=min_users_ge5,
            min_unique_courses=min_unique_courses,
            min_concept_coverage_percent=min_coverage,
        )
        rows.append(row)

    selected = select_feasible_domain(
        rows,
        min_users_ge5=min_users_ge5,
        min_unique_courses=min_unique_courses,
        min_concept_coverage_percent=min_coverage,
    )

    raw_hashes = {
        "user_json_sha256": _sha256(root / "entities" / "user.json"),
        "concept_json_sha256": _sha256(root / "entities" / "concept.json"),
        "concept_course_sha256": _sha256(
            root / "relations" / "concept-course.txt"
        ),
    }
    for domain in domains:
        raw_hashes[f"{domain}_prerequisite_sha256"] = _sha256(
            root / "prerequisites" / f"{domain}.json"
        )

    payload: dict[str, object] = {
        "experiment_id": "exp_016_secondary_domain_feasibility",
        "status": "completed",
        "phase": "fresh_setting_selection",
        "evidence_scope": "feasibility_only",
        "domains": rows,
        "feasibility_thresholds": {
            "min_users_ge5": min_users_ge5,
            "min_unique_courses": min_unique_courses,
            "min_concept_coverage_percent": min_coverage,
        },
        "selection_rule": (
            "among feasible domains: maximize users_ge5, then "
            "unique_mapped_courses_used, then concept coverage; "
            "alphabetical domain tie-break"
        ),
        "selected_domain": selected,
        "raw_data_hashes": raw_hashes,
        "recommendation_metrics_evaluated": False,
        "topology_target_signal_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_path.write_text(
        json.dumps(
            {
                "source_experiment": (
                    "exp_016_secondary_domain_feasibility"
                ),
                "selected_domain": selected,
                "selection_basis": "feasibility_only",
                "thresholds": payload["feasibility_thresholds"],
                "selection_rule": payload["selection_rule"],
                "recommendation_metrics_evaluated": False,
                "topology_target_signal_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_016 - Secondary-Domain Feasibility Audit",
        "",
        "- Evidence scope: feasibility only",
        "- Recommendation metrics evaluated: NO",
        "- Topology target-signal performance evaluated: NO",
        "- Development-test evaluated: NO",
        "",
        "## Predeclared feasibility thresholds",
        "",
        f"- Users with >=5 aligned events: >= {min_users_ge5:,}",
        f"- Unique mapped courses used: >= {min_unique_courses}",
        (
            "- DAG concepts reached by aligned users: >= "
            f"{min_coverage:.1f}%"
        ),
        "- Positive prerequisite graph must be a DAG",
        "",
        "## Domain audit",
        "",
    ]

    for row in rows:
        summary_lines.extend(
            [
                f"### {str(row['domain']).upper()}",
                "",
                (
                    f"- Graph: {int(row['graph_nodes']):,} nodes, "
                    f"{int(row['graph_edges']):,} edges, "
                    f"DAG={bool(row['graph_is_dag'])}"
                ),
                (
                    "- Users by aligned sequence length: "
                    f">=3 {int(row['users_ge3']):,}, "
                    f">=5 {int(row['users_ge5']):,}, "
                    f">=10 {int(row['users_ge10']):,}"
                ),
                (
                    "- Mapped courses used/available: "
                    f"{int(row['unique_mapped_courses_used']):,}/"
                    f"{int(row['mapped_courses_available']):,}"
                ),
                (
                    "- DAG concept user coverage: "
                    f"{float(row['dag_concept_user_coverage_percent']):.2f}%"
                ),
                (
                    "- Terminal-target mapping coverage: "
                    f"{float(row['terminal_target_coverage_percent']):.2f}%"
                ),
                f"- Feasible: {bool(row['feasible'])}",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Selection",
            "",
            f"- Selected untouched domain: {selected}",
            "",
            (
                "Selection uses only sample-size and structural-coverage "
                "criteria. No recommendation or topology-target performance "
                "is inspected in this experiment."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_016_secondary_domain_feasibility")
    for row in rows:
        print(
            f"{str(row['domain']).upper()}: "
            f"users>=5={int(row['users_ge5']):,}, "
            f"courses={int(row['unique_mapped_courses_used']):,}, "
            f"coverage={float(row['dag_concept_user_coverage_percent']):.2f}%, "
            f"feasible={bool(row['feasible'])}"
        )
    print(f"Selected domain: {selected}")
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
