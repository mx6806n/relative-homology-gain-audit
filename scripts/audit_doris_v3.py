"""Run the DORIS v3 conservative prerequisite audit.

v3 is a precision-repair audit motivated only by the independently observed
v2 text-matching failure modes. It preserves the original feasibility
thresholds and creates a new manual holdout disjoint from the v2 review sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.audit_doris_supplement as v2  # noqa: E402
from relhom.doris_text_match import extract_exact_course_ids, phrase_norm  # noqa: E402


HOLDOUT_SEED = 20261011


def _phrase_catalog(
    course_id_to_name: dict[str, str],
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for course_id, name in course_id_to_name.items():
        normalized = phrase_norm(name)
        if normalized:
            mapping[normalized].add(course_id)
    return mapping


def _extract_prerequisite_edges_v3(
    source: str,
    course_info: pd.DataFrame,
    course_id_to_name: dict[str, str],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    prereq_cols = v2._find_columns(course_info.columns, v2.PREREQ_HINTS)
    id_cols = v2._find_columns(course_info.columns, v2.COURSE_ID_HINTS)
    if len(prereq_cols) != 1 or len(id_cols) != 1:
        raise RuntimeError(
            "English CourseInformationTable must expose one Prerequisite and "
            "one CourseId column"
        )

    prereq_col = prereq_cols[0]
    target_id_col = id_cols[0]
    course_ids = set(course_id_to_name)
    phrase_to_ids = _phrase_catalog(course_id_to_name)

    edge_records: dict[tuple[str, str], dict[str, str]] = {}
    nonempty_rows = 0
    resolved_rows = 0
    rows_without_usable_nonself_match = 0

    for _, row in course_info.iterrows():
        raw = row[prereq_col]
        if pd.isna(raw) or not phrase_norm(raw):
            continue
        nonempty_rows += 1

        target_id = v2._norm_id(row[target_id_col])
        if not target_id or target_id not in course_ids:
            continue

        found_ids = extract_exact_course_ids(raw, phrase_to_ids)
        found_ids.discard(target_id)

        if not found_ids:
            rows_without_usable_nonself_match += 1
            continue

        resolved_rows += 1
        for source_id in sorted(found_ids):
            key = (source_id, target_id)
            edge_records.setdefault(
                key,
                {
                    "source_course_id": source_id,
                    "source_course_name": course_id_to_name.get(source_id, ""),
                    "target_course_id": target_id,
                    "target_course_name": course_id_to_name.get(target_id, ""),
                    "target_prerequisite_text": str(raw),
                    "table_source": source,
                },
            )

    records = list(edge_records.values())
    graph_courses = {
        row["source_course_id"] for row in records
    } | {
        row["target_course_id"] for row in records
    }

    graph = nx.DiGraph()
    graph.add_edges_from(
        (row["source_course_id"], row["target_course_id"])
        for row in records
    )
    scc_sizes = (
        sorted(
            (
                len(component)
                for component in nx.strongly_connected_components(graph)
            ),
            reverse=True,
        )
        if graph.number_of_nodes()
        else []
    )

    diagnostics: dict[str, object] = {
        "canonical_source": source,
        "resolver": (
            "v3 exact unambiguous course-phrase resolver; no arbitrary "
            "substring matches; no bare numeric CourseId matches"
        ),
        "nonempty_prerequisite_rows": nonempty_rows,
        "resolved_rows": resolved_rows,
        "row_resolution_fraction": (
            resolved_rows / nonempty_rows if nonempty_rows else 0.0
        ),
        "rows_without_usable_nonself_match": (
            rows_without_usable_nonself_match
        ),
        "unique_nonself_edges": len(records),
        "distinct_graph_courses": len(graph_courses),
        "graph_is_dag": nx.is_directed_acyclic_graph(graph),
        "strongly_connected_components_gt1": sum(
            1 for size in scc_sizes if size > 1
        ),
        "largest_strongly_connected_component": (
            scc_sizes[0] if scc_sizes else 0
        ),
    }
    return diagnostics, records


def _load_excluded_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "v3 holdout requires the committed v2 manual-review sample so "
            "reviewed edges can be excluded"
        )

    excluded: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_course_id", "target_course_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("invalid v2 review CSV")
        for row in reader:
            source = v2._norm_id(row["source_course_id"])
            target = v2._norm_id(row["target_course_id"])
            if source and target:
                excluded.add((source, target))
    return excluded


def _write_outputs(
    output_path: Path,
    records: list[dict[str, str]],
    *,
    excluded_pairs: set[tuple[str, str]],
) -> tuple[Path, Path, int]:
    edge_path = output_path.with_name(
        "doris_candidate_prerequisite_edges_v3.csv"
    )
    sample_path = output_path.with_name(
        "doris_prerequisite_manual_holdout_v3.csv"
    )

    fields = [
        "source_course_id",
        "source_course_name",
        "target_course_id",
        "target_course_name",
        "target_prerequisite_text",
        "table_source",
    ]
    ordered = sorted(
        records,
        key=lambda row: (
            row["target_course_id"],
            row["source_course_id"],
        ),
    )

    with edge_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    eligible = [
        row
        for row in ordered
        if (
            row["source_course_id"],
            row["target_course_id"],
        )
        not in excluded_pairs
    ]
    sample_size = min(100, len(eligible))
    rng = random.Random(HOLDOUT_SEED)
    sample = rng.sample(eligible, sample_size) if sample_size else []

    review_fields = fields + ["manual_supported_0_or_1", "manual_note"]
    with sample_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in sample:
            writer.writerow(
                {
                    **row,
                    "manual_supported_0_or_1": "",
                    "manual_note": "",
                }
            )

    return edge_path, sample_path, len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument(
        "--exclude-review",
        type=Path,
        default=Path("results/doris_prerequisite_manual_review_sample.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/doris_feasibility_audit_v3.json"),
    )
    args = parser.parse_args()

    paths = [args.s1, args.s2]
    frames = v2._collect_frames(paths)

    course_source, course_info = v2._select_primary_frame(
        frames,
        v2.COURSE_INFO_TOKEN,
    )
    selection_source, selection = v2._select_primary_frame(
        frames,
        v2.COURSE_SELECTION_TOKEN,
    )

    course_id_to_name, _, catalog_meta = v2._build_catalog(
        course_source,
        course_info,
    )
    prerequisite_diag, edge_records = _extract_prerequisite_edges_v3(
        course_source,
        course_info,
        course_id_to_name,
    )
    graph_course_ids = {
        row["source_course_id"] for row in edge_records
    } | {
        row["target_course_id"] for row in edge_records
    }

    sequence_diag = v2._sequence_and_alignment(
        selection_source,
        selection,
        graph_course_ids,
    )

    excluded_pairs = _load_excluded_pairs(args.exclude_review)
    edge_path, sample_path, holdout_pool_size = _write_outputs(
        args.output,
        edge_records,
        excluded_pairs=excluded_pairs,
    )

    thresholds = {
        "students_ge5_events": 2000,
        "students_ge3_semesters": 1500,
        "prerequisite_row_resolution_fraction": 0.25,
        "unique_prerequisite_edges": 500,
        "distinct_graph_courses": 300,
        "manual_edge_precision": 0.90,
        "graph_courses_in_histories_min": 500,
        "graph_course_history_coverage_fraction_if_graph_lt500": 0.80,
        "students_ge3_graph_mapped_events": 2000,
    }

    preliminary_gate = {
        "students_ge5_events": (
            sequence_diag["students_ge5_events"]
            >= thresholds["students_ge5_events"]
        ),
        "students_ge3_semesters": (
            sequence_diag["students_ge3_semesters"]
            >= thresholds["students_ge3_semesters"]
        ),
        "prerequisite_row_resolution_fraction": (
            prerequisite_diag["row_resolution_fraction"]
            >= thresholds["prerequisite_row_resolution_fraction"]
        ),
        "unique_prerequisite_edges": (
            prerequisite_diag["unique_nonself_edges"]
            >= thresholds["unique_prerequisite_edges"]
        ),
        "distinct_graph_courses": (
            prerequisite_diag["distinct_graph_courses"]
            >= thresholds["distinct_graph_courses"]
        ),
        "graph_course_history_alignment": (
            sequence_diag["graph_courses_in_histories"]
            >= thresholds["graph_courses_in_histories_min"]
            if prerequisite_diag["distinct_graph_courses"] >= 500
            else sequence_diag["graph_course_history_coverage_fraction"]
            >= thresholds[
                "graph_course_history_coverage_fraction_if_graph_lt500"
            ]
        ),
        "students_ge3_graph_mapped_events": (
            sequence_diag["students_ge3_graph_mapped_events"]
            >= thresholds["students_ge3_graph_mapped_events"]
        ),
    }

    payload = {
        "audit_version": 3,
        "status": "preliminary_pending_disjoint_manual_precision_review",
        "repair_basis": (
            "v2 independent precision error taxonomy only; topology outcomes "
            "were not used to define v3 matching rules"
        ),
        "mirror_policy": (
            "English supplement is canonical; Chinese supplement is archive "
            "consistency metadata only."
        ),
        "holdout_policy": {
            "seed": HOLDOUT_SEED,
            "excluded_v2_review_edges": len(excluded_pairs),
            "eligible_disjoint_holdout_pool": holdout_pool_size,
            "sample_size": min(100, holdout_pool_size),
        },
        "archives": [v2._scan_zip(path) for path in paths],
        "catalog": catalog_meta,
        "prerequisites": prerequisite_diag,
        "sequence_and_alignment": sequence_diag,
        "feasibility_thresholds": thresholds,
        "preliminary_gate": preliminary_gate,
        "preliminary_all_nonmanual_gates_pass": all(
            preliminary_gate.values()
        ),
        "manual_precision_gate_evaluated": False,
        "edge_output": str(edge_path),
        "manual_review_sample": str(sample_path),
        "note": (
            "Scientific use remains blocked until the new disjoint v3 manual "
            "holdout reaches the unchanged 0.90 precision threshold."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload["catalog"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["prerequisites"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["sequence_and_alignment"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["preliminary_gate"], ensure_ascii=False, indent=2))
    print(
        "Preliminary all non-manual gates pass: "
        f"{payload['preliminary_all_nonmanual_gates_pass']}"
    )
    print(f"Disjoint manual holdout: {sample_path}")
    print(f"Audit written to: {args.output}")


if __name__ == "__main__":
    main()
