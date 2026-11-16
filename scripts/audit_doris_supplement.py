"""Audit DORIS supplementary ZIPs without training or target evaluation.

The English supplement is treated as the canonical mirror for graph extraction
and sequence diagnostics. The Chinese supplement is scanned for archive
consistency only so bilingual copies are never double-counted as independent
records.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")
PREREQ_HINTS = ("prereq", "prerequisite", "先修")
COURSE_ID_HINTS = ("courseid", "course_id", "课程号", "课程编号")
COURSE_NAME_HINTS = ("coursename", "course_name", "name", "课程名", "课程名称")
STUDENT_HINTS = (
    "studentid",
    "student_id",
    "studedntid",
    "学生",
)
YEAR_HINTS = ("academicyear", "academic_year", "学年")
SEMESTER_HINTS = ("semester", "学期")

PRIMARY_MIRROR_TOKEN = "datasetinenglish"
COURSE_INFO_TOKEN = "courseinformationtable"
COURSE_SELECTION_TOKEN = "courseselectiontable"


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,；;、/\\]+", "|", text)
    return text


def _norm_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _read_member(blob: bytes, name: str) -> pd.DataFrame | None:
    lower = name.lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(blob))
        except ImportError as exc:
            raise RuntimeError(
                f"spreadsheet member {name!r} requires an Excel reader "
                "dependency not present in the project environment"
            ) from exc
        except Exception:
            return None

    if not lower.endswith((".csv", ".tsv", ".txt")):
        return None

    for encoding in TEXT_ENCODINGS:
        try:
            text = blob.decode(encoding)
        except UnicodeDecodeError:
            continue
        for sep in (None, ",", "\t"):
            try:
                return pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception:
                continue
    return None


def _scan_zip(path: Path) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            blob = archive.read(info.filename)
            frame = _read_member(blob, info.filename)
            suffix = Path(info.filename).suffix.lower()
            item = {
                "archive": str(path),
                "member": info.filename,
                "bytes": info.file_size,
                "suffix": suffix,
                "tabular": frame is not None,
            }
            if frame is not None:
                item["rows"] = int(len(frame))
                item["columns"] = [str(col) for col in frame.columns]
            rows.append(item)
    return rows


def _collect_frames(paths: list[Path]) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                frame = _read_member(archive.read(info.filename), info.filename)
                if frame is not None:
                    frames.append((f"{path.name}:{info.filename}", frame))
    return frames


def _find_columns(columns, hints):
    result = []
    for col in columns:
        normalized = _norm(col)
        if any(_norm(hint) == normalized for hint in hints):
            result.append(col)
    return result


def _select_primary_frame(
    frames: list[tuple[str, pd.DataFrame]],
    token: str,
) -> tuple[str, pd.DataFrame]:
    candidates = [
        (source, frame)
        for source, frame in frames
        if PRIMARY_MIRROR_TOKEN in _norm(source)
        and token in _norm(source)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one English primary frame for {token!r}; "
            f"found {[source for source, _ in candidates]}"
        )
    return candidates[0]


def _build_catalog(
    course_info_source: str,
    course_info: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, str]]:
    id_cols = _find_columns(course_info.columns, COURSE_ID_HINTS)
    name_cols = _find_columns(course_info.columns, COURSE_NAME_HINTS)
    if len(id_cols) != 1 or not name_cols:
        raise RuntimeError(
            "English CourseInformationTable must expose one CourseId and a "
            "course-name column"
        )

    id_col = id_cols[0]
    name_col = next(
        (col for col in name_cols if _norm(col) == "coursename"),
        name_cols[0],
    )

    course_id_to_name: dict[str, str] = {}
    normalized_name_to_ids: dict[str, set[str]] = defaultdict(set)
    for _, row in course_info.iterrows():
        course_id = _norm_id(row[id_col])
        name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        normalized_name = _norm(name)
        if not course_id:
            continue
        course_id_to_name[course_id] = name
        if normalized_name:
            normalized_name_to_ids[normalized_name].add(course_id)

    unambiguous_name_to_id = {
        name: next(iter(ids))
        for name, ids in normalized_name_to_ids.items()
        if len(ids) == 1
    }
    ambiguous_name_count = sum(
        1 for ids in normalized_name_to_ids.values() if len(ids) > 1
    )

    meta = {
        "source": course_info_source,
        "course_id_column": str(id_col),
        "course_name_column": str(name_col),
        "course_ids": len(course_id_to_name),
        "normalized_names": len(normalized_name_to_ids),
        "unambiguous_normalized_names": len(unambiguous_name_to_id),
        "ambiguous_normalized_names": ambiguous_name_count,
    }
    return course_id_to_name, normalized_name_to_ids, meta


def _extract_prerequisite_edges(
    source: str,
    course_info: pd.DataFrame,
    course_id_to_name: dict[str, str],
    normalized_name_to_ids: dict[str, set[str]],
) -> tuple[dict, list[dict]]:
    prereq_cols = _find_columns(course_info.columns, PREREQ_HINTS)
    id_cols = _find_columns(course_info.columns, COURSE_ID_HINTS)
    if len(prereq_cols) != 1 or len(id_cols) != 1:
        raise RuntimeError(
            "English CourseInformationTable must expose one Prerequisite and "
            "one CourseId column"
        )

    prereq_col = prereq_cols[0]
    target_id_col = id_cols[0]
    course_ids = set(course_id_to_name)

    name_to_id = {
        name: next(iter(ids))
        for name, ids in normalized_name_to_ids.items()
        if len(ids) == 1 and len(name) >= 3
    }
    searchable_names = sorted(name_to_id, key=len, reverse=True)

    edge_records: dict[tuple[str, str], dict] = {}
    nonempty_rows = 0
    resolved_rows = 0
    rows_with_only_ambiguous_or_self = 0

    for _, row in course_info.iterrows():
        raw = row[prereq_col]
        if pd.isna(raw) or not _norm(raw):
            continue

        nonempty_rows += 1
        text = _norm(raw)
        target_id = _norm_id(row[target_id_col])
        if not target_id or target_id not in course_ids:
            continue

        found_ids: set[str] = set()

        for token in re.findall(r"(?<!\d)\d{2,}(?!\d)", text):
            normalized_token = _norm_id(token)
            if normalized_token in course_ids:
                found_ids.add(normalized_token)

        for name in searchable_names:
            if name in text:
                found_ids.add(name_to_id[name])

        found_ids.discard(target_id)
        if not found_ids:
            rows_with_only_ambiguous_or_self += 1
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
        record["source_course_id"] for record in records
    } | {
        record["target_course_id"] for record in records
    }

    graph = nx.DiGraph()
    graph.add_edges_from(
        (record["source_course_id"], record["target_course_id"])
        for record in records
    )

    scc_sizes = sorted(
        (len(component) for component in nx.strongly_connected_components(graph)),
        reverse=True,
    ) if graph.number_of_nodes() else []

    diagnostics = {
        "canonical_source": source,
        "nonempty_prerequisite_rows": nonempty_rows,
        "resolved_rows": resolved_rows,
        "row_resolution_fraction": (
            resolved_rows / nonempty_rows if nonempty_rows else 0.0
        ),
        "rows_without_usable_nonself_match": rows_with_only_ambiguous_or_self,
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


def _sequence_and_alignment(
    source: str,
    selection: pd.DataFrame,
    graph_course_ids: set[str],
) -> dict:
    student_cols = _find_columns(selection.columns, STUDENT_HINTS)
    year_cols = _find_columns(selection.columns, YEAR_HINTS)
    semester_cols = _find_columns(selection.columns, SEMESTER_HINTS)
    course_cols = _find_columns(selection.columns, COURSE_ID_HINTS)

    if (
        len(student_cols) != 1
        or len(year_cols) != 1
        or len(semester_cols) != 1
        or len(course_cols) != 1
    ):
        raise RuntimeError(
            "English CourseSelectionTable columns could not be identified "
            f"uniquely: student={student_cols}, year={year_cols}, "
            f"semester={semester_cols}, course={course_cols}"
        )

    student_col = student_cols[0]
    year_col = year_cols[0]
    semester_col = semester_cols[0]
    course_col = course_cols[0]

    frame = selection.copy()
    frame["_student"] = frame[student_col].map(_norm_id)
    frame["_course"] = frame[course_col].map(_norm_id)
    frame = frame[(frame["_student"] != "") & (frame["_course"] != "")]

    event_counts = frame.groupby("_student").size()
    sessions = (
        frame[["_student", year_col, semester_col]]
        .drop_duplicates()
        .groupby("_student")
        .size()
    )

    mapped = frame[frame["_course"].isin(graph_course_ids)]
    mapped_counts = mapped.groupby("_student").size()

    history_course_ids = set(frame["_course"].unique())
    graph_courses_in_histories = graph_course_ids.intersection(history_course_ids)

    return {
        "canonical_source": source,
        "rows": int(len(frame)),
        "students": int(event_counts.size),
        "students_ge5_events": int((event_counts >= 5).sum()),
        "students_ge3_semesters": int((sessions >= 3).sum()),
        "distinct_history_courses": len(history_course_ids),
        "graph_courses_in_histories": len(graph_courses_in_histories),
        "graph_course_history_coverage_fraction": (
            len(graph_courses_in_histories) / len(graph_course_ids)
            if graph_course_ids else 0.0
        ),
        "graph_mapped_events": int(len(mapped)),
        "students_ge3_graph_mapped_events": int((mapped_counts >= 3).sum()),
    }


def _write_edge_outputs(
    output_path: Path,
    records: list[dict],
) -> tuple[Path, Path]:
    edge_path = output_path.with_name("doris_candidate_prerequisite_edges.csv")
    sample_path = output_path.with_name(
        "doris_prerequisite_manual_review_sample.csv"
    )

    fieldnames = [
        "source_course_id",
        "source_course_name",
        "target_course_id",
        "target_course_name",
        "target_prerequisite_text",
        "table_source",
    ]
    with edge_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(
            records,
            key=lambda row: (
                row["target_course_id"],
                row["source_course_id"],
            ),
        ))

    rng = random.Random(20261009)
    sample_size = min(100, len(records))
    sample = rng.sample(records, sample_size) if sample_size else []
    review_fields = fieldnames + ["manual_supported_0_or_1", "manual_note"]
    with sample_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in sample:
            writer.writerow({
                **row,
                "manual_supported_0_or_1": "",
                "manual_note": "",
            })

    return edge_path, sample_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/doris_feasibility_audit.json"),
    )
    args = parser.parse_args()

    paths = [args.s1, args.s2]
    frames = _collect_frames(paths)

    course_source, course_info = _select_primary_frame(
        frames,
        COURSE_INFO_TOKEN,
    )
    selection_source, selection = _select_primary_frame(
        frames,
        COURSE_SELECTION_TOKEN,
    )

    course_id_to_name, name_to_ids, catalog_meta = _build_catalog(
        course_source,
        course_info,
    )
    prerequisite_diag, edge_records = _extract_prerequisite_edges(
        course_source,
        course_info,
        course_id_to_name,
        name_to_ids,
    )
    graph_course_ids = {
        record["source_course_id"] for record in edge_records
    } | {
        record["target_course_id"] for record in edge_records
    }

    sequence_diag = _sequence_and_alignment(
        selection_source,
        selection,
        graph_course_ids,
    )

    edge_path, sample_path = _write_edge_outputs(
        args.output,
        edge_records,
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
        "audit_version": 2,
        "status": "preliminary_pending_manual_precision_review",
        "mirror_policy": (
            "English supplement is canonical; Chinese supplement is archive "
            "consistency metadata only."
        ),
        "archives": [_scan_zip(path) for path in paths],
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
            "Primary graph edges use stable CourseId->CourseId mappings from "
            "the English mirror only. Scientific use remains blocked until "
            "manual precision review reaches the predeclared 0.90 threshold."
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
    print(f"Manual review sample: {sample_path}")
    print(f"Audit written to: {args.output}")


if __name__ == "__main__":
    main()
