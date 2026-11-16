"""Run exp_021: structural analysis of relative H1 gains on DORIS semester bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
import re
import sys
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.topology import GF2RelativeHomologyEvaluator  # noqa: E402


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _load_graph(edges_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    with edges_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_course_id", "target_course_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "edge CSV must contain source_course_id and target_course_id"
            )
        for row in reader:
            source = _norm_id(row["source_course_id"])
            target = _norm_id(row["target_course_id"])
            if not source or not target or source == target:
                continue
            graph.add_edge(source, target)

    if graph.number_of_edges() == 0:
        raise ValueError("no prerequisite edges loaded")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("DORIS prerequisite graph must be a DAG")
    return graph


def _read_selection_table(s2_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(s2_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if (
                "datasetinenglish" in name.lower()
                and "courseselectiontable.xlsx" in name.lower()
                and "__macosx" not in name.lower()
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "expected exactly one English CourseSelectionTable.xlsx; "
                f"found {candidates}"
            )
        return pd.read_excel(io.BytesIO(archive.read(candidates[0])))


def _find_column(columns: list[object], accepted: set[str]) -> object:
    matches = [
        column
        for column in columns
        if str(column).strip().lower().replace("_", "") in accepted
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one column matching {sorted(accepted)}; "
            f"found {matches}"
        )
    return matches[0]


def _academic_year_key(value: object) -> int:
    """Return the starting calendar year for a DORIS academic-year label.

    The released table uses compact labels such as 18-19 in addition to
    ordinary numeric/full-year forms. Only chronological ordering is needed
    here, so 18-19 is interpreted as the 2018-2019 academic year and keyed by
    its starting year, 2018.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not math.isnan(value):
        return int(value)

    text = str(value).strip()

    full_year = re.search(r"(?:19|20)\d{2}", text)
    if full_year:
        return int(full_year.group(0))

    compact_range = re.fullmatch(
        r"(?P<start>\d{2})\s*[-–—/]\s*(?P<end>\d{2})",
        text,
    )
    if compact_range:
        start = int(compact_range.group("start"))
        end = int(compact_range.group("end"))
        expected_end = (start + 1) % 100
        if end != expected_end:
            raise ValueError(
                "unexpected compact AcademicYear range "
                f"{value!r}; expected consecutive years"
            )
        return 2000 + start

    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"cannot order AcademicYear value {value!r}") from exc


def _semester_key(value: object) -> float:
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, float) and not math.isnan(value):
        return float(value)

    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        pass

    aliases = {
        "first": 1.0,
        "1st": 1.0,
        "fall": 1.0,
        "autumn": 1.0,
        "second": 2.0,
        "2nd": 2.0,
        "spring": 2.0,
        "third": 3.0,
        "3rd": 3.0,
        "summer": 3.0,
        "winter": 4.0,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        f"cannot order Semester value {value!r}; add an explicit mapping "
        "before running the experiment"
    )


def _build_student_sessions(
    frame: pd.DataFrame,
    graph_nodes: set[str],
) -> tuple[dict[str, dict[tuple[int, float], set[str]]], dict[str, object]]:
    columns = list(frame.columns)
    student_col = _find_column(
        columns,
        {"studentid", "studedntid"},
    )
    year_col = _find_column(columns, {"academicyear"})
    semester_col = _find_column(columns, {"semester"})
    course_col = _find_column(columns, {"courseid"})

    sessions: dict[str, dict[tuple[int, float], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    graph_mapped_rows = 0

    semester_values = sorted(
        {str(value) for value in frame[semester_col].dropna().unique()}
    )
    year_values = sorted(
        {str(value) for value in frame[year_col].dropna().unique()}
    )

    for _, row in frame.iterrows():
        student = _norm_id(row[student_col])
        course = _norm_id(row[course_col])
        if not student or course not in graph_nodes:
            continue
        session = (
            _academic_year_key(row[year_col]),
            _semester_key(row[semester_col]),
        )
        sessions[student][session].add(course)
        graph_mapped_rows += 1

    metadata = {
        "student_column": str(student_col),
        "academic_year_column": str(year_col),
        "semester_column": str(semester_col),
        "course_column": str(course_col),
        "raw_semester_values": semester_values,
        "raw_academic_year_values": year_values,
        "graph_mapped_rows": graph_mapped_rows,
    }
    return sessions, metadata


def _rho(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    if not math.isfinite(value):
        return None
    return value


def _cluster_bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    students: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    by_student: dict[str, np.ndarray] = {}
    for student in np.unique(students):
        by_student[str(student)] = np.flatnonzero(students == student)

    student_ids = np.array(sorted(by_student), dtype=object)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []

    for _ in range(replicates):
        sampled = rng.choice(
            student_ids,
            size=student_ids.size,
            replace=True,
        )
        indices = np.concatenate([by_student[str(student)] for student in sampled])
        estimate = _rho(left[indices], right[indices])
        if estimate is not None:
            estimates.append(estimate)

    if not estimates:
        return {
            "replicates_requested": replicates,
            "replicates_defined": 0,
            "seed": seed,
            "ci95_percentile": [None, None],
        }

    values = np.asarray(estimates, dtype=np.float64)
    return {
        "replicates_requested": replicates,
        "replicates_defined": int(values.size),
        "seed": seed,
        "ci95_percentile": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
    }


def _bundle_summary(values: np.ndarray) -> dict[str, object]:
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "distinct_sizes": int(np.unique(values).size),
        "fraction_ge2": float(np.mean(values >= 2)),
    }


def run(
    config_path: Path,
    *,
    s2_path: Path,
    edges_path: Path,
    feasibility_path: Path,
) -> dict[str, object]:
    config = _load_config(config_path)
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_021 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    feasibility = json.loads(
        feasibility_path.read_text(encoding="utf-8")
    )
    if int(feasibility.get("audit_version", 0)) != 3:
        raise RuntimeError(
            "exp_021 requires DORIS audit_version=3"
        )
    if not bool(feasibility.get("final_feasibility_pass", False)):
        raise RuntimeError(
            "DORIS v3 final feasibility gate has not passed"
        )
    manual_review = feasibility.get("manual_review", {})
    if not bool(manual_review.get("review_is_disjoint_from_v2_sample", False)):
        raise RuntimeError(
            "exp_021 requires the disjoint v3 holdout validation"
        )

    graph = _load_graph(edges_path)
    selection = _read_selection_table(s2_path)
    sessions, session_metadata = _build_student_sessions(
        selection,
        set(graph.nodes),
    )

    evaluator = GF2RelativeHomologyEvaluator(
        graph,
        validate_dag=True,
        cache_size=int(config["evaluator_cache_size"]),
    )

    records: list[dict[str, object]] = []
    repeats_removed = 0
    students_with_states: set[str] = set()
    identity_mismatches = 0
    d1_identity_mismatches = 0

    for student, student_sessions in sessions.items():
        history: set[str] = set()
        for session in sorted(student_sessions):
            current = set(student_sessions[session])
            repeated = current.intersection(history)
            repeats_removed += len(repeated)
            candidate = current.difference(history)

            if history and candidate:
                decomposition = evaluator.relative_h1_gain_decomposition(
                    history,
                    candidate,
                )
                if (
                    decomposition.reconstructed_delta_beta1
                    != decomposition.delta_beta1
                ):
                    identity_mismatches += 1
                if (
                    decomposition.delta_rank_d1
                    != decomposition.new_vertex_count
                    - decomposition.delta_beta0
                ):
                    d1_identity_mismatches += 1

                records.append(
                    {
                        "student": student,
                        "year": session[0],
                        "semester_order": session[1],
                        "m": decomposition.new_vertex_count,
                        "q": decomposition.absorbed_relative_edges,
                        "h": decomposition.delta_beta0,
                        "t": decomposition.delta_rank_d2,
                        "d1": decomposition.delta_rank_d1,
                        "g": decomposition.delta_beta1,
                    }
                )
                students_with_states.add(student)

            history.update(current)

    if not records:
        raise RuntimeError("no eligible student-semester states")

    students = np.asarray(
        [str(record["student"]) for record in records],
        dtype=object,
    )
    m = np.asarray([int(record["m"]) for record in records], dtype=np.int64)
    q = np.asarray([int(record["q"]) for record in records], dtype=np.int64)
    h = np.asarray([int(record["h"]) for record in records], dtype=np.int64)
    t = np.asarray([int(record["t"]) for record in records], dtype=np.int64)
    g = np.asarray([int(record["g"]) for record in records], dtype=np.int64)
    abs_g = np.abs(g)

    correlations = {
        "abs_gain_vs_bundle_size": _rho(abs_g, m),
        "gain_vs_bundle_size": _rho(g, m),
        "absorbed_edges_vs_bundle_size": _rho(q, m),
        "d2_rank_drop_vs_bundle_size": _rho(t, m),
        "delta_beta0_vs_bundle_size": _rho(h, m),
        "abs_gain_vs_absorbed_edges": _rho(abs_g, q),
    }

    replicates = int(config["bootstrap_replicates"])
    seed = int(config["bootstrap_seed"])
    bootstrap = {
        "abs_gain_vs_bundle_size": _cluster_bootstrap_ci(
            abs_g,
            m,
            students,
            replicates=replicates,
            seed=seed,
        ),
        "absorbed_edges_vs_bundle_size": _cluster_bootstrap_ci(
            q,
            m,
            students,
            replicates=replicates,
            seed=seed + 1,
        ),
    }

    payload: dict[str, object] = {
        "experiment_id": str(config["experiment_id"]),
        "status": "completed",
        "phase": "external_mechanism_replication",
        "evidence_scope": "target_independent_external",
        "dataset": "DORIS",
        "state_definition": (
            "prior graph-mapped semesters S; current-semester previously "
            "unseen graph-mapped course bundle C"
        ),
        "inputs": {
            "s2_zip": str(s2_path),
            "s2_sha256": _sha256(s2_path),
            "edges": str(edges_path),
            "edges_sha256": _sha256(edges_path),
            "feasibility": str(feasibility_path),
            "feasibility_sha256": _sha256(feasibility_path),
            "manual_precision": feasibility["manual_review"]["precision"],
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "directed_triangles": evaluator.directed_triangle_count,
            "is_dag": nx.is_directed_acyclic_graph(graph),
        },
        "session_metadata": session_metadata,
        "eligible_students": len(students_with_states),
        "eligible_states": len(records),
        "repeated_graph_courses_removed_from_candidate_bundles": repeats_removed,
        "bundle_size": _bundle_summary(m),
        "gain_sign": {
            "positive": int(np.sum(g > 0)),
            "zero": int(np.sum(g == 0)),
            "negative": int(np.sum(g < 0)),
            "positive_fraction": float(np.mean(g > 0)),
            "zero_fraction": float(np.mean(g == 0)),
            "negative_fraction": float(np.mean(g < 0)),
        },
        "correlations": correlations,
        "cluster_bootstrap": bootstrap,
        "exact_checks": {
            "decomposition_identity_mismatches": identity_mismatches,
            "delta_rank_d1_equals_m_minus_h_mismatches": d1_identity_mismatches,
        },
        "selection_performed": False,
        "recommendation_metrics_evaluated": False,
        "target_performance_evaluated": False,
        "corrected_score_evaluated": False,
    }

    if identity_mismatches or d1_identity_mismatches:
        raise RuntimeError(
            "exact relative-H1 decomposition check failed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    primary = correlations["abs_gain_vs_bundle_size"]
    primary_ci = bootstrap["abs_gain_vs_bundle_size"]["ci95_percentile"]
    q_corr = correlations["absorbed_edges_vs_bundle_size"]
    q_ci = bootstrap["absorbed_edges_vs_bundle_size"]["ci95_percentile"]

    lines = [
        "# exp_021 - DORIS v3 Semester-Bundle Relative-H1 Confirmatory Replication",
        "",
        "- Evidence scope: target-independent external structural replication",
        "- Recommendation metrics evaluated: NO",
        "- Target performance evaluated: NO",
        "- Corrected score evaluated: NO",
        "",
        f"- Graph nodes/edges/triangles: {graph.number_of_nodes():,}/"
        f"{graph.number_of_edges():,}/{evaluator.directed_triangle_count:,}",
        f"- Eligible students/states: {len(students_with_states):,}/"
        f"{len(records):,}",
        (
            "- Bundle size min/median/max: "
            f"{int(m.min())}/{float(np.median(m)):.1f}/{int(m.max())}"
        ),
        f"- Fraction of bundles with m>=2: {float(np.mean(m >= 2)):.4f}",
        "",
        "## Primary external estimand",
        "",
        f"- Spearman(|gain|, m): {primary}",
        f"- Student-cluster bootstrap 95% CI: {primary_ci}",
        "",
        "## Secondary mechanism",
        "",
        f"- Spearman(q, m): {q_corr}",
        f"- Student-cluster bootstrap 95% CI: {q_ci}",
        f"- Spearman(gain, m): {correlations['gain_vs_bundle_size']}",
        f"- Spearman(t, m): {correlations['d2_rank_drop_vs_bundle_size']}",
        f"- Spearman(h, m): {correlations['delta_beta0_vs_bundle_size']}",
        f"- Spearman(|gain|, q): {correlations['abs_gain_vs_absorbed_edges']}",
        "",
        "## Exact checks",
        "",
        f"- Decomposition mismatches: {identity_mismatches}",
        f"- Delta rank(d1)=m-h mismatches: {d1_identity_mismatches}",
        "",
        "Interpretation: report this result regardless of magnitude. "
        "It measures external empirical materiality, not theorem validity.",
    ]
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("exp_021_doris_semester_bundle_mechanism")
    print(
        f"Eligible students/states: {len(students_with_states):,}/"
        f"{len(records):,}"
    )
    print(f"Bundle-size summary: {payload['bundle_size']}")
    print(f"Spearman(|gain|,m): {primary} CI={primary_ci}")
    print(f"Spearman(q,m): {q_corr} CI={q_ci}")
    print(f"Exact checks: {payload['exact_checks']}")
    print("Recommendation metrics evaluated: False")
    print("Corrected score evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    args = parser.parse_args()

    run(
        args.config,
        s2_path=args.s2,
        edges_path=args.edges,
        feasibility_path=args.feasibility,
    )


if __name__ == "__main__":
    main()
