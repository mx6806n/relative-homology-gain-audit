"""Dataset loading and concept-space alignment for MOOCCube2x.

The central design rule is that recommendation items and topological nodes live
in different spaces:

* users interact with courses such as C_682129;
* the expert-labelled prerequisite DAG is defined over prerequisite concepts.

MOOCCube2x prerequisite files use concept names in c1/c2 while
relations/concept-course.txt uses entity concept IDs. The course-to-concept
map therefore resolves entity IDs back to the exact concept names used as DAG
nodes. This prevents incompatible identifiers from entering the Ghost Vertex
construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv


MOOCCUBE2X_DOMAIN_FILES = {
    "cs": "cs.json",
    "math": "math.json",
    "psy": "psy.json",
}

MOOCCUBE2X_DOMAIN_FIELDS = {
    "cs": "计算机科学与技术",
    "math": "数学",
    "psy": "心理学",
}


@dataclass(frozen=True)
class DummyDataset:
    """Container used by the synthetic demonstration."""

    curriculum_graph: nx.DiGraph
    interactions: pd.DataFrame
    transcripts: dict[str, list[Hashable]]
    ground_truth_next: dict[str, set[Hashable]]
    ground_truth_capstones: set[Hashable]


@dataclass(frozen=True)
class PrerequisiteLoadStats:
    """Diagnostics for one prerequisite JSONL file."""

    total_records: int
    positive_records: int
    ignored_records: int
    malformed_records: int


@dataclass(frozen=True)
class CourseProjectionStats:
    """Diagnostics for the legacy derived course-graph projection."""

    prerequisite_concepts: int
    matched_concept_ids: int
    linked_courses: int
    projected_edges: int
    skipped_large_expansions: int


def build_curriculum_graph(
    edges: Iterable[tuple[Hashable, Hashable]],
    *,
    courses: Iterable[Hashable] | None = None,
    validate_dag: bool = True,
) -> nx.DiGraph:
    """Build a directed prerequisite graph with validation."""
    graph = nx.DiGraph()
    if courses is not None:
        graph.add_nodes_from(courses)
    graph.add_edges_from(edges)

    if nx.number_of_selfloops(graph):
        raise ValueError("curriculum prerequisite graph cannot contain self-loops")
    if validate_dag and not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        raise ValueError(f"curriculum contains a prerequisite cycle: {cycle}")
    return graph


def generate_dummy_dataset(seed: int = 7) -> DummyDataset:
    """Generate the compact higher-education example used by unit tests."""
    courses = ["W", "X", "Y", "C", "CAP", "ELECTIVE_A", "ELECTIVE_B", "Z"]
    edges = [
        ("W", "X"),
        ("W", "Y"),
        ("X", "C"),
        ("Y", "C"),
        ("C", "CAP"),
        ("Z", "ELECTIVE_B"),
    ]
    graph = build_curriculum_graph(edges, courses=courses)

    transcripts: dict[str, list[Hashable]] = {
        "student_001": ["W", "X", "Y"],
        "student_002": ["W", "X"],
        "student_003": ["Z", "ELECTIVE_A"],
    }
    ground_truth_next: dict[str, set[Hashable]] = {
        "student_001": {"C"},
        "student_002": {"Y"},
        "student_003": {"ELECTIVE_B"},
    }

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for student_id, transcript in transcripts.items():
        base_time = int(rng.integers(1_700_000_000, 1_710_000_000))
        for offset, course in enumerate(transcript):
            rows.append(
                {
                    "student_id": student_id,
                    "course_id": course,
                    "timestamp": base_time + 86_400 * offset,
                    "completed": True,
                }
            )

    interactions = (
        pd.DataFrame(rows)
        .sort_values(["student_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    return DummyDataset(
        curriculum_graph=graph,
        interactions=interactions,
        transcripts=transcripts,
        ground_truth_next=ground_truth_next,
        ground_truth_capstones={"CAP"},
    )


def resolve_mooccube2x_root(
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve the MOOCCube2x directory from an argument or local .env."""
    load_dotenv()
    raw = (
        str(root)
        if root is not None
        else os.getenv("MOOCCUBE2X_ROOT") or os.getenv("DATA_ROOT")
    )
    if not raw:
        raise FileNotFoundError(
            "Set MOOCCUBE2X_ROOT in .env or pass root=... explicitly."
        )

    candidate = Path(raw).expanduser()
    if not (candidate / "entities" / "user.json").exists():
        nested = candidate / "mooccube2x"
        if (nested / "entities" / "user.json").exists():
            candidate = nested

    required = [
        candidate / "entities" / "user.json",
        candidate / "entities" / "course.json",
        candidate / "entities" / "concept.json",
        candidate / "relations" / "concept-course.txt",
        candidate / "prerequisites" / "cs.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MOOCCube2x root is incomplete or incorrect. "
            f"Missing: {missing}"
        )
    return candidate.resolve()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    """Stream newline-delimited JSON without loading the whole file."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"expected a JSON object in {path} at line {line_number}"
                )
            yield line_number, payload


def normalize_mooccube2x_course_id(value: object) -> str:
    """Normalize numeric course IDs to the canonical C_ form."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty MOOCCube2x course id")
    return text if text.startswith("C_") else f"C_{text}"


def normalize_mooccube2x_user_id(value: object) -> str:
    """Normalize numeric user IDs to the canonical U_ form."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty MOOCCube2x user id")
    return text if text.startswith("U_") else f"U_{text}"


def iter_mooccube2x_user_transcripts(
    root: str | os.PathLike[str] | None = None,
    *,
    min_courses: int = 1,
) -> Iterator[tuple[str, list[str]]]:
    """Yield user course histories one record at a time."""
    if min_courses < 1:
        raise ValueError("min_courses must be >= 1")

    dataset_root = resolve_mooccube2x_root(root)
    path = dataset_root / "entities" / "user.json"

    for line_number, record in _iter_jsonl(path):
        raw_user_id = record.get("id")
        raw_order = record.get("course_order")
        if raw_user_id is None or not isinstance(raw_order, list):
            continue

        courses = [normalize_mooccube2x_course_id(item) for item in raw_order]
        if len(courses) < min_courses:
            continue

        enroll_time = record.get("enroll_time")
        if isinstance(enroll_time, list) and len(enroll_time) not in (0, len(courses)):
            user_id = normalize_mooccube2x_user_id(raw_user_id)
            raise ValueError(
                f"user {user_id} at line {line_number} has "
                f"{len(courses)} course_order entries but "
                f"{len(enroll_time)} enroll_time entries"
            )

        yield normalize_mooccube2x_user_id(raw_user_id), courses


def load_mooccube2x_user_transcripts(
    root: str | os.PathLike[str] | None = None,
    *,
    user_ids: Iterable[str] | None = None,
    max_users: int | None = 1000,
    min_courses: int = 1,
) -> dict[str, list[str]]:
    """Load selected course histories while retaining streaming behaviour."""
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be >= 1 or None")

    wanted = (
        {normalize_mooccube2x_user_id(user_id) for user_id in user_ids}
        if user_ids is not None
        else None
    )
    transcripts: dict[str, list[str]] = {}

    for user_id, courses in iter_mooccube2x_user_transcripts(
        root,
        min_courses=min_courses,
    ):
        if wanted is not None and user_id not in wanted:
            continue

        transcripts[user_id] = courses
        if wanted is not None and wanted.issubset(transcripts):
            break
        if wanted is None and max_users is not None and len(transcripts) >= max_users:
            break

    if wanted is not None:
        missing = wanted.difference(transcripts)
        if missing:
            raise KeyError(
                "requested users were not found with the requested constraints: "
                f"{sorted(missing)}"
            )
    return transcripts


def load_mooccube2x_course_ids(
    root: str | os.PathLike[str] | None = None,
) -> frozenset[str]:
    """Return the full course catalogue from entities/course.json."""
    dataset_root = resolve_mooccube2x_root(root)
    courses: set[str] = set()

    for _, record in _iter_jsonl(dataset_root / "entities" / "course.json"):
        course_id = record.get("id")
        if course_id is not None:
            courses.add(normalize_mooccube2x_course_id(course_id))

    return frozenset(courses)


def load_mooccube2x_prerequisite_graph(
    root: str | os.PathLike[str] | None = None,
    *,
    domain: str = "cs",
    positive_label: int = 1,
    validate_dag: bool = False,
) -> tuple[nx.DiGraph, PrerequisiteLoadStats]:
    """Load the ground-truth concept prerequisite graph.

    Only ground_truth == positive_label is retained. Prediction columns are
    intentionally ignored. Prerequisite concept names are the canonical
    topological node identifiers used throughout the codebase.
    """
    domain = domain.lower()
    if domain not in MOOCCUBE2X_DOMAIN_FILES:
        raise ValueError(
            f"domain must be one of {sorted(MOOCCUBE2X_DOMAIN_FILES)}"
        )

    dataset_root = resolve_mooccube2x_root(root)
    path = dataset_root / "prerequisites" / MOOCCUBE2X_DOMAIN_FILES[domain]
    graph = nx.DiGraph(
        dataset="MOOCCube2x",
        domain=domain,
        granularity="concept",
        concept_identifier="name",
        edge_source="ground_truth",
    )

    total = positive = ignored = malformed = 0
    for _, record in _iter_jsonl(path):
        total += 1
        c1 = record.get("c1")
        c2 = record.get("c2")
        label = record.get("ground_truth")

        if not isinstance(c1, str) or not isinstance(c2, str) or label is None:
            malformed += 1
            continue
        try:
            is_positive = int(label) == positive_label
        except (TypeError, ValueError):
            malformed += 1
            continue
        if not is_positive:
            ignored += 1
            continue
        if c1 == c2:
            malformed += 1
            continue

        graph.add_edge(
            c1,
            c2,
            ground_truth=positive_label,
            source_file=MOOCCUBE2X_DOMAIN_FILES[domain],
        )
        positive += 1

    if validate_dag and not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        raise ValueError(
            "positive prerequisite labels do not form a DAG under c1->c2. "
            f"Example cycle: {cycle[:10]}"
        )

    return graph, PrerequisiteLoadStats(
        total_records=total,
        positive_records=positive,
        ignored_records=ignored,
        malformed_records=malformed,
    )


def _load_domain_concept_ids_by_name(
    dataset_root: Path,
    *,
    concept_names: set[str],
    domain: str,
) -> dict[str, str]:
    """Map entity concept IDs to prerequisite-graph node names."""
    field = MOOCCUBE2X_DOMAIN_FIELDS[domain]
    suffix = f"_{field}"
    id_to_name: dict[str, str] = {}

    for _, record in _iter_jsonl(dataset_root / "entities" / "concept.json"):
        concept_id = record.get("id")
        concept_name = record.get("name")
        if (
            isinstance(concept_id, str)
            and isinstance(concept_name, str)
            and concept_name in concept_names
            and concept_id.endswith(suffix)
        ):
            id_to_name[concept_id] = concept_name

    return id_to_name


def build_course_to_concept_map(
    dataset_root: str | os.PathLike[str] | None = None,
    domain: str = "cs",
    *,
    allowed_concepts: Iterable[str] | None = None,
) -> dict[str, set[str]]:
    """Map each course to concept nodes from the native prerequisite DAG.

    The relation file uses entity concept IDs, while the labelled prerequisite
    graph uses concept names. This function joins through entities/concept.json
    and returns the graph node names, not raw entity IDs.

    Passing allowed_concepts=concept_graph.nodes avoids rereading the large
    prerequisite file when the graph is already available.
    """
    domain = domain.lower()
    if domain not in MOOCCUBE2X_DOMAIN_FILES:
        raise ValueError(
            f"domain must be one of {sorted(MOOCCUBE2X_DOMAIN_FILES)}"
        )

    root = resolve_mooccube2x_root(dataset_root)
    if allowed_concepts is None:
        prerequisite_graph, _ = load_mooccube2x_prerequisite_graph(
            root,
            domain=domain,
        )
        concept_names = set(prerequisite_graph.nodes)
    else:
        concept_names = set(allowed_concepts)

    id_to_name = _load_domain_concept_ids_by_name(
        root,
        concept_names=concept_names,
        domain=domain,
    )

    mapping: dict[str, set[str]] = defaultdict(set)
    relation_path = root / "relations" / "concept-course.txt"
    with relation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                continue

            concept_id, raw_course_id = parts
            concept_name = id_to_name.get(concept_id)
            if concept_name is None:
                continue

            try:
                course_id = normalize_mooccube2x_course_id(raw_course_id)
            except ValueError as exc:
                raise ValueError(
                    f"invalid course id in {relation_path} at line {line_number}"
                ) from exc
            mapping[course_id].add(concept_name)

    return {course: set(concepts) for course, concepts in mapping.items()}


def map_transcript_to_concepts(
    transcript_courses: Iterable[str],
    course_to_concept_map: dict[str, set[str]],
) -> frozenset[str]:
    """Return the union of mapped prerequisite concepts for a course history."""
    concepts: set[str] = set()
    for course in transcript_courses:
        concepts.update(course_to_concept_map.get(course, ()))
    return frozenset(concepts)


def analyze_aligned_cohort(
    root: str | os.PathLike[str] | None,
    *,
    course_to_concept_map: dict[str, set[str]],
    concept_graph: nx.DiGraph,
    export_dir: str | os.PathLike[str] | None = None,
    min_export_courses: int = 3,
) -> dict[str, object]:
    """Analyze the topology-aligned sequential cohort in one streaming pass.

    A user's aligned sequence is the original course order restricted to
    courses with at least one concept in the native prerequisite DAG. No
    artificial course-prerequisite graph is constructed.

    If export_dir is provided, users with at least min_export_courses aligned
    course events are written to sequences.jsonl. Deterministic vocabularies,
    the course-to-concept mapping, prerequisite edges, and cohort statistics
    are written beside the sequence file.
    """
    if min_export_courses < 1:
        raise ValueError("min_export_courses must be >= 1")

    dataset_root = resolve_mooccube2x_root(root)
    mapped_courses = {
        course
        for course, concepts in course_to_concept_map.items()
        if concepts
    }

    thresholds = (1, 2, 3, 5, 10)
    threshold_counts = {threshold: 0 for threshold in thresholds}
    history_thresholds = (2, 3, 5)
    eligible_history_counts = {
        history: 0 for history in history_thresholds
    }

    aligned_lengths: list[int] = []
    course_frequency: Counter[str] = Counter()
    reached_concepts: set[str] = set()

    total_users = 0
    total_enrollment_events = 0
    mapped_enrollment_events = 0
    aligned_transitions = 0
    terminal_target_users = 0
    mapped_terminal_targets = 0
    exported_users = 0

    export_path: Path | None = None
    temporary_sequences_path: Path | None = None
    sequence_handle = None

    if export_dir is not None:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        temporary_sequences_path = export_path / "sequences.jsonl.tmp"
        sequence_handle = temporary_sequences_path.open(
            "w",
            encoding="utf-8",
        )

    try:
        for user_id, transcript in iter_mooccube2x_user_transcripts(
            dataset_root,
            min_courses=1,
        ):
            total_users += 1
            total_enrollment_events += len(transcript)

            if len(transcript) >= 2:
                terminal_target_users += 1
                if transcript[-1] in mapped_courses:
                    mapped_terminal_targets += 1

            aligned_sequence = [
                course
                for course in transcript
                if course in mapped_courses
            ]
            aligned_length = len(aligned_sequence)
            if aligned_length == 0:
                continue

            aligned_lengths.append(aligned_length)
            mapped_enrollment_events += aligned_length
            aligned_transitions += max(aligned_length - 1, 0)
            course_frequency.update(aligned_sequence)

            for course in set(aligned_sequence):
                reached_concepts.update(
                    course_to_concept_map[course]
                )

            for threshold in thresholds:
                if aligned_length >= threshold:
                    threshold_counts[threshold] += 1

            for history in history_thresholds:
                if aligned_length >= history + 1:
                    eligible_history_counts[history] += 1

            if (
                sequence_handle is not None
                and aligned_length >= min_export_courses
            ):
                sequence_handle.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "courses": aligned_sequence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                exported_users += 1
    finally:
        if sequence_handle is not None:
            sequence_handle.close()

    if aligned_lengths:
        values = np.percentile(
            np.asarray(aligned_lengths, dtype=float),
            [25, 50, 75, 90, 95],
        )
        percentiles = {
            "p25": float(values[0]),
            "p50": float(values[1]),
            "p75": float(values[2]),
            "p90": float(values[3]),
            "p95": float(values[4]),
        }
    else:
        percentiles = {
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    graph_nodes = set(concept_graph.nodes)
    mapped_concepts = {
        concept
        for concepts in course_to_concept_map.values()
        for concept in concepts
    }.intersection(graph_nodes)

    terminal_target_coverage = (
        100.0 * mapped_terminal_targets / terminal_target_users
        if terminal_target_users
        else 0.0
    )
    reached_concept_coverage = (
        100.0 * len(reached_concepts) / len(graph_nodes)
        if graph_nodes
        else 0.0
    )

    stats: dict[str, object] = {
        "total_users": total_users,
        "total_enrollment_events": total_enrollment_events,
        "users_with_aligned_courses": threshold_counts[1],
        "users_by_min_aligned_courses": {
            str(key): value
            for key, value in threshold_counts.items()
        },
        "aligned_sequence_length_percentiles": percentiles,
        "eligible_users_by_min_history": {
            str(key): value
            for key, value in eligible_history_counts.items()
        },
        "mapped_enrollment_events": mapped_enrollment_events,
        "aligned_next_course_transitions": aligned_transitions,
        "unique_mapped_courses_used": len(course_frequency),
        "mapped_courses_available": len(mapped_courses),
        "dag_concepts": len(graph_nodes),
        "dag_concepts_mapped_by_courses": len(mapped_concepts),
        "dag_concepts_reached_by_users": len(reached_concepts),
        "dag_concept_user_coverage_percent": reached_concept_coverage,
        "terminal_target_users": terminal_target_users,
        "mapped_terminal_targets": mapped_terminal_targets,
        "terminal_target_coverage_percent": terminal_target_coverage,
        "top_20_mapped_courses": [
            {"course": course, "count": count}
            for course, count in course_frequency.most_common(20)
        ],
        "export_min_aligned_courses": (
            min_export_courses if export_path is not None else None
        ),
        "exported_users": exported_users,
    }

    if export_path is not None:
        assert temporary_sequences_path is not None
        temporary_sequences_path.replace(
            export_path / "sequences.jsonl"
        )

        course_vocab = {
            course: index
            for index, course in enumerate(sorted(mapped_courses))
        }
        concept_vocab = {
            concept: index
            for index, concept in enumerate(
                sorted(graph_nodes, key=str)
            )
        }
        serializable_map = {
            course: sorted(concepts, key=str)
            for course, concepts in sorted(
                course_to_concept_map.items(),
                key=lambda item: str(item[0]),
            )
            if concepts
        }

        (export_path / "course_vocab.json").write_text(
            json.dumps(
                course_vocab,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (export_path / "concept_vocab.json").write_text(
            json.dumps(
                concept_vocab,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (export_path / "course_to_concepts.json").write_text(
            json.dumps(
                serializable_map,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with (export_path / "prerequisite_edges.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "target"])
            for source, target in concept_graph.edges:
                writer.writerow([str(source), str(target)])

        (export_path / "cohort_stats.json").write_text(
            json.dumps(
                stats,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return stats

def project_concept_prerequisites_to_courses(
    concept_graph: nx.DiGraph,
    root: str | os.PathLike[str] | None = None,
    *,
    domain: str = "cs",
    min_support: int = 1,
    max_pair_expansion: int = 10_000,
    validate_dag: bool = False,
) -> tuple[nx.DiGraph, CourseProjectionStats]:
    """Legacy diagnostic: derive course edges from concept prerequisites.

    This projection is retained only for analysis. It is not the graph used by
    the hybrid recommender after the concept-space pivot.
    """
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    if max_pair_expansion < 1:
        raise ValueError("max_pair_expansion must be >= 1")

    course_to_concepts = build_course_to_concept_map(
        root,
        domain,
        allowed_concepts=concept_graph.nodes,
    )

    courses_by_concept: dict[str, set[str]] = defaultdict(set)
    for course, concepts in course_to_concepts.items():
        for concept in concepts:
            courses_by_concept[concept].add(course)

    support: Counter[tuple[str, str]] = Counter()
    skipped_large_expansions = 0
    for c1, c2 in concept_graph.edges:
        sources = courses_by_concept.get(c1, set())
        targets = courses_by_concept.get(c2, set())
        if not sources or not targets:
            continue
        if len(sources) * len(targets) > max_pair_expansion:
            skipped_large_expansions += 1
            continue
        for source_course in sources:
            for target_course in targets:
                if source_course != target_course:
                    support[(source_course, target_course)] += 1

    course_graph = nx.DiGraph(
        dataset="MOOCCube2x",
        domain=domain,
        granularity="course",
        edge_source="derived_concept_projection",
    )
    for (source, target), count in support.items():
        if count >= min_support:
            course_graph.add_edge(
                source,
                target,
                support=count,
                provenance="concept_prerequisite_projection",
            )

    linked_courses = set(course_to_concepts)
    course_graph.add_nodes_from(linked_courses)

    if validate_dag and not nx.is_directed_acyclic_graph(course_graph):
        cycle = nx.find_cycle(course_graph, orientation="original")
        raise ValueError(
            "derived course projection is cyclic and cannot be treated as a "
            f"strict prerequisite DAG. Example cycle: {cycle[:10]}"
        )

    matched_concepts = {
        concept
        for concepts in course_to_concepts.values()
        for concept in concepts
    }
    stats = CourseProjectionStats(
        prerequisite_concepts=concept_graph.number_of_nodes(),
        matched_concept_ids=len(matched_concepts),
        linked_courses=len(linked_courses),
        projected_edges=course_graph.number_of_edges(),
        skipped_large_expansions=skipped_large_expansions,
    )
    return course_graph, stats
