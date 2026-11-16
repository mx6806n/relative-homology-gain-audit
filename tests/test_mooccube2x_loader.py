"""Tests for MOOCCube2x concept-space alignment."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from relhom.data_loader import (
    analyze_aligned_cohort,
    build_course_to_concept_map,
    load_mooccube2x_course_ids,
    load_mooccube2x_prerequisite_graph,
    load_mooccube2x_user_transcripts,
    map_transcript_to_concepts,
    normalize_mooccube2x_course_id,
    project_concept_prerequisites_to_courses,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_fixture(root: Path) -> None:
    _write_jsonl(
        root / "entities" / "user.json",
        [
            {
                "id": "U_1",
                "course_order": [100, 200, 999],
                "enroll_time": [
                    "2020-01-01 00:00:00",
                    "2020-02-01 00:00:00",
                    "2020-03-01 00:00:00",
                ],
            }
        ],
    )
    _write_jsonl(
        root / "entities" / "course.json",
        [
            {"id": "C_100", "name": "Course 100"},
            {"id": "C_200", "name": "Course 200"},
            {"id": "C_300", "name": "Course 300"},
            {"id": "C_999", "name": "Unmapped course"},
        ],
    )
    _write_jsonl(
        root / "prerequisites" / "cs.json",
        [
            {"c1": "A", "c2": "B", "ground_truth": 1},
            {"c1": "B", "c2": "C", "ground_truth": 1},
            {"c1": "C", "c2": "A", "ground_truth": 0},
            {"c1": "ignored", "c2": "unknown", "ground_truth": -1},
        ],
    )
    _write_jsonl(
        root / "entities" / "concept.json",
        [
            {"id": "K_A_计算机科学与技术", "name": "A", "context": []},
            {"id": "K_B_计算机科学与技术", "name": "B", "context": []},
            {"id": "K_C_计算机科学与技术", "name": "C", "context": []},
            {"id": "K_A_数学", "name": "A", "context": []},
        ],
    )

    relation = root / "relations" / "concept-course.txt"
    relation.parent.mkdir(parents=True, exist_ok=True)
    relation.write_text(
        "K_A_计算机科学与技术\tC_100\n"
        "K_B_计算机科学与技术\tC_200\n"
        "K_C_计算机科学与技术\tC_300\n"
        "K_A_数学\tC_999\n",
        encoding="utf-8",
    )


def test_course_id_normalization() -> None:
    assert normalize_mooccube2x_course_id(682129) == "C_682129"
    assert normalize_mooccube2x_course_id("C_682129") == "C_682129"


def test_stream_users_and_positive_prerequisites(tmp_path: Path) -> None:
    _build_fixture(tmp_path)

    transcripts = load_mooccube2x_user_transcripts(
        tmp_path,
        user_ids=["U_1"],
    )
    assert transcripts == {
        "U_1": ["C_100", "C_200", "C_999"]
    }

    graph, stats = load_mooccube2x_prerequisite_graph(
        tmp_path,
        domain="cs",
        validate_dag=True,
    )
    assert set(graph.edges) == {("A", "B"), ("B", "C")}
    assert nx.is_directed_acyclic_graph(graph)
    assert stats.total_records == 4
    assert stats.positive_records == 2
    assert stats.ignored_records == 2


def test_course_to_concept_map_uses_dag_node_names(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    graph, _ = load_mooccube2x_prerequisite_graph(
        tmp_path,
        domain="cs",
    )

    mapping = build_course_to_concept_map(
        tmp_path,
        "cs",
        allowed_concepts=graph.nodes,
    )

    assert mapping == {
        "C_100": {"A"},
        "C_200": {"B"},
        "C_300": {"C"},
    }
    assert set().union(*mapping.values()).issubset(graph.nodes)

    concepts = map_transcript_to_concepts(
        ["C_100", "C_200", "C_999"],
        mapping,
    )
    assert concepts == frozenset({"A", "B"})


def test_full_course_catalog_includes_unmapped_courses(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    assert load_mooccube2x_course_ids(tmp_path) == frozenset(
        {"C_100", "C_200", "C_300", "C_999"}
    )


def test_course_projection_is_explicitly_derived(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    concept_graph, _ = load_mooccube2x_prerequisite_graph(
        tmp_path,
        domain="cs",
    )

    course_graph, stats = project_concept_prerequisites_to_courses(
        concept_graph,
        tmp_path,
        domain="cs",
        validate_dag=True,
    )

    assert set(course_graph.edges) == {
        ("C_100", "C_200"),
        ("C_200", "C_300"),
    }
    assert course_graph.graph["edge_source"] == "derived_concept_projection"
    assert stats.projected_edges == 2


def test_aligned_cohort_analysis_and_export(tmp_path: Path) -> None:
    _build_fixture(tmp_path)
    concept_graph, _ = load_mooccube2x_prerequisite_graph(
        tmp_path,
        domain="cs",
    )
    mapping = build_course_to_concept_map(
        tmp_path,
        "cs",
        allowed_concepts=concept_graph.nodes,
    )

    export_dir = tmp_path / "processed"
    stats = analyze_aligned_cohort(
        tmp_path,
        course_to_concept_map=mapping,
        concept_graph=concept_graph,
        export_dir=export_dir,
        min_export_courses=2,
    )

    assert stats["total_users"] == 1
    assert stats["mapped_enrollment_events"] == 2
    assert stats["aligned_next_course_transitions"] == 1
    assert stats["users_by_min_aligned_courses"]["1"] == 1
    assert stats["users_by_min_aligned_courses"]["2"] == 1
    assert stats["users_by_min_aligned_courses"]["3"] == 0
    assert stats["eligible_users_by_min_history"]["2"] == 0
    assert stats["dag_concepts_reached_by_users"] == 2
    assert stats["exported_users"] == 1

    sequence = json.loads(
        (export_dir / "sequences.jsonl").read_text(
            encoding="utf-8"
        ).strip()
    )
    assert sequence == {
        "user_id": "U_1",
        "courses": ["C_100", "C_200"],
    }

    for filename in (
        "course_vocab.json",
        "concept_vocab.json",
        "course_to_concepts.json",
        "prerequisite_edges.csv",
        "cohort_stats.json",
    ):
        assert (export_dir / filename).exists()
