"""Prepare the MOOCCubeX prerequisite graph and the aligned course histories.

Run with ``--export-aligned`` to write the files that the experiments read from
``data/processed/mooccube2x_<domain>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.data_loader import (  # noqa: E402
    analyze_aligned_cohort,
    build_course_to_concept_map,
    iter_mooccube2x_user_transcripts,
    load_mooccube2x_course_ids,
    load_mooccube2x_prerequisite_graph,
    load_mooccube2x_user_transcripts,
    map_transcript_to_concepts,
    project_concept_prerequisites_to_courses,
    resolve_mooccube2x_root,
)


def print_coverage_report(
    root: object,
    *,
    domain: str,
    concept_graph: nx.DiGraph,
    student_id: str | None,
) -> None:
    """Stream all users and report course-to-DAG-concept coverage."""
    course_to_concepts = build_course_to_concept_map(
        root,
        domain,
        allowed_concepts=concept_graph.nodes,
    )
    course_catalog = load_mooccube2x_course_ids(root)

    mapped_courses_per_user: list[int] = []
    concepts_per_user: list[int] = []
    total_users = 0
    users_with_mapped_concept = 0

    selected_id = None
    if student_id:
        selected_id = (
            student_id
            if student_id.startswith("U_")
            else f"U_{student_id}"
        )
    selected_transcript: list[str] | None = None

    for user_id, transcript in iter_mooccube2x_user_transcripts(
        root,
        min_courses=1,
    ):
        total_users += 1

        unique_courses = set(transcript)
        mapped_courses = {
            course
            for course in unique_courses
            if course_to_concepts.get(course)
        }
        transcript_concepts = map_transcript_to_concepts(
            transcript,
            course_to_concepts,
        )

        mapped_courses_per_user.append(len(mapped_courses))
        concepts_per_user.append(len(transcript_concepts))
        if transcript_concepts:
            users_with_mapped_concept += 1

        if selected_id is not None and user_id == selected_id:
            selected_transcript = transcript

    if total_users == 0:
        raise ValueError("no users with course histories were found")

    mapped_catalog_courses = set(course_to_concepts).intersection(
        course_catalog
    )
    user_coverage = 100.0 * users_with_mapped_concept / total_users
    candidate_coverage = (
        100.0 * len(mapped_catalog_courses) / len(course_catalog)
        if course_catalog
        else 0.0
    )

    print("\nMOOCCube2x concept-space coverage report")
    print("----------------------------------------")
    print(f"Domain: {domain}")
    print(f"Prerequisite DAG concepts: {concept_graph.number_of_nodes():,}")
    print(f"Total users with >=1 enrolled course: {total_users:,}")
    print(
        "Median mapped courses per user: "
        f"{median(mapped_courses_per_user):.1f}"
    )
    print(
        "Median prerequisite concepts covered per transcript: "
        f"{median(concepts_per_user):.1f}"
    )
    print(
        "Users with >=1 mapped prerequisite concept: "
        f"{users_with_mapped_concept:,}/{total_users:,} "
        f"({user_coverage:.2f}%)"
    )
    print(
        "Candidate-course concept coverage: "
        f"{len(mapped_catalog_courses):,}/{len(course_catalog):,} "
        f"catalogue courses ({candidate_coverage:.2f}%)"
    )

    if selected_id is None:
        return
    if selected_transcript is None:
        raise KeyError(f"selected user was not found: {selected_id}")

    selected_concepts = map_transcript_to_concepts(
        selected_transcript,
        course_to_concepts,
    )
    selected_mapped_courses = [
        course
        for course in selected_transcript
        if course_to_concepts.get(course)
    ]

    print(f"\nSelected-user diagnostic: {selected_id}")
    print(f"  enrolled courses: {len(selected_transcript)}")
    print(
        "  courses mapped into prerequisite DAG: "
        f"{len(set(selected_mapped_courses))}"
    )
    print(f"  unique prerequisite concepts: {len(selected_concepts)}")
    print("  enrolled course sequence: " + ", ".join(selected_transcript))

    if selected_mapped_courses:
        print("  mapped course -> DAG concepts:")
        for course in dict.fromkeys(selected_mapped_courses):
            concepts = sorted(course_to_concepts[course])
            print(f"    {course}: {', '.join(concepts)}")
    else:
        print("  mapped course -> DAG concepts: none")


def print_aligned_cohort_report(
    root: object,
    *,
    domain: str,
    concept_graph: nx.DiGraph,
    export_aligned: bool,
    export_dir: str | None,
    min_aligned_courses: int,
) -> None:
    """Report the sequential cohort after restricting histories to mapped courses."""
    course_to_concepts = build_course_to_concept_map(
        root,
        domain,
        allowed_concepts=concept_graph.nodes,
    )

    output_dir = None
    if export_aligned:
        output_dir = (
            Path(export_dir)
            if export_dir is not None
            else Path("data") / "processed" / f"mooccube2x_{domain}"
        )

    stats = analyze_aligned_cohort(
        root,
        course_to_concept_map=course_to_concepts,
        concept_graph=concept_graph,
        export_dir=output_dir,
        min_export_courses=min_aligned_courses,
    )

    counts = stats["users_by_min_aligned_courses"]
    percentiles = stats["aligned_sequence_length_percentiles"]
    eligible = stats["eligible_users_by_min_history"]

    print("\nMOOCCube2x topology-aligned cohort report")
    print("-----------------------------------------")
    print(
        "Users by minimum aligned course events: "
        f">=1: {counts['1']:,}, "
        f">=2: {counts['2']:,}, "
        f">=3: {counts['3']:,}, "
        f">=5: {counts['5']:,}, "
        f">=10: {counts['10']:,}"
    )
    print(
        "Aligned sequence-length percentiles "
        "(among users with >=1 aligned course):"
    )
    print(
        f"  p25={percentiles['p25']:.1f}, "
        f"p50={percentiles['p50']:.1f}, "
        f"p75={percentiles['p75']:.1f}, "
        f"p90={percentiles['p90']:.1f}, "
        f"p95={percentiles['p95']:.1f}"
    )
    print("Eligible users for next aligned-course prediction:")
    print(
        f"  history >=2: {eligible['2']:,} "
        f"(sequence length >=3)"
    )
    print(
        f"  history >=3: {eligible['3']:,} "
        f"(sequence length >=4)"
    )
    print(
        f"  history >=5: {eligible['5']:,} "
        f"(sequence length >=6)"
    )
    print(
        "Mapped enrollment events: "
        f"{stats['mapped_enrollment_events']:,}"
    )
    print(
        "Aligned next-course transitions: "
        f"{stats['aligned_next_course_transitions']:,}"
    )
    print(
        "Unique mapped courses used: "
        f"{stats['unique_mapped_courses_used']:,}/"
        f"{stats['mapped_courses_available']:,}"
    )
    print(
        "DAG concepts mapped by the aligned course catalogue: "
        f"{stats['dag_concepts_mapped_by_courses']:,}/"
        f"{stats['dag_concepts']:,}"
    )
    print(
        "DAG concepts reached by aligned users: "
        f"{stats['dag_concepts_reached_by_users']:,}/"
        f"{stats['dag_concepts']:,} "
        f"({stats['dag_concept_user_coverage_percent']:.2f}%)"
    )
    print(
        "Full-sequence terminal targets that are topology-mapped: "
        f"{stats['mapped_terminal_targets']:,}/"
        f"{stats['terminal_target_users']:,} "
        f"({stats['terminal_target_coverage_percent']:.2f}%)"
    )

    print("\nTop 20 mapped courses by enrollment frequency:")
    for rank, item in enumerate(
        stats["top_20_mapped_courses"],
        start=1,
    ):
        print(
            f"  {rank:2d}. {item['course']}: "
            f"{item['count']:,}"
        )

    if output_dir is not None:
        print(
            "\nAligned dataset exported to: "
            f"{output_dir.resolve()}"
        )
        print(
            "Exported users with >= "
            f"{min_aligned_courses} aligned courses: "
            f"{stats['exported_users']:,}"
        )

def run_mooccube2x(args: argparse.Namespace) -> None:
    """Run MOOCCube2x diagnostics using the native concept DAG."""
    root = resolve_mooccube2x_root(args.data_root)
    concept_graph, prereq_stats = load_mooccube2x_prerequisite_graph(
        root,
        domain=args.domain,
        validate_dag=False,
    )

    print(f"MOOCCube2x root: {root}")
    print(f"Domain: {args.domain}")
    print(
        "Concept prerequisite graph: "
        f"{concept_graph.number_of_nodes():,} nodes, "
        f"{concept_graph.number_of_edges():,} edges"
    )
    print(
        "Ground-truth records: "
        f"{prereq_stats.positive_records:,} positive / "
        f"{prereq_stats.total_records:,} total "
        f"({prereq_stats.malformed_records:,} malformed)"
    )
    print(
        "Concept graph is DAG: "
        f"{nx.is_directed_acyclic_graph(concept_graph)}"
    )

    if args.coverage_report:
        print_coverage_report(
            root,
            domain=args.domain,
            concept_graph=concept_graph,
            student_id=args.student_id,
        )
        return

    if args.aligned_cohort_report or args.export_aligned:
        print_aligned_cohort_report(
            root,
            domain=args.domain,
            concept_graph=concept_graph,
            export_aligned=args.export_aligned,
            export_dir=args.export_dir,
            min_aligned_courses=args.min_aligned_courses,
        )
        return

    requested_users = [args.student_id] if args.student_id else None
    transcripts = load_mooccube2x_user_transcripts(
        root,
        user_ids=requested_users,
        max_users=args.max_users,
        min_courses=args.min_courses,
    )

    print(f"Loaded user transcripts: {len(transcripts):,}")
    for user_id, courses in list(transcripts.items())[:5]:
        preview = ", ".join(courses[:8])
        suffix = " ..." if len(courses) > 8 else ""
        print(
            f"  {user_id}: {len(courses)} enrolled courses -> "
            f"{preview}{suffix}"
        )

    if not args.project_to_courses:
        print(
            "\nUse --coverage-report or --aligned-cohort-report for details."
        )
        return

    course_graph, projection_stats = project_concept_prerequisites_to_courses(
        concept_graph,
        root,
        domain=args.domain,
        min_support=args.min_support,
        validate_dag=False,
    )

    print("\nConcept prerequisites projected onto courses:")
    print(f"  linked courses: {projection_stats.linked_courses:,}")
    print(f"  course edges: {projection_stats.projected_edges:,}")
    print(
        "  projected course graph is DAG: "
        f"{nx.is_directed_acyclic_graph(course_graph)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MOOCCubeX data for the experiments."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override MOOCCUBE2X_ROOT or DATA_ROOT.",
    )
    parser.add_argument(
        "--domain",
        choices=("cs", "math", "psy"),
        default="cs",
    )
    parser.add_argument(
        "--student-id",
        default=None,
        help="Optional MOOCCube2x user such as U_24.",
    )
    parser.add_argument("--max-users", type=int, default=5)
    parser.add_argument("--min-courses", type=int, default=2)
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help=(
            "Stream all users and report course-to-prerequisite-concept "
            "coverage. Combine with --student-id for an exact user diagnostic."
        ),
    )
    parser.add_argument(
        "--aligned-cohort-report",
        action="store_true",
        help=(
            "Analyze user sequences after restricting them to courses mapped "
            "into the native prerequisite concept DAG."
        ),
    )
    parser.add_argument(
        "--export-aligned",
        action="store_true",
        help=(
            "Export the aligned sequential dataset and metadata. This also "
            "runs the aligned cohort report."
        ),
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help=(
            "Optional aligned-data output directory. Default: "
            "data/processed/mooccube2x_<domain>."
        ),
    )
    parser.add_argument(
        "--min-aligned-courses",
        type=int,
        default=3,
        help=(
            "Minimum aligned course events required for an exported sequence "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--project-to-courses",
        action="store_true",
        help="Project concept prerequisites onto courses and report whether the result is acyclic.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="Minimum concept-pair support for a projected course edge.",
    )
    return parser


def main() -> None:
    run_mooccube2x(build_parser().parse_args())


if __name__ == "__main__":
    main()
