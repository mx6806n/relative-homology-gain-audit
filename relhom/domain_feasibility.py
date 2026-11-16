"""Deterministic feasibility selection for untouched MOOCCube2x domains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def domain_is_feasible(
    row: Mapping[str, object],
    *,
    min_users_ge5: int,
    min_unique_courses: int,
    min_concept_coverage_percent: float,
) -> bool:
    """Check predeclared structural/sample-size feasibility only."""
    return bool(
        row["graph_is_dag"]
        and int(row["users_ge5"]) >= min_users_ge5
        and int(row["unique_mapped_courses_used"]) >= min_unique_courses
        and float(row["dag_concept_user_coverage_percent"])
        >= min_concept_coverage_percent
    )


def select_feasible_domain(
    rows: Sequence[Mapping[str, object]],
    *,
    min_users_ge5: int,
    min_unique_courses: int,
    min_concept_coverage_percent: float,
) -> str | None:
    """Select the most statistically stable feasible domain.

    Order is predeclared: larger >=5 cohort, then more used courses, then
    higher DAG-concept coverage, then alphabetical domain name.
    """
    feasible = [
        row
        for row in rows
        if domain_is_feasible(
            row,
            min_users_ge5=min_users_ge5,
            min_unique_courses=min_unique_courses,
            min_concept_coverage_percent=min_concept_coverage_percent,
        )
    ]
    if not feasible:
        return None

    ranked = sorted(
        feasible,
        key=lambda row: (
            -int(row["users_ge5"]),
            -int(row["unique_mapped_courses_used"]),
            -float(row["dag_concept_user_coverage_percent"]),
            str(row["domain"]),
        ),
    )
    return str(ranked[0]["domain"])
