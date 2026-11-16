"""Tests for untouched-domain feasibility selection."""

from __future__ import annotations

from relhom.domain_feasibility import (
    domain_is_feasible,
    select_feasible_domain,
)


def _row(
    domain: str,
    *,
    users: int,
    courses: int,
    coverage: float,
    dag: bool = True,
):
    return {
        "domain": domain,
        "graph_is_dag": dag,
        "users_ge5": users,
        "unique_mapped_courses_used": courses,
        "dag_concept_user_coverage_percent": coverage,
    }


def test_feasibility_thresholds() -> None:
    row = _row("math", users=2500, courses=40, coverage=60.0)
    assert domain_is_feasible(
        row,
        min_users_ge5=2000,
        min_unique_courses=30,
        min_concept_coverage_percent=50.0,
    )


def test_selection_prefers_larger_cohort_before_course_count() -> None:
    rows = [
        _row("math", users=5000, courses=35, coverage=60.0),
        _row("psy", users=3000, courses=80, coverage=90.0),
    ]
    selected = select_feasible_domain(
        rows,
        min_users_ge5=2000,
        min_unique_courses=30,
        min_concept_coverage_percent=50.0,
    )
    assert selected == "math"


def test_selection_returns_none_when_no_domain_is_feasible() -> None:
    rows = [
        _row("math", users=1000, courses=40, coverage=60.0),
        _row("psy", users=3000, courses=10, coverage=90.0),
    ]
    assert (
        select_feasible_domain(
            rows,
            min_users_ge5=2000,
            min_unique_courses=30,
            min_concept_coverage_percent=50.0,
        )
        is None
    )
