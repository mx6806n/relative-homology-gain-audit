"""Tests for exact course title matching on the DORIS prerequisite text."""

from relhom.doris_text_match import extract_exact_course_ids, phrase_norm


def _catalog(*pairs: tuple[str, str]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for course_id, name in pairs:
        mapping.setdefault(phrase_norm(name), set()).add(course_id)
    return mapping


def test_exact_list_entries_match() -> None:
    catalog = _catalog(
        ("162", "Probability Theory"),
        ("4888", "Linear Algebra"),
        ("177", "Mathematical Analysis"),
    )
    found = extract_exact_course_ids(
        "Probability Theory, Linear Algebra, Mathematical Analysis",
        catalog,
    )
    assert found == {"162", "4888", "177"}


def test_short_name_does_not_match_inside_longer_course_title() -> None:
    catalog = _catalog(
        ("2632", "Economics"),
        ("2569", "Intermediate Microeconomics"),
    )
    found = extract_exact_course_ids(
        "Intermediate Microeconomics",
        catalog,
    )
    assert "2632" not in found
    assert "2569" in found


def test_management_does_not_match_inside_compound_title() -> None:
    catalog = _catalog(
        ("2628", "Management"),
        ("999", "Urban Planning Management and Regulations"),
    )
    found = extract_exact_course_ids(
        "Principles of Urban Planning, Urban Planning Management and Regulations",
        catalog,
    )
    assert "2628" not in found
    assert "999" in found


def test_geometry_does_not_match_inside_algebraic_geometry() -> None:
    catalog = _catalog(("199", "Geometry"))
    found = extract_exact_course_ids(
        "abstract algebra, commutative algebra and some preliminary algebraic geometry",
        catalog,
    )
    assert found == set()


def test_no_prerequisite_row_is_rejected() -> None:
    catalog = _catalog(("3189", "Introduction to Political Science"))
    found = extract_exact_course_ids(
        "There are no prerequisites, but a course in Introduction to Political "
        "Science would be helpful.",
        catalog,
    )
    assert found == set()


def test_concurrent_course_is_not_a_prior_prerequisite() -> None:
    catalog = _catalog(
        ("466", "Particle Physics"),
        ("524", "Atomic Physics"),
    )
    found = extract_exact_course_ids(
        "Atomic Physics. Can be taken concurrently with Particle Physics.",
        catalog,
    )
    assert "524" in found
    assert "466" not in found


def test_conjunction_list_can_resolve_two_exact_courses() -> None:
    catalog = _catalog(
        ("652", "General Chemistry"),
        ("697", "Inorganic Chemistry"),
    )
    found = extract_exact_course_ids(
        "General Chemistry or Inorganic Chemistry",
        catalog,
    )
    assert found == {"652", "697"}
