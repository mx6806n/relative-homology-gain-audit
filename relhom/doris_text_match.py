"""Conservative DORIS prerequisite extraction helpers for audit v3."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable


NEGATIVE_ROW_PATTERNS = (
    r"\bno\s+pre[- ]?requisites?\b",
    r"\bdo\s+not\s+need\s+to\s+study\s+any\s+pre[- ]?requisite",
)
NON_PRIOR_FRAGMENT_PATTERNS = (
    r"\bconcurrent(?:ly)?\b",
    r"\bsimultaneous(?:ly)?\b",
    r"\bwould\s+be\s+helpful\b",
)

LEADING_PREFIXES = (
    r"^prerequisites?\s*:\s*",
    r"^prerequisite\s+courses?\s*(?:are|:)\s*",
    r"^take\s+courses?\s+such\s+as\s+",
    r"^courses?\s+such\s+as\s+",
    r"^i\s+have\s+studied\s+",
    r"^studied\s+",
)


def phrase_norm(value: object) -> str:
    """Normalize a course-title phrase without destroying word boundaries."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("_x000d_", " ")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([()])\s*", r"\1", text)
    text = re.sub(r"\s*[-–—]\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .:")
    return text


def _strip_prefix(text: str) -> str:
    value = text.strip()
    for pattern in LEADING_PREFIXES:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return value.strip(" .:")


def _split_fragment(fragment: str, exact_names: set[str]) -> list[str]:
    """Split conjunction lists only when the whole phrase is not a catalog title."""
    normalized = phrase_norm(_strip_prefix(fragment))
    if not normalized:
        return []
    if normalized in exact_names:
        return [normalized]

    parts = re.split(r"\s+(?:and|or)\s+", normalized)
    if len(parts) > 1:
        result: list[str] = []
        for part in parts:
            result.extend(_split_fragment(part, exact_names))
        return result
    return [normalized]


def extract_exact_course_ids(
    prerequisite_text: object,
    normalized_name_to_ids: dict[str, set[str]],
) -> set[str]:
    """Resolve only exact, unambiguous course-title phrases.

    v3 intentionally does not use arbitrary substring matching or bare numeric
    CourseId tokens.  This sacrifices recall to protect edge precision.
    """
    raw = unicodedata.normalize("NFKC", str(prerequisite_text))
    lowered = raw.lower().replace("_x000d_", "\n")

    if any(re.search(pattern, lowered) for pattern in NEGATIVE_ROW_PATTERNS):
        return set()

    name_to_id = {
        phrase_norm(name): next(iter(ids))
        for name, ids in normalized_name_to_ids.items()
        if len(ids) == 1 and phrase_norm(name)
    }
    exact_names = set(name_to_id)

    aliases: set[str] = set()
    for match in re.finditer(r"([^,;\n]+?)\(([^()]+)\)", lowered):
        outer = phrase_norm(match.group(1))
        inner = phrase_norm(match.group(2))
        if outer in exact_names and inner in exact_names:
            aliases.add(inner)

    found: set[str] = {name_to_id[name] for name in aliases}

    fragments = re.split(r"[,;，；、\n]+|(?<=[.!?])\s+", lowered)
    for raw_fragment in fragments:
        if any(
            re.search(pattern, raw_fragment, flags=re.IGNORECASE)
            for pattern in NON_PRIOR_FRAGMENT_PATTERNS
        ):
            continue

        for phrase in _split_fragment(raw_fragment, exact_names):
            if phrase in name_to_id:
                found.add(name_to_id[phrase])

    return found
