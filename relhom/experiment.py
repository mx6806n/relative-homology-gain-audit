"""Leakage-safe sequence splitting and deterministic recommendation baselines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from math import log2
from pathlib import Path


@dataclass(frozen=True)
class LeaveTwoOutSplit:
    """Temporal split for one aligned user sequence."""

    user_id: str
    train: tuple[str, ...]
    validation_target: str
    test_target: str

    @property
    def validation_history(self) -> tuple[str, ...]:
        """History available immediately before the validation target."""
        return self.train

    @property
    def test_history(self) -> tuple[str, ...]:
        """History available immediately before the test target."""
        return (*self.train, self.validation_target)

    @property
    def full_sequence(self) -> tuple[str, ...]:
        """Reconstruct the original aligned sequence."""
        return (*self.train, self.validation_target, self.test_target)


@dataclass(frozen=True)
class RepeatDiagnostics:
    """Repeated-course diagnostics for the selected experimental cohort."""

    users: int
    users_with_any_repeat: int
    repeated_course_events: int
    validation_targets_seen_in_train: int
    test_targets_seen_in_history: int


def load_course_universe(path: str | Path) -> tuple[str, ...]:
    """Load the deterministic course universe from course_vocab.json."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("course vocabulary must be a JSON object")

    pairs: list[tuple[int, str]] = []
    for course, index in payload.items():
        if not isinstance(course, str) or not course:
            raise ValueError("course vocabulary contains an invalid course id")
        if not isinstance(index, int) or index < 0:
            raise ValueError(
                f"course vocabulary index for {course!r} must be a non-negative integer"
            )
        pairs.append((index, course))

    indices = [index for index, _ in pairs]
    if len(indices) != len(set(indices)):
        raise ValueError("course vocabulary contains duplicate indices")

    pairs.sort()
    return tuple(course for _, course in pairs)


def load_leave_two_out_splits(
    sequences_path: str | Path,
    *,
    min_sequence_length: int = 5,
    item_universe: Iterable[str] | None = None,
) -> list[LeaveTwoOutSplit]:
    """Load aligned sequences and create deterministic temporal splits.

    For sequence [c1, ..., cn], the split is:
      train = [c1, ..., c(n-2)]
      validation_target = c(n-1)
      test_target = cn

    The source JSONL order is preserved. No random sampling is performed.
    """
    if min_sequence_length < 3:
        raise ValueError("min_sequence_length must be >= 3")

    universe = set(item_universe) if item_universe is not None else None
    seen_users: set[str] = set()
    splits: list[LeaveTwoOutSplit] = []

    path = Path(sequences_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"expected object in {path} at line {line_number}"
                )

            user_id = record.get("user_id")
            courses = record.get("courses")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError(
                    f"invalid user_id in {path} at line {line_number}"
                )
            if user_id in seen_users:
                raise ValueError(f"duplicate user_id in aligned data: {user_id}")
            seen_users.add(user_id)

            if not isinstance(courses, list) or not all(
                isinstance(course, str) and course
                for course in courses
            ):
                raise ValueError(
                    f"invalid courses list for {user_id} at line {line_number}"
                )
            if len(courses) < min_sequence_length:
                continue

            if universe is not None:
                unknown = set(courses).difference(universe)
                if unknown:
                    preview = ", ".join(sorted(unknown)[:10])
                    raise ValueError(
                        f"{user_id} contains courses outside the candidate universe: "
                        f"{preview}"
                    )

            splits.append(
                LeaveTwoOutSplit(
                    user_id=user_id,
                    train=tuple(courses[:-2]),
                    validation_target=courses[-2],
                    test_target=courses[-1],
                )
            )

    if not splits:
        raise ValueError(
            "no aligned sequences satisfy the requested minimum sequence length"
        )
    return splits


def analyze_repeats(
    splits: Sequence[LeaveTwoOutSplit],
) -> RepeatDiagnostics:
    """Quantify repeated courses without silently modifying sequences."""
    users_with_any_repeat = 0
    repeated_events = 0
    validation_seen = 0
    test_seen = 0

    for split in splits:
        full = split.full_sequence
        repeated_events += len(full) - len(set(full))
        if len(full) != len(set(full)):
            users_with_any_repeat += 1
        if split.validation_target in split.train:
            validation_seen += 1
        if split.test_target in split.test_history:
            test_seen += 1

    return RepeatDiagnostics(
        users=len(splits),
        users_with_any_repeat=users_with_any_repeat,
        repeated_course_events=repeated_events,
        validation_targets_seen_in_train=validation_seen,
        test_targets_seen_in_history=test_seen,
    )


def fit_popularity_ranking(
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Iterable[str],
) -> tuple[list[str], Counter[str]]:
    """Fit a deterministic popularity baseline using training events only."""
    universe = tuple(dict.fromkeys(item_universe))
    if not universe:
        raise ValueError("item_universe must not be empty")

    universe_set = set(universe)
    counts: Counter[str] = Counter()

    for split in splits:
        unknown = set(split.train).difference(universe_set)
        if unknown:
            preview = ", ".join(sorted(unknown)[:10])
            raise ValueError(
                f"training history contains items outside the universe: {preview}"
            )
        counts.update(split.train)

    ranking = sorted(
        universe,
        key=lambda course: (-counts[course], course),
    )
    return ranking, counts


def evaluate_fixed_ranking(
    ranking: Sequence[str],
    splits: Sequence[LeaveTwoOutSplit],
    *,
    target: str,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Evaluate one global full-catalog ranking against single next-course targets.

    This baseline intentionally ranks the complete candidate universe without
    sampled negatives or user-history filtering. The repeated-course diagnostic
    is reported separately so that a later seen-item policy can be chosen from
    evidence rather than silently imposed.
    """
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    if not splits:
        raise ValueError("splits must not be empty")

    rank_by_item = {
        item: rank
        for rank, item in enumerate(ranking, start=1)
    }

    recall_hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    reciprocal_rank_sum = 0.0
    missing_targets = 0

    for split in splits:
        target_item = (
            split.validation_target
            if target == "validation"
            else split.test_target
        )
        rank = rank_by_item.get(target_item)
        if rank is None:
            missing_targets += 1
            continue

        reciprocal_rank_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1
                ndcg_sums[k] += 1.0 / log2(rank + 1.0)

    denominator = len(splits)
    metrics: dict[str, float | int] = {
        "users": denominator,
        "missing_targets": missing_targets,
        "mrr": reciprocal_rank_sum / denominator,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / denominator
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / denominator
    return metrics


def evaluate_history_filtered_ranking(
    ranking: Sequence[str],
    splits: Sequence[LeaveTwoOutSplit],
    *,
    target: str,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Evaluate a global ranking after removing each user's observed history.

    This is the primary evaluation policy after exp_001 established that the
    selected cohort contains no repeated-course targets. For validation, the
    training prefix is removed. For test, the training prefix plus validation
    item are removed.

    The held-out target must not occur in the corresponding history; if it
    does, the unseen-item protocol is not valid for that split and execution
    stops rather than silently changing the target semantics.
    """
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    if not splits:
        raise ValueError("splits must not be empty")
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking must not contain duplicate items")

    rank_by_item = {
        item: rank
        for rank, item in enumerate(ranking, start=1)
    }

    recall_hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    reciprocal_rank_sum = 0.0
    missing_targets = 0

    for split in splits:
        if target == "validation":
            target_item = split.validation_target
            history = split.validation_history
        else:
            target_item = split.test_target
            history = split.test_history

        if target_item in history:
            raise ValueError(
                f"{split.user_id} has held-out {target} target "
                "already present in history; unseen-item evaluation is invalid"
            )

        base_rank = rank_by_item.get(target_item)
        if base_rank is None:
            missing_targets += 1
            continue

        seen_before_target = sum(
            1
            for item in set(history)
            if rank_by_item.get(item, base_rank + 1) < base_rank
        )
        filtered_rank = base_rank - seen_before_target

        reciprocal_rank_sum += 1.0 / filtered_rank
        for k in ks:
            if filtered_rank <= k:
                recall_hits[k] += 1
                ndcg_sums[k] += 1.0 / log2(filtered_rank + 1.0)

    denominator = len(splits)
    metrics: dict[str, float | int] = {
        "users": denominator,
        "missing_targets": missing_targets,
        "mrr": reciprocal_rank_sum / denominator,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / denominator
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / denominator
    return metrics


def fit_first_order_markov(
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Iterable[str],
) -> tuple[dict[str, Counter[str]], Counter[str], int]:
    """Fit train-only first-order transition counts with popularity backoff."""
    universe = tuple(dict.fromkeys(item_universe))
    if not universe:
        raise ValueError("item_universe must not be empty")

    universe_set = set(universe)
    transitions: dict[str, Counter[str]] = {}
    popularity: Counter[str] = Counter()
    transition_events = 0

    for split in splits:
        unknown = set(split.train).difference(universe_set)
        if unknown:
            preview = ", ".join(sorted(unknown)[:10])
            raise ValueError(
                f"training history contains items outside the universe: {preview}"
            )

        popularity.update(split.train)
        for source, target in zip(split.train, split.train[1:]):
            transitions.setdefault(source, Counter())[target] += 1
            transition_events += 1

    return transitions, popularity, transition_events


def rank_markov_candidates(
    history: Sequence[str],
    item_universe: Iterable[str],
    transitions: dict[str, Counter[str]],
    popularity: Counter[str],
) -> list[str]:
    """Rank unseen courses by first-order transitions then global popularity."""
    if not history:
        raise ValueError("history must not be empty")

    seen = set(history)
    candidates = [
        item
        for item in dict.fromkeys(item_universe)
        if item not in seen
    ]
    source = history[-1]
    outgoing = transitions.get(source, Counter())

    return sorted(
        candidates,
        key=lambda item: (
            -outgoing[item],
            -popularity[item],
            item,
        ),
    )


def evaluate_markov_ranking(
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Iterable[str],
    transitions: dict[str, Counter[str]],
    popularity: Counter[str],
    *,
    target: str,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Evaluate first-order Markov ranking under the unseen-course policy."""
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")
    if not splits:
        raise ValueError("splits must not be empty")

    universe = tuple(dict.fromkeys(item_universe))
    recall_hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    reciprocal_rank_sum = 0.0
    missing_targets = 0
    contexts_with_train_outgoing = 0
    targets_with_observed_transition = 0

    for split in splits:
        if target == "validation":
            history = split.validation_history
            target_item = split.validation_target
        else:
            history = split.test_history
            target_item = split.test_target

        if target_item in history:
            raise ValueError(
                f"{split.user_id} has held-out {target} target "
                "already present in history; unseen-item evaluation is invalid"
            )

        source = history[-1]
        outgoing = transitions.get(source, Counter())
        if sum(outgoing.values()) > 0:
            contexts_with_train_outgoing += 1
        if outgoing[target_item] > 0:
            targets_with_observed_transition += 1

        ranking = rank_markov_candidates(
            history,
            universe,
            transitions,
            popularity,
        )
        try:
            rank = ranking.index(target_item) + 1
        except ValueError:
            missing_targets += 1
            continue

        reciprocal_rank_sum += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall_hits[k] += 1
                ndcg_sums[k] += 1.0 / log2(rank + 1.0)

    denominator = len(splits)
    metrics: dict[str, float | int] = {
        "users": denominator,
        "missing_targets": missing_targets,
        "contexts_with_train_outgoing": contexts_with_train_outgoing,
        "targets_with_observed_transition": targets_with_observed_transition,
        "mrr": reciprocal_rank_sum / denominator,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / denominator
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / denominator
    return metrics

def count_cold_targets(
    counts: Counter[str],
    splits: Sequence[LeaveTwoOutSplit],
    *,
    target: str,
) -> int:
    """Count held-out targets never observed in any training prefix."""
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")

    cold = 0
    for split in splits:
        target_item = (
            split.validation_target
            if target == "validation"
            else split.test_target
        )
        if counts[target_item] == 0:
            cold += 1
    return cold
