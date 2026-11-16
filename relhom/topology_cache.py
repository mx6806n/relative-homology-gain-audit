"""Build and validate dense validation-time topology score caches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import time

import networkx as nx
import numpy as np

from relhom.experiment import LeaveTwoOutSplit
from relhom.topology import GF2RelativeHomologyEvaluator


TOPOLOGY_SENTINEL = np.int16(-32768)


@dataclass(frozen=True)
class TopologyCacheStats:
    users: int
    courses: int
    candidate_evaluations: int
    seen_entries: int
    elapsed_seconds: float
    cache_info: dict[str, int | float]
    beta_before_min: int
    beta_before_max: int
    delta_min: int
    delta_max: int
    delta_histogram: dict[str, int]
    positive_fraction: float
    zero_fraction: float
    negative_fraction: float
    target_positive: int
    target_zero: int
    target_negative: int
    users_with_discriminative_signal: int


def _check_int16(value: int, *, field: str) -> None:
    info = np.iinfo(np.int16)
    if value < info.min + 1 or value > info.max:
        raise OverflowError(
            f"{field}={value} does not fit in int16 with sentinel reserved"
        )


def build_validation_topology_cache(
    graph: nx.DiGraph,
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Sequence[str],
    course_to_concepts: Mapping[str, set[str] | frozenset[str]],
    output_path: str | Path,
    *,
    evaluator_cache_size: int = 524_288,
) -> TopologyCacheStats:
    """Score every unseen validation candidate on the raw prerequisite DAG.

    The resulting NPZ contains:
    - deltas: int16 [users, courses], TOPOLOGY_SENTINEL for seen courses;
    - beta_before: int16 [users];
    - validation_target_indices: int16 [users];
    - history_lengths: int16 [users];
    - user_ids: fixed-width Unicode [users];
    - course_ids: fixed-width Unicode [courses].

    Course order is exactly item_universe order.
    """
    if not splits:
        raise ValueError("splits must not be empty")
    if not item_universe:
        raise ValueError("item_universe must not be empty")

    course_ids = tuple(item_universe)
    if len(course_ids) != len(set(course_ids)):
        raise ValueError("item_universe contains duplicate courses")
    course_index = {
        course: index for index, course in enumerate(course_ids)
    }

    evaluator = GF2RelativeHomologyEvaluator(
        graph,
        cache_size=evaluator_cache_size,
    )

    n_users = len(splits)
    n_courses = len(course_ids)
    deltas = np.full(
        (n_users, n_courses),
        TOPOLOGY_SENTINEL,
        dtype=np.int16,
    )
    beta_before_values = np.empty(n_users, dtype=np.int16)
    target_indices = np.empty(n_users, dtype=np.int16)
    history_lengths = np.empty(n_users, dtype=np.int16)

    delta_hist: Counter[int] = Counter()
    target_positive = 0
    target_zero = 0
    target_negative = 0
    discriminative_users = 0
    candidate_evaluations = 0
    seen_entries = 0

    started = time.perf_counter()

    for user_index, split in enumerate(splits):
        history_courses = tuple(split.validation_history)
        history_course_set = set(history_courses)
        history_lengths[user_index] = len(history_courses)

        history_concepts: set[str] = set()
        for course in history_course_set:
            history_concepts.update(course_to_concepts.get(course, ()))

        beta_before = evaluator.compute_relative_betti(history_concepts)
        _check_int16(beta_before, field="beta_before")
        beta_before_values[user_index] = beta_before

        try:
            target_index = course_index[split.validation_target]
        except KeyError as exc:
            raise ValueError(
                f"validation target outside candidate universe: "
                f"{split.validation_target}"
            ) from exc
        target_indices[user_index] = target_index

        user_deltas: list[int] = []

        for course_index_value, course in enumerate(course_ids):
            if course in history_course_set:
                seen_entries += 1
                continue

            candidate_concepts = course_to_concepts.get(course, ())
            beta_after = evaluator.compute_relative_betti(
                history_concepts.union(candidate_concepts)
            )
            delta = beta_before - beta_after
            _check_int16(delta, field="delta_beta1")
            deltas[user_index, course_index_value] = delta
            delta_hist[delta] += 1
            user_deltas.append(delta)
            candidate_evaluations += 1

        target_delta = int(deltas[user_index, target_index])
        if target_delta == int(TOPOLOGY_SENTINEL):
            raise ValueError(
                f"validation target is masked as seen for user "
                f"{split.user_id}"
            )

        if target_delta > 0:
            target_positive += 1
        elif target_delta < 0:
            target_negative += 1
        else:
            target_zero += 1

        if len(set(user_deltas)) > 1:
            discriminative_users += 1

    elapsed = time.perf_counter() - started

    if candidate_evaluations + seen_entries != n_users * n_courses:
        raise RuntimeError("cache matrix accounting mismatch")

    candidate_values = deltas[deltas != TOPOLOGY_SENTINEL]
    if candidate_values.size != candidate_evaluations:
        raise RuntimeError("candidate count does not match cache entries")

    positive_count = int((candidate_values > 0).sum())
    zero_count = int((candidate_values == 0).sum())
    negative_count = int((candidate_values < 0).sum())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        deltas=deltas,
        beta_before=beta_before_values,
        validation_target_indices=target_indices,
        history_lengths=history_lengths,
        user_ids=np.asarray(
            [split.user_id for split in splits],
            dtype=str,
        ),
        course_ids=np.asarray(course_ids, dtype=str),
        sentinel=np.asarray([TOPOLOGY_SENTINEL], dtype=np.int16),
    )

    return TopologyCacheStats(
        users=n_users,
        courses=n_courses,
        candidate_evaluations=candidate_evaluations,
        seen_entries=seen_entries,
        elapsed_seconds=elapsed,
        cache_info=evaluator.cache_info(),
        beta_before_min=int(beta_before_values.min()),
        beta_before_max=int(beta_before_values.max()),
        delta_min=int(candidate_values.min()),
        delta_max=int(candidate_values.max()),
        delta_histogram={
            str(key): delta_hist[key]
            for key in sorted(delta_hist)
        },
        positive_fraction=positive_count / candidate_evaluations,
        zero_fraction=zero_count / candidate_evaluations,
        negative_fraction=negative_count / candidate_evaluations,
        target_positive=target_positive,
        target_zero=target_zero,
        target_negative=target_negative,
        users_with_discriminative_signal=discriminative_users,
    )


def load_validation_topology_cache(
    path: str | Path,
) -> dict[str, np.ndarray]:
    """Load a topology cache without pickle/object arrays."""
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "deltas",
            "beta_before",
            "validation_target_indices",
            "history_lengths",
            "user_ids",
            "course_ids",
            "sentinel",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                "topology cache missing arrays: "
                + ", ".join(sorted(missing))
            )
        return {
            key: np.array(payload[key], copy=True)
            for key in required
        }


def validate_topology_cache(
    cache: Mapping[str, np.ndarray],
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Sequence[str],
) -> None:
    """Validate ordering, dimensions, target visibility, and seen masking."""
    deltas = cache["deltas"]
    beta_before = cache["beta_before"]
    targets = cache["validation_target_indices"]
    histories = cache["history_lengths"]
    user_ids = cache["user_ids"]
    course_ids = cache["course_ids"]
    sentinel_values = cache["sentinel"]

    if sentinel_values.size != 1:
        raise ValueError("cache sentinel array must contain one value")
    sentinel = int(sentinel_values.reshape(-1)[0])

    expected_shape = (len(splits), len(item_universe))
    if deltas.shape != expected_shape:
        raise ValueError(
            f"delta matrix shape {deltas.shape} != {expected_shape}"
        )
    if beta_before.shape != (len(splits),):
        raise ValueError("beta_before shape mismatch")
    if targets.shape != (len(splits),):
        raise ValueError("target index shape mismatch")
    if histories.shape != (len(splits),):
        raise ValueError("history length shape mismatch")

    if list(map(str, user_ids.tolist())) != [
        split.user_id for split in splits
    ]:
        raise ValueError("cache user ordering does not match splits")
    if list(map(str, course_ids.tolist())) != list(item_universe):
        raise ValueError("cache course ordering does not match universe")

    course_index = {
        course: index for index, course in enumerate(item_universe)
    }
    for row, split in enumerate(splits):
        expected_target = course_index[split.validation_target]
        if int(targets[row]) != expected_target:
            raise ValueError(
                f"target index mismatch for {split.user_id}"
            )
        if int(deltas[row, expected_target]) == sentinel:
            raise ValueError(
                f"target masked in cache for {split.user_id}"
            )
        if int(histories[row]) != len(split.validation_history):
            raise ValueError(
                f"history length mismatch for {split.user_id}"
            )
        for course in split.validation_history:
            if int(deltas[row, course_index[course]]) != sentinel:
                raise ValueError(
                    f"seen course not masked for {split.user_id}: {course}"
                )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology_cache_stats_to_dict(
    stats: TopologyCacheStats,
) -> dict[str, object]:
    return asdict(stats)
