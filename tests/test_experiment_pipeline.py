"""Tests for leakage-safe temporal splitting and popularity evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from relhom.experiment import (
    analyze_repeats,
    count_cold_targets,
    evaluate_fixed_ranking,
    evaluate_history_filtered_ranking,
    evaluate_markov_ranking,
    fit_first_order_markov,
    fit_popularity_ranking,
    load_course_universe,
    load_leave_two_out_splits,
    rank_markov_candidates,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    sequences = root / "sequences.jsonl"
    rows = [
        {"user_id": "U_1", "courses": ["A", "B", "C", "D", "E"]},
        {"user_id": "U_2", "courses": ["A", "C", "B", "E", "D"]},
        {"user_id": "U_3", "courses": ["A", "B", "C", "D"]},
    ]
    with sequences.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    vocab = root / "course_vocab.json"
    vocab.write_text(
        json.dumps({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}),
        encoding="utf-8",
    )
    return sequences, vocab


def test_leave_two_out_is_temporal_and_min_length_filtered(
    tmp_path: Path,
) -> None:
    sequences, vocab_path = _write_fixture(tmp_path)
    universe = load_course_universe(vocab_path)

    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=universe,
    )

    assert len(splits) == 2
    assert splits[0].train == ("A", "B", "C")
    assert splits[0].validation_target == "D"
    assert splits[0].test_target == "E"
    assert splits[0].validation_history == ("A", "B", "C")
    assert splits[0].test_history == ("A", "B", "C", "D")


def test_popularity_uses_training_prefixes_only(tmp_path: Path) -> None:
    sequences, vocab_path = _write_fixture(tmp_path)
    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=universe,
    )

    ranking, counts = fit_popularity_ranking(splits, universe)

    assert counts["A"] == 2
    assert counts["B"] == 2
    assert counts["C"] == 2
    assert counts["D"] == 0
    assert counts["E"] == 0
    assert ranking == ["A", "B", "C", "D", "E", "F"]
    assert count_cold_targets(counts, splits, target="validation") == 2
    assert count_cold_targets(counts, splits, target="test") == 2

    validation = evaluate_fixed_ranking(
        ranking,
        splits,
        target="validation",
        cutoffs=(3, 5),
    )
    assert validation["recall@3"] == 0.0
    assert validation["recall@5"] == 1.0
    assert validation["mrr"] == pytest.approx((1 / 4 + 1 / 5) / 2)


def test_repeat_diagnostics_do_not_mutate_sequences(tmp_path: Path) -> None:
    sequences = tmp_path / "sequences.jsonl"
    sequences.write_text(
        json.dumps(
            {
                "user_id": "U_repeat",
                "courses": ["A", "B", "A", "C", "A"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=("A", "B", "C"),
    )
    diagnostics = analyze_repeats(splits)

    assert diagnostics.users_with_any_repeat == 1
    assert diagnostics.repeated_course_events == 2
    assert diagnostics.validation_targets_seen_in_train == 0
    assert diagnostics.test_targets_seen_in_history == 1
    assert splits[0].full_sequence == ("A", "B", "A", "C", "A")


def test_history_filtered_ranking_removes_seen_items(tmp_path: Path) -> None:
    sequences, vocab_path = _write_fixture(tmp_path)
    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=universe,
    )

    ranking, _ = fit_popularity_ranking(splits, universe)
    validation = evaluate_history_filtered_ranking(
        ranking,
        splits,
        target="validation",
        cutoffs=(1, 2),
    )

    assert validation["recall@1"] == 0.5
    assert validation["recall@2"] == 1.0
    assert validation["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_history_filtered_ranking_rejects_seen_target(tmp_path: Path) -> None:
    sequences = tmp_path / "sequences.jsonl"
    sequences.write_text(
        json.dumps(
            {
                "user_id": "U_repeat",
                "courses": ["A", "B", "C", "D", "A"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=("A", "B", "C", "D"),
    )

    with pytest.raises(ValueError, match="unseen-item evaluation is invalid"):
        evaluate_history_filtered_ranking(
            ["A", "B", "C", "D"],
            splits,
            target="test",
        )


def test_markov_baseline_uses_training_transitions_only(tmp_path: Path) -> None:
    sequences, vocab_path = _write_fixture(tmp_path)
    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=universe,
    )

    transitions, popularity, transition_events = fit_first_order_markov(
        splits,
        universe,
    )

    assert transition_events == 4
    assert transitions["A"]["B"] == 1
    assert transitions["A"]["C"] == 1
    assert transitions["B"]["C"] == 1
    assert transitions["C"]["B"] == 1
    assert transitions.get("D", {}).get("E", 0) == 0

    ranking = rank_markov_candidates(
        ("A", "B", "C"),
        universe,
        transitions,
        popularity,
    )
    assert "A" not in ranking
    assert "B" not in ranking
    assert "C" not in ranking
    assert set(ranking) == {"D", "E", "F"}


def test_markov_evaluation_is_full_unseen_ranking(tmp_path: Path) -> None:
    sequences, vocab_path = _write_fixture(tmp_path)
    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences,
        min_sequence_length=5,
        item_universe=universe,
    )
    transitions, popularity, _ = fit_first_order_markov(
        splits,
        universe,
    )

    metrics = evaluate_markov_ranking(
        splits,
        universe,
        transitions,
        popularity,
        target="validation",
        cutoffs=(1, 3),
    )

    assert metrics["users"] == 2
    assert metrics["missing_targets"] == 0
    assert metrics["contexts_with_train_outgoing"] == 2
    assert metrics["recall@3"] == 1.0
