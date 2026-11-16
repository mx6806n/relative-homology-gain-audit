"""Tests for the SASRec-style sequential baseline."""

from __future__ import annotations

import torch

from relhom.experiment import LeaveTwoOutSplit
from relhom.sasrec import (
    SASRecModel,
    build_sasrec_dataset,
    evaluate_sasrec_model,
    make_sasrec_training_batch,
)


def _splits() -> list[LeaveTwoOutSplit]:
    return [
        LeaveTwoOutSplit(
            user_id="U_1",
            train=("A", "B", "C"),
            validation_target="D",
            test_target="E",
        ),
        LeaveTwoOutSplit(
            user_id="U_2",
            train=("B", "C", "D"),
            validation_target="E",
            test_target="A",
        ),
    ]


def test_sasrec_dataset_uses_training_prefix_only() -> None:
    dataset = build_sasrec_dataset(
        _splits(),
        ("A", "B", "C", "D", "E", "F"),
    )

    assert dataset.train_sequences == (
        (0, 1, 2),
        (1, 2, 3),
    )
    assert dataset.validation_items.tolist() == [3, 4]
    assert dataset.test_items.tolist() == [4, 0]


def test_training_batch_is_shifted_next_item_prediction() -> None:
    inputs, targets, valid = make_sasrec_training_batch(
        [(0, 1, 2), (1, 2, 3)],
        max_seq_len=5,
        device=torch.device("cpu"),
    )

    assert inputs.tolist() == [[1, 2], [2, 3]]
    assert targets.tolist() == [[1, 2], [2, 3]]
    assert valid.tolist() == [[True, True], [True, True]]


def test_sasrec_forward_and_evaluation_shapes() -> None:
    torch.manual_seed(3)
    dataset = build_sasrec_dataset(
        _splits(),
        ("A", "B", "C", "D", "E", "F"),
    )
    model = SASRecModel(
        6,
        max_seq_len=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )

    inputs, _, _ = make_sasrec_training_batch(
        dataset.train_sequences,
        max_seq_len=5,
        device=torch.device("cpu"),
    )
    logits = model(inputs)
    assert logits.shape == (2, 2, 6)

    validation = evaluate_sasrec_model(
        model,
        dataset,
        target="validation",
        cutoffs=(1, 3),
        batch_size=2,
        device=torch.device("cpu"),
    )
    test = evaluate_sasrec_model(
        model,
        dataset,
        target="test",
        cutoffs=(1, 3),
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert validation["users"] == 2
    assert test["users"] == 2
    assert validation["missing_targets"] == 0
    assert test["missing_targets"] == 0
