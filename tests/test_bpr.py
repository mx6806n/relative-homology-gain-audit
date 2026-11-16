"""Tests for the BPR matrix-factorization baseline."""

from __future__ import annotations

import torch

from relhom.bpr import (
    BPRMatrixFactorization,
    build_bpr_dataset,
    evaluate_bpr_model,
    sample_bpr_negatives,
)
from relhom.experiment import LeaveTwoOutSplit


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


def test_bpr_dataset_contains_training_positives_only() -> None:
    dataset = build_bpr_dataset(
        _splits(),
        ("A", "B", "C", "D", "E", "F"),
    )

    assert dataset.positive_users.numel() == 6
    assert int(dataset.train_mask.sum().item()) == 6
    assert not bool(
        dataset.train_mask[
            torch.arange(2),
            dataset.validation_items,
        ].any()
    )
    assert not bool(
        dataset.train_mask[
            torch.arange(2),
            dataset.test_items,
        ].any()
    )


def test_negative_sampler_never_returns_training_positive() -> None:
    dataset = build_bpr_dataset(
        _splits(),
        ("A", "B", "C", "D", "E", "F"),
    )
    users = dataset.positive_users.repeat(20)
    negatives = sample_bpr_negatives(
        users,
        dataset.train_mask,
        generator=torch.Generator().manual_seed(7),
    )

    assert not bool(dataset.train_mask[users, negatives].any())


def test_bpr_evaluator_filters_history() -> None:
    dataset = build_bpr_dataset(
        _splits()[:1],
        ("A", "B", "C", "D", "E", "F"),
    )
    model = BPRMatrixFactorization(
        n_users=1,
        n_items=6,
        embedding_dim=1,
    )

    with torch.no_grad():
        model.user_embedding.weight[:] = 1.0
        model.item_embedding.weight[:, 0] = torch.tensor(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
        )
        model.item_bias.weight.zero_()

    validation = evaluate_bpr_model(
        model,
        dataset,
        target="validation",
        cutoffs=(1, 2),
    )
    assert validation["recall@1"] == 1.0

    test = evaluate_bpr_model(
        model,
        dataset,
        target="test",
        cutoffs=(1, 2),
    )
    assert test["recall@1"] == 1.0
