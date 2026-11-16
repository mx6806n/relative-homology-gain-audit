"""Tests for SASRec validation score caching."""

from __future__ import annotations

import numpy as np
import torch

from relhom.experiment import LeaveTwoOutSplit
from relhom.sasrec import SASRecModel, build_sasrec_dataset
from relhom.sasrec_score_cache import (
    evaluate_cached_validation_scores,
    load_sasrec_score_cache,
    score_validation_logits,
    validate_alignment,
    validate_sasrec_score_cache,
    write_sasrec_score_cache,
)


def _dataset():
    splits = [
        LeaveTwoOutSplit(
            user_id="U_1",
            train=("A", "B", "C"),
            validation_target="D",
            test_target="E",
        )
    ]
    dataset = build_sasrec_dataset(
        splits,
        ("A", "B", "C", "D", "E", "F"),
    )
    topology = {
        "deltas": np.asarray(
            [[-32768, -32768, -32768, 1, 0, -1]],
            dtype=np.int16,
        ),
        "validation_target_indices": np.asarray([3], dtype=np.int16),
        "user_ids": np.asarray(["U_1"]),
        "course_ids": np.asarray(["A", "B", "C", "D", "E", "F"]),
        "sentinel": np.asarray([-32768], dtype=np.int16),
    }
    return dataset, topology


def test_cached_scores_align_and_mask_seen_items(tmp_path) -> None:
    dataset, topology = _dataset()
    validate_alignment(dataset, topology)

    model = SASRecModel(
        6,
        max_seq_len=5,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    logits, truncated = score_validation_logits(
        model,
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert logits.shape == (1, 6)
    assert truncated == 0

    metrics = evaluate_cached_validation_scores(
        logits,
        topology,
        cutoffs=(1, 3),
    )
    assert metrics["users"] == 1
    assert 0.0 <= metrics["recall@1"] <= 1.0

    path = tmp_path / "scores.npz"
    write_sasrec_score_cache(
        path,
        logits,
        dataset,
        seed=42,
        topology_cache_sha256="abc",
    )
    loaded = load_sasrec_score_cache(path)
    validate_sasrec_score_cache(
        loaded,
        dataset,
        expected_seed=42,
        expected_topology_cache_sha256="abc",
    )
    assert np.array_equal(loaded["logits"], logits)


def test_cached_score_ranking_respects_topology_seen_mask() -> None:
    dataset, topology = _dataset()
    logits = np.asarray(
        [[100.0, 90.0, 80.0, 70.0, 60.0, 50.0]],
        dtype=np.float32,
    )
    metrics = evaluate_cached_validation_scores(
        logits,
        topology,
        cutoffs=(1, 2),
    )
    assert metrics["recall@1"] == 1.0
