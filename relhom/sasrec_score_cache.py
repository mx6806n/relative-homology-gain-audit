"""SASRec validation score cache utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import torch

from relhom.sasrec import SASRecDataset, SASRecModel, _make_history_batch


@dataclass(frozen=True)
class SASRecScoreCacheStats:
    users: int
    courses: int
    seen_entries: int
    candidate_entries: int
    score_min: float
    score_max: float
    truncated_histories: int


def score_validation_logits(
    model: SASRecModel,
    dataset: SASRecDataset,
    *,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> tuple[np.ndarray, int]:
    """Return raw validation logits for every user/course pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device is None:
        device = next(model.parameters()).device

    logits = np.empty(
        (len(dataset.user_ids), len(dataset.item_ids)),
        dtype=np.float32,
    )
    truncated_histories = 0
    model.eval()

    for start in range(0, len(dataset.user_ids), batch_size):
        stop = min(start + batch_size, len(dataset.user_ids))
        histories = dataset.train_sequences[start:stop]
        truncated_histories += sum(
            len(history) > model.max_seq_len
            for history in histories
        )
        input_ids, last_positions = _make_history_batch(
            histories,
            max_seq_len=model.max_seq_len,
            device=device,
        )

        with torch.no_grad():
            hidden = model.encode(input_ids)
            rows = torch.arange(hidden.shape[0], device=device)
            final_hidden = hidden[rows, last_positions]
            batch_logits = model.logits_from_hidden(final_hidden)

        logits[start:stop] = (
            batch_logits.detach().cpu().numpy().astype(np.float32, copy=False)
        )

    return logits, truncated_histories


def validate_alignment(
    dataset: SASRecDataset,
    topology_cache: Mapping[str, np.ndarray],
) -> None:
    """Require exact user/course/target alignment with the topology cache."""
    user_ids = list(map(str, topology_cache["user_ids"].tolist()))
    course_ids = list(map(str, topology_cache["course_ids"].tolist()))
    targets = topology_cache["validation_target_indices"]

    if user_ids != list(dataset.user_ids):
        raise ValueError("SASRec users do not match topology cache ordering")
    if course_ids != list(dataset.item_ids):
        raise ValueError("SASRec courses do not match topology cache ordering")
    if targets.shape != (len(dataset.user_ids),):
        raise ValueError("topology target-index shape mismatch")
    if not np.array_equal(
        targets.astype(np.int64, copy=False),
        dataset.validation_items.detach().cpu().numpy(),
    ):
        raise ValueError("validation targets do not match topology cache")


def evaluate_cached_validation_scores(
    logits: np.ndarray,
    topology_cache: Mapping[str, np.ndarray],
    *,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Evaluate cached logits using topology cache seen-item masking."""
    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")

    deltas = topology_cache["deltas"]
    targets = topology_cache["validation_target_indices"].astype(
        np.int64,
        copy=False,
    )
    sentinel_values = topology_cache["sentinel"]
    if sentinel_values.size != 1:
        raise ValueError("topology sentinel must contain exactly one value")
    sentinel = int(sentinel_values.reshape(-1)[0])

    if logits.shape != deltas.shape:
        raise ValueError(
            f"logit shape {logits.shape} != topology shape {deltas.shape}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("raw SASRec logits must all be finite")

    scores = torch.from_numpy(
        np.asarray(logits, dtype=np.float32)
    ).clone()
    mask = torch.from_numpy(deltas == sentinel)
    target_tensor = torch.from_numpy(targets)
    rows = torch.arange(scores.shape[0])

    if bool(mask[rows, target_tensor].any()):
        raise ValueError("validation target is masked as seen")

    scores.masked_fill_(mask, float("-inf"))
    target_scores = scores[rows, target_tensor]
    item_indices = torch.arange(scores.shape[1]).unsqueeze(0)

    better = scores > target_scores.unsqueeze(1)
    tied_before = (
        (scores == target_scores.unsqueeze(1))
        & (item_indices < target_tensor.unsqueeze(1))
    )
    ranks = (
        1
        + better.sum(dim=1)
        + tied_before.sum(dim=1)
    ).to(torch.float64)

    metrics: dict[str, float | int] = {
        "users": int(scores.shape[0]),
        "missing_targets": 0,
        "mrr": float((1.0 / ranks).mean().item()),
    }
    for k in ks:
        hits = ranks <= k
        metrics[f"recall@{k}"] = float(
            hits.to(torch.float64).mean().item()
        )
        gains = torch.where(
            hits,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        )
        metrics[f"ndcg@{k}"] = float(gains.mean().item())
    return metrics


def write_sasrec_score_cache(
    path: str | Path,
    logits: np.ndarray,
    dataset: SASRecDataset,
    *,
    seed: int,
    topology_cache_sha256: str,
) -> None:
    """Persist raw logits with exact alignment metadata."""
    if logits.shape != (
        len(dataset.user_ids),
        len(dataset.item_ids),
    ):
        raise ValueError("logit matrix shape does not match dataset")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        logits=np.asarray(logits, dtype=np.float32),
        validation_target_indices=(
            dataset.validation_items.detach().cpu().numpy().astype(np.int16)
        ),
        user_ids=np.asarray(dataset.user_ids, dtype=str),
        course_ids=np.asarray(dataset.item_ids, dtype=str),
        seed=np.asarray([seed], dtype=np.int64),
        topology_cache_sha256=np.asarray(
            [topology_cache_sha256],
            dtype=str,
        ),
    )


def load_sasrec_score_cache(
    path: str | Path,
) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "logits",
            "validation_target_indices",
            "user_ids",
            "course_ids",
            "seed",
            "topology_cache_sha256",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(
                "SASRec score cache missing arrays: "
                + ", ".join(sorted(missing))
            )
        return {
            key: np.array(payload[key], copy=True)
            for key in required
        }


def validate_sasrec_score_cache(
    cache: Mapping[str, np.ndarray],
    dataset: SASRecDataset,
    *,
    expected_seed: int,
    expected_topology_cache_sha256: str,
) -> SASRecScoreCacheStats:
    logits = cache["logits"]
    targets = cache["validation_target_indices"]
    user_ids = cache["user_ids"]
    course_ids = cache["course_ids"]
    seeds = cache["seed"]
    topology_hashes = cache["topology_cache_sha256"]

    expected_shape = (
        len(dataset.user_ids),
        len(dataset.item_ids),
    )
    if logits.shape != expected_shape:
        raise ValueError(
            f"SASRec logit shape {logits.shape} != {expected_shape}"
        )
    if logits.dtype != np.float32:
        raise ValueError("SASRec logits must be float32")
    if not np.isfinite(logits).all():
        raise ValueError("SASRec score cache contains non-finite logits")
    if list(map(str, user_ids.tolist())) != list(dataset.user_ids):
        raise ValueError("SASRec score-cache user order mismatch")
    if list(map(str, course_ids.tolist())) != list(dataset.item_ids):
        raise ValueError("SASRec score-cache course order mismatch")
    if not np.array_equal(
        targets.astype(np.int64, copy=False),
        dataset.validation_items.detach().cpu().numpy(),
    ):
        raise ValueError("SASRec score-cache target mismatch")
    if seeds.size != 1 or int(seeds.reshape(-1)[0]) != expected_seed:
        raise ValueError("SASRec score-cache seed mismatch")
    if (
        topology_hashes.size != 1
        or str(topology_hashes.reshape(-1)[0])
        != expected_topology_cache_sha256
    ):
        raise ValueError("SASRec cache references wrong topology cache")

    return SASRecScoreCacheStats(
        users=expected_shape[0],
        courses=expected_shape[1],
        seen_entries=0,
        candidate_entries=expected_shape[0] * expected_shape[1],
        score_min=float(logits.min()),
        score_max=float(logits.max()),
        truncated_histories=0,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
