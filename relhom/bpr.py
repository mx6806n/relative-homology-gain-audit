"""Bayesian Personalized Ranking matrix-factorization baseline."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log2

import torch
from torch import nn
from torch.nn import functional as F

from relhom.experiment import LeaveTwoOutSplit


@dataclass(frozen=True)
class BPRDataset:
    """Indexed train-only implicit-feedback data for one fixed split."""

    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    positive_users: torch.Tensor
    positive_items: torch.Tensor
    train_mask: torch.Tensor
    validation_items: torch.Tensor
    test_items: torch.Tensor


class BPRMatrixFactorization(nn.Module):
    """User/item latent-factor model optimized with pairwise BPR loss."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if n_users < 1 or n_items < 2:
            raise ValueError("BPR requires at least 1 user and 2 items")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.item_bias = nn.Embedding(n_items, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)
        bias = self.item_bias(items).squeeze(-1)
        return (user_vectors * item_vectors).sum(dim=-1) + bias

    def score_all_items(self) -> torch.Tensor:
        return (
            self.user_embedding.weight
            @ self.item_embedding.weight.transpose(0, 1)
            + self.item_bias.weight.squeeze(-1)
        )


def build_bpr_dataset(
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Iterable[str],
) -> BPRDataset:
    """Index the fixed leave-two-out split without using held-out targets as positives."""
    if not splits:
        raise ValueError("splits must not be empty")

    item_ids = tuple(dict.fromkeys(item_universe))
    if len(item_ids) < 2:
        raise ValueError("item_universe must contain at least two items")

    item_index = {item: index for index, item in enumerate(item_ids)}
    user_ids = tuple(split.user_id for split in splits)
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("split user ids must be unique")

    positive_users: list[int] = []
    positive_items: list[int] = []
    train_mask = torch.zeros(
        (len(splits), len(item_ids)),
        dtype=torch.bool,
    )
    validation_items = torch.empty(len(splits), dtype=torch.long)
    test_items = torch.empty(len(splits), dtype=torch.long)

    for user_idx, split in enumerate(splits):
        try:
            validation_idx = item_index[split.validation_target]
            test_idx = item_index[split.test_target]
        except KeyError as exc:
            raise ValueError(
                f"held-out item is outside the candidate universe: {exc}"
            ) from exc

        validation_items[user_idx] = validation_idx
        test_items[user_idx] = test_idx

        for item in split.train:
            try:
                item_idx = item_index[item]
            except KeyError as exc:
                raise ValueError(
                    f"training item is outside the candidate universe: {item}"
                ) from exc
            if train_mask[user_idx, item_idx]:
                raise ValueError(
                    f"duplicate training item for user {split.user_id}: {item}"
                )
            train_mask[user_idx, item_idx] = True
            positive_users.append(user_idx)
            positive_items.append(item_idx)

        if train_mask[user_idx, validation_idx]:
            raise ValueError(
                f"validation target already appears in training history: "
                f"{split.user_id}"
            )
        if train_mask[user_idx, test_idx] or validation_idx == test_idx:
            raise ValueError(
                f"test target already appears in earlier history: "
                f"{split.user_id}"
            )

    return BPRDataset(
        user_ids=user_ids,
        item_ids=item_ids,
        positive_users=torch.tensor(positive_users, dtype=torch.long),
        positive_items=torch.tensor(positive_items, dtype=torch.long),
        train_mask=train_mask,
        validation_items=validation_items,
        test_items=test_items,
    )


def sample_bpr_negatives(
    users: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample items not observed in each user's training prefix.

    Held-out validation/test items are intentionally NOT excluded: doing so
    would reveal future information to the training sampler.
    """
    if users.ndim != 1:
        raise ValueError("users must be a 1D tensor")
    if train_mask.ndim != 2:
        raise ValueError("train_mask must be a 2D tensor")

    n_items = train_mask.shape[1]
    negatives = torch.randint(
        n_items,
        (users.numel(),),
        generator=generator,
    )
    invalid = train_mask[users, negatives]

    while bool(invalid.any()):
        negatives[invalid] = torch.randint(
            n_items,
            (int(invalid.sum().item()),),
            generator=generator,
        )
        invalid = train_mask[users, negatives]

    return negatives


def evaluate_bpr_model(
    model: BPRMatrixFactorization,
    dataset: BPRDataset,
    *,
    target: str,
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float | int]:
    """Full unseen-course ranking for the static BPR user representation."""
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")

    model.eval()
    with torch.no_grad():
        scores = model.score_all_items().detach().cpu()

    mask = dataset.train_mask.clone()
    if target == "validation":
        targets = dataset.validation_items
    else:
        targets = dataset.test_items
        rows = torch.arange(len(dataset.user_ids))
        mask[rows, dataset.validation_items] = True

    rows = torch.arange(len(dataset.user_ids))
    if bool(mask[rows, targets].any()):
        raise ValueError("held-out target is masked as previously observed")

    scores = scores.masked_fill(mask, float("-inf"))
    target_scores = scores[rows, targets]

    item_indices = torch.arange(len(dataset.item_ids)).unsqueeze(0)
    better = scores > target_scores.unsqueeze(1)
    tied_before = (
        (scores == target_scores.unsqueeze(1))
        & (item_indices < targets.unsqueeze(1))
    )
    ranks = (
        1
        + better.sum(dim=1)
        + tied_before.sum(dim=1)
    ).to(torch.float64)

    metrics: dict[str, float | int] = {
        "users": len(dataset.user_ids),
        "missing_targets": 0,
        "mrr": float((1.0 / ranks).mean().item()),
    }
    for k in ks:
        hits = ranks <= k
        metrics[f"recall@{k}"] = float(hits.to(torch.float64).mean().item())
        gains = torch.where(
            hits,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        )
        metrics[f"ndcg@{k}"] = float(gains.mean().item())

    return metrics


def train_bpr(
    dataset: BPRDataset,
    *,
    embedding_dim: int,
    learning_rate: float,
    l2: float,
    batch_size: int,
    max_epochs: int,
    eval_every: int,
    patience: int,
    seed: int,
    cutoffs: Iterable[int] = (5, 10, 20),
    selection_metric: str = "ndcg@10",
) -> tuple[BPRMatrixFactorization, dict[str, object]]:
    """Train BPR on CPU with validation-only early stopping."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if batch_size < 1 or max_epochs < 1:
        raise ValueError("batch_size and max_epochs must be positive")
    if eval_every < 1 or patience < 1:
        raise ValueError("eval_every and patience must be positive")

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    model = BPRMatrixFactorization(
        len(dataset.user_ids),
        len(dataset.item_ids),
        embedding_dim,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    generator = torch.Generator().manual_seed(seed)

    positive_users = dataset.positive_users
    positive_items = dataset.positive_items
    n_events = positive_users.numel()

    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    evaluations_without_improvement = 0
    history: list[dict[str, float | int]] = []
    validation_negative_samples = 0
    test_negative_samples = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(
            n_events,
            generator=generator,
        )
        epoch_loss_sum = 0.0
        batches = 0

        for start in range(0, n_events, batch_size):
            indices = permutation[start : start + batch_size]
            users = positive_users[indices]
            positives = positive_items[indices]
            negatives = sample_bpr_negatives(
                users,
                dataset.train_mask,
                generator=generator,
            )

            validation_negative_samples += int(
                (negatives == dataset.validation_items[users]).sum().item()
            )
            test_negative_samples += int(
                (negatives == dataset.test_items[users]).sum().item()
            )

            optimizer.zero_grad()
            pos_scores = model.score_pairs(users, positives)
            neg_scores = model.score_pairs(users, negatives)
            ranking_loss = F.softplus(neg_scores - pos_scores).mean()

            user_vectors = model.user_embedding(users)
            pos_vectors = model.item_embedding(positives)
            neg_vectors = model.item_embedding(negatives)
            regularization = l2 * (
                user_vectors.pow(2).mean()
                + pos_vectors.pow(2).mean()
                + neg_vectors.pow(2).mean()
            )
            loss = ranking_loss + regularization
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            batches += 1

        should_evaluate = (
            epoch == 1
            or epoch % eval_every == 0
            or epoch == max_epochs
        )
        if not should_evaluate:
            continue

        validation = evaluate_bpr_model(
            model,
            dataset,
            target="validation",
            cutoffs=cutoffs,
        )
        if selection_metric not in validation:
            raise ValueError(
                f"selection metric not produced by evaluator: {selection_metric}"
            )
        current = float(validation[selection_metric])
        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss_sum / max(batches, 1),
                selection_metric: current,
            }
        )

        if current > best_metric + 1e-12:
            best_metric = current
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1

        if evaluations_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("BPR training produced no validation checkpoint")
    model.load_state_dict(best_state)

    diagnostics: dict[str, object] = {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "selection_metric": selection_metric,
        "best_validation_metric": best_metric,
        "validation_target_negative_samples": validation_negative_samples,
        "test_target_negative_samples": test_negative_samples,
        "history": history,
    }
    return model, diagnostics
