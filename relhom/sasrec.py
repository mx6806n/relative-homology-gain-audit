"""SASRec-style causal self-attention baseline with full-softmax training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from relhom.experiment import LeaveTwoOutSplit


@dataclass(frozen=True)
class SASRecDataset:
    """Indexed leave-two-out data for sequential recommendation."""

    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    train_sequences: tuple[tuple[int, ...], ...]
    validation_items: torch.Tensor
    test_items: torch.Tensor


class SASRecModel(nn.Module):
    """Causal self-attention sequential recommender.

    Item index 0 is reserved for padding; catalogue item indices are shifted by
    +1 inside the embedding table. Output logits are computed against all
    catalogue item embeddings.
    """

    def __init__(
        self,
        n_items: int,
        *,
        max_seq_len: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items < 2:
            raise ValueError("SASRec requires at least two items")
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be >= 2")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if num_heads < 1 or embedding_dim % num_heads != 0:
            raise ValueError(
                "num_heads must divide embedding_dim"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.item_embedding = nn.Embedding(
            n_items + 1,
            embedding_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.zeros_(self.item_bias)

    def encode(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        """Encode padded item-id sequences with a causal attention mask."""
        if sequence_ids.ndim != 2:
            raise ValueError("sequence_ids must have shape [batch, length]")
        batch_size, length = sequence_ids.shape
        if length < 1 or length > self.max_seq_len:
            raise ValueError(
                f"sequence length must be in [1, {self.max_seq_len}]"
            )

        positions = torch.arange(
            length,
            device=sequence_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        hidden = (
            self.item_embedding(sequence_ids)
            + self.position_embedding(positions)
        )
        hidden = self.input_dropout(hidden)

        causal_mask = torch.triu(
            torch.ones(
                (length, length),
                dtype=torch.bool,
                device=sequence_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = sequence_ids.eq(0)

        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output_norm(hidden)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        catalogue_embeddings = self.item_embedding.weight[1:]
        return (
            hidden @ catalogue_embeddings.transpose(0, 1)
            + self.item_bias
        )

    def forward(self, sequence_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden(self.encode(sequence_ids))


def build_sasrec_dataset(
    splits: Sequence[LeaveTwoOutSplit],
    item_universe: Iterable[str],
) -> SASRecDataset:
    """Index training prefixes and held-out items without target leakage."""
    if not splits:
        raise ValueError("splits must not be empty")

    item_ids = tuple(dict.fromkeys(item_universe))
    if len(item_ids) < 2:
        raise ValueError("item_universe must contain at least two items")

    item_index = {item: index for index, item in enumerate(item_ids)}
    user_ids = tuple(split.user_id for split in splits)
    if len(user_ids) != len(set(user_ids)):
        raise ValueError("split user ids must be unique")

    train_sequences: list[tuple[int, ...]] = []
    validation_items = torch.empty(len(splits), dtype=torch.long)
    test_items = torch.empty(len(splits), dtype=torch.long)

    for user_idx, split in enumerate(splits):
        try:
            encoded_train = tuple(
                item_index[item]
                for item in split.train
            )
            validation_idx = item_index[split.validation_target]
            test_idx = item_index[split.test_target]
        except KeyError as exc:
            raise ValueError(
                f"sequence item is outside the candidate universe: {exc}"
            ) from exc

        if len(encoded_train) < 2:
            raise ValueError(
                "SASRec training prefix must contain at least two items"
            )
        if len(encoded_train) != len(set(encoded_train)):
            raise ValueError(
                f"duplicate item in training sequence: {split.user_id}"
            )
        if validation_idx in encoded_train:
            raise ValueError(
                f"validation target already in training history: "
                f"{split.user_id}"
            )
        if (
            test_idx in encoded_train
            or test_idx == validation_idx
        ):
            raise ValueError(
                f"test target already in earlier history: {split.user_id}"
            )

        train_sequences.append(encoded_train)
        validation_items[user_idx] = validation_idx
        test_items[user_idx] = test_idx

    return SASRecDataset(
        user_ids=user_ids,
        item_ids=item_ids,
        train_sequences=tuple(train_sequences),
        validation_items=validation_items,
        test_items=test_items,
    )


def _truncate(sequence: Sequence[int], max_seq_len: int) -> tuple[int, ...]:
    return tuple(sequence[-max_seq_len:])


def make_sasrec_training_batch(
    sequences: Sequence[Sequence[int]],
    *,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create padded teacher-forcing inputs/targets from training prefixes.

    For [c1, c2, ..., cL], inputs are [c1, ..., c(L-1)] and targets are
    [c2, ..., cL]. Padding targets are marked -100 for cross entropy.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    examples: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    max_length = 0
    for sequence in sequences:
        truncated = _truncate(sequence, max_seq_len + 1)
        if len(truncated) < 2:
            raise ValueError(
                "every training sequence must contain at least two items"
            )
        inputs = truncated[:-1]
        targets = truncated[1:]
        examples.append((inputs, targets))
        max_length = max(max_length, len(inputs))

    batch_size = len(examples)
    input_ids = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
        device=device,
    )
    valid = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
        device=device,
    )

    for row, (inputs, targets) in enumerate(examples):
        length = len(inputs)
        input_ids[row, :length] = torch.tensor(
            [item + 1 for item in inputs],
            dtype=torch.long,
            device=device,
        )
        target_ids[row, :length] = torch.tensor(
            targets,
            dtype=torch.long,
            device=device,
        )
        valid[row, :length] = True

    return input_ids, target_ids, valid


def _make_history_batch(
    histories: Sequence[Sequence[int]],
    *,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad histories and return last valid positions."""
    if not histories:
        raise ValueError("histories must not be empty")

    truncated = [
        _truncate(history, max_seq_len)
        for history in histories
    ]
    if any(not history for history in truncated):
        raise ValueError("evaluation histories must not be empty")

    max_length = max(len(history) for history in truncated)
    input_ids = torch.zeros(
        (len(truncated), max_length),
        dtype=torch.long,
        device=device,
    )
    last_positions = torch.empty(
        len(truncated),
        dtype=torch.long,
        device=device,
    )

    for row, history in enumerate(truncated):
        length = len(history)
        input_ids[row, :length] = torch.tensor(
            [item + 1 for item in history],
            dtype=torch.long,
            device=device,
        )
        last_positions[row] = length - 1

    return input_ids, last_positions


def evaluate_sasrec_model(
    model: SASRecModel,
    dataset: SASRecDataset,
    *,
    target: str,
    cutoffs: Iterable[int] = (5, 10, 20),
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    """Evaluate full unseen-course ranking for validation or test."""
    if target not in {"validation", "test"}:
        raise ValueError("target must be 'validation' or 'test'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] <= 0:
        raise ValueError("cutoffs must contain positive integers")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    reciprocal_rank_sum = 0.0
    ndcg_sums = {k: 0.0 for k in ks}
    recall_hits = {k: 0 for k in ks}
    truncated_histories = 0

    for start in range(0, len(dataset.user_ids), batch_size):
        stop = min(start + batch_size, len(dataset.user_ids))
        histories: list[tuple[int, ...]] = []
        full_histories: list[tuple[int, ...]] = []
        target_items: list[int] = []

        for user_idx in range(start, stop):
            train = dataset.train_sequences[user_idx]
            if target == "validation":
                history = train
                target_item = int(
                    dataset.validation_items[user_idx].item()
                )
            else:
                validation_item = int(
                    dataset.validation_items[user_idx].item()
                )
                history = (*train, validation_item)
                target_item = int(
                    dataset.test_items[user_idx].item()
                )

            if target_item in history:
                raise ValueError(
                    f"{dataset.user_ids[user_idx]} has target already "
                    "present in available history"
                )

            if len(history) > model.max_seq_len:
                truncated_histories += 1
            histories.append(history)
            full_histories.append(history)
            target_items.append(target_item)

        input_ids, last_positions = _make_history_batch(
            histories,
            max_seq_len=model.max_seq_len,
            device=device,
        )

        with torch.no_grad():
            hidden = model.encode(input_ids)
            rows = torch.arange(
                hidden.shape[0],
                device=device,
            )
            final_hidden = hidden[rows, last_positions]
            scores = model.logits_from_hidden(final_hidden)

        for row, history in enumerate(full_histories):
            seen = torch.tensor(
                sorted(set(history)),
                dtype=torch.long,
                device=device,
            )
            scores[row, seen] = float("-inf")

        targets = torch.tensor(
            target_items,
            dtype=torch.long,
            device=device,
        )
        rows = torch.arange(scores.shape[0], device=device)
        target_scores = scores[rows, targets]
        item_indices = torch.arange(
            len(dataset.item_ids),
            device=device,
        ).unsqueeze(0)

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

        reciprocal_rank_sum += float((1.0 / ranks).sum().item())
        for k in ks:
            hits = ranks <= k
            recall_hits[k] += int(hits.sum().item())
            gains = torch.where(
                hits,
                1.0 / torch.log2(ranks + 1.0),
                torch.zeros_like(ranks),
            )
            ndcg_sums[k] += float(gains.sum().item())

    denominator = len(dataset.user_ids)
    metrics: dict[str, float | int] = {
        "users": denominator,
        "missing_targets": 0,
        "truncated_histories": truncated_histories,
        "mrr": reciprocal_rank_sum / denominator,
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / denominator
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / denominator
    return metrics


def train_sasrec(
    dataset: SASRecDataset,
    *,
    max_seq_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    eval_batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    cutoffs: Iterable[int] = (5, 10, 20),
    selection_metric: str = "ndcg@10",
    device: torch.device | None = None,
) -> tuple[SASRecModel, dict[str, object]]:
    """Train SASRec with full-softmax next-item cross entropy."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if batch_size < 1 or eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    model = SASRecModel(
        len(dataset.item_ids),
        max_seq_len=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)

    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    n_users = len(dataset.user_ids)

    truncated_train_sequences = sum(
        len(sequence) > max_seq_len + 1
        for sequence in dataset.train_sequences
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(
            n_users,
            generator=generator,
        )
        epoch_loss_sum = 0.0
        epoch_targets = 0

        for start in range(0, n_users, batch_size):
            indices = permutation[start : start + batch_size].tolist()
            sequences = [
                dataset.train_sequences[index]
                for index in indices
            ]
            input_ids, target_ids, valid = make_sasrec_training_batch(
                sequences,
                max_seq_len=max_seq_len,
                device=device,
            )

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_targets = int(valid.sum().item())
            normalized_loss = loss / valid_targets
            normalized_loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_targets += valid_targets

        validation = evaluate_sasrec_model(
            model,
            dataset,
            target="validation",
            cutoffs=cutoffs,
            batch_size=eval_batch_size,
            device=device,
        )
        if selection_metric not in validation:
            raise ValueError(
                f"selection metric not produced by evaluator: {selection_metric}"
            )
        current = float(validation[selection_metric])
        history.append(
            {
                "epoch": epoch,
                "loss_per_target": (
                    epoch_loss_sum / max(epoch_targets, 1)
                ),
                selection_metric: current,
            }
        )

        if current > best_metric + 1e-12:
            best_metric = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("SASRec training produced no validation checkpoint")

    model.load_state_dict(best_state)
    model.to(device)

    diagnostics: dict[str, object] = {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "selection_metric": selection_metric,
        "best_validation_metric": best_metric,
        "training_objective": "full_softmax_next_item_cross_entropy",
        "truncated_train_sequences": truncated_train_sequences,
        "history": history,
    }
    return model, diagnostics
