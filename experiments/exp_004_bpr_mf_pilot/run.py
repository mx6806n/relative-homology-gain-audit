"""Run exp_004: exploratory BPR matrix-factorization baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.bpr import (  # noqa: E402
    build_bpr_dataset,
    evaluate_bpr_model,
    train_bpr,
)
from relhom.experiment import (  # noqa: E402
    analyze_repeats,
    load_course_universe,
    load_leave_two_out_splits,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON/YAML object")
    return payload


def _format_metrics(metrics: dict[str, float | int]) -> str:
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(
            "exp_004 output already exists; refusing to overwrite evidence. "
            f"Existing output directory: {output_dir}"
        )

    min_sequence_length = int(config["min_sequence_length"])
    cutoffs = tuple(int(k) for k in config["cutoffs"])
    seed = int(config["seed"])

    sequences_path = data_dir / "sequences.jsonl"
    vocab_path = data_dir / "course_vocab.json"
    if not sequences_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            "aligned dataset is missing. Expected sequences.jsonl and "
            f"course_vocab.json under {data_dir}"
        )

    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences_path,
        min_sequence_length=min_sequence_length,
        item_universe=universe,
    )
    repeats = analyze_repeats(splits)
    if (
        repeats.validation_targets_seen_in_train
        or repeats.test_targets_seen_in_history
    ):
        raise ValueError(
            "unseen-course evaluation requires held-out targets not to occur "
            "earlier in user history"
        )

    dataset = build_bpr_dataset(splits, universe)
    model, training = train_bpr(
        dataset,
        embedding_dim=int(config["embedding_dim"]),
        learning_rate=float(config["learning_rate"]),
        l2=float(config["l2"]),
        batch_size=int(config["batch_size"]),
        max_epochs=int(config["max_epochs"]),
        eval_every=int(config["eval_every"]),
        patience=int(config["patience"]),
        seed=seed,
        cutoffs=cutoffs,
        selection_metric=str(config["selection_metric"]),
    )

    validation_metrics = evaluate_bpr_model(
        model,
        dataset,
        target="validation",
        cutoffs=cutoffs,
    )
    test_metrics = evaluate_bpr_model(
        model,
        dataset,
        target="test",
        cutoffs=cutoffs,
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_004_bpr_mf_pilot",
        "status": "completed",
        "phase": "exploratory",
        "parent_experiment": "exp_003_markov_baseline",
        "code_commit": _git_commit(),
        "seed": seed,
        "deterministic_cpu": True,
        "torch_version": torch.__version__,
        "data": {
            "sequences_path": str(sequences_path.relative_to(REPO_ROOT)),
            "sequences_sha256": _sha256(sequences_path),
            "course_vocab_path": str(vocab_path.relative_to(REPO_ROOT)),
            "course_vocab_sha256": _sha256(vocab_path),
            "candidate_courses": len(universe),
        },
        "split": {
            "protocol": "temporal_leave_two_out",
            "min_sequence_length": min_sequence_length,
            "users": len(splits),
            "train_events": int(dataset.positive_users.numel()),
        },
        "ranking_policy": "full_catalog_exclude_history",
        "model": {
            "type": "bpr_matrix_factorization",
            "embedding_dim": int(config["embedding_dim"]),
            "learning_rate": float(config["learning_rate"]),
            "l2": float(config["l2"]),
            "batch_size": int(config["batch_size"]),
            "max_epochs": int(config["max_epochs"]),
            "eval_every": int(config["eval_every"]),
            "patience": int(config["patience"]),
            "selection_metric": str(config["selection_metric"]),
            "test_representation_policy": (
                "static_user_embedding_from_training_prefix; "
                "validation_item_masked_as_observed_for_test_candidates"
            ),
            "negative_sampling_policy": (
                "uniform_from_items_not_in_training_prefix; held-out future "
                "items are not excluded"
            ),
        },
        "training": training,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_004 — Exploratory BPR Matrix Factorization",
        "",
        f"- Code commit: {metrics['code_commit']}",
        f"- Users: {len(splits):,}",
        f"- Training events: {dataset.positive_users.numel():,}",
        f"- Embedding dimension: {config['embedding_dim']}",
        f"- Best epoch: {training['best_epoch']}",
        f"- Epochs run: {training['epochs_ran']}",
        (
            "- Validation/test held-out items sampled as negatives during "
            f"training: {training['validation_target_negative_samples']:,}/"
            f"{training['test_target_negative_samples']:,}"
        ),
        "",
        "## Validation",
        "",
        _format_metrics(validation_metrics),
        "",
        "## Test",
        "",
        _format_metrics(test_metrics),
        "",
        "## Protocol note",
        "",
        (
            "Negative sampling excludes only training-prefix positives. "
            "Future validation/test items remain unobserved at training time "
            "and are therefore eligible negatives; excluding them would leak "
            "future information."
        ),
        (
            "For test ranking, the validation course is masked as already "
            "observed, but the static user embedding is not updated from it."
        ),
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_004_bpr_mf_pilot")
    print(f"Users: {len(splits):,}")
    print(f"Training events: {dataset.positive_users.numel():,}")
    print(
        "Best epoch / epochs run: "
        f"{training['best_epoch']}/{training['epochs_ran']}"
    )
    print(
        "Best validation "
        f"{training['selection_metric']}: "
        f"{training['best_validation_metric']:.6f}"
    )
    print(
        "Held-out validation/test items sampled as negatives: "
        f"{training['validation_target_negative_samples']:,}/"
        f"{training['test_target_negative_samples']:,}"
    )
    print("Validation: " + _format_metrics(validation_metrics))
    print("Test: " + _format_metrics(test_metrics))
    print(f"Metrics written to: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
