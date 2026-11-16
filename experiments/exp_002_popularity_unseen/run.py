"""Run exp_002: popularity baseline with unseen-course full ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.experiment import (  # noqa: E402
    analyze_repeats,
    count_cold_targets,
    evaluate_history_filtered_ranking,
    fit_popularity_ranking,
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
    fields: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            fields.append(f"{key}={value:.6f}")
        else:
            fields.append(f"{key}={value}")
    return ", ".join(fields)


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(
            "exp_002 output already exists; refusing to overwrite evidence. "
            f"Existing output directory: {output_dir}"
        )

    min_sequence_length = int(config["min_sequence_length"])
    cutoffs = tuple(int(k) for k in config["cutoffs"])
    seed = int(config["seed"])
    ranking_policy = str(config["ranking_policy"])

    if ranking_policy != "full_catalog_exclude_history":
        raise ValueError(
            "exp_002 only supports full_catalog_exclude_history"
        )

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
            "unseen-item protocol requires held-out targets not to occur "
            "earlier in history"
        )

    ranking, counts = fit_popularity_ranking(splits, universe)

    validation_metrics = evaluate_history_filtered_ranking(
        ranking,
        splits,
        target="validation",
        cutoffs=cutoffs,
    )
    test_metrics = evaluate_history_filtered_ranking(
        ranking,
        splits,
        target="test",
        cutoffs=cutoffs,
    )

    train_events = sum(len(split.train) for split in splits)
    train_catalog_items = sum(1 for item in universe if counts[item] > 0)

    metrics: dict[str, object] = {
        "experiment_id": "exp_002_popularity_unseen",
        "status": "completed",
        "parent_experiment": "exp_001_popularity_baseline",
        "code_commit": _git_commit(),
        "seed": seed,
        "deterministic": True,
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
            "train_events": train_events,
            "validation_targets": len(splits),
            "test_targets": len(splits),
        },
        "ranking_policy": ranking_policy,
        "repeat_diagnostics": {
            "users": repeats.users,
            "users_with_any_repeat": repeats.users_with_any_repeat,
            "repeated_course_events": repeats.repeated_course_events,
            "validation_targets_seen_in_train": (
                repeats.validation_targets_seen_in_train
            ),
            "test_targets_seen_in_history": (
                repeats.test_targets_seen_in_history
            ),
        },
        "training_catalog": {
            "items_observed_in_train": train_catalog_items,
            "zero_frequency_items": len(universe) - train_catalog_items,
            "cold_validation_targets": count_cold_targets(
                counts,
                splits,
                target="validation",
            ),
            "cold_test_targets": count_cold_targets(
                counts,
                splits,
                target="test",
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "top_20_training_popularity": [
            {"course": course, "count": counts[course]}
            for course in ranking[:20]
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_002 — Popularity Baseline with Unseen-Course Filtering",
        "",
        f"- Parent: {metrics['parent_experiment']}",
        f"- Code commit: {metrics['code_commit']}",
        f"- Selected users: {len(splits):,}",
        f"- Minimum aligned sequence length: {min_sequence_length}",
        f"- Candidate universe: {len(universe)} courses",
        f"- Training events: {train_events:,}",
        f"- Ranking policy: {ranking_policy}",
        "",
        "## Protocol rationale",
        "",
        (
            "exp_001 found zero repeated-course users and zero held-out targets "
            "that appeared earlier in history. Therefore previously observed "
            "courses are removed from each user's candidate list."
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
        "## Evidence note",
        "",
        (
            "Popularity frequencies use training prefixes only. History "
            "filtering is applied only at ranking/evaluation time."
        ),
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_002_popularity_unseen")
    print(f"Users: {len(splits):,}")
    print(f"Training events: {train_events:,}")
    print(
        "Repeated-course users: "
        f"{repeats.users_with_any_repeat:,}/{repeats.users:,}"
    )
    print(
        "Cold validation/test targets: "
        f"{metrics['training_catalog']['cold_validation_targets']:,}/"
        f"{metrics['training_catalog']['cold_test_targets']:,}"
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
