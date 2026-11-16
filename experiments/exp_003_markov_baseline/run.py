"""Run exp_003: train-only first-order Markov sequential baseline."""

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
    evaluate_markov_ranking,
    fit_first_order_markov,
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
            "exp_003 output already exists; refusing to overwrite evidence. "
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

    transitions, popularity, transition_events = fit_first_order_markov(
        splits,
        universe,
    )

    validation_metrics = evaluate_markov_ranking(
        splits,
        universe,
        transitions,
        popularity,
        target="validation",
        cutoffs=cutoffs,
    )
    test_metrics = evaluate_markov_ranking(
        splits,
        universe,
        transitions,
        popularity,
        target="test",
        cutoffs=cutoffs,
    )

    train_events = sum(len(split.train) for split in splits)
    source_courses = len(transitions)
    unique_transition_edges = sum(
        len(counter)
        for counter in transitions.values()
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_003_markov_baseline",
        "status": "completed",
        "parent_experiment": "exp_002_popularity_unseen",
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
        },
        "ranking_policy": "full_catalog_exclude_history",
        "model": {
            "type": "first_order_markov_with_popularity_backoff",
            "transition_fit_scope": "training_prefixes_only",
            "transition_events": transition_events,
            "source_courses_with_outgoing": source_courses,
            "unique_transition_edges": unique_transition_edges,
        },
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
            "items_observed_in_train": sum(
                1 for item in universe if popularity[item] > 0
            ),
            "zero_frequency_items": sum(
                1 for item in universe if popularity[item] == 0
            ),
            "cold_validation_targets": count_cold_targets(
                popularity,
                splits,
                target="validation",
            ),
            "cold_test_targets": count_cold_targets(
                popularity,
                splits,
                target="test",
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_003 — First-Order Markov Sequential Baseline",
        "",
        f"- Parent: {metrics['parent_experiment']}",
        f"- Code commit: {metrics['code_commit']}",
        f"- Users: {len(splits):,}",
        f"- Training events: {train_events:,}",
        f"- Training transition events: {transition_events:,}",
        f"- Unique transition edges: {unique_transition_edges:,}",
        f"- Transition source courses: {source_courses:,}",
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
            "Transition and popularity counts are fit only from training "
            "prefixes. At test time the validation course is available as "
            "the latest observed context but is not added to fitted counts."
        ),
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_003_markov_baseline")
    print(f"Users: {len(splits):,}")
    print(f"Training events: {train_events:,}")
    print(f"Training transition events: {transition_events:,}")
    print(f"Unique transition edges: {unique_transition_edges:,}")
    print(
        "Validation context/target transition support: "
        f"{validation_metrics['contexts_with_train_outgoing']:,}/"
        f"{validation_metrics['targets_with_observed_transition']:,}"
    )
    print(
        "Test context/target transition support: "
        f"{test_metrics['contexts_with_train_outgoing']:,}/"
        f"{test_metrics['targets_with_observed_transition']:,}"
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
