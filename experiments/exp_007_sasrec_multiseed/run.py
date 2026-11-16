"""Run exp_007: five-seed confirmation of the selected SASRec config."""

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

from relhom.experiment import (  # noqa: E402
    analyze_repeats,
    load_course_universe,
    load_leave_two_out_splits,
)
from relhom.result_stats import summarize_seed_metrics  # noqa: E402
from relhom.sasrec import (  # noqa: E402
    build_sasrec_dataset,
    evaluate_sasrec_model,
    train_sasrec,
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


def _format_aggregate(
    aggregate: dict[str, dict[str, float | int]],
    metric: str,
) -> str:
    values = aggregate[metric]
    return (
        f"{metric}: mean={values['mean']:.6f}, "
        f"std={values['std']:.6f}, "
        f"95% CI=[{values['ci95_low']:.6f}, "
        f"{values['ci95_high']:.6f}]"
    )


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(
            "exp_007 output already exists; refusing to overwrite evidence. "
            f"Existing output directory: {output_dir}"
        )

    raw_seeds = config.get("seeds")
    if not isinstance(raw_seeds, list) or len(raw_seeds) < 2:
        raise ValueError("seeds must be a list with at least two values")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")

    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("model config must be an object")

    cutoffs = tuple(int(k) for k in config["cutoffs"])
    selection_metric = str(model_config["selection_metric"])
    device = torch.device("cpu")

    sequences_path = data_dir / "sequences.jsonl"
    vocab_path = data_dir / "course_vocab.json"
    universe = load_course_universe(vocab_path)
    splits = load_leave_two_out_splits(
        sequences_path,
        min_sequence_length=int(config["min_sequence_length"]),
        item_universe=universe,
    )
    repeats = analyze_repeats(splits)
    if (
        repeats.validation_targets_seen_in_train
        or repeats.test_targets_seen_in_history
    ):
        raise ValueError(
            "unseen-course evaluation requires held-out targets not to occur "
            "earlier in history"
        )

    dataset = build_sasrec_dataset(splits, universe)

    seed_runs: list[dict[str, object]] = []
    validation_runs: list[dict[str, float | int]] = []
    development_test_runs: list[dict[str, float | int]] = []

    for seed in seeds:
        model, training = train_sasrec(
            dataset,
            max_seq_len=int(model_config["max_seq_len"]),
            embedding_dim=int(model_config["embedding_dim"]),
            num_heads=int(model_config["num_heads"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            learning_rate=float(model_config["learning_rate"]),
            weight_decay=float(model_config["weight_decay"]),
            batch_size=int(model_config["batch_size"]),
            eval_batch_size=int(model_config["eval_batch_size"]),
            max_epochs=int(model_config["max_epochs"]),
            patience=int(model_config["patience"]),
            seed=seed,
            cutoffs=cutoffs,
            selection_metric=selection_metric,
            device=device,
        )

        validation = evaluate_sasrec_model(
            model,
            dataset,
            target="validation",
            cutoffs=cutoffs,
            batch_size=int(model_config["eval_batch_size"]),
            device=device,
        )
        development_test = evaluate_sasrec_model(
            model,
            dataset,
            target="test",
            cutoffs=cutoffs,
            batch_size=int(model_config["eval_batch_size"]),
            device=device,
        )

        validation_runs.append(validation)
        development_test_runs.append(development_test)
        seed_runs.append(
            {
                "seed": seed,
                "best_epoch": int(training["best_epoch"]),
                "epochs_ran": int(training["epochs_ran"]),
                "best_validation_metric": float(
                    training["best_validation_metric"]
                ),
                "validation": validation,
                "development_test": development_test,
            }
        )

        print(
            f"seed={seed}: "
            f"best_epoch={training['best_epoch']}, "
            f"val_ndcg@10={validation['ndcg@10']:.6f}, "
            f"devtest_ndcg@10={development_test['ndcg@10']:.6f}"
        )

    metric_names = (
        "mrr",
        "recall@5",
        "ndcg@5",
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
    )
    validation_aggregate = summarize_seed_metrics(
        validation_runs,
        metric_names,
    )
    development_test_aggregate = summarize_seed_metrics(
        development_test_runs,
        metric_names,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_007_sasrec_multiseed",
        "status": "completed",
        "phase": "baseline_confirmation",
        "parent_experiment": "exp_006_sasrec_validation_sweep",
        "code_commit": _git_commit(),
        "seeds": list(seeds),
        "selection_seed_excluded": 42 not in seeds,
        "device": "cpu",
        "torch_version": torch.__version__,
        "data": {
            "sequences_path": str(sequences_path.relative_to(REPO_ROOT)),
            "sequences_sha256": _sha256(sequences_path),
            "course_vocab_path": str(vocab_path.relative_to(REPO_ROOT)),
            "course_vocab_sha256": _sha256(vocab_path),
            "candidate_courses": len(universe),
            "users": len(splits),
            "train_events": sum(len(split.train) for split in splits),
        },
        "ranking_policy": "full_catalog_exclude_history",
        "model": {
            "type": "sasrec_style_causal_transformer",
            "selected_candidate_id": "wide_128x2_d20_lr1e3",
            "training_objective": (
                "full_softmax_next_item_cross_entropy"
            ),
            **model_config,
            "parameter_count": parameter_count,
        },
        "seed_runs": seed_runs,
        "validation_aggregate": validation_aggregate,
        "development_test_aggregate": development_test_aggregate,
        "uncertainty_note": (
            "95% Student-t intervals summarize variability across the five "
            "training seeds; they are not user-level confidence intervals."
        ),
        "evidence_status": (
            "development_test is development evidence, not final confirmatory "
            "evidence"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_007 - SASRec Five-Seed Stage-1 Confirmation",
        "",
        f"- Code commit: {metrics['code_commit']}",
        f"- Seeds: {list(seeds)}",
        "- Hyperparameters: frozen from exp_006 validation-only selection",
        "- Selection seed 42 reused: NO",
        f"- Users: {len(splits):,}",
        f"- Training events: {metrics['data']['train_events']:,}",
        f"- Model parameters: {parameter_count:,}",
        "",
        "## Per-seed results",
        "",
    ]
    for result in seed_runs:
        summary_lines.append(
            f"- seed {result['seed']}: "
            f"best_epoch={result['best_epoch']}, "
            f"val NDCG@10={result['validation']['ndcg@10']:.6f}, "
            f"dev-test NDCG@10="
            f"{result['development_test']['ndcg@10']:.6f}"
        )

    summary_lines.extend(
        [
            "",
            "## Validation aggregate",
            "",
        ]
    )
    for metric in metric_names:
        summary_lines.append(
            "- " + _format_aggregate(validation_aggregate, metric)
        )

    summary_lines.extend(
        [
            "",
            "## Development-test aggregate",
            "",
        ]
    )
    for metric in metric_names:
        summary_lines.append(
            "- "
            + _format_aggregate(
                development_test_aggregate,
                metric,
            )
        )

    summary_lines.extend(
        [
            "",
            "## Evidence note",
            "",
            (
                "95% intervals are Student-t intervals across five random "
                "training seeds. The CS development-test split is not treated "
                "as pristine confirmatory evidence."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nValidation aggregate:")
    for metric in ("mrr", "recall@10", "ndcg@10"):
        print(_format_aggregate(validation_aggregate, metric))

    print("\nDevelopment-test aggregate:")
    for metric in ("mrr", "recall@10", "ndcg@10"):
        print(_format_aggregate(development_test_aggregate, metric))

    print(f"Metrics written to: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
