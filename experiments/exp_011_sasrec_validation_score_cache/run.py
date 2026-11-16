"""Run exp_011: frozen SASRec validation-score cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.experiment import (  # noqa: E402
    load_course_universe,
    load_leave_two_out_splits,
)
from relhom.sasrec import (  # noqa: E402
    build_sasrec_dataset,
    evaluate_sasrec_model,
    train_sasrec,
)
from relhom.sasrec_score_cache import (  # noqa: E402
    evaluate_cached_validation_scores,
    load_sasrec_score_cache,
    score_validation_logits,
    sha256_file,
    validate_alignment,
    validate_sasrec_score_cache,
    write_sasrec_score_cache,
)
from relhom.topology_cache import (  # noqa: E402
    TOPOLOGY_SENTINEL,
    load_validation_topology_cache,
    validate_topology_cache,
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


def _assert_metric_close(
    observed: dict[str, float | int],
    expected: dict[str, float | int],
    metric_names: tuple[str, ...],
    *,
    tolerance: float,
    context: str,
) -> None:
    for metric in metric_names:
        difference = abs(
            float(observed[metric]) - float(expected[metric])
        )
        if difference > tolerance:
            raise RuntimeError(
                f"{context} metric mismatch for {metric}: "
                f"observed={observed[metric]}, "
                f"expected={expected[metric]}, "
                f"difference={difference}"
            )


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    topology_cache_path = (
        REPO_ROOT / str(config["topology_cache_path"])
    )
    output_dir = REPO_ROOT / str(config["output_dir"])
    score_cache_path = output_dir / "validation_sasrec_scores.npz"
    manifest_path = output_dir / "cache_manifest.json"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (
        score_cache_path,
        manifest_path,
        metrics_path,
        summary_path,
    ):
        if path.exists():
            raise FileExistsError(
                "exp_011 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    expected_topology_sha = str(
        config["expected_topology_cache_sha256"]
    )
    actual_topology_sha = sha256_file(topology_cache_path)
    if actual_topology_sha != expected_topology_sha:
        raise RuntimeError(
            "topology cache SHA-256 mismatch: "
            f"{actual_topology_sha} != {expected_topology_sha}"
        )

    sequences_path = data_dir / "sequences.jsonl"
    course_vocab_path = data_dir / "course_vocab.json"
    universe = load_course_universe(course_vocab_path)
    splits = load_leave_two_out_splits(
        sequences_path,
        min_sequence_length=int(config["min_sequence_length"]),
        item_universe=universe,
    )

    topology_cache = load_validation_topology_cache(
        topology_cache_path
    )
    validate_topology_cache(
        topology_cache,
        splits,
        universe,
    )

    dataset = build_sasrec_dataset(splits, universe)
    validate_alignment(dataset, topology_cache)

    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise ValueError("model config must be an object")
    seed = int(config["seed"])
    device = torch.device("cpu")
    cutoffs = tuple(int(k) for k in config["cutoffs"])

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
        selection_metric=str(model_config["selection_metric"]),
        device=device,
    )

    direct_metrics = evaluate_sasrec_model(
        model,
        dataset,
        target="validation",
        cutoffs=cutoffs,
        batch_size=int(model_config["eval_batch_size"]),
        device=device,
    )

    expected_metrics = config["expected_validation_metrics"]
    if not isinstance(expected_metrics, dict):
        raise ValueError(
            "expected_validation_metrics must be an object"
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
    tolerance = float(config["reproduction_tolerance"])
    _assert_metric_close(
        direct_metrics,
        expected_metrics,
        metric_names,
        tolerance=tolerance,
        context="exp006 reproduction",
    )

    expected_best_epoch = int(config["expected_best_epoch"])
    if int(training["best_epoch"]) != expected_best_epoch:
        raise RuntimeError(
            "best epoch does not reproduce exp006: "
            f"{training['best_epoch']} != {expected_best_epoch}"
        )

    logits, truncated_histories = score_validation_logits(
        model,
        dataset,
        batch_size=int(model_config["eval_batch_size"]),
        device=device,
    )
    cached_metrics = evaluate_cached_validation_scores(
        logits,
        topology_cache,
        cutoffs=cutoffs,
    )
    _assert_metric_close(
        cached_metrics,
        direct_metrics,
        metric_names,
        tolerance=1e-12,
        context="direct-vs-cache",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_sasrec_score_cache(
        score_cache_path,
        logits,
        dataset,
        seed=seed,
        topology_cache_sha256=actual_topology_sha,
    )
    reloaded = load_sasrec_score_cache(score_cache_path)
    validate_sasrec_score_cache(
        reloaded,
        dataset,
        expected_seed=seed,
        expected_topology_cache_sha256=actual_topology_sha,
    )
    reloaded_metrics = evaluate_cached_validation_scores(
        reloaded["logits"],
        topology_cache,
        cutoffs=cutoffs,
    )
    _assert_metric_close(
        reloaded_metrics,
        direct_metrics,
        metric_names,
        tolerance=1e-12,
        context="reloaded-cache",
    )

    seen_mask = (
        topology_cache["deltas"] == int(TOPOLOGY_SENTINEL)
    )
    seen_entries = int(seen_mask.sum())
    candidate_entries = int((~seen_mask).sum())
    score_cache_sha = sha256_file(score_cache_path)

    data_hashes = {
        "sequences_sha256": _sha256(sequences_path),
        "course_vocab_sha256": _sha256(course_vocab_path),
    }

    manifest = {
        "experiment_id": "exp_011_sasrec_validation_score_cache",
        "cache_file": score_cache_path.name,
        "cache_sha256": score_cache_sha,
        "topology_cache": {
            "path": str(topology_cache_path.relative_to(REPO_ROOT)),
            "sha256": actual_topology_sha,
        },
        "data_hashes": data_hashes,
        "model": {
            "selected_candidate_id": (
                "wide_128x2_d20_lr1e3"
            ),
            "seed": seed,
            **model_config,
            "best_epoch": int(training["best_epoch"]),
        },
        "arrays": {
            "logits": {
                "dtype": str(reloaded["logits"].dtype),
                "shape": list(reloaded["logits"].shape),
            },
            "validation_target_indices": {
                "dtype": str(
                    reloaded["validation_target_indices"].dtype
                ),
                "shape": list(
                    reloaded["validation_target_indices"].shape
                ),
            },
            "user_ids": {
                "dtype": str(reloaded["user_ids"].dtype),
                "shape": list(reloaded["user_ids"].shape),
            },
            "course_ids": {
                "dtype": str(reloaded["course_ids"].dtype),
                "shape": list(reloaded["course_ids"].shape),
            },
        },
        "alignment": {
            "users": len(dataset.user_ids),
            "courses": len(dataset.item_ids),
            "seen_entries_from_topology_mask": seen_entries,
            "candidate_entries": candidate_entries,
            "topology_alignment_passed": True,
            "direct_vs_cached_metrics_passed": True,
            "exp006_reproduction_passed": True,
        },
        "development_test_evaluated": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_011_sasrec_validation_score_cache",
        "status": "completed",
        "phase": "hybrid_input_cache",
        "code_commit": _git_commit(),
        "seed": seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "data_hashes": data_hashes,
        "topology_cache_sha256": actual_topology_sha,
        "score_cache": {
            "path": str(score_cache_path.relative_to(REPO_ROOT)),
            "sha256": score_cache_sha,
            "bytes": score_cache_path.stat().st_size,
        },
        "training": {
            "best_epoch": int(training["best_epoch"]),
            "epochs_ran": int(training["epochs_ran"]),
            "best_validation_metric": float(
                training["best_validation_metric"]
            ),
            "truncated_train_sequences": int(
                training["truncated_train_sequences"]
            ),
        },
        "score_matrix": {
            "users": len(dataset.user_ids),
            "courses": len(dataset.item_ids),
            "dtype": str(logits.dtype),
            "score_min": float(logits.min()),
            "score_max": float(logits.max()),
            "seen_entries": seen_entries,
            "candidate_entries": candidate_entries,
            "truncated_histories": truncated_histories,
        },
        "validation": direct_metrics,
        "cached_validation": cached_metrics,
        "checks": {
            "topology_cache_hash": True,
            "topology_alignment": True,
            "exp006_metric_reproduction": True,
            "expected_best_epoch": True,
            "direct_vs_cache_metrics": True,
            "reload_validation": True,
        },
        "development_test_evaluated": False,
        "alpha_tuned": False,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_011 - Frozen SASRec Validation Score Cache",
        "",
        f"- Seed: {seed}",
        "- Model: frozen wide_128x2_d20_lr1e3",
        f"- Best epoch: {training['best_epoch']}",
        f"- Epochs run: {training['epochs_ran']}",
        f"- Users: {len(dataset.user_ids):,}",
        f"- Courses: {len(dataset.item_ids):,}",
        f"- Candidate entries: {candidate_entries:,}",
        f"- Seen/masked entries: {seen_entries:,}",
        f"- Score-cache bytes: {score_cache_path.stat().st_size:,}",
        f"- Score-cache SHA-256: {score_cache_sha}",
        f"- Topology-cache SHA-256: {actual_topology_sha}",
        "",
        "## Reproduction checks",
        "",
        "- exp_006 validation metrics reproduced: PASS",
        f"- Expected best epoch {expected_best_epoch}: PASS",
        "- Direct model vs cached-score metrics: PASS",
        "- Reloaded cache metrics: PASS",
        "- User/course/target topology alignment: PASS",
        "",
        "## Validation metrics",
        "",
        f"- MRR: {direct_metrics['mrr']:.6f}",
        f"- Recall@5: {direct_metrics['recall@5']:.6f}",
        f"- NDCG@5: {direct_metrics['ndcg@5']:.6f}",
        f"- Recall@10: {direct_metrics['recall@10']:.6f}",
        f"- NDCG@10: {direct_metrics['ndcg@10']:.6f}",
        f"- Recall@20: {direct_metrics['recall@20']:.6f}",
        f"- NDCG@20: {direct_metrics['ndcg@20']:.6f}",
        "",
        (
            f"- Truncated validation histories: "
            f"{truncated_histories:,}"
        ),
        "",
        "No alpha tuning or development-test evaluation is performed.",
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_011_sasrec_validation_score_cache")
    print(
        f"Best epoch / epochs run: "
        f"{training['best_epoch']}/{training['epochs_ran']}"
    )
    print(
        "Validation NDCG@10 direct/cache: "
        f"{direct_metrics['ndcg@10']:.6f}/"
        f"{cached_metrics['ndcg@10']:.6f}"
    )
    print(
        "Candidate / seen entries: "
        f"{candidate_entries:,}/{seen_entries:,}"
    )
    print(f"Score-cache bytes: {score_cache_path.stat().st_size:,}")
    print(f"Score-cache SHA-256: {score_cache_sha}")
    print("exp006 reproduction: PASS")
    print("Topology alignment: PASS")
    print("Cache reload validation: PASS")
    print("Development-test evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
