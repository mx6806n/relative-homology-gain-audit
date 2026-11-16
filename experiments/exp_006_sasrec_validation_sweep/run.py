"""Run exp_006: validation-only SASRec hyperparameter sweep."""

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


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    results_path = output_dir / "validation_results.json"
    selected_path = output_dir / "selected_config.json"
    summary_path = output_dir / "summary.md"

    if (
        results_path.exists()
        or selected_path.exists()
        or summary_path.exists()
    ):
        raise FileExistsError(
            "exp_006 output already exists; refusing to overwrite evidence. "
            f"Existing output directory: {output_dir}"
        )

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
    seed = int(config["seed"])
    cutoffs = tuple(int(k) for k in config["cutoffs"])
    selection_metric = str(config["selection_metric"])
    device = torch.device("cpu")

    raw_candidates = config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty list")

    candidate_results: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = str(raw_candidate["id"])
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen_ids.add(candidate_id)

        model, training = train_sasrec(
            dataset,
            max_seq_len=int(config["max_seq_len"]),
            embedding_dim=int(raw_candidate["embedding_dim"]),
            num_heads=int(raw_candidate["num_heads"]),
            num_layers=int(raw_candidate["num_layers"]),
            dropout=float(raw_candidate["dropout"]),
            learning_rate=float(raw_candidate["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            eval_batch_size=int(config["eval_batch_size"]),
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
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
            batch_size=int(config["eval_batch_size"]),
            device=device,
        )

        result = {
            "id": candidate_id,
            "config": {
                "max_seq_len": int(config["max_seq_len"]),
                "embedding_dim": int(raw_candidate["embedding_dim"]),
                "num_heads": int(raw_candidate["num_heads"]),
                "num_layers": int(raw_candidate["num_layers"]),
                "dropout": float(raw_candidate["dropout"]),
                "learning_rate": float(raw_candidate["learning_rate"]),
                "weight_decay": float(config["weight_decay"]),
                "batch_size": int(config["batch_size"]),
                "eval_batch_size": int(config["eval_batch_size"]),
                "max_epochs": int(config["max_epochs"]),
                "patience": int(config["patience"]),
                "selection_metric": selection_metric,
            },
            "parameter_count": _parameter_count(model),
            "best_epoch": int(training["best_epoch"]),
            "epochs_ran": int(training["epochs_ran"]),
            "best_validation_metric": float(
                training["best_validation_metric"]
            ),
            "truncated_train_sequences": int(
                training["truncated_train_sequences"]
            ),
            "validation": validation,
        }
        candidate_results.append(result)

        print(
            f"{candidate_id}: "
            f"{selection_metric}={result['best_validation_metric']:.6f}, "
            f"best_epoch={result['best_epoch']}, "
            f"epochs_ran={result['epochs_ran']}, "
            f"params={result['parameter_count']:,}"
        )

    candidate_results.sort(
        key=lambda result: (
            -float(result["best_validation_metric"]),
            str(result["id"]),
        )
    )
    selected = candidate_results[0]

    output: dict[str, object] = {
        "experiment_id": "exp_006_sasrec_validation_sweep",
        "status": "completed",
        "phase": "baseline_tuning",
        "parent_experiment": "exp_005_sasrec_pilot",
        "code_commit": _git_commit(),
        "seed": seed,
        "device": "cpu",
        "test_evaluated": False,
        "selection_metric": selection_metric,
        "data": {
            "sequences_path": str(sequences_path.relative_to(REPO_ROOT)),
            "sequences_sha256": _sha256(sequences_path),
            "course_vocab_path": str(vocab_path.relative_to(REPO_ROOT)),
            "course_vocab_sha256": _sha256(vocab_path),
            "candidate_courses": len(universe),
            "users": len(splits),
        },
        "candidates": candidate_results,
        "selected_candidate_id": selected["id"],
        "selected_validation_metric": selected[
            "best_validation_metric"
        ],
    }

    selected_config = {
        "source_experiment": "exp_006_sasrec_validation_sweep",
        "selection_metric": selection_metric,
        "selected_candidate_id": selected["id"],
        "seed_used_for_selection": seed,
        "test_evaluated_during_selection": False,
        "config": selected["config"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(selected_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_006 — SASRec Validation-Only Hyperparameter Sweep",
        "",
        f"- Code commit: {output['code_commit']}",
        f"- Users: {len(splits):,}",
        f"- Candidate configurations: {len(candidate_results)}",
        f"- Selection metric: {selection_metric}",
        "- Test evaluated during sweep: NO",
        "",
        "## Validation ranking",
        "",
    ]
    for rank, result in enumerate(candidate_results, start=1):
        summary_lines.append(
            f"{rank}. {result['id']}: "
            f"{selection_metric}="
            f"{result['best_validation_metric']:.6f}; "
            f"best_epoch={result['best_epoch']}; "
            f"params={result['parameter_count']:,}"
        )

    summary_lines.extend(
        [
            "",
            "## Selected configuration",
            "",
            f"- ID: {selected['id']}",
            (
                f"- Validation {selection_metric}: "
                f"{selected['best_validation_metric']:.6f}"
            ),
            "",
            (
                "The development-test split was not evaluated for any sweep "
                "candidate. The selected configuration is therefore based "
                "only on validation evidence."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Selected: "
        f"{selected['id']} with validation "
        f"{selection_metric}={selected['best_validation_metric']:.6f}"
    )
    print("Test evaluated: False")
    print(f"Results written to: {results_path}")
    print(f"Selected config written to: {selected_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
