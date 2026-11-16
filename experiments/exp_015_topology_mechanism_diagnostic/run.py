"""Run exp_015: diagnose what drives cached Delta beta_1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.experiment import (  # noqa: E402
    load_course_universe,
    load_leave_two_out_splits,
)
from relhom.hybrid_cache import validate_hybrid_cache_alignment  # noqa: E402
from relhom.sasrec_score_cache import (  # noqa: E402
    load_sasrec_score_cache,
    sha256_file as sasrec_sha256,
)
from relhom.topology_cache import (  # noqa: E402
    load_validation_topology_cache,
    sha256_file as topology_sha256,
)
from relhom.topology_mechanism import (  # noqa: E402
    candidate_structural_features,
    correlation_summary,
    linear_structural_r2,
    target_vs_candidates,
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_graph_and_mapping(
    data_dir: Path,
) -> tuple[nx.DiGraph, dict[str, set[str]]]:
    concept_vocab = json.loads(
        (data_dir / "concept_vocab.json").read_text(encoding="utf-8")
    )
    ordered = [
        concept
        for concept, _ in sorted(
            concept_vocab.items(),
            key=lambda item: int(item[1]),
        )
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(ordered)

    with (data_dir / "prerequisite_edges.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValueError("unexpected prerequisite CSV header")
        for row in reader:
            graph.add_edge(row["source"], row["target"])

    raw_mapping = json.loads(
        (data_dir / "course_to_concepts.json").read_text(encoding="utf-8")
    )
    mapping = {
        str(course): set(map(str, concepts))
        for course, concepts in raw_mapping.items()
    }
    return graph, mapping


def _corr_dict(summary) -> dict[str, object]:
    return asdict(summary)


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    topology_path = REPO_ROOT / str(config["topology_cache_path"])
    sasrec_path = REPO_ROOT / str(config["sasrec_cache_path"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_015 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    expected_topology_sha = str(
        config["expected_topology_cache_sha256"]
    )
    expected_sasrec_sha = str(
        config["expected_sasrec_cache_sha256"]
    )
    actual_topology_sha = topology_sha256(topology_path)
    actual_sasrec_sha = sasrec_sha256(sasrec_path)
    if actual_topology_sha != expected_topology_sha:
        raise RuntimeError("topology cache SHA-256 mismatch")
    if actual_sasrec_sha != expected_sasrec_sha:
        raise RuntimeError("SASRec cache SHA-256 mismatch")

    universe = load_course_universe(data_dir / "course_vocab.json")
    splits = load_leave_two_out_splits(
        data_dir / "sequences.jsonl",
        min_sequence_length=int(config["min_sequence_length"]),
        item_universe=universe,
    )
    graph, mapping = _load_graph_and_mapping(data_dir)

    topology = load_validation_topology_cache(topology_path)
    sasrec = load_sasrec_score_cache(sasrec_path)
    validate_hybrid_cache_alignment(topology, sasrec)

    if list(map(str, topology["course_ids"].tolist())) != list(universe):
        raise RuntimeError("course ordering mismatch")
    if list(map(str, topology["user_ids"].tolist())) != [
        split.user_id for split in splits
    ]:
        raise RuntimeError("user ordering mismatch")

    deltas = topology["deltas"].astype(np.int32, copy=False)
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen_mask = deltas == sentinel
    logits = sasrec["logits"].astype(np.float64, copy=False)
    targets = topology["validation_target_indices"].astype(
        np.int64,
        copy=False,
    )

    features = candidate_structural_features(
        splits,
        universe,
        mapping,
        graph,
        seen_mask,
    )

    candidate_mask = ~seen_mask
    abs_delta = np.abs(deltas)

    correlations = {
        "delta_vs_bundle_size": _corr_dict(
            correlation_summary(
                deltas,
                features["bundle_size"],
                seen_mask,
            )
        ),
        "delta_vs_new_concept_count": _corr_dict(
            correlation_summary(
                deltas,
                features["new_concept_count"],
                seen_mask,
            )
        ),
        "abs_delta_vs_bundle_size": _corr_dict(
            correlation_summary(
                abs_delta,
                features["bundle_size"],
                seen_mask,
            )
        ),
        "abs_delta_vs_new_concept_count": _corr_dict(
            correlation_summary(
                abs_delta,
                features["new_concept_count"],
                seen_mask,
            )
        ),
        "delta_vs_unmet_direct_count": _corr_dict(
            correlation_summary(
                deltas,
                features["unmet_direct_count"],
                seen_mask,
            )
        ),
        "delta_vs_unmet_ancestor_count": _corr_dict(
            correlation_summary(
                deltas,
                features["unmet_ancestor_count"],
                seen_mask,
            )
        ),
        "delta_vs_sasrec_logit": _corr_dict(
            correlation_summary(
                deltas,
                logits,
                seen_mask,
            )
        ),
    }

    structural_r2 = linear_structural_r2(
        deltas,
        (
            features["bundle_size"],
            features["new_concept_count"],
            features["unmet_direct_count"],
            features["unmet_ancestor_count"],
        ),
        seen_mask,
    )

    target_profiles = {
        "new_concept_count": target_vs_candidates(
            deltas,
            features["new_concept_count"],
            seen_mask,
            targets,
        ),
        "bundle_size": target_vs_candidates(
            deltas,
            features["bundle_size"],
            seen_mask,
            targets,
        ),
        "unmet_direct_count": target_vs_candidates(
            deltas,
            features["unmet_direct_count"],
            seen_mask,
            targets,
        ),
        "unmet_ancestor_count": target_vs_candidates(
            deltas,
            features["unmet_ancestor_count"],
            seen_mask,
            targets,
        ),
    }

    course_rows: list[dict[str, object]] = []
    for col, course in enumerate(universe):
        mask = candidate_mask[:, col]
        values = deltas[mask, col].astype(np.float64)
        course_rows.append(
            {
                "course_id": course,
                "candidate_count": int(mask.sum()),
                "bundle_size": int(features["bundle_size"][0, col]),
                "mean_delta_beta1": float(values.mean()),
                "median_delta_beta1": float(np.median(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "zero_fraction": float(np.mean(values == 0)),
                "negative_fraction": float(np.mean(values < 0)),
            }
        )

    course_mean_delta = np.asarray(
        [row["mean_delta_beta1"] for row in course_rows],
        dtype=np.float64,
    )
    course_bundle_size = np.asarray(
        [row["bundle_size"] for row in course_rows],
        dtype=np.float64,
    )
    from scipy.stats import spearmanr

    if np.unique(course_bundle_size).size > 1:
        course_level_bundle_corr = float(
            spearmanr(course_mean_delta, course_bundle_size).statistic
        )
    else:
        course_level_bundle_corr = None

    payload: dict[str, object] = {
        "experiment_id": "exp_015_topology_mechanism_diagnostic",
        "status": "completed",
        "phase": "mechanism_diagnostic",
        "evidence_scope": "validation_only",
        "inputs": {
            "users": len(splits),
            "courses": len(universe),
            "candidate_entries": int(candidate_mask.sum()),
            "topology_cache_sha256": actual_topology_sha,
            "sasrec_cache_sha256": actual_sasrec_sha,
            "sequences_sha256": _sha256(data_dir / "sequences.jsonl"),
            "course_to_concepts_sha256": _sha256(
                data_dir / "course_to_concepts.json"
            ),
            "prerequisite_edges_sha256": _sha256(
                data_dir / "prerequisite_edges.csv"
            ),
        },
        "correlations": correlations,
        "linear_structural_count_r2": structural_r2,
        "target_profiles": target_profiles,
        "course_level": {
            "mean_delta_vs_bundle_size_spearman": (
                course_level_bundle_corr
            ),
            "courses": course_rows,
        },
        "signal": {
            "candidate_delta_mean": float(
                deltas[candidate_mask].mean()
            ),
            "candidate_delta_median": float(
                np.median(deltas[candidate_mask])
            ),
            "candidate_abs_delta_mean": float(
                abs_delta[candidate_mask].mean()
            ),
        },
        "selection_performed": False,
        "new_reranker_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    def corr_line(name: str) -> str:
        item = correlations[name]
        return (
            f"global={item['global_spearman']}, "
            f"per-user median={item['per_user_median']}, "
            f"defined users={item['per_user_defined']:,}"
        )

    summary_lines = [
        "# exp_015 - Topology Mechanism Diagnostic",
        "",
        "- Evidence scope: validation only",
        f"- Users: {len(splits):,}",
        f"- Courses: {len(universe):,}",
        f"- Unseen candidate states: {int(candidate_mask.sum()):,}",
        "- Selection performed: NO",
        "- New reranker evaluated: NO",
        "- Development-test evaluated: NO",
        "",
        "## What correlates with signed Delta beta_1?",
        "",
        (
            "- Delta vs course bundle size: "
            + corr_line("delta_vs_bundle_size")
        ),
        (
            "- Delta vs new-concept count: "
            + corr_line("delta_vs_new_concept_count")
        ),
        (
            "- |Delta| vs course bundle size: "
            + corr_line("abs_delta_vs_bundle_size")
        ),
        (
            "- |Delta| vs new-concept count: "
            + corr_line("abs_delta_vs_new_concept_count")
        ),
        (
            "- Delta vs unmet direct-prerequisite count: "
            + corr_line("delta_vs_unmet_direct_count")
        ),
        (
            "- Delta vs unmet ancestor-prerequisite count: "
            + corr_line("delta_vs_unmet_ancestor_count")
        ),
        (
            "- Delta vs SASRec raw logit: "
            + corr_line("delta_vs_sasrec_logit")
        ),
        "",
        "## Structural-count explanation",
        "",
        (
            "- OLS R^2 using bundle size, new-concept count, unmet direct "
            f"count, and unmet ancestor count: {structural_r2:.6f}"
        ),
        (
            "- Course-level Spearman(mean Delta, bundle size): "
            f"{course_level_bundle_corr}"
        ),
        "",
        "## Target diagnostics",
        "",
    ]

    for name, profile in target_profiles.items():
        summary_lines.append(
            f"- {name}: target feature mean="
            f"{profile['target_feature_mean']:.4f}, "
            f"mean feature weak percentile="
            f"{profile['mean_target_feature_weak_percentile']:.4f}, "
            f"mean Delta weak percentile="
            f"{profile['mean_target_signal_weak_percentile']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            (
                "This experiment diagnoses the existing topology signal only. "
                "It does not justify a normalized or residualized reranker on "
                "the same CS validation data. Any new formulation motivated "
                "by these diagnostics must be specified before testing it in "
                "a fresh setting."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_015_topology_mechanism_diagnostic")
    print(
        "Delta vs bundle/new-concept global Spearman: "
        f"{correlations['delta_vs_bundle_size']['global_spearman']}/"
        f"{correlations['delta_vs_new_concept_count']['global_spearman']}"
    )
    print(
        "|Delta| vs bundle/new-concept global Spearman: "
        f"{correlations['abs_delta_vs_bundle_size']['global_spearman']}/"
        f"{correlations['abs_delta_vs_new_concept_count']['global_spearman']}"
    )
    print(
        "Delta vs SASRec logit global Spearman: "
        f"{correlations['delta_vs_sasrec_logit']['global_spearman']}"
    )
    print(f"Structural-count OLS R^2: {structural_r2:.6f}")
    print("Selection performed: False")
    print("Development-test evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
