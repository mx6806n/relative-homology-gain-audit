"""Run exp_014: descriptive accuracy-vs-prerequisite-validity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.curriculum_validity import (  # noqa: E402
    course_prerequisite_sets,
    prerequisite_status_for_items,
    reranked_shortlist_indices,
    target_prerequisite_profile,
    violation_metrics,
)
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
from relhom.topn_hybrid import (  # noqa: E402
    stage_one_topn_indices,
    topn_rerank_grid,
)


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON/YAML object")
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
    ordered_concepts = [
        concept
        for concept, _ in sorted(
            concept_vocab.items(),
            key=lambda item: int(item[1]),
        )
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_concepts)

    with (data_dir / "prerequisite_edges.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "target"]:
            raise ValueError(
                "prerequisite_edges.csv must have source,target header"
            )
        for row in reader:
            graph.add_edge(row["source"], row["target"])

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("prerequisite graph must be a DAG")

    raw_mapping = json.loads(
        (data_dir / "course_to_concepts.json").read_text(encoding="utf-8")
    )
    mapping = {
        str(course): set(map(str, concepts))
        for course, concepts in raw_mapping.items()
    }
    return graph, mapping


def _validity_for_ranked(
    ranked: np.ndarray,
    direct_full: np.ndarray,
    ancestor_full: np.ndarray,
) -> dict[str, float]:
    direct = np.take_along_axis(direct_full, ranked, axis=1)
    ancestor = np.take_along_axis(ancestor_full, ranked, axis=1)

    result: dict[str, float] = {}
    for key, value in violation_metrics(
        direct,
        cutoffs=(5, 10),
    ).items():
        result[f"direct_{key}"] = value
    for key, value in violation_metrics(
        ancestor,
        cutoffs=(5, 10),
    ).items():
        result[f"ancestor_{key}"] = value
    return result


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
                "exp_014 output already exists; refusing to overwrite "
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
        raise RuntimeError("processed universe does not match frozen caches")
    if list(map(str, topology["user_ids"].tolist())) != [
        split.user_id for split in splits
    ]:
        raise RuntimeError("processed split order does not match frozen caches")

    deltas = topology["deltas"]
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen_mask = deltas == sentinel
    logits = sasrec["logits"]
    targets = topology["validation_target_indices"].astype(
        np.int64,
        copy=False,
    )

    top_ns = tuple(int(n) for n in config["top_ns"])
    alphas = tuple(float(a) for a in config["alphas"])
    max_top_n = max(top_ns)

    direct_requirements, ancestor_requirements = course_prerequisite_sets(
        graph,
        mapping,
    )

    stage1_top = stage_one_topn_indices(
        logits,
        seen_mask,
        top_n=max_top_n,
    ).astype(np.int64, copy=False)

    direct_status, direct_counts = prerequisite_status_for_items(
        splits,
        universe,
        stage1_top,
        mapping,
        direct_requirements,
    )
    ancestor_status, ancestor_counts = prerequisite_status_for_items(
        splits,
        universe,
        stage1_top,
        mapping,
        ancestor_requirements,
    )

    direct_full = np.zeros_like(seen_mask, dtype=bool)
    ancestor_full = np.zeros_like(seen_mask, dtype=bool)
    rows = np.arange(len(splits))
    direct_full[rows[:, None], stage1_top] = direct_status
    ancestor_full[rows[:, None], stage1_top] = ancestor_status

    baseline_ranked = stage1_top
    baseline_validity = _validity_for_ranked(
        baseline_ranked,
        direct_full,
        ancestor_full,
    )

    direct_target = target_prerequisite_profile(
        splits,
        mapping,
        direct_requirements,
    )
    ancestor_target = target_prerequisite_profile(
        splits,
        mapping,
        ancestor_requirements,
    )

    shortlist_results: list[dict[str, object]] = []
    for top_n in top_ns:
        accuracy = topn_rerank_grid(
            logits,
            deltas,
            seen_mask,
            targets,
            top_n=top_n,
            alphas=alphas,
            cutoffs=(5, 10, 20),
        )
        accuracy_by_alpha = {
            float(result["alpha"]): result
            for result in accuracy["results"]
        }

        rows_for_n: list[dict[str, object]] = []
        for alpha in alphas:
            ranked = reranked_shortlist_indices(
                logits,
                deltas,
                seen_mask,
                top_n=top_n,
                alpha=alpha,
            )
            validity = _validity_for_ranked(
                ranked,
                direct_full,
                ancestor_full,
            )
            accuracy_result = accuracy_by_alpha[alpha]
            accuracy_metrics = accuracy_result["metrics"]
            rows_for_n.append(
                {
                    "top_n": top_n,
                    "alpha": alpha,
                    "accuracy": accuracy_metrics,
                    "validity": validity,
                    "validity_delta_vs_sasrec": {
                        key: validity[key] - baseline_validity[key]
                        for key in baseline_validity
                    },
                }
            )

        shortlist_results.append(
            {
                "top_n": top_n,
                "rows": rows_for_n,
            }
        )

    payload: dict[str, object] = {
        "experiment_id": "exp_014_accuracy_validity_pareto",
        "status": "completed",
        "phase": "mechanism_diagnostic",
        "evidence_scope": "validation_only",
        "prerequisite_semantics": {
            "edge_direction": "source_c1_prerequisite_to_target_c2_postrequisite",
            "source_verification": (
                "official MOOCCubeX docs/prerequisites-en.md"
            ),
            "direct_requirement": (
                "union of immediate graph predecessors of candidate concepts, "
                "excluding concepts inside the same candidate course bundle"
            ),
            "ancestor_requirement": (
                "union of all graph ancestors of candidate concepts, "
                "excluding concepts inside the same candidate course bundle"
            ),
        },
        "input_hashes": {
            "topology_cache_sha256": actual_topology_sha,
            "sasrec_cache_sha256": actual_sasrec_sha,
            "sequences_sha256": _sha256(data_dir / "sequences.jsonl"),
            "course_vocab_sha256": _sha256(data_dir / "course_vocab.json"),
            "course_to_concepts_sha256": _sha256(
                data_dir / "course_to_concepts.json"
            ),
            "prerequisite_edges_sha256": _sha256(
                data_dir / "prerequisite_edges.csv"
            ),
        },
        "baseline_sasrec_validity": baseline_validity,
        "observed_validation_target": {
            "direct": direct_target,
            "ancestor": ancestor_target,
        },
        "shortlist_results": shortlist_results,
        "diagnostics": {
            "mean_direct_required_concepts_in_stage1_top20": float(
                direct_counts.mean()
            ),
            "mean_ancestor_required_concepts_in_stage1_top20": float(
                ancestor_counts.mean()
            ),
        },
        "selection_performed": False,
        "sasrec_retrained": False,
        "topology_recomputed": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_014 - Accuracy vs Prerequisite-Validity Diagnostic",
        "",
        "- Evidence scope: validation only",
        "- Selection performed: NO",
        "- SASRec retrained: NO",
        "- Topology recomputed: NO",
        "- Development-test evaluated: NO",
        "- Edge direction: c1 prerequisite -> c2 post-requisite",
        "",
        "## Observed validation-target prerequisite alignment",
        "",
        (
            "- Direct-prerequisite target violation rate: "
            f"{float(direct_target['violation_rate']):.4f} "
            f"({int(direct_target['violations']):,}/"
            f"{int(direct_target['users']):,})"
        ),
        (
            "- Ancestor-prerequisite target violation rate: "
            f"{float(ancestor_target['violation_rate']):.4f} "
            f"({int(ancestor_target['violations']):,}/"
            f"{int(ancestor_target['users']):,})"
        ),
        "",
        "## Frozen SASRec structural validity",
        "",
        (
            f"- Direct PVR@5: "
            f"{baseline_validity['direct_pvr@5']:.6f}"
        ),
        (
            f"- Direct PVR@10: "
            f"{baseline_validity['direct_pvr@10']:.6f}"
        ),
        (
            f"- Ancestor PVR@5: "
            f"{baseline_validity['ancestor_pvr@5']:.6f}"
        ),
        (
            f"- Ancestor PVR@10: "
            f"{baseline_validity['ancestor_pvr@10']:.6f}"
        ),
        "",
        "## Existing exp_013 grid with validity diagnostics",
        "",
    ]

    for shortlist in shortlist_results:
        top_n = int(shortlist["top_n"])
        summary_lines.extend([f"### Top-N = {top_n}", ""])
        for row_data in shortlist["rows"]:
            alpha = float(row_data["alpha"])
            accuracy_metrics = row_data["accuracy"]
            validity = row_data["validity"]
            summary_lines.append(
                f"- alpha={alpha:.2f}: "
                f"NDCG@10={float(accuracy_metrics['ndcg@10']):.6f}, "
                f"direct-PVR@5={float(validity['direct_pvr@5']):.6f}, "
                f"direct-PVR@10={float(validity['direct_pvr@10']):.6f}, "
                f"ancestor-PVR@5={float(validity['ancestor_pvr@5']):.6f}, "
                f"ancestor-PVR@10={float(validity['ancestor_pvr@10']):.6f}"
            )
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Interpretation rule",
            "",
            (
                "This experiment is descriptive only. It does not choose a new "
                "alpha or shortlist depth. Any apparent accuracy-validity "
                "trade-off must be reviewed before defining a new hypothesis."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_014_accuracy_validity_pareto")
    print(
        "Observed target direct/ancestor violation rates: "
        f"{float(direct_target['violation_rate']):.4f}/"
        f"{float(ancestor_target['violation_rate']):.4f}"
    )
    print(
        "SASRec direct PVR@5/@10: "
        f"{baseline_validity['direct_pvr@5']:.6f}/"
        f"{baseline_validity['direct_pvr@10']:.6f}"
    )
    print(
        "SASRec ancestor PVR@5/@10: "
        f"{baseline_validity['ancestor_pvr@5']:.6f}/"
        f"{baseline_validity['ancestor_pvr@10']:.6f}"
    )
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
