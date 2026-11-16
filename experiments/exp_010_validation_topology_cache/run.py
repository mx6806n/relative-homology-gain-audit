"""Run exp_010: build the full raw-DAG validation topology cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.experiment import (  # noqa: E402
    load_course_universe,
    load_leave_two_out_splits,
)
from relhom.topology_cache import (  # noqa: E402
    build_validation_topology_cache,
    load_validation_topology_cache,
    sha256_file,
    topology_cache_stats_to_dict,
    validate_topology_cache,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON/YAML object")
    return payload


def _load_graph_and_mapping(
    data_dir: Path,
) -> tuple[nx.DiGraph, dict[str, set[str]]]:
    concept_vocab = json.loads(
        (data_dir / "concept_vocab.json").read_text(encoding="utf-8")
    )
    if not isinstance(concept_vocab, dict):
        raise ValueError("concept_vocab.json must contain an object")

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
        raise ValueError("processed prerequisite graph must be a DAG")

    raw_mapping = json.loads(
        (data_dir / "course_to_concepts.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw_mapping, dict):
        raise ValueError("course_to_concepts.json must contain an object")

    mapping = {
        str(course): set(map(str, concepts))
        for course, concepts in raw_mapping.items()
    }

    unknown = {
        concept
        for concepts in mapping.values()
        for concept in concepts
        if concept not in graph
    }
    if unknown:
        raise ValueError(
            "course_to_concepts contains concepts absent from graph: "
            + ", ".join(sorted(unknown)[:10])
        )

    return graph, mapping


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)

    data_dir = REPO_ROOT / str(config["data_dir"])
    output_dir = REPO_ROOT / str(config["output_dir"])
    cache_path = output_dir / "validation_topology_cache.npz"
    manifest_path = output_dir / "cache_manifest.json"
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (cache_path, manifest_path, metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_010 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    sequences_path = data_dir / "sequences.jsonl"
    course_vocab_path = data_dir / "course_vocab.json"
    concept_vocab_path = data_dir / "concept_vocab.json"
    mapping_path = data_dir / "course_to_concepts.json"
    edges_path = data_dir / "prerequisite_edges.csv"

    universe = load_course_universe(course_vocab_path)
    splits = load_leave_two_out_splits(
        sequences_path,
        min_sequence_length=int(config["min_sequence_length"]),
        item_universe=universe,
    )
    graph, mapping = _load_graph_and_mapping(data_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = build_validation_topology_cache(
        graph,
        splits,
        universe,
        mapping,
        cache_path,
        evaluator_cache_size=int(config["evaluator_cache_size"]),
    )

    cache = load_validation_topology_cache(cache_path)
    validate_topology_cache(cache, splits, universe)

    data_hashes = {
        "sequences_sha256": _sha256(sequences_path),
        "course_vocab_sha256": _sha256(course_vocab_path),
        "concept_vocab_sha256": _sha256(concept_vocab_path),
        "course_to_concepts_sha256": _sha256(mapping_path),
        "prerequisite_edges_sha256": _sha256(edges_path),
    }
    cache_hash = sha256_file(cache_path)

    manifest = {
        "experiment_id": "exp_010_validation_topology_cache",
        "cache_file": cache_path.name,
        "cache_sha256": cache_hash,
        "data_hashes": data_hashes,
        "topology": {
            "graph_variant": "native_raw_prerequisite_dag",
            "coeff": 2,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "evaluator": "GF2RelativeHomologyEvaluator",
            "evaluator_cache_size": int(config["evaluator_cache_size"]),
        },
        "split": {
            "protocol": "temporal_leave_two_out",
            "scope": "validation_histories_only",
            "min_sequence_length": int(config["min_sequence_length"]),
            "users": len(splits),
            "courses": len(universe),
        },
        "arrays": {
            "deltas": {
                "dtype": str(cache["deltas"].dtype),
                "shape": list(cache["deltas"].shape),
                "seen_sentinel": int(cache["sentinel"][0]),
            },
            "beta_before": {
                "dtype": str(cache["beta_before"].dtype),
                "shape": list(cache["beta_before"].shape),
            },
            "validation_target_indices": {
                "dtype": str(cache["validation_target_indices"].dtype),
                "shape": list(cache["validation_target_indices"].shape),
            },
            "history_lengths": {
                "dtype": str(cache["history_lengths"].dtype),
                "shape": list(cache["history_lengths"].shape),
            },
            "user_ids": {
                "dtype": str(cache["user_ids"].dtype),
                "shape": list(cache["user_ids"].shape),
            },
            "course_ids": {
                "dtype": str(cache["course_ids"].dtype),
                "shape": list(cache["course_ids"].shape),
            },
        },
        "validation": {
            "reloaded_without_pickle": True,
            "ordering_and_mask_checks_passed": True,
        },
        "development_test_evaluated": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_010_validation_topology_cache",
        "status": "completed",
        "phase": "topology_cache",
        "evidence_scope": "validation_histories_only",
        "data_hashes": data_hashes,
        "cache": {
            "path": str(cache_path.relative_to(REPO_ROOT)),
            "sha256": cache_hash,
            "bytes": cache_path.stat().st_size,
        },
        "topology": {
            "graph_variant": "native_raw_prerequisite_dag",
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "coeff": 2,
        },
        "stats": topology_cache_stats_to_dict(stats),
        "cache_validation_passed": True,
        "development_test_evaluated": False,
        "sasrec_used": False,
        "alpha_tuned": False,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_010 - Full Validation Topology Cache",
        "",
        "- Graph variant: native raw prerequisite DAG",
        "- Evidence scope: validation histories only",
        f"- Users: {stats.users:,}",
        f"- Courses: {stats.courses:,}",
        f"- Candidate evaluations: {stats.candidate_evaluations:,}",
        f"- Seen/masked matrix entries: {stats.seen_entries:,}",
        f"- Elapsed topology scoring: {stats.elapsed_seconds:.3f} s",
        f"- Cache file bytes: {cache_path.stat().st_size:,}",
        f"- Cache SHA-256: {cache_hash}",
        "",
        "## Evaluator cache",
        "",
        (
            f"- hits={stats.cache_info['hits']:,}, "
            f"misses={stats.cache_info['misses']:,}, "
            f"currsize={stats.cache_info['currsize']:,}, "
            f"hit_rate={stats.cache_info['hit_rate']:.4f}"
        ),
        "",
        "## Topology ranges",
        "",
        (
            f"- beta_1 before range: "
            f"[{stats.beta_before_min}, {stats.beta_before_max}]"
        ),
        f"- Delta beta_1 range: [{stats.delta_min}, {stats.delta_max}]",
        (
            "- Candidate positive/zero/negative fractions: "
            f"{stats.positive_fraction:.4f}/"
            f"{stats.zero_fraction:.4f}/"
            f"{stats.negative_fraction:.4f}"
        ),
        (
            "- Validation targets positive/zero/negative: "
            f"{stats.target_positive:,}/"
            f"{stats.target_zero:,}/"
            f"{stats.target_negative:,}"
        ),
        (
            "- Users with discriminative topology: "
            f"{stats.users_with_discriminative_signal:,}/"
            f"{stats.users:,}"
        ),
        "",
        "## Validation checks",
        "",
        "- Cache reloaded with allow_pickle=False: PASS",
        "- User ordering: PASS",
        "- Course ordering: PASS",
        "- Seen-course masking: PASS",
        "- Validation target visibility: PASS",
        "",
        "No SASRec scores, alpha tuning, or development-test data are used.",
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_010_validation_topology_cache")
    print(
        f"Users/courses: {stats.users:,}/{stats.courses:,}"
    )
    print(
        "Candidate evaluations / seen entries: "
        f"{stats.candidate_evaluations:,}/"
        f"{stats.seen_entries:,}"
    )
    print(
        f"Elapsed topology scoring: {stats.elapsed_seconds:.3f}s"
    )
    print(
        "Evaluator cache hits/misses/hit-rate: "
        f"{stats.cache_info['hits']:,}/"
        f"{stats.cache_info['misses']:,}/"
        f"{stats.cache_info['hit_rate']:.4f}"
    )
    print(
        "Candidate positive/zero/negative fractions: "
        f"{stats.positive_fraction:.4f}/"
        f"{stats.zero_fraction:.4f}/"
        f"{stats.negative_fraction:.4f}"
    )
    print(
        "Validation targets positive/zero/negative: "
        f"{stats.target_positive:,}/"
        f"{stats.target_zero:,}/"
        f"{stats.target_negative:,}"
    )
    print(
        "Discriminative users: "
        f"{stats.users_with_discriminative_signal:,}/"
        f"{stats.users:,}"
    )
    print(f"Cache bytes: {cache_path.stat().st_size:,}")
    print(f"Cache SHA-256: {cache_hash}")
    print("Cache validation: PASS")
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
