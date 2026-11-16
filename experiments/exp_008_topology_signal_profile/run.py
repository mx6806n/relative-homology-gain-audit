"""Run exp_008: validation-only topology signal and runtime profile."""

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
from relhom.topology_profile import (  # noqa: E402
    profile_topology_signal,
    topology_profile_to_dict,
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
        (data_dir / "concept_vocab.json").read_text(
            encoding="utf-8"
        )
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
        (data_dir / "course_to_concepts.json").read_text(
            encoding="utf-8"
        )
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
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(
            "exp_008 output already exists; refusing to overwrite evidence. "
            f"Existing output directory: {output_dir}"
        )

    sequences_path = data_dir / "sequences.jsonl"
    course_vocab_path = data_dir / "course_vocab.json"
    concept_vocab_path = data_dir / "concept_vocab.json"
    mapping_path = data_dir / "course_to_concepts.json"
    edges_path = data_dir / "prerequisite_edges.csv"

    item_universe = load_course_universe(course_vocab_path)
    splits = load_leave_two_out_splits(
        sequences_path,
        min_sequence_length=int(config["min_sequence_length"]),
        item_universe=item_universe,
    )
    graph, mapping = _load_graph_and_mapping(data_dir)

    profile = profile_topology_signal(
        graph,
        splits,
        item_universe,
        mapping,
        sample_users=int(config["sample_users"]),
        seed=int(config["sample_seed"]),
        reference_checks=int(config["reference_checks"]),
        cache_size=int(config["cache_size"]),
    )
    profile_dict = topology_profile_to_dict(profile)

    average_candidates = (
        profile.candidate_evaluations / profile.sampled_users
    )
    estimated_primary_candidate_evaluations = int(
        round(average_candidates * len(splits))
    )

    metrics: dict[str, object] = {
        "experiment_id": "exp_008_topology_signal_profile",
        "status": "completed",
        "phase": "topology_profiling",
        "evidence_scope": "validation_histories_only",
        "sample": {
            "users": profile.sampled_users,
            "seed": int(config["sample_seed"]),
            "all_unseen_candidates_per_sampled_user": True,
        },
        "data": {
            "sequences_sha256": _sha256(sequences_path),
            "course_vocab_sha256": _sha256(course_vocab_path),
            "concept_vocab_sha256": _sha256(concept_vocab_path),
            "course_to_concepts_sha256": _sha256(mapping_path),
            "prerequisite_edges_sha256": _sha256(edges_path),
            "primary_users": len(splits),
            "candidate_courses": len(item_universe),
        },
        "evaluator": {
            "type": "exact_relative_h1_gf2_bitset",
            "reference": "pyflagser_mapping_cone",
            "reference_checks": profile.reference_checks,
            "reference_mismatches": profile.reference_mismatches,
            "cache_size": int(config["cache_size"]),
        },
        "profile": profile_dict,
        "scaling": {
            "average_unseen_candidates_per_sampled_user": (
                average_candidates
            ),
            "estimated_candidate_evaluations_for_all_primary_users": (
                estimated_primary_candidate_evaluations
            ),
            "note": (
                "This is a count extrapolation only, not a runtime forecast; "
                "cache reuse can differ on the full cohort."
            ),
        },
        "test_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# exp_008 - Topology Signal and Runtime Profile",
        "",
        "- Evidence scope: validation histories only",
        f"- Sampled users: {profile.sampled_users:,}",
        (
            "- Candidate evaluations: "
            f"{profile.candidate_evaluations:,}"
        ),
        (
            "- Unique history concept states: "
            f"{profile.unique_history_concept_states:,}"
        ),
        (
            "- Raw prerequisite graph: "
            f"{graph.number_of_nodes():,} nodes, "
            f"{profile.raw_edges:,} edges, "
            f"{profile.directed_triangles:,} directed 2-simplices"
        ),
        (
            "- Transitive-closure edge count: "
            f"{profile.transitive_closure_edges:,}"
        ),
        "",
        "## Fast evaluator validation",
        "",
        (
            f"- Reference states checked with pyflagser: "
            f"{profile.reference_checks:,}"
        ),
        (
            f"- Fast/reference mismatches: "
            f"{profile.reference_mismatches:,}"
        ),
        (
            f"- Fast profiling time: "
            f"{profile.fast_elapsed_seconds:.3f} s"
        ),
        (
            f"- Reference-check time: "
            f"{profile.reference_elapsed_seconds:.3f} s"
        ),
        (
            "- Cache: "
            f"hits={profile.cache_info['hits']}, "
            f"misses={profile.cache_info['misses']}, "
            f"hit_rate={profile.cache_info['hit_rate']:.4f}"
        ),
        "",
        "## Delta beta_1 candidate signal",
        "",
        f"- Histogram: {profile.delta_histogram}",
        (
            "- Candidate fractions: "
            f"positive={profile.positive_candidate_fraction:.4f}, "
            f"zero={profile.zero_candidate_fraction:.4f}, "
            f"negative={profile.negative_candidate_fraction:.4f}"
        ),
        (
            "- Users with >=2 distinct candidate deltas: "
            f"{profile.users_with_discriminative_signal:,}/"
            f"{profile.sampled_users:,}"
        ),
        (
            "- Users with at least one positive candidate: "
            f"{profile.users_with_positive_candidate:,}/"
            f"{profile.sampled_users:,}"
        ),
        (
            "- Users with at least one negative candidate: "
            f"{profile.users_with_negative_candidate:,}/"
            f"{profile.sampled_users:,}"
        ),
        "",
        "## Validation-target topology",
        "",
        f"- Target delta histogram: {profile.target_delta_histogram}",
        (
            "- Target positive/zero/negative: "
            f"{profile.targets_with_positive_delta}/"
            f"{profile.targets_with_zero_delta}/"
            f"{profile.targets_with_negative_delta}"
        ),
        (
            "- Targets at user maximum delta: "
            f"{profile.targets_at_user_max_delta:,}/"
            f"{profile.sampled_users:,}"
        ),
        (
            "- Targets strictly above candidate median delta: "
            f"{profile.targets_above_candidate_median:,}/"
            f"{profile.sampled_users:,}"
        ),
        (
            "- Mean weak target percentile among unseen candidates: "
            f"{profile.mean_target_weak_percentile:.4f}"
        ),
        "",
        "## State distributions",
        "",
        f"- beta_1 before histogram: {profile.beta_before_histogram}",
        (
            "- History concept-count percentiles: "
            f"{profile.history_concept_percentiles}"
        ),
        (
            "- Per-user distinct-delta-count percentiles: "
            f"{profile.unique_delta_count_percentiles}"
        ),
        "",
        "## Scaling count",
        "",
        (
            "- Average unseen candidates/sample user: "
            f"{average_candidates:.2f}"
        ),
        (
            "- Estimated candidate evaluations for all primary users: "
            f"{estimated_primary_candidate_evaluations:,}"
        ),
        "",
        "No development-test targets or metrics are evaluated in exp_008.",
    ]
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("exp_008_topology_signal_profile")
    print(
        f"Graph: {graph.number_of_nodes()} nodes, "
        f"{profile.raw_edges} edges, "
        f"{profile.directed_triangles} directed triangles"
    )
    print(
        "Transitive closure edges: "
        f"{profile.transitive_closure_edges}"
    )
    print(
        "Reference checks/mismatches: "
        f"{profile.reference_checks}/{profile.reference_mismatches}"
    )
    print(
        "Candidate evaluations / elapsed: "
        f"{profile.candidate_evaluations:,} / "
        f"{profile.fast_elapsed_seconds:.3f}s"
    )
    print(
        "Cache hits/misses/hit-rate: "
        f"{profile.cache_info['hits']}/"
        f"{profile.cache_info['misses']}/"
        f"{profile.cache_info['hit_rate']:.4f}"
    )
    print(
        "Delta histogram: "
        f"{profile.delta_histogram}"
    )
    print(
        "Users with discriminative topology: "
        f"{profile.users_with_discriminative_signal:,}/"
        f"{profile.sampled_users:,}"
    )
    print(
        "Target positive/zero/negative: "
        f"{profile.targets_with_positive_delta}/"
        f"{profile.targets_with_zero_delta}/"
        f"{profile.targets_with_negative_delta}"
    )
    print(
        "Targets at max delta: "
        f"{profile.targets_at_user_max_delta:,}/"
        f"{profile.sampled_users:,}"
    )
    print(
        "Mean weak target percentile: "
        f"{profile.mean_target_weak_percentile:.4f}"
    )
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
