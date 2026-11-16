"""Run exp_022: size-stratified diagnostic of the cached Delta beta_1 signal.

Predeclared in docs/analysis_plan.md (D-074) before any result was computed.
Validation caches only; no selection, no new score, no development-test data.
"""

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
from relhom.size_stratified_diagnostic import (  # noqa: E402
    absorbed_edge_counts,
    bin_sizes,
    bootstrap_mean_ci,
    decide_outcome,
    per_user_target_auc,
    topk_size_enrichment,
)
from relhom.topology_cache import (  # noqa: E402
    load_validation_topology_cache,
    sha256_file as topology_sha256,
)
from relhom.topology_mechanism import candidate_structural_features  # noqa: E402

EXPERIMENT_ID = "exp_022_size_stratified_diagnostic"


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
    """Identical to exp_015 so features align with the frozen caches."""
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


def _paired_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    both = ~np.isnan(left) & ~np.isnan(right)
    return bootstrap_mean_ci(
        left[both] - right[both],
        replicates=replicates,
        seed=seed,
    )


def _fmt_ci(item: dict[str, object]) -> str:
    if item["mean"] is None:
        return "undefined"
    low, high = item["ci95"]
    return f"{item['mean']:.4f} [{low:.4f}, {high:.4f}] (users={item['n']:,})"


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
                f"{EXPERIMENT_ID} output already exists; refusing to "
                f"overwrite evidence: {path}"
            )

    actual_topology_sha = topology_sha256(topology_path)
    actual_sasrec_sha = sasrec_sha256(sasrec_path)
    if actual_topology_sha != str(config["expected_topology_cache_sha256"]):
        raise RuntimeError("topology cache SHA-256 mismatch")
    if actual_sasrec_sha != str(config["expected_sasrec_cache_sha256"]):
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

    deltas = topology["deltas"].astype(np.int64, copy=False)
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen_mask = deltas == sentinel
    candidate_mask = ~seen_mask
    logits = sasrec["logits"].astype(np.float64, copy=False)
    targets = topology["validation_target_indices"].astype(np.int64, copy=False)

    features = candidate_structural_features(
        splits, universe, mapping, graph, seen_mask
    )
    m = features["new_concept_count"].astype(np.int64)
    q = absorbed_edge_counts(splits, universe, mapping, graph, seen_mask).astype(
        np.int64
    )

    d = deltas[candidate_mask]
    lower_violations = int(np.count_nonzero(d < -m[candidate_mask]))
    upper_violations = int(np.count_nonzero(d > q[candidate_mask]))
    if lower_violations or upper_violations:
        raise RuntimeError(
            "bound -m <= Delta <= q violated "
            f"(lower={lower_violations}, upper={upper_violations}); "
            "features are misaligned with the topology cache"
        )

    bins = [int(x) for x in config["m_bin_lower_edges"]]
    m_bin = bin_sizes(m, bins)
    mq_key = m * 100_000 + q

    replicates = int(config["bootstrap_replicates"])
    seed = int(config["bootstrap_seed"])
    threshold = float(config["min_meaningful_auc"])
    k = int(config["top_k"])

    auc = {
        "delta_unstratified": per_user_target_auc(deltas, seen_mask, targets),
        "delta_within_exact_m": per_user_target_auc(
            deltas, seen_mask, targets, strata=m
        ),
        "delta_within_m_bins": per_user_target_auc(
            deltas, seen_mask, targets, strata=m_bin
        ),
        "delta_within_exact_m_and_q": per_user_target_auc(
            deltas, seen_mask, targets, strata=mq_key
        ),
        "m_unstratified": per_user_target_auc(
            m.astype(np.float64), seen_mask, targets
        ),
        "sasrec_unstratified": per_user_target_auc(logits, seen_mask, targets),
        "sasrec_within_exact_m": per_user_target_auc(
            logits, seen_mask, targets, strata=m
        ),
    }
    summaries = {
        name: bootstrap_mean_ci(values, replicates=replicates, seed=seed + i)
        for i, (name, values) in enumerate(auc.items())
    }

    primary = summaries["delta_within_exact_m"]
    difference = _paired_difference(
        auc["delta_within_exact_m"],
        auc["delta_unstratified"],
        replicates=replicates,
        seed=seed + 100,
    )
    decision = decide_outcome(
        primary, difference, min_meaningful_auc=threshold
    )

    same_m_competitors: list[int] = []
    for row in range(deltas.shape[0]):
        t = int(targets[row])
        mask = candidate_mask[row].copy()
        mask[t] = False
        same_m_competitors.append(int(np.count_nonzero(mask & (m[row] == m[row, t]))))
    competitors = np.asarray(same_m_competitors)

    enrichment = {
        "delta_topk_by_new_concepts": asdict(
            topk_size_enrichment(deltas.astype(np.float64), m, seen_mask, k)
        ),
        "sasrec_topk_by_new_concepts": asdict(
            topk_size_enrichment(logits, m, seen_mask, k)
        ),
    }

    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "phase": "size_stratified_mechanism_diagnostic",
        "evidence_scope": "validation_only",
        "predeclaration": "docs/analysis_plan.md D-074",
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
        "config": config,
        "auc_definition": (
            "per-user P(target outranks a competitor), ties count 0.5, "
            "target excluded; chance = 0.5; mean over users with >=1 "
            "competitor; user-level percentile bootstrap"
        ),
        "exact_checks": {
            "bound_lower_violations": lower_violations,
            "bound_upper_violations": upper_violations,
        },
        "auc": summaries,
        "primary": {
            "estimand": "delta_within_exact_m",
            "result": primary,
            "minus_unstratified_paired": difference,
            "min_meaningful_auc": threshold,
            "decision": decision,
        },
        "stratum_coverage": {
            "users_with_same_m_competitor": int(np.count_nonzero(competitors > 0)),
            "users_without_same_m_competitor": int(
                np.count_nonzero(competitors == 0)
            ),
            "median_same_m_competitors": float(np.median(competitors)),
        },
        "topk_enrichment": enrichment,
        "selection_performed": False,
        "new_score_evaluated": False,
        "development_test_evaluated": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    e_delta = enrichment["delta_topk_by_new_concepts"]
    e_sas = enrichment["sasrec_topk_by_new_concepts"]
    lines = [
        "# exp_022 - Size-Stratified Topology Diagnostic",
        "",
        "- Evidence scope: validation only",
        "- Predeclared: docs/analysis_plan.md D-074",
        f"- Users: {len(splits):,}",
        f"- Unseen candidate states: {int(candidate_mask.sum()):,}",
        "- Selection performed: NO",
        "- New score evaluated: NO",
        "- Development-test evaluated: NO",
        "",
        "## Exact check",
        "",
        f"- Bound -m <= Delta <= q violations: {lower_violations + upper_violations}",
        "",
        "## Target AUC (tie-aware, chance = 0.5)",
        "",
        f"- Delta, unstratified: {_fmt_ci(summaries['delta_unstratified'])}",
        f"- **Delta within equal new-concept count (PRIMARY):** {_fmt_ci(primary)}",
        f"- Delta within new-concept bins: {_fmt_ci(summaries['delta_within_m_bins'])}",
        f"- Delta within equal (m, q): {_fmt_ci(summaries['delta_within_exact_m_and_q'])}",
        f"- New-concept count itself, unstratified: {_fmt_ci(summaries['m_unstratified'])}",
        f"- SASRec logit, unstratified (reference): {_fmt_ci(summaries['sasrec_unstratified'])}",
        f"- SASRec logit within equal m (reference): {_fmt_ci(summaries['sasrec_within_exact_m'])}",
        "",
        "## Primary decision",
        "",
        f"- Stratified minus unstratified (paired): {_fmt_ci(difference)}",
        f"- Minimum meaningful AUC (predeclared): {threshold}",
        f"- **Outcome: {decision['outcome']}**",
        f"- {decision['meaning']}",
        "",
        "## Stratum coverage",
        "",
        (
            "- Users with >=1 same-size competitor: "
            f"{payload['stratum_coverage']['users_with_same_m_competitor']:,}"
        ),
        (
            "- Median same-size competitors per user: "
            f"{payload['stratum_coverage']['median_same_m_competitors']:.1f}"
        ),
        "",
        f"## Top-{k} size enrichment (new-concept count)",
        "",
        (
            f"- Delta top-{k}: median size ratio vs pool="
            f"{e_delta['median_size_ratio']:.3f}, share of users with ratio>1="
            f"{e_delta['share_users_ratio_above_1']:.3f}"
        ),
        (
            f"- SASRec top-{k} (reference): median size ratio vs pool="
            f"{e_sas['median_size_ratio']:.3f}, share of users with ratio>1="
            f"{e_sas['share_users_ratio_above_1']:.3f}"
        ),
        "",
        "## Interpretation rule",
        "",
        (
            "Outcome A supports a 'size masks topology' account; B means a "
            "size-independent signal exists but size does not explain the "
            "recommender failure; C means the score is weak regardless of "
            "size. No score, weight, or shortlist is selected from this "
            "experiment."
        ),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(EXPERIMENT_ID)
    print(f"Bound violations: {lower_violations + upper_violations}")
    print(f"Delta AUC unstratified: {_fmt_ci(summaries['delta_unstratified'])}")
    print(f"Delta AUC within equal m (PRIMARY): {_fmt_ci(primary)}")
    print(f"Stratified - unstratified: {_fmt_ci(difference)}")
    print(f"Outcome: {decision['outcome']}")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
