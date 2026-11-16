"""Run exp_024: normalization and near-tie sensitivity of the recommender null.

Predeclared in docs/analysis_plan.md (D-080) before any result was computed.
Validation caches only (exp_010/exp_011); no retraining, no recomputation,
no development-test data. Reports whatever it finds.

Question: does the exp_012/exp_013 null (validation selects pure SASRec)
survive (i) rank-normalizing the topology score exactly like SASRec instead of
MinMax, and (ii) a finer near-tie alpha grid {0.96, 0.97, 0.98, 0.99}?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.hybrid_cache import (  # noqa: E402
    candidate_rank_normalize,
    candidate_topology_minmax,
    evaluate_hybrid_grid,
    target_ranks_from_scores,
    validate_hybrid_cache_alignment,
)
from relhom.normalization_sensitivity import (  # noqa: E402
    classify_configuration,
    overall_classification,
    paired_bootstrap_difference,
    per_user_ndcg_at_k,
    select_best,
)
from relhom.sasrec_score_cache import (  # noqa: E402
    load_sasrec_score_cache,
    sha256_file as sasrec_sha256,
)
from relhom.topn_hybrid import topn_rerank_grid  # noqa: E402
from relhom.topology_cache import (  # noqa: E402
    load_validation_topology_cache,
    sha256_file as topology_sha256,
)

EXPERIMENT_ID = "exp_024_normalization_sensitivity"
TOL = 1e-12


def _near_tie(results):
    """Largest alpha < 1 with any rank movement, and its sign-test counts."""
    moving = [
        r for r in results
        if float(r["alpha"]) < 1.0
        and "rank_movement_vs_sasrec" in r
        and (r["rank_movement_vs_sasrec"]["improved"]
             + r["rank_movement_vs_sasrec"]["worsened"]) > 0
    ]
    if not moving:
        return None
    r = max(moving, key=lambda x: float(x["alpha"]))
    from scipy.stats import binomtest
    imp = int(r["rank_movement_vs_sasrec"]["improved"])
    wor = int(r["rank_movement_vs_sasrec"]["worsened"])
    return {
        "alpha": float(r["alpha"]),
        "improved": imp,
        "worsened": wor,
        "two_sided_binomial_p": float(binomtest(wor, imp + wor, 0.5).pvalue),
    }


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"
    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                f"{EXPERIMENT_ID} output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    topology_path = REPO_ROOT / str(config["topology_cache_path"])
    sasrec_path = REPO_ROOT / str(config["sasrec_cache_path"])
    if topology_sha256(topology_path) != config["expected_topology_cache_sha256"]:
        raise RuntimeError("topology cache SHA-256 mismatch")
    if sasrec_sha256(sasrec_path) != config["expected_sasrec_cache_sha256"]:
        raise RuntimeError("SASRec cache SHA-256 mismatch")

    topology = load_validation_topology_cache(topology_path)
    sasrec = load_sasrec_score_cache(sasrec_path)
    validate_hybrid_cache_alignment(topology, sasrec)

    deltas = topology["deltas"].astype(np.int32, copy=False)
    sentinel = int(topology["sentinel"].reshape(-1)[0])
    seen = deltas == sentinel
    logits = sasrec["logits"].astype(np.float64, copy=False)
    targets = topology["validation_target_indices"].astype(np.int64, copy=False)

    frozen = [float(a) for a in config["frozen_alphas"]]
    fine = [float(a) for a in config["fine_alphas"]]
    grid = sorted(set(frozen) | set(fine))
    cutoffs = [int(k) for k in config["cutoffs"]]
    reps = int(config["bootstrap_replicates"])
    seed = int(config["bootstrap_seed"])

    exp012 = json.loads((REPO_ROOT / config["exp012_metrics"]).read_text(encoding="utf-8"))
    exp013 = json.loads((REPO_ROOT / config["exp013_metrics"]).read_text(encoding="utf-8"))

    baseline_ranks = target_ranks_from_scores(logits, seen, targets)
    baseline_ndcg = per_user_ndcg_at_k(baseline_ranks, 10)
    sasrec_norm = candidate_rank_normalize(logits, seen)

    topo_norms = {
        "minmax": candidate_topology_minmax(deltas, seen),
        "rank": candidate_rank_normalize(deltas.astype(np.float64), seen),
    }

    configs: dict[str, dict[str, object]] = {}
    labels: list[str] = []
    offset = 0

    for norm, topo in topo_norms.items():
        results = evaluate_hybrid_grid(
            sasrec_norm, topo, seen, targets, grid, cutoffs=cutoffs
        )
        if norm == "minmax":
            ref = {float(r["alpha"]): r["metrics"]["ndcg@10"]
                   for r in exp012["signed"]["results"]}
            for r in results:
                a = float(r["alpha"])
                if a in ref and abs(r["metrics"]["ndcg@10"] - ref[a]) > TOL:
                    raise RuntimeError(
                        f"global MinMax alpha={a} does not reproduce exp_012"
                    )
        for r in results:
            a = float(r["alpha"])
            ranks = target_ranks_from_scores(
                a * sasrec_norm.astype(np.float64)
                + (1 - a) * topo.astype(np.float64),
                seen, targets,
            )
            r["rank_movement_vs_sasrec"] = {
                "improved": int((ranks < baseline_ranks).sum()),
                "worsened": int((ranks > baseline_ranks).sum()),
            }
            if a == float(select_best(results)["alpha"]):
                best_ranks = ranks
        best = select_best(results)
        boot = paired_bootstrap_difference(
            per_user_ndcg_at_k(best_ranks, 10), baseline_ndcg,
            replicates=reps, seed=seed + offset,
        )
        label = classify_configuration(float(best["alpha"]), boot["ci95"])
        labels.append(label)
        configs[f"global_{norm}"] = {
            "integration": "global fusion",
            "topology_normalization": norm,
            "selected_alpha": float(best["alpha"]),
            "selected_ndcg@10": float(best["metrics"]["ndcg@10"]),
            "sasrec_ndcg@10": float(baseline_ndcg.mean()),
            "paired_bootstrap_vs_sasrec": boot,
            "classification": label,
            "near_tie": _near_tie(results),
            "grid": [{"alpha": r["alpha"], "ndcg@10": r["metrics"]["ndcg@10"],
                      **r["rank_movement_vs_sasrec"]} for r in results],
            "reproduces_exp_012": norm == "minmax",
        }
        offset += 1

    for top_n in [int(n) for n in config["top_ns"]]:
        for norm in ("minmax", "rank"):
            out = topn_rerank_grid(
                logits, deltas, seen, targets,
                top_n=top_n, alphas=grid, cutoffs=cutoffs,
                topology_normalization=norm, return_ranks=True,
            )
            results = out["results"]
            if norm == "minmax":
                shortlist = next(s for s in exp013["shortlists"] if int(s["top_n"]) == top_n)
                ref = {float(r["alpha"]): r["metrics"]["ndcg@10"] for r in shortlist["results"]}
                for r in results:
                    a = float(r["alpha"])
                    if a in ref and abs(r["metrics"]["ndcg@10"] - ref[a]) > TOL:
                        raise RuntimeError(
                            f"top-{top_n} MinMax alpha={a} does not reproduce exp_013"
                        )
            best = select_best(results)
            boot = paired_bootstrap_difference(
                per_user_ndcg_at_k(best["ranks"], 10), baseline_ndcg,
                replicates=reps, seed=seed + offset,
            )
            label = classify_configuration(float(best["alpha"]), boot["ci95"])
            labels.append(label)
            configs[f"top{top_n}_{norm}"] = {
                "integration": f"top-{top_n} reranking",
                "topology_normalization": norm,
                "selected_alpha": float(best["alpha"]),
                "selected_ndcg@10": float(best["metrics"]["ndcg@10"]),
                "sasrec_ndcg@10": float(baseline_ndcg.mean()),
                "paired_bootstrap_vs_sasrec": boot,
                "classification": label,
                "near_tie": _near_tie(results),
                "grid": [{"alpha": r["alpha"], "ndcg@10": r["metrics"]["ndcg@10"],
                          "improved": r["rank_movement_vs_sasrec"]["improved"],
                          "worsened": r["rank_movement_vs_sasrec"]["worsened"]}
                         for r in results],
                "reproduces_exp_013": norm == "minmax",
            }
            offset += 1

    overall = overall_classification(labels)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "phase": "recommender_null_sensitivity",
        "evidence_scope": "validation_only",
        "predeclaration": "docs/analysis_plan.md D-080",
        "config": config,
        "grid": grid,
        "configurations": configs,
        "overall_classification": overall,
        "selection_optimism_note": (
            "alpha is selected and evaluated on the same validation data, so "
            "any improvement is optimistic and exploratory"
        ),
        "development_test_evaluated": False,
        "sasrec_retrained": False,
        "topology_recomputed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# exp_024 - Normalization and Near-Tie Sensitivity of the Recommender Null",
        "",
        "- Evidence scope: validation only",
        "- Predeclared: docs/analysis_plan.md D-080",
        f"- Alpha grid: frozen 21 values plus {fine}",
        "- MinMax configurations reproduce exp_012 / exp_013: PASS",
        "",
        "| Integration | Topology norm. | Selected alpha | NDCG@10 | "
        "Delta vs SASRec [95% CI] | Near-tie alpha: improved / worsened (p) | "
        "Classification |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, c in configs.items():
        b = c["paired_bootstrap_vs_sasrec"]
        nt = c["near_tie"]
        nt_text = (
            f"{nt['alpha']:.2f}: {nt['improved']} / {nt['worsened']} "
            f"(p={nt['two_sided_binomial_p']:.2e})" if nt else "no movement"
        )
        lines.append(
            f"| {c['integration']} | {c['topology_normalization']} | "
            f"{c['selected_alpha']:.2f} | {c['selected_ndcg@10']:.6f} | "
            f"{b['mean_difference']:+.6f} [{b['ci95'][0]:+.6f}, {b['ci95'][1]:+.6f}] | "
            f"{nt_text} | {c['classification']} |"
        )
    lines += [
        "",
        f"SASRec NDCG@10 reference: {float(baseline_ndcg.mean()):.6f}",
        "",
        f"**Overall: {overall}**",
        "",
        "NULL_ROBUST: no configuration improves NDCG@10 over SASRec with a "
        "paired-bootstrap CI above zero. NORMALIZATION_OR_GRID_SENSITIVE: at "
        "least one does; it must be reported, and the null claim qualified to "
        "the MinMax protocol. Any improvement here is selected on validation "
        "and therefore exploratory.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
