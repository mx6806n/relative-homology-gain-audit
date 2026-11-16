"""Run exp_023: no-topology null benchmark for size dependence of raw gain.

Predeclared in docs/analysis_plan.md (D-076) before any result was computed.
Target-independent: no recommendation metric, no selection, no new score.

Part A (MOOCCube2x CS/Math/Psychology): per-candidate g, m, q exactly as in
exp_018; primary estimand = mean per-user excess
Spearman_u(|g|, m) - Spearman_u(|N1|, m).

Part B (DORIS v3): per-state g, m, q exactly as in exp_021; primary estimands =
Spearman(|g|, m) - Spearman(|N|, m) for N1 and N2, student-cluster bootstrap;
BEYOND_BOUND only if both excesses clear the margin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.data_loader import resolve_mooccube2x_root  # noqa: E402
from relhom.null_models import (  # noqa: E402
    classify_excess,
    cluster_bootstrap_rho_difference,
    fit_vertex_probability,
    independent_vertex_null,
    per_row_spearman,
    safe_spearman,
    uniform_bound_null,
)
from relhom.size_stratified_diagnostic import bootstrap_mean_ci  # noqa: E402
from relhom.topology import GF2RelativeHomologyEvaluator  # noqa: E402
from relhom.topology_mechanism import correlation_summary  # noqa: E402

EXPERIMENT_ID = "exp_023_bound_null_benchmark"
REPRODUCTION_TOLERANCE = 1e-9


def _load_module(relative: str, name: str):
    """Import an earlier experiment script so data loading is identical."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _fmt_ci(ci: list[float | None]) -> str:
    if ci[0] is None:
        return "[undefined]"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _domain_arrays(graph, mapping, splits, universe, cache_size):
    """Per-candidate g, m, q with the exp_018 state definition."""
    evaluator = GF2RelativeHomologyEvaluator(
        graph, validate_dag=True, cache_size=cache_size
    )
    shape = (len(splits), len(universe))
    seen = np.zeros(shape, dtype=bool)
    g = np.zeros(shape, dtype=np.int32)
    m = np.zeros(shape, dtype=np.int32)
    q = np.zeros(shape, dtype=np.int32)
    course_index = {course: i for i, course in enumerate(universe)}

    for row, split in enumerate(splits):
        history: set[str] = set()
        for course in split.validation_history:
            history.update(mapping.get(course, ()))
        for course in set(split.validation_history):
            index = course_index.get(course)
            if index is not None:
                seen[row, index] = True
        before = evaluator.relative_h1_components(history)
        for col, course in enumerate(universe):
            if seen[row, col]:
                continue
            bundle = mapping.get(course, set())
            after = evaluator.relative_h1_components(history.union(bundle))
            g[row, col] = before.beta1 - after.beta1
            m[row, col] = len(bundle.difference(history))
            q[row, col] = before.relative_edge_count - after.relative_edge_count
    return g, m, q, seen


def _analyze_mooccube_domain(
    name, graph, mapping, splits, universe, *, config, reference, seed_offset
):
    started = time.perf_counter()
    g, m, q, seen = _domain_arrays(
        graph, mapping, splits, universe, int(config["evaluator_cache_size"])
    )
    candidates = ~seen

    check = correlation_summary(np.abs(g), m, seen)
    ref = reference["domains"][name]["correlations"]["abs_delta_vs_new_concepts"]
    for field in ("global_spearman", "per_user_median"):
        if abs(getattr(check, field) - float(ref[field])) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(
                f"{name}: {field} {getattr(check, field)} does not reproduce "
                f"exp_018 value {ref[field]}; data or code differ"
            )

    lower = int(np.count_nonzero(g[candidates] < -m[candidates]))
    upper = int(np.count_nonzero(g[candidates] > q[candidates]))
    if lower or upper:
        raise RuntimeError(f"{name}: bound violated (lower={lower}, upper={upper})")

    rng = np.random.default_rng(int(config["seed"]) + seed_offset)
    n1 = np.zeros_like(g)
    n1[candidates] = uniform_bound_null(m[candidates], q[candidates], rng)
    n3 = np.where(candidates, q - m, 0)

    obs_u = per_row_spearman(np.abs(g), m, candidates)
    n1_u = per_row_spearman(np.abs(n1), m, candidates)
    n3_u = per_row_spearman(np.abs(n3), m, candidates)
    both = ~np.isnan(obs_u) & ~np.isnan(n1_u)
    excess = bootstrap_mean_ci(
        obs_u[both] - n1_u[both],
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]) + seed_offset + 1,
    )
    margin = float(config["excess_margin"])

    def pooled(values):
        return safe_spearman(np.abs(values[candidates]), m[candidates])

    return {
        "users": len(splits),
        "courses": len(universe),
        "candidate_states": int(candidates.sum()),
        "reproduces_exp_018": True,
        "bound_violations": lower + upper,
        "pooled_spearman_abs_vs_m": {
            "observed": pooled(g),
            "n1_bound_uniform": pooled(n1),
            "n3_one_dimensional_q_minus_m": pooled(n3),
        },
        "per_user_median_spearman_abs_vs_m": {
            "observed": float(np.nanmedian(obs_u)),
            "n1_bound_uniform": float(np.nanmedian(n1_u)),
            "n3_one_dimensional_q_minus_m": float(np.nanmedian(n3_u)),
        },
        "primary_per_user_excess_over_n1": excess,
        "classification": classify_excess(excess["ci95"], margin),
        "fraction_g_equals_q_minus_m": float(
            np.mean(g[candidates] == (q - m)[candidates])
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _analyze_doris(*, config, s2_path, edges_path, feasibility_path, reference):
    started = time.perf_counter()
    doris = _load_module(
        "experiments/exp_021_doris_v3_semester_bundle_mechanism/run.py",
        "exp021_run",
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if int(feasibility.get("audit_version", 0)) != 3:
        raise RuntimeError("exp_023 requires DORIS audit_version=3")
    if not bool(feasibility.get("final_feasibility_pass", False)):
        raise RuntimeError("DORIS v3 final feasibility gate has not passed")

    graph = doris._load_graph(edges_path)
    selection = doris._read_selection_table(s2_path)
    sessions, _ = doris._build_student_sessions(selection, set(graph.nodes))
    evaluator = GF2RelativeHomologyEvaluator(
        graph, validate_dag=True, cache_size=int(config["evaluator_cache_size"])
    )

    students: list[str] = []
    histories: list[frozenset[str]] = []
    g_list: list[int] = []
    m_list: list[int] = []
    q_list: list[int] = []
    for student, student_sessions in sessions.items():
        history: set[str] = set()
        for session in sorted(student_sessions):
            current = set(student_sessions[session])
            candidate = current.difference(history)
            if history and candidate:
                dec = evaluator.relative_h1_gain_decomposition(history, candidate)
                students.append(student)
                histories.append(frozenset(history))
                g_list.append(dec.delta_beta1)
                m_list.append(dec.new_vertex_count)
                q_list.append(dec.absorbed_relative_edges)
            history.update(current)

    g = np.asarray(g_list, dtype=np.int64)
    m = np.asarray(m_list, dtype=np.int64)
    q = np.asarray(q_list, dtype=np.int64)
    clusters = np.asarray(students, dtype=object)

    observed = safe_spearman(np.abs(g), m)
    ref_rho = float(reference["correlations"]["abs_gain_vs_bundle_size"])
    if len(g) != int(reference["eligible_states"]) or abs(
        observed - ref_rho
    ) > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"DORIS: {len(g)} states / rho={observed} do not reproduce exp_021 "
            f"({reference['eligible_states']} / {ref_rho}); data or code differ"
        )
    if np.any(g < -m) or np.any(g > q):
        raise RuntimeError("DORIS: bound -m <= g <= q violated")

    seed = int(config["seed"]) + 1000
    rng = np.random.default_rng(seed)
    n1 = uniform_bound_null(m, q, rng)
    zero_fraction = float(np.mean(g == 0))
    p = fit_vertex_probability(m, zero_fraction)
    n2 = independent_vertex_null(m, p, rng)
    n3 = q - m

    reps = int(config["bootstrap_replicates"])
    primary = cluster_bootstrap_rho_difference(
        np.abs(g), np.abs(n1), m, clusters, replicates=reps, seed=seed + 1
    )
    vs_n2 = cluster_bootstrap_rho_difference(
        np.abs(g), np.abs(n2), m, clusters, replicates=reps, seed=seed + 2
    )

    nodes = np.asarray(sorted(graph.nodes), dtype=object)
    random_rhos: list[float] = []
    for rep in range(int(config["random_bundle_replicates"])):
        rep_rng = np.random.default_rng(seed + 10 + rep)
        g_rand = np.empty_like(g)
        for i, (history, size) in enumerate(zip(histories, m)):
            pool = nodes[~np.isin(nodes, list(history))]
            bundle = rep_rng.choice(pool, size=int(size), replace=False)
            g_rand[i] = evaluator.topological_gain(history, set(bundle))
        value = safe_spearman(np.abs(g_rand), m)
        if value is not None:
            random_rhos.append(value)

    margin = float(config["excess_margin"])
    beyond_both = (
        classify_excess(primary["ci95"], margin) == "BEYOND_BOUND"
        and classify_excess(vs_n2["ci95"], margin) == "BEYOND_BOUND"
    )
    return {
        "states": int(len(g)),
        "students": int(len(set(students))),
        "reproduces_exp_021": True,
        "observed_spearman_abs_vs_m": observed,
        "excess_over_n1": primary,
        "excess_over_n2": vs_n2,
        "classification_n1": classify_excess(primary["ci95"], margin),
        "classification_n2": classify_excess(vs_n2["ci95"], margin),
        "classification": "BEYOND_BOUND" if beyond_both else "BOUND_DOMINATED",
        "n2_fitted_vertex_probability": p,
        "observed_zero_fraction": zero_fraction,
        "n3_one_dimensional_spearman": safe_spearman(np.abs(n3), m),
        "n4_random_bundle": {
            "replicates": len(random_rhos),
            "spearman_values": random_rhos,
            "mean": float(np.mean(random_rhos)) if random_rhos else None,
        },
        "fraction_g_equals_q_minus_m": float(np.mean(g == n3)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(config_path: Path, *, s2_path: Path, edges_path: Path, feasibility_path: Path):
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

    exp018_path = REPO_ROOT / str(config["exp018_metrics"])
    exp021_path = REPO_ROOT / str(config["exp021_metrics"])
    exp018 = json.loads(exp018_path.read_text(encoding="utf-8"))
    exp021 = json.loads(exp021_path.read_text(encoding="utf-8"))

    exp018_run = _load_module(
        "experiments/exp_018_relative_chain_decomposition/run.py", "exp018_run"
    )
    domains: dict[str, object] = {}
    graph, mapping, splits, universe = exp018_run._load_processed_cs(
        REPO_ROOT / str(config["processed_cs_dir"])
    )
    domains["cs"] = _analyze_mooccube_domain(
        "cs", graph, mapping, splits, universe,
        config=config, reference=exp018, seed_offset=0,
    )
    print("cs done", flush=True)
    raw_root = resolve_mooccube2x_root(None)
    for offset, name in ((100, "math"), (200, "psy")):
        graph, mapping, splits, universe = exp018_run._load_raw_domain(
            raw_root, domain=name,
            min_sequence_length=int(config["min_sequence_length"]),
        )
        domains[name] = _analyze_mooccube_domain(
            name, graph, mapping, splits, universe,
            config=config, reference=exp018, seed_offset=offset,
        )
        print(f"{name} done", flush=True)

    doris = _analyze_doris(
        config=config, s2_path=s2_path, edges_path=edges_path,
        feasibility_path=feasibility_path, reference=exp021,
    )
    print("doris done", flush=True)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "phase": "no_topology_null_benchmark",
        "evidence_scope": "target_independent",
        "predeclaration": "docs/analysis_plan.md D-076",
        "config": config,
        "inputs": {
            "exp018_metrics_sha256": _sha256(exp018_path),
            "exp021_metrics_sha256": _sha256(exp021_path),
            "doris_edges_sha256": _sha256(edges_path),
            "doris_s2_sha256": _sha256(s2_path),
        },
        "nulls": {
            "N1": "g* ~ Uniform{-m..q} (bound only)",
            "N2": "g* = -Binomial(m, p), p fitted to observed zero fraction",
            "N3": "g* = q - m (one-dimensional circuit-rank term)",
            "N4": "same-size random bundle in the same graph (DORIS)",
        },
        "mooccube2x": domains,
        "doris": doris,
        "recommendation_metrics_evaluated": False,
        "selection_performed": False,
        "new_score_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# exp_023 - No-Topology Null Benchmark",
        "",
        "- Evidence scope: target-independent",
        "- Predeclared: docs/analysis_plan.md D-076",
        f"- Excess margin (predeclared): {config['excess_margin']}",
        "- Reproduction of exp_018 / exp_021 observed values: PASS",
        "",
        "## MOOCCube2x (within-user estimand)",
        "",
        "| Domain | Per-user median rho(|g|,m) | N1 bound-uniform | N3 q-m | "
        "Mean per-user excess over N1 [95% CI] | Classification |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("cs", "math", "psy"):
        d = domains[name]
        pu = d["per_user_median_spearman_abs_vs_m"]
        ex = d["primary_per_user_excess_over_n1"]
        lines.append(
            f"| {name.upper()} | {_fmt(pu['observed'])} | "
            f"{_fmt(pu['n1_bound_uniform'])} | "
            f"{_fmt(pu['n3_one_dimensional_q_minus_m'])} | "
            f"{_fmt(ex['mean'])} {_fmt_ci(ex['ci95'])} | {d['classification']} |"
        )
    lines += ["", "Pooled rho(|g|,m), point estimates:", ""]
    for name in ("cs", "math", "psy"):
        po = domains[name]["pooled_spearman_abs_vs_m"]
        lines.append(
            f"- {name.upper()}: observed {_fmt(po['observed'])}, "
            f"N1 {_fmt(po['n1_bound_uniform'])}, "
            f"N3 {_fmt(po['n3_one_dimensional_q_minus_m'])}; "
            f"share of candidates with g = q-m: "
            f"{domains[name]['fraction_g_equals_q_minus_m']:.4f}"
        )
    pr = doris["excess_over_n1"]
    s2 = doris["excess_over_n2"]
    lines += [
        "",
        "## DORIS v3 (across-state estimand, student-cluster bootstrap)",
        "",
        f"- States / students: {doris['states']:,} / {doris['students']:,}",
        f"- Observed rho(|g|,m): {_fmt(doris['observed_spearman_abs_vs_m'])}",
        f"- N1 bound-uniform rho: {_fmt(pr['null_rho'])}; excess "
        f"{_fmt(pr['difference'])} {_fmt_ci(pr['ci95'])} "
        f"({doris['classification_n1']})",
        f"- N2 independent-vertex rho: {_fmt(s2['null_rho'])} "
        f"(p={doris['n2_fitted_vertex_probability']:.3f}); excess "
        f"{_fmt(s2['difference'])} {_fmt_ci(s2['ci95'])} "
        f"({doris['classification_n2']})",
        f"- **PRIMARY classification (must exceed both N1 and N2): "
        f"{doris['classification']}**",
        f"- N3 q-m rho: {_fmt(doris['n3_one_dimensional_spearman'])}",
        f"- N4 random same-size bundles: mean rho "
        f"{_fmt(doris['n4_random_bundle']['mean'])} over "
        f"{doris['n4_random_bundle']['replicates']} replicates",
        f"- Share of states with g = q-m: {doris['fraction_g_equals_q_minus_m']:.4f}",
        "",
        "## Interpretation rule",
        "",
        "BEYOND_BOUND: the excess CI lower bound is at least the margin, so the "
        "observed size dependence exceeds what the bound alone produces, and the "
        "correlation may be cited together with its null. BOUND_DOMINATED: the "
        "association is reproduced by the bound; cite it only as consistent "
        "with the extensive bound. N4 asks whether the size dependence is "
        "generic for any bundle of that size in the same graph.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.config,
        s2_path=args.s2,
        edges_path=args.edges,
        feasibility_path=args.feasibility,
    )


if __name__ == "__main__":
    main()
