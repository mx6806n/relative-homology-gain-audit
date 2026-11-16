"""Run exp_019: deterministic synthetic relative-H1 counterexamples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relhom.topology import GF2RelativeHomologyEvaluator


def _load_config(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be an object")
    return payload


def _clone_graph(
    *,
    attachments: int,
    clones: int,
) -> tuple[nx.DiGraph, set[str], set[str]]:
    graph = nx.DiGraph()
    transcript = {"W"}
    graph.add_node("W")

    sources = [f"S{i}" for i in range(attachments)]
    candidates = [f"C{j}" for j in range(clones)]

    for source in sources:
        graph.add_edge("W", source)
        transcript.add(source)

    for candidate in candidates:
        graph.add_node(candidate)
        for source in sources:
            graph.add_edge(source, candidate)

    return graph, transcript, set(candidates)


def _row(
    evaluator: GF2RelativeHomologyEvaluator,
    transcript: set[str],
    candidate: set[str],
) -> dict[str, int]:
    item = evaluator.relative_h1_gain_decomposition(
        transcript,
        candidate,
    )
    return {
        "new_vertices": item.new_vertex_count,
        "absorbed_edges": item.absorbed_relative_edges,
        "delta_rank_d1": item.delta_rank_d1,
        "delta_rank_d2": item.delta_rank_d2,
        "delta_beta0": item.delta_beta0,
        "delta_beta1": item.delta_beta1,
        "reconstructed_delta_beta1": (
            item.reconstructed_delta_beta1
        ),
    }


def run(config_path: Path) -> dict[str, object]:
    config = _load_config(config_path)
    output_dir = REPO_ROOT / str(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.md"

    for path in (metrics_path, summary_path):
        if path.exists():
            raise FileExistsError(
                "exp_019 output already exists; refusing to overwrite "
                f"evidence: {path}"
            )

    max_attachments = int(config["max_clone_attachments"])
    max_clones = int(config["max_clones"])
    max_pendant_length = int(config["max_pendant_length"])

    clone_rows: list[dict[str, object]] = []
    for attachments in range(1, max_attachments + 1):
        for clones in range(1, max_clones + 1):
            graph, transcript, candidate = _clone_graph(
                attachments=attachments,
                clones=clones,
            )
            evaluator = GF2RelativeHomologyEvaluator(graph)
            decomposition = _row(
                evaluator,
                transcript,
                candidate,
            )
            expected = clones * (attachments - 1)

            if evaluator.directed_triangle_count != 0:
                raise RuntimeError("clone family must be triangle-free")
            if decomposition["delta_beta1"] != expected:
                raise RuntimeError("clone family formula failed")
            if (
                decomposition["reconstructed_delta_beta1"]
                != expected
            ):
                raise RuntimeError("clone decomposition failed")

            clone_rows.append(
                {
                    "attachments_per_clone": attachments,
                    "clones": clones,
                    "expected_gain": expected,
                    **decomposition,
                }
            )

    pendant_graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    previous = "C"
    for index in range(1, max_pendant_length + 1):
        current = f"L{index}"
        pendant_graph.add_edge(previous, current)
        previous = current

    pendant_evaluator = GF2RelativeHomologyEvaluator(
        pendant_graph
    )
    pendant_transcript = {"W", "X", "Y"}
    pendant_rows: list[dict[str, object]] = []
    for length in range(0, max_pendant_length + 1):
        candidate = {"C"}
        candidate.update(
            f"L{i}" for i in range(1, length + 1)
        )
        decomposition = _row(
            pendant_evaluator,
            pendant_transcript,
            candidate,
        )
        if decomposition["delta_beta1"] != 1:
            raise RuntimeError("pendant invariance formula failed")
        pendant_rows.append(
            {
                "pendant_length": length,
                "expected_gain": 1,
                **decomposition,
            }
        )

    filled_graph = nx.DiGraph(
        [
            ("W", "X"),
            ("W", "Y"),
            ("X", "Y"),
            ("X", "C"),
            ("Y", "C"),
        ]
    )
    filled_evaluator = GF2RelativeHomologyEvaluator(
        filled_graph
    )
    filled = _row(
        filled_evaluator,
        {"W", "X", "Y"},
        {"C"},
    )
    if filled["delta_beta1"] != 0:
        raise RuntimeError("filled-diamond compensation failed")
    if filled["absorbed_edges"] - filled["new_vertices"] != 1:
        raise RuntimeError("filled-diamond one-dimensional excess changed")
    if filled["delta_rank_d2"] != 1:
        raise RuntimeError("filled-diamond d2 compensation missing")

    payload: dict[str, object] = {
        "experiment_id": "exp_019_relative_h1_counterexamples",
        "status": "completed",
        "evidence_scope": "synthetic_theory_validation",
        "clone_family": clone_rows,
        "pendant_family": pendant_rows,
        "filled_diamond": {
            "directed_triangle_count": (
                filled_evaluator.directed_triangle_count
            ),
            **filled,
        },
        "checks": {
            "clone_formula_k_times_d_minus_one": True,
            "pendant_extension_invariance": True,
            "filled_diamond_d2_compensation": True,
            "all_candidate_decompositions_exact": True,
        },
        "recommendation_metrics_evaluated": False,
        "empirical_hyperparameters_tuned": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# exp_019 - Relative-H1 Synthetic Counterexamples",
        "",
        "- Evidence scope: synthetic theory validation",
        "- Recommendation metrics evaluated: NO",
        "- Empirical hyperparameters tuned: NO",
        "",
        "## Clone-cap family",
        "",
        (
            "For k clone concepts with d identical transcript attachments, "
            "the triangle-free proposition predicts gain k(d-1)."
        ),
        "",
    ]
    for item in clone_rows:
        lines.append(
            f"- d={int(item['attachments_per_clone'])}, "
            f"k={int(item['clones'])}: "
            f"q={int(item['absorbed_edges'])}, "
            f"m={int(item['new_vertices'])}, "
            f"gain={int(item['delta_beta1'])}"
        )

    lines.extend(
        [
            "",
            "## Pendant-extension control",
            "",
            (
                "Adding one pendant edge for every new pendant vertex keeps "
                "q-m constant and therefore preserves the core gain."
            ),
            "",
        ]
    )
    for item in pendant_rows:
        lines.append(
            f"- pendant length={int(item['pendant_length'])}: "
            f"q={int(item['absorbed_edges'])}, "
            f"m={int(item['new_vertices'])}, "
            f"gain={int(item['delta_beta1'])}"
        )

    lines.extend(
        [
            "",
            "## Filled-diamond higher-dimensional compensation",
            "",
            (
                f"- directed flag 2-simplices: "
                f"{filled_evaluator.directed_triangle_count}"
            ),
            (
                f"- absorbed edges q={filled['absorbed_edges']}, "
                f"new vertices m={filled['new_vertices']}"
            ),
            (
                f"- q-m={filled['absorbed_edges'] - filled['new_vertices']}"
            ),
            f"- Delta rank(d2)={filled['delta_rank_d2']}",
            f"- final Delta beta1={filled['delta_beta1']}",
            "",
            "## Result",
            "",
            (
                "All predeclared synthetic formulas passed exactly. "
                "The examples demonstrate both score inflation and score "
                "invariance under candidate-volume changes, governed by the "
                "relative-edge/vertex balance and higher-dimensional "
                "boundary-rank compensation."
            ),
        ]
    )
    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("exp_019_relative_h1_counterexamples")
    print(
        f"Clone family cases: {len(clone_rows)} "
        "(all k(d-1) checks PASS)"
    )
    print(
        f"Pendant cases: {len(pendant_rows)} "
        "(all invariant-gain checks PASS)"
    )
    print(
        "Filled diamond: "
        f"q-m={filled['absorbed_edges'] - filled['new_vertices']}, "
        f"Delta rank(d2)={filled['delta_rank_d2']}, "
        f"Delta beta1={filled['delta_beta1']} (PASS)"
    )
    print("Recommendation metrics evaluated: False")
    print(f"Metrics written to: {metrics_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
