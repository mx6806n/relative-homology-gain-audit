"""Finalize DORIS v3 feasibility from the disjoint manual holdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(
        (p * (1.0 - p) / n) + (z2 / (4.0 * n * n))
    ) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _load_labels(path: Path) -> tuple[int, int]:
    labels: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "manual_supported_0_or_1" not in (reader.fieldnames or []):
            raise ValueError(
                "manual holdout CSV must contain manual_supported_0_or_1"
            )
        for row_number, row in enumerate(reader, start=2):
            raw = (row.get("manual_supported_0_or_1") or "").strip()
            if raw not in {"0", "1"}:
                raise ValueError(
                    f"row {row_number}: manual_supported_0_or_1 must be 0 or 1; "
                    f"got {raw!r}"
                )
            labels.append(int(raw))

    if not labels:
        raise ValueError("manual holdout contains no labeled rows")
    return sum(labels), len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("results/doris_feasibility_audit_v3.json"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("results/doris_prerequisite_manual_holdout_v3.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/doris_feasibility_final_v3.json"),
    )
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if int(audit.get("audit_version", 0)) != 3:
        raise ValueError("finalizer requires DORIS audit_version=3")

    supported, n = _load_labels(args.review)
    precision = supported / n
    threshold = float(
        audit["feasibility_thresholds"]["manual_edge_precision"]
    )
    ci_low, ci_high = _wilson_interval(supported, n)

    nonmanual_pass = bool(
        audit.get("preliminary_all_nonmanual_gates_pass", False)
    )
    manual_pass = precision >= threshold
    final_pass = nonmanual_pass and manual_pass

    payload = {
        "status": "feasible" if final_pass else "not_feasible",
        "audit_version": 3,
        "nonmanual_gates_pass": nonmanual_pass,
        "manual_review": {
            "sample_size": n,
            "supported_edges": supported,
            "unsupported_edges": n - supported,
            "precision": precision,
            "threshold": threshold,
            "gate_pass": manual_pass,
            "wilson_95_ci": [ci_low, ci_high],
            "ci_is_descriptive_not_gate": True,
            "review_is_disjoint_from_v2_sample": True,
        },
        "final_feasibility_pass": final_pass,
        "decision_rule": (
            "final_feasibility_pass = all unchanged non-manual gates pass "
            "AND disjoint manual holdout precision >= unchanged 0.90 threshold"
        ),
        "source_audit": str(args.audit),
        "source_manual_review": str(args.review),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Final DORIS v3 feasibility written to: {args.output}")


if __name__ == "__main__":
    main()
