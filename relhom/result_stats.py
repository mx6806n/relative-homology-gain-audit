"""Aggregate stochastic experiment metrics across random seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from statistics import mean, stdev

from scipy.stats import t as student_t


def summarize_seed_metrics(
    runs: Sequence[Mapping[str, float | int]],
    metric_names: Sequence[str],
    *,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Return mean, sample SD, and Student-t CI across independent seeds."""
    if len(runs) < 2:
        raise ValueError("at least two seed runs are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not metric_names:
        raise ValueError("metric_names must not be empty")

    n = len(runs)
    critical = float(
        student_t.ppf(
            (1.0 + confidence) / 2.0,
            df=n - 1,
        )
    )

    summary: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        values: list[float] = []
        for run in runs:
            if metric not in run:
                raise ValueError(
                    f"metric {metric!r} missing from a seed run"
                )
            values.append(float(run[metric]))

        average = mean(values)
        sample_sd = stdev(values)
        margin = critical * sample_sd / sqrt(n)
        summary[metric] = {
            "n": n,
            "mean": average,
            "std": sample_sd,
            "ci95_low": average - margin,
            "ci95_high": average + margin,
            "min": min(values),
            "max": max(values),
        }

    return summary
