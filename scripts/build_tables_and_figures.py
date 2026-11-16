"""Build the figures and tables of the article from the stored result files.

The generator uses only the Python standard library and reads nothing but the
JSON files under ``results/``. Outputs are written to ``outputs/``.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

WIDTH = 1000
HEIGHT = 650
MARGIN_LEFT = 105
MARGIN_RIGHT = 45
MARGIN_TOP = 80
MARGIN_BOTTOM = 105
PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

PALETTE = ("#1f4e79", "#b45f06", "#38761d", "#674ea7", "#990000")
GRID = "#d9d9d9"
AXIS = "#333333"
TEXT = "#111111"

EXP24_CONFIGURATIONS = (
    "global_minmax",
    "global_rank",
    "top10_minmax",
    "top10_rank",
    "top20_minmax",
    "top20_rank",
)


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _metric_value(container: dict, metric: str) -> float:
    """Read a metric from either a direct metric dict or nested metrics block."""
    if metric in container:
        return float(container[metric])
    nested = container.get("metrics")
    if isinstance(nested, dict) and metric in nested:
        return float(nested[metric])
    raise KeyError(
        f"metric {metric!r} not found directly or under a metrics block"
    )


def _binomial_two_sided_half(successes: int, trials: int) -> float:
    """Exact two-sided binomial p-value for H0: p = 0.5 (stdlib only)."""
    import math

    if trials <= 0:
        return 1.0
    k = min(successes, trials - successes)
    log_half_n = trials * math.log(0.5)
    terms = [
        math.lgamma(trials + 1) - math.lgamma(i + 1) - math.lgamma(trials - i + 1)
        + log_half_n
        for i in range(0, k + 1)
    ]
    peak = max(terms)
    tail = math.exp(peak) * sum(math.exp(x - peak) for x in terms)
    return min(1.0, 2.0 * tail)


def _esc(value: object) -> str:
    return html.escape(str(value))


def _svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 18,
    anchor: str = "middle",
    weight: str = "normal",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{TEXT}"{transform}>'
        f"{_esc(text)}</text>"
    )


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        return [low]
    return [low + (high - low) * i / count for i in range(count + 1)]


def _line_chart(
    filename: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    x_ticks: Sequence[tuple[float, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    reference_y: float | None = None,
) -> None:
    all_x = [value for _, xs, _ in series for value in xs]
    all_y = [value for _, _, ys in series for value in ys]
    if not all_x or not all_y:
        raise ValueError("line chart requires data")

    x_min, x_max = min(all_x), max(all_x)
    if x_max == x_min:
        x_max = x_min + 1.0

    if y_limits is None:
        y_min, y_max = min(all_y), max(all_y)
        span = y_max - y_min
        pad = max(span * 0.08, 0.01)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = y_limits
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return MARGIN_LEFT + (x - x_min) / (x_max - x_min) * PLOT_W

    def sy(y: float) -> float:
        return MARGIN_TOP + (y_max - y) / (y_max - y_min) * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(WIDTH / 2, 38, title, size=24, weight="bold"),
    ]

    for tick in _nice_ticks(y_min, y_max):
        y = sy(tick)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{WIDTH-MARGIN_RIGHT}" '
            f'y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_svg_text(MARGIN_LEFT - 12, y + 6, f"{tick:.3f}", size=14, anchor="end"))

    parts.extend([
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" '
        f'y2="{HEIGHT-MARGIN_BOTTOM}" stroke="{AXIS}" stroke-width="2"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{HEIGHT-MARGIN_BOTTOM}" '
        f'x2="{WIDTH-MARGIN_RIGHT}" y2="{HEIGHT-MARGIN_BOTTOM}" '
        f'stroke="{AXIS}" stroke-width="2"/>',
    ])

    ticks = x_ticks
    if ticks is None:
        ticks = [(x, f"{x:.2f}") for x in _nice_ticks(x_min, x_max)]
    for value, label in ticks:
        if value < x_min or value > x_max:
            continue
        x = sx(value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{HEIGHT-MARGIN_BOTTOM}" x2="{x:.2f}" '
            f'y2="{HEIGHT-MARGIN_BOTTOM+7}" stroke="{AXIS}" stroke-width="1"/>'
        )
        parts.append(_svg_text(x, HEIGHT - MARGIN_BOTTOM + 28, label, size=14))

    if reference_y is not None and y_min <= reference_y <= y_max:
        y = sy(reference_y)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{WIDTH-MARGIN_RIGHT}" '
            f'y2="{y:.2f}" stroke="{AXIS}" stroke-width="1.5" stroke-dasharray="8,6"/>'
        )

    legend_y = 58
    legend_x = MARGIN_LEFT
    for index, (name, xs, ys) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        coords = " ".join(
            f"{sx(float(x)):.2f},{sy(float(y)):.2f}"
            for x, y in zip(xs, ys, strict=True)
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="3"/>'
        )
        for x, y in zip(xs, ys, strict=True):
            parts.append(
                f'<circle cx="{sx(float(x)):.2f}" cy="{sy(float(y)):.2f}" '
                f'r="4.2" fill="{color}"/>'
            )
        if len(series) > 1:
            lx = legend_x + index * 240
            parts.append(
                f'<line x1="{lx}" y1="{legend_y}" x2="{lx+28}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="4"/>'
            )
            parts.append(_svg_text(lx + 36, legend_y + 6, name, size=14, anchor="start"))

    parts.append(_svg_text(WIDTH / 2, HEIGHT - 28, x_label, size=18))
    parts.append(_svg_text(27, HEIGHT / 2, y_label, size=18, rotate=-90))
    parts.append("</svg>")

    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _bar_chart(
    filename: str,
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
    y_limits: tuple[float, float],
) -> None:
    if len(labels) != len(values) or not labels:
        raise ValueError("bar chart labels and values must align")

    y_min, y_max = y_limits

    def sy(y: float) -> float:
        return MARGIN_TOP + (y_max - y) / (y_max - y_min) * PLOT_H

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(WIDTH / 2, 38, title, size=24, weight="bold"),
    ]

    for tick in _nice_ticks(y_min, y_max):
        y = sy(tick)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" y1="{y:.2f}" x2="{WIDTH-MARGIN_RIGHT}" '
            f'y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_svg_text(MARGIN_LEFT - 12, y + 6, f"{tick:.2f}", size=14, anchor="end"))

    parts.extend([
        f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" '
        f'y2="{HEIGHT-MARGIN_BOTTOM}" stroke="{AXIS}" stroke-width="2"/>',
        f'<line x1="{MARGIN_LEFT}" y1="{HEIGHT-MARGIN_BOTTOM}" '
        f'x2="{WIDTH-MARGIN_RIGHT}" y2="{HEIGHT-MARGIN_BOTTOM}" '
        f'stroke="{AXIS}" stroke-width="2"/>',
    ])

    slot = PLOT_W / len(labels)
    bar_w = slot * 0.55
    zero_y = sy(max(0.0, y_min))
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = MARGIN_LEFT + slot * index + (slot - bar_w) / 2
        y = sy(float(value))
        height = max(0.0, zero_y - y)
        color = PALETTE[index % len(PALETTE)]
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" '
            f'height="{height:.2f}" fill="{color}"/>'
        )
        parts.append(_svg_text(x + bar_w / 2, HEIGHT - MARGIN_BOTTOM + 28, label, size=15))
        parts.append(_svg_text(x + bar_w / 2, y - 10, f"{value:.3f}", size=14))

    parts.append(_svg_text(27, HEIGHT / 2, y_label, size=18, rotate=-90))
    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _correlation_interval_chart(
    filename: str,
    *,
    title: str,
    rows: Sequence[tuple[str, float, float | None, float | None]],
    x_limits: tuple[float, float] = (-0.65, 0.65),
) -> None:
    """Draw category-wise correlation estimates with optional confidence intervals."""
    if not rows:
        raise ValueError("correlation interval chart requires data")

    width = WIDTH
    height = 520
    left = 330
    right = 55
    top = 85
    bottom = 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = x_limits
    if x_max <= x_min:
        raise ValueError("x_limits must be increasing")

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 38, title, size=23, weight="bold"),
    ]

    ticks = [x_min + (x_max - x_min) * i / 6 for i in range(7)]
    for tick in ticks:
        x = sx(tick)
        stroke = AXIS if abs(tick) < 1e-12 else GRID
        stroke_width = 2.0 if abs(tick) < 1e-12 else 1.0
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{height-bottom}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        parts.append(
            _svg_text(x, height - bottom + 28, f"{tick:.2f}", size=13)
        )

    slot = plot_h / len(rows)
    for index, (label, estimate, ci_low, ci_high) in enumerate(rows):
        y = top + slot * (index + 0.5)
        color = PALETTE[index % len(PALETTE)]

        parts.append(
            _svg_text(left - 18, y + 6, label, size=15, anchor="end")
        )

        if ci_low is not None and ci_high is not None:
            x1 = sx(float(ci_low))
            x2 = sx(float(ci_high))
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" '
                f'y2="{y:.2f}" stroke="{color}" stroke-width="4"/>'
            )
            for x in (x1, x2):
                parts.append(
                    f'<line x1="{x:.2f}" y1="{y-8:.2f}" x2="{x:.2f}" '
                    f'y2="{y+8:.2f}" stroke="{color}" stroke-width="2"/>'
                )

        x_est = sx(float(estimate))
        parts.append(
            f'<circle cx="{x_est:.2f}" cy="{y:.2f}" r="7" fill="{color}"/>'
        )
        anchor = "start" if estimate < 0.50 else "end"
        label_x = x_est + 14 if anchor == "start" else x_est - 14
        parts.append(
            _svg_text(
                label_x,
                y - 12,
                f"{estimate:.3f}",
                size=13,
                anchor=anchor,
                weight="bold",
            )
        )

    parts.append(
        _svg_text(width / 2, height - 24, "Spearman correlation", size=17)
    )
    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _cross_setting_dot_figure(
    filename: str,
    *,
    settings: Sequence[str],
    abs_gain: Sequence[float],
    absorbed_edges: Sequence[float],
    abs_gain_null: Sequence[float] | None = None,
) -> None:
    """Publication-style two-panel categorical dot plot with no in-figure title.

    If ``abs_gain_null`` is given, panel (a) also shows the exp_023
    no-topology bound-uniform null as a hollow marker for every setting.
    """
    if not (
        len(settings)
        == len(abs_gain)
        == len(absorbed_edges)
        and settings
    ):
        raise ValueError("cross-setting plot inputs must align")

    width = 1240
    height = 610
    top = 70
    bottom = 92
    panel_gap = 72
    left_label = 215
    right = 52
    panel_w = (width - left_label - right - panel_gap) / 2
    plot_h = height - top - bottom
    x_min, x_max = 0.0, 1.0
    colors = ("#285A7B", "#B76516")

    def sx(value: float, panel_left: float) -> float:
        return panel_left + (value - x_min) / (x_max - x_min) * panel_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    panels = [
        ("(a)", "|g| vs m", abs_gain, colors[0]),
        ("(b)", "q vs m", absorbed_edges, colors[1]),
    ]

    for p_index, (panel_label, x_caption, values, color) in enumerate(panels):
        panel_left = left_label + p_index * (panel_w + panel_gap)
        axis_bottom = height - bottom

        parts.append(
            _svg_text(
                panel_left - 28,
                42,
                panel_label,
                size=18,
                anchor="start",
                weight="bold",
            )
        )

        for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
            x = sx(tick, panel_left)
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                f'y2="{axis_bottom}" stroke="{GRID}" stroke-width="1"/>'
            )
            parts.append(
                _svg_text(x, axis_bottom + 29, f"{tick:.2f}", size=13)
            )

        parts.append(
            f'<line x1="{panel_left:.2f}" y1="{axis_bottom}" '
            f'x2="{panel_left + panel_w:.2f}" y2="{axis_bottom}" '
            f'stroke="{AXIS}" stroke-width="1.6"/>'
        )

        slot = plot_h / len(settings)
        for i, (setting, value) in enumerate(zip(settings, values, strict=True)):
            y = top + slot * (i + 0.5)
            if p_index == 0:
                parts.append(
                    _svg_text(
                        panel_left - 18,
                        y + 5,
                        setting,
                        size=14,
                        anchor="end",
                    )
                )
            else:
                parts.append(
                    f'<line x1="{panel_left:.2f}" y1="{y:.2f}" '
                    f'x2="{panel_left + panel_w:.2f}" y2="{y:.2f}" '
                    f'stroke="#eeeeee" stroke-width="0.8"/>'
                )

            x = sx(float(value), panel_left)
            if p_index == 0 and abs_gain_null is not None:
                xn = sx(float(abs_gain_null[i]), panel_left)
                parts.append(
                    f'<line x1="{min(x, xn):.2f}" y1="{y:.2f}" '
                    f'x2="{max(x, xn):.2f}" y2="{y:.2f}" '
                    f'stroke="{color}" stroke-width="1.2" '
                    f'stroke-dasharray="3 3"/>'
                )
                parts.append(
                    f'<circle cx="{xn:.2f}" cy="{y:.2f}" r="6.2" '
                    f'fill="white" stroke="{color}" stroke-width="1.8"/>'
                )
                parts.append(
                    _svg_text(
                        xn,
                        y + 22,
                        f"{float(abs_gain_null[i]):.3f}",
                        size=11,
                    )
                )
            if setting == "DORIS v3":
                r = 7.0
                points = " ".join([
                    f"{x:.2f},{y-r:.2f}",
                    f"{x+r:.2f},{y:.2f}",
                    f"{x:.2f},{y+r:.2f}",
                    f"{x-r:.2f},{y:.2f}",
                ])
                parts.append(
                    f'<polygon points="{points}" fill="{color}" '
                    f'stroke="white" stroke-width="1.2"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.2" '
                    f'fill="{color}" stroke="white" stroke-width="1.2"/>'
                )

            parts.append(
                _svg_text(
                    min(x + 12, panel_left + panel_w - 2),
                    y - 10,
                    f"{float(value):.3f}",
                    size=12,
                    anchor="start" if x < panel_left + panel_w - 62 else "end",
                )
            )


        parts.append(
            _svg_text(
                panel_left + panel_w / 2,
                height - 28,
                f"Spearman rho ({x_caption})",
                size=15,
            )
        )

    if abs_gain_null is not None:
        lx = left_label + 40
        ly = 22
        parts.append(
            f'<circle cx="{lx:.2f}" cy="{ly:.2f}" r="6" fill="{colors[0]}"/>'
        )
        parts.append(_svg_text(lx + 12, ly + 5, "observed", size=13, anchor="start"))
        parts.append(
            f'<circle cx="{lx + 100:.2f}" cy="{ly:.2f}" r="6" fill="white" '
            f'stroke="{colors[0]}" stroke-width="1.8"/>'
        )
        parts.append(
            _svg_text(
                lx + 112, ly + 5,
                "no-topology null within the same bound",
                size=13, anchor="start",
            )
        )

    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _size_diagnosis_figure(
    filename: str,
    *,
    auc_rows: Sequence[tuple[str, float, float, float, bool]],
    ratio_rows: Sequence[tuple[str, float]],
) -> None:
    """Main Figure 4 (exp_022): target AUC within size strata + top-10 size.

    auc_rows: (label, mean, ci_low, ci_high, is_primary)
    ratio_rows: (label, median top-10 size ratio)
    No in-figure title; categorical rows are never connected by lines.
    """
    width = 1240
    height = 470
    top = 64
    bottom = 86
    left_label = 230
    right = 40
    gap = 150
    panel_w_a = 520
    panel_w_b = width - left_label - right - gap - panel_w_a
    plot_h = height - top - bottom
    topo, count, sas = "#285A7B", "#7F7F7F", "#B76516"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    a_left = left_label
    a_min, a_max = 0.45, 0.65

    def ax(v: float) -> float:
        return a_left + (v - a_min) / (a_max - a_min) * panel_w_a

    axis_bottom = height - bottom
    parts.append(_svg_text(a_left - 205, 40, "(a)", size=18, anchor="start", weight="bold"))
    for tick in (0.45, 0.50, 0.55, 0.60, 0.65):
        x = ax(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{axis_bottom}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_svg_text(x, axis_bottom + 26, f"{tick:.2f}", size=13))
    xc = ax(0.5)
    parts.append(
        f'<line x1="{xc:.2f}" y1="{top - 8}" x2="{xc:.2f}" y2="{axis_bottom}" '
        f'stroke="{AXIS}" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    parts.append(_svg_text(xc, top - 14, "chance", size=12))
    parts.append(
        f'<line x1="{a_left:.2f}" y1="{axis_bottom}" x2="{a_left + panel_w_a:.2f}" '
        f'y2="{axis_bottom}" stroke="{AXIS}" stroke-width="1.6"/>'
    )
    slot = plot_h / len(auc_rows)
    for i, (label, mean, low, high, primary) in enumerate(auc_rows):
        y = top + slot * (i + 0.5)
        color = count if label.startswith("New-concept") else topo
        parts.append(
            _svg_text(a_left - 16, y + 5, label, size=14, anchor="end",
                      weight="bold" if primary else "normal")
        )
        parts.append(
            f'<line x1="{ax(low):.2f}" y1="{y:.2f}" x2="{ax(high):.2f}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="2.4"/>'
        )
        for edge in (low, high):
            parts.append(
                f'<line x1="{ax(edge):.2f}" y1="{y - 6:.2f}" x2="{ax(edge):.2f}" '
                f'y2="{y + 6:.2f}" stroke="{color}" stroke-width="1.6"/>'
            )
        parts.append(
            f'<circle cx="{ax(mean):.2f}" cy="{y:.2f}" r="6.4" fill="{color}" '
            f'stroke="white" stroke-width="1.2"/>'
        )
        parts.append(
            _svg_text(ax(high) + 12, y + 5, f"{mean:.3f}", size=12, anchor="start")
        )
    parts.append(
        _svg_text(a_left + panel_w_a / 2, height - 30,
                  "Target AUC (tie-aware; 95% CI)", size=15)
    )

    b_left = a_left + panel_w_a + gap
    b_min, b_max = 0.0, 6.0

    def bx(v: float) -> float:
        return b_left + (v - b_min) / (b_max - b_min) * panel_w_b

    parts.append(_svg_text(b_left - 110, 40, "(b)", size=18, anchor="start", weight="bold"))
    for tick in (0, 1, 2, 3, 4, 5, 6):
        x = bx(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{axis_bottom}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(_svg_text(x, axis_bottom + 26, f"{tick}", size=13))
    x1 = bx(1.0)
    parts.append(
        f'<line x1="{x1:.2f}" y1="{top - 8}" x2="{x1:.2f}" y2="{axis_bottom}" '
        f'stroke="{AXIS}" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    parts.append(_svg_text(x1, top - 14, "pool average", size=12))
    parts.append(
        f'<line x1="{b_left:.2f}" y1="{axis_bottom}" x2="{b_left + panel_w_b:.2f}" '
        f'y2="{axis_bottom}" stroke="{AXIS}" stroke-width="1.6"/>'
    )
    slot_b = plot_h / len(ratio_rows)
    for i, (label, ratio) in enumerate(ratio_rows):
        y = top + slot_b * (i + 0.5)
        color = sas if label.startswith("SASRec") else topo
        parts.append(_svg_text(b_left - 14, y + 5, label, size=14, anchor="end"))
        parts.append(
            f'<rect x="{bx(0):.2f}" y="{y - 11:.2f}" width="{bx(ratio) - bx(0):.2f}" '
            f'height="22" fill="{color}" opacity="0.85"/>'
        )
        parts.append(
            _svg_text(bx(ratio) + 10, y + 5, f"{ratio:.2f}x", size=12, anchor="start")
        )
    parts.append(
        _svg_text(b_left + panel_w_b / 2, height - 30,
                  "Median top-10 size / pool size", size=15)
    )

    parts.append("</svg>")
    (OUT / filename).write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_size_diagnosis_figure() -> None:
    exp22 = _load("results/exp_022_size_stratified_diagnostic/metrics.json")
    auc = exp22["auc"]

    def row(key: str, label: str, primary: bool = False):
        item = auc[key]
        return (label, float(item["mean"]), float(item["ci95"][0]),
                float(item["ci95"][1]), primary)

    auc_rows = [
        row("delta_unstratified", "Raw gain, all candidates"),
        row("delta_within_exact_m", "Raw gain, equal size m", True),
        row("delta_within_exact_m_and_q", "Raw gain, equal (m, q)"),
        row("m_unstratified", "New-concept count m"),
    ]
    enrich = exp22["topk_enrichment"]
    ratio_rows = [
        ("Raw gain", float(enrich["delta_topk_by_new_concepts"]["median_size_ratio"])),
        ("SASRec", float(enrich["sasrec_topk_by_new_concepts"]["median_size_ratio"])),
    ]
    _size_diagnosis_figure(
        "fig3_size_diagnosis.svg",
        auc_rows=auc_rows,
        ratio_rows=ratio_rows,
    )


def build_global_hybrid_figure() -> None:
    data = _load("results/exp_012_hybrid_alpha_ablation/metrics.json")
    rows = data["signed"]["results"]
    x = [float(row["alpha"]) for row in rows]
    y = [float(row["metrics"]["ndcg@10"]) for row in rows]
    baseline = float(data["signed"]["selected"]["metrics"]["ndcg@10"])

    _line_chart(
        "fig4a_global_fusion.svg",
        title="Global raw-topology fusion",
        x_label="SASRec weight alpha",
        y_label="Validation NDCG@10",
        series=[("Signed raw topology", x, y)],
        y_limits=(0.05, 0.43),
        reference_y=baseline,
    )


def build_topn_figure() -> None:
    data = _load("results/exp_013_topn_topology_rerank/metrics.json")
    series = []
    for shortlist in data["shortlists"]:
        top_n = int(shortlist["top_n"])
        rows = shortlist["results"]
        series.append(
            (
                f"Top-N={top_n}",
                [float(row["alpha"]) for row in rows],
                [float(row["metrics"]["ndcg@10"]) for row in rows],
            )
        )

    _line_chart(
        "fig4b_shortlist_reranking.svg",
        title="Local topology reranking",
        x_label="SASRec weight alpha",
        y_label="Validation NDCG@10",
        series=series,
        y_limits=(0.20, 0.43),
        reference_y=_metric_value(data["sasrec_baseline"], "ndcg@10"),
    )


def build_mechanism_figure() -> None:
    data = _load("results/exp_018_relative_chain_decomposition/metrics.json")
    domains = ["cs", "math", "psy"]
    labels = ["CS", "Math", "Psychology"]
    x = [0.0, 1.0, 2.0]

    delta_new = [
        float(data["domains"][d]["correlations"]["abs_delta_vs_new_concepts"]["global_spearman"])
        for d in domains
    ]
    edge_new = [
        float(data["domains"][d]["correlations"]["edge_drop_vs_new_concepts"]["global_spearman"])
        for d in domains
    ]

    _line_chart(
        "extra_cross_domain_correlations.svg",
        title="Cross-domain structural-volume mechanism",
        x_label="Prerequisite domain",
        y_label="Global Spearman correlation",
        series=[
            ("|Delta beta1| vs new concepts", x, delta_new),
            ("Absorbed edges vs new concepts", x, edge_new),
        ],
        x_ticks=list(zip(x, labels, strict=True)),
        y_limits=(0.0, 1.0),
    )

    cancellation = [
        float(data["domains"][d]["component_summary"]["mean_cancellation_fraction"])
        for d in domains
    ]
    _bar_chart(
        "extra_cancellation.svg",
        title="Cancellation among exact chain/rank terms",
        labels=labels,
        values=cancellation,
        y_label="Mean cancellation fraction",
        y_limits=(0.0, 1.0),
    )


def build_exact_controls_figure() -> None:
    """Three-panel vector schematic for the exact synthetic controls."""
    width = 1320
    height = 570
    top = 70
    bottom = 72
    gap = 42
    left = 46
    panel_w = (width - 2 * left - 2 * gap) / 3

    transcript_fill = "#D9EAD3"
    candidate_fill = "#F9CB9C"
    simplex_fill = "#D9EAF7"
    edge = "#333333"
    muted = "#777777"

    def node(x: float, y: float, label: str, fill: str) -> str:
        return (
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="19" '
            f'fill="{fill}" stroke="{edge}" stroke-width="1.6"/>'
            + _svg_text(x, y + 5, label, size=13, weight="bold")
        )

    def arrow(x1: float, y1: float, x2: float, y2: float, *, color: str = edge) -> str:
        return (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#arrow)"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<defs>',
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" '
        'orient="auto" markerUnits="strokeWidth">',
        f'<polygon points="0 0, 8 3.5, 0 7" fill="{edge}"/>',
        '</marker>',
        '</defs>',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    panel_lefts = [
        left,
        left + panel_w + gap,
        left + 2 * (panel_w + gap),
    ]
    for i, px in enumerate(panel_lefts):
        if i:
            sep_x = px - gap / 2
            parts.append(
                f'<line x1="{sep_x:.2f}" y1="48" x2="{sep_x:.2f}" '
                f'y2="{height-42}" stroke="#dddddd" stroke-width="1"/>'
            )
        parts.append(
            _svg_text(px, 36, f"({chr(ord('a') + i)})", size=18, anchor="start", weight="bold")
        )

    px = panel_lefts[0]
    cx = px + panel_w / 2
    parts.append(_svg_text(cx, 66, "Clone-cap inflation", size=17, weight="bold"))

    wy = 126
    sy = 230
    cy = 348
    parts.append(node(cx, wy, "W", transcript_fill))
    source_x = [cx - 95, cx, cx + 95]
    cand_x = [cx - 95, cx, cx + 95]

    for j, sx in enumerate(source_x, start=1):
        parts.append(arrow(cx, wy + 20, sx, sy - 22))
        parts.append(node(sx, sy, f"S{j}", transcript_fill))
    for j, qx in enumerate(cand_x, start=1):
        parts.append(node(qx, cy, f"C{j}", candidate_fill))
    for sx in source_x:
        for qx in cand_x:
            parts.append(arrow(sx, sy + 21, qx, cy - 22))

    parts.append(_svg_text(cx, 416, "triangle-free", size=13, weight="bold"))
    parts.append(
        _svg_text(cx, 448, "m = k,   q = kd,   t = h = 0", size=14)
    )
    parts.append(
        _svg_text(cx, 480, "g = k(d - 1)", size=17, weight="bold")
    )
    parts.append(
        _svg_text(cx, 506, "representation multiplicity can change raw gain", size=12)
    )

    px = panel_lefts[1]
    cx = px + panel_w / 2
    parts.append(_svg_text(cx, 66, "Pendant invariance", size=17, weight="bold"))

    W=(cx,126); X=(cx-82,210); Y=(cx+82,210); C=(cx,298)
    L1=(cx,366); L2=(cx,430)
    for a,b in [(W,X),(W,Y),(X,C),(Y,C),(C,L1),(L1,L2)]:
        parts.append(arrow(a[0],a[1]+18,b[0],b[1]-20))
    for pos,label in [(W,"W"),(X,"X"),(Y,"Y")]:
        parts.append(node(pos[0],pos[1],label,transcript_fill))
    for pos,label in [(C,"C"),(L1,"L1"),(L2,"L2")]:
        parts.append(node(pos[0],pos[1],label,candidate_fill))

    parts.append(
        _svg_text(cx, 476, "each extension:  Δq = Δm = 1", size=14)
    )
    parts.append(
        _svg_text(cx, 507, "g = 1 for every tested chain length", size=16, weight="bold")
    )

    px = panel_lefts[2]
    cx = px + panel_w / 2
    parts.append(_svg_text(cx, 66, "Higher-order compensation", size=17, weight="bold"))

    W=(cx,132); X=(cx-92,242); Y=(cx+42,242); C=(cx+92,356)
    parts.append(
        f'<polygon points="{W[0]:.2f},{W[1]:.2f} {X[0]:.2f},{X[1]:.2f} '
        f'{Y[0]:.2f},{Y[1]:.2f}" fill="{simplex_fill}" opacity="0.75"/>'
    )
    parts.append(
        f'<polygon points="{X[0]:.2f},{X[1]:.2f} {Y[0]:.2f},{Y[1]:.2f} '
        f'{C[0]:.2f},{C[1]:.2f}" fill="{simplex_fill}" opacity="0.75"/>'
    )
    for a,b in [(W,X),(W,Y),(X,Y),(X,C),(Y,C)]:
        parts.append(arrow(a[0],a[1]+18,b[0],b[1]-20))
    for pos,label in [(W,"W"),(X,"X"),(Y,"Y")]:
        parts.append(node(pos[0],pos[1],label,transcript_fill))
    parts.append(node(C[0],C[1],"C",candidate_fill))

    parts.append(_svg_text(cx, 418, "q = 2,   m = 1", size=14))
    parts.append(
        _svg_text(cx, 450, "q - m = 1,   t = 1", size=14)
    )
    parts.append(
        _svg_text(cx, 482, "g = q - m - t = 0", size=17, weight="bold")
    )
    parts.append(
        _svg_text(cx, 508, "2-simplex filling cancels the 1D excess", size=12)
    )

    legend_y = height - 24
    parts.append(
        f'<circle cx="470" cy="{legend_y}" r="8" fill="{transcript_fill}" '
        f'stroke="{edge}" stroke-width="1"/>'
    )
    parts.append(_svg_text(485, legend_y + 5, "transcript", size=12, anchor="start"))
    parts.append(
        f'<circle cx="575" cy="{legend_y}" r="8" fill="{candidate_fill}" '
        f'stroke="{edge}" stroke-width="1"/>'
    )
    parts.append(_svg_text(590, legend_y + 5, "candidate", size=12, anchor="start"))
    parts.append(
        f'<rect x="690" y="{legend_y-8}" width="16" height="16" '
        f'fill="{simplex_fill}" stroke="none"/>'
    )
    parts.append(_svg_text(713, legend_y + 5, "directed 2-simplex", size=12, anchor="start"))

    parts.append("</svg>")
    (OUT / "fig2_exact_controls.svg").write_text(
        "\n".join(parts) + "\n",
        encoding="utf-8",
    )


def build_synthetic_figure() -> None:
    data = _load("results/exp_019_relative_h1_counterexamples/metrics.json")

    clone_series = []
    for d in sorted({int(row["attachments_per_clone"]) for row in data["clone_family"]}):
        rows = [
            row for row in data["clone_family"]
            if int(row["attachments_per_clone"]) == d
        ]
        rows.sort(key=lambda row: int(row["clones"]))
        clone_series.append(
            (
                f"d={d}",
                [float(row["clones"]) for row in rows],
                [float(row["delta_beta1"]) for row in rows],
            )
        )

    _line_chart(
        "extra_clone_cap.svg",
        title="Clone-cap family: g = k(d-1)",
        x_label="Clone multiplicity k",
        y_label="Raw relative-H1 gain",
        series=clone_series,
        x_ticks=[(float(k), str(k)) for k in range(1, 6)],
        y_limits=(-0.5, 15.8),
    )

    rows = sorted(data["pendant_family"], key=lambda row: int(row["pendant_length"]))
    x = [float(row["new_vertices"]) for row in rows]
    y = [float(row["delta_beta1"]) for row in rows]
    _line_chart(
        "extra_pendant_invariance.svg",
        title="Pendant extension: cardinality grows, gain stays invariant",
        x_label="Candidate vertices",
        y_label="Raw relative-H1 gain",
        series=[("Pendant family", x, y)],
        x_ticks=[(float(k), str(k)) for k in range(1, 7)],
        y_limits=(0.0, 2.0),
    )


def build_cross_setting_mechanism_figure() -> None:
    exp18 = _load("results/exp_018_relative_chain_decomposition/metrics.json")
    exp21 = _load(
        "results/exp_021_doris_v3_semester_bundle_mechanism/metrics.json"
    )

    domains = ("cs", "math", "psy")
    settings = (
        "MOOCCube2x CS",
        "MOOCCube2x Math",
        "MOOCCube2x Psychology",
        "DORIS v3",
    )
    abs_gain = [
        float(
            exp18["domains"][d]["correlations"]["abs_delta_vs_new_concepts"][
                "global_spearman"
            ]
        )
        for d in domains
    ] + [float(exp21["correlations"]["abs_gain_vs_bundle_size"])]

    absorbed_edges = [
        float(
            exp18["domains"][d]["correlations"]["edge_drop_vs_new_concepts"][
                "global_spearman"
            ]
        )
        for d in domains
    ] + [float(exp21["correlations"]["absorbed_edges_vs_bundle_size"])]

    exp23 = _load("results/exp_023_bound_null_benchmark/metrics.json")
    abs_gain_null = [
        float(exp23["mooccube2x"][d]["pooled_spearman_abs_vs_m"]["n1_bound_uniform"])
        for d in domains
    ] + [float(exp23["doris"]["excess_over_n1"]["null_rho"])]

    _cross_setting_dot_figure(
        "extra_size_dependence_vs_null.svg",
        settings=settings,
        abs_gain=abs_gain,
        absorbed_edges=absorbed_edges,
        abs_gain_null=abs_gain_null,
    )


def build_external_replication_figure() -> None:
    data = _load(
        "results/exp_021_doris_v3_semester_bundle_mechanism/metrics.json"
    )
    primary_ci = data["cluster_bootstrap"]["abs_gain_vs_bundle_size"][
        "ci95_percentile"
    ]
    edge_ci = data["cluster_bootstrap"]["absorbed_edges_vs_bundle_size"][
        "ci95_percentile"
    ]

    _correlation_interval_chart(
        "extra_doris_structural.svg",
        title="DORIS v3 external semester-bundle replication",
        rows=[
            (
                "|gain| vs bundle size m (95% CI)",
                float(data["correlations"]["abs_gain_vs_bundle_size"]),
                float(primary_ci[0]),
                float(primary_ci[1]),
            ),
            (
                "Absorbed edges q vs m (95% CI)",
                float(data["correlations"]["absorbed_edges_vs_bundle_size"]),
                float(edge_ci[0]),
                float(edge_ci[1]),
            ),
            (
                "Signed gain g vs m",
                float(data["correlations"]["gain_vs_bundle_size"]),
                None,
                None,
            ),
        ],
    )


def build_tables() -> None:
    exp2 = _load("results/exp_002_popularity_unseen/metrics.json")
    exp3 = _load("results/exp_003_markov_baseline/metrics.json")
    exp4 = _load("results/exp_004_bpr_mf_pilot/metrics.json")
    exp7 = _load("results/exp_007_sasrec_multiseed/metrics.json")
    exp12 = _load("results/exp_012_hybrid_alpha_ablation/metrics.json")
    exp13 = _load("results/exp_013_topn_topology_rerank/metrics.json")
    exp18 = _load("results/exp_018_relative_chain_decomposition/metrics.json")
    exp19 = _load("results/exp_019_relative_h1_counterexamples/metrics.json")
    exp21 = _load(
        "results/exp_021_doris_v3_semester_bundle_mechanism/metrics.json"
    )
    exp24 = _load("results/exp_024_normalization_sensitivity/metrics.json")

    sasrec_reference = float(exp13["sasrec_baseline"]["ndcg@10"])
    for key in EXP24_CONFIGURATIONS:
        cfg = exp24["configurations"][key]
        if abs(float(cfg["sasrec_ndcg@10"]) - sasrec_reference) > 1e-12:
            raise RuntimeError(
                f"exp_024 {key}: SASRec reference {cfg['sasrec_ndcg@10']!r} "
                f"differs from exp_013 {sasrec_reference!r}"
            )

    tables_md = [
        "# Tables generated from the result files",
        "",
        "## SASRec five-seed validation",
        "",
    ]

    agg = exp7["validation_aggregate"]["ndcg@10"]
    tables_md.extend([
        f"- NDCG@10 mean: {float(agg['mean']):.6f}",
        f"- SD: {float(agg['std']):.6f}",
        f"- 95% CI: [{float(agg['ci95_low']):.6f}, {float(agg['ci95_high']):.6f}]",
        "",
        "## Raw topology integration",
        "",
        f"- Global selected alpha: {float(exp12['signed']['selected']['alpha']):.2f}",
        f"- Global NDCG@10: {float(exp12['signed']['selected']['metrics']['ndcg@10']):.6f}",
    ])

    for shortlist in exp13["shortlists"]:
        rows = shortlist["results"]
        selected = max(
            rows,
            key=lambda row: (
                float(row["metrics"]["ndcg@10"]),
                float(row["alpha"]),
            ),
        )
        tables_md.append(
            f"- Top-N={int(shortlist['top_n'])}: alpha={float(selected['alpha']):.2f}, "
            f"NDCG@10={float(selected['metrics']['ndcg@10']):.6f}"
        )

    fine_alphas = ", ".join(
        f"{float(a):.2f}" for a in exp24["config"]["fine_alphas"]
    )
    tables_md.extend([
        "",
        "### exp_024 normalization and near-tie sensitivity",
        "",
        f"- Overall: {exp24['overall_classification']}",
        f"- Alpha grid: {len(exp24['grid'])} values "
        f"(original {len(exp24['config']['frozen_alphas'])} plus {fine_alphas})",
    ])
    for key in EXP24_CONFIGURATIONS:
        cfg = exp24["configurations"][key]
        boot = cfg["paired_bootstrap_vs_sasrec"]
        tie = cfg["near_tie"]
        tie_text = (
            "no rank movement at any alpha < 1"
            if tie is None
            else (
                f"near-tie alpha={float(tie['alpha']):.2f}: "
                f"{int(tie['improved'])} improved / {int(tie['worsened'])} worsened "
                f"(p={float(tie['two_sided_binomial_p']):.3g})"
            )
        )
        tables_md.append(
            f"- {cfg['integration']}, {cfg['topology_normalization']}: "
            f"alpha={float(cfg['selected_alpha']):.2f}, "
            f"NDCG@10={float(cfg['selected_ndcg@10']):.6f}, "
            f"ΔNDCG@10={float(boot['mean_difference']):+.6f} "
            f"[{float(boot['ci95'][0]):+.6f}, {float(boot['ci95'][1]):+.6f}], "
            f"{tie_text}, {cfg['classification']}"
        )

    tables_md.extend(["", "## Exact decomposition", ""])
    for domain in ("cs", "math", "psy"):
        row = exp18["domains"][domain]
        tables_md.append(
            f"- {domain.upper()}: states={int(row['candidate_states']):,}, "
            f"identity mismatches={int(row['identity_mismatches'])}, "
            f"cancellation={float(row['component_summary']['mean_cancellation_fraction']):.4f}"
        )

    tables_md.extend([
        "",
        "## Synthetic checks",
        "",
        f"- Clone formula: {exp19['checks']['clone_formula_k_times_d_minus_one']}",
        f"- Pendant invariance: {exp19['checks']['pendant_extension_invariance']}",
        f"- d2 compensation: {exp19['checks']['filled_diamond_d2_compensation']}",
        f"- Exact decompositions: {exp19['checks']['all_candidate_decompositions_exact']}",
        "",
        "## DORIS v3 external structural replication",
        "",
        f"- Graph nodes/edges/triangles: {int(exp21['graph']['nodes']):,}/"
        f"{int(exp21['graph']['edges']):,}/"
        f"{int(exp21['graph']['directed_triangles']):,}",
        f"- Eligible students/states: {int(exp21['eligible_students']):,}/"
        f"{int(exp21['eligible_states']):,}",
        f"- Held-out edge-review precision: "
        f"{float(exp21['inputs']['manual_precision']):.3f}",
        f"- Spearman(|gain|, m): "
        f"{float(exp21['correlations']['abs_gain_vs_bundle_size']):.6f}",
        f"- Student-cluster bootstrap 95% CI: "
        f"[{float(exp21['cluster_bootstrap']['abs_gain_vs_bundle_size']['ci95_percentile'][0]):.6f}, "
        f"{float(exp21['cluster_bootstrap']['abs_gain_vs_bundle_size']['ci95_percentile'][1]):.6f}]",
        f"- Spearman(gain, m): "
        f"{float(exp21['correlations']['gain_vs_bundle_size']):.6f}",
        f"- Spearman(q, m): "
        f"{float(exp21['correlations']['absorbed_edges_vs_bundle_size']):.6f}",
        f"- Decomposition mismatches: "
        f"{int(exp21['exact_checks']['decomposition_identity_mismatches'])}",
        f"- d1 identity mismatches: "
        f"{int(exp21['exact_checks']['delta_rank_d1_equals_m_minus_h_mismatches'])}",
        "- Recommendation metrics evaluated: NO",
        "- Corrected score evaluated: NO",
        "",
    ])

    (OUT / "summary_tables.md").write_text(
        "\n".join(tables_md),
        encoding="utf-8",
    )

    with (OUT / "decomposition_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "domain", "users", "courses", "candidate_states", "graph_nodes",
            "graph_edges", "triangles", "identity_mismatches",
            "mean_abs_delta", "mean_edge_drop", "mean_abs_d1_drop",
            "mean_abs_d2_drop", "mean_cancellation"
        ])
        for domain in ("cs", "math", "psy"):
            row = exp18["domains"][domain]
            summary = row["component_summary"]
            writer.writerow([
                domain,
                row["users"],
                row["courses"],
                row["candidate_states"],
                row["graph_nodes"],
                row["graph_edges"],
                row["directed_triangles"],
                row["identity_mismatches"],
                summary["mean_abs_delta"],
                summary["mean_edge_drop"],
                summary["mean_abs_d1_rank_drop"],
                summary["mean_abs_d2_rank_drop"],
                summary["mean_cancellation_fraction"],
            ])


    with (OUT / "table8_9_recommendation.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "panel",
            "method",
            "evidence_status",
            "replicates",
            "mrr",
            "recall_at_5",
            "ndcg_at_5",
            "recall_at_10",
            "ndcg_at_10",
            "recall_at_20",
            "ndcg_at_20",
            "ndcg_at_10_sd",
            "selected_alpha",
            "shortlist_n",
            "delta_ndcg_at_10_vs_sasrec_reference",
        ])

        baseline_rows = [
            (
                "Popularity",
                "deterministic baseline",
                exp2["validation"],
                1,
                None,
            ),
            (
                "First-order Markov",
                "deterministic sequential baseline",
                exp3["validation"],
                1,
                None,
            ),
            (
                "BPR-MF",
                "single-seed exploratory learned baseline",
                exp4["validation"],
                1,
                None,
            ),
        ]
        for method, status, metrics, replicates, ndcg_sd in baseline_rows:
            writer.writerow([
                "A: standard baselines",
                method,
                status,
                replicates,
                metrics["mrr"],
                metrics["recall@5"],
                metrics["ndcg@5"],
                metrics["recall@10"],
                metrics["ndcg@10"],
                metrics["recall@20"],
                metrics["ndcg@20"],
                ndcg_sd,
                "",
                "",
                "",
            ])

        sasrec_metrics = {
            metric: exp7["validation_aggregate"][metric]["mean"]
            for metric in (
                "mrr",
                "recall@5",
                "ndcg@5",
                "recall@10",
                "ndcg@10",
                "recall@20",
                "ndcg@20",
            )
        }
        writer.writerow([
            "A: standard baselines",
            "SASRec",
            "frozen five-seed baseline confirmation",
            exp7["validation_aggregate"]["ndcg@10"]["n"],
            sasrec_metrics["mrr"],
            sasrec_metrics["recall@5"],
            sasrec_metrics["ndcg@5"],
            sasrec_metrics["recall@10"],
            sasrec_metrics["ndcg@10"],
            sasrec_metrics["recall@20"],
            sasrec_metrics["ndcg@20"],
            exp7["validation_aggregate"]["ndcg@10"]["std"],
            "",
            "",
            "",
        ])

        sasrec_ref = exp13["sasrec_baseline"]
        writer.writerow([
            "B: raw-topology hypothesis test",
            "SASRec reference",
            "exact cached reference used by exp_012/013",
            1,
            sasrec_ref["mrr"],
            sasrec_ref["recall@5"],
            sasrec_ref["ndcg@5"],
            sasrec_ref["recall@10"],
            sasrec_ref["ndcg@10"],
            sasrec_ref["recall@20"],
            sasrec_ref["ndcg@20"],
            "",
            1.0,
            "",
            0.0,
        ])

        global_selected = exp12["signed"]["selected"]
        gm = global_selected["metrics"]
        writer.writerow([
            "B: raw-topology hypothesis test",
            "Global raw-topology fusion",
            "validation-selected; alpha=1 means pure SASRec",
            1,
            gm["mrr"],
            gm["recall@5"],
            gm["ndcg@5"],
            gm["recall@10"],
            gm["ndcg@10"],
            gm["recall@20"],
            gm["ndcg@20"],
            "",
            global_selected["alpha"],
            "full catalog",
            gm["ndcg@10"] - sasrec_ref["ndcg@10"],
        ])

        for shortlist in exp13["shortlists"]:
            selected = max(
                shortlist["results"],
                key=lambda row: (
                    float(row["metrics"]["ndcg@10"]),
                    float(row["alpha"]),
                ),
            )
            metrics = selected["metrics"]
            writer.writerow([
                "B: raw-topology hypothesis test",
                f"Local raw-topology reranking (N={int(shortlist['top_n'])})",
                "validation-selected; alpha=1 means pure SASRec",
                1,
                metrics["mrr"],
                metrics["recall@5"],
                metrics["ndcg@5"],
                metrics["recall@10"],
                metrics["ndcg@10"],
                metrics["recall@20"],
                metrics["ndcg@20"],
                "",
                selected["alpha"],
                int(shortlist["top_n"]),
                metrics["ndcg@10"] - sasrec_ref["ndcg@10"],
            ])


    with (OUT / "table5_comparability_controls.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "result_or_control",
            "condition_or_construction",
            "exact_statement",
            "verification",
            "evidence_scope",
            "interpretation_boundary",
        ])

        mooc_states = sum(
            int(exp18["domains"][domain]["candidate_states"])
            for domain in ("cs", "math", "psy")
        )
        doris_states = int(exp21["eligible_states"])
        total_states = mooc_states + doris_states
        total_mismatches = (
            sum(
                int(exp18["domains"][domain]["identity_mismatches"])
                for domain in ("cs", "math", "psy")
            )
            + int(
                exp21["exact_checks"]["decomposition_identity_mismatches"]
            )
        )
        writer.writerow([
            "Exact candidate decomposition",
            "Any finite directed flag complex and induced transcript/candidate subcomplexes",
            "g = q - m + h - t",
            (
                "independent cross-checks: pyflagser mapping cone "
                "(exhaustive toy subsets; 25 real states, exp_008) and an "
                "independent re-implementation (2,316 random states, "
                "scripts/verify_relative_h1.py). The empirical reconstruction "
                f"({total_mismatches} mismatches over {total_states} states) "
                "holds by construction and is not counted as evidence"
            ),
            "formal theorem + independent computational cross-checks",
            "An exact identity; not a predictive-performance claim",
        ])

        writer.writerow([
            "Universal score bounds",
            "General case",
            "-m <= g <= q",
            "proved from 0 <= h <= m and 0 <= t <= q",
            "formal",
            "Bounds do not imply that large candidates are always favored",
        ])

        writer.writerow([
            "Connected nonempty-transcript condition",
            "|K| connected and S nonempty",
            "g = q - m - t; positive g requires q > m + t",
            "formal corollary",
            "formal",
            "Necessary condition for positive gain, not a utility preference",
        ])

        writer.writerow([
            "Extension-invariance criterion",
            "Candidate extension D = C union R",
            "Delta q + Delta h = Delta m + Delta t",
            "proved exactly; pendant family supplies a constructive control",
            "formal + synthetic",
            "Comparability issue arises only when the application expects semantic neutrality/invariance",
        ])

        clone_pass = sum(
            int(row["delta_beta1"]) == int(row["expected_gain"])
            for row in exp19["clone_family"]
        )
        writer.writerow([
            "Clone-cap sensitivity",
            "Connected triangle-free clone family",
            "g = k(d - 1)",
            f"{clone_pass}/{len(exp19['clone_family'])} predeclared cases exact",
            "synthetic theory validation",
            "Shows representation-granularity sensitivity, not topological equivalence of clones",
        ])

        pendant_pass = sum(
            int(row["delta_beta1"]) == int(row["expected_gain"]) == 1
            for row in exp19["pendant_family"]
        )
        writer.writerow([
            "Pendant invariance",
            "Hollow-diamond candidate extended by a directed pendant chain",
            "Delta q = Delta m for every extension; g = 1",
            f"{pendant_pass}/{len(exp19['pendant_family'])} predeclared cases exact",
            "synthetic theory validation",
            "Candidate volume alone does not determine gain",
        ])

        filled = exp19["filled_diamond"]
        writer.writerow([
            "Higher-order compensation",
            "Filled-diamond directed flag complex",
            "q - m = 1 and t = 1, hence g = 0",
            (
                f"q={int(filled['absorbed_edges'])}, "
                f"m={int(filled['new_vertices'])}, "
                f"t={int(filled['delta_rank_d2'])}, "
                f"g={int(filled['delta_beta1'])}"
            ),
            "synthetic theory validation",
            "No simple gain-per-vertex correction follows from the theorem",
        ])

        writer.writerow([
            "Disconnected-component compensation",
            "Candidate first reaches a previously untouched ambient component",
            "m=1, q=0, h=1, t=0 implies g=0",
            "formal control retained in the proof/regression package",
            "formal + regression",
            "The h term cannot be dropped unless connectedness/nonempty-transcript assumptions are stated",
        ])

    exp13_near_tie = {}
    for shortlist in exp13["shortlists"]:
        moving = [
            r for r in shortlist["results"]
            if float(r["alpha"]) < 1.0
            and (r["rank_movement_vs_sasrec"]["improved"]
                 + r["rank_movement_vs_sasrec"]["worsened"]) > 0
        ]
        nearest = max(moving, key=lambda r: float(r["alpha"]))
        exp13_near_tie[f"top{int(shortlist['top_n'])}_minmax"] = (
            float(nearest["alpha"]),
            int(nearest["rank_movement_vs_sasrec"]["improved"]),
            int(nearest["rank_movement_vs_sasrec"]["worsened"]),
        )

    with (OUT / "table10_near_tie_sign_tests.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "panel",
            "integration",
            "topology_normalization",
            "selected_alpha",
            "selected_ndcg_at_10",
            "near_tie_alpha",
            "targets_improved",
            "targets_worsened",
            "worsened_share",
            "two_sided_exact_binomial_p",
            "classification",
        ])
        for key in EXP24_CONFIGURATIONS:
            cfg = exp24["configurations"][key]
            if key in exp13_near_tie:
                alpha13, imp13, wor13 = exp13_near_tie[key]
                row24 = next(
                    r for r in cfg["grid"] if float(r["alpha"]) == alpha13
                )
                if (int(row24["improved"]), int(row24["worsened"])) != (imp13, wor13):
                    raise RuntimeError(
                        f"exp_024 {key} at alpha={alpha13}: rank movement "
                        f"{row24['improved']}/{row24['worsened']} differs from "
                        f"exp_013 {imp13}/{wor13}"
                    )
            tie = cfg["near_tie"]
            if tie is None:
                near_alpha = imp = wor = share = p_value = ""
            else:
                near_alpha = float(tie["alpha"])
                imp = int(tie["improved"])
                wor = int(tie["worsened"])
                share = wor / (imp + wor)
                p_value = float(tie["two_sided_binomial_p"])
                recomputed = _binomial_two_sided_half(wor, imp + wor)
                if abs(recomputed - p_value) > 1e-9 * p_value:
                    raise RuntimeError(
                        f"exp_024 {key}: stored p={p_value!r} differs from the "
                        f"standard-library value {recomputed!r}"
                    )
            integration = str(cfg["integration"])
            writer.writerow([
                "C: near-tie sign test",
                integration[:1].upper() + integration[1:],
                cfg["topology_normalization"],
                float(cfg["selected_alpha"]),
                float(cfg["selected_ndcg@10"]),
                near_alpha,
                imp,
                wor,
                share,
                p_value,
                cfg["classification"],
            ])

    exp22 = _load("results/exp_022_size_stratified_diagnostic/metrics.json")
    exp23 = _load("results/exp_023_bound_null_benchmark/metrics.json")

    with (OUT / "table6_size_diagnosis.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "quantity",
            "comparison_set",
            "value",
            "ci95_low",
            "ci95_high",
            "users",
            "reference",
            "note",
        ])
        auc_rows = (
            ("delta_unstratified", "Raw-gain target AUC", "all unseen candidates", ""),
            ("delta_within_exact_m", "Raw-gain target AUC (PRIMARY)", "candidates with equal new-concept count m", "predeclared (docs/analysis_plan.md, D-074)"),
            ("delta_within_m_bins", "Raw-gain target AUC", "candidates in the same predeclared m bin", ""),
            ("delta_within_exact_m_and_q", "Raw-gain target AUC", "candidates with equal (m, q)", "varies only through h - t"),
            ("m_unstratified", "New-concept-count target AUC", "all unseen candidates", "size alone"),
            ("sasrec_unstratified", "SASRec target AUC", "all unseen candidates", "reference model"),
            ("sasrec_within_exact_m", "SASRec target AUC", "candidates with equal m", "reference model"),
        )
        for key, quantity, comparison, note in auc_rows:
            item = exp22["auc"][key]
            writer.writerow([
                quantity, comparison, item["mean"], item["ci95"][0],
                item["ci95"][1], item["n"], "chance = 0.5", note,
            ])
        for key, label in (
            ("delta_topk_by_new_concepts", "Raw-gain top-10"),
            ("sasrec_topk_by_new_concepts", "SASRec top-10"),
        ):
            item = exp22["topk_enrichment"][key]
            writer.writerow([
                f"{label} median size ratio (top-10 mean m / pool mean m)",
                "per user",
                item["median_size_ratio"], "", "", item["users"],
                "pool average = 1",
                f"share of users with ratio > 1: {item['share_users_ratio_above_1']:.4f}",
            ])
        decision = exp22["primary"]["decision"]
        diff = exp22["primary"]["minus_unstratified_paired"]
        writer.writerow([
            "Primary minus unstratified (paired)", "same users",
            diff["mean"], diff["ci95"][0], diff["ci95"][1], diff["n"],
            f"predeclared outcome: {decision['outcome']}",
            "minimum meaningful AUC 0.55",
        ])

    with (OUT / "table7_size_dependence_vs_null.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "setting",
            "dataset",
            "topology_granularity",
            "users",
            "evaluated_states",
            "rho_abs_gain_vs_m_pooled",
            "null_n1_rho_pooled",
            "rho_q_vs_m_pooled",
            "primary_estimand",
            "primary_excess_over_null",
            "primary_excess_ci95_low",
            "primary_excess_ci95_high",
            "classification",
            "note",
        ])
        labels = {
            "cs": "MOOCCube2x CS",
            "math": "MOOCCube2x Math",
            "psy": "MOOCCube2x Psychology",
        }
        for domain in ("cs", "math", "psy"):
            row = exp18["domains"][domain]
            null = exp23["mooccube2x"][domain]
            excess = null["primary_per_user_excess_over_n1"]
            writer.writerow([
                labels[domain], "MOOCCube2x", "concept-level",
                row["users"], row["candidate_states"],
                row["correlations"]["abs_delta_vs_new_concepts"]["global_spearman"],
                null["pooled_spearman_abs_vs_m"]["n1_bound_uniform"],
                row["correlations"]["edge_drop_vs_new_concepts"]["global_spearman"],
                "mean per-user excess over N1 (user bootstrap)",
                excess["mean"], excess["ci95"][0], excess["ci95"][1],
                null["classification"],
                "consistent with the extensive bound",
            ])
        doris = exp23["doris"]
        writer.writerow([
            "DORIS v3", "DORIS", "course-level",
            exp21["eligible_students"], exp21["eligible_states"],
            exp21["correlations"]["abs_gain_vs_bundle_size"],
            doris["excess_over_n1"]["null_rho"],
            exp21["correlations"]["absorbed_edges_vs_bundle_size"],
            "excess over N1 and N2 (student-cluster bootstrap)",
            doris["excess_over_n1"]["difference"],
            doris["excess_over_n1"]["ci95"][0],
            doris["excess_over_n1"]["ci95"][1],
            doris["classification"],
            (
                f"excess over N2 {doris['excess_over_n2']['difference']:.4f}; "
                f"random same-size bundles rho {doris['n4_random_bundle']['mean']:.4f}"
            ),
        ])

    with (OUT / "table2_analysis_settings.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "setting",
            "dataset",
            "evidence_role",
            "topology_granularity",
            "graph_nodes",
            "graph_edges",
            "directed_triangles",
            "users",
            "evaluated_states",
            "candidate_definition",
            "sequence_or_state_definition",
        ])
        moo_roles = {
            "cs": (
                "MOOCCube2x CS",
                "primary recommendation + mechanism",
            ),
            "math": (
                "MOOCCube2x Math",
                "target-independent mechanism replication",
            ),
            "psy": (
                "MOOCCube2x Psychology",
                "target-independent mechanism replication",
            ),
        }
        for domain in ("cs", "math", "psy"):
            row = exp18["domains"][domain]
            label, role = moo_roles[domain]
            writer.writerow([
                label,
                "MOOCCube2x",
                role,
                "concept-level directed prerequisite flag complex",
                row["graph_nodes"],
                row["graph_edges"],
                row["directed_triangles"],
                row["users"],
                row["candidate_states"],
                "course candidate mapped to newly introduced prerequisite concepts",
                "aligned enrollment history; target-independent candidate-state analysis",
            ])
        writer.writerow([
            "DORIS v3",
            "DORIS",
            "external target-independent structural replication",
            "course-level directed prerequisite flag complex",
            exp21["graph"]["nodes"],
            exp21["graph"]["edges"],
            exp21["graph"]["directed_triangles"],
            exp21["eligible_students"],
            exp21["eligible_states"],
            "previously unseen graph-mapped courses in the current semester",
            "strictly prior semesters form S; current-semester bundle forms C",
        ])

    with (OUT / "doris_structural_table.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        primary_ci = exp21["cluster_bootstrap"]["abs_gain_vs_bundle_size"][
            "ci95_percentile"
        ]
        edge_ci = exp21["cluster_bootstrap"]["absorbed_edges_vs_bundle_size"][
            "ci95_percentile"
        ]
        writer.writerow([
            "dataset",
            "graph_nodes",
            "graph_edges",
            "directed_triangles",
            "eligible_students",
            "eligible_states",
            "bundle_mean",
            "bundle_median",
            "bundle_fraction_ge2",
            "holdout_edge_precision",
            "rho_abs_gain_vs_m",
            "rho_abs_gain_vs_m_ci95_low",
            "rho_abs_gain_vs_m_ci95_high",
            "rho_gain_vs_m",
            "rho_q_vs_m",
            "rho_q_vs_m_ci95_low",
            "rho_q_vs_m_ci95_high",
            "decomposition_mismatches",
            "d1_identity_mismatches",
        ])
        writer.writerow([
            "DORIS v3",
            exp21["graph"]["nodes"],
            exp21["graph"]["edges"],
            exp21["graph"]["directed_triangles"],
            exp21["eligible_students"],
            exp21["eligible_states"],
            exp21["bundle_size"]["mean"],
            exp21["bundle_size"]["median"],
            exp21["bundle_size"]["fraction_ge2"],
            exp21["inputs"]["manual_precision"],
            exp21["correlations"]["abs_gain_vs_bundle_size"],
            primary_ci[0],
            primary_ci[1],
            exp21["correlations"]["gain_vs_bundle_size"],
            exp21["correlations"]["absorbed_edges_vs_bundle_size"],
            edge_ci[0],
            edge_ci[1],
            exp21["exact_checks"]["decomposition_identity_mismatches"],
            exp21["exact_checks"]["delta_rank_d1_equals_m_minus_h_mismatches"],
        ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_global_hybrid_figure()
    build_topn_figure()
    build_mechanism_figure()
    build_synthetic_figure()
    build_exact_controls_figure()
    build_cross_setting_mechanism_figure()
    build_size_diagnosis_figure()
    build_external_replication_figure()
    build_tables()
    print(f"Figures and tables written to: {OUT}")


if __name__ == "__main__":
    main()
