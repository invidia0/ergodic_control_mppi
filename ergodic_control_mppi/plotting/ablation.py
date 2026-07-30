"""Publication figures for the ablation campaign.

Reads only the archive (``experiments/analyze.load_index`` / ``load_run``), so
re-plotting never touches the GPU.

    python -m ergodic_control_mppi.plotting.ablation --stage all

Colour policy (plotting/style.py): interaction heatmaps show % change against the
shipped default on a diverging map centred at zero -- blue better, red worse;
timing uses single-hue tints; nothing rainbow.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from ergodic_control_mppi.experiments.analyze import (
    load_index,
    load_series,
    steps_to_threshold,
    summarize_stage,
)
from ergodic_control_mppi.plotting.style import (
    ACCENT,
    DIVERGING_CMAP,
    NEUTRAL,
    PRIMARY,
    paper_style,
    save,
)

METRIC_LABELS = {
    "occupancy_mse": r"Occupancy MSE  $\mathcal{E}$",
    "fourier_ergodic": r"Ergodic metric  $\varepsilon$",
    "ms_per_step": "Time per step [ms]",
    "steps_to_threshold": r"Steps to threshold",
}

AXIS_LABELS = {
    "stein.memory_time": r"$\tau_{\mathcal{M}}$ [s]",
    "stein.memory_balance": r"$a$",
    "stein.memory_gain": r"$k_{\mathcal{M}}$",
    "stein.memory_scales": r"$Q$",
    "stein.fill_resolution": r"$\delta_{\mathrm{res}}$ [m]",
    "stein.fine_bandwidth": r"$h_f$",
    "stein.coarse_bandwidth": r"$h_c$",
    "stein.theta": r"$\theta$ [deg]",
    "stein.weight_stein": r"$\gamma$ (flow weight)",
    "stein.ell_self": r"$\ell_0$",
    "stein.reference_speed": r"$v$ [m/s]",
    "mppi.lambda": r"$\lambda$",
    "mppi.T": r"$T$",
    "mppi.K": r"$K$",
    "mppi.memory_length": r"$P$",
    "map.obstacles.num_obstacles": "obstacles",
}

LOG_AXES = {"stein.weight_stein", "mppi.lambda", "stein.coarse_bandwidth", "stein.ell_self"}

ARM_LABELS = {
    "full": "Full", "default": "Full",
    "memory_off": r"$k_{\mathcal{M}}=0$",
    "trail_only": r"$a=0$ (trail)",
    "excess_only": r"$a=1$ (excess)",
    "fine_only": r"$Q=1$ (fine)",
    "one_good_scale": r"$Q=1$ (midpoint)",
    "two_scale": r"$Q=2$",
    "no_curl": r"$\theta=0$",
    "weak_flow": r"low $\gamma$",
    "no_speed_gauge": r"$v=0$",
    "coarse_tuned": r"$h_c=3$",
    "long_buffer": r"$P=5\tau$",
    "best": "Best",
}


def _variant_label(row: dict[str, Any]) -> str:
    """Human label for a summary row: the arm name, or the axis=level it sets."""
    axes = json.loads(row.get("axes") or "{}")
    if not axes:
        return ARM_LABELS.get(row["arm"], row["arm"])
    parts = [f"{_label(name)} = {_format_level(float(value))}" for name, value in sorted(axes.items())]
    return ", ".join(parts)


def _label(name: str) -> str:
    return AXIS_LABELS.get(name, name.split(".")[-1].replace("_", " "))


def _by_axis(rows: list[dict[str, str]], metric: str) -> dict[str, dict[float, np.ndarray]]:
    """OFAT rows -> {axis: {level: values over seeds}}."""
    out: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        axes = json.loads(row.get("axes") or "{}")
        if len(axes) != 1:
            continue
        (axis, level), = axes.items()
        out[axis][float(level)].append(float(row[metric]))
    return {
        axis: {level: np.asarray(values) for level, values in sorted(levels.items())}
        for axis, levels in out.items()
    }


def _baseline(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    values = [
        float(row[metric]) for row in rows
        if (json.loads(row.get("axes") or "{}") == {}) or row["arm"] in ("default", "full")
    ]
    return np.asarray(values, dtype=np.float64)


def plot_tornado(
    campaign_dir: Path, stage: str, output: Path, metric: str = "occupancy_mse"
) -> Path:
    """Rank every swept axis by the span its levels induce in the metric.

    The first figure the reader should see: what matters, before any detail.
    """
    rows = load_index(campaign_dir, stage)
    grouped = _by_axis(rows, metric)
    base = _baseline(rows, metric)
    reference = float(np.median(base)) if base.size else np.nan

    entries = []
    for axis, levels in grouped.items():
        medians = np.asarray([np.median(v) for v in levels.values()])
        # Include the default so the span covers the level set actually explored.
        medians = np.append(medians, reference)
        span = 100.0 * (medians.max() - medians.min()) / reference
        entries.append((axis, span, 100.0 * (medians.min() - reference) / reference,
                        100.0 * (medians.max() - reference) / reference))
    entries.sort(key=lambda e: e[1])

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.35, 0.24 * len(entries) + 0.85))
        positions = np.arange(len(entries))
        for position, (_, _, low, high) in zip(positions, entries):
            axis.plot([low, high], [position, position], color=NEUTRAL, linewidth=3.2,
                      solid_capstyle="butt", zorder=2)
            axis.plot([low], [position], "o", color=PRIMARY, markersize=3.0, zorder=3)
            axis.plot([high], [position], "o", color=ACCENT, markersize=3.0, zorder=3)
        axis.axvline(0.0, color="#33415C", linewidth=0.7, zorder=1)
        axis.set_yticks(positions)
        axis.set_yticklabels([_label(name) for name, *_ in entries])
        axis.set_xlabel(f"change in {METRIC_LABELS.get(metric, metric)} vs default [%]")
        axis.set_title("Parameter influence, ranked")
        axis.grid(axis="y", visible=False)
        figure.tight_layout(pad=0.3)
        path = save(figure, output)
        plt.close(figure)
    return path


def plot_violins(
    campaign_dir: Path,
    stage: str,
    output: Path,
    metric: str = "occupancy_mse",
    columns: int = 4,
) -> Path:
    """One violin panel per axis: level on x, seed spread on y."""
    rows = load_index(campaign_dir, stage)
    grouped = _by_axis(rows, metric)
    base = _baseline(rows, metric)
    reference = float(np.median(base)) if base.size else np.nan
    names = sorted(grouped, key=lambda n: -max(np.median(v) for v in grouped[n].values()))

    count = len(names)
    rows_n = int(np.ceil(count / columns))
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(
            rows_n, columns, figsize=(6.9, 1.75 * rows_n), squeeze=False
        )
        for index, name in enumerate(names):
            axis = axes[index // columns][index % columns]
            levels = grouped[name]
            # Include the default level so each panel shows a complete sweep.
            merged = dict(levels)
            data = [merged[level] for level in sorted(merged)]
            xs = np.arange(len(data))
            parts = axis.violinplot(data, positions=xs, widths=0.7,
                                    showextrema=False, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(PRIMARY)
                body.set_alpha(0.45)
                body.set_edgecolor(PRIMARY)
                body.set_linewidth(0.5)
            parts["cmedians"].set_color("#23272F")
            parts["cmedians"].set_linewidth(0.9)
            for position, values in zip(xs, data):
                axis.plot(np.full(len(values), position), values, ".", color="#23272F",
                          markersize=1.6, alpha=0.75)
            if np.isfinite(reference):
                axis.axhline(reference, color=ACCENT, linewidth=0.7, linestyle="--")
            axis.set_xticks(xs)
            axis.set_xticklabels([_format_level(level) for level in sorted(merged)],
                                 fontsize=5.5, rotation=45, ha="right")
            axis.set_xlabel(_label(name), labelpad=1)
            if index % columns == 0:
                axis.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=6.5)
            axis.tick_params(axis="y", labelsize=5.5)
            n_seeds = min(len(v) for v in data)
            axis.set_title(f"$n={n_seeds}$", fontsize=6.0, fontweight="normal")
            if name in LOG_AXES:
                axis.set_yscale("log")
        for index in range(count, rows_n * columns):
            axes[index // columns][index % columns].axis("off")
        figure.tight_layout(pad=0.35, w_pad=0.7, h_pad=0.8)
        path = save(figure, output)
        plt.close(figure)
    return path


def _format_level(level: float) -> str:
    return f"{int(level)}" if float(level).is_integer() else f"{level:g}"


def plot_interactions(
    campaign_dir: Path, stage: str, output: Path, metric: str = "occupancy_mse"
) -> Path:
    """Grid of pairwise heatmaps: % change vs default, diverging, shared colorbar."""
    rows = load_index(campaign_dir, stage)
    base = _baseline(rows, metric)

    pairs: dict[str, dict[tuple[float, float], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    axis_names: dict[str, tuple[str, str]] = {}
    for row in rows:
        axes = json.loads(row.get("axes") or "{}")
        if len(axes) != 2:
            continue
        arm = row["arm"]
        (name_a, level_a), (name_b, level_b) = sorted(axes.items())
        axis_names[arm] = (name_a, name_b)
        pairs[arm][(float(level_a), float(level_b))].append(float(row[metric]))

    if not pairs:
        raise ValueError(f"stage '{stage}' has no two-axis cells to plot")

    # Reference: the default cell if present, else the best observed cell, so the
    # diverging scale still has a defensible centre.
    if base.size:
        reference = float(np.median(base))
    else:
        reference = float(
            min(np.median(v) for cells in pairs.values() for v in cells.values())
        )

    names = sorted(pairs)
    columns = min(3, len(names))
    rows_n = int(np.ceil(len(names) / columns))
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(
            rows_n, columns, figsize=(6.9, 2.35 * rows_n), squeeze=False
        )
        images = []
        for index, arm in enumerate(names):
            axis = axes[index // columns][index % columns]
            cells = pairs[arm]
            levels_a = sorted({key[0] for key in cells})
            levels_b = sorted({key[1] for key in cells})
            grid = np.full((len(levels_b), len(levels_a)), np.nan)
            for (level_a, level_b), values in cells.items():
                grid[levels_b.index(level_b), levels_a.index(level_a)] = np.median(values)
            change = 100.0 * (grid - reference) / reference
            limit = float(np.nanmax(np.abs(change))) or 1.0
            image = axis.pcolormesh(
                np.arange(len(levels_a) + 1) - 0.5,
                np.arange(len(levels_b) + 1) - 0.5,
                change, cmap=DIVERGING_CMAP,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
                shading="flat",
            )
            images.append(image)
            name_a, name_b = axis_names[arm]
            axis.set_xticks(np.arange(len(levels_a)))
            axis.set_xticklabels([_format_level(v) for v in levels_a], fontsize=5.5,
                                 rotation=45, ha="right")
            axis.set_yticks(np.arange(len(levels_b)))
            axis.set_yticklabels([_format_level(v) for v in levels_b], fontsize=5.5)
            axis.set_xlabel(_label(name_a), labelpad=1)
            axis.set_ylabel(_label(name_b), labelpad=1)
            axis.set_title(f"{_label(name_a)} $\\times$ {_label(name_b)}", fontsize=7.0)
            axis.grid(visible=False)
            # Mark the best cell so the reader is not left eyeballing the colour.
            flat = np.nanargmin(grid)
            best_b, best_a = np.unravel_index(flat, grid.shape)
            axis.plot(best_a, best_b, "*", color="#111111", markersize=5.0,
                      markeredgewidth=0)
            # The reach L = v*T*dt is predicted to set performance on this pair,
            # so overlay its iso-contours as a falsifiable claim.
            if {name_a, name_b} == {"stein.reference_speed", "mppi.T"}:
                _iso_reach(axis, levels_a, levels_b)

            bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
            bar.set_label("% vs default", labelpad=1, fontsize=6)
            bar.ax.tick_params(labelsize=5.5)
        for index in range(len(names), rows_n * columns):
            axes[index // columns][index % columns].axis("off")
        figure.tight_layout(pad=0.35, w_pad=0.9, h_pad=0.9)
        path = save(figure, output)
        plt.close(figure)
    return path


def _iso_reach(axis, levels_a, levels_b, delta_t: float = 0.02) -> None:
    """Overlay iso-``L = v*T*dt`` contours on the speed x horizon panel.

    ``levels_a`` is the x axis and ``levels_b`` the y axis, both drawn at integer
    positions, so the mesh is built on the same index grid the pcolormesh uses.
    Which of the two is the speed does not matter: the reach is their product.
    """
    grid_a, grid_b = np.meshgrid(
        np.asarray(levels_a, dtype=np.float64), np.asarray(levels_b, dtype=np.float64)
    )
    reach = grid_a * grid_b * delta_t
    contours = axis.contour(
        np.arange(len(levels_a)), np.arange(len(levels_b)), reach,
        levels=[2.0, 5.0, 10.0, 20.0, 40.0], colors="#111111",
        linewidths=0.5, alpha=0.65,
    )
    axis.clabel(contours, inline=True, fontsize=4.5, fmt="L=%.0f")


def plot_arms(
    campaign_dir: Path, stage: str, output: Path, threshold_factor: float = 1.5
) -> Path:
    """Structural-arm bars: error, spectral metric, transient, with win counts."""
    summary = summarize_stage(campaign_dir, stage, threshold_factor=threshold_factor)
    summary = [row for row in summary if row["arm"] != "default" or len(summary) == 1]
    order = sorted(summary, key=lambda r: r["occupancy_mse_median"])
    labels = [ARM_LABELS.get(row["arm"], row["arm"]) for row in order]
    metrics = ("occupancy_mse", "fourier_ergodic", "steps_to_threshold")

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.6))
        for axis, metric in zip(axes, metrics):
            key = f"{metric}_median"
            values = np.asarray([float(row.get(key, np.nan)) for row in order])
            errors = np.asarray([float(row.get(f"{metric}_iqr", 0.0) or 0.0) for row in order])
            positions = np.arange(len(order))
            colors = [
                ACCENT if row["arm"] in ("full", "default") else PRIMARY for row in order
            ]
            axis.barh(positions, values, xerr=errors if errors.any() else None,
                      color=colors, alpha=0.9, height=0.72,
                      error_kw={"ecolor": "#33415C", "elinewidth": 0.7, "capsize": 1.5})
            axis.set_yticks(positions)
            axis.set_yticklabels(labels if metric == metrics[0] else [], fontsize=6)
            axis.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=6.5)
            axis.tick_params(axis="x", labelsize=5.5)
            axis.grid(axis="y", visible=False)
            axis.invert_yaxis()
            if metric == "occupancy_mse":
                for position, row in zip(positions, order):
                    wins = row.get("occupancy_mse_wins")
                    if wins:
                        axis.text(values[position], position, f"  {wins}", va="center",
                                  fontsize=5.0, color="#33415C")
        figure.suptitle("Structural ablation (median over seeds, IQR bars)", fontsize=8)
        figure.tight_layout(pad=0.35, w_pad=0.6, rect=(0, 0, 1, 0.94))
        path = save(figure, output)
        plt.close(figure)
    return path


def plot_convergence(
    campaign_dir: Path, stage: str, output: Path, threshold_factor: float = 1.5
) -> Path:
    """Convergence bands plus the transient metric that quantifies them."""
    rows = load_index(campaign_dir, stage)
    # Group by arm AND axes: on an OFAT stage every level shares an arm name, and
    # merging distinct configurations into one median band would be quietly wrong.
    by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_arm[_variant_label(row)].append(row)

    series_by_arm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, members in by_arm.items():
        collected = []
        steps = None
        for member in members:
            steps, values = load_series(campaign_dir, member["stage"], member["cell_id"])
            collected.append(values)
        shortest = min(len(v) for v in collected)
        stacked = np.stack([v[-shortest:] for v in collected], axis=0)
        series_by_arm[arm] = (steps[-shortest:], stacked)

    finals = series_by_arm.get("full", series_by_arm.get("default"))
    threshold = (
        threshold_factor * float(np.median(finals[1][:, -1])) if finals is not None else np.nan
    )

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.6),
                                    gridspec_kw={"width_ratios": [1.5, 1.0]})
        axis = axes[0]
        for index, (arm, (steps, stacked)) in enumerate(sorted(series_by_arm.items())):
            median = np.median(stacked, axis=0)
            low, high = np.quantile(stacked, [0.25, 0.75], axis=0)
            axis.plot(steps, median, linewidth=0.9, label=ARM_LABELS.get(arm, arm))
            axis.fill_between(steps, low, high, alpha=0.16, linewidth=0)
        if np.isfinite(threshold):
            axis.axhline(threshold, color="#111111", linewidth=0.7, linestyle="--")
            axis.text(steps[len(steps) // 2], threshold, r"  $\tau$", fontsize=6,
                      va="bottom", color="#111111")
        axis.set_yscale("log")
        axis.set_xlabel("Step")
        axis.set_ylabel(METRIC_LABELS["occupancy_mse"])
        axis.set_title("(a) convergence, median and IQR")
        axis.legend(fontsize=5.0, ncol=2, loc="upper right", handlelength=1.1,
                    labelspacing=0.2, columnspacing=0.8)

        axis = axes[1]
        data, labels, censored = [], [], []
        for arm, (steps, stacked) in sorted(series_by_arm.items()):
            reached = [
                steps_to_threshold(steps, stacked[i], threshold)
                for i in range(stacked.shape[0])
            ]
            finite = [v for v in reached if np.isfinite(v)]
            if not finite:
                continue
            data.append(finite)
            labels.append(ARM_LABELS.get(arm, arm))
            censored.append(len(reached) - len(finite))
        if data:
            positions = np.arange(len(data))
            parts = axis.violinplot(data, positions=positions, widths=0.7,
                                    showextrema=False, showmedians=True)
            for body in parts["bodies"]:
                body.set_facecolor(PRIMARY)
                body.set_alpha(0.45)
                body.set_edgecolor(PRIMARY)
                body.set_linewidth(0.5)
            parts["cmedians"].set_color("#23272F")
            for position, values in zip(positions, data):
                axis.plot(np.full(len(values), position), values, ".", color="#23272F",
                          markersize=1.8, alpha=0.8)
            for position, count in zip(positions, censored):
                if count:
                    axis.text(position, axis.get_ylim()[1], f"+{count}", ha="center",
                              va="top", fontsize=5.0, color=ACCENT)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, fontsize=5.5, rotation=45, ha="right")
        axis.set_ylabel(METRIC_LABELS["steps_to_threshold"], fontsize=6.5)
        axis.set_title(rf"(b) transient ($\tau={threshold_factor:g}\times$ full)")
        figure.tight_layout(pad=0.35, w_pad=0.9)
        path = save(figure, output)
        plt.close(figure)
    return path


def plot_generalization(
    campaign_dir: Path, stage: str, output: Path, metric: str = "occupancy_mse"
) -> Path:
    """Density x obstacle-count grid for each arm, % change vs the `full` arm."""
    rows = load_index(campaign_dir, stage)
    cells: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        axes = json.loads(row.get("axes") or "{}")
        count = float(axes.get("map.obstacles.num_obstacles", np.nan))
        cells[(row["arm"], row["density"], count)].append(float(row[metric]))
    medians = {key: float(np.median(v)) for key, v in cells.items()}

    arms = sorted({key[0] for key in medians} - {"full"})
    densities = sorted({key[1] for key in medians})
    counts = sorted({key[2] for key in medians})

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(
            1, max(len(arms), 1), figsize=(6.9, 2.6), squeeze=False
        )
        for index, arm in enumerate(arms):
            axis = axes[0][index]
            grid = np.full((len(counts), len(densities)), np.nan)
            for row_index, count in enumerate(counts):
                for column, density in enumerate(densities):
                    mine = medians.get((arm, density, count))
                    theirs = medians.get(("full", density, count))
                    if mine is not None and theirs:
                        grid[row_index, column] = 100.0 * (mine - theirs) / theirs
            limit = float(np.nanmax(np.abs(grid))) or 1.0
            image = axis.pcolormesh(
                np.arange(len(densities) + 1) - 0.5,
                np.arange(len(counts) + 1) - 0.5,
                grid, cmap=DIVERGING_CMAP,
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), shading="flat",
            )
            for row_index in range(len(counts)):
                for column in range(len(densities)):
                    if np.isfinite(grid[row_index, column]):
                        axis.text(column, row_index, f"{grid[row_index, column]:+.0f}",
                                  ha="center", va="center", fontsize=5.0, color="#111111")
            axis.set_xticks(np.arange(len(densities)))
            axis.set_xticklabels(densities, fontsize=5.5, rotation=45, ha="right")
            axis.set_yticks(np.arange(len(counts)))
            axis.set_yticklabels([f"{int(c)}" for c in counts], fontsize=5.5)
            axis.set_xlabel("target density", labelpad=1)
            if index == 0:
                axis.set_ylabel("obstacles", labelpad=1)
            axis.set_title(f"{ARM_LABELS.get(arm, arm)} vs Full", fontsize=7)
            axis.grid(visible=False)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03).ax.tick_params(
                labelsize=5.5
            )
        figure.suptitle("Generalization: % change vs Full (blue better)", fontsize=8)
        figure.tight_layout(pad=0.35, w_pad=0.9, rect=(0, 0, 1, 0.93))
        path = save(figure, output)
        plt.close(figure)
    return path


STRUCTURE_ROWS = {"a0": r"$a=0$", "a085": r"$a=0.85$", "a1": r"$a=1$"}
STRUCTURE_COLS = {
    "Q1f": r"$Q{=}1$ ($h_f$)", "Q1m": r"$Q{=}1$ (mid)",
    "Q2": r"$Q{=}2$", "Q3": r"$Q{=}3$",
}


def plot_structure(
    campaign_dir: Path, stage: str, output: Path, metric: str = "occupancy_mse"
) -> Path:
    """Balance x scale-bank matrix for the structural cross.

    Arm names are the grid coordinates (``<balance>_<scale>``); arms that do not
    parse -- ``memory_off``, where both axes are meaningless -- are drawn as a
    reference line instead of forced into a cell.
    """
    rows = load_index(campaign_dir, stage)
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    outside: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        parts = row["arm"].split("_")
        if len(parts) == 2 and parts[0] in STRUCTURE_ROWS and parts[1] in STRUCTURE_COLS:
            cells[(parts[0], parts[1])].append(float(row[metric]))
        else:
            outside[row["arm"]].append(float(row[metric]))
    if not cells:
        raise ValueError(f"stage '{stage}' has no <balance>_<scale> arms to plot")

    row_keys = [k for k in STRUCTURE_ROWS if any(k == r for r, _ in cells)]
    col_keys = [k for k in STRUCTURE_COLS if any(k == c for _, c in cells)]
    grid = np.full((len(row_keys), len(col_keys)), np.nan)
    for (r, c), values in cells.items():
        grid[row_keys.index(r), col_keys.index(c)] = np.median(values)

    # Centre on the shipped default so the map reads as "better/worse than what
    # we ship", consistent with every other interaction figure.
    default = grid[row_keys.index("a085"), col_keys.index("Q3")] if (
        "a085" in row_keys and "Q3" in col_keys
    ) else np.nanmedian(grid)
    change = 100.0 * (grid - default) / default
    limit = float(np.nanmax(np.abs(change))) or 1.0

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.35, 2.5))
        image = axis.pcolormesh(
            np.arange(len(col_keys) + 1) - 0.5, np.arange(len(row_keys) + 1) - 0.5,
            change, cmap=DIVERGING_CMAP,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit), shading="flat",
        )
        for r in range(len(row_keys)):
            for c in range(len(col_keys)):
                if np.isfinite(change[r, c]):
                    axis.text(c, r, f"{change[r, c]:+.0f}", ha="center", va="center",
                              fontsize=6.0, color="#111111")
        # Offset into the cell corner: the best cell is often the default, whose
        # "+0" label would otherwise sit under the marker.
        best_r, best_c = np.unravel_index(np.nanargmin(grid), grid.shape)
        axis.plot(best_c - 0.33, best_r - 0.30, "*", color="#111111",
                  markersize=5.5, markeredgewidth=0)
        axis.set_xticks(np.arange(len(col_keys)))
        axis.set_xticklabels([STRUCTURE_COLS[c] for c in col_keys], fontsize=6)
        axis.set_yticks(np.arange(len(row_keys)))
        axis.set_yticklabels([STRUCTURE_ROWS[r] for r in row_keys], fontsize=6)
        axis.set_xlabel("scale bank")
        axis.set_ylabel("trail / excess balance")
        axis.set_title("% vs shipped default (blue better)", fontsize=7)
        axis.grid(visible=False)
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=5.5)
        if outside:
            note = ", ".join(
                f"{ARM_LABELS.get(k, k)} {100.0 * (np.median(v) - default) / default:+.0f}%"
                for k, v in sorted(outside.items())
            )
            axis.set_xlabel(f"scale bank\n({note})", fontsize=6)
        figure.tight_layout(pad=0.3)
        path = save(figure, output)
        plt.close(figure)
    return path


def plot_timing(timing_json: Path, output: Path) -> Path:
    """Per-stage time budget (donut) beside the cost scaling in K, T, P, Q.

    A pie asserts an additive decomposition of a total, which holds for code
    stages but not for parameters -- parameter effects are elasticities and do
    not sum to 100%. Hence: stages in (a), per-parameter scaling laws in (b).
    """
    report = json.loads(Path(timing_json).read_text(encoding="utf-8"))
    stages = report["stages"]
    labels = {
        "rollouts_KT": r"rollouts $O(KT)$",
        "memory_QP2": r"occupancy KDE $O(QP^2)$",
        "attraction_T2": r"attraction $O(T^2)$",
        "sample_epsilon": "sampling",
    }
    values = [stages["stages"][key]["ms_median"] for key in labels]
    residual = stages["residual_ms"]
    if residual > 0:
        labels_list = list(labels.values()) + ["residual / fusion"]
        values = values + [residual]
    else:
        labels_list = list(labels.values())

    tints = ["#2E4A6B", "#4E79A7", "#7FA6CC", "#B3C8DF", "#D9E2EC"][: len(values)]

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.7),
                                    gridspec_kw={"width_ratios": [1.0, 1.25]})
        axis = axes[0]
        wedges, *_ = axis.pie(
            values, labels=None, colors=tints, startangle=90,
            wedgeprops={"width": 0.42, "edgecolor": "#FFFFFF", "linewidth": 0.6},
        )
        shape = stages["shape"]
        axis.text(0, 0, f"{stages['total_ms']:.2f}\nms/step", ha="center", va="center",
                  fontsize=7.5, fontweight="bold")
        axis.legend(
            wedges,
            [f"{name}  {value:.2f} ms ({100 * value / stages['total_ms']:.0f}%)"
             for name, value in zip(labels_list, values)],
            loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=5.2,
            frameon=False, handlelength=1.0,
        )
        axis.set_title(
            f"(a) per-step budget\n$K$={shape['K']}, $T$={shape['T']}, "
            f"$P$={shape['P']}, $Q$={shape['Q']}",
            fontsize=7,
        )

        axis = axes[1]
        scaling = report.get("scaling", {})
        # Each parameter is plotted against the stage it actually drives, not the
        # total: the total carries a fixed ~2.4 ms overhead, so no pure power law
        # can fit it and a slope reference against it would be misleading.
        drives = {
            "K": ("rollouts_ms", 1.0, r"rollouts $\propto K$"),
            "T": ("rollouts_ms", 1.0, r"rollouts $\propto T$"),
            "P": ("memory_ms", 2.0, r"KDE $\propto P^2$"),
            "Q": ("memory_ms", 1.0, r"KDE $\propto Q$"),
        }
        for (name, rows_list), color in zip(
            sorted(scaling.items()), ["#2E4A6B", "#4E79A7", ACCENT, "#59A14F"]
        ):
            key, slope, label = drives[name]
            levels = np.asarray([row["level"] for row in rows_list], dtype=np.float64)
            times = np.asarray([row[key] for row in rows_list], dtype=np.float64)
            relative = levels / levels[0]
            axis.plot(relative, times / times[0], "o-", color=color,
                      markersize=2.4, linewidth=0.9, label=label)
            axis.plot(relative, relative ** slope, color=color, linewidth=0.5,
                      linestyle=":", alpha=0.7)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("parameter, relative to its smallest level")
        axis.set_ylabel("stage time, relative")
        axis.set_title("(b) cost scaling (dotted: predicted slope)", fontsize=7)
        axis.legend(fontsize=5.5, loc="upper left", handlelength=1.2, labelspacing=0.25)
        figure.tight_layout(pad=0.35, w_pad=1.0)
        path = save(figure, output)
        plt.close(figure)
    return path


def write_best_table(
    campaign_dir: Path,
    stage: str,
    output: Path,
    metrics: tuple[str, ...] = ("occupancy_mse", "fourier_ergodic", "steps_to_threshold", "ms_per_step"),
    palette: str = "warm",
    threshold_factor: float = 1.5,
) -> Path:
    """LaTeX table with the top three ranks colour-coded per metric column.

    ``warm`` is red/orange/yellow for 1st/2nd/3rd as specified. ``mono`` uses
    three tints of one hue instead: red-for-best inverts the usual convention and
    the warm triple is hard to rank under deuteranopia, so the camera-ready has
    an out.
    """
    palettes = {
        "warm": ("bestred", "bestorange", "bestyellow"),
        "mono": ("bestdark", "bestmid", "bestlight"),
    }
    definitions = {
        "warm": [
            r"\definecolor{bestred}{HTML}{F8B4B4}",
            r"\definecolor{bestorange}{HTML}{FBD5A5}",
            r"\definecolor{bestyellow}{HTML}{FDF0A9}",
        ],
        "mono": [
            r"\definecolor{bestdark}{HTML}{9CB6D4}",
            r"\definecolor{bestmid}{HTML}{C3D3E5}",
            r"\definecolor{bestlight}{HTML}{E4EBF3}",
        ],
    }
    if palette not in palettes:
        raise ValueError(f"palette must be one of {sorted(palettes)}")

    summary = summarize_stage(campaign_dir, stage, threshold_factor=threshold_factor)
    columns = [m for m in metrics if any(f"{m}_median" in row for row in summary)]
    # Key on (arm, axes): in an OFAT stage every level shares an arm name, so
    # keying on the arm alone would silently drop all but one level.
    table = {
        _variant_label(row): {m: float(row.get(f"{m}_median", np.nan)) for m in columns}
        for row in summary
    }
    arms = sorted(table, key=lambda a: table[a].get("occupancy_mse", np.inf))

    # Lower is better for every metric here, transient included.
    ranks: dict[str, dict[str, int]] = {}
    for metric in columns:
        values = [(arm, table[arm][metric]) for arm in arms if np.isfinite(table[arm][metric])]
        for position, (arm, _) in enumerate(sorted(values, key=lambda p: p[1])):
            ranks.setdefault(arm, {})[metric] = position

    def cell(arm: str, metric: str) -> str:
        value = table[arm][metric]
        if not np.isfinite(value):
            return "--"
        text = f"{value:.3g}" if metric != "steps_to_threshold" else f"{value:.0f}"
        position = ranks.get(arm, {}).get(metric, 99)
        if position < 3:
            return rf"\cellcolor{{{palettes[palette][position]}}}{text}"
        return text

    lines = [
        "% Generated by ergodic_control_mppi.plotting.ablation -- do not edit by hand.",
        r"% Requires \usepackage{xcolor} and \usepackage{colortbl}.",
        *definitions[palette],
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\caption{{Ablation summary ({stage}); median over seeds, lower is better. "
        r"Best \colorbox{" + palettes[palette][0] + r"}{first}, "
        r"\colorbox{" + palettes[palette][1] + r"}{second}, "
        r"\colorbox{" + palettes[palette][2] + r"}{third}.}",
        rf"\label{{tab:ablation_{stage}}}",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        "Variant & " + " & ".join(
            METRIC_LABELS.get(m, m).replace("$", "$") for m in columns
        ) + r" \\",
        r"\midrule",
    ]
    for arm in arms:
        lines.append(arm.replace("_", r"\_") if "$" not in arm and "=" not in arm else arm)
        lines[-1] += " & " + " & ".join(cell(arm, m) for m in columns) + r" \\"
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


STAGE_FIGURES = {
    "screening": ("tornado", "violins"),
    "interactions": ("interactions",),
    "core": ("arms", "convergence", "generalization", "table"),
    "structure": ("structure", "arms", "table"),
    "components": ("arms", "convergence", "table"),
    "generalization": ("generalization",),
    "smoke": ("tornado", "violins", "arms", "convergence", "table"),
    "smoke_grid": ("interactions",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=Path("results/campaign"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/campaign/figures"))
    parser.add_argument("--stage", action="append", dest="stages",
                        help="stage name, or 'all'. Repeatable.")
    parser.add_argument("--timing-json", type=Path,
                        default=Path("results/campaign/timing/timing.json"))
    parser.add_argument("--threshold-factor", type=float, default=1.5)
    parser.add_argument("--palette", default="warm", choices=("warm", "mono"))
    args = parser.parse_args()

    stages = args.stages or ["all"]
    if "all" in stages:
        stages = [s for s in STAGE_FIGURES if (args.campaign_dir / f"{s}.csv").exists()]

    written: list[Path] = []
    for stage in stages:
        for kind in STAGE_FIGURES.get(stage, ()):
            target = args.output_dir / f"{stage}_{kind}"
            try:
                if kind == "tornado":
                    written.append(plot_tornado(args.campaign_dir, stage, target.with_suffix(".pdf")))
                elif kind == "violins":
                    written.append(plot_violins(args.campaign_dir, stage, target.with_suffix(".pdf")))
                elif kind == "interactions":
                    written.append(plot_interactions(args.campaign_dir, stage, target.with_suffix(".pdf")))
                elif kind == "arms":
                    written.append(plot_arms(args.campaign_dir, stage, target.with_suffix(".pdf"),
                                             args.threshold_factor))
                elif kind == "convergence":
                    written.append(plot_convergence(args.campaign_dir, stage, target.with_suffix(".pdf"),
                                                    args.threshold_factor))
                elif kind == "generalization":
                    written.append(plot_generalization(args.campaign_dir, stage, target.with_suffix(".pdf")))
                elif kind == "structure":
                    written.append(plot_structure(args.campaign_dir, stage, target.with_suffix(".pdf")))
                elif kind == "table":
                    written.append(write_best_table(
                        args.campaign_dir, stage, target.with_suffix(".tex"),
                        palette=args.palette, threshold_factor=args.threshold_factor))
            except (ValueError, FileNotFoundError, KeyError) as error:
                print(f"skipped {stage}/{kind}: {error}")

    if args.timing_json.exists():
        written.append(plot_timing(args.timing_json, args.output_dir / "timing.pdf"))
    else:
        print(f"skipped timing: no {args.timing_json}")

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
