"""Publication plots for ablation experiment CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

ABLATION_ORDER = ["Full", "No Curl", "No Cross", "Weak Stein", "Reduced Horizon"]
METRIC_LABELS = {
    "team_ergodic_error": r"Team Ergodic Error ($\mathcal{E}_{\mathrm{team}}$)",
    "redundancy_metric": r"Redundancy ($\mathcal{R}_{\mathrm{pair}}$)",
    "safety_metric": r"Safety ($\mathcal{S}$)",
    "R_pair": r"Mean Pairwise Distance ($\overline{R}_{\mathrm{pair}}$)",
    "D_min_pair": r"Minimum Pairwise Distance ($D_{\min,\mathrm{pair}}$)",
    "pairwise_overlap": r"Pairwise Overlap ($\mathcal{O}_{\mathrm{pair}}$)",
}
GROUPED_METRICS = ["team_ergodic_error", "redundancy_metric", "R_pair"]
GROUPED_LEGEND_LABELS = {
    "team_ergodic_error": r"$\mathcal{E}_{\mathrm{team}}$",
    "redundancy_metric": r"$\mathcal{R}_{\mathrm{pair}}$",
    "R_pair": r"$\overline{R}_{\mathrm{pair}}$",
}
ABLATION_DISPLAY_LABELS = {
    "Full": "Full",
    "No Curl": "No Curl",
    "No Cross": "No Cross",
    "Weak Stein": "Weak Stein",
    "Reduced Horizon": "Reduced\nHorizon",
}


def _paper_plot_style() -> dict[str, object]:
    """Return publication-focused style settings for ablation bar plots."""
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#DCE2EC",
        "axes.edgecolor": "#98A4BA",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8ECF4",
        "grid.alpha": 0.9,
        "grid.linewidth": 0.75,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
    }


def _display_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    return metric.replace("_", " ").title()


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_by_ablation(
    rows: list[dict[str, str]],
    metrics: list[str],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    present = {str(r["ablation_name"]) for r in rows}
    ablations = [name for name in ABLATION_ORDER if name in present]
    ablations.extend(sorted(name for name in present if name not in set(ablations)))
    if len(ablations) == 0:
        raise ValueError("ablation_name column is required in ablation CSV")
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for metric in metrics:
        mean_vals = []
        std_vals = []
        for name in ablations:
            vals = np.asarray(
                [float(r[metric]) for r in rows if str(r["ablation_name"]) == name],
                dtype=np.float64,
            )
            mean_vals.append(float(np.mean(vals)))
            std_vals.append(float(np.std(vals)))
        means[metric] = np.asarray(mean_vals, dtype=np.float64)
        stds[metric] = np.asarray(std_vals, dtype=np.float64)
    return ablations, means, stds


def _plot_metric_bar(
    ax: plt.Axes,
    metric: str,
    metric_idx: int,
    ablations: list[str],
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> None:
    x = np.arange(len(ablations))
    cmap = plt.get_cmap("tab10")
    color = cmap(metric_idx % cmap.N)
    ax.bar(
        x,
        means[metric],
        yerr=stds[metric],
        capsize=5,
        color=color,
        alpha=0.9,
        edgecolor="#23272F",
        linewidth=0.9,
        error_kw={"ecolor": "#111111", "elinewidth": 1.25, "capthick": 1.25},
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(yfmt)
    ax.set_xticks(x)
    ax.set_xticklabels(ablations, rotation=16, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1)
    ax.set_title(_display_label(metric))
    ax.set_ylabel("Value")


def plot_ablation_grouped_three_metrics(
    csv_path: str = "results/dars2026/ablations/open_multimodal_ablations.csv",
    output_path: str | None = "results/dars2026/ablations/open_multimodal_ablations_grouped_three_metrics.pdf",
    show: bool = False,
) -> None:
    rows = _load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")
    for metric in GROUPED_METRICS:
        if metric not in rows[0]:
            raise ValueError(f"metric '{metric}' not found in CSV columns")

    ablations, means, stds = _aggregate_by_ablation(rows, GROUPED_METRICS)
    eps = 1e-12
    norm_means: dict[str, np.ndarray] = {}
    norm_stds: dict[str, np.ndarray] = {}
    for metric in GROUPED_METRICS:
        scale = max(float(np.max(means[metric])), eps)
        norm_means[metric] = means[metric] / scale
        norm_stds[metric] = stds[metric] / scale

    with plt.rc_context(rc=_paper_plot_style()):
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.3), constrained_layout=True)
        x = np.arange(len(ablations), dtype=np.float64)
        width = 0.22
        offsets = np.array([-width, 0.0, width], dtype=np.float64)
        cmap = plt.get_cmap("tab10")

        for i, metric in enumerate(GROUPED_METRICS):
            ax.bar(
                x + offsets[i],
                norm_means[metric],
                width=width,
                yerr=norm_stds[metric],
                capsize=2.5,
                color=cmap(i % cmap.N),
                alpha=0.92,
                edgecolor="#23272F",
                linewidth=0.7,
                error_kw={"ecolor": "#111111", "elinewidth": 0.9, "capthick": 0.9},
                label=GROUPED_LEGEND_LABELS.get(metric, _display_label(metric)),
            )

        ax.set_ylabel("Normalized metric value")
        ax.set_xticks(x)
        tick_labels = [ABLATION_DISPLAY_LABELS.get(a, a) for a in ablations]
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=1.0)
        ymax_with_err = max(
            float(np.max(norm_means[m] + norm_stds[m])) for m in GROUPED_METRICS
        )
        y_top = max(1.05, 1.08 * ymax_with_err)
        ax.set_ylim(0.0, y_top)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.legend(
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 1.01),
            framealpha=0.85,
            borderpad=0.25,
            columnspacing=0.8,
            handlelength=1.1,
        )

        if output_path is not None and len(output_path) > 0:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def plot_ablation_bars(
    csv_path: str = "results/dars2026/ablations/open_multimodal_ablations.csv",
    metrics: list[str] | None = None,
    output_path: str | None = "results/dars2026/ablations/open_multimodal_ablations_bars.pdf",
    show: bool = False,
) -> None:
    metric_list = metrics if metrics is not None else [
        "team_ergodic_error",
        "redundancy_metric",
        "R_pair",
        "D_min_pair",
    ]
    rows = _load_rows(csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")
    for metric in metric_list:
        if metric not in rows[0]:
            raise ValueError(f"metric '{metric}' not found in CSV columns")

    ablations, means, stds = _aggregate_by_ablation(rows, metric_list)
    with plt.rc_context(rc=_paper_plot_style()):
        n_metrics = len(metric_list)
        n_cols = 2 if n_metrics <= 4 else 3
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.3 * n_cols, 4.0 * n_rows),
            constrained_layout=True,
        )
        axes_flat = np.atleast_1d(axes).ravel()

        for i, metric in enumerate(metric_list):
            _plot_metric_bar(axes_flat[i], metric, i, ablations, means, stds)

        for j in range(n_metrics, len(axes_flat)):
            axes_flat[j].axis("off")

        if output_path is not None and len(output_path) > 0:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300, bbox_inches="tight")
            for i, metric in enumerate(metric_list):
                single_fig, single_ax = plt.subplots(
                    1,
                    1,
                    figsize=(5.3, 4.0),
                    constrained_layout=True,
                )
                _plot_metric_bar(single_ax, metric, i, ablations, means, stds)
                metric_suffix = metric.replace(" ", "_")
                single_out = out.with_name(f"{metric_suffix}{out.suffix}")
                single_fig.savefig(single_out, dpi=300, bbox_inches="tight")
                plt.close(single_fig)
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bar charts for ablation metrics.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="results/dars2026/ablations/open_multimodal_ablations.csv",
        help="Path to per-seed ablation CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/dars2026/ablations/open_multimodal_ablations_bars.pdf",
        help="Optional output image path. Set empty string to disable saving.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive figure window.",
    )
    parser.add_argument(
        "--grouped-three-metrics",
        action="store_true",
        help="Plot normalized grouped bars for team_ergodic_error, redundancy_metric, and R_pair.",
    )
    parser.add_argument(
        "--grouped-output-path",
        type=str,
        default="results/dars2026/ablations/open_multimodal_ablations_grouped_three_metrics.pdf",
        help="Output path for grouped-three-metrics plot.",
    )
    args = parser.parse_args()
    if args.grouped_three_metrics:
        plot_ablation_grouped_three_metrics(
            csv_path=args.csv_path,
            output_path=args.grouped_output_path,
            show=args.show,
        )
    else:
        plot_ablation_bars(
            csv_path=args.csv_path,
            output_path=args.output_path,
            show=args.show,
        )
