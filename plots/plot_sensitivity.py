from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

SWEEP_ORDER = ["theta", "alpha_cross", "horizon", "weight_stein"]
SWEEP_XLABELS = {
    "theta": r"$\theta$ (deg)",
    "alpha_cross": r"$\alpha_{\mathrm{cross}}$",
    "horizon": r"Horizon $T$",
    "weight_stein": r"$w_{\mathrm{stein}}$",
}
SWEEP_TITLE = {
    "theta": r"Sensitivity: $\theta$",
    "alpha_cross": r"Sensitivity: $\alpha_{\mathrm{cross}}$",
    "horizon": r"Sensitivity: $T$",
    "weight_stein": r"Sensitivity: $w_{\mathrm{stein}}$",
}
SWEEP_LEGEND = {
    "theta": r"$\theta$",
    "alpha_cross": r"$\alpha_{\mathrm{cross}}$",
    "horizon": r"$T$",
    "weight_stein": r"$w_{\mathrm{stein}}$",
}


def _paper_plot_style() -> dict[str, object]:
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
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
    }


def _load_summary_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_series(
    rows: list[dict[str, str]],
    sweep_name: str,
    max_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [r for r in rows if str(r["sweep_name"]) == sweep_name]
    if len(series) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )
    series_sorted = sorted(series, key=lambda r: float(r["sweep_value"]))
    if max_k is not None:
        if max_k < 1:
            raise ValueError("max_k must be >= 1 when provided")
        series_sorted = series_sorted[:max_k]
    x = np.asarray([float(r["sweep_value"]) for r in series_sorted], dtype=np.float64)
    y_mean = np.asarray(
        [float(r["team_ergodic_error_mean"]) for r in series_sorted], dtype=np.float64
    )
    y_std = np.asarray(
        [float(r["team_ergodic_error_std"]) for r in series_sorted], dtype=np.float64
    )
    return x, y_mean, y_std


def _plot_single_sweep(
    ax: plt.Axes,
    sweep_name: str,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    eps = 1e-12
    scale = max(float(np.max(y_mean)), eps)
    y_norm = y_mean / scale
    y_std_norm = y_std / scale
    lower = np.clip(y_norm - y_std_norm, 0.0, None)
    upper = y_norm + y_std_norm

    ax.plot(x, y_norm, marker="o", markersize=3.0, linewidth=1.6, color="tab:blue")
    ax.fill_between(x, lower, upper, color="tab:blue", alpha=0.22)
    if sweep_name == "weight_stein":
        ax.set_xscale("log")

    ax.set_title(SWEEP_TITLE.get(sweep_name, sweep_name))
    ax.set_xlabel(SWEEP_XLABELS.get(sweep_name, sweep_name))
    ax.set_ylabel(r"Normalized $\mathcal{E}_{\mathrm{team}}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x], rotation=45, ha="right", rotation_mode="anchor")
    ymax = max(1.05, 1.08 * float(np.max(upper)))
    ax.set_ylim(0.0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def _normalize_series(
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-12
    scale = max(float(np.max(y_mean)), eps)
    y_norm = y_mean / scale
    y_std_norm = y_std / scale
    lower = np.clip(y_norm - y_std_norm, 0.0, None)
    upper = y_norm + y_std_norm
    return y_norm, lower, upper


def plot_sensitivity_overlay(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    max_k: int | None = None,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_rc = {
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    }
    with plt.rc_context(rc=local_rc):
        fig, ax = plt.subplots(1, 1, figsize=(1.70, 1.40), constrained_layout=True)
        cmap = plt.get_cmap("tab10")
        upper_global_max = 0.0

        for i, sweep_name in enumerate(SWEEP_ORDER):
            x, y_mean, y_std = _build_series(rows, sweep_name, max_k=max_k)
            if x.size == 0:
                continue
            x_step = np.arange(1, x.size + 1, dtype=np.float64)
            y_norm, lower, upper = _normalize_series(y_mean, y_std)
            upper_global_max = max(upper_global_max, float(np.max(upper)))
            color = cmap(i % cmap.N)
            ax.plot(
                x_step,
                y_norm,
                marker="o",
                markersize=1.9,
                linewidth=0.95,
                color=color,
                label=SWEEP_LEGEND.get(sweep_name, sweep_name),
            )
            ax.fill_between(x_step, lower, upper, color=color, alpha=0.12)

        ax.set_xlabel(r"Sweep step index $k$")
        ax.set_ylabel(r"Normalized $\mathcal{E}_{\mathrm{team}}$")
        x_max = max_k if max_k is not None else max((len(_build_series(rows, s)[0]) for s in SWEEP_ORDER), default=1)
        x_max = max(int(x_max), 1)
        ax.set_xlim(1.0, float(x_max))
        ax.set_xticks(np.arange(1, x_max + 1, dtype=np.int32))
        y_top = max(1.03, 1.06 * upper_global_max)
        ax.set_ylim(0.0, y_top)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(
            loc="upper right",
            framealpha=0.82,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.0,
            handletextpad=0.3,
        )

        out_path = out_dir / "sensitivity_overlay_single_graph.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return str(out_path)


def plot_sensitivity_scores(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    max_k: int | None = None,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_labels: list[str] = []
    scores: list[float] = []
    score_errs: list[float] = []

    for sweep_name in SWEEP_ORDER:
        x, y_mean, y_std = _build_series(rows, sweep_name, max_k=max_k)
        if x.size == 0:
            continue
        y_norm, _, _ = _normalize_series(y_mean, y_std)
        eps = 1e-12
        scale = max(float(np.max(y_mean)), eps)
        y_std_norm = y_std / scale
        score_labels.append(SWEEP_LEGEND.get(sweep_name, sweep_name))
        scores.append(float(np.max(y_norm) - np.min(y_norm)))
        # Contextual uncertainty indicator: pooled normalized std
        score_errs.append(float(np.sqrt(np.mean(np.square(y_std_norm)))))

    local_rc = {
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
    with plt.rc_context(rc=local_rc):
        fig, ax = plt.subplots(1, 1, figsize=(1.70, 1.40), constrained_layout=True)
        x = np.arange(len(scores), dtype=np.float64)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(scores))]
        ax.bar(
            x,
            np.asarray(scores, dtype=np.float64),
            yerr=np.asarray(score_errs, dtype=np.float64),
            capsize=2.2,
            color=colors,
            alpha=0.9,
            edgecolor="#23272F",
            linewidth=0.7,
            error_kw={"ecolor": "#111111", "elinewidth": 0.75, "capthick": 0.75},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(score_labels)
        ax.set_ylabel(r"Sensitivity score ($\max-\min$)")
        min_top = 0.05
        headroom = 1.15
        ymax = max(min_top, headroom * float(np.max(np.asarray(scores) + np.asarray(score_errs))))
        ax.set_ylim(0.0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        out_path = out_dir / "sensitivity_scores_single_graph.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return str(out_path)


def plot_sensitivity_sweeps(
    summary_csv_path: str = "results/dars2026/sensitivity/open_multimodal_sensitivity_summary.csv",
    output_dir: str = "results/dars2026/sensitivity/plots",
    save_compact: bool = True,
    save_standalone: bool = True,
    save_overlay_single: bool = True,
    save_score_single: bool = True,
    max_k: int | None = None,
) -> list[str]:
    rows = _load_summary_rows(summary_csv_path)
    if len(rows) == 0:
        raise ValueError("no rows found for plotting")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    with plt.rc_context(rc=_paper_plot_style()):
        if save_compact:
            fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), constrained_layout=True)
            axes_flat = np.atleast_1d(axes).ravel()
            for i, sweep_name in enumerate(SWEEP_ORDER):
                x, y_mean, y_std = _build_series(rows, sweep_name, max_k=max_k)
                if x.size == 0:
                    axes_flat[i].axis("off")
                    continue
                _plot_single_sweep(axes_flat[i], sweep_name, x, y_mean, y_std)
            compact_out = out_dir / "sensitivity_compact_2x2.pdf"
            fig.savefig(compact_out, dpi=300, bbox_inches="tight")
            saved_paths.append(str(compact_out))
            plt.close(fig)

        if save_standalone:
            for sweep_name in SWEEP_ORDER:
                x, y_mean, y_std = _build_series(rows, sweep_name, max_k=max_k)
                if x.size == 0:
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(3.35, 2.3), constrained_layout=True)
                _plot_single_sweep(ax, sweep_name, x, y_mean, y_std)
                out_path = out_dir / f"sensitivity_{sweep_name}.pdf"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                saved_paths.append(str(out_path))
                plt.close(fig)

        if save_overlay_single:
            saved_paths.append(plot_sensitivity_overlay(rows=rows, output_dir=out_dir, max_k=max_k))

        if save_score_single:
            saved_paths.append(plot_sensitivity_scores(rows=rows, output_dir=out_dir, max_k=max_k))

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensitivity sweep summary results.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/dars2026/sensitivity/open_multimodal_sensitivity_summary.csv",
        help="Path to sensitivity summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dars2026/sensitivity/plots",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable compact 2x2 panel output.",
    )
    parser.add_argument(
        "--no-standalone",
        action="store_true",
        help="Disable standalone per-sweep outputs.",
    )
    parser.add_argument(
        "--with-overlay-single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable single-graph normalized trend overlay output.",
    )
    parser.add_argument(
        "--with-score-single",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable single-graph sensitivity score bar output.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=0,
        help="Optional max sweep step index k to include (0 means use all available points).",
    )
    args = parser.parse_args()
    max_k = args.max_k if args.max_k > 0 else None
    plot_sensitivity_sweeps(
        summary_csv_path=args.summary_csv,
        output_dir=args.output_dir,
        save_compact=not args.no_compact,
        save_standalone=not args.no_standalone,
        save_overlay_single=args.with_overlay_single,
        save_score_single=args.with_score_single,
        max_k=max_k,
    )
