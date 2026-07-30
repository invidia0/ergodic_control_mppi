"""Publication plots for literature-comparison CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from ergodic_control_mppi.plotting.style import (
    SCENARIO_COLORS,
    SCENARIO_LINESTYLES,
    paper_style,
)

METHOD_ORDER = ["mppi", "smc", "hedac", "traj_opt", "dec"]
METHOD_LABELS = {
    "mppi": "Proposed",
    "smc": "SMC [18]",
    "hedac": "HEDAC [10]",
    "traj_opt": "Traj. Opt. [19]",
    "dec": "DEC [1]",
}
SCENARIO_ORDER = ["unimodal", "bimodal", "trimodal", "four_mode"]
SCENARIO_LABELS = {
    "unimodal": "Unimodal",
    "bimodal": "Bimodal",
    "trimodal": "Trimodal",
    "four_mode": "Four-mode",
}
def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))



def _ordered(values: set[str], order: list[str]) -> list[str]:
    out = [v for v in order if v in values]
    out.extend(sorted(v for v in values if v not in set(out)))
    return out


def _slugify(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _build_convergence_stats(
    rows: list[dict[str, str]],
    scenarios: list[str],
    methods: list[str],
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    for row in rows:
        key = (str(row["scenario_name"]), str(row["method_name"]), int(row["seed"]))
        grouped.setdefault(key, []).append((int(row["step"]), float(row["team_ergodic_error"])))

    stats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scenario_name in scenarios:
        for method in methods:
            seed_series = [
                sorted(v, key=lambda p: p[0])
                for (s_name, m_name, _seed), v in grouped.items()
                if s_name == scenario_name and m_name == method
            ]
            if len(seed_series) == 0:
                continue
            min_len = min(len(v) for v in seed_series)
            steps = np.asarray([p[0] for p in seed_series[0][:min_len]], dtype=np.int64)
            vals = np.stack(
                [np.asarray([p[1] for p in series[:min_len]], dtype=np.float64) for series in seed_series],
                axis=0,
            )
            mean = np.mean(vals, axis=0)
            std = np.std(vals, axis=0)
            stats[(scenario_name, method)] = (steps, mean, std)
    return stats



def plot_literature_bars(
    summary_csv_path: str = "results/dars2026/literature/literature_comparison_summary.csv",
    output_path: str = "results/dars2026/literature/plots/literature_comparison_bars_by_method.pdf",
) -> str:
    rows = _load_rows(summary_csv_path)
    if len(rows) == 0:
        raise ValueError("no summary rows found for plotting")

    scenarios = _ordered({str(r["scenario_name"]) for r in rows}, SCENARIO_ORDER)
    methods = _ordered({str(r["method_name"]) for r in rows}, METHOD_ORDER)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    value_map: dict[tuple[str, str], tuple[float, float]] = {}
    for row in rows:
        value_map[(str(row["method_name"]), str(row["scenario_name"]))] = (
            float(row["team_ergodic_error_mean"]),
            float(row["team_ergodic_error_std"]),
        )

    with plt.rc_context(rc=paper_style("poster")):
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.4), constrained_layout=True)
        x = np.arange(len(methods), dtype=np.float64)
        width = 0.8 / max(len(scenarios), 1)
        half_span = (len(scenarios) - 1) * width / 2.0

        for s_idx, scenario_name in enumerate(scenarios):
            means = []
            stds = []
            for method in methods:
                val = value_map.get((method, scenario_name))
                if val is None:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(float(val[0]))
                    stds.append(float(val[1]))
            xpos = x - half_span + s_idx * width
            ax.bar(
                xpos,
                np.asarray(means, dtype=np.float64),
                yerr=np.asarray(stds, dtype=np.float64),
                width=width * 0.95,
                capsize=4.0,
                color=SCENARIO_COLORS.get(scenario_name, plt.get_cmap("tab10")(s_idx)),
                alpha=0.92,
                edgecolor="#23272F",
                linewidth=1.0,
                error_kw={"ecolor": "#111111", "elinewidth": 1.2, "capthick": 1.2},
                label=SCENARIO_LABELS.get(scenario_name, scenario_name),
            )
        # ax.set_title("Literature Comparison by Method and Scenario")
        ax.set_ylabel(r"Team Ergodic Error ($\mathcal{E}_{\mathrm{team}}$)")
        ax.set_xlabel("Method")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=0, ha="center")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", pad=6)
        ax.tick_params(axis="y", pad=4)
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            framealpha=0.92,
            title="Scenario",
            borderpad=0.4,
            handlelength=1.3,
        )
        leg.get_title().set_fontsize(16)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return str(out)



def plot_literature_convergence(
    convergence_csv_path: str = "results/dars2026/literature/literature_comparison_convergence.csv",
    output_dir: str = "results/dars2026/literature/plots",
    use_log_y: bool = True,
) -> list[str]:
    rows = _load_rows(convergence_csv_path)
    if len(rows) == 0:
        raise ValueError("no convergence rows found for plotting")

    scenarios = _ordered({str(r["scenario_name"]) for r in rows}, SCENARIO_ORDER)
    methods = _ordered({str(r["method_name"]) for r in rows}, METHOD_ORDER)
    stats = _build_convergence_stats(rows, scenarios, methods)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    with plt.rc_context(rc=paper_style("poster")):
        cmap = plt.get_cmap("tab10")
        for scenario_name in scenarios:
            fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.4), constrained_layout=True)
            for m_idx, method in enumerate(methods):
                sm_stats = stats.get((scenario_name, method))
                if sm_stats is None:
                    continue
                steps, mean, std = sm_stats

                color = cmap(m_idx % cmap.N)
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linewidth=2.2,
                    label=METHOD_LABELS.get(method, method),
                )
                ax.fill_between(steps, np.maximum(mean - std, 0.0), mean + std, color=color, alpha=0.2)

            ax.set_title(SCENARIO_LABELS.get(scenario_name, scenario_name))
            ax.set_xlabel("Step")
            ax.set_ylabel(r"Cumulative Team Ergodic Error ($\mathcal{E}_{\mathrm{team}}$)")
            if use_log_y:
                ax.set_yscale("log")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="x", pad=6)
            ax.tick_params(axis="y", pad=4)
            leg = ax.legend(
                loc="upper right",
                framealpha=0.92,
                borderpad=0.4,
                handlelength=1.8,
            )
            leg.get_title().set_fontsize(16)
            for line in leg.get_lines():
                line.set_linewidth(2.2)
            out = out_root / f"literature_comparison_convergence_{_slugify(scenario_name)}.pdf"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(out))

    return saved_paths


def plot_literature_convergence_single_axis_5(
    convergence_csv_path: str = "results/dars2026/literature/literature_comparison_convergence.csv",
    output_path: str = (
        "results/dars2026/literature/plots/literature_comparison_convergence_single_axis_5_methods.pdf"
    ),
    use_log_y: bool = True,
) -> str:
    rows = _load_rows(convergence_csv_path)
    if len(rows) == 0:
        raise ValueError("no convergence rows found for plotting")

    scenarios = _ordered({str(r["scenario_name"]) for r in rows}, SCENARIO_ORDER)
    methods = _ordered({str(r["method_name"]) for r in rows}, METHOD_ORDER)
    stats = _build_convergence_stats(rows, scenarios, methods)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(rc=paper_style("poster")):
        fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.4), constrained_layout=True)
        cmap = plt.get_cmap("tab10")
        for m_idx, method in enumerate(methods):
            scenario_series: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
                stats[(scenario_name, method)]
                for scenario_name in scenarios
                if (scenario_name, method) in stats
            ]
            if len(scenario_series) == 0:
                continue

            min_len = min(len(t[0]) for t in scenario_series)
            steps = scenario_series[0][0][:min_len]
            means_stack = np.stack([t[1][:min_len] for t in scenario_series], axis=0)
            agg_mean = np.mean(means_stack, axis=0)
            agg_std = np.std(means_stack, axis=0)

            color = cmap(m_idx % cmap.N)
            ax.plot(
                steps,
                agg_mean,
                color=color,
                linewidth=2.4,
                label=METHOD_LABELS.get(method, method),
            )
            ax.fill_between(
                steps,
                np.maximum(agg_mean - agg_std, 0.0),
                agg_mean + agg_std,
                color=color,
                alpha=0.18,
            )

        ax.set_title("Convergence (Single Axis, 5 Methods)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Team Ergodic Error")
        if use_log_y:
            ax.set_yscale("log")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", pad=6)
        ax.tick_params(axis="y", pad=4)
        ax.legend(loc="upper right", framealpha=0.92, borderpad=0.4, handlelength=1.8)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return str(out)


def plot_literature_convergence_single_axis_20(
    convergence_csv_path: str = "results/dars2026/literature/literature_comparison_convergence.csv",
    output_path: str = (
        "results/dars2026/literature/plots/literature_comparison_convergence_single_axis_20_lines.pdf"
    ),
    use_log_y: bool = True,
) -> str:
    rows = _load_rows(convergence_csv_path)
    if len(rows) == 0:
        raise ValueError("no convergence rows found for plotting")

    scenarios = _ordered({str(r["scenario_name"]) for r in rows}, SCENARIO_ORDER)
    methods = _ordered({str(r["method_name"]) for r in rows}, METHOD_ORDER)
    stats = _build_convergence_stats(rows, scenarios, methods)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(rc=paper_style("poster")):
        fig, ax = plt.subplots(1, 1, figsize=(9.8, 6.0), constrained_layout=True)
        cmap = plt.get_cmap("tab10")
        method_colors: dict[str, tuple[float, float, float, float]] = {
            m: cmap(i % cmap.N) for i, m in enumerate(methods)
        }

        for scenario_name in scenarios:
            linestyle = SCENARIO_LINESTYLES.get(scenario_name, "-")
            for method in methods:
                sm_stats = stats.get((scenario_name, method))
                if sm_stats is None:
                    continue
                steps, mean, std = sm_stats
                color = method_colors[method]
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.9,
                )
                ax.fill_between(
                    steps,
                    np.maximum(mean - std, 0.0),
                    mean + std,
                    color=color,
                    alpha=0.08,
                )

        ax.set_title("Convergence (Single Axis, 20 Method-Scenario Lines)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Team Ergodic Error")
        if use_log_y:
            ax.set_yscale("log")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="x", pad=6)
        ax.tick_params(axis="y", pad=4)

        method_handles = [
            Line2D(
                [0],
                [0],
                color=method_colors[m],
                lw=2.6,
                linestyle="-",
                label=METHOD_LABELS.get(m, m),
            )
            for m in methods
        ]
        scenario_handles = [
            Line2D(
                [0],
                [0],
                color="#202020",
                lw=2.2,
                linestyle=SCENARIO_LINESTYLES.get(s, "-"),
                label=SCENARIO_LABELS.get(s, s),
            )
            for s in scenarios
        ]

        leg_methods = ax.legend(
            handles=method_handles,
            title="Method (color)",
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=0.92,
            borderpad=0.35,
            handlelength=2.0,
        )
        leg_scenarios = ax.legend(
            handles=scenario_handles,
            title="Scenario (linestyle)",
            loc="upper right",
            bbox_to_anchor=(1.0, 0.63),
            framealpha=0.92,
            borderpad=0.35,
            handlelength=2.0,
        )
        ax.add_artist(leg_methods)
        leg_methods.get_title().set_fontsize(13)
        leg_scenarios.get_title().set_fontsize(13)

        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return str(out)



def plot_literature_comparison(
    summary_csv_path: str = "results/dars2026/literature/literature_comparison_summary.csv",
    convergence_csv_path: str = "results/dars2026/literature/literature_comparison_convergence.csv",
    output_dir: str = "results/dars2026/literature/plots",
    use_log_y: bool = True,
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = plot_literature_bars(
        summary_csv_path=summary_csv_path,
        output_path=str(out_dir / "literature_comparison_bars_by_method.pdf"),
    )
    conv_paths = plot_literature_convergence(
        convergence_csv_path=convergence_csv_path,
        output_dir=str(out_dir),
        use_log_y=use_log_y,
    )
    conv_single_5 = plot_literature_convergence_single_axis_5(
        convergence_csv_path=convergence_csv_path,
        output_path=str(out_dir / "literature_comparison_convergence_single_axis_5_methods.pdf"),
        use_log_y=use_log_y,
    )
    conv_single_20 = plot_literature_convergence_single_axis_20(
        convergence_csv_path=convergence_csv_path,
        output_path=str(out_dir / "literature_comparison_convergence_single_axis_20_lines.pdf"),
        use_log_y=use_log_y,
    )
    return [bars, *conv_paths, conv_single_5, conv_single_20]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot literature comparison bars and convergence curves.")
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/dars2026/literature/literature_comparison_summary.csv",
        help="Path to grouped summary CSV.",
    )
    parser.add_argument(
        "--convergence-csv",
        type=str,
        default="results/dars2026/literature/literature_comparison_convergence.csv",
        help="Path to per-step convergence CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dars2026/literature/plots",
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear y-axis for convergence plots (default is log).",
    )
    args = parser.parse_args()

    plot_literature_comparison(
        summary_csv_path=args.summary_csv,
        convergence_csv_path=args.convergence_csv,
        output_dir=args.output_dir,
        use_log_y=not args.linear_y,
    )
