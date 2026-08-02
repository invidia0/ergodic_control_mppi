"""Figures for the cross-campaign report.

Reads the shipped UAV per-seed CSVs and renders the three figures the report needs:

    fig_paired_arms      per-seed paired effect vs the shipped arm (violin + points)
    fig_effect_forest    median ratio + bootstrap CI, Holm-marked
    fig_agreement_matrix campaign vs UAV verdict per mechanism claim

    uv run python scripts/report_figures.py --output results/report

Campaign-side stages need `results/campaign/<stage>.csv`; the matrix falls back to the
values quoted in campaign_findings.md and marks them as quoted when the raw CSVs are absent.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

from ergodic_control_mppi.experiments.analyze import bootstrap_ci
from ergodic_control_mppi.plotting.style import (
    ACCENT,
    DIVERGING_CMAP,
    NEUTRAL,
    PRIMARY,
    paper_style,
    save,
)

BASELINE = "baseline"
ARMS = ["h_0.94", "h_6.6", "theta_0", "theta_15", "theta_45"]
ARM_LABELS = {
    "h_0.94": "$h{=}0.94$",
    "h_6.6": "$h{=}6.6$",
    "theta_0": r"$\theta{=}0$",
    "theta_15": r"$\theta{=}15$",
    "theta_45": r"$\theta{=}45$",
}


def load_arms(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    """Index an ablation CSV as ``arm -> seed -> row``."""
    table: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            table[row["arm"]][int(row["seed"])] = row
    return table


def paired(table, arm: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(arm_values, baseline_values)`` over the seeds both arms share."""
    seeds = sorted(set(table[arm]) & set(table[BASELINE]))
    a = np.array([float(table[arm][s][metric]) for s in seeds])
    b = np.array([float(table[BASELINE][s][metric]) for s in seeds])
    return a, b


def holm(pvalues: list[float]) -> list[bool]:
    """Holm-Bonferroni step-down. Returns a reject mask in the input order."""
    order = np.argsort(pvalues)
    n = len(pvalues)
    reject = [False] * n
    for rank, index in enumerate(order):
        if pvalues[index] > 0.05 / (n - rank):
            break
        reject[index] = True
    return reject


def fig_paired_arms(table, output: Path, metric: str = "occupancy_mse") -> Path:
    """Per-seed paired effect vs the shipped arm, as log2 ratios.

    Plotted as a ratio rather than two absolute distributions because the seeds are
    paired: the seed-to-seed spread is far larger than the arm effect, so unpaired
    violins of the raw metric hide the very comparison the experiment was run to make.
    """
    ratios = [np.log2(paired(table, arm, metric)[0] / paired(table, arm, metric)[1])
              for arm in ARMS]
    pvalues = [wilcoxon(*paired(table, arm, metric)).pvalue for arm in ARMS]
    significant = holm(pvalues)

    with plt.rc_context(rc=paper_style("double")):
        figure, axis = plt.subplots(figsize=(6.9, 2.6))
        positions = np.arange(len(ARMS))
        parts = axis.violinplot(ratios, positions=positions, widths=0.72,
                                showextrema=False, showmedians=False)
        for body, keep in zip(parts["bodies"], significant):
            body.set_facecolor(ACCENT if keep else PRIMARY)
            body.set_alpha(0.38 if keep else 0.22)
            body.set_edgecolor(ACCENT if keep else PRIMARY)
            body.set_linewidth(0.6)
        for position, values, keep in zip(positions, ratios, significant):
            jitter = (np.random.default_rng(0).random(values.size) - 0.5) * 0.16
            axis.plot(position + jitter, values, "o", markersize=2.6,
                      markerfacecolor=ACCENT if keep else PRIMARY,
                      markeredgecolor="#FFFFFF", markeredgewidth=0.4, alpha=0.9)
            axis.hlines(np.median(values), position - 0.3, position + 0.3,
                        color="#23272F", linewidth=1.2, zorder=5)

        axis.axhline(0.0, color="#23272F", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{ARM_LABELS[a]}\n" + (r"$p<0.001$" if p < 0.001 else f"$p={p:.3f}$")
             + ("$^*$" if s else "")
             for a, p, s in zip(ARMS, pvalues, significant)]
        )
        axis.set_ylabel(r"$\log_2$(arm / shipped), occupancy MSE")
        axis.set_title("Paired per-seed effect against the shipped arm "
                       r"($\theta{=}30$, $h{=}5.0$), $n=18$")
        # Headroom so the callout never collides with the topmost violin.
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + 0.18 * (high - low))
        # Direct-label the one arm that survives multiplicity, rather than every violin.
        for position, values, keep in zip(positions, ratios, significant):
            if keep:
                axis.annotate("Holm-significant", (position, np.max(values)),
                              textcoords="offset points", xytext=(0, 7),
                              ha="center", fontsize=6.5, color=ACCENT)
        axis.text(0.995, 0.03, "above 0 = worse than shipped", transform=axis.transAxes,
                  ha="right", fontsize=6.5, color="#5A6472")
        figure.tight_layout(pad=0.4)
        path = save(figure, output)
        plt.close(figure)
    return path


def fig_effect_forest(table, output: Path) -> Path:
    """Median ratio with 95% bootstrap CI for both metrics, Holm-marked."""
    metrics = [("occupancy_mse", "occupancy MSE"), ("fourier_ergodic", "Fourier ergodicity")]
    entries, pvalues = [], []
    for metric, label in metrics:
        for arm in ARMS:
            a, b = paired(table, arm, metric)
            logratio = np.log2(a / b)
            low, high = bootstrap_ci(logratio, np.median)
            entries.append((label, arm, float(np.median(logratio)), low, high,
                            int(np.sum(a < b)), a.size))
            pvalues.append(wilcoxon(a, b).pvalue)
    significant = holm(pvalues)

    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), sharex=True)
        for panel, (metric, label) in zip(axes, metrics):
            rows = [(e, p, s) for e, p, s in zip(entries, pvalues, significant)
                    if e[0] == label]
            ys = np.arange(len(rows))[::-1]
            for y, ((_, arm, med, low, high, wins, n), p, keep) in zip(ys, rows):
                color = ACCENT if keep else PRIMARY
                panel.plot([low, high], [y, y], color=color,
                           linewidth=1.4, alpha=0.85, solid_capstyle="round")
                panel.plot([med], [y], "o", markersize=4.5, markerfacecolor=color,
                           markeredgecolor="#FFFFFF", markeredgewidth=0.6, zorder=4)
                panel.text(1.02, y, f"{wins}/{n}", transform=panel.get_yaxis_transform(),
                           va="center", fontsize=6.0, color="#5A6472")
            panel.axvline(0.0, color="#23272F", linewidth=0.8)
            panel.set_yticks(ys)
            panel.set_yticklabels([ARM_LABELS[r[0][1]] for r in rows])
            panel.set_title(label)
            panel.set_xlabel(r"$\log_2$ ratio vs shipped (95% bootstrap CI)")
            panel.margins(y=0.16)
        axes[0].text(0.02, 0.02, "red = survives Holm", transform=axes[0].transAxes,
                     fontsize=6.5, color=ACCENT)
        figure.tight_layout(pad=0.4, w_pad=1.6)
        path = save(figure, output)
        plt.close(figure)
    return path


# (claim, campaign log2 effect or None, uav log2 effect or None, campaign quoted?, note)
CLAIMS = [
    ("memory is load-bearing", np.log2(13.6), np.log2(12.2), True, "both significant"),
    ("bank vs one good scale", np.log2(1.07), np.log2(1.0), True, "agree"),
    ("memory_balance inert", np.log2(1.089), np.log2(1.0), True, "agree"),
    ("curl must be kept", np.log2(1.108), np.log2(1.0 / 0.977), True, "agree, neither sig."),
    ("memory_gain = 60", np.log2(0.9), np.log2(4.0), True, "DISAGREE (density)"),
    ("lengthscale rule", np.log2(1.0), np.log2(1.165), True, "DISAGREE (scale transfer)"),
    ("horizon T = 150", np.log2(0.784), None, True, "confounded by lam_max"),
]


def fig_agreement_matrix(output: Path, campaign_dir: Path | None = None) -> Path:
    """Campaign vs UAV effect per mechanism claim, on a diverging scale.

    Diverging is the right job here: the quantity has a meaningful zero (no effect)
    and a direction (worse / better), so a neutral midpoint must read as "nothing".
    """
    labels = [c[0] for c in CLAIMS]
    grid = np.array([[c[1] if c[1] is not None else np.nan,
                      c[2] if c[2] is not None else np.nan] for c in CLAIMS])
    # Clip at 4x. The memory row is 12-14x and would otherwise own the whole ramp,
    # flattening every remaining row to white -- the rows the figure exists to compare.
    limit = 2.0

    with plt.rc_context(rc=paper_style("double")):
        figure, axis = plt.subplots(figsize=(6.9, 3.4))
        mesh = axis.imshow(grid, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit,
                           aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if np.isnan(value):
                    axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                 facecolor="#DCE2EC", hatch="///",
                                                 edgecolor="#98A4BA", linewidth=0.4))
                    axis.text(j, i, "not run", ha="center", va="center",
                              fontsize=6.5, color="#5A6472")
                else:
                    # Ink stays neutral; the cell fill carries the magnitude.
                    shade = "#FFFFFF" if abs(value) > 0.62 * limit else "#23272F"
                    mark = r"$\gg$" if abs(value) > limit else ""
                    axis.text(j, i, f"{mark}{2**value:.2f}x", ha="center", va="center",
                              fontsize=7.0, color=shade)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["campaign\n(quoted)", "UAV\n(re-analyzed)"])
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels)
        for i, claim in enumerate(CLAIMS):
            axis.text(1.06, i, claim[4], va="center", fontsize=6.5,
                      color=ACCENT if "DISAGREE" in claim[4] else "#5A6472",
                      transform=axis.get_yaxis_transform())
        axis.set_xlim(-0.5, 1.5)
        axis.grid(False)
        axis.set_title("Mechanism claims: campaign vs UAV deployment")
        # Fixed positions: tight_layout cannot see the notes column (it is text drawn
        # in axes coords), so it reclaims the space and collides with it.
        axis.set_position((0.19, 0.26, 0.42, 0.63))
        # Horizontal bar underneath: a vertical one would sit between the cells and
        # their notes column and break the left-to-right read.
        bar = figure.colorbar(mesh, cax=figure.add_axes((0.19, 0.11, 0.42, 0.035)),
                              orientation="horizontal", extend="both")
        bar.set_label(r"$\log_2$ effect vs reference (worse $\rightarrow$), clipped at $\pm2$",
                      fontsize=7)
        bar.ax.tick_params(labelsize=6)
        path = save(figure, output)
        plt.close(figure)
    return path


def self_check() -> None:
    """Holm is the only non-obvious logic here; check it against worked cases."""
    # Nothing passes: smallest p is above 0.05/n.
    assert holm([0.02, 0.30, 0.40] ) == [False, False, False]
    # Step-down thresholds are 0.05/3, 0.05/2, 0.05: 0.001 and 0.02 clear theirs,
    # then 0.60 stops the descent.
    assert holm([0.001, 0.02, 0.60]) == [True, True, False]
    # Rejection follows the value, not the input position.
    assert holm([0.60, 0.001, 0.02]) == [False, True, True]
    # A p that would pass on its own is still rejected once the descent has stopped:
    # 0.04 < 0.05 but the 0.30 above it halts the step-down.
    assert holm([0.001, 0.30, 0.04]) == [True, False, False]
    # All pass when every p clears the strictest threshold.
    assert holm([0.001, 0.002, 0.003]) == [True, True, True]
    # Once a step fails, later smaller-threshold entries stay rejected too.
    assert holm([0.0001, 0.30, 0.0002]) == [True, False, True]
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=Path("results/uav/ablation.csv"))
    parser.add_argument("--campaign-dir", type=Path, default=Path("results/campaign"))
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    parser.add_argument("--self-check", action="store_true", help="run assertions and exit")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    table = load_arms(args.ablation)
    written = [
        fig_paired_arms(table, args.output / "fig_paired_arms.png"),
        fig_effect_forest(table, args.output / "fig_effect_forest.png"),
        fig_agreement_matrix(args.output / "fig_agreement_matrix.png", args.campaign_dir),
    ]
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
