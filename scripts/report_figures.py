"""Figures for the cross-campaign report.

Reads the shipped UAV per-seed CSVs and renders the three figures the report needs:

    fig_paired_arms      per-seed paired effect vs the shipped arm (violin + points)
    fig_effect_forest    median ratio + bootstrap CI, Holm-marked
    fig_agreement_matrix campaign vs UAV verdict per mechanism claim
    fig_dot_matrix       every run as one dot, axes ranked by spread of their medians

    uv run python scripts/report_figures.py --output results/report

Campaign-side stages need `results/campaign/<stage>.csv`; the matrix falls back to the
values quoted in campaign_findings.md and marks them as quoted when the raw CSVs are absent.
"""

import argparse
import csv
import json
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
    OUTSIDE_TICKS,
    PRIMARY,
    paper_style,
    save,
)

BASELINE = "baseline"
ARMS = ["h_0.94", "h_6.6", "theta_0", "theta_15", "theta_45"]
ARM_LABELS = {
    "h_0.94": "$h{=}0.94$",
    "h_2.35": "$h{=}2.35$",
    "h_6.6": "$h{=}6.6$",
    "h_8.5": "$h{=}8.5$",
    "theta_0": r"$\theta{=}0$",
    "theta_15": r"$\theta{=}15$",
    "theta_45": r"$\theta{=}45$",
    "theta_60": r"$\theta{=}60$",
    "theta_75": r"$\theta{=}75$",
    "gain_8": r"$k_{\mathcal{M}}{=}8$",
    "gain_30": r"$k_{\mathcal{M}}{=}30$",
    "gain_60": r"$k_{\mathcal{M}}{=}60$",
    "tau_3": r"$\tau_{\mathcal{M}}{=}3$",
    "tau_11": r"$\tau_{\mathcal{M}}{=}11$",
    "tau_20": r"$\tau_{\mathcal{M}}{=}20$",
    "tau_30": r"$\tau_{\mathcal{M}}{=}30$",
    "T_150": "$T{=}150$",
    "T_500": "$T{=}500$",
    "T_750": "$T{=}750$",
    "K_125": "$N{=}125$",
    "K_500": "$N{=}500$",
    "ell_self_0.25": r"$\ell_{\min}{=}0.25$",
    "ell_self_4.0": r"$\ell_{\min}{=}4.0$",
    "balance_0.5": "$a{=}0.5$",
    "flow_1500": r"$\gamma{=}1500$",
    "flow_6000": r"$\gamma{=}6000$",
    "penalty_0.1": r"$w_{\rm obs}{\times}0.1$",
    "boundary_0.1": r"$w_{\partial}{\times}0.1$",
    "lam_max_1e5": r"$\lambda_{\max}{=}10^5$",
    "explore_0": r"$f_{\rm ex}{=}0$",
    "explore_0.3": r"$f_{\rm ex}{=}0.3$",
    # Added for the final nine-map campaign.
    "theta_30": r"$\theta{=}30$",
    "alpha_0.8": r"$\alpha{=}0.80$",
    "alpha_0.9": r"$\alpha{=}0.90$",
    "alpha_0.99": r"$\alpha{=}0.99$",
    "alpha_1.0": r"$\alpha{=}1.0$",
    "h_4.0": "$h{=}4.0$",
    "h_11.0": "$h{=}11.0$",
    "ell_self_2.0": r"$\ell_{\min}{=}2.0$",
    "K_1000": "$N{=}1000$",
    "lam_max_1e4": r"$\lambda_{\max}{=}10^4$",
    "refspeed_2.5": "$v{=}2.5$",
    "refspeed_3.0": "$v{=}3.0$",
    "memory_off": "Memory off",
    "Q2": "$Q{=}2$",
    "Q3_fine": "$Q{=}3$ fine",
    "Q3_coarse": "$Q{=}3$ coarse",
    "baseline@108": "Baseline",
    "baseline@27": "Baseline (w27)",
}

# Two lines each, so the widest line rather than the whole phrase has to fit the block --
# an axis block is only 0.14in per arm at 6.9in, and three of these axes hold a single arm.
# The first line names what the knob is *for*, the second names the symbol, so the header
# gives the intuition and the arm labels below give the levels.
AXIS_LABELS = {
    "memory_gain": "Memory\ngain $k_{\\mathcal{M}}$",
    "T": "Horizon\n$T$",
    "alpha": "Control\ncost $\\alpha$",
    "K": "Rollout\nsamples $N$",
    "theta": "Curl\n$\\theta$",
    "memory_time": "Memory\ntime $\\tau_{\\mathcal{M}}$",
    "exploration": "Explore\nfraction $f_{\\rm ex}$",
    "fine_bandwidth": "Memory\nbandwidth $h_f$",
    "reference_speed": "Ref.\nspeed $v$",
    "penalty_scale": "Obstacle\npenalty $w_{\\rm obs}$",
    "memory_scales": "Scale\nbank $Q$",
    "boundary_scale": "Wall\npenalty $w_{\\partial}$",
    "lam_max": "Temp.\ncap $\\lambda$",
    "ell_self": "Attraction\nfloor $\\ell_{\\min}$",
    "memory_balance": "Memory\nbalance $a$",
    "flow_weight": "Stein\nflow $\\gamma$",
}

# The five outcomes the sensitivity panel decomposes over, with the transform that makes a
# paired difference meaningful. `occupancy_mse` is deliberately absent: it is ~redundant
# with the Fourier metric, and the joint tick already shows what redundancy costs.
#
# Direction does not matter here. The sensitivity is the *magnitude* of the standardised
# effect, so an arm that halves the dwell and one that doubles it are equally influential --
# which is the right reading for "how much does this knob move the system".
OUTCOMES = (
    ("fourier_ergodic", "log", "Fourier ergodicity"),
    ("tours", "raw", "Tours"),
    ("mode_dwell_median_s", "log", "Dwell"),
    ("in_mode_fraction", "logit", "In-mode fraction"),
    ("speed_mps", "raw", "Achieved speed"),
)
# One colour per outcome, listed **bottom-to-top** in stacking order -- the reverse of how a
# published legend reads down the page, so a palette lifted from one has to be flipped.
# Amber through magenta to navy. Deliberately shares no hue with the diverging red/green
# above it: the panels sit on one x axis but measure different things, and a warmer palette
# here -- Nature's red-orange-to-blue was the alternative -- puts "worse" reds directly under
# the dot panel's "worse" reds and invites reading the two as one scale. Being sequential
# also makes the stack read as ordered layers rather than five unrelated categories.
# Matplotlib's `Blues` sampled at 0.22/0.40/0.58/0.76/0.97 -- the near-white end trimmed off,
# or the bottom band would be indistinguishable from the page. Single-hue on purpose: the
# panel is one quantity decomposed into parts, not five categories, and leaving the colour to
# the two verdict panels above keeps the reading order right. Amber-to-navy, cyan-to-magenta
# and a grey ramp were all tried; the saturated ones pulled the eye away from the verdicts.
BAND_COLOURS = ("#ccdff1", "#94c4df", "#519ccc", "#1f6eb3", "#083776")


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


# 25p/525 is a second copy of a map already in the campaign, not a ninth map. Seed 525
# qualified at both 15 and 25 pillars -- 492 against 824 occupied cells -- but
# `final_ablation.py`'s `_configs` cached map arrays by seed alone, so every lane labelled
# 25p/525 flew the 15-pillar field. Its rows are bit-identical to 15p/525's. Left in, one map
# would cast two votes in every per-map gate and the campaign would claim nine maps it does
# not have. The driver is fixed; this drops the rows the broken driver already wrote.
DUPLICATE_MAPS = {(25, 525)}


def load_final(path: Path) -> dict[str, dict[tuple, dict[str, str]]]:
    """Index the campaign as ``arm -> cell -> row`` with ``cell`` the paired unit.

    The cell is ``(obs_num, map_seed, seed)``, not the seed: the same map seed can be
    selected at two densities, where it is a completely different field. Keying on the seed
    alone would pair rows across densities.

    Baselines are stored per lane count as ``baseline@<lanes>``. The campaign ran a second
    baseline at width 27 so the quarantined axes would have a comparator on their own
    numerical branch, and a single ``baseline`` key let whichever group came last in the file
    serve both -- which silently compared 41 width-108 arms against a width-27 baseline. Same
    width means same branch is the assumption the whole cross-group design rests on, so the
    width belongs in the key. Resolve an arm's own baseline with :func:`baseline_for`.
    """
    table: dict[str, dict[tuple, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cell = (int(row["obs_num"]), int(row["map_seed"]), int(row["seed"]))
            if cell[:2] in DUPLICATE_MAPS:
                continue
            arm = row["arm"]
            table[f"{arm}@{row['lanes']}" if arm == BASELINE else arm][cell] = row
    return table


def baseline_for(table, arm: str) -> str:
    """The baseline measured at this arm's own lane count -- i.e. on its own branch."""
    return f"{BASELINE}@{next(iter(table[arm].values()))['lanes']}"


def outcome_values(rows: list[dict[str, str]], name: str, transform: str) -> np.ndarray:
    """Extract one outcome from rows, on the scale its paired difference is taken on."""
    if name == "tours":
        raw = np.array([float(r["all_modes_reached"]) + float(r["mode_cycles"])
                        for r in rows])
    elif name == "speed_mps":
        # Achieved speed is not a stored column; it is path length over flown time, and it
        # is the quantity the alpha finding is actually about.
        raw = np.array([float(r["path_length_m"]) / (float(r["steps"]) * 0.02)
                        for r in rows])
    else:
        raw = np.array([float(r[name]) for r in rows])
    if transform == "log":
        # A dwell of exactly zero happens when a cell never qualified as in-mode. Flooring
        # rather than dropping keeps the pairing balanced; the floor is far below any real
        # value so it cannot manufacture an effect.
        return np.log(np.maximum(raw, 1e-6))
    if transform == "logit":
        clipped = np.clip(raw, 1e-3, 1.0 - 1e-3)
        return np.log(clipped / (1.0 - clipped))
    return raw


TYPICAL = "typical"


def add_typical_reference(table, name: str = TYPICAL):
    """Add a synthetic reference arm: the per-cell median over every real arm.

    Paired against the baseline, the baseline itself is zero on every cell by construction,
    so it cannot be drawn -- the figure shows 45 arms falling away from an invisible origin.
    Re-referencing to the median arm puts the baseline back in as a column and lets it be
    read against its alternatives rather than assumed as the origin.

    The reference is a column-wise median, so it is a summary of the arm table and *not* a
    run that happened; it is a yardstick, not a configuration. Everything inferential
    (`final_report.py`) stays paired against the baseline, where the contrast is the knob
    change and the comparison is a real one.
    """
    # Width-108 arms only: the quarantined width-27 set includes a second copy of the
    # baseline, and a median that counted the shipped profile twice would be pulled toward
    # the very column this reference exists to place fairly.
    members = [a for a in table
               if a != name and next(iter(table[a].values()))["lanes"] == "108"]
    cells = set.intersection(*(set(table[a]) for a in members))
    reference = {}
    for cell in cells:
        rows = [table[arm][cell] for arm in members]
        merged = dict(rows[0])
        for key in merged:
            numbers = []
            for row in rows:
                try:
                    numbers.append(float(row[key]))
                except (TypeError, ValueError):
                    break
            else:
                merged[key] = repr(float(np.median(numbers)))
        merged["arm"] = name
        reference[cell] = merged
    table[name] = reference
    return table


def paired_final(table, arm: str, metric: str, transform: str = "raw",
                 reference: str | None = None):
    """Return ``(arm_values, reference_values, cells)`` over the cells both arms share.

    ``reference`` defaults to this arm's own-width baseline -- see :func:`baseline_for`.
    """
    reference = reference or baseline_for(table, arm)
    cells = sorted(set(table[arm]) & set(table[reference]))
    arm_rows = [table[arm][c] for c in cells]
    base_rows = [table[reference][c] for c in cells]
    return (outcome_values(arm_rows, metric, transform),
            outcome_values(base_rows, metric, transform), cells)


def sensitivity(table, arm: str,
                reference: str | None = None) -> tuple[dict[str, float], float]:
    """Per-outcome and joint Fisher sensitivity of one arm against the baseline.

    For outcome ``m`` with paired differences ``d`` over the shared cells, the returned
    per-outcome value is the standardised paired effect in **noise units**

        z_m = | mean(d) | / sd(d)

    and its square is the Fisher information one run carries about that arm's contrast under
    a Gaussian working model -- the scalar case of ``I = (dmu/dp)^T Sigma^-1 (dmu/dp)``. It
    is dimensionless, so knobs with wildly different units are comparable.

    Noise units rather than the squared form because the squared form spans five orders of
    magnitude across this arm table (a broken arm reaches ~3e4 while a null sits at 1e-2),
    which leaves every column but one invisible. The square root costs nothing: it is
    monotone, so the ranking is identical.

    The joint value returned alongside is the Mahalanobis distance

        z = sqrt( g^T Sigma^-1 g )

    with ``g`` the vector of mean differences and ``Sigma`` their covariance across cells.

    **The stack is not a bound on the joint value in either direction.** Summing z_m treats
    the outcomes as independent, and they are not -- they come from one trajectory. Where
    they are redundant the joint falls below the sum; where they carry complementary
    information (one outcome sharpening another once its variance is projected out) the
    joint can exceed it. Measured on synthetic arms it does both. The figure draws both
    numbers for exactly that reason: the gap is a property of the outcome set, and reporting
    only one of them would assert a relationship that does not hold.
    """
    columns, means = [], []
    per_outcome: dict[str, float] = {}
    for name, transform, _ in OUTCOMES:
        arm_values, base_values, _ = paired_final(table, arm, name, transform, reference)
        difference = arm_values - base_values
        spread = float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0
        per_outcome[name] = abs(float(np.mean(difference)) / spread) if spread > 0 else 0.0
        columns.append(difference)
        means.append(float(np.mean(difference)))

    matrix = np.vstack(columns)
    gradient = np.array(means)
    covariance = np.cov(matrix)
    # Ridge on the scale of the diagonal: with 108 cells and 5 outcomes the covariance is
    # well determined, but two near-collinear outcomes can still make it ill-conditioned,
    # and an inflated joint value would understate exactly the redundancy this measures.
    ridge = 1e-8 * float(np.trace(covariance)) / covariance.shape[0]
    joint = float(gradient @ np.linalg.solve(
        covariance + ridge * np.eye(covariance.shape[0]), gradient
    ))
    return per_outcome, float(np.sqrt(max(joint, 0.0)))


def per_map_effects(table, arm: str, metric: str = "fourier_ergodic",
                    standardize: bool = True,
                    reference: str | None = None) -> dict[tuple, float]:
    """Per-map effect keyed ``(obs_num, map_seed)``, standardised by its own noise.

    This is what the consistency strip draws and what the promotion gate counts. The dot
    matrix and the sensitivity panel both pool over maps, and pooling is precisely what hid
    the two findings this campaign exists to avoid repeating.

    Standardised, not raw, and that matters: a per-map median over twelve seeds carries a
    standard error of roughly 0.13 in log2 units on this data, so a fixed neutral band of
    +/-0.1 would colour pure coin flips as effects and the strip would report nine
    independent noise draws as a consistency pattern. Dividing by the median's own standard
    error (1.253 * sd / sqrt(n) for a normal) makes each cell read "this map's effect, in
    its own sigmas", so the same threshold means the same thing on every map and every arm.

    Pass ``standardize=False`` for the raw median log2 ratio, which is the effect *size*
    rather than its reliability.
    """
    arm_values, base_values, cells = paired_final(table, arm, metric, "raw", reference)
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for (obs_num, map_seed, _), a, b in zip(cells, arm_values, base_values):
        grouped[(obs_num, map_seed)].append(np.log2(b / a))
    effects = {}
    for key, values in grouped.items():
        median = float(np.median(values))
        if not standardize:
            effects[key] = median
            continue
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        error = 1.253 * spread / np.sqrt(len(values)) if spread > 0 else 0.0
        effects[key] = median / error if error > 0 else 0.0
    return effects


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


def holm_by_axis(table, arms: list[str], pvalues: list[float]) -> list[bool]:
    """Holm-Bonferroni applied within each one-factor axis rather than across all arms.

    A one-factor-at-a-time sweep asks a separate question per axis, so the multiplicity to
    correct is the levels of that axis, not the whole table. Correcting across ~30 arms at
    12 seeds would demand p < 0.0017 against a Wilcoxon floor of ~4.9e-4, which only an
    11-of-12 unanimous arm can reach; per axis the bar is ~0.01, i.e. 10 of 12.
    """
    families: dict[str, list[int]] = defaultdict(list)
    for index, arm in enumerate(arms):
        row = next(iter(table[arm].values()))
        families[row.get("axis") or arm].append(index)
    reject = [False] * len(arms)
    for indices in families.values():
        for index, keep in zip(indices, holm([pvalues[i] for i in indices])):
            reject[index] = keep
    return reject


# Eleven bins for the dots, five for the strip, in matched pairs so both panels always speak
# the same diverging language. `rdylgn` is the original red-yellow-green; `rdblu` is drawn
# from the sensitivity panel's own endpoints -- its coral `#ff6361` and navy `#003f5c` -- so
# all three panels come from one family. Both carry a light grey neutral: the warm ivory that
# preceded it sat within a few points of the shipped column's highlight.
# Eleven bins for the dots, five for the strip, sharing one neutral so both panels speak the
# same diverging language. Red-yellow-green: red and green carry "bad" and "good" without a
# legend, which a red-blue or magenta-amber ramp -- both tried -- does not. The neutral is a
# light grey rather than the warm ivory it replaced: a warm neutral on a warm ramp reads as a
# low value on the scale instead of as the absence of one.
NEUTRAL_BIN = "#d9d9d9"
DOT_COLOURS = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", NEUTRAL_BIN,
               "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#2c7fb8"]
STRIP_COLOURS = ["#d73027", "#fdae61", NEUTRAL_BIN, "#a6d96a", "#1a9850"]
DOT_EDGES = [-9, -2.0, -1.5, -1.0, -0.5, -0.15, 0.15, 0.5, 1.0, 1.5, 2.0, 9]


def fig_dot_matrix(table, output: Path, metric: str = "fourier_ergodic") -> Path:
    """Every run in the campaign as one dot, arms grouped by axis and ranked by spread.

    The forest and violin figures summarise each arm to a median and an interval, which
    hides the thing this campaign most needs to show: at twelve seeds most axes move
    nothing, and a six-up/six-down column is what that looks like. Here nothing is smoothed
    and nothing is pooled -- for each arm the seeds that beat the baseline *on the same
    seed* stack upward and the rest stack down, so column height reads as consistency and
    colour as size. Axes are ordered by the spread of their level medians, which makes the
    "which parameter matters" ranking a consequence of the sort rather than an assertion.

    A ridgeline over the same data was rejected: a kernel density over twelve points draws
    shape the data does not contain, and the axes here have two to five levels, not the long
    ordered sequence that form needs.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    arms = sorted(set(table) - {BASELINE})
    effects, tours, axis_of, value_of = {}, {}, {}, {}
    for arm in arms:
        a, b = paired(table, arm, metric)
        # Inverted: lower error is better, so positive means the arm beat the baseline.
        effects[arm] = np.log2(b / a)
        rows = list(table[arm].values())
        axis_of[arm] = rows[0].get("axis") or arm
        value_of[arm] = float(rows[0].get("value") or 0.0)
        tours[arm] = sum(
            int(r["all_modes_reached"]) + float(r["mode_cycles"]) for r in rows
        )

    by_axis: dict[str, list[str]] = defaultdict(list)
    for arm in arms:
        by_axis[axis_of[arm]].append(arm)
    spans = {
        axis: max(np.median(effects[a]) for a in members)
        - min(np.median(effects[a]) for a in members)
        for axis, members in by_axis.items()
    }
    order = sorted(by_axis, key=lambda a: -spans[a])
    columns = [
        (axis, arm)
        for axis in order
        for arm in sorted(by_axis[axis], key=lambda a: value_of[a])
    ]

    cmap = ListedColormap(DOT_COLOURS)
    norm = BoundaryNorm(DOT_EDGES, cmap.N)
    seeds = max(len(v) for v in effects.values())
    with plt.rc_context(paper_style("double")):
        figure, axes = plt.subplots(figsize=(11.5, 4.6))
        # The shared paper style tints the panel and draws a grid, both of which fight a
        # chart whose only ink should be the dots. Overridden here rather than in the style,
        # which the other figures depend on.
        figure.patch.set_facecolor("white")
        axes.set_facecolor("white")
        axes.grid(False)
        lead = [i for i, (axis, _) in enumerate(columns) if axis == order[0]]
        axes.axvspan(min(lead) - 0.5, max(lead) + 0.5, color="#f0f0f0", zorder=0)
        axes.text(float(np.mean(lead)), seeds + 1.4, "W I D E S T   S P R E A D",
                  ha="center", va="center", fontsize=7.0, color="#9a9a9a")

        for index, (_, arm) in enumerate(columns):
            values = effects[arm]
            up = np.sort(values[values > 0])
            down = np.sort(values[values <= 0])[::-1]
            for stack, sign in ((up, 1), (down, -1)):
                if stack.size:
                    axes.scatter(np.full(stack.size, index),
                                 sign * np.arange(1, stack.size + 1),
                                 c=stack, cmap=cmap, norm=norm, s=26,
                                 linewidths=0, zorder=3)
            # A separate gutter for the failure a coverage metric cannot express: an arm
            # that never completed a tour in any seed.
            if tours[arm] == 0:
                axes.scatter([index], [-seeds - 2.2], s=26, color="#4d4d4d",
                             linewidths=0, zorder=3)

        axes.axhline(0.0, color="#cfcfcf", linewidth=0.8, zorder=2)
        for axis in order[1:]:
            first = min(i for i, (a, _) in enumerate(columns) if a == axis)
            axes.axvline(first - 0.5, color="#e8e8e8", linewidth=0.7, zorder=1)

        singles = [i for i, (a, _) in enumerate(columns) if len(by_axis[a]) == 1]
        for axis in order:
            members = [i for i, (a, _) in enumerate(columns) if a == axis]
            if len(members) > 1:
                axes.text(float(np.mean(members)), seeds + 3.4, axis.replace("_", " "),
                          ha="center", va="center", fontsize=7.5, color="#6f6f6f")
        if singles:
            axes.text(float(np.mean(singles)), seeds + 3.4, "one level only", ha="center",
                      va="center", fontsize=7.5, color="#a8a8a8", style="italic")

        ticks = [t for t in range(-seeds, seeds + 1, 4)]
        axes.set_xticks(range(len(columns)))
        axes.set_xticklabels([ARM_LABELS.get(a, a) for _, a in columns],
                             rotation=90, fontsize=6.5, color=NEUTRAL)
        axes.set_yticks(ticks)
        axes.set_yticklabels([str(abs(t)) for t in ticks], fontsize=7, color=NEUTRAL)
        axes.text(-2.6, seeds / 2.0, "SEEDS IMPROVED >", rotation=90, ha="center",
                  va="center", fontsize=6.8, color=NEUTRAL)
        axes.text(-2.6, -seeds / 2.0, "< SEEDS WORSENED", rotation=90, ha="center",
                  va="center", fontsize=6.8, color=NEUTRAL)
        axes.text(-1.2, -seeds - 2.2, "No tour", ha="right", va="center", fontsize=6.5,
                  color=NEUTRAL)
        axes.set_xlim(-0.8, len(columns) - 0.2)
        axes.set_ylim(-seeds - 3.4, seeds + 4.4)
        for side in ("top", "right", "left", "bottom"):
            axes.spines[side].set_visible(False)
        axes.tick_params(length=0, which="both")
        axes.minorticks_off()

        left, width = 0.60, 0.30
        for slot, colour in enumerate(DOT_COLOURS):
            figure.patches.append(plt.Rectangle(
                (left + slot * width / len(DOT_COLOURS), 0.935),
                width / len(DOT_COLOURS), 0.028, transform=figure.transFigure,
                facecolor=colour, edgecolor="white", linewidth=0.6, clip_on=False))
        for fraction, label in ((0.0, "4x worse"), (0.5, "no change"), (1.0, "4x better")):
            figure.text(left + fraction * width, 0.905, label, ha="center", va="top",
                        fontsize=6.5, color=NEUTRAL)
        figure.text(left, 0.985,
                    f"paired change in {metric.replace('_', ' ')} vs baseline, per seed",
                    ha="left", va="top", fontsize=7, color="#6f6f6f")

        figure.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.22)
        path = save(figure, output)
        plt.close(figure)
        return path


def fig_final_ablation(table, output: Path, metric: str = "fourier_ergodic",
                       reference: str | None = None, dots_per_row: int = 4,
                       consistency: bool = True) -> Path:
    """The nine-map campaign in one figure: dots, per-map consistency, sensitivity.

    Three panels on one categorical x axis of arms, blocked by axis and ordered by joint
    sensitivity so "which parameter matters most" is a consequence of the sort:

    **Top -- one dot per run.** One per (map, seed), above the line if that run beat the
    reference on the same cell, below otherwise. Height is how often; colour is by how much
    (log2 ratio, positive = the arm won). Each stack is sorted **whole** and filled
    row-major at ``dots_per_row`` dots per row, so a column reads outward from the line as
    best run to worst, the colour ramp is monotone, and rows times the row width is the run
    count. An earlier version blocked the stacks by map to keep a one-map win from reading
    like an all-map win; it restarted the ramp once per map and cost more than it bought,
    and the strip below carries that distinction properly.

    **Middle -- the consistency strip.** One mark per (arm, map), grouped by density,
    coloured by that map's median effect **divided by its own standard error**. A promotable
    arm is a row of one colour; the map-agreement gate is legible rather than asserted. Five
    coarse bins, not the dots' eleven: eight medians of twelve seeds do not support finer
    gradation, and at this cell size finer bins would not survive print.

    The two panels share a palette but **not a scale**, which is the one thing a reader can
    get wrong here: the dots are an effect size in log2, the strip is a signal-to-noise
    ratio in sigmas. Because most arms are worse on the pooled median while most per-map
    effects sit inside their own error bars, the dots run red-heavy and the strip runs
    cream-heavy, and the mismatch is systematic rather than occasional. Hence the divider
    between them and the two separately captioned legends: strip cream means "not resolved",
    the absence of an answer, not a midpoint on the dots' scale.

    **Bottom -- the sensitivity panel.** Stacked squared paired effect per outcome, with a
    tick at the correlation-corrected joint value. See :func:`sensitivity`. Zero-gap bars
    rather than a filled area: x is categorical, and a slope between two unrelated arms
    would assert an interpolation that does not exist.

    ``reference`` selects what every dot is paired against. The default is the shipped
    profile, which is the contrast the campaign was designed around but which cannot draw
    the profile itself: baseline minus baseline is zero on all 108 cells. Pass ``TYPICAL``
    (after :func:`add_typical_reference`) to pair against the per-cell median arm instead,
    which puts the baseline in as a column alongside its alternatives. The panels are
    otherwise identical -- only the origin moves.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    # Under the default reference each arm pairs against the baseline at its own lane count,
    # so no baseline is drawable and both are dropped. Under a neutral reference the shipped
    # profile is a column like any other, and it is the width-108 one -- the branch 41 of the
    # 45 arms ran on.
    shipped = f"{BASELINE}@108"
    hidden = {reference, TYPICAL} | (
        {a for a in table if a.startswith(BASELINE + "@")}
        if reference is None else {f"{BASELINE}@27"}
    )
    arms = [a for a in table if a not in hidden]
    effects, tours, axis_of, value_of = {}, {}, {}, {}
    per_map, bands, joints = {}, {}, {}
    for arm in arms:
        arm_values, base_values, cells = paired_final(table, arm, metric,
                                                      reference=reference)
        effects[arm] = (np.log2(base_values / arm_values), cells)
        rows = list(table[arm].values())
        axis_of[arm] = rows[0].get("axis") or arm
        try:
            value_of[arm] = float(rows[0].get("value") or 0.0)
        except ValueError:
            value_of[arm] = 0.0
        tours[arm] = sum(int(float(r["all_modes_reached"])) + float(r["mode_cycles"])
                         for r in rows)
        per_map[arm] = per_map_effects(table, arm, metric, reference=reference)
        bands[arm], joints[arm] = sensitivity(table, arm, reference)

    by_axis: dict[str, list[str]] = defaultdict(list)
    for arm in arms:
        by_axis[axis_of[arm]].append(arm)
    # Ranked by the *joint* sensitivity of an axis's strongest arm: a multi-outcome
    # statement, unlike the prototype's spread of medians in a single metric.
    order = sorted(by_axis, key=lambda a: -max(joints[x] for x in by_axis[a]))
    # The shipped profile is a column only under a neutral reference, and then it is the one
    # column the reader is looking for. Sensitivity ranking would bury it: it is closest to
    # the median arm precisely because most knobs do nothing, which is the finding, not a
    # reason to hide it. Pin it leftmost instead.
    if shipped in arms:
        order.insert(0, order.pop(order.index(axis_of[shipped])))
    columns = [(axis, arm) for axis in order
               for arm in sorted(by_axis[axis], key=lambda a: value_of[a])]

    maps = sorted({key for arm in arms for key in per_map[arm]})
    runs = max(len(v[0]) for v in effects.values())
    cmap = ListedColormap(DOT_COLOURS)
    norm = BoundaryNorm(DOT_EDGES, cmap.N)
    # Binned in units of the per-map median's own standard error, so "coloured" means
    # "resolved above this map's seed noise" rather than "exceeded a number someone picked".
    #
    # The first edge is at 2 sigma, not 1. At 1 sigma a pure null colours 32% of its cells,
    # so a knob that does nothing would render as a patchy 3-of-9 pattern that reads as
    # structure; at 2 sigma that falls to 4.6%, i.e. well under one cell of the nine. The
    # strip's whole job is to stop noise being read as consistency, so its threshold has to
    # be one a null actually fails.
    strip_colours = ListedColormap(STRIP_COLOURS)
    strip_norm = BoundaryNorm([-1e3, -3.5, -2.0, 2.0, 3.5, 1e3], strip_colours.N)

    with plt.rc_context(paper_style("double")):
        # Type scale for a 6.9in canvas. Points are absolute, so these are the sizes that
        # actually reach the page: 4.4pt is the floor IEEE figure text stays legible at.
        tiny, small, mid, lead = 4.7, 5.2, 5.7, 6.2
        # 6.9in is this project's full-width `figure*` size -- every other double-column
        # figure in the repo uses it. Sized for inclusion at 1:1, not scaled down by LaTeX:
        # matplotlib text is in absolute points, so drawing at 13in and letting
        # \includegraphics shrink it to 6.9 would have taken 6pt labels down to 3.2pt.
        #
        # `consistency=False` drops the middle panel and the canvas height it occupied, so
        # the two surviving panels keep their absolute size rather than stretching to fill.
        # The sensitivity panel is read for rank order and for whether a bar clears the
        # 3-sigma floor, not for its absolute height, so it can be squeezed harder than the
        # dot grid above it. Tightened hspace pulls it up against the dots, which also makes
        # the shared x axis easier to read across the two panels.
        heights = [3.0, 0.60, 1.45] if consistency else [3.0, 0.95]
        figure, panels = plt.subplots(
            len(heights), 1, figsize=(6.9, 5.2 if consistency else 3.6), sharex=True,
            gridspec_kw={"height_ratios": heights, "hspace": 0.085 if consistency else 0.04},
        )
        top, bottom = panels[0], panels[-1]
        strip = panels[1] if consistency else None
        # Applied here rather than at the end: both the dots and the strip size their markers
        # in points from their own realised geometry, so the panel boxes have to be final
        # before either draws. The margins are larger fractions than at 13in because the text
        # inside them did not shrink with the canvas.
        # Margins converted from the 5.2in layout by height, not copied: they hold text, and
        # text is absolute points, so a shorter canvas needs a *larger* fraction to leave the
        # arm labels and the keys the same room they had.
        grew = 5.2 / figure.get_figheight()
        figure.subplots_adjust(left=0.080, right=0.995,
                               top=1.0 - 0.072 * grew, bottom=0.120 * grew)
        figure.patch.set_facecolor("white")
        for axes in panels:
            axes.set_facecolor("white")
            axes.grid(False)
            for side in ("top", "right", "left", "bottom"):
                axes.spines[side].set_visible(False)
            axes.tick_params(length=0, which="both")
            axes.minorticks_off()

        # ---- the shipped column, called out behind everything else
        #
        # A warm off-white. It has to clear the `#d9d9d9` neutral in *both* panels or the
        # shipped column's own unresolved cells vanish into their own highlight -- the one
        # column a reader goes looking for. A cool grey was tried and sat too close to it.
        if shipped in arms:
            slot = next(i for i, (_, a) in enumerate(columns) if a == shipped)
            for axes in panels:
                axes.axvspan(slot - 0.5, slot + 0.5, color="#f9f4ea", zorder=0)

        # ---- top: one dot per run, one sort per column
        #
        # One sort, not nine map-blocked ones. The blocked version was meant to keep a
        # one-map win from reading like a nine-map win, but each block sorts separately, so
        # the colour ramp restarted nine times going up a column -- a deep blue dot could
        # land at y=30 purely because its map's block fell low in the concatenation. Colour
        # *is* the effect size, so that made the panel's main channel unreadable vertically.
        # Sorted whole, a column reads top-to-bottom as best run to worst. Per-map structure
        # is the strip's job below, where it is nine explicit cells instead of nine stretches
        # of dots nobody can count.
        #
        # `dots_per_row` spends the column's horizontal slack: a column is 18.8 pt wide and a
        # dot is 2.65, while vertically 108 slots share 0.95 pt each, so the dots overlap
        # ~2.8x and render as a ribbon. Filling row-major at width 4 cuts the stack to 27
        # rows and buys ~3.8 pt of pitch in both directions. Row-major keeps the ramp
        # monotone (each row is four adjacent ranks) and keeps height proportional to count.
        limit = int(np.ceil(runs / dots_per_row))
        span = 0.62 if dots_per_row > 1 else 0.0
        # Dot size read off the realised panel rather than hard-coded: the pitch depends on
        # the figure width, the column count and the fill width all at once, and a constant
        # that looked right at 13in drew overlapping blobs at 6.9. Take whichever of the two
        # pitches is tighter -- horizontally the sub-slot inside a column, vertically one row.
        box = top.get_window_extent()
        points = 72.0 / figure.dpi
        column_pt = box.width * points / len(columns)
        row_pt = box.height * points / (2 * limit + 15.0)
        slot_pt = span * column_pt / max(dots_per_row - 1, 1) if dots_per_row > 1 else column_pt
        size = (0.82 * min(slot_pt, row_pt)) ** 2
        for index, (_, arm) in enumerate(columns):
            values, _cells = effects[arm]
            up = np.sort(values[values > 0])
            down = np.sort(values[values <= 0])[::-1]
            for stack, sign in ((up, 1), (down, -1)):
                if not stack.size:
                    continue
                row, column = np.divmod(np.arange(stack.size), dots_per_row)
                offset = (column - (dots_per_row - 1) / 2) * span / max(dots_per_row - 1, 1)
                top.scatter(index + offset, sign * (row + 1),
                            c=stack, cmap=cmap, norm=norm, s=size, linewidths=0, zorder=3)
            if tours[arm] == 0:
                top.scatter([index], [-limit - 4.0], s=size, color="#4d4d4d",
                            linewidths=0, zorder=3)
        top.axhline(0.0, color="#cfcfcf", linewidth=0.8, zorder=2)
        top.set_ylim(-limit - 6.0, limit + 9.5)  # provisional; finalised after the headers
        # Ticks stay labelled in runs whatever the fill width -- the reader counts flights,
        # not rows -- so the position is the run count divided by the row width.
        ticks = list(range(-runs, runs + 1, 24))
        top.set_yticks([t / dots_per_row for t in ticks])
        top.set_yticklabels([str(abs(t)) for t in ticks], fontsize=small, color="black")
        top.text(-3.3, limit * 0.55, "IMPROVED >", rotation=90, ha="center", va="center",
                 fontsize=small, color="black")
        top.text(-3.3, -limit * 0.55, "< WORSENED", rotation=90, ha="center",
                 va="center", fontsize=small, color="black")
        top.text(-3.3, 0.0, "RUNS", rotation=90, ha="center", va="center",
                 fontsize=small, color="black")
        top.text(-1.0, -limit - 4.0, "No tour", ha="right", va="center",
                 fontsize=tiny, color="black")

        if strip is not None:
            # ---- middle: per-map consistency, one mark per map, densities blocked
            #
            # Rectangles in data units rather than points-sized markers. A circle has to stay
            # round, so its diameter is bounded by the *shorter* of the row pitch and the column
            # pitch and the panel cannot be made flatter than eight round rows. A rectangle takes
            # the cell it is given, so the panel's height is free to shrink; the gap left around
            # each cell is what keeps the baseline column's highlight visible underneath.
            for index, (_, arm) in enumerate(columns):
                strip.bar(
                    np.full(len(maps), index), 0.80, width=0.82,
                    bottom=np.arange(len(maps)) - 0.40,
                    color=[strip_colours(strip_norm(per_map[arm].get(key, 0.0))) for key in maps],
                    linewidth=0, zorder=3)
            strip.set_ylim(-0.55, len(maps) - 0.45)
            # Labelled by density, not by map: which of the three fields at a density a cell
            # belongs to is not a question the figure is asked, and eight tiny `25p/516` labels
            # cost more legibility than they buy. The per-map breakdown stays in final_report.md.
            groups = defaultdict(list)
            for row, (obs, _) in enumerate(maps):
                groups[obs].append(row)
            # Named rather than numbered: a pillar count makes the reader convert a number into a
            # notion of clutter, which is the only thing the row grouping is for. The counts
            # themselves belong in the caption. The axis still says "by map" -- a group label
            # spans two or three map rows, it is not one row per density.
            names = ("Low", "Med.", "High")
            counts = sorted(groups)
            strip.set_yticks([float(np.mean(rows)) for rows in groups.values()])
            strip.set_yticklabels(
                [names[i] if len(counts) == len(names) else f"{obs}p"
                 for i, obs in enumerate(counts)], fontsize=small, color="black")
            # Names the factor the three group labels belong to. Turned on its side and set to
            # their left rather than stacked above them: the panel is now only ~0.6 of a unit
            # tall, so a two-line horizontal label had to overhang the top edge, and this column
            # of margin is free anyway. Sits between the group labels and the axis title at -3.3.
            strip.text(-2.5, (len(maps) - 1) / 2, "Obstacle density", rotation=90, ha="center",
                       va="center", fontsize=tiny, color="black")
            # Read off where the density actually changes, not every third row: the campaign is
            # 3 / 2 / 3 after 25p/525 was dropped as a duplicate, and a fixed stride would rule
            # the wrong rows and split a density group in half.
            for row in range(1, len(maps)):
                if maps[row][0] != maps[row - 1][0]:
                    strip.axhline(row - 0.5, color="#9a9a9a", linewidth=0.5)
            strip.text(-3.3, (len(maps) - 1) / 2, "CONSISTENCY BY MAP", rotation=90, ha="center",
                       va="center", fontsize=small, color="black")

        # ---- bottom: stacked sensitivity, joint tick on top
        #
        # `memory_off` runs to ~44 sigma while the next arm sits near 12, so a shared axis has
        # to choose which it serves. Scaling to the tallest column is strictly honest and was
        # tried: it costs the other 44 columns 3.3x of their height, drops `T_750` to a
        # quarter of the panel and turns everything past `h=4.0` into 1-2% slivers, which
        # destroys the 45-way ranking that is the panel's main job. So the ceiling comes from
        # the *second*-largest column, and the one stack above it is squeezed to fit under a
        # hatched cap -- its five bands stay readable, its true total is printed, and the
        # hatch says the height is not to scale.
        totals = {arm: sum(bands[arm].values()) for _, arm in columns}
        ranked = sorted(totals.values(), reverse=True)
        ceiling = (ranked[1] if len(ranked) > 1 else ranked[0]) * 1.18 or 1.0
        for index, (_, arm) in enumerate(columns):
            scale = ceiling * 0.93 / totals[arm] if totals[arm] > ceiling else 1.0
            base = 0.0
            for (name, _, _), colour in zip(OUTCOMES, BAND_COLOURS):
                height = bands[arm][name] * scale
                bottom.bar(index, height, bottom=base, width=1.0, color=colour,
                           linewidth=0, zorder=2)
                base += height
            if scale < 1.0:
                bottom.bar(index, ceiling * 0.07, bottom=ceiling * 0.93, width=1.0,
                           facecolor="none", edgecolor="#9a9a9a", hatch="////",
                           linewidth=0.4, zorder=2)
            if totals[arm] > ceiling:
                # Inside the hatch, not inside the stack: the top band is dark in two of the
                # three palettes and the label vanished into it.
                bottom.text(index, ceiling * 0.965, f"{totals[arm]:.0f}", ha="center",
                            va="center", fontsize=tiny, color="#1a1a1a", zorder=5)
        bottom.set_ylim(0, ceiling)
        # Step chosen so the shorter panel never draws more labels than it has room for:
        # a fixed 3-sigma step crowds into an unreadable smear once the ceiling is high.
        step = max(3, int(ceiling) // 5 // 3 * 3 or 3)
        bottom.set_yticks(range(0, int(ceiling) + 1, step))
        bottom.tick_params(axis="y", right=False)
        bottom.tick_params(axis="y", labelsize=small, colors="black", length=2)
        bottom.text(-3.3, bottom.get_ylim()[1] / 2, "SENSITIVITY", rotation=90,
                    ha="center", va="center", fontsize=small, color="black")

        # ---- shared x, axis blocks
        for axes in panels:
            for axis in order[1:]:
                first = min(i for i, (a, _) in enumerate(columns) if a == axis)
                axes.axvline(first - 0.5, color="#e8e8e8", linewidth=0.7, zorder=1)
        # One row, every axis. Stacking each header onto two lines is what makes this fit:
        # the constraint is the widest *line*, not the phrase, so "Obstacle / penalty" clears
        # a one-arm block where "obstacle penalty" never could. The baseline's pseudo-axis is
        # skipped -- its single arm label already reads "Baseline".
        #
        # Where two lines are still not enough -- three of these axes hold a single arm --
        # the label is turned on its side rather than left to run over its neighbours.
        # Decided by measuring the drawn text against the block, not by a hand-kept list,
        # because the widths depend on the figure size and on which arms an axis holds.
        renderer = figure.canvas.get_renderer()
        headers, upright = [], []
        for axis in (a for a in order if a != axis_of.get(shipped)):
            members = [i for i, (a, _) in enumerate(columns) if a == axis]
            label = AXIS_LABELS.get(axis, axis.replace("_", " "))
            text = top.text(float(np.mean(members)), limit + 1.6, label, ha="center",
                            va="bottom", fontsize=small, color="black", linespacing=1.15)
            edges = top.transData.transform(
                [(min(members) - 0.5, 0.0), (max(members) + 0.5, 0.0)])
            headers.append(text)
            if text.get_window_extent(renderer=renderer).width > edges[1][0] - edges[0][0]:
                text.set_text(label.replace("\n", " "))
                text.set_rotation(90)
                text.set_fontsize(tiny)
            else:
                upright.append(text)
        # Centre the sideways labels on the upright band rather than standing them on its
        # baseline: bottom-anchored, a rotated label spends its whole length upward and every
        # point of that is headroom stolen from the dots. Centred, it spends half.
        if upright:
            band = upright[0].get_window_extent(renderer=renderer)
            middle = top.transData.inverted().transform((0.0, (band.y0 + band.y1) / 2))[1]
            for text in headers:
                if text.get_rotation():
                    text.set_va("center")
                    text.set_position((text.get_position()[0], middle))
        # Then fit the ceiling to whatever the tallest header actually needs. Iterated,
        # because the labels are placed in data units and moving the limit moves them.
        for _ in range(3):
            tallest = max(t.get_window_extent(renderer=renderer).y1 for t in headers)
            top.set_ylim(-limit - 6.0,
                         top.transData.inverted().transform((0.0, tallest))[1] + 1.0)
        bottom.set_xticks(range(len(columns)))
        bottom.set_xticklabels([ARM_LABELS.get(a, a) for _, a in columns], rotation=90,
                               fontsize=tiny, color="black")
        # With a neutral reference the shipped profile is just another column, and the one
        # thing the reader is looking for is which column it is.
        for label, (_, arm) in zip(bottom.get_xticklabels(), columns):
            if arm == shipped:
                label.set_color("#1a1a1a")
                label.set_fontweight("bold")
        bottom.set_xlim(-0.7, len(columns) - 0.3)

        # ---- legends, side by side so the two scales are contrasted rather than conflated
        #
        # The panels share a palette but not a scale: the dots are an effect size in log2,
        # the strip is a signal-to-noise ratio in sigmas. Read as one scale, a red-heavy
        # column promises red cells below it -- and because most arms are worse on the pooled
        # median while most per-map effects sit inside their own error bars, that promise
        # fails systematically. Two captioned keys, adjacent, are the cheapest way to say
        # "these measure different things"; the divider below reinforces it.
        def _swatches(left, width, top_y, colours, ticks, caption):
            for slot, colour in enumerate(colours):
                figure.patches.append(plt.Rectangle(
                    (left + slot * width / len(colours), top_y),
                    width / len(colours), 0.018, transform=figure.transFigure,
                    facecolor=colour, edgecolor="white", linewidth=0.5, clip_on=False))
            for fraction, label in ticks:
                figure.text(left + fraction * width, top_y - 0.004, label, ha="center",
                            va="top", fontsize=tiny, color="black")
            figure.text(left + width / 2, top_y + 0.023, caption, ha="center", va="bottom",
                        fontsize=small, color="black")

        def _key_row(right, top_y, height, colours, labels):
            """Swatch-plus-label key laid out right to left, blocks sized like `_swatches`.

            Drawn by hand rather than with `figure.legend`, whose handle box is sized from the
            font and came out visibly smaller than the two ramps beside it. Widths are
            estimated from the character count -- exact extents need a renderer, and the row
            only has to not collide.
            """
            block = height * figure.get_figheight() / figure.get_figwidth()
            per_char = 0.45 * tiny / 72.0 / figure.get_figwidth()
            widths = [block + 0.005 + len(text) * per_char for text in labels]
            x = right - sum(widths) - 0.010 * (len(labels) - 1)
            for colour, text, span in zip(colours, labels, widths):
                figure.patches.append(plt.Rectangle(
                    (x, top_y), block, height, transform=figure.transFigure,
                    facecolor=colour, edgecolor="none", clip_on=False))
                figure.text(x + block + 0.005, top_y + height / 2, text, ha="left",
                            va="center", fontsize=tiny, color="black")
                x += span + 0.010

        # One baseline for all three keys, and it is the thing that sets the top margin: the
        # tick captions hang ~0.016 below it and the axis headers start at `top`, so the two
        # numbers move together or a gap opens between them.
        keys_y = 1.0 - 0.055 * grew
        _swatches(0.082, 0.200, keys_y, DOT_COLOURS,
                  ((0.0, r"4$\times$ worse"), (0.5, "No change"), (1.0, r"4$\times$ better")),
                  r"Per run, $\log_2$ ratio")
        # Middle swatch is "not resolved", not "no change": a cream cell is the absence of an
        # answer at twelve seeds, not a measured zero.
        if strip is not None:
            _swatches(0.330, 0.110, keys_y,
                      [strip_colours(i) for i in range(strip_colours.N)],
                      ((0.1, "Worse"), (0.5, "Not resolved"), (0.9, "Better")),
                      r"Per map, effect $/\ \sigma$")
        # No standfirst here: what the panels measure and what they are paired against goes
        # in the LaTeX caption, where it can be read at body-text size instead of 5.9pt.
        # Third key on the same baseline as the other two rather than under the arm labels:
        # the bottom strip it used to occupy is now panel height instead.
        _key_row(0.995, keys_y, 0.018, BAND_COLOURS, [label for _, _, label in OUTCOMES])

        path = save(figure, output)
        plt.close(figure)
        return path


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
                       r"($\theta{=}30$, $h{=}5.0$), "
                       f"$n={paired(table, ARMS[0], metric)[0].size}$")
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


def fig_effect_forest(table, output: Path, arms: list[str] | None = None,
                      per_axis: bool = False) -> Path:
    """Median ratio with 95% bootstrap CI for both metrics, Holm-marked.

    Args:
        table: ``arm -> seed -> row`` from :func:`load_arms`.
        output: Image path.
        arms: Arms to plot; defaults to the five-arm paper selection.
        per_axis: Correct within each axis instead of across the whole table. Set for the
            broad one-factor sweep, where the arms answer separate questions.
    """
    arms = ARMS if arms is None else arms
    metrics = [("occupancy_mse", "occupancy MSE"), ("fourier_ergodic", "Fourier ergodicity")]
    entries, pvalues = [], []
    for metric, label in metrics:
        for arm in arms:
            a, b = paired(table, arm, metric)
            logratio = np.log2(a / b)
            low, high = bootstrap_ci(logratio, np.median)
            entries.append((label, arm, float(np.median(logratio)), low, high,
                            int(np.sum(a < b)), a.size))
            pvalues.append(wilcoxon(a, b).pvalue)
    # The two metrics are two families of their own; correcting them jointly would charge
    # each arm twice for being measured twice.
    significant = []
    for offset in range(0, len(pvalues), len(arms)):
        block = pvalues[offset:offset + len(arms)]
        significant.extend(
            holm_by_axis(table, arms, block) if per_axis else holm(block)
        )

    height = max(2.8, 0.22 * len(arms) + 1.0)
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 2, figsize=(6.9, height), sharex=True)
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
            panel.set_yticklabels([ARM_LABELS.get(r[0][1], r[0][1]) for r in rows])
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


STEP_STAGES = (
    ("rollouts_KT", "Rollouts", r"$K{\times}T$ dynamics + stage cost"),
    ("memory_QP2", "Memory feedback", r"$Q$ scales over $T{\times}P$"),
    ("sample_epsilon", "Noise sampling", r"$K{\times}T{\times}3$ Gaussians"),
    ("attraction_T2", "Stein attraction", r"$T^2$ kernel"),
)

# Keyed by stage, not by rank: a wedge keeps its colour if the timings ever re-sort.
# Same bright, lifted-toward-white register as VIOLIN_COLOURS and the Fig. 3 edge fills,
# but a different hue set -- these four are validated as a *ring*, where the last wedge
# touches the first. Checked with the dataviz palette validator against a white surface:
# lightness band, chroma floor, adjacent-pair CVD and normal-vision separation all pass in
# this cyclic order (worst adjacent pair dE 12.7 tritan / 16.7 protan). Two pairs that look
# fine and are not: periwinkle/lavender is dE 1.1 for deuteranopes, and periwinkle/teal is
# 14.6 even in normal vision -- neither may end up adjacent, which is why the order below
# separates them rather than following the violin palette's hues.
STEP_COLOURS = {
    "rollouts_KT": "#7FADFF",     # periwinkle, the paper's subject colour
    "memory_QP2": "#F0A04B",      # amber
    "sample_epsilon": "#C08CFF",  # lavender -- never adjacent to the periwinkle
    "attraction_T2": "#FF8F87",   # coral
    "_residual": "#B9C0CC",       # grey: unattributed overhead is not a stage
}


def fig_step_budget(report: Path, output: Path) -> Path:
    """Where one control step's milliseconds go, as a donut over the measured stages.

    Reads the JSON written by ``ergodic_control_mppi.experiments.timing``. That module
    times each stage jitted on its own with an explicit ``block_until_ready()`` and
    reports the difference against the jitted whole step as ``residual`` rather than
    absorbing it -- which is the only honest way to attribute cost inside a single fused
    XLA program, and the reason this figure needs a branch.

    The residual is **signed**: XLA routinely fuses stages more cheaply together than
    apart, and a pie cannot draw a negative wedge. So the wedges are always normalised to
    the stages' own sum and the fused total is printed separately in the centre. A positive
    residual gets its own wedge (real overhead the stages miss); a negative one has no
    wedge, because there is no part of the step it corresponds to.
    """
    data = json.loads(Path(report).read_text(encoding="utf-8"))["stages"]
    stages = {name: data["stages"][name]["ms_median"] for name, _, _ in STEP_STAGES}
    total, residual = data["total_ms"], data["residual_ms"]
    if residual > 0:
        stages["_residual"] = residual
        wedges = list(STEP_STAGES) + [("_residual", "Fusion + launch",
                                       "unattributed to any stage")]
    else:
        wedges = list(STEP_STAGES)

    order = sorted(wedges, key=lambda w: -stages[w[0]])
    values = [stages[name] for name, _, _ in order]
    # One qualitative colour per wedge -- a same-hue ramp made same-sized slices hard to
    # tell apart at a glance -- taken per stage rather than per rank, see STEP_COLOURS.
    colours = [STEP_COLOURS[name] for name, _, _ in order]

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.4, 1.95))
        axis.set_facecolor("white")
        figure.patch.set_facecolor("white")
        axis.grid(False)
        axis.set_axis_off()
        share = 100.0 * np.asarray(values) / sum(values)
        # A ring smaller than the axes, with the labels pushed well past its edge:
        # the labels are two lines each, so at the default distance the lower line
        # of a label sitting near the vertical runs into the wedge behind it.
        radius = 0.78
        axis.pie(
            values, colors=colours, startangle=90, counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.8},
            labels=[f"{label}\n{ms:.2f} ms ({pct:.0f}%)"
                    for (_, label, _), ms, pct in zip(order, values, share)],
            labeldistance=1.30, textprops={"fontsize": 6.8, "color": "#23272F"},
            radius=radius,
        )
        # The ring is square but the canvas is not, and `pie` leaves the axes at the data
        # limits its labels need. Pin the vertical extent to the ring instead: with equal
        # aspect the horizontal range then follows the box, which is where the labels go.
        # Asymmetric, because only the upper half carries a label above the ring --
        # a symmetric range pads the bottom with a sixth of the figure in blank paper.
        axis.set_ylim(-0.88, 1.14)
        # Centre carries the one number the reader needs -- the fused step.
        axis.text(0, 0, f"{total:.2f} ms", ha="center", va="center",
                  fontsize=9.5, fontweight="bold", color="#23272F")
        shape = data["shape"]
        figure.text(0.5, 0.995, "Step computation time breakdown",
                    ha="center", va="top", fontsize=8.5, fontweight="bold",
                    color="#23272F")
        figure.text(0.5, 0.90,
                    f"$K{{=}}{shape['K']}$, $T{{=}}{shape['T']}$, "
                    f"$P{{=}}{shape['P']}$, $Q{{=}}{shape['Q']}$ on GPU",
                    ha="center", va="top", fontsize=6.5, color="#5A6472")
        figure.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.03)
        path = save(figure, output)
        plt.close(figure)
    return path


BASELINE_LABELS = {
    "ours": "Ours", "hedac": "HEDAC", "sves": "SVES", "fmec": "FMEC", "smc": "SMC",
}


def load_baselines(*paths: Path) -> dict:
    """Index the baseline archives as ``tier -> method -> (map, seed) -> row``.

    Takes several files because the tiers are run as separate jobs -- the clutter tier is
    hours long and is checkpointed on its own -- and each row already names its tier, so
    merging is just concatenation.
    """
    table: dict = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                table[row["tier"]][row["method"]][(row["map"], int(row["seed"]))] = row
    return {tier: dict(methods) for tier, methods in table.items()}


# Lighter accents on the blue panel: one hue per baseline, held across all three panels so a
# method keeps its colour. Ours is the indigo, deliberately the only cool-neutral among four
# saturated hues, so the subject of the comparison reads as the subject.
# The ggplot hue palette lifted 40% toward white. At full saturation those hues are heavy
# against the pale blue panel -- the green in particular dominated its neighbours -- and a
# large filled area wants a lighter tint than a thin line would. Ours keeps the periwinkle,
# the one cool-neutral among four saturated hues.
VIOLIN_COLOURS = {
    "ours": "#A0C4FF", "hedac": "#66D9DC", "sves": "#F9A2EE",
    "fmec": "#66D688", "smc": "#FBADA7",
}


def _violin(axes, data, colours, labels, *, width=0.78):
    """Filled violins with a seaborn-style inner box.

    Matplotlib draws neither the quartile box nor the median dot, and both are what make a
    violin readable at column width: the silhouette shows the shape, the box shows where the
    mass actually is.
    """
    parts = axes.violinplot(data, positions=range(len(data)), widths=width,
                            showextrema=False, showmedians=False)
    for body, colour in zip(parts["bodies"], colours):
        body.set_facecolor(colour)
        body.set_edgecolor("none")
        body.set_alpha(0.95)
    for slot, values in enumerate(data):
        low, mid, high = np.percentile(values, [25, 50, 75])
        spread = high - low
        whisker = [max(np.min(values), low - 1.5 * spread),
                   min(np.max(values), high + 1.5 * spread)]
        axes.plot([slot, slot], whisker, color="#46506A", linewidth=0.7, zorder=3)
        axes.plot([slot, slot], [low, high], color="#46506A", linewidth=3.2,
                  solid_capstyle="butt", zorder=4)
        axes.plot([slot], [mid], marker="o", markersize=2.6, markerfacecolor="white",
                  markeredgecolor="#46506A", markeredgewidth=0.5, zorder=5)
    axes.set_xticks(range(len(data)))
    axes.set_xticklabels(labels)
    axes.set_xlim(-0.62, len(data) - 0.38)


def _paired_effects(table: dict, tier: str, metric: str):
    """log2(baseline / ours) per shared (map, seed) cell, per baseline."""
    ours = table[tier].get("ours", {})
    out = []
    for method in (m for m in BASELINE_LABELS if m != "ours" and m in table[tier]):
        cells = sorted(set(ours) & set(table[tier][method]))
        if not cells:
            continue
        out.append((method, np.array([
            np.log2(float(table[tier][method][c][metric]) / max(float(ours[c][metric]), 1e-12))
            for c in cells])))
    return out


def fig_baselines_violins(table: dict, directory: Path, metric: str = "fourier_ergodic",
                          formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Three column-width violin panels, written as separate files.

    Separate rather than one tall image so the paper can stack them with ``subfigure`` and
    control the spacing in LaTeX; a single rendered image bakes in whitespace that cannot be
    recovered on the page.

    The split follows the argument rather than the data layout. The first two panels are the
    coverage comparison in each tier, which we lose; the third is the constraint outcome,
    which inverts the ranking. Keeping the third separate is deliberate -- it is a different
    quantity in different units, and overlaying it on a log-ratio axis would misrepresent it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with plt.rc_context({**paper_style("column"), **OUTSIDE_TICKS}):
        for tier, stem, title in (
                ("open", "open", "Paired coverage effect: obstacle-free field"),
                ("clutter", "clutter", "Paired coverage effect: pillar fields")):
            if tier not in table:
                continue
            effects = _paired_effects(table, tier, metric)
            if not effects:
                continue
            figure, axes = plt.subplots(figsize=(3.4, 1.85))
            _violin(axes, [e for _, e in effects],
                    [VIOLIN_COLOURS[m] for m, _ in effects],
                    [BASELINE_LABELS[m] for m, _ in effects])
            axes.axhline(0.0, color="#5A6472", linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
            axes.set_ylabel(r"$\log_2$(baseline / ours)")
            axes.set_title(title)
            # Which way is good, stated on the axis rather than left to the caption.
            axes.text(0.012, 0.955, "Ours better", transform=axes.transAxes, ha="left",
                      va="top", fontsize=6.2, fontweight="bold", color="#3A4357")
            axes.text(0.012, 0.045, "Baseline better", transform=axes.transAxes, ha="left",
                      va="bottom", fontsize=6.2, fontweight="bold", color="#3A4357")
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
            figure.tight_layout(pad=0.25)
            for suffix in formats:
                written.append(save(figure, directory / f"fig_baselines_{stem}.{suffix}"))
            plt.close(figure)

        if "clutter" in table:
            methods = [m for m in BASELINE_LABELS if m in table["clutter"]]
            struck, labels, colours = [], [], []
            for method in methods:
                runs = list(table["clutter"][method].values())
                if not runs:
                    continue
                struck.append(100.0 * sum(1 for r in runs
                                          if float(r.get("collisions", 0) or 0) > 0)
                              / len(runs))
                labels.append(BASELINE_LABELS[method])
                colours.append(VIOLIN_COLOURS[method])
            figure, axes = plt.subplots(figsize=(3.4, 1.75))
            positions = range(len(struck))
            axes.bar(positions, struck, width=0.66, linewidth=0, color=colours, zorder=3)
            for slot, (value, method) in enumerate(zip(struck, methods)):
                mine = method == "ours"
                axes.text(slot, value + max(struck) * 0.035, f"{value:.0f}%", ha="center",
                          va="bottom", fontsize=7.4 if mine else 6.8,
                          fontweight="bold" if mine else "normal",
                          color="#1F2430" if mine else "#2A2F3D", zorder=4)
            axes.set_xticks(list(positions))
            axes.set_xticklabels(labels)
            axes.set_xlim(-0.62, len(struck) - 0.38)
            axes.set_ylim(0, max(struck) * 1.28)
            axes.set_ylabel("Runs with a collision (%)")
            axes.set_title("Obstacle-constraint violations: pillar fields")
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
            figure.tight_layout(pad=0.25)
            for suffix in formats:
                written.append(save(figure, directory / f"fig_baselines_safety.{suffix}"))
            plt.close(figure)
    return written


def fig_baselines(table: dict, output: Path, metric: str = "fourier_ergodic") -> Path:
    """Paired effect of every baseline against ours, one panel per tier.

    Positive is *our* win, in log2 of the ratio, one dot per (map, seed) cell with the
    per-method median and its bootstrap interval over the top. Paired rather than pooled:
    each cell is the same map and the same seed flown by both, which removes the map-to-map
    spread that otherwise swamps a 40-metre workspace.

    The open tier is the honest coverage comparison -- no obstacles, every method at its own
    published formulation. The clutter tier is the one the paper is about, and methods whose
    papers define no obstacle behaviour are marked, because they are running with a term
    they were not published with.

    The bottom strip carries the constraint outcome, and it is not decoration: on the
    spectral metric we lose every comparison, while we are the only method that never
    collides. A figure showing only the top row would report half of the result, and the
    half that flatters the methods that fly closest to the pillars.
    """
    tiers = [t for t in ("open", "clutter") if t in table]
    if not tiers:
        raise ValueError("no tiers in the baseline archive")
    safety = "clutter" in table and any(
        "collisions" in row
        for method in table["clutter"].values() for row in method.values())

    with plt.rc_context(paper_style("double")):
        if safety:
            figure, grid = plt.subplots(
                2, len(tiers), figsize=(6.9, 3.2), sharey="row",
                gridspec_kw={"wspace": 0.06, "hspace": 0.30,
                             "height_ratios": [3.0, 1.0]})
            axes_row, strip_row = grid[0], grid[1]
        else:
            figure, axes_row = plt.subplots(
                1, len(tiers), figsize=(6.9, 2.5), sharey=True,
                gridspec_kw={"wspace": 0.06})
            strip_row = None
        axes_row = np.atleast_1d(axes_row)
        figure.patch.set_facecolor("white")
        for axes, tier in zip(axes_row, tiers):
            axes.set_facecolor("white")
            methods = [m for m in BASELINE_LABELS if m in table[tier] and m != "ours"]
            ours = table[tier].get("ours", {})
            for slot, method in enumerate(methods):
                cells = sorted(set(ours) & set(table[tier][method]))
                if not cells:
                    continue
                # log2(theirs / ours): positive means our metric is the smaller one, and
                # every outcome here is lower-is-better.
                effect = np.array([
                    np.log2(float(table[tier][method][c][metric])
                            / max(float(ours[c][metric]), 1e-12))
                    for c in cells])
                jitter = (np.random.default_rng(0).random(len(effect)) - 0.5) * 0.28
                axes.scatter(slot + jitter, effect, s=4.0, linewidths=0,
                             color="#94c4df", zorder=2)
                low, high = bootstrap_ci(effect)
                axes.plot([slot, slot], [low, high], color="#1a1a1a", linewidth=1.1,
                          zorder=3, solid_capstyle="butt")
                axes.plot([slot], [float(np.median(effect))], marker="o", markersize=3.2,
                          color="#1a1a1a", zorder=4)
                # An added obstacle term is a caveat on the number, so it is marked on the
                # number rather than left to the caption alone.
                if any(int(table[tier][method][c].get("added_avoidance", 0)) for c in cells):
                    axes.text(slot, axes.get_ylim()[0], "*", ha="center", va="bottom",
                              fontsize=7, color="#5A6472")
            axes.axhline(0.0, color="#9a9a9a", linewidth=0.7, zorder=1)
            axes.set_xticks(range(len(methods)))
            axes.set_xticklabels([BASELINE_LABELS[m] for m in methods], fontsize=7)
            axes.set_xlim(-0.6, len(methods) - 0.4)
            axes.set_title("Open field" if tier == "open" else "Pillar fields",
                           fontsize=8, pad=3)
            axes.grid(False)
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
        axes_row[0].set_ylabel(r"$\log_2$(baseline / ours)" "\n" r"ours better $\uparrow$",
                               fontsize=7.5)

        if strip_row is not None:
            strip_row = np.atleast_1d(strip_row)
            for axes, tier in zip(strip_row, tiers):
                axes.set_facecolor("white")
                methods = [m for m in BASELINE_LABELS if m in table[tier]]
                if tier != "clutter":
                    # No obstacles: the constraint cannot be violated, so an empty panel
                    # here would invite reading zero as a result rather than as N/A.
                    axes.text(0.5, 0.5, "no obstacles", ha="center", va="center",
                              transform=axes.transAxes, fontsize=7, color="#8A8A8A")
                    axes.set_xticks([]); axes.set_yticks([])
                    for side in ("top", "right", "left", "bottom"):
                        axes.spines[side].set_visible(False)
                    continue
                fractions, labels = [], []
                for method in methods:
                    runs = list(table[tier][method].values())
                    hit = sum(1 for r in runs if float(r.get("collisions", 0) or 0) > 0)
                    fractions.append(100.0 * hit / max(len(runs), 1))
                    labels.append(BASELINE_LABELS[method])
                positions = range(len(fractions))
                axes.bar(positions, fractions, width=0.62, linewidth=0,
                         color=["#1a1a1a" if m == "ours" else "#c0563a" for m in methods])
                for slot, value in zip(positions, fractions):
                    axes.text(slot, value + 1.5, f"{value:.0f}", ha="center", va="bottom",
                              fontsize=6.5, color="#333333")
                axes.set_xticks(list(positions))
                axes.set_xticklabels(labels, fontsize=7)
                axes.set_xlim(-0.6, len(fractions) - 0.4)
                axes.set_ylim(0, max(fractions) * 1.30 + 2)
                axes.set_ylabel("runs with a\ncollision (%)", fontsize=7)
                axes.grid(False)
                for side in ("top", "right"):
                    axes.spines[side].set_visible(False)

        figure.subplots_adjust(left=0.11, right=0.995, top=0.92,
                               bottom=0.11 if strip_row is not None else 0.12)
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
    # Per-axis families are more permissive than one family over every arm: 0.02 survives
    # a two-arm axis (0.05/2) but not a six-arm table (0.05/6).
    table = {
        arm: {0: {"axis": axis}}
        for arm, axis in [("a1", "a"), ("a2", "a"), ("b1", "b"), ("b2", "b"),
                          ("c1", "c"), ("c2", "c")]
    }
    arms = ["a1", "a2", "b1", "b2", "c1", "c2"]
    pvalues = [0.02, 0.60, 0.60, 0.60, 0.60, 0.60]
    assert holm(pvalues) == [False] * 6
    assert holm_by_axis(table, arms, pvalues) == [True, False, False, False, False, False]
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Both of these now default into the 2026-08-05 quarantine, not the live tree. They are
    # single-map campaigns whose conclusions the nine-map final campaign supersedes; they
    # still render, but pointing them at a live path invites reading them as current.
    parser.add_argument("--ablation", type=Path,
                        default=Path("results/archive/2026-08-05/csv/ablation.csv"))
    # The second UAV ablation campaign: a broad one-factor sweep on a different pillar map.
    # Too many arms for the violin, so it renders as a forest with per-axis Holm.
    parser.add_argument("--sweep-ablation", type=Path,
                        default=Path("results/archive/2026-08-05/csv/ablation_25_pillars.csv"))
    # The nine-map campaign. Rendered when present; it is the figure the paper uses.
    parser.add_argument("--final", type=Path,
                        default=Path("results/uav/ablation_final.csv"))
    parser.add_argument("--campaign-dir", type=Path, default=Path("results/campaign"))
    # Per-stage step timing at the deployment shape, from
    # `python -m ergodic_control_mppi.experiments.timing --config configs/uav_profile.yaml`.
    parser.add_argument("--baselines", type=Path, nargs="*",
                        default=[Path("results/uav/baselines_open.csv"),
                                 Path("results/uav/baselines_clutter.csv")])
    parser.add_argument("--timing", type=Path,
                        default=Path("results/campaign/timing/timing_uav.json"))
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
    if args.sweep_ablation.exists():
        sweep = load_arms(args.sweep_ablation)
        arms = sorted(set(sweep) - {BASELINE})
        written.append(
            fig_effect_forest(sweep, args.output / "fig_sweep_forest.png",
                              arms=arms, per_axis=True)
        )
        written.append(fig_dot_matrix(sweep, args.output / "fig_dot_matrix.png"))
    if args.final.exists():
        final = load_final(args.final)
        written.append(
            fig_final_ablation(final, args.output / "fig_final_ablation.png")
        )
        # The same campaign re-referenced to the median arm, so the shipped profile is drawn
        # as a column instead of being the invisible origin. Same data, different yardstick.
        typical = add_typical_reference(final)
        written.append(
            fig_final_ablation(typical, args.output / "fig_final_ablation_typical.png",
                               reference=TYPICAL)
        )
        # Two panels for a paper that has room for one figure, not three: the per-map strip
        # is the panel whose content already exists as a column in final_report.md.
        written.append(
            fig_final_ablation(typical,
                               args.output / "fig_final_ablation_typical_nostrip.png",
                               reference=TYPICAL, consistency=False)
        )
    baselines = load_baselines(*args.baselines)
    if baselines:
        written.append(fig_baselines(baselines, args.output / "fig_baselines.png"))
    if args.timing.exists():
        written.append(fig_step_budget(args.timing, args.output / "fig_step_budget.png"))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
