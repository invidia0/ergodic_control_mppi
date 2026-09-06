"""Figures for the cross-campaign report.

Reads the shipped UAV per-seed CSVs and renders the three figures the report needs:

    fig_paired_arms      per-seed paired effect vs the shipped arm (violin + points)
    fig_effect_forest    median ratio + bootstrap CI, Holm-marked
    fig_dot_matrix       every run as one dot, axes ranked by spread of their medians

    uv run python scripts/report_figures.py --output results/report

Campaign-side stages need `results/campaign/<stage>.csv`; the matrix falls back to the
values quoted in campaign_findings.md and marks them as quoted when the raw CSVs are absent.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

from ergodic_control_mppi.experiments.common import verified_rows
from ergodic_control_mppi.experiments.analyze import bootstrap_ci
from ergodic_control_mppi.plotting.style import (
    ACCENT,
    DIVERGING_CMAP,
    FIGSIZES,
    NEUTRAL,
    OUTSIDE_TICKS,
    PRIMARY,
    paper_style,
    save,
)

BASELINE = "baseline"
# The three necessity rows plus the two bandwidth anchors: the arms the
# mechanism argument stands or falls on.
ARMS = ["memory_off", "plan_off", "release_off", "h_0.47", "h_5.0"]
ARM_LABELS = {
    "h_0.94": "$h{=}0.94$",
    "h_2.35": "$h{=}2.35$",
    "gain_30": r"$k_{\mathcal{M}}{=}30$",
    "gain_60": r"$k_{\mathcal{M}}{=}60$",
    "tau_3": r"$\tau_{\mathcal{M}}{=}3$",
    "tau_11": r"$\tau_{\mathcal{M}}{=}11$",
    **{f"T_{t}": f"$T{{=}}{t}$" for t in (75, 100, 250, 350)},
    "T_150": "$T{=}150$",
    "T_500": "$T{=}500$",
    "T_750": "$T{=}750$",
    "K_125": "$N{=}125$",
    "K_500": "$N{=}500$",
    "balance_0.5": "$a{=}0.5$",
    "gamma_1500": r"$\gamma{=}1500$",
    "gamma_6000": r"$\gamma{=}6000$",
    "penalty_0.1": r"$w_{\rm obs}{\times}0.1$",
    "boundary_0.1": r"$w_{\partial}{\times}0.1$",
    "explore_0": r"$f_{\rm ex}{=}0$",
    # The mechanism axes of the gradient-field campaign.
    "plan_off": "Plan rep. off",
    "plan_3": "$g{=}3$",
    "plan_10": "$g{=}10$",
    "h_0.47": "$h{=}0.47$",
    "h_5.0": "$h{=}5.0$",
    "gain_120": r"$k_{\mathcal{M}}{=}120$",
    "release_off": r"Release off",
    "release_1.75": r"$\sigma^*{=}1.75$",
    "release_3.0": r"$\sigma^*{=}3.0$",
    "ceiling_0": "$c{=}0$",
    "ceiling_0.5": "$c{=}0.5$",
    "service_20": r"$\tau_s{=}20$",
    "service_90": r"$\tau_s{=}90$",
    "transit_1": r"$\beta{=}1$",
    "transit_8": r"$\beta{=}8$",
    "floor_1.0": r"$\varepsilon_s{=}1.0$",
    "alpha_0.9": r"$\alpha{=}0.90$",
    "alpha_0.9": r"$\alpha{=}0.90$",
    "K_1000": "$N{=}1000$",
    "lam_max_1e4": r"$\lambda_{\max}{=}10^4$",
    "refspeed_2.5": "$v{=}2.5$",
    "refspeed_3.0": "$v{=}3.0$",
    "memory_off": "Memory off",
}


def arm_label(arm: str) -> str:
    """Display name for an arm, including the per-width baselines.

    The baselines are keyed by the lane count they were measured at, which is a property of
    the campaign's chunking rather than of the design -- hardcoding the widths here meant a
    re-chunked campaign silently drew raw keys like ``baseline@36`` on an axis.
    """
    if arm.startswith(BASELINE + "@"):
        return "Baseline"
    return ARM_LABELS.get(arm, arm)

# Two lines each, so the widest line rather than the whole phrase has to fit the block --
# an axis block is only 0.14in per arm at 6.9in, and three of these axes hold a single arm.
# The first line names what the knob is *for*, the second names the symbol, so the header
# gives the intuition and the arm labels below give the levels.
AXIS_LABELS = {
    "memory_gain": "Memory\ngain $k_{\\mathcal{M}}$",
    "T": "Horizon\n$T$",
    "alpha": "Control\ncost $\\alpha$",
    "K": "Rollout\nsamples $N$",
    "memory_time": "Memory\ntime $\\tau_{\\mathcal{M}}$",
    "exploration": "Explore\nfraction $f_{\\rm ex}$",
    "fine_bandwidth": "Memory\nbandwidth $h_f$",
    "reference_speed": "Ref.\nspeed $v$",
    "penalty_scale": "Obstacle\npenalty $w_{\\rm obs}$",
    "boundary_scale": "Wall\npenalty $w_{\\partial}$",
    "lam_max": "Temp.\ncap $\\lambda$",
    "plan_gain": "Plan\nrepulsion $g$",
    "release_ratio": "Release\nratio $\\sigma^*$",
    "deficit_ceiling": "Destination\nbend $c$",
    "service_time": "Service\ntime $\\tau_s$",
    "transit_speedup": "Transit\nspeedup $\\beta$",
    "service_floor": "Service\nfloor $\\varepsilon_s$",
    "memory_balance": "Memory\nbalance $a$",
    "track_weight": "Reference\ntracking $\\gamma$",
}

# The five outcomes the sensitivity panel decomposes over, with the transform that makes a
# paired difference meaningful. `occupancy_mse` is deliberately absent: it is ~redundant
# with the Fourier metric, and the joint tick already shows what redundancy costs.
#
# Direction does not matter here. The sensitivity is the *magnitude* of the standardised
# effect, so an arm that halves the dwell and one that doubles it are equally influential --
# which is the right reading for "how much does this knob move the system".
OUTCOMES = (
    ("occupancy_mse", "log", "Occupancy MSE"),
    ("fourier_ergodic", "log", "Fourier ergodicity"),
    ("tours", "raw", "Tours"),
    ("mode_dwell_median_s", "log", "Dwell"),
    ("in_mode_fraction", "logit", "In-mode fraction"),
)
# One colour per outcome, listed **bottom-to-top** in stacking order -- the reverse of how a
# published legend reads down the page, so a palette lifted from one has to be flipped.
# Amber through magenta to navy. Deliberately shares no hue with the diverging red/green
# above it: the panels sit on one x axis but measure different things, and a warmer palette
# here -- Nature's red-orange-to-blue was the alternative -- puts "worse" reds directly under
# the dot panel's "worse" reds and invites reading the two as one scale. Being sequential
# also makes the stack read as ordered layers rather than five unrelated categories.
# Five outcomes, five hues from the project palette, in stacking order. An identity
# encoding, not a magnitude one, so discrete hues say what a single-hue ramp only implied by
# position. Green and orange are a step darker than the palette's own: at full brightness
# both sat above the lightness band and washed out against the white panel.
#
# Validated against that panel -- lightness band, chroma floor, adjacent-pair CVD (worst
# dE 22.4 protan) and normal-vision separation (26.7) all pass. The order is load-bearing:
# red is flanked by blue and purple, the only two hues here it is not confusable with. Beside
# orange it is dE 12.0 even in normal vision, and beside green dE 4.7 for deuteranopes.
BAND_COLOURS = ("#0078FF", "#00C98A", "#FF6B6B", "#F09A4C", "#9B7BFF")


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
    rows = verified_rows(path, ("arm", "obs_num", "map_seed", "seed", "lanes"), legacy=True)
    contexts = {(r.get("steps"), r.get("hardware"), r.get("device"), r.get("jax_version")) for r in rows}
    if len(contexts) > 1:
        raise ValueError(f"{path}: mixed execution configurations")
    hashes = defaultdict(set)
    for row in rows:
        cell = (int(row["obs_num"]), int(row["map_seed"]), int(row["seed"]))
        arm = row["arm"]
        key = f"{arm}@{row['lanes']}" if arm == BASELINE else arm
        hashes[(key, cell[:2])].add(row.get("config_hash", ""))
        if len(hashes[(key, cell[:2])]) > 1:
            raise ValueError(f"{path}: mixed configurations for {key}/{cell[:2]}")
        table[key][cell] = row
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


def _main_width(table) -> str:
    """Return the lane width used by the most arms in a campaign table."""
    widths = [next(iter(rows.values()))["lanes"] for rows in table.values()]
    return Counter(widths).most_common(1)[0][0]


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
    # Main-width arms only: the quarantined set includes a second copy of the
    # baseline, and a median that counted the shipped profile twice would be pulled toward
    # the very column this reference exists to place fairly.
    main_width = _main_width(table)
    members = [a for a in table
               if a != name and next(iter(table[a].values()))["lanes"] == main_width]
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
    if not np.any(matrix):
        return per_outcome, 0.0
    covariance = np.cov(matrix)
    # Ridge on the scale of the diagonal: with 108 cells and 5 outcomes the covariance is
    # well determined, but two near-collinear outcomes can still make it ill-conditioned,
    # and an inflated joint value would understate exactly the redundancy this measures.
    ridge = 1e-8 * float(np.trace(covariance)) / covariance.shape[0]
    joint = float(gradient @ np.linalg.solve(
        covariance + ridge * np.eye(covariance.shape[0]), gradient
    ))
    return per_outcome, float(np.sqrt(max(joint, 0.0)))


def per_map_effects(table, arm: str, metric: str = "occupancy_mse",
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
    pvalues = np.nan_to_num(pvalues, nan=1.0).tolist()
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


# Eleven bins for the dots, five for the strip, sharing one neutral so both panels speak the
# same diverging language: red - orange - yellow - grey - green - cyan - blue, worst to best.
# Multi-hue on each side is the point -- red and green carry "bad" and "good" without
# consulting a legend, and eleven bins need more separation than one hue gives over eleven
# steps.
#
# The named stops are project-palette values and the four bins between them are straight
# midpoints of their neighbours, so the ramp interpolates the palette rather than
# approximating it. One exception: the yellow stop is pushed off the palette's #FFD166,
# which is a golden amber and read as a second orange next to the real one; and the cyan
# stop is a saturated cyan rather than the palette's muted #4DD5E7, which read as teal.
#
# Two things the validator would flag, both accepted. The scale is not single-hue -- that
# rule is for sequential ramps and this is a polarity encoding. And the bins flanking the
# neutral sit under the 2:1 contrast floor, which is what they are for: they mean "almost no
# change", so they have to stay recessive. Darkening them makes a null look like an effect.
# Clear space in data units between the top row of dots and a sideways group header.
HEADER_GAP = 0.4

NEUTRAL_BIN = "#C0D4E9"
DOT_COLOURS = ["#FF6B6B", "#FF8A64", "#FFAA5C", "#FFC861", "#FFE566", NEUTRAL_BIN,
               "#00E89D", "#00DEC6", "#00D5F0", "#00A6F8", "#0078FF"]
STRIP_COLOURS = ["#FF6B6B", "#FFAA5C", NEUTRAL_BIN, "#00D5F0", "#0078FF"]
DOT_EDGES = [-9, -2.0, -1.5, -1.0, -0.5, -0.15, 0.15, 0.5, 1.0, 1.5, 2.0, 9]


def fig_dot_matrix(table, output: Path, metric: str = "occupancy_mse") -> Path:
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
        axes.set_xticklabels([arm_label(a) for _, a in columns],
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


def paired_effect_summary(table, metric: str = "occupancy_mse", *, replicates: int = 10000,
                          seed: int = 20260906) -> dict:
    """Resample maps and paired seeds jointly across all arms, retaining matched widths.

    Returns:
        Pooled and per-map medians with percentile hierarchical bootstrap intervals.
        Only the sampled maps support between-map inference; six maps are not a
        large-map asymptotic justification.
    """
    arms = [a for a in table if not a.startswith(BASELINE) and a != TYPICAL]
    if not arms or replicates < 1:
        raise ValueError("paired effects need arms and positive bootstrap replicates")
    cells = sorted(table[arms[0]])
    maps = sorted({c[:2] for c in cells})
    seeds = sorted({c[2] for c in cells})
    expected = [(d, m, s) for d, m in maps for s in seeds]
    if cells != expected:
        raise ValueError("incomplete map/seed grid")
    effects = []
    for arm in arms:
        reference = baseline_for(table, arm)
        if sorted(table[arm]) != cells or sorted(table[reference]) != cells:
            raise ValueError(f"incomplete matched cells for {arm}")
        a, b, _ = paired_final(table, arm, metric)
        if not np.isfinite([a, b]).all() or np.any(a <= 0) or np.any(b <= 0):
            raise ValueError(f"{arm}: paired log effects need finite positive metrics")
        effects.append(np.log2(b / a).reshape(len(maps), len(seeds)))
    effects = np.asarray(effects)
    rng = np.random.default_rng(seed)
    draw_maps = rng.integers(len(maps), size=(replicates, len(maps), 1))
    draw_seeds = rng.integers(len(seeds), size=(replicates, len(maps), len(seeds)))
    boot = np.median(effects[:, draw_maps, draw_seeds], axis=(-2, -1))
    intervals = np.percentile(boot, [2.5, 97.5], axis=1).T
    return {"metric": metric, "analysis_seed": seed, "replicates": replicates,
            "maps": maps, "seeds": seeds,
            "bundle_hashes": sorted({r.get("bundle_hash", "legacy")
                                      for a in arms for r in table[a].values()}),
            "arms": [{"arm": a, "axis": next(iter(table[a].values()))["axis"],
                      "median": float(np.median(effects[i])),
                      "map_medians": np.median(effects[i], axis=1).tolist(),
                      "interval": intervals[i].tolist()}
                     for i, a in enumerate(arms)]}


def fig_final_ablation(table, output: Path, metric: str = "occupancy_mse") -> Path:
    """Draw two aligned paired-effects panels on the shared journal gray surface."""
    from matplotlib.colors import to_rgb

    summary = paired_effect_summary(table, metric)
    mppi_axes = {"T", "K", "alpha", "exploration", "lam_max", "track_weight",
                 "reference_speed", "penalty_scale", "boundary_scale"}
    groups = [[r for r in summary["arms"] if (r["axis"] in mppi_axes) == is_mppi]
              for is_mppi in (False, True)]
    colours = {10: "#0078FF", 15: "#00C98A", 20: "#9B7BFF", 0: "#0078FF"}
    markers = {10: "o", 15: "s", 20: "^", 0: "o"}
    extent = max(abs(x) for r in summary["arms"] for x in (*r["interval"], *r["map_medians"]))
    extent = max(0.25, extent * 1.08)
    with plt.rc_context({**paper_style("double"), **OUTSIDE_TICKS}):
        figure, axes = plt.subplots(1, 2, figsize=(FIGSIZES["double"][0], 4.5), sharex=True)
        height = max(map(len, groups))
        for ax, rows, title in zip(axes, groups, ("(a) Mechanisms", "(b) MPPI settings")):
            for y, row in enumerate(rows):
                lo, hi = row["interval"]
                ax.hlines(y, lo, hi, color="#30343B", linewidth=0.65, zorder=3)
                for offset, ((density, _), value) in enumerate(zip(summary["maps"], row["map_medians"])):
                    colour = colours[density]
                    pale = 0.4 * np.asarray(to_rgb(colour)) + 0.6
                    ax.scatter(value, y + (offset - (len(summary["maps"]) - 1) / 2) * 0.075,
                               s=11, marker=markers[density], facecolor=pale,
                               edgecolor=colour, linewidth=0.45, zorder=4)
                ax.plot(row["median"], y, "D", color="#252931", markersize=3, zorder=5)
                if y and row["axis"] != rows[y-1]["axis"]:
                    ax.axhline(y-0.5, color="#C5C7CB", linewidth=0.45)
            ax.set_yticks(range(len(rows)), [arm_label(r["arm"]) for r in rows])
            ax.set_ylim(height - 0.4, -0.8)
            ax.set_xlim(-extent, extent)
            ax.grid(False)
            ax.xaxis.grid(True, color="#EDEEF0", linewidth=0.6)
            ax.axvline(0, color="#30343B", linewidth=0.8)
            ax.set_title(title, loc="left")
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel(r"$\log_2$(profile MSE / alternative MSE)")
        for density in sorted({m[0] for m in summary["maps"]}):
            axes[0].scatter([], [], marker=markers[density], color=colours[density],
                            s=12, label=f"{density} pillars")
        figure.legend(loc="upper center", bbox_to_anchor=(0.54, 1.015), ncol=3, frameon=False)
        figure.subplots_adjust(left=0.16, right=0.99, bottom=0.11, top=0.91, wspace=0.58)
        path = save(figure, output)
        plt.close(figure)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
            [f"{arm_label(a)}\n" + (r"$p<0.001$" if p < 0.001 else f"$p={p:.3f}$")
             + ("$^*$" if s else "")
             for a, p, s in zip(ARMS, pvalues, significant)]
        )
        axis.set_ylabel(r"$\log_2$(arm / shipped), occupancy MSE")
        axis.set_title("Paired per-seed effect against the shipped arm "
                       r"($h{=}0.94$, $g{=}6$, $\sigma^*{=}2.24$), "
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
            panel.set_yticklabels([arm_label(r[0][1]) for r in rows])
            panel.set_title(label)
            panel.set_xlabel(r"$\log_2$ ratio vs shipped (95% bootstrap CI)")
            panel.margins(y=0.16)
        axes[0].text(0.02, 0.02, "red = survives Holm", transform=axes[0].transAxes,
                     fontsize=6.5, color=ACCENT)
        figure.tight_layout(pad=0.4, w_pad=1.6)
        path = save(figure, output)
        plt.close(figure)
    return path


STEP_STAGES = (
    ("rollouts_KT", "Rollouts", r"$K{\times}T$ dynamics + stage cost"),
    ("memory_P2", "Memory feedback", r"$T{\times}P$ kernel"),
    ("sample_epsilon", "Noise sampling", r"$K{\times}T{\times}3$ Gaussians"),
    ("plan_T2", "Plan repulsion", r"$T^2$ kernel"),
    ("attraction_T", "Score attraction", r"$T$ pointwise"),
)

# Keyed by stage, not by rank: a wedge keeps its colour if the timings ever re-sort. The
# paper's shared hues, so the donut reads as part of the same family as the violins and the
# sensitivity bands. Colour carries nothing on this chart -- every wedge is directly
# labelled with its name, milliseconds and share -- which is what makes a five-hue ring
# acceptable where the adjacent-pair CVD separation is tighter than the bands' 22.4 dE.
STEP_COLOURS = {
    "rollouts_KT": "#0078FF",     # blue
    "memory_P2": "#00C98A",       # green
    "sample_epsilon": "#FF6B6B",  # red
    "plan_T2": "#F09A4C",         # orange
    "attraction_T": "#9B7BD4",    # violet
    "_residual": "#B9C0CC",       # grey: unattributed overhead is not a stage
}


def fig_step_budget(report: Path, output: Path) -> Path:
    """Render measured stages and synchronized whole-loop timing as separate table rows."""
    from ergodic_control_mppi.plotting.style import SURFACE

    report_data = json.loads(Path(report).read_text(encoding="utf-8"))
    data = report_data["stages"]
    rows = [[label, f"{data['stages'][name]['ms_median']:.3f}"]
            for name, label, _ in STEP_STAGES]
    rows.append(["Fused MPPI step", f"{data['total_ms']:.3f}"])
    for label, values in report_data.get("endtoend", {}).items():
        if isinstance(values, dict) and "ms_per_step" in values:
            rows.append([f"Whole loop: {label.replace('_', ' ')}", f"{values['ms_per_step']:.3f}"])
    with plt.rc_context(paper_style("column")):
        figure, ax = plt.subplots(figsize=(FIGSIZES["column"][0], 0.28 * (len(rows) + 2)))
        ax.set_axis_off()
        table = ax.table(cellText=rows, colLabels=["Measurement", "ms / step"],
                         colWidths=[0.77, 0.23], cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.25)
        for (row, column), cell in table.get_celld().items():
            cell.set_facecolor(SURFACE)
            cell.set_edgecolor("#A9ABB0")
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_text_props(weight="bold")
        figure.tight_layout(pad=0.3)
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
        for row in verified_rows(path, ("tier", "method", "map", "seed"), legacy=True):
            key = (row["map"], int(row["seed"]))
            target = table[row["tier"]][row["method"]]
            if key in target:
                raise ValueError(f"duplicate baseline identity: {row['tier']}/{row['method']}/{key}")
            if target and {r.get("bundle_hash", "") for r in target.values()} != {row.get("bundle_hash", "")}:
                raise ValueError(f"mixed baseline bundles for {row['tier']}/{row['method']}")
            target[key] = row
    return {tier: dict(methods) for tier, methods in table.items()}


# One hue per baseline, held across all three panels so a method keeps its colour, and the
# same five the sensitivity bands use -- a reader who has learned the palette on one figure
# does not relearn it on the next. Ours is the blue, the subject of the comparison.
#
# Four steps for five keys, which is deliberate: ours draws no visible mark in any of the
# three panels -- the violin panels plot the baselines only, and in the safety panel its bar
# is 0% -- so HEDAC can take the blue and the four visible violins are the four hues. Should
# ours ever post a nonzero collision rate, give it its own step then.
#
# Validated against the panel (#E2E3E6): lightness band, chroma floor, adjacent-pair CVD
# (worst dE 8.8 protan) and normal-vision separation (23.4) all pass. Red leads because it is
# confusable with both orange and green, so it needs an end slot with a single neighbour. Contrast against the panel is
# under 3:1, which is only acceptable because every violin sits above its own axis label --
# colour is not carrying identity alone.
#
# The dict order is the plotting order and the checks are on adjacent pairs, so reordering
# the methods means revalidating.
VIOLIN_COLOURS = {
    "ours": "#0078FF", "hedac": "#0078FF", "sves": "#00C98A",
    "fmec": "#FF6B6B", "smc": "#F09A4C",
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
            figure, axes = plt.subplots(figsize=(FIGSIZES["column"][0], 1.85))
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
            figure, axes = plt.subplots(figsize=(FIGSIZES["column"][0], 1.75))
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


# (axis, levels in panel order, label template, figure builder). The level strings are the
# ones `mechanism_captures.py` writes into its filenames, so a missing capture is a missing
# panel rather than a mislabelled one.
MECHANISM_FIGURES = (
    ("plan_gain", ("0", "3", "6", "10"), "$g={}$", "fig_plan_gain"),
    ("release_ratio", ("off", "1.5", "2.24", "3.0"), r"$\sigma^*={}$", "fig_service_gate"),
)


def mechanism_figures(directory: Path, output: Path, seed: int = 43) -> list[Path]:
    """Render the empty-workspace mechanism figures from whatever captures exist.

    Open field, so nothing drawn here is attributable to obstacle avoidance -- which is the
    whole point: these figures make a mechanism claim, and the clutter tier makes the
    constraint claim.
    """
    from ergodic_control_mppi.plotting.trajectories import (
        figure_plan_gain, figure_service_gate, load_captures,
    )

    written = []
    for axis, levels, template, name in MECHANISM_FIGURES:
        paths, titles = [], []
        for level in levels:
            candidate = directory / f"{axis}_{level}_s{seed}.npz"
            if candidate.exists():
                paths.append(candidate)
                titles.append(template.format(level))
        if not paths:
            continue
        captures = load_captures(paths, titles)
        builder = figure_plan_gain if name == "fig_plan_gain" else figure_service_gate
        written.append(builder(captures, output / f"{name}.png"))
    return written


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
    # Empty-workspace mechanism captures, from `scripts/mechanism_captures.py`. Rendered
    # when present; these are the Sec. III-D and III-E figures.
    parser.add_argument("--captures", type=Path,
                        default=Path("results/report/captures"))
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    parser.add_argument("--self-check", action="store_true", help="run assertions and exit")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    written = []
    if args.ablation.exists():
        table = load_arms(args.ablation)
        written.extend([fig_paired_arms(table, args.output / "fig_paired_arms.png"),
                        fig_effect_forest(table, args.output / "fig_effect_forest.png")])
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
    baselines = load_baselines(*args.baselines)
    if baselines:
        written.append(fig_baselines(baselines, args.output / "fig_baselines.png"))
    if args.timing.exists():
        written.append(fig_step_budget(args.timing, args.output / "fig_step_budget.png"))
    written.extend(mechanism_figures(args.captures, args.output))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
