"""The figure behind Sec. "guarantees"'s audit: which side of Prop. 3 can be measured.

The proposition's left side is a ball metric -- an integral over balls, hence a *smoothed*
functional -- while its right side is a total variation, a supremum over every Borel set.
That distinction is invisible in the statement and decisive in practice: one is estimable
from a finite trajectory and the other is not, at any horizon we can fly.

A table cannot carry this. The content is a shape -- curves descending and then flattening,
and a noise floor crossing below its signal only at coarse resolution -- and 7 resolutions x
8 horizons is 56 numbers that hide the very thing they demonstrate. Hence two panels sharing
one x axis and one legend.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ergodic_control_mppi.plotting.style import TABLEAU, paper_style, save

#: Fraction of the signal below which the plug-in TV is treated as quotable. Arbitrary, but
#: it has to be *some* line, and it is drawn rather than left implicit.
NOISE_FLOOR = 0.10


def fig_estimability(sweep: dict, output: Path | str) -> Path:
    """Draw TV against horizon, and its split-half noise against the same.

    Args:
        sweep: ``{grid: {"k": [...], "tv": [...], "noise": [...]}}``, noise as a fraction
            of signal. Ordered coarse-to-fine on iteration.
        output: Destination path.

    Returns:
        The written path.
    """
    grids = list(sweep)
    # A single-hue ramp, dark for fine grids and light for coarse: the ordering *is* the
    # variable, so a qualitative cycle would misrepresent it as unordered categories.
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.82, len(grids)))

    with plt.rc_context(rc=paper_style("column")):
        figure, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})
        upper, lower = axes

        for index, grid in enumerate(grids):
            entry = sweep[grid]
            style = dict(color=colours[index], marker="o", markersize=2.2, linewidth=1.0)
            upper.plot(entry["k"], entry["tv"], label=fr"${grid}\times{grid}$", **style)
            lower.plot(entry["k"], np.asarray(entry["noise"]) * 100.0, **style)

        upper.set_ylabel(r"$\mathrm{TV}(\hat\rho_K, p^\star)$")
        upper.set_ylim(0.45, 1.0)

        lower.axhline(NOISE_FLOOR * 100.0, color=TABLEAU[3], linewidth=0.9,
                      linestyle="--", zorder=1)
        # Annotate the line rather than legending it: it is a threshold, not a series, and a
        # second legend entry would compete with the seven that carry the actual variable.
        lower.text(6.0e3, NOISE_FLOOR * 100.0 * 1.18, "10% of signal", color=TABLEAU[3],
                   fontsize=5.6, va="bottom")
        lower.set_ylabel("split-half noise [\\%]")
        lower.set_yscale("log")
        # A log axis defaults to decade labels, and this one spans well under a decade of
        # interest -- without explicit ticks the reader gets a single "10" and cannot read
        # any magnitude off the panel.
        lower.set_yticks([10, 20, 30, 50, 70])
        lower.set_yticklabels(["10", "20", "30", "50", "70"])
        lower.set_ylim(5.5, 85)
        lower.minorticks_off()
        lower.set_xlabel(r"replanning steps $K$")

        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlim(4.2e3, 9.0e5)

        upper.legend(ncol=3, loc="upper right", fontsize=5.2, handlelength=1.1,
                     columnspacing=0.8, handletextpad=0.4, borderpad=0.25)
        return save(figure, output)


def sweep_from_rows(rows: list[dict], halves: list[dict]) -> dict:
    """Fold the two sweep CSVs into the nested form :func:`fig_estimability` wants.

    The noise is the between-seed split-half divided by ``sqrt(2)`` -- two independent
    empirical measures sit about ``sqrt(2)`` times further from each other than either sits
    from the law they share, verified against an i.i.d. control -- and then expressed as a
    fraction of the measured TV.
    """
    import statistics as st

    grids = sorted({int(r["grid"]) for r in rows}, reverse=True)
    horizons = sorted({int(r["k"]) for r in rows})
    sweep = {}
    for grid in grids:
        tv, noise = [], []
        for horizon in horizons:
            measured = st.median(
                float(r["tv"]) for r in rows
                if int(r["grid"]) == grid and int(r["k"]) == horizon
            )
            split = st.median(
                float(r["tv_split"]) for r in halves
                if int(r["grid"]) == grid and int(r["k"]) == horizon
            )
            tv.append(measured)
            noise.append(split / np.sqrt(2.0) / measured)
        sweep[grid] = {"k": horizons, "tv": tv, "noise": noise}
    return sweep
