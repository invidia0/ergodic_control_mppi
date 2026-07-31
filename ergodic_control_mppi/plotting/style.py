"""Shared paper plot style, palette, and figure saving.

One source of truth for what used to be three copies of ``_paper_plot_style()``
(``plotting/literature.py`` plus the deleted ``sensitivity.py``/``ablations.py``,
identical except for font sizes).

The palette -- light blue-grey axes on white, near-white grid -- is unchanged.
Layered on top are the structural conventions of the SciencePlots ``ieee`` style
(inward ticks on all four spines, minor ticks, thin spines, tight bbox with a
small pad); its white background and color cycle are deliberately *not* adopted.
"""

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical palette (Tableau 10 subset). Shared with plotting/literature.py so
# a scenario keeps its color across every figure in the paper.
SCENARIO_COLORS = {
    "unimodal": "#4E79A7",
    "bimodal": "#F28E2B",
    "trimodal": "#59A14F",
    "four_mode": "#E15759",
    "multiscale": "#B07AA1",
}
SCENARIO_LINESTYLES = {
    "unimodal": "-",
    "bimodal": "--",
    "trimodal": "-.",
    "four_mode": ":",
    "multiscale": (0, (3, 1, 1, 1)),
}

PRIMARY = "#4E79A7"
ACCENT = "#E15759"
NEUTRAL = "#98A4BA"

SURFACE = "#DCE2EC"  # axes.facecolor; every ramp below is validated against it


def sequential(hue: str = "Blues", low: float = 0.55, high: float = 0.95) -> LinearSegmentedColormap:
    """Single-hue ramp clipped so its light end still reads against SURFACE.

    The chart surface is light *and* blue, so a ramp running to near-white loses
    its low end into the background -- measured WCAG contrast of the lightest
    step against SURFACE:

        Blues full   #f7fbff   1.25:1   invisible
        Blues @0.35  #a6cee4   1.28:1   invisible
        Blues @0.55  #5ba3d0   2.12:1   ok      <- default here
        Reds  @0.55  #f6583e   2.53:1   ok
        cividis      #f3db42   1.07:1   invisible (and multi-hue)

    2:1 is the floor for the lightest step of an ordinal ramp. Darkening the
    surface does not help -- it lowers contrast with a light-ended ramp further
    (#CBD5E3 gives 1.13:1) -- so the ramp is clipped instead of the surface
    changed. Verified with the dataviz palette validator, not by eye.
    """
    base = plt.get_cmap(hue)
    return LinearSegmentedColormap.from_list(
        f"{hue}_{low:g}_{high:g}", base(np.linspace(low, high, 256))
    )


# Diverging map for "% change vs the shipped default": blue = better (negative),
# white = default, red = worse. Its near-surface midpoint is deliberate -- on a
# diverging scale zero should read as nothing.
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = sequential("Blues")   # magnitude: recency, generic scalars
EXCESS_CMAP = sequential("Reds")        # magnitude: over-coverage / error

# Figure widths in inches. IEEE two-column: 3.35in single, 6.9in double.
FIGSIZES = {
    "column": (3.35, 2.4),
    "double": (6.9, 2.6),
    "poster": (9.2, 5.4),
}

_FONT_SIZES = {
    "column": {"title": 8.5, "label": 8.0, "tick": 7.0, "legend": 7.0},
    "double": {"title": 9.0, "label": 8.5, "tick": 7.5, "legend": 7.5},
    "poster": {"title": 16.0, "label": 16.0, "tick": 16.0, "legend": 16.0},
}


def paper_style(size: str = "column") -> dict[str, Any]:
    """Return rcParams for ``plt.rc_context``.

    Args:
        size: ``"column"`` (single-column figure), ``"double"`` (full width), or
            ``"poster"`` (the large sizes the literature figures already use).

    Raises:
        ValueError: If ``size`` is not a known preset.
    """
    if size not in _FONT_SIZES:
        raise ValueError(f"size must be one of {sorted(_FONT_SIZES)}, got {size!r}")
    fonts = _FONT_SIZES[size]
    return {
        # Typography: serif + STIX math, matching the IEEEtran body text.
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "text.usetex": False,
        # Palette: the blue-ish panel and grey-ish grid, unchanged.
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#DCE2EC",
        "axes.edgecolor": "#98A4BA",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8ECF4",
        "grid.alpha": 0.9,
        "grid.linewidth": 0.75 if size != "poster" else 0.9,
        # SciencePlots "ieee" structure: ticks in, on all four sides, minors on.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#98A4BA",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.6,
        "axes.prop_cycle": mpl.cycler(
            color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759",
                   "#B07AA1", "#76B7B2", "#EDC948", "#9C755F"]
        ),
        "figure.figsize": FIGSIZES[size],
        "axes.titlesize": fonts["title"],
        "axes.titleweight": "bold",
        "axes.labelsize": fonts["label"],
        "xtick.labelsize": fonts["tick"],
        "ytick.labelsize": fonts["tick"],
        "legend.fontsize": fonts["legend"],
        "figure.titlesize": fonts["title"],
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }


def save(figure, path: str | Path, dpi: int = 300) -> Path:
    """Write ``figure`` to ``path``, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    return output
