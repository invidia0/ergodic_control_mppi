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

TABLEAU = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759",
           "#B07AA1", "#76B7B2", "#EDC948", "#9C755F"]

# The blue panel of `paper_style`, but with ticks outside and only on the axes that carry a
# scale. The default puts minor ticks inside on all four sides, which on a filled panel reads
# as hatching against the fill.
OUTSIDE_TICKS = {
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": True,
    "axes.titlesize": 7.5,
}

SURFACE = "#E2E3E6"  # axes.facecolor; every ramp below is validated against it


def sequential(hue: str = "Blues", low: float = 0.55, high: float = 0.95) -> LinearSegmentedColormap:
    """Single-hue ramp clipped so its light end still reads against SURFACE.

    The chart surface is light, so a ramp running to near-white loses its low end
    into the background -- measured WCAG contrast of the lightest step against
    SURFACE:

        Blues full   #f7fbff   1.27:1   invisible
        Blues @0.35  #a6cee4   1.30:1   invisible
        Blues @0.55  #5ba3d0   2.15:1   ok      <- default here
        Reds  @0.55  #f6583e   2.56:1   ok
        cividis      #f3db42   1.08:1   invisible (and multi-hue)

    2:1 is the floor for the lightest step of an ordinal ramp. Darkening the
    surface does not help -- it lowers contrast with a light-ended ramp further --
    so the ramp is clipped instead of the surface changed. Verified with the
    dataviz palette validator, not by eye. (These figures moved a fraction when
    the panel went from blue-grey to neutral grey; the ordering did not.)
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

# Neutral occupancy shading at the same peak luminance as Blues@0.60.
OCCUPANCY_CMAP = sequential("Greys", 0.0, 0.53)

# Pillar height ramp for the deployment render: a strong blue at the base, through cyan and
# turquoise, to a bright spring green at the cap. Saturated at both ends rather than fading
# out -- the page behind the render is white, and a pale cap dissolves the top of every
# pillar into the background. The trail crosses this ramp in a pale blue-grey, so path and
# pillars separate by lightness wherever they overlap.
PILLAR_CMAP = LinearSegmentedColormap.from_list(
    "pillar", ["#0078FF", "#0095F0", "#00B2DC", "#00CFBE", "#00E89D"]
)

# Target density on the floor of the deployment render. `Greys` reads correctly but bottoms
# out at pure black, which at these contour levels turns each mode centre into a hard blob
# that outweighs everything standing on it. This runs white to a soft warm black instead, so
# the modes stay clearly the darkest thing on the plane without becoming holes in it. The
# steps carry #424140's warmth the whole way up rather than cooling to a blue-grey, which on
# a plane this pale reads as a colour cast rather than as a neutral.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "carbon", ["#FFFFFF", "#F1F0EF", "#DAD8D6", "#B6B3B1", "#8A8785", "#615F5D", "#424140"]
)

# Executed trail: soft old positions -> near-black newest positions.
TRAIL_CMAP = LinearSegmentedColormap.from_list(
    "trail", ["#D5DAE0", "#73808C", "#101820"]
)

# Deployment render, recast so the result is the salient thing in the scene.
#
# The pillars are context and their height is a depth cue, not a measurement, so they get one
# neutral hue running light at the base to mid-slate at the cap -- enough to read as volumes,
# not enough to compete. `PILLAR_CMAP` above is the earlier blue-through-green ramp; it is
# multi-hue for an axis that carries no quantity, which is the case a single hue exists to
# avoid, and it out-shouted the trail it was drawn behind.
PILLAR_NEUTRAL_CMAP = LinearSegmentedColormap.from_list(
    "pillar_neutral", ["#CDD2D9", "#AEB5BF", "#8C95A2"]
)

# The trail is the measurement, and elapsed time along it is what shows the tour structure a
# coverage controller is judged on. One hue, light (early) to dark (late), so it survives
# greyscale printing and every form of colour-vision deficiency: against the neutral scene it
# is the only chroma on the page, and against itself it separates by lightness alone.
TRAIL_TIME_CMAP = LinearSegmentedColormap.from_list(
    "trail_time", ["#B9D4E8", "#6A9EC6", "#4E79A7", "#2C4F72", "#16304A"]
)

# Figure widths in inches, measured off the class rather than guessed: IEEEtran journal
# reports \columnwidth = 252pt and \textwidth = 516pt (72.27 TeX points/inch). Rendering at
# exactly those widths makes `\includegraphics[width=\linewidth]` a no-op, which is what
# keeps the numbers below honest -- a figure drawn narrower is scaled up on the page and
# every point size in it grows by the same factor.
FIGSIZES = {
    "column": (252 / 72.27, 2.4),
    "double": (516 / 72.27, 2.6),
    "poster": (9.2, 5.4),
}

# Point sizes as they land on the page, given the widths above. The body text is 10pt and
# captions are 8pt, so 8pt labels sit exactly at caption size and read as part of the page
# rather than as something pasted onto it; ticks drop to 7pt (the class's scriptsize) and
# titles rise to 9pt (its small).
#
# "column" and "double" carry the *same* sizes on purpose. They used to differ, which only
# made sense while both were being rescaled by different factors; at 1:1 a double-width
# figure with larger type just looks like a different paper.
_FONT_SIZES = {
    "column": {"title": 9.0, "label": 8.0, "tick": 7.0, "legend": 7.5},
    "double": {"title": 9.0, "label": 8.0, "tick": 7.0, "legend": 7.5},
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
        # Palette: a neutral grey panel with a near-white grid. Grey rather than the
        # blue-grey this used to be, so the panel takes no side among the hues sitting
        # on it -- with a blue "ours" and a blue-ended diverging scale in the same
        # paper, a blue panel put a thumb on the scale.
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#A9ABB0",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EDEEF0",
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
        "legend.edgecolor": "#A9ABB0",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.6,
        "axes.prop_cycle": mpl.cycler(color=TABLEAU),
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


#: Nature research-figure geometry, in millimetres:
#: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
#: One column is 89 mm, two are 183 mm, and nothing may exceed 170 mm of height because the
#: legend has to fit on the page beneath it.
NATURE_WIDTH_MM = {"single": 89.0, "double": 183.0}
NATURE_MAX_HEIGHT_MM = 170.0

#: Arial and Helvetica are what the guide names. Neither ships on Linux, so the stack falls
#: through to their metrically exact clones -- Liberation Sans is Arial's, Nimbus Sans is
#: Helvetica's -- which typeset identically rather than approximately.
NATURE_SANS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]


def nature_style(width: str = "double", height_mm: float = 60.0) -> dict[str, Any]:
    """rcParams for a figure built to the Nature research-figure specification.

    Separate from :func:`paper_style` rather than replacing it: that one matches the
    IEEEtran body text in serif on a grey panel, which is right for the manuscript's own
    format and wrong here. This one is sans-serif at 5--7 pt on white, with text kept as
    text in the PDF (``fonttype`` 42) because the guide asks for editable layers and
    forbids outlined type.

    Args:
        width: ``"single"`` (89 mm) or ``"double"`` (183 mm).
        height_mm: Figure height; must not exceed :data:`NATURE_MAX_HEIGHT_MM`.

    Returns:
        rcParams for ``plt.rc_context``.

    Raises:
        ValueError: If ``width`` is unknown or the height exceeds the page allowance.
    """
    if width not in NATURE_WIDTH_MM:
        raise ValueError(f"width must be one of {sorted(NATURE_WIDTH_MM)}, got {width!r}")
    if height_mm > NATURE_MAX_HEIGHT_MM:
        raise ValueError(
            f"height {height_mm} mm exceeds the {NATURE_MAX_HEIGHT_MM} mm page allowance"
        )
    return {
        "font.family": "sans-serif",
        "font.sans-serif": NATURE_SANS,
        # stixsans, not dejavusans: the sans-serif requirement applies to the maths too, and
        # DejaVu's mathtext has no \mathcal, which the field-gain symbols need.
        "mathtext.fontset": "stixsans",
        "text.usetex": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.5,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "lines.linewidth": 0.8,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "axes.titlesize": 7,
        "axes.labelsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 5.5,
        "figure.titlesize": 7,
        "figure.figsize": (NATURE_WIDTH_MM[width] / 25.4, height_mm / 25.4),
        # Keep glyphs as text so the exported PDF has editable layers.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "#FFFFFF",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }


def save(figure, path: str | Path, dpi: int = 300) -> Path:
    """Write ``figure`` to ``path``, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    return output
