"""Explanatory figures for Sec. III-C, Fading-Memory Coverage Feedback.

Every field, weight and bandwidth drawn here is produced by calling the
controller's own functions -- ``kernel``, ``kernel_gradient``, ``smoothed``,
``pdf``, ``kde_repulsion`` -- on a memory buffer taken from a real run. Nothing
is hand-drawn, so a figure cannot drift away from the implementation; the
composition of those calls is pinned against ``memory_flow`` itself in
``tests/test_plotting.py``.

    python -m ergodic_control_mppi.plotting.mechanism --output-dir theory/pictures

Three figures, in the order Sec. III-C introduces them:

    fig_occupancy       o_t^h and the fading trail that produced it
    fig_excess_focus    the relative excess e_{t,i}^h, read off one memory point
    fig_memory_fields   the recency and over-coverage fields

The source run is cached; delete the .npz to regenerate it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.ticker import MultipleLocator

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.field import (
    ACTIVITY_FLOOR,
    kernel,
    kernel_gradient,
    pdf,
    smoothed,
    kde_repulsion,
)
from ergodic_control_mppi.plotting.style import (
        TRAIL_CMAP,
    ACCENT,
    NEUTRAL,
    OUTSIDE_TICKS,
    PRIMARY,
    paper_style,
    save,
    sequential,
)

MECHANISM_OCCUPANCY_CMAP = sequential("Greys", 0.05, 0.67)

# One hue for the target, everywhere it appears: the paper blue #0078FF that the
# deployment pillars (style.PILLAR_CMAP) and the ablation figures are keyed to.
# Two steps of it, because a line and a filled area need different weights of the
# same colour, and one darker text step:
#
#   TARGET_LINE  contours on the map, and the target's edge in a strip. Drawn
#     over the occupancy ramp, so it has to clear that ramp's whole range and not
#     just its ends -- the ramp runs #f9f9f9 -> #676767, and any mid-tone (the old
#     #356FA8 read 1.08:1, plain #0078FF reads 1.03:1) disappears somewhere along
#     it. #004899 is the paper hue taken below the ramp's dark end: 1.55:1 at the
#     worst point, 8.8:1 on white. No halo -- see the marks test.
#   TARGET_FILL  the strip areas, which sit on white, not on the ramp. Lifted
#     rather than darkened, because a large filled area wants a lighter tint than
#     a line does, and bright enough (2.32:1 on white, against #A0C4FF's 1.77:1)
#     that the grey standing clear of it is legible at 8pt.
#
# Occupancy keeps the grey it already has on the map, so a strip needs no legend.
# Red is left free to mean one thing only: the excess.
TARGET_LINE = "#004899"
TARGET_FILL = "#66AEFF"
TARGET_COLOR = TARGET_LINE  # target labels and curves outside the map

# The target on a map, filled rather than drawn as rings. Both densities being
# compared are then the same kind of object -- two fields, one grey and one blue --
# instead of one field and a ladder of lines around it, which read as a boundary
# of something rather than as an amount of something.
#
# The ramp carries its own alpha: transparent where the target is negligible, so
# the occupancy underneath is never tinted by a field that is not there, and only
# partly opaque at the modes, so the grey still shows through where the two
# overlap -- which is the whole comparison the figure is about.
# The level boundaries of that fill, drawn as hairlines on top: light, so they
# sit inside the blue as structure rather than on it as a second mark, and they
# fade out on their own where the fill does -- a line this pale over the bare grey
# is not there, which is the right answer in a region with no target in it.
TARGET_EDGE = "#DCEBFF"

TARGET_CMAP = LinearSegmentedColormap.from_list(
    "target_fill",
    [(*to_rgb(TARGET_FILL), 0.0), (*to_rgb(TARGET_FILL), 0.26),
     (*to_rgb(TARGET_LINE), 0.38)],
)
OCCUPANCY_FILL = MECHANISM_OCCUPANCY_CMAP(0.42)
OCCUPANCY_LABEL = "#4B5563"  # darker companion of OCCUPANCY_FILL, for text on white

# Level rules on the occupancy fill, the counterpart of TARGET_EDGE on the target.
# Dark where the target's are light, because each field's rules belong to its own
# colour and the two fills overlap: pale rules on both would be one ladder over
# two densities. Like TARGET_EDGE they fade where their own fill runs out -- here
# into the ridge core, which the fill has already said is the maximum.
OCCUPANCY_EDGE = "#5B6572"

# Construction ink: the kernel circle and its spoke, the profile cuts, the focus
# point, the locator's crop box. One colour for "this is drawing, not data", and
# dark enough (3.16:1) to clear the field ramp wherever it lands.
CONSTRUCTION = "#1F2937"

# The trail's own dark end, TRAIL_CMAP(1.0). Used wherever a path is drawn flat --
# pure black is a value the trail ramp itself is tested for not reaching.
INK = "#101820"

# The vehicle: a white dot with an ink ring, not the old `tab:red`. Red now belongs
# to the excess, and no mid-tone hue survives the occupancy ramp -- it clears the
# ramp's ends and vanishes into its middle. White is the one value that reads at
# every step of it (5.65:1 at the dark end), and a white marker on a dark ring is
# already the house marker for a point on top of a dense mark (the violin medians
# in scripts/report_figures.py).
ROBOT_COLOR = "#FFFFFF"
ROBOT_EDGE = INK
FLOW_COLOR = "#A12F3B"
RECENCY_COLOR = "#26747A"

# Tick spacing [m] for a zoomed workspace crop, shared by x and y so the two axes
# of a square crop carry the same number of labels.
ZOOM_TICK_STEP = 2.0

# Enough steps for the buffer to hold a representative mid-run trail rather than
# the initial transient: the shipped tau_M = 10 s gives P = 1500 at dt = 0.02.
SOURCE_STEPS = 4000


def _source(config_path: str, cache: Path, device: str, steps: int) -> dict[str, np.ndarray]:
    """Load or produce the run whose final memory buffer the figures use."""
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}

    import yaml

    from ergodic_control_mppi.simulation import run_simulation

    # Patch only the step count; everything else is the shipped config.
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data["steps"] = steps
    scratch = cache.parent / "mechanism_source_config.yaml"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    config = load_config(scratch)

    result = run_simulation(config, device)
    length = config.controller.mppi.memory_length
    path = result.paths[:, 0, :2].astype(np.float32)

    # Illustration instant. The buffer only ever holds the last P steps (30 s at
    # the shipped tau_M), so the *end* of a run is an arbitrary slice of it -- and
    # often a single dwell, which shows the mechanism at its least interesting.
    # Pick instead the instant whose buffer is most spatially spread, i.e. one
    # containing a dwell and a transit. This selects the illustration, not the
    # mechanism: every quantity below is still computed from a genuine buffer.
    cell = 2.0
    best_end, best_spread = length, -1
    for end in range(length, len(path) + 1, max(length // 10, 1)):
        window = path[end - length:end]
        occupied = len(set(map(tuple, np.floor(window / cell).astype(np.int64))))
        if occupied > best_spread:
            best_end, best_spread = end, occupied

    arrays = {
        "memory": path[best_end - length:best_end],
        "path": path[:best_end],
        "surrogate": result.surrogates[0].astype(np.float32),
        "position": path[best_end - 1],
        "buffer_end": np.asarray(best_end, dtype=np.int64),
        "buffer_cells": np.asarray(best_spread, dtype=np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def _context(config, arrays):
    """Per-step derived quantities, exactly as core.py builds them."""
    params = config.controller
    field, workspace, gmm = params.field, params.workspace, params.gmm
    memory = jnp.asarray(arrays["memory"])

    ages = jnp.arange(memory.shape[0])[::-1]
    recency = field.memory_decay ** ages
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )
    # The median rollout: the query set the field is evaluated on, and the plan the
    # self-repulsion term repels from. There is no estimated bandwidth any more -- `h` is
    # a design constant, so the panels draw the scale the controller actually uses.
    particles = jnp.asarray(arrays["surrogate"])
    bandwidth = float(field.fine_bandwidth)
    # Slice through whichever mode the buffer actually occupies, so panel (b)'s
    # three curves are comparable instead of all sitting near zero.
    means = np.asarray(gmm.means)
    buffer_np = np.asarray(arrays["memory"])
    occupancy_per_mode = [
        float(np.mean(np.linalg.norm(buffer_np - mean, axis=-1) < 3.0)) for mean in means
    ]
    return {
        "params": params, "field": field, "gmm": gmm, "workspace": workspace,
        "memory": memory, "recency": recency, "density_floor": density_floor,
        "particles": particles, "bandwidth": bandwidth,
        "full_path": np.asarray(arrays["path"]),
        "position": np.asarray(arrays["position"]),
        "slice_y": float(means[int(np.argmax(occupancy_per_mode))][1]),
        "limits_x": tuple(map(float, workspace.x_limits)),
        "limits_y": tuple(map(float, workspace.y_limits)),
    }


def _field_at(ctx, points, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupancy, scale-matched target and relative excess at arbitrary points.

    Term for term eq. (occupancy_density), (smoothed_target) and
    (relative_excess), which is what ``field.py:memory_weights`` evaluates at the memory
    points. Blocked over the query axis: the full ``(N, P)`` kernel matrix is
    ~200 MB at N = 180^2, P = 1500.
    """
    memory, recency = ctx["memory"], ctx["recency"]
    query = jnp.asarray(points, dtype=jnp.float32).reshape(-1, 2)
    blocks = max(1, int(np.ceil(query.shape[0] / 4096)))
    occupancy = np.concatenate([
        np.asarray(kernel(chunk[:, None, :], memory[None, :, :], bandwidth) @ recency)
        for chunk in jnp.array_split(query, blocks)
    ]) / float(jnp.sum(recency) * jnp.pi * bandwidth)
    target = np.asarray(pdf(query, smoothed(ctx["gmm"], bandwidth)))
    excess = np.maximum(occupancy - target, 0.0) / (target + ctx["density_floor"])
    return occupancy, target, excess


def _excess(ctx, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``_field_at`` on the memory buffer, plus the recency-mean excess S_t^h."""
    occupancy, target, excess = _field_at(ctx, ctx["memory"], bandwidth)
    recency = np.asarray(ctx["recency"])
    return occupancy, target, excess, float(recency @ excess / recency.sum())


def _grid(ctx, n: int = 180, limits=None):
    """Regular query grid over the workspace (or a sub-box), for the field maps."""
    limits_x, limits_y = limits or (ctx["limits_x"], ctx["limits_y"])
    grid_x, grid_y = np.meshgrid(
        np.linspace(*limits_x, n), np.linspace(*limits_y, n)
    )
    return grid_x, grid_y, np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)


def _rho(ctx, points, masses, bandwidth: float) -> np.ndarray:
    """Memory repulsion of eq. (memory_repulsion), as the controller computes it.

    ``kde_repulsion`` divides by the mass sum, so passing the raw recency gives
    exactly rho(.; q^rec).
    """
    return np.asarray(
        kde_repulsion(
            jnp.asarray(points, dtype=jnp.float32).reshape(-1, 2),
            ctx["memory"], jnp.asarray(masses), bandwidth,
        )
    )


def _rho_excess(ctx, points, bandwidth: float) -> np.ndarray:
    """The over-coverage field, gate included -- ``field.py:memory_weights`` verbatim."""
    _, _, excess, activity = _excess(ctx, bandwidth)
    gate = activity / (activity + ACTIVITY_FLOOR)
    return gate * _rho(ctx, points, np.asarray(ctx["recency"]) * excess, bandwidth)


def _trail(axis, ctx, linewidth: float = 2.0, zorder: int = 4, flat: str | None = None):
    """The executed trail, fading soft blue-grey to near-black, plus the robot.

    ``flat`` draws the path in one solid colour instead of the recency ramp, for
    panels that show the path as geometry only -- there the ramp's light end
    reads as a faded line rather than as an encoding of anything.
    """
    memory = np.asarray(ctx["memory"])
    collection = None
    if flat is not None:
        axis.plot(memory[:, 0], memory[:, 1], color=flat, linewidth=linewidth,
                  solid_capstyle="round", zorder=zorder)
    else:
        recency = np.asarray(ctx["recency"])
        axis.plot(memory[:, 0], memory[:, 1], color="#20252B", alpha=0.65,
                  linewidth=linewidth + 0.7, solid_capstyle="round", zorder=zorder - 1)
        collection = LineCollection(
            np.stack((memory[:-1], memory[1:]), axis=1),
            cmap=TRAIL_CMAP, linewidth=linewidth, capstyle="round", zorder=zorder,
        )
        collection.set_array(recency[1:])
        collection.set_clim(0.0, 1.0)
        axis.add_collection(collection)
    axis.plot(*ctx["position"], marker="o", markersize=6.5, color=ROBOT_COLOR,
              markeredgecolor=ROBOT_EDGE, markeredgewidth=1.0, zorder=zorder + 1)
    return collection


def _target_contours(axis, ctx, levels=10, cmap=TARGET_CMAP, limits=None, gmm=None,
                     rule: float = 0.3, rule_color: str = TARGET_EDGE) -> None:
    """Draw the target density as a filled blue field, under the trajectory marks.

    Filled rather than a ring ladder: the figure sets two densities against each
    other, so both should be fields. The levels are still drawn, as hairlines on
    the boundaries of the bands they belong to -- a fill alone says how much, and
    the rules say how much per step, which is what makes the falloff readable
    rather than merely visible.
    """
    grid_x, grid_y, points = _grid(ctx, n=160, limits=limits)
    density = np.asarray(
        pdf(jnp.asarray(points, jnp.float32), gmm if gmm is not None else ctx["gmm"])
    )
    field = density.reshape(grid_x.shape)
    # No `set_edgecolor("face")` here, unlike the opaque field in `_field_map`.
    # On a translucent fill the stroke composites a second time over its own band
    # and over its neighbour, and every boundary comes back as a heavy ring -- the
    # exact artefact the stroke removes when the fill is opaque. The rules below
    # are that boundary drawn deliberately, at a width and a colour chosen for it.
    filled = axis.contourf(grid_x, grid_y, field, levels=levels, cmap=cmap, zorder=1)
    if rule:
        axis.contour(grid_x, grid_y, field, levels=filled.levels[1:],
                     colors=rule_color, linewidths=rule, alpha=0.75, zorder=1)


def _map_axes(axis, ctx, *, ylabel: bool = False, limits=None) -> None:
    """Shared geometry for every workspace panel, so panels align exactly."""
    limits_x, limits_y = limits or (ctx["limits_x"], ctx["limits_y"])
    axis.set_xlim(*limits_x)
    axis.set_ylim(*limits_y)
    axis.set_aspect("equal")
    if limits is None:
        axis.set_xticks([-10, -5, 0, 5, 10])
        axis.set_yticks([-10, -5, 0, 5, 10])
    else:
        # Independent auto-locators can pick different step sizes for two
        # equal-span axes depending on where the numeric bounds happen to fall,
        # so a zoom box gets one explicit step shared by x and y instead.
        axis.xaxis.set_major_locator(MultipleLocator(ZOOM_TICK_STEP))
        axis.yaxis.set_major_locator(MultipleLocator(ZOOM_TICK_STEP))
    axis.set_xlabel(r"$x$ [m]", labelpad=1)
    if ylabel:
        axis.set_ylabel(r"$y$ [m]", labelpad=1)
    elif limits is None:
        # Panels sharing the workspace crop drop the repeated labels; a zoom keeps
        # its own, since its range is not the one the neighbour already stated.
        axis.set_yticklabels([])


def _modes(axis, ctx, label: str | None = None) -> None:
    """Mark the target modes identically in every panel."""
    means = np.asarray(ctx["params"].gmm.means)
    axis.plot(means[:, 0], means[:, 1], marker="x", linestyle="none",
              color="#33415C", markersize=3.4, markeredgewidth=0.9,
              zorder=6, label=label)


def _inline_colorbar(figure, axis, mappable, label: str, *, backing: bool = False,
                     corner: str = "lower") -> None:
    """Compact horizontal colorbar *inside* the panel.

    An external colorbar steals ~15% of a panel's width and, when only some
    panels carry one, leaves their titles at different heights. Putting it
    inside keeps every map the same size.

    ``backing`` lays a translucent white pad underneath, needed whenever the bar
    sits over a filled field map: the ink used for its label and ticks reads
    1.33:1 against the dark end of the ramp.
    """
    bottom = 0.075 if corner == "lower" else 0.80
    if backing:
        pad = axis.inset_axes((0.035, bottom - 0.04, 0.50, 0.185), zorder=7)
        pad.set_facecolor("#FFFFFF")
        pad.patch.set_alpha(0.84)
        pad.set_xticks([])
        pad.set_yticks([])
        for spine in pad.spines.values():
            spine.set_visible(False)
    cax = axis.inset_axes((0.075, bottom, 0.38, 0.032), zorder=8)
    bar = figure.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.outline.set_linewidth(0.4)
    bar.outline.set_edgecolor("#5C6B87")
    cax.xaxis.set_major_locator(plt.MaxNLocator(2))
    cax.minorticks_off()
    cax.tick_params(labelsize=5.6, length=1.4, pad=1, colors="#33415C")
    cax.set_title(label, fontsize=6.0, pad=2, color="#33415C")


def _field_map(axis, grid_x, grid_y, values, *, cmap=MECHANISM_OCCUPANCY_CMAP,
               norm=None, vmax=None, levels: int = 24, rules: int = 0,
               rule_color: str = OCCUPANCY_EDGE):
    """Filled field contours under everything else.

    ``rules`` draws that many level lines over the fill, as ``_target_contours``
    does for the target. They are spaced evenly *through the norm* rather than
    evenly in value: the fill's own bands are, so value-spaced rules would visibly
    disagree with the shading they are drawn on.
    """
    if norm is None:
        norm = PowerNorm(0.5, vmin=0.0, vmax=vmax if vmax is not None else values.max())
    # A grid over a filled field is noise, and axisbelow puts it above the field.
    axis.grid(False)
    # ``levels`` is a resolution knob, not a style one. 24 is right for a
    # workspace-wide map, where the field changes fast enough that the bands are
    # not separately visible; a zoomed crop of a smooth region needs far more, or
    # the terraces read as a topographic staircase belonging to the drawing rather
    # than to the density. Raising it also lightens the low end, since the lowest
    # band then spans a much smaller range -- fine on a crop, but it would wash out
    # the halo the workspace map exists to show.
    filled = axis.contourf(
        grid_x, grid_y, values.reshape(grid_x.shape), levels=levels,
        cmap=cmap, norm=norm, zorder=0,
    )
    if rules:
        axis.contour(
            grid_x, grid_y, values.reshape(grid_x.shape),
            levels=norm.inverse(np.linspace(0.0, 1.0, rules + 1)[1:-1]),
            colors=rule_color, linewidths=0.3, alpha=0.55, zorder=0,
        )
    # Each level is a separate polygon, so at 64 of them the antialiased seams
    # between neighbours show as white hairlines across the field. Stroking every
    # band in its own fill colour closes them.
    filled.set_edgecolor("face")
    return filled


def figure_occupancy(ctx, output: Path) -> Path:
    """Fig. 1 -- the occupancy proxy the memory maintains, and the trail behind it.

    One map, because there is one object: eq. (occupancy_density) at the deployed
    scale, with the buffer that generated it drawn on top fading from soft
    blue-grey (oldest) to near-black (newest), ending at the blue robot marker.
    PowerNorm(1/2) rather than a linear norm -- occupancy is a sum of P narrow
    kernels, so linearly the halo around the track is invisible and only a thin
    ridge survives.
    """
    bandwidth = float(ctx["field"].fine_bandwidth)
    grid_x, grid_y, points = _grid(ctx, n=220)
    occupancy, _, _ = _field_at(ctx, points, bandwidth)

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
        mesh = _field_map(axis, grid_x, grid_y, occupancy)
        _target_contours(axis, ctx)
        _trail(axis, ctx)
        _modes(axis, ctx)
        _map_axes(axis, ctx, ylabel=True)
        axis.set_title(rf"occupancy $o^{{h}}_t$,  $h={bandwidth:.2f}$")
        _inline_colorbar(figure, axis, mesh, r"$o^{h_c}_t(\mathbf{z})$  [m$^{-2}$]")
        path = save(figure, output)
        plt.close(figure)
    return path


def _focus_point(ctx, bandwidth: float) -> tuple[int, np.ndarray]:
    """The most over-served memory point *inside a mode*, and the excess vector.

    Restricted to Mahalanobis radius < 2 of the nearest mode on purpose. The
    unrestricted argmax lands on a transit point where p*_h ~ 0 and the excess
    collapses to o/eps_p -- that illustrates the density floor, not the coverage
    mechanism the figure is about.
    """
    memory = np.asarray(ctx["memory"])
    _, _, excess = _field_at(ctx, memory, bandwidth)
    means = np.asarray(ctx["gmm"].means)
    delta = memory[:, None, :] - means[None, :, :]
    radius = np.sqrt(
        np.einsum("pmi,mij,pmj->pm", delta, np.asarray(ctx["gmm"].covariance_inverse), delta)
    ).min(axis=1)
    inside = radius < 2.0
    return int(np.argmax(np.where(inside, excess, -np.inf))), excess


def _edge_profile(axis, coordinate, fields, vmax, excess_max, *, vertical: bool,
                  depth: float = 0.24, labels: tuple[str, ...] = ()) -> None:
    """One chromeless o-vs-p* cut, drawn just outside an edge of a map panel.

    The comparison eq. (relative_excess) makes is local, so it does not need
    axes of its own: two filled curves on the border say it in a fifth of the
    height a second panel costs. ``o`` is filled first and ``p*`` over it, so
    the sliver of occupancy colour left uncovered *is* the positive part -- the
    excess is shown rather than shaded and asserted.

    Over the two fills goes eq. (relative_excess) itself, the ratio the fills
    only imply, in bold red. It is dimensionless, so it cannot share the density
    axis; it is drawn against ``excess_max`` instead. Both strips take ``vmax``
    and ``excess_max`` from the caller rather than from their own cut, so the
    three curves are comparable between the two edges as well as within one.
    """
    occupancy, target, excess = fields
    # Scaled to sit just under the fills' ceiling, on its own common scale --
    # a strip carries no axis, so the only thing a height means here is a
    # comparison with the same curve on the other edge.
    excess_curve = 0.88 * vmax * excess / excess_max
    # A hair of clearance off the panel: the target fill is thin over most of a
    # cut, and flush against the spine it reads as a coloured rule on the map
    # rather than as the low end of a density.
    gap = 0.02
    strip = axis.inset_axes((1.0 + gap, 0.0, depth, 1.0) if vertical
                            else (0.0, 1.0 + gap, 1.0, depth))
    fill = strip.fill_betweenx if vertical else strip.fill_between
    fill(coordinate, 0.0, occupancy, color=OCCUPANCY_FILL, linewidth=0, alpha=0.9)
    fill(coordinate, 0.0, target, color=TARGET_FILL, linewidth=0, alpha=0.9)
    # Each fill also gets its own crest as a line. The claim the strip makes is
    # "where the grey stands clear of the blue", i.e. the gap between two curves --
    # as two washes alone that gap is bounded by the edge of a tint, and at 8pt the
    # thin tail of p* reads as nothing at all rather than as a small number.
    for values, colour, width in ((occupancy, OCCUPANCY_LABEL, 0.55),
                                  (target, TARGET_LINE, 0.8)):
        crest = (values, coordinate) if vertical else (coordinate, values)
        strip.plot(*crest, color=colour, linewidth=width, solid_capstyle="round",
                   zorder=3)
    line = (excess_curve, coordinate) if vertical else (coordinate, excess_curve)
    strip.plot(*line, color=ACCENT, linewidth=1.4, solid_capstyle="round", zorder=4)
    span, density = (strip.set_ylim, strip.set_xlim) if vertical else (strip.set_xlim, strip.set_ylim)
    span(coordinate[0], coordinate[-1])
    density(0.0, 1.05 * vmax)

    # No frame, no ticks, no background: the strip is a shape on the page, and
    # the panel it sits on already carries the position axis it shares.
    strip.patch.set_visible(False)
    strip.grid(False)
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)

    # Every curve is on both strips, but each label is written once: the density
    # pair on the top edge, the excess on the right. Three labels on one strip
    # crowd each other -- all three curves peak within a metre or two of the cut.
    # (where to put it, where along the cut, colour, text, offset, ha, va).
    # Each label goes in open white above its own curve. o cannot use its crest --
    # that is exactly where the excess curve crosses over it -- so it is anchored
    # down the shoulder, at the widest gap between the two curves, where the space
    # above the fill is clear all the way to the ceiling.
    gap_at = np.argmax(occupancy - excess_curve)
    specs = {
        "o": (occupancy, gap_at, OCCUPANCY_LABEL,
              r"$o^{h_c}$", (0, 3), "center", "bottom"),
        "p": (target, np.argmax(target), TARGET_COLOR,
              r"$p^\star_{h_c}$", (3, 3), "left", "bottom"),
        "e": (excess_curve, np.argmax(excess_curve), ACCENT,
              r"$e^{h_c}$", (3, 3), "left", "bottom"),
    }
    for name in labels:
        values, at, colour, text, offset, align, vertical_align = specs[name]
        peak = int(at)
        position = (values[peak], coordinate[peak]) if vertical else (coordinate[peak], values[peak])
        strip.annotate(text, xy=position, xytext=offset, textcoords="offset points",
                       fontsize=8.0, color=colour, ha=align, va=vertical_align,
                       clip_on=False, zorder=6)


def figure_excess_focus(ctx, output: Path) -> Path:
    """Fig. 3 -- the relative excess, read off one memory point.

    One panel: the neighbourhood around the most over-served memory point --
    occupancy ramp, target contours, kernel radius, executed trail in flat ink
    over the ramp -- with a small locator thumbnail (the target's modes and the
    trail, boxed at the crop) showing where in Omega it sits.

    Three inks, one meaning each: blue is the target wherever it appears, grey is
    the occupancy, red is the excess and nothing else. Everything that is drawing
    rather than data -- the cuts, the kernel circle and its spoke, the focus point,
    the crop box -- is in one construction ink.

    The arithmetic of eq. (relative_excess) is on the borders rather than in a
    second panel: ``_edge_profile`` puts o and p* along the two marked cuts
    through the point on the top and right edges, filled and overlapping, and
    the excess is the part of the occupancy fill that the target fill does not
    cover. The crop is centred exactly on the point, so both profiles peak at
    the middle of their edge.
    """
    bandwidth = float(ctx["field"].fine_bandwidth)
    index, excess = _focus_point(ctx, bandwidth)
    memory = np.asarray(ctx["memory"])
    focus = memory[index]
    # 2.5 tick steps of half-span: wide enough that both cuts run out to where
    # o and p* have decayed, so each profile shows a shape rather than a slab.
    half = 2.5 * ZOOM_TICK_STEP
    # Centred on the point, with no snap to the tick grid: the edge profiles are
    # cuts through it, and a snap of up to half a step would leave their peaks
    # visibly off-centre. The clip keeps the box inside Omega -- outside it the
    # fields are defined but meaningless -- and should never bite, since the
    # focus point lies inside a mode, well clear of every wall. Off-centre
    # profiles are the symptom that it did.
    box = tuple(
        (centre - half, centre + half)
        for centre in (float(np.clip(c, lo + half, hi - half))
                       for c, (lo, hi) in zip(focus, (ctx["limits_x"], ctx["limits_y"])))
    )

    # Full workspace, for the crop's shared vmax and the target's level ladder only
    # -- the locator no longer draws the field, so its mesh is not needed.
    _, _, points = _grid(ctx, n=200)
    occupancy, _, _ = _field_at(ctx, points, bandwidth)
    zoom_x, zoom_y, zoom_points = _grid(ctx, n=160, limits=box)
    zoom_occupancy, _, _ = _field_at(ctx, zoom_points, bandwidth)
    matched_target = smoothed(ctx["gmm"], bandwidth)
    target_density = np.asarray(pdf(jnp.asarray(points, jnp.float32), matched_target))
    # Levels of the target fill, and of the hairlines on its band boundaries. Cut
    # on the workspace-wide maximum, not the crop's, so a band means the same
    # amount of target here as it does in the locator.
    target_levels = np.linspace(0.0, target_density.max(), 11)[1:]

    # The two cuts through the focus point, and the single density scale they share.
    xs = np.linspace(*box[0], 400)
    ys = np.linspace(*box[1], 400)
    row = _field_at(ctx, np.stack((xs, np.full_like(xs, focus[1])), axis=-1), bandwidth)
    column = _field_at(ctx, np.stack((np.full_like(ys, focus[0]), ys), axis=-1), bandwidth)
    profile_max = max(row[0].max(), row[1].max(), column[0].max(), column[1].max())
    excess_max = max(row[2].max(), column[2].max())

    # The three numbers the caption quotes, straight from the controller's formula.
    occupancy_i, target_i, excess_i = (float(v[0]) for v in _field_at(ctx, focus[None], bandwidth))
    # Minor y ticks off: the crop already carries a labelled tick every
    # ZOOM_TICK_STEP metres, and the unlabelled ones between them only add ink.
    with plt.rc_context(rc={**paper_style("column"), **OUTSIDE_TICKS,
                            "ytick.minor.visible": False}):
        figure = plt.figure(figsize=(3.5, 3.5), constrained_layout=True)

        # A true magnification of the neighbourhood -- same quantity, same ramp,
        # at full column width. What is added is the target it is compared
        # against (contours) and the kernel scale that sets both. The trail goes
        # on flat black here: over the occupancy ramp its own light end washes out.
        zoom = figure.add_subplot()
        _field_map(zoom, zoom_x, zoom_y, zoom_occupancy, vmax=occupancy.max(),
                   levels=64, rules=10)
        _target_contours(zoom, ctx, levels=target_levels, limits=box,
                         gmm=matched_target)
        _trail(zoom, ctx, linewidth=1.3, flat=INK)
        # The two cuts the edge profiles are taken along. Dashed and in the
        # construction ink, so they read as drawing over the field rather than as
        # features of it -- and so that red is left saying one thing, the excess.
        for rule, value in ((zoom.axhline, focus[1]), (zoom.axvline, focus[0])):
            rule(value, color=CONSTRUCTION, linewidth=0.7, alpha=0.8, zorder=3,
                 linestyle=(0, (4, 2.2)))
        radius = float(np.sqrt(0.5 * bandwidth))
        radius_angle = np.deg2rad(35.0)
        radius_end = focus + radius * np.array(
            [np.cos(radius_angle), np.sin(radius_angle)]
        )
        zoom.add_patch(plt.Circle(focus, radius, fill=False, edgecolor=CONSTRUCTION,
                                  linewidth=1.0, zorder=5))
        zoom.plot([focus[0], radius_end[0]], [focus[1], radius_end[1]],
                  color=CONSTRUCTION, linewidth=0.9, linestyle=(0, (3, 2)), zorder=5)
        zoom.plot(*focus, marker="o", markersize=3.2, color=CONSTRUCTION,
                  markeredgewidth=0, zorder=6)
        zoom.annotate(r"$\mathbf{m}_{i^\star}$", xy=focus, xytext=(-6, 5),
                      textcoords="offset points", fontsize=7.5, color=CONSTRUCTION,
                      ha="right", va="bottom", zorder=7)
        radius_midpoint = 0.5 * (focus + radius_end)
        # Offset along the normal of the radius segment, not straight up: a vertical
        # offset leaves a slanted label lying across the dashed line it names.
        label_offset = 13.0 * np.array([-np.sin(radius_angle), np.cos(radius_angle)])
        zoom.annotate(r"$\sqrt{h_c/2}$", xy=radius_midpoint, xytext=tuple(label_offset),
                      textcoords="offset points", fontsize=7.5, color=CONSTRUCTION,
                      ha="center", va="bottom", zorder=7)
        _modes(zoom, ctx)
        _map_axes(zoom, ctx, ylabel=True, limits=box)

        # Locator thumbnail: the full workspace at reduced detail, standing in
        # for the separate context panel the earlier layout gave its own column.
        # The zoom box is square in data under equal aspect, so an axes-fraction
        # box is also a display-proportion box: giving it the domain's own
        # dx:dy ratio lets aspect="auto" fill it exactly -- undistorted, and
        # without the letterboxed white margins that "equal" leaves behind.
        # Bottom-left, because the annotated radius and the profiles now occupy
        # the upper half and the right edge.
        dom_dx = ctx["limits_x"][1] - ctx["limits_x"][0]
        dom_dy = ctx["limits_y"][1] - ctx["limits_y"][0]
        loc_w = 0.34
        loc_h = loc_w * dom_dy / dom_dx
        margin = 0.025
        pad = zoom.inset_axes(
            (margin, margin, loc_w + margin, loc_h + margin),
            zorder=8,
        )
        pad.set_facecolor("#FFFFFF")
        pad.patch.set_alpha(0.92)
        pad.set_xticks([])
        pad.set_yticks([])
        for spine in pad.spines.values():
            spine.set_visible(False)
        locator = zoom.inset_axes(
            (1.5 * margin, 1.5 * margin, loc_w, loc_h),
            zorder=9,
        )
        # No occupancy field in here. A 24-level filled ramp at a third of a column
        # is a grey smear, and the question the thumbnail answers -- whereabouts in
        # Omega is this crop -- is answered by the modes, the path and the box. The
        # panel's own surface is the background instead.
        locator.grid(False)
        _target_contours(locator, ctx, levels=8, rule=0.25,
                         rule_color=TARGET_LINE)
        # The executed path, flat: at thumbnail size the recency gradient of
        # `_trail` is not readable, and the shape is the only thing being asked for.
        locator.plot(memory[:, 0], memory[:, 1], color=INK, linewidth=0.45,
                     solid_capstyle="round", zorder=4)
        locator.set_aspect("auto")
        locator.set_xlim(*ctx["limits_x"])
        locator.set_ylim(*ctx["limits_y"])
        locator.set_xticks([])
        locator.set_yticks([])
        for spine in locator.spines.values():
            spine.set_edgecolor("#A9ABB0")
            spine.set_linewidth(0.5)
        # The crop box is the darkest mark in the thumbnail, so its own frame is
        # dropped to the panel edge colour rather than competing with it.
        locator.add_patch(plt.Rectangle(
            (box[0][0], box[1][0]), box[0][1] - box[0][0], box[1][1] - box[1][0],
            fill=False, edgecolor=CONSTRUCTION, linewidth=0.8, zorder=5,
        ))

        # A title, as every other mechanism panel carries one, naming the quantity
        # and the scale it is read at -- parallel with Fig. 1's occupancy panel. It
        # goes on the figure rather than the axes: `set_title` would land inside the
        # top edge profile, which is an inset and sits above the axes box.
        figure.suptitle(
            rf"Relative over-coverage $e^{{h_c}}$,  $h_c={bandwidth:.2f}$",
            fontweight="bold",
        )

        # The arithmetic, on the two borders: labelled once, on the top edge.
        _edge_profile(zoom, xs, row, profile_max, excess_max, vertical=False,
                      labels=("o", "p"))
        _edge_profile(zoom, ys, column, profile_max, excess_max, vertical=True,
                      labels=("e",))

        path = save(figure, output)
        plt.close(figure)

    print(f"  focus i* = {index}/{len(memory)} at {np.round(focus, 2).tolist()}  "
          f"o = {occupancy_i:.4f}  p* = {target_i:.4f}  e = {excess_i:.2f}  "
          f"(global max e = {excess.max():.2f})")
    return path


def figure_memory_fields(ctx, output: Path) -> Path:
    """Fig. 3 -- recency and over-coverage fields on one workspace map."""
    bandwidth = float(ctx["field"].fine_bandwidth)
    recency = np.asarray(ctx["recency"])
    stream_x, stream_y, stream_points = _grid(ctx, n=34)
    recency_field = _rho(ctx, stream_points, recency, bandwidth)
    recency_magnitude = np.linalg.norm(recency_field, axis=-1).reshape(stream_x.shape)
    quiver_x, quiver_y, quiver_points = _grid(ctx, n=11)
    excess_field = _rho_excess(ctx, quiver_points, bandwidth)
    excess_magnitude = np.linalg.norm(excess_field, axis=-1)
    excess_keep = excess_magnitude >= 0.18 * excess_magnitude.max()
    shared_max = max(recency_magnitude.max(), excess_magnitude.max())

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
        axis.grid(False)
        _target_contours(axis, ctx)
        memory = np.asarray(ctx["memory"])
        axis.plot(*memory.T, color="#30343B", linewidth=1.2, alpha=0.88, zorder=4)

        axis.streamplot(
            stream_x, stream_y,
            recency_field[:, 0].reshape(stream_x.shape),
            recency_field[:, 1].reshape(stream_x.shape),
            color=RECENCY_COLOR,
            linewidth=0.3 + 0.75 * recency_magnitude / shared_max,
            density=0.40, arrowsize=0.5, zorder=3,
        )
        axis.quiver(
            quiver_x.ravel()[excess_keep], quiver_y.ravel()[excess_keep],
            excess_field[excess_keep, 0], excess_field[excess_keep, 1],
            color=FLOW_COLOR, angles="xy", scale_units="xy",
            scale=max(excess_magnitude.max() / 1.25, 1e-12),
            width=0.005, headwidth=3.2, headlength=4.0, alpha=0.9, zorder=3,
        )
        axis.plot([], [], color=RECENCY_COLOR, linewidth=1.1,
                  label=r"recency $\boldsymbol{\rho}^{h_c,\mathrm{rec}}_t$")
        axis.plot([], [], color=FLOW_COLOR, marker=r"$\rightarrow$", linestyle="none",
                  markersize=7, label=r"over-coverage $\boldsymbol{\rho}^{h_c,\mathrm{exc}}_t$")

        axis.plot(*ctx["position"], marker="o", markersize=5.5, color=ROBOT_COLOR,
                  markeredgecolor=ROBOT_EDGE, markeredgewidth=0.9, clip_on=False,
                  zorder=5)
        _modes(axis, ctx)
        _map_axes(axis, ctx, ylabel=True)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                    fontsize=6.0, handletextpad=0.4, handlelength=1.8,
                    columnspacing=0.8, frameon=False, borderaxespad=0.0)

        path = save(figure, output)
        plt.close(figure)

    print(f"  |rho| max: recency {recency_magnitude.max():.4g}  "
          f"over-coverage {excess_magnitude.max():.4g}  "
          f"shown arrows {excess_keep.sum()}/{excess_keep.size}")
    return path


def figure_extra(ctx, output_dir: Path) -> list[Path]:
    """Rebuttal-only: matched target, activity gate, effective blend.

    Deliberately not in the submission. Kept runnable so the answer exists if a
    reviewer asks why the occupancy is compared against a smoothed target, what
    eps_S is for, or why the two fields are blended after normalization rather
    than blending their weights.
    """
    field = ctx["field"]
    paths = []
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))

        # Why p*_h and not p*: comparing a smoothed occupancy against an
        # unsmoothed target manufactures excess that is pure bandwidth artefact.
        axis = axes[0]
        bandwidth = float(field.fine_bandwidth)
        line_y = ctx["slice_y"]
        xs = np.linspace(*ctx["limits_x"], 400)
        slice_points = jnp.stack((xs, jnp.full_like(xs, line_y)), axis=-1)
        raw = np.asarray(pdf(slice_points, ctx["gmm"]))
        matched = np.asarray(pdf(slice_points, smoothed(ctx["gmm"], bandwidth)))
        axis.fill_between(xs, matched, raw, where=raw > matched, color=ACCENT,
                          alpha=0.16, linewidth=0)
        axis.plot(xs, matched, color=TARGET_COLOR, linewidth=1.1)
        axis.plot(xs, raw, color=ACCENT, linewidth=1.0)
        axis.annotate(r"$p^\star_{h_c}$", xy=(xs[300], matched[300]),
                      xytext=(3, 3), textcoords="offset points",
                      fontsize=6, color=TARGET_COLOR)
        axis.annotate(r"$p^\star$", xy=(xs[int(np.argmax(raw))], raw.max()),
                      xytext=(3, 1), textcoords="offset points",
                      fontsize=6, color=ACCENT)
        axis.set_xlabel(rf"$x$ [m] at $y={line_y:.0f}$", labelpad=1)
        axis.set_ylabel("density", labelpad=2)
        axis.set_title("shaded: excess that\nan unmatched target invents", fontsize=6.5)

        # Gate: field magnitude as total excess vanishes.
        axis = axes[1]
        activities = np.geomspace(1e-6, 1e1, 200)
        axis.plot(activities, activities / (activities + 1e-3), color=PRIMARY,
                  linewidth=1.1, label=r"gated: $S/(S+\varepsilon_S)$")
        axis.axhline(1.0, color=ACCENT, linewidth=0.9, linestyle="--",
                     label="ungated (scale-invariant)")
        axis.axvline(1e-3, color=NEUTRAL, linewidth=0.6, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel(r"total relative excess $S^h_t$")
        axis.set_ylabel("field scale")
        axis.set_title("the gate restores continuity\nas over-coverage vanishes", fontsize=6.5)
        axis.legend(loc="lower right", fontsize=6)

        # Effective blend coefficient of the retired weight-blending design.
        axis = axes[2]
        grid_a = np.linspace(0.0, 1.0, 200)
        grid_s = np.geomspace(1e-3, 1e1, 200)
        mesh_a, mesh_s = np.meshgrid(grid_a, grid_s)
        effective = mesh_a * mesh_s / ((1.0 - mesh_a) + mesh_a * mesh_s)
        image = axis.pcolormesh(mesh_a, mesh_s, effective - mesh_a, cmap="RdBu_r",
                                vmin=-0.5, vmax=0.5, shading="auto")
        axis.set_yscale("log")
        axis.set_xlabel(r"requested balance $a$")
        axis.set_ylabel(r"$S^h_t$")
        axis.set_title(r"drift $\alpha^h_t-a$ if the weights" "\n" r"are blended instead", fontsize=6.5)
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=5.5)

        figure.tight_layout(pad=0.35, w_pad=1.1)
        paths.append(save(figure, output_dir / "fig_design_choices.pdf"))
        plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mppi_params.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("theory/pictures"))
    parser.add_argument("--cache", type=Path,
                        default=Path("results/campaign/mechanism_source.npz"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--steps", type=int, default=SOURCE_STEPS)
    parser.add_argument("--extra", action="store_true",
                        help="also render the rebuttal-only design-choices figure")
    args = parser.parse_args()

    arrays = _source(args.config, args.cache, args.device, args.steps)
    config = load_config(args.config)
    ctx = _context(config, arrays)
    print(f"h = {float(ctx['field'].fine_bandwidth):.3f}  P = {len(arrays['memory'])}")

    written = [
        figure_occupancy(ctx, args.output_dir / "fig_occupancy.pdf"),
        figure_excess_focus(ctx, args.output_dir / "fig_excess_focus.pdf"),
        figure_memory_fields(ctx, args.output_dir / "fig_memory_fields.pdf"),
    ]
    if args.extra:
        written.extend(figure_extra(ctx, args.output_dir))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
