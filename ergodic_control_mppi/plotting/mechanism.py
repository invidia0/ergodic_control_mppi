"""Explanatory figures for Sec. III-C, Fading-Memory Coverage Feedback.

Every field, weight and bandwidth drawn here is produced by calling the
controller's own functions -- ``kernel``, ``kernel_gradient``, ``smoothed``,
``pdf``, ``stein_repulsion`` -- on a memory buffer taken from a real run. Nothing
is hand-drawn, so a figure cannot drift away from the implementation; the
composition of those calls is pinned against ``multiscale_memory_flow`` itself in
``tests/test_plotting.py``.

    python -m ergodic_control_mppi.plotting.mechanism --output-dir theory/pictures

Four figures, in the order Sec. III-C introduces them:

    fig_occupancy       o_t^h and the fading trail that produced it
    fig_excess_focus    the relative excess e_{t,i}^h, read off one memory point
    fig_memory_fields   the recency and over-coverage fields
    fig_scale_bank      the gauge, and what each scale contributes

The source run is cached; delete the .npz to regenerate it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.stein import (
    ACTIVITY_FLOOR,
    kernel,
    kernel_gradient,
    pdf,
    smoothed,
    stein_repulsion,
)
from ergodic_control_mppi.plotting.style import (
    SEQUENTIAL_CMAP,
    TRAIL_CMAP,
    ACCENT,
    NEUTRAL,
    PRIMARY,
    paper_style,
    save,
    sequential,
)

MECHANISM_OCCUPANCY_CMAP = sequential("Greys", 0.05, 0.67)
TARGET_CMAP = sequential("Blues", 0.45, 0.95)
TARGET_COLOR = "#356FA8"
ROBOT_COLOR = "tab:red"
FLOW_COLOR = "#A12F3B"
RECENCY_COLOR = "#26747A"

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
    stein, workspace, gmm = params.stein, params.workspace, params.gmm
    memory = jnp.asarray(arrays["memory"])

    ages = jnp.arange(memory.shape[0])[::-1]
    recency = stein.memory_decay ** ages
    density_floor = 1.0 / (
        (workspace.x_limits[1] - workspace.x_limits[0])
        * (workspace.y_limits[1] - workspace.y_limits[0])
    )
    # Attraction source and bandwidth: the median rollout, as core.py:171-176.
    particles = jnp.asarray(arrays["surrogate"])
    differences = particles[:, None, :] - particles[None, :, :]
    bandwidth = jnp.maximum(
        jnp.median(jnp.sum(differences * differences, axis=-1)), stein.self_bandwidth
    )
    scales = np.asarray(
        jnp.geomspace(stein.fine_bandwidth, stein.coarse_bandwidth, stein.memory_scales)
    )
    # Slice through whichever mode the buffer actually occupies, so panel (b)'s
    # three curves are comparable instead of all sitting near zero.
    means = np.asarray(gmm.means)
    buffer_np = np.asarray(arrays["memory"])
    occupancy_per_mode = [
        float(np.mean(np.linalg.norm(buffer_np - mean, axis=-1) < 3.0)) for mean in means
    ]
    return {
        "params": params, "stein": stein, "gmm": gmm, "workspace": workspace,
        "memory": memory, "recency": recency, "density_floor": density_floor,
        "particles": particles, "bandwidth": bandwidth, "scales": scales,
        "full_path": np.asarray(arrays["path"]),
        "position": np.asarray(arrays["position"]),
        "slice_y": float(means[int(np.argmax(occupancy_per_mode))][1]),
        "limits_x": tuple(map(float, workspace.x_limits)),
        "limits_y": tuple(map(float, workspace.y_limits)),
    }


def _field_at(ctx, points, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occupancy, scale-matched target and relative excess at arbitrary points.

    Term for term eq. (occupancy_density), (smoothed_target) and
    (relative_excess), which is what ``stein.py:174-178`` evaluates at the memory
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


def _rho(ctx, points, masses, bandwidth: float, rotation=None) -> np.ndarray:
    """Memory repulsion of eq. (memory_repulsion), as the controller computes it.

    ``stein_repulsion`` divides by the mass sum, so passing the raw recency gives
    exactly rho(.; q^rec); ``rotation`` overrides C = R(theta) for the panel that
    isolates the curl.
    """
    stein = ctx["stein"]
    if rotation is not None:
        stein = replace(stein, rotation=jnp.asarray(rotation, dtype=jnp.float32))
    return np.asarray(
        stein_repulsion(
            jnp.asarray(points, dtype=jnp.float32).reshape(-1, 2),
            ctx["memory"], jnp.asarray(masses), stein, bandwidth,
        )
    )


def _rho_excess(ctx, points, bandwidth: float, rotation=None) -> np.ndarray:
    """The over-coverage field, gate included -- ``stein.py:188-190`` verbatim."""
    _, _, excess, activity = _excess(ctx, bandwidth)
    gate = activity / (activity + ACTIVITY_FLOOR)
    return gate * _rho(ctx, points, np.asarray(ctx["recency"]) * excess, bandwidth,
                       rotation=rotation)


def _trail(axis, ctx, linewidth: float = 2.0, zorder: int = 4):
    """The executed trail, fading soft blue-grey to near-black, plus the robot."""
    memory = np.asarray(ctx["memory"])
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
    axis.plot(*ctx["position"], marker="o", markersize=5.5, color=ROBOT_COLOR,
              markeredgecolor="#20252B", markeredgewidth=0.45, zorder=zorder + 1)
    return collection


def _target_contours(axis, ctx, levels=12, cmap=TARGET_CMAP,
                     limits=None, gmm=None, linewidth: float = 0.7,
                     alpha: float = 1.0) -> None:
    """Draw visible target-density context beneath the trajectory marks."""
    grid_x, grid_y, points = _grid(ctx, n=160, limits=limits)
    density = np.asarray(
        pdf(jnp.asarray(points, jnp.float32), gmm if gmm is not None else ctx["gmm"])
    )
    axis.contour(grid_x, grid_y, density.reshape(grid_x.shape), levels=levels,
                 cmap=cmap, linewidths=linewidth, alpha=alpha, zorder=1)


def _map_axes(axis, ctx, *, ylabel: bool = False, limits=None) -> None:
    """Shared geometry for every workspace panel, so panels align exactly."""
    limits_x, limits_y = limits or (ctx["limits_x"], ctx["limits_y"])
    axis.set_xlim(*limits_x)
    axis.set_ylim(*limits_y)
    axis.set_aspect("equal")
    if limits is None:
        axis.set_xticks([-10, -5, 0, 5, 10])
        axis.set_yticks([-10, -5, 0, 5, 10])
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
               norm=None, vmax=None):
    """Filled field contours under everything else."""
    if norm is None:
        norm = PowerNorm(0.5, vmin=0.0, vmax=vmax if vmax is not None else values.max())
    # A grid over a filled field is noise, and axisbelow puts it above the field.
    axis.grid(False)
    return axis.contourf(
        grid_x, grid_y, values.reshape(grid_x.shape), levels=24,
        cmap=cmap, norm=norm, zorder=0,
    )


def figure_occupancy(ctx, output: Path) -> Path:
    """Fig. 1 -- the occupancy proxy the memory maintains, and the trail behind it.

    One map, because there is one object: eq. (occupancy_density) at the coarse
    scale, with the buffer that generated it drawn on top fading from soft
    blue-grey (oldest) to near-black (newest), ending at the blue robot marker.
    PowerNorm(1/2) rather than a linear norm -- occupancy is a sum of P narrow
    kernels, so linearly the halo around the track is invisible and only a thin
    ridge survives.
    """
    coarse = float(ctx["stein"].coarse_bandwidth)
    grid_x, grid_y, points = _grid(ctx, n=220)
    occupancy, _, _ = _field_at(ctx, points, coarse)

    with plt.rc_context(rc=paper_style("column")):
        figure, axis = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
        mesh = _field_map(axis, grid_x, grid_y, occupancy)
        _target_contours(axis, ctx)
        _trail(axis, ctx)
        _modes(axis, ctx)
        _map_axes(axis, ctx, ylabel=True)
        axis.set_title(rf"occupancy $o^{{h_c}}_t$,  $h_c={coarse:.2f}$")
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


def figure_excess_focus(ctx, output: Path) -> Path:
    """Fig. 2 -- the relative excess, read off one memory point.

    (a) locates the point in the run, (b) resolves its neighbourhood, and (c) is
    the arithmetic of eq. (relative_excess): the only place the positive part and
    the density floor are visible rather than asserted.
    """
    bandwidth = float(ctx["stein"].coarse_bandwidth)
    index, excess = _focus_point(ctx, bandwidth)
    memory = np.asarray(ctx["memory"])
    focus = memory[index]
    half = 3.5
    # Keep the box inside Omega: outside it the fields are defined but meaningless.
    box = tuple(
        (min(max(c, lo + half), hi - half) - half, min(max(c, lo + half), hi - half) + half)
        for c, (lo, hi) in zip(focus, (ctx["limits_x"], ctx["limits_y"]))
    )

    grid_x, grid_y, points = _grid(ctx, n=200)
    occupancy, _, _ = _field_at(ctx, points, bandwidth)
    zoom_x, zoom_y, zoom_points = _grid(ctx, n=160, limits=box)
    zoom_occupancy, _, _ = _field_at(ctx, zoom_points, bandwidth)
    matched_target = smoothed(ctx["gmm"], bandwidth)
    target_density = np.asarray(pdf(jnp.asarray(points, jnp.float32), matched_target))
    target_levels = np.linspace(0.0, target_density.max(), 10)[1:]

    # The three numbers the caption quotes, straight from the controller's formula.
    occupancy_i, target_i, excess_i = (float(v[0]) for v in _field_at(ctx, focus[None], bandwidth))
    floor = float(ctx["density_floor"])

    with plt.rc_context(rc=paper_style("double")):
        figure = plt.figure(figsize=(6.9, 2.5), constrained_layout=True)
        grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])

        # (a) where in the run this point is.
        axis = figure.add_subplot(grid[0, 0])
        _field_map(axis, grid_x, grid_y, occupancy, vmax=occupancy.max())
        _target_contours(axis, ctx, levels=target_levels, gmm=matched_target)
        _trail(axis, ctx, linewidth=1.4)
        _modes(axis, ctx)
        _map_axes(axis, ctx, ylabel=True)
        axis.set_title(r"(a) $o^{h_c}_t$ over $\Omega$")

        # (b) a true magnification of (a) -- same quantity, same ramp, same limits,
        # so the eye carries the colour across. What is added is the target it is
        # compared against (contours) and the kernel scale that sets both.
        zoom = figure.add_subplot(grid[0, 1])
        _field_map(zoom, zoom_x, zoom_y, zoom_occupancy, vmax=occupancy.max())
        _target_contours(zoom, ctx, levels=target_levels, limits=box,
                         gmm=matched_target)
        _trail(zoom, ctx, linewidth=1.6)
        zoom.axhline(focus[1], color=ACCENT, linewidth=0.65, alpha=0.8, zorder=3)
        radius = float(np.sqrt(0.5 * bandwidth))
        radius_angle = np.deg2rad(35.0)
        radius_end = focus + radius * np.array(
            [np.cos(radius_angle), np.sin(radius_angle)]
        )
        zoom.add_patch(plt.Circle(focus, radius, fill=False, edgecolor="#1F2937",
                                  linewidth=1.2, zorder=5))
        zoom.plot([focus[0], radius_end[0]], [focus[1], radius_end[1]],
                  color="#1F2937", linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
        zoom.plot(*focus, marker="o", markersize=3.6, color="#1F2937",
                  markeredgewidth=0, zorder=6)
        zoom.annotate(r"$\mathbf{m}_{t,i^\star}$", xy=focus, xytext=(-4, 4),
                      textcoords="offset points", fontsize=6.5, color="#1F2937",
                      ha="right", va="bottom", zorder=7)
        radius_midpoint = 0.5 * (focus + radius_end)
        zoom.annotate(r"$\sqrt{h_c/2}$", xy=radius_midpoint, xytext=(0, 4),
                      textcoords="offset points", fontsize=6.5, color="#1F2937",
                      ha="center", va="bottom", zorder=7)
        _modes(zoom, ctx)
        _map_axes(zoom, ctx, limits=box)
        zoom.set_title(r"(b) field: $o^{h_c}_t$,  contours: $p^\star_{h_c}$")
        axis.indicate_inset(
            (box[0][0], box[1][0], 2 * half, 2 * half), inset_ax=zoom,
            edgecolor="#1F2937", linewidth=0.6, alpha=0.9,
        )

        # (c) the arithmetic, on a cut through the point.
        cut = figure.add_subplot(grid[0, 2])
        xs = np.linspace(*box[0], 400)
        cut_points = np.stack((xs, np.full_like(xs, focus[1])), axis=-1)
        occupancy_cut, target_cut, _ = _field_at(ctx, cut_points, bandwidth)
        cut.fill_between(xs, target_cut, occupancy_cut, where=occupancy_cut > target_cut,
                         color=ACCENT, alpha=0.20, linewidth=0, zorder=2)
        cut.plot(xs, occupancy_cut, color="#1F2937", linewidth=1.1, zorder=4)
        cut.plot(xs, target_cut, color=TARGET_COLOR, linewidth=1.1, zorder=4)
        cut.axvline(focus[0], color=NEUTRAL, linewidth=0.6, zorder=1)
        peak = int(np.argmax(occupancy_cut))
        cut.annotate(r"$o^{h_c}_t$", xy=(xs[peak], occupancy_cut[peak]), xytext=(4, 3),
                     textcoords="offset points", fontsize=6.8, color="#1F2937")
        low = int(np.argmax(target_cut))
        cut.annotate(r"$p^\star_{h_c}$", xy=(xs[low], target_cut[low]), xytext=(-20, 6),
                     textcoords="offset points", fontsize=6.8, color=TARGET_COLOR,
                     va="bottom")
        cut.annotate(
            rf"$e^{{h_c}}_{{t,i^\star}}="
            rf"\dfrac{{[{occupancy_i:.4f}-{target_i:.4f}]_+}}"
            rf"{{{target_i:.4f}+{floor:.4f}}}={excess_i:.2f}$",
            xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
            fontsize=6.6, color="#1F2937",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", alpha=0.9,
                      edgecolor="#98A4BA", linewidth=0.4),
        )
        cut.set_xlim(*box[0])
        cut_top = max(occupancy_cut.max(), target_cut.max()) * 1.55
        cut.set_ylim(0, cut_top)
        cut.set_xlabel(rf"$x$ [m] at $y={focus[1]:.1f}$", labelpad=1)
        cut.set_ylabel("density [m$^{-2}$]", labelpad=2)
        cut.set_title(r"(c) shaded: $[o^{h_c}_t-p^\star_{h_c}]_+$")

        # Expand the spatial slice into the full density cross-section.
        for target_y in (0.0, cut_top):
            figure.add_artist(mpatches.ConnectionPatch(
                xyA=(box[0][1], focus[1]), coordsA="data", axesA=zoom,
                xyB=(box[0][0], target_y), coordsB="data", axesB=cut,
                color=ACCENT, linewidth=0.55, alpha=0.55,
                clip_on=False, zorder=0,
            ))

        path = save(figure, output)
        plt.close(figure)

    print(f"  focus i* = {index}/{len(memory)} at {np.round(focus, 2).tolist()}  "
          f"o = {occupancy_i:.4f}  p* = {target_i:.4f}  e = {excess_i:.2f}  "
          f"(global max e = {excess.max():.2f})")
    return path


def figure_memory_fields(ctx, output: Path) -> Path:
    """Fig. 3 -- recency and over-coverage fields on one workspace map."""
    bandwidth = float(ctx["stein"].coarse_bandwidth)
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
        _target_contours(axis, ctx, levels=4, linewidth=0.5, alpha=0.72)
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

        axis.plot(*ctx["position"], marker="o", markersize=4.2, color=ROBOT_COLOR,
                  markeredgewidth=0, clip_on=False, zorder=5)
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


def _scale_field(ctx, points, bandwidth: float) -> np.ndarray:
    """One term of the bank: the gauged, balance-blended field at ``bandwidth``.

    ``stein.py:186-191`` verbatim, for a single scale.
    """
    balance = float(ctx["stein"].memory_balance)
    blended = (1.0 - balance) * _rho(ctx, points, np.asarray(ctx["recency"]), bandwidth)
    blended += balance * _rho_excess(ctx, points, bandwidth)
    return float(np.sqrt(0.5 * np.e * bandwidth)) * blended


def figure_scale_bank(ctx, output: Path) -> Path:
    """Fig. 4 -- why there are Q scales, and why none carries its own gain.

    Left, stacked: raw above gauged, so "before/after" is spatial rather than
    encoded by dash-vs-solid, which would compete with the hue identifying the
    scale. Right: what each gauged scale contributes as a field, under one shared
    colour scale -- the point of the gauge is that they are commensurate, which
    per-panel normalization would hide.
    """
    stein = ctx["stein"]
    scales = ctx["scales"]
    # Ordinal ramp: h_0 < h_1 < h_2 is ordered, so it takes one hue light->dark,
    # not three categorical hues.
    colors = [SEQUENTIAL_CMAP(v) for v in np.linspace(0.12, 1.0, len(scales))]

    grid_x, grid_y, points = _grid(ctx, n=200)
    contributions = [np.linalg.norm(_scale_field(ctx, points, float(h)), axis=-1)
                     for h in scales]
    shared_max = max(c.max() for c in contributions)

    with plt.rc_context(rc=paper_style("double")):
        # Height is set so the two stacked profile axes and the square maps end
        # up the same height: taller and the maps float in slack, shorter and the
        # profiles are too squat to read.
        figure = plt.figure(figsize=(6.9, 2.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 4, width_ratios=[1.2, 1.0, 1.0, 1.0])
        ax_raw = figure.add_subplot(grid[0, 0])
        ax_gauged = figure.add_subplot(grid[1, 0], sharex=ax_raw)

        radii = np.linspace(1e-3, 3.6, 400)
        probe = jnp.stack((radii, jnp.zeros_like(radii)), axis=-1)
        origin = jnp.zeros_like(probe)
        peaks = []
        for index, (bandwidth, color) in enumerate(zip(scales, colors)):
            raw = np.linalg.norm(np.asarray(kernel_gradient(probe, origin, bandwidth)), axis=-1)
            gauged = float(np.sqrt(0.5 * np.e * bandwidth)) * raw
            peaks.append(gauged.max())
            ax_raw.plot(radii, raw, color=color, linewidth=1.0)
            ax_gauged.plot(radii, gauged, color=color, linewidth=1.0)
            # Direct label at the peak instead of a legend box.
            ax_raw.annotate(rf"$h_{index}\!=\!{bandwidth:.2f}$",
                            xy=(float(radii[int(np.argmax(raw))]), raw.max()),
                            xytext=(2.5, 1.0), textcoords="offset points",
                            fontsize=6.0, color=color)

        # Self-check, not decoration: the gauge is correct iff every scale peaks
        # at exactly 1. Fail loudly rather than ship a wrong figure.
        if not np.allclose(peaks, 1.0, atol=1e-5):
            raise AssertionError(f"gauge broken: peak magnitudes {peaks} != 1")

        ax_gauged.axhline(1.0, color="#5C6B87", linewidth=0.5, zorder=1)
        ax_raw.set_ylabel(r"$\|\nabla\kappa_h\|$", labelpad=2)
        ax_gauged.set_ylabel("gauged", labelpad=2)
        ax_gauged.set_xlabel(r"$r=\|\mathbf{z}-\mathbf{m}_{t,i}\|$ [m]", labelpad=1)
        ax_raw.tick_params(labelbottom=False)
        ax_raw.set_ylim(0, 4.0)
        ax_gauged.set_ylim(0, 1.3)
        ax_raw.set_title("(a) raw: peaks differ 8x", pad=3)
        ax_gauged.set_title(r"$\times\sqrt{he/2}$: peaks coincide at 1", pad=3,
                            fontsize=6.5, fontweight="normal")

        # (b)-(d) what each gauged scale actually contributes over the workspace.
        for index, (bandwidth, magnitude) in enumerate(zip(scales, contributions)):
            axis = figure.add_subplot(grid[:, index + 1])
            mesh = _field_map(axis, grid_x, grid_y, magnitude, vmax=shared_max)
            _trail(axis, ctx, linewidth=0.7, zorder=3)
            _modes(axis, ctx)
            _map_axes(axis, ctx, ylabel=(index == 0))
            axis.set_title(rf"({'bcd'[index]}) $h_{index}={bandwidth:.2f}$", pad=3)
            if index == len(scales) - 1:
                _inline_colorbar(figure, axis, mesh,
                                 r"$\sqrt{he/2}\,\|\boldsymbol{\rho}^h_t\|$  (shared)",
                                 backing=True)

        path = save(figure, output)
        plt.close(figure)

    print("  gauged scale peaks: "
          + "  ".join(f"h={h:.2f}: {c.max():.4g}" for h, c in zip(scales, contributions)))
    return path


def figure_extra(ctx, output_dir: Path) -> list[Path]:
    """Rebuttal-only: matched target, activity gate, effective blend.

    Deliberately not in the submission. Kept runnable so the answer exists if a
    reviewer asks why the occupancy is compared against a smoothed target, what
    eps_S is for, or why the two fields are blended after normalization rather
    than blending their weights.
    """
    stein = ctx["stein"]
    paths = []
    with plt.rc_context(rc=paper_style("double")):
        figure, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))

        # Why p*_h and not p*: comparing a smoothed occupancy against an
        # unsmoothed target manufactures excess that is pure bandwidth artefact.
        axis = axes[0]
        coarse = float(stein.coarse_bandwidth)
        line_y = ctx["slice_y"]
        xs = np.linspace(*ctx["limits_x"], 400)
        slice_points = jnp.stack((xs, jnp.full_like(xs, line_y)), axis=-1)
        raw = np.asarray(pdf(slice_points, ctx["gmm"]))
        matched = np.asarray(pdf(slice_points, smoothed(ctx["gmm"], coarse)))
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
    print(f"scale bank: {np.round(ctx['scales'], 3).tolist()}  P = {len(arrays['memory'])}")

    written = [
        figure_occupancy(ctx, args.output_dir / "fig_occupancy.pdf"),
        figure_excess_focus(ctx, args.output_dir / "fig_excess_focus.pdf"),
        figure_memory_fields(ctx, args.output_dir / "fig_memory_fields.pdf"),
        figure_scale_bank(ctx, args.output_dir / "fig_scale_bank.pdf"),
    ]
    if args.extra:
        written.extend(figure_extra(ctx, args.output_dir))
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
