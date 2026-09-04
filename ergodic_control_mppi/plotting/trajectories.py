"""Flat trajectory panels: the executed path against the target modes it is meant to serve.

The question these answer is not "how good is the number" but "what shape does the
controller draw". A grid of panels over one axis -- bandwidth, or method -- makes the
dwell/transit trade visible in a way the metrics table cannot: the same controller at
h=0.94 fills two basins and crosses once between them, and at h=5.0 shuttles.

Mode boundaries are drawn at the 2-sigma Mahalanobis ellipse because that is the boundary
``metrics/modes.py`` actually uses for ``in_mode_fraction`` (``enter_sigma=2.0``), so a
reader counting time inside an ellipse is counting the reported statistic and not a
decorative contour.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ergodic_control_mppi.plotting import style

# The 2-sigma level of a 2-D Gaussian: chi2 with 2 dof, so the Mahalanobis radius is the
# sigma multiple itself rather than a quantile lookup.
_ENTER_SIGMA = 2.0


def _ellipse_points(mean: np.ndarray, covariance: np.ndarray, sigma: float,
                    count: int = 181) -> np.ndarray:
    """Return the ``sigma``-Mahalanobis ellipse of a 2-D Gaussian as a closed polyline."""
    values, vectors = np.linalg.eigh(np.asarray(covariance, dtype=float))
    # eigh can return a tiny negative eigenvalue on a near-singular covariance; clipping
    # keeps the square root real without silently reshaping a well-conditioned one.
    radii = sigma * np.sqrt(np.clip(values, 0.0, None))
    angle = np.linspace(0.0, 2.0 * np.pi, count)
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=0)
    return (np.asarray(mean, dtype=float)[:, None] + vectors @ (radii[:, None] * unit)).T


def _draw_panel(axes, positions, means, covariances, occupancy=None, origin=None,
                resolution: float = 0.15, title: str | None = None,
                limits: tuple[float, float, float, float] | None = None) -> None:
    """Draw one trajectory panel: obstacles, path coloured by time, mode ellipses."""
    if occupancy is not None:
        occupied = np.asarray(occupancy, dtype=bool)
        rows, columns = np.nonzero(occupied)
        if rows.size:
            x = origin[0] + (columns + 0.5) * resolution
            y = origin[1] + (rows + 0.5) * resolution
            axes.scatter(x, y, s=(resolution * 72 / 0.15) ** 2 * 0.02,
                         c=style.NEUTRAL, marker="s", linewidths=0, zorder=1)

    path = np.asarray(positions, dtype=float)
    # One LineCollection rather than a scatter: a 20k-step path drawn as points is both
    # slower to render and heavier in the PDF than the same path as joined segments.
    from matplotlib.collections import LineCollection

    segments = np.stack([path[:-1], path[1:]], axis=1)
    collection = LineCollection(segments, cmap=style.SEQUENTIAL_CMAP, linewidths=0.5,
                                array=np.linspace(0.0, 1.0, len(segments)), zorder=2)
    axes.add_collection(collection)

    for mean, covariance in zip(np.asarray(means), np.asarray(covariances)):
        ring = _ellipse_points(mean[:2], np.asarray(covariance)[:2, :2], _ENTER_SIGMA)
        axes.plot(ring[:, 0], ring[:, 1], color=style.ACCENT, linewidth=1.0,
                  linestyle=(0, (4, 2)), zorder=3)

    if limits is not None:
        axes.set_xlim(limits[0], limits[1])
        axes.set_ylim(limits[2], limits[3])
    else:
        axes.autoscale_view()
    axes.set_aspect("equal", adjustable="box")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_linewidth(0.6)
    if title:
        axes.set_title(title)


def panel_grid(captures, path: str | Path, columns: int = 2, size: str = "double"):
    """Draw a grid of trajectory panels and save it.

    Args:
        captures: Sequence of mappings with keys ``positions``, ``means``,
            ``covariances``, ``title``, and optionally ``occupancy``, ``grid_origin``,
            ``grid_resolution`` and ``limits`` -- the field names ``capture.py`` writes,
            so a saved ``.npz`` can be passed through unchanged.
        path: Destination for the rendered figure.
        columns: Panels per row.
        size: A key of :data:`style.FIGSIZES`.

    Returns:
        The written path.
    """
    import matplotlib.pyplot as plt

    captures = list(captures)
    if not captures:
        raise ValueError("panel_grid needs at least one capture")
    rows = int(np.ceil(len(captures) / columns))

    with plt.rc_context(style.paper_style(size)):
        width, height = style.FIGSIZES[size]
        figure, grid = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(width, height / 2.0 * rows))
        for axes, capture in zip(grid.ravel(), captures):
            _draw_panel(
                axes,
                capture["positions"],
                capture["means"],
                capture["covariances"],
                occupancy=capture.get("occupancy"),
                origin=capture.get("grid_origin"),
                resolution=float(capture.get("grid_resolution", 0.15)),
                title=capture.get("title"),
                limits=capture.get("limits"),
            )
        for axes in grid.ravel()[len(captures):]:
            axes.set_visible(False)
        figure.tight_layout()
        return style.save(figure, path)


def load_captures(paths, titles=None):
    """Load ``.npz`` captures written by the capture harness, in the given order."""
    loaded = []
    for index, item in enumerate(paths):
        data = np.load(item, allow_pickle=False)
        capture = {key: data[key] for key in data.files}
        capture["title"] = (titles[index] if titles is not None
                            else Path(item).stem.replace("_", " "))
        loaded.append(capture)
    return loaded


# --------------------------------------------------------------------------- mechanism


def _potential_grid(capture, resolution: int = 220):
    """Evaluate ``Phi`` on a workspace grid from one frozen step of a capture.

    Only possible because the field is a gradient: there is a scalar to draw. The memory,
    recency and plan point sets are the frozen ones, so the surface is the landscape the
    controller was descending at that step and not a generic potential.
    """
    import jax.numpy as jnp

    from ergodic_control_mppi.mppi.field import potential
    from ergodic_control_mppi.parameters import FieldParams, GMMParams

    limits = np.asarray(capture["limits"], dtype=float)
    grid_x, grid_y = np.meshgrid(
        np.linspace(limits[0], limits[1], resolution),
        np.linspace(limits[2], limits[3], resolution),
    )
    points = jnp.asarray(
        np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1), dtype=jnp.float32
    )

    covariance = jnp.asarray(capture["covariances"], dtype=jnp.float32)
    gmm = GMMParams(
        means=jnp.asarray(capture["means"], dtype=jnp.float32),
        covariance=covariance,
        covariance_inverse=jnp.linalg.inv(covariance),
        log_weights=jnp.asarray(capture["log_weights"], dtype=jnp.float32),
        log_normalizers=-0.5 * (2 * jnp.log(2 * jnp.pi)
                                + jnp.linalg.slogdet(covariance)[1]),
    )
    # Only the fields `potential` reads; the rest carry their disabling values, so a
    # missing key in an older capture fails loudly rather than drawing a different surface.
    field = FieldParams(
        track_weight=0.0,
        fine_bandwidth=float(capture["fine_bandwidth"]),
        memory_decay=0.0,
        reference_speed=0.0,
        memory_gain=float(capture["memory_gain"]),
        memory_balance=float(capture["memory_balance"]),
        plan_gain=float(capture["plan_gain"]),
        transit_speedup=1.0,
        dwell_slowdown=1.0,
        service_floor=0.0,
        service_decay=0.0,
        deficit_ceiling=float(capture["deficit_ceiling"]),
        release_ratio=float(capture["release_ratio"]),
    )
    workspace_area = (limits[1] - limits[0]) * (limits[3] - limits[2])
    values = potential(
        points,
        jnp.asarray(capture["memory"], dtype=jnp.float32),
        jnp.asarray(capture["recency"], dtype=jnp.float32),
        jnp.asarray(capture["plan"], dtype=jnp.float32),
        gmm, field,
        jnp.asarray(1.0 / workspace_area, dtype=jnp.float32),
        jnp.asarray(capture["service_mass"], dtype=jnp.float32),
    )
    return grid_x, grid_y, np.asarray(values, dtype=float).reshape(grid_x.shape)


def figure_plan_gain(captures, path: str | Path, columns: int = 2):
    """Sec. III-D -- what the plan self-repulsion does, over ``g``.

    Each panel: the executed path coloured by time over the 2-sigma ellipses, the planned
    cloud at one frozen step drawn on top, and a ``Phi`` contour underlay.

    The claim is that ``f_mem`` cannot substitute for ``f_plan``. At ``g = 0`` the memory
    still repels the vehicle from where it has been, and a compact repeated circuit is
    perfectly admissible under that -- so the plan collapses to a tight loop. At the
    deployed ``g`` the horizon points repel each other and the plan fills the ellipse. Both
    facts are visible in the same panel: the drawn cloud is the plan, and the contour is
    the landscape it sits in.
    """
    import matplotlib.pyplot as plt

    captures = list(captures)
    rows = int(np.ceil(len(captures) / columns))
    with plt.rc_context(style.paper_style("double")):
        width, _ = style.FIGSIZES["double"]
        # Height from the *map* aspect, not from the style's default. `_draw_panel` sets an
        # equal aspect, so a cell taller than the map letterboxes it: the drawing shrinks to
        # fit the short side and the spare width opens as a gap between the columns. At a
        # 40x20 workspace that gap was as wide as the panels themselves.
        limits = np.asarray(captures[0]["limits"], dtype=float)
        cell = (limits[3] - limits[2]) / (limits[1] - limits[0])
        figure, grid = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(width, rows * (width / columns * cell + 0.32)))
        for axes, capture in zip(grid.ravel(), captures):
            limits = np.asarray(capture["limits"], dtype=float)
            grid_x, grid_y, phi = _potential_grid(capture)
            # Percentile clip, not min/max: log p* diverges downward far from every mode,
            # so a raw range spends the whole colour scale on empty corners.
            low, high = np.percentile(phi, (2.0, 100.0))
            shading = axes.contourf(grid_x, grid_y, np.clip(phi, low, high), levels=24,
                                    cmap=style.DENSITY_CMAP, alpha=0.55, zorder=0)
            _draw_panel(axes, capture["positions"], capture["means"],
                        capture["covariances"], title=capture.get("title"),
                        limits=(limits[0], limits[1], limits[2], limits[3]))
            plan = np.asarray(capture["plan"], dtype=float)
            axes.scatter(plan[:, 0], plan[:, 1], s=3.0, c=style.ACCENT,
                         linewidths=0, zorder=5,
                         label=f"plan at step {int(capture['freeze_step'])}")
        for axes in grid.ravel()[len(captures):]:
            axes.set_visible(False)
        handles, labels = grid.ravel()[0].get_legend_handles_labels()
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=shading.cmap(0.35), edgecolor="none")] + handles[:1]
        labels = [r"potential $\Phi$ (dark = low)"] + labels[:1]
        figure.tight_layout(h_pad=0.6)
        figure.legend(handles, labels, loc="lower center", frameon=False, ncol=2,
                      bbox_to_anchor=(0.5, -0.02))
        return style.save(figure, path)


def service_series(capture):
    """Return ``(time, sigma, log_weight)`` per mode from a capture's service mass.

    ``sigma_j = (mass_j / sum mass) / w_j`` is the per-mode service ratio the gate reads,
    and ``log_weight`` is the bent log-weight the attraction actually follows -- promotion
    from the deficit ceiling, demotion from the release ratio. Both are pure functions of
    the recorded mass, so nothing had to be instrumented into the control loop to get them.
    """
    import jax.numpy as jnp

    from ergodic_control_mppi.mppi.field import deficit_weighted, per_mode_weighted
    from ergodic_control_mppi.parameters import GMMParams

    mass = np.asarray(capture["service_mass_history"], dtype=float)
    weights = np.exp(np.asarray(capture["log_weights"], dtype=float))
    share = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
    sigma = share / np.maximum(weights, 1e-12)

    covariance = jnp.asarray(capture["covariances"], dtype=jnp.float32)
    gmm = GMMParams(
        means=jnp.asarray(capture["means"], dtype=jnp.float32),
        covariance=covariance,
        covariance_inverse=jnp.linalg.inv(covariance),
        log_weights=jnp.asarray(capture["log_weights"], dtype=jnp.float32),
        log_normalizers=-0.5 * (2 * jnp.log(2 * jnp.pi)
                                + jnp.linalg.slogdet(covariance)[1]),
    )
    ceiling = jnp.asarray(max(float(capture["deficit_ceiling"]), 1e-12), jnp.float32)
    release = float(capture["release_ratio"])
    bent = np.stack([
        np.asarray(
            per_mode_weighted(jnp.asarray(row, jnp.float32), gmm, ceiling, release)
            .log_weights if release > 0 else
            deficit_weighted(jnp.asarray(row, jnp.float32), gmm, ceiling).log_weights,
            dtype=float,
        )
        for row in mass
    ])
    time = (np.arange(len(mass)) * float(capture["stride"])
            * float(capture["delta_t"]))
    return time, sigma, bent


def figure_service_gate(captures, path: str | Path, deployed_release_ratio: float = 2.24):
    """Sec. III-E -- the release, and why it happens when it happens.

    Top row: one trajectory panel per ``sigma*`` level, so the basin trap at ``off`` and
    the full tour at the deployed level are the same picture at two settings.

    Bottom strip: for the deployed arm, ``sigma_j(t)`` for each mode with the release
    threshold drawn as a horizontal line, and the bent ``log w_j(t)`` beneath it. The
    prediction and the event are in one figure: the log-weight crosses the ``Delta_j``
    margin at the moment the path leaves.
    """
    import matplotlib.pyplot as plt

    captures = list(captures)
    if not captures:
        raise ValueError("figure_service_gate needs at least one capture")
    columns = len(captures)
    with plt.rc_context(style.paper_style("double")):
        width, height = style.FIGSIZES["double"]
        figure = plt.figure(figsize=(width, height * 1.15))
        grid = figure.add_gridspec(3, columns, height_ratios=[2.2, 1.0, 1.0], hspace=0.35)

        for index, capture in enumerate(captures):
            limits = np.asarray(capture["limits"], dtype=float)
            axes = figure.add_subplot(grid[0, index])
            _draw_panel(axes, capture["positions"], capture["means"],
                        capture["covariances"], title=capture.get("title"),
                        limits=(limits[0], limits[1], limits[2], limits[3]))

        # The strip reads the *deployed* arm: it explains the mechanism, and showing it for
        # a level where the gate is off would only show a flat line.
        chosen = next(c for c in captures
                      if np.isclose(float(c["release_ratio"]), deployed_release_ratio))
        time, sigma, bent = service_series(chosen)
        release = float(chosen["release_ratio"])
        colours = style.TABLEAU[: sigma.shape[1]]

        axis = figure.add_subplot(grid[1, :])
        for mode in range(sigma.shape[1]):
            axis.plot(time, sigma[:, mode], color=colours[mode], linewidth=0.8,
                      label=rf"mode {mode + 1}")
        if release > 0:
            # The point of *full* release, not a threshold. The penalty is continuous --
            # `kappa_j (sigma_j - 1)` with `kappa_j = Delta_j / (sigma* - 1)` -- so it equals
            # the log-odds margin exactly at sigma* and a fraction of it below. In closed
            # loop the vehicle leaves long before sigma* arrives, so the line is never
            # crossed; labelling it bare invited the reader to read that as a dead gate.
            axis.axhline(release, color=style.ACCENT, linewidth=0.9,
                         linestyle=(0, (4, 2)),
                         label=rf"$\sigma^*={release:g}$ (full release)")
            axis.annotate(rf"peak $\sigma_j={np.nanmax(sigma):.2f}$: the loop leaves first",
                          xy=(0.015, 0.06), xycoords="axes fraction", fontsize=5.5,
                          color=style.NEUTRAL)
        axis.set_ylabel(r"$\sigma_j$")
        axis.set_xticklabels([])
        axis.legend(loc="upper right", frameon=False, ncol=sigma.shape[1] + 1,
                    fontsize=6.0)

        axis = figure.add_subplot(grid[2, :])
        for mode in range(bent.shape[1]):
            axis.plot(time, bent[:, mode], color=colours[mode], linewidth=0.8)
        axis.set_ylabel(r"$\log \hat{w}_j$")
        axis.set_xlabel("time [s]")

        return style.save(figure, path)
