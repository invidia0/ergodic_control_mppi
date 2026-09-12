"""
Figures for the discrepancy audit: what the witness looks like, and how the bound behaves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ergodic_control_mppi.metrics.discrepancy import mean_embedding
from ergodic_control_mppi.plotting import style


def herding_points(
    support: np.ndarray, embedding: np.ndarray, bandwidth: float,
    count: int, start: np.ndarray,
) -> np.ndarray:
    """
    Run the exact occupation-descent oracle on the target's support.
    
    Args:
            support: Target support points, ``(S, 2)``.
            embedding: ``m_pi`` evaluated on ``support``, shape ``(S,)``.
            bandwidth: The kernel's ``h``.
            count: How many points to place.
            start: The first point, shape ``(2,)``.
    Returns:
            The placed points with shape ``(count, 2)``.
    """
    kernel_sum = np.zeros(support.shape[0])
    points = [np.asarray(start, dtype=np.float64)]
    for placed in range(1, count):
        kernel_sum += np.exp(-np.sum((support - points[-1]) ** 2, axis=1) / bandwidth)
        points.append(support[np.argmin(kernel_sum / placed - embedding)])
    return np.stack(points)


#: Flat obstacle footprints. `style.PILLAR_CMAP` is a blue-to-green ramp for the 3D
#: deployment render; on a top-down mask every cell holds the same value, so a ramp only
#: contributes a colour that means nothing. One neutral reads as geometry instead.
FOOTPRINT_CMAP = ListedColormap(["#4A4E57"])


def _map_panel(axes, occupancy, extent, limits_x, limits_y) -> None:
    """Draw the rasterized pillars and fix the panel to the workspace."""
    axes.imshow(np.ma.masked_where(~occupancy, occupancy), origin="lower", extent=extent,
                cmap=FOOTPRINT_CMAP, interpolation="nearest", zorder=3)
    axes.set_xlim(*limits_x)
    axes.set_ylim(*limits_y)
    axes.set_aspect("equal")
    axes.set_xlabel("x [m]")


def mechanism_figure(
    path: np.ndarray, support: np.ndarray, weights: np.ndarray, bandwidth: float,
    density: np.ndarray, mask: np.ndarray, occupancy: np.ndarray, extent,
    limits_x, limits_y, step_radius: float,
):
    """
    Draw the target, the witness at the end of the run, and the ideal oracle's points.
    
    Args:
            path: The audited executed positions, shape ``(N, 2)``.
            support: Target support points, ``(S, 2)``, from
                : func:`ergodic_control_mppi.metrics.discrepancy.grid_target`.
            weights: Target weights, ``(S,)``.
            bandwidth: The kernel's ``h``.
            density: The unrestricted target grid, ``(rows, columns)``, on ``linspace`` nodes.
            mask: Reachable component on that grid, ``(rows, columns)``.
            occupancy: Rasterized obstacle map for display, on its own raster extent.
            extent: ``occupancy``'s extent, which is *not* the workspace box the metric grid is
                drawn on -- the two grids have different resolutions and origins.
            limits_x: Workspace x limits.
            limits_y: Workspace y limits.
            step_radius: One-step reach, drawn as the vehicle's choice set.
    Returns:
            The figure.
    """
    field_extent = (limits_x[0], limits_x[1], limits_y[0], limits_y[1])
    rows, columns = density.shape
    mesh_x, mesh_y = np.meshgrid(np.linspace(*limits_x, columns),
                                 np.linspace(*limits_y, rows))
    grid = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    free = mask.ravel()

    embedding = mean_embedding(grid, support, weights, bandwidth)
    kernel_sum = np.zeros(grid.shape[0])
    for point in path[:-1]:
        kernel_sum += np.exp(-np.sum((grid - point) ** 2, axis=1) / bandwidth)
    witness = kernel_sum / (len(path) - 1) - embedding
    oracle = herding_points(
        support, mean_embedding(support, support, weights, bandwidth),
        bandwidth, len(path), path[0],
    )
    restricted = density * mask
    restricted = restricted / restricted.sum()

    with plt.rc_context(style.paper_style("double") | style.OUTSIDE_TICKS):
        figure, panels = plt.subplots(1, 3, figsize=(style.FIGSIZES["double"][0], 2.5))

        panels[0].imshow(restricted, origin="lower", extent=field_extent,
                         cmap=style.DENSITY_CMAP, interpolation="bilinear", zorder=1)
        panels[0].plot(path[:, 0], path[:, 1], color=style.PRIMARY, lw=0.7, alpha=0.9, zorder=4)
        panels[0].set_title(f"target and executed path ($N={len(path)}$)")
        panels[0].set_ylabel("y [m]")

        field = witness.reshape(rows, columns)
        limit = float(np.abs(field).max())
        image = panels[1].imshow(np.ma.masked_where(~mask, field),
                                 origin="lower", extent=field_extent,
                                 cmap=style.DIVERGING_CMAP, vmin=-limit, vmax=limit, zorder=1)
        panels[1].plot(path[:, 0], path[:, 1], color="0.25", lw=0.4, alpha=0.5, zorder=4)
        admissible = np.flatnonzero(free)
        best = grid[admissible[np.argmin(witness[admissible])]]
        here = path[-1]
        panels[1].add_patch(plt.Circle(here, step_radius, fill=False, lw=0.8,
                                       ec=style.ACCENT, ls="--", zorder=5))
        panels[1].plot(*here, "o", ms=4, color=style.ACCENT, zorder=6, label="vehicle")
        panels[1].plot(*best, "*", ms=9, color="#111111", zorder=6, label=r"$\arg\min w_n$")
        panels[1].legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3)
        panels[1].set_title(rf"witness $w_n$ at $n={len(path)}$")
        figure.colorbar(image, ax=panels[1], fraction=0.040, pad=0.02, shrink=0.86)

        panels[2].scatter(oracle[:, 0], oracle[:, 1], s=2.5, color=style.PRIMARY,
                          alpha=0.85, linewidths=0, zorder=4)
        panels[2].set_title(r"ideal oracle, $\delta_n = 0$")

        for panel in panels:
            _map_panel(panel, occupancy, extent, limits_x, limits_y)
        figure.tight_layout()
    return figure


def bound_figure(series: dict[str, np.ndarray], bandwidths=None, looseness=None,
                 constraint_share=None):
    """
    Draw the bound against the observed error and the gap against its reach proxy.
    
    Args:
            series: ``n``, ``error``, ``bound``, ``gap``, ``gap_reach``, ``trivial`` -- the keys
                ``scripts/theory_audit.py discrepancy`` writes into its series file.
            bandwidths: Optional kernel ``h`` values of a sensitivity sweep; when given, a third
                panel is drawn. The audit does not produce a sweep, so it is normally absent.
            looseness: Final ``bound / error`` at each swept bandwidth.
            constraint_share: Median ``(gap - gap_reach) / gap`` at each swept bandwidth.
    Returns:
            The figure.
    """
    count, error, bound = series["n"], series["error"], series["bound"]
    columns = 2 if bandwidths is None else 3
    with plt.rc_context(style.paper_style("double") | style.OUTSIDE_TICKS):
        figure, panels = plt.subplots(
            1, columns, figsize=(style.FIGSIZES["double"][0] * columns / 3.0, 2.3)
        )

        panels[0].loglog(count, error, color=style.PRIMARY, lw=1.2, label=r"observed $E_n$")
        panels[0].loglog(count[1:], bound[1:], color=style.ACCENT, lw=1.2,
                         label="theorem bound")
        panels[0].axhline(series["trivial"], color=style.NEUTRAL, lw=0.9, ls=":",
                          label="trivial bound")
        reference = error[4] * count[4] / count[4:]
        panels[0].loglog(count[4:], reference, color="0.35", lw=0.8, ls="--",
                         label=r"$O(1/n)$")
        panels[0].set_xlabel("audited samples $n$")
        panels[0].set_ylabel(r"squared MMD")
        panels[0].set_title(rf"bound within {bound[-1]/error[-1]:.1f}$\times$ at "
                            rf"$n={int(count[-1])}$")
        panels[0].legend(loc="upper right", borderaxespad=0.3)

        panels[1].plot(count[:-1], series["gap"][:-1], color=style.ACCENT, lw=1.0,
                       label=r"min over $\mathrm{supp}\,p^\star$")
        panels[1].plot(count[:-1], series["gap_reach"][:-1], color="#59A14F", lw=1.0,
                       label="over the reach proxy")
        panels[1].set_ylim(bottom=0.0)
        panels[1].set_xlabel("audited samples $n$")
        panels[1].set_ylabel(r"descent gap $\delta_n$")
        panels[1].set_title("the gap never decays")
        panels[1].legend(loc="upper right", borderaxespad=0.3)

        if bandwidths is not None:
            panels[2].semilogx(bandwidths, looseness, "o-", color=style.ACCENT, lw=1.1, ms=3.5)
            panels[2].set_xlabel(r"kernel bandwidth $h$")
            panels[2].set_ylabel(r"looseness  bound$/E_N$", color=style.ACCENT)
            panels[2].tick_params(axis="y", colors=style.ACCENT)
            twin = panels[2].twinx()
            twin.semilogx(bandwidths, constraint_share, "s--", color=style.PRIMARY,
                          lw=1.1, ms=3.5)
            twin.set_ylabel("reach-proxy share of the gap", color=style.PRIMARY)
            twin.tick_params(axis="y", colors=style.PRIMARY)
            twin.set_ylim(0.0, 1.0)
            panels[2].set_title("a wider kernel shifts the gap\nto the speed limit")

        figure.tight_layout()
    return figure
