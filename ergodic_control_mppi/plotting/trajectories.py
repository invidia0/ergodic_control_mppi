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
