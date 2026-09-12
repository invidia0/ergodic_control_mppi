"""Oblique 3D view of a volumetric run: every pillar at its own height, the path as flown."""

from pathlib import Path

import numpy as np

from ergodic_control_mppi.plotting import style
from ergodic_control_mppi.plotting.deployment import (
    FLOOR_EDGE,
    _crop_transparent,
    _draw_quadrotor,
    _heading,
    _resolve_cmap,
    _strip_axes,
)

# From the paper's own cycle, and of similar lightness: at one alpha a dark hue (teal)
# reads as solid next to pale ones.
MODE_COLOURS = (style.NPG[6], style.NPG[4], style.NPG[9])
# 2.5 sigma holds about 90% of a 3D Gaussian's mass (chi-square, 3 dof).
SHELL_SIGMA = 2.5
SHELL_ALPHA = 0.20
VEHICLE_COLOUR = "#111111"
# The bunny scan: a light matte grey under a headlight raised this far above the camera, so
# the two opposite views are lit alike.
BUNNY_GREY = "#C9CDD3"
BUNNY_LIGHT_RISE = 0.6
# A noise map as RViz shows one: occupied cells as translucent squares coloured by height.
# YlGnBu from 15% in, since its palest yellow vanishes on the white floor.
OBSTACLE_RAMP = ("YlGnBu", 0.15, 1.0)
OBSTACLE_ALPHA = 0.5
OBSTACLE_MARKER = 14      # points^2: about a 0.3 m cell at the render's scale


def _planar_normal(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Evaluate a planar Gaussian PDF at ``points``."""
    delta = points - mean
    quadratic = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
    return np.exp(-0.5 * quadratic) / (2.0 * np.pi * np.sqrt(np.linalg.det(covariance)))


def _draw_scene(axes, pillars, path, means, covariances, floor, ceiling, azimuth,
                altitude_map, low, high, hidden=None) -> int:
    """Pillars, mode shells and path pieces, painter-sorted along the view direction."""
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    towards_camera = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])
    theta = np.linspace(0.0, 2.0 * np.pi, 90)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    ramp = _obstacle_cmap()  # the Perlin cloud's height ramp, so both maps read alike

    def pillar(x, y, radius, top, order):
        levels = np.linspace(floor, top, max(int(48 * (top - floor) / (ceiling - floor)), 2))
        # One height scale for every pillar, so a short one is visibly short, not re-ramped.
        shade = np.asarray(ramp((levels - floor) / (ceiling - floor)), dtype=float)
        ring_x, ring_y = x + radius * np.cos(theta), y + radius * np.sin(theta)
        surface = axes.plot_surface(
            np.tile(ring_x, (levels.size, 1)), np.tile(ring_y, (levels.size, 1)),
            np.tile(levels[:, None], (1, theta.size)),
            facecolors=np.tile(shade[:, None, :], (1, theta.size, 1)),
            shade=False, linewidth=0, antialiased=True, zorder=order, rstride=1, cstride=1)
        surface.set_clip_on(False)
        surface.set_edgecolor("face")
        cap = Poly3DCollection([np.column_stack([ring_x, ring_y, np.full(theta.size, top)])],
                               facecolors=[shade[-1]], edgecolors="none", linewidths=0)
        cap.set_zorder(order + 0.1)
        cap.set_clip_on(False)
        axes.add_collection3d(cap)

    azimuths, polars = np.linspace(0.0, 2.0 * np.pi, 48), np.linspace(0.0, np.pi, 24)
    sphere = np.stack([np.outer(np.cos(azimuths), np.sin(polars)),
                       np.outer(np.sin(azimuths), np.sin(polars)),
                       np.outer(np.ones_like(azimuths), np.cos(polars))])

    def shell(mean, covariance, colour, order):
        points = mean[:, None, None] + np.einsum(
            "ij,jkl->ikl", np.linalg.cholesky(covariance), sphere)
        # Flat, not shaded: shading only darkens a translucent fill further.
        # Antialiasing off, or every facet edge of the translucent mesh draws as a seam.
        surface = axes.plot_surface(*points, color=colour, alpha=SHELL_ALPHA, shade=False,
                                    linewidth=0, antialiased=False, zorder=order)
        surface.set_clip_on(False)
        # Two outlines through the centre, in a darker shade of the fill so they hold up
        # against it: the horizontal cut at the mode's altitude, and the vertical cut along
        # its minor horizontal axis, which gives the shell its height profile.
        edge = tuple(0.65 * np.asarray(to_rgb(colour)))
        precision = np.linalg.inv(covariance)
        horizontal = np.linalg.inv(precision[:2, :2])
        ring = mean[:2, None] + SHELL_SIGMA * np.linalg.cholesky(horizontal) @ circle
        minor = np.r_[np.linalg.eigh(horizontal)[1][:, 0], 0.0]
        plane = np.column_stack([minor, [0.0, 0.0, 1.0]])
        section = np.linalg.cholesky(np.linalg.inv(plane.T @ precision @ plane))
        loop = mean[:, None] + SHELL_SIGMA * plane @ section @ circle
        axes.plot(ring[0], ring[1], np.full(theta.size, mean[2]), color=edge,
                  linewidth=0.9, zorder=order, clip_on=False)
        axes.plot(*loop, color=edge, linewidth=0.9, zorder=order, clip_on=False)
        axes.plot([mean[0]] * 2, [mean[1]] * 2, [floor, mean[2]], color=edge,
                  linestyle=":", linewidth=1.0, zorder=order, clip_on=False)

    def piece(points, order):
        axes.plot(points[:, 0], points[:, 1], points[:, 2], color=altitude_map(0.0),
                  linewidth=1.3, solid_capstyle="round", zorder=order, clip_on=False)

    drawables = [(float(p[:2] @ towards_camera) + p[2], pillar, tuple(p)) for p in pillars]
    drawables += [(float(m[:2] @ towards_camera), shell,
                   (m, c, MODE_COLOURS[j % len(MODE_COLOURS)]))
                  for j, (m, c) in enumerate(zip(means, covariances))]
    chunk = max(len(path) // 220, 2)
    for begin in range(0, len(path) - 1, chunk):
        points = path[begin:begin + chunk + 1]
        if hidden is not None:
            # matplotlib breaks a line at NaN, so a hidden point becomes a gap.
            points = np.where(hidden[begin:begin + chunk + 1, None], np.nan, points)
            if np.isnan(points).all():
                continue
        drawables.append((float(np.nanmean(points[:, :2], axis=0) @ towards_camera), piece,
                          (points,)))
    order = 3
    for order, (_, draw, args) in enumerate(sorted(drawables, key=lambda item: item[0]),
                                            start=3):
        draw(*args, order)
    return order + 1


def _draw_floor(axes, means, covariances, weights, limits) -> None:
    """The target's planar marginal on the floor, framed, under every volumetric view."""
    from matplotlib.colors import LinearSegmentedColormap

    floor = limits[2][0]
    x = np.linspace(*limits[0], 240)
    y = np.linspace(*limits[1], 120)
    grid = np.stack(np.meshgrid(x, y), axis=-1)
    density = sum(w * _planar_normal(grid, m[:2], c[:2, :2])
                  for w, m, c in zip(weights, means, covariances))
    floor_map = LinearSegmentedColormap.from_list(
        "_floor", _resolve_cmap("carbon")(np.linspace(0.16, 1.0, 9)))
    axes.contourf(x, y, density, levels=np.linspace(0.0, density.max(), 12), zdir="z",
                  offset=floor, cmap=floor_map, extend="max")
    (x0, x1), (y0, y1) = limits[0], limits[1]
    axes.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], np.full(5, floor),
              color=FLOOR_EDGE, linewidth=0.7, alpha=0.55)
    for collection in [*axes.collections, *axes.lines]:
        collection.set_clip_on(False)


def volumetric_snapshot(path, pillars, means, covariances, weights, limits, output: Path, *,
                        elevation: float = 38.0, azimuth: float = -90.0,
                        z_exaggeration: float = 2.0, dpi: int = 600) -> Path:
    """Render one volumetric run over its pillar field and the target's components.

Args:
        path: Executed positions, ``(N, 3)``.
        pillars: ``(n, 4)`` as ``(x, y, radius, top)``; a top above the ceiling is drawn to it.
        means: Target mode centres, ``(3, 3)``.
        covariances: Target covariances, ``(3, 3, 3)``; each component is drawn as a
            ``SHELL_SIGMA`` shell, and the floor shows the planar marginal.
        weights: Target weights, ``(3,)``.
        limits: Workspace bounds, ``(3, 2)``.
        output: Raster image path to write.
        elevation: Camera elevation in degrees.
        azimuth: Camera azimuth in degrees.
        z_exaggeration: Vertical exaggeration of the drawn box; state it in the caption.
        dpi: Raster resolution before cropping.
Returns:
        The path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    path, pillars = np.asarray(path, float), np.asarray(pillars, float)
    means, limits = np.asarray(means, float), np.asarray(limits, float)
    covariances = np.asarray(covariances, float)
    floor, ceiling = limits[2]
    with plt.rc_context({**style.paper_style("double"), "savefig.bbox": "standard"}):
        figure = plt.figure(figsize=(8.6, 7.4))
        axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

        _draw_floor(axes, means, covariances, weights, limits)
        (x0, x1), (y0, y1) = limits[0], limits[1]

        above = _draw_scene(axes, pillars, path, means, covariances, floor, ceiling, azimuth,
                            ListedColormap([VEHICLE_COLOUR]), 0.0, 1.0)
        _draw_quadrotor(axes, tuple(path[-1]), _heading(path[:, :2]), 1.5, VEHICLE_COLOUR,
                        linewidth=1.1, zorder=above)

        axes.set_xlim(x0, x1)
        axes.set_ylim(y0, y1)
        axes.set_zlim(floor, ceiling)
        axes.set_box_aspect((x1 - x0, y1 - y0, (ceiling - floor) * z_exaggeration))
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_position((0.06, 0.10, 0.88, 0.76))
        _strip_axes(axes)

        written = style.save(figure, output, dpi=dpi)
        plt.close(figure)
    # Near-zero side border, as the planar Fig. 7 panels: at column width every point of
    # horizontal padding is scene given away.
    return _crop_transparent(written, pad_fraction_x=0.005)


def _occluded(points: np.ndarray, body: np.ndarray, origin, pitch: float,
              towards: np.ndarray) -> np.ndarray:
    """Whether the ``(Z, Y, X)`` voxel body hides each point from an orthographic camera."""
    steps = np.arange(pitch, np.linalg.norm(np.asarray(body.shape) * pitch), pitch / 2)
    rays = points[:, None, :] + steps[None, :, None] * towards
    cell = np.floor((rays - np.asarray(origin)) / pitch).astype(int)
    inside = np.all((cell >= 0) & (cell < np.asarray(body.shape[::-1])), axis=-1)
    hit = np.zeros(inside.shape, bool)
    kept = cell[inside]
    hit[inside] = body[kept[:, 2], kept[:, 1], kept[:, 0]]
    return hit.any(axis=1)


def _lambert(normals: np.ndarray, towards: np.ndarray) -> np.ndarray:
    """Two-sided Lambert factor ``(F,)`` under a headlight ``BUNNY_LIGHT_RISE`` above the camera."""
    light = towards + [0.0, 0.0, BUNNY_LIGHT_RISE]
    return 0.45 + 0.55 * np.abs(normals @ (light / np.linalg.norm(light)))


def bunny_snapshot(path, vertices, faces, body, origin, pitch: float, output: Path, *,
                   azimuths=(-60.0, 120.0), elevation: float = 20.0, dpi: int = 600) -> Path:
    """One bunny run from opposite sides: the shaded scan, the path in black.

Args:
        path: Executed positions, ``(N, 3)``; the last is marked as the vehicle.
        vertices: Scan vertices, ``(V, 3)``.
        faces: Scan triangles, ``(F, 3)``.
        body: Solid voxels ``(Z, Y, X)`` from ``origin`` at ``pitch``, for occlusion.
        output: Raster image path to write.
        azimuths: One panel per camera azimuth, degrees.
        elevation: Camera elevation, degrees.
Returns:
        The path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    path = np.asarray(path, float)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    segments = np.stack([path[:-1], path[1:]], axis=1)
    everything = np.vstack([path, vertices])
    low, high = everything.min(axis=0), everything.max(axis=0)
    with plt.rc_context({**style.paper_style("double"), "savefig.bbox": "standard"}):
        figure = plt.figure(figsize=(4.3 * len(azimuths), 4.6))
        for index, azimuth in enumerate(azimuths):
            # Overlapping boxes, centres pulled in: a bare ortho scene fills only the middle
            # of its axes, and the boxes themselves are transparent.
            centre = 0.5 + (index - (len(azimuths) - 1) / 2) * 0.36
            axes = figure.add_axes((centre - 0.31, 0.20, 0.62, 0.80), projection="3d",
                                   computed_zorder=False)
            axes.set_proj_type("ortho")
            a, e = np.deg2rad(azimuth), np.deg2rad(elevation)
            towards = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            hidden = _occluded(path, body, origin, pitch, towards)
            seen = ~(hidden[:-1] | hidden[1:])
            # Two-sided: the scan's winding is not guaranteed consistent.
            colours = np.clip(_lambert(normals, towards)[:, None] * np.asarray(to_rgb(BUNNY_GREY)),
                              0.0, 1.0)
            axes.plot([low[0], high[0], high[0], low[0], low[0]],
                      [low[1], low[1], high[1], high[1], low[1]], np.zeros(5),
                      color=FLOOR_EDGE, linewidth=0.7, alpha=0.55, zorder=1)
            mesh = Poly3DCollection(triangles, facecolors=colours, edgecolors="none",
                                    linewidths=0)
            mesh.set_zorder(2)
            axes.add_collection3d(mesh)
            axes.add_collection3d(Line3DCollection(
                segments[seen], colors=VEHICLE_COLOUR, linewidths=0.7, zorder=3))
            axes.scatter(*path[-1], s=40, color=VEHICLE_COLOUR, linewidths=0,
                         depthshade=False, zorder=4)
            axes.set_xlim(low[0], high[0])
            axes.set_ylim(low[1], high[1])
            axes.set_zlim(0.0, high[2])
            axes.set_box_aspect((high[0] - low[0], high[1] - low[1], high[2]))
            axes.view_init(elev=elevation, azim=azimuth)
            _strip_axes(axes)
        written = style.save(figure, output, dpi=dpi)
        plt.close(figure)
    return _crop_transparent(written, pad_fraction_x=0.005)


def _obstacle_cmap():
    """Return the obstacle height colormap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    name, low, high = OBSTACLE_RAMP
    return LinearSegmentedColormap.from_list(f"_{name}",
                                             plt.get_cmap(name)(np.linspace(low, high, 64)))


def perlin_snapshot(path, cloud, means, covariances, weights, limits, body, origin,
                    pitch: float, output: Path, *, azimuth: float = -75.0,
                    elevation: float = 30.0, dpi: int = 600) -> Path:
    """One frame of a run through a noise map drawn as a point cloud, as RViz shows one.

Args:
        path: Executed positions up to this frame, ``(N, 3)``.
        cloud: Occupied cell centres, ``(M, 3)``.
        means, covariances, weights: The target, drawn as in `volumetric_snapshot`.
        limits: Workspace bounds, ``(3, 2)``.
        body: Solid voxels ``(Z, Y, X)`` from ``origin`` at ``pitch``, for occlusion.
        output: Raster image path to write.
Returns:
        The path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    path, cloud = np.asarray(path, float), np.asarray(cloud, float)
    means, limits = np.asarray(means, float), np.asarray(limits, float)
    covariances = np.asarray(covariances, float)
    floor, ceiling = limits[2]
    (x0, x1), (y0, y1) = limits[0], limits[1]
    a, e = np.deg2rad(azimuth), np.deg2rad(elevation)
    towards = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    hidden = _occluded(path, body, origin, pitch, towards)
    with plt.rc_context({**style.paper_style("double"), "savefig.bbox": "standard"}):
        figure = plt.figure(figsize=(8.6, 7.4))
        axes = figure.add_subplot(111, projection="3d", computed_zorder=False)
        axes.set_proj_type("ortho")
        _draw_floor(axes, means, covariances, weights, limits)
        segments = np.stack([path[:-1], path[1:]], axis=1)
        behind = hidden[:-1] | hidden[1:]
        under = Line3DCollection(segments[behind], colors=VEHICLE_COLOUR, linewidths=1.3)
        under.set_zorder(1.5)
        under.set_clip_on(False)
        axes.add_collection3d(under)
        heights = np.clip((cloud[:, 2] - floor) / (ceiling - floor), 0.0, 1.0)
        points = axes.scatter(*cloud.T, c=_obstacle_cmap()(heights), s=OBSTACLE_MARKER,
                              marker="s", alpha=OBSTACLE_ALPHA, linewidths=0, depthshade=False,
                              zorder=2)
        points.set_clip_on(False)
        above = _draw_scene(axes, np.zeros((0, 4)), path, means, covariances, floor, ceiling,
                            azimuth, ListedColormap([VEHICLE_COLOUR]), 0.0, 1.0, hidden)
        _draw_quadrotor(axes, tuple(path[-1]), _heading(path[:, :2]), 1.5, VEHICLE_COLOUR,
                        linewidth=1.1, zorder=above)
        axes.set_xlim(x0, x1)
        axes.set_ylim(y0, y1)
        axes.set_zlim(floor, ceiling)
        axes.set_box_aspect((x1 - x0, y1 - y0, ceiling - floor))
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_position((0.06, 0.10, 0.88, 0.76))
        _strip_axes(axes)
        written = style.save(figure, output, dpi=dpi)
        plt.close(figure)
    return _crop_transparent(written, pad_fraction_x=0.005)

