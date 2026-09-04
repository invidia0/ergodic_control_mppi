"""Oblique 3D view of a flown trial: pillars, flown path, target modes, MPPI rollouts.

Rendered from the recorded arrays rather than screenshotted from RViz, so it is
reproducible headlessly and versioned with the run it depicts. The camera looks from a
long side of the workspace obliquely down onto a corner of the workspace, high enough
that the path stays visible between the pillars and oblique enough that obstacle height and
flight altitude read as three-dimensional.

Obstacle footprints come from the *raw* occupancy, never the inflated planning grid. The
pillars stand *on* the target-density plane and rise above it, which is the geometry the
vehicle actually flies: it holds a fixed altitude inside a field of 2-3 m obstacles, so it
threads between them rather than passing over them.

Their drawn height is the shortest pillar the manifest guarantees, not a recovered one --
the archived occupancy is a single planar slice, so the individual heights are not in the
data. Every pillar really is at least that tall; none is drawn taller than the map allows.
"""

from pathlib import Path

import numpy as np

from ergodic_control_mppi.plotting import style

# Obliquely down onto a corner. 90 elevation would be a plan view with no height cue at
# all; 45 keeps the pillars readable as volumes without the near ones hiding the path.
#
# The azimuth is deliberately off the -90 axis: looking straight down -y maps both y and z
# to pure vertical screen motion, so mplot3d stacks their ticks in one corner and drops the
# y label entirely. -60 separates the two axes into their own screen directions.
DEFAULT_ELEVATION = 45.0
DEFAULT_AZIMUTH = -60.0

# The workspace outline on the density plane: a pale grey, present but never competing
# with the modes it frames.
FLOOR_EDGE = "#9AA1AC"



def _cell_centres(occupancy: np.ndarray, origin, resolution: float) -> np.ndarray:
    """World ``(x, y)`` of every occupied cell centre, shape ``(N, 2)``."""
    rows, columns = np.nonzero(occupancy)
    return np.column_stack(
        [
            origin[0] + (columns + 0.5) * resolution,
            origin[1] + (rows + 0.5) * resolution,
        ]
    )


def _draw_pillars(axes, centres: np.ndarray, resolution: float, base: float, top: float) -> None:
    """Draw each occupied cell as a box standing from ``base`` up to ``top``."""
    half = 0.5 * resolution
    for x, y in centres:
        corners = np.array(
            [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]
        )
        # Sides, then the cap. Drawing the cap last keeps it visible from above.
        for index in range(4):
            first, second = corners[index], corners[(index + 1) % 4]
            axes.plot_surface(
                np.array([[first[0], second[0]], [first[0], second[0]]]),
                np.array([[first[1], second[1]], [first[1], second[1]]]),
                np.array([[base, base], [top, top]]),
                color=style.NEUTRAL, alpha=0.55, linewidth=0, shade=True,
            )
        axes.plot_surface(
            np.array([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]]),
            np.array([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]]),
            np.full((2, 2), top),
            color=style.NEUTRAL, alpha=0.8, linewidth=0, shade=True,
        )


def _pillar_height(run_directory: Path, fallback: float = 2.0) -> float:
    """Shortest pillar the manifest guarantees, so the drawing cannot overstate the map.

    ``map_parameters.pillar_height_m`` is the generator's ``[min, max]`` range. Every
    pillar is at least the minimum, so drawing them all at that height is true of the
    whole field; the archived occupancy is a planar slice, so the individual heights that
    would let each be drawn exactly are not recoverable.
    """
    import json

    manifest = run_directory / "manifest.json"
    if not manifest.exists():
        return fallback
    heights = json.loads(manifest.read_text(encoding="utf-8")).get(
        "map_parameters", {}
    ).get("pillar_height_m")
    return float(heights[0]) if heights else fallback


def snapshot(
    run_directory: Path,
    output: Path,
    *,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    altitude: float = 0.75,
    pillar_height: float | None = None,
    z_exaggeration: float = 3.0,
    rollout_step: int | None = None,
    max_rollouts: int = 60,
    size: str = "double",
):
    """Render the oblique snapshot for one recorded run.

    Args:
        run_directory: Recorder output holding ``arrays.npz``; ``figure_data.npz`` is used
            too when present, for the rollout overlay.
        output: Image path to write.
        elevation: Camera elevation in degrees; 90 is straight down.
        azimuth: Camera azimuth in degrees; keep it off -90, which degenerates y and z
            onto the same screen direction.
        altitude: Flight altitude, for the path's z. Also the plane the pillars stand on,
            so the vehicle reads as threading between them rather than flying over them.
        pillar_height: Height above ``altitude`` to extrude the exact planar footprints.
            Defaults to the shortest pillar the run's manifest guarantees.
        z_exaggeration: Vertical exaggeration factor of the rendered box, literally: at
            3.0 a metre of height draws three times as long as a metre of ground. Needed
            because at true scale a 2 m pillar over a 40 m span vanishes. State it in the
            caption.
        rollout_step: Which recorded snapshot to draw rollouts from; ``None`` picks the
            middle one. Ignored when no ``figure_data.npz`` exists.
        max_rollouts: How many sampled rollouts to draw.
        size: Key into ``style.FIGSIZES``; "double" is the two-column figure width.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = np.load(run_directory / "arrays.npz", allow_pickle=False)
    occupancy = np.asarray(arrays["occupancy"]).astype(bool)
    origin = tuple(float(v) for v in np.asarray(arrays["grid_origin"]))
    resolution = float(arrays["grid_resolution"])
    # Odometry columns are [t, x, y, z, ...]; the flown path is the executed one.
    odometry = np.asarray(arrays["odometry"])
    path = odometry[:, 1:3]

    with plt.rc_context(style.paper_style(size)):
        figure = plt.figure(figsize=style.FIGSIZES[size])
        axes = figure.add_subplot(111, projection="3d")

        # Pillars stand on the target-density plane the vehicle flies in, rising above it.
        # Drawn from z=0 they would sit under the path and the vehicle would read as
        # flying over the field instead of avoiding it.
        top = altitude + (
            _pillar_height(run_directory) if pillar_height is None else pillar_height
        )
        _draw_pillars(axes, _cell_centres(occupancy, origin, resolution), resolution,
                      altitude, top)

        # Flown path, coloured by time so the tour order is readable.
        points = np.column_stack([path, np.full(path.shape[0], altitude)])
        stride = max(1, points.shape[0] // 4000)
        shown = points[::stride]
        axes.scatter(
            shown[:, 0], shown[:, 1], shown[:, 2],
            c=np.linspace(0.0, 1.0, shown.shape[0]), cmap=style.TRAIL_CMAP,
            s=1.2, linewidths=0, depthshade=False,
        )

        figure_data = run_directory / "figure_data.npz"
        if figure_data.exists():
            _overlay_rollouts(axes, figure_data, altitude, rollout_step, max_rollouts)

        _draw_modes(axes, run_directory, altitude)

        _configure_axes(axes, occupancy, origin, resolution, altitude, top,
                        elevation, azimuth, z_exaggeration)
        return style.save(figure, output)


def _configure_axes(axes, occupancy, origin, resolution, altitude, top,
                    elevation, azimuth, z_exaggeration, bare: bool = False) -> None:
    """Apply the shared camera, box aspect, limits and label workarounds."""
    import matplotlib.pyplot as plt

    x_limits = (origin[0], origin[0] + occupancy.shape[1] * resolution)
    y_limits = (origin[1], origin[1] + occupancy.shape[0] * resolution)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    # Start at the flight plane, not at 0: nothing is drawn below it, and the empty
    # band cost a third of the axis height and put a third z tick right on top of the
    # y ticks, which share that corner of the projection.
    axes.set_zlim(altitude, top)
    # x:y is true metric proportion (2:1 here) so the workspace is not read as square.
    # z is deliberately exaggerated -- at true scale a 2 m pillar over a 40 m span is
    # 5% of the width and vanishes.
    #
    # Because x is one box unit per metre, making the z extent `z_exaggeration` box
    # units per metre of drawn height makes `z_exaggeration` the vertical exaggeration
    # factor itself, which is the number the caption has to state.
    span = x_limits[1] - x_limits[0]
    axes.set_box_aspect(
        (span, y_limits[1] - y_limits[0], (top - altitude) * z_exaggeration)
    )
    axes.view_init(elev=elevation, azim=azimuth)
    # Fill the canvas: the default 3D axes box leaves half the figure empty at this
    # box aspect, and it fixes where the two manual axis labels below have to go.
    #
    # Bare renders get the whole canvas. The 14% reserved on the right is for the two
    # manual y/z labels, which `_strip_axes` removes -- and the crop afterwards is a pixel
    # operation, so every fraction of the canvas the scene does not occupy is resolution
    # thrown away. At the old (0.86, 0.96) box the cropped scene came out near 750 px from
    # a 2070 px canvas, which is where the pixelated look came from.
    # Inset rather than filling the canvas, for the same reason: mplot3d happily
    # draws outside its own axes box, so the slack is what keeps the scene whole.
    # `_crop_transparent` reclaims whatever is unused.
    axes.set_position((0.06, 0.06, 0.88, 0.88) if bare else (0.0, 0.02, 0.86, 0.96))
    axes.set_xlabel("x [m]", labelpad=14)
    # mplot3d reports these two labels as visible and positioned, then draws
    # neither, at any labelpad or axes position -- only the x label survives this
    # projection. Place them on the figure instead, beside the tick columns they
    # belong to. Positions are tied to the default camera; a caller overriding
    # elevation or azimuth should expect to move them.
    axes.set_ylabel("")
    axes.set_zlabel("")
    # Axes coordinates, not figure ones: paper_style saves with a tight bounding
    # box, so figure-fraction text placed past the content just grows the canvas.
    axes.text2D(1.02, -0.02, "y [m]", transform=axes.transAxes,
                ha="center", va="center")
    axes.text2D(1.00, 0.52, "z [m]", transform=axes.transAxes,
                ha="center", va="center", rotation=90)
    # paper_style turns on minor ticks, which crowd unreadably in a projected 3D axis.
    # The flight/density plane and the guaranteed pillar top: the two heights a reader
    # needs to see that the vehicle passes between the obstacles rather than over them.
    axes.set_zticks([altitude, top])
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_minor_locator(plt.NullLocator())
        axis.pane.set_alpha(0.0)
    axes.tick_params(axis="z", pad=-2)
    axes.grid(False)


def _pillar_cloud(centres: np.ndarray, resolution: float, base: float, top: float,
                  density: int = 3):
    """Fill each occupied cell with a column of points, as the simulator's cloud is.

    The archived occupancy is a planar slice, so the columns are synthesised at the
    guaranteed height rather than recovered -- the same caveat the extruded boxes carry.

    ``density`` is how many samples span one cell pitch in each of the three directions.
    At 1 the cloud is one column of beads per occupied cell, which is what made the
    columns read as stacks of dots rather than surfaces; at 3 each cell contributes a
    3x3 lattice of columns at a third of the pitch, so neighbouring markers overlap and
    close up. Cost is cubic in ``density`` -- 25 pillars at 3 is ~200k points, which
    mplot3d still depth-sorts in about a second, and 4 is the practical ceiling.
    """
    step = resolution / max(density, 1)
    # Sub-cell offsets centred on the cell, so the lattice stays inside the footprint the
    # occupancy actually claims and the pillars do not fatten as the density rises.
    offsets = (np.arange(density) - 0.5 * (density - 1)) * step
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    planar = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    levels = np.arange(base, top + 0.5 * step, step)

    spots = (centres[:, None, :] + planar[None, :, :]).reshape(-1, 2)
    x = np.repeat(spots[:, 0], levels.size)
    y = np.repeat(spots[:, 1], levels.size)
    z = np.tile(levels, spots.shape[0])
    return x, y, z


def _draw_quadrotor(axes, centre, heading: float, span: float, colour: str,
                    linewidth: float, zorder: float = 5.0) -> None:
    """A quadrotor glyph: four arms on the diagonals, four rotor discs, a body hub.

    Drawn rather than meshed. The simulator's airframe ships as a 6.4 MB Ogre binary that
    matplotlib cannot read, and at this scale the silhouette is all that survives anyway:
    the real hummingbird is ~0.55 m tip to tip across a 40 m workspace, so any faithful
    render is sub-pixel. ``span`` is therefore an exaggeration like ``z_exaggeration`` and
    has to be stated in the caption.

    Called last and with ``computed_zorder=False`` in force, so the glyph sits above every
    pillar instead of being buried by whichever column happens to be nearer the camera.
    """
    x, y, z = centre
    arm = 0.5 * span
    # Diagonals, i.e. the "X" airframe, offset by the travel heading so the glyph points
    # along the path rather than along the world axes.
    angles = heading + np.pi / 4 + np.arange(4) * (np.pi / 2)
    hubs = np.column_stack([x + arm * np.cos(angles), y + arm * np.sin(angles)])
    for hub in hubs:
        axes.plot([x, hub[0]], [y, hub[1]], [z, z],
                  color=colour, linewidth=linewidth, solid_capstyle="round",
                  zorder=zorder)

    # Rotor discs as outlines in the flight plane. A scatter marker would keep a fixed
    # pixel size and drift out of proportion with the arms as the figure is scaled.
    circle = np.linspace(0.0, 2.0 * np.pi, 32)
    radius = 0.30 * span
    for hub in hubs:
        axes.plot(hub[0] + radius * np.cos(circle), hub[1] + radius * np.sin(circle),
                  np.full(circle.size, z), color=colour, linewidth=0.8 * linewidth,
                  zorder=zorder)
    axes.scatter(hubs[:, 0], hubs[:, 1], np.full(4, z), s=(2.2 * linewidth) ** 2,
                 color=colour, depthshade=False, linewidths=0, zorder=zorder + 1)
    axes.scatter([x], [y], [z], s=(4.5 * linewidth) ** 2, color=colour,
                 depthshade=False, linewidths=0, zorder=zorder + 1)


def trajectory_snapshot(
    positions: np.ndarray,
    map_source: Path,
    output: Path,
    *,
    title: str | None = None,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    altitude: float = 0.75,
    pillar_height: float | None = None,
    z_exaggeration: float = 3.0,
    density_levels: int = 12,
    point_size: float = 1.0,
    pillar_cmap: str = "turbo_r",
    density_cmap: str = "Blues",
    pillar_alpha: float = 1.0,
    pillar_style: str = "points",
    trail_size: float = 1.6,
    vehicle_span: float = 1.5,
    trail_colour: str = "#4a4f59",
    vehicle_colour: str = "#111111",
    cloud_density: int = 3,
    dpi: int = 600,
    flight_fraction: float = 0.5,
    bare: bool = False,
    size: str = "double",
    gmm=None,
):
    """Render an offline trajectory over the pillar cloud and the target density.

    Unlike ``snapshot`` this takes a bare path array, so it draws runs the sweep produced
    but never archived. Pillars are drawn as a height-coloured point cloud rather than
    extruded boxes, matching how the field appears in the simulator; the target density is
    a filled contour on the plane the vehicle flies in; the trail is one solid colour, so
    it shows where the vehicle went rather than encoding time it cannot also show.

    Args:
        positions: Executed planar path, shape ``(N, 2)``.
        map_source: Run directory holding the ``arrays.npz`` whose map this path was run on.
        output: Image path to write.
        title: Optional heading, e.g. the arm name.
        elevation: Camera elevation in degrees.
        azimuth: Camera azimuth in degrees. -90 puts the long side of the workspace
            parallel to the bottom edge; it degenerates y and z onto one screen direction,
            which only matters when ``bare`` is False and the ticks have to be readable.
        altitude: Target-density plane, and the base the pillars stand on.
        pillar_height: Height above ``altitude``; defaults to the manifest's guarantee.
        z_exaggeration: Vertical exaggeration factor; state it in the caption.
        density_levels: Filled contour levels for the target density.
        point_size: Marker area for the pillar cloud points.
        trail_size: Marker area for the trail points; large enough that they close into
            a continuous line at the sampled spacing.
        vehicle_span: Tip-to-tip width of the drawn quadrotor, in metres. An exaggeration
            like ``z_exaggeration``: the real airframe is ~0.55 m across a 40 m workspace
            and would be sub-pixel. State it in the caption.
        trail_colour: Trail ink. Pure black buried the path in the pillar cloud; a slate
            grey separates from the ``turbo_r`` columns without going pale.
        vehicle_colour: Quadrotor glyph ink. Black reads against both the pale density
            plane and the pillar cloud, which no single hue in the ``turbo_r`` ramp does.
        cloud_density: Samples per cell pitch in each direction for the pillar cloud. 1 is
            one bead column per occupied cell; 3 closes the columns into surfaces.
        dpi: Raster resolution. High by default because a bare render is cropped to the
            scene afterwards, so the saved pixels are only the fraction the scene occupies.
        flight_fraction: Where the trail is drawn between the density plane and the
            pillar tops. Presentational: the deployment is planar, so the executed path
            has no height of its own, and drawing it mid-column is what makes the vehicle
            read as flying *between* the pillars rather than skimming the floor. State it
            in the caption; it is not a flown altitude.
        bare: Strip every tick, label and axis line, leaving only the scene.
        size: Key into ``style.FIGSIZES``.
        gmm: Target mixture to contour. Pass it when the map directory carries no
            sibling profile YAML, which is the case for the campaign's maps.

    Returns:
        The path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    arrays = np.load(map_source / "arrays.npz", allow_pickle=False)
    occupancy = np.asarray(arrays["occupancy"]).astype(bool)
    origin = tuple(float(v) for v in np.asarray(arrays["grid_origin"]))
    resolution = float(arrays["grid_resolution"])
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    top = altitude + (
        _pillar_height(map_source) if pillar_height is None else pillar_height
    )

    with plt.rc_context(style.paper_style(size)):
        # mplot3d fits the projected box to the *shorter* side of the axes, so on the
        # paper's 6.9x2.6 canvas the scene was pinned by the 2.6 in height and left three
        # quarters of the width empty. A bare render is cropped to the scene and carries no
        # type, so its canvas is free to be shaped for pixels rather than for the page --
        # the figure it lands in is sized by LaTeX regardless. Labelled renders keep the
        # paper size, because there the type has to come out at its stated point size.
        # Wider than the page: the bare render is cropped to its content afterwards,
        # so the canvas only has to be large enough that no part of the projection
        # falls off it. At an oblique azimuth the rotated floor corners reach well
        # past the axes box and were being clipped by the figure edge before the
        # alpha crop ever ran, which is what cut the near edge of the field.
        canvas = (8.6, 6.4) if bare else style.FIGSIZES[size]
        figure = plt.figure(figsize=canvas)
        # computed_zorder=False: by default mplot3d overrides call order with each artist's
        # centroid depth, which buries the vehicle dot inside the cloud whatever order it
        # is added in. Off, the three layers stack as written. Points *within* a collection
        # are still depth-sorted, which is what the merged scatter below relies on.
        axes = figure.add_subplot(111, projection="3d", computed_zorder=False)

        # Footprints first: the floor border below needs their extent, and the cylinder
        # renderer needs the components themselves, so the labelling runs once.
        centres = _cell_centres(occupancy, origin, resolution)
        components = (_cylinder_components(centres, resolution)
                      if pillar_style == "cylinders" and centres.size else [])

        # Target density first, on the floor, so the cloud and trail draw over it.
        density, extent = _target_density(map_source, occupancy, origin, resolution, gmm)
        if density is not None:
            axes.contourf(
                *extent, density, levels=density_levels, zdir="z", offset=altitude,
                cmap=_resolve_cmap(density_cmap),
            )
        # The workspace boundary, drawn on the same plane. The density fades to the page
        # long before the map ends, so without it the floor has no edge and the pillars at
        # the rim look like they are standing on nothing.
        #
        # Widened to contain the pillars as *drawn*. Every pillar is inside the workspace
        # in the data, but a cylinder's radius is inflated by half a cell for looks, so a
        # pillar built on the boundary cells overhangs the true extent by up to that much
        # and would otherwise stand across its own floor's edge.
        edge_x = [origin[0], origin[0] + occupancy.shape[1] * resolution]
        edge_y = [origin[1], origin[1] + occupancy.shape[0] * resolution]
        for (middle, radius) in components:
            edge_x[0] = min(edge_x[0], middle[0] - radius)
            edge_x[1] = max(edge_x[1], middle[0] + radius)
            edge_y[0] = min(edge_y[0], middle[1] - radius)
            edge_y[1] = max(edge_y[1], middle[1] + radius)
        axes.plot(
            [edge_x[0], edge_x[1], edge_x[1], edge_x[0], edge_x[0]],
            [edge_y[0], edge_y[0], edge_y[1], edge_y[1], edge_y[0]],
            np.full(5, altitude), color=FLOOR_EDGE, linewidth=0.7, alpha=0.55, zorder=1,
        )

        # Height-coloured point cloud, as the simulator renders the pillar field. A rainbow
        # ramp is a poor quantitative colormap but this axis is a depth cue, not a
        # measurement, and matching the simulator is what makes the two views comparable.
        colour_map = _resolve_cmap(pillar_cmap)
        if pillar_style == "cylinders":
            # Surfaces, not points. mplot3d painter-sorts whole artists, so the trail is
            # drawn separately and *segmented*: one artist per short run, each sorted on its
            # own centroid, which restores most of the interleaving the merged scatter gives
            # for free. It is an approximation -- a segment straddling a pillar still lands
            # wholly in front or behind -- so this style is offered as an alternative look,
            # not as a replacement for the point cloud.
            flight = altitude + flight_fraction * (top - altitude)
            above = _draw_cylinder_scene(
                axes, components, altitude, top, colour_map, pillar_alpha,
                azimuth, positions, flight_fraction, trail_colour, trail_size,
            )
            # Explicitly above the whole depth stack: the vehicle marks where the run ended
            # and is the one thing that must never be occluded, least of all by its own trail.
            _draw_quadrotor(axes, (positions[-1, 0], positions[-1, 1], flight),
                            _heading(positions), vehicle_span, vehicle_colour,
                            linewidth=1.1, zorder=above)
            _configure_axes(axes, occupancy, origin, resolution, altitude, top,
                            elevation, azimuth, z_exaggeration, bare=bare)
            if bare:
                _strip_axes(axes)
            if title:
                axes.set_title(title, pad=0.0)
            written = style.save(figure, output, dpi=dpi)
            plt.close(figure)
            return _crop_transparent(written, pad_fraction_x=0.005) if bare else written

        x, y, z = _pillar_cloud(
            centres, resolution, altitude, top, density=cloud_density,
        )

        # Trail and cloud go in as one scatter. mplot3d cannot occlude one artist by
        # another per-fragment -- whichever is drawn second wins everywhere they overlap,
        # so a separate trail either floats over the whole field or vanishes behind all of
        # it. Within a single collection the points *are* depth-sorted, so merging them is
        # what makes the path pass between the columns instead of in front of or behind
        # them. The cost is that the trail is a dense run of dots rather than a stroked
        # line; at the sampled spacing it closes up into one.
        flight = altitude + flight_fraction * (top - altitude)
        colours = np.vstack(
            [
                colour_map((z - altitude) / max(top - altitude, 1e-9))
                * np.array([1.0, 1.0, 1.0, pillar_alpha]),
                np.tile(to_rgba(trail_colour), (positions.shape[0], 1)),
            ]
        )
        axes.scatter(
            np.concatenate([x, positions[:, 0]]),
            np.concatenate([y, positions[:, 1]]),
            np.concatenate([z, np.full(positions.shape[0], flight)]),
            c=colours,
            s=np.concatenate(
                [np.full(x.size, point_size), np.full(positions.shape[0], trail_size)]
            ),
            linewidths=0, depthshade=False,
        )

        # Last, so no column can hide it. Heading from the final leg of the path, so the
        # airframe points along travel; a stationary end state falls back to +x.
        step = positions[-1] - positions[max(positions.shape[0] - 20, 0)]
        heading = float(np.arctan2(step[1], step[0])) if np.hypot(*step) > 1e-6 else 0.0
        _draw_quadrotor(axes, (positions[-1, 0], positions[-1, 1], flight), heading,
                        vehicle_span, vehicle_colour, linewidth=1.1)

        _configure_axes(axes, occupancy, origin, resolution, altitude, top,
                        elevation, azimuth, z_exaggeration, bare=bare)
        if bare:
            _strip_axes(axes)
        if title:
            axes.set_title(title, pad=0.0)
        written = style.save(figure, output, dpi=dpi)
        return _crop_transparent(written) if bare else written


def _resolve_cmap(name: str):
    """Look a colormap up in matplotlib, falling back to the Scientific Colour Maps.

    ``pillar`` and ``carbon`` are this paper's own ramps, defined in ``style``.
    Crameri's maps (batlow, acton, oslo, devon, ...) are perceptually uniform and
    colour-vision safe, which the default rainbow ramp is not. They ship in `cmcrameri`
    under a ``cmc.`` prefix; accept the bare name too so ``--pillar-cmap batlow`` works.
    """
    import matplotlib.pyplot as plt

    # This module's own ramps first: matplotlib would not know them, and registering into
    # its global namespace to look them up would be a side effect on import.
    local = {"pillar": style.PILLAR_CMAP, "carbon": style.DENSITY_CMAP}
    if name in local:
        return local[name]
    try:
        return plt.get_cmap(name)
    except (ValueError, KeyError):
        pass
    from cmcrameri import cm as crameri

    return getattr(crameri, name.removeprefix("cmc."))


def _heading(positions: np.ndarray) -> float:
    """Travel direction from the tail of the path."""
    tail = positions[-min(20, len(positions)):]
    delta = tail[-1] - tail[0]
    return float(np.arctan2(delta[1], delta[0]))


def _cylinder_components(centres, resolution):
    """Connected pillar footprints as ``(centre_xy, radius)``, one per pillar.

    The occupancy grid is square cells, so the columns it produces are square; fitting a
    circle to each connected component recovers the geometry the map generator sampled.
    """
    from scipy import ndimage

    pitch = resolution
    keys = np.round(centres / pitch).astype(int)
    grid = np.zeros(keys.max(axis=0) - keys.min(axis=0) + 3, dtype=bool)
    offset = keys.min(axis=0) - 1
    grid[tuple((keys - offset).T)] = True
    labels, count = ndimage.label(grid)
    out = []
    for index in range(1, count + 1):
        cells = np.argwhere(labels == index) + offset
        middle = cells.mean(axis=0) * pitch
        radius = max(np.max(np.linalg.norm(cells * pitch - middle, axis=1)), pitch) \
            + 0.5 * pitch
        out.append((middle, radius))
    return out


def _draw_cylinder_scene(axes, components, base, top, colour_map, alpha,
                         azimuth, positions, flight_fraction, trail_colour, trail_size,
                         edge: str = "#8A93A6", trail_width: float = 0.82):
    """Capped cylinders and the trail, drawn back to front in one depth order.

    Why surfaces rather than the point cloud: the cloud's cap is a lattice of markers
    spanning a fraction of a cell, so at an azimuth near -90 it projects to a sliver a
    couple of pixels tall and every column ends in a flat edge. Zooming does not help --
    the sliver scales with the picture. A real top face and a drawn silhouette read at any
    camera, which is what gives these their solid, game-like geometry.

    Why one combined sort: the panel runs with ``computed_zorder=False``, so mplot3d does
    not depth-sort artists at all and every line would otherwise draw over every surface --
    a pillar at the back putting its outline straight through one at the front. Depth along
    the view direction is the painter's order, and the trail is cut into short pieces and
    sorted into the *same* sequence, so it passes behind near pillars and in front of far
    ones instead of floating over the whole field or hiding behind it.
    """
    view = np.deg2rad(azimuth)
    towards_camera = np.array([np.cos(view), np.sin(view)])
    # Fine enough that neither the silhouette nor the vertical ramp shows its facets:
    # at 48 segments the rim was visibly polygonal and 14 colour bands striped the tube.
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    levels = np.linspace(base, top, 96)
    # The two generators bounding a vertical cylinder on screen, for this camera.
    silhouette = (view + 0.5 * np.pi, view - 0.5 * np.pi)
    height = base + flight_fraction * (top - base)

    def draw_pillar(middle, radius, order):
        circle_x = middle[0] + radius * np.cos(theta)
        circle_y = middle[1] + radius * np.sin(theta)
        shade = colour_map((levels - base) / max(top - base, 1e-9))
        axes.plot_surface(
            np.tile(circle_x, (levels.size, 1)), np.tile(circle_y, (levels.size, 1)),
            np.tile(levels[:, None], (1, theta.size)),
            facecolors=np.tile(shade[:, None, :], (1, theta.size, 1)),
            shade=False, linewidth=0, edgecolor="none", antialiased=False,
            alpha=alpha, zorder=order,
        )
        span = np.linspace(0.0, 1.0, 2)
        cap_x = middle[0] + np.outer(span, radius * np.cos(theta))
        cap_y = middle[1] + np.outer(span, radius * np.sin(theta))
        # linewidth 0 is not enough on its own: plot_surface still strokes each quad in
        # its face colour, which at this mesh density reads as a wireframe over the tube.
        axes.plot_surface(cap_x, cap_y, np.full_like(cap_x, top),
                          color=colour_map(1.0), shade=False, linewidth=0,
                          edgecolor="none", antialiased=False, alpha=alpha,
                          zorder=order + 0.1)
        # Both rims and the silhouette, in a soft grey: enough to say "this is a solid
        # volume" and not enough to draw the eye off the trail. Lighter than this and the
        # base arc stops reading as the pillar's foot and starts reading as a pale line
        # drawn across it. The base rim matters as much as the top one -- without it a
        # pillar reads as fading into the density plane rather than standing on it.
        axes.plot(circle_x, circle_y, np.full_like(circle_x, top),
                  color=edge, linewidth=0.5, alpha=0.55, zorder=order + 0.2)
        # Only the near arc of the base: the far half is hidden by the tube itself, and
        # drawing the whole ellipse puts it straight through the pillar.
        near = np.cos(theta - view) > 0.0
        axes.plot(circle_x[near], circle_y[near], np.full(near.sum(), base),
                  color=edge, linewidth=0.5, alpha=0.55, zorder=order + 0.2)
        for angle in silhouette:
            edge_x = middle[0] + radius * np.cos(angle)
            edge_y = middle[1] + radius * np.sin(angle)
            axes.plot([edge_x, edge_x], [edge_y, edge_y], [base, top],
                      color=edge, linewidth=0.5, alpha=0.55, zorder=order + 0.2)

    def draw_trail(piece, order):
        axes.plot(piece[:, 0], piece[:, 1], np.full(len(piece), height),
                  color=trail_colour, linewidth=trail_size * trail_width,
                  solid_capstyle="round", zorder=order)

    drawables = []
    for middle, radius in components:
        # Nearest point of the footprint, not its centre: a wide pillar the trail passes
        # beside should sort on the side facing the camera.
        depth = float(middle @ towards_camera) + radius
        drawables.append((depth, draw_pillar, (middle, radius)))
    chunk = max(len(positions) // 220, 2)
    for begin in range(0, len(positions) - 1, chunk):
        piece = positions[begin:begin + chunk + 1]
        drawables.append((float(piece.mean(axis=0) @ towards_camera), draw_trail, (piece,)))

    order = 3
    for order, (_, draw, args) in enumerate(
        sorted(drawables, key=lambda item: item[0]), start=3
    ):
        draw(*args, order)
    # The caller stacks the vehicle on top of this, and the stack is as deep as the scene.
    return order + 1


def _crop_transparent(path: Path, pad_fraction: float = 0.02,
                      pad_fraction_x: float | None = None) -> Path:
    """Trim the fully transparent border off a saved figure, leaving a small one.

    A bare 3D axis fills the canvas with an invisible projection box, so matplotlib's tight
    bounding box has nothing to crop against and leaves the scene floating in a wide
    margin. With the panels transparent, the alpha channel gives the true extent.

    Raster only, and silently so: this is a pixel operation, and a vector output has no
    alpha channel to measure. PDF and SVG are returned untouched rather than handed to PIL,
    which would raise. A point-cloud scene of this size should be a high-DPI PNG anyway --
    as vector it carries a quarter of a million individual point paths.
    """
    if path.suffix.lower() not in {".png", ".tif", ".tiff", ".webp"}:
        return path
    from PIL import Image

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        box = rgba.getchannel("A").getbbox()
        if box:
            crop = rgba.crop(box)
            # Cropping to the alpha bbox alone puts the outermost pillar caps hard against
            # the border, which reads as the scene being cut off rather than framed. The
            # border is pasted on rather than taken from the canvas: whether any slack is
            # left there depends on the camera, and at azimuth -90 there is none at the top.
            #
            # Per axis, not one pad off the long edge: this scene is about 2:1, so a single
            # pad sized on the width came to roughly a seventh of the height on each of the
            # top and bottom, which is a band of empty page the figure pays for twice.
            #
            # The sides get less again. The figure is placed at \linewidth, so horizontal
            # slack is not free space -- it is scaled away, and everything else grows with
            # it. A narrow side margin means a larger scene for the same column width.
            pad_x = round((pad_fraction if pad_fraction_x is None else pad_fraction_x)
                          * crop.width)
            pad_y = round(pad_fraction * crop.height)
            if pad_x or pad_y:
                bordered = Image.new(
                    "RGBA", (crop.width + 2 * pad_x, crop.height + 2 * pad_y), (0, 0, 0, 0)
                )
                bordered.paste(crop, (pad_x, pad_y))
                crop = bordered
            crop.save(path)
    return path


def _strip_axes(axes) -> None:
    """Leave only the scene: no ticks, labels, axis lines or panes."""
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_ticks([])
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_visible(False)
    axes.set_xlabel("")
    axes.set_ylabel("")
    axes.set_zlabel("")
    # _configure_axes places these two by hand, as mplot3d refuses to draw them itself.
    for text in list(axes.texts):
        text.remove()
    axes.set_axis_off()
    # Reclaim the margin _configure_axes reserved for the labels that just went away, and
    # drop the panel fill so the figure drops cleanly onto a page of any colour.
    axes.set_position((0.0, 0.0, 1.0, 1.0))
    axes.patch.set_alpha(0.0)
    axes.get_figure().patch.set_alpha(0.0)


def _target_density(map_source: Path, occupancy, origin, resolution, gmm=None):
    """Evaluate the mixture on the workspace grid; ``(None, None)`` if unavailable.

    ``gmm`` short-circuits the lookup. The disk path resolves the manifest's ``profile``
    to a sibling YAML, which only exists for runs archived next to their config -- the
    campaign's map directories hold arrays and a manifest and nothing else, so a caller
    that already has the config must hand it over or the density silently does not draw.
    """
    if gmm is None:
        import json

        manifest = map_source / "manifest.json"
        if not manifest.exists():
            return None, None
        profile = json.loads(manifest.read_text(encoding="utf-8")).get("profile")
        if not profile:
            return None, None
        from ergodic_control_mppi.config import load_config

        candidate = map_source.parent / f"{profile}.yaml"
        if not candidate.exists():
            return None, None
        gmm = load_config(candidate).controller.gmm
    means = np.asarray(gmm.means)
    inverses = np.asarray(gmm.covariance_inverse)
    weights = np.exp(np.asarray(gmm.log_weights))
    xs = np.linspace(origin[0], origin[0] + occupancy.shape[1] * resolution, 200)
    ys = np.linspace(origin[1], origin[1] + occupancy.shape[0] * resolution, 120)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    density = np.zeros(points.shape[0])
    for weight, mean, inverse in zip(weights, means, inverses):
        offset = points - mean
        density += weight * np.exp(
            -0.5 * np.einsum("ni,ij,nj->n", offset, inverse, offset)
        ) * np.sqrt(np.linalg.det(inverse))
    return density.reshape(grid_x.shape), (grid_x, grid_y)


def _overlay_rollouts(axes, figure_data: Path, altitude: float, rollout_step, count: int):
    """Draw the MPPI sample cloud and the selected plan at one recorded step."""
    from dataclasses import replace

    import jax.numpy as jnp

    from ergodic_control_mppi.config import load_config
    from ergodic_control_mppi.mppi.replay import replay_step, restore_snapshot

    data = np.load(figure_data, allow_pickle=False)
    steps = data["snapshot_steps"]
    index = len(steps) // 2 if rollout_step is None else int(np.argmin(abs(steps - rollout_step)))

    config = load_config("configs/uav_profile.yaml")
    params = replace(
        config.controller,
        workspace=replace(
            config.controller.workspace,
            grid=jnp.asarray(data["grid"].astype(np.float32)),
            grid_origin=jnp.asarray(data["grid_origin"].astype(np.float32)),
            grid_resolution=float(data["grid_resolution"]),
        ),
    )
    bundle = replay_step(params, restore_snapshot(data, index))

    # Show the rollouts that carry the weight, not an arbitrary slice: a uniform sample
    # would be dominated by trajectories the update effectively discarded.
    order = np.argsort(bundle.weights)[::-1][:count]
    for sample in order:
        track = bundle.positions[sample]
        axes.plot(
            track[:, 0], track[:, 1], np.full(track.shape[0], altitude),
            color=style.PRIMARY, alpha=0.12, linewidth=0.4,
        )
    axes.plot(
        bundle.optimal[:, 0], bundle.optimal[:, 1],
        np.full(bundle.optimal.shape[0], altitude),
        color=style.ACCENT, linewidth=2.0, label="MPPI plan",
    )


def _draw_modes(axes, run_directory: Path, altitude: float) -> None:
    """Mark the target mode centres."""
    import json

    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    for candidate in (manifest.get("config"), manifest.get("config_relative")):
        if candidate and Path(candidate).exists():
            from ergodic_control_mppi.config import load_config

            means = np.asarray(load_config(candidate).controller.gmm.means)
            axes.scatter(
                means[:, 0], means[:, 1], np.full(means.shape[0], altitude),
                marker="x", s=44, c=style.ACCENT, linewidths=1.6, depthshade=False,
            )
            return


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    parser.add_argument("--pillar-height", type=float, default=None,
                        help="Height above the flight plane; default is the map's minimum")
    parser.add_argument("--z-exaggeration", type=float, default=3.0)
    parser.add_argument("--rollout-step", type=int, default=None)
    arguments = parser.parse_args()
    output = arguments.output or arguments.run_dir / "snapshot.png"
    written = snapshot(
        arguments.run_dir, output,
        elevation=arguments.elevation, azimuth=arguments.azimuth,
        pillar_height=arguments.pillar_height, z_exaggeration=arguments.z_exaggeration,
        rollout_step=arguments.rollout_step,
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
