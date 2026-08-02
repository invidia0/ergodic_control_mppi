"""Oblique 3D view of a flown trial: pillars, flown path, target modes, MPPI rollouts.

Rendered from the recorded arrays rather than screenshotted from RViz, so it is
reproducible headlessly and versioned with the run it depicts. The camera looks from a
long side of the workspace at a high elevation -- close to a plan view but oblique enough
that obstacle height and flight altitude read as three-dimensional.

Obstacle footprints come from the *raw* occupancy, never the inflated planning grid.
Recorded arrays contain only the planar slice, so ``pillar_height`` is a visualization
extrusion capped at the target-density plane, not a recovered physical height.
"""

from pathlib import Path

import numpy as np

from ergodic_control_mppi.plotting import style

# Looking along -y from a long side, steeply down. 90 would be a plan view with no height
# cue at all; 60 keeps the pillars readable as volumes.
DEFAULT_ELEVATION = 60.0
DEFAULT_AZIMUTH = -90.0


def _cell_centres(occupancy: np.ndarray, origin, resolution: float) -> np.ndarray:
    """World ``(x, y)`` of every occupied cell centre, shape ``(N, 2)``."""
    rows, columns = np.nonzero(occupancy)
    return np.column_stack(
        [
            origin[0] + (columns + 0.5) * resolution,
            origin[1] + (rows + 0.5) * resolution,
        ]
    )


def _draw_pillars(axes, centres: np.ndarray, resolution: float, height: float) -> None:
    """Draw each occupied cell as an extruded box of the given height."""
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
                np.array([[0.0, 0.0], [height, height]]),
                color=style.NEUTRAL, alpha=0.55, linewidth=0, shade=True,
            )
        axes.plot_surface(
            np.array([[corners[0, 0], corners[1, 0]], [corners[3, 0], corners[2, 0]]]),
            np.array([[corners[0, 1], corners[1, 1]], [corners[3, 1], corners[2, 1]]]),
            np.full((2, 2), height),
            color=style.NEUTRAL, alpha=0.8, linewidth=0, shade=True,
        )


def snapshot(
    run_directory: Path,
    output: Path,
    *,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    altitude: float = 0.75,
    pillar_height: float = 0.04,
    z_exaggeration: float = 1.0,
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
        azimuth: Camera azimuth in degrees; -90 looks along -y from a long side.
        altitude: Flight altitude, for the path's z.
        pillar_height: Schematic height used to extrude the exact planar footprints.
        z_exaggeration: Vertical stretch of the rendered box only; 1.0 already
            exaggerates z relative to the 40 m span so pillars stay legible.
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

        _draw_pillars(axes, _cell_centres(occupancy, origin, resolution), resolution,
                      pillar_height)

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

        x_limits = (origin[0], origin[0] + occupancy.shape[1] * resolution)
        y_limits = (origin[1], origin[1] + occupancy.shape[0] * resolution)
        axes.set_xlim(*x_limits)
        axes.set_ylim(*y_limits)
        axes.set_zlim(0.0, max(pillar_height, altitude) * 1.1)
        # True metric proportions; without this the long axis is squashed to a cube and
        # the workspace reads as square when it is 2:1.
        # x:y is true metric proportion (2:1 here) so the workspace is not read as
        # square. z is deliberately exaggerated -- at true scale a 2.5 m pillar over a 40 m
        # span is 6% of the width and vanishes. State the factor in the caption.
        span = x_limits[1] - x_limits[0]
        axes.set_box_aspect(
            (span, y_limits[1] - y_limits[0], span * z_exaggeration / 8.0)
        )
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_xlabel("x [m]", labelpad=2)
        axes.set_ylabel("y [m]", labelpad=2)
        axes.set_zlabel("z [m]", labelpad=-8)
        # paper_style turns on minor ticks, which crowd unreadably in a projected 3D axis.
        axes.set_zticks([0.0, altitude])
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_minor_locator(plt.NullLocator())
            axis.pane.set_alpha(0.0)
        axes.tick_params(axis="z", pad=-3)
        axes.grid(False)
        return style.save(figure, output)


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
    parser.add_argument("--pillar-height", type=float, default=0.04)
    parser.add_argument("--z-exaggeration", type=float, default=1.0)
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
