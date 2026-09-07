"""Render the opening figure from a trajectory the campaign actually flew.

The previous opening figure was flown on a 25-pillar map. The campaign uses 10, 15 and 20
pillars, so that map is in no reported table and the figure showed a run no number in the
paper describes. This replays one path out of the frozen bundle's `short_paths.npz` instead,
on the campaign's own map, so the picture and the results are the same experiment.

Replay, not re-flight: the paths are already recorded, so this needs no GPU and cannot drift
from the bundle the manuscript cites.

    uv run python scripts/teaser_figure.py --obs-num 20 --map-seed 516 --seed 0
"""

import argparse
from pathlib import Path

import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.plotting.deployment import trajectory_snapshot, wrap_pdf


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/uav/T150"))
    parser.add_argument("--obs-num", type=int, default=20)
    parser.add_argument("--map-seed", type=int, default=516)
    parser.add_argument("--seed", type=int, default=0,
                        help="audit seed index; the audit stage flies 0..11 per map")
    parser.add_argument("--output", type=Path,
                        default=Path("results/report/fig_deployment_snapshot.png"))
    parser.add_argument("--azimuth", type=float, default=-90.0)
    parser.add_argument("--elevation", type=float, default=38.0)
    parser.add_argument("--axes", action="store_true", help="keep ticks and axis lines")
    arguments = parser.parse_args()

    paths = np.load(arguments.bundle / "audit/short_paths.npz", allow_pickle=False)
    match = np.flatnonzero((paths["obs_num"] == arguments.obs_num)
                           & (paths["map_seed"] == arguments.map_seed)
                           & (paths["seed"] == arguments.seed))
    if match.size != 1:
        available = sorted({(int(o), int(m)) for o, m in
                            zip(paths["obs_num"], paths["map_seed"])})
        raise SystemExit(f"no single path for {arguments.obs_num}p/{arguments.map_seed}/"
                         f"s{arguments.seed}; bundle holds {available}")
    positions = np.asarray(paths["positions"][int(match[0])], dtype=np.float64)

    map_source = (arguments.bundle
                  / f"maps/density_{arguments.obs_num}_map_{arguments.map_seed}")
    config = load_config(str(arguments.bundle / "config.yaml"))

    written = trajectory_snapshot(
        positions, map_source, arguments.output,
        gmm=config.controller.gmm,
        elevation=arguments.elevation, azimuth=arguments.azimuth,
        bare=not arguments.axes,
        # Neutral scene, one coloured trail. The pillars and the target density are context
        # and stay greyscale; the trail carries elapsed time and is the only chroma present.
        pillar_cmap="pillar_neutral",
        density_cmap="carbon",
        trail_cmap="trail_time",
        # Up from 1.6: the trail is now the subject of the picture rather than a trace
        # threaded behind the columns, and has to hold its own against a 20-pillar field.
        trail_size=2.6,
        vehicle_colour="#101820",
    )
    print(f"wrote {written}")
    print(f"wrote {wrap_pdf(written)}")


if __name__ == "__main__":
    main()
