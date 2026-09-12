"""Render the volumetric-run figure from a path the clutter3d study stored.

uv run python scripts/dimension_figure.py --pillars 20 --seed 0 --until 150"""

import argparse
from pathlib import Path

import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.dimension import LIMITS_3D, PATH_STRIDE, PROFILE, target_3d
from ergodic_control_mppi.plotting.deployment import wrap_pdf
from ergodic_control_mppi.plotting.dimension import volumetric_snapshot


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("results/dimension/clutter3d_paths"))
    parser.add_argument("--pillars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    # The whole 400 s run is a tangle at column width; by 150 s all three modes are reached.
    parser.add_argument("--until", type=float, default=150.0, help="seconds of the run to draw")
    parser.add_argument("--azimuth", type=float, default=-75.0, help="camera azimuth, degrees")
    parser.add_argument("--elevation", type=float, default=30.0, help="camera elevation, degrees")
    parser.add_argument("--output", type=Path, default=Path("results/report/fig_volumetric.png"))
    arguments = parser.parse_args()

    stored = np.load(arguments.paths / f"volumetric_{arguments.pillars}_s{arguments.seed}.npz")
    params = load_config(PROFILE).controller
    means, covariances, weights = target_3d(params)
    step = PATH_STRIDE * float(params.model.delta_t)
    path = stored["path"][: int(round(arguments.until / step)) + 1]
    written = volumetric_snapshot(path, stored["pillars"], means, covariances,
                                  weights, LIMITS_3D, arguments.output,
                                  azimuth=arguments.azimuth, elevation=arguments.elevation)
    print(f"wrote {wrap_pdf(written)}")


if __name__ == "__main__":
    main()
