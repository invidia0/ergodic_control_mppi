"""Why does HEDAC pin against a pillar in the clutter tier?

The clutter rows say it travels 145 m in 400 s at a median clearance of 0.005 m and reaches
all three modes in 2 of 72 runs. That is not bad coverage, it is a vehicle pressed against an
obstacle, and a 10x "win" over the classical clutter baseline is worthless until we know
whether the pin is HEDAC's or ours.

Two candidate mechanisms, both testable here:

``iterations``
    Eight warm-started Jacobi sweeps converge a smooth open field. With reflecting obstacles
    the potential has to propagate *around* each pillar, and a stale local solve has a
    gradient that need not point anywhere useful.

``gradient stencil``
    ``_jacobi_neumann`` is Neumann by construction -- the stencil reflects at occupied
    faces and never reads an obstacle cell. But ``_hedac_velocity`` then takes
    ``np.gradient`` of the same array, and *that* does read the obstacle cells, which hold
    a hard zero. So the solve sees no-flux walls while the steering sees a Dirichlet cliff
    at every pillar. The two disagree exactly in the band where the vehicle pins.

Reports, per configuration: distance travelled, closest approach, the step at which motion
effectively stops, and at the pin, the gradient magnitude and whether the commanded
direction points into the nearest pillar.

    uv run python scripts/hedac_diagnostic.py
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ergodic_control_mppi.experiments import baselines
from ergodic_control_mppi.experiments.baselines import BaselineConfig

# One cell that pinned, from results/uav/baselines_clutter.csv.
MAP = ("25p_516", 25, 516)


def _scenario(map_seed: int, obs_num: int):
    """Load the campaign map exactly as the clutter tier does.

    The clutter scenario *is* the open one renamed: obstacles reach the methods through the
    occupancy grid handed to ``run_method``, not through the scenario. Built the same way
    here so the diagnostic cannot differ from the run it is explaining.
    """
    from ergodic_control_mppi.experiments.uav_pillar_tuning import _grid_config

    directory = Path(f"results/uav/density_{obs_num}/maps/map_{map_seed}")
    config, manifest, arrays = _grid_config(directory)
    scenario = replace(baselines._open_scenario(config), name=f"{obs_num}p_{map_seed}")
    return config, scenario, arrays, manifest


def _trace(scenario, state0, occupancy, origin, resolution, cfg, steps, avoidance=False):
    """Fly HEDAC and record what the steering saw at every step.

    A copy of the clutter loop in `run_method`, instrumented. Duplicated rather than
    hooked because the point is to watch intermediate quantities the loop does not return,
    and threading a callback through the shared runner for one diagnostic is the worse
    trade.
    """
    from ergodic_control_mppi.experiments.literature_methods import (
        _boundary_bias, _limit_speed, _tracker_step_np,
    )

    shape, pitch = baselines._solver_shape(scenario, cfg.grid_size)
    blocked = baselines._blocked_mask(occupancy, shape)
    target = baselines._resample_target(scenario, shape)
    target = np.where(blocked, 0.0, target)
    target = target / max(target.sum(), 1e-12)
    centres, radii = baselines._pillar_circles(occupancy, origin, resolution)

    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    x_edges = np.linspace(x_min, x_max, shape[1] + 1)
    y_edges = np.linspace(y_min, y_max, shape[0] + 1)

    states = np.asarray(state0, dtype=np.float64)[None, :].copy()
    counts = np.zeros(shape)
    warm = None
    log = np.zeros((steps, 5))  # x, y, |grad u|, clearance, into-pillar cosine
    for step in range(steps):
        xy = states[:, :2]
        column = np.clip(np.searchsorted(x_edges, xy[:, 0]) - 1, 0, shape[1] - 1)
        row = np.clip(np.searchsorted(y_edges, xy[:, 1]) - 1, 0, shape[0] - 1)
        np.add.at(counts, (row, column), 1.0)
        coverage = counts / max(counts.sum(), 1.0)

        field, warm = baselines._hedac_velocity(
            xy, coverage, target, blocked, scenario, cfg, shape, warm, pitch=pitch)
        raw = float(np.linalg.norm(field))

        desired = baselines._unit_field(field, cfg.desired_speed)
        desired = desired + _boundary_bias(xy, scenario.map_x_limits, scenario.map_y_limits)
        if avoidance:
            desired = desired + baselines._avoidance(
                xy, centres, radii, cfg.avoid_clearance, cfg.avoid_gain)
        desired = _limit_speed(desired, cfg.desired_speed)
        states = _tracker_step_np(states, desired, scenario, cfg.tracker_gain)

        # Clearance to the nearest pillar *surface*, and whether we are commanding into it.
        offsets = xy[0] - centres
        distance = np.linalg.norm(offsets, axis=1)
        nearest = int(np.argmin(distance - radii))
        clearance = float(distance[nearest] - radii[nearest])
        inward = -offsets[nearest] / max(distance[nearest], 1e-12)
        heading = desired[0] / max(np.linalg.norm(desired[0]), 1e-12)
        log[step] = (xy[0, 0], xy[0, 1], raw, clearance, float(heading @ inward))
    return log


def _report(name, log, dt):
    """Where the distance goes, and how much of the run is spent inside a pillar.

    The clutter row reports one closest approach and one total distance, which cannot
    distinguish "drives into a pillar once" from "spends the run embedded in one". Both the
    time inside and the distance per quarter are needed to tell those apart.
    """
    xy = log[:, :2]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    quarter = len(steps) // 4
    legs = [float(steps[i * quarter:(i + 1) * quarter].sum()) for i in range(4)]
    inside = float(np.mean(log[:, 3] < 0.0))
    print(f"  {name:16s} travelled {steps.sum():6.1f} m   "
          f"per quarter {'/'.join(f'{v:.0f}' for v in legs)} m")
    print(f"  {'':16s} inside a pillar {inside:5.1%} of the run   "
          f"min clearance {log[:, 3].min():+.2f} m   "
          f"median |grad u| {np.median(log[:, 2]):.2e}")
    # Only while embedded: is the steering pushing further in, or working its way out?
    embedded = log[log[:, 3] < 0.0]
    if embedded.size:
        print(f"  {'':16s} while embedded: median cos(command, into pillar) "
              f"{np.median(embedded[:, 4]):+.2f}   "
              f"({np.mean(embedded[:, 4] > 0):.0%} of those steps point inward)")
    return float(steps.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--iterations", default="8,64,256")
    arguments = parser.parse_args()

    name, obs_num, map_seed = MAP
    config, scenario, arrays, manifest = _scenario(map_seed, obs_num)
    occupancy = np.asarray(arrays["occupancy"]).astype(bool)
    origin = tuple(map(float, np.asarray(arrays["grid_origin"])))
    resolution = float(arrays["grid_resolution"])
    state0 = baselines.seed_state(
        np.asarray(arrays["initial_state"], dtype=np.float64), scenario, arguments.seed)

    dt = float(config.controller.model.delta_t)
    print(f"HEDAC on {name}, seed {arguments.seed}, {arguments.steps} steps "
          f"({arguments.steps * dt:.0f} s)\n")

    for iterations in (int(v) for v in arguments.iterations.split(",")):
        for avoidance in (False, True):
            cfg = BaselineConfig(hedac_iterations=iterations)
            log = _trace(scenario, state0, occupancy, origin, resolution, cfg,
                         arguments.steps, avoidance=avoidance)
            tag = "+avoid" if avoidance else "neumann"
            _report(f"it={iterations} {tag}", log, dt)
            print()


if __name__ == "__main__":
    main()
