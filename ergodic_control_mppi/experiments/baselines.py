"""
Single-robot comparison against ergodic-coverage baselines, in the open and in clutter.

uv run python -m ergodic_control_mppi.experiments.baselines --tier open
uv run python -m ergodic_control_mppi.experiments.baselines --tier clutter
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ergodic_control_mppi.experiments.common import Scenario
from ergodic_control_mppi.experiments.literature_methods import (
    _boundary_bias,
    _limit_speed,
    _tracker_step_np,
)

METHODS = ("ours", "hedac", "sves", "fmec", "smc")
# Three seeds, decided on the median: one run cannot settle whether a chaotic
# closed loop reproduces. See `fidelity_check`.
FIDELITY_SEEDS = (43, 44, 45)
# Methods that score occupancy natively. Others get `_avoidance` in the clutter tier.
NATIVE_OBSTACLES = {"ours"}


# --------------------------------------------------------------------------- obstacles


def _solver_shape(scenario: Scenario, long_side: int) -> tuple[tuple[int, int], float]:
    """Grid dimensions with *square* cells, plus the cell pitch in metres."""
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    width, height = float(x_max - x_min), float(y_max - y_min)
    pitch = max(width, height) / long_side
    return (max(int(round(height / pitch)), 1), max(int(round(width / pitch)), 1)), pitch


def _cell_index(values: np.ndarray, edges: np.ndarray, count: int) -> np.ndarray:
    """Which grid cell each coordinate falls in."""
    return np.clip(np.searchsorted(edges, values) - 1, 0, count - 1)


def _blocked_mask(occupancy: np.ndarray, shape: tuple[int, int], scenario: Scenario,
                  origin=None, resolution: float | None = None) -> np.ndarray:
    """Resample an occupancy grid onto the solver grid by nearest neighbour."""
    if origin is None or resolution is None:
        rows = np.clip((np.arange(shape[0]) + 0.5) * occupancy.shape[0] / shape[0],
                       0, occupancy.shape[0] - 1).astype(int)
        columns = np.clip((np.arange(shape[1]) + 0.5) * occupancy.shape[1] / shape[1],
                          0, occupancy.shape[1] - 1).astype(int)
        return occupancy[np.ix_(rows, columns)].astype(bool)

    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    centres_x = x_min + (np.arange(shape[1]) + 0.5) * (x_max - x_min) / shape[1]
    centres_y = y_min + (np.arange(shape[0]) + 0.5) * (y_max - y_min) / shape[0]
    columns = np.clip(((centres_x - origin[0]) / resolution).astype(int),
                      0, occupancy.shape[1] - 1)
    rows = np.clip(((centres_y - origin[1]) / resolution).astype(int),
                   0, occupancy.shape[0] - 1)
    return occupancy[np.ix_(rows, columns)].astype(bool)


def _jacobi_neumann(source: np.ndarray, blocked: np.ndarray, gain: float,
                    damping: float, iterations: int, warm=None) -> np.ndarray:
    """
    Solve the HEDAC screened-Poisson problem with no-flux obstacle boundaries.
    """
    # The four reflection masks depend only on `blocked`, which is fixed for a whole run,
    # but were being rebuilt inside every sweep: four `np.roll`s of a boolean grid, eight
    # sweeps deep, 20 000 steps long. Hoisting them is bit-for-bit identical and removes
    # half the rolls in the solve.
    walls = _reflection_masks(blocked)

    def neighbour(potential, index, axis, amount):
        """Value across one face, reflected where there is no fluid cell behind it."""
        return np.where(walls[index], potential, np.roll(potential, amount, axis))

    potential = np.zeros_like(source) if warm is None else warm
    free = ~blocked
    scaled_source = gain * source
    for _ in range(iterations):
        stencil = (
            neighbour(potential, 0, 0, 1) + neighbour(potential, 1, 0, -1)
            + neighbour(potential, 2, 1, 1) + neighbour(potential, 3, 1, -1)
        )
        potential = np.where(free, (stencil + scaled_source) / (4.0 + damping), 0.0)
    return potential


def _reflection_masks(blocked: np.ndarray) -> list[np.ndarray]:
    """
    Where the five-point stencil must reflect: obstacles, plus the domain edge.
    """
    masks = []
    for axis, amount in ((0, 1), (0, -1), (1, 1), (1, -1)):
        wall = np.roll(blocked, amount, axis).copy()
        edge = [slice(None), slice(None)]
        edge[axis] = 0 if amount > 0 else -1
        wall[tuple(edge)] = True
        masks.append(wall)
    return masks


def _unit_field(field: np.ndarray, speed: float) -> np.ndarray:
    """
    Rescale a coverage-law field to a commanded speed, keeping its direction.
    """
    magnitude = np.linalg.norm(field, axis=1, keepdims=True)
    return np.where(magnitude > 1e-9, field / np.maximum(magnitude, 1e-12) * speed, 0.0)


def _avoidance(xy: np.ndarray, centres: np.ndarray, radius: np.ndarray,
               clearance: float, gain: float) -> np.ndarray:
    """Shared repulsion for baselines whose formulation defines none."""
    if centres.size == 0:
        return np.zeros_like(xy)
    offsets = xy[:, None, :] - centres[None, :, :]
    distance = np.linalg.norm(offsets, axis=-1)
    reach = radius[None, :] + clearance
    # Only obstacles inside the clearance band push, and the push saturates at the surface
    # rather than diverging, so a state that starts inside one is still recoverable.
    depth = np.clip(reach - distance, 0.0, None)
    unit = offsets / np.maximum(distance, 1e-9)[..., None]
    return gain * np.einsum("nm,nmi->ni", depth, unit)


def _pillar_circles(occupancy: np.ndarray, origin, resolution: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Fit one circle per connected component of the occupancy grid."""
    from scipy import ndimage

    labels, count = ndimage.label(occupancy)
    if count == 0:
        return np.zeros((0, 2)), np.zeros(0)
    centres, radii = [], []
    for index in range(1, count + 1):
        rows, columns = np.nonzero(labels == index)
        x = origin[0] + (columns + 0.5) * resolution
        y = origin[1] + (rows + 0.5) * resolution
        centre = np.array([x.mean(), y.mean()])
        # Half a cell added: the furthest *centre* understates the footprint by the
        # half-pitch of the cell it sits in.
        radii.append(float(np.max(np.hypot(x - centre[0], y - centre[1])) + 0.5 * resolution))
        centres.append(centre)
    return np.asarray(centres), np.asarray(radii)


# ----------------------------------------------------------------------------- methods
#
# Every runner takes the same arguments and returns the executed state history, shape
# ``(steps, 6)``. They share `_tracker_step_np`, the double integrator and the speed limit,
# so what differs between them is only the desired-velocity law.


def _hedac_velocity(state_xy, coverage, target, blocked, scenario, cfg, shape,
                    warm=None, *, pitch=1.0, x_edges=None, y_edges=None):
    """
    HEDAC: follow the gradient of the potential driven by the coverage deficit.
    """
    from scipy import ndimage

    coverage = ndimage.gaussian_filter(coverage, cfg.hedac_sensor / pitch, mode="nearest")
    coverage = coverage / max(coverage.sum(), 1e-12)
    source = np.where(blocked, 0.0, target - coverage)
    # damping = h^2/alpha keeps the diffusion length sqrt(alpha) fixed as the mesh changes;
    # gain = damping keeps the source scale at unity. See `hedac_alpha`.
    damping = pitch ** 2 / cfg.hedac_alpha
    potential = _jacobi_neumann(source, blocked, damping, damping,
                                cfg.hedac_iterations, warm)
    grad_y, grad_x = np.gradient(potential, pitch)
    column = _cell_index(state_xy[:, 0], x_edges, shape[1])
    row = _cell_index(state_xy[:, 1], y_edges, shape[0])
    field = cfg.hedac_gradient_gain * np.stack(
        [grad_x[row, column], grad_y[row, column]], axis=1)
    return field, potential


def _smc_velocity(state_xy, ctx, coefficients, elapsed):
    """SMC (Mathew & Mezic): descend the Fourier ergodic metric pointwise."""
    from ergodic_control_mppi.experiments.literature_methods import _basis_and_grad_np

    basis, grad = _basis_and_grad_np(state_xy, ctx)
    del basis
    weight = ctx.lambda_k_np * (coefficients / max(elapsed, 1.0) - ctx.phi_k_np)
    return -np.einsum("k,nki->ni", weight, grad)


def _sves_planner(ctx, model_params, cfg):
    """
    Stein Variational Ergodic Search: SVGD over a population of control sequences.
    """
    import jax
    import jax.numpy as jnp

    from ergodic_control_mppi.experiments.literature_methods import (
        _basis_values_jax,
        _rollout_controls,
    )

    horizon, particles = cfg.sves_horizon, cfg.sves_particles
    limit = float(model_params.max_accel_lin_abs)

    def ergodic_cost(controls, state, history, elapsed):
        trajectory = _rollout_controls(state, controls, model_params)
        basis = _basis_values_jax(trajectory[..., :2], ctx)
        # `history` is the running *sum* of per-step basis values, not their mean, so it is
        # added to the horizon's sum and the pair is averaged once. Scaling it by `elapsed`
        # first counts the past twice over and grows without bound, which drowns the
        # horizon term completely -- the particles then optimise a constant and the vehicle
        # flies fast in no useful pattern. That is how this first failed the gate, at 708 m
        # travelled and a metric that got worse.
        total = (history + jnp.sum(basis, axis=0)) / (elapsed + horizon)
        return jnp.mean(ctx.lambda_k_jax * jnp.square(total - ctx.phi_k_jax))

    def update(controls, state, history, elapsed):
        flat = controls.reshape(particles, -1)
        costs, grads = jax.vmap(
            jax.value_and_grad(ergodic_cost), in_axes=(0, None, None, None)
        )(controls, state, history, elapsed)
        score = -grads.reshape(particles, -1) / cfg.sves_temperature

        # RBF kernel over control sequences, bandwidth by the median heuristic -- the
        # standard SVGD choice, and it matters here because the particle spread changes by
        # orders of magnitude between a converged and a re-planning step.
        square = jnp.sum((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
        bandwidth = jnp.maximum(jnp.median(square), 1e-6) / jnp.log(particles + 1.0)
        kernel = jnp.exp(-square / bandwidth)
        # grad_{U_j} k(U_j, U_i) summed over j, the repulsion term.
        repulsion = (kernel @ flat - jnp.sum(kernel, axis=1, keepdims=True) * flat)
        drive = (kernel @ score + 2.0 * repulsion / bandwidth) / particles

        stepped = flat + cfg.sves_step * drive
        return jnp.clip(stepped, -limit, limit).reshape(controls.shape), costs

    rollout = jax.jit(lambda controls, state:
                      _rollout_controls(state, controls, model_params)[..., :2])
    return jax.jit(update), rollout


def _fmec_velocity(state_xy, coverage, target, ctx, cfg, x_edges, y_edges, *, pitch=1.0):
    """
    Flow-Matching Ergodic Coverage: follow the transport field from coverage to target.
    """
    from scipy import ndimage

    smoothed = ndimage.gaussian_filter(coverage, cfg.fmec_bandwidth / pitch, mode="nearest")
    # Floors, not epsilons on the log: a zero-coverage cell has an unbounded score, which
    # would fling the agent at the first unvisited cell it can see rather than transport it.
    smoothed = smoothed / max(smoothed.sum(), 1e-12) + cfg.fmec_floor
    reference = target / max(target.sum(), 1e-12) + cfg.fmec_floor

    grad_y, grad_x = np.gradient(np.log(reference) - np.log(smoothed), pitch)
    column = _cell_index(state_xy[:, 0], x_edges, coverage.shape[1])
    row = _cell_index(state_xy[:, 1], y_edges, coverage.shape[0])
    return cfg.fmec_gain * np.stack([grad_x[row, column], grad_y[row, column]], axis=1)


from ergodic_control_mppi.experiments.common import (
    ensure_bundle, execution_record, fingerprint, numerical_record, verified_rows,
)

# ------------------------------------------------------------------------------ config


class BaselineConfig:
    """Tunables for every baseline, in one place so the settings are auditable."""

    def __init__(self, **overrides):
        self.steps = 20000
        self.desired_speed = 1.8       # the shipped profile's reference speed
        self.tracker_gain = 3.0
        self.fourier_order = 5
        # Cells along the *long* workspace axis; the short axis follows so cells stay
        # square (see `_solver_shape`). 160 puts the pitch at 0.25 m, which resolves a
        # 0.59 m pillar across roughly five cells. At 80 a pillar was two cells wide and
        # the potential barrier around it was thinner than the vehicle's stopping
        # distance, so HEDAC flew through the obstacles it was solving for.
        self.grid_size = 160
        # HEDAC. `hedac_alpha` is the screened-Poisson diffusivity in m^2, so the potential's
        # reach is a physical length and not a property of the mesh. The iteration's fixed
        # point is `-(pitch^2/damping) lap(u) + u = (gain/damping) q`, so alpha = h^2/damping
        # and the source is faithful only when gain == damping; both are therefore derived
        # from the pitch at run time rather than configured. Previously they were constants,
        # which made alpha = 5 h^2: changing `grid_size` from 80 to 160 quartered it, and the
        # measured 1/e decay of a point source moved from 1.00 m to 0.50 m without anything
        # in the configuration appearing to change. 1.25 m^2 preserves the reach the solver
        # had at the grid it was originally tuned on.
        self.hedac_alpha = 1.25
        self.hedac_iterations = 8
        self.hedac_gradient_gain = 8.0
        # Sensor footprint radius in metres, selected on the open field by the gate's own
        # criterion (GPU, 8 seeds -- the device the tiers fly, because a CPU sweep disagreed
        # with the GPU gate on mode counts). 1.0, 2.0 and
        # 3.0 m all reach every mode in 8/8; best metric is 1.97e-4, 2.63e-4 and 8.71e-4, so
        # the rule -- mode reach first, `ergodic_best` as tie-break -- picks 1.0 m. That is
        # the setting where the baseline is *strongest*, which is the direction this choice
        # must err in. 4.0 m fails outright at 0/3.
        self.hedac_sensor = 1.0
        # SVES
        # Retained SVES horizon, 350; the T150 controller uses a shorter horizon.
        # 16 particles over 350 steps costs
        # about the same wall time as 8 over 40 -- the rollout is vmapped.
        self.sves_particles = 16
        self.sves_horizon = 350
        self.sves_step = 0.05
        self.sves_temperature = 1e-3
        self.sves_replan_every = 10
        self.sves_lookahead = 100      # plan steps ahead the tracker aims at
        self.sves_init = 0.35          # prior spread, as a fraction of a_max
        # FMEC
        # Metres, not cells (see `_solver_shape`). Selected on the open field against the
        # fidelity gate's own criterion by `scripts/fmec_bandwidth_sweep.py`: 0.35, 0.50,
        # 0.70 and 1.00 m all reach every mode, 2.00 m reaches none -- the over-wide-kernel
        # under-exploration that Fig. 2 of Sun et al. describes. The passing range spans a
        # factor of three and the scores within it are non-monotonic (0.70 m is worse than
        # both its neighbours), which on a loop this chaotic means three seeds cannot
        # resolve them. So this is not "the best number": it is the value at which the
        # baseline performed best, taken deliberately because choosing a competitor's
        # weaker setting is the failure mode worth guarding against here.
        self.fmec_bandwidth = 1.0      # metres
        self.fmec_floor = 1e-4
        self.fmec_gain = 4.0
        # Shared avoidance. The clearance is set so that every method plans against the
        # *same* effective keep-out radius as our own controller does. Ours is handed the
        # inflated occupancy grid, whose area-equivalent pillar radius is 1.30 m; the
        # baselines are handed circles fitted to the raw footprints, median radius 0.594 m.
        # 0.71 m of clearance puts their keep-out at 1.30 m too. At the previous 0.6 m ours
        # kept 1.30 m clear while the baselines kept 1.19 m, a 9% advantage to us on the one
        # outcome -- collisions -- where we have the strongest claim. Scoring was already
        # symmetric: `score_run` measures every method against the raw geometry inflated by
        # the 0.30 m robot radius.
        self.avoid_clearance = 0.71
        self.avoid_gain = 6.0
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ValueError(f"unknown baseline setting {key!r}")
            setattr(self, key, value)

    def as_dict(self) -> dict:
        return dict(sorted(vars(self).items()))

    # Which settings each method's trajectory actually depends on. Explicit rather than by
    # prefix, because the answer is not obvious: `grid_size` builds the coverage histogram
    # for every method but only HEDAC and FMEC ever read it, and `ours` reads nothing here
    # at all -- `run_method` returns before this config is touched, so its behaviour is
    # fixed by the profile YAML, whose own hash the map manifest already records.
    _SHARED = ("desired_speed", "tracker_gain", "avoid_clearance", "avoid_gain")
    DEPENDS_ON = {
        "ours": (),
        "hedac": _SHARED + ("fourier_order", "grid_size", "hedac_alpha", "hedac_iterations",
                            "hedac_gradient_gain", "hedac_sensor"),
        "fmec": _SHARED + ("fourier_order", "grid_size", "fmec_bandwidth", "fmec_floor",
                           "fmec_gain"),
        "smc": _SHARED + ("fourier_order",),
        "sves": _SHARED + ("fourier_order", "sves_particles", "sves_horizon", "sves_step",
                           "sves_temperature", "sves_replan_every", "sves_lookahead",
                           "sves_init"),
    }

    def fingerprint_for(self, method: str, config, arrays, scoring: dict) -> str:
        """
        Hash method-specific control inputs and the shared resolved world and scorer.
        """
        if method not in self.DEPENDS_ON:
            raise ValueError(f"unknown method {method!r}")
        params = config.controller
        shared = {"model": params.model, "gmm": params.gmm, "workspace": params.workspace,
                  "arrays": arrays, "scoring": scoring, "steps": self.steps}
        controller = (params if method == "ours" else
                      {k: getattr(self, k) for k in self.DEPENDS_ON[method]})
        return fingerprint({"shared": shared, "controller": controller})

    def fingerprint(self) -> str:
        """Short hash of every setting, stamped on each row."""
        import hashlib

        settings = {k: v for k, v in self.as_dict().items() if k != "steps"}
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------ driver


def run_method(method: str, scenario: Scenario, state0: np.ndarray, *, steps: int,
               seed: int, cfg: BaselineConfig, occupancy=None, origin=None,
               resolution: float = 0.15) -> np.ndarray:
    """
    Fly one method for ``steps`` and return the executed states, shape ``(steps, 6)``.
    """
    from ergodic_control_mppi.experiments.literature_methods import _fourier_context

    if method == "ours":
        return _run_ours(scenario, state0, steps=steps, seed=seed)

    ctx = _fourier_context(scenario, cfg.fourier_order)
    shape, pitch = _solver_shape(scenario, cfg.grid_size)
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    x_edges = np.linspace(x_min, x_max, shape[1] + 1)
    y_edges = np.linspace(y_min, y_max, shape[0] + 1)

    blocked = (np.zeros(shape, bool) if occupancy is None
               else _blocked_mask(occupancy, shape, scenario, origin, resolution))
    if occupancy is None or method in NATIVE_OBSTACLES:
        centres, radii = np.zeros((0, 2)), np.zeros(0)
    else:
        centres, radii = _pillar_circles(occupancy, origin, resolution)

    target = _resample_target(scenario, shape)
    target = np.where(blocked, 0.0, target)
    target /= max(target.sum(), 1e-12)

    states = np.asarray(state0, dtype=np.float64).reshape(1, 6)
    path = np.zeros((steps, 6), dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    # Running Fourier coefficients of the executed path. SMC descends their mismatch
    # directly; SVES carries them as the coverage history its horizon is appended to.
    coefficients = np.zeros(ctx.phi_k_np.shape, dtype=np.float64)

    planner = rollout = controls = warm = None
    best = 0
    if method == "sves":
        import jax

        planner, rollout = _sves_planner(ctx, scenario.params.model, cfg)
        # Sampled from the prior, not zeroed. SVGD's repulsion is
        # ``grad_{U_j} k(U_j, U_i)``, which is identically zero when every particle sits at
        # the same point, so a zero-initialised population can never separate and the
        # method degenerates to one trajectory that never leaves its initial guess. That is
        # what made the first three attempts at this fly the *same* 708 m straight line
        # whatever else was changed.
        limit = float(scenario.params.model.max_accel_lin_abs)
        controls = cfg.sves_init * limit * jax.random.normal(
            jax.random.PRNGKey(seed), (cfg.sves_particles, cfg.sves_horizon, 3),
            dtype=jax.numpy.float32)

    for step in range(steps):
        xy = states[:, :2]
        column = _cell_index(xy[:, 0], x_edges, shape[1])
        row = _cell_index(xy[:, 1], y_edges, shape[0])
        np.add.at(counts, (row, column), 1.0)
        coverage = counts / max(counts.sum(), 1.0)

        # Accumulated from the *executed* position, not from any plan: the coverage history
        # both spectral methods condition on is where the vehicle actually went.
        if method in ("smc", "sves"):
            from ergodic_control_mppi.experiments.literature_methods import _basis_and_grad_np

            basis, _ = _basis_and_grad_np(xy, ctx)
            coefficients += basis[0]

        if method == "hedac":
            desired, warm = _hedac_velocity(xy, coverage, target, blocked, scenario,
                                            cfg, shape, warm, pitch=pitch,
                                            x_edges=x_edges, y_edges=y_edges)
        elif method == "fmec":
            desired = _fmec_velocity(xy, coverage, target, ctx, cfg, x_edges, y_edges,
                                     pitch=pitch)
        elif method == "smc":
            desired = _smc_velocity(xy, ctx, coefficients, step + 1)
        elif method == "sves":
            desired, controls, best = _sves_step(
                planner, rollout, controls, best, states, coefficients, step, cfg)
        else:
            raise ValueError(f"unknown method {method!r}")

        # Coverage law sets the direction and is flown at the commanded speed; the wall
        # bias and the obstacle push are corrections added on top, and the total is capped.
        # Same gauge our own controller uses, so no method is faster than another.
        desired = _unit_field(desired, cfg.desired_speed)
        desired = desired + _boundary_bias(xy, scenario.map_x_limits, scenario.map_y_limits)
        if centres.size:
            desired = desired + _avoidance(xy, centres, radii, cfg.avoid_clearance,
                                           cfg.avoid_gain)
        desired = _limit_speed(desired, cfg.desired_speed)
        states = _tracker_step_np(states, desired, scenario, cfg.tracker_gain)
        path[step] = states[0]
    return path


def _resample_target(scenario: Scenario, shape: tuple[int, int]) -> np.ndarray:
    """Put the scenario's target density on the solver grid."""
    from ergodic_control_mppi.experiments.literature_methods import _resize_grid_bilinear

    return _resize_grid_bilinear(
        np.asarray(scenario.target_density_grid, dtype=np.float64), shape)


def _run_ours(scenario: Scenario, state0: np.ndarray, *, steps: int, seed: int
              ) -> np.ndarray:
    """Our controller, through the same entry point the campaign uses."""
    import jax
    import jax.numpy as jnp

    from ergodic_control_mppi.mppi.single import run_single
    from ergodic_control_mppi.simulation import controller_key

    params = scenario.params
    controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
    result = jax.jit(run_single, static_argnames=("steps",))(
        params, jnp.asarray(state0, dtype=jnp.float32), controls,
        controller_key(seed), steps)
    return np.asarray(result.path, dtype=np.float64)


def _sves_step(planner, rollout, controls, best, states, coefficients, step, cfg):
    """
    One SVES control: refresh the particle population, then track the best plan.
    """
    import jax.numpy as jnp

    state = jnp.asarray(states[0], dtype=jnp.float32)
    history = jnp.asarray(coefficients, dtype=jnp.float32)
    if step % cfg.sves_replan_every == 0:
        controls, costs = planner(controls, state, history, float(max(step, 1)))
        best = int(jnp.argmin(costs))
    # Only the state at `index` is wanted, and `_rollout_controls` is a `lax.scan`, so
    # simulating the prefix gives that state bit for bit while doing a third of the work.
    # Rolling the full 350-step horizon to read step 100 was the per-step cost of SVES.
    index = min(cfg.sves_lookahead, cfg.sves_horizon - 1)
    waypoint = np.asarray(rollout(controls[best][:index + 1], state)[-1], dtype=np.float64)
    return (waypoint - states[0, :2])[None, :], controls, best


# ---------------------------------------------------------------------- fidelity gate


def fidelity_check(method: str, scenario: Scenario, state0: np.ndarray, *,
                   cfg: BaselineConfig, steps: int = 20000, seeds=(43, 44, 45)) -> dict:
    """Does this implementation actually do ergodic coverage on an open field?"""
    from ergodic_control_mppi.experiments.literature_methods import (
        _basis_values_np,
        _fourier_context,
    )
    from ergodic_control_mppi.metrics.modes import compute_mode_metrics

    ctx = _fourier_context(scenario, cfg.fourier_order)
    gmm = scenario.params.gmm
    means = np.asarray(gmm.means)
    inverses = np.asarray(gmm.covariance_inverse)
    delta_t = float(scenario.params.model.delta_t)

    def ergodic(points):
        basis = _basis_values_np(points, ctx.k_arr_np, ctx.x_min, ctx.x_max,
                                 ctx.y_min, ctx.y_max)
        coefficients = basis.mean(axis=0)
        return float(np.mean(ctx.lambda_k_np * (coefficients - ctx.phi_k_np) ** 2))

    reached, finals, bests, distances = [], [], [], []
    for seed in seeds:
        path = run_method(method, scenario, seed_state(state0, scenario, seed),
                          steps=steps, seed=seed, cfg=cfg)
        xy = path[:, :2]
        modes = compute_mode_metrics(xy, means, inverses, delta_t)
        reached.append(bool(np.isfinite(modes["first_all_modes_s"])))
        finals.append(ergodic(xy))
        # Best over the run, so a method that converges and then drifts is recorded as
        # having converged. The drift is reported separately, not used to disqualify.
        bests.append(min(ergodic(xy[:int(steps * f)]) for f in (0.25, 0.5, 0.75, 1.0)))
        distances.append(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))

    majority = sum(reached) > len(seeds) / 2
    travelled = float(np.median(distances))
    passed = bool(majority and travelled > 1.0)
    return {
        "method": method, "passed": passed, "seeds": list(seeds),
        "modes_reached": f"{sum(reached)}/{len(reached)}",
        "ergodic_best": float(np.median(bests)),
        "ergodic_final": float(np.median(finals)),
        "degrades_after_convergence": bool(np.median(finals) > 1.5 * np.median(bests)),
        "distance_m": travelled,
        "note": "" if passed else "not reproduced: did not visit all modes, or did not move",
    }


def seed_state(state0: np.ndarray, scenario: Scenario, seed: int) -> np.ndarray:
    """Per-seed start pose, shared by every method."""
    rng = np.random.default_rng(seed)
    state = np.asarray(state0, dtype=np.float64).copy()
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    state[0] = float(np.clip(state[0] + rng.uniform(-1.5, 1.5), x_min + 1.0, x_max - 1.0))
    state[1] = float(np.clip(state[1] + rng.uniform(-1.5, 1.5), y_min + 1.0, y_max - 1.0))
    return state


# ---------------------------------------------------------------------------- harness


def _open_scenario(config, span=(-20.0, 20.0, -10.0, 10.0), shape=(120, 240)):
    """
    An obstacle-free scenario on the deployment's workspace and target mixture.
    """
    from ergodic_control_mppi.experiments.common import build_target_grid

    params = config.controller
    grid = build_target_grid(params, shape)
    return Scenario(
        name="open", params=params, run_config=config.run,
        target_density_grid=grid,
        map_x_limits=(float(params.workspace.x_limits[0]),
                      float(params.workspace.x_limits[1])),
        map_y_limits=(float(params.workspace.y_limits[0]),
                      float(params.workspace.y_limits[1])),
        obstacle_map=np.zeros((0, 3)), safety_radius=0.30,
    )


def _open_arrays(scenario: Scenario, resolution: float = 0.15) -> dict:
    """Synthetic ``arrays.npz`` contents for an obstacle-free map."""
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    shape = (int((y_max - y_min) / resolution), int((x_max - x_min) / resolution))
    return {
        "target_grid": _resample_target(scenario, shape),
        "reachable_mask": np.ones(shape, dtype=bool),
        "occupancy": np.zeros(shape, dtype=bool),
        "grid_origin": np.array([x_min, y_min]),
        "grid_resolution": resolution,
        "initial_state": np.array([x_min + 0.5 * (x_max - x_min), 0.0, 0.0, 0.0, 0.0, 0.0]),
    }


#: Metric-grid shape the finite-trajectory certificate is defined on. Every coverage metric
#: in the paper compares on this grid, and the certificate has to be the *same* discrete
#: target for every method or the comparison is not paired. The clutter tier's archived
#: `target_grid` is already this shape, so for it nothing is resampled.
CERTIFICATE_BINS = (80, 80)

#: Sub-sampling of the executed path the certificate is applied to, matching
#: `scripts/theory_audit.py discrepancy --stride`. rho_n is then the empirical measure of
#: the audited samples and delta_n the gap at an audited step, which is how the bound is
#: stated.
CERTIFICATE_STRIDE = 20


def _certificate_target(tier: str, arrays: dict, scenario, limits_x, limits_y):
    """
    Support points and weights of the discrete target the certificate is stated against.
    """
    from ergodic_control_mppi.metrics.discrepancy import grid_target

    density = np.asarray(arrays["target_grid"], dtype=np.float64)
    mask = np.asarray(arrays["reachable_mask"], dtype=bool)
    if density.shape != CERTIFICATE_BINS:
        density = _resample_target(scenario, CERTIFICATE_BINS)
        mask = np.ones(CERTIFICATE_BINS, dtype=bool) if tier == "open" else mask
    return grid_target(density, mask, limits_x, limits_y)


def certificate_columns(positions: np.ndarray, support, weights, bandwidth: float) -> dict:
    """Score one executed path with the finite-trajectory MMD certificate."""
    from ergodic_control_mppi.metrics.discrepancy import walk

    # A descent gap needs a successor, so a path shorter than two strides is thinned less
    # rather than left unscored: the column set must not depend on the run length. Every
    # deployed run is 20,000 steps, where this clamps to CERTIFICATE_STRIDE exactly.
    stride = min(CERTIFICATE_STRIDE, max(1, len(positions) // 2))
    audited = np.asarray(positions[::stride], dtype=np.float64)
    result = walk(audited, support, weights, bandwidth)
    return {
        "mmd_final": float(result["error"][-1]),
        "mmd_bound": float(result["bound"][-1]),
        "mmd_trivial": float(result["trivial"]),
        "mmd_weighted_gap": float(result["weighted_gap"]),
        "mmd_prefix_holds": int(np.all(result["error"] <= result["bound"] + 1e-9)),
        "mmd_beats_trivial": int(result["bound"][-1] < result["trivial"]),
        "mmd_samples": int(len(audited)),
        "mmd_stride": stride,
    }


def run_tier(tier: str, methods, seeds, cfg: BaselineConfig, config_path: str,
             maps_path: Path, output: Path | None = None, *, overwrite: bool = False) -> list[dict]:
    """
    Fly every (method, map, seed) cell of one tier and score it like a campaign row.
    """
    from ergodic_control_mppi.config import load_config
    from ergodic_control_mppi.experiments.uav_pillar_tuning import _grid_config, score_run

    import jax

    # Recorded from the backend actually in use, not asserted. This was hardcoded to "cpu"
    # while the tier ran on cuda:0, so every row claimed a device it had not been flown on
    # -- and the backend moves these chaotic loops in the last float32 bits, which is the
    # whole reason the column exists.
    device = str(jax.devices()[0])
    rows: list[dict] = []
    if tier == "open":
        config = load_config(config_path)
        scenario = _open_scenario(config)
        cells = [("open", 0, config, scenario, _open_arrays(scenario),
                  {"map_seed": 0, "robot_radius": 0.30, "deadline_ms": 20.0})]
    else:
        manifest_all = json.loads(maps_path.read_text(encoding="utf-8"))
        cells = []
        for entry in manifest_all["maps"]:
            config, manifest, arrays = _grid_config(Path(entry["run_dir"]), config_path)
            scenario = _open_scenario(config)
            scenario = replace(scenario, name=f"{entry['obs_num']}p_{entry['map_seed']}")
            cells.append((scenario.name, entry["obs_num"], config, scenario, arrays,
                          manifest))

    fingerprints = {(method, name): cfg.fingerprint_for(method, config, arrays, manifest)
                    for name, _, config, _, arrays, manifest in cells for method in METHODS}
    record = {
        "execution": execution_record("ergodic_control_mppi/experiments/baselines.py", device),
        "tier": tier, "steps": cfg.steps, "seeds": list(seeds),
        "settings": cfg.as_dict(),
        "cells": [{"name": name, "controller": numerical_record(config.controller),
                   "arrays": numerical_record(arrays), "scoring": manifest,
                   "method_hashes": {m: fingerprints[m, name] for m in METHODS}}
                  for name, _, config, _, arrays, manifest in cells],
    }
    bundle_hash = ensure_bundle(output, record, overwrite) if output is not None else fingerprint(record)
    columns = ("method", "map", "seed", "steps", "config_hash")
    rows = verified_rows(output, columns) if output is not None else []
    for row in rows:
        if (row["config_hash"] != fingerprints.get((row["method"], row["map"]))
                or int(row["steps"]) != cfg.steps):
            raise ValueError(f"{output}: incompatible method inputs; use a fresh path or --overwrite")
    done = {(r["method"], r["map"], int(r["seed"]), int(r["steps"]), r["config_hash"])
            for r in rows}

    paths_directory = (output.with_name(output.stem + "_paths") if output is not None
                       else None)
    for name, obs_num, config, scenario, arrays, manifest in cells:
        occupancy = None if tier == "open" else np.asarray(arrays["occupancy"]).astype(bool)
        limits_x = tuple(float(v) for v in config.controller.workspace.x_limits)
        limits_y = tuple(float(v) for v in config.controller.workspace.y_limits)
        # One target per map, shared by every method on it: the certificate is only a paired
        # comparison if all five methods are scored against the identical discrete measure.
        support, weights = _certificate_target(tier, arrays, scenario, limits_x, limits_y)
        bandwidth = float(config.controller.field.fine_bandwidth)
        origin = tuple(map(float, np.asarray(arrays["grid_origin"])))
        resolution = float(arrays["grid_resolution"])
        state0 = np.asarray(arrays["initial_state"], dtype=np.float64)
        for method in methods:
            for seed in seeds:
                if (method, name, seed, cfg.steps, fingerprints[method, name]) in done:
                    continue
                started = time.perf_counter()
                path = run_method(method, scenario, seed_state(state0, scenario, seed),
                                  steps=cfg.steps, seed=seed, cfg=cfg,
                                  occupancy=occupancy, origin=origin,
                                  resolution=resolution)
                wall = time.perf_counter() - started
                row = score_run(
                    config, arrays, manifest, seed, cfg.steps,
                    positions=path[:, :2], velocities=path[:, 2:4],
                    # Not MPPI: there is no effective sample size or temperature to report,
                    # and a zero would read as a measured value rather than an absence.
                    ess_fractions=np.full(cfg.steps, np.nan),
                    temperatures=np.full(cfg.steps, np.nan),
                    wall=wall, device=device,
                )
                row.update({
                    "method": method, "tier": tier, "map": name, "obs_num": obs_num,
                    "seed": seed, "wall_seconds": wall,
                    # Stated per row so the caption cannot overclaim: which baselines were
                    # given an obstacle term they do not publish.
                    "added_avoidance": int(tier == "clutter"
                                           and method not in NATIVE_OBSTACLES),
                    "config_hash": fingerprints[method, name],
                    "bundle_hash": bundle_hash,
                    "effective_horizon": (config.controller.mppi.horizon if method == "ours"
                                          else cfg.sves_horizon if method == "sves" else 0),
                })
                if not np.isfinite(path).all():
                    raise ValueError(f"{method}/{name}/{seed}: nonfinite trajectory")
                if method == "ours" and int(row["collisions"]):
                    raise ValueError(f"ours/{name}/{seed}: collision; stopped")
                row.update(certificate_columns(path[:, :2], support, weights, bandwidth))
                rows.append(row)
                if paths_directory is not None:
                    # One file per cell rather than one bundle at the end: the tier runs for
                    # hours and resumes by identity, so a bundle written last would be lost
                    # on any interruption and would be incomplete after every resume.
                    paths_directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        paths_directory / f"{method}_{name}_s{seed}.npz",
                        positions=np.asarray(path[:, :2], dtype=np.float32),
                        method=method, map=name, obs_num=obs_num, seed=seed,
                        steps=cfg.steps, config_hash=fingerprints[method, name],
                    )
                if output is not None:
                    _append_row(output, row, rows)
                print(f"  [{tier}] {method:6s} {name:12s} s{seed} "
                      f"E_N={float(row['mmd_final']):.4g} "
                      f"fourier={float(row['fourier_ergodic']):.4g} {wall:.0f}s", flush=True)
    return rows


def _append_row(output: Path, row: dict, rows: list[dict]) -> None:
    """Append one scored cell, refusing a changed header."""
    fields = sorted({k for existing in rows for k in existing})
    header_ok = False
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
        header_ok = header == fields
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not header_ok:
        raise ValueError(f"{output}: stale header; use a fresh path or --overwrite")
    if not output.exists():
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
    with output.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(row)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("open", "clutter"), default="open")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="43,44,45,46,47,48,49,50,51,52,53,54",
                        help="comma-separated seed values (not a seed count)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--maps", type=Path,
                        default=Path("results/uav/campaign_maps.json"))
    parser.add_argument("--output", type=Path, default=Path("results/uav/baselines.csv"))
    parser.add_argument("--check-only", action="store_true",
                        help="run the open-field fidelity gate and exit")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("steps must be positive")

    cfg = BaselineConfig(steps=arguments.steps)
    methods = [m.strip() for m in arguments.methods.split(",")]
    seeds = [int(s) for s in arguments.seeds.split(",")]
    # `--seeds 6` reads as "six seeds" and parses as "the single seed 6". That silently ran
    # a whole clutter tier at one seed, outside the campaign's 43.. family, and the result
    # looked plausible enough to nearly ship. A lone small integer is always the mistake.
    if len(seeds) == 1 and seeds[0] < 43:
        raise SystemExit(
            f"--seeds takes a comma-separated list, not a count: got {arguments.seeds!r}, "
            f"which means the single seed {seeds[0]}. The campaign family starts at 43, "
            f"e.g. --seeds {','.join(str(43 + i) for i in range(seeds[0]))}"
        )

    from ergodic_control_mppi.config import load_config

    import jax

    fidelity_path = arguments.output.with_suffix(".fidelity.json")
    fidelity_hash = ensure_bundle(fidelity_path, {
        "config": numerical_record(load_config(arguments.config).controller),
        "settings": cfg.as_dict(), "methods": methods, "tier": arguments.tier,
        "execution": execution_record("ergodic_control_mppi/experiments/baselines.py", str(jax.devices()[0])),
    }, arguments.overwrite)
    config = load_config(arguments.config)
    scenario = _open_scenario(config)
    state0 = _open_arrays(scenario)["initial_state"]

    import jax

    # Recorded, because it changes the answer: the target grid is built with `jnp`, so the
    # backend moves it in the last float32 bits, and these loops are chaotic enough for
    # that to matter. A gate result without its device is not reproducible.
    device = str(jax.devices()[0])
    print(f"fidelity gate on {device}, {len(FIDELITY_SEEDS)} seeds", flush=True)
    # Ours is measured here too, though it is never a candidate for exclusion: the check
    # records `ergodic_best` alongside `ergodic_final`, and the gap between them is the
    # converge-then-degrade behaviour that two of the baselines show and `score_run` cannot
    # see, since it keeps only the final value. Running ours through the same measurement on
    # the same field is the only way that comparison is like-for-like.
    checks = (json.loads(fidelity_path.read_text()) if fidelity_path.exists() else
              [fidelity_check(m, scenario, state0, cfg=cfg, seeds=FIDELITY_SEEDS)
               for m in methods])
    for check in checks:
        check["device"] = device
        check["bundle_hash"] = fidelity_hash
    for check in checks:
        print(f"fidelity {check['method']:6s} "
              f"{'PASS' if check['passed'] else 'FAIL'}  "
              f"modes {check['modes_reached']}  "
              f"best {check['ergodic_best']:.2e} final {check['ergodic_final']:.2e}"
              f"{' (degrades)' if check['degrades_after_convergence'] else ''}  "
              f"{check['distance_m']:.0f} m  {check['note']}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".fidelity.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    if arguments.check_only:
        return

    if any(not c["passed"] for c in checks):
        print("Fidelity failures are retained and must be reported; no method is excluded.")

    rows = run_tier(arguments.tier, methods, seeds, cfg, arguments.config, arguments.maps,
                    output=arguments.output, overwrite=arguments.overwrite)
    print(f"wrote {arguments.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
