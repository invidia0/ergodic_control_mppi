"""Single-robot comparison against ergodic-coverage baselines, in the open and in clutter.

Separate from :mod:`literature` on purpose. That module scores a *team*
(``team_ergodic_error``, ``pairwise_overlap``, ``R_pair``) for a different, multi-robot
paper, and it builds its scenarios through ``make_no_obstacle_scenario``, which hard-codes
an empty obstacle map. This one scores exactly what the ablation campaign scores, through
the same :func:`~ergodic_control_mppi.experiments.uav_pillar_tuning.score_run`, so a
baseline row and an ablation row are the same measurement and can sit in one table.

Two tiers, because the baselines are not all obstacle-aware:

``open``
    No obstacles. Every method runs at its own published formulation with nothing added,
    so the coverage law is compared with nobody handicapped. This is the tier that answers
    "is the coverage better".

``clutter``
    The campaign's pillar maps. HEDAC additionally gets the Neumann boundaries its own
    paper specifies. Every baseline also gets the shared penalty of :func:`_avoidance`,
    identical across methods so none is advantaged by better-tuned avoidance, and the fact
    is recorded per row so the caption can state it rather than quietly present a helped
    baseline as the published one. See :data:`NATIVE_OBSTACLES` for why HEDAC needs it
    despite having its own obstacle handling.

Every method drives the *same* double-integrator through the *same* tracker
(``literature_methods._tracker_step_np``) at the same speed limit. The comparison is
between coverage laws, not between vehicle models.

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
# Methods that need no added obstacle term. Anything not listed here is given the shared
# penalty of `_avoidance` in the clutter tier, and the row records it so the caption can say
# so rather than present a helped baseline as the published one.
#
# HEDAC is deliberately *not* listed, even though its Neumann boundaries are genuine
# published obstacle handling and it still gets them. The formulation assumes a first-order
# vehicle that exactly follows ``v_max * grad(u)/|grad(u)|``, which by construction never
# enters an obstacle. Ours is a second-order vehicle at 1.8 m/s, and the potential barrier
# around a 0.6 m pillar is thinner than its stopping distance, so it does enter -- and
# inside an obstacle the solve holds ``u = 0`` identically, so ``grad(u)`` is exactly zero,
# `_unit_field` commands nothing, and the vehicle stops there for the rest of the run. That
# is a trap with no escape: measured on 25p_516, HEDAC spent 87% of the run embedded in a
# pillar and travelled 77 m. With the shared term it never penetrates at all (0%, minimum
# clearance +0.14 m) and travels 332 m. Reporting the trapped number would have handed us a
# 10x win over the classical clutter baseline that is an artefact of our vehicle choice
# rather than a property of the method.
NATIVE_OBSTACLES = {"ours"}


# --------------------------------------------------------------------------- obstacles


def _solver_shape(scenario: Scenario, long_side: int) -> tuple[tuple[int, int], float]:
    """Grid dimensions with *square* cells, plus the cell pitch in metres.

    The deployment workspace is $40\\times20$\\,m. A square ``grid_size`` grid over it puts
    the cell pitch at $0.5$\\,m in $x$ and $0.25$\\,m in $y$, and three separate pieces of
    physics silently assume it is isotropic: the five-point Jacobi stencil of
    :func:`_jacobi_neumann` weights all four neighbours equally, ``np.gradient`` called
    without a spacing argument returns a derivative per *index* rather than per metre, and
    the scalar ``sigma`` of the FMEC and HEDAC Gaussian filters is a count of cells. On an
    anisotropic grid all three are wrong by the aspect ratio, which tilts every field
    toward the coarse axis. Square cells make them right.
    """
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    width, height = float(x_max - x_min), float(y_max - y_min)
    pitch = max(width, height) / long_side
    return (max(int(round(height / pitch)), 1), max(int(round(width / pitch)), 1)), pitch


def _cell_index(values: np.ndarray, edges: np.ndarray, count: int) -> np.ndarray:
    """Which grid cell each coordinate falls in.

    One helper because there were two conventions in play: the coverage histogram and FMEC
    binned by cell, while HEDAC read its gradient at ``(x - x0)/W * (N - 1)``, a *node*
    mapping half a cell offset from the other. HEDAC was therefore sampling its gradient
    0.125 m from where its own coverage had been deposited.
    """
    return np.clip(np.searchsorted(edges, values) - 1, 0, count - 1)


def _blocked_mask(occupancy: np.ndarray, shape: tuple[int, int], scenario: Scenario,
                  origin=None, resolution: float | None = None) -> np.ndarray:
    """Resample an occupancy grid onto the solver grid by nearest neighbour.

    Nearest neighbour rather than area averaging: a partially covered cell must come out
    blocked, not half blocked. A diffusion solve with a fractional obstacle leaks through
    it, which is exactly the failure the Neumann treatment exists to prevent.

    Sampled by *world coordinate*, not by index ratio. The two are not the same here: the
    occupancy grid is 267x134 at 0.15 m, so it spans 40.05 x 20.10 m while the workspace is
    40 x 20. Assuming coincident extents stretches the mask by up to 0.05 m in x and 0.10 m
    in y, growing toward the far corner, so HEDAC's walls drifted away from the pillar
    circles that `_pillar_circles` places from the same grid in true coordinates.
    """
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
    """Solve the HEDAC screened-Poisson problem with no-flux obstacle boundaries.

    The canonical HEDAC formulation solves ``alpha * laplacian(u) - u = -q`` subject to
    ``du/dn = 0`` on the boundary *and on every obstacle*. Discretely, zero flux across a
    face means the neighbour behind that face contributes the centre cell's own value: the
    stencil reflects instead of reading into the obstacle. Occupied cells carry no source
    and hold no potential, so nothing diffuses out of an obstacle either.

    Getting this right is what makes HEDAC a fair baseline in clutter rather than a
    handicapped one -- with a plain stencil the potential diffuses straight through the
    pillars and the gradient points into them.

    The *domain* boundary is no-flux for the same reason: the workspace is closed and no
    heat leaves it. Forcing ``u = 0`` at the edge instead, as a naive Jacobi loop over the
    interior does, makes the potential rise monotonically toward every wall, so the
    gradient points out of the domain from everywhere inside it. That version failed the
    fidelity gate by driving straight into the south wall and pinning there for the
    remaining 3500 steps.

    ``warm`` continues from the previous step's potential. HEDAC *evolves* its field in
    time rather than re-solving to steady state at every instant, so carrying it forward is
    the faithful reading as well as the cheap one: a handful of relaxation sweeps per
    control step tracks a source that barely moves, where a cold solve needs dozens.
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
    """Where the five-point stencil must reflect: obstacles, plus the domain edge.

    ``np.roll`` wraps, so the row or column that comes round from the far side is not a
    real neighbour. Marking it a wall is what makes the *domain* boundary no-flux as well:
    forcing ``u = 0`` there instead puts a ramp against every wall, which is what once
    drove HEDAC into the south boundary and pinned it there.
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
    """Rescale a coverage-law field to a commanded speed, keeping its direction.

    HEDAC, SMC and FMEC all produce a *direction*: their magnitudes are gradients of a
    potential, a spectral mismatch and a log-density ratio respectively, in three unrelated
    and arbitrarily scaled units, all of which decay toward zero as coverage improves. The
    published control laws move along that direction at the vehicle's speed -- HEDAC's is
    literally ``v_max * grad(u)/|grad(u)|``.

    Capping with `_limit_speed` alone is what made the first run of the fidelity gate fail:
    the raw fields came out at a fraction of a metre per second and fell from there, so
    every baseline crawled and would have "lost" for a reason that is an artefact of unit
    choice rather than of the method. Normalising is both faithful and the only way the
    comparison is about coverage rather than about gain tuning.

    A field that has genuinely collapsed (below ``1e-9``) yields zero rather than a
    direction amplified out of numerical noise.
    """
    magnitude = np.linalg.norm(field, axis=1, keepdims=True)
    return np.where(magnitude > 1e-9, field / np.maximum(magnitude, 1e-12) * speed, 0.0)


def _avoidance(xy: np.ndarray, centres: np.ndarray, radius: np.ndarray,
               clearance: float, gain: float) -> np.ndarray:
    """Shared repulsion for baselines whose formulation defines none.

    A single inverse-distance push out of the inflated footprint, identical for every
    method that needs it, so no baseline is advantaged by a better-tuned avoidance term
    than another. Any row produced with this active is flagged ``added_avoidance``.
    """
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
    """Fit one circle per connected component of the occupancy grid.

    The campaign's manifests record the pillar count and a radius *range*, never the
    individual centres, so the circles the baselines need for their avoidance term have to
    be recovered from the grid. Labelling is a four-neighbour flood fill over occupied
    cells; each component's centroid and its furthest cell give the centre and radius.
    """
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
    """HEDAC: follow the gradient of the potential driven by the coverage deficit.

    The coverage is smoothed by the sensor footprint before the deficit is formed, which is
    part of the formulation and not a numerical nicety: HEDAC's coverage field is the
    integral of a *sensor function* along the path, not a histogram of visited cells.
    Depositing into a single cell instead makes the potential a symmetric spike centred on
    the agent, and the central difference of a symmetric spike at its own centre is zero --
    the agent sits at the bottom of a well it dug itself and never leaves. That is exactly
    how this failed the fidelity gate the first time, at 10 m travelled in 80 s.
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
    """SMC (Mathew & Mezic): descend the Fourier ergodic metric pointwise.

    The first-order feedback law: move along the negative gradient of the ergodic metric
    with respect to the current position, with the coefficients accumulated over the
    trajectory so far.
    """
    from ergodic_control_mppi.experiments.literature_methods import _basis_and_grad_np

    basis, grad = _basis_and_grad_np(state_xy, ctx)
    del basis
    weight = ctx.lambda_k_np * (coefficients / max(elapsed, 1.0) - ctx.phi_k_np)
    return -np.einsum("k,nki->ni", weight, grad)


def _sves_planner(ctx, model_params, cfg):
    """Stein Variational Ergodic Search: SVGD over a population of control sequences.

    The method represents the trajectory posterior by particles rather than by a single
    optimum, and pushes them with Stein variational gradient descent against an ergodic
    objective. With ``M`` control-sequence particles ``U_i`` over a horizon ``H``,

        phi(U_i) = (1/M) sum_j [ k(U_j, U_i) grad_{U_j} log p(U_j) + grad_{U_j} k(U_j, U_i) ]

    where ``log p(U) = -J_erg(U) / temperature``. The first term drives every particle
    downhill on the shared objective; the second is the repulsion that keeps the population
    from collapsing onto one mode, which is the whole point of the method and the reason it
    explores where a single-trajectory optimiser stalls.

    Implemented receding-horizon: the ergodic cost is evaluated on the *cumulative* Fourier
    coefficients -- what the vehicle has already covered plus what the horizon would add --
    so the plan responds to coverage history rather than re-solving from scratch. Returns a
    jitted closure; building it once per run keeps the compile out of the step loop.
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
    """Flow-Matching Ergodic Coverage: follow the transport field from coverage to target.

    FMEC replaces the spectral ergodic objective with a flow: it builds a velocity field
    that transports the agent's current coverage distribution onto the target and tracks
    it. The field used here is the Wasserstein gradient flow of ``KL(c || p)``,

        v(x) = grad log p(x) - grad log c(x),

    i.e. the score difference between the target and the kernel-smoothed empirical
    coverage. This is the transport direction that annihilates the mismatch, it is what
    "matching the flow between the two densities" reduces to for this pair, and it needs no
    Fourier truncation -- which is the property the method is published for.

    Both densities are smoothed on the same grid, so the two scores are differenced at
    equal resolution and the field does not inherit the sampling noise of the coverage
    histogram.
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


# ------------------------------------------------------------------------------ config


class BaselineConfig:
    """Tunables for every baseline, in one place so the settings are auditable.

    Defaults are the published-formulation settings where a paper states one and the
    existing `literature_comparison.yaml` values otherwise, so this is not a fresh tuning
    pass dressed up as a comparison.
    """

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
        # criterion (`scripts/baseline_param_sweep.py`, GPU, 8 seeds -- the device the tiers
        # fly, because a CPU sweep disagreed with the GPU gate on mode counts). 1.0, 2.0 and
        # 3.0 m all reach every mode in 8/8; best metric is 1.97e-4, 2.63e-4 and 8.71e-4, so
        # the rule -- mode reach first, `ergodic_best` as tie-break -- picks 1.0 m. That is
        # the setting where the baseline is *strongest*, which is the direction this choice
        # must err in. 4.0 m fails outright at 0/3.
        self.hedac_sensor = 1.0
        # SVES
        # Matched to our own controller's planning horizon (T = 350) so the
        # comparison is not won on lookahead. 16 particles over 350 steps costs
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

    def fingerprint_for(self, method: str) -> str:
        """Hash of only the settings ``method`` reads.

        A whole-config hash is too coarse: retuning ``hedac_sensor`` would mark every SMC
        row stale and re-fly 96 cells that could not possibly have changed. That is not
        hypothetical -- it happened once already, between stamping the surviving rows and
        settling the last parameter.
        """
        import hashlib

        settings = self.as_dict()
        keys = self.DEPENDS_ON.get(method, tuple(k for k in settings if k != "steps"))
        payload = json.dumps({k: settings[k] for k in keys}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def fingerprint(self) -> str:
        """Short hash of every setting, stamped on each row.

        Resume-by-identity keys on ``(method, map, seed)``, so a row flown under different
        settings is indistinguishable from a current one and gets skipped. That happened
        three times in one afternoon -- the solver grid changed, then two parameters changed
        units -- and each time the stale rows had to be found and deleted by hand, which is
        a step that will eventually be forgotten. With the fingerprint on the row, resume
        skips only what genuinely matches and re-flies the rest on its own.

        ``steps`` is excluded: it is a property of the invocation and already recorded.
        """
        import hashlib

        settings = {k: v for k, v in self.as_dict().items() if k != "steps"}
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------------------ driver


def run_method(method: str, scenario: Scenario, state0: np.ndarray, *, steps: int,
               seed: int, cfg: BaselineConfig, occupancy=None, origin=None,
               resolution: float = 0.15) -> np.ndarray:
    """Fly one method for ``steps`` and return the executed states, shape ``(steps, 6)``.

    ``occupancy`` is ``None`` for the open tier. In the clutter tier every method sees the
    obstacles: HEDAC through its Neumann boundaries, the rest through :func:`_avoidance`.
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
    """One SVES control: refresh the particle population, then track the best plan.

    Replanning every step would spend the whole budget in SVGD for no benefit -- the
    population is warm and the objective moves slowly -- so the particles are updated every
    ``sves_replan_every`` steps and the best plan is otherwise followed open-loop, which is
    how a receding-horizon sampler is run in practice.

    The plan is executed as a **waypoint**, not as its first acceleration. The particles
    parameterise acceleration, and one integration step of it changes the current velocity
    by at most ``a_max * dt``, so a desired velocity built that way points wherever the
    vehicle was already going. Normalised to the commanded speed like the gradient laws,
    that is a straight line: the first version of this flew 708 m without covering
    anything, at a metric that got *worse*. Steering at a point ``sves_lookahead`` steps
    down the planned trajectory instead makes the plan actually determine the direction.
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
    """Does this implementation actually do ergodic coverage on an open field?

    A reimplementation that silently does not work would show up as a loss for the baseline
    and a win for us, which is the single most dishonest failure mode available here. The
    check is deliberately weak and mechanical, so it catches breakage rather than grading
    quality: on an obstacle-free map with a multi-modal target, a working ergodic
    controller must (i) end with a lower Fourier ergodic metric over the whole path than it
    had over the first half, and (ii) not sit still.

    Run at the campaign's own 20 000 steps, not at some shorter convenience horizon. HEDAC
    is a local gradient law and needs most of that to reach modes 24 m apart; failing it at
    4 000 steps would have recorded "not reproduced" for a method that was merely still in
    transit, which is the same dishonesty in the other direction.

    **The criterion is absolute: every mode must be visited.** An earlier version asked
    only whether the metric improved relative to the method's own first half, and it got
    both interesting verdicts backwards. It passed FMEC, which sat between $4$ and
    $6\\times10^{-3}$ for the whole run and never came within $13$\\,m of the third mode --
    uniformly mediocre scores well on a self-relative test precisely because it never got
    good. It failed HEDAC, which reached $2.7\\times10^{-4}$ by mid-run, the best of any
    baseline, and was penalised for degrading afterwards. Coverage of a trimodal target is
    the thing being reproduced, so mode visitation is what the gate asks about, using the
    campaign's own ``compute_mode_metrics`` definition rather than a second opinion.

    **Over several seeds, and decided on the median.** These are closed feedback loops on a
    chaotic system: this project has already established that a one-ULP perturbation is
    enough to change a run's outcome, and the target grid is built with ``jnp``, so merely
    running on the GPU instead of the CPU moves it in the last float32 bits. A single-run
    version of this check duly passed HEDAC on CPU and failed it on GPU with identical code.
    Seeds vary the *initial state*, shared across methods: three of the four baselines are
    deterministic laws that ignore a controller seed entirely, so seeding anything else
    gives three identical runs and a false sense of replication.

    A method that fails is reported as ``not reproduced`` and excluded from the tables
    rather than presented as a beaten baseline. Degrading after convergence is *not* a
    failure -- it is a result, and it is reported as one.
    """
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
    """Per-seed start pose, shared by every method.

    Three of the four baselines are deterministic feedback laws: they consume no random
    numbers, so flying them twelve times from one start gives twelve identical rows, zero
    variance, and a paired test against our sampling-based controller that means nothing.
    The seed therefore has to vary something about the *trial* rather than something inside
    one method, and the start pose is the natural choice -- every method sees the same
    twelve starts and the pairing is over a shared quantity.

    Drawn in a band around the archived start rather than anywhere in the workspace, so a
    seed cannot begin inside a pillar on the cluttered maps.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(state0, dtype=np.float64).copy()
    x_min, x_max = scenario.map_x_limits
    y_min, y_max = scenario.map_y_limits
    state[0] = float(np.clip(state[0] + rng.uniform(-1.5, 1.5), x_min + 1.0, x_max - 1.0))
    state[1] = float(np.clip(state[1] + rng.uniform(-1.5, 1.5), y_min + 1.0, y_max - 1.0))
    return state


# ---------------------------------------------------------------------------- harness


def _open_scenario(config, span=(-20.0, 20.0, -10.0, 10.0), shape=(120, 240)):
    """An obstacle-free scenario on the deployment's workspace and target mixture.

    Tier 1 has to isolate the coverage law, so everything except the obstacles is held at
    the deployment's own settings -- same workspace, same GMM, same vehicle.
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
    """Synthetic ``arrays.npz`` contents for an obstacle-free map.

    `score_run` reads its geometry from a campaign map's archive. The open tier has no such
    archive, so one is synthesised with the same keys: everything free, everything
    reachable, and the target grid the scenario already carries. Scoring then goes through
    exactly the same code as the ablation rows.
    """
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


def run_tier(tier: str, methods, seeds, cfg: BaselineConfig, config_path: str,
             maps_path: Path, output: Path | None = None) -> list[dict]:
    """Fly every (method, map, seed) cell of one tier and score it like a campaign row.

    Rows are appended to ``output`` as they finish, not collected and written at the
    end. The clutter tier is $8$ maps by $5$ methods by $12$ seeds and runs for hours;
    holding all of it in memory until the last cell means one interruption discards the
    lot. Resuming is by identity: a cell already present in the file is skipped, so a
    re-run costs only what never finished.
    """
    from ergodic_control_mppi.config import load_config
    from ergodic_control_mppi.experiments.uav_pillar_tuning import _grid_config, score_run

    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    fingerprints = {m: cfg.fingerprint_for(m) for m in METHODS}
    if output is not None and output.exists():
        stale = 0
        with output.open(encoding="utf-8", newline="") as stream:
            for existing in csv.DictReader(stream):
                # Only a row flown under *these* settings is kept. Rows from other settings
                # are dropped rather than carried forward, so the output file is always the
                # product of a single harness: keeping them would leave two rows for the
                # same cell and let load order decide which one the paper reports.
                if existing.get("config_hash") == fingerprints.get(existing["method"]):
                    rows.append(existing)
                    done.add((existing["method"], existing["map"], int(existing["seed"])))
                else:
                    stale += 1
        print(f"resuming: {len(done)} cells match their method's settings"
              + (f", {stale} rows from other settings dropped and re-flown" if stale else ""),
              flush=True)

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

    for name, obs_num, config, scenario, arrays, manifest in cells:
        occupancy = None if tier == "open" else np.asarray(arrays["occupancy"]).astype(bool)
        origin = tuple(map(float, np.asarray(arrays["grid_origin"])))
        resolution = float(arrays["grid_resolution"])
        state0 = np.asarray(arrays["initial_state"], dtype=np.float64)
        for method in methods:
            for seed in seeds:
                if (method, name, seed) in done:
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
                    wall=wall, device="cpu",
                )
                row.update({
                    "method": method, "tier": tier, "map": name, "obs_num": obs_num,
                    "seed": seed, "wall_seconds": wall,
                    # Stated per row so the caption cannot overclaim: which baselines were
                    # given an obstacle term they do not publish.
                    "added_avoidance": int(tier == "clutter"
                                           and method not in NATIVE_OBSTACLES),
                    "config_hash": fingerprints[method],
                })
                rows.append(row)
                if output is not None:
                    _append_row(output, row, rows)
                print(f"  [{tier}] {method:6s} {name:12s} s{seed} "
                      f"fourier={float(row['fourier_ergodic']):.4g} {wall:.0f}s", flush=True)
    return rows


def _append_row(output: Path, row: dict, rows: list[dict]) -> None:
    """Append one scored cell, rewriting the header if a new column has appeared.

    `score_run` can return a different key set for a method that reports something the
    others do not, and a plain append would then silently misalign the columns. Cheap
    insurance: when the field set grows, the whole file is rewritten once.
    """
    fields = sorted({k for existing in rows for k in existing})
    header_ok = False
    if output.exists():
        with output.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), [])
        header_ok = header == fields
    output.parent.mkdir(parents=True, exist_ok=True)
    if not header_ok:
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
    parser.add_argument("--seeds", default="43,44,45,46,47,48,49,50,51,52,53,54")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--config", default="configs/uav_profile.yaml")
    parser.add_argument("--maps", type=Path,
                        default=Path("results/uav/campaign_maps.json"))
    parser.add_argument("--output", type=Path, default=Path("results/uav/baselines.csv"))
    parser.add_argument("--check-only", action="store_true",
                        help="run the open-field fidelity gate and exit")
    arguments = parser.parse_args()

    cfg = BaselineConfig(steps=arguments.steps)
    methods = [m.strip() for m in arguments.methods.split(",")]
    seeds = [int(s) for s in arguments.seeds.split(",")]

    from ergodic_control_mppi.config import load_config

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
    checks = [fidelity_check(m, scenario, state0, cfg=cfg, seeds=FIDELITY_SEEDS)
              for m in methods]
    for check in checks:
        check["device"] = device
    for check in checks:
        print(f"fidelity {check['method']:6s} "
              f"{'PASS' if check['passed'] else 'FAIL'}  "
              f"modes {check['modes_reached']}  "
              f"best {check['ergodic_best']:.2e} final {check['ergodic_final']:.2e}"
              f"{' (degrades)' if check['degrades_after_convergence'] else ''}  "
              f"{check['distance_m']:.0f} m  {check['note']}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    (arguments.output.parent / "baseline_fidelity.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    if arguments.check_only:
        return

    # A method that did not reproduce is dropped rather than reported as beaten. Ours is
    # exempt: the gate exists to certify *reimplementations* of other people's work, and
    # excluding our own controller from its own comparison on it would be incoherent.
    failed = {c["method"] for c in checks if not c["passed"] and c["method"] != "ours"}
    if failed:
        print(f"excluded, did not reproduce: {', '.join(sorted(failed))}")
    kept = [m for m in methods if m not in failed]

    rows = run_tier(arguments.tier, kept, seeds, cfg, arguments.config, arguments.maps,
                    output=arguments.output)
    print(f"wrote {arguments.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
