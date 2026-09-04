"""Immutable runtime parameters for the controller."""

from dataclasses import dataclass, field

import jax

from ergodic_control_mppi.models.double_integrator import DoubleIntegratorParams


@dataclass(frozen=True)
class RunConfig:
    """Non-controller simulation settings.

    Attributes:
        seed: JAX random seed.
        steps: Number of closed-loop control steps.
        resolution: Visualization grid resolution in workspace units.
    """

    seed: int
    steps: int
    resolution: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GMMParams:
    """Precomputed terms for a two-dimensional Gaussian mixture."""

    means: jax.Array
    covariance: jax.Array
    covariance_inverse: jax.Array
    log_weights: jax.Array
    log_normalizers: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FieldParams:
    """The reference potential field and its service gate.

    Three terms, all gradients of explicit potentials in the query position: the
    analytic score of the (possibly deficit-bent) target, KDE repulsion from the
    fading memory of executed positions, and KDE repulsion of the plan from itself.
    One bandwidth ``fine_bandwidth`` governs both kernels, so ``memory_gain`` and
    ``plan_gain`` are commensurate under the shared ``sqrt(he/2)`` gauge.

    There is no rotation. ``R(theta) grad Phi`` is not a gradient unless
    ``R = I``, and the potential is the point.
    """

    track_weight: float
    fine_bandwidth: float
    memory_decay: float
    reference_speed: float
    memory_gain: float
    memory_balance: float
    plan_gain: float
    transit_speedup: float
    dwell_slowdown: float
    service_floor: float
    service_decay: float
    deficit_ceiling: float
    # Per-mode release: every component leaves at sigma* = release_ratio times its fair
    # share, with kappa_j = Delta_j / (sigma* - 1) read off the target's own log-odds gaps.
    # Static: it selects which bend `attraction_target` applies. <= 0 keeps the promotion-
    # only bend, which cannot overturn a Delta_j margin -- the necessity arm of the campaign.
    release_ratio: float = field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MPPIParams:
    """Sampling and adaptive-temperature MPPI parameters."""

    samples: int = field(metadata={"static": True})
    horizon: int = field(metadata={"static": True})
    memory_length: int = field(metadata={"static": True})
    # Static: it selects the convolution shape, and at 1 the filter is skipped entirely.
    smooth_window: int = field(metadata={"static": True})
    temperature: float
    alpha: float
    exploration: float
    ess_target: float
    temperature_min: float
    temperature_max: float
    covariance: jax.Array
    covariance_inverse: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WorkspaceParams:
    """Workspace boundaries, obstacle geometry, and constraint costs.

    The occupancy grid is a runtime input rather than a YAML knob: offline runs leave
    it empty and keep the circular-obstacle behavior unchanged, while a deployment
    supplies a rasterized map. Both sources are charged ``obstacle_cost``.

    Attributes:
        grid: Occupancy with shape ``(H, W)``, ``1.0`` where blocked, empty when unused.
        grid_origin: World coordinates of the lower-left corner of cell ``(0, 0)``.
        grid_resolution: Grid cell size in workspace units.
    """

    x_limits: jax.Array
    y_limits: jax.Array
    out_of_map_cost: float
    obstacles: jax.Array
    obstacle_cost: float
    safe_distance: float
    boundary_margin: float
    boundary_weight: float
    grid: jax.Array
    grid_origin: jax.Array
    grid_resolution: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ControllerParams:
    """Complete immutable parameter tree consumed by controller functions."""

    mppi: MPPIParams
    gmm: GMMParams
    field: FieldParams
    workspace: WorkspaceParams
    model: DoubleIntegratorParams
