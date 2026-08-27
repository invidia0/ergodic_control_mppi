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
class SteinParams:
    """Stein-flow geometry and fading-memory coverage feedback.

    The memory term is a bank of ``memory_scales`` log-spaced bandwidths between
    ``fine_bandwidth`` and ``coarse_bandwidth``, each gauge-normalized so one
    ``memory_gain`` suffices; ``memory_balance`` interpolates trail avoidance
    against over-coverage correction. Both bandwidths are derived from the robot
    resolution and the target density rather than tuned.
    """

    memory_scales: int = field(metadata={"static": True})
    rotation: jax.Array
    self_bandwidth: float
    flow_weight: float
    coarse_bandwidth: float
    fine_bandwidth: float
    memory_decay: float
    reference_speed: float
    memory_gain: float
    memory_balance: float


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
    stein: SteinParams
    workspace: WorkspaceParams
    model: DoubleIntegratorParams
