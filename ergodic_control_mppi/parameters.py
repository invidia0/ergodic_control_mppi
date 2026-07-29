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
    covariance_inverse: jax.Array
    log_weights: jax.Array
    log_normalizers: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SteinParams:
    """Stein-flow geometry and cost weights."""

    rotation: jax.Array
    self_bandwidth: float
    flow_weight: float
    repulsion_weight: float
    repulsion_bandwidth: float
    memory_decay: float
    reference_speed: float
    deficit_gate: float
    spiral_bandwidth: float
    spiral_weight: float
    spiral_deficit: float
    eject_fill_gated: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MPPIParams:
    """Sampling and adaptive-temperature MPPI parameters."""

    samples: int = field(metadata={"static": True})
    horizon: int = field(metadata={"static": True})
    memory_length: int = field(metadata={"static": True})
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
    """Workspace boundaries, obstacle geometry, and constraint costs."""

    x_limits: jax.Array
    y_limits: jax.Array
    out_of_map_cost: float
    obstacles: jax.Array
    obstacle_cost: float
    safe_distance: float
    boundary_margin: float
    boundary_weight: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ControllerParams:
    """Complete immutable parameter tree consumed by controller functions."""

    mppi: MPPIParams
    gmm: GMMParams
    stein: SteinParams
    workspace: WorkspaceParams
    model: DoubleIntegratorParams
