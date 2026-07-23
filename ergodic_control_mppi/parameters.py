"""Immutable runtime parameters for the controller and experiments."""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from ergodic_control_mppi.models.double_integrator import DoubleIntegratorParams


@dataclass(frozen=True)
class RunConfig:
    """Non-controller simulation settings.

    Attributes:
        seed: JAX random seed.
        steps: Number of closed-loop control steps.
        num_robots: Number of independently controlled robots.
        resolution: Visualization grid resolution in workspace units.
    """

    seed: int
    steps: int
    num_robots: int
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
    cross_bandwidth: float
    flow_weight: float
    repulsion_weight: float


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MPPIParams:
    """Sampling and adaptive-temperature MPPI parameters."""

    samples: int = field(metadata={"static": True})
    horizon: int = field(metadata={"static": True})
    history_length: int = field(metadata={"static": True})
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ControllerParams:
    """Complete immutable parameter tree consumed by controller functions."""

    mppi: MPPIParams
    gmm: GMMParams
    stein: SteinParams
    workspace: WorkspaceParams
    model: DoubleIntegratorParams


@dataclass(frozen=True)
class ControllerVariant:
    """Typed optional overrides used by all experiment runners."""

    repulsion_weight: float | None = None
    cross_bandwidth: float | None = None
    flow_weight: float | None = None
    theta: float | None = None
    horizon: int | None = None
    history_length: int | None = None


def apply_variant(params: ControllerParams, variant: ControllerVariant) -> ControllerParams:
    """Apply a typed experiment variant through immutable nested replacements.

    Args:
        params: Base controller parameters.
        variant: Optional values to replace.

    Returns:
        A controller tree sharing all unchanged arrays with ``params``.

    Raises:
        ValueError: If an override is outside its supported range.
    """
    stein_changes: dict[str, object] = {}
    mppi_changes: dict[str, object] = {}
    if variant.repulsion_weight is not None:
        if variant.repulsion_weight < 0:
            raise ValueError("repulsion_weight must be >= 0")
        stein_changes["repulsion_weight"] = float(variant.repulsion_weight)
    if variant.cross_bandwidth is not None:
        if variant.cross_bandwidth <= 0:
            raise ValueError("cross_bandwidth must be > 0")
        stein_changes["cross_bandwidth"] = float(variant.cross_bandwidth)
    if variant.flow_weight is not None:
        if variant.flow_weight < 0:
            raise ValueError("flow_weight must be >= 0")
        stein_changes["flow_weight"] = float(variant.flow_weight)
    if variant.theta is not None:
        if not 0 <= variant.theta <= 90:
            raise ValueError("theta must be in [0, 90]")
        theta = jnp.deg2rad(jnp.asarray(variant.theta, dtype=jnp.float32))
        stein_changes["rotation"] = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]],
            dtype=jnp.float32,
        )
    if variant.horizon is not None:
        if variant.horizon < 1:
            raise ValueError("horizon must be >= 1")
        mppi_changes["horizon"] = int(variant.horizon)
    if variant.history_length is not None:
        if variant.history_length < 1:
            raise ValueError("history_length must be >= 1")
        mppi_changes["history_length"] = int(variant.history_length)
    return replace(
        params,
        stein=replace(params.stein, **stein_changes),
        mppi=replace(params.mppi, **mppi_changes),
    )
