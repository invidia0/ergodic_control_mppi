"""Device selection, initialization, dispatch, and host conversion."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import AppConfig
from ergodic_control_mppi.mppi.multi import run_multi
from ergodic_control_mppi.mppi.single import run_single
from ergodic_control_mppi.parameters import ControllerParams


@dataclass(frozen=True)
class SimulationResult:
    """Consumer-facing NumPy simulation outputs.

    Attributes:
        paths: Executed states with normalized shape ``(N, R, 6)``.
        optimal_trajectories: Final plans with shape ``(R, T, 6)``.
        surrogates: Final shared position paths with shape ``(R, T, 2)``.
        initial_states: Initial states with shape ``(R, 6)``.
        device: JAX device platform used for execution.
    """

    paths: np.ndarray
    optimal_trajectories: np.ndarray
    surrogates: np.ndarray
    initial_states: np.ndarray
    device: str


def select_device(requested: str = "auto") -> jax.Device:
    """Resolve ``auto``, ``cpu``, or ``gpu`` without import-time discovery.

    Raises:
        ValueError: If the selection is invalid or a requested GPU is absent.
    """
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu")
    if requested != "cpu":
        try:
            devices = jax.devices("gpu")
            if devices:
                return devices[0]
        except RuntimeError:
            if requested == "gpu":
                raise ValueError("GPU requested but no JAX GPU device is available") from None
    if requested == "gpu":
        raise ValueError("GPU requested but no JAX GPU device is available")
    return jax.devices("cpu")[0]


def random_state(key: jax.Array, params: ControllerParams) -> jax.Array:
    """Sample one finite initial state within configured map limits."""
    keys = jax.random.split(key, 6)
    workspace = params.workspace
    return jnp.array(
        [
            jax.random.uniform(keys[1], (), minval=workspace.x_limits[0], maxval=workspace.x_limits[1]),
            jax.random.uniform(keys[2], (), minval=workspace.y_limits[0], maxval=workspace.y_limits[1]),
            jax.random.uniform(keys[3], (), minval=-1.0, maxval=1.0),
            jax.random.uniform(keys[4], (), minval=-1.0, maxval=1.0),
            jax.random.uniform(keys[5], (), minval=-jnp.pi, maxval=jnp.pi),
            jax.random.uniform(keys[0], (), minval=-1.0, maxval=1.0),
        ],
        dtype=jnp.float32,
    )


def run_simulation(config: AppConfig, device: str = "auto") -> SimulationResult:
    """Initialize and execute the configured single- or multi-robot controller."""
    selected = select_device(device)
    params = jax.device_put(config.controller, selected)
    root_key = jax.random.PRNGKey(config.run.seed)

    if config.run.num_robots == 1:
        simulation_key, state_key = jax.random.split(root_key)
        initial_states = random_state(state_key, params)[None, :]
        controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
        result = jax.jit(run_single, static_argnames=("steps",))(
            params,
            initial_states[0],
            jax.device_put(controls, selected),
            simulation_key,
            steps=config.run.steps,
        )
        paths = result.path[:, None, :]
        optimal = result.optimal_trajectory[None, :, :]
        surrogates = result.surrogate[None, :, :]
    else:
        state_keys = jax.random.split(root_key, config.run.num_robots)
        initial_states = jax.vmap(random_state, in_axes=(0, None))(state_keys, params)
        simulation_keys = jax.random.split(jax.random.fold_in(root_key, 1), config.run.num_robots)
        controls = jnp.zeros(
            (config.run.num_robots, params.mppi.horizon, 3), dtype=jnp.float32
        )
        result = jax.jit(run_multi, static_argnames=("steps",))(
            params,
            initial_states,
            jax.device_put(controls, selected),
            simulation_keys,
            steps=config.run.steps,
        )
        paths = result.paths
        optimal = result.optimal_trajectories
        surrogates = result.surrogates
    return SimulationResult(
        paths=np.asarray(paths),
        optimal_trajectories=np.asarray(optimal),
        surrogates=np.asarray(surrogates),
        initial_states=np.asarray(initial_states),
        device=selected.platform,
    )
