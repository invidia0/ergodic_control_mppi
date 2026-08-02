"""Device selection, initialization, dispatch, and host conversion."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import AppConfig
from ergodic_control_mppi.mppi.single import run_single
from ergodic_control_mppi.parameters import ControllerParams


@dataclass(frozen=True)
class SimulationResult:
    """Consumer-facing NumPy simulation outputs.

    Attributes:
        paths: Executed states with normalized shape ``(N, R, 6)``.
        optimal_trajectories: Final plans with shape ``(R, T, 6)``.
        surrogates: Final shared position paths with shape ``(R, T, 2)``.
        ess_fractions: Per-step effective sample-size fractions with shape ``(N,)``.
        temperatures: Per-step adapted temperatures with shape ``(N,)``.
        initial_states: Initial states with shape ``(R, 6)``.
        device: JAX device platform used for execution.
    """

    paths: np.ndarray
    optimal_trajectories: np.ndarray
    surrogates: np.ndarray
    ess_fractions: np.ndarray
    temperatures: np.ndarray
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


def controller_key(seed: int) -> jax.Array:
    """Return the controller half of the stable run-seed split."""
    return jax.random.split(jax.random.PRNGKey(seed))[0]


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


def run_simulation(
    config: AppConfig,
    device: str = "auto",
    progress: bool = False,
    initial_state: jax.Array | np.ndarray | None = None,
    preflight_steps: int = 0,
) -> SimulationResult:
    """Initialize and execute the configured single-robot controller.

    Args:
        config: Validated simulation configuration.
        device: Requested JAX device selection.
        progress: Whether to print periodic closed-loop progress.
        initial_state: Explicit start state with shape ``(6,)``, or ``None`` to sample
            one from the configured seed. Supplying it lets a replayed trial start
            exactly where the original one did.
        preflight_steps: Stationary planning iterations retained before motion starts.

    Returns:
        Host-resident simulation arrays and the selected device platform.
    """
    selected = select_device(device)
    params = jax.device_put(config.controller, selected)
    # The key is split the same way either way, so a seed keeps its meaning.
    simulation_key, state_key = jax.random.split(jax.random.PRNGKey(config.run.seed))
    if initial_state is None:
        initial_states = random_state(state_key, params)[None, :]
    else:
        initial_states = jnp.asarray(initial_state, dtype=jnp.float32).reshape(1, 6)
    controls = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
    result = jax.jit(
        run_single, static_argnames=("steps", "progress", "preflight_steps")
    )(
        params,
        jax.device_put(initial_states[0], selected),
        jax.device_put(controls, selected),
        simulation_key,
        steps=config.run.steps,
        progress=progress,
        preflight_steps=preflight_steps,
    )
    paths = result.path[:, None, :]
    optimal = result.optimal_trajectory[None, :, :]
    surrogates = result.surrogate[None, :, :]
    return SimulationResult(
        paths=np.asarray(paths),
        optimal_trajectories=np.asarray(optimal),
        surrogates=np.asarray(surrogates),
        ess_fractions=np.asarray(result.ess_fraction),
        temperatures=np.asarray(result.temperature),
        initial_states=np.asarray(initial_states),
        device=selected.platform,
    )
