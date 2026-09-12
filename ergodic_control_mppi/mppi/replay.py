"""
Recover the MPPI sample cloud behind a recorded control step, for figures.
"""

from typing import NamedTuple

import jax
import numpy as np

from ergodic_control_mppi.mppi.core import _rollouts, mppi_step, sample_epsilon
from ergodic_control_mppi.mppi.single import SingleControllerState
from ergodic_control_mppi.parameters import ControllerParams


class RolloutBundle(NamedTuple):
    """Sample cloud and plan for one MPPI planning step.

    Attributes:
        positions: Rollout positions, shape ``(K, T, d)``.
        weights: Normalized MPPI weights, shape ``(K,)``.
        costs: Per-rollout costs, shape ``(K,)``.
        optimal: Selected state trajectory, shape ``(T, 2d + 2)``.
        surrogate: Median path for flow evaluation, shape ``(T, d)``.
        memory: Fading-memory buffer, shape ``(P, d)``.
        service_mass: Per-component visit mass, shape ``(J,)``.
        state: Planning state, shape ``(2d + 2,)``.
    """

    positions: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    optimal: np.ndarray
    surrogate: np.ndarray
    memory: np.ndarray
    service_mass: np.ndarray
    state: np.ndarray


def replay_step(params: ControllerParams, carry: SingleControllerState) -> RolloutBundle:
    """
    Re-run one planning step and return its sample cloud.
    
    Args:
            params: The same controller parameters the step ran under.
            carry: A recorded closed-loop carry.
    Returns:
            The rollout cloud, weights, plan, and memory for that step.
    """
    epsilon, _ = sample_epsilon(carry.key, params)
    costs, _, positions = _rollouts(
        params, carry.state, carry.controls, epsilon, carry.temperature
    )
    # `service_mass` is not optional here even though `mppi_step` defaults it: without it
    # the replay reads the service ratio out of the trail instead of the accumulator, which
    # is a different field and therefore a different step -- exactly the drift this module
    # exists to prevent.
    result = mppi_step(
        params, carry.controls, carry.state, carry.key, carry.temperature, carry.memory,
        carry.service_mass,
    )
    return RolloutBundle(
        positions=np.asarray(positions),
        weights=np.asarray(result.weights),
        costs=np.asarray(costs),
        optimal=np.asarray(result.optimal_trajectory),
        surrogate=np.asarray(result.surrogate),
        memory=np.asarray(carry.memory),
        service_mass=np.asarray(carry.service_mass),
        state=np.asarray(carry.state),
    )


def snapshot_arrays(snapshots: list[SingleControllerState]) -> dict[str, np.ndarray]:
    """
    Stack recorded carries into ``npz``-writable arrays.
    
    Args:
            snapshots: Carries captured during a run, in step order.
    Returns:
            Mapping of ``snap_*`` array names to stacked values.
    """
    # The key is a typed PRNG array and refuses np.asarray; store its raw words instead.
    # Without the key a snapshot cannot be replayed at all, so it is not optional.
    plain = [
        state._replace(key=jax.random.key_data(state.key)) for state in snapshots
    ]
    stacked = jax.tree.map(lambda *leaves: np.stack([np.asarray(x) for x in leaves]), *plain)
    return {f"snap_{name}": value for name, value in stacked._asdict().items()}


def restore_snapshot(arrays, index: int) -> SingleControllerState:
    """
    Rebuild one carry from the arrays written by :func:`snapshot_arrays`.
    
    Args:
            arrays: An ``npz`` mapping, or anything indexable by the ``snap_*`` names.
            index: Which snapshot to rebuild.
    Returns:
            The carry, ready to pass to : func:`replay_step`.
    """
    import jax.numpy as jnp

    values = {
        name: jnp.asarray(arrays[f"snap_{name}"][index])
        for name in SingleControllerState._fields
    }
    values["key"] = jax.random.wrap_key_data(values["key"])
    return SingleControllerState(**values)
