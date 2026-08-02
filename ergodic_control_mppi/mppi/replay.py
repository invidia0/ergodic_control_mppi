"""Recover the MPPI sample cloud behind a recorded control step, for figures.

The rollouts are the largest intermediate in ``mppi_step`` -- ``(K, T, 2)`` floats every
step -- so they are deliberately not returned by it and not carried through the scan. They
do not need to be: ``single_step`` is pure and ``SingleControllerState`` carries the PRNG
key, so re-drawing from the same key reproduces the same cloud exactly. Saving the small
carry and replaying it here costs ~7 kB per snapshot instead of ~700 kB.
"""

from typing import NamedTuple

import jax
import numpy as np

from ergodic_control_mppi.mppi.core import _rollouts, mppi_step, sample_epsilon
from ergodic_control_mppi.mppi.single import SingleControllerState
from ergodic_control_mppi.parameters import ControllerParams


class RolloutBundle(NamedTuple):
    """Everything needed to draw one planning step.

    Attributes:
        positions: Sampled rollout positions with shape ``(K, T, 2)``.
        weights: Normalized MPPI weights with shape ``(K,)``; sums to one.
        costs: Per-rollout costs with shape ``(K,)``.
        optimal: The selected plan's state trajectory with shape ``(T, 6)``.
        surrogate: The shared median path the flow was evaluated on, shape ``(T, 2)``.
        memory: The fading-memory buffer at this step, shape ``(P, 2)``, oldest first.
        state: The state the step planned from, shape ``(6,)``.
    """

    positions: np.ndarray
    weights: np.ndarray
    costs: np.ndarray
    optimal: np.ndarray
    surrogate: np.ndarray
    memory: np.ndarray
    state: np.ndarray


def replay_step(params: ControllerParams, carry: SingleControllerState) -> RolloutBundle:
    """Re-run one planning step and return its sample cloud.

    Both draws start from ``carry.key``, so the epsilon here is the one the recorded step
    used and the returned positions are the exact rollouts its weights were computed from.
    This does not advance anything -- ``carry`` is unchanged.

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
    result = mppi_step(
        params, carry.controls, carry.state, carry.key, carry.temperature, carry.memory
    )
    return RolloutBundle(
        positions=np.asarray(positions),
        weights=np.asarray(result.weights),
        costs=np.asarray(costs),
        optimal=np.asarray(result.optimal_trajectory),
        surrogate=np.asarray(result.surrogate),
        memory=np.asarray(carry.memory),
        state=np.asarray(carry.state),
    )


def snapshot_arrays(snapshots: list[SingleControllerState]) -> dict[str, np.ndarray]:
    """Stack recorded carries into ``npz``-writable arrays.

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
    """Rebuild one carry from the arrays written by :func:`snapshot_arrays`.

    Args:
        arrays: An ``npz`` mapping, or anything indexable by the ``snap_*`` names.
        index: Which snapshot to rebuild.

    Returns:
        The carry, ready to pass to :func:`replay_step`.
    """
    import jax.numpy as jnp

    values = {
        name: jnp.asarray(arrays[f"snap_{name}"][index])
        for name in SingleControllerState._fields
    }
    values["key"] = jax.random.wrap_key_data(values["key"])
    return SingleControllerState(**values)
