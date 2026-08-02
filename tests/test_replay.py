"""The replayed sample cloud must be the one the recorded step actually used."""

import shutil
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.replay import (
    replay_step,
    restore_snapshot,
    snapshot_arrays,
)
from ergodic_control_mppi.mppi.single import initialize_single, single_step
from tests.helpers import write_small_config


class ReplayTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        self.params = load_config(write_small_config(Path(directory))).controller
        self.carry = initialize_single(
            self.params,
            jnp.zeros(6, dtype=jnp.float32),
            jnp.zeros((self.params.mppi.horizon, 3), dtype=jnp.float32),
            jax.random.key(0),
        )

    def test_replay_matches_the_recorded_step(self):
        """The replayed plan and weights equal the ones the live step produced."""
        _, result = single_step(self.params, self.carry)
        bundle = replay_step(self.params, self.carry)
        np.testing.assert_array_equal(bundle.weights, np.asarray(result.weights))
        np.testing.assert_array_equal(bundle.optimal, np.asarray(result.optimal_trajectory))

    def test_rollout_cloud_has_the_sampled_shape(self):
        """Positions are (K, T, 2) and the weights they pair with sum to one."""
        bundle = replay_step(self.params, self.carry)
        self.assertEqual(
            bundle.positions.shape,
            (self.params.mppi.samples, self.params.mppi.horizon, 2),
        )
        self.assertAlmostEqual(float(bundle.weights.sum()), 1.0, places=5)

    def test_replay_does_not_advance_the_carry(self):
        """Replaying is read-only: the same carry replays identically twice."""
        first = replay_step(self.params, self.carry)
        second = replay_step(self.params, self.carry)
        np.testing.assert_array_equal(first.positions, second.positions)

    def test_snapshots_round_trip(self):
        """A carry survives being stacked into arrays and rebuilt."""
        carry = self.carry
        snapshots = []
        for _ in range(3):
            snapshots.append(carry)
            carry, _ = single_step(self.params, carry)
        arrays = snapshot_arrays(snapshots)
        rebuilt = restore_snapshot(arrays, 2)
        np.testing.assert_array_equal(
            np.asarray(rebuilt.memory), np.asarray(snapshots[2].memory)
        )
        np.testing.assert_array_equal(
            replay_step(self.params, rebuilt).positions,
            replay_step(self.params, snapshots[2]).positions,
        )


if __name__ == "__main__":
    unittest.main()
