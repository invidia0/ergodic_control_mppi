import tempfile
import unittest
from pathlib import Path

import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.multi import build_cross_particles, initialize_surrogates
from ergodic_control_mppi.simulation import run_simulation
from tests.helpers import write_small_config


class ControllerTest(unittest.TestCase):
    def test_tiny_single_and_multi_cpu_loops(self):
        with tempfile.TemporaryDirectory() as temporary:
            for robots in (1, 2):
                config = load_config(write_small_config(Path(temporary), robots=robots, steps=2))
                result = run_simulation(config, "cpu")
                self.assertEqual(result.paths.shape, (2, robots, 6))
                self.assertTrue(np.all(np.isfinite(result.paths)))

    def test_surrogates_start_at_actual_positions(self):
        states = np.array([[1, 2, 0, 0, 0, 0], [-3, 4, 0, 0, 0, 0]], dtype=np.float32)
        surrogates = np.asarray(initialize_surrogates(states, 3))
        np.testing.assert_array_equal(surrogates[0], np.tile([1, 2], (3, 1)))
        np.testing.assert_array_equal(surrogates[1], np.tile([-3, 4], (3, 1)))

    def test_cross_particles_exclude_self_synchronously(self):
        surrogates = np.stack([np.full((2, 2), robot) for robot in (0, 1, 2)])
        histories = np.stack([np.full((1, 2), robot) for robot in (0, 1, 2)])
        cross = np.asarray(build_cross_particles(surrogates, histories))
        self.assertEqual(cross.shape, (3, 6, 2))
        for robot in range(3):
            self.assertNotIn(robot, np.unique(cross[robot]))


if __name__ == "__main__":
    unittest.main()
