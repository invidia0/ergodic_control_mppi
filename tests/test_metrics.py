import unittest

import numpy as np

from ergodic_control_mppi.metrics.coordination import (
    compute_pairwise_min_distance,
    compute_pairwise_overlap,
    compute_redundancy_metric,
    compute_safety_metric,
)
from ergodic_control_mppi.metrics.ergodicity import (
    compute_reachable_mask,
    compute_team_ergodic_error,
)
from ergodic_control_mppi.metrics.evaluate import TrialData, compute_all_metrics


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.x_limits = (-5.0, 5.0)
        self.y_limits = (-5.0, 5.0)
        self.target = np.ones((10, 10))

    def test_single_robot_and_empty_obstacles(self):
        path = np.zeros((5, 1, 6))
        self.assertEqual(compute_pairwise_overlap(path, self.x_limits, self.y_limits), 0.0)
        self.assertTrue(np.isinf(compute_pairwise_min_distance(path)))
        self.assertEqual(compute_safety_metric(path, np.zeros((0, 3)), 0.5), 0.0)
        metrics = compute_all_metrics(TrialData(
            path, self.target, self.x_limits, self.y_limits, np.zeros((0, 3)), 0.5
        ))
        self.assertEqual(set(metrics), {
            "team_ergodic_error", "pairwise_overlap", "safety_metric",
            "redundancy_metric", "R_pair", "D_min_pair",
        })

    def test_identical_multi_paths_are_redundant(self):
        line = np.zeros((10, 6))
        line[:, 0] = np.linspace(-2, 2, 10)
        identical = np.stack((line, line), axis=1)
        separated = identical.copy()
        separated[:, 1, 1] = 4
        self.assertGreater(
            compute_redundancy_metric(identical, self.x_limits, self.y_limits),
            compute_redundancy_metric(separated, self.x_limits, self.y_limits),
        )

    def test_ergodic_error_is_finite(self):
        self.assertTrue(np.isfinite(compute_team_ergodic_error(
            np.zeros((4, 1, 6)), self.target, self.x_limits, self.y_limits
        )))

    def test_reachable_mask_blocks_obstacle_cells(self):
        obstacle = np.array([[0.0, 0.0, 1.0]])
        mask = compute_reachable_mask(obstacle, 0.5, self.x_limits, self.y_limits, (10, 10))
        self.assertEqual(mask.shape, (10, 10))
        # center cell is inside the obstacle+safe radius, corner is free
        self.assertFalse(bool(mask[5, 5]))
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(compute_reachable_mask(
            np.zeros((0, 3)), 0.5, self.x_limits, self.y_limits, (10, 10)
        ).all()))

    def test_mask_lowers_error_for_target_mass_under_obstacle(self):
        # Target mass split between a blocked cell and a reachable corner; the
        # robot fully covers the reachable corner.
        target = np.zeros((10, 10))
        target[5, 5] = 1.0  # under the obstacle -> unreachable
        target[0, 0] = 1.0  # reachable corner, where the robot dwells
        path = np.zeros((3, 1, 6))
        path[:, 0, 0] = -4.5  # bottom-left corner cell (ix=0, iy=0)
        path[:, 0, 1] = -4.5
        obstacle = np.array([[0.0, 0.0, 1.0]])
        mask = compute_reachable_mask(obstacle, 0.5, self.x_limits, self.y_limits, (10, 10))
        unmasked = compute_team_ergodic_error(path, target, self.x_limits, self.y_limits)
        masked = compute_team_ergodic_error(
            path, target, self.x_limits, self.y_limits, reachable_mask=mask
        )
        # Unmasked penalizes the unreachable peak; masked renormalizes to the
        # reachable target, which the robot covers perfectly -> ~0 error.
        self.assertLess(masked, unmasked)
        self.assertAlmostEqual(masked, 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
