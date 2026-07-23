import unittest

import numpy as np

from ergodic_control_mppi.metrics.coordination import (
    compute_pairwise_min_distance,
    compute_pairwise_overlap,
    compute_redundancy_metric,
    compute_safety_metric,
)
from ergodic_control_mppi.metrics.ergodicity import compute_team_ergodic_error
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


if __name__ == "__main__":
    unittest.main()
