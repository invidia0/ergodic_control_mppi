import unittest

import numpy as np

from metrics.ergodicity import compute_team_ergodic_error
from metrics.overlap import compute_pairwise_overlap
from metrics.safety import compute_safety_metric
from metrics.redundancy import compute_redundancy_metric


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.map_x = (-10.0, 10.0)
        self.map_y = (-10.0, 10.0)

    def test_identical_trajectories_high_overlap(self):
        steps = 200
        t = np.linspace(-5.0, 5.0, steps)
        r0 = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        r1 = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        paths = np.stack([r0, r1], axis=1)  # (steps, robots, state_dim)
        overlap = compute_pairwise_overlap(paths, self.map_x, self.map_y)
        self.assertGreater(overlap, 0.9)

    def test_separated_trajectories_low_overlap(self):
        steps = 200
        t = np.linspace(-5.0, 5.0, steps)
        r0 = np.stack([t, np.full_like(t, -6.0), np.zeros_like(t)], axis=1)
        r1 = np.stack([t, np.full_like(t, 6.0), np.zeros_like(t)], axis=1)
        paths = np.stack([r0, r1], axis=1)
        overlap = compute_pairwise_overlap(paths, self.map_x, self.map_y)
        self.assertLess(overlap, 0.2)

    def test_near_collision_poor_safety(self):
        steps = 100
        t = np.linspace(-1.0, 1.0, steps)
        r0 = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
        r1 = np.stack([t, np.full_like(t, 0.05), np.zeros_like(t)], axis=1)
        paths = np.stack([r0, r1], axis=1)
        obstacles = np.zeros((0, 3))
        unsafe = compute_safety_metric(paths, obstacles, safety_radius=0.3)
        safe = compute_safety_metric(paths, obstacles, safety_radius=0.01)
        self.assertGreater(unsafe, safe)

    def test_ergodic_error_improves_with_mode_coverage(self):
        target = np.zeros((40, 40), dtype=np.float64)
        target[10, 10] = 0.5
        target[30, 30] = 0.5

        steps = 200
        # Covers both modes.
        r0_x = np.concatenate([np.full(100, -5.0), np.full(100, 5.0)])
        r0_y = np.concatenate([np.full(100, -5.0), np.full(100, 5.0)])
        paths_cover = np.stack(
            [
                np.stack([r0_x, r0_y, np.zeros_like(r0_x)], axis=1),
            ],
            axis=1,
        )
        # Collapses on one mode.
        paths_collapse = np.stack(
            [
                np.stack([np.full(steps, -5.0), np.full(steps, -5.0), np.zeros(steps)], axis=1),
            ],
            axis=1,
        )
        err_cover = compute_team_ergodic_error(paths_cover, target, self.map_x, self.map_y)
        err_collapse = compute_team_ergodic_error(paths_collapse, target, self.map_x, self.map_y)
        self.assertLess(err_cover, err_collapse)

    def test_redundancy_higher_when_paths_identical(self):
        steps = 150
        t = np.linspace(-4.0, 4.0, steps)
        shared = np.stack([t, t, np.zeros_like(t)], axis=1)
        disjoint_a = np.stack([t, np.full_like(t, -5.0), np.zeros_like(t)], axis=1)
        disjoint_b = np.stack([t, np.full_like(t, 5.0), np.zeros_like(t)], axis=1)

        identical = np.stack([shared, shared], axis=1)
        disjoint = np.stack([disjoint_a, disjoint_b], axis=1)
        r_identical = compute_redundancy_metric(identical, self.map_x, self.map_y)
        r_disjoint = compute_redundancy_metric(disjoint, self.map_x, self.map_y)
        self.assertGreater(r_identical, r_disjoint)


if __name__ == "__main__":
    unittest.main()

