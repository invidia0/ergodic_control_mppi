import unittest

import numpy as np

from analysis.pareto import is_pareto_efficient, compute_pareto_front
from analysis.select_config import select_knee_point


class ParetoTest(unittest.TestCase):
    def test_is_pareto_efficient(self):
        points = np.array(
            [
                [1.0, 1.0],  # efficient
                [2.0, 2.0],  # dominated
                [0.5, 3.0],  # efficient
                [3.0, 0.5],  # efficient
            ]
        )
        mask = is_pareto_efficient(points)
        self.assertEqual(mask.tolist(), [True, False, True, True])

    def test_compute_pareto_front(self):
        rows = [
            {"a": 1.0, "b": 1.0},
            {"a": 2.0, "b": 2.0},
            {"a": 0.5, "b": 3.0},
        ]
        front = compute_pareto_front(rows, ["a", "b"])
        self.assertEqual(len(front), 2)

    def test_knee_scale_robustness(self):
        rows = [
            {"m1": 0.10, "m2": 100.0},
            {"m1": 0.20, "m2": 20.0},
            {"m1": 0.30, "m2": 10.0},
        ]
        pick = select_knee_point(rows, ["m1", "m2"])
        self.assertIn("m1", pick)
        self.assertIn("m2", pick)


if __name__ == "__main__":
    unittest.main()

