"""Regressions for the guard decision: every rejection path must actually reject."""

import unittest

import numpy as np

from ergodic_control_mppi_ros.safety_shield import Candidate, reject_reason

# A 10x10 m grid at 1 m cells, clear except for one blocked cell at (5.5, 5.5).
GRID = np.zeros((10, 10), dtype=bool)
GRID[5, 5] = True
ORIGIN = (0.0, 0.0)
RESOLUTION = 1.0
NOW = 100.0
TIMEOUT = 0.10
MAX_SPEED = 2.0
CLEAR_PATH = np.array([[1.5, 1.5], [1.5, 2.5], [1.5, 3.5]])


def _candidate(**overrides) -> Candidate:
    base = dict(
        command_stamp=NOW - 0.01,
        path_stamp=NOW - 0.01,
        odometry_stamp=NOW - 0.01,
        position=np.array([1.5, 1.5, 0.75]),
        velocity=np.array([1.0, 0.0, 0.0]),
        path=CLEAR_PATH,
    )
    base.update(overrides)
    return Candidate(**base)


def _reason(candidate) -> str | None:
    return reject_reason(candidate, NOW, TIMEOUT, MAX_SPEED, GRID, ORIGIN, RESOLUTION)


class AcceptTest(unittest.TestCase):
    def test_a_fresh_matched_safe_command_is_accepted(self):
        self.assertIsNone(_reason(_candidate()))


class RejectTest(unittest.TestCase):
    def test_no_command(self):
        self.assertEqual(_reason(None), "no command")

    def test_no_grid(self):
        reason = reject_reason(
            _candidate(), NOW, TIMEOUT, MAX_SPEED, None, ORIGIN, RESOLUTION
        )
        self.assertEqual(reason, "no safety grid")

    def test_stale_command(self):
        self.assertEqual(_reason(_candidate(command_stamp=NOW - 0.5)), "stale command")

    def test_stale_odometry(self):
        self.assertEqual(_reason(_candidate(odometry_stamp=NOW - 0.5)), "stale odometry")

    def test_mismatched_stamps(self):
        candidate = _candidate(path_stamp=NOW - 0.02)
        self.assertEqual(_reason(candidate), "command and path stamps differ")

    def test_non_finite_position(self):
        candidate = _candidate(position=np.array([np.nan, 1.5, 0.75]))
        self.assertEqual(_reason(candidate), "non-finite command")

    def test_non_finite_velocity(self):
        candidate = _candidate(velocity=np.array([np.inf, 0.0, 0.0]))
        self.assertEqual(_reason(candidate), "non-finite command")

    def test_non_finite_path(self):
        path = CLEAR_PATH.copy()
        path[1, 0] = np.nan
        self.assertEqual(_reason(_candidate(path=path)), "non-finite command")

    def test_empty_path(self):
        self.assertEqual(_reason(_candidate(path=np.zeros((0, 2)))), "non-finite command")

    def test_over_speed(self):
        candidate = _candidate(velocity=np.array([2.5, 0.0, 0.0]))
        self.assertEqual(_reason(candidate), "commanded speed over limit")

    def test_path_through_a_blocked_cell(self):
        path = np.array([[4.5, 5.5], [5.5, 5.5], [6.5, 5.5]])
        self.assertEqual(_reason(_candidate(path=path)), "safety path enters a blocked cell")

    def test_path_leaving_the_grid(self):
        path = np.array([[1.5, 1.5], [-5.0, 1.5]])
        self.assertEqual(_reason(_candidate(path=path)), "safety path enters a blocked cell")


class OrderingTest(unittest.TestCase):
    def test_staleness_is_reported_before_geometry(self):
        """A stale command must not be judged on its (possibly garbage) path."""
        candidate = _candidate(
            command_stamp=NOW - 0.5,
            path_stamp=NOW - 0.5,
            path=np.array([[5.5, 5.5], [5.5, 5.5]]),
        )
        self.assertEqual(_reason(candidate), "stale command")

    def test_speed_limit_is_exclusive_at_the_boundary(self):
        self.assertIsNone(_reason(_candidate(velocity=np.array([MAX_SPEED, 0.0, 0.0]))))


if __name__ == "__main__":
    unittest.main()
