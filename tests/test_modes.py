"""Regressions for the target-mode visit, dwell, and cycling metrics."""

import unittest

import numpy as np

from ergodic_control_mppi.metrics.modes import compute_mode_metrics

DELTA_T = 0.02
# Three unit-covariance modes far enough apart that the 2.5-sigma release of one never
# overlaps the 2-sigma entry of another.
MEANS = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
INVERSES = np.repeat(np.eye(2)[None, :, :], 3, axis=0)


def _hold(center, seconds):
    """Return samples parked on a mode center for the given duration."""
    return np.repeat(np.asarray(center, dtype=np.float64)[None, :], int(seconds / DELTA_T), axis=0)


def _metrics(positions, **kwargs):
    return compute_mode_metrics(positions, MEANS, INVERSES, DELTA_T, **kwargs)


class MinimumDwellTest(unittest.TestCase):
    def test_brief_touch_is_not_a_visit(self):
        # 0.5 s inside mode 0 is under the one-second qualification.
        positions = np.concatenate([_hold([0.0, 0.0], 0.5), _hold([100.0, 0.0], 2.0)])
        result = _metrics(positions)
        self.assertEqual(result["mode_visits"], 0.0)
        self.assertEqual(result["in_mode_fraction"], 0.0)

    def test_long_enough_touch_is_a_visit(self):
        positions = np.concatenate([_hold([0.0, 0.0], 1.5), _hold([100.0, 0.0], 1.5)])
        result = _metrics(positions)
        self.assertEqual(result["mode_visits"], 1.0)
        self.assertAlmostEqual(result["mode_dwell_median_s"], 1.5, places=6)
        self.assertAlmostEqual(result["in_mode_fraction"], 0.5, places=6)


class HysteresisTest(unittest.TestCase):
    def test_boundary_chatter_stays_one_visit(self):
        """Oscillating across the 2-sigma entry must not split a single dwell."""
        inside = _hold([0.0, 0.0], 1.0)
        # Alternate either side of the entry threshold but always within the 2.5 release.
        chatter = np.array(
            [[2.2 if index % 2 else 1.8, 0.0] for index in range(100)], dtype=np.float64
        )
        positions = np.concatenate([inside, chatter, _hold([100.0, 0.0], 1.0)])
        result = _metrics(positions)
        self.assertEqual(result["mode_visits"], 1.0)
        self.assertEqual(result["mode_switches"], 0.0)

    def test_true_exit_ends_the_visit(self):
        """Leaving mode 0 and returning is a switch and a revisit, but not a transition."""
        positions = np.concatenate(
            [_hold(MEANS[0], 1.5), _hold([100.0, 0.0], 1.5), _hold(MEANS[0], 1.5)]
        )
        result = _metrics(positions)
        self.assertEqual(result["mode_visits"], 2.0)
        self.assertEqual(result["mode_switches"], 1.0)
        self.assertEqual(result["mode_revisits"], 1.0)
        self.assertEqual(result["mode_transitions"], 0.0)

    def test_moving_to_another_mode_is_a_transition(self):
        positions = np.concatenate([_hold(MEANS[0], 1.5), _hold(MEANS[1], 1.5)])
        result = _metrics(positions)
        self.assertEqual(result["mode_switches"], 1.0)
        self.assertEqual(result["mode_transitions"], 1.0)
        self.assertEqual(result["mode_revisits"], 0.0)


class CyclingTest(unittest.TestCase):
    def test_single_sweep_completes_no_cycle(self):
        positions = np.concatenate(
            [_hold(MEANS[0], 1.5), _hold(MEANS[1], 1.5), _hold(MEANS[2], 1.5)]
        )
        result = _metrics(positions)
        self.assertEqual(result["mode_visits"], 3.0)
        self.assertEqual(result["mode_cycles"], 0.0)
        # The first sweep closes at the end of the third dwell.
        self.assertAlmostEqual(result["first_all_modes_s"], 4.5, places=6)

    def test_second_sweep_counts_one_cycle(self):
        sweep = [_hold(MEANS[0], 1.5), _hold(MEANS[1], 1.5), _hold(MEANS[2], 1.5)]
        result = _metrics(np.concatenate(sweep + sweep))
        self.assertEqual(result["mode_cycles"], 1.0)
        self.assertAlmostEqual(result["first_all_modes_s"], 4.5, places=6)

    def test_never_reaching_all_modes_reports_nan(self):
        positions = np.concatenate([_hold(MEANS[0], 2.0), _hold(MEANS[1], 2.0)])
        result = _metrics(positions)
        self.assertTrue(np.isnan(result["first_all_modes_s"]))
        self.assertEqual(result["mode_cycles"], 0.0)


class DegenerateInputTest(unittest.TestCase):
    def test_empty_path(self):
        result = _metrics(np.zeros((0, 2)))
        self.assertEqual(result["mode_visits"], 0.0)
        self.assertEqual(result["in_mode_fraction"], 0.0)
        self.assertTrue(np.isnan(result["first_all_modes_s"]))


if __name__ == "__main__":
    unittest.main()
