"""Regressions for trajectory figures and the 2-sigma mode boundary."""
import numpy as np
import tempfile
import unittest
from unittest.mock import patch
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")

from ergodic_control_mppi.plotting.trajectories import (
    _ellipse_points,
    _mode_labels,
    figure_service_gate,
    figure_service_speed,
    panel_grid,
    speed_snapshots,
)


def test_ellipse_is_two_sigma_mahalanobis():
    mean = np.array([1.0, -2.0])
    covariance = np.array([[8.0, 2.0], [2.0, 3.0]])
    ring = _ellipse_points(mean, covariance, 2.0)
    inverse = np.linalg.inv(covariance)
    delta = ring - mean
    distance = np.sqrt(np.einsum("ni,ij,nj->n", delta, inverse, delta))
    assert np.allclose(distance, 2.0, atol=1e-9)


def test_ellipse_handles_singular_covariance():
    ring = _ellipse_points(np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]]), 2.0)
    assert np.isfinite(ring).all()


def test_panel_grid_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    capture = {
        "positions": np.cumsum(rng.normal(size=(200, 2)), axis=0),
        "means": np.array([[0.0, 6.0], [-12.0, -4.0]]),
        "covariances": np.array([[[12.0, 0.0], [0.0, 2.0]],
                                 [[8.0, 2.0], [2.0, 3.0]]]),
        "title": "test",
    }
    out = panel_grid([capture, capture], tmp_path / "panels.pdf")
    assert out.exists() and out.stat().st_size > 0


class ServiceGateFigureTest(unittest.TestCase):
    def test_mode_labels_match_the_drawn_rings(self):
        means = np.array([[0.0, 0.0], [10.0, 0.0]])
        covariances = np.array([np.eye(2), np.eye(2)])
        positions = np.array([[0.0, 0.0], [1.9, 0.0], [2.1, 0.0], [10.0, 0.0]])
        labels = _mode_labels(positions, means, covariances, sigma=2.0)
        np.testing.assert_array_equal(labels, [0, 0, -1, 1])

    def test_series_uses_deployed_release_ratio(self):
        captures = [
            {
                "positions": np.zeros((2, 2)),
                "means": np.zeros((1, 2)),
                "covariances": np.eye(2)[None],
                "limits": np.array([-1.0, 1.0, -1.0, 1.0]),
                "release_ratio": np.asarray(value),
            }
            for value in (0.0, 1.5, 2.24, 3.0)
        ]
        series = (np.arange(2), np.ones((2, 1)), np.zeros((2, 1)))
        with tempfile.TemporaryDirectory() as directory, patch(
            "ergodic_control_mppi.plotting.trajectories.service_series",
            return_value=series,
        ) as mocked:
            figure_service_gate(captures, f"{directory}/service.png")
        self.assertEqual(float(mocked.call_args.args[0]["release_ratio"]), 2.24)


def _speed_capture():
    """Tiny capture with a known hold at step 0 and release at step 10."""
    weights = np.full(3, 1.0 / 3.0)
    low = np.array([0.2, 0.4, 0.4])
    high = np.array([0.5, 0.25, 0.25])
    mass = np.vstack([np.tile(low, (10, 1)), np.tile(high, (10, 1))])
    return {
        "positions": np.zeros((20, 2)),
        "service_mass_history": mass,
        "stride": np.asarray(1),
        "delta_t": np.asarray(0.02),
        "service_decay": np.asarray(0.0),
        "means": np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]),
        "covariances": np.stack([np.eye(2), np.eye(2), np.eye(2)]),
        "log_weights": np.log(weights),
        "limits": np.array([-2.0, 22.0, -2.0, 2.0]),
        "reference_speed": np.asarray(1.8),
        "transit_speedup": np.asarray(4.0),
        "dwell_slowdown": np.asarray(1.0),
        "service_floor": np.asarray(0.3),
        "fine_bandwidth": np.asarray(0.94),
        "deficit_ceiling": np.asarray(0.05),
        "release_ratio": np.asarray(2.24),
    }


class ServiceSpeedFigureTest(unittest.TestCase):
    def test_snapshots_are_the_first_hold_then_release(self):
        hold, release = speed_snapshots(_speed_capture())
        self.assertEqual((hold, release), (0, 10))

    def test_snapshots_are_deterministic(self):
        capture = _speed_capture()
        self.assertEqual(speed_snapshots(capture), speed_snapshots(capture))

    def test_figure_writes_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = figure_service_speed(_speed_capture(), f"{directory}/speed.png")
            self.assertTrue(out.exists() and out.stat().st_size > 0)
