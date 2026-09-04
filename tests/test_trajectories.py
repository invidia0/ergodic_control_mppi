"""The 2-sigma ellipse must be the boundary metrics/modes.py scores against.

If these drift apart the figures stop illustrating the reported statistic, which is the
one thing this module exists to guarantee.
"""
import numpy as np

from ergodic_control_mppi.plotting.trajectories import _ellipse_points, panel_grid


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
