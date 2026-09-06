"""The executable half of the certificate: it must hold on every prefix, against the
same discrete target the coverage metrics use."""

import unittest

import numpy as np

from ergodic_control_mppi.metrics.discrepancy import (
    grid_target,
    mean_embedding,
    radius_squared,
    self_inner,
    trivial_bound,
    walk,
)


LIMITS = (0.0, 10.0)
BANDWIDTH = 1.0
MEANS = np.array([[3.0, 3.0], [7.0, 6.5]])
COVARIANCES = np.array([[[0.8, 0.2], [0.2, 0.5]], [[0.6, -0.1], [-0.1, 0.9]]])
WEIGHTS = np.array([0.6, 0.4])
#: A pillar in the middle of the workspace, so the mask removes target mass rather than
#: only empty corners -- which is the case the restriction exists to handle.
OBSTACLE, RADIUS = np.array([5.0, 5.0]), 1.5


def mixture_density(query: np.ndarray) -> np.ndarray:
    """Evaluate the test mixture at ``(Q, 2)`` positions."""
    delta = query[:, None, :] - MEANS
    inverse = np.linalg.inv(COVARIANCES)
    quadratic = np.einsum("qji,jik,qjk->qj", delta, inverse, delta)
    normalizer = 2.0 * np.pi * np.sqrt(np.linalg.det(COVARIANCES))
    return (np.exp(-0.5 * quadratic) / normalizer) @ WEIGHTS


def nodes(bins: int) -> np.ndarray:
    """The ``linspace`` node grid `grid_target` places the target on, as ``(B*B, 2)``."""
    axis = np.linspace(*LIMITS, bins)
    x, y = np.meshgrid(axis, axis)
    return np.column_stack([x.ravel(), y.ravel()])


def target(bins: int, obstacle: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Build the discrete target on a ``bins x bins`` grid, optionally with the pillar cut."""
    grid = nodes(bins)
    density = mixture_density(grid).reshape(bins, bins)
    mask = np.ones((bins, bins), dtype=bool)
    if obstacle:
        mask = (np.sum((grid - OBSTACLE) ** 2, axis=1) > RADIUS ** 2).reshape(bins, bins)
    return grid_target(density, mask, LIMITS, LIMITS)


def continuous_embedding(query: np.ndarray) -> np.ndarray:
    """``m_pi`` for the *untruncated* mixture, in closed form.

    The reference the discrete embedding must converge to on an unmasked grid, and only
    that: the certified object is the discrete target, not this one.
    """
    inflated = COVARIANCES + 0.5 * BANDWIDTH * np.eye(2)
    delta = query[:, None, :] - MEANS
    inverse = np.linalg.inv(inflated)
    quadratic = np.einsum("qji,jik,qjk->qj", delta, inverse, delta)
    normalizer = 2.0 * np.pi * np.sqrt(np.linalg.det(inflated))
    return np.pi * BANDWIDTH * (np.exp(-0.5 * quadratic) / normalizer) @ WEIGHTS


class TargetTest(unittest.TestCase):
    """The finite sums must be the finite sums, and the restriction must restrict."""

    def test_embedding_matches_direct_enumeration(self):
        """On a grid small enough to sum by hand, both must agree exactly."""
        support, weights = target(5)
        query = np.array([[3.0, 3.0], [5.0, 5.0], [9.0, 1.0]])
        direct = np.array([
            sum(w * np.exp(-np.dot(z - c, z - c) / BANDWIDTH)
                for c, w in zip(support, weights))
            for z in query
        ])
        np.testing.assert_allclose(
            mean_embedding(query, support, weights, BANDWIDTH), direct, rtol=0.0, atol=1e-15
        )

    def test_self_inner_matches_direct_enumeration(self):
        support, weights = target(5)
        direct = sum(
            wa * wb * np.exp(-np.dot(a - b, a - b) / BANDWIDTH)
            for a, wa in zip(support, weights) for b, wb in zip(support, weights)
        )
        self.assertAlmostEqual(self_inner(support, weights, BANDWIDTH), direct, places=14)

    def test_mask_drops_the_obstacle_mass_and_renormalizes(self):
        support, weights = target(60, obstacle=True)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(np.linalg.norm(support - OBSTACLE, axis=1) > RADIUS))
        # The pillar sits between the two modes, so this is real mass, not tail.
        self.assertLess(len(weights), 60 * 60)
        self.assertGreater(
            1.0 - len(weights) / (60 * 60), 0.0
        )

    def test_unmasked_grid_matches_the_continuous_embedding(self):
        """A discretization diagnostic, not the certificate's correctness test.

        The certified target is the discrete measure at whatever resolution the metrics
        use; this only shows the discretization is the one it claims to be. It does not
        converge to zero: the discrete target lives on the workspace box and the mixture
        does not, so the residual settles on the truncated tail (0.03% here) rather than
        on the grid spacing. Certifying the continuous mixture would mean bounding that
        floor, which is exactly the claim this module declines to make.
        """
        query = np.array([[3.0, 3.0], [5.0, 5.0], [8.0, 2.0]])
        reference = continuous_embedding(query)
        support, weights = target(160)
        error = np.max(np.abs(mean_embedding(query, support, weights, BANDWIDTH) - reference))
        self.assertLess(error, 1e-3 * float(np.max(reference)))
        # A grid too coarse to resolve the mixture is a different object entirely.
        coarse, coarse_weights = target(10)
        self.assertGreater(
            np.max(np.abs(mean_embedding(query, coarse, coarse_weights, BANDWIDTH)
                          - reference)),
            10.0 * error,
        )

    def test_comparator_bounds_the_error_and_beats_the_loose_one(self):
        """``1 + ||m_pi||^2`` must dominate every observed ``E_n`` and be tighter."""
        support, weights = target(60, obstacle=True)
        comparator = trivial_bound(support, weights, BANDWIDTH)
        loose = (1.0 + np.sqrt(self_inner(support, weights, BANDWIDTH))) ** 2
        self.assertLess(comparator, loose)
        self.assertAlmostEqual(comparator, radius_squared(support, weights, BANDWIDTH))
        rng = np.random.default_rng(7)
        for path in (rng.uniform(*LIMITS, size=(60, 2)),
                     np.full((60, 2), 9.9),
                     np.tile(MEANS[0], (60, 1))):
            result = walk(path, support, weights, BANDWIDTH)
            self.assertLessEqual(float(result["error"].max()), comparator)


class WalkTest(unittest.TestCase):
    """``walk`` must reproduce the recurrence, not merely produce plausible numbers."""

    def setUp(self):
        self.support, self.weights = target(60, obstacle=True)

    def audit(self, path, **kwargs):
        return walk(path, self.support, self.weights, BANDWIDTH, **kwargs)

    def test_bound_holds_on_every_prefix_with_obstacles(self):
        rng = np.random.default_rng(1)
        path = np.clip(np.cumsum(rng.normal(0.0, 0.5, size=(150, 2)), axis=0) + 5.0, *LIMITS)
        result = self.audit(path)
        self.assertTrue(np.all(result["error"] >= -1e-12))
        self.assertTrue(np.all(result["error"] <= result["bound"] + 1e-9))

    def test_bound_holds_for_an_independent_sample(self):
        rng = np.random.default_rng(2)
        component = rng.choice(2, size=200, p=WEIGHTS)
        path = np.stack([
            rng.multivariate_normal(MEANS[j], COVARIANCES[j]) for j in component
        ])
        result = self.audit(np.clip(path, *LIMITS))
        self.assertTrue(np.all(result["error"] <= result["bound"] + 1e-9))
        # An iid sample is not the greedy oracle, but it still averages the target down.
        self.assertLess(result["error"][-1], result["error"][4])

    def test_the_reach_proxy_lowers_the_gap(self):
        """Restricting where the vehicle may go raises min w, so it *lowers* delta.

        Only the support gap is valid in the recurrence; the ordering here is what makes
        the reach gap a diagnostic split rather than a cheaper bound.
        """
        rng = np.random.default_rng(3)
        path = np.clip(np.cumsum(rng.normal(0.0, 0.4, size=(80, 2)), axis=0) + 5.0, *LIMITS)
        result = self.audit(path, step_radius=1.0)
        finite = np.isfinite(result["gap_reach"])
        self.assertTrue(finite.any())
        self.assertTrue(np.all(result["gap"][finite] >= result["gap_reach"][finite] - 1e-12))
        self.assertGreater(np.nanmax(result["gap"] - result["gap_reach"]), 0.0)

    def test_greedy_oracle_drives_the_error_toward_zero(self):
        """delta_n ~ 0 must give the O(1/N) decay the ideal mechanism claims.

        The oracle steps on the target's own support, so with the restricted target there
        is no obstacle mass left to floor the error -- which is the point of the fix.
        """
        embedded = mean_embedding(self.support, self.support, self.weights, BANDWIDTH)
        kernel_sum = np.zeros(self.support.shape[0])
        path = [self.support[0]]
        for count in range(1, 220):
            kernel_sum += np.exp(
                -np.sum((self.support - path[-1]) ** 2, axis=1) / BANDWIDTH
            )
            path.append(self.support[np.argmin(kernel_sum / count - embedded)])
        result = self.audit(np.stack(path))
        errors, counts = result["error"], result["n"]
        self.assertLess(errors[-1], 0.05 * errors[9])
        # O(1/N) means N * E_N stays bounded; a stalled mechanism would grow it.
        self.assertLess((counts * errors)[-1], 3.0 * (counts * errors)[49])
        self.assertTrue(np.all(errors <= result["bound"] + 1e-9))
        self.assertTrue(np.all(result["gap"][:-1] >= -1e-9))

    def test_short_path_is_refused(self):
        with self.assertRaises(ValueError):
            self.audit(np.array([[1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
