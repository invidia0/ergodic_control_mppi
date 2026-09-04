"""The reference field, and the one claim the whole theory now rests on.

``mppi/field.py:potential`` writes down a scalar ``Phi`` and asserts that the controller
tracks ``Gamma_v(grad Phi)``. That is only true if every weight in the three KDE terms is
constant in the query ``z``, and it is only true because there is no rotation:
``R(theta) grad Phi`` is not the gradient of anything unless ``R = I``.

:class:`PotentialTest` finite-differences ``Phi`` against the pre-gauge field and fails
loudly the moment any weight acquires a ``z`` dependence.
"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.mppi.field import (
    deficit_weighted,
    kde_repulsion,
    memory_flow,
    per_mode_weighted,
    potential,
    responsibility_gaps,
    score_pdf,
)
from tests.helpers import write_small_config


def _params(directory: Path):
    return load_config(write_small_config(Path(directory))).controller


def _unit(vector):
    return vector / jnp.maximum(jnp.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


class PotentialTest(unittest.TestCase):
    """Phi is the potential of the pre-gauge field. This is Sec. III-F's whole content."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.params = _params(Path(cls.temp.name))
        rng = np.random.default_rng(0)
        cls.memory = jnp.asarray(rng.uniform(-6.0, 6.0, size=(48, 2)), dtype=jnp.float32)
        cls.plan = jnp.asarray(rng.uniform(-6.0, 6.0, size=(24, 2)), dtype=jnp.float32)
        cls.queries = jnp.asarray(rng.uniform(-5.0, 5.0, size=(9, 2)), dtype=jnp.float32)
        cls.recency = jnp.asarray(
            float(cls.params.field.memory_decay) ** np.arange(48)[::-1], dtype=jnp.float32
        )
        cls.floor = jnp.asarray(1.0 / 400.0, dtype=jnp.float32)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _field(self, field, service_mass=None):
        """The pre-gauge field: the three closed-form gradients ``field_at`` sums."""
        from ergodic_control_mppi.mppi.field import attraction_target

        gauge = jnp.sqrt(0.5 * jnp.e * field.fine_bandwidth)
        flow = score_pdf(
            self.queries, attraction_target(self.params.gmm, field, service_mass)
        )
        flow += field.memory_gain * memory_flow(
            self.queries, self.memory, self.recency, self.params.gmm, field, self.floor
        )
        flow += field.plan_gain * gauge * kde_repulsion(
            self.queries, self.plan,
            jnp.ones((self.plan.shape[0],), dtype=jnp.float32),
            field.fine_bandwidth,
        )
        return flow

    def _grad_phi(self, field, service_mass=None):
        """``grad Phi`` by autodiff, with the memory and plan point sets held fixed."""
        def scalar(point):
            return potential(
                point[None, :], self.memory, self.recency, self.plan,
                self.params.gmm, field, self.floor, service_mass,
            )[0]

        return jax.vmap(jax.grad(scalar))(self.queries)

    def test_grad_phi_is_the_field(self):
        """The claim, at the deployed settings."""
        field = self.params.field
        produced = np.asarray(self._field(field), dtype=np.float64)
        gradient = np.asarray(self._grad_phi(field), dtype=np.float64)
        self.assertGreater(np.abs(produced).max(), 1e-3, "both fields are zero")
        # float32 throughout, and the memory term is a P-point kernel sum, so the agreement
        # is roundoff-limited rather than exact. Relative to the field magnitude.
        scale = np.abs(produced).max()
        self.assertLess(np.abs(produced - gradient).max() / scale, 1e-3,
                        f"grad Phi differs from the field:\n{produced}\n{gradient}")

    def test_grad_phi_is_the_field_under_the_destination_bend(self):
        """The bent mixture must not introduce a z dependence in any weight.

        This is the arm of the claim most at risk: ``attraction_target`` re-weights the
        mixture from the service mass, and if that ever read the query position the field
        would stop being a gradient without anything else changing.
        """
        field = self.params.field
        mass = jnp.asarray([4.0, 0.2, 0.05], dtype=jnp.float32)[
            : self.params.gmm.log_weights.shape[0]
        ]
        produced = np.asarray(self._field(field, mass), dtype=np.float64)
        gradient = np.asarray(self._grad_phi(field, mass), dtype=np.float64)
        scale = np.abs(produced).max()
        self.assertLess(np.abs(produced - gradient).max() / scale, 1e-3)

    def test_each_term_is_separately_a_gradient(self):
        """Isolate the three terms, so a failure names which one broke."""
        base = self.params.field
        cases = {
            "score only": replace(base, memory_gain=0.0, plan_gain=0.0),
            "memory only": replace(base, plan_gain=0.0),
            "plan only": replace(base, memory_gain=0.0),
            "excess only": replace(base, memory_balance=1.0, plan_gain=0.0),
            "trail only": replace(base, memory_balance=0.0, plan_gain=0.0),
        }
        for name, field in cases.items():
            with self.subTest(term=name):
                produced = np.asarray(self._field(field), dtype=np.float64)
                gradient = np.asarray(self._grad_phi(field), dtype=np.float64)
                scale = max(np.abs(produced).max(), 1e-9)
                self.assertGreater(scale, 1e-3, f"{name} is identically zero")
                self.assertLess(np.abs(produced - gradient).max() / scale, 1e-3, name)

    def test_the_field_is_curl_free(self):
        """The coordinate-free half of the same statement.

        A gradient has zero curl. Checking it separately catches the case where ``Phi`` and
        the field are edited together into two consistent but non-conservative expressions,
        which the finite-difference test alone would pass.
        """
        field = self.params.field

        def at(point):
            saved = self.queries
            type(self).queries = point[None, :]
            try:
                return self._field(field)[0]
            finally:
                type(self).queries = saved

        for point in np.asarray(self.queries)[:4]:
            jacobian = jax.jacobian(at)(jnp.asarray(point, dtype=jnp.float32))
            curl = float(jacobian[1, 0] - jacobian[0, 1])
            self.assertLess(abs(curl), 1e-3, f"field has curl {curl:.3g} at {point}")


class ServiceGateTest(unittest.TestCase):
    """Sec. III-E: promotion is capped, demotion is not, and that asymmetry is the point."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.params = _params(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_promotion_is_capped_below_the_margins(self):
        """``log((c+1)/c)`` against ``Delta_j``: the pre-registered null on ``c``."""
        gaps = np.asarray(responsibility_gaps(self.params.gmm), dtype=np.float64)
        for ceiling in (0.05, 0.5):
            promotion = np.log((ceiling + 1.0) / ceiling)
            self.assertLess(promotion, gaps.min(),
                            f"c={ceiling} promotes by {promotion:.2f} nats, which could "
                            f"overturn the {gaps.min():.2f}-nat margin")

    def test_per_mode_release_equalizes_the_threshold(self):
        """``kappa_j = Delta_j / (sigma* - 1)`` releases every mode at the same over-service.

        A scalar penalty does not: ``sigma*_j = 1 + Delta_j / kappa`` spreads by the ratio of
        the gaps, so the shallowest mode leaves earliest and is under-served by construction.
        """
        gaps = np.asarray(responsibility_gaps(self.params.gmm), dtype=np.float64)
        for target in (1.75, 2.24, 3.0):
            kappa = gaps / (target - 1.0)
            np.testing.assert_allclose(1.0 + gaps / kappa, target, rtol=1e-6)
        if gaps.max() / gaps.min() > 1.05:
            scalar = 1.0 + gaps / 25.0
            self.assertGreater(scalar.max() / scalar.min(), 1.0)

    def test_demotion_holds_the_shallow_mode_longer_than_promotion_alone(self):
        """With every mode equally over-served, the demotion must keep the shallow one."""
        gmm = self.params.gmm
        ceiling = jnp.asarray(0.05)
        mass = jnp.full((gmm.log_weights.shape[0],), 2.0)
        promoted = np.asarray(deficit_weighted(mass, gmm, ceiling).log_weights)
        demoted = np.asarray(per_mode_weighted(mass, gmm, ceiling, 2.24).log_weights)
        shallow = int(np.argmin(np.asarray(responsibility_gaps(gmm))))
        self.assertGreater(demoted[shallow], promoted[shallow], (demoted, promoted))

    def test_a_unimodal_target_does_not_produce_nan(self):
        """One component has no rival, so Delta = +inf and the demotion must go inert.

        Left unguarded, ``inf * 0`` at exactly fair share is NaN and takes the whole field
        down -- which is how a unimodal literature scenario flew a path of NaNs.
        """
        import jax.numpy as jnp
        from ergodic_control_mppi.parameters import GMMParams

        covariance = jnp.asarray([[[4.0, 0.0], [0.0, 4.0]]], dtype=jnp.float32)
        single = GMMParams(
            means=jnp.zeros((1, 2), dtype=jnp.float32),
            covariance=covariance,
            covariance_inverse=jnp.linalg.inv(covariance),
            log_weights=jnp.zeros((1,), dtype=jnp.float32),
            log_normalizers=-0.5 * (2 * jnp.log(2 * jnp.pi)
                                    + jnp.linalg.slogdet(covariance)[1]),
        )
        self.assertFalse(np.isfinite(np.asarray(responsibility_gaps(single))).any())
        for mass in (jnp.ones((1,)), jnp.asarray([5.0]), jnp.asarray([1e-6])):
            bent = per_mode_weighted(mass, single, jnp.asarray(0.05), 2.24)
            self.assertTrue(np.all(np.isfinite(np.asarray(bent.log_weights))), mass)
            self.assertTrue(np.all(np.isfinite(
                np.asarray(score_pdf(jnp.asarray([[1.0, 2.0]], jnp.float32), bent)))))

    def test_the_bend_moves_the_score_field(self):
        """The destination bias must actually reach the attraction, or it does nothing."""
        gmm = self.params.gmm
        components = gmm.log_weights.shape[0]
        query = jnp.asarray([[0.0, 0.0], [-6.0, 1.0]], dtype=jnp.float32)
        ceiling = jnp.asarray(0.05)
        even = jnp.ones((components,))
        starved = jnp.asarray([1.0] * (components - 1) + [1e-3])
        a = score_pdf(query, deficit_weighted(even, gmm, ceiling))
        b = score_pdf(query, deficit_weighted(starved, gmm, ceiling))
        self.assertGreater(float(jnp.abs(a - b).max()), 1e-3,
                           "service mass does not move the score field")


class PlanRepulsionTest(unittest.TestCase):
    """Sec. III-D: the identity, and why the memory cannot substitute for it."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.params = _params(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_repulsion_is_minus_grad_of_the_kde(self):
        """``sum_m grad_zm kappa(z_m, z) == -grad_z sum_m kappa(z_m, z)``, exactly."""
        from ergodic_control_mppi.mppi.field import kde_potential

        rng = np.random.default_rng(3)
        plan = jnp.asarray(rng.uniform(-4.0, 4.0, size=(16, 2)), dtype=jnp.float32)
        weights = jnp.ones((16,), dtype=jnp.float32)
        bandwidth = jnp.asarray(0.94, dtype=jnp.float32)
        query = jnp.asarray([[0.4, -1.1]], dtype=jnp.float32)

        produced = kde_repulsion(query, plan, weights, bandwidth)[0]
        gradient = jax.grad(
            lambda z: kde_potential(z[None, :], plan, weights, bandwidth)[0]
        )(query[0])
        np.testing.assert_allclose(
            np.asarray(produced), -np.asarray(gradient), rtol=1e-4, atol=1e-6
        )

    def test_it_spreads_a_clumped_plan(self):
        """The claim the memory cannot make: a compact plan repels *itself* apart.

        A trail-repulsion term can only push away from the past. Given a plan clumped at one
        point the memory has never visited, it produces nothing; the plan term produces an
        outward field at every horizon point but the centre.
        """
        field = replace(self.params.field, fine_bandwidth=0.94)
        rng = np.random.default_rng(5)
        centre = np.array([3.0, -2.0])
        plan = jnp.asarray(centre + rng.normal(scale=0.15, size=(24, 2)), dtype=jnp.float32)
        flow = kde_repulsion(
            plan, plan, jnp.ones((24,), dtype=jnp.float32), field.fine_bandwidth
        )
        outward = np.asarray(plan) - centre
        alignment = np.sum(np.asarray(_unit(flow)) * (
            outward / np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
        ), axis=1)
        # The centre-most point has no preferred direction, so require the bulk, not all.
        self.assertGreater(float(np.mean(alignment > 0)), 0.85, alignment)


if __name__ == "__main__":
    unittest.main()
