import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.core import _rollouts, mppi_step, sample_epsilon, stage_cost
from ergodic_control_mppi.mppi.stein import kernel, kernel_gradient, logpdf, score_pdf
from tests.helpers import write_small_config


class NumericalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.params = load_config(write_small_config(Path(cls.temp.name))).controller

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_batched_dynamics_and_clamping(self):
        states = jnp.zeros((2, 6))
        controls = jnp.array([[100, -100, 100], [-100, 100, -100]], dtype=jnp.float32)
        limited = clamp(controls, self.params.model)
        self.assertEqual(step(states, controls, self.params.model).shape, (2, 6))
        self.assertTrue(np.all(np.abs(limited[:, :2]) <= self.params.model.max_accel_lin_abs))

    def test_analytic_gmm_score_matches_autodiff(self):
        point = jnp.array([1.2, -0.7], dtype=jnp.float32)
        expected = jax.grad(lambda value: logpdf(value, self.params.gmm))(point)
        np.testing.assert_allclose(score_pdf(point, self.params.gmm), expected, rtol=2e-5, atol=2e-5)

    def test_analytic_kernel_gradient_matches_autodiff(self):
        x = jnp.array([1.0, 2.0])
        y = jnp.array([-0.5, 0.25])
        bandwidth = jnp.array(3.0)
        expected = jax.grad(lambda value: kernel(value, y, bandwidth))(x)
        np.testing.assert_allclose(kernel_gradient(x, y, bandwidth), expected, rtol=1e-6)

    def test_collision_and_empty_obstacle_costs(self):
        state = jnp.zeros(6)
        workspace = replace(
            self.params.workspace,
            obstacles=jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            obstacle_cost=7.0,
        )
        self.assertEqual(float(stage_cost(state, replace(self.params, workspace=workspace))), 7.0)
        empty = replace(workspace, obstacles=jnp.zeros((0, 3)))
        self.assertEqual(float(stage_cost(state, replace(self.params, workspace=empty))), 0.0)

    def test_mppi_weights_shapes_and_prng_determinism(self):
        controls = jnp.zeros((self.params.mppi.horizon, 3))
        key = jax.random.PRNGKey(4)
        args = (self.params, controls, jnp.zeros(6), key, jnp.array(1.0), jnp.zeros((0, 2)))
        first = mppi_step(*args)
        second = mppi_step(*args)
        self.assertEqual(first.weights.shape, (self.params.mppi.samples,))
        self.assertEqual(first.optimal_trajectory.shape, (self.params.mppi.horizon, 6))
        self.assertAlmostEqual(float(first.weights.sum()), 1.0, places=6)
        np.testing.assert_array_equal(first.control, second.control)
        eps_a, _ = sample_epsilon(key, self.params)
        eps_b, _ = sample_epsilon(key, self.params)
        np.testing.assert_array_equal(eps_a, eps_b)

    def test_control_cost_uses_current_temperature(self):
        workspace = replace(self.params.workspace, obstacles=jnp.zeros((0, 3)), out_of_map_cost=0.0)
        params = replace(self.params, workspace=workspace)
        controls = jnp.ones((params.mppi.horizon, 3))
        epsilon = jnp.zeros((params.mppi.samples, params.mppi.horizon, 3))
        cost_one, _, _ = _rollouts(params, jnp.zeros(6), controls, epsilon, jnp.array(1.0))
        cost_two, _, _ = _rollouts(params, jnp.zeros(6), controls, epsilon, jnp.array(2.0))
        np.testing.assert_allclose(cost_two[:-1], 2 * cost_one[:-1], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
