import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.models.double_integrator import clamp, step
from ergodic_control_mppi.mppi.core import (
    _flow_tracking_cost,
    _rollouts,
    mppi_step,
    sample_epsilon,
    stage_cost,
)
from ergodic_control_mppi.mppi.stein import (
    kernel,
    kernel_gradient,
    logpdf,
    score_pdf,
    stein_gradient,
    stein_repulsion,
)
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
        memory = jnp.zeros((self.params.mppi.memory_length, 2))
        args = (self.params, controls, jnp.zeros(6), key, jnp.array(1.0), memory)
        first = mppi_step(*args)
        second = mppi_step(*args)
        self.assertEqual(first.weights.shape, (self.params.mppi.samples,))
        self.assertEqual(first.optimal_trajectory.shape, (self.params.mppi.horizon, 6))
        self.assertAlmostEqual(float(first.weights.sum()), 1.0, places=6)
        np.testing.assert_array_equal(first.control, second.control)
        shifted_memory = memory + jnp.array([1.0, -1.0])
        shifted = mppi_step(*args[:-1], shifted_memory)
        self.assertFalse(np.allclose(first.weights, shifted.weights))
        eps_a, _ = sample_epsilon(key, self.params)
        eps_b, _ = sample_epsilon(key, self.params)
        np.testing.assert_array_equal(eps_a, eps_b)

    def test_spiral_term_toggles_control(self):
        controls = jnp.zeros((self.params.mppi.horizon, 3))
        key = jax.random.PRNGKey(0)
        state = jnp.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        # trail ~0.3 m from the state so the fine (0.4 m) spiral term is active
        memory = jnp.broadcast_to(jnp.array([5.3, 5.0]), (self.params.mppi.memory_length, 2))
        active = mppi_step(self.params, controls, state, key, jnp.array(1.0), memory)
        off = replace(self.params, stein=replace(self.params.stein, spiral_weight=0.0))
        disabled = mppi_step(off, controls, state, key, jnp.array(1.0), memory)
        self.assertFalse(np.allclose(active.control, disabled.control))

    def test_control_cost_uses_current_temperature(self):
        workspace = replace(self.params.workspace, obstacles=jnp.zeros((0, 3)), out_of_map_cost=0.0)
        params = replace(self.params, workspace=workspace)
        controls = jnp.ones((params.mppi.horizon, 3))
        epsilon = jnp.zeros((params.mppi.samples, params.mppi.horizon, 3))
        cost_one, _, _ = _rollouts(params, jnp.zeros(6), controls, epsilon, jnp.array(1.0))
        cost_two, _, _ = _rollouts(params, jnp.zeros(6), controls, epsilon, jnp.array(2.0))
        np.testing.assert_allclose(cost_two[:-1], 2 * cost_one[:-1], rtol=1e-6)

    def test_displacement_flow_cost_and_translation_invariance(self):
        flow = jnp.array(
            [[[1.0, 0.5], [0.25, -1.0]], [[-0.5, 0.75], [1.5, 0.25]]]
        )
        displacements = jnp.array(
            [[[0.2, -0.1], [0.4, 0.3]], [[-0.3, 0.2], [0.1, -0.4]]]
        )
        time_step = 0.2
        expected = jnp.sum(
            -time_step * jnp.sum(flow * displacements, axis=-1)
            + 0.5 * jnp.sum(displacements**2, axis=-1),
            axis=-1,
        )
        np.testing.assert_allclose(
            _flow_tracking_cost(flow, displacements, time_step), expected
        )
        self.assertTrue(
            np.all(
                _flow_tracking_cost(flow, time_step * flow, time_step)
                < _flow_tracking_cost(flow, -time_step * flow, time_step)
            )
        )

        particles = jnp.array([[0.0, 0.5], [0.4, 0.8]])
        increments = jnp.array(
            [[[0.1, 0.0], [0.2, -0.1]], [[-0.1, 0.2], [0.0, 0.1]]]
        )
        bandwidth = jnp.array(1.5)

        def cost_at(source_particles, gmm):
            flow = stein_gradient(
                source_particles, source_particles, gmm, self.params.stein, bandwidth
            )
            return _flow_tracking_cost(flow[None], increments, time_step)

        shift = jnp.array([3.0, -2.0])
        shifted_gmm = replace(self.params.gmm, means=self.params.gmm.means + shift)
        np.testing.assert_allclose(
            cost_at(particles, self.params.gmm),
            cost_at(particles + shift, shifted_gmm),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_weighted_repulsion_normalizes_and_selects(self):
        positions = jnp.array([[0.5, -0.5], [1.0, 2.0]])
        particles = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        bandwidth = jnp.array(3.0)
        stein = self.params.stein
        uniform = jnp.ones((particles.shape[0],))
        rotated = kernel_gradient(particles[None], positions[:, None], bandwidth) @ stein.rotation.T
        # Uniform weights reproduce the plain (unweighted) mean over particles.
        np.testing.assert_allclose(
            stein_repulsion(positions, particles, uniform, stein, bandwidth),
            jnp.mean(rotated, axis=1),
            rtol=1e-6,
        )
        # Scaling all weights by a constant leaves the normalized result unchanged.
        np.testing.assert_allclose(
            stein_repulsion(positions, particles, 5.0 * uniform, stein, bandwidth),
            stein_repulsion(positions, particles, uniform, stein, bandwidth),
            rtol=1e-6,
        )
        # A one-hot weight selects that particle's rotated contribution.
        one_hot = jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            stein_repulsion(positions, particles, one_hot, stein, bandwidth),
            rotated[:, 1, :],
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
