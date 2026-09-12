"""Regressions for volumetric (d > 2) controller behavior."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.bunny import fourier_target, surface_target
from ergodic_control_mppi.experiments.dimension import continuous_mmd, lift
from ergodic_control_mppi.metrics.ergodicity import fourier_wavenumbers
from ergodic_control_mppi.mppi.core import stage_cost
from ergodic_control_mppi.mppi.field import (
    attraction_target,
    kde_repulsion,
    memory_repulsion,
    potential,
    responsibility_gaps,
    score_pdf,
)
from ergodic_control_mppi.mppi.single import run_single
from tests.helpers import write_small_config

MEANS = [[-4.0, 0.0, 1.0], [4.0, 2.0, 3.0]]
COVARIANCES = [np.diag([2.0, 2.0, 0.5])] * 2
LIMITS = [[-10.0, 10.0], [-6.0, 6.0], [0.0, 5.0]]
SHORT_PILLAR = [[0.0, 0.0, 0.5, 2.0]]  # top at 2 m


class VolumetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            planar = load_config(write_small_config(Path(directory))).controller
        cls.params = lift(planar, MEANS, COVARIANCES, [0.5, 0.5], LIMITS, SHORT_PILLAR)
        cls.open = lift(planar, MEANS, COVARIANCES, [0.5, 0.5], LIMITS)

    def test_grad_phi_is_the_field_in_3d(self):
        """The pre-gauge field is grad Phi in 3D too (the planar `PotentialTest`, lifted)."""
        rng = np.random.default_rng(0)
        memory = jnp.asarray(rng.uniform(-5, 5, (48, 3)), jnp.float32)
        plan = jnp.asarray(rng.uniform(-5, 5, (24, 3)), jnp.float32)
        queries = jnp.asarray(rng.uniform(-4, 4, (9, 3)), jnp.float32)
        field, gmm = self.params.field, self.params.gmm
        recency = jnp.asarray(float(field.memory_decay) ** np.arange(48)[::-1], jnp.float32)
        floor = jnp.float32(1.0 / 1200.0)
        gauge = jnp.sqrt(0.5 * jnp.e * field.fine_bandwidth)
        flow = score_pdf(queries, attraction_target(gmm, field, None))
        flow += field.memory_gain * memory_repulsion(queries, memory, recency, gmm, field, floor)
        flow += field.plan_gain * gauge * kde_repulsion(
            queries, plan, jnp.ones((24,), jnp.float32), field.fine_bandwidth)
        gradient = jax.vmap(jax.grad(lambda point: potential(
            point[None], memory, recency, plan, gmm, field, floor)[0]))(queries)
        produced, gradient = np.asarray(flow), np.asarray(gradient)
        scale = max(np.abs(gradient).max(), 1e-6)
        self.assertLess(np.abs(produced - gradient).max() / scale, 1e-3)

    def test_a_short_pillar_is_cleared_from_above(self):
        """Over the pillar's top it costs nothing; beside it, exactly the obstacle cost."""
        states = np.zeros((2, 8), np.float32)
        states[:, 2] = [3.0, 1.8]  # above the 2 m top plus clearance, then below it
        states = jnp.asarray(states)
        extra = np.asarray(stage_cost(states, self.params) - stage_cost(states, self.open))
        np.testing.assert_allclose(
            extra, [0.0, float(self.params.workspace.obstacle_cost)], rtol=1e-6)

    def test_a_voxel_map_is_charged_in_3d(self):
        """A ``(Z, Y, X)`` grid charges the blocked voxel and not the one above it."""
        grid = np.zeros((5, 12, 20), np.float32)
        grid[1, 8, 13] = 1.0  # the voxel holding (3.5, 2.5, 1.5)
        voxels = replace(self.open, workspace=replace(
            self.open.workspace, grid=jnp.asarray(grid),
            grid_origin=jnp.asarray([-10.0, -6.0, 0.0], jnp.float32), grid_resolution=1.0))
        states = np.zeros((2, 8), np.float32)
        states[:, :3] = [[3.5, 2.5, 1.5], [3.5, 2.5, 3.5]]
        states = jnp.asarray(states)
        extra = np.asarray(stage_cost(states, voxels) - stage_cost(states, self.open))
        np.testing.assert_allclose(
            extra, [float(self.open.workspace.obstacle_cost), 0.0], rtol=1e-6)

    def test_surface_patches_tile_a_sphere_shell(self):
        """Fitted 1 m off a sphere, the patches sit on the shell and the gate stays live."""
        pitch, lower, radius = 0.1, np.array([-4.0, -4.0, 1.0]), 2.0
        z, y, x = np.meshgrid(*(lower[k] + (np.arange(80) + 0.5) * pitch for k in (2, 1, 0)),
                              indexing="ij")
        distance = np.maximum(np.sqrt(x ** 2 + y ** 2 + (z - 5.0) ** 2) - radius, 0.0)
        means, covariances, weights = surface_target(distance, lower, pitch, patches=12)
        reach = np.linalg.norm(means - [0.0, 0.0, 5.0], axis=1)
        self.assertTrue(np.all((reach > 2.4) & (reach < 3.05)), reach)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        box = [[-4.0, 4.0], [-4.0, 4.0], [1.0, 9.0]]
        gaps = np.asarray(responsibility_gaps(lift(self.open, means, covariances, weights,
                                                   box).gmm))
        self.assertTrue(np.all(np.isfinite(gaps) & (gaps > 0.0)), gaps)

    def test_fourier_coefficients_match_a_direct_sum(self):
        """The factored 3D coefficients equal summing every basis function over the grid."""
        rng = np.random.default_rng(2)
        centres = [np.linspace(-6.5, 6.5, 7), np.linspace(-6.0, 6.0, 5), np.linspace(-1.0, 8.5, 4)]
        target = rng.random((4, 5, 7))
        k, _, phi = fourier_target(target, centres, 3)
        z, y, x = np.meshgrid(centres[2], centres[1], centres[0], indexing="ij")
        points = np.stack([x, y, z], -1).reshape(-1, 3)
        scale = np.pi * k / np.array([14.0, 14.0, 10.5])
        basis = np.prod(np.cos(scale[None] * (points[:, None, :] - [-7.0, -7.0, -1.5])), -1)
        np.testing.assert_allclose(phi, target.reshape(-1) @ basis, rtol=1e-10, atol=1e-10)

    def test_avoidance_points_out_of_a_body_from_inside_too(self):
        """Off a sphere's signed distance, the push is radial and outward on both sides."""
        from ergodic_control_mppi.experiments import bunny

        z, y, x = np.meshgrid(*(bunny.LOWER[k] + (np.arange(bunny.SHAPE[2 - k]) + 0.5)
                                * bunny.VOXEL for k in (2, 1, 0)), indexing="ij")
        centre = np.array([0.0, 0.0, 4.0])
        clearance = bunny.signed_distance(x ** 2 + y ** 2 + (z - 4.0) ** 2 <= 4.0)
        surfaces = bunny.surface_trees(clearance)
        # Just outside, inside the outermost solid voxel, and deep inside. Within 60 degrees
        # of radial: a sub-voxel offset to the nearest voxel centre tilts the direction.
        for point in ([0.0, 0.0, 6.15], [0.0, 1.97, 4.02], [1.0, -1.0, 3.0]):
            push = bunny.avoidance(np.array(point), clearance, surfaces, 0.3, 6.0)
            radial = (np.array(point) - centre) / np.linalg.norm(np.array(point) - centre)
            self.assertGreater(float(push @ radial), 0.5 * np.linalg.norm(push))
            self.assertGreater(np.linalg.norm(push), 0.0)
        far = bunny.avoidance(np.array([0.0, 0.0, 7.0]), clearance, surfaces, 0.3, 6.0)
        np.testing.assert_array_equal(far, np.zeros(3))

    def test_no_flux_gradient_reflects_at_a_wall(self):
        """Open space: a central difference. Beside a blocked cell: the wall reflects."""
        from ergodic_control_mppi.experiments.bunny import no_flux_gradient

        field = np.arange(27.0).reshape(3, 3, 3)  # 9 z + 3 y + x
        blocked = np.zeros((3, 3, 3), bool)
        np.testing.assert_allclose(no_flux_gradient(field, blocked, (1, 1, 1), 0.5),
                                   [2.0, 6.0, 18.0])
        blocked[1, 1, 2] = True  # the +x neighbour
        np.testing.assert_allclose(no_flux_gradient(field, blocked, (1, 1, 1), 0.5),
                                   [1.0, 6.0, 18.0])

    def test_perlin_noise_and_its_drawn_surface(self):
        """Noise vanishes on the lattice and repeats per seed; the cloud keeps solid cells."""
        from ergodic_control_mppi.experiments import perlin

        rng = np.random.default_rng(3)
        np.testing.assert_array_equal(
            perlin.perlin(rng.integers(-40, 40, (64, 3)).astype(float), 0), 0.0)
        points = rng.uniform(-40, 40, (512, 3))
        first = perlin.perlin(points, 0)
        np.testing.assert_array_equal(first, perlin.perlin(points, 0))
        self.assertFalse(np.allclose(first, perlin.perlin(points, 1)))
        self.assertLess(np.abs(first).max(), 1.5)
        solid = np.zeros((6, 6, 6), bool)
        solid[3:6, 0:3, 0:3] = True  # one whole cell at pitch 3, one layer up
        solid[0, 5, 5] = True        # and a stray voxel, too little to draw its cell
        np.testing.assert_allclose(perlin.cloud(solid, pitch=3),
                                   [perlin.LOWER + [0.15, 0.15, 0.45]])

    def test_closed_loop_runs_in_3d(self):
        start = jnp.asarray([5.0, 0.0, 2.5, 0, 0, 0, 0, 0], jnp.float32)
        path = run_single(self.params, start, jnp.zeros((self.params.mppi.horizon, 4)),
                          jax.random.PRNGKey(0), 3).path
        self.assertEqual(path.shape, (3, 8))
        self.assertTrue(np.isfinite(np.asarray(path)).all())


class ContinuousMMDTest(unittest.TestCase):
    def test_matches_monte_carlo(self):
        """The closed-form embedding and norm agree with sampling the mixture in 3D."""
        rng = np.random.default_rng(1)
        means, variance, bandwidth = rng.uniform(-3, 3, (2, 3)), 1.0, 0.94
        path = rng.normal(0.0, 2.0, (50, 3))
        count = 200_000
        first = means[rng.integers(0, 2, count)] + rng.normal(0, np.sqrt(variance), (count, 3))
        second = means[rng.integers(0, 2, count)] + rng.normal(0, np.sqrt(variance), (count, 3))

        def kernel(a, b):
            return np.exp(-np.sum((a - b) ** 2, axis=-1) / bandwidth)

        sampled = (kernel(path[:, None], path[None]).mean()
                   - 2.0 * np.mean([kernel(z, first).mean() for z in path])
                   + kernel(first, second).mean())
        self.assertAlmostEqual(continuous_mmd(path, means, variance, bandwidth), sampled,
                               delta=6e-3)


class WavenumberTest(unittest.TestCase):
    def test_planar_modes_are_unchanged(self):
        k, lam = fourier_wavenumbers(3)
        expected = np.asarray([(kx, ky) for kx in range(4) for ky in range(4)][1:], float)
        np.testing.assert_array_equal(k, expected)
        np.testing.assert_array_equal(lam, np.power(1.0 + np.sum(expected ** 2, 1), -1.5))
        self.assertEqual(len(fourier_wavenumbers(5, 3)[0]), 6 ** 3 - 1)


if __name__ == "__main__":
    unittest.main()
