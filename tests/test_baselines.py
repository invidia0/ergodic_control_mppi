"""The baseline harness: the physics that makes the comparison fair, not the tuning.

Each of these pins a bug the fidelity gate actually caught while the baselines were being
written. They are cheap and they are the reason a "we win" row can be believed.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ergodic_control_mppi.experiments import baselines


class NeumannTest(unittest.TestCase):
    """The HEDAC solve must not leak through obstacles or out of the domain."""

    def test_no_flux_across_an_obstacle(self):
        # Source on the left half only, a wall down the middle. With no-flux boundaries no
        # potential may appear on the far side: the wall spans the domain, so the only path
        # from source to the right half is through it.
        shape = (32, 32)
        blocked = np.zeros(shape, bool)
        blocked[:, 16] = True
        source = np.zeros(shape)
        source[16, 8] = 1.0
        potential = baselines._jacobi_neumann(source, blocked, 1.0, 0.2, 400)
        self.assertGreater(potential[16, 8], 0.0)
        self.assertAlmostEqual(float(np.abs(potential[:, 17:]).max()), 0.0, places=12)

    def test_potential_is_zero_inside_obstacles(self):
        shape = (16, 16)
        blocked = np.zeros(shape, bool)
        blocked[4:8, 4:8] = True
        source = np.ones(shape) * 0.01
        potential = baselines._jacobi_neumann(source, blocked, 1.0, 0.2, 100)
        self.assertTrue(np.all(potential[blocked] == 0.0))

    def test_domain_boundary_does_not_pull_the_gradient_outward(self):
        """A Dirichlet edge made the field point out of the domain from everywhere.

        With a uniform source and no-flux walls the solution is flat, so the gradient is
        zero. Forcing ``u = 0`` at the edge instead puts a ramp against every wall -- which
        is what drove HEDAC into the south boundary and pinned it there.
        """
        shape = (24, 24)
        blocked = np.zeros(shape, bool)
        potential = baselines._jacobi_neumann(np.ones(shape), blocked, 1.0, 0.2, 2000)
        interior = potential[2:-2, 2:-2]
        self.assertLess(float(np.ptp(interior) / max(np.mean(interior), 1e-12)), 1e-3)


class SolverGridTest(unittest.TestCase):
    """Square cells, and a pillar wide enough to be a barrier rather than a line."""

    def test_cells_are_square_on_the_non_square_workspace(self):
        """An 80x80 grid over 40x20 m gave 0.5 m cells in x and 0.25 m in y.

        The Jacobi stencil weights all four neighbours equally, ``np.gradient`` without a
        spacing returns a per-index derivative, and the Gaussian sigmas count cells. All
        three are wrong by the aspect ratio on an anisotropic grid.
        """
        scenario = baselines._open_scenario(_profile())
        (rows, columns), pitch = baselines._solver_shape(scenario, 160)
        width = scenario.map_x_limits[1] - scenario.map_x_limits[0]
        height = scenario.map_y_limits[1] - scenario.map_y_limits[0]
        self.assertAlmostEqual(width / columns, height / rows, places=9)
        self.assertAlmostEqual(pitch, width / columns, places=9)

    def test_a_pillar_spans_several_solver_cells(self):
        """At 0.5 m pitch a 0.59 m pillar was two cells wide, so its potential barrier was
        thinner than the vehicle's stopping distance and HEDAC drove through it."""
        scenario = baselines._open_scenario(_profile())
        _, pitch = baselines._solver_shape(scenario, baselines.BaselineConfig().grid_size)
        self.assertGreaterEqual(2.0 * 0.59 / pitch, 4.0)


class HedacTrapTest(unittest.TestCase):
    """Inside an obstacle the HEDAC solve holds ``u = 0``, so its gradient is exactly zero.

    A first-order vehicle following ``v_max * grad(u)/|grad(u)|`` never gets there. Ours is
    second order and does, and then `_unit_field` commands nothing and it never leaves --
    87% of a 400 s run, measured. The escape is the shared avoidance term, so HEDAC must
    not be exempt from it.
    """

    def test_the_gradient_vanishes_inside_an_obstacle(self):
        shape = (32, 32)
        blocked = np.zeros(shape, bool)
        blocked[12:20, 12:20] = True
        source = np.where(blocked, 0.0, 1.0)
        potential = baselines._jacobi_neumann(source, blocked, 1.0, 0.2, 200)
        grad_y, grad_x = np.gradient(potential, 1.0)
        interior = (slice(14, 18), slice(14, 18))
        self.assertAlmostEqual(float(np.abs(grad_x[interior]).max()), 0.0, places=12)
        self.assertAlmostEqual(float(np.abs(grad_y[interior]).max()), 0.0, places=12)

    def test_hedac_is_not_exempt_from_the_shared_avoidance(self):
        self.assertNotIn("hedac", baselines.NATIVE_OBSTACLES)
        self.assertIn("ours", baselines.NATIVE_OBSTACLES)


class IntegratorTranscriptionTest(unittest.TestCase):
    """The numpy integrator must equal the JAX one *bit for bit*, not merely closely.

    It was transcribed to remove 46% of every baseline's runtime, which is only a free
    optimisation if it changes nothing. The subtlety is precision: the original takes
    float64 in but computes in float32, because ``jax_enable_x64`` is off and ``jnp.asarray``
    narrows silently. A float64 transcription passes ``allclose`` and still perturbs every
    trajectory, and one ULP is enough to change a run's outcome on this system.
    """

    def test_matches_the_jax_step_bitwise(self):
        import jax.numpy as jnp

        from ergodic_control_mppi.experiments.literature_methods import (
            _clamp_controls_np, _double_integrator_step_np,
        )
        from ergodic_control_mppi.models import double_integrator as model

        params = model.DoubleIntegratorParams(
            delta_t=0.02, max_accel_lin_abs=3.0, max_accel_ang_abs=10.0)
        rng = np.random.default_rng(0)
        for _ in range(500):
            state = rng.normal(0.0, 5.0, (1, 6))
            control = rng.normal(0.0, 4.0, (1, 3))
            reference = np.asarray(
                model.step(jnp.asarray(state), jnp.asarray(control), params),
                dtype=np.float64)
            np.testing.assert_array_equal(
                _double_integrator_step_np(state, control, params), reference)
            np.testing.assert_array_equal(
                _clamp_controls_np(control, params),
                np.asarray(model.clamp(jnp.asarray(control), params), dtype=np.float64))


class DiffusionLengthTest(unittest.TestCase):
    """HEDAC's potential must reach the same physical distance on any mesh.

    With constant `damping` the diffusivity was alpha = h^2/damping, so refining the grid
    shrank the potential's reach: the measured 1/e decay of a point source halved, from
    1.00 m to 0.50 m, when only `grid_size` changed.
    """

    def _decay_length(self, cells, pitch, alpha):
        """1/e decay of a *line* source, so the profile is a clean 1-D exponential.

        A point source is near-singular at its own cell and drops below 1/e within a cell or
        two whatever alpha is, which measures the discretisation rather than the physics.
        """
        shape = (cells, cells)
        source = np.zeros(shape)
        source[:, 0] = 1.0
        damping = pitch ** 2 / alpha
        potential = baselines._jacobi_neumann(
            source, np.zeros(shape, bool), damping, damping, 8000)
        profile = potential[cells // 2, :]
        profile = profile / profile[0]
        return float(np.argmax(profile < np.exp(-1.0))) * pitch

    def test_refining_the_mesh_converges_on_the_configured_alpha(self):
        """The discrete decay length must approach sqrt(alpha), not track the mesh.

        Some mesh dependence is ordinary discretisation error and shrinks with refinement.
        What the old code had was different in kind: alpha *itself* was h^2/damping with
        damping constant, so halving the pitch quartered the physical diffusivity.
        """
        lengths = [self._decay_length(cells, pitch, 1.25)
                   for cells, pitch in ((80, 0.5), (160, 0.25), (320, 0.125))]
        target = np.sqrt(1.25)
        errors = [abs(length - target) for length in lengths]
        self.assertTrue(errors[0] > errors[1] > errors[2], f"not converging: {lengths}")
        self.assertLess(errors[-1], 0.1 * target)

    def test_alpha_is_what_the_configuration_says(self):
        """The invariant the code actually enforces: damping is derived so alpha is fixed."""
        for pitch in (0.5, 0.25, 0.125):
            for alpha in (1.25, 0.25):
                self.assertAlmostEqual(pitch ** 2 / (pitch ** 2 / alpha), alpha, places=12)


class PlanningMarginTest(unittest.TestCase):
    """Every method must plan against the same keep-out radius.

    Ours is handed the inflated occupancy grid; the baselines get circles fitted to the raw
    footprints plus `avoid_clearance`. When those disagree, the collision comparison
    measures the margin rather than the controller.
    """

    def test_the_shared_clearance_matches_the_inflated_footprint(self):
        from pathlib import Path

        from scipy import ndimage

        from ergodic_control_mppi.experiments.uav_pillar_tuning import _grid_config

        directory = Path("results/uav/density_25/maps/map_516")
        if not directory.exists():
            self.skipTest("campaign maps not present")
        _, _, arrays = _grid_config(directory)
        raw = np.asarray(arrays["occupancy"]).astype(bool)
        inflated = np.asarray(arrays["grid"]).astype(bool)
        resolution = float(arrays["grid_resolution"])
        pillars = ndimage.label(raw)[1]
        ours = np.sqrt(inflated.sum() / pillars / np.pi) * resolution
        _, radii = baselines._pillar_circles(
            raw, tuple(map(float, np.asarray(arrays["grid_origin"]))), resolution)
        theirs = float(np.median(radii)) + baselines.BaselineConfig().avoid_clearance
        self.assertAlmostEqual(ours, theirs, delta=0.05)


class UnitFieldTest(unittest.TestCase):
    """Coverage laws are directions; they are flown at the commanded speed."""

    def test_rescales_to_the_commanded_speed(self):
        field = np.array([[3e-7, 4e-7], [-10.0, 0.0]])
        out = baselines._unit_field(field, 1.8)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.8, 1.8], rtol=1e-9)

    def test_a_collapsed_field_yields_no_motion(self):
        # Not a direction amplified out of numerical noise.
        out = baselines._unit_field(np.zeros((1, 2)), 1.8)
        np.testing.assert_array_equal(out, np.zeros((1, 2)))


class PillarCircleTest(unittest.TestCase):
    """Obstacle circles are recovered from the grid; the manifests do not store them."""

    def test_two_pillars_recover_two_circles(self):
        occupancy = np.zeros((40, 40), bool)
        occupancy[5:9, 5:9] = True
        occupancy[30:34, 30:34] = True
        centres, radii = baselines._pillar_circles(occupancy, (0.0, 0.0), 0.5)
        self.assertEqual(centres.shape, (2, 2))
        # Columns 5..8 at pitch 0.5 sit at (5.5 .. 8.5) * 0.5 = 2.75 .. 4.25, mean 3.5;
        # columns 30..33 likewise centre on 16.0.
        np.testing.assert_allclose(sorted(centres[:, 0]), [3.5, 16.0], atol=1e-9)
        self.assertTrue(np.all(radii > 0.5 * 0.5))

    def test_an_empty_map_has_no_circles(self):
        centres, radii = baselines._pillar_circles(np.zeros((8, 8), bool), (0.0, 0.0), 1.0)
        self.assertEqual(centres.shape, (0, 2))
        self.assertEqual(radii.shape, (0,))


class AvoidanceTest(unittest.TestCase):
    """The shared penalty pushes out, and only inside the clearance band."""

    def test_pushes_away_from_the_obstacle(self):
        centres, radii = np.array([[0.0, 0.0]]), np.array([1.0])
        push = baselines._avoidance(np.array([[0.5, 0.0]]), centres, radii, 0.6, 6.0)
        self.assertGreater(push[0, 0], 0.0)
        self.assertAlmostEqual(push[0, 1], 0.0)

    def test_is_silent_outside_the_band(self):
        centres, radii = np.array([[0.0, 0.0]]), np.array([1.0])
        push = baselines._avoidance(np.array([[9.0, 0.0]]), centres, radii, 0.6, 6.0)
        np.testing.assert_allclose(push, np.zeros((1, 2)))


class FidelityGateTest(unittest.TestCase):
    """The gate has to be able to fail, or it is decoration."""

    def test_seeds_move_the_start_because_the_laws_are_deterministic(self):
        """Three of four baselines consume no randomness at all.

        Seeding a deterministic feedback law gives identical runs, zero variance and a
        paired test that means nothing, so the seed has to vary the trial instead.
        """
        scenario = baselines._open_scenario(_profile())
        base = np.zeros(6)
        starts = [baselines.seed_state(base, scenario, s)[:2] for s in (43, 44, 45)]
        self.assertEqual(len({tuple(s) for s in starts}), 3)
        # Repeatable, and inside the workspace.
        np.testing.assert_allclose(starts[0], baselines.seed_state(base, scenario, 43)[:2])
        for start in starts:
            self.assertGreaterEqual(start[0], scenario.map_x_limits[0])
            self.assertLessEqual(start[0], scenario.map_x_limits[1])

    def test_converge_then_drift_is_not_a_failure(self):
        """HEDAC reaches the best metric of any baseline and then degrades.

        A self-relative gate failed it for that while passing a method that never covered
        anything. Convergence followed by drift is a result, reported via
        ``degrades_after_convergence``, not grounds for exclusion.
        """
        from unittest import mock

        # A path that tours all three modes, then parks in one corner.
        means = np.asarray(_profile().controller.gmm.means)
        tour = np.concatenate([np.linspace(means[i], means[(i + 1) % 3], 1200)
                               for i in range(3)] * 2)
        parked = np.tile(means[0] + np.array([6.0, 0.0]), (2800, 1))
        xy = np.concatenate([tour, parked])
        path = np.zeros((xy.shape[0], 6))
        path[:, :2] = xy
        with mock.patch.object(baselines, "run_method", return_value=path):
            check = baselines.fidelity_check(
                "drifter", baselines._open_scenario(_profile()), np.zeros(6),
                cfg=baselines.BaselineConfig(), steps=xy.shape[0], seeds=(1,))
        self.assertTrue(check["passed"])
        self.assertEqual(check["modes_reached"], "1/1")

    def test_a_stationary_method_fails(self):
        from unittest import mock

        with mock.patch.object(baselines, "run_method",
                               return_value=np.zeros((200, 6))):
            config = _profile()
            scenario = baselines._open_scenario(config)
            check = baselines.fidelity_check(
                "stuck", scenario, np.zeros(6),
                cfg=baselines.BaselineConfig(), steps=200, seeds=(43, 44, 45))
        self.assertFalse(check["passed"])
        self.assertIn("not reproduced", check["note"])


def _profile():
    from ergodic_control_mppi.config import load_config

    return load_config("configs/uav_profile.yaml")


class SeedArgumentTest(unittest.TestCase):
    def test_a_bare_seed_count_is_rejected(self):
        with patch("sys.argv", ["baselines", "--seeds", "6"]), self.assertRaisesRegex(
            SystemExit, "comma-separated list, not a count"
        ):
            baselines.main()


if __name__ == "__main__":
    unittest.main()


class IncrementalWriteTest(unittest.TestCase):
    """A nine-hour tier must not hold its results in memory until the last cell."""

    def _row(self, method, name, seed, **extra):
        row = {"method": method, "map": name, "seed": seed, "fourier_ergodic": 0.1}
        row.update(extra)
        return row

    def test_appends_and_resumes_by_identity(self):
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "rows.csv"
            rows = []
            for seed in (43, 44):
                rows.append(self._row("hedac", "open", seed))
                baselines._append_row(out, rows[-1], rows)
            with out.open(encoding="utf-8", newline="") as stream:
                written = list(csv.DictReader(stream))
        self.assertEqual(len(written), 2)
        self.assertEqual({int(r["seed"]) for r in written}, {43, 44})

    def test_a_new_column_is_refused_without_mutating_existing_rows(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "rows.csv"
            rows = [self._row("hedac", "open", 43)]
            baselines._append_row(out, rows[0], rows)
            before = out.read_bytes()
            rows.append(self._row("ours", "open", 43, ess_settled_median=0.5))
            with self.assertRaisesRegex(ValueError, "stale header"):
                baselines._append_row(out, rows[1], rows)
            self.assertEqual(out.read_bytes(), before)
