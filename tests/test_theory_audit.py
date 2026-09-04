"""Checks that the closed-loop analysis' inequalities hold on the loop they describe.

These are the executable half of the paper's Sec. "guarantees": each test asserts one
statement's inequality on real planning steps, so a change that breaks the analysis' premises
fails here rather than in a reviewer's reading.
"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.theory_audit import (
    endpoint_jacobian,
    step_residuals,
)
from ergodic_control_mppi.metrics.ergodicity import compute_ball_ergodic_metric
from ergodic_control_mppi.mppi.single import initialize_single, single_step
from tests.helpers import write_small_config

LIMITS_X = (-20.0, 20.0)
LIMITS_Y = (-10.0, 10.0)


def _params(directory: Path):
    return load_config(write_small_config(Path(directory))).controller


def _walk(params, steps: int, first: int = 2, stride: int = 1):
    """Replay ``params`` for ``steps`` steps, returning residuals from ``first`` on."""
    carry = initialize_single(
        params,
        jnp.zeros((6,), jnp.float32),
        jnp.zeros((params.mppi.horizon, 3), jnp.float32),
        jax.random.key(43),
    )
    advance = jax.jit(single_step)
    rows = []
    for index in range(steps):
        if index >= first and index % stride == 0:
            rows.append(step_residuals(params, carry))
        carry, _ = advance(params, carry)
    return rows


class ExecutedTrackingBoundTest(unittest.TestCase):
    """Prop. "executed_flow_tracking": eps_track <= 2 eps_avg + 2 eps_FM."""

    def test_bound_holds_at_every_step(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = _walk(_params(directory), steps=12)
        self.assertTrue(rows)
        for row in rows:
            # Both forms: as the proof builds it (k=0 only) and as the paper states it.
            self.assertLessEqual(row.eps_track, row.rhs_k0 * (1.0 + 1e-5))
            self.assertLessEqual(row.eps_track, row.rhs_full * (1.0 + 1e-5))

    def test_k0_slack_is_exactly_the_weighted_rollout_spread(self):
        """With eps_avg = 0 the sharp k=0 bound collapses to eps_fm_k0, whose slack is Var_w(v).

        No factor 2: the bound is ``(sqrt(eps_avg) + sqrt(eps_fm))^2``, so at eps_avg = 0 it
        *is* eps_fm_k0. That the remaining slack is exactly the weighted rollout spread is
        what says the k=0 step is Jensen and nothing else.
        """
        with tempfile.TemporaryDirectory() as directory:
            rows = _walk(_params(directory), steps=12)
        for row in rows:
            # The identity is a difference of two ~250 m^2/s^2 quantities whose gap is
            # ~1e-4, so it is only resolvable to the float32 ulp at that magnitude -- about
            # 1.5e-5 here. A fixed decimal tolerance would be asserting precision the
            # representation does not have; four ulps is what the arithmetic can carry.
            tolerance = 4.0 * float(np.spacing(np.float32(row.eps_track)))
            self.assertAlmostEqual(
                row.rhs_k0 - row.eps_track, row.jensen_slack, delta=tolerance
            )

    def test_averaging_gap_vanishes_under_a_convex_update(self):
        """The executed control is a convex combination of clamped rollout controls, and the
        position update is affine in it, so v_exec = v_bar exactly -- not approximately."""
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            self.assertEqual(params.mppi.smooth_window, 1)
            rows = _walk(params, steps=12)
        self.assertLess(max(row.eps_avg for row in rows), 1e-8)

    def test_averaging_gap_is_not_vacuous(self):
        """Smoothing breaks the convex combination, and the gap must then be measurable.

        Without this the previous test is satisfiable by an eps_avg that is identically zero
        for the wrong reason (a broken measurement), which would make the bound untestable.
        """
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            smoothed = replace(params, mppi=replace(params.mppi, smooth_window=3))
            shipped_rows = _walk(params, steps=12)
            smoothed_rows = _walk(smoothed, steps=12)
        shipped = max(row.eps_avg for row in shipped_rows)
        smoothed_gap = max(row.eps_avg for row in smoothed_rows)
        self.assertGreater(smoothed_gap, 1e3 * max(shipped, 1e-12))
        for row in smoothed_rows:  # and the bound still holds once the term is live
            self.assertLessEqual(row.eps_track, row.rhs_k0 * (1.0 + 1e-5))


class AssumptionTest(unittest.TestCase):
    """As. "endpoint" and the Sec. III-E margins, both checkable exactly."""

    def test_endpoint_map_has_full_row_rank_at_two_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
        state = jnp.zeros((6,), jnp.float32)
        single = endpoint_jacobian(params, state, jnp.zeros((1, 3), jnp.float32))
        self.assertEqual(np.linalg.matrix_rank(single), 3)  # one step cannot span the state
        pair = endpoint_jacobian(params, state, jnp.zeros((2, 3), jnp.float32))
        self.assertEqual(np.linalg.matrix_rank(pair), 6)
        # sigma_min is small only because it is the position channel's dt^2 sensitivity --
        # exactly dt^2/sqrt(2) for the two-step map -- not because the map is near-degenerate.
        self.assertAlmostEqual(
            float(np.linalg.svd(pair, compute_uv=False).min())
            / (params.model.delta_t ** 2 / np.sqrt(2.0)),
            1.0,
            places=3,
        )

    def test_saturated_witness_is_rank_deficient(self):
        """Why the assumption asks for an *interior* witness: clamp zeroes the derivative."""
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
        saturated = endpoint_jacobian(
            params, jnp.zeros((6,), jnp.float32), jnp.full((2, 3), 1e3, jnp.float32)
        )
        self.assertEqual(np.linalg.matrix_rank(saturated), 0)

    def test_promotion_cannot_overturn_the_smallest_margin(self):
        """Sec. III-E's split: the bend promotes, but only the demotion empties a basin.

        The multiplicative bend shifts a log-weight by at most ``log((c+1)/c)``. If that
        ever exceeds the smallest ``Delta_j``, promotion alone could free a mode and the
        pre-registered null on ``c`` would have no basis.
        """
        from ergodic_control_mppi.mppi.field import responsibility_gaps

        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
        ceiling = float(params.field.deficit_ceiling)
        if ceiling <= 0:
            self.skipTest("destination bias disabled in this config")
        promotion = np.log((ceiling + 1.0) / ceiling)
        gaps = np.asarray(responsibility_gaps(params.gmm), dtype=np.float64)
        self.assertLess(promotion, gaps.min())


class BallErgodicMetricTest(unittest.TestCase):
    """The metric Prop. "ergodic_error_decomposition" is stated in."""

    def test_matching_measures_score_zero(self):
        grid = np.random.default_rng(0).random((40, 80))
        self.assertEqual(compute_ball_ergodic_metric(grid, grid, LIMITS_X, LIMITS_Y), 0.0)

    def test_two_disjoint_atoms_match_the_closed_form(self):
        """Residual per radius is two disjoint discs, so E = int_0^R 2 pi r^2 dr."""
        left = np.zeros((200, 400))
        left[100, 120] = 1.0
        right = np.zeros((200, 400))
        right[100, 280] = 1.0
        measured = compute_ball_ergodic_metric(
            left, right, LIMITS_X, LIMITS_Y, max_radius=5.0, radii=256
        )
        self.assertAlmostEqual(measured / (2.0 * np.pi * 5.0 ** 3 / 3.0), 1.0, places=3)

    def test_zero_padding_prevents_wraparound(self):
        """Two atoms one cell apart *across the seam* are far apart, not adjacent.

        A circular convolution would make these two cases score identically, since west and
        east are also one cell apart the wrong way round. They must not.
        """
        grid_shape = (100, 200)
        west = np.zeros(grid_shape)
        west[50, 0] = 1.0
        east = np.zeros(grid_shape)
        east[50, -1] = 1.0
        near = np.zeros(grid_shape)
        near[50, 100] = 1.0
        far = np.zeros(grid_shape)
        far[50, 101] = 1.0
        seam = compute_ball_ergodic_metric(west, east, LIMITS_X, LIMITS_Y, max_radius=2.0)
        adjacent = compute_ball_ergodic_metric(near, far, LIMITS_X, LIMITS_Y, max_radius=2.0)
        self.assertGreater(seam, 3.0 * adjacent)

    def test_non_square_cells_still_measure_circles(self):
        """The deployment raster is square only to 0.4%, so the disc is built in world units.

        On a deliberately 2:1 grid, two atoms separated the same world distance along x and
        along y must score the same -- an ellipse would split them.
        """
        shape = (100, 100)  # 0.4 x 0.2 m cells over the 40 x 20 workspace
        centre = np.zeros(shape)
        centre[50, 50] = 1.0
        east = np.zeros(shape)
        east[50, 55] = 1.0  # +2.0 m in x
        north = np.zeros(shape)
        north[60, 50] = 1.0  # +2.0 m in y
        along_x = compute_ball_ergodic_metric(centre, east, LIMITS_X, LIMITS_Y, max_radius=3.0)
        along_y = compute_ball_ergodic_metric(centre, north, LIMITS_X, LIMITS_Y, max_radius=3.0)
        self.assertAlmostEqual(along_x / along_y, 1.0, places=2)




class TVEstimatorSweepTest(unittest.TestCase):
    """The (resolution, K) sweep behind the TV-estimator bias correction."""

    @staticmethod
    def _sweep_module():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_cli", Path(__file__).resolve().parents[1] / "scripts" / "theory_audit.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_coarsen_conserves_mass_and_widens_mask(self):
        module = self._sweep_module()
        grid = np.arange(64, dtype=float).reshape(8, 8)
        for factor in (1, 2, 4, 8):
            reduced = module._coarsen(grid, factor, "sum")
            self.assertEqual(reduced.shape, (8 // factor, 8 // factor))
            # Block-summing is a partition of the cells, so total mass is invariant. This is
            # what lets a coarse target stay a probability measure without renormalizing.
            self.assertAlmostEqual(float(reduced.sum()), float(grid.sum()), places=9)

        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True
        # One reachable cell must keep its coarse block reachable, or coarsening would
        # silently delete free space and inflate every TV computed on it.
        self.assertTrue(bool(module._coarsen(mask, 4, "any")[0, 0]))
        self.assertEqual(int(module._coarsen(mask, 4, "any").sum()), 1)

    def test_tv_is_zero_when_occupancy_matches_target(self):
        module = self._sweep_module()
        limits_x, limits_y = (-1.0, 1.0), (-1.0, 1.0)
        # A path visiting each cell of a 2x2 grid equally, against a uniform target: the two
        # normalized measures coincide, so the estimator must return exactly zero.
        positions = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
        target = np.ones((2, 2))
        mask = np.ones((2, 2), dtype=bool)
        self.assertAlmostEqual(
            module._tv(positions, target, mask, limits_x, limits_y), 0.0, places=9
        )

    def test_tv_reaches_one_on_disjoint_support(self):
        module = self._sweep_module()
        limits_x, limits_y = (-1.0, 1.0), (-1.0, 1.0)
        # Path confined to the left column, target supported only on the right: mutually
        # singular, so TV is 1. Guards the normalization -- a wrong denominator shows up here.
        positions = np.array([[-0.5, -0.5], [-0.5, 0.5]])
        target = np.array([[0.0, 1.0], [0.0, 1.0]])
        mask = np.ones((2, 2), dtype=bool)
        self.assertAlmostEqual(
            module._tv(positions, target, mask, limits_x, limits_y), 1.0, places=9
        )


class PropositionThreeFormsTest(unittest.TestCase):
    """Prop. 3 states one bound, a rewriting of it, and a relaxation of it -- not three."""

    @staticmethod
    def _terms():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_cli", Path(__file__).resolve().parents[1] / "scripts" / "theory_audit.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(0)
        target = np.zeros((16, 16))
        target[4:8, 4:8] = 1.0
        target[10:14, 10:14] = 2.0
        arrays = {"target_grid": target, "reachable_mask": np.ones((16, 16), dtype=bool)}
        positions = rng.uniform(-1.0, 1.0, size=(4000, 2))
        return module.coverage_terms(positions, arrays, (-1.0, 1.0), (-1.0, 1.0))

    def test_l1_bound_is_the_tv_bound(self):
        terms = self._terms()
        # TV = ||.||_1 / 2 by definition, so |O|R/4 ||.||_1^2 == |O|R TV^2. The two Prop. 3
        # forms are one inequality; a divergence here would mean the renormalization broke.
        self.assertAlmostEqual(terms["l1"], 2.0 * terms["tv"], places=12)
        self.assertTrue(terms["bound_l1_matches_tv"])
        self.assertAlmostEqual(terms["bound_l1"], terms["bound_tv"], places=9)

    def test_kl_bound_is_never_tighter_than_tv(self):
        terms = self._terms()
        # Pinsker gives TV^2 <= KL/2, so the KL form is a *relaxation*: it can only be looser.
        # If this ever inverts, the KL is being computed against a different normalization.
        self.assertGreaterEqual(terms["bound_kl"], terms["bound_tv"] - 1e-12)


class IdealFlowKernelTest(unittest.TestCase):
    """The comparison kernel of As. 7 must track the reference field exactly, by construction."""

    def test_executed_velocity_equals_the_reference_field(self):
        from ergodic_control_mppi.experiments.theory_audit import ideal_step
        from ergodic_control_mppi.mppi.core import _rollouts, reference_flow, sample_epsilon

        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            carry = initialize_single(
                params,
                jnp.zeros((6,), jnp.float32),
                jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                jax.random.key(11),
            )
            nxt = ideal_step(params, carry)

            # Recompute the field independently of the step under test.
            epsilon, _ = sample_epsilon(carry.key, params)
            _, _, sampled = _rollouts(
                params, carry.state, carry.controls, epsilon, carry.temperature
            )
            origin = carry.state[:2]
            initial = jnp.broadcast_to(origin, (params.mppi.samples, 1, 2))
            evaluation = jnp.concatenate((initial, sampled[:, :-1]), axis=1)
            flow = reference_flow(params, evaluation, carry.memory, carry.service_mass)[0]

            # eps_track is zero for this kernel wherever the projection is inactive: the
            # executed spatial velocity IS the reference. If this drifts, the "ideal" run is
            # not ideal and every conclusion drawn from it about As. 7 is void. The step below
            # starts at the origin and moves 3.6 cm, so no constraint is anywhere near active.
            executed = (nxt.state[:2] - origin) / params.model.delta_t
            np.testing.assert_allclose(
                np.asarray(executed), np.asarray(flow), rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                np.asarray(nxt.state[2:4]), np.asarray(flow), rtol=1e-6, atol=1e-6
            )

    def test_memory_records_the_executed_position(self):
        from ergodic_control_mppi.experiments.theory_audit import ideal_step

        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            carry = initialize_single(
                params,
                jnp.zeros((6,), jnp.float32),
                jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                jax.random.key(12),
            )
            nxt = ideal_step(params, carry)
            # The fading memory must hold executed positions, or the coverage feedback is
            # reacting to a trajectory that was never flown.
            np.testing.assert_allclose(
                np.asarray(nxt.memory[-1]), np.asarray(nxt.state[:2]), rtol=0, atol=0
            )


class ProjectionTest(unittest.TestCase):
    """The confined ideal kernel must stay in the admissible set without moving otherwise."""

    def test_admissible_points_are_untouched(self):
        from ergodic_control_mppi.experiments.theory_audit import project_admissible

        with tempfile.TemporaryDirectory() as directory:
            workspace = _params(directory).workspace
            inside = jnp.asarray([0.0, 0.0], jnp.float32)
            # Only meaningful if the origin really is admissible in the small test config.
            gaps = jnp.linalg.norm(inside - workspace.obstacles[:, :2], axis=-1) - (
                workspace.obstacles[:, 2] + workspace.safe_distance
            )
            self.assertGreater(float(gaps.min()), 0.0)
            np.testing.assert_allclose(
                np.asarray(project_admissible(inside, workspace)), np.zeros(2), atol=0
            )

    def test_points_outside_the_box_are_clipped(self):
        from ergodic_control_mppi.experiments.theory_audit import project_admissible

        with tempfile.TemporaryDirectory() as directory:
            workspace = _params(directory).workspace
            far = jnp.asarray([1e3, -1e3], jnp.float32)
            got = np.asarray(project_admissible(far, workspace))
            self.assertLessEqual(got[0], float(workspace.x_limits[1]) + 1e-6)
            self.assertGreaterEqual(got[1], float(workspace.y_limits[0]) - 1e-6)

    def test_points_inside_a_pillar_are_pushed_to_the_keepout_radius(self):
        from ergodic_control_mppi.experiments.theory_audit import project_admissible

        with tempfile.TemporaryDirectory() as directory:
            workspace = _params(directory).workspace
            centre = workspace.obstacles[0, :2]
            keepout = float(workspace.obstacles[0, 2] + workspace.safe_distance)
            # Offset rather than the exact centre: the push direction is undefined there.
            got = project_admissible(centre + jnp.asarray([0.01, 0.0], jnp.float32), workspace)
            self.assertAlmostEqual(
                float(jnp.linalg.norm(got - centre)), keepout, places=4
            )

    def test_recorded_velocity_aligns_with_the_move_it_produced(self):
        """``scripts/theory_audit.py`` recovers eps_track from the path using this alignment.

        ``state[k][2:4]`` is the flow that produced the move *into* ``k``, so the reference for
        the step ``k-1 -> k`` is ``state[k]``. Reading ``state[k-1]`` instead is silent -- it
        yields a plausible O(0.1) residual rather than an error -- and inflated the measured
        projection rate tenfold before this was pinned.
        """
        from ergodic_control_mppi.experiments.theory_audit import ideal_walk

        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            states = np.asarray(ideal_walk(
                params,
                jnp.zeros((6,), jnp.float32),
                jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                jax.random.key(5),
                steps=60,
            ))
            realized = np.diff(states[:, :2], axis=0) / float(params.model.delta_t)
            aligned = np.sum((realized - states[1:, 2:4]) ** 2, axis=1)
            shifted = np.sum((realized - states[:-1, 2:4]) ** 2, axis=1)
            # Interior walk, so the projection is inactive and the aligned residual is float32
            # round-trip noise only; the shifted one is not.
            self.assertLess(aligned.max(), 1e-6)
            self.assertGreater(shifted.max(), 1e-4)

    def test_the_ideal_walk_never_leaves_the_admissible_set(self):
        """The whole point of the projection: As. 1 must hold for the comparison kernel."""
        from ergodic_control_mppi.experiments.theory_audit import ideal_walk

        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            states = ideal_walk(
                params,
                jnp.zeros((6,), jnp.float32),
                jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                jax.random.key(7),
                steps=200,
            )
            position = np.asarray(states[:, :2])
            self.assertLessEqual(position[:, 0].max(), float(params.workspace.x_limits[1]) + 1e-5)
            self.assertGreaterEqual(position[:, 0].min(), float(params.workspace.x_limits[0]) - 1e-5)
            self.assertLessEqual(position[:, 1].max(), float(params.workspace.y_limits[1]) + 1e-5)
            self.assertGreaterEqual(position[:, 1].min(), float(params.workspace.y_limits[0]) - 1e-5)
            gaps = np.linalg.norm(
                position[:, None, :] - np.asarray(params.workspace.obstacles[:, :2]), axis=-1
            ) - np.asarray(params.workspace.obstacles[:, 2] + params.workspace.safe_distance)
            self.assertGreaterEqual(gaps.min(), -1e-5)


if __name__ == "__main__":
    unittest.main()
