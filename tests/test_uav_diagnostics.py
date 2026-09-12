"""Regression checks for JAX-to-UAV discrepancy attribution."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.uav_diagnostics import (
    _perturb_state,
    _run_first_perturbation,
    _run_stepwise,
    attribute_map,
    build_discrepancy_report,
    feedback_residuals,
    nudge_float32,
    overwrite_observation,
    path_sha256,
    tour_count,
)
from ergodic_control_mppi.mppi.single import initialize_single
from ergodic_control_mppi.simulation import controller_key, run_simulation
from tests.helpers import write_small_config


def _rows(map_seed, null, ideal, so3):
    rows = [
        {"source": "jax", "condition": "ulp", "vehicle": "none",
         "map_seed": str(map_seed), "tour_count": str(value)}
        for value in null
    ]
    rows.extend(
        {"source": "ros", "condition": vehicle, "vehicle": vehicle,
         "map_seed": str(map_seed), "tour_count": str(value)}
        for vehicle, values in (("ideal", ideal), ("so3", so3)) for value in values
    )
    return rows


def _complete_exact(map_seed, tours=0):
    return [{
        "source": "jax", "condition": "exact", "vehicle": "none",
        "map_seed": str(map_seed), "tour_count": str(tours), "hardware": hardware,
        "path_sha256": f"hash-{hardware}",
    } for hardware in ("jeff", "laptop") for _ in range(5)]


class PerturbationTest(unittest.TestCase):
    def test_zero_impulse_matches_run_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_small_config(Path(directory), steps=3))
            initial = np.array([-1.0, 0.5, 0.1, -0.2, 0.3, 0.0], dtype=np.float32)
            exact = run_simulation(
                config, device="cpu", initial_state=initial, preflight_steps=1
            ).paths[:, 0]
            diagnostic, _, _ = _run_first_perturbation(
                config, initial, np.zeros(6, dtype=np.float32), 0, 0, "cpu", 1
            )
            np.testing.assert_allclose(diagnostic, exact, rtol=1e-6, atol=1e-6)

    def test_stepwise_matches_scanned_simulation_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_small_config(Path(directory), steps=3))
            initial = np.array([-1.0, 0.5, 0.1, -0.2, 0.3, 0.0], dtype=np.float32)
            exact = run_simulation(
                config, device="cpu", initial_state=initial, preflight_steps=1
            ).paths[:, 0]
            stepwise = _run_stepwise(config, initial, "cpu", 1)
            np.testing.assert_allclose(stepwise, exact, rtol=1e-5, atol=1e-6)

    def test_ulp_generation_changes_only_intended_float32_values(self):
        values = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        changed = nudge_float32(values, sign_mask=0b0101, ulps=1)
        expected = values.copy()
        for axis in range(4):
            expected[axis] = np.nextafter(
                values[axis], np.float32(np.inf if 0b0101 & (1 << axis) else -np.inf),
                dtype=np.float32,
            )
        np.testing.assert_array_equal(changed.view(np.uint32), expected.view(np.uint32))
        state = np.array([*values, 0.25, -0.5], dtype=np.float32)
        jax_changed = np.asarray(
            _perturb_state(jnp.asarray(state), jnp.zeros(6), sign_mask=0b0101, ulps=1)
        )
        np.testing.assert_array_equal(jax_changed[:4].view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(jax_changed[4:], state[4:])

    def test_observation_updates_state_and_newest_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_small_config(Path(directory)))
            state = jnp.zeros(6, dtype=jnp.float32)
            carry = initialize_single(
                config.controller, state,
                jnp.zeros((config.controller.mppi.horizon, 3), dtype=jnp.float32),
                controller_key(config.run.seed),
            )
            observation = jnp.arange(6, dtype=jnp.float32)
            changed = overwrite_observation(carry, observation)
            np.testing.assert_array_equal(changed.state, observation)
            np.testing.assert_array_equal(changed.memory[-1], observation[:2])

    @classmethod
    def tearDownClass(cls):
        jax.clear_caches()


class ScoringTest(unittest.TestCase):
    def test_failures_score_zero_tours(self):
        self.assertEqual(tour_count({"first_all_modes_s": np.nan, "mode_cycles": 0}), 0)
        self.assertEqual(tour_count({"first_all_modes_s": 12.0, "mode_cycles": 2}), 3)

    def test_path_hash_is_stable_and_value_sensitive(self):
        path = np.arange(18, dtype=np.float32).reshape(3, 6)
        self.assertEqual(path_sha256(path), path_sha256(path.copy()))
        changed = path.copy()
        changed[0, 0] += 1
        self.assertNotEqual(path_sha256(path), path_sha256(changed))

    def test_feedback_residual_uses_latest_odometry_before_next_command(self):
        commands = np.zeros((3, 8))
        commands[:, 0] = [1.0, 2.0, 3.0]
        commands[0, [1, 2, 4, 5]] = [10.0, 20.0, 1.0, 2.0]
        odometry = np.zeros((2, 8))
        odometry[:, 0] = [1.9, 2.9]
        odometry[0, [1, 2, 4, 5]] = [10.1, 19.8, 1.3, 1.6]
        residual = feedback_residuals({"cmd_raw": commands, "odometry": odometry})
        np.testing.assert_allclose(residual["state"][0], [0.1, -0.2, 0.3, -0.4])
        self.assertAlmostEqual(residual["ages"][0], 0.1)


class AttributionTest(unittest.TestCase):
    def test_all_attribution_outcomes(self):
        self.assertEqual(
            attribute_map(_rows(516, [0, 1], [0, 1] * 12, [0, 1] * 12), 516),
            "numerical/rare-event",
        )
        self.assertEqual(
            attribute_map(_rows(516, [2, 2], [0] * 24, [0] * 24), 516),
            "ros-feedback/scheduling",
        )
        self.assertEqual(
            attribute_map(_rows(516, [0, 1], [1] * 24, [0] * 24), 516),
            "vehicle-dynamics",
        )
        self.assertEqual(
            attribute_map(_rows(516, [0, 1], [0, 2] * 12, [0, 2] * 12), 516),
            "unresolved",
        )

    def test_synthetic_reports_exercise_the_four_outcomes(self):
        cases = {
            "numerical/rare-event": ([0, 1], [0, 1] * 12, [0, 1] * 12),
            "ros-feedback/scheduling": ([2, 2], [0] * 24, [0] * 24),
            "vehicle-dynamics": ([0, 1], [1] * 24, [0] * 24),
            "unresolved": ([0, 1], [0, 2] * 12, [0, 2] * 12),
        }
        for outcome, values in cases.items():
            rows = _rows(516, *values) + _complete_exact(516, values[0][0])
            self.assertIn(f"Attribution: {outcome}.", build_discrepancy_report(rows))

    def test_report_identifies_map_specific_geometry_interaction(self):
        rows = _rows(516, [0, 1], [0, 1] * 12, [0, 1] * 12)
        rows += _rows(539, [2, 2], [0] * 24, [0] * 24)
        for map_seed in (516, 539):
            for hardware in ("jeff", "laptop"):
                rows.extend({
                    "source": "jax", "condition": "exact", "vehicle": "none",
                    "map_seed": str(map_seed),
                    "tour_count": "0" if map_seed == 516 else "2",
                    "hardware": hardware,
                    "path_sha256": f"hash-{map_seed}-{hardware}",
                } for _ in range(5))
        report = build_discrepancy_report(rows)
        self.assertIn("Geometry interaction", report)
        self.assertIn("3 SO3 flights", report)


if __name__ == "__main__":
    unittest.main()
