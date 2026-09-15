"""Mission spec compilation: frames, geometry, and the reasons a mission is refused."""

import copy
import math
import unittest
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.grid import world_to_cell
from ergodic_control_mppi.deploy.grid import path_blocked
from ergodic_control_mppi.deploy.mission import (
    EARTH_RADIUS_M,
    command_vectors,
    compile_flight,
    compile_mission,
    dry_run_failure,
    flight_step,
    ned_from_wgs84,
    plan_blocked,
    start_failure,
    unpack_flight,
)
from ergodic_control_mppi.mppi.single import initialize_single, measured_step
from ergodic_control_mppi.simulation import controller_key

REFERENCE = (44.63, 10.95)
DEGREES_PER_METRE = math.degrees(1.0 / EARTH_RADIUS_M)


def at(north: float, east: float) -> list[float]:
    """``[lat, lon]`` of a point ``north``, ``east`` metres from ``REFERENCE``."""
    return [
        REFERENCE[0] + north * DEGREES_PER_METRE,
        REFERENCE[1] + east * DEGREES_PER_METRE / math.cos(math.radians(REFERENCE[0])),
    ]


SPEC = {
    "schema": "ergodic/1",
    "mission_id": "test",
    "command_mode": "velocity",
    "duration_s": 60,
    "vehicle": {
        "radius_m": 0.45,
        "clearance_m": 0.15,
        "tracking_allowance_m": 0.2,
        "max_speed_mps": 1.5,
        "max_accel_mps2": 3.0,
        "brake_accel_mps2": 3.0,
        "reaction_time_s": 0.1,
    },
    "area": [at(-10, -10), at(10, -10), at(10, 10), at(-10, 10)],
    "obstacles": [],
    "density": [
        {"mean": at(5, 5), "sigma_major_m": 3, "sigma_minor_m": 1, "bearing_deg": 0, "weight": 2},
        {"mean": at(-5, -5), "sigma_major_m": 2, "sigma_minor_m": 2, "bearing_deg": 45, "weight": 1},
    ],
}


def spec(**changes) -> dict:
    document = copy.deepcopy(SPEC)
    document.update(changes)
    return document


class ProjectionTest(unittest.TestCase):
    def test_reference_is_the_origin(self):
        np.testing.assert_allclose(ned_from_wgs84(np.array([REFERENCE]), REFERENCE), [[0.0, 0.0]])

    def test_north_and_east_offsets_land_on_their_axes(self):
        ned = ned_from_wgs84(np.array([at(500, 0), at(0, 500)]), REFERENCE)
        np.testing.assert_allclose(ned, [[500.0, 0.0], [0.0, 500.0]], atol=0.05)


class CompileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_config("configs/uav_profile.yaml").controller

    def compile(self, document: dict, start=(0.0, 0.0)):
        return compile_mission(document, self.base, REFERENCE, start)

    def blocked(self, mission, north: float, east: float) -> bool:
        row, column = world_to_cell(np.array([north, east]), mission.origin, mission.resolution)
        return bool(mission.grid[row, column])

    def test_valid_spec_becomes_a_ned_workspace(self):
        mission = self.compile(spec())
        workspace = mission.params.workspace
        np.testing.assert_allclose(workspace.x_limits, [-10.0, 10.0], atol=0.05)
        np.testing.assert_allclose(workspace.y_limits, [-10.0, 10.0], atol=0.05)
        np.testing.assert_allclose(mission.params.gmm.means, [[5.0, 5.0], [-5.0, -5.0]], atol=0.05)
        np.testing.assert_allclose(np.exp(mission.params.gmm.log_weights), [2 / 3, 1 / 3], rtol=1e-5)
        self.assertEqual(workspace.grid.shape, mission.grid.shape)
        self.assertFalse(self.blocked(mission, 0.0, 0.0))
        # The area edge sits inside the safety margin.
        self.assertTrue(self.blocked(mission, 9.9, 0.0))

    def test_bearing_orients_the_major_axis(self):
        north = self.compile(spec()).params.gmm.covariance[0]
        np.testing.assert_allclose(north, [[9.0, 0.0], [0.0, 1.0]], atol=1e-4)
        turned = spec()
        turned["density"][0]["bearing_deg"] = 90
        east = self.compile(turned).params.gmm.covariance[0]
        np.testing.assert_allclose(east, [[1.0, 0.0], [0.0, 9.0]], atol=1e-4)

    def test_vehicle_limits_reach_the_controller(self):
        params = self.compile(spec()).params
        self.assertAlmostEqual(params.field.reference_speed * params.field.transit_speedup, 1.5)
        self.assertEqual(params.model.max_accel_lin_abs, 3.0)

    def test_circle_and_polygon_obstacles_are_rasterized(self):
        mission = self.compile(spec(obstacles=[
            {"type": "circle", "center": at(0, 5), "radius_m": 1.0},
            {"type": "polygon", "points": [at(-1, -8), at(1, -8), at(0, -6)]},
        ]))
        self.assertTrue(self.blocked(mission, 0.0, 5.0))
        self.assertTrue(self.blocked(mission, 0.0, -7.0))
        self.assertFalse(self.blocked(mission, 0.0, 0.0))

    def test_mode_inside_an_obstacle_is_refused(self):
        document = spec(obstacles=[{"type": "circle", "center": at(5, 5), "radius_m": 1.0}])
        with self.assertRaisesRegex(ValueError, r"density modes \[0\] sit inside"):
            self.compile(document)

    def test_mode_behind_a_wall_is_refused(self):
        wall = {"type": "polygon", "points": [at(2, -12), at(3, -12), at(3, 12), at(2, 12)]}
        with self.assertRaisesRegex(ValueError, r"density modes \[0\] are cut off"):
            self.compile(spec(obstacles=[wall]))

    def test_drone_outside_the_area_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not in the free part"):
            self.compile(spec(), start=(50.0, 0.0))

    def test_malformed_specs_are_refused(self):
        cases = {
            "unknown keys": spec(extra=1),
            "schema": spec(schema="ergodic/0"),
            "command_mode": spec(command_mode="jerk"),
            "duration_s must be positive": spec(duration_s=0),
            "at least 3": spec(area=[at(0, 0), at(1, 1)]),
            "type must be": spec(obstacles=[{"type": "cone"}]),
        }
        for reason, document in cases.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.compile(document)

    def test_oversized_area_is_refused(self):
        huge = spec(area=[at(-500, -500), at(500, -500), at(500, 500), at(-500, 500)])
        with self.assertRaisesRegex(ValueError, "cells"):
            self.compile(huge)


class FlightRulesTest(unittest.TestCase):
    """The per-tick rules the node applies: start gate and command shaping."""

    @classmethod
    def setUpClass(cls):
        base = load_config("configs/uav_profile.yaml").controller
        cls.mission = compile_mission(spec(), base, REFERENCE, (0.0, 0.0))

    def test_start_gate(self):
        self.assertIsNone(start_failure(self.mission, (0.0, 0.0), 0.02, False))
        self.assertIn("fresh", start_failure(self.mission, (0.0, 0.0), 0.5, False))
        self.assertIn("EKF origin", start_failure(self.mission, (0.0, 0.0), 0.02, True))
        self.assertIn("safety margin", start_failure(self.mission, (9.9, 0.0), 0.02, False))
        self.assertIn("safety margin", start_failure(self.mission, (40.0, 0.0), 0.02, False))

    def test_velocity_mode_clamps_speed_and_keeps_feedforward(self):
        planned = np.array([0.0, 0.0, 3.0, 4.0, 0.0, 0.0])
        velocity, acceleration = command_vectors(self.mission, planned, np.array([0.5, -0.2, 0.0]), (0, 0))
        np.testing.assert_allclose(velocity[:2], [0.9, 1.2], rtol=1e-6)  # 5 m/s scaled to 1.5
        np.testing.assert_allclose(acceleration[:2], [0.5, -0.2], rtol=1e-6)
        self.assertTrue(np.isnan(velocity[2]) and np.isnan(acceleration[2]))

    def test_acceleration_mode_never_speeds_up_past_the_limit(self):
        mission = self.mission._replace(command_mode="acceleration")
        velocity, acceleration = command_vectors(
            mission, np.zeros(6), np.array([1.0, 1.0, 0.0]), (1.5, 0.0)
        )
        self.assertTrue(np.isnan(velocity).all())
        np.testing.assert_allclose(acceleration[:2], [0.0, 1.0], atol=1e-6)
        # Below the limit, and when braking, the acceleration passes through.
        _, slow = command_vectors(mission, np.zeros(6), np.array([1.0, 1.0, 0.0]), (1.0, 0.0))
        _, brake = command_vectors(mission, np.zeros(6), np.array([-1.0, 0.0, 0.0]), (1.5, 0.0))
        np.testing.assert_allclose(slow[:2], [1.0, 1.0])
        np.testing.assert_allclose(brake[:2], [-1.0, 0.0])
        # Above the limit the cap brakes even when the plan asks for nothing.
        _, over = command_vectors(mission, np.zeros(6), np.zeros(3), (2.0, 0.0))
        np.testing.assert_allclose(over[:2], [-1.0, 0.0], atol=1e-6)

    def test_velocity_feedforward_is_capped_at_the_limit(self):
        planned = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
        _, feedforward = command_vectors(self.mission, planned, np.array([1.0, 0.5, 0.0]), (1.5, 0.0))
        np.testing.assert_allclose(feedforward[:2], [0.0, 0.5], atol=1e-6)


class DryRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = load_config("configs/uav_profile.yaml").controller
        base = replace(base, mppi=replace(base.mppi, samples=8, horizon=4))
        cls.mission = compile_mission(spec(), base, REFERENCE, (0.0, 0.0))

    def test_clean_flight_passes(self):
        state = jnp.zeros(6, dtype=jnp.float32)
        self.assertIsNone(dry_run_failure(self.mission, state, controller_key(0), 0.5))

    def test_device_placed_parameters_run(self):
        """The node dry-runs parameters after device_put, which turns floats into arrays."""
        device = jax.devices("cpu")[0]
        mission = self.mission._replace(params=jax.device_put(self.mission.params, device))
        state = jax.device_put(jnp.zeros(6, dtype=jnp.float32), device)
        self.assertIsNone(dry_run_failure(mission, state, controller_key(0), 0.1))

    def test_start_inside_a_margin_fails(self):
        state = jnp.asarray([9.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        reason = dry_run_failure(self.mission, state, controller_key(0), 0.1)
        self.assertIn("safety margin", reason)

    def test_flight_step_packs_what_the_tick_reads(self):
        params = self.mission.params
        state = jnp.asarray([1.0, -2.0, 0.3, 0.1, 0.0, 0.0], dtype=jnp.float32)
        carry = initialize_single(params, state, jnp.zeros((params.mppi.horizon, 3)), controller_key(0))
        expected_carry, expected = measured_step(params, carry, state)
        _, packed = jax.jit(flight_step, static_argnames="plan_steps")(params, carry, state, plan_steps=3)
        plan, next_state, control = unpack_flight(np.asarray(packed), 3)
        np.testing.assert_allclose(plan, expected.optimal_trajectory[:3, :2], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(next_state, expected_carry.state, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(control, expected.control, rtol=1e-5, atol=1e-6)

    def test_flight_never_recompiles_after_the_preflight(self):
        """A mid-flight compile stalled the Orin's first tick for 3.4 s."""
        device = jax.devices("cpu")[0]
        params = jax.device_put(self.mission.params, device)
        state = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        zeros = jnp.zeros((params.mppi.horizon, 3), dtype=jnp.float32)
        carry = jax.device_put(initialize_single(params, jnp.asarray(state), zeros, controller_key(0)), device)
        step = compile_flight(params, carry, state, plan_steps=3)
        compiled = step._cache_size()
        # The node's sequence: fresh start carry, then measurements and predictions interleaved.
        start = jax.device_put(initialize_single(params, jnp.asarray(state), zeros, controller_key(0)), device)
        carry, _ = step(params, start, state)
        for index in range(4):
            carry, _ = step(params, carry, state + 0.1 * index if index % 2 else carry.state)
        self.assertEqual(step._cache_size(), compiled)

    def test_plan_check_matches_the_segment_exact_check(self):
        dense = np.column_stack((np.linspace(-2.0, 2.0, 60), np.zeros(60)))
        into_margin = np.column_stack((np.linspace(8.0, 9.95, 60), np.zeros(60)))
        sparse = np.array([[0.0, -9.0], [0.0, 9.0], [0.0, 12.0]])
        for path in (dense, into_margin, sparse):
            with self.subTest(end=path[-1].tolist()):
                self.assertEqual(
                    plan_blocked(self.mission, path),
                    path_blocked(self.mission.grid, self.mission.origin, self.mission.resolution, path),
                )
        self.assertFalse(plan_blocked(self.mission, dense))
        self.assertTrue(plan_blocked(self.mission, into_margin))

    def test_non_finite_state_fails(self):
        state = jnp.full(6, jnp.nan, dtype=jnp.float32)
        self.assertIn("non-finite", dry_run_failure(self.mission, state, controller_key(0), 0.1))


if __name__ == "__main__":
    unittest.main()
