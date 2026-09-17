"""Mission spec compilation: frames, geometry, and the reasons a mission is refused."""

import copy
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.grid import path_blocked, world_to_cell
from ergodic_control_mppi.deploy.mission import (
    COMMAND_MODES,
    EARTH_RADIUS_M,
    altitude_failure,
    command_position,
    command_vectors,
    compile_flight,
    compile_mission,
    dimension,
    dry_run_failure,
    flight_step,
    ned_from_wgs84,
    plan_blocked,
    replans_from_measurement,
    start_failure,
    unpack_flight,
)
from ergodic_control_mppi.mppi.core import stage_cost
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
    "schema": "ergodic/3",
    "mission_id": "test",
    "command_mode": "velocity",
    "duration_s": 60,
    "max_altitude_m": 12,
    "vehicle": {
        "radius_m": 0.45,
        "clearance_m": 0.15,
        "tracking_allowance_m": 0.2,
        "max_speed_mps": 1.5,
        "max_accel_mps2": 3.0,
        "brake_accel_mps2": 3.0,
        "reaction_time_s": 0.1,
        "max_vertical_speed_mps": 1.0,
    },
    "area": [at(-10, -10), at(10, -10), at(10, 10), at(-10, 10)],
    "obstacles": [],
    "density": [
        {"mean": [*at(5, 5), 6], "sigma_major_m": 3, "sigma_minor_m": 1, "sigma_vertical_m": 1,
         "bearing_deg": 0, "pitch_deg": 0, "roll_deg": 0, "weight": 2},
        {"mean": [*at(-5, -5), 6], "sigma_major_m": 2, "sigma_minor_m": 2, "sigma_vertical_m": 1,
         "bearing_deg": 45, "pitch_deg": 0, "roll_deg": 0, "weight": 1},
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

    def compile(self, document: dict):
        return compile_mission(document, self.base, REFERENCE)

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
        # A planar mission flies the footprint: pitching the major axis up shrinks it to minor.
        pitched = spec()
        pitched["density"][0]["pitch_deg"] = 90
        footprint = self.compile(pitched).params.gmm.covariance[0]
        np.testing.assert_allclose(footprint, [[1.0, 0.0], [0.0, 1.0]], atol=1e-4)

    def test_vehicle_limits_reach_the_controller(self):
        params = self.compile(spec()).params
        self.assertAlmostEqual(params.field.reference_speed * params.field.transit_speedup, 1.5)
        self.assertEqual(params.model.max_accel_lin_abs, 3.0)

    def test_circle_and_polygon_obstacles_are_rasterized(self):
        mission = self.compile(spec(obstacles=[
            {"type": "circle", "center": at(0, 5), "radius_m": 1.0, "height_m": 2},
            {"type": "polygon", "points": [at(-1, -8), at(1, -8), at(0, -6)], "height_m": 2},
        ]))
        self.assertTrue(self.blocked(mission, 0.0, 5.0))
        self.assertTrue(self.blocked(mission, 0.0, -7.0))
        self.assertFalse(self.blocked(mission, 0.0, 0.0))

    def test_mode_inside_an_obstacle_is_refused(self):
        document = spec(obstacles=[{"type": "circle", "center": at(5, 5), "radius_m": 1.0, "height_m": 2}])
        with self.assertRaisesRegex(ValueError, r"density modes \[0\] sit inside"):
            self.compile(document)

    def test_mode_behind_a_wall_is_refused(self):
        wall = {"type": "polygon", "points": [at(2, -12), at(3, -12), at(3, 12), at(2, 12)], "height_m": 2}
        with self.assertRaisesRegex(ValueError, r"density modes \[1\] are cut off from the heaviest mode"):
            self.compile(spec(obstacles=[wall]))

    def test_planar_modes_block_every_obstacle_at_all_altitudes(self):
        """A 1 m obstacle is still a wall to a planar mission: it cannot change altitude."""
        low = {"type": "circle", "center": at(0, 5), "radius_m": 1.0, "height_m": 1}
        self.assertTrue(self.blocked(self.compile(spec(obstacles=[low])), 0.0, 5.0))

    def test_heaviest_mode_is_the_dry_run_start(self):
        mission = self.compile(spec())
        np.testing.assert_allclose(mission.dry_run_start, [5.0, 5.0], atol=0.05)
        self.assertTrue(mission.reachable[tuple(world_to_cell(np.array([0.0, 0.0]), mission.origin, mission.resolution))])

    def test_malformed_specs_are_refused(self):
        cases = {
            "unknown keys": spec(extra=1),
            "schema": spec(schema="ergodic/0"),
            "command_mode": spec(command_mode="jerk"),
            "duration_s must be positive": spec(duration_s=0),
            "at least 3": spec(area=[at(0, 0), at(1, 1)]),
            "type must be": spec(obstacles=[{"type": "cone"}]),
            "crosses itself": spec(area=[at(-10, -10), at(10, 10), at(10, -10), at(-10, 10)]),
            "height_m must be positive": spec(
                obstacles=[{"type": "circle", "center": at(0, 5), "radius_m": 1.0, "height_m": 0}]
            ),
            "max_altitude_m must be positive": spec(max_altitude_m=0),
        }
        density = copy.deepcopy(SPEC["density"])
        density[0]["mean"] = [*at(5, 5), 20]
        cases["above the 12 m ceiling"] = spec(density=density)
        density = copy.deepcopy(SPEC["density"])
        density[0]["mean"] = at(5, 5)
        cases[r"mean must be \[lat, lon, altitude_m\]"] = spec(density=density)
        density = copy.deepcopy(SPEC["density"])
        density[0]["sigma_vertical_m"] = -1
        cases["sigma_vertical_m must be positive"] = spec(density=density)
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
        cls.mission = compile_mission(spec(), base, REFERENCE)

    def test_start_gate(self):
        self.assertIsNone(start_failure(self.mission, (0.0, 0.0), 2.0, 0.02, False))
        self.assertIn("fresh", start_failure(self.mission, (0.0, 0.0), 2.0, 0.5, False))
        self.assertIn("EKF origin", start_failure(self.mission, (0.0, 0.0), 2.0, 0.02, True))
        self.assertIn("safety margin", start_failure(self.mission, (9.9, 0.0), 2.0, 0.02, False))
        self.assertIn("inside the area", start_failure(self.mission, (40.0, 0.0), 2.0, 0.02, False))
        self.assertIn("ceiling", start_failure(self.mission, (0.0, 0.0), 12.5, 0.02, False))

    def test_start_gate_refuses_a_pocket_cut_off_from_the_modes(self):
        """Free but walled off from every mode: the drone could never reach the density."""
        base = load_config("configs/uav_profile.yaml").controller
        wall = {"type": "polygon", "points": [at(-3, -12), at(-2, -12), at(-2, 12), at(-3, 12)], "height_m": 2}
        density = [dict(SPEC["density"][0])]
        mission = compile_mission(spec(obstacles=[wall], density=density), base, REFERENCE)
        self.assertIsNone(start_failure(mission, (3.0, 0.0), 2.0, 0.02, False))
        self.assertIn("connected", start_failure(mission, (-7.0, 0.0), 2.0, 0.02, False))

    def test_altitude_failure_is_the_ceiling(self):
        self.assertIsNone(altitude_failure(self.mission, 12.0))
        self.assertIn("12.0 m mission ceiling", altitude_failure(self.mission, 12.1))

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

    def test_trajectory_mode_sends_the_planned_position_with_velocity_feedforward(self):
        trajectory = self.mission._replace(command_mode="trajectory")
        planned = np.array([1.0, -2.0, 3.0, 4.0, 0.0, 0.0])
        np.testing.assert_allclose(command_position(trajectory, planned)[:2], [1.0, -2.0])
        self.assertTrue(np.isnan(command_position(trajectory, planned)[2]))
        velocity, _ = command_vectors(trajectory, planned, np.zeros(3), (0, 0))
        np.testing.assert_allclose(velocity[:2], [0.9, 1.2], rtol=1e-6)
        self.assertTrue(np.isnan(command_position(self.mission, planned)).all())  # velocity mode

    def test_trajectory_mode_replans_from_its_reference_until_the_drone_strays(self):
        trajectory = self.mission._replace(command_mode="trajectory")  # 0.2 m tracking allowance
        reference = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        near = np.array([1.1, 1.1, 0.3, 0.0, 0.0, 0.0])  # 0.14 m off
        far = np.array([1.2, 1.2, 0.3, 0.0, 0.0, 0.0])   # 0.28 m off
        self.assertFalse(replans_from_measurement(trajectory, reference, near))
        self.assertTrue(replans_from_measurement(trajectory, reference, far))
        self.assertTrue(replans_from_measurement(self.mission, reference, near))  # velocity mode


class DryRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = load_config("configs/uav_profile.yaml").controller
        base = replace(base, mppi=replace(base.mppi, samples=8, horizon=4))
        cls.mission = compile_mission(spec(), base, REFERENCE)

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
        plan, next_state, control = unpack_flight(np.asarray(packed), 3, 2)
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
                    path_blocked(self.mission.guard_grid, self.mission.origin, self.mission.resolution, path),
                )
        self.assertFalse(plan_blocked(self.mission, dense))
        self.assertTrue(plan_blocked(self.mission, into_margin))

    def test_the_plan_check_allows_the_buffer_but_not_the_core(self):
        # Area edge at 10 m. Margin 1.43 m: radius 0.45 + clearance 0.15 + cell 0.11 (the core,
        # 0.71 m, rounded to 0.75 m) + tracking 0.2 + stopping 0.525.
        mission = self.mission
        self.assertAlmostEqual(mission.margin, 1.431, places=3)
        in_buffer = np.column_stack((np.linspace(8.0, 8.9, 60), np.zeros(60)))
        in_core = np.column_stack((np.linspace(8.0, 9.4, 60), np.zeros(60)))
        self.assertTrue(path_blocked(mission.grid, mission.origin, mission.resolution, in_buffer))
        self.assertFalse(plan_blocked(mission, in_buffer))
        self.assertTrue(plan_blocked(mission, in_core))
        self.assertTrue((mission.guard_grid <= mission.grid).all())  # the core never exceeds the margin

    def test_non_finite_state_fails(self):
        state = jnp.full(6, jnp.nan, dtype=jnp.float32)
        self.assertIn("non-finite", dry_run_failure(self.mission, state, controller_key(0), 0.1))


class VolumetricTest(unittest.TestCase):
    """The ``_3d`` modes: a voxel grid, fly-over, the altitude band and 3D commands."""

    SHORT = {"type": "circle", "center": at(0, 5), "radius_m": 1.0, "height_m": 3}
    TALL = {"type": "circle", "center": at(0, -5), "radius_m": 1.0, "height_m": 30}

    @classmethod
    def setUpClass(cls):
        base = load_config("configs/uav_profile.yaml").controller
        cls.base = replace(base, mppi=replace(base.mppi, samples=8, horizon=4))
        cls.mission = compile_mission(
            spec(command_mode="velocity_3d", obstacles=[cls.SHORT, cls.TALL]), cls.base, REFERENCE
        )

    def blocked(self, north: float, east: float, altitude: float) -> bool:
        cell = world_to_cell(np.array([north, east, -altitude]), self.mission.origin, self.mission.resolution)
        return bool(self.mission.grid[tuple(cell)])

    def test_the_density_is_a_ned_3d_mixture(self):
        gmm = self.mission.params.gmm
        np.testing.assert_allclose(gmm.means, [[5.0, 5.0, -6.0], [-5.0, -5.0, -6.0]], atol=0.05)
        np.testing.assert_allclose(gmm.covariance[0], np.diag([9.0, 1.0, 1.0]), atol=1e-4)
        self.assertEqual(self.mission.params.mppi.covariance.shape, (4, 4))
        np.testing.assert_allclose(self.mission.params.workspace.extra_limits, [[-12.0, 0.0]])

    def test_pitch_and_roll_tilt_the_blob_as_the_paddock_draws_it(self):
        # Body (major 3, minor 1, up 1), NED z down. Pitch 90 lifts major onto up; roll 90
        # lifts minor onto up. A 45 degree pitch leans major to the north and up (NED: z < 0).
        def covariance(pitch, roll):
            density = [{**SPEC["density"][0], "pitch_deg": pitch, "roll_deg": roll, "sigma_vertical_m": 2}]
            mission = compile_mission(spec(command_mode="velocity_3d", density=density), self.base, REFERENCE)
            return np.asarray(mission.params.gmm.covariance[0])

        np.testing.assert_allclose(covariance(90, 0), np.diag([4.0, 1.0, 9.0]), atol=1e-4)
        np.testing.assert_allclose(covariance(0, 90), np.diag([9.0, 4.0, 1.0]), atol=1e-4)
        leaning = covariance(45, 0)
        np.testing.assert_allclose(leaning[0, 0], leaning[2, 2], rtol=1e-5)
        self.assertLess(leaning[0, 2], 0.0)

    def test_short_obstacles_are_flown_over_and_tall_ones_are_not(self):
        self.assertTrue(self.blocked(0.0, 5.0, 2.0))
        self.assertFalse(self.blocked(0.0, 5.0, 7.0))
        self.assertTrue(self.blocked(0.0, -5.0, 7.0))
        self.assertFalse(self.blocked(0.0, 0.0, 7.0))

    def test_ground_and_ceiling_are_occupied(self):
        self.assertTrue(self.blocked(0.0, 0.0, 0.5))
        self.assertTrue(self.blocked(0.0, 0.0, 11.5))
        self.assertTrue(self.blocked(0.0, 0.0, 13.0))

    def test_a_mode_above_a_short_obstacle_is_reachable_only_in_3d(self):
        above = [{**SPEC["density"][0], "mean": [*at(0, 5), 8]}]
        compile_mission(
            spec(command_mode="velocity_3d", obstacles=[self.SHORT], density=above), self.base, REFERENCE
        )
        with self.assertRaisesRegex(ValueError, "sit inside"):
            compile_mission(spec(obstacles=[self.SHORT], density=above), self.base, REFERENCE)

    def test_a_ceiling_with_no_room_between_the_margins_is_refused(self):
        low = [{**component, "mean": [*component["mean"][:2], 2]} for component in SPEC["density"]]
        with self.assertRaisesRegex(ValueError, "max_altitude_m must exceed"):
            compile_mission(
                spec(command_mode="velocity_3d", max_altitude_m=3, density=low), self.base, REFERENCE
            )

    def test_start_gate_uses_the_voxel(self):
        self.assertIsNone(start_failure(self.mission, (0.0, 0.0, -6.0), 6.0, 0.02, False))
        self.assertIn("safety margin", start_failure(self.mission, (0.0, 5.0, -2.0), 2.0, 0.02, False))

    def test_the_controller_charges_the_voxels(self):
        beside = np.zeros((2, 8), np.float32)
        beside[:, :3] = [[0.0, 5.0, -2.0], [0.0, 5.0, -7.0]]
        costs = np.asarray(stage_cost(jnp.asarray(beside), self.mission.params))
        self.assertGreaterEqual(costs[0] - costs[1], 0.99 * float(self.mission.params.workspace.obstacle_cost))

    def test_commands_carry_z(self):
        planned = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0])
        velocity, acceleration = command_vectors(
            self.mission, planned, np.array([0.5, 0.0, -0.2, 0.0]), (0.0, 0.0, 0.0)
        )
        # Norm-capped to 1.5 m/s, then vz clipped to the 1.0 m/s vertical limit.
        np.testing.assert_allclose(velocity, [0.9, 0.0, 1.0], rtol=1e-6)
        np.testing.assert_allclose(acceleration, [0.5, 0.0, -0.2], rtol=1e-6)
        climbing = self.mission._replace(command_mode="acceleration_3d")
        velocity, capped = command_vectors(climbing, planned, np.array([0.0, 0.0, -1.0, 0.0]), (0.0, 0.0, -1.5))
        self.assertTrue(np.isnan(velocity).all())
        # At the speed limit the norm cap removes the climb; 0.5 m/s over the vertical limit
        # then brakes the climb.
        np.testing.assert_allclose(capped, [0.0, 0.0, 1.0], atol=1e-6)
        _, level = command_vectors(climbing, planned, np.array([0.0, 0.0, -1.0, 0.0]), (0.0, 0.0, -0.9))
        np.testing.assert_allclose(level, [0.0, 0.0, -0.2], atol=1e-6)

    def test_model_flight_and_packed_step_run_in_3d(self):
        state = np.array([*self.mission.dry_run_start, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertIsNone(dry_run_failure(self.mission, jnp.asarray(state), controller_key(0), 0.5))
        params = self.mission.params
        carry = initialize_single(params, jnp.asarray(state), jnp.zeros((params.mppi.horizon, 4)), controller_key(0))
        step = compile_flight(params, carry, state, plan_steps=3)
        _, packed = step(params, carry, state)
        plan, next_state, control = unpack_flight(np.asarray(packed), 3, 3)
        self.assertEqual((plan.shape, next_state.shape, control.shape), ((3, 3), (8,), (4,)))
        self.assertFalse(plan_blocked(self.mission, plan))


class PaddockFixtureTest(unittest.TestCase):
    """The spec the Paddock writes for the field scene (its ergodicDraft test holds the same file)."""

    def test_every_mode_compiles_the_paddock_field_scene(self):
        document = json.loads((Path(__file__).parent / "fixtures" / "paddock_3d_spec.json").read_text())
        base = load_config("configs/uav_profile.yaml").controller
        for mode in COMMAND_MODES:
            with self.subTest(mode=mode):
                mission = compile_mission({**document, "command_mode": mode}, base, REFERENCE)
                d = dimension(mode)
                self.assertEqual(mission.params.gmm.means.shape, (3, d))
                self.assertEqual(mission.max_vertical_speed, 0.5)
                # The takeoff point at 3 m is a legal start.
                self.assertIsNone(start_failure(mission, (0.0, 0.0, -3.0)[:d], 3.0, 0.02, False))
        three_d = compile_mission(document, base, REFERENCE)
        # 2.2-2.6 m pillars are flown over at 5 m; the 8 m ones are not.
        for (north, east), free in (((-8.0, 2.5), True), ((16.0, 5.0), True), ((8.0, 2.5), False), ((4.5, -2.0), False)):
            cell = world_to_cell(np.array([north, east, -5.0]), three_d.origin, three_d.resolution)
            self.assertEqual(bool(three_d.grid[tuple(cell)]), not free, (north, east))


if __name__ == "__main__":
    unittest.main()
