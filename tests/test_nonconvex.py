"""A non-convex environment: the controller must go around a V notch, never across it.

The area's outside is occupied like any obstacle, so a notch between two density modes puts
blocked space on the straight line between them. Slow (a minute of closed loop on the CPU), so
it runs with the full suite at image build, not in the node's boot self-test.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.deploy.grid import world_to_cell
from ergodic_control_mppi.deploy.mission import EARTH_RADIUS_M, compile_mission
from ergodic_control_mppi.mppi.single import run_single
from ergodic_control_mppi.simulation import controller_key

REFERENCE = (44.63, 10.95)
DEGREES_PER_METRE = math.degrees(1.0 / EARTH_RADIUS_M)
# A 30 m notch from the north edge down to north = -10, leaving a 10 m passage below its tip.
NOTCH = [(-20, -25), (20, -25), (20, -2), (-10, 0), (20, 2), (20, 25), (-20, 25)]
SECONDS = 60.0


def at(north: float, east: float) -> list[float]:
    return [
        REFERENCE[0] + north * DEGREES_PER_METRE,
        REFERENCE[1] + east * DEGREES_PER_METRE / math.cos(math.radians(REFERENCE[0])),
    ]


class NonConvexAreaTest(unittest.TestCase):
    def test_modes_across_a_notch_are_reached_around_its_tip(self):
        mode = {"sigma_major_m": 3, "sigma_minor_m": 3, "sigma_vertical_m": 1, "bearing_deg": 0,
                "pitch_deg": 0, "roll_deg": 0, "weight": 1}
        spec = {
            "schema": "ergodic/3", "mission_id": "notch", "command_mode": "velocity", "duration_s": SECONDS,
            "max_altitude_m": 12,
            "vehicle": {"radius_m": 0.45, "clearance_m": 0.15, "tracking_allowance_m": 0.2,
                        "max_speed_mps": 1.5, "max_accel_mps2": 3.0, "brake_accel_mps2": 3.0,
                        "reaction_time_s": 0.1, "max_vertical_speed_mps": 0.5},
            "area": [at(north, east) for north, east in NOTCH],
            "obstacles": [],
            "density": [{**mode, "mean": [*at(12, -12), 5]}, {**mode, "mean": [*at(12, 12), 5]}],
        }
        mission = compile_mission(spec, load_config("configs/uav_profile.yaml").controller, REFERENCE)
        params = mission.params
        start = jnp.asarray([12.0, -12.0, 0.0, 0.0, 0.0, 0.0], jnp.float32)
        steps = round(SECONDS / float(params.model.delta_t))
        path = np.asarray(
            jax.jit(run_single, static_argnames="steps")(
                params, start, jnp.zeros((params.mppi.horizon, 3)), controller_key(0), steps=steps
            ).path
        )

        rows, columns = world_to_cell(path[:, :2], mission.origin, mission.resolution).T
        height, width = mission.grid.shape
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        blocked = ~inside | mission.grid[np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1)]
        self.assertFalse(blocked.any(), "the flight entered a safety margin or left the area")
        east = path[:, 1] > 0
        self.assertTrue(east.any(), "never reached the mode across the notch")
        crossing = path[max(0, int(np.argmax(east)) - 50): int(np.argmax(east)) + 50]
        self.assertLess(crossing[:, 0].min(), -10.0, "crossed without going around the notch tip")
        self.assertGreater(east.mean(), 0.25, "stayed in one arm")


if __name__ == "__main__":
    unittest.main()
