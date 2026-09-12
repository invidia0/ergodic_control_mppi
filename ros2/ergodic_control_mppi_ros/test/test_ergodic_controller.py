"""Regressions for the controller node's message/state conversions."""

import math
import unittest

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry

from ergodic_control_mppi_ros.ergodic_controller import (
    grid_from,
    limit_yaw_rate,
    observation_from,
    state_trace_u32,
    yaw_of,
)


def _odometry(x, y, vx, vy, yaw, yaw_rate) -> Odometry:
    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    message.twist.twist.linear.x = vx
    message.twist.twist.linear.y = vy
    message.twist.twist.angular.z = yaw_rate
    return message


class ObservationTest(unittest.TestCase):
    def test_six_state_order_matches_the_controller_contract(self):
        state = observation_from(_odometry(1.0, -2.0, 0.3, -0.4, 0.5, 0.6))
        self.assertEqual(state.shape, (6,))
        np.testing.assert_allclose(state, [1.0, -2.0, 0.3, -0.4, 0.5, 0.6], atol=1e-6)

    def test_yaw_recovers_across_the_full_circle(self):
        for yaw in (-3.0, -1.0, 0.0, 1.0, 3.0):
            message = _odometry(0, 0, 0, 0, yaw, 0)
            self.assertAlmostEqual(yaw_of(message.pose.pose.orientation), yaw, places=6)

    def test_state_trace_preserves_float32_bits(self):
        predicted = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf], dtype=np.float32)
        observed = np.nextafter(predicted, np.float32(np.inf), dtype=np.float32)
        values = np.fromstring(state_trace_u32(7, predicted, observed), dtype=np.uint32, sep=",")
        self.assertEqual(int(values[0]), 7)
        np.testing.assert_array_equal(values[1:7], predicted.view(np.uint32))
        np.testing.assert_array_equal(values[7:], observed.view(np.uint32))


class YawRateLimitTest(unittest.TestCase):
    def test_small_error_is_reached_in_one_step(self):
        result = limit_yaw_rate(0.0, 0.01, max_rate=math.pi, delta_t=0.02)
        self.assertAlmostEqual(result, 0.01, places=9)

    def test_large_error_is_clamped_to_the_rate(self):
        result = limit_yaw_rate(0.0, 3.0, max_rate=math.pi, delta_t=0.02)
        self.assertAlmostEqual(result, math.pi * 0.02, places=9)

    def test_shortest_way_around_the_wrap(self):
        """From +3.0 to -3.0 the short way is forward through pi, not back through zero."""
        result = limit_yaw_rate(3.0, -3.0, max_rate=math.pi, delta_t=0.02)
        self.assertAlmostEqual(result, 3.0 + math.pi * 0.02, places=9)

    def test_wrap_converges_to_the_target(self):
        """Stepping repeatedly across the wrap must arrive, not oscillate."""
        yaw = 3.0
        for _ in range(20):
            yaw = limit_yaw_rate(yaw, -3.0, max_rate=math.pi, delta_t=0.02)
        error = math.atan2(math.sin(-3.0 - yaw), math.cos(-3.0 - yaw))
        self.assertLess(abs(error), 1e-9)

    def test_result_stays_wrapped(self):
        for target in (-6.0, -3.0, 0.0, 3.0, 6.0):
            result = limit_yaw_rate(3.1, target, max_rate=100.0, delta_t=0.02)
            self.assertTrue(-math.pi - 1e-9 <= result <= math.pi + 1e-9)


class GridFromTest(unittest.TestCase):
    def test_unpacks_occupancy_origin_and_resolution(self):
        message = OccupancyGrid()
        message.info.width = 3
        message.info.height = 2
        message.info.resolution = 0.15
        message.info.origin.position.x = -20.0
        message.info.origin.position.y = -10.0
        message.data = [0, 100, 0, 0, 0, 100]

        occupancy, origin, resolution = grid_from(message)
        self.assertEqual(occupancy.shape, (2, 3))
        self.assertEqual(occupancy.dtype, np.float32)
        np.testing.assert_array_equal(occupancy, [[0, 1, 0], [0, 0, 1]])
        self.assertEqual(origin, (-20.0, -10.0))
        self.assertAlmostEqual(resolution, 0.15)


if __name__ == "__main__":
    unittest.main()
