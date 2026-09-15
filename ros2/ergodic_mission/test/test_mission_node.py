"""The node's own glue: name joining, PX4 topic names, self-test parsing, and the deadline gate."""

import time
import unittest
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from ergodic_mission.mission_node import (
    WARMUP_MIN_STEPS,
    failed_tests,
    join_path,
    position_problem,
    px4_topic,
    warmup_p99,
)


class Carry(NamedTuple):
    state: np.ndarray


class JoinPathTest(unittest.TestCase):
    def test_namespaces(self):
        self.assertEqual(join_path("", "command"), "/command")
        self.assertEqual(join_path("/", "command"), "/command")
        self.assertEqual(join_path("/px4_1", "command"), "/px4_1/command")
        self.assertEqual(join_path("/px4_1/", "mission/ergodic/load"), "/px4_1/mission/ergodic/load")


class Px4TopicTest(unittest.TestCase):
    def test_versioned_messages_get_the_suffix(self):
        v1 = SimpleNamespace(MESSAGE_VERSION=1)
        v0 = SimpleNamespace(MESSAGE_VERSION=0)
        unversioned = SimpleNamespace()
        self.assertEqual(px4_topic("/uav_1/fmu/out/vehicle_local_position", v1),
                         "/uav_1/fmu/out/vehicle_local_position_v1")
        self.assertEqual(px4_topic("/fmu/out/vehicle_land_detected", v0), "/fmu/out/vehicle_land_detected")
        self.assertEqual(px4_topic("/fmu/in/offboard_control_mode", unversioned), "/fmu/in/offboard_control_mode")


class FailedTestsTest(unittest.TestCase):
    def test_parses_failures_and_errors(self):
        report = (
            "..F.E\n"
            "======\n"
            "FAIL: test_start_gate (tests.test_mission.FlightRulesTest.test_start_gate)\n"
            "ERROR: test_config (unittest.loader._FailedTest.test_config)\n"
        )
        self.assertEqual(
            failed_tests(report),
            [
                "test_start_gate (tests.test_mission.FlightRulesTest.test_start_gate)",
                "test_config (unittest.loader._FailedTest.test_config)",
            ],
        )


class PositionProblemTest(unittest.TestCase):
    def test_each_reason_is_named(self):
        valid = SimpleNamespace(xy_valid=True, v_xy_valid=True)
        self.assertIsNone(position_problem(valid, received_at=10.0, now=10.05))
        self.assertIn("yet", position_problem(None, received_at=0.0, now=10.0))
        invalid = SimpleNamespace(xy_valid=True, v_xy_valid=False)
        self.assertIn("invalid", position_problem(invalid, received_at=10.0, now=10.0))
        self.assertIn("0.50 s old", position_problem(valid, received_at=10.0, now=10.5))


class WarmupGateTest(unittest.TestCase):
    def test_hopeless_loop_stops_early_and_reports_its_time(self):
        calls = []

        def slow_step(params, carry, observation):
            calls.append(None)
            time.sleep(0.005)
            return carry, None

        p99 = warmup_p99(slow_step, None, Carry(np.zeros(6)), None, deadline_ms=0.1)
        self.assertEqual(len(calls), WARMUP_MIN_STEPS)
        self.assertGreater(p99, 0.1)


if __name__ == "__main__":
    unittest.main()
