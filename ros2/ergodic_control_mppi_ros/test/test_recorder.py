"""Regressions for the recorder's output guard.

Refusing to overwrite is the behaviour worth pinning here: the recorder is what a screening
script runs unattended, so silently replacing a finished run would destroy evidence.
"""

import tempfile
import unittest
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from ergodic_control_mppi_ros.recorder import Recorder, file_hash

from helpers import CONFIG


def _recorder(directory: Path, run_id: str, overwrite: bool, bag: bool = False) -> Recorder:
    arguments = [
        "--ros-args",
        "-p", f"config:={CONFIG}",
        "-p", f"run_id:={run_id}",
        "-p", f"output_root:={directory}",
        "-p", f"overwrite:={'true' if overwrite else 'false'}",
        "-p", f"bag:={'true' if bag else 'false'}",
        "-p", "steps:=10",
    ]
    rclpy.init(args=arguments)
    try:
        return Recorder()
    except BaseException:
        rclpy.shutdown()
        raise


class OverwriteGuardTest(unittest.TestCase):
    def test_fresh_run_id_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            node = _recorder(Path(directory), "fresh", overwrite=False)
            try:
                self.assertEqual(node.steps, 10)
                self.assertEqual(node.run_id, "fresh")
            finally:
                node.destroy_node()
                rclpy.shutdown()

    def test_existing_run_id_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "taken").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                _recorder(Path(directory), "taken", overwrite=False)

    def test_existing_run_id_is_replaced_when_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "taken").mkdir(parents=True)
            node = _recorder(Path(directory), "taken", overwrite=True)
            try:
                self.assertTrue(node.overwrite)
            finally:
                node.destroy_node()
                rclpy.shutdown()

    def test_fresh_bag_directory_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bagged" / "bag").mkdir(parents=True)
            node = _recorder(Path(directory), "bagged", overwrite=False, bag=True)
            try:
                self.assertTrue(node.bag)
            finally:
                node.destroy_node()
                rclpy.shutdown()


class FileHashTest(unittest.TestCase):
    def test_hash_is_stable_and_content_dependent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "f.json"
            path.write_text("a", encoding="utf-8")
            first = file_hash(path)
            self.assertEqual(first, file_hash(path))
            path.write_text("b", encoding="utf-8")
            self.assertNotEqual(first, file_hash(path))

    def test_missing_file_is_reported_not_raised(self):
        self.assertEqual(file_hash("/nonexistent/path.json"), "unknown")


class DiagnosticArchiveTest(unittest.TestCase):
    def test_controller_diagnostics_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            node = _recorder(Path(directory), "diagnostics", overwrite=False)
            try:
                status = DiagnosticStatus(
                    name="ergodic_controller",
                    hardware_id="gpu",
                    values=[
                        KeyValue(key="ess_fraction", value="0.25"),
                        KeyValue(key="temperature", value="123.0"),
                        KeyValue(key="temperature_at_cap", value="True"),
                        KeyValue(key="state_trace_u32", value=",".join(str(i) for i in range(13))),
                    ],
                )
                node.on_diagnostics(DiagnosticArray(status=[status]))
                self.assertEqual(node.ess_fractions, [0.25])
                self.assertEqual(node.temperatures, [123.0])
                self.assertEqual(node.temperature_at_cap, [True])
                self.assertEqual(node.controller_state_bits[0].tolist(), list(range(13)))
            finally:
                node.destroy_node()
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
