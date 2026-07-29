import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.simulation import run_simulation
from tests.helpers import write_small_config


class ControllerTest(unittest.TestCase):
    def test_tiny_single_robot_cpu_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_small_config(Path(temporary), steps=2))
            silent_output = io.StringIO()
            with redirect_stdout(silent_output):
                result = run_simulation(config, "cpu")
            self.assertEqual(result.paths.shape, (2, 1, 6))
            self.assertTrue(np.all(np.isfinite(result.paths)))
            self.assertNotIn("Progress:", silent_output.getvalue())

            progress_output = io.StringIO()
            with redirect_stdout(progress_output):
                run_simulation(config, "cpu", progress=True)
            self.assertIn("Progress: 1/2 (50%)", progress_output.getvalue())
            self.assertIn("Progress: 2/2 (100%)", progress_output.getvalue())


if __name__ == "__main__":
    unittest.main()
