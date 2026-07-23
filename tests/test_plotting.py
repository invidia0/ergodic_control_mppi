import tempfile
import unittest
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.plotting.simulation import plot_simulation
from ergodic_control_mppi.simulation import run_simulation
from tests.helpers import write_small_config


class PlottingTest(unittest.TestCase):
    def test_simulation_plot_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = load_config(write_small_config(directory, steps=2))
            result = run_simulation(config, "cpu")
            output = directory / "plot.png"
            figure = plot_simulation(config, result, output=output, show=False)
            self.assertTrue(output.exists())
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
