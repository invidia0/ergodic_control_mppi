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
from ergodic_control_mppi.plotting.style import (
    EXCESS_CMAP,
    SEQUENTIAL_CMAP,
    SURFACE,
    paper_style,
)
from ergodic_control_mppi.simulation import run_simulation
from tests.helpers import write_small_config


def _relative_luminance(rgb) -> float:
    channels = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(a, b) -> float:
    first, second = _relative_luminance(a), _relative_luminance(b)
    light, dark = max(first, second), min(first, second)
    return (light + 0.05) / (dark + 0.05)


class RampContrastTest(unittest.TestCase):
    """A sequential ramp whose light end sinks into the surface is invisible.

    The chart surface is light AND blue, so an unclipped Blues ramp lands at
    1.25:1 and `cividis` at 1.07:1 -- both unreadable. The shipped ramps are
    clipped to clear a 2:1 floor; this guards that without needing the node
    validator in CI.
    """

    @staticmethod
    def _lightest_contrast(cmap, surface) -> float:
        """Contrast of the ramp's *lightest* step, wherever along it that sits.

        Not `cmap(0.0)`: cividis and viridis run dark->light, so sampling the
        low end would pass them on their dark blue and miss the pale end that is
        the actual problem.
        """
        steps = [cmap(v) for v in [i / 32 for i in range(33)]]
        lightest = max(steps, key=_relative_luminance)
        return _contrast(lightest, surface)

    def test_light_end_clears_the_surface(self):
        surface = matplotlib.colors.to_rgb(SURFACE)
        for name, cmap in (("sequential", SEQUENTIAL_CMAP), ("excess", EXCESS_CMAP)):
            with self.subTest(ramp=name):
                ratio = self._lightest_contrast(cmap, surface)
                self.assertGreaterEqual(
                    ratio, 2.0,
                    f"{name} ramp's lightest step is {ratio:.2f}:1 against the "
                    f"surface {SURFACE}; clip the ramp further",
                )

    def test_guard_rejects_the_ramps_it_was_written_for(self):
        """The guard is only worth having if it fails what we removed."""
        surface = matplotlib.colors.to_rgb(SURFACE)
        for name in ("cividis", "viridis", "Blues"):
            with self.subTest(ramp=name):
                ratio = self._lightest_contrast(plt.get_cmap(name), surface)
                self.assertLess(ratio, 2.0, f"{name} unexpectedly passes at {ratio:.2f}:1")

    def test_surface_matches_the_style(self):
        """SURFACE must track axes.facecolor, or the check above is vacuous."""
        self.assertEqual(
            matplotlib.colors.to_hex(paper_style()["axes.facecolor"]).lower(),
            SURFACE.lower(),
        )


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
