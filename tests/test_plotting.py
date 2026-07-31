import tempfile
import unittest
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.plotting.simulation import plot_simulation
from ergodic_control_mppi.plotting.style import (
    ACCENT,
    EXCESS_CMAP,
    OCCUPANCY_CMAP,
    SEQUENTIAL_CMAP,
    SURFACE,
    TRAIL_CMAP,
    TRAIL_STROKE,
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

    def test_marks_clear_the_field_ramp(self):
        """A field map is full-bleed, so its ramp is the background for the marks.

        The occupancy/field maps are drawn edge to edge, so what a mark on top has
        to clear is the darkest step of OCCUPANCY_CMAP, not SURFACE. The trail's
        recent end and the robot wear ACCENT; the faded end and every halo are
        white. Both must stay legible if the ramp is ever re-clipped.
        """
        darkest = OCCUPANCY_CMAP(1.0)
        for name, color in (("white", (1.0, 1.0, 1.0)), ("accent", ACCENT)):
            with self.subTest(mark=name):
                ratio = _contrast(matplotlib.colors.to_rgb(color), darkest)
                self.assertGreaterEqual(
                    ratio, 3.0,
                    f"{name} reads {ratio:.2f}:1 on the darkest field step "
                    f"{matplotlib.colors.to_hex(darkest)}; clip OCCUPANCY_CMAP lighter",
                )

    def test_trail_ramp_is_exempt_but_stroked(self):
        """TRAIL_CMAP's white end is deliberate; the underlay is what saves it.

        It is the one ramp that may fail the surface rule -- fading to invisible is
        the message. This asserts the exemption is paid for: the ramp does start at
        white, and TRAIL_STROKE is dark enough to outline it.
        """
        white_end = matplotlib.colors.to_hex(TRAIL_CMAP(0.0)).lower()
        self.assertEqual(white_end, "#ffffff")
        self.assertGreaterEqual(
            _contrast(matplotlib.colors.to_rgb(TRAIL_STROKE), (1.0, 1.0, 1.0)), 3.0,
            "TRAIL_STROKE cannot outline a white trail",
        )


class MechanismFieldTest(unittest.TestCase):
    """The figures must compute what the controller computes, not a lookalike.

    ``_field_at`` / ``_rho`` / ``_rho_excess`` transcribe ``stein.py:174-191`` so
    they can be evaluated on a grid instead of only at the memory points. Pinning
    them against ``multiscale_memory_flow`` itself -- with the bank collapsed to a
    single scale, which is the only configuration in which the two are comparable
    term for term -- is what stops the figures drifting from the implementation.
    """

    def test_scale_field_matches_the_controller(self):
        from dataclasses import replace

        import jax.numpy as jnp

        from ergodic_control_mppi.mppi.stein import multiscale_memory_flow
        from ergodic_control_mppi.plotting.mechanism import _scale_field

        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_small_config(Path(temporary), steps=2))
        params = config.controller

        rng = np.random.default_rng(0)
        memory = jnp.asarray(rng.uniform(-4.0, 4.0, size=(40, 2)), dtype=jnp.float32)
        recency = jnp.asarray(0.99 ** np.arange(40)[::-1], dtype=jnp.float32)
        points = jnp.asarray(rng.uniform(-5.0, 5.0, size=(11, 2)), dtype=jnp.float32)
        bandwidth = 0.7
        floor = 1.0 / 400.0

        single = replace(params.stein, memory_scales=1,
                         fine_bandwidth=bandwidth, coarse_bandwidth=bandwidth)
        ctx = {"stein": single, "gmm": params.gmm, "memory": memory,
               "recency": recency, "density_floor": floor}

        expected = multiscale_memory_flow(points, memory, recency, params.gmm, single, floor)
        got = _scale_field(ctx, points, bandwidth)
        # Guard against passing on two zero fields.
        self.assertGreater(float(np.abs(np.asarray(expected)).max()), 1e-3)
        self.assertTrue(
            np.allclose(got, expected, atol=1e-5),
            f"figure field differs from the controller by "
            f"{float(np.abs(got - expected).max()):.3g}",
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
