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
    EXCESS_CMAP,
    OCCUPANCY_CMAP,
    SEQUENTIAL_CMAP,
    SURFACE,
    TRAIL_CMAP,
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
        """Marks remain legible without decorative white halos."""
        from ergodic_control_mppi.plotting.mechanism import (
            FLOW_COLOR,
            ROBOT_COLOR,
            TARGET_COLOR,
        )

        darkest = OCCUPANCY_CMAP(1.0)
        for name, color, background, minimum in (
            ("old trail", TRAIL_CMAP(0.0), OCCUPANCY_CMAP(0.0), 1.3),
            ("new trail", TRAIL_CMAP(1.0), darkest, 5.0),
            ("robot", ROBOT_COLOR, darkest, 1.5),
            ("target", TARGET_COLOR, darkest, 1.5),
            ("mode", "#33415C", darkest, 3.0),
            ("flow", FLOW_COLOR, darkest, 2.0),
        ):
            with self.subTest(mark=name):
                ratio = _contrast(matplotlib.colors.to_rgb(color), background)
                self.assertGreaterEqual(
                    ratio, minimum,
                    f"{name} reads only {ratio:.2f}:1 against its field background",
                )

    def test_trail_ramp_darkens_with_recency(self):
        samples = np.asarray([TRAIL_CMAP(value)[:3] for value in np.linspace(0, 1, 9)])
        luminance = np.asarray([_relative_luminance(color) for color in samples])
        self.assertTrue(np.all(np.diff(luminance) < 0.0))
        for endpoint in (TRAIL_CMAP(0.0), TRAIL_CMAP(1.0)):
            self.assertNotIn(
                matplotlib.colors.to_hex(endpoint).lower(), ("#000000", "#ffffff")
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


def test_cylinder_scene_sorts_back_to_front():
    """The cylinder pillars and the trail share one painter's order.

    `computed_zorder=False` means mplot3d does no depth sorting of its own, so this
    ordering is the only thing keeping a far pillar's outline from drawing through a near
    one, and the trail weaving between them rather than floating over the field.
    """
    import numpy as np
    from unittest.mock import MagicMock

    from ergodic_control_mppi.plotting import deployment

    # Two pillars on the camera axis at azimuth -90 (camera at -y, so smaller y is nearer)
    # and a trail between them, all wide apart so the ranks are unambiguous.
    centres = np.array([[0.0, -8.0], [0.0, 8.0]])
    positions = np.zeros((40, 2))
    drawn: list[tuple[str, float]] = []
    axes = MagicMock()
    axes.plot_surface.side_effect = lambda *a, **k: drawn.append(("pillar", k["zorder"]))
    axes.plot.side_effect = lambda *a, **k: drawn.append(("line", k["zorder"]))

    components = deployment._cylinder_components(centres, 1.0)
    deployment._draw_cylinder_scene(
        axes, components, base=0.0, top=2.0,
        colour_map=lambda v: np.zeros((np.size(v), 4)), alpha=1.0, azimuth=-90.0,
        positions=positions, flight_fraction=0.5, trail_colour="#000000", trail_size=1.0,
    )

    orders = [z for _, z in drawn]
    assert orders == sorted(orders), "artists must be emitted back to front"
    # The far pillar (y = +8, away from a camera at -y) is drawn before the near one.
    pillars = [z for kind, z in drawn if kind == "pillar"]
    assert pillars[0] < pillars[-1]
