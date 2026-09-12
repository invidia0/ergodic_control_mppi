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
        """Marks remain legible without decorative white halos.

        Checked against the ramp the panels actually fill with. This used to use
        ``style.OCCUPANCY_CMAP``, whose dark end is #8e8e8e, while every filled
        panel draws ``MECHANISM_OCCUPANCY_CMAP``, which runs to #676767 -- so a
        mark could pass here and still vanish into the field on the page.
        """
        from ergodic_control_mppi.plotting.mechanism import (
            CONSTRUCTION,
            INK,
            MECHANISM_OCCUPANCY_CMAP,
            ROBOT_COLOR,
            ROBOT_EDGE,
            TARGET_LINE,
        )

        darkest = MECHANISM_OCCUPANCY_CMAP(1.0)
        for name, color, background, minimum in (
            ("old trail", TRAIL_CMAP(0.0), MECHANISM_OCCUPANCY_CMAP(0.0), 1.3),
            ("new trail", TRAIL_CMAP(1.0), darkest, 2.5),
            ("flat trail", INK, darkest, 2.5),
            # A solid jet-black dot, as in every figure: it has to clear the field as the
            # trail does, not its own ring.
            ("robot", ROBOT_COLOR, darkest, 2.5),
            ("target contour", TARGET_LINE, darkest, 1.5),
            ("construction", CONSTRUCTION, darkest, 2.0),
            ("mode", "#33415C", darkest, 1.7),
        ):
            with self.subTest(mark=name):
                ratio = _contrast(matplotlib.colors.to_rgb(color),
                                  matplotlib.colors.to_rgb(background))
                self.assertGreaterEqual(
                    ratio, minimum,
                    f"{name} reads only {ratio:.2f}:1 against its field background",
                )

    def test_marks_clear_the_bare_panel(self):
        """The vector-field panel has no fill under it, so its marks sit on SURFACE."""
        from ergodic_control_mppi.plotting.mechanism import FLOW_COLOR, RECENCY_COLOR

        surface = matplotlib.colors.to_rgb(SURFACE)
        for name, color in (("flow", FLOW_COLOR), ("recency", RECENCY_COLOR)):
            with self.subTest(mark=name):
                ratio = _contrast(matplotlib.colors.to_rgb(color), surface)
                self.assertGreaterEqual(
                    ratio, 3.0,
                    f"{name} reads only {ratio:.2f}:1 against the panel {SURFACE}",
                )

    def test_target_contours_clear_every_step_of_the_field(self):
        """A line drawn over a filled ramp has to clear the whole ramp, not its ends.

        The failure this catches is silent and was live: a mid-tone blue matches the
        ramp's luminance somewhere in the middle and the contour disappears exactly
        where the occupancy ridge is, which is the part of the figure being argued
        about. The rejects below are the two obvious candidates -- the colour this
        replaced (#356FA8, 1.08:1), the old paper blue (#0078FF), and PRIMARY
        taken straight -- so the guard fails what it was written for.
        """
        from ergodic_control_mppi.plotting.mechanism import (
            MECHANISM_OCCUPANCY_CMAP,
            TARGET_LINE,
        )
        from ergodic_control_mppi.plotting.style import PRIMARY

        def worst(color):
            return min(
                _contrast(matplotlib.colors.to_rgb(color), MECHANISM_OCCUPANCY_CMAP(v))
                for v in np.linspace(0.0, 1.0, 33)
            )

        ratio = worst(TARGET_LINE)
        self.assertGreaterEqual(
            ratio, 1.5,
            f"target contours read {ratio:.2f}:1 at the worst step of the field ramp",
        )
        for reject in ("#356FA8", "#0078FF", PRIMARY):
            with self.subTest(reject=reject):
                self.assertLess(worst(reject), 1.5)

    def test_trail_ramp_darkens_with_recency(self):
        samples = np.asarray([TRAIL_CMAP(value)[:3] for value in np.linspace(0, 1, 9)])
        luminance = np.asarray([_relative_luminance(color) for color in samples])
        self.assertTrue(np.all(np.diff(luminance) < 0.0))
        for endpoint in (TRAIL_CMAP(0.0), TRAIL_CMAP(1.0)):
            self.assertNotIn(
                matplotlib.colors.to_hex(endpoint).lower(), ("#000000", "#ffffff")
            )


class ManuscriptPaletteTest(unittest.TestCase):
    """One categorical cycle and two named ramps for every manuscript figure."""

    def test_baseline_methods_have_unique_hues(self):
        from ergodic_control_mppi.plotting.style import METHOD_COLORS

        keys = ("ours", "hedac", "sves", "fmec", "smc")
        hues = [METHOD_COLORS[key].lower() for key in keys]
        self.assertEqual(len(hues), len(set(hues)))
        self.assertEqual(METHOD_COLORS["ours"], METHOD_COLORS["mppi"])

    def test_diverging_maps(self):
        from ergodic_control_mppi.plotting.style import DIVERGING_CMAP, POTENTIAL_CMAP

        self.assertIn("roma", DIVERGING_CMAP.name.lower())
        self.assertEqual("RdYlBu_r", POTENTIAL_CMAP.name)

    def test_density_ramp_is_white_to_jetblack(self):
        from ergodic_control_mppi.plotting.style import DENSITY_CMAP

        light = matplotlib.colors.to_hex(DENSITY_CMAP(0.0)).lower()
        dark = DENSITY_CMAP(1.0)
        self.assertEqual(light, "#ffffff")
        self.assertLess(_relative_luminance(dark), 0.1)


class MechanismFieldTest(unittest.TestCase):
    """The figures must compute what the controller computes, not a lookalike.

    ``_field_at`` / ``_rho`` / ``_rho_excess`` transcribe ``field.py:memory_weights`` and
    ``kde_repulsion`` so they can be evaluated on a grid instead of only at the memory
    points. Pinning their ``memory_balance`` blend against ``memory_repulsion`` itself is what
    stops the figures drifting from the implementation.
    """

    def test_memory_field_matches_the_controller(self):
        from dataclasses import replace

        import jax.numpy as jnp

        from ergodic_control_mppi.mppi.field import memory_repulsion
        from ergodic_control_mppi.plotting.mechanism import _rho, _rho_excess

        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(write_small_config(Path(temporary), steps=2))
        params = config.controller

        rng = np.random.default_rng(0)
        memory = jnp.asarray(rng.uniform(-4.0, 4.0, size=(40, 2)), dtype=jnp.float32)
        recency = jnp.asarray(0.99 ** np.arange(40)[::-1], dtype=jnp.float32)
        points = jnp.asarray(rng.uniform(-5.0, 5.0, size=(11, 2)), dtype=jnp.float32)
        bandwidth = 0.7
        floor = 1.0 / 400.0

        field = replace(params.field, fine_bandwidth=bandwidth)
        ctx = {"field": field, "gmm": params.gmm, "memory": memory,
               "recency": recency, "density_floor": floor}

        expected = memory_repulsion(points, memory, recency, params.gmm, field, floor)
        balance = float(field.memory_balance)
        got = float(np.sqrt(0.5 * np.e * bandwidth)) * (
            (1.0 - balance) * _rho(ctx, points, np.asarray(recency), bandwidth)
            + balance * _rho_excess(ctx, points, bandwidth)
        )
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

    def test_strip_axes_keeps_the_inset_that_saves_the_caps(self):
        """Bare 3D renders must not fill the canvas.

        mplot3d draws the projected box outside the axes rectangle. Setting the axes to
        (0, 0, 1, 1) leaves the far pillar caps with nowhere to land but the figure edge,
        which is the cut padding the crop cannot undo -- those pixels were never drawn.
        """
        from ergodic_control_mppi.plotting.deployment import _strip_axes

        figure = plt.figure()
        axes = figure.add_subplot(111, projection="3d")
        axes.set_position((0.06, 0.10, 0.88, 0.76))
        _strip_axes(axes)
        box = axes.get_position()
        plt.close(figure)
        self.assertLess(box.y0 + box.height, 0.95)
        self.assertGreater(box.y0, 0.02)


class MechanismMapTest(unittest.TestCase):
    """Visit-count metric and the two split Figure-5 builders."""

    def test_stacked_passes_count_higher_than_offset_passes(self):
        from ergodic_control_mppi.plotting.trajectories import neighboring_pass_count

        theta = np.linspace(0.0, 12.0 * 2.0 * np.pi, 800)
        coil = np.stack([0.4 * np.cos(theta), 0.4 * np.sin(theta)], axis=1)
        coil_ids = np.floor(np.linspace(0.0, 12.0, 800, endpoint=False)).astype(int)
        t = np.linspace(0.0, 8.0, 200)
        spread = np.concatenate([
            np.stack([t, np.zeros_like(t)], axis=1),
            np.stack([t, np.full_like(t, 3.0)], axis=1),
        ])
        spread_ids = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
        self.assertGreater(
            np.median(neighboring_pass_count(coil, coil_ids, 0.6)),
            np.median(neighboring_pass_count(spread, spread_ids, 0.6)),
        )

    def test_offset_passes_are_not_neighbours(self):
        from ergodic_control_mppi.plotting.trajectories import neighboring_pass_count

        t = np.linspace(0.0, 8.0, 80)
        xy = np.concatenate([
            np.stack([t, np.zeros_like(t)], axis=1),
            np.stack([t, np.full_like(t, 3.0)], axis=1),
        ])
        ids = np.concatenate([np.zeros(80, dtype=int), np.ones(80, dtype=int)])
        self.assertEqual(int(neighboring_pass_count(xy, ids, radius=0.5).max()), 0)

    def test_plan_gain_and_potential_write_files(self):
        from ergodic_control_mppi.plotting.trajectories import (
            figure_plan_gain, figure_potential,
        )

        t = np.linspace(0.0, 2.0 * np.pi, 40)
        xy = np.stack([np.cos(t), np.sin(t)], axis=1)
        capture = {
            "positions": xy,
            "limits": np.array([-3.0, 3.0, -3.0, 3.0]),
            "means": np.array([[0.0, 0.0], [1.5, 0.0]]),
            "covariances": np.stack([np.eye(2), np.eye(2)]),
            "log_weights": np.array([np.log(0.5), np.log(0.5)]),
            "memory": xy[:20],
            "recency": 0.99 ** np.arange(20)[::-1],
            "plan": xy[:8],
            "fine_bandwidth": np.array(0.94),
            "memory_gain": np.array(1.0),
            "memory_balance": np.array(0.5),
            "plan_gain": np.array(6.0),
            "deficit_ceiling": np.array(0.05),
            "release_ratio": np.array(2.24),
            "service_mass": np.array([0.5, 0.5]),
            "title": "$g=6$",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = figure_plan_gain([capture], directory / "plan.png")
            potential = figure_potential(capture, directory / "phi.png", resolution=16)
            self.assertTrue(plan.exists())
            self.assertTrue(potential.exists())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()


def test_cylinder_scene_sorts_back_to_front():
    """The cylinder pillars and the trail share one painter's order.

    `computed_zorder=False` means mplot3d does no depth sorting of its own, so this
    ordering is the only thing keeping a far pillar from drawing through a near one,
    and the trail weaving between them rather than floating over the field.
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
