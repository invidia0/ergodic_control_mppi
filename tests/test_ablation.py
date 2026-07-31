"""Campaign expansion, archive round-trip, and the derived metrics."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from ergodic_control_mppi.experiments.ablation import (
    Cell,
    _estimate_ms,
    _get_dotted,
    _patched,
    _set_dotted,
    _static_key,
    expand,
    load_campaign,
    run_campaign,
)
from ergodic_control_mppi.experiments.analyze import (
    bootstrap_ci,
    load_index,
    load_run,
    load_series,
    steps_to_threshold,
    summarize_stage,
    write_summary,
)
from ergodic_control_mppi.metrics.ergodicity import (
    compute_cumulative_fourier_ergodic_metric,
    compute_cumulative_team_ergodic_error,
    compute_fourier_ergodic_metric,
    compute_team_ergodic_error,
    fourier_wavenumbers,
)

CAMPAIGN = "configs/experiments/ablation_campaign.yaml"


class DottedPathTest(unittest.TestCase):
    def test_set_and_get_roundtrip(self):
        data = {"a": {"b": 1}, "top": 2}
        _set_dotted(data, "a.b", 7)
        _set_dotted(data, "top", 9)
        self.assertEqual(_get_dotted(data, "a.b"), 7)
        self.assertEqual(_get_dotted(data, "top"), 9)
        self.assertIsNone(_get_dotted(data, "a.missing"))

    def test_non_mapping_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            _set_dotted({"a": 1}, "a.b", 2)


class ExpansionTest(unittest.TestCase):
    def setUp(self):
        self.campaign = load_campaign(CAMPAIGN)

    def test_cell_id_is_stable_and_distinguishing(self):
        first = Cell("s", "arm", {"stein.memory_gain": 5.0}, 43, 43, "trimodal", 100)
        same = Cell("s", "arm", {"stein.memory_gain": 5.0}, 43, 43, "trimodal", 100)
        other = Cell("s", "arm", {"stein.memory_gain": 6.0}, 43, 43, "trimodal", 100)
        seeded = Cell("s", "arm", {"stein.memory_gain": 5.0}, 44, 43, "trimodal", 100)
        self.assertEqual(first.cell_id, same.cell_id)
        self.assertNotEqual(first.cell_id, other.cell_id)
        self.assertNotEqual(first.cell_id, seeded.cell_id)

    def test_static_signatures_are_contiguous(self):
        """Cells sharing an XLA-static signature must not be interleaved."""
        cells = expand(self.campaign, ["screening", "interactions"])
        base = yaml.safe_load(self.campaign.base_config.read_text(encoding="utf-8"))
        keys = [_static_key(_patched(base, self.campaign, cell)) for cell in cells]
        seen = []
        for key in keys:
            if not seen or seen[-1] != key:
                self.assertNotIn(key, seen, "static signature revisited: recompile churn")
                seen.append(key)

    def test_expansion_is_deduplicated(self):
        cells = expand(self.campaign)
        self.assertEqual(len({c.cell_id for c in cells}), len(cells))

    def test_patched_applies_density_seeds_and_axes(self):
        cell = Cell("s", "arm", {"stein.memory_gain": 42.0, "mppi.T": 7},
                    11, 22, "bimodal", 33)
        base = yaml.safe_load(self.campaign.base_config.read_text(encoding="utf-8"))
        data = _patched(base, self.campaign, cell)
        self.assertEqual(data["seed"], 11)
        self.assertEqual(data["steps"], 33)
        self.assertEqual(data["map"]["obstacles"]["seed"], 22)
        self.assertEqual(data["stein"]["memory_gain"], 42.0)
        self.assertEqual(data["mppi"]["T"], 7)
        self.assertEqual(len(data["density"]["weights"]), 2)
        # The base config must not be mutated in place.
        self.assertNotEqual(base["seed"], 11)

    def test_unknown_density_is_rejected(self):
        cell = Cell("s", "arm", {}, 1, 1, "does_not_exist", 10)
        base = yaml.safe_load(self.campaign.base_config.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            _patched(base, self.campaign, cell)

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            expand(self.campaign, ["not_a_stage"])

    def test_cost_model_orders_the_expensive_corners(self):
        base = yaml.safe_load(self.campaign.base_config.read_text(encoding="utf-8"))
        shipped = _estimate_ms(base)
        heavy = dict(base)
        heavy["mppi"] = {**base["mppi"], "T": 700}
        self.assertGreater(_estimate_ms(heavy), shipped)
        self.assertGreater(shipped, 1.0)


class TransientMetricTest(unittest.TestCase):
    def test_requires_the_threshold_to_hold_to_the_end(self):
        steps = np.arange(0, 100, 10)
        # Dips below 1.0 at step 20 but comes back up: not converged.
        transient = np.array([5, 4, 0.5, 3, 2, 0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(steps_to_threshold(steps, transient, 1.0), 50)
        monotone = np.array([5, 4, 3, 2, 1.5, 0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(steps_to_threshold(steps, monotone, 1.0), 50)
        never = np.full(10, 9.0)
        self.assertTrue(np.isnan(steps_to_threshold(steps, never, 1.0)))

    def test_bootstrap_ci_brackets_the_median(self):
        values = np.array([-3.0, -2.0, -2.5, -1.0, -4.0])
        low, high = bootstrap_ci(values)
        self.assertLess(low, np.median(values))
        self.assertGreater(high, np.median(values))


class FourierMetricTest(unittest.TestCase):
    def setUp(self):
        self.limits = (-10.0, 10.0)
        self.uniform = np.ones((48, 48))

    def test_excludes_the_constant_mode(self):
        k_arr, lambda_k = fourier_wavenumbers(3)
        self.assertFalse(np.any(np.all(k_arr == 0.0, axis=1)))
        self.assertEqual(len(k_arr), 4 * 4 - 1)
        # Sobolev weights must decay with wavenumber.
        self.assertGreater(lambda_k[0], lambda_k[-1])
        with self.assertRaises(ValueError):
            fourier_wavenumbers(0)

    def test_space_filling_path_beats_a_corner_path(self):
        t = np.linspace(0.0, 400.0 * np.pi, 20000)
        filling = np.stack([9.8 * np.sin(0.0301 * t), 9.8 * np.sin(0.0299 * t + 0.5)], -1)
        corner = np.stack([-9.0 + 0.5 * np.cos(t), -9.0 + 0.5 * np.sin(t)], -1)
        good = compute_fourier_ergodic_metric(filling, self.uniform, self.limits, self.limits, order=6)
        bad = compute_fourier_ergodic_metric(corner, self.uniform, self.limits, self.limits, order=6)
        self.assertLess(good, bad)

    def test_interleaving_beats_sequential_mid_run(self):
        """Same point set, better ordering -> lower metric before the end."""
        rng = np.random.default_rng(0)
        left = np.stack([rng.uniform(-9, -1, 4000), rng.uniform(-9, 9, 4000)], -1)
        right = np.stack([rng.uniform(1, 9, 4000), rng.uniform(-9, 9, 4000)], -1)
        sequential = np.concatenate([left, right])
        interleaved = np.empty_like(sequential)
        interleaved[0::2], interleaved[1::2] = left, right
        a = compute_cumulative_fourier_ergodic_metric(
            sequential, self.uniform, self.limits, self.limits, order=6
        )
        b = compute_cumulative_fourier_ergodic_metric(
            interleaved, self.uniform, self.limits, self.limits, order=6
        )
        middle = len(a) // 2
        self.assertLess(b[middle], a[middle])
        # Identical point sets must score identically once both are complete.
        self.assertAlmostEqual(a[-1], b[-1], places=12)

    def test_series_last_entry_matches_the_scalar(self):
        rng = np.random.default_rng(1)
        path = np.stack([rng.uniform(-9, 9, 500), rng.uniform(-9, 9, 500)], -1)
        for stride in (1, 7, 25):
            with self.subTest(stride=stride):
                series = compute_cumulative_fourier_ergodic_metric(
                    path, self.uniform, self.limits, self.limits, order=5, stride=stride
                )
                scalar = compute_fourier_ergodic_metric(
                    path, self.uniform, self.limits, self.limits, order=5
                )
                self.assertAlmostEqual(series[-1], scalar, places=12)


class StridedOccupancySeriesTest(unittest.TestCase):
    def test_stride_does_not_change_the_final_value(self):
        """Striding must skip evaluations, not samples."""
        rng = np.random.default_rng(2)
        path = np.stack([rng.uniform(-9, 9, 400), rng.uniform(-9, 9, 400)], -1)
        target = np.ones((20, 20))
        limits = (-10.0, 10.0)
        scalar = compute_team_ergodic_error(path, target, limits, limits, (20, 20))
        for stride in (1, 9, 25):
            with self.subTest(stride=stride):
                series = compute_cumulative_team_ergodic_error(
                    path, target, limits, limits, (20, 20), stride=stride
                )
                self.assertAlmostEqual(series[-1], scalar, places=12)
        with self.assertRaises(ValueError):
            compute_cumulative_team_ergodic_error(
                path, target, limits, limits, (20, 20), stride=0
            )


class InvalidCellTest(unittest.TestCase):
    """One impossible parameter combination must not abort the whole stage.

    Regression: a fine x coarse sweep asked for h_c = 1.0 with delta_res = 0.8
    (h_f = 1.28), which load_config rightly rejects -- and the traceback killed
    the remaining 510 cells of a 30-hour run.
    """

    def test_stage_survives_and_records_the_skip(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = yaml.safe_load(Path(CAMPAIGN).read_text(encoding="utf-8"))
            data["output_root"] = str(root / "out")
            data["stages"] = {
                "mixed": {
                    "kind": "arms",
                    "steps": 40,
                    "seeds": [43],
                    "arms": {
                        "ok": {},
                        # h_f = 2 * 0.8^2 = 1.28 > h_c = 1.0 -> rejected by config.py
                        "impossible": {
                            "stein.fill_resolution": 0.8,
                            "stein.coarse_bandwidth": 1.0,
                        },
                        "ok_too": {"stein.memory_gain": 5.0},
                    },
                }
            }
            config = root / "campaign.yaml"
            config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            out = Path(data["output_root"])

            executed = run_campaign(config, stages=["mixed"], device="cpu")

            self.assertEqual(executed, 2, "the two valid arms must still run")
            arms = {row["arm"] for row in load_index(out, "mixed")}
            self.assertEqual(arms, {"ok", "ok_too"})

            skipped = out / "mixed_skipped.csv"
            self.assertTrue(skipped.exists(), "the skip must be recorded, not swallowed")
            with skipped.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["arm"], "impossible")
            self.assertIn("coarse_bandwidth", rows[0]["error"])


class ArchiveRoundTripTest(unittest.TestCase):
    """The archive must be sufficient to re-derive any metric without the GPU."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        data = yaml.safe_load(Path(CAMPAIGN).read_text(encoding="utf-8"))
        data["output_root"] = str(root / "out")
        # A two-cell, two-seed, 60-step stage: enough to exercise every path.
        data["stages"] = {
            "tiny": {
                "kind": "arms",
                "steps": 60,
                "seeds": [43, 44],
                "arms": {"full": {}, "memory_off": {"stein.memory_gain": 0.0}},
            }
        }
        config = root / "campaign.yaml"
        config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        cls.root = Path(data["output_root"])
        run_campaign(config, stages=["tiny"], device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_index_and_configs_exist_for_every_cell(self):
        rows = load_index(self.root, "tiny")
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue((self.root / "runs" / "tiny" / f"{row['cell_id']}.npz").exists())
            self.assertTrue((self.root / "configs" / "tiny" / f"{row['cell_id']}.yaml").exists())
        self.assertTrue((self.root / "manifest.json").exists())

    def test_metrics_recompute_exactly_from_the_archive(self):
        for row in load_index(self.root, "tiny"):
            run = load_run(self.root, "tiny", row["cell_id"])
            limits_x, limits_y = (tuple(v) for v in run["map_limits"])
            target, mask = run["target_grid"], run["reachable_mask"]
            bins = (target.shape[1], target.shape[0])
            mse = compute_team_ergodic_error(
                run["paths"], target, limits_x, limits_y, bins, reachable_mask=mask
            )
            self.assertAlmostEqual(mse, float(row["occupancy_mse"]), places=12)
            epsilon = compute_fourier_ergodic_metric(
                run["paths"], target, limits_x, limits_y, order=10, reachable_mask=mask
            )
            self.assertAlmostEqual(epsilon, float(row["fourier_ergodic"]), places=12)

    def test_stored_series_ends_at_the_stored_scalar(self):
        for row in load_index(self.root, "tiny"):
            steps, values = load_series(self.root, "tiny", row["cell_id"])
            self.assertEqual(len(steps), len(values))
            self.assertAlmostEqual(values[-1], float(row["occupancy_mse"]), places=12)
            self.assertEqual(steps[-1], int(row["steps"]) - 1)

    def test_resume_skips_completed_cells(self):
        data = yaml.safe_load((Path(self.temp.name) / "campaign.yaml").read_text())
        config = Path(self.temp.name) / "campaign.yaml"
        executed = run_campaign(config, stages=["tiny"], device="cpu")
        self.assertEqual(executed, 0, "a completed campaign must be a no-op on resume")
        self.assertEqual(len(load_index(self.root, "tiny")), 4)

    def test_summary_is_written_and_pairs_against_the_reference(self):
        path = write_summary(self.root, "tiny", threshold_factor=1.5)
        self.assertTrue(path.exists())
        rows = summarize_stage(self.root, "tiny", threshold_factor=1.5)
        by_arm = {row["arm"]: row for row in rows}
        self.assertIn("full", by_arm)
        self.assertIn("memory_off", by_arm)
        self.assertEqual(by_arm["full"]["n_seeds"], 2)
        # The reference arm must not be compared against itself.
        self.assertNotIn("occupancy_mse_wins", {k for k, v in by_arm["full"].items() if v != ""})
        self.assertIn("occupancy_mse_wins", by_arm["memory_off"])
        wins, total = by_arm["memory_off"]["occupancy_mse_wins"].split("/")
        self.assertEqual(int(total), 2)
        self.assertLessEqual(int(wins), 2)

    def test_threshold_factor_changes_the_transient_not_the_error(self):
        loose = summarize_stage(self.root, "tiny", threshold_factor=3.0)
        tight = summarize_stage(self.root, "tiny", threshold_factor=1.1)
        for a, b in zip(loose, tight):
            self.assertAlmostEqual(a["occupancy_mse_median"], b["occupancy_mse_median"])
        loose_full = next(r for r in loose if r["arm"] == "full")
        tight_full = next(r for r in tight if r["arm"] == "full")
        self.assertGreater(loose_full["threshold"], tight_full["threshold"])


if __name__ == "__main__":
    unittest.main()
