"""Regressions for the gated seven-cell pillar tuning campaign."""

import csv
import tempfile
import unittest
from pathlib import Path

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.uav_ablation import ARMS
from ergodic_control_mppi.experiments.uav_ablation import FIELDS as ABLATION_FIELDS
from ergodic_control_mppi.experiments.uav_ablation import _apply
from ergodic_control_mppi.experiments.uav_pillar_tuning import (
    ABLATION_ARMS,
    APPROACH_ARMS,
    CAP_FIELDS,
    FIELDS,
    SCREEN_ARMS,
    SWEEP_ARMS,
    build_report,
    dry_run,
    evaluate,
    run_stage,
    select_split,
    stage_arms,
    write_ablation_copy,
    write_profile,
)


class SelectionTest(unittest.TestCase):
    def test_first_six_split_before_controller_evaluation(self):
        rows = [
            {"map_seed": str(seed), "qualifies": "1", "free_fraction": str(free)}
            for seed, free in zip(range(511, 518), (0.8, 0.6, 0.7, 0.9, 0.5, 0.65, 0.1))
        ]
        result = select_split(rows)
        self.assertEqual(result["development"], [511, 512, 513])
        self.assertEqual(result["holdout"], [514, 515, 516])
        self.assertEqual(result["development_representative"], 513)
        self.assertEqual(result["holdout_representative"], 516)

    def test_measured_blocked_mass_outranks_seed_order(self):
        """The six least-blocked qualifying seeds win, and lead their own split.

        Guards the failure this replaced: seed order put the map carrying 24.7% of the
        target inside inflation in front of one carrying 10.4%, and that map would have
        been the one the whole sweep ran on.
        """
        rows = [
            {"map_seed": str(seed), "qualifies": "1", "free_fraction": "0.84"}
            for seed in range(511, 519)
        ]
        blocked = {511: 0.25, 512: 0.10, 513: 0.21, 514: 0.11,
                   515: 0.09, 516: 0.13, 517: 0.12, 518: 0.24}
        result = select_split(rows, blocked)
        # 511 and 518 are the two worst and are the two dropped, despite 511 leading by seed.
        self.assertEqual(result["development"], [512, 514, 515])
        self.assertEqual(result["holdout"], [513, 516, 517])
        self.assertEqual(result["development_representative"], 515)
        self.assertEqual(result["holdout_representative"], 517)

    def test_worst_mode_beats_a_good_average(self):
        """A map whose lobes average well but starve one mode must lose to a balanced one.

        This is the real map 539 against the real map 516: 539 had the better aggregate
        (10.1% vs 12.5%) and one mode at 22.5%, and both flights on it missed exactly that
        mode. Ranking must be on the worst lobe, so the balanced map wins.
        """
        worst = {516: 0.138, 530: 0.160, 525: 0.167, 514: 0.177, 531: 0.197,
                 537: 0.214, 539: 0.225, 512: 0.254}
        rows = [{"map_seed": str(s), "qualifies": "1", "free_fraction": "0.84"} for s in worst]
        result = select_split(rows, worst)
        self.assertEqual(result["development_representative"], 516)
        self.assertNotIn(539, result["development"] + result["holdout"])
        self.assertNotIn(512, result["development"] + result["holdout"])

    def test_unmeasured_candidates_are_refused_not_ranked_as_zero(self):
        """A seed with no built map must not outrank a measured one by defaulting to 0."""
        rows = [
            {"map_seed": str(seed), "qualifies": "1", "free_fraction": "0.84"}
            for seed in range(511, 519)
        ]
        with self.assertRaisesRegex(ValueError, "need 6"):
            select_split(rows, {511: 0.10, 512: 0.11})


class ArmTest(unittest.TestCase):
    def test_stage_shapes_and_overrides_are_fixed(self):
        self.assertEqual(
            dry_run(),
            {
                "screen_cap": 30, "screen_full_max": 30,
                "approach_cap": 108, "approach_full_max": 108,
                "holdout_cap": 108, "holdout_full_max": 108,
                "sweep_cap": 372, "sweep_full_max": 372,
            },
        )
        approach = stage_arms("approach", "T500", "base")
        self.assertEqual(approach["tau11"]["T"], 500)
        self.assertEqual(approach["tau11"]["lam_max"], 1e5)
        self.assertEqual(approach["tau11"]["memory_time"], 11.0)
        holdout = stage_arms("holdout", "K500", "h2.35")
        self.assertEqual(holdout["winner"]["K"], 500)
        self.assertEqual(holdout["winner"]["fine_bandwidth"], 2.35)
        self.assertEqual(set(stage_arms("screen", "shipped", "base")), set(SCREEN_ARMS))
        self.assertEqual(len(APPROACH_ARMS), 6)

    def test_sweep_arms_come_from_the_ablation_table(self):
        """One arm definition serves the sweep and the standalone ablation runner."""
        sweep = stage_arms("sweep", "shipped", "base")
        self.assertEqual(set(sweep), set(SWEEP_ARMS))
        self.assertEqual(sweep["baseline"], {})
        # The base arm is unmodified, so the shipped profile is its own control.
        self.assertEqual(sweep["theta_75"], {"theta": 75.0})
        for name, _, _, overrides in ARMS:
            if name in SWEEP_ARMS:
                self.assertEqual(sweep[name], overrides)

    def test_every_sweep_arm_is_applicable_and_writable(self):
        """An arm the runner honours but the profile cannot express is unflyable."""
        config = load_config("configs/uav_profile.yaml")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile.yaml"
            for name in SWEEP_ARMS:
                overrides = ABLATION_ARMS[name][2]
                _apply(config, overrides)
                load_config(str(write_profile(overrides, output)))

    def test_cap_must_precede_full_run(self):
        with self.assertRaisesRegex(ValueError, "smaller than steps"):
            run_stage(
                Path("missing"), Path("missing.csv"), "screen", "shipped", "base",
                43, 1, 10000, "cpu", set(), Path("cap.csv"), 10000,
            )


class GateTest(unittest.TestCase):
    def _row(self, arm, seed, reached, steps=10000):
        row = dict.fromkeys(CAP_FIELDS, "")
        row.update(
            stage="screen", arm=arm, map_seed=511, seed=seed, steps=steps,
            ess_settled_median=0.3, temperature_cap_fraction=0.0,
            occupancy_mse=1e-6, all_modes_reached=int(reached),
            first_all_modes_s=100.0 if reached else float("nan"),
            mode_cycles=0, collisions=0, accepted=int(reached),
        )
        return row

    def test_screen_requires_a_material_visitation_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self._row("common_cap", seed, False) for seed in range(43, 49)]
            rows += [self._row("T500", seed, seed < 47) for seed in range(43, 49)]
            with (root / "run_cap.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CAP_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            full = [self._row("T500", seed, True, 20000) for seed in range(43, 47)]
            with (root / "offline.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows({field: row[field] for field in FIELDS} for row in full)
            result = evaluate(root)
        self.assertEqual(result["screen_winner"], "T500")
        self.assertTrue(result["screen"]["T500"]["eligible"])
        self.assertFalse(result["screen"]["common_cap"]["eligible"])

    def test_report_exposes_per_map_terminal_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self._row("base", seed, False) for seed in range(43, 49)]
            for row in rows:
                row["stage"] = "approach"
            with (root / "run_cap.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CAP_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            report = build_report(root)
        self.assertIn("| base | 511 | 6 | 0 | 0 | 0 |", report)
        self.assertIn("Holdout primary: **NOT RUN**", report)
        self.assertIn("**Negative result:**", report)
        self.assertIn("one dwell-qualified visit to each target mode", report)


class AblationExportTest(unittest.TestCase):
    """The sweep's 10k layer doubles as a paired ablation campaign, so it must be balanced."""

    def _write(self, root, arms):
        rows = []
        for arm, seeds in arms.items():
            for seed in seeds:
                row = dict.fromkeys(CAP_FIELDS, "")
                row.update(
                    stage="sweep", arm=arm, axis=ABLATION_ARMS[arm][0],
                    value=ABLATION_ARMS[arm][1], map_seed=543, seed=seed, steps=10000,
                    occupancy_mse=1e-6, fourier_ergodic=0.05, accepted=1,
                )
                rows.append(row)
        with (root / "sweep_cap.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CAP_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_export_carries_the_ablation_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seeds = range(43, 55)
            self._write(root, {"baseline": seeds, "theta_75": seeds})
            output = write_ablation_copy(root, root / "ablation_543.csv")
            with output.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
        # Leading columns are the ablation schema, so report_figures reads this archive
        # exactly as it reads the first campaign's.
        self.assertEqual(fields[: len(ABLATION_FIELDS)], ABLATION_FIELDS)
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["axis"] for row in rows}, {"-", "theta"})

    def test_export_refuses_an_unpaired_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, {"baseline": range(43, 55), "theta_75": range(43, 50)})
            with self.assertRaisesRegex(ValueError, "theta_75"):
                write_ablation_copy(root, root / "ablation_543.csv")


if __name__ == "__main__":
    unittest.main()
