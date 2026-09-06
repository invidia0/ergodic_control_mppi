"""Regressions for the final nine-map ablation driver.

The expensive part of this campaign is unrunnable in a test, so what is pinned here is
everything that decides *whether the expensive part is valid*: the grouping widths that
define the numerical branch, the resume identity that must not let one branch satisfy
another, the atomic group write that a corrupt archive already cost us once, and the map
guard that keeps a perlin map out of a pillar campaign.
"""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "final_ablation", ROOT / "scripts" / "final_ablation.py"
)
final_ablation = importlib.util.module_from_spec(_spec)
sys.modules["final_ablation"] = final_ablation
_spec.loader.exec_module(final_ablation)

from ergodic_control_mppi.experiments.uav_ablation import (  # noqa: E402
    _BY_NAME,
    FINAL_ARMS,
)


def _map(map_seed: int, obs_num: int) -> dict:
    return {
        "map_seed": map_seed,
        "obs_num": obs_num,
        "run_dir": f"results/uav/density_{obs_num}/maps/map_{map_seed}",
        "occupied_cells": 800,
        "occupancy_digest": f"digest{obs_num}_{map_seed}",
        "reachable_fraction": 0.84,
        "grid_shape": [134, 267],
        "initial_state": [-15.57, 0.42, 0.0, 0.0, 0.0, 0.0],
    }


SIX = [_map(500 + i, obs) for obs in (10, 15, 20) for i in range(2)]
SEEDS = range(43, 49)


class ArmTableTest(unittest.TestCase):
    def test_every_final_arm_exists_and_is_unique(self):
        self.assertEqual(len(set(FINAL_ARMS)), len(FINAL_ARMS))
        for arm in FINAL_ARMS:
            self.assertIn(arm, _BY_NAME)

    def test_baseline_levels_are_not_swept_against_themselves(self):
        # An arm identical to the shipped profile would pair the control against itself
        # and dilute its axis with a guaranteed null.
        self.assertIn("baseline", FINAL_ARMS)
        # The withdrawn Stein axes must not be reachable as arms at all.
        for gone in ("theta_0", "theta_15", "Q2", "Q3_fine", "ell_self_0.25"):
            self.assertNotIn(gone, FINAL_ARMS)

    def test_campaign_size_matches_the_registered_design(self):
        self.assertEqual(len(FINAL_ARMS), 40)

    def test_the_three_necessity_rows_are_present(self):
        """One per term of Phi the mechanism argument claims is load-bearing."""
        for row in ("memory_off", "plan_off", "release_off"):
            self.assertIn(row, FINAL_ARMS)
        self.assertEqual(_BY_NAME["memory_off"][2], {"memory_gain": 0.0})
        self.assertEqual(_BY_NAME["plan_off"][2], {"plan_gain": 0.0})
        self.assertEqual(_BY_NAME["release_off"][2], {"release_ratio": 0.0})


class GroupingTest(unittest.TestCase):
    """The grouping *is* the numerical-branch contract, so it is pinned exactly."""

    def setUp(self):
        self.groups = list(final_ablation.groups(SIX, SEEDS))

    def test_every_group_has_exactly_one_width(self):
        for label, execution, lanes in self.groups:
            self.assertEqual(execution, f"batch{len(lanes)}", label)

    def test_unquarantined_arm_is_one_group_of_36(self):
        arm = [g for g in self.groups if g[0] == "plan_off"]
        self.assertEqual(len(arm), 1)
        self.assertEqual(len(arm[0][2]), 36)

    def test_quarantined_axis_is_chunked_at_its_own_width(self):
        chunks = [g for g in self.groups if g[0].startswith("K_1000")]
        self.assertEqual(len(chunks), 4)
        for _, execution, lanes in chunks:
            self.assertEqual(len(lanes), 9)
            self.assertEqual(execution, "batch9")

    def test_quarantined_axis_gets_a_baseline_at_its_own_width(self):
        # Without this the K arms have no comparator on their own branch, and the
        # comparison silently reaches across widths -- the exact error the width is for.
        replicates = [g for g in self.groups if g[0].startswith("baseline_K")]
        self.assertEqual(len(replicates), 4)
        self.assertTrue(all(len(lanes) == 9 for _, _, lanes in replicates))
        self.assertTrue(all(arm == "baseline" for _, _, lanes in replicates
                            for _, arm, _ in lanes))

    def test_each_arm_covers_every_map_and_seed_exactly_once(self):
        seen = {}
        for _, _, lanes in self.groups:
            for entry, arm, seed in lanes:
                key = (arm, entry["obs_num"], entry["map_seed"], seed)
                # baseline appears twice by design: once on each branch.
                seen[key] = seen.get(key, 0) + 1
        for arm in FINAL_ARMS:
            cells = [k for k in seen if k[0] == arm]
            self.assertEqual(len(cells), 36, arm)
            expected = 2 if arm == "baseline" else 1
            self.assertTrue(all(seen[k] == expected for k in cells), arm)

    def test_total_cell_count(self):
        # 38 arms x 36, plus the 36-cell baseline replicate the K quarantine needs.
        self.assertEqual(sum(len(lanes) for _, _, lanes in self.groups), 40 * 36 + 36)


class IdentityTest(unittest.TestCase):
    def setUp(self):
        self.lane = (SIX[0], "plan_off", 43)

    def test_execution_and_hardware_are_part_of_identity(self):
        base = final_ablation.identity(self.lane, 20000, "thinkpad", "batch108")
        self.assertNotEqual(
            base, final_ablation.identity(self.lane, 20000, "thinkpad", "batch27")
        )
        self.assertNotEqual(
            base, final_ablation.identity(self.lane, 20000, "jeff", "batch108")
        )
        self.assertNotEqual(
            base, final_ablation.identity(self.lane, 10000, "thinkpad", "batch108")
        )

    def test_same_seed_at_two_densities_is_two_cells(self):
        # All three prepare runs probe seeds 511-610, so one seed can be selected at two
        # densities where it is a completely different field. Keyed on the seed alone those
        # collide: resume skips a cell that never ran, and the analysis pairs rows across
        # densities. SIX deliberately reuses seeds to hold this.
        sparse = final_ablation.identity((_map(501, 15), "plan_off", 43), 20000, "h", "b108")
        dense = final_ablation.identity((_map(501, 35), "plan_off", 43), 20000, "h", "b108")
        self.assertNotEqual(sparse, dense)

    def test_same_cell_on_same_branch_is_the_same_key(self):
        self.assertEqual(
            final_ablation.identity(self.lane, 20000, "thinkpad", "batch108"),
            final_ablation.identity((dict(SIX[0]), "plan_off", 43), 20000,
                                    "thinkpad", "batch108"),
        )


class ConfigCacheTest(unittest.TestCase):
    """The seed-collision bug that made the campaign claim nine maps it did not have.

    `_configs` cached each map's config, manifest and arrays under the map seed alone. Seed
    525 was selected at both 15 and 25 pillars -- 492 against 824 occupied cells -- so the
    second density silently reused the first's arrays and 564 rows were labelled with a
    density they were not flown at. Nothing downstream could catch it: every check keys on
    the label, not on the field.
    """

    def test_same_seed_at_two_densities_loads_two_maps(self):
        loaded = []

        def fake_grid_config(run_dir, config_path=None):
            loaded.append(run_dir)
            return (f"config:{run_dir}", f"manifest:{run_dir}", f"arrays:{run_dir}")

        cache = {}
        with unittest.mock.patch.object(final_ablation, "_grid_config", fake_grid_config):
            sparse = final_ablation._configs(cache, _map(525, 15))
            dense = final_ablation._configs(cache, _map(525, 25))
            again = final_ablation._configs(cache, _map(525, 15))

        self.assertNotEqual(sparse, dense)
        self.assertEqual(sparse, again, "the cache must still cache")
        self.assertEqual(len(loaded), 2, f"loaded {loaded}")


class AppendRowsTest(unittest.TestCase):
    def _row(self, arm="plan_off", seed=43):
        row = dict.fromkeys(final_ablation.FIELDS, "")
        row.update({"arm": arm, "map_seed": 500, "seed": seed, "steps": 20000,
                    "hardware": "test", "execution": "batch108"})
        return row

    def test_group_is_written_whole_and_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final.csv"
            from ergodic_control_mppi.experiments.common import ensure_bundle
            bundle_hash = ensure_bundle(output, {"test": True})
            rows = [self._row(seed=s) for s in range(43, 55)]
            for row in rows:
                row.update(bundle_hash=bundle_hash, config_hash="test")
            final_ablation.append_rows(output, rows)
            extra = self._row(arm="plan_3", seed=43)
            extra.update(bundle_hash=bundle_hash, config_hash="test")
            final_ablation.append_rows(output, [extra])
            with output.open(encoding="utf-8", newline="") as stream:
                written = list(csv.DictReader(stream))
            self.assertEqual(len(written), 13)
            self.assertEqual(len(final_ablation.completed(output)), 13)

    def test_stale_header_refuses_rather_than_shifting_columns(self):
        # A DictWriter emits values in FIELDS order whatever the file says, so appending a
        # new column to an old archive shifts every later row by one and looks like it
        # worked. It happened once; this is the guard.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final.csv"
            with output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(final_ablation.FIELDS[:-1])
                writer.writerow(["x"] * (len(final_ablation.FIELDS) - 1))
            with self.assertRaises(SystemExit):
                final_ablation.append_rows(output, [self._row()])


class MapGuardTest(unittest.TestCase):
    def _write_map(self, directory: Path, meta: dict, obs_num: int = 25) -> Path:
        # Under a density_<obs>/ root: the density label comes from the path, and the guard
        # that enforces that is itself under test below.
        run_dir = directory / f"density_{obs_num}" / "maps" / f"map_{meta['map_seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez(
            run_dir / "arrays.npz",
            occupancy=np.zeros((134, 267), dtype=np.float32),
            reachable_mask=np.ones((80, 80), dtype=bool),
            grid=np.zeros((134, 267), dtype=np.float32),
            initial_state=np.array([-15.57, 0.42, 0, 0, 0, 0], dtype=np.float32),
        )
        return run_dir

    def test_pillar_map_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_map(
                Path(directory), {"map_seed": 514, "map_source": "random_forest"}
            )
            entry = final_ablation._check_map(run_dir, 514, 25)
            self.assertEqual(entry["obs_num"], 25)
            self.assertEqual(entry["grid_shape"], [134, 267])

    def test_perlin_map_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_map(
                Path(directory), {"map_seed": 518, "map_source": "perlin3d"}
            )
            with self.assertRaises(SystemExit):
                final_ablation._check_map(run_dir, 518, 25)

    def test_density_label_must_match_the_path(self):
        # The recorder writes no obs_num into a map manifest, so the path is the only
        # witness that a map labelled 35-pillar is not a 15-pillar one under a wrong root.
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_map(
                Path(directory), {"map_seed": 514, "map_source": "random_forest"},
                obs_num=15,
            )
            with self.assertRaises(SystemExit):
                final_ablation._check_map(run_dir, 514, 35)

    def test_duplicate_labels_are_refused(self):
        """Seed 525 qualified at two densities and one field flew under both labels."""
        entries = [_map(513, 15), _map(513, 15)]
        with self.assertRaises(SystemExit):
            final_ablation._assert_distinct(entries)

    def test_identical_fields_under_two_labels_are_refused(self):
        """The failure that actually happened: two labels, one field.

        The digest is the witness the label cannot forge -- it hashes the occupancy array
        itself, so a collision means the same field reached the manifest twice.
        """
        first, second = _map(513, 15), _map(525, 25)
        second["occupancy_digest"] = first["occupancy_digest"]
        with self.assertRaises(SystemExit):
            final_ablation._assert_distinct([first, second])

    def test_equal_cell_counts_are_not_a_duplicate(self):
        """Regression: an integer count over a quantized grid collides by coincidence.

        Maps 513 and 530 of density_15 both have 457 occupied cells and differ in 908 of
        them. Guarding on the count rejected that manifest outright, which would have
        blocked the campaign on a false positive.
        """
        first, second = _map(513, 15), _map(530, 15)
        self.assertEqual(first["occupied_cells"], second["occupied_cells"])
        self.assertNotEqual(first["occupancy_digest"], second["occupancy_digest"])
        final_ablation._assert_distinct([first, second])

    def test_real_rebuilt_516_matches_its_flight(self):
        rebuilt = ROOT / "results" / "uav" / "density_25" / "maps" / "map_516"
        if not (rebuilt / "arrays.npz").exists():
            self.skipTest("516 not rebuilt yet")
        entry = final_ablation._check_map(rebuilt, 516, 25)
        self.assertEqual(entry["occupied_cells"], 882)

    def test_map_with_no_source_key_is_refused(self):
        # results/uav/paper01 is exactly this: a perlin map whose manifest predates the
        # map_source field. It must fail by construction, not by being remembered.
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_map(Path(directory), {"map_seed": 518})
            with self.assertRaises(SystemExit):
                final_ablation._check_map(run_dir, 518, 25)

    def test_real_paper01_is_refused_if_present(self):
        paper01 = ROOT / "results" / "uav" / "paper01"
        if not (paper01 / "manifest.json").exists():
            self.skipTest("paper01 not present")
        with self.assertRaises(SystemExit):
            final_ablation._check_map(paper01, 518, 25)


class LoadMapsTest(unittest.TestCase):
    def _manifest(self, directory: Path, entries: list[dict]) -> Path:
        path = Path(directory) / "maps.json"
        path.write_text(json.dumps({"maps": entries}), encoding="utf-8")
        return path

    def test_six_consistent_maps_load(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                len(final_ablation.load_maps(self._manifest(Path(directory), SIX))), 6
            )

    def test_mismatched_grid_shape_refuses(self):
        odd = dict(SIX[0], grid_shape=[100, 200])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                final_ablation.load_maps(
                    self._manifest(Path(directory), [odd] + SIX[1:])
                )

    def test_mismatched_start_state_refuses(self):
        # run_batch broadcasts one start across lanes, so a second start would silently fly
        # some maps from the wrong place rather than erroring.
        odd = dict(SIX[0], initial_state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                final_ablation.load_maps(
                    self._manifest(Path(directory), [odd] + SIX[1:])
                )


if __name__ == "__main__":
    unittest.main()


class SensitivityMathTest(unittest.TestCase):
    """The Fisher panel's arithmetic, against cases computed by hand."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "report_figures", ROOT / "scripts" / "report_figures.py"
        )
        self.rf = importlib.util.module_from_spec(spec)
        sys.modules["report_figures"] = self.rf
        spec.loader.exec_module(self.rf)

    def _table(self, arm_shift: dict, cells=6, maps=(10, 15, 20), seed=0):
        rng = np.random.default_rng(seed)
        table = {}
        keys = [(obs, 500 + i, 43 + s) for obs in maps for i in range(2)
                for s in range(cells)]
        # The baseline is keyed by width, as `load_final` writes it, so `baseline_for`
        # resolves here the same way it does on the real archive.
        for arm, shift in {"baseline@108": 0.0, **arm_shift}.items():
            table[arm] = {}
            for key in keys:
                table[arm][key] = {
                    "arm": arm, "axis": "x", "value": "1", "steps": "20000", "lanes": "108",
                    "obs_num": str(key[0]), "map_seed": str(key[1]), "seed": str(key[2]),
                    # Both outcomes carry the shift: occupancy_mse is the primary metric
                    # now, and a constant column would make every sensitivity read zero.
                    "fourier_ergodic": str(0.05 * np.exp(-shift + 0.1 * rng.standard_normal())),
                    "occupancy_mse": str(7.3e-07 * np.exp(-shift + 0.1 * rng.standard_normal())),
                    "all_modes_reached": "1", "mode_cycles": "3",
                    "mode_dwell_median_s": "9.0", "in_mode_fraction": "0.4",
                    "path_length_m": "340.0",
                }
        return table

    def test_zero_effect_gives_zero_sensitivity(self):
        table = self._table({"null": 0.0})
        bands, joint = self.rf.sensitivity(table, "null")
        # Only fourier varies in this fixture; a null arm's standardised effect should sit
        # near zero rather than at some arbitrary floor.
        self.assertLess(bands["fourier_ergodic"], 0.5)
        self.assertLess(joint, 0.5)

    def test_bit_identical_arm_has_zero_joint_sensitivity(self):
        table = self._table({"same": 0.0})
        table["same"] = {cell: dict(row) for cell, row in table["baseline@108"].items()}
        bands, joint = self.rf.sensitivity(table, "same")
        self.assertEqual(joint, 0.0)
        self.assertTrue(all(value == 0.0 for value in bands.values()))

    def test_holm_does_not_reject_a_nan_pvalue(self):
        self.assertEqual(self.rf.holm([float("nan"), 0.001]), [False, True])

    def test_standardised_effect_matches_hand_calculation(self):
        table = self._table({"shifted": 0.4})
        arm, base, _ = self.rf.paired_final(table, "shifted", "fourier_ergodic", "log")
        difference = arm - base
        expected = abs(difference.mean() / difference.std(ddof=1))
        bands, _ = self.rf.sensitivity(table, "shifted")
        self.assertAlmostEqual(bands["fourier_ergodic"], expected, places=9)

    def test_sensitivity_is_direction_agnostic(self):
        # It is a magnitude: an arm that halves the metric and one that doubles it are
        # equally influential, which is the right reading for "how much does this move".
        # Built as an exact negation rather than two random draws -- with independent noise
        # the two means are -0.4+eps and +0.4+eps, whose magnitudes genuinely differ.
        table = self._table({"up": 0.0, "down": 0.0}, seed=1)
        for index, key in enumerate(table["baseline@108"]):
            base = float(table["baseline@108"][key]["fourier_ergodic"])
            offset = 0.3 + 0.02 * index
            table["up"][key]["fourier_ergodic"] = str(base * np.exp(offset))
            table["down"][key]["fourier_ergodic"] = str(base * np.exp(-offset))
        self.assertAlmostEqual(
            self.rf.sensitivity(table, "up")[0]["fourier_ergodic"],
            self.rf.sensitivity(table, "down")[0]["fourier_ergodic"],
            places=9,
        )

    def _archive(self, path: Path):
        """A miniature archive with both baseline widths and a seed reused across densities."""
        fields = ["arm", "obs_num", "map_seed", "seed", "lanes", "occupancy_mse",
                  "fourier_ergodic",
                  "all_modes_reached", "mode_cycles", "mode_dwell_median_s",
                  "in_mode_fraction", "path_length_m", "steps", "axis", "value"]
        maps = [(15, 513), (15, 525), (25, 516), (25, 525)]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            # baseline@108 first and baseline@27 last, matching the real file: with a
            # width-blind key the trailing width-27 rows win and serve every arm.
            for arm, lanes, value in (("baseline", 108, 1.0), ("wide", 108, 2.0),
                                      ("narrow", 27, 4.0), ("baseline", 27, 8.0)):
                for obs, map_seed in maps:
                    for seed in range(43, 46):
                        writer.writerow({
                            "arm": arm, "obs_num": obs, "map_seed": map_seed, "seed": seed,
                            "lanes": lanes, "occupancy_mse": value,
                            "fourier_ergodic": value,
                            "all_modes_reached": 1, "mode_cycles": 2,
                            "mode_dwell_median_s": 9.0, "in_mode_fraction": 0.4,
                            "path_length_m": 340.0, "steps": 20000, "axis": "x",
                            "value": "1",
                        })

    def test_load_final_keeps_a_baseline_per_lane_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            self._archive(path)
            table = self.rf.load_final(path)

        self.assertIn("baseline@108", table)
        self.assertIn("baseline@27", table)
        self.assertNotIn("baseline", table)
        # Each arm resolves to the baseline on its own numerical branch. Collapsed to one
        # key, the trailing width-27 rows served everything and `wide` was scored 2.0 vs 8.0
        # instead of 2.0 vs 1.0 -- which is what the real campaign was doing for 41 arms.
        self.assertEqual(self.rf.baseline_for(table, "wide"), "baseline@108")
        self.assertEqual(self.rf.baseline_for(table, "narrow"), "baseline@27")
        arm, base, _ = self.rf.paired_final(table, "wide", "fourier_ergodic")
        self.assertTrue(np.allclose(arm, 2.0) and np.allclose(base, 1.0))
        arm, base, _ = self.rf.paired_final(table, "narrow", "fourier_ergodic")
        self.assertTrue(np.allclose(arm, 4.0) and np.allclose(base, 8.0))

    def test_load_final_keys_cells_by_density_and_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            self._archive(path)
            table = self.rf.load_final(path)

        # The same map seed at two densities is two different fields, so the cell key
        # carries obs_num. Keyed on the seed alone these three would collapse to two.
        keys = {(obs, seed) for obs, seed, _ in table["wide"]}
        self.assertEqual(keys, {(15, 513), (15, 525), (25, 516), (25, 525)})
        self.assertEqual(len(self.rf.per_map_effects(table, "wide")), 4)

    def test_typical_reference_is_the_per_cell_median_arm(self):
        # Why the reference exists at all: paired against itself the baseline is exactly
        # zero on every cell, so under the default reference it cannot be drawn as a column.
        table = self._table({"a": 0.3, "b": -0.3})
        table["baseline@27"] = {
            cell: {**row, "lanes": "27"}
            for cell, row in table["baseline@108"].items()
        }
        arm, base, _ = self.rf.paired_final(table, "baseline@108", "fourier_ergodic")
        self.assertTrue(np.array_equal(arm, base))

        self.rf.add_typical_reference(table)
        cell = next(iter(table["baseline@108"]))
        self.assertAlmostEqual(
            float(table[self.rf.TYPICAL][cell]["fourier_ergodic"]),
            float(np.median([float(table[a][cell]["fourier_ergodic"])
                             for a in ("baseline@108", "a", "b")])),
            places=12,
        )
        # And with it the baseline becomes a real column. Bracket it deterministically --
        # one arm strictly better on every cell, one strictly worse -- so the median arm is
        # the baseline itself and its drawn effect is exactly zero rather than approximately.
        for cell in table["baseline@108"]:
            value = float(table["baseline@108"][cell]["fourier_ergodic"])
            table["a"][cell]["fourier_ergodic"] = repr(value * 0.5)
            table["b"][cell]["fourier_ergodic"] = repr(value * 2.0)
        table.pop(self.rf.TYPICAL)
        self.rf.add_typical_reference(table)
        arm, base, cells = self.rf.paired_final(table, "baseline@108", "fourier_ergodic",
                                                reference=self.rf.TYPICAL)
        self.assertEqual(len(cells), len(table["baseline@108"]))
        self.assertTrue(np.allclose(arm, base))
        # The bracketing arms are drawn at exactly +/-1 log2 unit, which is what the dot
        # matrix colours: the reference is a yardstick, not one of the runs.
        worse, reference, _ = self.rf.paired_final(table, "b", "fourier_ergodic",
                                                   reference=self.rf.TYPICAL)
        self.assertTrue(np.allclose(np.log2(reference / worse), -1.0))

    def test_joint_is_not_bounded_by_the_marginal_sum(self):
        # Recorded because the opposite was asserted while designing this panel and the
        # synthetic data disproved it: correlated outcomes can make the Mahalanobis
        # distance exceed the sum of marginals, so neither bounds the other.
        table = self._table({"shifted": 0.4})
        bands, joint = self.rf.sensitivity(table, "shifted")
        self.assertGreater(joint, 0.0)
        self.assertGreater(sum(bands.values()), 0.0)

    def test_per_map_effects_are_standardised_by_their_own_noise(self):
        table = self._table({"shifted": 0.4})
        raw = self.rf.per_map_effects(table, "shifted", standardize=False)
        standard = self.rf.per_map_effects(table, "shifted")
        self.assertEqual(len(raw), 6)
        # 0.4 in log units is ~0.58 in log2; standardising must not change the sign, and
        # over 6 seeds it should push a real effect well past 1 sigma.
        for key in raw:
            self.assertGreater(raw[key], 0.0)
            self.assertGreater(standard[key], 1.0)

    def test_one_map_effect_is_not_read_as_consistent(self):
        # The 516 trap in miniature: a large effect on exactly one of six maps must not
        # produce a strip that looks consistent, which is what a fixed neutral band did.
        #
        # Thresholded at 2 sigma, the strip's first colour edge, not at 1: a 1-sigma edge
        # fires on 32% of null cells, so two of the five untouched maps would light up by
        # chance and the arm would read as patchy-but-real. That is the whole reason the
        # edge sits at 2.
        table = self._table({"trap": 0.0})
        for key, row in table["trap"].items():
            if (key[0], key[1]) == (15, 501):
                row["occupancy_mse"] = str(float(row["occupancy_mse"]) * 0.4)
        standard = self.rf.per_map_effects(table, "trap")
        resolved = {k for k, v in standard.items() if abs(v) > 2.0}
        self.assertIn((15, 501), resolved)
        self.assertLessEqual(len(resolved), 2)
        self.assertGreater(standard[(15, 501)], 10.0)


class StepBudgetTest(unittest.TestCase):
    """The one branch in `fig_step_budget`: the residual is signed.

    XLA fuses the whole step, so the stages timed in isolation can sum to *more* than the
    fused total. A pie cannot draw a negative wedge, and silently clamping one to zero
    would invent cost that does not exist -- so a negative residual must never reach the
    ring at all.
    """

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "report_figures", ROOT / "scripts" / "report_figures.py"
        )
        self.rf = importlib.util.module_from_spec(spec)
        sys.modules["report_figures"] = self.rf
        spec.loader.exec_module(self.rf)

    def _report(self, total: float) -> dict:
        stages = {"rollouts_KT": 3.0, "memory_P2": 0.8, "sample_epsilon": 0.6,
                  "plan_T2": 0.4, "attraction_T": 0.1}  # five stages, so five wedges
        accounted = sum(stages.values())
        return {"stages": {
            "shape": {"K": 250, "T": 350, "P": 825},
            "device": "cuda:0",
            "stages": {k: {"ms_median": v} for k, v in stages.items()},
            "accounted_ms": accounted,
            "total_ms": total,
            "residual_ms": total - accounted,
        }}

    def test_table_keeps_fused_total_separate_from_stage_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "timing.json"
            data = self._report(4.5)
            data["endtoend"] = {"with_memory": {"ms_per_step": 6.2}}
            report.write_text(json.dumps(data))
            with unittest.mock.patch.object(self.rf, "save", lambda figure, path: figure):
                figure = self.rf.fig_step_budget(report, Path(directory) / "f.pdf")
            values = [cell.get_text().get_text()
                      for cell in figure.axes[0].tables[0].get_celld().values()]
            self.assertIn("4.500", values)
            self.assertIn("6.200", values)
            self.assertIn("Fused MPPI step", values)
