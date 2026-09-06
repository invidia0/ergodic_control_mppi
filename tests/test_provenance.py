"""Small resume regressions: changed inputs must never suppress a requested run."""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments import baselines
from ergodic_control_mppi.experiments.common import ensure_bundle, fingerprint, verified_rows
from tests.helpers import write_small_config
from tests.test_final_ablation import final_ablation, SIX
from scripts import theory_audit


class ProvenanceTest(unittest.TestCase):
    def test_resolved_dependencies_and_method_specific_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write_small_config(Path(directory)))
        cfg = baselines.BaselineConfig()
        arrays = {"initial_state": np.zeros(6), "occupancy": np.zeros((3, 3))}
        horizon = replace(config, controller=replace(config.controller,
                          mppi=replace(config.controller.mppi, horizon=150)))
        for method in baselines.METHODS:
            original = cfg.fingerprint_for(method, config, arrays, {})
            changed = cfg.fingerprint_for(method, horizon, arrays, {})
            self.assertEqual(original == changed, method != "ours")
            for variant in (
                replace(config, controller=replace(config.controller,
                    model=replace(config.controller.model, delta_t=0.1))),
                replace(config, controller=replace(config.controller,
                    workspace=replace(config.controller.workspace, safe_distance=0.9))),
            ):
                self.assertNotEqual(original, cfg.fingerprint_for(method, variant, arrays, {}))
            changed_arrays = dict(arrays, initial_state=np.ones(6))
            self.assertNotEqual(original, cfg.fingerprint_for(method, config, changed_arrays, {}))
            longer = baselines.BaselineConfig(steps=cfg.steps + 1)
            self.assertNotEqual(original, longer.fingerprint_for(method, config, arrays, {}))
        reference = replace(config, controller=replace(config.controller,
                            field=replace(config.controller.field, plan_gain=10)))
        self.assertNotEqual(cfg.fingerprint_for("ours", config, arrays, {}),
                            cfg.fingerprint_for("ours", reference, arrays, {}))

    def test_bundle_resume_invalidation_legacy_duplicates_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            record = {"configurations": {}, "source": "a", "width": 9, "steps": 20000}
            digest = ensure_bundle(path, record)
            self.assertEqual(ensure_bundle(path, dict(record)), digest)
            for key, value in (("source", "b"), ("width", 36), ("steps", 750000)):
                with self.assertRaisesRegex(ValueError, "incompatible bundle"):
                    ensure_bundle(path, dict(record, **{key: value}))
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, ["seed", "bundle_hash"])
                writer.writeheader()
                writer.writerow({"seed": 43, "bundle_hash": digest})
            self.assertEqual(len(verified_rows(path, ("seed",))), 1)
            with path.open("a") as stream:
                stream.write(f"43,{digest}\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                verified_rows(path, ("seed",))
            ensure_bundle(path, dict(record, source="b"), overwrite=True)
            self.assertFalse(path.exists())
            path.with_suffix(".manifest.json").unlink()
            path.write_text("seed\n43\n")
            self.assertEqual(len(verified_rows(path, ("seed",), legacy=True)), 1)
            with self.assertRaisesRegex(ValueError, "legacy provenance"):
                ensure_bundle(path, record)
            with self.assertRaisesRegex(ValueError, "unknown bundle"):
                verified_rows(path, ("seed",))

    def test_baseline_matching_rows_resume_without_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_small_config(Path(directory))
            output = Path(directory) / "baselines.csv"
            cfg = baselines.BaselineConfig(steps=2)
            score = {"steps": 2, "collisions": 0, "fourier_ergodic": 0.1}
            with patch.object(baselines, "run_method", return_value=np.zeros((2, 6))) as run, \
                 patch("ergodic_control_mppi.experiments.uav_pillar_tuning.score_run", return_value=score):
                baselines.run_tier("open", ["ours"], [43], cfg, str(config_path), Path("unused"), output)
                self.assertEqual(run.call_count, 1)
                baselines.run_tier("open", ["ours"], [43], cfg, str(config_path), Path("unused"), output)
                self.assertEqual(run.call_count, 1)
                cfg.steps = 3
                with self.assertRaisesRegex(ValueError, "incompatible bundle"):
                    baselines.run_tier("open", ["ours"], [43], cfg, str(config_path), Path("unused"), output)
                self.assertEqual(run.call_count, 1)

    def test_profile_hash_is_part_of_ablation_identity(self):
        lane = (SIX[0], "baseline", 43)
        self.assertNotEqual(final_ablation.identity(lane, 20000, "cpu", "batch9", "a"),
                            final_ablation.identity(lane, 20000, "cpu", "batch9", "b"))

    def test_explicit_initial_starts_keep_all_seeds_and_use_map_resolution(self):
        arrays = {"initial_state": np.zeros(6), "reachable_mask": np.ones((4, 5), bool),
                  "grid_origin": np.array([3., 6.]), "grid_resolution": 0.2}
        args = SimpleNamespace(inits=4, seeds=12, start_index=0)
        config = SimpleNamespace(controller=SimpleNamespace(workspace=SimpleNamespace(
            x_limits=[3., 4.], y_limits=[6., 6.8])))
        starts = []
        for index in range(4):
            args.start_index = index
            positions = [theory_audit.dispersed_initial_state(arrays, SIX[0], s, args, config) for s in range(12)]
            np.testing.assert_array_equal(positions, np.repeat([positions[0]], 12, axis=0))
            starts.append(tuple(positions[0][:2]))
        self.assertEqual(len(set(starts)), 4)
        self.assertTrue(all(3 <= x <= 4 and 6 <= y <= 6.8 for x, y in starts))
        args.start_index = 4
        with self.assertRaisesRegex(ValueError, "start-index"):
            theory_audit.dispersed_initial_state(arrays, SIX[0], 0, args, config)

    def test_equal_cost_witness_has_no_temperature_derivative(self):
        import jax
        import jax.numpy as jnp
        from ergodic_control_mppi.mppi.core import adapt_temperature

        with tempfile.TemporaryDirectory() as directory:
            params = load_config(write_small_config(Path(directory))).controller
        update = lambda costs: adapt_temperature(jnp.asarray(1.0), jax.nn.softmax(-costs), params)
        derivative = jax.grad(update)(jnp.zeros(params.mppi.samples))
        np.testing.assert_allclose(derivative, 0, atol=1e-8)

    def test_tv_stage_requires_both_outputs_and_matching_receipt(self):
        from scripts.run_t150_revision import check_artifacts

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tv.csv"
            split = Path(directory) / "tv_split.csv"
            receipt = output.with_suffix(".artifacts.json")
            for path in (output, split):
                path.write_text("bundle_hash\na\n")
            receipt.write_text(json.dumps(theory_audit.artifact_digests([output, split])))
            check_artifacts([output, split, receipt], 1)
            split.write_text("bundle_hash\nb\n")
            with self.assertRaisesRegex(ValueError, "receipt"):
                check_artifacts([output, split, receipt], 1)

    def test_timing_overwrite_preserves_old_report_until_commit(self):
        from ergodic_control_mppi.experiments.timing import _commit_timing_output, _timing_output

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "timing.json"
            output.write_text("old")
            output.with_suffix(".manifest.json").write_text("old manifest")
            staged, _ = _timing_output(output, {"session": "new"}, True)
            self.assertEqual(output.read_text(), "old")
            staged.write_text("new")
            _commit_timing_output(staged, output)
            self.assertEqual(output.read_text(), "new")

    def test_hierarchical_effects_keep_pairing_and_render(self):
        from scripts.report_figures import paired_effect_summary, fig_final_ablation
        table = {"baseline@36": {}, "memory_off": {}, "T_350": {}}
        for density in (10, 15, 20):
            for map_seed in (1, 2):
                for seed in range(6):
                    cell = (density, map_seed, seed)
                    value = 1 + seed + map_seed
                    table["baseline@36"][cell] = {"occupancy_mse": value, "lanes": "36", "axis": "-"}
                    for arm, axis in (("memory_off", "memory_gain"), ("T_350", "T")):
                        table[arm][cell] = {"occupancy_mse": value / 2, "lanes": "36", "axis": axis}
        summary = paired_effect_summary(table)
        for row in summary["arms"]:
            self.assertEqual(row["median"], 1)
            self.assertEqual(row["interval"], [1, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effects.pdf"
            fig_final_ablation(table, path)
            self.assertTrue(path.is_file())
        del table["memory_off"][(10, 1, 0)]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            paired_effect_summary(table)

    def test_branch_gate_actually_uses_odd_requested_width(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_small_config(Path(directory))
            config = load_config(config_path)
            widths = []
            def fake_batch(params, initial, controls, keys, **kwargs):
                widths.append(len(keys))
                return SimpleNamespace(path=np.zeros((len(keys), 2, 6)))
            args = SimpleNamespace(config=str(config_path), device="cpu", verify_lanes=9,
                                   seeds=6, steps=2, output=None)
            with patch.object(final_ablation, "_configs", return_value=(config, {}, {"initial_state": np.zeros(6)})), \
                 patch.object(final_ablation, "run_batch", side_effect=fake_batch), \
                 patch.object(final_ablation.jax, "jit", side_effect=lambda function, **kw: function):
                self.assertTrue(final_ablation.verify_branch(SIX, args))
            self.assertEqual(widths, [9, 9])

    def test_capture_field_source_starts_at_frozen_current_position(self):
        from scripts.mechanism_captures import capture
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_small_config(Path(directory))
            data = capture(str(config_path), {}, 43, steps=4, stride=1, freeze=2)
        np.testing.assert_array_equal(data["plan"][0], data["position"])
        np.testing.assert_array_equal(data["position"], data["positions"][1])

    def test_campaign_design_and_receipts(self):
        from scripts.run_t150_revision import stages, check_artifacts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduled = {name: (command, count, paths) for name, command, count, paths in stages(root)}
            self.assertEqual(scheduled["ablation_clutter"][1], 1476)
            self.assertEqual(scheduled["ablation_open"][1], 276)
            self.assertEqual(len([n for n in scheduled if n.startswith("nt_")]), 6)
            for i in range(4):
                command = scheduled[f"inits_start{i}"][0]
                self.assertEqual(command[command.index("--start-index") + 1], str(i))
                self.assertEqual(command[command.index("--seeds") + 1], "12")
            output = root / "gate.json"
            output.write_text('{"passed": false}')
            with self.assertRaisesRegex(ValueError, "verification failed"):
                check_artifacts([output], 0)
            output.write_text('{"passed": true}')
            self.assertIn("gate.json", check_artifacts([output], 0))


if __name__ == "__main__":
    unittest.main()
