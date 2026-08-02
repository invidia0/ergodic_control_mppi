"""Regressions for the deployment summary schema and the acceptance report."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ergodic_control_mppi.deploy.summary import (
    SUMMARY_FIELDS,
    append_summary,
    compute_row,
    steps_to_threshold,
    timing_stats,
    tracking_stats,
)
from ergodic_control_mppi.experiments.uav_report import (
    acceptance,
    build_report,
    read_rows,
    screen_table,
    split_modes,
)

# A row that satisfies every acceptance criterion; tests mutate one field at a time.
GOOD = {
    "run_id": "r1",
    "profile": "baseline",
    "mode": "uav",
    "seed": 43,
    "map_seed": 511,
    "map_fill": 0.02,
    "steps": 5000,
    "occupancy_mse": 1.0e-6,
    "fourier_ergodic": 0.25,
    "steps_to_threshold": 100,
    "mode_visits": 6,
    "mode_switches": 5,
    "mode_revisits": 3,
    "mode_dwell_median_s": 4.0,
    "mode_dwell_total_s": 24.0,
    "mode_transitions": 5,
    "mode_cycles": 1,
    "first_all_modes_s": 30.0,
    "in_mode_fraction": 0.5,
    "collisions": 0,
    "min_clearance_m": 1.4,
    "guard_interventions": 1,
    "guard_fraction": 0.001,
    "guard_duration_s": 0.1,
    "max_speed_mps": 1.9,
    "pos_rmse_m": 0.12,
    "pos_p95_m": 0.19,
    "vel_rmse_mps": 0.2,
    "vel_p95_mps": 0.4,
    "compile_s": 12.0,
    "step_p50_ms": 8.0,
    "step_p95_ms": 11.0,
    "step_p99_ms": 12.0,
    "step_max_ms": 14.0,
    "deadline_miss_fraction": 0.0,
    "achieved_rate_hz": 50.0,
    "wall_seconds": 100.0,
    "real_time_factor": 1.0,
    "run_hash": "abc",
    "config_hash": "def",
    "git_sha": "0123",
    "seed_controller": 43,
    "jax_version": "0.11.0",
    "ros_distro": "jazzy",
    "device": "gpu",
}


def _row(**overrides):
    return {**GOOD, **overrides}


def _write(path: Path, rows):
    for row in rows:
        append_summary(path, row)


class SchemaTest(unittest.TestCase):
    def test_header_is_exactly_the_frozen_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            append_summary(path, _row())
            with path.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(next(csv.reader(stream)), SUMMARY_FIELDS)

    def test_unknown_field_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                append_summary(Path(directory) / "s.csv", _row(bogus=1))

    def test_missing_fields_are_written_blank_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            append_summary(path, {"run_id": "r", "mode": "uav"})
            rows = read_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["occupancy_mse"], "")

    def test_read_rows_rejects_a_drifted_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.write_text("run_id,mode\nr,uav\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(path)


class StatisticsTest(unittest.TestCase):
    def test_timing_percentiles_and_misses(self):
        stats = timing_stats(np.array([1.0, 2.0, 3.0, 100.0]), deadline_ms=16.0)
        self.assertEqual(stats["step_max_ms"], 100.0)
        self.assertAlmostEqual(stats["deadline_miss_fraction"], 0.25)

    def test_timing_on_empty_input_is_nan_not_zero(self):
        stats = timing_stats(np.zeros(0), deadline_ms=16.0)
        self.assertTrue(np.isnan(stats["step_p99_ms"]))

    def test_tracking_error_of_a_constant_offset(self):
        times = np.array([0.0, 0.02, 0.04])
        actual = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        commanded = actual + np.array([0.3, 0.0])
        stats = tracking_stats(times, actual, times, commanded)
        self.assertAlmostEqual(stats["pos_rmse_m"], 0.3, places=6)
        # A constant offset means identical velocities, so velocity error is zero.
        self.assertAlmostEqual(stats["vel_rmse_mps"], 0.0, places=6)

    def test_streams_are_aligned_by_time_not_by_index(self):
        """The real failure: odometry runs faster than the control loop.

        Differencing the two by index compares unrelated instants and reports an error the
        size of the workspace. Here a perfectly tracked straight line is sampled at 100 Hz
        against 50 Hz setpoints; the error must be zero, not the index skew.
        """
        actual_times = np.arange(0, 2.0, 0.01)
        commanded_times = np.arange(0, 2.0, 0.02)
        line = lambda t: np.column_stack((5.0 * t, np.zeros_like(t)))
        stats = tracking_stats(
            actual_times, line(actual_times), commanded_times, line(commanded_times)
        )
        self.assertAlmostEqual(stats["pos_rmse_m"], 0.0, places=9)
        self.assertAlmostEqual(stats["vel_rmse_mps"], 0.0, places=6)

    def test_offset_clocks_do_not_extrapolate(self):
        """Only the overlapping window is scored, so no sample meets an invented setpoint."""
        actual_times = np.arange(10.0, 12.0, 0.01)
        commanded_times = np.arange(11.0, 13.0, 0.02)
        line = lambda t: np.column_stack((2.0 * t, np.zeros_like(t)))
        stats = tracking_stats(
            actual_times, line(actual_times), commanded_times, line(commanded_times)
        )
        self.assertAlmostEqual(stats["pos_rmse_m"], 0.0, places=9)

    def test_disjoint_windows_report_nan(self):
        stats = tracking_stats(
            np.array([0.0, 1.0]), np.zeros((2, 2)), np.array([50.0, 51.0]), np.zeros((2, 2))
        )
        self.assertTrue(np.isnan(stats["pos_rmse_m"]))

    def test_steps_to_threshold_ignores_a_transient_dip(self):
        # Dips under the threshold at index 1, then rises again before settling.
        series = np.array([10.0, 1.0, 10.0, 1.1, 1.0])
        self.assertEqual(steps_to_threshold(series, stride=1, factor=1.5), 3.0)


class ComputeRowTest(unittest.TestCase):
    def test_collisions_count_episodes_not_samples(self):
        # A 4x4 m map at 1 m cells with one obstacle; the path scrapes it for 3 samples.
        occupancy = np.zeros((4, 4), dtype=bool)
        occupancy[2, 2] = True
        positions = np.array([[0.5, 0.5], [2.5, 2.5], [2.5, 2.5], [2.5, 2.5], [0.5, 0.5]])
        row = compute_row(
            identity={"run_id": "r", "mode": "uav"},
            positions=positions,
            target_grid=np.full((8, 8), 1.0 / 64.0),
            x_limits=(0.0, 4.0),
            y_limits=(0.0, 4.0),
            reachable_mask=np.ones((8, 8), dtype=bool),
            gmm_means=np.array([[2.0, 2.0]]),
            gmm_inverses=np.eye(2)[None],
            delta_t=0.02,
            occupancy=occupancy,
            grid_origin=(0.0, 0.0),
            grid_resolution=1.0,
            robot_radius=0.30,
            guard_states=np.array(["pass", "brake", "brake", "pass", "brake"]),
            guard_period=0.01,
            speeds=np.array([1.0, 2.0, 1.0, 0.5, 0.0]),
            actual_times=np.arange(5) * 0.02,
            commanded_times=np.arange(5) * 0.02,
            commanded=positions,
            step_ms=np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
            deadline_ms=16.0,
            wall_seconds=1.0,
            odometry_seconds=1.0,
            control_seconds=0.1,
        )
        self.assertEqual(row["collisions"], 1)
        self.assertEqual(row["guard_interventions"], 2)
        # Guard duration integrates at the guard's rate, not the controller's.
        self.assertAlmostEqual(row["guard_duration_s"], 3 * 0.01, places=9)
        self.assertAlmostEqual(row["max_speed_mps"], 2.0)
        self.assertEqual(set(row) - set(SUMMARY_FIELDS), set())


class SamplePeriodTest(unittest.TestCase):
    """Dwell is timed by the spacing of the position samples, not the control period.

    Online, positions come from odometry running ~20x faster than the control loop. Timing
    them with the control period reported a 175 s dwell inside a 160 s flight.
    """

    def _row(self, count, duration):
        positions = np.tile([2.0, 2.0], (count, 1))  # parked on the single mode
        return compute_row(
            identity={"run_id": "r", "mode": "uav"},
            positions=positions,
            target_grid=np.full((8, 8), 1.0 / 64.0),
            x_limits=(0.0, 4.0),
            y_limits=(0.0, 4.0),
            reachable_mask=np.ones((8, 8), dtype=bool),
            gmm_means=np.array([[2.0, 2.0]]),
            gmm_inverses=np.eye(2)[None],
            delta_t=0.02,
            occupancy=np.zeros((4, 4), dtype=bool),
            grid_origin=(0.0, 0.0),
            grid_resolution=1.0,
            robot_radius=0.30,
            guard_states=np.full(count, "pass"),
            guard_period=0.01,
            speeds=np.zeros(count),
            actual_times=np.linspace(0.0, duration, count),
            commanded_times=np.linspace(0.0, duration, count),
            commanded=positions,
            step_ms=np.full(count, 5.0),
            deadline_ms=16.0,
            wall_seconds=duration,
            odometry_seconds=duration,
            control_seconds=duration,
        )

    def test_dwell_cannot_exceed_the_run(self):
        # 10 s of flight sampled at 1000 Hz, the real odometry rate. Each sample counts as
        # one period, so a fully-dwelling run overhangs the span by exactly one period.
        duration, count = 10.0, 10_000
        row = self._row(count=count, duration=duration)
        period = duration / (count - 1)
        self.assertLessEqual(row["mode_dwell_total_s"], duration + period + 1e-9)
        self.assertGreater(row["mode_dwell_total_s"], duration - period)
        self.assertAlmostEqual(row["mode_dwell_median_s"], duration, places=1)

    def test_same_flight_scores_the_same_at_any_sample_rate(self):
        fast = self._row(count=10_000, duration=10.0)
        slow = self._row(count=500, duration=10.0)
        self.assertAlmostEqual(
            fast["mode_dwell_median_s"], slow["mode_dwell_median_s"], places=1
        )
        self.assertAlmostEqual(fast["in_mode_fraction"], slow["in_mode_fraction"], places=6)


class AcceptanceTest(unittest.TestCase):
    def test_all_criteria_pass_on_good_data(self):
        results = acceptance([_row()], [_row(mode="ideal")])
        self.assertTrue(all(entry["status"] == "PASS" for entry in results), results)
        self.assertTrue(all(entry["measured"] for entry in results))

    def test_a_collision_flips_exactly_one_criterion(self):
        baseline = acceptance([_row()], [_row(mode="ideal")])
        broken = acceptance([_row(collisions=1)], [_row(mode="ideal")])
        failed = [
            entry["criterion"]
            for entry, before in zip(broken, baseline)
            if entry["status"] != before["status"]
        ]
        self.assertEqual(failed, ["zero collisions in every run"])

    def test_a_slow_step_flips_exactly_one_criterion(self):
        baseline = acceptance([_row()], [_row(mode="ideal")])
        broken = acceptance([_row(step_p99_ms=20.0)], [_row(mode="ideal")])
        failed = [
            entry["criterion"]
            for entry, before in zip(broken, baseline)
            if entry["status"] != before["status"]
        ]
        self.assertEqual(failed, ["GPU p99 MPPI time < 16 ms"])

    def test_missing_data_fails_rather_than_passes(self):
        results = acceptance([], [])
        self.assertTrue(all(entry["status"] == "FAIL" for entry in results))

    def test_never_reaching_all_modes_fails(self):
        results = acceptance(
            [_row(first_all_modes_s="", mode_cycles=0)], [_row(mode="ideal")]
        )
        statuses = {entry["criterion"]: entry["status"] for entry in results}
        self.assertEqual(statuses["at least one completed all-mode cycle"], "FAIL")


class PairingTest(unittest.TestCase):
    def test_unpaired_rows_are_ignored(self):
        rows = [_row(run_id="a"), _row(run_id="a", mode="ideal"), _row(run_id="b")]
        uav, ideal = split_modes(rows)
        self.assertEqual([row["run_id"] for row in uav], ["a"])
        self.assertEqual(len(ideal), 1)


class ScreenTest(unittest.TestCase):
    def test_baseline_is_retained_within_one_iqr(self):
        rows = [
            _row(run_id="b1", profile="baseline", seed=43, occupancy_mse=1.0e-6),
            _row(run_id="b2", profile="baseline", seed=44, occupancy_mse=3.0e-6),
            # Marginally better median, well inside the baseline IQR: not a real win.
            _row(run_id="c1", profile="fast", seed=43, occupancy_mse=0.9e-6),
            _row(run_id="c2", profile="fast", seed=44, occupancy_mse=2.9e-6),
        ]
        selected = {e["profile"]: e["selected"] for e in screen_table(rows)}
        self.assertEqual(selected["baseline"], "yes")
        self.assertEqual(selected["fast"], "")

    def test_a_clear_winner_is_selected(self):
        rows = [
            _row(run_id="b1", profile="baseline", seed=43, occupancy_mse=1.0e-5),
            _row(run_id="b2", profile="baseline", seed=44, occupancy_mse=1.1e-5),
            _row(run_id="c1", profile="fast", seed=43, occupancy_mse=1.0e-7),
            _row(run_id="c2", profile="fast", seed=44, occupancy_mse=1.1e-7),
        ]
        selected = {e["profile"]: e["selected"] for e in screen_table(rows)}
        self.assertEqual(selected["fast"], "yes")

    def test_unsafe_arm_is_not_shortlisted(self):
        rows = [_row(run_id="x1", profile="risky", collisions=1)]
        self.assertEqual(screen_table(rows)[0]["shortlisted"], "no")


class ReportTest(unittest.TestCase):
    def test_report_renders_and_states_a_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _write(path, [_row(), _row(mode="ideal")])
            report = build_report(read_rows(path))
        self.assertIn("**Verdict: ACCEPTED**", report)
        self.assertIn("## Acceptance checklist", report)
        self.assertIn("## Paired UAV vs ideal", report)
        self.assertIn("## Screen", report)
        self.assertNotIn("| n/a |", report.split("## Screen")[0])

    def test_failing_run_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            _write(path, [_row(collisions=2), _row(mode="ideal")])
            report = build_report(read_rows(path))
        self.assertIn("NOT ACCEPTED", report)


if __name__ == "__main__":
    unittest.main()
