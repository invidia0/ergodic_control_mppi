"""Turn the deployment summary CSV into an auto-evaluated acceptance report.

Reads only ``summary.csv``: no GPU, no re-running, so a report can always be regenerated
from an archived run. Every acceptance criterion renders with the number it was judged on,
which is the point -- the paper claim should be a table read, not a manual audit.

    python -m ergodic_control_mppi.experiments.uav_report \\
        --summary results/uav/summary.csv --output results/uav/report.md
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ergodic_control_mppi.experiments.analyze import bootstrap_ci
from ergodic_control_mppi.experiments.common import prepare_outputs
from ergodic_control_mppi.deploy.summary import SUMMARY_FIELDS

# Metrics compared UAV against ideal, and whether lower is better.
PAIRED_METRICS = [
    ("occupancy_mse", True),
    ("fourier_ergodic", True),
    ("in_mode_fraction", False),
    ("mode_dwell_median_s", False),
    ("mode_cycles", False),
]
DEGRADATION_LIMIT = 0.10
POINT_LIMIT = 0.10


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read the summary CSV, checking it still matches the frozen schema."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError(
                f"{path} does not match the summary schema; "
                f"got {reader.fieldnames}, expected {SUMMARY_FIELDS}"
            )
        return list(reader)


def _numbers(rows: list[dict[str, str]], field: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (TypeError, ValueError):
            continue
    return np.asarray(values, dtype=np.float64)


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.median(values)),
        float(np.percentile(values, 75) - np.percentile(values, 25)),
    )


def split_modes(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Split rows into UAV and ideal, keeping only run ids present on both sides."""
    by_mode = defaultdict(dict)
    for row in rows:
        by_mode[row["run_id"]][row["mode"]] = row
    paired = [entry for entry in by_mode.values() if "uav" in entry and "ideal" in entry]
    return [entry["uav"] for entry in paired], [entry["ideal"] for entry in paired]


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"criterion": name, "status": "PASS" if passed else "FAIL", "measured": detail}


def acceptance(uav: list[dict], ideal: list[dict]) -> list[dict[str, Any]]:
    """Evaluate every acceptance criterion against the paired rows.

    A criterion with no data to judge it on fails rather than passing silently: an absent
    measurement is not evidence of safety.
    """
    results = []
    if not uav:
        return [check("paired runs present", False, "no run id appears as both uav and ideal")]

    collisions = _numbers(uav, "collisions")
    results.append(
        check(
            "zero collisions in every run",
            collisions.size > 0 and bool((collisions == 0).all()),
            f"max {int(collisions.max())} over {collisions.size} runs"
            if collisions.size
            else "no data",
        )
    )
    visits = _numbers(uav, "mode_visits")
    cycles = _numbers(uav, "mode_cycles")
    first_all = _numbers(uav, "first_all_modes_s")
    results.append(
        check(
            "every mode visited (all modes reached at least once)",
            first_all.size > 0 and bool(np.isfinite(first_all).all()),
            f"{int(np.isfinite(first_all).sum())}/{first_all.size} runs reached all modes; "
            f"median visits {np.median(visits) if visits.size else float('nan'):.1f}",
        )
    )
    results.append(
        check(
            "at least one completed all-mode cycle",
            cycles.size > 0 and bool((cycles >= 1).all()),
            f"min {cycles.min() if cycles.size else float('nan'):.0f} cycles",
        )
    )

    for field, limit_kind in (("occupancy_mse", "ratio"), ("fourier_ergodic", "ratio")):
        uav_values, ideal_values = _numbers(uav, field), _numbers(ideal, field)
        uav_median, _ = _median_iqr(uav_values)
        ideal_median, ideal_iqr = _median_iqr(ideal_values)
        degradation = (uav_median - ideal_median) / ideal_median if ideal_median else np.nan
        within_iqr = abs(uav_median - ideal_median) <= ideal_iqr
        results.append(
            check(
                f"{field} degradation <= 10% or within the ideal seed IQR",
                bool(degradation <= DEGRADATION_LIMIT or within_iqr),
                f"uav {uav_median:.4g} vs ideal {ideal_median:.4g} "
                f"({degradation:+.1%}, ideal IQR {ideal_iqr:.4g})",
            )
        )

    for field in ("in_mode_fraction", "mode_dwell_median_s"):
        uav_values, ideal_values = _numbers(uav, field), _numbers(ideal, field)
        uav_median, _ = _median_iqr(uav_values)
        ideal_median, ideal_iqr = _median_iqr(ideal_values)
        gap = uav_median - ideal_median
        results.append(
            check(
                f"{field} within 10 points or one ideal IQR",
                bool(abs(gap) <= POINT_LIMIT or abs(gap) <= ideal_iqr),
                f"uav {uav_median:.4g} vs ideal {ideal_median:.4g} "
                f"(gap {gap:+.4g}, ideal IQR {ideal_iqr:.4g})",
            )
        )

    for field, limit, comparison, label in (
        ("step_p99_ms", 16.0, "max", "GPU p99 MPPI time < 16 ms"),
        ("deadline_miss_fraction", 0.001, "max", "deadline misses < 0.1%"),
        ("guard_fraction", 0.01, "max", "guard intervention < 1%"),
    ):
        values = _numbers(uav, field)
        worst = float(values.max()) if values.size else float("nan")
        results.append(
            check(label, values.size > 0 and worst < limit, f"worst {worst:.4g} (limit {limit})")
        )

    for field, low, high, label in (
        ("achieved_rate_hz", 49.0, 51.0, "achieved control rate 49-51 Hz"),
        ("real_time_factor", 0.98, 1.02, "real-time factor 0.98-1.02"),
    ):
        values = _numbers(uav, field)
        inside = values.size > 0 and bool(((values >= low) & (values <= high)).all())
        results.append(
            check(
                label,
                inside,
                f"range {values.min():.4g}-{values.max():.4g}" if values.size else "no data",
            )
        )
    return results


def paired_table(uav: list[dict], ideal: list[dict]) -> list[dict[str, Any]]:
    """Median, IQR, delta and a paired bootstrap CI for each compared metric."""
    table = []
    for field, lower_is_better in PAIRED_METRICS:
        uav_values, ideal_values = _numbers(uav, field), _numbers(ideal, field)
        uav_median, uav_iqr = _median_iqr(uav_values)
        ideal_median, ideal_iqr = _median_iqr(ideal_values)
        count = min(uav_values.size, ideal_values.size)
        differences = (
            uav_values[:count] - ideal_values[:count] if count else np.zeros(0)
        )
        low, high = bootstrap_ci(differences)
        table.append(
            {
                "metric": field,
                "better": "lower" if lower_is_better else "higher",
                "uav": f"{uav_median:.4g} ± {uav_iqr:.3g}",
                "ideal": f"{ideal_median:.4g} ± {ideal_iqr:.3g}",
                "delta_pct": (
                    f"{(uav_median - ideal_median) / ideal_median:+.1%}"
                    if ideal_median
                    else "n/a"
                ),
                "ci": f"[{low:.3g}, {high:.3g}]",
            }
        )
    return table


def screen_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Rank UAV arms by the shortlist rule: safety and timing gate first, error decides."""
    by_profile = defaultdict(list)
    for row in rows:
        if row["mode"] == "uav":
            by_profile[row["profile"]].append(row)

    entries = []
    for profile, profile_rows in sorted(by_profile.items()):
        collisions = _numbers(profile_rows, "collisions")
        guard = _numbers(profile_rows, "guard_fraction")
        p99 = _numbers(profile_rows, "step_p99_ms")
        error = _numbers(profile_rows, "occupancy_mse")
        median, iqr = _median_iqr(error)
        shortlisted = bool(
            collisions.size
            and (collisions == 0).all()
            and (guard < 0.01).all()
            and (p99 < 16.0).all()
        )
        entries.append(
            {
                "profile": profile,
                "seeds": len(profile_rows),
                "occupancy_mse": f"{median:.4g} ± {iqr:.3g}",
                "_median": median,
                "_iqr": iqr,
                "collisions": int(collisions.max()) if collisions.size else -1,
                "guard_fraction": f"{guard.max():.4g}" if guard.size else "n/a",
                "step_p99_ms": f"{p99.max():.4g}" if p99.size else "n/a",
                "shortlisted": "yes" if shortlisted else "no",
            }
        )

    shortlist = [entry for entry in entries if entry["shortlisted"] == "yes"]
    if shortlist:
        best = min(shortlist, key=lambda entry: entry["_median"])
        baseline = next((e for e in shortlist if e["profile"] == "baseline"), None)
        # Retain the inherited setting unless the winner clears it by more than one IQR.
        if baseline is not None and best["_median"] > baseline["_median"] - baseline["_iqr"]:
            best = baseline
        for entry in entries:
            entry["selected"] = "yes" if entry is best else ""
    for entry in entries:
        entry.pop("_median", None)
        entry.pop("_iqr", None)
        entry.setdefault("selected", "")
    return entries


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_no data_\n"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(row[key]) for key in headers) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def build_report(rows: list[dict[str, str]]) -> str:
    """Render the full markdown report."""
    uav, ideal = split_modes(rows)
    criteria = acceptance(uav, ideal)
    failures = sum(1 for entry in criteria if entry["status"] == "FAIL")
    verdict = "ACCEPTED" if failures == 0 else f"NOT ACCEPTED ({failures} criteria failed)"
    return (
        "# UAV deployment report\n\n"
        f"{len(uav)} paired runs, {len(rows)} summary rows.\n\n"
        f"**Verdict: {verdict}**\n\n"
        "## Acceptance checklist\n\n"
        + _markdown_table(criteria)
        + "\n## Paired UAV vs ideal\n\n"
        "Median ± IQR across seeds. The CI is a percentile bootstrap over the paired\n"
        "per-run differences (uav - ideal); an interval spanning zero means the runs do\n"
        "not separate at this sample size.\n\n"
        + _markdown_table(paired_table(uav, ideal))
        + "\n## Screen\n\n"
        "Shortlisted requires zero collisions, guard intervention under 1%, and p99 under\n"
        "16 ms in every seed. Among those, lowest median occupancy error wins, except that\n"
        "the inherited baseline is retained unless beaten by more than one of its IQRs.\n\n"
        + _markdown_table(screen_table(rows))
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/uav/summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/uav/report.md"))
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    rows = read_rows(arguments.summary)
    prepare_outputs([arguments.output], arguments.overwrite)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(build_report(rows), encoding="utf-8")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
