"""Read the campaign archive and apply the pre-registered promotion gate.

The gate is written down here, in code, rather than applied by eye after the fact. Its
shape comes from what went wrong before: ``alpha = 1.0`` had a pooled p-value of 0.0005 and
tripled the tour count, and was still wrong, because the entire effect lived on one map. A
small pooled p-value is necessary and nowhere near sufficient.

So an arm is promoted only if **all three** hold:

  1. the pooled paired Wilcoxon over all 108 cells survives Holm within its own axis,
  2. the per-map median effect has the **same sign on at least 6 of 8 maps**, and
  3. the joint sensitivity clears :data:`SENSITIVITY_FLOOR` sigma.

Condition 2 is the one that would have caught the two retracted findings. It is checked on
sign, not on significance: nine per-map tests at twelve seeds have no power to spare, and
demanding significance on each would promote nothing.

Condition 3 was added after a null arm was promoted on synthetic data at p = 0.011 with
every map agreeing. Two things make that possible and neither is a fluke. Holm applied
within a *two-arm* axis sets the bar at p < 0.025, and with sixteen axes the familywise
error across the table is over 50%; and a paired Wilcoxon over 108 cells has enough power
to resolve effects far too small to act on. An effect-size floor is the standard answer,
and the joint sensitivity is the right quantity for it because it is a signal-to-noise
ratio rather than a raw magnitude -- the false positive had a median effect indistinguishable
from a real arm's (+0.214 against +0.210 log2) and a sensitivity fifty times smaller
(0.3 against 14.5). Consistency, not size, is what separated them.

    uv run python scripts/final_report.py --output results/uav/final_report.md
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "report_figures", ROOT / "scripts" / "report_figures.py"
)
rf = importlib.util.module_from_spec(_spec)
sys.modules["report_figures"] = rf
_spec.loader.exec_module(rf)

# 6 of 8, not the 7 of 9 registered before the run: seed 525 was selected at two densities
# and the driver's config cache flew the 15-pillar field under both labels, so the campaign
# covers eight distinct maps. 6/8 is a 75% bar against 7/9's 77.8% -- the same strictness,
# not a threshold relaxed after seeing results. See `DUPLICATE_MAPS` in report_figures.py.
MAP_AGREEMENT = 6
# Joint sensitivity, in sigmas of a single run's noise across the five-outcome vector. An
# arm below this moved the system less than three noise units however small its p-value is,
# and 108 paired cells can resolve a great deal less than that.
SENSITIVITY_FLOOR = 3.0
DENSITIES = (15, 25, 35)


def analyse(table, metric: str = "fourier_ergodic") -> list[dict]:
    """One record per arm: pooled test, per-map agreement, per-density medians."""
    # Both baseline widths are excluded, not just the plain name: `load_final` stores them as
    # `baseline@108` / `baseline@27`, and an arm paired against itself has zero variance on
    # every outcome, which makes the joint sensitivity's covariance singular.
    arms = sorted(a for a in table if not a.startswith(rf.BASELINE))
    records = []
    for arm in arms:
        arm_values, base_values, cells = rf.paired_final(table, arm, metric)
        # Lower is better for the coverage metrics, so a positive effect means the arm won.
        effect = np.log2(base_values / arm_values)
        try:
            pvalue = float(wilcoxon(arm_values, base_values).pvalue)
        except ValueError:
            pvalue = 1.0
        signed = rf.per_map_effects(table, arm, metric, standardize=False)
        positive = sum(1 for v in signed.values() if v > 0)
        by_density: dict[int, list[float]] = defaultdict(list)
        for (obs_num, _, _), value in zip(cells, effect):
            by_density[obs_num].append(value)
        rows = list(table[arm].values())
        records.append({
            "arm": arm,
            "axis": rows[0].get("axis") or arm,
            "cells": len(cells),
            "median_effect": float(np.median(effect)),
            "pvalue": pvalue,
            "maps_positive": positive,
            "maps_total": len(signed),
            "agreement": max(positive, len(signed) - positive),
            "density_medians": {d: float(np.median(by_density[d]))
                                for d in DENSITIES if by_density[d]},
            "sensitivity": rf.sensitivity(table, arm)[1],
            "tours": sum(int(float(r["all_modes_reached"])) + float(r["mode_cycles"])
                         for r in rows),
        })

    by_axis: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_axis[record["axis"]].append(index)
    for indices in by_axis.values():
        keep = rf.holm([records[i]["pvalue"] for i in indices])
        for index, significant in zip(indices, keep):
            records[index]["holm"] = bool(significant)

    for record in records:
        resolved = record["holm"] and record["sensitivity"] >= SENSITIVITY_FLOOR
        consistent = record["agreement"] >= MAP_AGREEMENT
        record["promoted"] = bool(resolved and consistent and record["median_effect"] > 0)
        # Four outcomes, not two. A consistently *harmful* arm is a finding, not a failed
        # promotion, and lumping it in with the map-dependent ones produced the nonsense
        # line "only 8/8 maps agree" for the arm that loses on every map.
        if record["promoted"]:
            record["verdict"] = "promoted"
        elif resolved and consistent:
            record["verdict"] = "harmful"
        elif resolved:
            record["verdict"] = "map-dependent"
        elif record["holm"]:
            record["verdict"] = "negligible"
        else:
            record["verdict"] = "null"
    return records


def render(records: list[dict], metric: str) -> str:
    """Markdown, ordered by joint sensitivity so the ranking leads."""
    records = sorted(records, key=lambda r: -r["sensitivity"])
    lines = [
        f"# Final ablation campaign -- {metric}",
        "",
        f"{len(records)} arms against the shipped profile, {records[0]['cells']} paired "
        "cells each (8 maps over 3 densities x 12 seeds, 20 000 steps).",
        "",
        "`promoted` requires all three of: a Holm-surviving pooled Wilcoxon within the "
        f"arm's own axis, the same effect sign on at least {MAP_AGREEMENT} of 8 maps, and "
        f"a joint sensitivity of at least {SENSITIVITY_FLOOR} sigma. The map condition is "
        "the one that matters most: the retracted `alpha = 1.0` finding had p = 0.0005 "
        "pooled and agreed on 1 map of 3.",
        "",
        "| arm | axis | effect (log2) | p | Holm | maps | 15p | 25p | 35p | sens | promoted |",
        "| --- | --- | ---: | ---: | :-: | :-: | ---: | ---: | ---: | ---: | :-: |",
    ]
    for record in records:
        density = record["density_medians"]
        lines.append(
            f"| `{record['arm']}` | {record['axis']} | {record['median_effect']:+.3f} | "
            f"{record['pvalue']:.2e} | {'yes' if record['holm'] else '--'} | "
            f"{record['maps_positive']}/{record['maps_total']} | "
            + " | ".join(f"{density.get(d, float('nan')):+.3f}" for d in DENSITIES)
            + f" | {record['sensitivity']:.1f} | "
            f"{'**YES**' if record['promoted'] else record['verdict']} |"
        )

    promoted = [r for r in records if r["verdict"] == "promoted"]
    harmful = [r for r in records if r["verdict"] == "harmful"]
    split = [r for r in records if r["verdict"] == "map-dependent"]
    negligible = [r for r in records if r["verdict"] == "negligible"]
    dead = [r for r in records if r["tours"] == 0]
    lines += [
        "",
        "## Promoted",
        "",
        ("\n".join(f"- `{r['arm']}` ({r['axis']}): {r['median_effect']:+.3f} log2, "
                   f"{r['maps_positive']}/{r['maps_total']} maps, p = {r['pvalue']:.2e}"
                   for r in promoted) if promoted else "None."),
        "",
        "## Consistently harmful",
        "",
        "Resolved, consistent across maps, and in the wrong direction. These are results, "
        "not failures -- an axis that degrades everywhere is evidence the term carries the "
        "method.",
        "",
        ("\n".join(f"- `{r['arm']}` ({r['axis']}): {r['median_effect']:+.3f} log2 on "
                   f"{r['agreement']}/{r['maps_total']} maps, p = {r['pvalue']:.2e}"
                   for r in harmful) if harmful else "None."),
        "",
        "## Significant but map-dependent -- NOT promoted",
        "",
        "These are the alpha = 1.0 shape: a real pooled effect that does not hold across "
        "maps. Report them as conditional, with the density or map they depend on named.",
        "",
        ("\n".join(f"- `{r['arm']}` ({r['axis']}): p = {r['pvalue']:.2e} but only "
                   f"{r['agreement']}/{r['maps_total']} maps agree; per density "
                   + ", ".join(f"{d}p {r['density_medians'].get(d, float('nan')):+.3f}"
                               for d in DENSITIES)
                   for r in split) if split else "None."),
        "",
        f"## Resolved but below the {SENSITIVITY_FLOOR} sigma floor",
        "",
        "Holm-significant and practically negligible. With 108 paired cells the test "
        "resolves effects far too small to act on; these arms prove it.",
        "",
        ("\n".join(f"- `{r['arm']}` ({r['axis']}): p = {r['pvalue']:.2e}, "
                   f"sensitivity {r['sensitivity']:.2f} sigma"
                   for r in negligible) if negligible else "None."),
        "",
        "## Never completed a tour",
        "",
        ("\n".join(f"- `{r['arm']}`" for r in dead) if dead else "None."),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path("results/uav/ablation_final.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/uav/final_report.md"))
    parser.add_argument("--metric", default="fourier_ergodic")
    arguments = parser.parse_args()

    table = rf.load_final(arguments.archive)
    records = analyse(table, arguments.metric)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render(records, arguments.metric), encoding="utf-8")
    promoted = [r["arm"] for r in records if r["promoted"]]
    print(f"wrote {arguments.output}")
    print(f"promoted: {', '.join(promoted) if promoted else 'none'}")


if __name__ == "__main__":
    main()
