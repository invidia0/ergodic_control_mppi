"""Check numerical claims in the manuscript against the frozen T=150 bundle.

Run from the repository root:

    uv run python scripts/verify_manuscript_numbers.py
"""

import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("fr", "scripts/final_report.py")
fr = importlib.util.module_from_spec(spec)
sys.modules["fr"] = fr
spec.loader.exec_module(fr)
rf = sys.modules["report_figures"]

ROOT = Path("results/uav/T150")
MANUSCRIPT = Path("69f1b707cd917a58478ed643/main.tex")
TEXT = MANUSCRIPT.read_text(encoding="utf-8") if MANUSCRIPT.is_file() else None


def renderings(value: float) -> list[str]:
    """Return the numeric spellings used in prose and LaTeX scientific notation."""
    out = []
    for text in (f"{value:g}", f"{value:,g}", f"{abs(value):g}"):
        out.append(text)
        if text.startswith("0."):
            out.append(text[1:])
    if value and abs(value) < 1e-3:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10**exponent
        out += [f"{mantissa:.2f}", f"{mantissa:.3g}"]
    return out


ok = bad = missing = 0


def check(label: str, claimed: float, actual: float, tol: float = 0.005) -> None:
    """Compare one manuscript claim with its value reconstructed from the archive."""
    global ok, bad, missing
    good = abs(claimed - actual) <= tol * max(1.0, abs(actual))
    found = TEXT is not None and any(value in TEXT for value in renderings(claimed))
    ok += good
    bad += not good
    missing += not found
    print(
        f"  {'OK ' if good else 'BAD'} {label:<46} claimed {claimed:<12} actual {actual:.4g}"
        f"{'' if found else '   [not found in main.tex]'}"
    )


print("--- ablation ---")
table = rf.load_final(ROOT / "clutter/ablation.csv")
records = {record["arm"]: record for record in fr.analyse(table)}
check("arms", 39, len(records), 0)
check("axes", 19, len({record["axis"] for record in records.values()}), 0)
check("cells per arm", 36, records["T_350"]["cells"], 0)
for arm, effect, sensitivity in (
    ("memory_off", -2.88, 13.0),
    ("plan_off", -1.31, None),
    ("ceiling_0", -0.70, 4.2),
    ("release_off", -0.12, 2.9),
    ("transit_1", -1.33, 8.5),
    ("h_0.47", -0.11, 4.4),
    ("h_2.35", -0.37, 5.1),
    ("h_5.0", -0.78, 7.9),
    ("alpha_0.9", -1.16, 5.0),
    ("T_350", -0.55, 3.3),
    ("T_500", -0.75, 4.5),
    ("T_100", 0.17, None),
    ("T_75", 0.29, None),
    ("ceiling_0.5", -0.04, 0.48),
):
    check(f"{arm} effect", effect, records[arm]["median_effect"], 0.02)
    if sensitivity is not None:
        check(f"{arm} sensitivity", sensitivity, records[arm]["sensitivity"], 0.02)
check("promotions", 0, sum(r["verdict"] == "promoted" for r in records.values()), 0)
check(
    "Holm-significant sub-3sigma",
    12,
    sum(r["holm"] and r["sensitivity"] < 3.0 for r in records.values()),
    0,
)
check("null verdicts", 18, sum(r["verdict"] == "null" for r in records.values()), 0)
open_records = {
    record["arm"]: record
    for record in fr.analyse(rf.load_final(ROOT / "open/ablation.csv"))
}
check("open h_2.35 effect", -0.19, open_records["h_2.35"]["median_effect"], 0.02)
check("open h_2.35 sensitivity", 7.4, open_records["h_2.35"]["sensitivity"], 0.02)


def load_baselines(path: Path) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    """Index a baseline CSV by method and paired map/seed cell."""
    data = collections.defaultdict(dict)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            data[row["method"]][(row["map"], row["seed"])] = row
    return data


print("--- baselines ---")
# (Fourier, occupancy MSE, certified E_N) as quoted in Sec. V-C, at T=150.
for tier, claims in (
    ("open", {"fmec": (2.53, 0.39, 0.30), "hedac": (0.85, 1.71, -0.05),
              "sves": (0.70, 0.57, 1.46), "smc": (-1.01, 1.90, 0.79)}),
    ("clutter", {"fmec": (1.36, 0.04, -0.31), "hedac": (0.12, 1.08, -0.23),
                 "sves": (0.58, 0.88, 1.46), "smc": (-1.29, 1.72, 0.76)}),
):
    data = load_baselines(ROOT / tier / "baselines.csv")
    ours = data["ours"]
    for method, (fourier, occupancy, certified) in claims.items():
        cells = sorted(set(ours) & set(data[method]))
        for metric, claimed in (("fourier_ergodic", fourier), ("occupancy_mse", occupancy),
                                ("mmd_final", certified)):
            baseline = np.array([float(data[method][cell][metric]) for cell in cells])
            reference = np.array([float(ours[cell][metric]) for cell in cells])
            check(f"{tier} {method} {metric}", claimed, float(np.median(np.log2(baseline / reference))), 0.02)
    if tier == "clutter":
        for method, claimed in (("sves", 36.1), ("smc", 27.8), ("fmec", 13.9), ("hedac", 8.3), ("ours", 0.0)):
            rows = list(data[method].values())
            check(f"clutter {method} collision %", claimed, 100 * sum(int(r["collisions"]) > 0 for r in rows) / len(rows), 0.02)
        check("clutter ours clearance", 0.94, np.median([float(r["min_clearance_m"]) for r in ours.values()]), 0.02)
        check("clutter ours path m", 953, np.median([float(r["path_length_m"]) for r in ours.values()]), 0.01)
        check("clutter ours speed", 2.38,
              np.median([float(r["path_length_m"]) for r in ours.values()]) / 400.0, 0.01)
        check("clutter fastest baseline path m", 685,
              max(np.median([float(r["path_length_m"]) for r in data[m].values()])
                  for m in ("hedac", "sves", "fmec", "smc")), 0.01)
        check("clutter sves modes", 31, sum(int(float(r["all_modes_reached"])) for r in data["sves"].values()), 0)
        # The certificate's own validity, counted over every method rather than ours alone.
        for field, label in (("mmd_prefix_holds", "prefix"), ("mmd_beats_trivial", "nonvacuous")):
            check(f"all-method certificate {label}", 240,
                  sum(int(r[field]) for tier_data in (
                      load_baselines(ROOT / "open/baselines.csv"),
                      load_baselines(ROOT / "clutter/baselines.csv"))
                      for rows in tier_data.values() for r in rows.values()), 0)
    else:
        check("open ours tours", 6, np.median([float(r["mode_cycles"]) for r in ours.values()]), 0)
        check("open sves tours", 7.5, np.median([float(r["mode_cycles"]) for r in data["sves"].values()]), 0)


print("--- certificate ---")
with (ROOT / "audit/certificate.csv").open(newline="") as stream:
    certificate = list(csv.DictReader(stream))
check("certificate paths", 72, len(certificate), 0)
check("certificate prefix passes", 72, sum(r["prefix_holds"] == "True" for r in certificate), 0)
check("certificate nonvacuous", 72, sum(r["beats_trivial"] == "True" for r in certificate), 0)
check("certificate final discrepancy", 0.00382, np.median([float(r["error_final"]) for r in certificate]), 0.01)
check("certificate final bound", 0.0212, np.median([float(r["bound_final"]) for r in certificate]), 0.01)
check("certificate looseness", 5.56, np.median([float(r["looseness"]) for r in certificate]), 0.01)


print("--- timing ---")
timing = json.loads((ROOT / "audit/timing_t150.json").read_text(encoding="utf-8"))
stages = timing["stages"]
check("isolated stage sum ms", 1.96, stages["accounted_ms"], 0.01)
check("isolated/fused gap percent", 10.6, 100 * stages["residual_ms"] / stages["total_ms"], 0.01)
check("end-to-end step ms", 2.65, timing["endtoend"]["with_memory"]["ms_median"], 0.01)
check("fused step ms", 2.19, stages["total_ms"], 0.01)
check("unattributed ms", 0.23, stages["residual_ms"], 0.02)
check("rollout share percent", 64, 100 * stages["stages"]["rollouts_KT"]["ms_median"] / stages["total_ms"], 0.01)
for name, claimed in (("rollouts_KT", 1.41), ("memory_P2", 0.22), ("sample_epsilon", 0.15),
                      ("plan_T2", 0.11), ("attraction_T", 0.08)):
    check(f"{name} ms", claimed, stages["stages"][name]["ms_median"], 0.01)

print(f"\n{ok} verified, {bad} WRONG, {missing} not located in the manuscript text")
if bad:
    raise SystemExit(f"{bad} of {ok + bad} manuscript numbers do not match the data")
