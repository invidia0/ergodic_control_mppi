#!/usr/bin/env bash
# Read-only progress summary. Usage:
# ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_pillar_tuning.sh
set -u
cd /home/ars-admin/ergodic_control_mppi

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path("results/uav/pillar_tuning")

def rows(name):
    path = root / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

maps = rows("qualification.csv")
selection = json.loads((root / "selection.json").read_text()) if (root / "selection.json").exists() else {}
print(f"maps       {len(maps)}/100  selected={selection or 'pending'}")
caps, full = rows("run_cap.csv"), rows("offline.csv")
# The sweep writes its own pair of files so the gated campaign archive stays untouched.
caps += rows("sweep_cap.csv")
full += rows("sweep.csv")
print(f"{'stage':10} {'10k':>8} {'accepted':>10} {'20k':>8} {'repeats':>9}")
for stage in ("screen", "approach", "holdout", "sweep"):
    attempted = [row for row in caps if row["stage"] == stage]
    completed = [row for row in full if row["stage"] == stage]
    accepted = sum(int(row["all_modes_reached"]) for row in attempted)
    repeats = sum(float(row["mode_cycles"]) >= 1 for row in completed)
    print(f"{stage:10} {len(attempted):8} {accepted:10} {len(completed):8} {repeats:9}")

# Per-arm progress for the sweep, which is the stage that varies arm by arm.
sweep = [row for row in caps if row["stage"] == "sweep"]
if sweep:
    print()
    print(f"{'arm':16} {'10k':>5} {'tours':>6} {'20k':>5} {'dwell s':>8} {'m/s':>6}")
    for arm in sorted({row["arm"] for row in sweep}):
        mine = [row for row in sweep if row["arm"] == arm]
        done = [row for row in full if row["stage"] == "sweep" and row["arm"] == arm]
        dwell = sorted(float(row["mode_dwell_median_s"]) for row in mine)
        speed = sorted(
            float(row["path_length_m"]) / (int(row["steps"]) * 0.02) for row in mine
        )
        middle = len(mine) // 2
        print(
            f"{arm:16} {len(mine):5} "
            f"{sum(int(row['accepted']) for row in mine):6} {len(done):5} "
            f"{dwell[middle]:8.1f} {speed[middle]:6.3f}"
        )
PY

pgrep -f "scripts/run_pillar_tuning.sh" >/dev/null \
    && state=RUNNING || state="NOT RUNNING"
echo "state      $state"
echo "gpu        $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for log in scratch/pillar_tuning_campaign.log scratch/sweep_primary.log scratch/sweep_rest.log; do
    last="$(grep -E '^(ROW=|DISCARD=|wrote )' "$log" 2>/dev/null | tail -1)"
    [[ -n "$last" ]] && echo "last       ${log##*/}: $last"
done
