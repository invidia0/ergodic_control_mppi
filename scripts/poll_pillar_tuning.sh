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
print(f"{'stage':10} {'10k':>8} {'accepted':>10} {'20k':>8} {'repeats':>9}")
for stage in ("screen", "approach", "holdout"):
    attempted = [row for row in caps if row["stage"] == stage]
    completed = [row for row in full if row["stage"] == stage]
    accepted = sum(int(row["all_modes_reached"]) for row in attempted)
    repeats = sum(float(row["mode_cycles"]) >= 1 for row in completed)
    print(f"{stage:10} {len(attempted):8} {accepted:10} {len(completed):8} {repeats:9}")
PY

pgrep -f "scripts/run_pillar_tuning.sh campaign" >/dev/null \
    && state=RUNNING || state="NOT RUNNING"
echo "state      $state"
echo "gpu        $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "last       $(grep -E '^(ROW=|DISCARD=|wrote )' scratch/pillar_tuning_campaign.log 2>/dev/null | tail -1)"
