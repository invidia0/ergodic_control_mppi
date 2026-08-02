#!/usr/bin/env bash
# Screen UAV controller settings by flying each one headless, then pair every run with an
# ideal offline twin. Results accumulate in results/uav/summary.csv, one row per run.
#
# Designed to be launched detached on the GPU box and polled from elsewhere:
#
#   ssh ars-admin@HOST 'cd ~/ergodic_control_mppi && nohup bash scripts/run_uav_screen.sh \
#       > scratch/uav_screen.log 2>&1 &'
#
# Poll by counting rows in results/uav/summary.csv and wait for the EXIT_ sentinel.
# Resumable: a run whose directory already exists is skipped rather than repeated, so
# re-running after a crash costs only the runs that never finished.

set -u -o pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p scratch results/uav

COMPOSE="docker compose -f docker/ros2/compose.yaml"
STEPS="${STEPS:-5000}"
SEEDS="${SEEDS:-43 44 45}"
DEVICE="${DEVICE:-gpu}"
# Host path for patching, container path for launching: the repo is bind-mounted.
PROFILE_CONFIG_HOST="${PROFILE_CONFIG_HOST:-configs/uav_profile.yaml}"
MAP_SEED="${MAP_SEED:-511}"
MAP_FILL="${MAP_FILL:-0.02}"

# arm name -> dotted config overrides. The inherited baseline carries no overrides; every
# other arm changes exactly one axis so a win is attributable.
declare -A ARMS=(
    [baseline]=""
    [speed_1p0]="stein.reference_speed=1.0"
    [speed_1p5]="stein.reference_speed=1.5"
    [speed_2p0]="stein.reference_speed=2.0"
    [accel_4]="model.double_integrator.max_accel_lin_abs=4.0"
    [accel_8]="model.double_integrator.max_accel_lin_abs=8.0"
    [horizon_100]="mppi.T=100"
    [horizon_200]="mppi.T=200"
    [samples_512]="mppi.K=512"
    [memory_5]="stein.memory_time=5.0"
)

echo "=== uav screen start $(date -u +%FT%TZ) ==="
echo "commit : $(git rev-parse --short HEAD)"
echo "steps  : $STEPS"
echo "seeds  : $SEEDS"
echo "map    : seed=$MAP_SEED fill=$MAP_FILL"

FAILED=()
for ARM in "${!ARMS[@]}"; do
    for SEED in $SEEDS; do
        RUN_ID="${ARM}_s${SEED}"
        if [ -d "results/uav/${RUN_ID}" ]; then
            echo "SKIP=${RUN_ID} already recorded"
            continue
        fi

        # Materialize the arm's config once, reusing the campaign runner's dotted-path
        # patcher so an axis here means exactly what it means in the ablation stages.
        CONFIG="configs/generated/uav_${ARM}.yaml"
        mkdir -p configs/generated
        OVERRIDES="${ARMS[$ARM]}" BASE="$PROFILE_CONFIG_HOST" OUT="$CONFIG" uv run python - <<'PY' \
            || { FAILED+=("$RUN_ID config"); continue; }
import os, pathlib, yaml
from ergodic_control_mppi.experiments.ablation import _set_dotted

data = yaml.safe_load(pathlib.Path(os.environ["BASE"]).read_text(encoding="utf-8"))
for override in os.environ["OVERRIDES"].split():
    dotted, _, raw = override.partition("=")
    _set_dotted(data, dotted, yaml.safe_load(raw))
pathlib.Path(os.environ["OUT"]).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

        echo "=== run $RUN_ID start $(date -u +%FT%TZ) ==="
        if $COMPOSE run --rm uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                    "config:=/workspace/${CONFIG}" "run_id:=${RUN_ID}" \
                    "profile:=${ARM}" "steps:=${STEPS}" "seed:=${SEED}" \
                    "map_seed:=${MAP_SEED}" "map_fill:=${MAP_FILL}" \
                    "device:=${DEVICE}" "rviz:=false" \
                2>&1 | tee "scratch/uav_${RUN_ID}.log"; then
            # Pair immediately: it is pure post-processing on the saved grid, so a later
            # flight crashing never costs the comparison for the flights that finished.
            uv run python -m ergodic_control_mppi.experiments.uav_pair \
                --run-dir "results/uav/${RUN_ID}" --device "$DEVICE" \
                2>&1 | tee -a "scratch/uav_${RUN_ID}.log" \
                || FAILED+=("$RUN_ID pair")
            echo "=== run $RUN_ID done $(date -u +%FT%TZ) ==="
        else
            echo "=== run $RUN_ID FAILED $(date -u +%FT%TZ) ==="
            FAILED+=("$RUN_ID")
        fi
    done
done

uv run python -m ergodic_control_mppi.experiments.uav_report \
    --summary results/uav/summary.csv --output results/uav/report.md --overwrite \
    || echo "report pass reported problems"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "EXIT_OK all runs complete $(date -u +%FT%TZ)"
else
    echo "EXIT_FAIL runs: ${FAILED[*]}"
    exit 1
fi
