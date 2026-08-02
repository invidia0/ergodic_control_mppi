#!/usr/bin/env bash
# Resumable seven-cell pillar tuning. Stages are explicit so no remote work starts by accident.

set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:---dry-run}"
ROOT="results/uav/pillar_tuning"
CONFIG="configs/uav_profile.yaml"
COMPOSE="docker compose -f docker/ros2/compose.yaml"
STEPS=20000
CAP_STEPS=10000

launch_args=(
    "config:=/workspace/$CONFIG"
    "map_source:=random_forest" "map_fill:=0.0" "obs_num:=45"
    "pillar_min_radius:=0.3" "pillar_max_radius:=0.6"
    "pillar_min_height:=2.0" "pillar_max_height:=3.0"
    "pillar_min_distance:=1.2" "clearance:=0.05" "rviz:=false"
)

value() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning value \
        --path "$1" --field "$2"
}

report() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning report \
        --root "$ROOT" --output "$ROOT/report.md"
}

prepare() {
    mkdir -p "$ROOT/maps" scratch
    for map_seed in $(seq 511 610); do
        log="scratch/pillar_tuning_probe_${map_seed}.log"
        if [[ ! -f "$ROOT/qualification.csv" ]] || \
           ! awk -F, -v seed="$map_seed" 'NR > 1 && $1 == seed { found=1 } END { exit !found }' "$ROOT/qualification.csv"; then
            set +e
            ROS_DOMAIN_ID=$((map_seed - 410)) timeout 6 $COMPOSE run --rm \
                --name "pillar-tuning-probe-$map_seed" uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                "${launch_args[@]}" "map_seed:=$map_seed" "steps:=1" "device:=gpu" \
                "output_root:=/tmp/pillar-tuning-probe" "run_id:=probe_$map_seed" \
                "qualify_only:=true" >"$log" 2>&1
            docker stop --timeout 1 "pillar-tuning-probe-$map_seed" >/dev/null 2>&1 || true
            set -e
        fi
        uv run python -m ergodic_control_mppi.experiments.uav_pillars probe \
            --log "$log" --seed "$map_seed" --output "$ROOT/qualification.csv"
    done
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning select \
        --qualification "$ROOT/qualification.csv" --output "$ROOT/selection.json"

    maps="$(value "$ROOT/selection.json" development) $(value "$ROOT/selection.json" holdout)"
    index=0
    for map_seed in $maps; do
        run_id="map_$map_seed"
        if [[ ! -f "$ROOT/maps/$run_id/arrays.npz" ]]; then
            ROS_DOMAIN_ID=$((220 + index)) $COMPOSE run --rm uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                "${launch_args[@]}" "map_seed:=$map_seed" "steps:=1" "device:=gpu" \
                "preflight_steps:=0" "output_root:=/workspace/$ROOT/maps" "run_id:=$run_id"
        fi
        index=$((index + 1))
    done
}

run_screen() {
    representative="$(value "$ROOT/selection.json" development_representative)"
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning run \
        --run-dir "$ROOT/maps/map_$representative" --output "$ROOT/offline.csv" \
        --cap-output "$ROOT/run_cap.csv" --cap-steps "$CAP_STEPS" \
        --stage screen --first-seed 43 --seeds 6 --steps "$STEPS" --device gpu
    report
}

run_approach() {
    report
    base="$(value "$ROOT/gate.json" screen_winner)"
    for map_seed in $(value "$ROOT/selection.json" development); do
        uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning run \
            --run-dir "$ROOT/maps/map_$map_seed" --output "$ROOT/offline.csv" \
            --cap-output "$ROOT/run_cap.csv" --cap-steps "$CAP_STEPS" \
            --stage approach --base-arm "$base" --first-seed 43 --seeds 6 \
            --steps "$STEPS" --device gpu
    done
    report
}

run_holdout() {
    report
    base="$(value "$ROOT/gate.json" screen_winner)"
    winner="$(value "$ROOT/gate.json" approach_winner)"
    [[ -n "$winner" ]] || { echo "No development arm passed; holdout not run." >&2; exit 1; }
    for map_seed in $(value "$ROOT/selection.json" holdout); do
        uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning run \
            --run-dir "$ROOT/maps/map_$map_seed" --output "$ROOT/offline.csv" \
            --cap-output "$ROOT/run_cap.csv" --cap-steps "$CAP_STEPS" \
            --stage holdout --base-arm "$base" --winner "$winner" \
            --first-seed 43 --seeds 18 --steps "$STEPS" --device gpu
    done
    report
}

run_online() {
    report
    [[ "$(value "$ROOT/gate.json" holdout_pass)" == "True" ]] || {
        echo "Holdout gate has not passed; online flights not run." >&2
        exit 1
    }
    profile="$ROOT/pillar_profile.yaml"
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning config \
        --root "$ROOT" --output "$profile"
    representative="$(value "$ROOT/selection.json" holdout_representative)"
    online_args=("${launch_args[@]}")
    online_args[0]="config:=/workspace/$profile"
    for seed in $(seq 43 47); do
        run_id="online_${representative}_s${seed}"
        if [[ ! -f "$ROOT/$run_id/arrays.npz" ]]; then
            ROS_DOMAIN_ID=$((250 + seed - 43)) $COMPOSE run --rm uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                "${online_args[@]}" \
                "map_seed:=$representative" "steps:=$STEPS" "seed:=$seed" \
                "preflight_steps:=200" "profile:=pillar_tuned" "bag:=true" "device:=gpu" \
                "output_root:=/workspace/$ROOT" "run_id:=$run_id"
        fi
        if [[ ! -f "$ROOT/summary.csv" ]] || \
           ! awk -F, -v id="$run_id" 'NR > 1 && $1 == id && $3 == "ideal" { found=1 } END { exit !found }' "$ROOT/summary.csv"; then
            uv run python -m ergodic_control_mppi.experiments.uav_pair \
                --run-dir "$ROOT/$run_id" --device gpu --summary "$ROOT/summary.csv"
        fi
    done
    uv run python -m ergodic_control_mppi.plotting.deployment \
        --run-dir "$ROOT/online_${representative}_s43" --output "$ROOT/snapshot.png"
    report
}

run_campaign() {
    prepare
    run_screen
    run_approach
    run_holdout
    run_online
}

case "$ACTION" in
    --dry-run)
        uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning dry-run
        echo "Stages: prepare -> screen -> approach -> holdout -> online"
        ;;
    campaign) run_campaign ;;
    prepare) prepare ;;
    screen) run_screen ;;
    approach) run_approach ;;
    holdout) run_holdout ;;
    online) run_online ;;
    report) report ;;
    *) echo "usage: $0 [--dry-run|campaign|prepare|screen|approach|holdout|online|report]" >&2; exit 2 ;;
esac
