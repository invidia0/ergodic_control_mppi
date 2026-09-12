#!/usr/bin/env bash
# Preregister, run, pair, and report the planar pillar campaign.

set -u -o pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

ROOT="results/uav/pillar"
CONFIG="configs/uav_profile.yaml"
COMPOSE="docker compose -f docker/ros2/compose.yaml"
STEPS=20000
OBS_NUM=45
RADIUS_MIN=0.3
RADIUS_MAX=0.6
HEIGHT_MIN=2.0
HEIGHT_MAX=3.0
MIN_DISTANCE=1.2

if [[ "${1:-}" == "--overwrite" ]]; then
    [[ "$ROOT" == "results/uav/pillar" ]] || exit 2
    rm -rf -- "$ROOT"
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--overwrite]" >&2
    exit 2
fi

mkdir -p "$ROOT/maps" scratch

launch_args=(
    "config:=/workspace/$CONFIG"
    "map_source:=random_forest"
    "map_fill:=0.0"
    "obs_num:=$OBS_NUM"
    "pillar_min_radius:=$RADIUS_MIN"
    "pillar_max_radius:=$RADIUS_MAX"
    "pillar_min_height:=$HEIGHT_MIN"
    "pillar_max_height:=$HEIGHT_MAX"
    "pillar_min_distance:=$MIN_DISTANCE"
    "clearance:=0.15"
    "preflight_steps:=0"
    "rviz:=false"
)

echo "=== pillar geometry sweep $(date -u +%FT%TZ) ==="
for map_seed in $(seq 511 610); do
    log="scratch/pillar_probe_${map_seed}.log"
    if [[ ! -f "$ROOT/qualification.csv" ]] || ! awk -F, -v seed="$map_seed" 'NR > 1 && $1 == seed { found=1 } END { exit !found }' "$ROOT/qualification.csv"; then
        set +e
        ROS_DOMAIN_ID=$((map_seed - 510)) timeout 6 $COMPOSE run --rm \
            --name "pillar-probe-$map_seed" uav \
            ros2 launch ergodic_control_mppi_ros uav.launch.py \
            "${launch_args[@]}" "map_seed:=$map_seed" "steps:=1" "device:=gpu" \
            "output_root:=/tmp/pillar-probe" "run_id:=probe_$map_seed" \
            "qualify_only:=true" \
            >"$log" 2>&1
        docker stop --timeout 1 "pillar-probe-$map_seed" >/dev/null 2>&1 || true
        set -e
    fi
    uv run python -m ergodic_control_mppi.experiments.uav_pillars probe \
        --log "$log" --seed "$map_seed" --output "$ROOT/qualification.csv"
done
set -e

uv run python -m ergodic_control_mppi.experiments.uav_pillars select \
    --qualification "$ROOT/qualification.csv" --output "$ROOT/selection.json"
maps="$(uv run python -m ergodic_control_mppi.experiments.uav_pillars value \
    --selection "$ROOT/selection.json" --field maps)"
representative="$(uv run python -m ergodic_control_mppi.experiments.uav_pillars value \
    --selection "$ROOT/selection.json" --field representative)"

capture_map() {
    local map_seed="$1" run_id="$2" domain="$3"
    if [[ -f "$ROOT/maps/$run_id/arrays.npz" ]]; then
        echo "SKIP $run_id already captured"
        return
    fi
    ROS_DOMAIN_ID="$domain" $COMPOSE run --rm uav \
        ros2 launch ergodic_control_mppi_ros uav.launch.py \
        "${launch_args[@]}" "map_seed:=$map_seed" "steps:=1" "device:=gpu" \
        "output_root:=/workspace/$ROOT/maps" "run_id:=$run_id"
}

index=0
for map_seed in $maps; do
    capture_map "$map_seed" "map_$map_seed" $((121 + index))
    index=$((index + 1))
done
capture_map "$representative" "map_${representative}_repeat" 124

echo "=== offline sensitivity $(date -u +%FT%TZ) ==="
for map_seed in $maps; do
    uv run python -m ergodic_control_mppi.experiments.uav_ablation \
        --run-dir "$ROOT/maps/map_$map_seed" --output "$ROOT/offline.csv" \
        --first-seed 43 --seeds 18 --steps "$STEPS" --device gpu \
        --arms baseline,h_0.94,h_6.6
done

echo "=== online flights on map $representative $(date -u +%FT%TZ) ==="
for seed in $(seq 43 47); do
    run_id="online_${representative}_s${seed}"
    if [[ -d "$ROOT/$run_id" && ! -f "$ROOT/$run_id/arrays.npz" ]]; then
        mkdir -p "$ROOT/failed"
        mv -- "$ROOT/$run_id" \
            "$ROOT/failed/${run_id}_$(date -u +%Y%m%dT%H%M%S%NZ)"
    fi
    if [[ ! -f "$ROOT/$run_id/arrays.npz" ]]; then
        ROS_DOMAIN_ID=$((150 + seed - 43)) $COMPOSE run --rm uav \
            ros2 launch ergodic_control_mppi_ros uav.launch.py \
            "${launch_args[@]}" "map_seed:=$representative" "steps:=$STEPS" \
            "seed:=$seed" "profile:=pillar" "bag:=true" "device:=gpu" \
            "output_root:=/workspace/$ROOT" "run_id:=$run_id"
    else
        echo "SKIP $run_id already recorded"
    fi
    if [[ ! -f "$ROOT/summary.csv" ]] || ! awk -F, -v id="$run_id" 'NR > 1 && $1 == id && $3 == "ideal" { found=1 } END { exit !found }' "$ROOT/summary.csv"; then
        uv run python -m ergodic_control_mppi.experiments.uav_pair \
            --run-dir "$ROOT/$run_id" --device gpu --summary "$ROOT/summary.csv"
    fi
done

representative_run="online_${representative}_s43"
uv run python -m ergodic_control_mppi.plotting.deployment \
    --run-dir "$ROOT/$representative_run" --output "$ROOT/snapshot.png"
uv run python -m ergodic_control_mppi.experiments.uav_pillars report \
    --root "$ROOT" --output "$ROOT/report.md"

echo "EXIT_OK pillar campaign complete $(date -u +%FT%TZ)"
