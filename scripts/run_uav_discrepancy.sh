#!/usr/bin/env bash
# Causal JAX -> ROS -> SO3 attribution. Every expensive stage is explicit and resumable.

set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

ACTION="${1:---help}"
ROOT="results/uav/discrepancy"
PROFILE="results/uav/pillar_25/sweep_theta_15.yaml"
CSV="$ROOT/results.csv"
SHORT_CSV="$ROOT/short.csv"
STEPS=20000
SHORT_STEPS=500
SEED=52
PREFLIGHT=200
DEVICE="${DISCREPANCY_DEVICE:-gpu}"
HARDWARE="${DISCREPANCY_HARDWARE:-$(hostname)}"
REFERENCE_HARDWARE="${DISCREPANCY_REFERENCE_HARDWARE:-jeff}"
COMPOSE=(docker compose -f docker/ros2/compose.yaml)

map_run() {
    case "$1" in
        516) echo "results/uav/pillar_25/flight_theta_15_516_s52" ;;
        # s46, not s52: the s52 run was never exported -- it held only a bag, which the
        # 2026-08-05 cleanup removed. s46 is the 539 flight that carries arrays.npz.
        539) echo "results/uav/pillar_25/flight_theta_15_539_s46" ;;
        *) echo "unsupported map: $1" >&2; exit 2 ;;
    esac
}

canonical() {
    echo "$ROOT/canonical_${1}_${REFERENCE_HARDWARE}.npz"
}

jax_cell() {
    map_seed="$1"
    shift
    uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-jax \
        --run-dir "$(map_run "$map_seed")" --config "$PROFILE" --output "$CSV" \
        --map-seed "$map_seed" --seed "$SEED" --steps "$STEPS" \
        --preflight-steps "$PREFLIGHT" --device "$DEVICE" --hardware "$HARDWARE" "$@"
}

run_determinism() {
    mkdir -p "$ROOT"
    for map_seed in 516 539; do
        for repeat in 0 1 2 3 4; do
            jax_cell "$map_seed" --condition exact --repeat "$repeat" \
                --canonical "$ROOT/canonical_${map_seed}_${HARDWARE}.npz"
        done
    done
}

run_envelope() {
    mkdir -p "$ROOT"
    [[ "$HARDWARE" == "$REFERENCE_HARDWARE" ]] || {
        echo "the ULP/impulse envelope must run on reference hardware '$REFERENCE_HARDWARE'" >&2
        exit 1
    }
    for map_seed in 516 539; do
        reference="$(canonical "$map_seed")"
        [[ -f "$reference" ]] || {
            echo "missing $reference; run determinism on the reference machine first" >&2
            exit 1
        }
        jax_cell "$map_seed" --condition zero --repeat 0 --canonical "$reference"
        for ulps in 1 8; do
            for mask in $(seq 0 15); do
                jax_cell "$map_seed" --condition ulp --ulps "$ulps" \
                    --sign-mask "$mask" --repeat 0 --canonical "$reference"
            done
        done
        mapfile -t residuals < <(
            find results/uav/pillar_25 -maxdepth 2 -type f -name arrays.npz \
                \( -path '*/flight_theta_15_*/*' -o -path '*/jeff_539_s52/*' \) \
                -printf '%h\n' | sort -u | head -4
        )
        [[ "${#residuals[@]}" -eq 4 ]] || {
            echo "four archived theta-15 flights are required" >&2
            exit 1
        }
        index=0
        for residual in "${residuals[@]}"; do
            jax_cell "$map_seed" --condition measured --residual-run "$residual" \
                --repeat "$index" --canonical "$reference"
            index=$((index + 1))
        done
    done
}

launch_ros() {
    map_seed="$1"
    repeat="$2"
    vehicle="$3"
    steps="$4"
    output_root="$5"
    score_output="${6:-}"
    run_id="map_${map_seed}_${vehicle}_r$(printf '%02d' "$repeat")"
    run_dir="$output_root/$run_id"
    if [[ ! -f "$run_dir/arrays.npz" ]]; then
        ROS_DOMAIN_ID=$((100 + (map_seed + repeat) % 100)) "${COMPOSE[@]}" run --rm uav \
            ros2 launch ergodic_control_mppi_ros uav.launch.py \
            "config:=/workspace/$PROFILE" "map_source:=random_forest" "map_fill:=0.0" \
            "obs_num:=25" "pillar_min_radius:=0.3" "pillar_max_radius:=0.6" \
            "pillar_min_height:=2.0" "pillar_max_height:=3.0" \
            "pillar_min_distance:=1.2" "clearance:=0.0" \
            "tracking_allowance:=0.025" "reaction_time:=0.06" \
            "map_seed:=$map_seed" "steps:=$steps" "seed:=$SEED" \
            "preflight_steps:=$PREFLIGHT" "profile:=sweep_theta_15" \
            "bag:=false" "rviz:=false" "device:=gpu" "deadline_ms:=20.0" \
            "vehicle:=$vehicle" "output_root:=/workspace/$output_root" "run_id:=$run_id"
    fi
    if [[ -n "$score_output" ]]; then
        uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-ros \
            --run-dir "$run_dir" --config "$PROFILE" --output "$score_output" \
            --vehicle "$vehicle" --repeat "$repeat" --hardware "$HARDWARE" \
            --canonical "$(canonical "$map_seed")"
    fi
}

run_short_screen() {
    for map_seed in 516 539; do
        [[ -f "$(canonical "$map_seed")" ]] || {
            echo "missing $(canonical "$map_seed"); the reference JAX path is required" >&2
            exit 1
        }
        for repeat in $(seq 0 7); do
            if (( repeat % 2 == 0 )); then order=(ideal so3); else order=(so3 ideal); fi
            for vehicle in "${order[@]}"; do
                launch_ros "$map_seed" "$repeat" "$vehicle" "$SHORT_STEPS" \
                    "$ROOT/short" "$SHORT_CSV"
            done
        done
    done
}

run_stepwise_screen() {
    for map_seed in 516 539; do
        for repeat in 0 1 2 3 4; do
            uv run python -m ergodic_control_mppi.experiments.uav_diagnostics discrepancy-jax \
                --run-dir "$(map_run "$map_seed")" --config "$PROFILE" \
                --output "$SHORT_CSV" --map-seed "$map_seed" --seed "$SEED" \
                --steps "$SHORT_STEPS" --preflight-steps "$PREFLIGHT" --device "$DEVICE" \
                --hardware "$HARDWARE" --condition stepwise --repeat "$repeat" \
                --canonical "$(canonical "$map_seed")"
        done
    done
}

run_smoke() {
    mkdir -p "$ROOT/smoke"
    launch_ros 516 0 ideal 200 "$ROOT/smoke"
    launch_ros 516 0 so3 200 "$ROOT/smoke"
    touch "$ROOT/smoke/PASSED"
}

run_ros_campaign() {
    [[ -f "$ROOT/smoke/PASSED" ]] || {
        echo "run '$0 smoke' successfully before the repeated campaign" >&2
        exit 1
    }
    for map_seed in 516 539; do
        [[ -f "$(canonical "$map_seed")" ]] || {
            echo "missing $(canonical "$map_seed"); the reference JAX path is required" >&2
            exit 1
        }
        for repeat in $(seq 0 23); do
            if (( repeat % 2 == 0 )); then
                order=(ideal so3)
            else
                order=(so3 ideal)
            fi
            for vehicle in "${order[@]}"; do
                launch_ros "$map_seed" "$repeat" "$vehicle" "$STEPS" "$ROOT/ros" "$CSV"
            done
        done
    done
}

case "$ACTION" in
    determinism) run_determinism ;;
    envelope) run_envelope ;;
    short) run_short_screen ;;
    stepwise) run_stepwise_screen ;;
    smoke) run_smoke ;;
    ros) run_ros_campaign ;;
    report)
        uv run python -m ergodic_control_mppi.experiments.uav_diagnostics \
            discrepancy-report --input "$CSV" --output "$ROOT/report.md"
        ;;
    *)
        echo "usage: $0 determinism|envelope|short|stepwise|smoke|ros|report" >&2
        echo "Set DISCREPANCY_HARDWARE and DISCREPANCY_REFERENCE_HARDWARE explicitly across machines." >&2
        exit 2
        ;;
esac
