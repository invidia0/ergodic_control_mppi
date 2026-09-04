#!/usr/bin/env bash
# Resumable pillar tuning. Stages are explicit so no remote work starts by accident.
#
# Two obstacle densities run through this script, differing only in the pillar count and
# the safety margin, so the same arms are directly comparable across them. The defaults are
# the 25-pillar geometry. To resume the 45-pillar dense campaign:
#
#   PILLAR_ROOT=results/uav/pillar_tuning PILLAR_OBS_NUM=45 PILLAR_CLEARANCE=0.05 \
#   PILLAR_TRACKING_ALLOWANCE=0.05 PILLAR_REACTION_TIME=0.10 \
#       scripts/run_pillar_tuning.sh sweep
#
# Why 25 and a thinner margin: at 45 pillars and a 1.039 m inflation, 22.9% of the target
# probability mass fell inside inflated obstacles -- mass the vehicle is pulled toward and
# can never cover, which is what produced the observed local minima. At 25 pillars and
# 0.884 m it is 9.8%, with free space up from 0.671 to 0.840 and 80% of maps still blocking
# at least two of three mode-to-mode segments.

set -euo pipefail
cd "$(dirname "$0")/.."
# Launched detached over ssh, where the login profile that puts uv on PATH never runs.
# Same reason as scripts/run_uav_screen.sh.
export PATH="$HOME/.local/bin:$PATH"

ACTION="${1:---dry-run}"
ROOT="${PILLAR_ROOT:-results/uav/pillar_25}"
CONFIG="configs/uav_profile.yaml"
COMPOSE="docker compose -f docker/ros2/compose.yaml"
STEPS=20000
CAP_STEPS=10000
SWEEP_SEEDS=12

OBS_NUM="${PILLAR_OBS_NUM:-25}"
# The 15% margin cut, taken from measured slack rather than uniformly: clearance is
# documented as discretionary, the flown tracking error was pos_p95 = 0.022 m against the
# 0.05 allowed, and 0.06 s is three control periods at 50 Hz.
#
# Stated assumption, not a measurement: the stopping term inside inflation_radius assumes
# max_speed = 2.0 m/s, and a flight recorded 2.54 m/s, so the budget is already optimistic
# by ~0.26 m before this cut. Empirically there was room -- min clearance 0.94 m against a
# 0.30 m footprint, guard on 0.99% of samples -- but the assumption is written down here
# rather than buried in a default.
CLEARANCE="${PILLAR_CLEARANCE:-0.0}"
TRACKING_ALLOWANCE="${PILLAR_TRACKING_ALLOWANCE:-0.025}"
REACTION_TIME="${PILLAR_REACTION_TIME:-0.06}"

launch_args=(
    "config:=/workspace/$CONFIG"
    "map_source:=random_forest" "map_fill:=0.0" "obs_num:=$OBS_NUM"
    "pillar_min_radius:=0.3" "pillar_max_radius:=0.6"
    "pillar_min_height:=2.0" "pillar_max_height:=3.0"
    "pillar_min_distance:=1.2" "rviz:=false"
    "clearance:=$CLEARANCE" "tracking_allowance:=$TRACKING_ALLOWANCE"
    "reaction_time:=$REACTION_TIME"
)

value() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning value \
        --path "$1" --field "$2"
}

report() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning report \
        --root "$ROOT" --output "$ROOT/report.md"
}

# Probe range. Narrow it to sanity-check a new geometry before spending the full 100:
#   PILLAR_SEED_LAST=525 scripts/run_pillar_tuning.sh prepare
# Re-running with the full range fills in the rest; probed seeds are skipped by identity.
SEED_FIRST="${PILLAR_SEED_FIRST:-511}"
SEED_LAST="${PILLAR_SEED_LAST:-610}"
# How many qualifying seeds get a map built so selection can rank them. Six are kept; the
# rest is headroom, at roughly half a minute each.
CANDIDATES="${PILLAR_CANDIDATES:-15}"

# The arming gate *and* the metric denominator for snapshot flights. 20.0 is the real
# constraint: one 50 Hz control period.
#
# Raise it only to record a visualization flight on a machine whose warmup estimate trips
# the gate, and then judge the timing on the recorded `step_p99_ms` against 20 ms, not on
# this value. The warmup estimate is not a reliable stand-in for either direction --
# measured here it over-predicted the flight p99 by 32% on a laptop RTX PRO 500 and
# under-predicted it by ~40% on an RTX 5090.
DEADLINE_MS="${PILLAR_DEADLINE_MS:-20.0}"

# "so3" is the deployment. "ideal" swaps the airframe and attitude loop for a perfect
# tracker to measure what the vehicle costs against the offline model; runs recorded
# that way are diagnostics and carry no real-time or tracking meaning.
VEHICLE="${PILLAR_VEHICLE:-so3}"

prepare() {
    mkdir -p "$ROOT/maps" scratch
    for map_seed in $(seq "$SEED_FIRST" "$SEED_LAST"); do
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
    # Build the leading candidates before selecting, because the number the six are chosen
    # on -- target mass inside inflation -- needs a generated map, not the 6 s probe. Over
    # the first 25-pillar maps it ranged 10.4% to 24.7% while free space was flat at
    # 0.834-0.849, so selecting on the probe alone was choosing that number at random.
    build_maps $(uv run python -c '
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    seeds = [row["map_seed"] for row in csv.DictReader(stream) if int(row["qualifies"])]
print(" ".join(seeds[:int(sys.argv[2])]))
' "$ROOT/qualification.csv" "$CANDIDATES")

    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning select \
        --qualification "$ROOT/qualification.csv" --output "$ROOT/selection.json" \
        --candidates "$ROOT"

    # No-op when the six all came from the candidate pool; the guard is for a resumed run
    # whose selection.json predates it.
    build_maps $(value "$ROOT/selection.json" development) \
               $(value "$ROOT/selection.json" holdout)
}

build_maps() {
    index=0
    for map_seed in "$@"; do
        run_id="map_$map_seed"
        if [[ ! -f "$ROOT/maps/$run_id/arrays.npz" ]]; then
            # Bounded like the probe above. A launch whose controller node dies leaves the
            # others spinning and never exits, so one dead node wedges the build stage
            # indefinitely -- which is what a transient GPU failure did. A map either
            # records well inside this or it is not going to.
            set +e
            ROS_DOMAIN_ID=$((220 + index % 12)) \
                timeout "${PILLAR_BUILD_TIMEOUT:-180}" $COMPOSE run --rm \
                --name "pillar-build-$map_seed" uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                "${launch_args[@]}" "map_seed:=$map_seed" "steps:=1" "device:=gpu" \
                "preflight_steps:=0" "output_root:=/workspace/$ROOT/maps" "run_id:=$run_id"
            status=$?
            docker stop --timeout 1 "pillar-build-$map_seed" >/dev/null 2>&1 || true
            set -e
            if [[ ! -f "$ROOT/maps/$run_id/arrays.npz" ]]; then
                echo "  build of $run_id wrote no arrays (rc=$status); continuing" >&2
            fi
        fi
        index=$((index + 1))
    done
}

# The sweep map is the development representative of *this* geometry, not a remembered
# seed: changing obs_num changes the generator's draw sequence, so every seed yields a
# different field and the previous campaign's 543 means nothing here.
sweep_map() {
    echo "${PILLAR_SWEEP_MAP:-$(value "$ROOT/selection.json" development_representative)}"
}

sweep_report() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning sweep-report \
        --root "$ROOT" --output "$ROOT/sweep_report.md"
}

# Optional arm list, so the primary axes can be read before the rest is committed:
#   scripts/run_pillar_tuning.sh sweep baseline,theta_0,theta_15,theta_45,theta_60,theta_75
run_sweep() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning run \
        --run-dir "$ROOT/maps/map_$(sweep_map)" --output "$ROOT/sweep.csv" \
        --cap-output "$ROOT/sweep_cap.csv" --cap-steps "$CAP_STEPS" \
        --stage sweep --first-seed 43 --seeds "$SWEEP_SEEDS" --steps "$STEPS" \
        --device gpu --arms "${1:-}"
    sweep_report
}

# Separate from `sweep` because the export refuses an unbalanced campaign, which is the
# normal state part-way through a run and should not abort the progress report.
# Keyed by obstacle density, not map seed: the two campaigns share arm names, so the pair
# of exports is the density-scaling comparison.
export_ablation() {
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning sweep-report \
        --root "$ROOT" --output "$ROOT/sweep_report.md" \
        --ablation "results/uav/ablation_${OBS_NUM}_pillars.csv"
}

# Flights for the paper snapshot and video. Not gated: the sweep is exploratory and this
# produces a figure, not an accepted claim. Bags stay on the machine that records them.
#
# 200 preflight updates, matching the sweep the arm was selected in.
#
# 500 was tried, on the reasoning that 200 hands over an unsettled temperature: from the
# lambda = 1 start the adaptive loop reaches only 19 of its settled 80 by update 200, at an
# achieved ESS of 0.006 against the 0.3 target, and does not stay within 5% until ~470.
# Measured on one seed of T_150, that made the run worse, not better -- mode_cycles 1 -> 0
# and Fourier 0.0855 -> 0.2108 on the same machine. A cold handover is a *greedy* handover,
# and greedy is what tours: ESS correlates negatively with completing a tour (r = -0.21 over
# 138 cells) and the temperature-clipped `shipped` arm out-toured the uncapped ones. Settling
# the temperature first hands the vehicle a smooth, indecisive plan.
#
# The real-time constraint is the 50 Hz control period: 20 ms. Set to exactly that, so
# deadline_miss_fraction means "the solve did not fit in one control period" rather than
# "the solve exceeded a margin someone chose". The campaign's 16 ms is an 80% margin,
# strict enough to reject a loop that would have flown fine.
#
# Caveat if a future arm lands near the limit: publish, DDS and the shield share the same
# period, so a solve at 19 ms leaves nothing for them and the rate will slip even though
# the metric passes. Measured here T_150 warms up at 7.4 ms p99, so the question is moot.
run_snapshot() {
    arm="${1:?usage: $0 snapshot <arm-name> [seed ...]}"
    shift
    # Default to the five-flight set; a seed list flies just those, for a quick look.
    seeds=("$@")
    [[ ${#seeds[@]} -gt 0 ]] || mapfile -t seeds < <(seq 43 47)
    profile="$ROOT/sweep_${arm}.yaml"
    uv run python -m ergodic_control_mppi.experiments.uav_pillar_tuning config \
        --root "$ROOT" --arm "$arm" --output "$profile"
    map_seed="$(sweep_map)"
    snapshot_args=("${launch_args[@]}")
    snapshot_args[0]="config:=/workspace/$profile"
    for seed in "${seeds[@]}"; do
        run_id="flight_${arm}_${map_seed}_s${seed}"
        [[ "$VEHICLE" == "so3" ]] || run_id="${run_id}_${VEHICLE}"
        if [[ ! -f "$ROOT/$run_id/arrays.npz" ]]; then
            ROS_DOMAIN_ID=$((60 + seed - 43)) $COMPOSE run --rm uav \
                ros2 launch ergodic_control_mppi_ros uav.launch.py \
                "${snapshot_args[@]}" \
                "map_seed:=$map_seed" "steps:=$STEPS" "seed:=$seed" \
                "preflight_steps:=200" "profile:=sweep_$arm" "bag:=true" "device:=gpu" \
                "deadline_ms:=$DEADLINE_MS" "vehicle:=$VEHICLE" \
                "output_root:=/workspace/$ROOT" "run_id:=$run_id"
        fi
        if [[ ! -f "$ROOT/flights.csv" ]] || \
           ! awk -F, -v id="$run_id" 'NR > 1 && $1 == id && $3 == "ideal" { found=1 } END { exit !found }' "$ROOT/flights.csv"; then
            uv run python -m ergodic_control_mppi.experiments.uav_pair \
                --run-dir "$ROOT/$run_id" --device gpu --summary "$ROOT/flights.csv"
        fi
        uv run python -m ergodic_control_mppi.plotting.deployment \
            --run-dir "$ROOT/$run_id" --output "$ROOT/$run_id/snapshot.png"
    done
    # Two files, because the pair is written by two different processes: the recorder logs
    # the flown row (mode=uav) to summary.csv as the flight ends, and uav_pair appends the
    # ideal twin here afterwards. Judge a flight on summary.csv; flights.csv is the twin it
    # is being compared against.
    echo "Flown rows: $ROOT/summary.csv   ideal twins: $ROOT/flights.csv"
    echo "Then: scripts/replay_uav.sh ${ROOT#results/uav/}/<run_id> 0.25"
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
            ROS_DOMAIN_ID=$((65 + seed - 43)) $COMPOSE run --rm uav \
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
    sweep) run_sweep "${2:-}" ;;
    sweep-report) sweep_report ;;
    ablation) export_ablation ;;
    snapshot) shift; run_snapshot "$@" ;;
    *)
        echo "usage: $0 [--dry-run|campaign|prepare|screen|approach|holdout|online|report]" >&2
        echo "       $0 sweep [arm,arm,...] | sweep-report | ablation | snapshot <arm>" >&2
        exit 2
        ;;
esac
