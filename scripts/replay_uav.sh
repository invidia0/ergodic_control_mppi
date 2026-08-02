#!/usr/bin/env bash
# Bring up the replay container and play a recorded run's bag in RViz.
#
# Playback only: no simulator, no controller, no guard. Everything on screen is recorded
# data, so a screenshot taken here can never be a picture of a fresh, different run.
#
#   scripts/replay_uav.sh                 # newest run that has a bag
#   scripts/replay_uav.sh paper01         # a named run under results/uav/
#   scripts/replay_uav.sh pillar/online_511_s43  # nested campaign run
#   scripts/replay_uav.sh paper01 0.25    # quarter speed, for screenshots
#
# Requires a bag, which only exists for runs launched with `bag:=true`.

set -euo pipefail

RUN_ID="${1:-}"
RATE="${2:-1.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$REPO/docker/ros2/compose.yaml"

cd "$REPO"

if [[ -z "$RUN_ID" ]]; then
    # Newest directory that actually contains a bag, not merely the newest run.
    RUN_ID="$(find results/uav -mindepth 2 -type d -name bag -printf '%T@ %h\n' 2>/dev/null \
              | sort -rn | head -1 | awk '{print $2}' | sed 's#^results/uav/##')"
    [[ -n "$RUN_ID" ]] || {
        echo "No run under results/uav/ has a bag." >&2
        echo "Record one with:  ros2 launch ergodic_control_mppi_ros uav.launch.py bag:=true ..." >&2
        exit 1
    }
    echo "Using newest bagged run: $RUN_ID"
fi

BAG="results/uav/$RUN_ID/bag"
[[ -d "$REPO/$BAG" ]] || { echo "No bag at $BAG" >&2; exit 1; }

: "${DISPLAY:?DISPLAY is not set -- run this from a graphical session}"
# compose.yaml requires XAUTHORITY; on most desktops it is set, but fall back to the
# conventional location rather than failing with a bare substitution error.
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
[[ -f "$XAUTHORITY" ]] || { echo "XAUTHORITY file not found: $XAUTHORITY" >&2; exit 1; }
export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
# Keep playback off the default domain so it cannot discover, or be discovered by, a
# headless trial running at the same time.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

# Let the container's X client talk to the host server. Scoped to the local user, and
# reverted on exit however the script ends.
if command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
    trap 'xhost -SI:localuser:"$(id -un)" >/dev/null 2>&1 || true' EXIT
fi

echo "Replaying $BAG at ${RATE}x on ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "(RViz opens on \$DISPLAY=$DISPLAY; Ctrl-C to stop)"

exec docker compose -f "$COMPOSE" run --rm replay \
    ros2 launch ergodic_control_mppi_ros replay.launch.py \
        bag:="/workspace/$BAG" rate:="$RATE"
