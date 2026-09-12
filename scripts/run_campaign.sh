#!/usr/bin/env bash
# Run the ablation campaign stage by stage, logging to scratch/<stage>.log.
#
# Designed to be launched detached on the GPU box and polled from elsewhere:
#
#   ssh ars-admin@HOST 'cd ~/ergodic_control_mppi && nohup bash scripts/run_campaign.sh \
#       screening interactions components > scratch/campaign.log 2>&1 &'
#
# Poll by counting ROW= lines against the --dry-run total, and wait for the
# EXIT_ sentinel. Every stage is resumable: cells already in the stage CSV are
# skipped, so re-running after a crash or a reboot costs nothing.

set -u -o pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p scratch

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(screening interactions components)
fi

DEVICE="${DEVICE:-gpu}"
CONFIG="${CONFIG:-configs/experiments/ablation_campaign.yaml}"

echo "=== campaign start $(date -u +%FT%TZ) ==="
echo "commit : $(git rev-parse --short HEAD)"
echo "stages : ${STAGES[*]}"
echo "device : $DEVICE"
uv run python -m ergodic_control_mppi.experiments.ablation --config "$CONFIG" --dry-run

FAILED=()
for STAGE in "${STAGES[@]}"; do
    echo "=== stage $STAGE start $(date -u +%FT%TZ) ==="
    if uv run python -m ergodic_control_mppi.experiments.ablation \
            --config "$CONFIG" --stage "$STAGE" --device "$DEVICE" \
            2>&1 | tee "scratch/${STAGE}.log"; then
        # Derive the summary immediately: it is pure post-processing, so a later
        # stage crashing never costs the analysis of the stages that finished.
        uv run python -m ergodic_control_mppi.experiments.analyze \
            --campaign-dir results/campaign --stage "$STAGE" \
            2>&1 | tee -a "scratch/${STAGE}.log"
        echo "=== stage $STAGE done $(date -u +%FT%TZ) ==="
    else
        echo "=== stage $STAGE FAILED $(date -u +%FT%TZ) ==="
        FAILED+=("$STAGE")
    fi
done

# Figures are cheap and read only the archive, so refresh them at the end.
uv run python -m ergodic_control_mppi.plotting.ablation \
    --campaign-dir results/campaign --output-dir results/campaign/figures \
    --stage all || echo "figure pass reported problems"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "EXIT_OK all stages complete $(date -u +%FT%TZ)"
else
    echo "EXIT_FAIL stages: ${FAILED[*]}"
    exit 1
fi
