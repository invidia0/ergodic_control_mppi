#!/usr/bin/env bash
# Progress of a running campaign. Safe to run any time; reads logs only.
#   ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_campaign.sh
# Cell counts come from `ablation.py --dry-run`; update them if stages change.
set -u
cd /home/ars-admin/ergodic_control_mppi
STAGES="screening:225 interactions:882 core:560 structure:130 components:60 generalization:560"
LOGS=""; for s in $STAGES; do LOGS="$LOGS scratch/${s%%:*}.log"; done
printf '%-14s %9s %6s\n' stage done pct
tot=0; dn=0
for s in $STAGES; do
  n=${s%%:*}; t=${s##*:}
  # Count the stage CSVs, not the logs: run_campaign.sh truncates <stage>.log on
  # a resumed re-run, and a resume re-emits no ROW= for cells already done.
  c=0; [ -f results/campaign/$n.csv ] && c=$(( $(wc -l < results/campaign/$n.csv) - 1 ))
  k=0; [ -f results/campaign/${n}_skipped.csv ] && k=$(( $(wc -l < results/campaign/${n}_skipped.csv) - 1 ))
  sk=""; [ "$k" -gt 0 ] && sk=" (${k} skipped)"
  printf '%-14s %4d/%-4d %5d%%%s\n' "$n" "$c" "$t" "$((100*c/t))" "$sk"
  tot=$((tot+t)); dn=$((dn+c))
done
printf '%-14s %4d/%-4d %5d%%\n' TOTAL "$dn" "$tot" "$((100*dn/tot))"
pgrep -f "^bash scripts/run_campaign" >/dev/null && st=RUNNING || st="NOT RUNNING"
echo "state : $st   gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "last  : $(grep -hE '^ROW=' $LOGS 2>/dev/null | tail -1)"
# Pace from the 40 most recent runs only, so it tracks the stage actually running.
sec=$(grep -hoE ' [0-9]+\.[0-9]s$' $LOGS 2>/dev/null | tr -d ' s' | tail -40 \
      | awk '{s+=$1;n++} END{if(n)printf "%.1f", s/n}')
[ -n "$sec" ] && echo "pace  : ${sec}s/run (last 40) -> $(awk -v s="$sec" -v r="$((tot-dn))" 'BEGIN{printf "%.1f", s*r/3600}')h left at this rate"
# Read the newest driver log, not campaign.log: a stage resumed in its own run
# logs elsewhere, leaving campaign.log's stale EXIT_FAIL to be reported forever.
drv=$(grep -lE '^EXIT_' scratch/*.log 2>/dev/null | xargs -r ls -t | head -1)
err=$(grep -hciE "RESOURCE_EXHAUSTED|out of memory|Traceback" $LOGS $drv 2>/dev/null | paste -sd+ | bc)
echo "errors: ${err:-0}   $(grep -hE '^EXIT_' $drv 2>/dev/null | tail -2)"
