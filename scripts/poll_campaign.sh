#!/usr/bin/env bash
# Progress of a running campaign. Safe to run any time; reads logs only.
#   ssh ars-admin@155.185.245.31 "bash -s" < scripts/poll_campaign.sh
# Cell counts come from `ablation.py --dry-run`; update them if stages change.
set -u
cd /home/ars-admin/ergodic_control_mppi
STAGES="screening:225 interactions:882 core:560 structure:130 components:60"
LOGS=""; for s in $STAGES; do LOGS="$LOGS scratch/${s%%:*}.log"; done
printf '%-14s %9s %6s\n' stage done pct
tot=0; dn=0
for s in $STAGES; do
  n=${s%%:*}; t=${s##*:}
  c=$(grep -cE '^ROW=' scratch/$n.log 2>/dev/null | head -1); c=${c:-0}
  k=$(grep -cE '^SKIP=' scratch/$n.log 2>/dev/null | head -1); k=${k:-0}
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
err=$(grep -hciE "RESOURCE_EXHAUSTED|out of memory|Traceback" $LOGS scratch/campaign.log 2>/dev/null | paste -sd+ | bc)
echo "errors: ${err:-0}   $(grep -hE '^EXIT_' scratch/campaign.log 2>/dev/null | tail -2)"
