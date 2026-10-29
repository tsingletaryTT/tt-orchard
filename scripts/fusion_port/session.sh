#!/bin/bash
# One config: start the slice server, wait for it, run bench.py (1 warm-up + 5 measured, 128 tokens) and a 16-token greedy check, stop it.
# Usage: session.sh LABEL MODELDIR [extra docker args...]
S=${ORCHARD_SCRATCH:?set the scratch directory}
LABEL=$1; MODEL=$2; shift 2
cn=fusion-$LABEL
$S/run/work/serve.sh $cn $MODEL 8033 "$@" > $S/serve_$LABEL.log 2>&1 &
t0=$(date +%s)
until curl -s -m 2 http://127.0.0.1:8033/v1/models >/dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 30 ] && ! docker ps --format "{{.Names}}" | grep -q "^$cn$" && { echo "EXITED $LABEL"; exit 1; }
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo "TIMEOUT $LABEL"; docker stop -t 60 $cn >/dev/null; exit 1; }
  sleep 5
done
echo "UP $LABEL after $(( $(date +%s) - t0 )) s"
python3 -I $S/run/work/bench.py 8033 5 128 $S/run/work/prompt.txt m 16 > $S/bench_$LABEL.log 2>&1
grep -E "^BENCH" $S/bench_$LABEL.log | cut -c1-900
docker stop -t 90 $cn >/dev/null 2>&1
echo "STOPPED $LABEL"
