#!/bin/bash
# waits on ticket $1, takes the lease owned by this script's pid, runs matrix.sh into $2, releases
T=$1; OUT=$2
until gozer wait $T --timeout 300 >/tmp/claude-1000/sc/wait.out 2>&1; do
  grep -qi "granted" /tmp/claude-1000/sc/wait.out && break
  sleep 2
done
gozer acquire --ticket $T --exact 2 --who "claude:granite-repeat" --reason "Granite repeat removal and trace: measurement matrix" --expect 15m --owner-pid $$ > /tmp/claude-1000/sc/acq.out 2>&1
cat /tmp/claude-1000/sc/acq.out
LEASE=$(grep -oE "lease [0-9a-f]{6}" /tmp/claude-1000/sc/acq.out | head -1 | cut -d' ' -f2)
export $(grep -oE "TT_VISIBLE_DEVICES=[^ ]+" /tmp/claude-1000/sc/acq.out | head -1)
bash ~/orchard-runs/ibm-granite--granite-4.0-h-1b/port2/matrix.sh $OUT
gozer release $LEASE
echo "RELEASED $LEASE"
