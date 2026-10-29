#!/bin/bash
# queue for the board, take the lease the wait grants, run $MATRIX into $1 inside the 15-minute detached grace window, release
OUT=$1
LOG=<scratch>/acq2.out
gozer acquire --exact 2 --who "claude:granite-repeat" --reason "Granite repeat removal and trace: speed rounds" --expect 15m > $LOG 2>&1
rc=$?
if [ $rc -eq 10 ]; then
  T=$(grep -oE "ticket [0-9a-f]+" $LOG | head -1 | cut -d' ' -f2)
  echo "queued $T" ; 
  gozer wait $T --timeout 28000 > $LOG 2>&1
fi
cat $LOG
LEASE=$(grep -oE "lease [0-9a-f]{6}" $LOG | head -1 | cut -d' ' -f2)
[ -z "$LEASE" ] && { echo NO LEASE; exit 3; }
export $(grep -oE "TT_VISIBLE_DEVICES=[^ ]+" $LOG | head -1)
bash ~/orchard-runs/ibm-granite--granite-4.0-h-1b/port2/${MATRIX:-matrix.sh} $OUT
gozer release $LEASE
echo "RELEASED $LEASE"
