#!/bin/bash
# queue for board 1 (2 chips), take the lease the wait grants, run $1 (a script) with OUTDIR $2 inside the 15-minute detached grace window, release
LOG=<scratch>/acq3.out
gozer acquire --exact 2 --who "claude:granite-package" --reason "${REASON:-Granite package measurements}" --expect 15m > $LOG 2>&1
if [ $? -eq 10 ]; then
  T=$(grep -oE "ticket [0-9a-f]+" $LOG | head -1 | cut -d' ' -f2); echo "queued $T"
  gozer wait $T --timeout 28000 > $LOG 2>&1
fi
cat $LOG
LEASE=$(grep -oE "lease [0-9a-f]{6}" $LOG | head -1 | cut -d' ' -f2)
[ -z "$LEASE" ] && { echo NO LEASE; exit 3; }
export $(grep -oE "TT_VISIBLE_DEVICES=[^ ]+" $LOG | head -1)
echo "LEASED $LEASE $(date -Is)"
bash "$1" "$2"
gozer release $LEASE
echo "RELEASED $LEASE $(date -Is)"
