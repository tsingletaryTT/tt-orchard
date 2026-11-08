#!/bin/bash
# Take a 2-chip lease (queued behind earlier tickets; waits with `gozer wait`, never loops `gozer run`), run one command under it, release.
# usage: lease_run.sh "<reason>" <command...>
reason="$1"; shift
acq=$(mktemp /tmp/can-cad-acq.XXXXXX)
if ! gozer acquire --chips 2 --who "claude:can-cad" --reason "$reason" --expect 90m --owner-pid $$ > "$acq" 2>&1; then
  t=$(grep -o -E 'ticket[^0-9a-f]*[0-9a-f]{4}' "$acq" | grep -o -E '[0-9a-f]{4}$' | head -1)
  echo "queued ticket $t" >&2
  while true; do
    gozer wait --timeout 8m "$t" > "$acq" 2>&1          # the grant is in wait's own output
    if grep -q -E '^granted' "$acq"; then break; fi
    if gozer acquire --chips 2 --who "claude:can-cad" --reason "$reason" --expect 90m --ticket "$t" --owner-pid $$ > "$acq" 2>&1; then break; fi
  done
fi
cat "$acq" >&2
lease=$(grep -o -E 'lease [0-9a-f]{6}' "$acq" | head -1 | cut -d' ' -f2)
export $(grep -o -E 'TT_VISIBLE_DEVICES=[0-9a-fA-F:.,]+' "$acq" | head -1)
"$@"; rc=$?
[ -n "$lease" ] && gozer release "$lease" >&2
rm -f "$acq"
exit $rc
