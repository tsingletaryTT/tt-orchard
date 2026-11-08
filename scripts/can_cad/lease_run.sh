#!/bin/bash
# Take a two-chip gozer lease, run one command under it, release. Queue handling follows orchard/adapters/gozer.py:
# - every lease is taken with --owner-pid set to this script's pid, so gozer judges it by this process and it cannot expire while a device is idle;
# - this script never calls `gozer wait` (it grants a lease with no owner pid and a 15 minute detached window);
# - only exit 10 with a ticket in gozer's JSON means "queued": the ticket is claimed by running `acquire --ticket` again, every LEASE_POLL_S seconds;
# - exit 14 (board serial unreadable while a neighbour resets) is tried again a few times, any other exit is an error and is passed on.
# usage: lease_run.sh "<reason>" <command...>
# Exports TT_VISIBLE_DEVICES, TT_LEASE_ID, TT_LEASE_OWNER_PID and TT_LEASE_DEV_INDICES (comma separated) to the command.
reason="$1"; shift
poll="${LEASE_POLL_S:-15}"; serial_wait="${LEASE_SERIAL_WAIT_S:-10}"
out=$(mktemp /tmp/can-cad-acq.XXXXXX)
trap 'rm -f "$out"' EXIT
base=(acquire --chips 2 --who "claude:can-cad" --reason "$reason" --expect 90m --owner-pid $$ --json)
ticket=""; serial_tries=0
while true; do
  if [ -n "$ticket" ]; then gozer "${base[@]}" --ticket "$ticket" > "$out" 2>&1; else gozer "${base[@]}" > "$out" 2>&1; fi
  rc=$?
  if [ $rc -eq 0 ] && grep -q '"granted"' "$out"; then break; fi
  if [ $rc -eq 10 ]; then
    t=$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p.get("ticket") or "") if p.get("queued") else print("")' "$out" 2>/dev/null)
    if [ -z "$t" ]; then echo "gozer said queued without a ticket: $(head -c 300 "$out")" >&2; exit 10; fi
    ticket="$t"; sleep "$poll"; continue
  fi
  if [ $rc -eq 14 ] && grep -q "unreadable tt_serial" "$out" && [ $serial_tries -lt 3 ]; then
    serial_tries=$((serial_tries + 1)); sleep "$serial_wait"; continue
  fi
  echo "gozer acquire failed (exit $rc): $(head -c 400 "$out")" >&2
  exit $rc
done
read -r lease owner devs vis < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["lease_id"], p["owner_pid"], ",".join(str(i) for i in p["dev_indices"]), ",".join(p["chips"]))' "$out")
export TT_LEASE_ID="$lease" TT_LEASE_OWNER_PID="$owner" TT_LEASE_DEV_INDICES="$devs" TT_VISIBLE_DEVICES="$vis"
echo "lease $lease on devices $devs" >&2
"$@"; rc=$?
gozer release "$lease" >&2
exit $rc
