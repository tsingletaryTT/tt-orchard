"""A child process for the crash tests (see tests/dbtools.py): one operation of the real code, killed at a named step.

usage: crash_child.py KIND DIRECTORY STEP [ARGUMENT ...]    STEP is a hook name such as "after:3", or "-" for none.

  recover DIR STEP              open the database (a damaged file triggers the recovery steps)
  restore DIR STEP BACKUP       restore a backup
  migrate DIR STEP [N]          migrate the telegraph directory (N is --accept-malformed)
  confirm DIR STEP              confirm a recovery
  hold-after-reserve DIR - ID   reserve, print RESERVED, wait to be killed
  hold-in-transaction DIR - ID  reserve and wait inside the transaction (printing INSIDE) to be killed
  storm DIR - PREFIX EVERY      reserve and settle until killed, printing each commit
  probe DIR - ID                ask for one reservation and print GRANTED or REFUSED <reason>
"""
import os
import signal
import sys
import time


def killer(target):
    def hook(step):
        if step == target:
            os.kill(os.getpid(), signal.SIGKILL)
    return hook


def main(argv):
    kind, directory, step, *rest = argv
    hook = killer(step)
    from orchard import db as dbm
    from orchard import telegraph_migrate as mig
    from orchard import telegraph_store as ts
    if kind == "recover":
        with dbm.Database(directory, hook=hook):
            pass
    elif kind == "restore":
        dbm.restore_backup(directory, rest[0], hook=hook)
    elif kind == "migrate":
        accept = int(rest[0]) if rest else None
        mig.migrate(directory, hook=hook, accept_malformed=accept)
    elif kind == "confirm":
        with dbm.Database(directory, hook=hook) as db:
            db.confirm_reconciled("checked")
    elif kind in ("hold-after-reserve", "hold-in-transaction"):
        call_id = rest[0]
        with ts.TelegraphStore.open(directory) as store:
            reservation = {"reserve": call_id, "problem": "p", "model": "small", "tokens": 1500, "cost_usd": 0.5}
            refusal = lambda why: {"call_id": call_id, "outcome": "budget", "reason": why}      # noqa: E731
            if kind == "hold-after-reserve":
                store.reserve(reservation, lambda records: None, refusal)
                print("RESERVED", flush=True)
            else:
                def decide(records):
                    return None
                with store.db.write() as conn:
                    store._insert(conn, reservation)
                    print("INSIDE", flush=True)
                    time.sleep(120)
            time.sleep(120)
    elif kind == "sidecar-kill":                                # killed halfway through writing the torn-tail sidecar
        from orchard import safeio

        real_write = safeio._write_all

        def half_then_die(fd, data):
            if b"half-written-fragment" not in bytes(data):          # only the sidecar write is cut short (the marker write is whole)
                return real_write(fd, data)
            os.write(fd, data[: len(data) // 2])
            os.kill(os.getpid(), signal.SIGKILL)
        safeio._write_all = half_then_die
        mig.migrate(directory)
    elif kind == "probe":                                       # a second process asks for a reservation and says what happened
        with ts.TelegraphStore.open(directory) as store:
            why = store.reserve({"reserve": rest[0], "problem": "p", "model": "small", "tokens": 10, "cost_usd": 0.0},
                                lambda records: None, lambda why: {"call_id": rest[0], "outcome": "budget", "reason": why})
        print("GRANTED" if why is None else "REFUSED " + why, flush=True)
    elif kind == "storm":
        prefix, every = rest[0], int(rest[1])
        with ts.TelegraphStore.open(directory) as store:
            i = 0
            while True:
                call_id = f"{prefix}-{i}"
                store.reserve({"reserve": call_id, "problem": "p", "model": "small", "tokens": 1000 + i % 7, "cost_usd": 0.25},
                              lambda records: None, lambda why: {"call_id": call_id, "outcome": "budget", "reason": why})
                print("R", call_id, flush=True)
                if i % every:
                    store.settle({"call_id": call_id, "problem": "p", "model": "small", "outcome": "ok", "cached": False,
                                  "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0625, "usage_known": bool(i % 2)})
                    print("S", call_id, flush=True)
                i += 1
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
