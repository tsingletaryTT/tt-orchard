"""Helpers shared by the database tests: a small history, a fake wire, and a runner that kills a child at a named step.

`run_child(kind, directory, step)` starts `tests/crash_child.py` in a new process, which carries out one operation of the real
code (an open that recovers, a restore, a migration, a confirmation) and sends itself SIGKILL when the named step is reached. The
parent then reopens the directory from a second process, as a restart would.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from orchard import db as dbm
from orchard import telegraph_store as ts

REPO = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).resolve().parent / "crash_child.py"


def reserve(store, call_id, tokens=1000, cost=0.25, problem="p", decide=lambda records: None):
    """One reservation through the store; returns the refusal reason or None."""
    return store.reserve({"reserve": call_id, "problem": problem, "model": "small", "tokens": tokens, "cost_usd": cost}, decide,
                         lambda why: {"call_id": call_id, "problem": problem, "model": "small", "outcome": "budget", "reason": why,
                                      "cached": False, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})


def settle(store, call_id, tokens_in=60, tokens_out=40, cost=0.0, problem="p"):
    store.settle({"call_id": call_id, "problem": problem, "model": "small", "outcome": "ok", "cached": False, "tokens_in": tokens_in,
                  "tokens_out": tokens_out, "cost_usd": cost, "usage_known": True, "ts": 1.0})


def make_history(directory, n=5):
    """A directory whose ledger holds `n` unsettled reservations."""
    with ts.TelegraphStore.open(directory) as store:
        for i in range(n):
            assert reserve(store, f"c{i}") is None
    return Path(directory)


def run_child(kind, directory, step=None, *args, timeout=60):
    """Run one operation in a child process and kill it at `step`. Returns the completed process (returncode -9 when killed)."""
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    return subprocess.run([sys.executable, str(CHILD), kind, str(directory), step or "-", *map(str, args)], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=timeout)


KILLED = -signal.SIGKILL


def snapshot(directory):
    """Every regular file of a directory (not the subdirectories) with its bytes: a test that nothing was touched."""
    d = Path(directory)
    return sorted((p.name, p.read_bytes()) for p in d.iterdir() if p.is_file())


def blocked(directory):
    """The refusal a new opener gets when it asks for a reservation (None when it would be granted)."""
    with ts.TelegraphStore.open(directory) as store:
        return reserve(store, "probe-" + os.urandom(3).hex())


# --- a worker for the multi-process test: spawned children need a function they can import ------------------------------------

RETRY_PATIENCE_S = 120.0       # how long a worker keeps asking while the ledger says it is unavailable


def retry_when_unavailable(operation, what, *, patience_s=RETRY_PATIENCE_S, interval_s=0.02, clock=time.monotonic, sleep=time.sleep):
    """Run `operation()` and return its value, asking again while the ledger answers `LedgerUnavailable`.

    That answer means the write lock was not free inside the busy timeout (30 s). Nothing was recorded, and that is the designed
    refusal: a reservation that cannot be recorded cannot pass a cap. On a box at load 80 a process can wait that long for its time
    slice, so a worker that wants an answer asks again. The wait between asks grows to half a second. After `patience_s` seconds
    without an answer this raises AssertionError that names `what` and the last answer, so a ledger that is really stuck still fails
    the test. Any other exception passes through."""
    from orchard.db import LedgerUnavailable
    deadline, tries = clock() + patience_s, 0
    while True:
        try:
            return operation()
        except LedgerUnavailable as exc:
            tries += 1
            if clock() >= deadline:
                raise AssertionError(f"{what}: the ledger was unavailable for {patience_s:.0f} s ({tries} tries); last answer: {exc}") from exc
            sleep(min(0.5, interval_s * tries))


def store_worker(directory, table, index, count, problems, tokens, cost, settle_calls=False, busy_timeout_ms=None):
    """One process of the multi-process test: `count` reservations through the real Telegraph budget rules, optionally settled.
    An open, a reservation or a settle that the ledger could not take (`LedgerUnavailable`) is asked again until it is answered
    (`retry_when_unavailable`), so every reservation ends as granted or refused. `busy_timeout_ms` shortens the lock wait, which makes
    that answer common on purpose. Returns [(call_id, granted)]."""
    import types

    import orchard.telegraph as tg
    cfg = tg.load(table, base=Path(directory))
    stub = types.SimpleNamespace(cfg=cfg)
    out = []
    kw = {} if busy_timeout_ms is None else {"busy_timeout_ms": busy_timeout_ms}
    # Opening takes locks too (`open.lock`), and gets the same answer when they are not free in time.
    with retry_when_unavailable(lambda: ts.TelegraphStore.open(directory, **kw), f"worker {index} opening the ledger") as store:
        for k in range(count):
            call_id, problem = f"w{index}-{k}", problems[(index + k) % len(problems)]
            worst = {"tokens": tokens, "cost": cost}
            why = retry_when_unavailable(
                lambda: reserve(store, call_id, tokens=tokens, cost=cost, problem=problem,
                                decide=lambda records: tg.Telegraph._over_budget(stub, "small", worst, tg.accounting(records), problem)),
                f"reservation {call_id}")
            out.append((call_id, why is None))
            if settle_calls and why is None:
                retry_when_unavailable(lambda: settle(store, call_id, tokens_in=60, tokens_out=40, cost=0.0, problem=problem),
                                       f"settle of {call_id}")
    return out
