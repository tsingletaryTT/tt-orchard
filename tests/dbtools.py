"""Helpers shared by the database tests: a small history, a fake wire, and a runner that kills a child at a named step.

`run_child(kind, directory, step)` starts `tests/crash_child.py` in a new process, which carries out one operation of the real
code (an open that recovers, a restore, a migration, a confirmation) and sends itself SIGKILL when the named step is reached. The
parent then reopens the directory from a second process, as a restart would.
"""
import os
import signal
import subprocess
import sys
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

def store_worker(directory, table, index, count, problems, tokens, cost, settle_calls=False):
    """One process of the multi-process test: `count` reservations through the real Telegraph budget rules, optionally settled.
    Returns [(call_id, granted)]."""
    import types

    import orchard.telegraph as tg
    cfg = tg.load(table, base=Path(directory))
    stub = types.SimpleNamespace(cfg=cfg)
    out = []
    with ts.TelegraphStore.open(directory) as store:
        for k in range(count):
            call_id, problem = f"w{index}-{k}", problems[(index + k) % len(problems)]
            worst = {"tokens": tokens, "cost": cost}
            why = reserve(store, call_id, tokens=tokens, cost=cost, problem=problem,
                          decide=lambda records: tg.Telegraph._over_budget(stub, "small", worst, tg.accounting(records), problem))
            out.append((call_id, why is None))
            if settle_calls and why is None:
                settle(store, call_id, tokens_in=60, tokens_out=40, cost=0.0, problem=problem)
    return out
