# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agent's `hardware_check` tool takes its chip through gozer and lets gozer keep the books.

The probe asks for the lease with `gozer acquire --owner-pid <the probe's pid>`, so gozer itself owns the cleanup of anything
that escapes: a lease whose owner pid is dead is reaped (`gozer reconcile`, run by the next `acquire`), whatever killed the probe.
A busy board gives a queue ticket, and the probe claims that same ticket with `gozer acquire --ticket T --owner-pid P` every
few seconds (verified against the real gozer: a claim that is not yet possible answers exit 10 with the same ticket, so the call
keeps its queue place and no second ticket is made; a possible claim is granted with `owner_pid` P). The harness then runs under
the lease's environment and the lease is released in one `finally`.

Nothing is remembered in this process. What the probe holds is read back from gozer: `mine()` is the unreleased leases that
`gozer history` shows granted to this pid, and the queue tickets that `gozer queue` shows with this pid. That makes the answer the
same for every attempt of a probe (there is one pid), for a probe that was interrupted and started again (the old pid's lease is
gozer's to reap), and for a grant whose output was lost (it is found by its pid). The history window is the last
HISTORY_EVENTS events; a lease older than that would not be seen.

The design note (docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md) compares this with the in-memory ledger it
replaced and lists every path from the ticket to the final release with the test of each. Nothing here opens a chip."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from typing import NamedTuple

from orchard.commands import CommandResult, run_command

RELEASE_TIMEOUT_S = 90.0         # choice: a release resets the chips (about 42 s measured); an overrun is left running, not killed
CLAIM_POLL_S = 2.0               # between claims of the kept ticket (gozer wait itself looks once a second)
ACQUIRE_TERM_GRACE_S = 5.0       # an interrupted `gozer acquire` gets this long to end after SIGTERM before it is killed
CLAIM_TIMEOUT_S = 60.0           # the most one `gozer acquire` call may take; gozer answers at once
CLAIM_MIN_TIMEOUT_S = 5.0        # the least: a claim is given the wait that is left, but never less than this
RECOVERY_TIMEOUT_S = 15.0        # one read of gozer's history or queue while recovering inside a wait or an exception
HISTORY_EVENTS = 200

# `gozer release` exits 0 even when its reset failed. These phrases in its message say so (orchard/adapters/gozer.py).
RELEASE_RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")

EXIT_QUEUED = 10                 # gozer acquire: queued, ticket in the JSON
_NAME = re.compile(r"[0-9A-Za-z]+")


def _run_gozer(argv, timeout, **kw):
    """The real command runner, looked up at call time so that the test suite can refuse it (tests/conftest.py). Used for the reads,
    `cancel` and `release`: a read or a cancel that an interrupt leaves running is harmless, and a release must never be cut short."""
    return run_command(argv, timeout, **kw)


def _run_acquire(argv, timeout, **kw):
    """The runner for `gozer acquire` (a request or a claim): it stops and reaps its child on any interrupt (see run_reaping)."""
    return run_reaping(argv, timeout, **kw)


def _popen(*args, **kw):
    """subprocess.Popen, looked up at call time so that the test suite can refuse it (tests/conftest.py)."""
    return subprocess.Popen(*args, **kw)


def _signal_group(proc, sig, killpg) -> None:
    """Signal the child's own process group (it was started in its own session). Never pid 1 or below, never our own group."""
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 1:
        return
    try:
        if pid == os.getpgrp():
            return
        killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _stop_and_reap(proc, killpg) -> None:
    """SIGTERM, up to ACQUIRE_TERM_GRACE_S for the child to end, then SIGKILL, then wait for it: no acquire is left running (it
    could still grant a lease after the probe moved on) and no zombie is left behind."""
    _signal_group(proc, signal.SIGTERM, killpg)
    try:
        proc.wait(timeout=ACQUIRE_TERM_GRACE_S)
        return
    except Exception:
        pass
    _signal_group(proc, signal.SIGKILL, killpg)
    try:
        proc.wait(timeout=ACQUIRE_TERM_GRACE_S)
    except Exception:
        pass


def run_reaping(argv, timeout: float, *, env: dict | None = None, popen=None, killpg=None, **_ignored) -> CommandResult:
    """Run one command like orchard.commands.run_command, but never leave the child behind. A timeout kills the group and waits
    for it. Any other exception, a KeyboardInterrupt above all, stops the child (SIGTERM, then SIGKILL after a grace) and waits
    for it before the exception goes on, so a `gozer acquire` that was interrupted cannot grant a lease after the probe has
    moved on. Whatever it granted before it ended is in gozer's history, and the caller's reclaim finds it."""
    argv = tuple(str(a) for a in argv)
    spawn = popen or _popen
    kill = killpg or os.killpg
    try:
        proc = spawn(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, env=env)
    except FileNotFoundError as exc:
        return CommandResult(argv, 127, "", f"not found: {exc}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(proc, signal.SIGKILL, kill)
        try:
            out, err = proc.communicate()
        except Exception:
            out, err = "", ""
        return CommandResult(argv, None, out, err, timed_out=True, pid=getattr(proc, "pid", None))
    except BaseException:
        _stop_and_reap(proc, kill)
        raise
    return CommandResult(argv, proc.returncode, out, err, pid=getattr(proc, "pid", None))


class ReleaseResult(NamedTuple):
    ok: bool        # the lease is released and its chips were reset
    held: bool      # the lease may still be held (refused, failed, overran or could not run)
    text: str       # what to tell a person when not ok


class Mine(NamedTuple):
    leases: list    # [{"lease_id", "chips", "env", "granted_at"}] granted to this pid and not released
    tickets: list   # [ticket, ...] queued by this pid
    readable: bool  # False when history or the queue could not be read


class Reclaimed(NamedTuple):
    left: str | None        # what gozer still shows for this pid afterwards (see unresolved), else None
    leases: list            # the leases that were found, each {"lease_id", "chips", "env", "granted_at"}


class LeaseAnswer(NamedTuple):
    grant: dict | None      # {"lease_id", "chips", "env"} when granted
    ticket: str | None      # when queued
    error: str | None       # when gozer refused or could not be asked


class GozerLeases:
    def __init__(self, *, who: str, pid: int | None = None, run=None, gozer: str = "gozer", clock=None, sleep=None):
        import time
        self.who, self.pid = who, os.getpid() if pid is None else pid
        # Every command but `release` runs under the reaping runner, so an interrupt never leaves a gozer child behind (an acquire
        # could still grant a lease, a cancel or a read could still be running). `release` keeps the leave-running runner: a reset
        # must not be cut short, and the lease it releases is tracked in `in_flight`. An injected runner serves every command.
        self.run = run if run is not None else _run_acquire
        self.run_release = run if run is not None else _run_gozer
        self.gozer = gozer
        self.run_acquire = self.run
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        # Releases this process started and has not seen finish (an interrupt, or an overrun that was left running): lease id ->
        # the left-running command's pid. A reclaim waits for gozer to say these are released instead of starting a second one.
        self.in_flight: dict[str, int | None] = {}

    # ---- asking ---------------------------------------------------------------------------------------------------

    def _argv(self, reason: str, *extra: str) -> list[str]:
        return [self.gozer, "acquire", "--json", "--chips", "1", "--exact", "1", "--who", self.who, "--reason", reason,
                "--owner-pid", str(self.pid), *extra]

    def request(self, reason: str) -> LeaseAnswer:
        """The first `gozer acquire --owner-pid <pid>`: a grant, a queue ticket, or the reason gozer said no."""
        return self._answer(self._argv(reason))

    def claim(self, ticket: str, reason: str, timeout: float = CLAIM_TIMEOUT_S) -> LeaseAnswer:
        """Claim the kept ticket. Exit 10 with the same ticket means not yet; a grant means the lease is ours."""
        if not _NAME.fullmatch(ticket or ""):
            return LeaseAnswer(None, None, f"not a ticket name: {ticket!r}")
        return self._answer(self._argv(reason, "--ticket", ticket), timeout)

    def _answer(self, argv: list[str], timeout: float = CLAIM_TIMEOUT_S) -> LeaseAnswer:
        try:
            got = self.run_acquire(argv, timeout)
        except Exception as exc:
            return LeaseAnswer(None, None, str(exc))
        try:
            data = json.loads(got.stdout)
        except (TypeError, ValueError):
            data = None
        data = data if isinstance(data, dict) else {}
        if got.returncode == 0 and data.get("granted") is True and isinstance(data.get("lease_id"), str) \
                and _NAME.fullmatch(data["lease_id"]):
            chips = [c for c in data.get("chips") or [] if isinstance(c, str)]
            env = data.get("env") if isinstance(data.get("env"), dict) else {}
            devices = env.get("TT_VISIBLE_DEVICES") if isinstance(env.get("TT_VISIBLE_DEVICES"), str) else None
            return LeaseAnswer({"lease_id": data["lease_id"], "chips": chips,
                                "env": {"TT_VISIBLE_DEVICES": devices or ",".join(chips)} if (devices or chips) else {}},
                               None, None)
        if got.returncode == EXIT_QUEUED and isinstance(data.get("ticket"), str) and _NAME.fullmatch(data["ticket"]):
            return LeaseAnswer(None, data["ticket"], None)
        text = ((got.stdout or "") + (got.stderr or "")).strip()
        return LeaseAnswer(None, None, text or f"gozer acquire exited {got.returncode}")

    def wait(self, ticket: str, reason: str, wait_s: float) -> tuple[dict | None, str | None]:
        """Claim `ticket` every CLAIM_POLL_S until `wait_s` has passed. (grant, cancel error): the grant, or None after the
        ticket is cancelled (the error is why it could not be, else None). A claim that could not be answered, or a cancel
        that finds the ticket gone, is checked against gozer's own record before the answer is None: a lease granted to this
        pid while the probe was not looking is returned, not lost."""
        end = self.clock() + max(wait_s, 0.0)
        while True:
            # A claim is given the wait that is left (never less than the floor), so a short wait is not stretched to a minute
            # by one slow call. The first claim is made even for a zero wait: that is the immediate check.
            left = end - self.clock()
            answer = self.claim(ticket, reason, timeout=min(CLAIM_TIMEOUT_S, max(left, CLAIM_MIN_TIMEOUT_S)))
            if answer.grant is not None:
                return answer.grant, None
            if answer.ticket is None:
                # Not a ticket and not a grant: gozer erred for a moment (seen on the real gozer while another lease's reset was
                # starting), the claim timed out, or the ticket is gone. A lease granted to this pid whose answer was lost is
                # taken; otherwise the wait goes on while the queue still shows the ticket (or cannot be read) and ends, with the
                # ticket cancelled, when the queue shows it gone.
                found = self._found_lease()
                if found is not None:
                    return found, None
                if self._ticket_queued(ticket) is False:
                    break
            if self.clock() >= end:
                break
            self.sleep(min(CLAIM_POLL_S, max(end - self.clock(), 0.0)))
        error = self.cancel(ticket)
        found = self._found_lease()
        return found, (error if found is None else None)

    def _ticket_queued(self, ticket: str) -> bool | None:
        queue = self._records("queue", [self.gozer, "queue", "--json"], "queue", RECOVERY_TIMEOUT_S)
        return None if queue is None else any(str(q.get("ticket")) == ticket for q in queue)

    def _found_lease(self) -> dict | None:
        found = self.mine(RECOVERY_TIMEOUT_S)
        return found.leases[0] if found.leases else None

    # ---- giving back ----------------------------------------------------------------------------------------------

    def cancel(self, ticket: str) -> str | None:
        """`gozer cancel TICKET`: None when the ticket is gone (cancelled, or already consumed), else the reason it is not."""
        if not _NAME.fullmatch(ticket or ""):
            return f"not a ticket name: {ticket!r}"
        try:
            got = self.run([self.gozer, "cancel", ticket], 60)
        except Exception as exc:
            return str(exc)
        if got.returncode in (0, 13):                        # 13: no such ticket, so it is gone already
            return None
        return ((got.stdout or "") + (got.stderr or "")).strip() or f"gozer cancel exited {got.returncode}"

    def release(self, lease_id: str) -> ReleaseResult:
        """`gozer release LEASE --json` (it resets the chips). Never killed when it overruns: a reset must not be cut short.
        What is still held afterwards is read back from gozer by mine(), not remembered here."""
        if not _NAME.fullmatch(lease_id or ""):
            return ReleaseResult(False, False, f"not a lease id: {lease_id!r}")
        self.in_flight[lease_id] = None              # an interrupt in the command leaves it here: the child may still run
        try:
            got = self.run_release([self.gozer, "release", lease_id, "--json"], RELEASE_TIMEOUT_S, kill_on_timeout=False)
        except Exception as exc:
            self.in_flight.pop(lease_id, None)       # it could not start (or failed to answer): nothing is running
            return ReleaseResult(False, True, str(exc))
        if got.timed_out:
            self.in_flight[lease_id] = got.pid
            return ReleaseResult(False, True, f"gozer release {lease_id} is still running after {RELEASE_TIMEOUT_S:.0f} s "
                                              "and was left running")
        self.in_flight.pop(lease_id, None)
        text = ((got.stdout or "") + (got.stderr or "")).strip()
        if got.returncode == 0:
            message = _message(got.stdout) or text
            if any(t in message for t in RELEASE_RESET_FAILED_TEXT):
                return ReleaseResult(False, False, f"the lease was released but its chip reset did not succeed: {message}")
            return ReleaseResult(True, False, "")
        if got.returncode == 13:                             # no such lease: nothing is left to release
            return ReleaseResult(True, False, "")
        return ReleaseResult(False, True, text or f"gozer release exited {got.returncode}")

    def reclaim(self, on_found=None) -> Reclaimed:
        """Give back everything gozer shows for this pid: cancel its tickets, release its leases. For an exception path.
        `on_found(lease)` is called for each lease before it is released, so the caller can record the grant (and start its
        chip-time clock) first. A lease whose release this process started and has not seen finish is not released again: the
        reclaim waits up to RELEASE_TIMEOUT_S for gozer's history to show it released, then leaves it, shown as unresolved.
        Returns what was found and what gozer still shows. Never raises Exception."""
        found = Mine([], [], False)
        try:
            found = self.mine(RECOVERY_TIMEOUT_S)
            for ticket in found.tickets:
                self.cancel(ticket)
            for lease in found.leases:
                if on_found is not None:
                    try:
                        on_found(lease)
                    except Exception:
                        pass
                if lease["lease_id"] in self.in_flight:
                    self._await_release(lease["lease_id"])
                else:
                    self.release(lease["lease_id"])
        except Exception:
            pass
        try:
            left = self.unresolved()
        except Exception:
            left = "cannot confirm that nothing is held (gozer could not be read)"
        return Reclaimed(left, found.leases)

    def _await_release(self, lease_id: str) -> bool:
        """Wait (up to RELEASE_TIMEOUT_S) for gozer's history to show `lease_id` released: the release that is still running
        finishes by itself. True when it did."""
        end = self.clock() + RELEASE_TIMEOUT_S
        while True:
            events = self._records("history", [self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], "history",
                                   RECOVERY_TIMEOUT_S)
            if events is not None and any(e.get("lease_id") == lease_id and e.get("event") in ("released", "reaped")
                                          for e in events):
                self.in_flight.pop(lease_id, None)
                return True
            if self.clock() >= end:
                return False
            self.sleep(min(CLAIM_POLL_S, max(end - self.clock(), 0.0)))

    # ---- what gozer shows ------------------------------------------------------------------------------------------

    def mine(self, timeout: float = 60.0) -> Mine:
        """What gozer shows for this pid. A read that fails does not discard the other: the leases from a readable history and
        the tickets from a readable queue are kept, and `readable` says whether both could be read."""
        events = self._records("history", [self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], "history", timeout)
        queue = self._records("queue", [self.gozer, "queue", "--json"], "queue", timeout)
        leases, tickets = [], []
        if events is not None:
            ended = {e.get("lease_id") for e in events if e.get("event") in ("released", "reaped")}
            for e in events:
                if (e.get("event") == "granted" and e.get("pid") == self.pid and e.get("who") == self.who
                        and isinstance(e.get("lease_id"), str) and e["lease_id"] not in ended):
                    chips = [c for c in e.get("chips") or [] if isinstance(c, str)]
                    leases.append({"lease_id": e["lease_id"], "chips": chips,
                                   "env": {"TT_VISIBLE_DEVICES": ",".join(chips)} if chips else {},
                                   "granted_at": e.get("ts")})
        if queue is not None:
            tickets = [str(q["ticket"]) for q in queue if q.get("pid") == self.pid and q.get("who") == self.who
                       and isinstance(q.get("ticket"), (str, int))]
        return Mine(leases, tickets, events is not None and queue is not None)

    def unresolved(self) -> str | None:
        """None when gozer shows no lease granted to this pid and no ticket queued by it; else one sentence naming them (or
        saying gozer could not be read)."""
        found = self.mine()
        notes = [f"lease {x['lease_id']} may still be held" for x in found.leases]
        notes += [f"ticket {t} is still queued" for t in found.tickets]
        if not found.readable:
            notes.append("cannot confirm that nothing else is held (gozer history or queue is not readable)")
        return "; ".join(notes) + "; look with `gozer status`" if notes else None

    def duration(self, lease_id: str) -> float | None:
        """The seconds gozer says the lease was held (its `released` event), or None while it has none."""
        events = self._records("history", [self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], "history") or []
        for e in reversed(events):
            if e.get("event") == "released" and e.get("lease_id") == lease_id:
                d = e.get("duration_s")
                return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) and d >= 0 else None
        return None

    def _records(self, key: str, argv: list[str], field: str, timeout: float = 60.0) -> list[dict] | None:
        try:
            got = self.run(argv, timeout)
            data = json.loads(got.stdout) if got.returncode == 0 else None
        except Exception:
            return None
        rows = data.get(field) if isinstance(data, dict) else None
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else None


def _message(stdout: str) -> str:
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError):
        return ""
    return str(data.get("message", "")) if isinstance(data, dict) else ""
