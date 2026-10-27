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
import subprocess
from typing import NamedTuple

from orchard.commands import run_command

RELEASE_TIMEOUT_S = 90.0         # choice: a release resets the chips (about 42 s measured); an overrun is left running, not killed
CLAIM_POLL_S = 2.0               # between claims of the kept ticket (gozer wait itself looks once a second)
CLAIM_TIMEOUT_S = 60.0           # one `gozer acquire` call; gozer answers at once
HISTORY_EVENTS = 200

# `gozer release` exits 0 even when its reset failed. These phrases in its message say so (orchard/adapters/gozer.py).
RELEASE_RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")

EXIT_QUEUED = 10                 # gozer acquire: queued, ticket in the JSON
_NAME = re.compile(r"[0-9A-Za-z]+")


def _run_gozer(argv, timeout, **kw):
    """The real command runner, looked up at call time so that the test suite can refuse it (tests/conftest.py)."""
    return run_command(argv, timeout, **kw)


class ReleaseResult(NamedTuple):
    ok: bool        # the lease is released and its chips were reset
    held: bool      # the lease may still be held (refused, failed, overran or could not run)
    text: str       # what to tell a person when not ok


class Mine(NamedTuple):
    leases: list    # [{"lease_id", "chips", "env", "granted_at"}] granted to this pid and not released
    tickets: list   # [ticket, ...] queued by this pid
    readable: bool  # False when history or the queue could not be read


class LeaseAnswer(NamedTuple):
    grant: dict | None      # {"lease_id", "chips", "env"} when granted
    ticket: str | None      # when queued
    error: str | None       # when gozer refused or could not be asked


class GozerLeases:
    def __init__(self, *, who: str, pid: int | None = None, run=None, gozer: str = "gozer", clock=None, sleep=None):
        import time
        self.who, self.pid = who, os.getpid() if pid is None else pid
        self.run, self.gozer = run if run is not None else _run_gozer, gozer
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep

    # ---- asking ---------------------------------------------------------------------------------------------------

    def _argv(self, reason: str, *extra: str) -> list[str]:
        return [self.gozer, "acquire", "--json", "--chips", "1", "--exact", "1", "--who", self.who, "--reason", reason,
                "--owner-pid", str(self.pid), *extra]

    def request(self, reason: str) -> LeaseAnswer:
        """The first `gozer acquire --owner-pid <pid>`: a grant, a queue ticket, or the reason gozer said no."""
        return self._answer(self._argv(reason))

    def claim(self, ticket: str, reason: str) -> LeaseAnswer:
        """Claim the kept ticket. Exit 10 with the same ticket means not yet; a grant means the lease is ours."""
        if not _NAME.fullmatch(ticket or ""):
            return LeaseAnswer(None, None, f"not a ticket name: {ticket!r}")
        return self._answer(self._argv(reason, "--ticket", ticket))

    def _answer(self, argv: list[str]) -> LeaseAnswer:
        try:
            got = self.run(argv, CLAIM_TIMEOUT_S)
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
            answer = self.claim(ticket, reason)
            if answer.grant is not None:
                return answer.grant, None
            if answer.ticket is None:
                # Not a ticket and not a grant: gozer erred for a moment (seen on the real gozer while another lease's reset was
                # starting) or the ticket is gone. A lease granted to this pid whose answer was lost is taken; otherwise the wait
                # goes on while the queue still shows the ticket (or cannot be read) and ends, with the ticket cancelled, when
                # the queue shows it gone.
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
        queue = self._records("queue", [self.gozer, "queue", "--json"], "queue")
        return None if queue is None else any(str(q.get("ticket")) == ticket for q in queue)

    def _found_lease(self) -> dict | None:
        found = self.mine()
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
        try:
            got = self.run([self.gozer, "release", lease_id, "--json"], RELEASE_TIMEOUT_S, kill_on_timeout=False)
        except Exception as exc:
            return ReleaseResult(False, True, str(exc))
        if got.timed_out:
            return ReleaseResult(False, True, f"gozer release {lease_id} is still running after {RELEASE_TIMEOUT_S:.0f} s "
                                              "and was left running")
        text = ((got.stdout or "") + (got.stderr or "")).strip()
        if got.returncode == 0:
            message = _message(got.stdout) or text
            if any(t in message for t in RELEASE_RESET_FAILED_TEXT):
                return ReleaseResult(False, False, f"the lease was released but its chip reset did not succeed: {message}")
            return ReleaseResult(True, False, "")
        if got.returncode == 13:                             # no such lease: nothing is left to release
            return ReleaseResult(True, False, "")
        return ReleaseResult(False, True, text or f"gozer release exited {got.returncode}")

    def reclaim(self) -> str | None:
        """Give back everything gozer shows for this pid: cancel its tickets, release its leases. For an exception path.
        Returns what could not be given back (see unresolved), else None. Never raises Exception."""
        try:
            found = self.mine()
            for ticket in found.tickets:
                self.cancel(ticket)
            for lease in found.leases:
                self.release(lease["lease_id"])
        except Exception:
            pass
        try:
            return self.unresolved()
        except Exception:
            return "cannot confirm that nothing is held (gozer could not be read)"

    # ---- what gozer shows ------------------------------------------------------------------------------------------

    def mine(self) -> Mine:
        events = self._records("history", [self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], "history")
        queue = self._records("queue", [self.gozer, "queue", "--json"], "queue")
        if events is None or queue is None:
            return Mine([], [], False)
        ended = {e.get("lease_id") for e in events if e.get("event") in ("released", "reaped")}
        leases = []
        for e in events:
            if (e.get("event") == "granted" and e.get("pid") == self.pid and e.get("who") == self.who
                    and isinstance(e.get("lease_id"), str) and e["lease_id"] not in ended):
                chips = [c for c in e.get("chips") or [] if isinstance(c, str)]
                leases.append({"lease_id": e["lease_id"], "chips": chips,
                               "env": {"TT_VISIBLE_DEVICES": ",".join(chips)} if chips else {},
                               "granted_at": e.get("ts")})
        tickets = [str(q["ticket"]) for q in queue if q.get("pid") == self.pid and q.get("who") == self.who
                   and isinstance(q.get("ticket"), (str, int))]
        return Mine(leases, tickets, True)

    def unresolved(self) -> str | None:
        """None when gozer shows no lease granted to this pid and no ticket queued by it; else one sentence naming them (or
        saying gozer could not be read)."""
        found = self.mine()
        if not found.readable:
            return "cannot confirm that nothing is held (gozer history or queue is not readable)"
        notes = [f"lease {x['lease_id']} may still be held" for x in found.leases]
        notes += [f"ticket {t} is still queued" for t in found.tickets]
        return "; ".join(notes) + "; look with `gozer status`" if notes else None

    def duration(self, lease_id: str) -> float | None:
        """The seconds gozer says the lease was held (its `released` event), or None while it has none."""
        events = self._records("history", [self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], "history") or []
        for e in reversed(events):
            if e.get("event") == "released" and e.get("lease_id") == lease_id:
                d = e.get("duration_s")
                return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) and d >= 0 else None
        return None

    def _records(self, key: str, argv: list[str], field: str) -> list[dict] | None:
        try:
            got = self.run(argv, 60)
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
