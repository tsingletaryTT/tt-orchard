# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lease and ticket bookkeeping behind the agent's `hardware_check` tool.

A busy board is waited for on one queue ticket (`gozer wait`), and the harness then runs under the lease that wait granted.
Between taking the ticket and the final release there are several ways to end up holding something: a grant that arrives while
the waiter is being stopped, an interrupt at any point, a release that is refused, overruns or exits 0 without resetting the
chips. This module keeps one rule for all of them: register before acting. A ticket is registered when the probe keeps it, a
lease the moment it is granted (or found in gozer's history), and each is dropped only when gozer confirms it is gone. The tool
asks `unresolved()` before it starts another call, so a lease or ticket that an earlier call left is never stacked under a
new one. The table of paths and the test that covers each is in docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md.

Nothing here opens a chip. The commands are `gozer wait`, `release`, `cancel`, `history` and `queue`."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from typing import NamedTuple

from orchard.commands import run_command

RELEASE_TIMEOUT_S = 90.0         # choice: a release resets the chips, about 42 s measured; an overrun is left running, not killed
WAIT_SLACK_S = 10.0              # how long the probe lets `gozer wait` run past its own timeout before it stops it
HISTORY_EVENTS = 200             # events of `gozer history` read when looking for a grant or a release
MIN_TOKEN = 6                    # a token shorter than this could match any reason

# `gozer release` exits 0 even when its reset failed. These phrases in its message say so (orchard/adapters/gozer.py).
RELEASE_RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")

_NAME = re.compile(r"[0-9A-Za-z]+")


class ReleaseResult(NamedTuple):
    ok: bool        # the lease is released and its chips were reset
    held: bool      # the lease may still be held (refused, failed, overran or could not run)
    text: str       # what to tell a person when not ok


class LeaseLedger:
    """Pending tickets and leases of the calls of one attempt, and the gozer commands that settle them."""

    def __init__(self, *, run=run_command, gozer: str = "gozer", popen=subprocess.Popen, alive=None):
        self.run, self.gozer, self.popen = run, gozer, popen
        self.alive = alive if alive is not None else _pid_alive
        self.pending_tickets: dict[str, dict] = {}
        self.pending_leases: dict[str, dict] = {}

    # ---- registration -----------------------------------------------------------------------------------------

    def add_ticket(self, ticket: str) -> None:
        self.pending_tickets[ticket] = {}

    def add_lease(self, lease_id: str, *, why: str, pid: int | None = None) -> None:
        self.pending_leases[lease_id] = {"why": why, "pid": pid}

    # ---- the commands -----------------------------------------------------------------------------------------

    def cancel(self, ticket: str) -> str | None:
        """`gozer cancel TICKET`. None when the ticket is gone (cancelled, or already consumed), else the reason it is not.
        A ticket that could not be cancelled stays pending."""
        if not _NAME.fullmatch(ticket or ""):
            return f"not a ticket name: {ticket!r}"
        try:
            got = self.run([self.gozer, "cancel", ticket], 60)
        except Exception as exc:
            self.pending_tickets.setdefault(ticket, {})
            return str(exc)
        if got.returncode in (0, 13):                    # 13: no such ticket, so it is gone already
            self.pending_tickets.pop(ticket, None)
            return None
        self.pending_tickets.setdefault(ticket, {})
        return ((got.stdout or "") + (got.stderr or "")).strip() or f"gozer cancel exited {got.returncode}"

    def release(self, lease_id: str) -> ReleaseResult:
        """`gozer release LEASE --json` (it resets the chips). The lease is registered before the command runs, so an
        interrupt cannot lose it, and dropped only when gozer says it is gone. The command is never killed when it overruns:
        a reset must not be cut short. See ReleaseResult."""
        if not _NAME.fullmatch(lease_id or ""):
            return ReleaseResult(False, False, f"not a lease id: {lease_id!r}")
        self.pending_leases.setdefault(lease_id, {"why": "release", "pid": None})
        try:
            got = self.run([self.gozer, "release", lease_id, "--json"], RELEASE_TIMEOUT_S, kill_on_timeout=False)
        except Exception as exc:
            return ReleaseResult(False, True, str(exc))
        if got.timed_out:
            self.pending_leases[lease_id]["pid"] = got.pid
            return ReleaseResult(False, True, f"gozer release {lease_id} is still running after {RELEASE_TIMEOUT_S:.0f} s "
                                              "and was left running")
        text = ((got.stdout or "") + (got.stderr or "")).strip()
        if got.returncode == 0:
            message = _message(got.stdout) or text
            self.pending_leases.pop(lease_id, None)
            if any(t in message for t in RELEASE_RESET_FAILED_TEXT):
                return ReleaseResult(False, False, f"the lease was released but its chip reset did not succeed: {message}")
            return ReleaseResult(True, False, "")
        if got.returncode == 13:                         # no such lease: nothing is left to release
            self.pending_leases.pop(lease_id, None)
            return ReleaseResult(True, False, "")
        return ReleaseResult(False, True, text or f"gozer release exited {got.returncode}")

    # ---- what an earlier call left behind ---------------------------------------------------------------------

    def unresolved(self) -> str | None:
        """None when no ticket or lease of an earlier call is pending (or gozer confirms each is gone, and it is dropped);
        else one sentence that names them. Never raises."""
        notes = []
        if self.pending_leases:
            events = self._history()
            for lease_id, info in list(self.pending_leases.items()):
                if events is None:
                    notes.append(f"lease {lease_id}: cannot confirm its release (gozer history is not readable)")
                elif any(e.get("lease_id") == lease_id and e.get("event") in ("released", "reaped") for e in events):
                    self.pending_leases.pop(lease_id, None)
                elif info.get("pid") and self.alive(info["pid"]):
                    notes.append(f"lease {lease_id}: its release is still running (pid {info['pid']})")
                else:
                    notes.append(f"lease {lease_id} may still be held")
        if self.pending_tickets:
            queue = self._queue()
            for ticket in list(self.pending_tickets):
                if queue is None:
                    notes.append(f"ticket {ticket}: cannot confirm it left the queue (gozer queue is not readable)")
                elif ticket not in queue:
                    self.pending_tickets.pop(ticket, None)
                else:
                    notes.append(f"ticket {ticket} is still queued")
        return "; ".join(notes) + "; look with `gozer status`" if notes else None

    def orphan_grant(self, token: str) -> dict | None:
        """A lease that gozer granted for a request whose reason carries `token` and that was never released, from `gozer
        history`: the grant a stopped waiter did not get to print. {"lease_id", "chips", "env"} or None."""
        if not token or len(token) < MIN_TOKEN:
            return None
        events = self._history()
        if events is None:
            return None
        ended = {e.get("lease_id") for e in events if e.get("event") in ("released", "reaped")}
        for e in reversed(events):
            if (e.get("event") == "granted" and isinstance(e.get("lease_id"), str) and e["lease_id"] not in ended
                    and token in str(e.get("reason") or "") and _NAME.fullmatch(e["lease_id"])):
                chips = [c for c in e.get("chips") or [] if isinstance(c, str)]
                return {"lease_id": e["lease_id"], "chips": chips, "env": {"TT_VISIBLE_DEVICES": ",".join(chips)}}
        return None

    # ---- the waiter -------------------------------------------------------------------------------------------

    def wait_for_grant(self, ticket: str, wait_s: float, token: str) -> dict | None:
        """Wait on `ticket` with `gozer wait --json --timeout Ns` (bare digits mean minutes to gozer, so seconds are
        spelled out). gozer ends the wait itself; the probe stops the waiter only if it runs WAIT_SLACK_S past that. A grant
        is returned ({"lease_id", "chips", "env"}) and registered as a pending lease at once. After a stop or an interrupt the
        waiter's last output and gozer's history are read, so a lease granted while the waiter was being stopped is not lost:
        it is used after a timeout and released before an interrupt goes on. None when the ticket was not granted (the caller
        cancels it). Interrupts are re-raised after the ticket is cancelled and any grant released."""
        if not _NAME.fullmatch(ticket or ""):
            return None
        seconds = max(math.ceil(wait_s), 1)
        try:
            proc = self.popen([self.gozer, "wait", "--json", "--timeout", f"{seconds}s", ticket],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError:
            return None
        grant = None
        try:
            try:
                out, _ = proc.communicate(timeout=wait_s + WAIT_SLACK_S)
                grant = _parse_grant(out) if proc.returncode == 0 else None
            except subprocess.TimeoutExpired:
                proc.kill()
                grant = _parse_grant(self._drain(proc)) or self.orphan_grant(token)
            if grant is not None:
                self.add_lease(grant["lease_id"], why="granted")
                self.pending_tickets.pop(ticket, None)
            return grant
        except BaseException:
            # An interrupt (or any error) at any point: stop the waiter, find a grant it made (in the output it printed, in
            # what was already registered, or in gozer's history), release it, and otherwise cancel the ticket.
            proc.kill()
            found = grant
            try:
                found = found or _parse_grant(self._drain(proc))
            finally:
                found = found or self.orphan_grant(token)
                lease_id = (found or {}).get("lease_id")
                if lease_id:
                    self.add_lease(lease_id, why="granted during an interrupt")
                    self.release(lease_id)
                else:
                    self.cancel(ticket)
            raise

    @staticmethod
    def _drain(proc) -> str:
        try:
            return proc.communicate()[0] or ""
        except Exception:
            return ""

    # ---- gozer's own records ----------------------------------------------------------------------------------

    def _history(self) -> list[dict] | None:
        try:
            got = self.run([self.gozer, "history", "--json", "-n", str(HISTORY_EVENTS)], 60)
            data = json.loads(got.stdout) if got.returncode == 0 else None
        except Exception:
            return None
        events = data.get("history") if isinstance(data, dict) else None
        return [e for e in events if isinstance(e, dict)] if isinstance(events, list) else None

    def _queue(self) -> set[str] | None:
        try:
            got = self.run([self.gozer, "queue", "--json"], 60)
            data = json.loads(got.stdout) if got.returncode == 0 else None
        except Exception:
            return None
        entries = data.get("queue") if isinstance(data, dict) else None
        return ({str(e.get("ticket")) for e in entries if isinstance(e, dict)} if isinstance(entries, list) else None)


def _parse_grant(out: str) -> dict | None:
    try:
        data = json.loads(out)
    except (TypeError, ValueError):
        return None
    if not (isinstance(data, dict) and data.get("granted") is True and isinstance(data.get("lease_id"), str)
            and _NAME.fullmatch(data["lease_id"])):
        return None
    devices = (data.get("env") or {}).get("TT_VISIBLE_DEVICES") if isinstance(data.get("env"), dict) else None
    chips = [c for c in data.get("chips") or [] if isinstance(c, str)]
    return {"lease_id": data["lease_id"], "chips": chips,
            "env": {"TT_VISIBLE_DEVICES": devices} if isinstance(devices, str) and devices else {}}


def _message(stdout: str) -> str:
    try:
        data = json.loads(stdout)
    except (TypeError, ValueError):
        return ""
    return str(data.get("message", "")) if isinstance(data, dict) else ""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
