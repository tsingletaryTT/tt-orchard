"""Lease adapter for tt-gozer.

This module owns the translation between the supervisor's lease calls and the `gozer` command
line. It reads only gozer's `--json` output and exit codes (spec section 8, item 3). The flags and
exit codes below were read from tt-gozer `gozer/cli.py` on 2026-10-02. Committed `main` is gozer
0.3.2 and carries the `reset` command and the owner-pid ownership code (the working tree has an
uncommitted 0.3.3 version bump for its report archive).

Rules this adapter keeps:
- Every lease is taken with `--owner-pid` set to the supervisor's pid. gozer then judges the lease
  by that pid's liveness, so it cannot expire while no device is open during a swap.
- It never calls `gozer wait`. `wait` grants a lease with no owner pid, which falls back to gozer's
  15 minute detached window. A queued ticket is claimed by running `acquire --ticket` again.
- It never passes `--force`.
- `reset` and `release` are never killed on a timeout. Killing gozer would leave its `tt-smi -r`
  running with nobody watching. After such a timeout the adapter refuses a second reset or
  release of that lease.
"""
from __future__ import annotations

import json
import os

from orchard.adapters import (AdapterError, ChipState, Lease, LeaseLost, Queued, Refused,
                              ResetFailed, TicketGone)
from orchard.commands import CommandResult, run_command
from orchard.defaults import CMD_TIMEOUT_S, RESET_TIMEOUT_S

# gozer exit codes (tt-gozer gozer/cli.py).
EXIT_OK = 0
EXIT_QUEUED = 10
EXIT_UNAVAILABLE = 12          # acquire --no-queue found no chips; also a bad argument
EXIT_NO_LEASE = 13             # no such lease or ticket
EXIT_REFUSED = 15              # reset refused, or release refused after it may have reset first
EXIT_RESET_FAILED = 17         # reset ran tt-smi and it failed; the lease is untouched
EXIT_RESET_CHANGED_HANDS = 18  # reset ran, and afterwards the lease was gone or taken

# `gozer release` exits 0 even when its reset failed. These phrases in its message say so
# (orchard/hardware_check.py makes the same check).
RELEASE_RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")


class GozerAdapter:
    def __init__(self, *, gozer: str = "gozer", owner_pid: int | None = None, run=run_command,
                 timeout: float = CMD_TIMEOUT_S, reset_timeout: float = RESET_TIMEOUT_S):
        pid = os.getpid() if owner_pid is None else owner_pid
        # gozer counts the owner's descendants as the owner's own work only for owner pids above
        # 1, because every process descends from pid 1 (spec section 8, item 2).
        if pid <= 1:
            raise ValueError(f"owner pid must be above 1, got {pid}")
        self.gozer, self.owner_pid, self.run = gozer, pid, run
        self.timeout, self.reset_timeout = timeout, reset_timeout
        self.in_flight: set[str] = set()

    # ---- plumbing ---------------------------------------------------------------------------

    def _call(self, *args: str, long: bool = False) -> tuple[CommandResult, dict | None]:
        res = self.run([self.gozer, *args], self.reset_timeout if long else self.timeout,
                       kill_on_timeout=not long)
        try:
            payload = json.loads(res.stdout) if res.stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        return res, payload if isinstance(payload, dict) else None

    @staticmethod
    def _detail(res: CommandResult, payload: dict | None) -> str:
        msg = str((payload or {}).get("message") or (payload or {}).get("error") or "")
        return (msg or res.stderr.strip() or res.stdout.strip())[:500]

    def _acquire_argv(self, chips: int, who: str, reason: str) -> list[str]:
        if chips < 1:
            raise ValueError(f"chips must be at least 1, got {chips}")
        return ["acquire", "--chips", str(chips), "--who", who, "--reason", reason,
                "--owner-pid", str(self.owner_pid), "--json"]

    def _granted(self, res: CommandResult, payload: dict | None, *, op: str,
                 ticket: str | None = None) -> Lease:
        if res.timed_out:
            raise AdapterError(f"gozer {op} timed out after {self.timeout} s")
        rc = res.returncode
        if rc == EXIT_OK and payload and payload.get("granted"):
            return self._lease_from(payload)
        if rc == EXIT_QUEUED and payload and payload.get("queued") and payload.get("ticket"):
            raise Queued(str(payload["ticket"]), payload.get("position"))
        if rc == EXIT_NO_LEASE and ticket is not None:
            raise TicketGone(ticket)
        if rc == EXIT_UNAVAILABLE:
            raise Refused(self._detail(res, payload) or "no chips free")
        raise AdapterError(f"gozer {op} exited {rc}: {self._detail(res, payload)}")

    def _lease_from(self, p: dict) -> Lease:
        lid, chips, devs = p.get("lease_id"), p.get("chips"), p.get("dev_indices")
        env, units = p.get("env"), p.get("units")
        problems = []
        if not isinstance(lid, str) or not lid:
            problems.append("no lease_id")
        if not isinstance(chips, list) or not chips:
            problems.append("no chips")
            chips = []
        if not isinstance(devs, list) or len(devs) != len(chips):
            problems.append("dev_indices do not match chips")
        if not isinstance(units, list) or not units:
            problems.append("no units")
        if not isinstance(env, dict) or env.get("TT_VISIBLE_DEVICES") != ",".join(chips):
            problems.append("env does not list the granted chips")
        # The check that matters most: a lease judged by another pid is not held through a swap.
        if p.get("owner_pid") != self.owner_pid:
            problems.append(f"owner_pid {p.get('owner_pid')!r} is not this supervisor ({self.owner_pid})")
        if problems:
            note = ""
            if isinstance(lid, str) and lid:
                # Keep no lease the supervisor cannot use. The release resets the chips (about 42 s).
                r, _ = self._call("release", lid, "--json", long=True)
                note = f"; released it (gozer exit {r.returncode})"
            raise AdapterError("gozer granted a lease the adapter cannot use: "
                               + "; ".join(problems) + note)
        return Lease(lease_id=lid, chips=tuple(chips), dev_indices=tuple(int(i) for i in devs),
                     env={str(k): str(v) for k, v in env.items()}, units=tuple(units))

    def _not_in_flight(self, lease: Lease) -> None:
        if lease.lease_id in self.in_flight:
            raise Refused(f"a reset or release of lease {lease.lease_id} is still running; "
                          "wait for it to end and read `gozer status`")

    # ---- the LeaseAdapter calls ---------------------------------------------------------------

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease:
        argv = self._acquire_argv(chips, who, reason)
        if exact:
            argv += ["--exact", exact]
        if not queue:
            argv.append("--no-queue")
        return self._granted(*self._call(*argv), op="acquire")

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease:
        argv = self._acquire_argv(chips, who, reason) + ["--ticket", ticket]
        return self._granted(*self._call(*argv), op="claim", ticket=ticket)

    def cancel(self, ticket: str) -> None:
        res, payload = self._call("cancel", ticket, "--json")
        if res.returncode in (EXIT_OK, EXIT_NO_LEASE):   # 13: the ticket is already gone
            return
        raise AdapterError(f"gozer cancel exited {res.returncode}: {self._detail(res, payload)}")

    def release(self, lease: Lease) -> None:
        self._not_in_flight(lease)
        res, payload = self._call("release", lease.lease_id, "--json", long=True)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"gozer release {lease.lease_id} still running after "
                              f"{self.reset_timeout} s; left running", left_running=True)
        if res.returncode == EXIT_OK:
            msg = str((payload or {}).get("message", ""))
            if any(t in msg for t in RELEASE_RESET_FAILED_TEXT):
                raise ResetFailed(f"lease {lease.lease_id} released, but its reset did not "
                                  f"succeed: {msg}")
            return
        if res.returncode == EXIT_NO_LEASE:
            return                       # already gone: nothing is left to release
        if res.returncode == EXIT_REFUSED:
            raise Refused(self._detail(res, payload))
        raise AdapterError(f"gozer release exited {res.returncode}: {self._detail(res, payload)}")

    def reset(self, lease: Lease) -> None:
        self._not_in_flight(lease)
        res, payload = self._call("reset", lease.lease_id, "--json", long=True)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"gozer reset {lease.lease_id} still running after "
                              f"{self.reset_timeout} s; left running", left_running=True)
        rc, detail = res.returncode, self._detail(res, payload)
        if rc == EXIT_OK and (payload or {}).get("status") == "reset":
            return
        if rc == EXIT_NO_LEASE:
            raise LeaseLost(f"gozer has no lease {lease.lease_id}: {detail}")
        if rc == EXIT_REFUSED:
            # Nothing was done. Some refusals leave a valid lease in place (spec section 8,
            # item 1), so the caller reads the chip states and decides.
            raise Refused(detail)
        if rc == EXIT_RESET_FAILED:
            raise ResetFailed(detail)
        if rc == EXIT_RESET_CHANGED_HANDS:
            raise LeaseLost(f"the reset ran, and afterwards lease {lease.lease_id} was gone or "
                            f"taken: {detail}")
        raise AdapterError(f"gozer reset exited {rc}: {detail}")

    def status(self) -> list[ChipState]:
        res, payload = self._call("status", "--json")
        if res.returncode != EXIT_OK or not payload or not isinstance(payload.get("chips"), list):
            raise AdapterError(f"gozer status failed (exit {res.returncode}): "
                               f"{self._detail(res, payload)}")
        out = []
        for c in payload["chips"]:
            try:
                out.append(ChipState(bdf=str(c["bdf"]), state=str(c["state"]), who=c.get("who"),
                                     board=str(c.get("board") or ""), dev_index=c.get("dev_index"),
                                     lease_pid=c.get("pid"),
                                     pids_holding=tuple(c.get("pids_holding") or ())))
            except (KeyError, TypeError, ValueError) as exc:
                raise AdapterError(f"gozer status listed a chip the adapter cannot read: "
                                   f"{c!r} ({exc})") from exc
        return out
