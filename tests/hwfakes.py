# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the agent hardware tool: a gozer that keeps state, and an adapter that runs the REAL wrapper (orchard/hwtool.py) in
process, so a fake `run_watched` sees the harness command as it did before the wrapper existed. No test using these opens a chip or
reaches the real gozer (tests/conftest.py refuses hwtool._sh)."""
from __future__ import annotations

import json

from orchard import hwtool, kernelprobe as kp

PID = 4321
CHIPS = ["0000:01:00.0", "0000:02:00.0"]


def _r(code, out="", err=""):
    return code, out, err


class FakeGozer:
    """A gozer with state, as the wrapper and the status check use it: acquire, wait, cancel, release and status.

    `busy`: how many acquires answer "queued" before one is granted. `wait`: "grant" (the wait ends with a grant), "timeout" (it ends
    without one) or "gone". `cancel`: "ok", "fail" or "gone". `release`: "ok", "reset_failed" (exit 0 with the reset-failed text),
    "refused" (exit 15, the lease stays held) or "hold" (answers ok but the lease stays held, a gozer that lies). `status_fails`.
    `acquire_fail`. `no_env`: a grant with no device list."""

    def __init__(self, *, busy=0, wait="grant", cancel="ok", release="ok", status_fails=False, acquire_fail=False, no_env=False):
        self.busy, self.wait_mode, self.cancel_mode, self.release_mode = busy, wait, cancel, release
        self.status_fails, self.acquire_fail, self.no_env = status_fails, acquire_fail, no_env
        self.calls: list[list[str]] = []
        self.leases: dict[str, dict] = {}              # lease id -> {"who", "pid", "chips"} for every lease gozer still holds
        self.tickets: dict[str, dict] = {}             # ticket -> {"who", "pid"}
        self.counter = 0
        self.history: list[tuple[str, str]] = []       # ("granted" | "released" | "cancelled", id)

    def sh(self, argv, timeout):
        argv = list(argv)
        self.calls.append(argv)
        return getattr(self, f"_{argv[1]}")(argv)

    def subs(self):
        return [c[1] for c in self.calls]

    # ---- the commands -----------------------------------------------------------------------------------------------
    def _grant(self, who, pid):
        self.counter += 1
        lease = f"L{self.counter}"
        self.leases[lease] = {"who": who, "pid": pid, "chips": list(CHIPS)}
        self.history.append(("granted", lease))
        env = {} if self.no_env else {"TT_VISIBLE_DEVICES": ",".join(CHIPS)}
        return {"granted": True, "lease_id": lease, "chips": [] if self.no_env else list(CHIPS), "env": env}

    def _acquire(self, argv):
        if self.acquire_fail:
            return _r(1, "", "boom")
        who = argv[argv.index("--who") + 1]
        pid = int(argv[argv.index("--owner-pid") + 1])
        if self.busy > 0:
            self.busy -= 1
            ticket = f"T{len(self.tickets) + self.counter + 1}"
            self.tickets[ticket] = {"who": who, "pid": pid}
            return _r(10, json.dumps({"granted": False, "queued": True, "ticket": ticket}))
        return _r(0, json.dumps({**self._grant(who, pid), "owner_pid": pid}))

    def _wait(self, argv):
        ticket = argv[-1]
        entry = self.tickets.get(ticket)
        if self.wait_mode == "grant" and entry is not None:
            del self.tickets[ticket]
            return _r(0, json.dumps(self._grant(entry["who"], None)))      # `gozer wait` grants a detached lease (no pid)
        if self.wait_mode == "gone":
            return _r(13, json.dumps({"granted": False, "reason": "gone"}))
        return _r(11, json.dumps({"granted": False, "ticket": ticket}))

    def _cancel(self, argv):
        if self.cancel_mode == "fail":
            return _r(1, "", "boom")
        if self.cancel_mode == "gone" or argv[2] not in self.tickets:
            return _r(13, "no such ticket")
        del self.tickets[argv[2]]
        self.history.append(("cancelled", argv[2]))
        return _r(0, "cancelled")

    def _release(self, argv):
        lease = argv[2]
        if self.release_mode == "refused":
            return _r(15, json.dumps({"released": False, "message": "device still open"}))
        if lease not in self.leases:
            return _r(13, json.dumps({"released": False, "message": "not found"}))
        if self.release_mode != "hold":
            del self.leases[lease]
            self.history.append(("released", lease))
        message = "released; NOT marked clean" if self.release_mode == "reset_failed" else "released"
        return _r(0, json.dumps({"released": True, "message": message}))

    def _status(self, argv):
        if self.status_fails:
            return _r(1, "", "boom")
        held = {bdf: v for v in self.leases.values() for bdf in v["chips"]}
        chips = [{"bdf": bdf, "pid": held[bdf]["pid"], "who": held[bdf]["who"], "state": "CLAIMED"} if bdf in held
                 else {"bdf": bdf, "pid": None, "who": None, "state": "FREE"} for bdf in CHIPS]
        queue = [{"ticket": t, "who": v["who"], "pid": v["pid"]} for t, v in self.tickets.items()]
        return _r(0, json.dumps({"chips": chips, "queue": queue}))

    # ---- reading back -----------------------------------------------------------------------------------------------
    def released(self):
        return [c[2] for c in self.calls if c[1] == "release"]

    def cancelled(self):
        return [c[2] for c in self.calls if c[1] == "cancel"]

    def acquires(self):
        return [c for c in self.calls if c[1] == "acquire"]

    def clean(self):
        """Ground truth: gozer holds nothing of ours."""
        return not self.leases and not self.tickets


def wrapped(inner, gozer: FakeGozer, now, *, record=None):
    """A fake `run_watched` that runs the real wrapper when it is given one, and `inner` for every other command (the smoke test, the
    probe's own hardware phase). `inner(command, **kw)` plays the harness: it is what ran as `gozer run` before the wrapper. The
    wrapper's own record is written where the probe reads it. If `inner` raises, the wrapper is stopped (it releases its lease, as the
    real one does on a group kill) and the exception goes on to the probe. If `inner` returns a killed run (the stall watcher), the
    wrapper is stopped the same way."""
    def run_watched(argv, **kw):
        argv = list(argv)
        if len(argv) < 3 or not str(argv[1]).endswith("hwtool.py"):
            return inner(argv, **kw)
        cut = argv.index("--")
        command = argv[cut + 1:]
        state = {"w": None, "exc": None}

        class Proc:
            def wait(self, timeout=None):
                if state["w"] is not None or state["exc"] is not None:       # the command has run: a later wait (after a stop) is instant
                    return -15 if state["exc"] is not None or state["w"].killed else state["w"].returncode
                try:
                    state["w"] = inner(command, **kw)
                except BaseException as exc:
                    state["exc"] = exc
                    raise KeyboardInterrupt
                if state["w"].killed:
                    raise hwtool._Stop()
                return state["w"].returncode

            def terminate(self):
                pass

            def kill(self):
                pass

        def out(line):
            with open(kw["log_path"], "a") as f:
                f.write(line + "\n")
        status = hwtool.wrapper_cli(argv[2:cut + 1] + command, install_signals=False, sh=gozer.sh, spawn=lambda cmd, env=None: Proc(),
                                    now=now, out=out, pid=PID)
        if record is not None:
            record.append(list(argv))
        if state["exc"] is not None:
            raise state["exc"]
        w = state["w"]
        if w is None:                                  # the wrapper ended without running the command (busy, no lease, changed)
            return kp.Watched(status, None, False, 0.5)
        if w.killed:
            return kp.Watched(status, w.reason, True, w.elapsed_s)
        return kp.Watched(status, None, False, w.elapsed_s)
    return run_watched
