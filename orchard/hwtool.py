# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `hardware_check` tool of a kernel-probe attempt: the policy around one chip run.

The agent's shell cannot run gozer (the command runner refuses it and the shell has a no-chip device mask), so the probe
gives the agent a tool. This module holds what makes the tool safe to leave on. It does not run anything itself: the probe
passes two functions, `prepare` and `run`, which do the work with the probe's own lease, copy of the kernel, static gate
and stall watcher (orchard/kernelprobe.py, `make_hardware_check`).

Rules, each tested in tests/test_agent_hw.py:
- The tool takes at most a short `note`. Any other argument is an error and nothing runs.
- At most `max_calls` calls per attempt (a call that waits for a busy board counts, so the agent cannot poll for ever;
  a call refused before it reached the lease does not) and at most `lease_s` seconds of chip time in all. `cleanup_s`
  (gozer's term grace, which the stop and the chip reset can use in full) is reserved: a run is stopped at
  `call_timeout_s` or at the seconds left minus `cleanup_s`, whichever is less, and a call needs `min_call_s` seconds of
  run time left after that reserve. So the worst call (the run until its deadline, then the cleanup) fits the cap.
- One call at a time. A second call while one runs is answered at once.
- A busy board is waited for on one queue ticket, for at most `wait_s` seconds: the lease is requested with `gozer acquire
  --owner-pid`, and when gozer queues the request the probe claims that same ticket every few seconds, so the call keeps its
  place in the queue and never asks again in a loop. When the wait runs out the ticket is cancelled and the agent is told the board
  is busy and to go on with the simulator (orchard/hwlease.py).
- What a call leaves behind is read from gozer, not remembered (`health`): while a lease or ticket of this probe is still shown, no
  new call starts. The chip time of a call whose release was not seen is replaced by gozer's own duration for the lease (`settle`).
- After a hang, hardware checks are off for the rest of the attempt.
- Every call is recorded, and the probe writes the record into attempt.json and the report.

The choices (6 calls, 600 lease seconds, 240 s per call, 120 s of waiting, 210 s of cleanup reserve, 60 s minimum) are not
measured against a population of attempts. A whole call took about 47 s when measured (the device was open for about
0.5 s; the rest is the lease and the reset that its release makes) (docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

NOTE_MAX = 200                      # characters of the agent's note that are kept
TAIL_LINES = 15                     # lines of the run's output shown after a failure that is not a numeric one
TAIL_CHARS = 1200
VIOLATIONS_SHOWN = 5


@dataclass(frozen=True)
class HwPolicy:
    """The limits of one attempt's hardware checks. `max_calls` 0 turns the tool off."""
    max_calls: int = 6
    lease_s: float = 600.0          # chip seconds in all: the seconds the probe's gozer runs took, reset included
    call_timeout_s: float = 240.0   # a run's backstop; the stall watcher usually stops a hang sooner
    wait_s: float = 120.0           # how long one call waits for a busy board
    min_call_s: float = 60.0        # seconds of run time that must be left, after the cleanup reserve, for a call to start
    cleanup_s: float = 210.0        # reserved for the stop (gozer's term grace, 120 s) and the separate release and reset (90 s)

    @property
    def enabled(self) -> bool:
        return self.max_calls > 0

    def as_dict(self) -> dict:
        return {"max_calls": self.max_calls, "lease_s": self.lease_s, "call_timeout_s": self.call_timeout_s,
                "wait_s": self.wait_s, "min_call_s": self.min_call_s,
                "cleanup_s": self.cleanup_s}


def schema(policy: HwPolicy) -> dict:
    """The tool as the model sees it. The description states the cost and the limit of this attempt's policy."""
    return {"type": "function", "function": {
        "name": "hardware_check",
        "description": ("Run your current kernel.py on a Tenstorrent chip and get the result of the probe's hardware check. "
                        "A whole call takes about 47 s (the kernel runs in under a second; the rest is the lease and the chip reset), and each attempt has "
                        f"a limited number of calls ({policy.max_calls} in this attempt; every answer says how many are "
                        "left). The simulator check after each write is much cheaper: use this for a kernel the "
                        "simulator cannot run (a FAIL: sim_gap), for a numeric check at hardware precision after the "
                        "simulator passes, or for a question about the hardware itself. If every board is busy it waits "
                        f"up to {policy.wait_s:.0f} s and then says so. The probe judges the copy of kernel.py it takes "
                        "now; nothing you say changes the verdict."),
        "parameters": {"type": "object",
                       "properties": {"note": {"type": "string", "description":
                                               f"optional, at most {NOTE_MAX} characters, kept in the record"}},
                       "required": [], "additionalProperties": False}}}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


class HardwareCheck:
    """One attempt's hardware tool.

    prepare(k) -> (blocked, sha256): `blocked` is None when kernel.py was copied and passed the static gate, else the hw
    block that says why it cannot run (no chip is used).
    run(k, timeout_s) -> (hw, event, text): one gozer run of the prepared copy. `event` is None, "queued" (the board was
    busy; the probe's call cancelled the ticket and `hw["queued"]` says which) or "hang" (the stall watcher or the
    backstop stopped it). `hw` carries `elapsed_s`, the whole gozer run including the lease and the reset, and may carry
    `device_s`, `numbers` and the judging fields; `text` is the run's output."""

    def __init__(self, policy: HwPolicy, *, prepare, run, clock=time.monotonic, sleep=time.sleep,
                 stamp=_utc_stamp, on_record=None, health=None, settle=None):
        self.policy, self.prepare, self.run = policy, prepare, run
        self.settle = settle                # (lease_id) -> the seconds gozer says the lease was held, or None
        self._pending = None                # the last answer of `health`, kept for summary()
        self.health = health                # () -> None, or a sentence about a lease or ticket an earlier call left
        self.on_record = on_record          # called with summary() each time a call is recorded; its errors are ignored
        self.clock, self.sleep, self.stamp = clock, sleep, stamp
        self._lock = threading.Lock()
        self.records: list[dict] = []
        self.counted = 0
        self.lease_used = 0.0
        self.bad_calls = 0
        self.hung = False

    def schema(self) -> dict:
        return schema(self.policy)

    # ---- what the probe writes into attempt.json ----------------------------------------------------------------

    def summary(self) -> dict:
        out = {"enabled": self.policy.enabled, "policy": self.policy.as_dict(), "calls": [dict(r) for r in self.records],
               "calls_counted": self.counted, "bad_calls": self.bad_calls, "lease_s_used": round(self.lease_used, 1)}
        if self._pending:
            out["unresolved"] = self._pending
        return out

    # ---- the call -----------------------------------------------------------------------------------------------

    def call(self, arguments) -> str:
        note = arguments.get("note", "") if isinstance(arguments, dict) else None
        if not isinstance(arguments, dict) or set(arguments) - {"note"} or not isinstance(note, str):
            self.bad_calls += 1
            return ("error: hardware_check takes no command and no path, only an optional `note` (a short string). "
                    "Nothing was run.")
        if not self._lock.acquire(blocking=False):
            return "error: a hardware check is already running; wait for its answer. Nothing was started."
        try:
            return self._call(note[:NOTE_MAX])
        finally:
            before = self._pending
            settled = self._settle()
            self._refresh()                 # what this call left behind, for the record and the next call
            if (settled or self._pending != before) and self.on_record is not None:
                try:
                    self.on_record(self.summary())
                except Exception:
                    pass
            self._lock.release()

    def _settle(self) -> bool:
        """Replace the estimated chip time of a call whose lease was not seen released (an interrupt, an overrunning
        release) by the duration gozer records for the lease, as soon as gozer has one. True when a record changed."""
        changed = False
        if self.settle is None:
            return False
        for rec in self.records:
            if rec.get("lease_incomplete") and rec.get("lease_id"):
                try:
                    seconds = self.settle(rec["lease_id"])
                except Exception:
                    seconds = None
                if _number(seconds):
                    self.lease_used += max(seconds - rec["lease_s"], 0.0)
                    rec["lease_s"] = round(float(seconds), 1)
                    rec["lease_incomplete"], rec["lease_reconciled"] = False, True
                    rec.pop("lease_upper_bound", None)
                    changed = True
        return changed

    def _refresh(self):
        try:
            self._pending = self.health() if self.health is not None else None
        except Exception:
            self._pending = "the lease check failed"
        return self._pending

    def _left(self) -> str:
        return (f"budget: {max(self.policy.max_calls - self.counted, 0)} calls left, "
                f"{max(self.policy.lease_s - self.lease_used, 0):.0f} lease seconds left")

    def _record(self, **fields) -> dict:
        rec = {"at": self.stamp(), "k": None, "counted": False, "outcome": None, "category": None, "passed": False,
               "total_s": 0.0, "device_s": None, "lease_s": 0.0, "waited_s": 0.0, "queued_polls": 0,
               "cancelled": None, "timed_out": False, "note": "", "first_line": "", "pcc_min": None,
               "kernel_sha256": None, "lease_id": None, "lease_incomplete": False}
        rec.update(fields)
        self.records.append(rec)
        if self.on_record is not None:
            try:
                self.on_record(self.summary())
            except Exception:
                pass                                # keeping the record on disk must never end the call
        return rec

    def _refuse(self, note: str, why: str) -> str:
        self._record(outcome="refused", note=note, first_line=why)
        return f"hardware_check refused: {why}"

    def _call(self, note: str) -> str:
        p = self.policy
        self._settle()                              # an earlier call's overlong release may have finished since
        if self.hung:
            return self._refuse(note, "an earlier hardware run of this attempt hung, so hardware checks are off for the "
                                      "rest of it. Use the simulator (the check after each write).")
        if self.counted >= p.max_calls:
            return self._refuse(note, f"no hardware checks left: {self.counted} of {p.max_calls} used in this attempt. "
                                      "Use the simulator (the check after each write).")
        left = p.lease_s - self.lease_used - p.cleanup_s        # run time that is left once the release is reserved
        if left < p.min_call_s:
            return self._refuse(note, f"only {max(p.lease_s - self.lease_used, 0):.0f} lease seconds are left of "
                                      f"{p.lease_s:.0f}; {p.cleanup_s:.0f} are reserved for the cleanup (the stop and the "
                                      f"chip reset) and a call needs at least {p.min_call_s:.0f} to run. Use the "
                                      "simulator (the check after each write).")
        pending = self._refresh()
        if pending:
            return self._refuse(note, f"an earlier call left something unresolved ({pending}). No new hardware call starts "
                                      "until gozer shows it is gone. Use the simulator (the check after each write).")
        blocked, sha = self.prepare(self.counted + 1)
        if blocked is not None:
            return self._blocked(note, blocked, sha)
        self.counted += 1
        k = self.counted
        start = self.clock()
        phase = {"wait_began": start, "run_began": None}   # the call waits for a lease from its start; a grant starts the run
        ctx: dict = {}                              # the run closure puts the lease id here the moment it is granted
        finished = False

        def on_grant():
            phase["wait_began"], phase["run_began"] = None, self.clock()
        try:
            timeout = min(p.call_timeout_s, p.lease_s - self.lease_used - p.cleanup_s)
            hw, event, text = self.run(k, timeout, max(p.wait_s, 0.0), on_grant, ctx)
            finished = True
        finally:
            if not finished:                        # an interrupt or an error: keep what is known
                now = self.clock()
                # A run that was in progress may have held the board until now: count that time, as an upper bound
                # (the run may have stopped earlier, and the lease release may still be going on; the duration gozer
                # reports for the lease replaces it when it is known, see _settle). Time spent waiting for the lease is
                # the wait.
                run_began = phase["run_began"]
                held = max(now - run_began, 0.0) if run_began is not None else 0.0
                waited_so_far = max((run_began if run_began is not None else now) - start, 0.0)
                self.lease_used += held
                self._record(k=k, counted=True, outcome="interrupted", note=note, kernel_sha256=sha,
                             total_s=round(now - start, 1), queued_polls=1 if ctx.get("queued") else 0,
                             lease_s=round(held, 1), waited_s=round(waited_so_far, 1), lease_id=ctx.get("lease_id"),
                             lease_incomplete=bool(ctx.get("lease_id")),
                             lease_upper_bound=True if run_began is not None else None)
        total = round(self.clock() - start, 1)
        polls = 1 if (ctx.get("queued") or event == "queued") else 0
        if event == "queued":
            q = hw.get("queued") if isinstance(hw.get("queued"), dict) else {}
            cancelled = bool(q.get("ticket")) and q.get("cancel_error") is None
            self._record(k=k, counted=True, outcome="busy", category="queued", note=note, kernel_sha256=sha,
                         total_s=total, waited_s=total, queued_polls=polls, cancelled=cancelled,
                         first_line=str(hw.get("first_line") or ""))
            return self._busy_text(k, total, polls, cancelled, hw)
        cancelled = None
        waited = float(hw["waited_s"]) if _number(hw.get("waited_s")) else 0.0
        elapsed = float(hw["elapsed_s"]) if _number(hw.get("elapsed_s")) else total - waited
        self.lease_used += elapsed
        device = hw.get("device_s") if _number(hw.get("device_s")) else None
        release_error = str(hw["release_error"]) if hw.get("release_error") else None
        lease_id = hw.get("lease_id") if isinstance(hw.get("lease_id"), str) else None
        incomplete = bool(lease_id and hw.get("lease_incomplete"))
        if event == "hang":
            self.hung = True
            self._record(k=k, counted=True, outcome="hang", category="hang", note=note, kernel_sha256=sha, total_s=total,
                         lease_s=round(elapsed, 1), waited_s=round(waited, 1), queued_polls=polls,
                         cancelled=cancelled, timed_out=True, release_error=release_error, lease_id=lease_id,
                         lease_incomplete=incomplete, first_line=str(hw.get("first_line") or ""))
            return (f"hardware_check {k} of {p.max_calls}: HANG: the run was stopped ({hw.get('first_line')}) after "
                    f"{elapsed:.1f} s. The probe then released the lease with `gozer release`, which resets the chip. Look with `gozer status` if in doubt (a lease marked stale is cleared "
                    "with `gozer reconcile`, the gozer-gatekeeper skill says how), and never reset a chip by hand. "
                    "Hardware checks are off for the rest of this attempt; the simulator still works."
                    f"{self._release_note(release_error)}\n{self._left()}")
        passed = hw.get("passed") is True
        category = None if passed else hw.get("category")
        self._record(k=k, counted=True, outcome="pass" if passed else str(category), category=category, passed=passed,
                     note=note, kernel_sha256=sha, total_s=total, device_s=device, lease_s=round(elapsed, 1),
                     waited_s=round(waited, 1), queued_polls=polls, cancelled=cancelled, lease_id=lease_id,
                     lease_incomplete=incomplete, release_error=release_error, first_line=str(hw.get("first_line") or ""),
                     pcc_min=hw.get("pcc_min") if _number(hw.get("pcc_min")) else None)
        return self._result_text(k, hw, text, total, waited, device, passed) + (
            "\n" + self._release_note(release_error).strip() if release_error else "")

    # ---- the answers --------------------------------------------------------------------------------------------

    @staticmethod
    def _release_note(error) -> str:
        if not error:
            return ""
        return (f" gozer could not release the lease ({error}); the lease may still be held: look with `gozer status`, "
                "and clear a stale one with `gozer reconcile`.")

    def _blocked(self, note: str, blocked: dict, sha) -> str:
        category = str(blocked.get("category"))
        self._record(outcome=category, category=category, note=note, kernel_sha256=sha,
                     first_line=str(blocked.get("first_line") or ""))
        lines = [f"hardware_check: FAIL: {category}: {blocked.get('first_line')}"]
        for v in [v for v in blocked.get("violations") or [] if isinstance(v, dict)][:VIOLATIONS_SHOWN]:
            lines.append(f"  {v.get('rule')} at line {v.get('line')}: {v.get('detail')}. {v.get('remedy')}")
        lines.append("The probe did not run this kernel, so no chip was used and this call was not counted.")
        lines.append(self._left())
        return "\n".join(lines)

    def _busy_text(self, k, total, polls, cancelled, hw) -> str:
        q = hw.get("queued") if isinstance(hw.get("queued"), dict) else {}
        tail = ("The queue tickets were cancelled." if cancelled else
                f"A ticket could not be cancelled ({q.get('cancel_error') or 'no ticket in gozer output'}; ticket "
                f"{q.get('ticket')}): look with `gozer status`.")
        return (f"hardware_check {k} of {self.policy.max_calls}: BUSY: the board was busy for {total:.0f} s "
                f"(waited on {polls} queue ticket). {tail} The chip was not used. Go on with the simulator (the check after each "
                f"write).\n{self._left()}")

    def _result_text(self, k, hw, text, total, waited, device, passed) -> str:
        p = self.policy
        if passed:
            head = (f"hardware_check {k} of {p.max_calls}: PASS (the probe judged this copy of kernel.py on the chip "
                    "with the hardware bars)")
        else:
            head = f"hardware_check {k} of {p.max_calls}: FAIL: {hw.get('category')}: {hw.get('first_line')}"
        lines = [head]
        sets = [s for s in hw.get("sets") or [] if isinstance(s, dict)]
        measures = []
        if _number(hw.get("pcc_min")):
            measures.append(f"pcc_min {hw['pcc_min']:.6g}")
        rng = hw.get("slope_range")
        if isinstance(rng, list) and len(rng) == 2 and all(_number(x) for x in rng):
            measures.append(f"slope {rng[0]:.4g} to {rng[1]:.4g}")
        if _number(hw.get("rel_l2_max")):
            measures.append(f"rel_l2_max {hw['rel_l2_max']:.3g}")
        if sets:
            measures.append(f"{len(sets)} input sets")
        if measures:
            lines.append(", ".join(measures))
        if not passed and hw.get("numbers"):
            lines.append(f"numbers: {hw['numbers']}")
        if not passed and hw.get("category") != "hw_numeric_mismatch" and text.strip():
            tail = "\n".join([l for l in text.splitlines() if l.strip()][-TAIL_LINES:])[-TAIL_CHARS:]
            lines.append("output (last lines):\n" + tail)
        dev = f"device {device:.1f} s" if device is not None else "device time not reported"
        wait = f", waited {waited:.1f} s for a lease" if waited >= 0.05 else ""
        lines.append(f"time: {total:.1f} s in all ({dev}; the rest is the lease and the chip reset){wait}")
        lines.append(self._left())
        return "\n".join(lines)
