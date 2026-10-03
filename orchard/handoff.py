"""Park and restore: swap the coder off a board and back (spec section 6).

This module owns the decision whether a stage needs a park, the park and restore sequence, and
recovery after a supervisor restart. The sequence is a function-driven state machine. Each step
writes a `park` or `restore` ledger entry when it completes, so `progress()` can replay the
ledger and say which step completed. A resumed `Handoff` skips the steps already recorded.
`recover()` compares the ledger with the machine after a restart, and the machine wins.

Park steps: note (the handoff note exists and parses), canary_before (the coder answers the
canary), standin_started (the CPU stand-in process exists; its pid is recorded before anything
else can fail), standin (the stand-in answers the canary), stop_sent (`tt-model stop`), stopped
(the stop is confirmed by the server's tooling and by the lease tool), reset (the lease is reset
in place and stays held). The stage test runs next; plan 4 owns it.
Restore steps: reset (the test is gone; the chips are reset again), serve, ready, canary (the
answer must equal the pre-park answer exactly), resumed (the stand-in stops and is checked gone).
A `restart` entry makes the restore start again from its reset.

A park that cannot go on before the coder was told to stop ends with an `abandoned` park entry:
the coder never left, the stand-in is stopped, and the run is not parked.

`orchard.ledger.replay_state` is the authority on whether the run is parked. `progress()` derives
the step detail from the same entries and must agree with it: parked exactly when the phase is
not "idle" (tests/test_handoff.py checks this after every entry).

Any condition the spec says needs a person raises `Blocked` after a ledger `notice`. Nothing is
retried blindly. The retry rule (spec section 3) allows one retry of a reset that gozer refused
because a device was busy (exit 15). A reset that ran and failed (exit 17) blocks at once
(spec section 10). Restarting a model server that died is plan 4's job (spec section 10).
"""
from __future__ import annotations

from dataclasses import dataclass

from orchard.adapters import ChipState, Lease, boards_of

import dataclasses
import json
import time
from pathlib import Path

from orchard.adapters import (AdapterError, LeaseLost, Queued, Refused, ResetFailed,
                              TicketGone)
from orchard.canary import CanaryError, CanaryResult, compare
from orchard.defaults import (COLD_BOOT_BUDGET_S, IDLE_RELEASE_S, MESH_RESET_EXTRA_S, POLL_S,
                              QUEUE_POLL_S, QUIET_WAIT_S)
from orchard.ledger import file_evidence
from orchard.server import NotReady, ServerError, ServerStarting

PARK_STEPS = ("note", "canary_before", "standin_started", "standin", "stop_sent", "stopped",
              "reset")
ABANDONED = "abandoned"          # closes a park before the coder was told to stop
RESTORE_STEPS = ("reset", "serve", "ready", "canary", "resumed")
NOTE_KEYS = ("goal", "stage", "evidence", "next_action", "check_on_return")


class Blocked(Exception):
    """The stage is blocked and the run pauses for the operator. The ledger has a notice."""

    def __init__(self, reason: str, **evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


@dataclass(frozen=True)
class ParkDecision:
    action: str                       # "use_free", "park" or "wait"
    free_boards: tuple[str, ...]
    server_boards: tuple[str, ...]
    boards_needed: int

    @property
    def park_needed(self) -> bool:
        return self.action == "park"


def decide_park(server_boards, chips: list[ChipState], boards_needed: int) -> ParkDecision:
    """Does a stage that needs `boards_needed` boards have to park the coder? (spec section 6)

    `server_boards` are the board serials the coder's lease holds: both boards for the large tier,
    one for the small tier. A board is free when every chip on it is FREE with no lease. "use_free"
    means: lease a free board (reacquire with exact=its first chip) and leave the coder loaded.
    "park" means the coder's boards are needed; take any further boards with a lease of their own.
    "wait" means parking cannot supply enough boards; wait in the queue.
    """
    boards = boards_of(chips)
    server = set(server_boards)
    unknown = server - set(boards)
    if unknown:
        raise ValueError(f"server boards {sorted(unknown)} are not in the chip list")
    if not 1 <= boards_needed <= len(boards):
        raise ValueError(f"boards_needed must be 1 to {len(boards)}, got {boards_needed}")
    free = tuple(b for b, cs in boards.items()
                 if b not in server and all(c.state == "FREE" and not c.who for c in cs))
    if len(free) >= boards_needed:
        action = "use_free"
    elif server and len(free) + len(server) >= boards_needed:
        action = "park"
    else:
        action = "wait"
    return ParkDecision(action, free, tuple(sorted(server)), boards_needed)


def chips_quiet(chips: list[ChipState], lease: Lease, accept=("CLAIMED",)) -> tuple[bool, dict[str, str]]:
    """Is every chip of the lease in an accepted state? CLAIMED means leased and no device open."""
    states = {c.bdf: c.state for c in chips if c.bdf in lease.chips}
    quiet = set(states) == set(lease.chips) and all(s in accept for s in states.values())
    return quiet, states


@dataclass(frozen=True)
class Progress:
    phase: str                                  # "idle", "parking", "parked" or "restoring"
    stage: int | None = None
    park_done: tuple[str, ...] = ()
    restore_done: tuple[str, ...] = ()
    lease: dict | None = None                   # the latest lease record
    owner_pid: int | None = None                # the supervisor pid that wrote the latest step
    server: dict | None = None                  # the coder's ServerControl record
    standin: dict | None = None
    canary_before: dict | None = None           # file_evidence of the pre-park answer
    mesh_reset: bool = False                    # `tt-model stop` reset the mesh itself


def progress(entries: list[dict]) -> Progress:
    """Replay the ledger: where is the current handoff? The ledger is the only state.

    Entries without a step (the plan 1 form) count as a park that starts and a restore that ends,
    as `replay_state` reads them, so the two always agree.
    """
    cur = None
    restoring = False
    for e in entries:
        ev = e["event"]
        if ev not in ("park", "restore"):
            continue
        d = e["data"]
        step = d.get("step")
        if ev == "park" and step in ("note", None):
            cur = {"stage": e["stage"], "park_done": [], "restore_done": [], "lease": None,
                   "owner_pid": None, "server": None, "standin": None, "canary_before": None,
                   "mesh_reset": False}
            restoring = False
        if cur is None:
            continue
        for k in ("lease", "owner_pid", "server", "standin"):
            if d.get(k) is not None:
                cur[k] = d[k]
        if ev == "park":
            if step == ABANDONED:
                cur = None
                continue
            if step not in cur["park_done"]:
                cur["park_done"].append(step)
            if step == "canary_before":
                cur["canary_before"] = d.get("canary")
            if step == "stop_sent" and isinstance(d.get("result"), dict):
                cur["mesh_reset"] = bool(d["result"].get("mesh_reset"))
            continue
        restoring = True
        if step in ("resumed", None):
            cur = None
        elif step == "restart":
            cur["restore_done"] = []
        elif step not in cur["restore_done"]:
            cur["restore_done"].append(step)
    if cur is None:
        return Progress("idle")
    if restoring:
        phase = "restoring"
    elif "reset" in cur["park_done"]:
        phase = "parked"
    else:
        phase = "parking"
    return Progress(phase, cur["stage"], tuple(cur["park_done"]), tuple(cur["restore_done"]),
                    cur["lease"], cur["owner_pid"], cur["server"], cur["standin"],
                    cur["canary_before"], cur["mesh_reset"])


@dataclass(frozen=True)
class Budgets:
    quiet_wait_s: float = QUIET_WAIT_S              # how long to wait for chips to go quiet
    poll_s: float = POLL_S
    cold_boot_s: float = COLD_BOOT_BUDGET_S         # the run's cold-boot budget (spec section 6)
    mesh_reset_extra_s: float = MESH_RESET_EXTRA_S  # added when tt-model reset the mesh itself


def wait_stopped(server, adapter, lease: Lease, *, quiet_wait_s: float, poll_s: float, clock,
                 sleep, accept=("CLAIMED",)) -> dict:
    """Poll until the server's own checks and the lease tool both show nothing running.

    Both must agree. docker can show a container gone while a vLLM worker it started still holds
    the device; the lease tool sees that holder. The lease tool can show a chip CLAIMED while
    another user's container still holds it; docker sees that one. During another board's reset
    our chips can look busy for about 42 s, so the wait lasts longer than that.
    """
    deadline = clock() + quiet_wait_s
    while True:
        check = server.confirm_stopped()
        try:
            quiet, states = chips_quiet(adapter.status(), lease, accept)
        except AdapterError as exc:
            quiet, states = False, {"error": str(exc)}
        if check.stopped and quiet:
            others = (check.evidence.get("docker_inspect") or {}).get("others_with_all_devices") or []
            return {"ok": True, "checks": check.checks, "chip_states": states,
                    "others_with_all_devices": others}
        if clock() >= deadline:
            return {"ok": False, "checks": check.checks, "chip_states": states,
                    "server_evidence": check.evidence}
        sleep(poll_s)


class Handoff:
    """One park and restore of the coder, step by step, each step recorded in the ledger."""

    def __init__(self, *, ledger, stage, adapter, server, standin, lease: Lease, canary_prompt: str,
                 note_path, evidence_dir, budgets: Budgets = Budgets(), clock=time.monotonic,
                 sleep=time.sleep):
        self.ledger, self.stage = ledger, stage
        self.adapter, self.server, self.standin = adapter, server, standin
        self.lease, self.canary_prompt = lease, canary_prompt
        self.note_path = Path(note_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.budgets, self.clock, self.sleep = budgets, clock, sleep

    # ---- ledger and evidence helpers ----------------------------------------------------------

    def _record(self, event: str, step: str, **data) -> None:
        self.ledger.append(event, self.stage, step=step, lease=self.lease.record(),
                           owner_pid=self.adapter.owner_pid, **data)

    def _block(self, reason: str, **evidence):
        self.ledger.append("notice", self.stage, blocked=True, reason=reason, evidence=evidence)
        raise Blocked(reason, **evidence)

    def _block_before_stop(self, reason: str, **evidence):
        """Block a park that never told the coder to stop: close it, so the run is not parked."""
        self.ledger.append("notice", self.stage, blocked=True, reason=reason, evidence=evidence)
        stopped = None
        if "standin_started" in progress(self.ledger.read()).park_done:
            stopped = self._stop_standin()
        self._record("park", ABANDONED, reason=reason, standin_stopped=stopped)
        raise Blocked(reason, **evidence)

    def _evidence_path(self, stem: str) -> Path:
        """A new file for every attempt, named by the next ledger sequence number.

        A ledger entry records each file's sha256, so a file is never written twice.
        """
        seq = len(self.ledger.read()) + 1
        n = 0
        while True:
            path = self.evidence_dir / (f"{stem}-{seq:05d}.txt" if n == 0 else
                                        f"{stem}-{seq:05d}-{n}.txt")
            if not path.exists():
                return path
            n += 1

    def _write_evidence(self, stem: str, text: str) -> Path:
        path = self._evidence_path(stem)
        with open(path, "x", encoding="utf-8") as f:       # "x": never overwrite
            f.write(text)
        return path

    def _stop_standin(self) -> bool:
        """Stop the stand-in and check it is gone. It is a process in its own session, so it
        outlives a supervisor crash; a stand-in left running holds host memory and its port."""
        try:
            self.standin.stop()
            check = self.standin.confirm_stopped()
        except Exception as exc:          # cleanup: record it and carry on
            self.ledger.append("notice", self.stage, what="stopping the stand-in failed",
                               error=str(exc))
            return False
        if not check.stopped:
            self.ledger.append("notice", self.stage, what="the stand-in is still running after "
                               "its stop", checks=check.checks, standin=self.standin.record())
        return check.stopped

    # ---- shared checks ------------------------------------------------------------------------

    def _wait_stopped(self, accept=("CLAIMED",), extra_s: float = 0.0) -> dict:
        check = wait_stopped(self.server, self.adapter, self.lease,
                             quiet_wait_s=self.budgets.quiet_wait_s + extra_s,
                             poll_s=self.budgets.poll_s, clock=self.clock, sleep=self.sleep,
                             accept=accept)
        if check["ok"] and check["others_with_all_devices"]:
            # Recorded for the operator; another agent's container does not block our stop.
            self.ledger.append("notice", self.stage,
                               what="other containers map the whole /dev/tenstorrent directory",
                               containers=check["others_with_all_devices"])
        return check

    def _reset(self, event: str) -> None:
        for attempt in (1, 2):
            t0 = self.clock()
            try:
                self.adapter.reset(self.lease)
            except LeaseLost as exc:
                self._block(f"the lease is gone or another tenant holds its chips; no server is "
                            f"started: {exc}")
            except Refused as exc:
                if exc.permanent:
                    self._block(f"the lease adapter cannot reset these chips: {exc}")
                # gozer's refusal ran nothing (spec section 8, item 1). The usual cause is a device
                # still open, so look again before the one retry the retry rule allows.
                if attempt == 2:
                    self._block(f"gozer refused the reset twice: {exc}")
                check = self._wait_stopped()
                if not check["ok"]:
                    self._block(f"the reset was refused and the chips are still in use: {exc}", **check)
                self.ledger.append("retry", self.stage, what="reset", attempt=2, reason=str(exc))
                continue
            except ResetFailed as exc:
                # The reset ran and failed, or is still running: the stage blocks (spec section 10).
                self._block(f"the reset failed: {exc}", left_running=exc.left_running)
            seconds = round(self.clock() - t0, 3)      # the successful call only
            self._record(event, "reset", seconds=seconds, attempts=attempt)
            self.ledger.append("measurement", self.stage, name=f"{event}_reset_seconds",
                               value=seconds, unit="s", label="measured")
            return

    # ---- park (spec section 6, steps 1 to 3) --------------------------------------------------

    def park(self) -> Lease:
        p = progress(self.ledger.read())
        if p.phase == "restoring":
            raise ValueError("a restore is in progress; call restore() or recover first")
        done = set(p.park_done) if p.phase in ("parking", "parked") else set()
        if "note" not in done:
            self._park_note()
        if "canary_before" not in done:
            self._park_canary()
        # Stand-in first (spec section 6, step 2): the coder stops only after the stand-in answered.
        if "standin" not in done:
            self._park_standin(started="standin_started" in done)
        mesh_reset = p.mesh_reset
        if "stop_sent" not in done:
            result = self.server.stop()
            mesh_reset = bool(result.get("mesh_reset"))
            self._record("park", "stop_sent", result=result)
        if "stopped" not in done:
            extra = self.budgets.mesh_reset_extra_s if mesh_reset else 0.0
            check = self._wait_stopped(extra_s=extra)
            if not check["ok"]:
                self._block("the coder is not confirmed stopped; no reset was run", **check)
            self._record("park", "stopped", **check)
        if "reset" not in done:
            self._reset("park")
        return self.lease

    def _park_note(self) -> None:
        path = self.note_path
        try:
            note = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            self._block(f"the handoff note {path} is missing or is not JSON: {exc}")
        if not isinstance(note, dict):
            self._block(f"the handoff note {path} is not a JSON object")
        missing = [k for k in NOTE_KEYS if note.get(k) in (None, "")]
        if missing:
            self._block(f"the handoff note {path} lacks {', '.join(missing)}", note=str(path))
        self._record("park", "note", note=file_evidence(path), server=self.server.record())

    def _park_canary(self) -> None:
        try:
            answer = self.server.ask(self.canary_prompt)
        except (CanaryError, OSError) as exc:
            self._block_before_stop(f"the coder did not answer the canary before the park: {exc}")
        if not answer.strip():
            self._block_before_stop("the coder gave an empty canary answer before the park")
        path = self._write_evidence("canary-before", answer)
        self._record("park", "canary_before", canary=file_evidence(path))

    def _park_standin(self, started: bool) -> None:
        if not started:
            try:
                self.standin.start()
            except (ServerError, OSError) as exc:
                self._record("park", "standin_started", standin=self.standin.record(), ok=False)
                self._block_before_stop(f"the stand-in did not start; the coder stays up: {exc}")
            # Recorded before the canary, so a crash from here on leaves the pid in the ledger.
            self._record("park", "standin_started", standin=self.standin.record())
        try:
            answer = self.standin.ask(self.canary_prompt)
        except (CanaryError, ServerError, OSError) as exc:
            self._block_before_stop(f"the stand-in did not answer; the coder stays up: {exc}")
        if not answer.strip():
            self._block_before_stop("the stand-in gave an empty answer; the coder stays up")
        path = self._write_evidence("standin-canary", answer)
        self._record("park", "standin", canary=file_evidence(path))

    # ---- restore (spec section 6, steps 4 to 6; the stage test itself is plan 4's) ------------

    def restore(self) -> CanaryResult | None:
        p = progress(self.ledger.read())
        if p.phase not in ("parked", "restoring"):
            raise ValueError(f"nothing to restore (phase {p.phase})")
        done = set(p.restore_done)
        if "reset" not in done:
            check = self._wait_stopped()
            if not check["ok"]:
                self._block("the chips are still in use after the stage; no reset was run", **check)
            self._reset("restore")
        if "serve" not in done:
            # Never start a second server while one may be up or coming up on the same port.
            if not self.server.confirm_stopped().stopped:
                self._block("a server is already up or coming up for the coder; not starting another")
            try:
                self.server.start(self.lease)
            except ServerStarting as exc:
                self._block(f"the coder start timed out while it was coming up; nothing else is "
                            f"started: {exc}", may_be_running=True)
            except ServerError as exc:
                self._block(f"starting the coder failed; nothing is retried: {exc}")
            self._record("restore", "serve", server=self.server.record())
        if "ready" not in done:
            try:
                seconds = self.server.wait_ready(self.budgets.cold_boot_s)
            except NotReady as exc:
                self._block("the coder did not return within the cold-boot budget; the run is "
                            "paused and nothing is retried", waited_s=exc.waited_s,
                            budget_s=self.budgets.cold_boot_s)
            except ServerError as exc:
                self._block(f"the coder exited before it was ready; nothing is retried: {exc}")
            self._record("restore", "ready", seconds=seconds)
            # The wait for readiness only: `tt-model serve` itself is not inside this number.
            self.ledger.append("measurement", self.stage, name="coder_ready_wait_seconds",
                               value=seconds, unit="s", label="measured")
        result = None
        if "canary" not in done:
            result = self._restore_canary(p.canary_before)
        if "resumed" not in done:
            stopped = self._stop_standin()
            # Plan 4 hands the note and a summary of the stage test to the coder.
            self._record("restore", "resumed", note=str(self.note_path), standin_stopped=stopped)
        return result

    def _restore_canary(self, before_ev: dict | None) -> CanaryResult | None:
        try:
            after = self.server.ask(self.canary_prompt)
        except (CanaryError, OSError) as exc:
            self._block(f"the coder did not answer the canary after the restore: {exc}")
        after_path = self._write_evidence("canary-after", after)
        if before_ev is None:
            # Only possible after a restart that interrupted the park before its canary.
            self.ledger.append("notice", self.stage,
                               what="canary not compared: no answer was recorded before the park")
            self._record("restore", "canary", compared=False, canary=file_evidence(after_path))
            return None
        try:
            before = Path(before_ev["path"]).read_text()
        except OSError as exc:
            self._block(f"the pre-park canary answer cannot be read: {exc}", before_file=before_ev)
        result = compare(before, after)
        if not result.match:
            self._block("the canary answer changed after the restore",
                        whitespace_only=result.whitespace_only, before=before[:500],
                        after=after[:500], before_file=before_ev,
                        after_file=file_evidence(after_path))
        self._record("restore", "canary", compared=True, match=True, canary=file_evidence(after_path))
        return result

    # ---- recovery after a restart (spec section 6, branch table; section 10) ----------------

    def recover_and_restore(self, recovery: Recovery, *, chips: int, who: str, reason: str,
                            wait_budget_s: float) -> CanaryResult | None:
        """Finish an interrupted handoff: bring the coder back under a lease this process holds.

        If the park never sent its stop and the coder is serving, the park is abandoned instead
        (see `recover`). Otherwise: a lease is judged by its owner pid, so after a restart the old lease belongs to a dead
        process. While the coder runs, its chips stay HELD-FOREIGN under that lease and nobody
        can reset or re-lease them. So: stop the coder if it runs, take a new lease (gozer reaps
        the orphan once no device is open, without a reset), and run the whole restore again from
        its reset. The stage test, if it was interrupted, is re-run by plan 4's stage machine.
        """
        if recovery.action == "none":
            return None
        self.ledger.append("decision", self.stage, decision="recover after restart; the machine wins",
                           **dataclasses.asdict(recovery))
        if recovery.action == "abandon":
            # The park never sent its stop and the coder is serving: close the park and leave the
            # coder up (spec section 6, step 2). Its lease belongs to the dead supervisor;
            # re-leasing a coder that serves outside a park is plan 4's job.
            stopped = self._stop_standin() if "standin_started" in recovery.park_done else None
            self._record("park", ABANDONED, recovered=True, standin_stopped=stopped)
            return None
        ours = recovery.lease_state == "ours"
        continuing = ours and recovery.coder_running and "serve" in recovery.restore_done
        if recovery.coder_running and not continuing:
            self._record("park", "stop_sent", result=self.server.stop(), recovered=True)
            check = self._wait_stopped(accept=("CLAIMED", "STALE", "FREE"))
            if not check["ok"]:
                self._block("after the restart the coder could not be confirmed stopped", **check)
            self._record("park", "stopped", recovered=True, **check)
        if not ours:
            old = self.lease.lease_id
            self.lease = reacquire(self.adapter, chips=chips, who=who, reason=reason,
                                   ledger=self.ledger, stage=self.stage, wait_budget_s=wait_budget_s,
                                   clock=self.clock, sleep=self.sleep)
            self.ledger.append("decision", self.stage, decision="new lease after restart",
                               old_lease_id=old, lease_id=self.lease.lease_id)
        if not continuing:
            self._record("restore", "restart", recovered=True)
        return self.restore()


def release_for_idle_phase(adapter, server, lease: Lease, *, ledger, stage,
                           expected_idle_s: float, budget_s: float = IDLE_RELEASE_S,
                           quiet_wait_s: float = QUIET_WAIT_S, poll_s: float = POLL_S,
                           clock=time.monotonic, sleep=time.sleep) -> bool:
    """Spec section 6, first branch row: release the lease for a long phase with no hardware use.

    A held board is unavailable to everyone else. Holding it costs nothing for a short phase;
    for a long one (an image build of 1.5 to 2.5 h, a CPU-only reference run) the supervisor
    releases it and takes a new lease, through `reacquire`, before the next hardware phase.
    gozer's release resets the chips (about 42 s), and gozer cannot see a container owned by
    another user, so the same two-sided stop check as a park runs first (`wait_stopped`). A
    ResetFailed from the release propagates to the caller.
    """
    info = {"lease_id": lease.lease_id, "expected_idle_s": expected_idle_s, "budget_s": budget_s}
    if expected_idle_s <= budget_s:
        ledger.append("decision", stage,
                      decision="hold the lease through a phase with no hardware use", **info)
        return False
    check = wait_stopped(server, adapter, lease, quiet_wait_s=quiet_wait_s, poll_s=poll_s,
                         clock=clock, sleep=sleep)
    if not check["ok"]:
        reason = "the server is not confirmed stopped; the lease was not released"
        ledger.append("notice", stage, blocked=True, reason=reason, evidence=check)
        raise Blocked(reason, **check)
    adapter.release(lease)
    ledger.append("decision", stage,
                  decision="released the lease for a phase with no hardware use", **info)
    return True


def reacquire(adapter, *, chips: int, who: str, reason: str, ledger, stage, wait_budget_s: float,
              clock=time.monotonic, sleep=time.sleep, poll_s: float = QUEUE_POLL_S,
              exact: str | None = None) -> Lease:
    """Take a lease, waiting in the lease tool's queue if the box is busy (spec section 10).

    The ticket is claimed by repeating the acquire with the ticket (the gozer-park skill); `gozer
    wait` is never used, because it grants a lease with no owner pid. The poll interval stays well
    inside gozer's 90 s claim window. gozer expires a ticket after one hour; the first expiry takes
    a new ticket and records the lost place, and a second one blocks. Waiting past the budget
    cancels the ticket and blocks. The ledger records the wait.
    """
    t0 = clock()

    def waited() -> float:
        return round(clock() - t0, 3)

    def block(why: str, **ev):
        ledger.append("notice", stage, blocked=True, reason=why, evidence=ev)
        raise Blocked(why, **ev)

    def granted(lease: Lease) -> Lease:
        ledger.append("decision", stage, decision="lease granted", lease_id=lease.lease_id,
                      chips=list(lease.chips), waited_s=waited())
        ledger.append("measurement", stage, name="queue_wait_seconds", value=waited(), unit="s",
                      label="measured")
        return lease

    def enqueue() -> tuple[Lease | None, str | None]:
        try:
            return granted(adapter.acquire(chips, who, reason, queue=True, exact=exact)), None
        except Queued as q:
            ledger.append("decision", stage, decision="queued for lease", ticket=q.ticket,
                          position=q.position, chips=chips)
            return None, q.ticket
        except Refused as exc:
            block(f"the lease tool refused the request: {exc}")

    lease, ticket = enqueue()
    if lease is not None:
        return lease
    replaced = False
    while True:
        if clock() - t0 >= wait_budget_s:
            try:
                adapter.cancel(ticket)
            except AdapterError as exc:
                ledger.append("notice", stage, what="cancelling the queue ticket failed",
                              ticket=ticket, error=str(exc))
            block("waited past the budget for a lease; the ticket was cancelled", ticket=ticket,
                  waited_s=waited(), budget_s=wait_budget_s)
        sleep(poll_s)
        try:
            return granted(adapter.claim(ticket, chips, who, reason))
        except Queued:
            continue
        except TicketGone:
            ledger.append("notice", stage, what="queue ticket expired; the place in the queue is lost",
                          ticket=ticket, waited_s=waited())
            if replaced:
                block("a second queue ticket expired", ticket=ticket)
            replaced = True
            lease, ticket = enqueue()
            if lease is not None:
                return lease
        except Refused as exc:
            block(f"the lease tool refused the claim: {exc}", ticket=ticket)


@dataclass(frozen=True)
class Recovery:
    phase: str
    park_done: tuple[str, ...]
    restore_done: tuple[str, ...]
    lease_id: str | None
    lease_state: str           # "none", "ours", "orphaned", "gone" or "taken"
    coder_running: bool
    standin_running: bool
    action: str                # "none", "abandon" or "restore"
    disagreements: tuple[str, ...]


def recover(entries: list[dict], *, adapter, server, standin=None) -> Recovery:
    """After a supervisor restart, compare the ledger with the machine. The machine wins.

    The coder's state comes from the server's own tooling (docker for a container, ps and pgrep
    for a process), because a container server can hold chips that gozer shows as free or as
    another holder (spec section 6, branch table). The lease's state comes from the lease tool:
    "ours" when this process owns it, "orphaned" when the previous supervisor pid still owns it,
    "gone" when its chips are free, "taken" otherwise. Every disagreement is listed.

    The action: "abandon" when the coder is running and the park never sent its stop (the coder
    never left, so it is not stopped; re-leasing a coder that serves outside a park is plan 4's
    job), "restore" for any other interrupted handoff, "none" when no handoff was in progress.
    The stand-in is a process in its own session, so it can outlive the crash; its recorded pid
    is adopted and checked.
    """
    p = progress(entries)
    if p.phase == "idle":
        return Recovery("idle", (), (), None, "none", False, False, "none", ())
    if p.server:
        server.adopt(p.server)
    if standin is not None and p.standin:
        standin.adopt(p.standin)
    running = not server.confirm_stopped().stopped
    standin_running = bool(standin is not None and p.standin
                           and not standin.confirm_stopped().stopped)
    lease_id = (p.lease or {}).get("lease_id")
    lease_chips = set((p.lease or {}).get("chips") or ())
    mine = [c for c in adapter.status() if c.bdf in lease_chips]
    if mine and all(c.lease_pid == adapter.owner_pid for c in mine):
        state = "ours"
    elif not mine or all(c.state == "FREE" for c in mine):
        state = "gone"
    elif p.owner_pid is not None and all(c.lease_pid == p.owner_pid for c in mine):
        state = "orphaned"
    else:
        state = "taken"
    disagreements = []
    if "stopped" in p.park_done and "serve" not in p.restore_done and running:
        disagreements.append("the ledger says the coder was stopped; the machine shows it running")
    if "serve" in p.restore_done and not running:
        disagreements.append("the ledger says the coder was started; the machine shows it stopped")
    if state != "ours":
        disagreements.append(f"the ledger holds lease {lease_id}; the lease tool shows it {state}")
    action = "abandon" if running and "stop_sent" not in p.park_done else "restore"
    return Recovery(p.phase, p.park_done, p.restore_done, lease_id, state, running,
                    standin_running, action, tuple(disagreements))
