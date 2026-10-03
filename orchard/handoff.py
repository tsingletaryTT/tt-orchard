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
