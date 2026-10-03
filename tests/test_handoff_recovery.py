"""Recovery: kill the supervisor after every ledger event and check the final state (spec section 13)."""
import signal

import pytest

from fakes import (CANARY, FakeProc, WHO, Crash, FakeAdapter, FakeServer, FakeStandIn, World, make_handoff,
                   steps)
from orchard.adapters import Lease
from orchard.handoff import Blocked, Handoff, progress, recover
from orchard.commands import CommandResult
from orchard.ledger import Ledger, replay_state
from orchard.server import ServerControl, ServerSpec, ServerStandIn

NEW_PID = 200


class CrashingLedger(Ledger):
    """A ledger whose process dies right after its k-th append is on disk."""

    def __init__(self, path, crash_after=0):
        super().__init__(path)
        self.crash_after, self.appended = crash_after, 0

    def append(self, *args, **kwargs):
        entry = super().append(*args, **kwargs)
        self.appended += 1
        if self.appended == self.crash_after:
            raise Crash(f"killed after ledger event {self.appended}: {entry['event']} "
                        f"{entry['data'].get('step')}")
        return entry


def restart(tmp_path, h, world):
    """The supervisor died. Start a new one with a new pid and let it recover.

    The stand-in and a bundle coder run in their own sessions (orchard/server.py), so they
    outlive the supervisor. The fake keeps them running across the restart.
    """
    h.ledger.close()
    world.owner_pid = NEW_PID
    ledger = Ledger(tmp_path / "ledger.jsonl")
    adapter, server, standin = FakeAdapter(world, NEW_PID), FakeServer(world), FakeStandIn(world)
    rec = recover(ledger.read(), adapter=adapter, server=server, standin=standin)
    p = progress(ledger.read())
    h2 = Handoff(ledger=ledger, stage=2, adapter=adapter, server=server, standin=standin,
                 lease=Lease.from_record(p.lease) if p.lease else h.lease, canary_prompt=CANARY,
                 note_path=h.note_path, evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    h2.recover_and_restore(rec, chips=2, who=WHO, reason="coder", wait_budget_s=600)
    return rec, ledger


def final_state(world, ledger):
    entries = ledger.read()
    return {"coder_running": world.coder_running, "standin_running": world.standin_running,
            "leases": len(world.leases),
            "lease_owner_is_live": all(o == world.owner_pid for _, o in world.leases.values()),
            "parked": replay_state(entries)["parked"], "phase": progress(entries).phase}


EXPECTED = {"coder_running": True, "standin_running": False, "leases": 1,
            "lease_owner_is_live": True, "parked": False, "phase": "idle"}
# A crash before `stop_sent`: the park is abandoned and the coder, which never left, keeps
# serving under the dead supervisor's lease. Re-leasing it is plan 4's job.
EXPECTED_ABANDONED = dict(EXPECTED, lease_owner_is_live=False)

# A clean park and restore appends 15 entries: park note, canary_before, standin_started,
# standin, stop_sent, stopped, reset, measurement; restore reset, measurement, serve, ready,
# measurement, canary, resumed.
CLEAN_APPENDS = 15
STOP_SENT = 5        # the append number of `park stop_sent`


def stand_in_answered_before_any_stop(world):
    if "coder_stop" not in world.events:
        return True
    return ("standin_ask" in world.events
            and world.events.index("standin_ask") < world.events.index("coder_stop"))


def test_a_clean_run_appends_the_expected_entries(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger)
    h.park()
    h.restore()
    assert ledger.appended == CLEAN_APPENDS
    entries = ledger.read()
    assert entries[STOP_SENT - 1]["data"]["step"] == "stop_sent"
    assert final_state(world, ledger) == EXPECTED


def test_a_crash_after_any_ledger_event_reaches_the_expected_final_state(tmp_path):
    # The last event is `resumed`: after it the handoff is over, and re-leasing a coder that is
    # serving belongs to plan 4 (see test_a_crash_after_resume_leaves_nothing_to_recover).
    failures = []
    for k in range(1, CLEAN_APPENDS):
        d = tmp_path / f"k{k}"
        d.mkdir()
        h, world, ledger = make_handoff(d, ledger_cls=CrashingLedger, ledger_kw={"crash_after": k})
        try:
            h.park()
            h.restore()
            failures.append((k, "no crash happened"))
            continue
        except Crash:
            pass
        rec, ledger2 = restart(d, h, world)
        state = final_state(world, ledger2)
        want = EXPECTED_ABANDONED if k < STOP_SENT else EXPECTED
        if state != want:
            failures.append((k, rec, state))
        if k < STOP_SENT and "coder_stop" in world.events:
            failures.append((k, "a coder whose park never sent its stop was stopped"))
        if not stand_in_answered_before_any_stop(world):
            failures.append((k, "the coder stopped before the stand-in answered", world.events))
        ledger2.close()
    assert failures == []


def test_an_orphaned_standin_is_found_from_the_ledger_and_stopped(tmp_path):
    # Killed right after the stand-in started, before its canary: its pid is in the ledger.
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger, ledger_kw={"crash_after": 3})
    try:
        h.park()
    except Crash:
        pass
    assert world.standin_running                 # it outlived the supervisor
    rec, ledger2 = restart(tmp_path, h, world)
    assert rec.action == "abandon" and rec.standin_running
    assert not world.standin_running
    closing = [e["data"] for e in ledger2.read() if e["event"] == "park"][-1]
    assert closing["step"] == "abandoned" and closing["standin_stopped"] is True


def test_a_crash_after_resume_leaves_nothing_to_recover(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger,
                                    ledger_kw={"crash_after": CLEAN_APPENDS})
    h.park()
    try:
        h.restore()
    except Crash:
        pass
    ledger.close()
    with Ledger(tmp_path / "ledger.jsonl") as led:
        rec = recover(led.read(), adapter=FakeAdapter(world, NEW_PID), server=FakeServer(world))
    assert rec.action == "none" and rec.phase == "idle"


def test_killed_between_tt_model_stop_and_the_reset(tmp_path):
    world = World()
    world.crash_in_stop = True          # the coder stopped, and the ledger never heard of it
    h, world, ledger = make_handoff(tmp_path, world)
    try:
        h.park()
    except Crash:
        pass
    assert ("park", "stop_sent") not in steps(ledger)
    rec, ledger2 = restart(tmp_path, h, world)
    assert rec.phase == "parking" and not rec.coder_running and rec.lease_state == "orphaned"
    assert final_state(world, ledger2) == EXPECTED
    assert world.resets == ["L2"]       # the new lease is reset before the coder starts on it
    decisions = [e["data"]["decision"] for e in ledger2.read() if e["event"] == "decision"]
    assert decisions[0] == "recover after restart; the machine wins"
    assert "new lease after restart" in decisions


def test_the_machine_wins_when_the_ledger_says_stopped_but_docker_shows_the_coder(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger, ledger_kw={"crash_after": 6})
    try:
        h.park()                        # dies right after the `stopped` entry
    except Crash:
        pass
    world.coder_running = True          # someone started the coder again meanwhile
    rec, ledger2 = restart(tmp_path, h, world)
    assert "the ledger says the coder was stopped; the machine shows it running" in rec.disagreements
    assert final_state(world, ledger2) == EXPECTED


def test_recover_with_nothing_in_progress_does_nothing(tmp_path):
    world = World()
    with Ledger(tmp_path / "ledger.jsonl") as led:
        rec = recover(led.read(), adapter=FakeAdapter(world), server=FakeServer(world))
    assert rec.action == "none" and rec.lease_state == "none"


# ---- the stand-in is adopted by its recorded identity (review I2) -----------------------------

STANDIN_SPEC = ServerSpec("t/standin", "process", 20990, "fake", argv=("python3", "fake.py"))


def real_standin(killed, identity):
    """A ServerStandIn over a real ServerControl with fake commands and a fake signal."""
    def run(argv, timeout, *, env=None, kill_on_timeout=True):
        rc = 7 if argv[0] == "curl" else 1 if argv[0] in ("ps", "pgrep") else 0
        header = "State Recv-Q\n"
        return CommandResult(tuple(argv), rc, header if argv[0] == "ss" else "", "")

    ctl = ServerControl(STANDIN_SPEC, run=run, spawn=lambda argv, env, log=None: FakeProc(5150),
                        killpg=lambda pgid, sig: killed.append((pgid, sig)), identity=identity,
                        sleep=lambda s: None)
    return ServerStandIn(ctl, ready_budget_s=1)


def crash_after_the_standin_spawned(tmp_path, killed, boot="boot-A", start=777):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger,
                                    ledger_kw={"crash_after": 3})   # note, canary, standin_started
    h.standin = real_standin(killed, lambda pid: (boot, start))
    with pytest.raises(Crash):
        h.park()
    assert steps(ledger)[-1] == ("park", "standin_started")
    return h, world


def recover_with(tmp_path, h, world, killed, identity):
    h.ledger.close()
    world.owner_pid = NEW_PID
    ledger = Ledger(tmp_path / "ledger.jsonl")
    adapter, server = FakeAdapter(world, NEW_PID), FakeServer(world)
    standin = real_standin(killed, identity)
    rec = recover(ledger.read(), adapter=adapter, server=server, standin=standin)
    p = progress(ledger.read())
    h2 = Handoff(ledger=ledger, stage=2, adapter=adapter, server=server, standin=standin,
                 lease=Lease.from_record(p.lease), canary_prompt=CANARY, note_path=h.note_path,
                 evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    h2.recover_and_restore(rec, chips=2, who=WHO, reason="coder", wait_budget_s=600)
    return rec, ledger


def test_recovery_adopts_the_standin_and_signals_its_recorded_group(tmp_path):
    # Guards recover()'s adopt call: without it the new supervisor has no pgid and sends nothing.
    killed = []
    h, world = crash_after_the_standin_spawned(tmp_path, killed)
    entry = [e["data"] for e in h.ledger.read() if e["data"].get("step") == "standin_started"][0]
    assert entry["standin"]["pgid"] == 5150 and entry["standin"]["start_time"] == 777
    recover_with(tmp_path, h, world, killed, lambda pid: ("boot-A", 777))
    assert killed == [(5150, signal.SIGTERM)]


def test_recovery_does_not_signal_a_group_that_belongs_to_another_process(tmp_path):
    # While the supervisor was down the stand-in exited and its pid went to a stranger.
    killed = []
    h, world = crash_after_the_standin_spawned(tmp_path, killed)
    rec, ledger = recover_with(tmp_path, h, world, killed, lambda pid: ("boot-A", 99999))
    assert killed == []
    assert not rec.standin_running                     # treated as gone
    said = [e["data"].get("what") or e["data"].get("reason") or "" for e in ledger.read()
            if e["event"] == "notice"]
    assert any("is not the process orchard started" in n for n in said)
    assert rec.notices and "5150" in rec.notices[0]


def test_recovery_blocks_when_docker_still_lists_the_coder_after_its_stop(tmp_path):
    # The supervisor died after tt-model stop was sent. On restart docker still lists the
    # container, and gozer shows the chips CLAIMED or STALE because it cannot see that container.
    # Only the server check can tell. No lease is taken and nothing is reset.
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger,
                                    ledger_kw={"crash_after": 5})        # ... stop_sent
    with pytest.raises(Crash):
        h.park()
    assert steps(ledger)[-1] == ("park", "stop_sent")
    h.ledger.close()
    world.owner_pid = NEW_PID
    world.invisible_container = True
    ledger2 = Ledger(tmp_path / "ledger.jsonl")
    adapter, server, standin = FakeAdapter(world, NEW_PID), FakeServer(world), FakeStandIn(world)
    rec = recover(ledger2.read(), adapter=adapter, server=server, standin=standin)
    assert rec.coder_running and rec.action == "restore"
    h2 = Handoff(ledger=ledger2, stage=2, adapter=adapter, server=server, standin=standin,
                 lease=h.lease, canary_prompt=CANARY, note_path=h.note_path,
                 evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    with pytest.raises(Blocked, match="after the restart the coder could not be confirmed stopped"):
        h2.recover_and_restore(rec, chips=2, who=WHO, reason="coder", wait_budget_s=600)
    assert world.resets == [] and not [c for c in adapter.calls if c[0] in ("acquire", "reset")]
    assert world.coder_starts == 1
