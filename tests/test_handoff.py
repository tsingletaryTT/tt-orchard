"""Park and restore against a fake machine (tests/fakes.py)."""
import pytest

from fakes import Crash, World, make_handoff, steps
from orchard.handoff import Blocked, Handoff, progress
from orchard.ledger import file_evidence, replay_state
from orchard.server import NotReady

PARK = [("park", s) for s in ("note", "canary_before", "standin_started", "standin", "stop_sent",
                              "stopped", "reset")]
RESTORE = [("restore", s) for s in ("reset", "serve", "ready", "canary", "resumed")]


def ledger_of(h):
    return h.ledger


def notices(ledger):
    return [e["data"] for e in ledger.read() if e["event"] == "notice"]


def test_park_then_restore_runs_every_step_in_order(tmp_path):
    h, world, ledger = make_handoff(tmp_path)
    lease = h.park()
    assert lease.lease_id == "L1" and world.resets == ["L1"] and not world.coder_running
    assert replay_state(ledger.read())["parked"] is True
    result = h.restore()
    assert result.match and steps(ledger) == PARK + RESTORE
    assert world.resets == ["L1", "L1"] and world.coder_running and not world.standin_running
    assert replay_state(ledger.read())["parked"] is False
    names = [e["data"]["name"] for e in ledger.read() if e["event"] == "measurement"]
    assert names == ["park_reset_seconds", "restore_reset_seconds", "coder_ready_wait_seconds"]


def test_coder_is_not_stopped_until_the_standin_answers(tmp_path):
    h, world, _ = make_handoff(tmp_path)
    h.park()
    assert world.events.index("standin_ask") < world.events.index("coder_stop")


def test_a_standin_that_does_not_answer_leaves_the_coder_up(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="stand-in did not answer"):
        h.park()
    assert world.coder_running and "coder_stop" not in world.events
    assert not world.standin_running
    assert notices(ledger)[-1]["blocked"] is True
    # The park is closed: the coder never left, so the run is not parked.
    assert steps(ledger)[-1] == ("park", "abandoned")
    assert replay_state(ledger.read())["parked"] is False and progress(ledger.read()).phase == "idle"


def test_after_an_abandoned_park_a_new_park_starts_from_its_note(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.standin_fails = False
    h.park()
    assert steps(ledger)[-len(PARK):] == PARK and world.resets == ["L1"]


def test_parked_and_progress_agree_after_every_entry(tmp_path):
    # replay_state is the authority on "parked"; progress() must agree after every entry,
    # including an abandoned park.
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.standin_fails = False
    h.park()
    h.restore()
    entries = ledger.read()
    for n in range(len(entries) + 1):
        prefix = entries[:n]
        assert replay_state(prefix)["parked"] == (progress(prefix).phase != "idle"), n


def test_the_standin_pid_is_recorded_before_its_canary(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    started = [e["data"] for e in ledger.read()
               if e["event"] == "park" and e["data"]["step"] == "standin_started"]
    assert started[0]["standin"]["pid"] == 5150


def test_the_standin_pid_is_recorded_before_its_readiness_wait(tmp_path):
    # The readiness wait is the longest step of a park (a CPU model load, up to 600 s). The
    # supervisor dies inside it: the ledger must already hold the pid and pgid to stop.
    world = World()
    world.standin_wait_raises = Crash("killed while the stand-in loads")
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Crash):
        h.park()
    assert world.standin_running                      # the orphan the ledger has to point at
    started = [e["data"] for e in ledger.read()
               if e["event"] == "park" and e["data"]["step"] == "standin_started"]
    assert len(started) == 1
    assert started[0]["standin"]["pid"] == 5150 and started[0]["standin"]["pgid"] == 5150
    assert world.events.index("standin_start") < world.events.index("standin_wait_ready")


def test_a_standin_that_never_becomes_ready_is_stopped_and_the_park_abandoned(tmp_path):
    world = World()
    world.standin_wait_raises = NotReady(600.0)
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="stand-in did not start"):
        h.park()
    assert world.coder_running and "coder_stop" not in world.events
    assert not world.standin_running                  # it was stopped, so no orphan is left
    assert steps(ledger)[-1] == ("park", "abandoned")


def test_a_standin_that_survives_its_stop_is_reported(tmp_path):
    world = World()
    world.standin_survives_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    h.restore()
    resumed = [e["data"] for e in ledger.read()
               if e["event"] == "restore" and e["data"]["step"] == "resumed"]
    assert resumed[0]["standin_stopped"] is False
    assert any(n.get("what", "").startswith("the stand-in is still running") for n in notices(ledger))


def test_an_empty_standin_answer_leaves_the_coder_up(tmp_path):
    world = World()
    world.standin_answer = "  "
    h, world, _ = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="empty answer"):
        h.park()
    assert world.coder_running and steps(ledger_of(h))[-1] == ("park", "abandoned")


@pytest.mark.parametrize("content", [None, "not json", '{"goal": "x"}'])
def test_a_missing_or_incomplete_note_blocks_before_anything_runs(tmp_path, content):
    h, world, ledger = make_handoff(tmp_path)
    if content is None:
        h.note_path.unlink()
    else:
        h.note_path.write_text(content)
    with pytest.raises(Blocked, match="handoff note"):
        h.park()
    assert "standin_start" not in world.events and world.coder_running
    assert steps(ledger) == []


def test_a_leftover_worker_blocks_the_reset(tmp_path):
    world = World()
    world.leave_worker_on_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="not confirmed stopped"):
        h.park()
    assert world.resets == [] and not any(c[0] == "reset" for c in h.adapter.calls)
    assert notices(ledger)[-1]["evidence"]["chip_states"]["0000:01:00.0"] == "HELD-FOREIGN"


def test_a_neighbours_reset_that_ends_inside_the_wait_does_not_block(tmp_path):
    world = World()
    world.foreign_polls = 3          # tt-smi -r of the other board opens every device for ~42 s
    h, world, _ = make_handoff(tmp_path, world)
    h.park()
    assert world.resets == ["L1"]


def test_a_refused_reset_is_retried_once_then_blocks(tmp_path):
    world = World()
    world.refuse_resets = 2
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="refused the reset twice"):
        h.park()
    assert [c for c in h.adapter.calls if c[0] == "reset"] == [("reset", "L1"), ("reset", "L1")]
    assert [e["data"]["what"] for e in ledger.read() if e["event"] == "retry"] == ["reset"]
    assert world.resets == []


def test_the_stop_is_checked_again_before_the_reset_is_retried(tmp_path):
    # Spec Review Focus 2. After the refusal a container that only docker sees shows up. The
    # chips still look CLAIMED, so only the server check can stop the retry.
    world = World()
    world.refuse_resets = 1
    world.container_appears_on_refusal = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="reset was refused and the chips are still in use") as exc:
        h.park()
    assert [c for c in h.adapter.calls if c[0] == "reset"] == [("reset", "L1")]
    assert world.resets == [] and exc.value.evidence["checks"] == {"docker_ps": False}
    assert not [e for e in ledger.read() if e["event"] == "retry"]


def test_one_refusal_then_success_records_two_attempts(tmp_path):
    world = World()
    world.refuse_resets = 1
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    reset = [e for e in ledger.read() if e["event"] == "park" and e["data"]["step"] == "reset"]
    assert reset[0]["data"]["attempts"] == 2


def test_a_reset_left_running_is_never_retried(tmp_path):
    world = World()
    world.reset_hangs = True
    h, world, _ = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="reset failed"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1


def test_a_failed_reset_blocks_at_once(tmp_path):
    # Spec section 10: if the reset fails the stage blocks. Only a busy refusal is retried.
    world = World()
    world.reset_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="reset failed"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1
    assert [e for e in ledger.read() if e["event"] == "retry"] == []


def test_an_adapter_that_cannot_reset_blocks_at_once_and_says_why(tmp_path):
    world = World()
    world.reset_unavailable = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="cannot reset these chips"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1
    assert [e for e in ledger.read() if e["event"] == "retry"] == []
    assert h.clock() < h.budgets.quiet_wait_s      # no quiet wait was spent on it


@pytest.mark.parametrize("mesh_reset,parks", [(True, True), (False, False)])
def test_a_mesh_reset_by_tt_model_stop_extends_the_quiet_wait(tmp_path, mesh_reset, parks):
    # tt-model's own reset (its SIGKILL path) runs outside gozer: our chips look busy for longer.
    world = World()
    world.stop_mesh_reset = mesh_reset
    world.foreign_polls = 40                   # 80 s at a 2 s poll: past 60 s, inside 120 s
    h, world, _ = make_handoff(tmp_path, world)
    if parks:
        h.park()
        assert world.resets == ["L1"]
    else:
        with pytest.raises(Blocked, match="not confirmed stopped"):
            h.park()


def test_evidence_files_are_never_overwritten(tmp_path):
    world = World()
    world.coder_answer_after = "5"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked):
        h.restore()
    first = notices(ledger)[-1]["evidence"]["after_file"]
    world.coder_answer_after = "4"             # the operator found the cause
    assert h.restore().match
    # The blocked attempt's file is still the file its notice hashed.
    assert file_evidence(first["path"]) == first
    afters = sorted(p.name for p in h.evidence_dir.glob("canary-after-*.txt"))
    assert len(afters) == 2


def test_a_start_that_times_out_while_the_coder_boots_starts_nothing_else(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.start_hangs_but_comes_up = True
    with pytest.raises(Blocked, match="coming up"):
        h.restore()
    world.start_hangs_but_comes_up = False
    with pytest.raises(Blocked, match="not starting another"):
        h.restore()
    assert world.coder_starts == 2             # the first start and the one that timed out


def test_a_lost_lease_blocks_without_starting_a_server(tmp_path):
    h, world, _ = make_handoff(tmp_path)
    h.park()
    world.lose_lease_on_reset = True
    with pytest.raises(Blocked, match="lease is gone"):
        h.restore()
    assert world.coder_starts == 1


def test_a_coder_that_never_returns_blocks_without_a_retry(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.never_ready = True
    with pytest.raises(Blocked, match="cold-boot budget"):
        h.restore()
    assert world.coder_starts == 2               # the first start, and one start in the restore
    assert ("restore", "ready") not in steps(ledger)


def test_a_changed_canary_blocks(tmp_path):
    world = World()
    world.coder_answer_after = "5"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked, match="canary answer changed") as exc:
        h.restore()
    assert exc.value.evidence["whitespace_only"] is False
    assert ("restore", "canary") not in steps(ledger)


def test_a_whitespace_only_canary_difference_still_blocks_and_says_so(tmp_path):
    world = World()
    world.coder_answer_after = "4\n"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked) as exc:
        h.restore()
    assert exc.value.evidence["whitespace_only"] is True
    assert notices(ledger)[-1]["evidence"]["after"] == "4\n"


def test_park_resumes_after_the_last_recorded_step(tmp_path):
    world = World()
    world.leave_worker_on_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.worker_left = False                   # the operator stopped the worker
    again = Handoff(ledger=ledger, stage=2, adapter=h.adapter, server=h.server, standin=h.standin,
                    lease=h.lease, canary_prompt=h.canary_prompt, note_path=h.note_path,
                    evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    again.park()
    assert world.events.count("standin_start") == 1 and world.events.count("coder_stop") == 1
    assert steps(ledger) == PARK


def test_restore_after_a_block_does_not_repeat_a_finished_step(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.never_ready = True
    with pytest.raises(Blocked):
        h.restore()
    world.never_ready = False                   # the operator looked; the coder came up late
    assert h.restore().match
    assert world.coder_starts == 2
    assert steps(ledger) == PARK + RESTORE


def test_restore_with_nothing_parked_is_an_error(tmp_path):
    h, _, _ = make_handoff(tmp_path)
    with pytest.raises(ValueError, match="nothing to restore"):
        h.restore()
