# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""decide_park, chips_quiet and progress: pure functions over chip states and ledger entries."""
import pytest

from fakes import LEASE
from orchard.adapters import ChipState
from orchard.handoff import chips_quiet, decide_park, progress


def board(serial, first, state, who=None):
    return [ChipState(f"0000:0{first}:00.0", state, who, board=serial),
            ChipState(f"0000:0{first + 1}:00.0", state, who, board=serial)]


# The large tier serves Qwen3.8-27B on all 4 chips; the small tier on 2 (operator, 2026-10-02).
LARGE = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "HELD-FOREIGN", "orchard:run")
SMALL = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "FREE")


@pytest.mark.parametrize("need", [1, 2])
def test_large_tier_on_both_boards_parks_for_every_hardware_stage(need):
    d = decide_park({"B0", "B1"}, LARGE, need)
    assert d.action == "park" and d.park_needed and d.free_boards == ()


def test_small_tier_on_one_board_leaves_the_other_free():
    d = decide_park({"B0"}, SMALL, 1)
    assert d.action == "use_free" and not d.park_needed and d.free_boards == ("B1",)


def test_small_tier_and_a_two_board_stage_parks():
    assert decide_park({"B0"}, SMALL, 2).action == "park"


def test_a_board_leased_by_someone_else_is_not_free():
    chips = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "CLAIMED", "claude:x")
    assert decide_park({"B0"}, chips, 1).action == "park"


def test_a_board_busy_during_a_neighbours_reset_is_not_free():
    chips = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "BUSY-UNTRACKED")
    assert decide_park({"B0"}, chips, 1).action == "park"


def test_no_server_board_and_nothing_free_waits():
    chips = board("B0", 1, "CLAIMED", "claude:x") + board("B1", 3, "CLAIMED", "claude:y")
    assert decide_park(set(), chips, 1).action == "wait"


def test_bad_arguments_are_errors():
    with pytest.raises(ValueError, match="not in the chip list"):
        decide_park({"B9"}, SMALL, 1)
    with pytest.raises(ValueError, match="boards_needed"):
        decide_park({"B0"}, SMALL, 3)


def chip(bdf, state):
    return ChipState(bdf, state, "orchard:run")


def test_chips_quiet_needs_every_lease_chip_claimed():
    ok, states = chips_quiet([chip("0000:01:00.0", "CLAIMED"), chip("0000:02:00.0", "CLAIMED")], LEASE)
    assert ok and states == {"0000:01:00.0": "CLAIMED", "0000:02:00.0": "CLAIMED"}


@pytest.mark.parametrize("second", ["HELD", "HELD-FOREIGN", "BUSY-UNTRACKED", "STALE", "FREE"])
def test_chips_quiet_refuses_any_other_state(second):
    ok, _ = chips_quiet([chip("0000:01:00.0", "CLAIMED"), chip("0000:02:00.0", second)], LEASE)
    assert not ok


def test_chips_quiet_refuses_a_lease_chip_missing_from_status():
    ok, _ = chips_quiet([chip("0000:01:00.0", "CLAIMED")], LEASE)
    assert not ok


def test_chips_quiet_accepts_extra_states_when_asked():
    ok, _ = chips_quiet([chip("0000:01:00.0", "STALE"), chip("0000:02:00.0", "STALE")], LEASE,
                        accept=("CLAIMED", "STALE"))
    assert ok


def e(event, step, stage=2, **data):
    return {"event": event, "stage": stage, "data": {"step": step, **data}}


def test_an_empty_ledger_is_idle():
    assert progress([]).phase == "idle"


def test_park_steps_are_collected_in_order():
    entries = [e("park", "note", lease=LEASE.record(), owner_pid=100, server={"kind": "container"}),
               e("park", "canary_before", canary={"path": "c.txt", "sha256": "ab"}),
               {"event": "measurement", "stage": 2, "data": {"label": "measured"}}]
    p = progress(entries)
    assert (p.phase, p.park_done, p.stage) == ("parking", ("note", "canary_before"), 2)
    assert p.lease == LEASE.record() and p.owner_pid == 100
    assert p.canary_before == {"path": "c.txt", "sha256": "ab"} and p.server == {"kind": "container"}


def test_reset_means_parked_and_restore_steps_mean_restoring():
    entries = [e("park", s) for s in ("note", "canary_before", "standin", "stop_sent", "stopped", "reset")]
    assert progress(entries).phase == "parked"
    entries += [e("restore", "reset"), e("restore", "serve", server={"pid": 9})]
    p = progress(entries)
    assert p.phase == "restoring" and p.restore_done == ("reset", "serve") and p.server == {"pid": 9}


def test_restart_clears_the_restore_steps():
    entries = [e("park", "note"), e("park", "reset"), e("restore", "reset"), e("restore", "serve"),
               e("restore", "restart")]
    p = progress(entries)
    assert p.phase == "restoring" and p.restore_done == ()


def test_resumed_ends_the_handoff_and_a_new_note_starts_another():
    entries = [e("park", "note"), e("park", "reset"), e("restore", "resumed")]
    assert progress(entries).phase == "idle"
    p = progress(entries + [e("park", "note", stage=3)])
    assert p.phase == "parking" and p.park_done == ("note",) and p.stage == 3


def test_the_latest_lease_record_wins():
    new = dict(LEASE.record(), lease_id="new")
    p = progress([e("park", "note", lease=LEASE.record()), e("restore", "restart", lease=new)])
    assert p.lease["lease_id"] == "new"


def test_an_abandoned_park_is_idle():
    entries = [e("park", "note"), e("park", "canary_before"), e("park", "standin_started"),
               e("park", "abandoned")]
    assert progress(entries).phase == "idle"


def test_the_plan_1_form_without_steps_is_read_as_replay_state_reads_it():
    # replay_state: a park entry with no step parks, a restore entry with no step ends the park.
    park = {"event": "park", "stage": 4, "data": {}}
    restore = {"event": "restore", "stage": 4, "data": {"canary": "ok"}}
    assert progress([park]).phase == "parking"
    assert progress([park, restore]).phase == "idle"


def test_a_stop_that_reset_the_mesh_is_remembered():
    entries = [e("park", "note"), e("park", "stop_sent", result={"how": "x", "mesh_reset": True})]
    assert progress(entries).mesh_reset is True
