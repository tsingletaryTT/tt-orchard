# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 591: the "hardware phase" decision records what `decide_park` saw.

In run 24b of the tiny Qwen4 rehearsal the supervisor parked the coder with `free_boards=[]` while `gozer status` showed two chips FREE. The park
cost two chip resets and a second coder load. The ledger held only the board lists, so the cause could not be read afterwards. The decision now also
carries `board_facts` (`handoff.decision_facts`): each board's chips with their state and `who`, whether the coder holds the board, and the reason a
board that shows FREE chips was not counted free. The decision itself is unchanged.
"""
import json

import pytest

from orchard import handoff
from orchard.adapters import ChipState
from orchard.handoff import decide_park, decision_facts
from orchard.supervisor import EXIT_READY
from test_supervisor import rig  # noqa: F401
from test_supervisor_container_base import wide_rig


def chip(bdf, board, state="FREE", who=None):
    return ChipState(bdf=bdf, state=state, who=who, board=board)


TWO_FREE = [chip("0000:01:00.0", "A"), chip("0000:02:00.0", "A"), chip("0000:03:00.0", "B"), chip("0000:04:00.0", "B")]


def test_the_facts_list_each_board_with_its_chips_and_whether_the_coder_holds_it():
    chips = [chip("0000:01:00.0", "A", "HELD", "coder"), chip("0000:02:00.0", "A", "HELD", "coder"),
             chip("0000:03:00.0", "B"), chip("0000:04:00.0", "B")]
    f = decision_facts(["A"], chips)
    assert f["boards"]["A"]["coder"] is True and f["boards"]["B"]["coder"] is False
    assert f["boards"]["B"]["chips"] == [{"bdf": "0000:03:00.0", "state": "FREE", "who": None},
                                         {"bdf": "0000:04:00.0", "state": "FREE", "who": None}]
    assert f["not_counted_free"] == [{"board": "A", "why": "the coder's lease holds this board"}]
    assert f["free_chips_not_on_free_boards"] == []


def test_free_chips_on_the_coders_own_board_are_named_because_that_is_the_run_24b_picture():
    """The coder's lease lists a board whose chips gozer shows FREE: the board is not counted free and the chips are listed."""
    f = decision_facts(["A", "B"], TWO_FREE)
    assert decide_park(["A", "B"], TWO_FREE, 1).free_boards == ()
    assert {x["board"] for x in f["not_counted_free"]} == {"A", "B"}
    assert f["free_chips_not_on_free_boards"] == ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]


def test_a_free_chip_beside_a_busy_one_says_which_chip_kept_the_board_from_counting():
    chips = [chip("0000:01:00.0", "A"), chip("0000:02:00.0", "A", "HELD", "someone"), chip("0000:03:00.0", "B", "HELD", "coder"),
             chip("0000:04:00.0", "B", "HELD", "coder")]
    f = decision_facts(["B"], chips)
    assert {"board": "A", "why": "chip 0000:02:00.0 is HELD (who: someone)"} in f["not_counted_free"]
    assert f["free_chips_not_on_free_boards"] == ["0000:01:00.0"]


def test_a_free_chip_that_lists_a_who_is_not_counted_free_and_the_facts_say_so():
    chips = [chip("0000:01:00.0", "A", "FREE", "left-over"), chip("0000:02:00.0", "A")]
    f = decision_facts([], chips)
    assert decide_park([], chips, 1).free_boards == ()
    assert f["not_counted_free"] == [{"board": "A", "why": "chip 0000:01:00.0 is FREE but lists who: left-over"}]


def test_chips_with_no_board_serial_are_named():
    chips = [chip("0000:01:00.0", ""), chip("0000:02:00.0", "")]
    f = decision_facts([], chips)
    assert any("no board serial" in x["why"] for x in f["not_counted_free"])


def test_a_board_that_is_counted_free_has_no_reason():
    f = decision_facts([], TWO_FREE)
    assert f["not_counted_free"] == [] and f["free_chips_not_on_free_boards"] == []


def test_the_facts_agree_with_the_decision_on_which_boards_are_free():
    """The facts and the decision use one rule, so a board listed in `not_counted_free` is never in `free_boards`."""
    for server in ([], ["A"], ["B"], ["A", "B"]):
        d = decide_park(server, TWO_FREE, 1)
        f = decision_facts(server, TWO_FREE)
        assert not {x["board"] for x in f["not_counted_free"]} & set(d.free_boards)
        assert set(f["boards"]) - {x["board"] for x in f["not_counted_free"]} == set(d.free_boards)


def test_the_facts_are_plain_json():
    json.dumps(decision_facts(["A"], TWO_FREE))


def test_a_who_that_is_not_text_does_not_crash_the_facts():
    chips = [ChipState(bdf="0000:01:00.0", state="FREE", who=["x"], board="A")]
    assert decision_facts([], chips)["boards"]["A"]["chips"][0]["who"] == "['x']"


def test_the_supervisors_hardware_phase_decision_carries_the_facts(tmp_path):
    r = wide_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        phase = [d for d in r.decisions() if d.get("decision") == "hardware phase"]
        assert phase and all(isinstance(d.get("board_facts"), dict) and "boards" in d["board_facts"] for d in phase), phase
    finally:
        r.close()


def test_every_hardware_phase_decision_of_a_plain_run_carries_the_facts_and_they_name_the_coders_board(rig):  # noqa: F811
    assert rig.run() == EXIT_READY
    phase = [d for d in rig.decisions() if d.get("decision") == "hardware phase"]
    assert phase
    for d in phase:
        facts = d["board_facts"]
        assert set(facts["boards"]) >= set(d["server_boards"]) and all(facts["boards"][b]["coder"] for b in d["server_boards"])
        assert {x["board"] for x in facts["not_counted_free"]} >= set(d["server_boards"])
        assert not {x["board"] for x in facts["not_counted_free"]} & set(d["free_boards"])
