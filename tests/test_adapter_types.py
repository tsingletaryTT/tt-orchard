# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Adapter types: the lease record, chip states and the exceptions."""
import pytest

from orchard.adapters import (ChipState, Lease, Queued, Refused, ResetFailed, TicketGone,
                              boards_of)

LEASE = Lease("fb9995", ("0000:01:00.0", "0000:02:00.0"), (0, 1),
              {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, ("0000046131924062",))


def test_lease_record_round_trips_through_json_types():
    rec = LEASE.record()
    assert rec == {"lease_id": "fb9995", "chips": ["0000:01:00.0", "0000:02:00.0"],
                   "dev_indices": [0, 1], "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"},
                   "units": ["0000046131924062"]}
    assert Lease.from_record(rec) == LEASE


def test_lease_is_hashable_even_with_an_env_dict():
    assert hash(LEASE) == hash(Lease.from_record(LEASE.record()))


def test_unknown_chip_state_is_refused():
    with pytest.raises(ValueError, match="unknown chip state"):
        ChipState("0000:01:00.0", "WEDGED", None)


def test_boards_of_groups_by_serial_in_listed_order():
    chips = [ChipState("a", "FREE", None, board="B0"), ChipState("c", "FREE", None, board="B1"),
             ChipState("b", "CLAIMED", "x", board="B0")]
    groups = boards_of(chips)
    assert list(groups) == ["B0", "B1"]
    assert [c.bdf for c in groups["B0"]] == ["a", "b"]


def test_exceptions_carry_what_the_caller_needs():
    q = Queued("t-1", 3)
    assert (q.ticket, q.position) == ("t-1", 3)
    assert TicketGone("t-1").ticket == "t-1"
    assert ResetFailed("x").left_running is False
    assert ResetFailed("x", left_running=True).left_running is True
    assert Refused("busy").permanent is False and Refused("no reset", permanent=True).permanent
