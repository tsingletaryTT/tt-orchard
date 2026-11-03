# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The board that holds the lowest device index is reset last before a 4-chip test (issue 26).

Measured on 2026-11-03 on a two-board box, one 1x4 mesh open per trial with the fabric on (1-D, strict init), each trial
starting from a release and a fresh lease:
  one lease on all four chips, one `gozer reset`:              0 of 11 trials failed;
  two leases, board 1 reset and then board 0:                  0 of 12 failed;
  two leases, board 0 reset and then board 1:                  6 of 12 failed, every time with `Fabric Router Sync: Timeout after
                                                               10000 ms on Device 0` and one router stuck at STARTED.
The supervisor resets two leases in the order of their events: the further board's lease at take time, the coder's board at the
park, and both again in list order on a startup retry. With the coder on board 0 the retry order was the failing one, and with the
coder on board 1 the first order was."""
import pytest

import run_fakes
from orchard.supervisor import EXIT_READY, reset_order
from run_fakes import BOARDS, DEV, FAKE_SWAP_TEST, MachineAdapter, bringup
from test_supervisor import Rig
from test_supervisor_fastretry import FLAKY, with_config

pytestmark = pytest.mark.usefixtures("stub_tools")


class FakeLease:
    def __init__(self, *dev_indices):
        self.dev_indices = tuple(dev_indices)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def lowest_index(monkeypatch):
    """lease id -> the lowest device index of the lease, for every lease the supervisor resets."""
    seen = {}
    real = MachineAdapter.reset

    def recording(self, lease):
        seen[lease.lease_id] = min(lease.dev_indices)
        return real(self, lease)
    monkeypatch.setattr(MachineAdapter, "reset", recording)
    return seen


def four_chip_entries(rig):
    return [e for e in rig.entries() if e["stage"] == 4 and (e["data"].get("config") == 4 or e["event"] == "park")]


def test_the_lease_with_the_lowest_device_index_is_reset_last():
    board0, board1 = FakeLease(0, 1), FakeLease(2, 3)
    assert reset_order([board0, board1]) == [board1, board0]
    assert reset_order([board1, board0]) == [board1, board0]


def test_reset_order_keeps_every_lease_and_does_not_change_the_list_it_is_given():
    leases = [FakeLease(0, 1), FakeLease(2, 3)]
    ordered = reset_order(leases)
    assert [l.dev_indices for l in leases] == [(0, 1), (2, 3)] and len(ordered) == 2


def test_a_startup_retry_resets_the_board_with_device_zero_last(rig, lowest_index):
    """The park lease (the coder's board, device 0) comes first in the list the retry walks. It must still be reset last."""
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    first, second = [i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"]
    between = [lowest_index[e["data"]["lease_id"]] for e in es[first:second]
               if e["data"].get("decision") == "test chips reset"]
    assert between == [DEV[BOARDS["B1"][0]], DEV[BOARDS["B0"][0]]]


def test_with_the_coder_on_board_1_the_further_board_is_reset_again_after_the_park(rig, lowest_index, monkeypatch):
    """Coder on board 1: the further board (board 0) was reset at take time, then the park reset board 1. Board 0 must be reset
    once more after the park, so that the lowest device index is reset last."""
    monkeypatch.setattr(run_fakes, "BOARDS", {"B1": BOARDS["B1"], "B0": BOARDS["B0"]})
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    park = next(i for i, e in enumerate(es) if e["event"] == "park")
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    before = [e for e in es[:park] if e["data"].get("decision") == "test chips reset"]
    after = [e for e in es[park:started] if e["data"].get("decision") == "test chips reset"]
    assert len(before) == 1 and len(after) == 1
    assert before[0]["data"]["lease_id"] == after[0]["data"]["lease_id"]       # the further board, twice
    assert lowest_index[after[0]["data"]["lease_id"]] == DEV[BOARDS["B0"][0]]


def test_with_the_coder_on_board_0_the_order_is_already_right_and_nothing_is_reset_again(rig, lowest_index):
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    park = next(i for i, e in enumerate(es) if e["event"] == "park")
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    assert [e for e in es[park:started] if e["data"].get("decision") == "test chips reset"] == []
    assert len([e for e in es[:park] if e["data"].get("decision") == "test chips reset"]) == 1
