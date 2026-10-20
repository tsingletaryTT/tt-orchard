# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A test that opens more than one board starts on chips that were reset after their last use.

Evidence (2026-10-13, RAG-Gate-27B): the park resets only the coder's board, and the 4-chip test failed twice with
fabric router timeouts on the chips the 1-chip and 2-chip tests had used. Taking all four chips and resetting them
let the same command pass. The cause was not isolated, so this is a cheap precaution (about 42 s), not a fix."""
import json

import pytest

from orchard.supervisor import EXIT_READY
from run_fakes import MachineAdapter
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def taken(rig, config):
    return [e for e in rig.entries() if e["stage"] == 4 and e["data"].get("decision") == "test lease taken"
            and e["data"].get("config") == config]


def test_the_further_board_is_reset_in_place_before_the_coder_is_parked(rig):
    assert rig.run() == EXIT_READY
    lease_id = taken(rig, 4)[0]["data"]["test_lease"]["lease_id"]
    assert ("reset", lease_id) in rig.m.resets
    es = [e for e in rig.entries() if e["stage"] == 4]
    reset = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test chips reset"
                 and e["data"].get("config") == 4)
    assert es[reset]["data"]["lease_id"] == lease_id
    take = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test lease taken"
                and e["data"].get("config") == 4)
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"
                   and e["data"].get("config") == 4)
    parked = [i for i, e in enumerate(es) if e["event"] == "park"]
    assert take < reset < started
    assert not parked or reset < parked[0]


def test_one_and_two_chip_tests_are_not_reset_by_the_precaution(rig):
    assert rig.run() == EXIT_READY
    for config in (1, 2):
        lease_id = taken(rig, config)[0]["data"]["test_lease"]["lease_id"]
        assert ("reset", lease_id) not in rig.m.resets
    assert not [e for e in rig.entries() if e["data"].get("decision") == "test chips reset"
                and e["data"].get("config") in (1, 2)]


def test_a_refused_reset_blocks_before_the_test_runs(rig, monkeypatch):
    def refuse(self, lease):
        from orchard.adapters import Refused
        raise Refused("device still open")
    monkeypatch.setattr(MachineAdapter, "reset", refuse)
    rig.on_sleep = lambda: (_ for _ in ()).throw(Stop())
    with pytest.raises(Stop):
        rig.run()
    es = [e for e in rig.entries() if e["stage"] == 4]
    blocked = [e["data"] for e in es if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked and blocked[-1]["reason"].startswith("resetting the chips for the 4-chip test failed")
    assert not [e for e in es if e["data"].get("decision") == "hardware test started"
                and e["data"].get("config") == 4]
