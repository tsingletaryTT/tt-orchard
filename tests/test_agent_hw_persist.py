# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, round 10, finding 2: a call that finished (the wrapper has exited and released) is written to the record, with its chip time,
before any interrupt that lands afterwards is handled. The interrupt may land in the status check, in the creation of the record, or
in the check of an interrupted call. Fakes only."""
from __future__ import annotations

import json

import pytest

from hwfakes import PID  # noqa: F401
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, attempt_dir, fake_gozer, make_check  # noqa: F401
from test_agent_hw import fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_agent_hw_states import inner_ok  # noqa: F401

GOOD = {"passed": True, "lease_s": 40.0, "waited_s": 2.0, "lease_id": "L1", "wrapper_pid": PID, "device_s": 0.5}


def tool(*, run=None, check=None, stamp=None, snapshots=None):
    return hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=lambda k: (None, "ab" * 32),
                                run=run or (lambda k, t, w: (dict(GOOD), None, "")), check=check,
                                on_record=(lambda s: snapshots.append(json.loads(json.dumps(s)))) if snapshots is not None else None,
                                **({"stamp": stamp} if stamp else {}))


def boom(*a, **k):
    raise KeyboardInterrupt


def test_an_interrupt_in_the_status_check_after_a_finished_call_leaves_the_call_recorded_with_its_chip_time():
    snaps = []
    c = tool(check=boom, snapshots=snaps)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    last = snaps[-1]
    assert last["calls"][0]["outcome"] == "pass" and last["calls"][0]["lease_s"] == 40.0 and last["lease_s_used"] == 40.0
    assert last["stopped"] and "interrupted" in last["stopped"]                 # fail closed: it is not known whether a chip is held
    assert c.state == "stopped"


def test_the_record_is_written_before_the_status_check_starts():
    snaps, seen = [], []
    c = tool(check=lambda pids: seen.append(len(snaps)) or None, snapshots=snaps)
    c.call({})
    assert seen == [1]                                                           # one snapshot (the call) existed when the check ran


def test_an_interrupt_in_the_check_of_an_interrupted_call_keeps_the_call_recorded():
    snaps = []

    def run(k, t, w):
        raise KeyboardInterrupt
    c = tool(run=run, check=boom, snapshots=snaps)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert snaps[0]["calls"][0]["outcome"] == "interrupted" and snaps[-1]["stopped"]


def test_an_interrupt_while_the_record_is_being_made_still_records_the_finished_call_once_with_its_chip_time():
    snaps, state = [], {"n": 0}

    def stamp():
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt                                              # lands as the call's record is created
        return "2026-10-27T00:00:00Z"
    looks = []
    c = tool(stamp=stamp, snapshots=snaps, check=lambda pids: looks.append(set(pids)) or None)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = snaps[-1]["calls"]
    assert len(rec) == 1 and rec[0]["outcome"] == "interrupted" and rec[0]["lease_s"] == 40.0 and rec[0]["lease_id"] == "L1"
    assert snaps[-1]["lease_s_used"] == 40.0                                     # charged once, not twice
    assert len(looks) == 1                                    # and the one look at gozer still happened


def test_an_interrupt_after_the_chip_time_was_charged_does_not_charge_it_again():
    snaps = []
    c = tool(check=boom, snapshots=snaps)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert c.lease_used == 40.0


def test_through_the_probe_the_calls_file_holds_the_finished_call_when_the_status_read_is_interrupted(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer()
    original = gz.sh

    def sh(argv, timeout):
        if argv[1] == "status":
            raise KeyboardInterrupt
        return original(argv, timeout)
    gz.sh = sh
    check = make_check(adir, tmp_path, inner_ok, gozer=gz)
    with pytest.raises(KeyboardInterrupt):
        check.call({})
    saved = json.loads(kp.hwcheck_calls_path(adir.parent.parent, "add", 1).read_text())
    assert saved["calls"][0]["outcome"] == "pass" and saved["calls"][0]["lease_s"] > 0 and saved["stopped"]
