# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, round 10, finding 3: a lease record that never finished is charged as an upper bound, never as zero or as the partial time.

A wrapper that was killed has no end stamp, so nobody knows when its lease ended. The call is charged the larger of two figures: the
full call timeout plus the release grace (the longest the lease could have lasted before the group kill), or the seconds the call has
been running up to now. The same rule covers an interrupted call. A record that finished is trusted: its own stamps, and zero seconds
when it was never granted. Fakes only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwfakes import PID
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, attempt_dir, fake_gozer, make_check  # noqa: F401
from test_agent_hw import fake_kenv, skills_current, ttlang_skill  # noqa: F401

FULL = 240.0 + hwtool.RELEASE_GRACE_S                    # the default call timeout plus the release grace


def killed_run(record, elapsed, *, policy=None):
    """A probe-side check whose wrapper was killed after `elapsed` seconds, leaving `record` (or no record when None)."""
    def run_watched(argv, **kw):
        result = Path(argv[argv.index("--result") + 1])
        if record is not None:
            result.write_text(json.dumps(record))
        CLOCK["now"] += elapsed
        return kp.Watched(-9, None, False, elapsed)
    return run_watched


def check_for(tmp_path, fake_kenv, run_watched, **policy):
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer()
    CLOCK["now"] = 0.0
    return kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(**policy),
                                  run_watched=run_watched, status=gz.sh, clock=lambda: CLOCK["now"], sleep=lambda s: None)


HELD = {"pid": PID, "started_at": 1.0, "granted_at": 2.0, "ended_at": None, "lease_id": "L1", "outcome": None, "phase": "held",
        "release_ok": None}


def test_a_killed_wrapper_is_charged_the_full_call_timeout_plus_the_release_grace_when_that_is_larger(tmp_path, fake_kenv):
    c = check_for(tmp_path, fake_kenv, killed_run(HELD, 20.0))
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == FULL and rec["lease_upper_bound"] is True and c.summary()["lease_s_used"] == FULL


def test_a_killed_wrapper_is_charged_the_time_up_to_now_when_that_is_larger(tmp_path, fake_kenv):
    c = check_for(tmp_path, fake_kenv, killed_run(HELD, 500.0))
    c.call({})
    assert c.summary()["calls"][0]["lease_s"] == 500.0


@pytest.mark.parametrize("record", [None, {"pid": PID, "started_at": 1.0, "granted_at": None, "ended_at": None, "lease_id": None,
                                           "outcome": None, "phase": "starting"}], ids=["no record", "record from before the grant"])
def test_a_wrapper_that_left_no_end_stamp_is_charged_in_full_even_when_no_grant_was_recorded(tmp_path, fake_kenv, record):
    c = check_for(tmp_path, fake_kenv, killed_run(record, 5.0))
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == FULL and rec["lease_upper_bound"] is True


def test_a_record_that_finished_without_a_grant_is_charged_nothing(tmp_path, fake_kenv):
    done = {"pid": PID, "started_at": 1.0, "granted_at": None, "ended_at": 7.0, "lease_id": None, "outcome": "busy", "phase": "done",
            "ticket": "T1", "cancel_error": None}
    c = check_for(tmp_path, fake_kenv, killed_run(done, 6.0))
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == 0.0 and not rec.get("lease_upper_bound")


def test_a_record_that_finished_is_trusted_for_its_own_stamps(tmp_path, fake_kenv):
    done = dict(HELD, ended_at=52.0, outcome="ran", phase="done", release_ok=True)
    c = check_for(tmp_path, fake_kenv, killed_run(done, 60.0))
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == 50.0 and not rec.get("lease_upper_bound")


def test_the_call_timeout_in_the_charge_is_this_calls_own_not_the_default(tmp_path, fake_kenv):
    c = check_for(tmp_path, fake_kenv, killed_run(HELD, 10.0), call_timeout_s=100.0)
    c.call({})
    assert c.summary()["calls"][0]["lease_s"] == 100.0 + hwtool.RELEASE_GRACE_S


def test_the_charge_of_a_killed_wrapper_counts_against_the_cap_so_the_next_call_is_refused(tmp_path, fake_kenv):
    c = check_for(tmp_path, fake_kenv, killed_run(HELD, 20.0), lease_s=500.0)
    c.call({})
    assert c.call({}).startswith("hardware_check refused")                   # 500 - 330 leaves less than the 210 reserve plus 60


def test_an_interrupted_call_is_charged_the_full_call_timeout_plus_the_release_grace_when_that_is_larger():
    now = {"t": 0.0}

    def run(k, t, w):
        now["t"] += 12.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=lambda k: (None, "ab" * 32), run=run, clock=lambda: now["t"])
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == FULL and rec["total_s"] == 12.0 and rec["lease_upper_bound"] is True and c.lease_used == FULL


def test_an_interrupted_call_is_charged_the_time_up_to_now_when_that_is_larger():
    now = {"t": 0.0}

    def run(k, t, w):
        now["t"] += 400.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=lambda k: (None, "ab" * 32), run=run, clock=lambda: now["t"])
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert c.summary()["calls"][0]["lease_s"] == 400.0


def test_the_interrupted_charge_uses_the_run_deadline_that_call_was_given():
    now = {"t": 0.0}
    seen = {}

    def run(k, t, w):
        seen["timeout"] = t
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(lease_s=400.0, cleanup_s=100.0), prepare=lambda k: (None, "ab" * 32), run=run,
                             clock=lambda: now["t"])
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    assert seen["timeout"] == 240.0 and c.summary()["calls"][0]["lease_s"] == 240.0 + hwtool.RELEASE_GRACE_S


def test_the_charge_helper_is_the_larger_of_the_two_figures():
    assert hwtool.unfinished_charge_s(240.0, 20.0) == 330.0
    assert hwtool.unfinished_charge_s(240.0, 500.0) == 500.0
    assert hwtool.unfinished_charge_s(240.0, 330.0) == 330.0
