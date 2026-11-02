# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A relaunch blocked by the granted-board check launched nothing, and a resume must treat it that way (issue 351).

Before the fix the blocked check left a "coder starting" record. A resume read it as a launch: a foreign server that answered the coder's port
could be accepted, and the missing container counted as the coder's one allowed death."""
import pytest

from orchard.stages import coder_start_blocked, coder_state, run_progress
from orchard.ledger import read_entries
from orchard.supervisor import Control
from run_fakes import BOARDS, DEV
from test_supervisor import (EXIT_READY, Crash, ForeignContainerCoder, GrantsSecondBoardAdapter, Rig, Stop, _blocked_reasons,
                                   _stop_once_blocked)


class GrantedBoardContainerCoder(ForeignContainerCoder):
    foreign_dev = DEV[BOARDS["B1"][0]]         # on the board gozer grants (the checked board is the first one)
    adopted = []                                # every record this coder was told to adopt, across instances
    explode_on = None                           # dev indices at which confirm_stopped raises Crash (a signal during the check)
    stops = []

    def __init__(self, machine):
        super().__init__(machine)
        self.owned = None                       # as ServerControl: None unknown, False another process's, True ours

    def adopt(self, record):
        type(self).adopted.append(record)
        self.owned = False if self.owned is False else record.get("owned")
        super().adopt(record)

    def start(self, lease):
        self.owned = None
        super().start(lease)
        self.owned = True

    def record(self):
        return {**super().record(), "owned": self.owned}

    def stop(self):
        type(self).stops.append(self.owned)
        return super().stop()

    def confirm_stopped(self):
        if type(self).explode_on is not None and self.dev_indices == type(self).explode_on:
            raise Crash()
        return super().confirm_stopped()


def _blocked_reasons_file(rig):
    return [e["data"].get("reason") for e in read_entries(rig.run_dir / "ledger.jsonl") if e["event"] == "notice" and e["data"].get("blocked")][-1:]


def _blocked_rig(tmp_path):
    """A run crashed after stage 0, then resumed on the one-board profile: the relaunch blocks on a foreign container on the granted board."""
    GrantedBoardContainerCoder.foreign_dev = DEV[BOARDS["B1"][0]]
    GrantedBoardContainerCoder.adopted, GrantedBoardContainerCoder.stops, GrantedBoardContainerCoder.explode_on = [], [], None
    rig = Rig(tmp_path, chips=4)
    rig.coder_cls, rig.adapter_cls = GrantedBoardContainerCoder, GrantsSecondBoardAdapter
    rig.on_sleep = _stop_once_blocked(rig)
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    rig.args.coder_chips = 2
    GrantedBoardContainerCoder.stops = []         # only stops after the first run's start count
    with pytest.raises(Stop):
        rig.run(pid=200)
    assert len(_blocked_reasons(rig)) == 1
    return rig


def test_a_blocked_granted_board_check_leaves_a_blocked_record_and_no_starting_record(tmp_path):
    rig = _blocked_rig(tmp_path)
    try:
        entries = rig.entries()
        assert coder_start_blocked(entries)
        names = [e["data"]["decision"] for e in entries if e["event"] == "decision" and str(e["data"].get("decision", "")).startswith("coder st")]
        assert names[-1] == "coder start blocked" and names.count("coder starting") == 1       # the first run's start only
        lease_rec, server_rec, _ = coder_state(entries)         # the granted lease and the granted board's devices stay findable
        assert set(lease_rec["chips"]) == set(BOARDS["B1"])
        assert server_rec["dev_indices"] == [DEV[c] for c in BOARDS["B1"]]
    finally:
        rig.close()


def test_a_resume_after_a_blocked_check_adopts_no_server_and_spends_no_coder_death(tmp_path):
    """Scenarios 1 and 2 of issue 351: the container is gone on the next resume. The recorded server is not adopted, the unfinished-boot
    path does not run, the one restart allowance stays unspent, and the coder starts once under a lease of the new supervisor."""
    rig = _blocked_rig(tmp_path)
    try:
        GrantedBoardContainerCoder.foreign_dev = 99      # the operator removed the foreign container
        GrantedBoardContainerCoder.adopted = []
        pauses = []

        def resume_once():
            # The operator resumes the paused run once. A second pause means the resume went wrong: fail now instead of waiting forever.
            if pauses:
                raise AssertionError("the run paused again after the resume: " + str(_blocked_reasons_file(rig)))
            pauses.append(1)
            Control(rig.run_dir).write("resume")
        rig.on_sleep = resume_once
        assert rig.run(pid=300) == EXIT_READY
        decisions = [d["decision"] for d in rig.decisions() if "decision" in d]
        assert "coder died; restarting it once" not in decisions
        assert "finishing an unfinished coder boot" not in decisions
        assert "retrying a blocked coder start" in decisions
        names = [d["decision"] for d in rig.decisions() if "decision" in d]
        after = names[len(names) - 1 - names[::-1].index("coder start blocked"):]
        assert after.count("coder starting") == 1 and after.count("coder started") == 1
        assert run_progress(rig.entries()).coder_deaths == 0
        assert not coder_start_blocked(rig.entries())
    finally:
        GrantedBoardContainerCoder.foreign_dev = DEV[BOARDS["B1"][0]]
        rig.close()


def test_a_foreign_server_that_answers_with_a_matching_canary_is_not_taken_as_the_coder(tmp_path):
    """Scenario 1: something answers on the coder's port with the baseline canary answer while the foreign container still holds the granted
    board. The resume retries the check, which blocks again; the foreign server is never asked, adopted or accepted, and no stage runs."""
    rig = _blocked_rig(tmp_path)
    try:
        rig.m.coder_running, rig.m.coder_chips = True, ()   # the foreign server: it answers the canary question
        starts = rig.m.coder_starts
        asked = len(rig.m.asked)
        GrantedBoardContainerCoder.adopted = []

        class StillBlocked(GrantedBoardContainerCoder):
            def confirm_stopped(self):
                check = super().confirm_stopped()
                return type(check)(False, check.checks, {})
        rig.coder_cls = StillBlocked

        sleeps = []

        def resume_once():
            sleeps.append(1)
            if len(sleeps) > 40:                         # a run that waits this long is not retrying the check: fail now
                raise AssertionError("the run kept waiting instead of retrying the blocked start")
            if sum(e["data"].get("decision") == "coder start blocked" for e in read_entries(rig.run_dir / "ledger.jsonl")) >= 2:
                raise Stop()                             # blocked again after the resume: the operator stops here
            Control(rig.run_dir).write("resume")
        rig.on_sleep = resume_once
        with pytest.raises(Stop):
            rig.run(pid=300)
        assert rig.m.coder_starts == starts
        assert len(rig.m.asked) == asked                 # nothing was put to the foreign server
        assert not any(d.get("decision") == "coder started" for d in rig.decisions()[-6:])
        assert coder_start_blocked(rig.entries())
    finally:
        rig.close()


def test_a_kill_right_after_the_blocked_record_is_retried_like_a_block(tmp_path):
    """Kill point: the "coder start blocked" entry is written and the process dies before the pause notice. The next resume sees the same
    last start state as after a block, retries the check and launches, with no coder death counted."""
    GrantedBoardContainerCoder.foreign_dev = DEV[BOARDS["B1"][0]]
    rig = Rig(tmp_path, chips=4)
    rig.coder_cls, rig.adapter_cls = GrantedBoardContainerCoder, GrantsSecondBoardAdapter
    try:
        with pytest.raises(Crash):
            rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
        rig.args.coder_chips = 2
        with pytest.raises(Crash):
            rig.run(pid=200, crash_if=lambda e: e["event"] == "decision" and e["data"].get("decision") == "coder start blocked")
        assert coder_start_blocked(rig.entries())
        GrantedBoardContainerCoder.foreign_dev = 99
        calls = []

        def resume_once():
            if calls:
                raise AssertionError("the run paused again after the resume")
            calls.append(1)
            Control(rig.run_dir).write("resume")
        rig.on_sleep = resume_once
        assert rig.run(pid=300) == EXIT_READY
        names = [d["decision"] for d in rig.decisions() if "decision" in d]
        assert "coder died; restarting it once" not in names and "retrying a blocked coder start" in names
        assert run_progress(rig.entries()).coder_deaths == 0
    finally:
        GrantedBoardContainerCoder.foreign_dev = DEV[BOARDS["B1"][0]]
        rig.close()


def test_cleanup_after_a_blocked_check_stops_no_container_and_releases_the_lease_without_a_reset(tmp_path):
    """Review of PR 403: a container on the granted board is another session's. Cleanup of the block must not call stop() on it."""
    rig = _blocked_rig(tmp_path)
    try:
        entries = rig.entries()
        blocked_at = max(i for i, e in enumerate(entries) if e["data"].get("decision") == "coder start blocked")
        record = entries[blocked_at]["data"]["server"]
        assert record["owned"] is False
        after = entries[blocked_at:]
        assert not any(e["data"].get("decision") == "stopping the coder" for e in after)
        released = [e["data"] for e in after if e["data"].get("decision") == "hardware released"]
        assert released and released[0].get("reset") is False
        # stop() ran for the relaunch of the first run's coder (before the check) and never for the foreign container
        assert all(owned is not False for owned in GrantedBoardContainerCoder.stops)
    finally:
        rig.close()


def test_an_interrupt_during_the_granted_board_check_leaves_the_granted_lease_in_the_ledger(tmp_path):
    """Review of PR 403: the lease must be in the ledger before the check runs, so cleanup finds it. The state is a start that launched nothing."""
    GrantedBoardContainerCoder.adopted, GrantedBoardContainerCoder.stops = [], []
    GrantedBoardContainerCoder.foreign_dev = 99
    rig = Rig(tmp_path, chips=4)
    rig.coder_cls, rig.adapter_cls = GrantedBoardContainerCoder, GrantsSecondBoardAdapter
    try:
        with pytest.raises(Crash):
            rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
        rig.args.coder_chips = 2
        GrantedBoardContainerCoder.explode_on = tuple(DEV[c] for c in BOARDS["B1"])
        with pytest.raises(Crash):
            rig.run(pid=200)
        GrantedBoardContainerCoder.explode_on = None
        entries = rig.entries()
        lease_rec, server_rec, _ = coder_state(entries)
        assert set(lease_rec["chips"]) == set(BOARDS["B1"]) and server_rec["owned"] is False
        assert coder_start_blocked(entries)
        # the next resume retries and goes on: no coder death
        rig.on_sleep = None
        assert rig.run(pid=300) == EXIT_READY
        assert run_progress(rig.entries()).coder_deaths == 0
    finally:
        GrantedBoardContainerCoder.explode_on = None
        GrantedBoardContainerCoder.foreign_dev = DEV[BOARDS["B1"][0]]
        rig.close()
