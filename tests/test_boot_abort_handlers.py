# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 603 and 604 (Copilot review of PR 599): the handlers that catch a blocked coder boot or a refused coder start.

603. `_restore_coder` and `_ensure_coder` catch the hook's `Blocked` and probe the PCIe bus once more (`_block_if_off_bus`). The boot abort has
     already read the device state (a probe that is allowed DEVICE_STATE_READ_S); a second synchronous probe can hang before the pause and the
     cleanup. The probe is skipped when the exception's evidence already holds `device_state`, and kept for every other handoff failure.
604. A refused start launched nothing, so the ledger must say `owned: False` before any lookup that can crash (the leftover removal, the evidence
     gathering of the second refusal). `owned: None` before each launch says "unknown": a crash during a retry launch can leave a container
     that is this run's own. A restart reads the last coder record of the ledger (`stages.coder_state`) and `ServerControl.adopt`: False means
     recovery never stops that container; None means it may stop one under the coder's name."""
from types import SimpleNamespace

import pytest

from fakes import World, make_handoff
from orchard.handoff import Blocked
from orchard.stages import coder_state
from orchard.supervisor import Supervisor
from test_coder_chips_held import NAME, REFUSAL, ScriptedCoder, earlier_launch, launch, rig  # noqa: F401  (rig is a fixture)


# ---- 603 -------------------------------------------------------------------------------------------------------------------------------

def sup_counting_probes(tmp_path, restore_exc):
    """A Supervisor whose bus probe is counted and whose handoff raises `restore_exc` from restore() and recover_and_restore()."""
    h, world, ledger = make_handoff(tmp_path, World())

    def raiser(*a, **k):
        raise restore_exc
    h.restore, h.recover_and_restore = raiser, raiser
    sup = Supervisor.__new__(Supervisor)
    sup.probes = []
    sup._off_bus = lambda held: sup.probes.append(list(held)) or None          # the bus is healthy: the probe returns no finding
    sup.ledger, sup.adapter, sup.coder, sup.standin = ledger, h.adapter, h.server, h.standin
    sup.coder_chips, sup.coder_lease, sup.keep_on_stall = 2, None, False
    sup._helpers_running = lambda: []
    sup._handoff = lambda stage, lease: h
    return sup, h


def test_a_restore_blocked_with_the_device_state_already_read_is_not_probed_again(tmp_path):
    exc = Blocked("the boot aborted", device_state={"bus": None})
    sup, h = sup_counting_probes(tmp_path, exc)
    with pytest.raises(Blocked) as got:
        sup._restore_coder(2, h)
    assert got.value is exc and sup.probes == []


def test_a_restore_blocked_for_another_reason_is_still_probed_once(tmp_path):
    exc = Blocked("the lease was not released")
    sup, h = sup_counting_probes(tmp_path, exc)
    with pytest.raises(Blocked) as got:
        sup._restore_coder(2, h)
    assert got.value is exc and len(sup.probes) == 1


def interrupted(tmp_path, exc):
    sup, h = sup_counting_probes(tmp_path, exc)
    h.park()                                       # a park that has not been restored: progress() is not idle, so _ensure_coder recovers
    return sup, h


def test_an_interrupted_restore_blocked_with_the_device_state_already_read_is_not_probed_again(tmp_path):
    exc = Blocked("the boot aborted", device_state={"bus": None})
    sup, h = interrupted(tmp_path, exc)
    with pytest.raises(Blocked) as got:
        sup._ensure_coder()
    assert got.value is exc and sup.probes == []


def test_an_interrupted_restore_blocked_for_another_reason_is_still_probed_once(tmp_path):
    exc = Blocked("the lease was not released")
    sup, h = interrupted(tmp_path, exc)
    with pytest.raises(Blocked):
        sup._ensure_coder()
    assert len(sup.probes) == 1


def test_a_device_state_in_the_evidence_must_be_the_key_not_a_substring_of_the_reason(tmp_path):
    exc = Blocked("device_state is mentioned in the reason only")
    sup, h = sup_counting_probes(tmp_path, exc)
    with pytest.raises(Blocked):
        sup._restore_coder(2, h)
    assert len(sup.probes) == 1


# ---- 604 -------------------------------------------------------------------------------------------------------------------------------

def last_coder_record(rig):
    return coder_state(rig.entries())[1]


class CrashingLeftoverLookup(ScriptedCoder):
    def find_container_rows(self):
        raise RuntimeError("the leftover lookup crashed")


def test_a_crash_in_the_leftover_lookup_leaves_owned_false_in_the_ledger(rig):
    """The first refusal: the coder's `owned` is False in memory, and the ledger must say so before the lookup runs."""
    earlier_launch(rig)
    launch(rig, lambda m: CrashingLeftoverLookup(m, serve=[(1, "", REFUSAL)]))
    with pytest.raises(RuntimeError, match="leftover lookup crashed"):
        rig.run()
    record = last_coder_record(rig)
    assert record is not None and record["owned"] is False
    refused = [d for d in rig.decisions() if d["decision"] == "coder not started"]
    assert refused and "already in use" in refused[0]["reason"]


def test_a_crash_while_collecting_the_second_refusals_evidence_leaves_owned_false(rig, monkeypatch):
    earlier_launch(rig)
    rig.args.coder_busy_retry_s = 1
    original = Supervisor._busy_evidence

    def crashing(self, refusal, lease, found, *, second=False):
        if second:
            raise RuntimeError("the evidence of the second look crashed")
        return original(self, refusal, lease, found, second=second)
    monkeypatch.setattr(Supervisor, "_busy_evidence", crashing)
    launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), (1, "", REFUSAL)], rows=[]))
    rig.on_sleep = None                                               # the wait before the second start is part of this scenario
    with pytest.raises(RuntimeError, match="second look crashed"):
        rig.run()
    record = last_coder_record(rig)
    assert record is not None and record["owned"] is False
    refused = [d for d in rig.decisions() if d["decision"] == "coder not started"]
    assert len(refused) >= 2                                          # one per refusal, the second written before its evidence was read


class CrashingSecondLaunch(ScriptedCoder):
    """After the leftover is removed the start is tried again: this one crashes the supervisor during that launch."""
    launches = 0

    def start(self, lease):
        type(self).launches += 1
        if type(self).launches == 2:
            raise RuntimeError("the second launch crashed")
        return super().start(lease)


def test_the_launch_after_a_leftover_removal_is_preceded_by_an_unknown_ownership_record(rig):
    earlier_launch(rig)
    CrashingSecondLaunch.launches = 0
    launch(rig, lambda m: CrashingSecondLaunch(m, serve=[(1, "", REFUSAL), (0, "", "", [(NAME, "running", "c2")])]))
    with pytest.raises(RuntimeError, match="second launch crashed"):
        rig.run()
    record = last_coder_record(rig)
    assert record is not None and record["owned"] is None             # the refusal said False; a launch is about to run, so unknown again
    order = [d["decision"] for d in rig.decisions() if d["decision"] in ("coder starting", "coder not started", "coder container removed")]
    assert order[-3:] == ["coder not started", "coder container removed", "coder starting"]


def test_the_refusal_is_recorded_once_per_refusal_not_twice(rig):
    earlier_launch(rig)
    rig.args.coder_busy_retry_s = 0
    launch(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], rows=[]))
    assert rig.run() == 5
    assert len([d for d in rig.decisions() if d["decision"] == "coder not started"]) == 1
