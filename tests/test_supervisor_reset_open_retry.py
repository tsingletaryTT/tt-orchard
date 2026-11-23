# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A reset refused because a device is open for a moment is retried a few times before the run blocks (issue 798).

Evidence: on a full-port run the 4-chip test's reset was refused ("chips [2, 3] still open by [<pid>]"), the holder
had exited seconds later, and the run blocked as hardware-unhealthy. A read-only `tt-smi -s` poll opens the device
nodes for a moment. These tests keep the retry bounded, make it visible in the ledger, and keep every other refusal
immediate."""
import pytest

from orchard import supervisor as sup
from orchard.adapters import AdapterError, Refused
from orchard.supervisor import EXIT_READY
from run_fakes import MachineAdapter
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


OPEN = "chips [2, 3] still open by [4242] -- stop the workload first, or pass --force"


def stage4(rig):
    return [e for e in rig.entries() if e["stage"] == 4]


def refusals(rig):
    return [e["data"] for e in stage4(rig) if e["data"].get("decision") == "test chip reset refused"]


def refuse_first(monkeypatch, times, message=OPEN, calls=None):
    real = MachineAdapter.reset
    seen = calls if calls is not None else []

    def reset(self, lease):
        # Only the first lease reset in a run is the 4-chip test's (the park resets the coder's board later).
        seen.append(lease.lease_id)
        if lease.lease_id == seen[0] and seen.count(lease.lease_id) <= times:
            raise Refused(message)
        return real(self, lease)
    monkeypatch.setattr(MachineAdapter, "reset", reset)


def test_a_device_open_for_a_moment_is_retried_and_the_run_goes_on(rig, monkeypatch):
    refuse_first(monkeypatch, 2)
    assert rig.run() == EXIT_READY
    es = stage4(rig)
    refused = refusals(rig)
    assert [r["attempt"] for r in refused] == [1, 2]
    assert all("still open by [4242]" in r["reason"] for r in refused)
    reset = next(i for i, e in enumerate(es) if e["data"].get("decision") == "test chips reset"
                 and e["data"].get("config") == 4)
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"
                   and e["data"].get("config") == 4)
    assert reset < started


def test_the_retries_are_spaced_by_the_wait_constant(rig, monkeypatch):
    refuse_first(monkeypatch, 2)
    waited = []
    real_sleep = rig.sleep
    monkeypatch.setattr(rig, "sleep", lambda s: (waited.append(s), real_sleep(s))[1])
    rig.run()
    assert waited.count(sup.RESET_OPEN_WAIT_S) >= 2


def test_a_device_that_stays_open_blocks_after_the_bounded_tries(rig, monkeypatch):
    calls = []
    refuse_first(monkeypatch, 10 ** 6, calls=calls)
    rig.on_sleep = lambda: (_ for _ in ()).throw(Stop()) if len(calls) >= sup.RESET_OPEN_TRIES else None
    with pytest.raises(Stop):
        rig.run()
    es = stage4(rig)
    blocked = [e["data"] for e in es if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked and "still open by [4242]" in blocked[-1]["reason"]
    assert blocked[-1]["reason"].startswith("resetting the chips for the 4-chip test failed")
    assert len(calls) == sup.RESET_OPEN_TRIES
    assert [r["attempt"] for r in refusals(rig)] == list(range(1, sup.RESET_OPEN_TRIES))
    assert not [e for e in es if e["data"].get("decision") == "hardware test started"
                and e["data"].get("config") == 4]


def test_a_refusal_that_is_not_about_an_open_device_blocks_at_once(rig, monkeypatch):
    calls = []
    refuse_first(monkeypatch, 10 ** 6, message="lease 5a disagrees with its units", calls=calls)
    rig.on_sleep = lambda: (_ for _ in ()).throw(Stop())      # the first wait of the run is the pause after the block
    with pytest.raises(Stop):
        rig.run()
    assert len(calls) == 1
    assert [e for e in stage4(rig) if e["event"] == "notice" and e["data"].get("blocked")]
    assert not refusals(rig)


def test_a_failed_reset_that_is_not_a_refusal_blocks_at_once(rig, monkeypatch):
    calls = []

    def fail(self, lease):
        calls.append(lease.lease_id)
        raise AdapterError("gozer reset exited 99: chips [2] still open by [7]")
    monkeypatch.setattr(MachineAdapter, "reset", fail)
    rig.on_sleep = lambda: (_ for _ in ()).throw(Stop())      # the first wait of the run is the pause after the block
    with pytest.raises(Stop):
        rig.run()
    assert len(calls) == 1
    assert [e for e in stage4(rig) if e["event"] == "notice" and e["data"].get("blocked")]
