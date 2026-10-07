# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GozerAdapter against hand-written gozer output (tests/test_gozer_contract.py checks the real CLI)."""
import pytest

from fakes import GRANT, LEASE, OWNER, FakeRun
from orchard.adapters import (AdapterError, ChipState, LeaseLost, Queued, Refused, ResetFailed,
                              TicketGone)
from orchard.adapters.gozer import GozerAdapter


def make(script):
    run = FakeRun(script)
    return GozerAdapter(owner_pid=OWNER, run=run), run


OK_RELEASE = (0, {"released": True, "message": "released fb9995; chips reset"}, "")
OK_RESET = (0, {"reset": True, "status": "reset", "message": "Resetting PCI BDFs"}, "")


def test_acquire_sends_the_documented_flags():
    a, run = make({"acquire": [(0, GRANT, "")]})
    a.acquire(2, "orchard:run1", "stage 2")
    assert run.argvs() == [["gozer", "acquire", "--chips", "2", "--who", "orchard:run1",
                            "--reason", "stage 2", "--owner-pid", str(OWNER), "--json",
                            "--no-queue"]]


def test_acquire_returns_the_granted_lease():
    a, _ = make({"acquire": [(0, GRANT, "")]})
    assert a.acquire(2, "w", "r") == LEASE


def test_exact_board_is_passed():
    a, run = make({"acquire": [(0, GRANT, "")]})
    a.acquire(2, "w", "r", exact="0000:01:00.0")
    argv = run.argvs()[0]
    assert argv[argv.index("--exact") + 1] == "0000:01:00.0"


def test_acquire_with_queue_raises_queued_with_the_ticket():
    a, run = make({"acquire": [(10, {"granted": False, "queued": True, "ticket": "t-7",
                                     "position": 2, "ahead": ["claude:x"]}, "")]})
    with pytest.raises(Queued) as exc:
        a.acquire(4, "w", "r", queue=True)
    assert (exc.value.ticket, exc.value.position) == ("t-7", 2)
    assert "--no-queue" not in run.argvs()[0]


def test_acquire_without_queue_is_refused_when_busy():
    a, _ = make({"acquire": [(12, {"granted": False, "queued": False}, "")]})
    with pytest.raises(Refused):
        a.acquire(4, "w", "r")


def test_a_grant_owned_by_another_pid_is_released_and_refused():
    a, run = make({"acquire": [(0, dict(GRANT, owner_pid=999), "")], "release": [OK_RELEASE]})
    with pytest.raises(AdapterError, match="owner_pid 999"):
        a.acquire(2, "w", "r")
    assert run.argvs()[1] == ["gozer", "release", "fb9995", "--json"]


def test_a_grant_whose_env_does_not_list_its_chips_is_refused():
    bad = dict(GRANT, env={"TT_VISIBLE_DEVICES": "0000:01:00.0"})
    a, run = make({"acquire": [(0, bad, "")], "release": [OK_RELEASE]})
    with pytest.raises(AdapterError, match="env does not list"):
        a.acquire(2, "w", "r")
    # An unusable lease is given back at once, or the board stays held until the supervisor dies.
    assert run.argvs()[1] == ["gozer", "release", "fb9995", "--json"]


def test_claim_passes_the_ticket_and_reports_a_gone_ticket():
    a, run = make({"acquire": [(13, "", "gozer: no such ticket t-7")]})
    with pytest.raises(TicketGone):
        a.claim("t-7", 4, "w", "r")
    argv = run.argvs()[0]
    assert argv[-2:] == ["--ticket", "t-7"] and "--no-queue" not in argv


def test_claim_that_is_still_queued_raises_queued():
    a, _ = make({"acquire": [(10, {"granted": False, "queued": True, "ticket": "t-7",
                                   "position": 1}, "")]})
    with pytest.raises(Queued):
        a.claim("t-7", 4, "w", "r")


@pytest.mark.parametrize("rc,payload,exc", [
    (13, {"reset": False, "status": "not-found", "message": "lease fb9995 not found"}, LeaseLost),
    (15, {"reset": False, "status": "refused", "message": "device still open"}, Refused),
    (17, {"reset": False, "status": "failed", "message": "tt-smi -r failed"}, ResetFailed),
    (18, {"reset": False, "status": "changed-hands", "message": "unit held by another lease"}, LeaseLost),
    (2, "", AdapterError),
])
def test_reset_exit_codes(rc, payload, exc):
    a, _ = make({"reset": [(rc, payload, "")]})
    with pytest.raises(exc):
        a.reset(LEASE)


def test_reset_success_needs_status_reset():
    a, run = make({"reset": [OK_RESET, (0, {"reset": True, "status": "odd"}, "")]})
    a.reset(LEASE)
    assert run.argvs()[0] == ["gozer", "reset", "fb9995", "--json"]
    with pytest.raises(AdapterError):
        a.reset(LEASE)


def test_reset_is_never_killed_and_a_hung_reset_blocks_another():
    a, run = make({"reset": [("timeout", "", "")]})
    with pytest.raises(ResetFailed) as exc:
        a.reset(LEASE)
    assert exc.value.left_running and run.calls[0]["kill_on_timeout"] is False
    with pytest.raises(Refused, match="still running"):
        a.reset(LEASE)
    with pytest.raises(Refused, match="still running"):
        a.release(LEASE)
    assert len(run.calls) == 1


def test_release_whose_reset_failed_raises():
    a, _ = make({"release": [(0, {"released": True,
                                  "message": "released fb9995; chips NOT marked clean"}, "")]})
    with pytest.raises(ResetFailed):
        a.release(LEASE)


def test_release_of_a_gone_lease_is_quiet():
    a, _ = make({"release": [(13, {"released": False, "message": "lease fb9995 not found"}, "")]})
    a.release(LEASE)


def test_release_refused():
    a, _ = make({"release": [(15, {"released": False, "message": "device still open"}, "")]})
    with pytest.raises(Refused, match="still open"):
        a.release(LEASE)


STATUS = {"grain": "board", "queue": [], "chips": [
    {"bdf": "0000:01:00.0", "board": "B0", "card": "p300c", "dev_index": 0,
     "state": "HELD-FOREIGN", "who": "orchard:run1", "pid": OWNER, "reason": "coder",
     "pids_holding": [777], "overstayed": False},
    {"bdf": "0000:03:00.0", "board": "B1", "card": "p300c", "dev_index": 2, "state": "FREE",
     "who": None, "pid": None, "reason": None, "pids_holding": [], "overstayed": False}]}


def test_status_reads_every_field():
    a, _ = make({"status": [(0, STATUS, "")]})
    chips = a.status()
    assert chips[0] == ChipState("0000:01:00.0", "HELD-FOREIGN", "orchard:run1", board="B0",
                                 dev_index=0, lease_pid=OWNER, pids_holding=(777,))
    assert chips[1].state == "FREE" and chips[1].who is None


def test_status_with_an_unknown_state_fails_closed():
    odd = {"chips": [dict(STATUS["chips"][0], state="WEDGED")]}
    a, _ = make({"status": [(0, odd, "")]})
    with pytest.raises(AdapterError):
        a.status()


def test_owner_pid_must_be_above_one():
    with pytest.raises(ValueError):
        GozerAdapter(owner_pid=1, run=FakeRun())


def test_the_adapter_never_forces_and_never_waits():
    refused = (15, {"reset": False, "status": "refused", "message": "device still open"}, "")
    failed = (17, {"reset": False, "status": "failed", "message": "tt-smi -r failed"}, "")
    a, run = make({"acquire": [(0, GRANT, "")], "reset": [OK_RESET, refused, failed],
                   "release": [OK_RELEASE, (15, {"released": False, "message": "still open"}, "")],
                   "status": [(0, STATUS, "")], "cancel": [(0, {"cancelled": True}, "")]})
    a.acquire(2, "w", "r", queue=True)
    a.claim("t", 2, "w", "r")
    a.status()
    a.cancel("t")
    a.reset(LEASE)
    # The refusal and failure paths are where a "--force" retry would appear, so drive them too.
    with pytest.raises(Refused):
        a.reset(LEASE)
    with pytest.raises(ResetFailed):
        a.reset(LEASE)
    a.release(LEASE)
    with pytest.raises(Refused):
        a.release(LEASE)
    assert not any("--force" in argv for argv in run.argvs())
    assert not any(argv[1] == "wait" for argv in run.argvs())
