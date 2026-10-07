# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""release_for_idle_phase and reacquire, against the fake machine."""
import pytest

from fakes import WHO, FakeAdapter, FakeClock, FakeServer, World
from orchard.adapters import Queued, Refused, TicketGone
from orchard.defaults import GOZER_CLAIM_WINDOW_S
from orchard.handoff import Blocked, reacquire, release_for_idle_phase
from orchard.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        yield led


def decisions(ledger):
    return [e["data"]["decision"] for e in ledger.read() if e["event"] == "decision"]


def run(ledger, world, budget=600.0):
    clock = FakeClock()
    adapter = FakeAdapter(world)
    lease = reacquire(adapter, chips=2, who=WHO, reason="stage 3", ledger=ledger, stage=3,
                      wait_budget_s=budget, clock=clock, sleep=clock.sleep)
    return lease, adapter, clock


def test_a_short_idle_phase_keeps_the_lease(ledger):
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    assert release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                                  expected_idle_s=600) is False
    assert ("release", "L1") not in adapter.calls
    assert decisions(ledger) == ["hold the lease through a phase with no hardware use"]


def test_a_long_idle_phase_releases_the_lease(ledger):
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    # An image build takes 1.5 to 2.5 h (spec section 3).
    assert release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                                  expected_idle_s=5400) is True
    assert ("release", "L1") in adapter.calls and world.leases == {}


def test_a_long_idle_phase_with_a_server_still_up_keeps_the_lease_and_blocks(ledger):
    # gozer's release resets the chips, and gozer cannot see another user's container.
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    world.coder_running = True
    clock = FakeClock()
    with pytest.raises(Blocked, match="not released"):
        release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                               expected_idle_s=5400, clock=clock, sleep=clock.sleep)
    assert ("release", "L1") not in adapter.calls and "L1" in world.leases


def test_a_free_box_grants_at_once(ledger):
    lease, adapter, _ = run(ledger, World())
    assert lease.lease_id == "L1" and adapter.calls == [("acquire", 2, True)]
    granted = [e["data"] for e in ledger.read() if e["event"] == "decision"][-1]
    assert granted["waited_s"] == 0


def test_a_busy_box_waits_on_the_ticket_and_records_the_wait(ledger):
    world = World()
    world.queue_script = [Queued("t1", 2), Queued("t1", 1), Queued("t1", 1)]
    lease, adapter, clock = run(ledger, world)
    assert lease.lease_id == "L1"
    assert adapter.calls == [("acquire", 2, True), ("claim", "t1"), ("claim", "t1"), ("claim", "t1")]
    assert decisions(ledger) == ["queued for lease", "lease granted"]
    wait = [e["data"] for e in ledger.read() if e["event"] == "measurement"][-1]
    assert wait["name"] == "queue_wait_seconds" and wait["value"] == 30.0
    assert all(s * 3 <= GOZER_CLAIM_WINDOW_S for s in clock.sleeps)


def test_an_expired_ticket_is_replaced_once_and_the_lost_place_recorded(ledger):
    world = World()
    world.queue_script = [Queued("t1"), TicketGone("t1"), Queued("t2")]
    lease, adapter, _ = run(ledger, world)
    assert lease.lease_id == "L1"
    lost = [e["data"] for e in ledger.read() if e["event"] == "notice"]
    assert lost[0]["what"].startswith("queue ticket expired") and lost[0]["ticket"] == "t1"
    queued = [e["data"]["ticket"] for e in ledger.read()
              if e["event"] == "decision" and e["data"]["decision"] == "queued for lease"]
    assert queued == ["t1", "t2"]
    assert ("claim", "t2") in adapter.calls


def test_a_second_expiry_blocks(ledger):
    world = World()
    world.queue_script = [Queued("t1"), TicketGone("t1"), Queued("t2"), TicketGone("t2")]
    with pytest.raises(Blocked, match="second queue ticket expired"):
        run(ledger, world)


def test_waiting_past_the_budget_cancels_the_ticket_and_blocks(ledger):
    world = World()
    world.queue_script = [Queued("t1")] * 100
    with pytest.raises(Blocked, match="past the budget"):
        run(ledger, world, budget=25)
    adapter_calls = [c for c in ledger.read() if c["event"] == "notice"]
    assert adapter_calls[-1]["data"]["blocked"] is True


def test_the_cancel_goes_to_the_lease_tool(ledger):
    world = World()
    world.queue_script = [Queued("t1")] * 100
    adapter = FakeAdapter(world)
    clock = FakeClock()
    with pytest.raises(Blocked):
        reacquire(adapter, chips=2, who=WHO, reason="r", ledger=ledger, stage=3, wait_budget_s=25,
                  clock=clock, sleep=clock.sleep)
    assert adapter.calls[-1] == ("cancel", "t1")
    assert len([c for c in adapter.calls if c[0] == "claim"]) == 3


def test_a_refusal_blocks(ledger):
    world = World()
    world.queue_script = [Refused("the single-tenant adapter has no queue")]
    with pytest.raises(Blocked, match="refused"):
        run(ledger, world)
