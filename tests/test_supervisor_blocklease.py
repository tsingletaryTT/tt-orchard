# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A block while a stage's test lease is held gives that lease back (review finding 7 of PR 11).

The pre-test reset and the fast-fail retry's reset block while the test lease is held. An unattended run ends in
`_end_blocked`, which used to release only the coder's lease, so the automatic retry (hardware-unhealthy is retried) ran
against chips a dead supervisor still held. A supervisor that starts again also gives back a local test lease an earlier
supervisor of the run took and never released, as it already did for a lab's."""
import json

import pytest

from orchard import supervisor
from orchard.adapters import Refused
from orchard.ledger import Ledger
from run_fakes import MachineAdapter, bringup, swap_entry, swap_prepare
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.args.unattended = True
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the unattended run waited for the operator"))
    yield r
    r.close()


def decisions(rig, name):
    return [d for d in rig.decisions() if d.get("decision") == name]


def test_a_refused_pre_test_reset_ends_the_run_with_the_test_lease_released(rig, monkeypatch):
    def refuse(self, lease):
        raise Refused("device still open")
    monkeypatch.setattr(MachineAdapter, "reset", refuse)
    assert rig.run() == supervisor.EXIT_BLOCKED
    blocked = decisions(rig, "blocked")
    assert blocked and blocked[-1]["code"] == "hardware-unhealthy"
    assert rig.m.leases == {}, "a lease is still held after the run ended: the automatic retry would fail on it"
    assert decisions(rig, "test lease released")


FLAKY = ("import sys\nprint('Fabric Router Sync: Timeout')\nsys.exit(4)\n")


def test_a_refused_reset_before_the_fast_fail_retry_also_releases_every_lease(rig, monkeypatch):
    refusing = {"on": False}
    real = supervisor.spawn_checked

    def spawn(command, run_dir, env, timeout, out):
        got = real(command, run_dir, env, timeout, out)
        if command.endswith("configs/4/serve_and_compare_container.py"):
            refusing["on"] = True                       # the first try of the 4-chip test has failed
        return got

    def reset(self, lease):
        if refusing["on"]:
            raise Refused("device still open")
        self.calls.append(("reset", lease.lease_id))
        self.m.resets.append(("reset", lease.lease_id))
    monkeypatch.setattr(supervisor, "spawn_checked", spawn)
    monkeypatch.setattr(MachineAdapter, "reset", reset)
    files = swap_prepare()
    files["configs/4/serve_and_compare_container.py"] = FLAKY
    finish = {"result.json": {"configs": [swap_entry(1), swap_entry(2), swap_entry(4, ok=False)]}}
    rig.script = lambda r: bringup(r, overrides={(4, "prepare"): files, (4, "finish"): finish})
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert rig.m.leases == {}
    assert not rig.m.coder_running


def test_a_local_test_lease_an_earlier_supervisor_left_is_given_back_on_start(rig):
    stale = MachineAdapter(rig.m, owner_pid=999).acquire(2, "orchard:supervisor", "stage 4 test")
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={})
        led.append("decision", 4, decision="test lease taken", config=4, test_lease=stale.record())
    rig.on_sleep = None
    rig.run()
    note = next(d for d in rig.decisions() if d.get("decision") == "test lease released"
                and d.get("lease_id") == stale.lease_id)
    assert "earlier supervisor" in note["note"] and note["note"].endswith("released")
    assert stale.lease_id not in rig.m.leases


def test_a_lease_the_ledger_shows_released_is_not_released_again(rig):
    other = MachineAdapter(rig.m, owner_pid=999).acquire(2, "someone-else", "not ours")
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={})
        led.append("decision", 4, decision="test lease taken", config=4, test_lease=other.record())
        led.append("decision", 4, decision="test lease released", lease_id=other.lease_id)
    rig.on_sleep = lambda: rig.m.leases.pop(other.lease_id, None) and None
    rig.run()
    assert not [d for d in rig.decisions() if "earlier supervisor" in str(d.get("note"))]
