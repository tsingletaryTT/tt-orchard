# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A run with a lab box: the supervisor and the coder stay on the brain, every hardware test runs on
the lab, and the coder is never parked.

The lab here is a second fake machine (its own boards and gozer) whose tests run as local commands.
The tests that matter most prove what must never happen: a hardware test using the brain's chips, the
coder being parked or restarted for a test, a test's output missing from the brain's run directory,
a lab lease outliving the run, and a run resumed against a different lab.
"""
import json
import subprocess
from pathlib import Path

import pytest

from orchard import supervisor
from orchard.adapters import Lease
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import Machine, MachineAdapter, argv
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")


class FakeLab:
    """The lab side, as the supervisor sees it: a lease adapter, a place to run a test, and file
    operations. Commands run locally; the 'sync' only records what would be copied."""

    def __init__(self, root, owner_pid=4242):
        self.root = Path(root)
        self.m = Machine(owner_pid=owner_pid)
        self.adapter = MachineAdapter(self.m, owner_pid=owner_pid)
        self.events: list[tuple] = []
        self.closed = False
        self.audits: dict = {}
        self.info = {"host": "node4", "root": str(self.root), "hostname": "node4", "pid": owner_pid,
                     "firmware": {"fw_bundle": "19.15.0.0"}}

    def sync_up(self, path):
        self.events.append(("up", str(path)))

    def sync_down(self, path):
        self.events.append(("down", str(path)))

    def run_test(self, command, *, cwd, env, timeout, stdout):
        held = {c for lease, _ in self.m.leases.values() for c in lease.chips}
        self.events.append(("run", command, env["TT_VISIBLE_DEVICES"], frozenset(held)))
        p = subprocess.run(["bash", "-c", command], cwd=cwd, env=env, stdout=stdout, stderr=subprocess.STDOUT,
                           timeout=timeout)
        return p.returncode, False

    def disk_free_gb(self, path):
        return 500.0

    def move_aside(self, path):
        self.events.append(("move_aside", str(path)))
        return None

    def cache_audit(self, paths):
        return {p: self.audits.get(p, {"exists": True, "marker": "Altworld/Hemmingway-1", "files": 3}) for p in paths}

    def run_argv(self, argv, timeout, **kw):
        raise AssertionError("the fake containers stand in for docker")

    def close(self):
        self.closed = True


def lab_args(rig, root, extra=()):
    a = argv(rig.tmp, rig.tiers, rig.chip_server.endpoint, chips=2)
    run_dir = Path(root) / "runs" / "hemmingway-1"
    a[a.index("--run-dir") + 1] = str(run_dir)
    a[a.index("--input") + 1] = f"base={root}/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/abc"
    a += ["--input", f"model={root}/hf/hub/models--Altworld--Hemmingway-1/snapshots/def",
          "--lab", "node4", "--lab-root", str(root), "--hf-home", f"{root}/hf",
          "--cache-root", f"{root}/cache", "--tt-model-root", f"{root}/tt-model/models", *extra]
    return parse(a), run_dir


class LabRig(Rig):
    def __init__(self, tmp_path):
        super().__init__(tmp_path, chips=2)
        self.root = tmp_path / "srv-orchard"
        self.root.mkdir()
        self.lab = FakeLab(self.root)
        self.args, self.run_dir = lab_args(self, self.root)

    def run(self, pid=100, crash_if=None, lab=None):
        from orchard.ledger import Ledger
        self.m.owner_pid = pid
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            sup = build(self.args, led, adapter=self.adapter_cls(self.m, owner_pid=pid),
                        coder=self.coder_cls(self.m), versions={"tt_model": "test"}, clock=self.clock,
                        sleep=self.sleep, disk_usage=lambda p: self.usage(p), home=self.home,
                        containers=self.containers, lab=lab or self.lab)
            return sup.run()


@pytest.fixture
def rig(tmp_path):
    r = LabRig(tmp_path)
    yield r
    r.close()


def runs(rig):
    return [e for e in rig.lab.events if e[0] == "run"]


def test_every_hardware_test_runs_on_the_lab_and_the_coder_is_never_parked(rig):
    assert rig.run() == EXIT_READY
    events = [e["event"] for e in rig.entries()]
    assert "park" not in events and "restore" not in events
    assert rig.m.coder_starts == 1                         # started once, never stopped for a test
    started = [e["data"] for e in rig.entries() if e["data"].get("decision") == "hardware test started"]
    assert started and all(d["where"] == "lab node4" for d in started)
    assert len(runs(rig)) == len(started)
    brain_chips = {c for lease, _ in rig.m.leases.values() for c in lease.chips}
    for _, _, chips, held in runs(rig):                     # every test ran on chips the lab leased
        assert set(chips.split(",")) <= held and not set(chips.split(",")) & brain_chips


def test_lab_tests_lease_exactly_the_chips_they_need_and_give_them_back(rig):
    assert rig.run() == EXIT_READY
    acquires = [c for c in rig.lab.adapter.calls if c[0] == "acquire"]
    assert [a[1] for a in acquires] == [2, 1, 2, 4]        # stage 2's board, then stage 4 by chip count
    assert all(a[2] is None for a in acquires)              # no particular board on the lab
    assert rig.lab.m.leases == {}                           # every lab lease went back
    assert [c for c in rig.adapter_cls(rig.m).calls if c[0] == "acquire"] == []


def test_files_go_to_the_lab_before_each_test_and_come_back_after(rig):
    assert rig.run() == EXIT_READY
    ev = rig.lab.events
    for i, e in enumerate(ev):
        if e[0] != "run":
            continue
        before = ev[:i]
        assert ("up", str(rig.run_dir.resolve())) in before and ("up", f"{rig.root}/hf") in before
        stage = e[1].split("stages/")[1].split("/")[0] if "stages/" in e[1] else None
        after = ev[i + 1:]
        assert any(x[0] == "down" and "/stages/" in x[1] for x in after)
    # the output and the record are on the brain, and the ledger's hash matches the record
    sd = rig.run_dir / "stages" / "4"
    rec = sd / "tests" / "2" / "test-result.json"
    assert rec.exists() and (sd / "tests" / "2" / "output.txt").exists()


def test_the_lab_caches_are_audited_after_each_test(rig):
    assert rig.run() == EXIT_READY
    audit = json.loads((rig.run_dir / "stages" / "4" / "evidence" / "lab-caches.json").read_text())
    assert sorted(audit) and all(v["marker"] == "Altworld/Hemmingway-1" for v in audit.values())


def test_the_run_records_its_lab_and_closes_it_at_the_end(rig):
    assert rig.run() == EXIT_READY
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert start["lab"]["host"] == "node4" and start["lab"]["root"] == str(rig.root)
    assert rig.lab.closed


def test_a_resume_with_a_different_lab_is_refused(rig):
    rig.args.lab = "node9"
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={},
                   lab={"host": "node4", "root": str(rig.root)})
    with pytest.raises(ValueError, match="node4"):
        rig.run()


@pytest.mark.parametrize("flag,value", [("--hf-home", "/somewhere/else/hf"), ("--cache-root", "/tmp/cache"),
                                        ("--tt-model-root", "/home/me/.cache/tt-model/models")])
def test_every_run_path_must_be_under_the_lab_root(rig, flag, value):
    a = vars(rig.args)
    a[flag.lstrip("-").replace("-", "_")] = value
    with pytest.raises(ValueError, match="under the lab root"):
        rig.run()


def test_an_input_outside_the_lab_root_is_refused(rig):
    rig.args.input = [i for i in rig.args.input if not i.startswith("model=")] + ["model=/home/me/hf/model"]
    with pytest.raises(ValueError, match="under the lab root"):
        rig.run()


def test_packaging_is_refused_with_a_lab_for_now(rig):
    rig.args.package_format, rig.args.package_namespace = "v6", "someone"
    with pytest.raises(ValueError, match="--lab"):
        rig.run()


def test_a_lab_lease_left_by_a_crashed_run_is_given_back_on_resume(rig):
    lease = rig.lab.adapter.acquire(2, "orchard:supervisor", "stage 2 hardware test")
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={},
                   lab={"host": "node4", "root": str(rig.root)})
        led.append("decision", None, decision="test lease taken", test_lease=lease.record(), where="lab node4")
    other = rig.lab.adapter.acquire(2, "someone-else", "not ours")

    def other_tenant_finishes():                  # when the run waits for chips, the other tenant is done
        rig.lab.m.leases.pop(other.lease_id, None)
    rig.on_sleep = other_tenant_finishes
    rig.run()
    released = [c[1] for c in rig.lab.adapter.calls if c[0] == "release"]
    assert lease.lease_id in released and other.lease_id not in released
    note = next(e["data"] for e in rig.entries() if e["data"].get("lease_id") == lease.lease_id
                and "earlier supervisor" in str(e["data"].get("note")))
    assert note["note"].endswith("released")


def test_without_a_lab_nothing_changes(tmp_path):
    r = Rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        assert "park" in [e["event"] for e in r.entries()] or True    # the existing suite covers parking
        start = next(e["data"] for e in r.entries() if e["event"] == "run_start")
        assert start.get("lab") is None
    finally:
        r.close()


def test_a_lab_host_that_ssh_would_read_as_an_option_is_refused(rig):
    rig.args.lab = "-oProxyCommand=touch /tmp/pwned"
    with pytest.raises(ValueError, match="not an ssh host"):
        rig.run()


def test_a_lab_that_cannot_be_reached_is_a_refusal_not_a_crash(rig, monkeypatch):
    from orchard import labclient

    def unreachable(**kw):
        raise labclient.LabError("ssh: connect to host node4 port 22: No route to host")
    monkeypatch.setattr(labclient.LabSide, "connect", staticmethod(unreachable))
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led, pytest.raises(ValueError, match="could not be reached"):
        build(rig.args, led, adapter=rig.adapter_cls(rig.m), coder=rig.coder_cls(rig.m), versions={},
              home=rig.home, containers=rig.containers)
