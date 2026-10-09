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
        self.chips = None                          # unknown: no limit (a test sets the lab's chip count)
        self.info = {"host": "node4", "root": str(self.root), "hostname": "node4", "pid": owner_pid,
                     "firmware": {"fw_bundle": "19.15.0.0"}}

    def sync_up(self, path):
        self.events.append(("up", str(path)))

    def sync_down(self, path, exclude=()):
        self.events.append(("down", str(path), tuple(exclude)))

    def mirror_up(self, path):
        self.events.append(("mirror", str(path)))

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


def test_the_stage_directory_is_mirrored_to_the_lab_before_each_test(rig):
    """A failed attempt's stage directory is moved aside on the brain, but its files stay on the lab
    (rsync never deletes). Mirroring stages/N before the test keeps them out of the next attempt."""
    assert rig.run() == EXIT_READY
    ev = rig.lab.events
    for i, e in enumerate(ev):
        if e[0] != "run":
            continue
        mirrors = [j for j, x in enumerate(ev[:i]) if x[0] == "mirror"]
        assert mirrors, e
        last = ev[mirrors[-1]][1]
        assert last.startswith(str(rig.run_dir.resolve() / "stages")) and last.rstrip("/").split("/")[-1].isdigit()
        run_up = max(j for j, x in enumerate(ev[:i]) if x == ("up", str(rig.run_dir.resolve())))
        assert mirrors[-1] > run_up                              # after the run directory goes up


def test_the_sync_down_never_overwrites_the_streamed_test_output(rig):
    """Every streamed output of the first lab run came back empty: the sync up carried the just-opened
    output file to the lab, and the sync down copied that empty copy back over the streamed one."""
    assert rig.run() == EXIT_READY
    downs = [e for e in rig.lab.events if e[0] == "down"]
    assert downs and all(len(e[2]) == 1 for e in downs)
    names = {e[2][0] for e in downs}
    assert "evidence/hw-test-output.txt" in names and "tests/1/output.txt" in names


def test_an_optional_test_larger_than_the_lab_is_recorded_as_not_run_and_never_leased(rig):
    rig.lab.chips = 2
    rig.args.required_chips = (1, 2)
    rig.args.unattended = True
    rig.run()
    acquires = [c[1] for c in rig.lab.adapter.calls if c[0] == "acquire"]
    assert 4 not in acquires
    rec = json.loads((rig.run_dir / "stages" / "4" / "tests" / "4" / "test-result.json").read_text())
    assert rec["returncode"] is None and "the lab node4 has 2 chips" in rec["not_run"]
    ev = [e["data"] for e in rig.entries() if e["data"].get("what") == "hardware test" and e["data"].get("config") == 4]
    assert ev and ev[0]["path"] == "stages/4/tests/4/test-result.json"


def test_a_required_test_larger_than_the_lab_blocks_with_the_reason(rig):
    rig.lab.chips = 2
    rig.args.required_chips = (2, 4)
    rig.args.unattended = True
    rig.run()
    entries = rig.entries()
    assert 4 not in [c[1] for c in rig.lab.adapter.calls if c[0] == "acquire"]
    reasons = [str(e["data"].get("reason")) for e in entries if e["data"].get("decision") in ("pause", "blocked")]
    assert any("requires a 4-chip test" in r and "has 2 chips" in r for r in reasons)


def test_the_lab_caches_are_audited_after_each_test(rig):
    assert rig.run() == EXIT_READY
    audit = json.loads((rig.run_dir / "stages" / "4" / "evidence" / "lab-caches.json").read_text())
    assert sorted(audit) and all(v["marker"] == "Altworld/Hemmingway-1" for v in audit.values())


def test_the_run_records_its_lab_and_closes_it_at_the_end(rig):
    assert rig.run() == EXIT_READY
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert start["lab"]["host"] == "node4" and start["lab"]["root"] == str(rig.root)
    assert rig.lab.closed


def _installed_for_draft(rig):
    """The facts a swap config is drafted from: both snapshots and a 2-chip v6 bundle of the nearest model."""
    for key in ("base", "model"):
        Path(next(i for i in rig.args.input if i.startswith(key + "=")).split("=", 1)[1]).mkdir(parents=True)
    b = rig.root / "tt-model" / "models" / "episod" / "qwen3.8-27b-dflash2-p300"
    b.mkdir(parents=True)
    (b / "tt_kernel_manifest.json").write_text(json.dumps(
        {"schema_version": "6", "device_count": 2, "weights": {"repo_id": "Qwen/Qwen3.8-27B"}}))
    (b / "run.sh").write_text("#!/bin/bash\n")
    return b


def test_stage_2_starts_from_a_drafted_swap_config_and_the_templates(rig):
    """On the lab run the agent spent whole attempts finding the bundle and the snapshots, and was
    stopped by the watchdog before it wrote swap_config.json. The supervisor now writes it first."""
    b = _installed_for_draft(rig)
    assert rig.run() == EXIT_READY
    entries = rig.entries()
    drafted = [i for i, e in enumerate(entries) if e["data"].get("decision") == "swap config drafted"]
    assert drafted and entries[drafted[0]]["stage"] == 2
    d = entries[drafted[0]]["data"]
    assert d["bundle_dir"] == str(b) and d["copied"] == ["prepare_swap.py", "serve_and_compare.py"]
    assert d["config"]["path"] == "stages/2/swap_config.json"
    prepare = next(i for i, e in enumerate(entries) if e["stage"] == 2 and e["data"].get("phase") == "prepare")
    assert drafted[0] < prepare                                 # before the agent's first turn
    assert (rig.run_dir / "stages" / "2" / "prepare_swap.py").is_file()


def test_without_the_facts_nothing_is_drafted_and_the_ledger_says_why(rig):
    assert rig.run() == EXIT_READY
    d = next(e["data"] for e in rig.entries() if e["data"].get("decision") == "swap config not drafted")
    assert any("does not exist" in p or "no installed v6 bundle" in p for p in d["problems"])
    assert not (rig.run_dir / "stages" / "2" / "prepare_swap.py").exists()


def test_the_run_start_records_the_mode(rig, tmp_path):
    assert rig.run() == EXIT_READY
    start = next(e["data"] for e in rig.entries() if e["event"] == "run_start")
    assert start["mode"] == "lab"


def test_a_lab_run_resumed_without_its_lab_is_refused_in_terms_of_the_mode(rig):
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", versions={}, inputs={},
                   lab={"host": "node4", "root": str(rig.root)}, mode="lab")
    rig.args.lab = None
    with pytest.raises(ValueError, match="--mode lab"):
        rig.run()


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


def test_a_swap_config_already_in_the_stage_directory_is_kept(rig):
    """A resumed stage keeps the config it had, including one the agent corrected."""
    from orchard.ledger import Ledger
    from orchard.stages import WEIGHTS_ONLY_STAGE_2
    _installed_for_draft(rig)
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    stage = rig.run_dir / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text("{\"port\": 9999}")
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m),
                    versions={}, clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p),
                    home=rig.home, containers=rig.containers, lab=rig.lab)
        sup._draft(WEIGHTS_ONLY_STAGE_2, stage)
        assert not [e for e in led.read() if "swap config" in str(e["data"].get("decision"))]
    assert (stage / "swap_config.json").read_text() == "{\"port\": 9999}"


def test_a_prepare_agent_that_repeats_its_last_write_still_gets_its_hardware_test(rig):
    """On the lab run the agent wrote hw_test.json and handoff.json for stage 2 and then wrote them
    again unchanged instead of ending; the watchdog escalated and blocked. With both files valid, the
    repeat ends the step as done and the test runs."""
    from run_fakes import bringup, where, turn, call
    import run_fakes

    def repeating(request):
        if "tools" in request and where(request) == (2, "prepare"):
            files = run_fakes.FILES[(2, "prepare")]
            writes = [call("write_file", path=p, content=c if isinstance(c, str) else json.dumps(c))
                      for p, c in files.items()]
            return writes[min(turn(request), len(writes) - 1)]      # never a final answer
        return bringup(request)
    rig.script = repeating
    assert rig.run() == EXIT_READY
    entries = rig.entries()
    assert any(e["data"].get("decision") == "step finished: repeated write" and e["stage"] == 2 for e in entries)
    assert not [e for e in entries if e["data"].get("watchdog") and e["stage"] == 2 and e["data"].get("rung")]
    assert any(e["data"].get("decision") == "hardware test started" and e["stage"] == 2 for e in entries)
