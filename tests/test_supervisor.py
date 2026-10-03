"""The supervisor run: escalation, pause and resume, abort, disk, the hardware test, the coder."""
import shutil

import pytest

from fake_model import FakeModel, turn
from fakes import Crash
from orchard.ledger import Ledger
from orchard.supervisor import EXIT_ABORTED, EXIT_READY, EXIT_REFUSED, Control, build, main, parse
from run_fakes import (BOARDS, FILES, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv,
                       bringup, clock, plenty, where, write_tiers)

# The hardware test and agent shells run real bash here, so stub tools come first on PATH.
pytestmark = pytest.mark.usefixtures("stub_tools")


class Stop(Exception):
    """Ends a test that would otherwise wait for the operator forever."""


class Rig:
    """A machine, two fake model servers (chip tier and CPU tier) and a run directory."""

    def __init__(self, tmp_path, chips=2):
        self.tmp, self.m = tmp_path, Machine()
        self.script = bringup
        self.chip_server = FakeModel(lambda r: self.script(r), models=["qwen-27b"])
        self.cpu_server = FakeModel(bringup, models=["cpu-model"])
        self.chip_server.__enter__()
        self.cpu_server.__enter__()
        self.tiers = write_tiers(tmp_path / "tiers.toml", self.chip_server.endpoint, self.cpu_server.endpoint)
        self.args = parse(argv(tmp_path, self.tiers, self.chip_server.endpoint, chips=chips))
        self.run_dir = tmp_path / "run"
        self.home = tmp_path / "operator-home"       # the preflight looks here, never in the real home
        self.clock, self.usage, self.on_sleep = clock(), plenty, None

    def sleep(self, s):
        self.clock.sleep(s)
        if self.on_sleep:
            self.on_sleep()

    def run(self, pid=100, crash_if=None):
        self.m.owner_pid = pid
        path = self.run_dir / "ledger.jsonl"
        with (CrashingLedger(path, crash_if=crash_if) if crash_if else Ledger(path)) as led:
            sup = build(self.args, led, adapter=MachineAdapter(self.m, owner_pid=pid),
                        coder=MachineCoder(self.m), versions={"tt_model": "test"}, clock=self.clock,
                        sleep=self.sleep, disk_usage=lambda p: self.usage(p), home=self.home)
            return sup.run()

    def entries(self):
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            return led.read()

    def decisions(self):
        return [e["data"] for e in self.entries() if e["event"] == "decision"]

    def ends(self, n):
        return [e["data"] for e in self.entries() if e["event"] == "stage_end" and e["stage"] == n]

    def close(self):
        self.chip_server.__exit__()
        self.cpu_server.__exit__()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def escalation_aware(stage, bad_files):
    """A script that fails `stage` until the context shows the stage was escalated."""
    def script(request):
        if "tools" in request and where(request)[0] == stage:
            if f"stage {stage}: escalate" not in request["messages"][1]["content"]:
                return bringup(request, overrides=bad_files)
        return bringup(request)
    return script


def test_main_writes_the_control_file(tmp_path, capsys):
    assert main(["control", "--run-dir", str(tmp_path), "pause"]) == 0
    assert (tmp_path / "control").read_text() == "pause\n"


def test_main_refuses_a_credential_before_running_anything(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + ["--env", "HF_TOKEN=hf_x", "--gozer", "/nonexistent"]
    assert main(a, home=tmp_path / "operator-home") == EXIT_REFUSED
    assert "looks like a credential" in capsys.readouterr().err
    assert not (tmp_path / "run" / "ledger.jsonl").exists()      # nothing was recorded or run


CREDENTIAL_FILES = (".cache/huggingface/token", ".config/gh/hosts.yml", ".ssh/id_ed25519",
                    ".ssh/id_rsa", ".netrc", ".docker/config.json")


def planted_home(root):
    """A fake operator home with every credential file the preflight knows, plus a public key."""
    for rel in CREDENTIAL_FILES + (".ssh/id_ed25519.pub",):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("planted")
    return root


def test_the_preflight_refuses_to_start_while_credential_files_are_visible(rig):
    rig.home = planted_home(rig.tmp / "operator-home")
    with pytest.raises(ValueError) as exc:
        rig.run()
    for rel in CREDENTIAL_FILES:
        assert str(rig.home / rel) in str(exc.value)
    assert "id_ed25519.pub" not in str(exc.value)          # a public key is not a credential
    assert "--accept-credentials-visible" in str(exc.value)
    assert rig.m.leases == {} and rig.m.coder_starts == 0
    assert not (rig.run_dir / "ledger.jsonl").exists() or rig.entries() == []


def test_main_names_each_visible_credential_file_and_exits_2(tmp_path, capsys, stub_tools):
    home = planted_home(tmp_path / "operator-home")
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + ["--gozer", "/nonexistent"]
    assert main(a, home=home) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.startswith("refused:") and all(str(home / rel) in err for rel in CREDENTIAL_FILES)
    assert stub_tools.calls() == []                 # refused before tt-model or tt-smi was asked
    assert not (tmp_path / "run" / "ledger.jsonl").exists()


def test_an_operator_who_accepts_visible_credentials_is_recorded(rig):
    rig.home = planted_home(rig.tmp / "operator-home")
    rig.args.accept_credentials_visible = True
    assert rig.run() == EXIT_READY
    accepted = [d for d in rig.decisions() if d["decision"] == "operator accepted visible credentials"]
    assert len(accepted) == 1
    assert sorted(accepted[0]["paths"]) == sorted(str(rig.home / rel) for rel in CREDENTIAL_FILES)


# ---- the run --------------------------------------------------------------------------------------

def test_a_failed_gate_escalates_once_to_the_diagnose_tier(rig):
    bad = {(2, "finish"): {"result.json": {"pcc": 0.9, "argmax_match": True,
                                           "evidence": ["stages/2/evidence/hw-test-output.txt"]}}}
    rig.script = escalation_aware(2, bad)
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(2)] == ["escalate", "pass"]
    esc = [e["data"] for e in rig.entries() if e["event"] == "escalate"]
    assert esc[0]["by"] == "stage machine" and "pcc must be" in esc[0]["reasons"][0]
    steps = [d for d in rig.decisions() if d["decision"] == "agent step" and d["escalated"]]
    assert steps and all(d["tier"] == "large" for d in steps)
    assert (rig.run_dir / "stages" / "2.partial-1" / "result.json").exists()   # kept, never deleted


def test_a_second_failure_pauses_the_run_until_the_operator_resumes(rig):
    bad = {(3, "finish"): {"result.json": {"parity": False, "top1": 0.2,
                                           "evidence": ["stages/3/evidence/hw-test-output.txt"]}}}
    rig.script = lambda r: bringup(r, overrides=bad)

    def fixed_and_resumed():
        rig.script = bringup
        Control(rig.run_dir).write("resume")
    rig.on_sleep = fixed_and_resumed
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(3)] == ["escalate", "fail", "pass"]
    pauses = [d for d in rig.decisions() if d["decision"] in ("pause", "resume")]
    assert pauses[0]["reason"].startswith("stage 3 failed after escalation")
    assert pauses[1] == {"decision": "resume", "by": "operator"}


def test_the_escalation_cap_pauses_the_run(rig):
    def bad(n):
        return {(n, "finish"): {"result.json": {"broken": True}}}

    def script(request):
        if "tools" in request and where(request)[0] in (2, 3, 4):
            n = where(request)[0]
            return escalation_aware(n, bad(n))(request)
        return bringup(request)
    rig.script = script
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    pauses = [e for e in rig.entries() if e["event"] == "decision" and e["data"]["decision"] == "pause"]
    assert [(e["stage"], e["data"]["reason"]) for e in pauses] == [
        (4, "3 escalations since the last resume (cap 3)")]   # before stage 4 runs again


def test_a_full_port_pauses_before_stage_2(rig):
    from run_fakes import DELTA
    full = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "path": "full-port"}}}
    rig.script = lambda r: bringup(r, overrides=full)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(0)] == ["pass"] and rig.ends(2) == []
    assert any(d.get("reason", "").startswith("stage 0 found a full port") for d in rig.decisions())


def test_an_operator_pause_waits_for_resume(rig):
    Control(rig.run_dir).write("pause")
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    assert {"decision": "pause", "reason": "operator"} in rig.decisions()


def test_abort_stops_the_coder_releases_its_lease_and_stays_aborted(rig):
    Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    assert not rig.m.coder_running and rig.m.leases == {}
    assert [d["decision"] for d in rig.decisions()][-3:] == ["abort", "stopping the coder",
                                                             "hardware released"]
    starts = rig.m.coder_starts
    assert rig.run() == EXIT_ABORTED and rig.m.coder_starts == starts


def test_a_stage_short_of_disk_pauses_until_the_operator_resumes(rig):
    from collections import namedtuple
    rig.usage = lambda p: namedtuple("U", "total used free")(0, 0, 2e9)

    def freed():
        rig.usage = plenty
        Control(rig.run_dir).write("resume")
    rig.on_sleep = freed
    assert rig.run() == EXIT_READY
    notes = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert notes == ["stage 1 needs 5.0 GB free on the run directory's disk; 2.0 GB is free"]


def test_a_refused_test_command_fails_the_stage_before_any_hardware_is_used(rig, stub_tools):
    # If the runner's reset rule regressed, `tt-smi -r 0` would run. It must reach a stub.
    assert shutil.which("tt-smi") == str(rig.tmp / "stub-bin" / "tt-smi")
    bad = {(2, "prepare"): {**FILES[(2, "prepare")],
                            "hw_test.json": {"command": "tt-smi -r 0", "deadline_s": 60}}}
    rig.script = escalation_aware(2, bad)
    assert rig.run() == EXIT_READY
    first = rig.ends(2)[0]
    assert first["result"] == "escalate" and "the test command is refused" in first["reasons"][0]
    started = [e for e in rig.entries() if e["event"] == "decision" and e["stage"] == 2
               and e["data"]["decision"] == "hardware test started"]
    assert len(started) == 1                    # only the escalated attempt ran a test
    assert stub_tools.calls() == []             # tt-smi never ran, not even the stub


def test_the_hardware_test_gets_the_leased_chips_and_no_token(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supervisor_secret")
    monkeypatch.setenv("GH_TOKEN", "ghp_supervisor_secret")
    assert rig.run() == EXIT_READY
    env = (rig.run_dir / "stages" / "2" / "evidence" / "devices.txt").read_text()
    assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B1'])}" in env
    # The agent shells' no-chip mask must not reach the hardware test in either variable.
    assert "TT_METAL_VISIBLE_DEVICES" not in env and "0000:ff:00.0" not in env
    assert "secret" not in env


def test_numbers_from_stage_6_become_labelled_measurements(rig):
    assert rig.run() == EXIT_READY
    got = [(e["data"]["name"], e["data"]["label"]) for e in rig.entries()
           if e["event"] == "measurement" and e["stage"] == 6]
    assert got[-2:] == [("decode", "measured"), ("ttft", "TODO")]   # after the queue-wait measurement


def test_a_relaunched_coder_with_a_different_canary_answer_blocks(rig):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    rig.m.coder_answer = "41"                   # the coder answers differently after its restart

    def stop():
        raise Stop()
    rig.on_sleep = stop
    with pytest.raises(Stop):
        rig.run(pid=200)
    d = [x["decision"] for x in rig.decisions()]
    assert "relaunch the coder under this supervisor's lease" in d
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == ["the coder's canary answer changed after it was started again"]


def test_a_coder_that_dies_is_restarted_once_and_a_second_death_blocks(rig):
    killed = set()

    def script(request):
        if "tools" in request:
            n, phase = where(request)
            if n in (1, 3) and turn(request) == 0 and n not in killed:
                killed.add(n)
                rig.m.coder_running = False         # the coder dies while the agent works
        return bringup(request)
    rig.script = script

    def stop():
        raise Stop()
    rig.on_sleep = stop
    with pytest.raises(Stop):
        rig.run()
    d = [x["decision"] for x in rig.decisions()]
    assert d.count("coder died; restarting it once") == 1
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == ["the coder died a second time since the last resume"]
    assert ("reset", "L1") in rig.m.resets        # the chips were reset before the single restart
