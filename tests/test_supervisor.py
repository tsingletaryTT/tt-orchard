"""The supervisor run: escalation, pause and resume, abort, disk, the hardware test, the coder."""
import os
import shutil
import signal

import pytest

from fake_model import FakeModel, turn
from fakes import Crash
from orchard import supervisor
from orchard.adapters import AdapterError
from orchard.ledger import Ledger, LedgerCorrupt
from orchard.supervisor import (EXIT_ABORTED, EXIT_ERROR, EXIT_READY, EXIT_REFUSED, Control, build,
                                main, parse)
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
        self.adapter_cls, self.coder_cls = MachineAdapter, MachineCoder

    def sleep(self, s):
        self.clock.sleep(s)
        if self.on_sleep:
            self.on_sleep()

    def run(self, pid=100, crash_if=None):
        self.m.owner_pid = pid
        path = self.run_dir / "ledger.jsonl"
        with (CrashingLedger(path, crash_if=crash_if) if crash_if else Ledger(path)) as led:
            sup = build(self.args, led, adapter=self.adapter_cls(self.m, owner_pid=pid),
                        coder=self.coder_cls(self.m), versions={"tt_model": "test"}, clock=self.clock,
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


# ---- signals and errors release the hardware ------------------------------------------------------

@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_mid_run_takes_the_abort_path(rig, sig):
    def script(request):
        if "tools" in request and where(request) == (1, "run") and turn(request) == 0:
            os.kill(os.getpid(), sig)               # Ctrl-C or kill while the agent works
        return bringup(request)
    rig.script = script
    before = signal.getsignal(signal.SIGINT)
    assert rig.run() == EXIT_ABORTED
    assert signal.getsignal(signal.SIGINT) is before          # the run's handlers are removed
    assert not rig.m.coder_running and rig.m.leases == {}
    decisions = rig.decisions()
    assert {"decision": "abort", "by": f"signal {sig.name}"} in decisions
    assert decisions[-1]["decision"] == "hardware released"
    starts = rig.m.coder_starts
    assert rig.run() == EXIT_ABORTED and rig.m.coder_starts == starts    # it stays aborted


def test_a_signal_during_the_hardware_test_kills_the_test_and_releases_its_lease(rig):
    pid_file = rig.run_dir / "stages" / "2" / "child.pid"
    command = ("python3 -c 'import os, signal, time; "
               f"open(\"{pid_file}\", \"w\").write(str(os.getpid())); "
               f"os.kill({os.getpid()}, signal.SIGTERM); time.sleep(30)'")
    rig.script = lambda r: bringup(r, overrides={(2, "prepare"): {
        **FILES[(2, "prepare")], "hw_test.json": {"command": command, "deadline_s": 60}}})
    assert rig.run() == EXIT_ABORTED
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)                 # the test process is gone
    assert rig.m.leases == {} and not rig.m.coder_running      # test lease and coder lease
    assert "test lease released" in [d["decision"] for d in rig.decisions()]


class StatusFailsOnce(MachineAdapter):
    """gozer status fails once, at the start of stage 2's hardware phase."""
    armed = False

    def status(self):
        if StatusFailsOnce.armed:
            StatusFailsOnce.armed = False
            raise AdapterError("gozer status exited 1")
        return super().status()


def test_an_adapter_error_mid_run_stops_the_coder_and_releases_its_lease(rig):
    def script(request):
        if "tools" in request and where(request) == (2, "prepare") and turn(request) == 0:
            StatusFailsOnce.armed = True
        return bringup(request)
    rig.script, rig.adapter_cls = script, StatusFailsOnce
    with pytest.raises(AdapterError):
        rig.run()
    assert not rig.m.coder_running and rig.m.leases == {}
    notes = [e["data"] for e in rig.entries() if e["event"] == "notice"]
    assert any(n.get("what") == "the supervisor stopped on an error" for n in notes)
    rig.script = bringup
    assert rig.run(pid=200) == EXIT_READY                       # a re-run resumes, unlike an abort


def test_a_corrupt_ledger_mid_run_still_releases_the_hardware(rig):
    def script(request):
        if "tools" in request and where(request) == (1, "run") and turn(request) == 0:
            with open(rig.run_dir / "ledger.jsonl", "a") as f:
                f.write("not json\n")             # an agent writing to the ledger with tee
        return bringup(request)
    rig.script = script
    with pytest.raises(LedgerCorrupt):
        rig.run()
    assert not rig.m.coder_running and rig.m.leases == {}


def test_a_ledger_broken_during_the_coder_boot_still_releases_the_coder(rig):
    # Before "coder started" is written, only this process knows the coder's lease.
    class BreaksTheLedger(MachineCoder):
        def start(self, lease):
            super().start(lease)
            with open(rig.run_dir / "ledger.jsonl", "a") as f:
                f.write("not json\n")

    rig.coder_cls = BreaksTheLedger
    with pytest.raises(LedgerCorrupt):
        rig.run()
    assert not rig.m.coder_running and rig.m.leases == {}


def test_a_mid_run_error_is_not_reported_as_a_refusal(tmp_path, capsys, monkeypatch):
    class Failing:
        stop_message = "The coder was stopped and the hardware released."

        def run(self):
            raise ValueError("no free board")
    monkeypatch.setattr(supervisor, "build", lambda args, ledger, home=None: Failing())
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    assert main(argv(tmp_path, tiers, "http://127.0.0.1:8000/v1"),
                home=tmp_path / "operator-home") == EXIT_ERROR
    err = capsys.readouterr().err
    assert "refused" not in err and "no free board" in err and "hardware released" in err


# ---- a crash during the coder's boot -------------------------------------------------------------

def test_a_crash_during_the_coder_boot_stops_that_container_and_re_leases(tmp_path):
    """The supervisor dies after the container started and before it answered (the 30 minute
    boot). The container outlives it on all four chips, under a lease whose pid is dead. The
    restart must find both in the ledger, stop the container and take a new lease."""
    rig = Rig(tmp_path, chips=4)
    try:
        class CrashesWhileBooting(MachineCoder):
            def start(self, lease):
                super().start(lease)
                raise Crash("killed during the boot")
        rig.coder_cls = CrashesWhileBooting
        with pytest.raises(Crash):
            rig.run()
        assert rig.m.coder_running and set(rig.m.leases) == {"L1"}
        rig.coder_cls = MachineCoder
        assert rig.run(pid=200) == EXIT_READY
        d = [x["decision"] for x in rig.decisions()]
        relaunch = d.index("relaunch the coder under this supervisor's lease")
        assert "stopping the coder" in d[relaunch:d.index("coder started")]
        started = next(x for x in rig.decisions() if x["decision"] == "coder started")
        assert started["lease"]["lease_id"] != "L1"            # a new lease, under the new pid
        assert rig.m.leases == {} and not rig.m.coder_running
    finally:
        rig.close()


# ---- a boot that never finished is finished before any stage --------------------------------------

def test_a_resume_after_a_failed_coder_boot_waits_for_ready_and_checks_the_canary(rig):
    """The review's probe: the coder never becomes ready, the run pauses, readiness returns and
    the operator resumes. The coder must answer its canary before any stage runs."""
    rig.m.never_ready = True

    def ready_and_resumed():
        rig.m.never_ready = False
        Control(rig.run_dir).write("resume")
    rig.on_sleep = ready_and_resumed
    assert rig.run() == EXIT_READY
    d = [x["decision"] for x in rig.decisions()]
    assert d.index("resume") < d.index("coder started") < d.index("agent step")
    started = next(x for x in rig.decisions() if x["decision"] == "coder started")
    assert started["canary_after"]["path"].startswith("evidence/coder-canary")


def test_a_canary_mismatch_still_blocks_after_a_resume(rig):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    rig.m.coder_answer = "41"                   # the relaunched coder answers differently
    resumed = []

    def resume_once_then_stop():
        if resumed:
            raise Stop()
        resumed.append(True)
        Control(rig.run_dir).write("resume")    # the operator resumes without fixing anything
    rig.on_sleep = resume_once_then_stop
    with pytest.raises(Stop):
        rig.run(pid=200)
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == ["the coder's canary answer changed after it was started again"] * 2
    assert rig.ends(1) == []                    # no stage ran on the unchecked coder
