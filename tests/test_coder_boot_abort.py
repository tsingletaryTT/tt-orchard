# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 578, 487 and 560 in the supervisor: what happens when the coder start or its ready wait goes wrong.

Boot abort (issues 578 and 487): a container that stalls, exits or passes the deadline is stopped, the chips are reset through the lease, the
ledger holds each step with the end of the container log and the device state, and the run blocks. When the chips read all-ones serials the
block text says so and points at the box reset; a failed release carries the device state.
Busy refusal (issue 560): `tt-model serve` refused as 'in use' with no container to explain it. The device holders and gozer's view go to the
ledger and the block text; one retry follows a wait when no holder is a container; a second refusal blocks.
Everything is a fake: no chip, lease, docker or tt-smi."""
import pytest

from fakes import FakeClock
from orchard import blocked, device_holders
from orchard.adapters import AdapterError
from orchard.server import BootStalled, CoderExited, NotReady, StopCheck
from run_fakes import MachineAdapter, MachineCoder
from test_coder_chips_held import NAME, REFUSAL, ScriptedCoder, launch, rms, serves
from test_supervisor import Rig, never_wait

ALL_ONES_ERROR = ("gozer status failed (exit 14): gozer: every Tenstorrent chip under /sys/class/tenstorrent reads an all-ones tt_serial "
                  "(the chips are not responding); nothing can be leased")
TOKEN = "hf_" + "a" * 34


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def reasons(rig):
    return [e["data"].get("reason", "") for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]


def decision_names(rig):
    return [d["decision"] for d in rig.decisions()]


class AbortCoder(MachineCoder):
    """A coder whose boot ends the way the test says (`exc`), with a stop that records what it was asked and whether it killed."""

    def __init__(self, machine, exc, *, killed=False, stops=True):
        super().__init__(machine)
        self.exc, self.killed, self.stops, self.abort_calls = exc, killed, stops, []

    def wait_ready(self, budget_s):
        raise self.exc

    def abort_stop(self, grace_s):
        self.abort_calls.append(grace_s)
        if self.stops:
            self.m.coder_running = False
        return {"how": "tt-model stop (SIGTERM first)", "killed": self.killed}


def run_with(rig, coder_factory, *, waits=False):
    """`waits`: the run may sleep (a release that polls the chips for a minute, as when they do not answer) instead of failing the test."""
    made = []
    rig.coder_cls = lambda machine: made.append(coder_factory(machine)) or made[0]
    rig.args.unattended = True
    if not waits:
        never_wait(rig)
    return made


# ---- the boot abort ---------------------------------------------------------------------------------------------------------------


def test_a_stalled_boot_stops_the_container_with_the_long_grace_resets_the_chips_and_blocks_with_the_log(rig):
    tail = f"loading layer 19\nloading layer 20 {TOKEN}\n"
    made = run_with(rig, lambda m: AbortCoder(m, BootStalled(900, 610, tail)))
    assert rig.run() == 5
    assert made[0].abort_calls == [900.0]                                # the long grace, not the 120 s of an ordinary stop
    names = decision_names(rig)
    assert names.index("coder boot abandoned") < names.index("coder boot abandoned: container stopped") < names.index(
        "coder boot abandoned: chips reset") < names.index("coder boot abandoned: device state")
    assert [r for r in rig.m.resets if r[0] == "reset"], "the chips were reset through the lease"
    entry = next(d for d in rig.decisions() if d["decision"] == "coder boot abandoned")
    assert entry["why"] == "stalled" and "loading layer 20" in entry["log_tail"] and TOKEN not in entry["log_tail"]
    reason = reasons(rig)[-1]
    assert reason.startswith("The coder boot stalled") and "loading layer 20" in reason and "10.2 minutes" in reason and TOKEN not in reason
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason.lower()}) == "coder-unusable"


def test_the_deadline_is_a_boot_abort_too(rig):
    made = run_with(rig, lambda m: AbortCoder(m, NotReady(2700, "", "still loading layer 30")))
    assert rig.run() == 5
    reason = reasons(rig)[-1]
    assert reason.startswith("The coder boot passed its deadline") and "still loading layer 30" in reason
    assert made[0].abort_calls == [900.0] and [r for r in rig.m.resets if r[0] == "reset"]


def test_an_exited_container_blocks_at_once_with_its_exit_code_and_a_fabric_hint(rig):
    log = "Fabric Router Sync: Timeout after 10000 ms on Device 2\nRuntimeError: Engine core initialization failed\n"
    made = run_with(rig, lambda m: AbortCoder(m, CoderExited(50, "exited", 1, log), stops=False))
    assert rig.run() == 5
    reason = reasons(rig)[-1]
    assert reason.startswith("The coder container exited with code 1") and "Engine core initialization failed" in reason
    assert "reset all four chips" in reason
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason.lower()}) == "coder-unusable"


def test_a_kill_after_the_grace_is_said_in_the_ledger_and_the_block(rig):
    made = run_with(rig, lambda m: AbortCoder(m, BootStalled(900, 610, "x"), killed=True))
    assert rig.run() == 5
    stopped = next(d for d in rig.decisions() if d["decision"] == "coder boot abandoned: container stopped")
    assert stopped["result"]["killed"] is True
    assert "docker kill" in reasons(rig)[-1] and "did not finish within 900 s" in reasons(rig)[-1]


def test_no_reset_runs_under_a_container_that_is_not_confirmed_stopped(rig):
    made = run_with(rig, lambda m: AbortCoder(m, BootStalled(900, 610, "x"), stops=False))
    rig.m.never_ready = False
    assert rig.run() == 5
    assert not [c for c in rig.m.resets if c[0] == "reset"]
    # the reset was not even tried: no entry for it and no entry for a refused one
    assert "coder boot abandoned: chips reset" not in decision_names(rig)
    assert not [e for e in rig.entries() if e["data"].get("what") == "the reset after an abandoned coder boot failed"]
    assert "not confirmed stopped, so the chips were not reset" in reasons(rig)[0]


def test_chips_that_read_all_ones_after_the_abort_are_named_with_the_way_out(rig):
    class DeadChips(MachineAdapter):
        def reset(self, lease):
            raise AdapterError("gozer reset exited 14: " + ALL_ONES_ERROR)

        def status(self):
            raise AdapterError(ALL_ONES_ERROR)
    rig.adapter_cls = DeadChips
    run_with(rig, lambda m: AbortCoder(m, BootStalled(900, 610, "loading layer 20")), waits=True)
    assert rig.run() == 5
    reason = reasons(rig)[0]
    assert "all-ones" in reason and "power cycle" in reason and "reboot" in reason
    state = next(d for d in rig.decisions() if d["decision"] == "coder boot abandoned: device state")["device_state"]
    assert state["gozer"]["all_ones"] is True
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason.lower()}) == "hardware-unhealthy"
    assert blocked.retryable("hardware-unhealthy", reason) is False          # a retry meets the same dead chips
    assert blocked.retryable("coder-unusable", "the coder boot stalled and the log had nothing") is True


def test_a_failed_release_carries_the_device_state_and_says_when_the_chips_do_not_answer(rig):
    class DeadOnRelease(MachineAdapter):
        dead = False

        def release(self, lease, keep_running=False):
            DeadOnRelease.dead = True
            raise AdapterError("gozer release exited 14: " + ALL_ONES_ERROR)

        def status(self):
            if DeadOnRelease.dead:
                raise AdapterError(ALL_ONES_ERROR)
            return super().status()
    rig.adapter_cls = DeadOnRelease
    run_with(rig, lambda m: AbortCoder(m, BootStalled(900, 610, "x")), waits=True)
    rig.run()
    failed = [e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("what") == "releasing the hardware failed"]
    assert failed and failed[0]["device_state"]["gozer"]["all_ones"] is True and "all-ones" in failed[0]["device_state"]["gozer"]["error"]


def test_the_summary_of_a_failed_release_names_the_chips_state_when_they_do_not_answer():
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    sup.chips_all_ones = True
    text = sup._release_summary(False)
    assert "do not answer" in text and "power cycle" in text and "device state" in text
    sup.chips_all_ones = False
    assert "device state" in sup._release_summary(False) and "power cycle" not in sup._release_summary(False)


# ---- the busy refusal (issue 560) ------------------------------------------------------------------------------------------------


def holders(*rows):
    return lambda: {"holders": [dict(pid=p, comm=c, devices=["/dev/tenstorrent/0"], container=k) for p, c, k in rows], "unreadable": 0, "fuser": None}


def started_ok():
    return (0, "", "", [(NAME, "running", "c2")])


def test_a_refusal_with_only_a_host_process_holder_is_retried_once_after_the_wait_and_the_run_goes_on(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders((4242, "python3", False)))
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[]), waits=True)
    assert rig.run() == 0
    assert decision_names(rig).count("retrying the coder start once: no container holds the chips") == 1 and 60.0 in rig.clock.sleeps
    seen = next(d for d in rig.decisions() if d["decision"] == "coder start refused: who holds the device")
    assert seen["holders"]["holders"][0]["pid"] == 4242 and seen["container_holder"] is False and "gozer" in seen
    assert "retrying the coder start once: no container holds the chips" in decision_names(rig)


def test_a_second_refusal_blocks_with_both_looks_and_the_pointer_to_gozer(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders((4242, "python3", False)))
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], rows=[]), waits=True)
    assert rig.run() == 5
    assert len(serves(made[0].run)) == 2                                   # one retry, not a loop
    looks = [d["look"] for d in rig.decisions() if d["decision"] == "coder start refused: who holds the device"]
    assert looks == ["first", "second"]
    reason = reasons(rig)[-1]
    assert "pid 4242 (python3, host process)" in reason and "gozer reconcile" in reason and "started again once after a wait" in reason
    assert "no container holds them" in reason and "another container holds them" not in reason
    assert blocked.block_code({"decision": "pause", "reason": "blocked: " + reason}) == "coder-unusable"
    assert blocked.retryable("coder-unusable", reason) is False


def test_a_refusal_with_nothing_visible_is_also_retried_once(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders())
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[]), waits=True)
    assert rig.run() == 0
    assert decision_names(rig).count("retrying the coder start once: no container holds the chips") == 1


def test_a_container_holder_is_not_retried(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders((77, "python3", True)))
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[]))
    assert rig.run() == 5
    assert len(serves(made[0].run)) == 1 and 60.0 not in rig.clock.sleeps
    assert "container" in reasons(rig)[-1] and "pid 77" in reasons(rig)[-1]


def test_a_container_of_the_coders_name_is_not_retried(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders())
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[(NAME, "exited", "c1")]))
    assert rig.run() == 5 and len(serves(made[0].run)) == 1


def test_the_wait_comes_from_the_config_and_zero_turns_the_retry_off(rig, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders())
    rig.args.coder_busy_retry_s = 7.0
    made = run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[]), waits=True)
    assert rig.run() == 0 and 7.0 in rig.clock.sleeps


def test_a_wait_of_zero_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(device_holders, "scan", holders())
    r = Rig(tmp_path)
    try:
        r.args.coder_busy_retry_s = 0.0
        made = run_with(r, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL), started_ok()], rows=[]))
        assert r.run() == 5 and len(serves(made[0].run)) == 1
    finally:
        r.close()


# ---- the device holder scan --------------------------------------------------------------------------------------------------------


def make_proc(tmp_path, pid, comm, links, cgroup="0::/user.slice/session-1.scope"):
    d = tmp_path / str(pid)
    (d / "fd").mkdir(parents=True)
    for i, target in enumerate(links):
        (d / "fd" / str(i)).symlink_to(target)
    (d / "comm").write_text(comm + "\n")
    (d / "cgroup").write_text(cgroup + "\n")


def test_the_scan_lists_processes_with_a_device_node_open_and_tells_containers_apart(tmp_path):
    make_proc(tmp_path, 10, "ttnn-report", ["/dev/tenstorrent/0", "/tmp/x"])
    make_proc(tmp_path, 11, "vllm", ["/dev/tenstorrent/1"], cgroup="0::/system.slice/docker-abc.scope")
    make_proc(tmp_path, 12, "bash", ["/dev/null", "/dev/tenstorrent-not"])         # not a device node of ours
    (tmp_path / "self").mkdir()
    out = device_holders.scan(proc_root=str(tmp_path), which=lambda name: None)
    assert [(h["pid"], h["comm"], h["container"]) for h in out["holders"]] == [(10, "ttnn-report", False), (11, "vllm", True)]
    assert device_holders.any_container(out) is True and out["fuser"] is None
    assert "pid 10 (ttnn-report, host process)" in device_holders.describe(out)


def test_the_scan_runs_fuser_when_it_exists_and_says_when_nothing_holds_the_device(tmp_path):
    seen = []

    def fake_run(argv, timeout, **kw):
        seen.append(argv)
        from orchard.commands import CommandResult
        return CommandResult(tuple(argv), 1, "", "")
    out = device_holders.scan(proc_root=str(tmp_path), which=lambda name: "/usr/bin/fuser", run=fake_run)
    assert out["holders"] == [] and seen == [["/usr/bin/fuser", "-v", "/dev/tenstorrent"]] and out["fuser"]["returncode"] == 1
    assert "no process visible" in device_holders.describe(out)


def test_the_scan_survives_an_unreadable_proc(tmp_path):
    out = device_holders.scan(proc_root=str(tmp_path / "missing"), which=lambda name: None)
    assert out["holders"] == [] and "cannot read" in out["error"]


def test_a_failed_release_of_a_lease_whose_container_belongs_to_another_process_carries_the_device_state_too(rig, monkeypatch):
    class DeadOnRelease(MachineAdapter):
        def release(self, lease, keep_running=False):
            raise AdapterError("gozer release exited 14: " + ALL_ONES_ERROR)

        def status(self):
            raise AdapterError(ALL_ONES_ERROR)
    monkeypatch.setattr(device_holders, "scan", holders((77, "python3", True)))
    rig.adapter_cls = DeadOnRelease
    run_with(rig, lambda m: ScriptedCoder(m, serve=[(1, "", REFUSAL)], rows=[(NAME, "running", "c1")]), waits=True)
    rig.run()
    failed = [e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("what") == "releasing the hardware failed"]
    assert failed and failed[0]["device_state"]["gozer"]["all_ones"] is True


def test_a_bus_read_that_never_returns_does_not_hang_the_device_state_report():
    import threading
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    sup.chips_all_ones = False
    sup.DEVICE_STATE_READ_S = 0.2                              # the instance overrides the class value
    gate = threading.Event()
    sup._gozer_state = lambda: {"chips": []}
    sup._off_bus = lambda held: gate.wait(30)                  # a sysfs read of a hung chip
    try:
        state = sup._device_state(("0000:01:00.0",))
    finally:
        gate.set()
    assert state["bus"] is None and "did not finish" in state["bus_error"]


@pytest.mark.parametrize("word", ["disk full", "chip reset", "gozer", "canary"])
def test_a_boot_abort_is_classed_by_its_fixed_start_not_by_words_in_the_log(word):
    text = f"blocked: the coder boot stalled: no growth. Last log lines: {word} | more"
    assert blocked.block_code({"decision": "pause", "reason": text}) == "coder-unusable"


def test_the_flags_reach_the_deadline_the_stall_check_and_the_retry_wait(rig):
    """The wiring: `build` hands the deadline to the supervisor's budgets, the stall time to the real ServerControl and the retry wait to the
    supervisor (a unit test of the arithmetic would pass with any of the three left unconnected)."""
    from orchard.ledger import Ledger
    from orchard.supervisor import build
    rig.args.coder_deadline_minutes, rig.args.coder_stall_minutes, rig.args.coder_busy_retry_s = 30.0, 4.0, 9.0
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=MachineAdapter(rig.m, owner_pid=100), coder=None, versions={"tt_model": "test"}, clock=rig.clock,
                    sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)
        assert sup.budgets.cold_boot_s == 1800.0 and sup.coder.stall_s == 240.0 and sup.busy_retry_wait_s == 9.0
    rig.args.coder_stall_minutes = 0
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=MachineAdapter(rig.m, owner_pid=100), coder=None, versions={"tt_model": "test"}, clock=rig.clock,
                    sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)
        assert sup.coder.stall_s is None


@pytest.mark.parametrize("flag,value", [("coder_deadline_minutes", 0), ("coder_deadline_minutes", -1), ("coder_stall_minutes", -1),
                                        ("coder_busy_retry_s", float("nan")), ("coder_busy_retry_s", -5)])
def test_a_wait_flag_that_is_not_sensible_is_refused(rig, flag, value):
    from orchard.ledger import Ledger
    from orchard.supervisor import build
    setattr(rig.args, flag, value)
    with Ledger(rig.run_dir / "ledger.jsonl") as led, pytest.raises(ValueError, match="coder"):
        build(rig.args, led, adapter=MachineAdapter(rig.m, owner_pid=100), coder=None, versions={"tt_model": "test"}, clock=rig.clock,
              sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)
