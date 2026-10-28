# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agent hardware tool after the redesign (PR 116, round 9): the lease is held by a short-lived wrapper process (orchard/hwtool.py),
the probe starts it, waits with a hard timeout, kills its process group on any exception, and reads `gozer status` once after the call.

Three states for the tool: idle, running, stopped. The invariant, checked on the fake gozer's own record: after a call, either gozer
holds nothing of ours, or the tool is stopped and every later call is refused without asking gozer for anything.

Fakes only: a stateful gozer (tests/hwfakes.py) and a fake command. One test starts the real wrapper as a child process with a stub
`gozer` on PATH and sends it SIGTERM. No test reaches the real gozer or a chip."""
from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hwfakes import CHIPS, PID, FakeGozer, wrapped
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, WHO, make_check, attempt_dir, fake_gozer, run_attempt, scripted_hw_agent  # noqa: F401
from test_agent_hw import fake_kenv, skills_current, ttlang_skill  # noqa: F401
from fake_model import FakeModel
from test_kernelprobe import GOOD_HW_SET, KERNELS, canned, env_ready, hw_writer, probe_args  # noqa: F401


# ---- the wrapper itself, in process -----------------------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0
        return self.t


class Cmd:
    """A fake command: wait() returns `code`, or raises `exc` (a stop that reaches the wrapper while the command runs)."""

    def __init__(self, code=0, exc=None, on_wait=None):
        self.code, self.exc, self.on_wait = code, exc, on_wait
        self.argv, self.env, self.terminated, self.killed = None, None, False, False

    def spawn(self, argv, env=None):
        self.argv, self.env = list(argv), env
        return self

    def wait(self, timeout=None):
        if self.on_wait:
            self.on_wait()
        if self.exc is not None and not self.terminated:
            raise self.exc
        return self.code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def wrap(tmp_path, gz, cmd, *, wait_s=5.0, sha=None, kernel=None, lines=None):
    out = lines if lines is not None else []
    status = hwtool.run_wrapper(who=WHO, reason="why", wait_s=wait_s, result_path=tmp_path / "lease.json", command=["harness", "--x"],
                                sh=gz.sh, spawn=cmd.spawn, now=Clock(), out=out.append, pid=PID, expect_sha256=sha, kernel_file=kernel)
    return status, json.loads((tmp_path / "lease.json").read_text()), out


def test_a_free_board_is_leased_by_the_wrappers_pid_the_command_runs_under_it_and_it_is_released(tmp_path):
    gz, cmd = FakeGozer(), Cmd(0)
    status, rec, _ = wrap(tmp_path, gz, cmd)
    acquire = gz.acquires()[0]
    assert acquire[acquire.index("--owner-pid") + 1] == str(PID) and acquire[acquire.index("--who") + 1] == WHO
    assert acquire[acquire.index("--chips") + 1] == "1" and acquire[acquire.index("--exact") + 1] == "1"
    assert cmd.argv == ["harness", "--x"] and cmd.env["TT_VISIBLE_DEVICES"] == ",".join(CHIPS)
    assert gz.released() == ["L1"] and gz.clean() and status == 0
    assert rec["outcome"] == "ran" and rec["release_ok"] is True and rec["detached"] is False and rec["pid"] == PID


def test_the_chip_time_is_the_wrappers_own_stamps_from_the_grant_to_the_end_of_the_release(tmp_path):
    status, rec, _ = wrap(tmp_path, FakeGozer(), Cmd(0))
    assert rec["granted_at"] > rec["started_at"] and rec["ended_at"] > rec["granted_at"]
    assert rec["phase"] == "done"


def test_a_busy_board_waits_once_on_its_ticket_and_the_granted_lease_is_released(tmp_path):
    gz, cmd = FakeGozer(busy=1), Cmd(0)
    status, rec, _ = wrap(tmp_path, gz, cmd, wait_s=4.2)
    waits = [c for c in gz.calls if c[1] == "wait"]
    assert len(waits) == 1 and len(gz.acquires()) == 1                       # one wait, no loop, no second request
    assert waits[0][waits[0].index("--timeout") + 1] == "5s" and waits[0][-1] == rec["ticket"]
    assert gz.cancelled() == [] and gz.released() == ["L1"] and gz.clean() and rec["detached"] is True and status == 0


def test_a_wait_that_ends_without_a_grant_cancels_the_ticket_and_runs_nothing(tmp_path):
    gz, cmd = FakeGozer(busy=1, wait="timeout"), Cmd(0)
    status, rec, out = wrap(tmp_path, gz, cmd)
    assert status == hwtool.WRAP_BUSY and cmd.argv is None and gz.released() == []
    assert gz.cancelled() == [rec["ticket"]] and gz.clean() and rec["outcome"] == "busy" and rec["cancel_error"] is None
    assert any("was cancelled" in line for line in out)


def test_a_ticket_that_cannot_be_cancelled_is_said_in_the_record(tmp_path):
    gz = FakeGozer(busy=1, wait="timeout", cancel="fail")
    status, rec, out = wrap(tmp_path, gz, Cmd(0))
    assert rec["cancel_error"] == "boom" and not gz.clean() and any("could not be cancelled" in line for line in out)


def test_a_ticket_gone_when_it_is_cancelled_counts_as_cancelled(tmp_path):
    status, rec, _ = wrap(tmp_path, FakeGozer(busy=1, wait="gone", cancel="gone"), Cmd(0))
    assert rec["outcome"] == "busy" and rec["cancel_error"] is None


def test_a_refused_acquire_runs_nothing_and_releases_nothing(tmp_path):
    gz, cmd = FakeGozer(acquire_fail=True), Cmd(0)
    status, rec, _ = wrap(tmp_path, gz, cmd)
    assert status == hwtool.WRAP_NO_LEASE and cmd.argv is None and gz.released() == [] and rec["outcome"] == "no_lease"
    assert "boom" in rec["release_text"]


def test_a_grant_with_no_device_list_is_released_and_nothing_runs(tmp_path):
    gz, cmd = FakeGozer(no_env=True), Cmd(0)
    status, rec, _ = wrap(tmp_path, gz, cmd)
    assert status == hwtool.WRAP_NO_LEASE and cmd.argv is None and gz.released() == ["L1"] and gz.clean()


def test_the_kernel_copy_is_checked_after_the_wait_and_a_changed_copy_is_not_run(tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("a = 1\n")
    import hashlib
    gz, cmd = FakeGozer(), Cmd(0)
    status, rec, _ = wrap(tmp_path, gz, cmd, sha=hashlib.sha256(b"something else\n").hexdigest(), kernel=str(kernel))
    assert status == hwtool.WRAP_NO_LEASE and cmd.argv is None and rec["outcome"] == "changed" and gz.released() == ["L1"]
    status, rec, _ = wrap(tmp_path, FakeGozer(), Cmd(0), sha=hashlib.sha256(kernel.read_bytes()).hexdigest(), kernel=str(kernel))
    assert status == 0 and rec["outcome"] == "ran"


def test_a_kernel_copy_that_became_a_link_is_not_run(tmp_path):
    import hashlib
    real = tmp_path / "real.py"
    real.write_text("a = 1\n")
    link = tmp_path / "kernel.py"
    link.symlink_to(real)
    cmd = Cmd(0)
    status, rec, _ = wrap(tmp_path, FakeGozer(), cmd, sha=hashlib.sha256(real.read_bytes()).hexdigest(), kernel=str(link))
    assert cmd.argv is None and rec["outcome"] == "changed"


def test_the_commands_exit_status_is_the_wrappers_and_a_signal_death_maps_above_128(tmp_path):
    assert wrap(tmp_path, FakeGozer(), Cmd(3))[0] == 3
    assert wrap(tmp_path, FakeGozer(), Cmd(-9))[0] == 137


@pytest.mark.parametrize("stop", [hwtool._Stop(), KeyboardInterrupt()], ids=["sigterm", "ctrl-c"])
def test_a_stop_while_the_command_runs_ends_the_command_and_still_releases(tmp_path, stop):
    gz, cmd = FakeGozer(), Cmd(0, exc=stop)
    status, rec, _ = wrap(tmp_path, gz, cmd)
    assert cmd.terminated and gz.released() == ["L1"] and gz.clean()
    assert rec["outcome"] == "interrupted" and status in (143, 130) and rec["release_ok"] is True


def test_a_stop_while_waiting_for_the_board_cancels_the_ticket(tmp_path):
    gz = FakeGozer(busy=1)
    original = gz.sh

    def sh(argv, timeout):
        if argv[1] == "wait":
            raise hwtool._Stop()
        return original(argv, timeout)
    gz.sh = sh
    status, rec, _ = wrap(tmp_path, gz, Cmd(0))
    assert gz.cancelled() == [rec["ticket"]] and gz.clean() and status == 143 and rec["outcome"] == "interrupted"


def test_the_cleanup_ignores_a_second_sigterm_so_the_release_is_not_cut_short(tmp_path):
    gz = FakeGozer()
    original, seen = gz.sh, []
    known = lambda signum, frame: None                                       # a known disposition, not one an earlier test left
    signal.signal(signal.SIGTERM, known)
    try:
        _ignored_during_release(tmp_path, gz, original, seen, known)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _ignored_during_release(tmp_path, gz, original, seen, known):

    def sh(argv, timeout):
        if argv[1] == "release":
            seen.append(signal.getsignal(signal.SIGTERM))
        return original(argv, timeout)
    gz.sh = sh
    wrap(tmp_path, gz, Cmd(0))
    assert seen == [signal.SIG_IGN] and signal.getsignal(signal.SIGTERM) == known       # ignored during the release, restored after


def test_a_release_that_resets_nothing_or_is_refused_is_said_in_the_record(tmp_path):
    _, rec, _ = wrap(tmp_path, FakeGozer(release="reset_failed"), Cmd(0))
    assert rec["release_ok"] is False and "reset did not succeed" in rec["release_text"]
    gz = FakeGozer(release="refused")
    _, rec, _ = wrap(tmp_path, gz, Cmd(0))
    assert rec["release_ok"] is False and "device still open" in rec["release_text"] and not gz.clean()


def test_a_wait_prints_a_new_line_every_heartbeat_so_the_stall_watcher_sees_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(hwtool, "HEARTBEAT_S", 0.01)
    gz = FakeGozer(busy=1)
    original = gz.sh

    def sh(argv, timeout):
        if argv[1] == "wait":
            time.sleep(0.12)
        return original(argv, timeout)
    gz.sh = sh
    status, rec, out = wrap(tmp_path, gz, Cmd(0))
    beats = [line for line in out if line.startswith("waiting for the board")]
    assert len(beats) >= 2 and len(set(beats)) == len(beats)                     # never the same line twice (a repeat is a hang)


def test_the_wrapper_record_is_written_after_the_grant_so_a_killed_wrapper_still_names_its_lease(tmp_path):
    gz = FakeGozer()
    path = tmp_path / "lease.json"
    seen = []

    def on_wait():
        seen.append(json.loads(path.read_text()))
    cmd = Cmd(0, on_wait=on_wait)
    wrap(tmp_path, gz, cmd)
    assert seen[0]["phase"] == "held" and seen[0]["lease_id"] == "L1" and seen[0]["ended_at"] is None


# ---- the real wrapper as a child process -------------------------------------------------------------------------------------------


STUB = '''#!/usr/bin/env python3
import json, os, sys
log = os.environ["STUB_LOG"]
with open(log, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
sub = sys.argv[1]
if sub == "acquire":
    print(json.dumps({"granted": True, "lease_id": "L1", "chips": ["0000:01:00.0"], "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0"}}))
elif sub == "release":
    print(json.dumps({"released": True, "message": "released"}))
'''


def test_a_real_wrapper_process_releases_its_lease_when_it_gets_sigterm(tmp_path):
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (stub_dir / "gozer").write_text(STUB)
    (stub_dir / "gozer").chmod(0o755)
    log, result = tmp_path / "stub.log", tmp_path / "lease.json"
    env = {"PATH": f"{stub_dir}:/usr/bin:/bin", "STUB_LOG": str(log)}
    proc = subprocess.Popen([sys.executable, hwtool.__file__, "run", "--who", WHO, "--reason", "r", "--wait-s", "5", "--result",
                             str(result), "--", "/bin/sleep", "60"], env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:                       # until the wrapper holds the lease and its command runs
            data = hwtool.read_wrapper_record(result) or {}
            if data.get("phase") == "held":
                break
            time.sleep(0.05)
        assert data.get("phase") == "held" and data["lease_id"] == "L1"
        time.sleep(0.3)
        os.kill(proc.pid, signal.SIGTERM)
        code = proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    lines = log.read_text().splitlines()
    assert [l.split()[0] for l in lines] == ["acquire", "release"] and "--owner-pid" in lines[0] and "L1" in lines[1]
    final = hwtool.read_wrapper_record(result)
    assert code == 143 and final["outcome"] == "interrupted" and final["release_ok"] is True and final["phase"] == "done"


# ---- the status check ---------------------------------------------------------------------------------------------------------------


def status_of(chips):
    return lambda argv, timeout: (0, json.dumps({"chips": chips}), "")


def test_nothing_held_under_our_tag_is_clean():
    chips = [{"bdf": "b1", "pid": None, "who": None}, {"bdf": "b2", "pid": 7, "who": "claude:someone-else"}]
    assert hwtool.held_by_us(WHO, {PID}, sh=status_of(chips)) is None


def test_a_chip_held_under_our_tag_is_named():
    text = hwtool.held_by_us(WHO, set(), sh=status_of([{"bdf": "b1", "pid": 9, "who": WHO}, {"bdf": "b2", "pid": 9, "who": WHO}]))
    assert "b1, b2" in text and WHO in text and "gozer status" in text


def test_a_chip_held_by_a_wrapper_pid_we_started_is_named_whatever_its_tag():
    assert "b1" in hwtool.held_by_us(WHO, {PID}, sh=status_of([{"bdf": "b1", "pid": PID, "who": "other-tag"}]))


def test_a_free_chip_with_no_pid_never_matches_a_pid():
    assert hwtool.held_by_us(WHO, {PID}, sh=status_of([{"bdf": "b1", "pid": None, "who": None}])) is None


@pytest.mark.parametrize("sh", [lambda a, t: (1, "", "boom"), lambda a, t: (0, "not json", ""), lambda a, t: (0, json.dumps({"chips": 3}), ""),
                                lambda a, t: (None, "", "timed out")], ids=["exit", "garbage", "shape", "timeout"])
def test_a_status_that_cannot_be_read_counts_as_held(sh):
    assert "cannot confirm" in hwtool.held_by_us(WHO, set(), sh=sh)


def test_a_status_runner_that_raises_counts_as_held():
    def boom(a, t):
        raise RuntimeError("no gozer")
    assert "cannot confirm" in hwtool.held_by_us(WHO, set(), sh=boom)


def test_the_check_reads_status_once_with_json_and_nothing_else():
    calls = []

    def sh(argv, timeout):
        calls.append(list(argv))
        return 0, json.dumps({"chips": []}), ""
    hwtool.held_by_us(WHO, set(), sh=sh)
    assert calls == [["gozer", "status", "--json"]]


# ---- the three states: idle, running, stopped, against the fake gozer ---------------------------------------------------------------


def inner_ok(argv, **kw):
    Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 0.5}))
    CLOCK["now"] += 40.0
    return kp.Watched(0, None, False, 40.0)


def inner_hang(argv, **kw):
    CLOCK["now"] += 66.0
    return kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)


_ONCE = {"done": False}


def inner_interrupt(argv, **kw):
    """Ctrl-C reaches the probe during the run, once (the world calms down for the next call)."""
    if _ONCE["done"]:
        return inner_ok(argv, **kw)
    _ONCE["done"] = True
    CLOCK["now"] += 12.0
    raise KeyboardInterrupt


def killed_wrapper(gz, *, after_grant=True):
    """A run_watched for a wrapper that is SIGKILLed: it took its lease (owned by its pid, now dead), wrote its record after the grant,
    and never released. gozer has not reaped the lease yet when the probe looks."""
    def run_watched(argv, **kw):
        if "hwtool.py" not in str(argv[1]):
            return inner_ok(argv, **kw)
        who = argv[argv.index("--who") + 1]
        result = Path(argv[argv.index("--result") + 1])
        if after_grant:
            code, out, _ = gz.sh(["gozer", "acquire", "--json", "--who", who, "--owner-pid", str(PID)], 1)
            lease = json.loads(out)["lease_id"]
            result.write_text(json.dumps({"pid": PID, "started_at": 1.0, "granted_at": 2.0, "ended_at": None, "lease_id": lease,
                                          "outcome": None, "phase": "held", "release_ok": None}))
        CLOCK["now"] += 20.0
        return kp.Watched(-9, None, False, 20.0)
    return run_watched


def build(name, tmp_path, fake_kenv):
    """(check, gozer) for one scenario of the table. The check is idle."""
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer(**GOZER_FOR.get(name, {}))
    inner = INNER_FOR.get(name, inner_ok)
    CLOCK["now"] = 0.0
    if name == "wrapper-killed":
        check = kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(),
                                       run_watched=killed_wrapper(gz), who=WHO, status=gz.sh, clock=lambda: CLOCK["now"],
                                       sleep=lambda s: None)
        check.gozer = gz
        return check, gz
    return make_check(adir, tmp_path, inner, gozer=gz), gz


GOZER_FOR = {
    "busy-then-grant": {"busy": 1, "wait": "grant"},
    "busy-timeout": {"busy": 1, "wait": "timeout"},
    "busy-timeout-cancel-fails": {"busy": 1, "wait": "timeout", "cancel": "fail"},
    "release-refused": {"release": "refused"},
    "release-lies": {"release": "hold"},
    "reset-failed": {"release": "reset_failed"},
    "status-unreadable": {"status_fails": True},
    "acquire-refused": {"acquire_fail": True},
}
INNER_FOR = {"hang": inner_hang, "interrupt": inner_interrupt}

# name -> (the state the tool is in afterwards, whether gozer holds nothing of ours afterwards, an optional word the answer must contain)
EXPECT = {
    "grant": ("idle", True, "PASS"),
    "busy-then-grant": ("idle", True, "PASS"),
    "busy-timeout": ("idle", True, "BUSY"),
    "busy-timeout-cancel-fails": ("stopped", False, "could not be cancelled"),
    "hang": ("idle", True, "HANG"),
    "interrupt": ("idle", True, None),
    "wrapper-killed": ("stopped", False, None),
    "release-refused": ("stopped", False, "STOPPED"),
    "release-lies": ("stopped", False, "STOPPED"),
    "reset-failed": ("idle", True, "reset"),
    "status-unreadable": ("stopped", True, "cannot confirm"),
    "acquire-refused": ("idle", True, "harness_error"),
}


def run_call(check):
    try:
        return "returned", check.call({})
    except BaseException as exc:
        return "raised", exc


@pytest.mark.parametrize("name", list(EXPECT))
def test_every_scenario_ends_idle_with_gozer_clean_or_stopped_and_refusing(name, tmp_path, fake_kenv):
    _ONCE["done"] = False
    check, gz = build(name, tmp_path, fake_kenv)
    assert check.state == "idle"
    outcome, text = run_call(check)
    want_state, want_clean, word = EXPECT[name]
    assert outcome == ("raised" if name == "interrupt" else "returned")
    if outcome == "returned" and word:
        assert word in text, text
    assert check.state == want_state
    if name != "status-unreadable" and name != "reset-failed":
        assert gz.clean() == want_clean, (gz.leases, gz.tickets)        # ground truth, not what the probe read
    # the invariant: clean, or stopped and refusing
    if check.state != "stopped":
        assert gz.clean()
        acquires = len(gz.acquires())
        outcome, text = run_call(check)                                  # an idle tool takes the next call (a hang turns it off)
        assert outcome == "returned"
        if name == "hang":
            assert text.startswith("hardware_check refused") and "hung" in text and len(gz.acquires()) == acquires
        else:
            assert len(gz.acquires()) == acquires + 1
    else:
        before = len(gz.calls)
        outcome, text = run_call(check)
        assert outcome == "returned" and text.startswith("hardware_check refused") and "stopped" in text
        assert len(gz.calls) == before                                   # the refused call asked gozer for nothing
        assert check.summary()["stopped"]


def test_a_second_call_while_one_runs_is_answered_at_once_and_the_state_is_running(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    seen = {}
    holder = {}

    def inner(argv, **kw):
        seen["state"] = holder["c"].state
        seen["second"] = holder["c"].call({})
        return inner_ok(argv, **kw)
    gz = fake_gozer()
    holder["c"] = make_check(adir, tmp_path, inner, gozer=gz)
    holder["c"].call({})
    assert seen["state"] == "running" and "already running" in seen["second"] and len(gz.acquires()) == 1


def test_the_status_is_read_exactly_once_after_each_call_that_reached_the_wrapper(tmp_path, fake_kenv):
    check, gz = build("grant", tmp_path, fake_kenv)
    check.call({})
    assert gz.subs().count("status") == 1 and gz.subs()[-1] == "status"           # after the release, once
    check.call({})
    assert gz.subs().count("status") == 2


def test_a_refused_or_blocked_call_reads_no_status(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv, "forbidden_import.py")
    gz = fake_gozer()
    check = make_check(adir, tmp_path, inner_ok, gozer=gz)
    check.call({})
    assert gz.calls == []                                                           # the static gate refused it: gozer was not asked


def test_an_interrupt_kills_nothing_here_and_reads_status_once_then_raises(tmp_path, fake_kenv):
    check, gz = build("interrupt", tmp_path, fake_kenv)
    outcome, exc = run_call(check)
    assert outcome == "raised" and isinstance(exc, KeyboardInterrupt)
    assert gz.subs().count("status") == 1 and gz.released() == ["L1"]               # the wrapper released; the probe only looked
    rec = check.summary()["calls"][0]
    assert rec["outcome"] == "interrupted" and rec["lease_upper_bound"] is True and rec["lease_s"] == hwtool.unfinished_charge_s(240.0, 12.0)


def test_an_interrupt_with_a_chip_still_held_stops_the_tool_and_the_interrupt_still_goes_on(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer(release="refused")
    check = make_check(adir, tmp_path, inner_interrupt, gozer=gz)
    with pytest.raises(KeyboardInterrupt):
        check.call({})
    assert check.state == "stopped" and "0000:01:00.0" in check.summary()["stopped"]


def test_the_probe_does_not_repair_a_stopped_tool_it_never_releases_cancels_or_acquires(tmp_path, fake_kenv):
    check, gz = build("release-refused", tmp_path, fake_kenv)
    check.call({})
    before = list(gz.calls)
    for _ in range(3):
        check.call({})
    assert gz.calls == before
    assert set(gz.subs()) <= {"acquire", "release", "status"} and gz.subs().count("release") == 1      # only the wrapper's own release


def test_the_chip_time_of_a_clean_call_is_the_wrappers_and_a_killed_wrappers_is_an_upper_bound(tmp_path, fake_kenv):
    check, _ = build("grant", tmp_path, fake_kenv)
    check.call({})
    rec = check.summary()["calls"][0]
    assert rec["lease_s"] > 0 and rec.get("lease_upper_bound") is None and rec["lease_id"] == "L1"
    check, _ = build("wrapper-killed", tmp_path / "two", fake_kenv)
    check.call({})
    rec = check.summary()["calls"][0]
    assert rec["lease_upper_bound"] is True and rec["lease_id"] == "L1"


def test_a_wrapper_that_cannot_get_a_lease_is_a_harness_error_not_a_stop(tmp_path, fake_kenv):
    check, gz = build("acquire-refused", tmp_path, fake_kenv)
    text = check.call({})
    assert "FAIL: harness_error" in text and "could not give a lease" in text and check.state == "idle"


def test_a_call_whose_kernel_changed_during_the_wait_is_not_run_and_is_not_a_stop(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer(busy=1)
    seen = []

    def inner(argv, **kw):
        seen.append(argv)
        return inner_ok(argv, **kw)
    original = gz.sh

    def sh(argv, timeout):
        if argv[1] == "wait":                                    # the agent rewrites the copy while the call waits for the board
            kernel = next(adir.parent.parent.rglob(".hwcheck/add-1-1/authoritative/kernel.py"), None) or next(
                (adir.parent.parent / ".hwcheck").rglob("kernel.py"))
            kernel.write_text("import os\n")
        return original(argv, timeout)
    gz.sh = sh
    check = make_check(adir, tmp_path, inner, gozer=gz)
    text = check.call({})
    assert seen == [] and "FAIL: harness_error" in text and "changed while" in text and check.state == "idle" and gz.clean()


# ---- the probe: exit 5, the record and the report -------------------------------------------------------------------------------------


def test_run_attempt_records_a_stopped_tool_in_the_attempt(tmp_path, fake_kenv, skills_current, ttlang_skill):
    good = (KERNELS / "add_good.py").read_text()
    gz = fake_gozer(release="refused")
    with FakeModel(scripted_hw_agent(good)) as server:
        record = run_attempt(tmp_path, fake_kenv, server, hw_writer([GOOD_HW_SET], []), gozer=gz, hw_policy=hwtool.HwPolicy())
    assert "0000:01:00.0" in record["agent_hw"]["stopped"] and record["agent_hw"]["calls"][0]["release_error"]


def test_main_ends_with_exit_5_after_an_attempt_whose_tool_stopped_and_starts_no_further_attempt(tmp_path, monkeypatch, fake_kenv, env_ready):
    seen = []

    def fake_run_attempt(*, op, n, **kw):
        seen.append(n)
        return dict(canned(op, n), agent_hw={"enabled": True, "calls": [{"k": 1}], "calls_counted": 1,
                                             "stopped": "gozer still shows chip b1 held by this probe's hardware call"})
    monkeypatch.setattr(kp, "run_attempt", fake_run_attempt)
    code = kp.main(probe_args(tmp_path, attempts=3), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True)
    assert code == kp.EXIT_BLOCKED == 5 and seen == [1]
    report = json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())
    assert "left a chip held" in report["header"]["stopped"] and len(report["attempts"]) == 1


def test_main_goes_on_when_the_tool_did_not_stop(tmp_path, monkeypatch, fake_kenv, env_ready):
    seen = []
    monkeypatch.setattr(kp, "run_attempt", lambda *, op, n, **kw: seen.append(n) or dict(
        canned(op, n), agent_hw={"enabled": True, "calls": [{"k": 1}], "calls_counted": 1}))
    assert kp.main(probe_args(tmp_path, attempts=2), env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True) == 0 and seen == [1, 2]


def test_the_report_says_a_tool_was_stopped(tmp_path):
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    md = _report_md(tmp_path, [_row_attempt(agent_hw=dict(hw_summary({}), stopped="gozer still shows chip b1 held"))])
    assert "STOPPED" in md and "gozer still shows chip b1 held" in md


def test_the_calls_file_carries_the_stop(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    gz = fake_gozer(release="refused")
    check = make_check(adir, tmp_path, inner_ok, gozer=gz)
    check.call({})
    saved = json.loads(kp.hwcheck_calls_path(adir.parent.parent, "add", 1).read_text())
    assert saved["stopped"] and "0000:01:00.0" in saved["stopped"]


def test_a_check_that_raises_stops_the_tool(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    check = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=lambda k: (None, "ab" * 32), run=lambda k, t, w: (
        {"passed": True, "lease_s": 1.0}, None, ""), check=lambda pids: (_ for _ in ()).throw(RuntimeError("x")))
    assert "STOPPED" in check.call({}) and check.state == "stopped"


@pytest.fixture(autouse=True)
def _calm_world():
    """Each test starts with the one-shot Ctrl-C of `inner_interrupt` not yet used."""
    _ONCE["done"] = False


# ---- the survivors of the first mutation table of round 9 -------------------------------------------------------------------------------


def test_an_interrupt_that_finds_a_chip_held_writes_the_stop_to_the_calls_file(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    check = make_check(adir, tmp_path, inner_interrupt, gozer=fake_gozer(release="refused"))
    with pytest.raises(KeyboardInterrupt):
        check.call({})
    saved = json.loads(kp.hwcheck_calls_path(adir.parent.parent, "add", 1).read_text())
    assert saved["stopped"] and saved["calls"][0]["outcome"] == "interrupted"


def test_the_check_after_a_call_is_given_the_pid_of_that_calls_own_wrapper():
    got = []
    check = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=lambda k: (None, "ab" * 32),
                                 run=lambda k, t, w: ({"passed": True, "lease_s": 1.0, "wrapper_pid": 777}, None, ""),
                                 check=lambda pids: got.append(set(pids)))
    check.call({})
    check.call({})
    assert got == [{777}, {777}]


def test_a_busy_call_whose_ticket_could_not_be_cancelled_says_stopped_in_its_answer(tmp_path, fake_kenv):
    check, gz = build("busy-timeout-cancel-fails", tmp_path, fake_kenv)
    text = check.call({})
    assert "BUSY" in text and "STOPPED" in text and check.state == "stopped"


def test_the_default_tag_names_the_probe_process_and_the_wrapper_gets_the_wait(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    seen = []

    def run_watched(argv, **kw):
        seen.append(list(argv))
        return kp.Watched(0, None, False, 1.0)
    gz = fake_gozer()
    check = kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(wait_s=33.0),
                                   run_watched=run_watched, status=gz.sh)
    check.call({})
    argv = seen[0]
    assert argv[argv.index("--who") + 1] == f"{kp.GOZER_WHO}:{os.getpid()}"
    assert argv[argv.index("--wait-s") + 1] == "33.0"
    assert argv[argv.index("--sha256") + 1] and Path(argv[argv.index("--file") + 1]).name == "kernel.py"
    assert argv[argv.index("--result") + 1].endswith("hw-lease.json")


def test_the_killed_wrappers_record_says_it_never_released_and_how_long_it_waited(tmp_path, fake_kenv):
    check, _ = build("wrapper-killed", tmp_path, fake_kenv)
    check.call({})
    rec = check.summary()["calls"][0]
    assert "before it released" in rec["release_error"] and rec["waited_s"] == 1.0
