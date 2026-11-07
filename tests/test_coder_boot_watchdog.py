# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 578 and 487: the wait for a coder container to boot.

(a) A container whose log stops growing while /health is down is abandoned after the stall time (issue 578; run 24 waited 4 h 11 min on a
    load that went silent at layer 20 of 48). A log that keeps growing, even slowly, is not abandoned. The absolute deadline still ends a wait
    whose log keeps growing.
(b) A container that is no longer running ends the wait at once with its exit code and the end of its log (issue 487).
(c) Abandoning a boot stops the container with `tt-model stop` and a long wait. `docker kill` runs only when that wait expires.
All commands are fakes; no docker, chip or lease is used."""
import pytest

from fakes import LEASE, FakeClock
from orchard.commands import CommandResult
from orchard.server import BootStalled, CoderExited, NotReady, ServerControl, ServerSpec

SPEC = ServerSpec("org/qwen-coder", "container", 8011, "qwen", profile="p300")
NAME = "tt-model-qwen-coder-p300"


class BootRun:
    """A docker that follows a clock. `log_at(t)` is the last log line at time t (a different value means the log grew), `state_at(t)` is
    (docker state, exit code), `health_at(t)` is the curl exit code and body. `stop_rc` is what `tt-model stop` returns; "timeout" makes it
    run out its wait. `after_stop` is the container state once a stop or a kill ran."""

    def __init__(self, clock, log_at, state_at=lambda t: ("running", 0), health_at=lambda t: (7, ""), stop_rc=0):
        self.clock, self.log_at, self.state_at, self.health_at, self.stop_rc = clock, log_at, state_at, health_at, stop_rc
        self.calls, self.stopped, self.killed = [], False, False

    def argvs(self):
        return [c["argv"] for c in self.calls]

    def __call__(self, argv, timeout, *, env=None, kill_on_timeout=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "timeout": timeout})
        t = self.clock()

        def out(rc, text="", err="", **kw):
            return CommandResult(tuple(argv), rc, text, err, **kw)
        if argv[0] == "curl":
            rc, body = self.health_at(t)
            return out(rc, body)
        if argv[:2] == ["docker", "inspect"]:
            st, code = ("exited", 137) if (self.stopped or self.killed) else self.state_at(t)
            return out(0, f"{st} {code}")
        if argv[:2] == ["docker", "logs"]:
            if "--tail" in argv and argv[argv.index("--tail") + 1] == "1":
                return out(0, self.log_at(t))
            return out(0, "line one\n" + self.log_at(t) + "\n", "a warning")
        if argv[:2] == ["tt-model", "stop"]:
            if self.stop_rc == "timeout":
                return out(None, timed_out=True)
            self.stopped = True
            return out(self.stop_rc, "stopping\n")
        if argv[:2] == ["docker", "kill"]:
            self.killed = True
            return out(0, NAME)
        if argv[:2] == ["docker", "ps"]:
            running = not (self.stopped or self.killed) and self.state_at(t)[0] == "running"
            return out(0, f"c1 img {NAME} \n" if running and "-q" not in argv else ("c1\n" if running else ""))
        if argv[:2] == ["docker", "inspect"] or argv[:2] == ["ss", "-ltn"]:
            return out(0, "State\n")
        raise AssertionError(f"unexpected call: {argv}")


def control(log_at, **kw):
    clock = FakeClock()
    stall = kw.pop("stall_s", 600.0)
    run = BootRun(clock, log_at, **kw)
    ctl = ServerControl(SPEC, run=run, clock=clock, sleep=clock.sleep, stall_s=stall)
    ctl.dev_indices = (0, 1)
    return ctl, run, clock


def frozen_after(t_stop):
    """A log that grows until `t_stop` and then says nothing more (run 24: layer 20 of 48)."""
    return lambda t: f"loading layer {int(min(t, t_stop) // 12)}"


# ---- (a) the progress watchdog and the deadline ------------------------------------------------------------------------------


def test_a_log_that_stops_growing_ends_the_wait_after_the_stall_time_with_the_log_tail():
    ctl, run, clock = control(frozen_after(240))
    with pytest.raises(BootStalled) as exc:
        ctl.wait_ready(2700)
    assert 600 <= exc.value.stalled_s < 640 and 840 <= exc.value.waited_s < 900
    assert "loading layer 20" in exc.value.log_tail and "a warning" in exc.value.log_tail
    assert clock.now < 1000                           # not the whole budget


def test_a_slow_log_that_keeps_growing_is_not_abandoned_and_the_boot_can_finish():
    # one new line every 300 s (half the stall time) for 40 minutes, then /health answers
    ctl, run, clock = control(lambda t: f"line {int(t // 300)}", health_at=lambda t: (0, "200") if t > 2400 else (7, ""))
    assert ctl.wait_ready(2700) >= 2400


def test_the_absolute_deadline_ends_a_wait_whose_log_keeps_growing():
    ctl, run, clock = control(lambda t: f"line {int(t // 5)}")
    with pytest.raises(NotReady) as exc:
        ctl.wait_ready(900)
    assert 900 <= exc.value.waited_s < 910 and "line" in exc.value.log_tail


def test_an_unreadable_log_is_no_evidence_of_a_stall():
    ctl, run, clock = control(frozen_after(0))
    real = run.__call__

    def broken_logs(argv, timeout, **kw):
        if list(argv)[:2] == ["docker", "logs"] and "1" in list(argv):
            return CommandResult(tuple(argv), 1, "", "daemon busy")
        return real(argv, timeout, **kw)
    ctl.run = broken_logs
    with pytest.raises(NotReady):                     # the deadline, not a stall
        ctl.wait_ready(1500)


def test_the_stall_check_can_be_turned_off():
    ctl, run, clock = control(frozen_after(0), stall_s=0)
    with pytest.raises(NotReady):
        ctl.wait_ready(1500)


def test_health_answering_wins_over_a_silent_log():
    ctl, run, clock = control(frozen_after(0), health_at=lambda t: (0, "200") if t >= 590 else (7, ""))
    assert ctl.wait_ready(2700) >= 590


# ---- (b) an exited container ---------------------------------------------------------------------------------------------------


def test_an_exited_container_ends_the_wait_at_once_with_its_exit_code_and_log():
    ctl, run, clock = control(lambda t: "RuntimeError: Engine core initialization failed",
                              state_at=lambda t: ("exited", 1) if t >= 50 else ("running", 0))
    with pytest.raises(CoderExited) as exc:
        ctl.wait_ready(2700)
    assert exc.value.exit_code == 1 and exc.value.status == "exited" and "Engine core" in exc.value.log_tail
    assert clock.now < 70                             # seconds, not the 43 minutes of issue 487


def test_a_container_that_is_still_starting_is_waited_for():
    ctl, run, clock = control(lambda t: f"line {int(t // 5)}", state_at=lambda t: ("created", 0) if t < 20 else ("running", 0),
                              health_at=lambda t: (0, "200") if t > 100 else (7, ""))
    assert ctl.wait_ready(2700) > 100


# ---- (c) stopping an abandoned boot -----------------------------------------------------------------------------------------------


def test_abort_stop_gives_tt_model_stop_the_long_grace_and_does_not_kill_a_stop_that_finished():
    ctl, run, clock = control(lambda t: "x")
    out = ctl.abort_stop(900)
    stops = [c for c in run.calls if c["argv"][:2] == ["tt-model", "stop"]]
    assert len(stops) == 1 and stops[0]["timeout"] == 900 and stops[0]["argv"][-2:] == ["--profile", "p300"]
    assert out["killed"] is False and not [a for a in run.argvs() if a[:2] == ["docker", "kill"]]


def test_abort_stop_kills_only_after_the_wait_expired_and_the_container_still_runs():
    ctl, run, clock = control(lambda t: "x", stop_rc="timeout")
    out = ctl.abort_stop(900)
    assert out["killed"] is True and "did not finish within 900 s" in out["kill_reason"]
    assert [a for a in run.argvs() if a[:2] == ["docker", "kill"]] == [["docker", "kill", NAME]]


def test_abort_stop_leaves_a_container_that_has_already_exited_alone():
    ctl, run, clock = control(lambda t: "x", state_at=lambda t: ("exited", 1))
    out = ctl.abort_stop(900)
    assert out["exit_code"] == 1 and not [a for a in run.argvs() if a[:2] in (["tt-model", "stop"], ["docker", "kill"])]


def test_abort_stop_does_nothing_to_a_container_another_process_owns():
    ctl, run, clock = control(lambda t: "x")
    ctl.owned = False
    assert ctl.abort_stop(900)["skipped"] is True and run.calls == []
