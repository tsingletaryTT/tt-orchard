"""The read-only status command (orchard/status.py).

The run directories here are built with the real Ledger writer, so the ledger the status command
reads has the same shape and hash chain as a real one. Time is injected: a fake clock stamps each
entry, and `collect` takes `now`, so wall times are exact. Nothing here touches hardware or gozer.
"""
import hashlib
import json
import os
import time as real_time
from pathlib import Path

import pytest

from orchard import ledger as ledger_module
from orchard import status, supervisor
from orchard.ledger import Ledger

T0 = 1_791_000_000          # an arbitrary UTC second


class FakeClock:
    """Stands in for the `time` module inside orchard.ledger, so entries get chosen timestamps."""

    def __init__(self):
        self.t = T0

    def gmtime(self, secs=None):
        return real_time.gmtime(self.t if secs is None else secs)

    def __getattr__(self, name):
        return getattr(real_time, name)


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(ledger_module, "time", c)
    return c


def write_run(run_dir: Path, clock, events, *, model="org/some-model"):
    """A run directory whose ledger holds `events`: (seconds after T0, event, stage, data)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model=model, versions={}, inputs={})
        for dt, event, stage, data in events:
            clock.t = T0 + dt
            led.append(event, stage, **data)
    return run_dir


def watchdog_pause(detector="no_file_written", summary="20 model turns in a row and no file written"):
    return {"decision": "pause", "watchdog": True, "rung": "pause", "agent": "stage-agent",
            "reason": "watchdog: " + summary,
            "findings": [{"detector": detector, "summary": summary, "evidence": {}}]}


PAUSED = [
    (10, "stage_start", 0, {}),
    (100, "decision", 0, watchdog_pause()),
]
READY = [
    (10, "stage_start", 0, {}),
    (70, "stage_end", 0, {"result": "pass"}),
    (80, "decision", None, {"decision": "ready for operator review", "bundle": "stages/8/bundle"}),
]


def facts(run_dir, *, alive=False, now=T0 + 1000, free=(500.0, 500.0), gozer="", lock=None):
    """collect() with every outside signal injected. `alive=True` with no pid file means the ledger
    lock has a holder (pid 1 here), which is how an old run without a pid file looks alive."""
    if alive and lock is None and not (Path(run_dir) / "supervisor.pid").exists():
        lock = 1
    return status.collect(run_dir, now=now, pid_alive=lambda pid: alive, lock_holder=lambda p: lock,
                          disk_free_gb=lambda p: free[0] if p == Path(run_dir) else free[1],
                          gozer_status=lambda: gozer, home=Path("/somewhere/home"))


def test_a_run_dir_with_no_ledger_is_not_started(tmp_path):
    f = facts(tmp_path)
    assert f["state"] == "not-started" and f["ledger"]["ok"] is True


@pytest.mark.parametrize("events,alive,state", [
    ([(10, "stage_start", 0, {})], True, "running"),
    (PAUSED, True, "paused"),
    (READY, False, "ready-for-operator-review"),
    (READY, True, "ready-for-operator-review"),       # a run that is finishing is still finished
    ([(10, "stage_start", 0, {}), (20, "decision", None, {"decision": "abort", "by": "operator"})],
     False, "aborted"),
    ([(10, "stage_start", 0, {})], False, "stopped-or-crashed"),
    (PAUSED, False, "stopped-or-crashed"),             # paused in the ledger, but nobody is waiting
])
def test_each_run_state(tmp_path, clock, events, alive, state):
    run = write_run(tmp_path / "run", clock, events)
    assert facts(run, alive=alive)["state"] == state


def test_a_resume_after_a_pause_makes_the_run_running_again(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED + [
        (200, "decision", None, {"decision": "resume", "by": "operator"})])
    f = facts(run, alive=True)
    assert f["state"] == "running" and f["pause"] is None


def test_the_state_says_how_it_was_decided(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    (run / "supervisor.pid").write_text("4242\n")
    f = status.collect(run, now=T0 + 500, pid_alive=lambda pid: pid == 4242, lock_holder=lambda p: None,
                       disk_free_gb=lambda p: 500.0, gozer_status=lambda: "", home=Path("/h"))
    assert f["supervisor"] == {"pid": 4242, "alive": True, "source": "supervisor.pid"}
    assert "4242" in status.render(f) and "alive" in status.render(f)


def test_an_old_run_without_a_pid_file_uses_the_ledger_lock_holder(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    f = facts(run, alive=False, lock=777)          # no pid file, the lock has a holder
    assert f["supervisor"] == {"pid": 777, "alive": True, "source": "ledger lock"}
    assert f["state"] == "paused"


def test_no_pid_signal_at_all_is_said_plainly(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    f = facts(run, alive=False)
    assert f["supervisor"] == {"pid": None, "alive": False, "source": "none"}
    assert "no supervisor pid" in status.render(f)


def test_the_pause_reason_and_detector(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    f = facts(run, alive=True)
    assert f["pause"]["detector"] == "no_file_written"
    assert "20 model turns in a row and no file written" in f["pause"]["reason"]
    text = status.render(f)
    assert "no_file_written" in text and "20 model turns" in text


@pytest.mark.parametrize("reason,kind", [
    ("operator", "operator"),
    ("stage 1 failed after escalation: reference.json is missing", "stage_failed"),
    ("blocked: the coder did not answer", "blocked"),
    ("3 escalations since the last resume (cap 3)", "budget"),
    ("stage 0 found a full port (new model code is needed); plan 4 runs weights-only", "full_port"),
])
def test_pause_kinds_without_a_detector(tmp_path, clock, reason, kind):
    run = write_run(tmp_path / "run", clock, [
        (10, "stage_start", 1, {}), (20, "decision", 1, {"decision": "pause", "reason": reason})])
    assert facts(run, alive=True)["pause"]["kind"] == kind


def test_stage_rows_show_status_and_wall_time(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, [
        (0, "stage_start", 0, {}), (60, "stage_end", 0, {"result": "pass"}),
        (70, "stage_start", 1, {}), (90, "stage_end", 1, {"result": "escalate", "reasons": ["x"]}),
        (100, "stage_start", 1, {}), (160, "stage_end", 1, {"result": "fail", "reasons": ["y"]}),
        (170, "stage_end", 3, {"result": "skipped"}),
        (180, "stage_start", 4, {}),
    ])
    rows = {r["stage"]: r for r in facts(run, alive=True, now=T0 + 480)["stages"]}
    assert (rows[0]["status"], rows[0]["wall_s"]) == ("pass", 60)
    assert (rows[1]["status"], rows[1]["wall_s"], rows[1]["attempts"]) == ("fail", 80, 2)
    assert rows[3]["status"] == "skipped"
    assert (rows[4]["status"], rows[4]["wall_s"]) == ("running", 300)     # now minus its start


def test_current_stage_and_attempt(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, [
        (0, "stage_start", 2, {}), (5, "stage_end", 2, {"result": "escalate", "reasons": []}),
        (6, "stage_start", 2, {})])
    f = facts(run, alive=True)
    assert f["stage"]["current"] == 2 and f["stage"]["attempt"] == 2


def test_counts(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, [
        (1, "stage_start", 0, {}),
        (2, "retry", 0, {"watchdog": True, "rung": "nudge", "message": "m"}),
        (3, "retry", 0, {"watchdog": True, "rung": "nudge", "message": "m"}),
        (4, "escalate", 0, {"watchdog": True, "rung": "escalate"}),
        (5, "decision", 0, watchdog_pause()),
        (6, "decision", None, {"decision": "resume", "by": "operator"}),
        (7, "decision", 0, {"decision": "pause", "reason": "operator"}),
    ])
    (run / "control.done-1").write_text("resume\n")
    (run / "control.done-2").write_text("pause\n")
    c = facts(run, alive=True)["counts"]
    assert c == {"retries": 2, "escalations": 1, "nudges": 2, "pauses": 2, "operator_commands": 2}


def test_the_last_five_events(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, [
        (i, "notice", 0, {"what": f"thing {i}"}) for i in range(1, 10)])
    ev = facts(run, alive=True)["last_events"]
    assert [e["seq"] for e in ev] == [6, 7, 8, 9, 10]
    assert "thing 9" in ev[-1]["summary"] and ev[-1]["event"] == "notice"


def test_a_pending_control_word_is_reported(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    (run / "control").write_text("resume\n")
    f = facts(run, alive=True)
    assert f["control_pending"] == "resume" and "control: resume" in status.render(f)


def test_disk_and_lease_lines(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    gozer = ("grain: board   (2 boards, 4 chips)\n"
             "board 0000046131924062  (p300c)\n"
             "  chip 0  0000:01:00.0  HELD-FOREIGN     claude:x pid 12\n"
             "  chip 1  0000:02:00.0  HELD-FOREIGN     claude:x pid 12\n"
             "board 0000046131924055  (p300c)\n"
             "  chip 2  0000:03:00.0  FREE\n"
             "  chip 3  0000:04:00.0  FREE\n")
    f = facts(run, alive=True, free=(66.2, 131.0), gozer=gozer)
    assert f["disk"] == {"run_dir_free_gb": 66.2, "home_free_gb": 131.0}
    assert len(f["leases"]) == 2 and "HELD-FOREIGN" in f["leases"][0] and "FREE" in f["leases"][1]
    text = status.render(f)
    assert "66.2 GB" in text and "board 0000046131924055" in text


def test_a_missing_or_failing_gozer_is_skipped_silently(tmp_path, clock, monkeypatch):
    run = write_run(tmp_path / "run", clock, PAUSED)
    assert facts(run, alive=True, gozer="")["leases"] == []
    monkeypatch.setenv("PATH", str(tmp_path))                  # no gozer here
    assert status._gozer_status() == ""


# ---- the hint table ----------------------------------------------------------------------------

def hint(**over):
    base = {"state": "running", "pause": None, "disk": {"run_dir_free_gb": 500.0},
            "control_pending": None, "pauses_at_stage": 0, "quiet_s": 60.0}
    base.update(over)
    return status.hint_for(base)


def pause(kind, detector=None, stage=0):
    return {"kind": kind, "detector": detector, "reason": "r", "stage": stage}


@pytest.mark.parametrize("over,words", [
    ({"state": "aborted"}, "terminal, a human decides"),
    ({"state": "ready-for-operator-review"}, "never publish"),
    ({"state": "not-started"}, "run script"),
    ({"state": "stopped-or-crashed"}, "do not abort"),
    ({"state": "paused", "pause": pause("watchdog", "no_file_written")},
     "resume once; if it pauses again at the same stage, stop and ask a human"),
    ({"state": "paused", "pause": pause("watchdog", "no_file_written"), "pauses_at_stage": 2},
     "same stage has paused 2 times"),
    ({"state": "paused", "pause": pause("full_port", None, 2)}, "ask a human"),
    ({"state": "paused", "pause": pause("blocked")}, "ask a human"),
    ({"state": "paused", "pause": pause("budget")}, "ask a human"),
    ({"state": "paused", "pause": pause("operator")}, "ask a human"),
    ({"state": "paused", "pause": pause("stage_failed")}, "resume once"),
    ({"state": "paused", "pause": pause("watchdog", "no_file_written"), "control_pending": "resume"},
     "wait"),
    ({"state": "running"}, "sleep 300"),
    ({"state": "running", "quiet_s": 6 * 3600.0}, "ask a human"),
    ({"state": "running", "disk": {"run_dir_free_gb": 39.0}}, "stop and ask a human"),
    ({"state": "stopped-or-crashed", "disk": {"run_dir_free_gb": 10.0}}, "stop and ask a human"),
])
def test_hint_table(over, words):
    assert words in hint(**over)


def test_disk_does_not_override_a_terminal_state():
    assert "never publish" in hint(state="ready-for-operator-review", disk={"run_dir_free_gb": 1.0})


def test_hint_is_table_driven():
    # The printer must get its hint from the table, so a new rule is one new row.
    assert isinstance(status.HINT_RULES, tuple) and len(status.HINT_RULES) >= 10
    assert all(len(r) == 3 and callable(r[1]) for r in status.HINT_RULES)


def test_the_run_state_line_ends_with_the_hint(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    assert status.render(facts(run, alive=True)).splitlines()[-1].startswith("next: ")


# ---- JSON, text size, corruption, read-only ----------------------------------------------------

JSON_KEYS = {"ledger", "model", "run_dir", "run_name", "state", "supervisor", "stage", "stages", "counts",
             "pause", "blocked", "last_events", "control_pending", "disk", "leases", "hint", "now"}


def test_json_keys_are_stable(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED, model="org/m")
    assert status.main(["--run-dir", str(run), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == JSON_KEYS and out["model"] == "org/m" and out["ledger"]["ok"] is True


def test_the_text_block_says_ledger_ok_and_names_the_model(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED, model="org/m")
    assert status.main(["--run-dir", str(run)]) == 0
    out = capsys.readouterr().out
    assert "ledger: ok" in out and "org/m" in out and "\x1b[" not in out


def test_a_long_run_still_prints_under_40_lines(tmp_path, clock):
    events = []
    for s in range(9):
        events += [(s * 100, "stage_start", s, {}), (s * 100 + 50, "stage_end", s, {"result": "pass"})]
    events += [(2000 + i, "notice", 0, {"what": "x" * 500 + "\nsecond line"}) for i in range(300)]
    events += [(3000, "decision", 8, {"decision": "pause", "reason": "y" * 2000})]
    run = write_run(tmp_path / "run", clock, events)
    gozer = "".join(f"board B{i}  (p300c)\n  chip {i}  0000:0{i}:00.0  FREE\n" for i in range(2))
    text = status.render(facts(run, alive=True, gozer=gozer))
    assert len(text.splitlines()) < 40
    assert all(len(line) < 400 for line in text.splitlines())


def test_a_corrupt_ledger_exits_2_with_a_clear_line(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED)
    lines = (run / "ledger.jsonl").read_text().splitlines()
    lines[1] = lines[1].replace("stage_start", "stage_end")      # an edit in the middle breaks the chain
    (run / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    assert status.main(["--run-dir", str(run)]) == 2
    out = capsys.readouterr().out
    assert out.startswith("ledger: CORRUPT") and "hash chain broken" in out and "ask a human" in out


def test_a_corrupt_ledger_in_json_mode(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED)
    (run / "ledger.jsonl").write_text("not json\n")
    assert status.main(["--run-dir", str(run), "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ledger"]["ok"] is False and "not JSON" in out["ledger"]["error"]


def test_a_bad_run_dir_exits_2(tmp_path, capsys):
    assert status.main(["--run-dir", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err


def snapshot(run: Path):
    out = {}
    for p in sorted(run.rglob("*")):
        st = p.stat()
        out[str(p)] = (st.st_mtime_ns, st.st_size,
                       hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)
    return out


def test_status_changes_nothing_in_the_run_dir(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    (run / "control").write_text("resume\n")
    before = snapshot(run)
    assert status.main(["--run-dir", str(run)]) == 0
    assert snapshot(run) == before


def test_status_works_while_a_writer_holds_the_ledger_lock(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED)
    with Ledger(run / "ledger.jsonl") as writer:            # holds the flock for the whole block
        before = snapshot(run)
        assert status.main(["--run-dir", str(run), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["ledger"]["ok"] is True
        writer.append("notice", 0, what="still writable")   # the writer was not disturbed
        assert snapshot(run) != before                      # only the writer's own append changed it


def test_status_never_creates_the_lock_file(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    (run / "ledger.jsonl.lock").unlink()
    status.main(["--run-dir", str(run)])
    assert not (run / "ledger.jsonl.lock").exists()


def test_an_unfinished_last_line_is_ignored_not_repaired(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, PAUSED)
    path = run / "ledger.jsonl"
    path.write_bytes(path.read_bytes() + b'{"seq": 99, "half')      # a writer mid-append
    before = snapshot(run)
    f = facts(run, alive=True)
    assert f["ledger"]["ok"] is True and f["state"] == "paused"
    assert snapshot(run) == before                                  # no .torn sidecar, no cut


# ---- the pid file the supervisor now writes, and the CLI wiring --------------------------------

def test_the_supervisor_writes_its_pid_at_start(tmp_path):
    supervisor.write_pid_file(tmp_path)
    assert (tmp_path / "supervisor.pid").read_text().strip() == str(os.getpid())
    assert status._read_pid_file(tmp_path) == os.getpid()


def test_the_status_subcommand_is_wired_into_the_supervisor_cli(tmp_path, clock, capsys):
    run = write_run(tmp_path / "run", clock, PAUSED)
    assert supervisor.main(["status", "--run-dir", str(run), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] in ("paused", "stopped-or-crashed")
    args = supervisor.parse(["status", "--run-dir", "x", "--json"])
    assert args.cmd == "status" and args.json is True


def test_a_live_pid_check_sees_this_process_and_a_dead_one():
    assert status._pid_alive(os.getpid()) is True
    assert status._pid_alive(2 ** 22 + 12345) is False


def test_the_lock_holder_is_found_from_proc_locks_for_a_real_ledger_lock(tmp_path, clock):
    # Not injected: a real Ledger writer in this process holds the flock, and /proc/locks names us.
    run = write_run(tmp_path / "run", clock, PAUSED)
    assert status._lock_holder(run) is None            # nobody holds it after write_run closed
    with Ledger(run / "ledger.jsonl"):
        assert status._lock_holder(run) == os.getpid()


# ---- the blocked state (an unattended run that named why it stopped) -----------------------------------

BLOCKED = [
    (10, "stage_start", 0, {}),
    (100, "decision", 2, {"decision": "pause", "reason": "stage 2 failed after escalation: serves is false"}),
    (110, "decision", None, {"decision": "blocked", "code": "stage-failed",
                             "reason": "stage 2 failed after escalation: serves is false"}),
]


def test_a_run_that_ended_blocked_says_so_whether_or_not_anyone_is_alive(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, BLOCKED)
    for alive in (False, True):
        f = facts(run, alive=alive)
        assert f["state"] == "blocked"
        assert f["blocked"] == {"code": "stage-failed",
                                "reason": "stage 2 failed after escalation: serves is false"}


def test_the_blocked_hint_names_the_bundle_and_says_to_stop(tmp_path, clock):
    f = facts(write_run(tmp_path / "run", clock, BLOCKED))
    assert "BLOCKED.md" in f["hint"] and "stop" in f["hint"].lower()


def test_a_retry_after_a_block_clears_it(tmp_path, clock):
    run = write_run(tmp_path / "run", clock, BLOCKED + [
        (200, "decision", None, {"decision": "resume", "by": "retry"})])
    f = facts(run, alive=True)
    assert f["state"] == "running" and f["blocked"] is None


def test_a_run_that_is_not_blocked_has_a_null_block(tmp_path, clock):
    assert facts(write_run(tmp_path / "run", clock, PAUSED), alive=True)["blocked"] is None


def test_the_text_block_prints_the_code_when_blocked(tmp_path, clock):
    text = status.render(facts(write_run(tmp_path / "run", clock, BLOCKED)))
    assert "blocked: stage-failed" in text and "state: blocked" in text
