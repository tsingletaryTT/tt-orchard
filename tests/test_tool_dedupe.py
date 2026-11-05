# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tool-result dedupe in the agent loop (docs/tool-result-dedupe.md).

A coder that reads the same unchanged file again and again fills its context with text it already holds and makes no progress. Within one
agent step, `Tools` answers a repeat read of an unchanged file, and a repeat of a read-only shell command whose output is unchanged, with a
short note that names the turn that holds the text. Everything here runs against scripted fakes: no model, no chip."""
import os
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard.agent import AgentStep, Tools, agent_env
from orchard.ledger import Ledger
from orchard.watchdog import DedupeStreak, Event, Ladder, Watchdog


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    return tmp_path / "run"


def tools(run, **kw):
    return Tools(run, run / "stages" / "0", agent_env(run), **kw)


def put(run, name, text):
    p = run / "stages" / "0" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def read(t, turn_no, path="evidence/a.txt", **args):
    t.begin_turn(turn_no)
    import json
    return t.call("read_file", json.dumps({"path": path, **args}))


def shell(t, turn_no, command):
    import json
    t.begin_turn(turn_no)
    return t.call("shell", json.dumps({"command": command}))


def write(t, turn_no, path, content):
    import json
    t.begin_turn(turn_no)
    return t.call("write_file", json.dumps({"path": path, "content": content}))


def is_note(text):
    return "unchanged since turn" in text and "do not read it again" in text


# ---- read_file ----------------------------------------------------------------------------------------------------------------------

def test_a_repeat_read_of_an_unchanged_file_is_answered_with_a_note_that_names_the_turn(run):
    put(run, "evidence/a.txt", "hello\n")
    t = tools(run)
    assert read(t, 1) == "hello\n"
    note = read(t, 4)
    assert is_note(note) and "turn 1" in note and "evidence/a.txt" in note
    assert t.dedupe_count == 1 and t.last_dedupe == {"tool": "read_file", "turn": 4, "first_turn": 1, "target": "stages/0/evidence/a.txt"}


def test_the_first_read_and_a_read_of_another_file_are_never_notes(run):
    put(run, "evidence/a.txt", "one")
    put(run, "evidence/b.txt", "two")
    t = tools(run)
    assert read(t, 1) == "one" and read(t, 2, "evidence/b.txt") == "two"
    assert t.dedupe_count == 0 and t.last_dedupe is None


def test_a_file_the_agent_wrote_is_read_again(run):
    put(run, "evidence/a.txt", "old")
    t = tools(run)
    assert read(t, 1) == "old"
    assert write(t, 2, "evidence/a.txt", "new content").startswith("wrote ")
    assert read(t, 3) == "new content" and t.dedupe_count == 0


def test_a_file_changed_from_outside_is_read_again(run):
    p = put(run, "evidence/a.txt", "old text")
    t = tools(run)
    assert read(t, 1) == "old text"
    p.write_text("new text", encoding="utf-8")
    os.utime(p, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    assert read(t, 2) == "new text" and t.dedupe_count == 0


def test_a_file_replaced_by_another_inode_with_the_same_text_is_read_again(run):
    p = put(run, "evidence/a.txt", "same")
    t = tools(run)
    assert read(t, 1) == "same"
    other = put(run, "evidence/a.tmp", "same")
    os.utime(other, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns))
    os.replace(other, p)                                       # same size, same text, same times, a different inode
    assert read(t, 2) == "same" and t.dedupe_count == 0
    assert is_note(read(t, 3))                                  # and the record now holds the new file


def test_pages_of_a_long_file_are_keyed_by_offset(run):
    put(run, "evidence/a.txt", "x" * 300)
    t = tools(run, limit=100)
    first = read(t, 1)
    assert "characters 0 to 100 of 300" in first
    assert is_note(read(t, 2))                                  # the same page again
    page2 = read(t, 3, offset=100)
    assert "characters 100 to 200 of 300" in page2              # another offset is another read
    assert is_note(read(t, 4, offset=100)) and t.dedupe_count == 2


def test_a_link_to_a_file_has_the_files_key(run):
    real = put(run, "evidence/a.txt", "linked text")
    (run / "stages" / "0" / "evidence" / "alias.txt").symlink_to(real)
    t = tools(run)
    assert read(t, 1) == "linked text"
    assert is_note(read(t, 2, "evidence/alias.txt"))


def test_an_error_and_a_refusal_are_never_recorded_or_deduped(run):
    t = tools(run)
    first = read(t, 1, "evidence/missing.txt")
    assert first.startswith("error:") and read(t, 2, "evidence/missing.txt") == first
    assert read(t, 3, "../../../etc/passwd").startswith("refused:") and t.dedupe_count == 0


def test_a_new_tools_object_starts_empty_so_a_new_step_never_dedupes(run):
    put(run, "evidence/a.txt", "text")
    assert read(tools(run), 1) == "text"
    assert read(tools(run), 1) == "text"


def test_reset_forgets_every_read_for_a_conversation_that_started_again(run):
    put(run, "evidence/a.txt", "text")
    t = tools(run)
    assert read(t, 1) == "text" and is_note(read(t, 2))
    t.reset_dedupe()
    assert read(t, 3) == "text"


def test_the_switch_off_answers_every_read_in_full(run):
    put(run, "evidence/a.txt", "text")
    t = tools(run, dedupe=False)
    assert [read(t, n) for n in (1, 2, 3)] == ["text"] * 3
    assert t.dedupe_count == 0 and t.last_dedupe is None
    assert shell(t, 4, "cat stages/0/evidence/a.txt") == shell(t, 5, "cat stages/0/evidence/a.txt")


# ---- shell --------------------------------------------------------------------------------------------------------------------------

def test_a_repeat_of_a_read_only_command_with_the_same_output_is_a_note(run):
    put(run, "evidence/a.txt", "hello")
    t = tools(run)
    cmd = "cat stages/0/evidence/a.txt"
    assert shell(t, 1, cmd) == "exit 0\nhello"
    note = shell(t, 3, cmd)
    assert is_note(note) and "turn 1" in note and cmd in note
    assert t.last_dedupe == {"tool": "shell", "turn": 3, "first_turn": 1, "target": cmd}


def test_the_allowlist_covers_the_read_only_forms_and_nothing_else(run):
    from orchard.agent import read_only_command
    ok = ["ls", "ls -la stages/0", "cat stages/0/evidence/a.txt", "head -n 5 stages/0/evidence/a.txt", "tail -5 stages/0/evidence/a.txt",
          "wc -l stages/0/evidence/a.txt", "stat stages/0/evidence/a.txt", "sed -n '10,20p' stages/0/evidence/a.txt",
          "grep -n needle stages/0/evidence/a.txt", "find stages/0 -name '*.txt' -type f"]
    no = ["echo hi", "cat a.txt > b.txt", "cat a.txt | head", "ls; rm -rf x", "cat $(echo a)", "grep -r needle stages", "grep -R needle .",
          "sed -i 's/a/b/' a.txt", "sed -n '1,2p' a.txt > b", "find . -delete", "find . -exec rm {} +", "cat /etc/passwd", "cat ../outside.txt",
          "head -n 5 a.txt\nrm a.txt", "python3 -c 'print(1)'", "cd stages && ls", "tee a.txt", "ls `pwd`",
          "cat stages/*/a.txt", "ls stages/0/ev*", "cat", "grep needle", "find stages/0 -name a.txt -fprint out", "cat a.txt b.txt c.txt d.txt e.txt"]
    for c in ok:
        assert read_only_command(c, run), c
    for c in no:
        assert not read_only_command(c, run), c


def test_a_write_file_clears_the_shell_records(run):
    put(run, "evidence/a.txt", "hello")
    t = tools(run)
    cmd = "cat stages/0/evidence/a.txt"
    assert shell(t, 1, cmd) == "exit 0\nhello"
    write(t, 2, "evidence/b.txt", "other")
    assert shell(t, 3, cmd) == "exit 0\nhello" and t.dedupe_count == 0       # the output is the same, and the note is not given
    assert is_note(shell(t, 4, cmd))                                         # recorded again after the write


def test_a_command_outside_the_allowlist_clears_the_shell_records(run):
    put(run, "evidence/a.txt", "hello")
    t = tools(run)
    cmd = "cat stages/0/evidence/a.txt"
    assert shell(t, 1, cmd) == "exit 0\nhello"
    assert shell(t, 2, "echo touched > stages/0/evidence/c.txt").startswith("exit 0")
    assert shell(t, 3, cmd) == "exit 0\nhello" and t.dedupe_count == 0


def test_a_command_whose_output_changed_is_answered_in_full(run):
    put(run, "evidence/a.txt", "one")
    t = tools(run)
    assert shell(t, 1, "ls stages/0/evidence") == "exit 0\na.txt\n"
    put(run, "evidence/b.txt", "two")                                         # changed from outside: the allowlisted command shows it
    assert shell(t, 2, "ls stages/0/evidence") == "exit 0\na.txt\nb.txt\n" and t.dedupe_count == 0


def test_a_failed_read_only_command_is_never_a_note(run):
    t = tools(run)
    out = shell(t, 1, "cat stages/0/evidence/missing.txt")
    assert out.startswith("exit 1") and shell(t, 2, "cat stages/0/evidence/missing.txt") == out and t.dedupe_count == 0


# ---- the step: events, ledger, nudge, restart ------------------------------------------------------------------------------------

def step(run, ledger, endpoint, *, feed=None, dedupe=True, control=None, **kw):
    class Control:
        def __init__(self):
            self.nudges, self.stop = [], None

        def nudge(self, agent, message):
            self.nudges.append(message)

        def escalate(self, agent, stage):
            self.stop = "escalate"

        def pause(self, reason, evidence):
            self.stop = "pause"

        def take_nudges(self, agent):
            out, self.nudges = self.nudges, []
            return out

        def stop_reason(self):
            return self.stop
    control = control or Control()
    sd = run / "stages" / "0"
    events = []
    s = AgentStep(agent="stage-agent", endpoint=endpoint, model="fake", tools=Tools(run, sd, agent_env(run), dedupe=dedupe), ledger=ledger,
                  stage=0, phase="run", feed=feed or events.append, control=control, run_dir=run, evidence_dir=sd / "evidence",
                  log_path=sd / "log" / "run-1.jsonl", clock=lambda: 1000.0, **kw)
    return s, events, control


SCRIPT = [call("read_file", path="evidence/a.txt"), call("read_file", path="evidence/a.txt"), call("read_file", path="evidence/a.txt"), final("done")]


def test_a_step_that_reads_one_file_three_times_gets_two_notes_and_a_ledger_count(run):
    put(run, "evidence/a.txt", "file text")
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: SCRIPT[turn(r)]) as fm:
        s, events, _ = step(run, led, fm.endpoint)
        out = s.run("system", "user")
        tool_msgs = [m["content"] for m in fm.requests[-1]["messages"] if m["role"] == "tool"]
        rec = [e["data"] for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "transcript"][-1]
    assert out.status == "done"
    assert tool_msgs[0] == "file text" and all(is_note(m) for m in tool_msgs[1:])
    assert rec["tool_deduped"] == 2 and rec["tool_deduped_calls"] == [["read_file", 2, 1], ["read_file", 3, 1]]
    assert [e.deduped for e in events if e.kind == "tool_result"] == [False, True, True]


def test_the_ledger_record_of_a_step_with_no_repeat_has_no_dedupe_keys(run):
    put(run, "evidence/a.txt", "file text")
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: [call("read_file", path="evidence/a.txt"), final()][turn(r)]) as fm:
        step(run, led, fm.endpoint)[0].run("s", "u")
        rec = [e["data"] for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "transcript"][-1]
    assert "tool_deduped" not in rec and "tool_deduped_calls" not in rec


def test_with_the_switch_off_a_step_sends_the_file_every_time_and_records_nothing(run):
    put(run, "evidence/a.txt", "file text")
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: SCRIPT[turn(r)]) as fm:
        s, events, _ = step(run, led, fm.endpoint, dedupe=False)
        s.run("system", "user")
        tool_msgs = [m["content"] for m in fm.requests[-1]["messages"] if m["role"] == "tool"]
        rec = [e["data"] for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "transcript"][-1]
    assert tool_msgs == ["file text"] * 3 and "tool_deduped" not in rec
    assert not any(e.deduped for e in events if e.kind == "tool_result")


def test_the_bound_on_the_listed_calls(run):
    put(run, "evidence/a.txt", "t")
    t = tools(run)
    read(t, 1)
    for n in range(2, 60):
        read(t, n)
    assert t.dedupe_count == 58 and len(t.dedupe_calls) <= 12



def test_a_write_ends_the_note_even_when_the_identity_and_the_text_come_back_the_same(run, monkeypatch):
    put(run, "evidence/a.txt", "old")
    monkeypatch.setattr(Tools, "_identity", staticmethod(lambda target: (1, 1, 3, 1)))      # a coarse clock: the file looks the same to the identity check
    t = tools(run)
    assert read(t, 1) == "old"
    write(t, 2, "evidence/a.txt", "new")
    write(t, 3, "evidence/a.txt", "old")
    assert read(t, 4) == "old" and t.dedupe_count == 0                                      # the agent wrote the file since: the answer is the text


def test_a_change_that_leaves_the_identity_alone_is_still_caught_by_the_text(run, monkeypatch):
    p = put(run, "evidence/a.txt", "aaa")
    monkeypatch.setattr(Tools, "_identity", staticmethod(lambda target: (1, 1, 3, 1)))
    t = tools(run)
    assert read(t, 1) == "aaa"
    p.write_text("bbb", encoding="utf-8")                                                    # the same size, and the identity check cannot tell
    assert read(t, 2) == "bbb" and t.dedupe_count == 0


def test_a_file_that_changed_while_it_was_read_is_not_recorded(run, monkeypatch):
    put(run, "evidence/a.txt", "text")
    real = Tools._identity
    calls = []

    def flaky(target):
        calls.append(1)
        n = len(calls)
        ident = real(target)
        return (ident[0], ident[1], ident[2], ident[3] + 1) if n == 2 else ident            # the second look (after the first read) differs
    monkeypatch.setattr(Tools, "_identity", staticmethod(flaky))
    t = tools(run)
    assert read(t, 1) == "text" and read(t, 2) == "text"                                    # neither is a note: the first was never recorded
    assert is_note(read(t, 3))


def test_a_step_that_starts_again_after_capped_replies_gets_the_file_again(run):
    from test_agent_runaway_round3 import LOOP, by_request
    from fake_model import truncated
    from orchard import runaway
    put(run, "evidence/a.txt", "file text")
    answers = [call("read_file", path="evidence/a.txt"), truncated(16384, content=LOOP), truncated(16384, content=LOOP),
               call("read_file", path="evidence/a.txt"), final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(by_request(*answers)) as fm:
        s, _, _ = step(run, led, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1)
        out = s.run("system", "task")
    assert out.status == "done"
    last_tool = [m["content"] for m in fm.requests[-1]["messages"] if m["role"] == "tool"]
    assert last_tool == ["file text"]                                                        # a fresh conversation: the first read is not in it


def test_the_kernel_probe_keeps_its_own_reads(tmp_path):
    from orchard.kernelprobe import attempt_tools
    assert attempt_tools(tmp_path).dedupe is False


def test_the_supervisors_watchdog_has_the_detector(tmp_path):
    from test_supervisor import Rig
    from orchard import supervisor
    (tmp_path / "r").mkdir()
    rig = Rig(tmp_path / "r")
    seen = []
    real = supervisor.Watchdog

    def spy(detectors, ladder):
        seen.append([type(d).__name__ for d in detectors])
        return real(detectors, ladder)
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(supervisor, "Watchdog", spy)
    try:
        rig.run()
    finally:
        mp.undo()
        rig.close()
    assert seen and all("DedupeStreak" in names for names in seen)

# ---- the detector and the ladder ----------------------------------------------------------------------------------------------------

def result(turn_no, deduped, name="stages/0/evidence/a.txt", ts=1.0):
    return Event(ts=ts, agent="a", kind="tool_result", tool="read_file", output_hash="h", turn=turn_no, deduped=deduped, name=name if deduped else None)


def test_three_deduped_results_in_the_last_six_nudge_once_with_the_file_named():
    d = DedupeStreak()
    found = [d.feed(result(n, True)) for n in (1, 2, 3)]
    assert found[0] is None and found[1] is None and found[2] is not None
    f = found[2]
    assert f.detector == "dedupe_streak" and f.nudge_only and not f.pause and not f.notice_only
    assert "stages/0/evidence/a.txt" in f.nudge and "write" in f.nudge.lower()
    assert d.feed(result(4, True)) is None and d.feed(result(5, True)) is None      # once per step


def test_deduped_results_spread_wider_than_six_do_not_fire():
    d = DedupeStreak()
    seq = [True, False, False, False, False, False, True, False, False, False, False, False, True]
    assert [d.feed(result(n, flag)) for n, flag in enumerate(seq, 1)] == [None] * len(seq)


def test_a_new_step_starts_the_count_again():
    d = DedupeStreak()
    for n in (1, 2, 3):
        d.feed(result(n, True))
    assert d.feed(result(1, True)) is None and d.feed(result(2, True)) is None
    assert d.feed(result(3, True)) is not None                                        # turn numbers went back: a new step, a new nudge


class Act:
    def __init__(self):
        self.calls = []

    def nudge(self, agent, message):
        self.calls.append(("nudge", message))

    def escalate(self, agent, stage):
        self.calls.append(("escalate", stage))

    def pause(self, reason, evidence):
        self.calls.append(("pause", reason))


def test_a_nudge_only_finding_never_escalates_or_pauses(tmp_path):
    from orchard.defaults import DEDUPE_NUDGES_PER_STAGE
    with Ledger(tmp_path / "l.jsonl") as led:
        act = Act()
        ladder = Ladder(act, led, ["a"])
        wd = Watchdog([DedupeStreak()], ladder)
        wd.stage = 0
        outcomes = []
        for step_no in range(DEDUPE_NUDGES_PER_STAGE + 3):       # more steps with a streak than the hints allow
            for n in (1, 2, 3):
                wd.feed(result(n, True, ts=float(step_no)))
        kinds = [c[0] for c in act.calls]
    assert kinds.count("nudge") == DEDUPE_NUDGES_PER_STAGE and "escalate" not in kinds and "pause" not in kinds
    notices = [e for e in Ledger(tmp_path / "l.jsonl").read() if e["event"] == "notice"]
    assert notices                                                                  # the streaks past the cap are noted, not acted on


def test_the_deduped_flag_reaches_the_watchdog_as_an_event_field():
    ev = result(1, True)
    assert ev.deduped is True and Event(ts=1.0, agent="a", kind="response").deduped is False
