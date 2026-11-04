# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 458: runaway coder output in the agent loop. A fake coder that loops, a fake coder that writes an enormous file."""
import json
import os

import pytest

from fake_model import FakeModel, call, final, truncated
from orchard import runaway
from orchard.agent import AgentStep, Tools, agent_env
from orchard.ledger import Ledger

# One paragraph repeated until the reply was cut off: the shape of the capped replies of the 2026-11-03 runs.
LOOP = ("Now I see the issue clearly. The model dictionary opens at position 5350 and never closes.\n"
        "- Position 17628: } (closes something, but not the model dictionary)\n") * 120


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


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    (tmp_path / "run" / "stages" / "0" / "port").mkdir(parents=True)
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


def step(run, ledger, endpoint, **kw):
    sd = run / "stages" / "0"
    return AgentStep(agent="stage-agent", endpoint=endpoint, model="fake",
                     tools=Tools(run, sd, agent_env(run)), ledger=ledger, stage=0, phase="run",
                     feed=lambda e: None, control=Control(), run_dir=run,
                     evidence_dir=sd / "evidence", log_path=sd / "log" / "run-1.jsonl",
                     clock=lambda: 1000.0, **kw)


def by_request(*answers):
    seen = []
    def script(request):
        seen.append(1)
        return answers[min(len(seen), len(answers)) - 1]
    return script


def capped_notices(ledger):
    return [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("what") == "reply cut off at max_tokens"]


# ---- (a) what an agent may write ----

def test_write_file_refuses_a_file_over_the_byte_cap_and_keeps_the_old_one(run):
    run_dir, _ = run
    tools = Tools(run_dir, run_dir / "stages" / "0", agent_env(run_dir))
    assert tools.write_file("port/port_model.py", "x = 1\n").startswith("wrote ")
    out = tools.write_file("port/port_model.py", "y" * (runaway.MAX_FILE_BYTES + 1))
    assert out.startswith("refused:") and "bytes" in out and "shorter" in out
    assert (run_dir / "stages" / "0" / "port" / "port_model.py").read_text() == "x = 1\n"


def test_write_file_refuses_a_looped_file_and_a_refused_new_file_is_not_created(run):
    run_dir, _ = run
    tools = Tools(run_dir, run_dir / "stages" / "0", agent_env(run_dir))
    out = tools.write_file("port/loop.py", "def f():\n    return 1\n" * 1500)
    assert out.startswith("refused:") and "degenerate repetition" in out
    assert not (run_dir / "stages" / "0" / "port" / "loop.py").exists()


def test_write_file_still_writes_an_ordinary_file(run):
    run_dir, _ = run
    tools = Tools(run_dir, run_dir / "stages" / "0", agent_env(run_dir))
    text = "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(500))
    assert tools.write_file("port/port_model.py", text) == f"wrote {len(text)} characters to stages/0/port/port_model.py"


GROW = ("import sys\nsys.stdout.write('')\nwith open('stages/0/port/port_model.py', 'w') as f:\n"
        "    f.write('x = 1\\n' * 200000)\n")


def test_a_command_that_grows_a_port_file_past_the_cap_is_undone_and_the_agent_is_told(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    tools.write_file("port/port_model.py", "def f():\n    return 1\n")
    tools.write_file("grow.py", GROW)
    out = tools.shell("python3 stages/0/grow.py")
    assert "refused and restored" in out and "port/port_model.py" in out and "bytes" in out
    assert (sd / "port" / "port_model.py").read_text() == "def f():\n    return 1\n"


def test_a_command_that_creates_a_huge_new_port_file_has_it_removed(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    tools.write_file("grow.py", GROW.replace("port_model.py", "extra.py"))
    out = tools.shell("python3 stages/0/grow.py")
    assert "refused and removed" in out and not (sd / "port" / "extra.py").exists()


def test_a_command_that_edits_a_port_file_normally_is_left_alone(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    tools.write_file("edit.py", "open('stages/0/port/port_model.py', 'w').write('def g():\\n    return 2\\n')\n")
    out = tools.shell("python3 stages/0/edit.py")
    assert "refused" not in out and (sd / "port" / "port_model.py").read_text() == "def g():\n    return 2\n"


def test_files_outside_the_port_directory_are_not_touched_by_the_shell_guard(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    tools.write_file("big.py", "import sys\nopen('stages/0/evidence/big.txt', 'w').write('z' * 500000)\n")
    out = tools.shell("python3 stages/0/big.py")
    assert "refused" not in out and (sd / "evidence" / "big.txt").stat().st_size == 500000


# ---- (b) the retry after a capped reply ----

def test_the_request_after_a_capped_reply_uses_the_retry_profile_and_the_next_good_one_does_not(run):
    run_dir, ledger = run
    script = by_request(truncated(16384, content=LOOP), call("shell", command="echo hi"), final("done"))
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    first, second, third = fm.requests
    assert first["temperature"] == 0 and first["max_tokens"] == 16384 and "frequency_penalty" not in first
    assert second["temperature"] == 0.5 and second["max_tokens"] == 8192 and second["frequency_penalty"] == 0.2
    assert third["temperature"] == 0 and third["max_tokens"] == 16384 and "frequency_penalty" not in third


def test_without_a_retry_profile_every_request_is_as_before(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(8192), call("shell", command="echo hi"), final("done"))) as fm:
        step(run_dir, ledger, fm.endpoint).run("s", "u")
    assert all(r["temperature"] == 0 and r["max_tokens"] == 16384 and "frequency_penalty" not in r for r in fm.requests)


def test_a_server_that_refuses_the_retry_profile_gets_the_plain_request_and_the_ledger_says_so(run):
    run_dir, ledger = run
    answers = [truncated(16384, content=LOOP), call("shell", command="echo hi"), final("done")]

    def script(request):
        if "frequency_penalty" in request:
            return 400                                    # this server does not take the profile
        plain_so_far = sum(1 for r in fm.requests if "frequency_penalty" not in r)
        return answers[plain_so_far - 1]
    with FakeModel(script) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "done"
    refused = [e["data"] for e in ledger.read() if e["event"] == "notice" and e["data"].get("what") == "retry sampling refused by the server"]
    assert len(refused) == 1 and refused[0]["retry_sampling"]["temperature"] == 0.5
    assert fm.requests[-1]["temperature"] == 0 and "frequency_penalty" not in fm.requests[-1]


# ---- (c) a capped reply is its own event ----

def test_a_capped_loop_is_recorded_as_a_repetition_loop_with_its_counts_and_gets_the_loop_nudge(run):
    run_dir, ledger = run
    script = by_request(truncated(16384, content=LOOP), final("ok"))
    with FakeModel(script) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    (note,) = capped_notices(ledger)
    assert note["reply_capped"] is True and note["repetition"] is True and "degenerate repetition" in note["repetition_detail"]
    assert note["capped_in_step"] == 1 and note["capped_in_stage"] == 1 and note["completion_tokens"] == 16384
    assert note["retry_sampling"] == runaway.Unstick().record()
    nudge = fm.requests[1]["messages"][-1]["content"]
    assert "repeated itself" in nudge and "one command" in nudge


def test_a_capped_reply_that_is_not_a_loop_gets_the_ordinary_nudge(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(8192), final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    (note,) = capped_notices(ledger)
    assert note["repetition"] is False and "repetition_detail" not in note
    assert fm.requests[1]["messages"][-1]["content"] == "Your last reply was cut off before you ran any command. Think briefly, then call a tool."


def test_the_stage_count_includes_capped_replies_of_earlier_steps(run):
    run_dir, ledger = run
    ledger.append("notice", 0, watchdog=True, what="reply cut off at max_tokens", reply_capped=True, capped_in_step=1)
    ledger.append("notice", 1, watchdog=True, what="reply cut off at max_tokens", reply_capped=True, capped_in_step=1)   # another stage
    with FakeModel(by_request(truncated(8192), final("ok"))) as fm:
        step(run_dir, ledger, fm.endpoint).run("s", "u")
    assert capped_notices(ledger)[-1]["capped_in_stage"] == 2


def test_two_capped_loops_end_the_step_and_the_detail_names_the_coders_replies(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(16384, content=LOOP), truncated(16384, content=LOOP))) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick()).run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2
    assert out.detail.startswith("the reply was cut off at max_tokens twice in a row (16384 tokens each); no action was taken")
    assert "2 of the 2 capped replies in this step were repetition loops" in out.detail and "coder's replies" in out.detail


def test_two_capped_replies_that_are_not_loops_keep_the_old_detail_exactly(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(8192), truncated(8192))) as fm:
        out = step(run_dir, ledger, fm.endpoint).run("s", "u")
    assert out.detail == "the reply was cut off at max_tokens twice in a row (8192 tokens each); no action was taken"


def test_a_step_that_may_restart_drops_the_loop_from_its_context_and_goes_on(run):
    run_dir, ledger = run
    (run_dir / "stages" / "0" / "port" / "port_model.py").write_text("def f():\n    return 1\n")
    answers = [call("shell", command="echo one"), truncated(16384, content=LOOP), truncated(16384, content=LOOP),
               call("shell", command="echo two"), final("recovered")]
    with FakeModel(by_request(*answers)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("system text", "task text")
    assert out.status == "done" and out.final_text == "recovered"
    after = fm.requests[3]["messages"]
    assert [m["role"] for m in after] == ["system", "user"] and after[0]["content"] == "system text"
    assert after[1]["content"].startswith("task text") and "restarted" in after[1]["content"]
    assert fm.requests[3]["temperature"] == 0.5 and fm.requests[3]["max_tokens"] == 8192      # the fresh context still uses the retry profile
    assert "port/port_model.py" in after[1]["content"]                        # the files on disk are named, so the state is not lost
    assert not any("position 5350" in m.get("content", "") for m in after)
    decisions = [e["data"] for e in ledger.read() if e["event"] == "decision" and e["data"].get("decision") == "context restarted after capped replies"]
    assert len(decisions) == 1 and decisions[0]["capped_in_step"] == 2 and decisions[0]["repetition_loops"] == 2


def test_a_step_restarts_once_and_a_second_pair_of_capped_replies_ends_it(run):
    run_dir, ledger = run
    with FakeModel(by_request(*[truncated(16384, content=LOOP)] * 6)) as fm:
        out = step(run_dir, ledger, fm.endpoint, unstick=runaway.Unstick(), restart_on_caps=1).run("s", "u")
    assert out.status == "error" and len(fm.requests) == 4
    assert "4 of the 4 capped replies in this step were repetition loops" in out.detail
    assert sum(1 for e in ledger.read() if e["event"] == "decision" and e["data"].get("decision") == "context restarted after capped replies") == 1


def test_a_step_without_restarts_ends_after_two_as_before(run):
    run_dir, ledger = run
    with FakeModel(by_request(truncated(8192), truncated(8192), final("never"))) as fm:
        out = step(run_dir, ledger, fm.endpoint).run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2


@pytest.mark.timeout(30)
def test_a_fifo_or_a_link_to_a_device_in_the_port_directory_does_not_block_or_confuse_the_shell_guard(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    tools.write_file("port/port_model.py", "def f():\n    return 1\n")
    os.mkfifo(sd / "port" / "pipe")                                   # a reader that opened it would wait for a writer
    os.symlink("/dev/zero", sd / "port" / "zero")                     # a reader that followed it would read forever
    out = tools.shell("echo hi")
    assert out.startswith("exit 0") and "refused" not in out
    assert (sd / "port" / "port_model.py").read_text() == "def f():\n    return 1\n"


@pytest.mark.timeout(60)
def test_a_port_tree_with_many_entries_is_scanned_within_the_bound(run):
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    for i in range(2500):
        (sd / "port" / f"f{i}.py").write_text("x = 1\n")
    seen = tools._port_files()
    assert len(seen) <= 200                                           # PORT_GUARD_FILES: the scan stops, whatever the tree holds


def test_the_scan_stops_after_the_entry_bound_whatever_a_directory_lists(run, monkeypatch):
    """The bound is on entries of any kind, so a directory of ten thousand subdirectories and no files cannot make the guard slow."""
    from orchard import agent as agent_module
    run_dir, _ = run
    sd = run_dir / "stages" / "0"
    tools = Tools(run_dir, sd, agent_env(run_dir))
    consumed = []

    class Entry:
        def __init__(self, i):
            self.name, self.path = f"d{i}", f"/nonexistent/d{i}"
            consumed.append(i)

        def is_symlink(self):
            return False

        def is_dir(self, follow_symlinks=True):
            return False

        def is_file(self, follow_symlinks=True):
            return False

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return (Entry(i) for i in range(10_000))

    monkeypatch.setattr(agent_module.os, "scandir", lambda d: Listing())
    assert tools._port_files() == []
    assert len(consumed) <= 2001                                      # the documented bound: 2,000 entries, and the one that crosses it
