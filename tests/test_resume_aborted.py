# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Resuming an operator's abort (PR 53 review, finding 1).

An abort is terminal by default: running the same command again returns EXIT_ABORTED and starts nothing
(test_supervisor.py pins that). The web page's `Resume run` for an aborted run launched exactly that command,
so it could not advance. Resuming is now an explicit, opt-in transition: `--resume-aborted` makes the
supervisor record `resume` with `by: retry, after: abort` for an abort that an operator or a signal made, and
replay then stops treating the run as aborted. Nothing resumes without the flag."""
import pytest

from orchard.stages import abort_can_resume, run_progress
from orchard.supervisor import EXIT_ABORTED, EXIT_READY, Control, parse
from test_supervisor import rig  # noqa: F401  (the fixture)

pytestmark = pytest.mark.usefixtures("stub_tools")

RESUME_AFTER_ABORT = {"decision": "resume", "by": "retry", "after": "abort"}


def abort_at_start(rig):
    Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    return rig.m.coder_starts


def abort_while_paused(rig):
    """A pause the run is waiting in, then the operator's abort: the ledger holds an open pause and an abort."""
    Control(rig.run_dir).write("pause")
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    rig.on_sleep = None
    return rig.m.coder_starts


def ev(event, data, stage=None, ts="2026-10-23T12:00:00Z"):
    return {"ts": ts, "event": event, "stage": stage, "data": data}


# ---- the finding: the old launch cannot advance ---------------------------------------------------------------

def test_without_the_flag_the_same_command_starts_nothing_and_stays_aborted(rig):
    """What the page launched before this change. It is the default and stays that way."""
    starts = abort_at_start(rig)
    assert rig.run() == EXIT_ABORTED and rig.m.coder_starts == starts
    assert not any(d == RESUME_AFTER_ABORT for d in rig.decisions())


# ---- the transition ---------------------------------------------------------------------------------------------

def test_with_the_flag_an_aborted_run_resumes_and_every_stage_passes(rig):
    starts = abort_at_start(rig)
    rig.args.resume_aborted = True
    assert rig.run() == EXIT_READY
    assert rig.m.coder_starts > starts                                   # the coder was started again
    assert RESUME_AFTER_ABORT in rig.decisions()
    assert [e["data"].get("result") for e in rig.entries() if e["event"] == "stage_end"][-1] in ("pass", "skipped")
    assert run_progress(rig.entries()).finished and not run_progress(rig.entries()).aborted


def test_the_resume_entry_comes_after_the_abort_and_before_any_new_work(rig):
    abort_at_start(rig)
    rig.args.resume_aborted = True
    rig.run()
    names = [(e["event"], e["data"].get("decision")) for e in rig.entries()]
    abort_at = names.index(("decision", "abort"))
    resume_at = names.index(("decision", "resume"))
    assert abort_at < resume_at
    assert not any(ev == "stage_start" for ev, _ in names[abort_at:resume_at])


def test_an_abort_made_while_paused_resumes_too_and_the_pause_is_cleared(rig):
    starts = abort_while_paused(rig)
    p = run_progress(rig.entries())
    assert p.aborted and p.paused is not None                            # the ledger as the abort left it
    rig.args.resume_aborted = True
    assert rig.run() == EXIT_READY
    assert rig.m.coder_starts > starts


def test_a_second_abort_after_a_resume_is_an_abort_again(rig):
    abort_at_start(rig)
    rig.args.resume_aborted = True
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    Control(rig.run_dir).write("pause")                                  # the resumed run waits, then is aborted again
    assert rig.run() == EXIT_ABORTED
    assert run_progress(rig.entries()).aborted
    rig.args.resume_aborted = False
    rig.on_sleep = None
    assert rig.run() == EXIT_ABORTED                                     # and without the flag it stays aborted


def test_a_finished_run_is_not_resumed_or_changed_by_the_flag(rig):
    assert rig.run() == EXIT_READY
    before = len(rig.entries())
    rig.args.resume_aborted = True
    assert rig.run() == EXIT_READY
    assert not any(d == RESUME_AFTER_ABORT for d in rig.decisions())
    assert len(rig.entries()) >= before                                   # nothing was re-run: no new stage_start
    assert sum(1 for e in rig.entries() if e["event"] == "stage_start") == sum(
        1 for e in rig.entries()[:before] if e["event"] == "stage_start")


# ---- replay -----------------------------------------------------------------------------------------------------

def base_entries():
    return [ev("run_start", {"model": "org/a"}), ev("stage_start", {}, 0)]


def test_a_plain_operator_resume_after_an_abort_does_not_clear_it():
    """The control file's resume is for a paused run a supervisor is waiting in; an aborted run has none."""
    entries = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}),
                                ev("decision", {"decision": "resume", "by": "operator"})]
    assert run_progress(entries).aborted


def test_a_resume_after_abort_clears_it_and_resets_the_counters():
    entries = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}),
                                ev("decision", RESUME_AFTER_ABORT, ts="2026-10-23T13:00:00Z")]
    p = run_progress(entries)
    assert not p.aborted and p.paused is None and p.started


def test_a_later_abort_wins_over_an_earlier_resume_after_abort():
    entries = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}),
                                ev("decision", RESUME_AFTER_ABORT),
                                ev("decision", {"decision": "abort", "by": "operator"})]
    assert run_progress(entries).aborted


def test_a_resume_after_abort_with_no_abort_before_it_changes_nothing():
    entries = base_entries() + [ev("decision", RESUME_AFTER_ABORT)]
    p = run_progress(entries)
    assert not p.aborted and not p.finished


@pytest.mark.parametrize("by", ["operator", "signal SIGINT", "signal SIGTERM"])
def test_an_operators_or_a_signals_abort_can_resume(by):
    assert abort_can_resume(base_entries() + [ev("decision", {"decision": "abort", "by": by})])


@pytest.mark.parametrize("by", [None, "", "watchdog", "signal", "signal ", "signalSIGTERM", "operator2", "Operator", 5, ["operator"]])
def test_an_abort_by_anything_else_cannot_resume(by):
    assert not abort_can_resume(base_entries() + [ev("decision", {"decision": "abort", "by": by})])


def test_a_run_that_is_not_aborted_or_is_finished_has_nothing_to_resume():
    assert not abort_can_resume(base_entries())
    assert not abort_can_resume([])
    finished = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}),
                                 ev("decision", {"decision": "ready for operator review"})]
    assert not abort_can_resume(finished)


def test_the_latest_abort_decides_who_made_it():
    entries = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}), ev("decision", RESUME_AFTER_ABORT),
                                ev("decision", {"decision": "abort", "by": "watchdog"})]
    assert not abort_can_resume(entries)


def test_an_already_resumed_run_is_not_aborted_so_there_is_nothing_to_resume():
    entries = base_entries() + [ev("decision", {"decision": "abort", "by": "operator"}), ev("decision", RESUME_AFTER_ABORT)]
    assert not abort_can_resume(entries)


# ---- the flag ---------------------------------------------------------------------------------------------------

def test_the_flag_defaults_off_and_is_a_switch(tmp_path):
    base = ["run", "--model", "org/a", "--run-dir", str(tmp_path), "--tiers", "t.toml", "--coder-target", "p/n",
            "--coder-kind", "container", "--coder-profile", "x", "--coder-port", "8000", "--coder-chips", "2"]
    assert parse(base).resume_aborted is False
    assert parse(base + ["--resume-aborted"]).resume_aborted is True


def test_the_unattended_run_does_not_resume_an_abort_by_itself(rig):
    """`--unattended` retries a block. It must not turn an abort into a resume: only the explicit flag does."""
    abort_at_start(rig)
    rig.args.unattended = True
    assert rig.run() == EXIT_ABORTED
    assert not any(d == RESUME_AFTER_ABORT for d in rig.decisions())


def test_a_ledger_with_an_abort_but_no_run_start_has_nothing_to_resume():
    assert not abort_can_resume([ev("decision", {"decision": "abort", "by": "operator"})])


def test_an_abort_of_another_kind_is_left_alone_by_a_supervisor_that_was_asked_to_resume(rig):
    from orchard.ledger import Ledger
    abort_at_start(rig)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("decision", None, decision="abort", by="watchdog")          # not an operator's and not a signal's
    rig.args.resume_aborted = True
    starts = rig.m.coder_starts
    assert rig.run() == EXIT_ABORTED and rig.m.coder_starts == starts
    assert not any(d == RESUME_AFTER_ABORT for d in rig.decisions())


def test_a_supervisor_built_directly_does_not_resume_an_abort():
    """The option is off unless something turns it on. `build()` passes it explicitly, so only the signature shows the default."""
    import inspect

    from orchard.supervisor import Supervisor
    assert inspect.signature(Supervisor.__init__).parameters["resume_aborted"].default is False


def test_the_operator_runbook_still_says_never_abort_and_no_longer_says_an_abort_cannot_be_resumed():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "orchard" / "skills" / "operator-runbook.md").read_text(encoding="utf-8")
    assert "Never send `control abort`." in text
    assert "cannot be resumed" not in text
    assert "only a person resumes it" in text
