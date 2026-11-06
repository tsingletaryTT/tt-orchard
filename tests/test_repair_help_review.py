# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The outside helper in the repair rounds: the findings of the Copilot review of PR 496. Each test names the finding."""
import hashlib
import os

import pytest

from orchard import phone_a_friend as paf
from orchard import repair_help as rh
from orchard.supervisor import Supervisor, RunActuator, Control
from orchard.stages import run_progress
from orchard.ledger import Ledger


def stage_with_port(tmp_path, text="x = 1\n"):
    stage = tmp_path / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(text)
    return stage


# ---- finding 0: a link at stages/2/port ------------------------------------------------------------------------------------

def test_a_linked_port_directory_is_not_read_and_not_written(tmp_path):
    out = tmp_path / "outside"
    out.mkdir()
    (out / "port_model.py").write_text("SECRET = 1\n")
    stage = tmp_path / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "port").symlink_to(out, target_is_directory=True)
    data, why = rh.read_port(stage)
    assert data is None and why
    placed, why, _ = rh.place(stage, "y = 2\n", hashlib.sha256(b"SECRET = 1\n").hexdigest(), coder_failed=True, call_id="c", model="m")
    assert placed is False
    assert (out / "port_model.py").read_text() == "SECRET = 1\n"


def test_a_linked_port_file_is_not_read(tmp_path):
    stage = tmp_path / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    target = tmp_path / "t.py"
    target.write_text("SECRET = 1\n")
    (stage / "port" / "port_model.py").symlink_to(target)
    assert rh.read_port(stage)[0] is None


def test_a_good_port_is_read_and_replaced(tmp_path):
    stage = stage_with_port(tmp_path)
    assert rh.read_port(stage) == (b"x = 1\n", "")
    placed, why, _ = rh.place(stage, "y = 2\n", hashlib.sha256(b"x = 1\n").hexdigest(), coder_failed=False, call_id="c", model="m")
    assert placed and "y = 2" in (stage / "port" / "port_model.py").read_text()


# ---- finding 4: signatures that the check cannot call ----------------------------------------------------------------------

@pytest.mark.parametrize("sig", ["load(a, b, required, *args)", "prefill(model, ids, *, required)"])
def test_a_function_the_check_cannot_call_is_a_draft_problem(sig):
    name = sig.split("(")[0]
    good = {"load": "def load(a, b):\n    return 1\n", "prefill": "def prefill(m, i):\n    return 1\n", "decode_step": "def decode_step(a, b, c):\n    return 1\n",
            "close": "def close(a):\n    return 1\n"}
    src = "".join(v for k, v in good.items() if k != name) + f"def {sig}:\n    return 1\n"
    problems = paf.draft_problems(src)
    assert any(p.startswith(name) for p in problems), problems


@pytest.mark.parametrize("sig", ["load(a, b, *args)", "prefill(m, i, extra=1, *, flag=2)", "load(a, *args, **kw)"])
def test_a_callable_signature_with_extras_is_accepted(sig):
    name = sig.split("(")[0]
    good = {"load": "def load(a, b):\n    return 1\n", "prefill": "def prefill(m, i):\n    return 1\n", "decode_step": "def decode_step(a, b, c):\n    return 1\n",
            "close": "def close(a):\n    return 1\n"}
    src = "".join(v for k, v in good.items() if k != name) + f"def {sig}:\n    return 1\n"
    assert not [p for p in paf.draft_problems(src) if p.startswith(name)]


# ---- finding 3: one problem id per run directory ------------------------------------------------------------------------------

def test_two_runs_with_the_same_directory_name_get_different_problem_ids(tmp_path):
    a, b = tmp_path / "x" / "run", tmp_path / "y" / "run"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    assert paf.problem_id(a) != paf.problem_id(b)
    assert paf.problem_id(a) == paf.problem_id(a)


def test_long_names_that_share_a_prefix_get_different_problem_ids(tmp_path):
    a, b = tmp_path / ("n" * 60 + "a"), tmp_path / ("n" * 60 + "b")
    a.mkdir()
    b.mkdir()
    assert paf.problem_id(a) != paf.problem_id(b)


def test_the_problem_id_holds_no_path(tmp_path):
    d = tmp_path / "secret-home-name"
    d.mkdir()
    assert str(tmp_path) not in paf.problem_id(d)


# ---- finding 5: the attempt-start ask counts the repair asks too ----------------------------------------------------------------

def test_an_attempt_start_ask_is_refused_when_the_repair_asks_have_spent_the_stage_limit(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("stage_start", 2, escalated=False, resumed=False)
    for k in range(rh.TOTAL_PER_STAGE):
        led.append("decision", 2, decision=rh.ASKED, round=k + 1)
    led.append("stage_end", 2, result="fail", reasons=["x"])
    led.append("stage_start", 2, escalated=True, resumed=False)
    s = object.__new__(Supervisor)
    s.ledger = led
    asked = []
    s.telegraph = type("T", (), {"ask": lambda *a, **k: asked.append(a)})()
    stage = tmp_path / "stages" / "2"
    stage.mkdir(parents=True)
    spec = type("S", (), {"number": 2})()
    assert s._phone_a_friend(spec, stage) is None
    assert not asked, "a seventh paid call was sent"
    refused = [e["data"] for e in led.read() if e["data"].get("decision") == paf.NOT_ASKED]
    assert refused and "limit" in refused[0]["reason"]


# ---- finding 1, the pause case: the ledger's pause is answered ------------------------------------------------------------------------

def stub_with_stop(tmp_path, status):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("run_start", None, model="m")
    led.append("stage_start", 2, escalated=False, resumed=False)
    s = object.__new__(Supervisor)
    s.ledger = led
    s.actuator = RunActuator(Control(tmp_path))
    if status == "pause":
        s.actuator.pause("watchdog", {})
        led.append("decision", 2, decision="pause", reason="the watchdog paused the stage")
    else:
        s.actuator.escalate("stage-agent", 2)
    return s


def test_a_handled_watchdog_pause_is_cleared_in_the_actuator_and_answered_in_the_ledger(tmp_path):
    s = stub_with_stop(tmp_path, "pause")
    assert s.actuator.stop_reason() == "pause" and run_progress(s.ledger.read()).paused
    s._consume_handled_stop(2, "pause")
    assert s.actuator.stop_reason() is None
    assert run_progress(s.ledger.read()).paused is None


def test_a_handled_escalation_is_cleared_in_the_actuator_and_writes_no_resume(tmp_path):
    s = stub_with_stop(tmp_path, "escalate")
    s._consume_handled_stop(2, "escalate")
    assert s.actuator.stop_reason() is None
    assert not [e for e in s.ledger.read() if e["data"].get("decision") == "resume"]


# ---- round 2 of the review of PR 496 ---------------------------------------------------------------------------------------------------

def test_a_mapping_pattern_rest_capture_is_a_bound_name():
    """Finding: `case {"weight": w, **rest}:` binds `rest`, and the scan reported a later use of it as undefined."""
    src = "def f(x):\n    match x:\n        case {'weight': w, **rest}:\n            return w, rest\n"
    assert paf.undefined_names(src) == []


def test_a_name_that_no_pattern_binds_is_still_reported():
    """The control for the test above: the scan still reports a name nothing binds."""
    src = "def f(x):\n    match x:\n        case {'weight': w}:\n            return w, rest\n"
    assert paf.undefined_names(src) == ["rest"]


def _ent(seq, event, **data):
    return {"seq": seq, "event": event, "stage": 2, "data": data}


def test_a_resumed_start_does_not_cut_the_attempts_history():
    """Finding: a resumed stage_start reset the history boundary, so round 1's progress was lost and the helper was asked again in round 2."""
    from orchard import port_repair
    entries = [_ent(1, "stage_start", resumed=False), _ent(2, "decision", decision=port_repair.DECISION, round=1, op="load"),
               _ent(3, "stage_start", resumed=True)]
    assert rh.previous_op(entries, 2, 2) == (True, "load")
    ask, why = rh.start_trigger(entries, 2, 2, {"op": "prefill"})
    assert ask is False, why                        # the failure moved from load to prefill: the coder goes first
    entries.append(_ent(4, "stage_start", resumed=False))            # a new attempt does cut it
    assert rh.previous_op(entries, 2, 2) == (False, None)


# ---- finding 5, the standing draft: the same combined limit decides whether an ask is coming -------------------------------------------

def _supervisor_after_failed_attempt(tmp_path, repair_asks):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("stage_start", 2, escalated=False, resumed=False)
    for k in range(repair_asks):
        led.append("decision", 2, decision=rh.ASKED, round=k + 1)
    led.append("stage_end", 2, result="fail", reasons=["x"])
    led.append("stage_start", 2, escalated=True, resumed=False)
    s = object.__new__(Supervisor)
    s.ledger = led
    s.telegraph = object()
    placed = []
    s._place_friend_draft = lambda n, stage_dir, draft, call_id: placed.append(draft.name)
    stage = tmp_path / "stages" / "2"
    (stage / "telegraph").mkdir(parents=True)
    (stage / "telegraph" / "draft-1.py").write_text("x = 1\n")
    return s, stage, placed


def test_the_standing_draft_is_placed_when_the_repair_asks_have_spent_the_stage_limit(tmp_path):
    """Finding: with six repair asks spent the attempt-start ask is refused, yet the standing draft was skipped as if an ask were coming."""
    s, stage, placed = _supervisor_after_failed_attempt(tmp_path, rh.TOTAL_PER_STAGE)
    s._place_standing_draft(type("S", (), {"number": 2})(), stage)
    assert placed == ["draft-1.py"]


def test_the_standing_draft_waits_for_the_ask_when_the_limit_is_not_spent(tmp_path):
    """The control: with asks left, the attempt asks the helper and the standing draft is left alone."""
    s, stage, placed = _supervisor_after_failed_attempt(tmp_path, rh.TOTAL_PER_STAGE - 1)
    s._place_standing_draft(type("S", (), {"number": 2})(), stage)
    assert placed == []


# ---- finding 2: a helper rescue clears the pause and nothing else ---------------------------------------------------------------------

def _paused_ledger(tmp_path, resume):
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("run_start", None, model="m")
    led.append("stage_start", 2, escalated=False, resumed=False)
    led.append("escalate", 2, reason="x")
    led.append("escalate", 2, reason="y")
    led.append("decision", 2, decision="coder died; restarting it once")
    led.append("decision", 2, decision="pause", reason="the watchdog paused the stage")
    led.append("decision", 2, decision="resume", **resume)
    return led


def test_a_helper_rescue_resume_keeps_the_run_budgets_and_the_clock(tmp_path):
    """Finding: `resume` cleared the escalation and coder-death counters and restarted the wall-clock budget."""
    led = _paused_ledger(tmp_path, {"by": "helper rescue", "keeps_budgets": True})
    p = run_progress(led.read())
    assert p.paused is None
    assert p.escalations == 2 and p.coder_deaths == 1
    assert p.clock_start == run_progress(led.read()[:1]).clock_start


def test_an_operator_resume_still_resets_the_budgets(tmp_path):
    """The control: the operator's resume is a fresh start, as before."""
    p = run_progress(_paused_ledger(tmp_path, {"by": "operator"}).read())
    assert p.paused is None and p.escalations == 0 and p.coder_deaths == 0


def _ladder_with_a_rung_taken(tmp_path, resume):
    from orchard import watchdog
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("run_start", None, model="m")
    rung = watchdog.RUNGS[0]
    led.append("decision", 2, watchdog=True, rung=rung, agent="coder", findings=[])
    led.append("decision", 2, decision="resume", **resume)
    return watchdog.Ladder(RunActuator(Control(tmp_path)), led, {"coder"}), rung


def test_a_helper_rescue_resume_does_not_give_back_a_watchdog_rung(tmp_path):
    """Finding: the ladder forgot every recorded rung at a resume, so a consumed action could repeat."""
    ladder, rung = _ladder_with_a_rung_taken(tmp_path, {"by": "helper rescue", "keeps_budgets": True})
    assert ladder._taken[("coder", 2, rung)] == 1


def test_an_operator_resume_still_gives_the_rungs_back(tmp_path):
    ladder, rung = _ladder_with_a_rung_taken(tmp_path, {"by": "operator"})
    assert ladder._taken[("coder", 2, rung)] == 0


def test_the_supervisor_writes_the_handled_pause_resume_with_the_marker(tmp_path):
    s = stub_with_stop(tmp_path, "pause")
    s._consume_handled_stop(2, "pause")
    (resume,) = [e for e in s.ledger.read() if e["data"].get("decision") == "resume"]
    assert resume["data"].get("keeps_budgets") is True


# ---- finding 4: a completed test is judged before the next ask and before the attempt-end fallback ----------------------------------------

import json as _json  # noqa: E402
from orchard import port_repair  # noqa: E402


def _failed_test_on_disk(run, stage, op):
    """A supervisor test record and output file that `port_repair.read_failure` accepts: a failure of the port at request `op`."""
    out = stage / port_repair.OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = ("FAILED: x\n" + port_repair.FAILURE_TOKEN + _json.dumps({"kind": "port", "op": op}) + "\n").encode()
    out.write_bytes(raw)
    (stage / port_repair.MARKER).write_text(_json.dumps({"returncode": 2, "timed_out": False,
                                                          "output": {"path": str(out.relative_to(run)), "sha256": hashlib.sha256(raw).hexdigest()}}))


def _answered_then_tested(tmp_path, *, tested, after="prefill", vouch=True):
    """A run whose helper answered in round 1 (the check failed at `load`) and whose supervisor stopped after the check's rerun finished."""
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text("x = 1\n")
    led = Ledger(run / "ledger.jsonl")
    led.append("run_start", None, model="m")
    led.append("stage_start", 2, escalated=False, resumed=False)
    led.append("decision", 2, decision=port_repair.DECISION, round=1, of=3, op="load")
    led.append("decision", 2, decision=rh.ASKED, round=1, failure_op="load")
    led.append("decision", 2, decision=rh.ANSWERED, ok=True, call_id="c1", cached=False)
    if tested:
        led.append("decision", 2, decision=port_repair.TESTED, sources=port_repair.digest(port_repair.fingerprint(stage / "port")))
    _failed_test_on_disk(run, stage, after)            # by default the test after the answer failed later in the chain: progress
    if vouch:
        # What Supervisor._run_test appends right after it writes the marker: the path and hash the verdict is tied to (review of PR 496, round 3).
        raw = (stage / port_repair.MARKER).read_bytes()
        led.append("evidence", 2, what="hardware test", config=[], cache=None, returncode=2, timed_out=False,
                   path=str((stage / port_repair.MARKER).relative_to(run)), sha256=hashlib.sha256(raw).hexdigest())
    s = object.__new__(Supervisor)
    s.run_dir, s.ledger, s.repair_help = run, led, True
    reported = []
    s.telegraph = type("T", (), {"report": lambda self, call_id, helped, note="": reported.append((call_id, helped, note))})()
    return s, stage, reported


def _progress(s):
    return [e["data"] for e in s.ledger.read() if e["data"].get("decision") == rh.PROGRESS]


def test_the_attempt_end_judges_an_answer_by_the_completed_test_not_by_the_attempt_outcome(tmp_path):
    """Finding: the end-of-attempt fallback gave the answer the attempt's outcome even when a test after the answer had judged it."""
    s, stage, reported = _answered_then_tested(tmp_path, tested=True)
    s._report_phone_a_friend(2, helped=False, stage_dir=stage)
    (p,) = _progress(s)
    assert p["outcome"] == "further" and p["helped"] is True and p["after_op"] == "prefill"
    assert reported[0][1] is True


def test_an_answer_no_test_followed_still_gets_the_attempt_outcome(tmp_path):
    """The control: with no `tested` entry after the answer, nothing judged it."""
    s, stage, reported = _answered_then_tested(tmp_path, tested=False)
    s._report_phone_a_friend(2, helped=False, stage_dir=stage)
    (p,) = _progress(s)
    assert p["outcome"] == "the attempt ended before the check judged it" and p["helped"] is False


def test_a_restart_judges_the_finished_test_before_it_asks_again(tmp_path):
    """Finding: after a stop between the `tested` entry and its verdict, the next round asked the helper first, so its model list saw no verdict."""
    s, stage, _ = _answered_then_tested(tmp_path, tested=True, after="load")          # the same request: the next round asks again
    s.port_repair_rounds = 3
    s._watchdog = lambda spec: None
    s._homes = lambda: ()
    seen = []

    class Stop(Exception):
        pass

    def fake_help(*a, **k):
        seen.append(len(_progress(s)))
        raise Stop

    s._repair_help = fake_help
    with pytest.raises(Stop):
        s._port_repair_rounds(type("S", (), {"number": 2})(), stage, False, False)
    assert seen == [1], "the helper was asked before the finished test was judged"


def test_a_failure_marker_the_ledger_never_vouched_for_judges_nothing(tmp_path):
    """Finding (round 3): the failure path trusted a marker and an output hash the coder can rewrite. Without the supervisor's evidence entry the
    answer is not judged by it; it gets the attempt's outcome."""
    s, stage, _ = _answered_then_tested(tmp_path, tested=True, vouch=False)
    s._report_phone_a_friend(2, helped=False, stage_dir=stage)
    (p,) = _progress(s)
    assert p["outcome"] == "the attempt ended before the check judged it" and p["helped"] is False


def test_a_rewritten_passing_marker_does_not_make_the_answer_helped(tmp_path):
    """Finding (round 3): with a pending answer and a later `tested` entry, a marker rewritten to exit code 0 produced a helped verdict."""
    s, stage, _ = _answered_then_tested(tmp_path, tested=True)
    marker = stage / port_repair.MARKER
    marker.write_text(_json.dumps({"returncode": 0, "timed_out": False}))           # the coder's rewrite: the ledger's hash no longer matches
    s._report_phone_a_friend(2, helped=False, stage_dir=stage)
    (p,) = _progress(s)
    assert p["helped"] is False and p["outcome"] == "the attempt ended before the check judged it"


def test_the_vouched_marker_still_judges_the_answer(tmp_path):
    s, stage, _ = _answered_then_tested(tmp_path, tested=True)
    passed, op, readable = s._repair_outcome(2, stage)
    assert (passed, op, readable) == (False, "prefill", True)


def test_the_implicit_class_cell_is_a_known_name():
    """Finding (round 3): a method that refers to `__class__` binds it implicitly, and the scope-blind check reported it as undefined, which made
    repair placement and skeleton merging refuse a valid port."""
    from orchard import phone_a_friend as paf
    src = "class A:\n    def cls(self):\n        return __class__\n"
    assert paf.undefined_names(src) == []
    assert paf.undefined_names("x = missing_name\n") == ["missing_name"]          # the control: a real undefined name is still reported


# ---- round 3: the placement is about one port directory and keeps a late edit -----------------------------------------------------------

def _port_run(tmp_path, text="x = 1\n"):
    stage = tmp_path / "run" / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(text)
    return stage


def test_a_port_directory_swapped_in_after_the_read_is_not_written(tmp_path, monkeypatch):
    """Finding: the pinned descriptor named the stage directory only. A `port` replaced by another real directory between the read and the write
    had its coder file overwritten."""
    stage = _port_run(tmp_path)
    sent = hashlib.sha256(b"x = 1\n").hexdigest()
    real = rh.safeio.read_beneath
    calls = []

    def swapping(fd, parts, **kw):
        out = real(fd, parts, **kw)
        if not calls:
            calls.append(1)
            os.rename(stage / "port", stage / "port.old")
            (stage / "port").mkdir()
            (stage / "port" / "port_model.py").write_text("coder_file = 2\n")
        return out
    monkeypatch.setattr(rh.safeio, "read_beneath", swapping)
    rh.place(stage, "y = 2\n", sent, coder_failed=False, call_id="c1", model="m")
    assert (stage / "port" / "port_model.py").read_text() == "coder_file = 2\n"


def test_an_edit_made_after_the_comparison_is_kept_and_the_placement_is_refused(tmp_path, monkeypatch):
    """Finding: the replacement was unconditional, so an edit between the comparison and the write was lost."""
    stage = _port_run(tmp_path)
    sent = hashlib.sha256(b"x = 1\n").hexdigest()
    real = rh.safeio.read_beneath
    calls = []

    def editing(fd, parts, **kw):
        out = real(fd, parts, **kw)
        if not calls:
            calls.append(1)
            (stage / "port" / "port_model.py").write_text("late_edit = 3\n")
        return out
    monkeypatch.setattr(rh.safeio, "read_beneath", editing)
    placed, why, _ = rh.place(stage, "y = 2\n", sent, coder_failed=False, call_id="c1", model="m")
    assert placed is False and "changed" in why
    assert (stage / "port" / "port_model.py").read_text() == "late_edit = 3\n"
    assert [p.name for p in (stage / "port").iterdir()] == ["port_model.py"], "a temporary file was left behind"
