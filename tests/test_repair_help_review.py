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
