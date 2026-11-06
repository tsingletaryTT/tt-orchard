# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The helper drives stage 2: the findings of the Copilot review of PR 497. Each test names the finding."""
import pytest

from orchard import port_repair, repair_help as rh
from orchard.ledger import Ledger
from orchard.supervisor import EXIT_ABORTED, Control, Supervisor
from port_check_fakes import HONEST
from test_helper_drives_port import BROKEN, PORT, PREFILL_BROKEN, STILL_LOAD, decs, port_text, rig, start, wire_helper  # noqa: F401
from test_supervisor_telegraph_in_repair import stage_ends
from test_telegraph import openai_reply


def edit_during_request(rig, code):
    """A helper reply that, as the request is answered, changes the port file (an edit made while the helper was working), once."""
    done = []

    def answer(body):
        if not done:
            done.append(1)
            path = rig.run_dir / "stages" / "2" / "port" / "port_model.py"
            path.write_text(path.read_text() + "# edited while the helper answered\n")
        return openai_reply(f"Here.\n```python\n{code}\n```\n", 1200, 400)
    return answer


def test_a_file_changed_while_the_helper_answered_is_not_overwritten_in_a_driven_round(rig, monkeypatch, tmp_path):
    """Finding: driven rounds passed coder_failed=True, which switched off the sent-hash conflict check."""
    wire_helper(monkeypatch, tmp_path, [edit_during_request(rig, PREFILL_BROKEN), HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    refused = [d for d in decs(rig, rh.NOT_PLACED)]
    assert any("coder changed" in d.get("reason", "") for d in refused), refused
    first = decs(rig, rh.ASKED)[0]
    assert first["coder_failed"] is False


def test_an_abort_queued_while_the_helper_answers_stops_the_run_before_the_check_and_before_another_ask(rig, monkeypatch, tmp_path):
    """Finding: a driven round never polled the operator's controls, so paid work went on after an abort."""
    sent = []

    def answer(body):
        if not sent:
            sent.append(1)
            Control(rig.run_dir).write("abort")
        return openai_reply(f"Here.\n```python\n{PREFILL_BROKEN}\n```\n", 1200, 400)
    tel, wire = wire_helper(monkeypatch, tmp_path, [answer, HONEST])
    start(rig, BROKEN, ["# c\n"])
    assert rig.run() == EXIT_ABORTED
    assert len(wire.requests) == 1, "a second paid request went out after the abort"
    assert not decs(rig, port_repair.RERUN), "the chip check ran after the abort"
    assert "pass" not in stage_ends(rig)


def entry(seq, event, data, stage=2):
    return {"seq": seq, "event": event, "stage": stage, "data": data}


def test_a_resumed_stage_start_does_not_hide_the_rounds_of_the_same_attempt():
    """Finding: _last_start took the latest stage_start, so a resume made the earlier rounds of the attempt invisible."""
    entries = [entry(1, "stage_start", {"resumed": False}),
               entry(2, "decision", {"decision": port_repair.DECISION, "round": 1, "op": "load", "failure": "boom"}),
               entry(3, "stage_start", {"resumed": True})]
    assert rh.previous_op(entries, 2, 2) == (True, "load")


def test_a_fresh_stage_start_still_starts_a_new_attempt():
    entries = [entry(1, "stage_start", {"resumed": False}),
               entry(2, "decision", {"decision": port_repair.DECISION, "round": 1, "op": "load"}),
               entry(3, "stage_start", {"resumed": False})]
    assert rh.previous_op(entries, 2, 2) == (False, None)


class Stub(Supervisor):
    def __init__(self, tmp_path, *, rounds=3, drives=True, rehearsal=False, ledger=None):
        self.port_repair_rounds, self.helper_drives, self.full_port_rehearsal = rounds, drives, rehearsal
        self.repair_help, self.telegraph, self.run_dir = True, object(), tmp_path
        self.ledger = ledger or Ledger(tmp_path / "ledger.jsonl")


def test_the_effective_round_limit_is_shared_by_the_loop_and_the_pending_check(tmp_path, monkeypatch):
    """Finding: _repair_pending used --port-repair-rounds, so a helper-driven attempt resumed after its third failed rerun looked exhausted."""
    s = Stub(tmp_path)
    assert s._repair_limit(2) == max(3, rh.HELPER_ROUNDS)
    monkeypatch.setattr(port_repair, "stopped_noted", lambda *a: False)
    monkeypatch.setattr(port_repair, "rerun_inflight", lambda *a: None)
    monkeypatch.setattr(port_repair, "read_failure", lambda *a: {"op": "load"})
    monkeypatch.setattr(port_repair, "rounds_used", lambda *a: 3)
    spec = type("S", (), {"draft": "full_port", "number": 2})()
    assert s._repair_pending(spec, tmp_path) is True
    s.helper_drives = False
    assert s._repair_pending(spec, tmp_path) is False        # coder-driven: three rounds are all there are


def test_a_resumed_rehearsal_keeps_the_helper_driving(tmp_path):
    """Finding: build() passes full_port_rehearsal=False for an existing run, so a resume switched back to coder-driven repairs."""
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("run_start", None, model="m", full_port_rehearsal=True)
    s = Stub(tmp_path, drives=None, rehearsal=False, ledger=led)
    assert s._helper_drive_on() is True
    led2 = Ledger(tmp_path / "ledger2.jsonl")
    led2.append("run_start", None, model="m")
    assert Stub(tmp_path, drives=None, rehearsal=False, ledger=led2)._helper_drive_on() is False


def test_the_failure_of_the_last_allowed_round_is_recorded_as_the_best_port(rig, monkeypatch, tmp_path):
    """Finding: the best port was recorded only when another round began, so the last round's progress never reached the ledger."""
    monkeypatch.setattr(rh, "HELPER_ROUNDS", 2)
    rig.args.port_repair_rounds = 2
    wire_helper(monkeypatch, tmp_path, [STILL_LOAD, PREFILL_BROKEN])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    entries = rig.entries()
    first_end = next(e["seq"] for e in entries if e["stage"] == 2 and e["event"] == "stage_end")
    first_attempt = [e for e in entries if e["seq"] < first_end]
    best = rh.best_state(first_attempt, 2)
    assert best is not None and best["op"] == "prefill", best      # the first attempt alone, not what a later attempt records


def test_an_abort_that_arrived_during_the_chip_check_stops_the_run_before_the_first_paid_request(rig, monkeypatch, tmp_path):
    original = Supervisor._hardware_phase

    def phase(self, *a, **k):
        original(self, *a, **k)
        Control(self.run_dir).write("abort")                 # the operator's command arrives while the first check ran
    monkeypatch.setattr(Supervisor, "_hardware_phase", phase)
    tel, wire = wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, ["# c\n"])
    assert rig.run() == EXIT_ABORTED
    assert not wire.requests, "a paid request went out after the abort"


def test_a_helper_driven_attempt_runs_more_rounds_than_port_repair_rounds(rig, monkeypatch, tmp_path):
    """The loop's limit is the helper's when the helper drives, whatever --port-repair-rounds says."""
    monkeypatch.setattr(rh, "HELPER_ROUNDS", 3)
    rig.args.port_repair_rounds = 1
    wire_helper(monkeypatch, tmp_path, [STILL_LOAD, PREFILL_BROKEN, PREFILL_BROKEN + "# x\n", HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    entries = rig.entries()
    first_end = next(e["seq"] for e in entries if e["stage"] == 2 and e["event"] == "stage_end")
    rounds = [e for e in entries if e["seq"] < first_end and e["data"].get("decision") == port_repair.DECISION]
    assert len(rounds) >= 2, len(rounds)
