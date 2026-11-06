# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The helper drives stage 2: the findings of the Copilot review of PR 497. Each test names the finding."""
from pathlib import Path

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


# ---- round 2 of the review of PR 497 ---------------------------------------------------------------------------------------------------

def test_a_helper_driven_round_past_the_stage_budget_pauses_before_the_chip_check_and_the_next_request(rig, monkeypatch, tmp_path):
    """Finding: driven rounds skip the agent step that feeds the watchdog, so slow helper calls took the attempt past its budget unnoticed."""
    from orchard.stages import STAGE_BUDGET_S
    over = STAGE_BUDGET_S[2] + 1.0

    def answer(body):
        rig.clock.now += over                                  # the helper call took longer than the stage's whole budget
        return openai_reply(f"Here.\n```python\n{PREFILL_BROKEN}\n```\n", 1200, 400)
    tel, wire = wire_helper(monkeypatch, tmp_path, [answer, HONEST])
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    assert len(wire.requests) == 1, "a second paid request went out past the stage budget"
    assert not decs(rig, port_repair.RERUN), "the chip check ran past the stage budget"
    assert "pass" not in stage_ends(rig)


def test_a_run_started_with_an_explicit_choice_keeps_it_on_a_resume(tmp_path):
    """Finding: `helper_drives` came only from the current arguments, so a resume without the flag lost an explicit choice."""
    from orchard import supervisor as sv
    for choice in (True, False):
        led = Ledger(tmp_path / f"ledger-{choice}.jsonl")
        led.append("run_start", None, model="m", helper_drives=choice)
        progress = sv.run_progress(led.read())
        assert progress.run_start["helper_drives"] is choice
    assert sv.recorded_helper_drives(None, {"helper_drives": True}) is True
    assert sv.recorded_helper_drives(None, {"helper_drives": False}) is False
    assert sv.recorded_helper_drives(True, {}) is True and sv.recorded_helper_drives(None, {}) is None
    assert sv.recorded_helper_drives(False, {"helper_drives": True}) is True          # the run's own record wins over a later flag


def test_build_gives_a_resumed_supervisor_the_choice_the_run_started_with(rig, monkeypatch, tmp_path):
    """The wiring: run_start records the explicit choice, and `build` reads it back for a resume that does not repeat the flag."""
    from orchard import supervisor
    rig.args.full_port_rehearsal = False
    rig.args.helper_drives_port = True
    start(rig, BROKEN, ["# c\n"])
    rig.run()
    starts = [e for e in Ledger(rig.run_dir / "ledger.jsonl").read() if e["event"] == "run_start"]
    assert starts[0]["data"].get("helper_drives") is True
    rig.args.helper_drives_port = None                         # the resume does not repeat the flag
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = supervisor.build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m),
                               versions={"tt_model": "test"}, clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p),
                               home=rig.home, containers=rig.containers)
        assert sup.helper_drives is True


def _passed_check_stub(tmp_path, *, vouched, fails_at=None):
    """A stage 2 directory whose supervisor test record says the check passed, and a supervisor that drives with the helper."""
    import hashlib
    import json
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text("x = 1\n")
    (stage / "port" / "helper.py").write_text("y = 2\n")
    # The chain the supervisor's test leaves: a record that lists the sources it ran, an output file that prints the record's hash once, and a
    # marker that names the output by hash (review of PR 497, round 3).
    from orchard import full_port
    (stage / "evidence").mkdir()
    record = json.dumps({"port_sources": [
        {"path": f"stages/2/port/{name}", "sha256": hashlib.sha256((stage / "port" / name).read_bytes()).hexdigest(), "role": "model"}
        for name in ("helper.py", "port_model.py")]}).encode()
    (stage / full_port.RECORD_FILE).write_bytes(record)
    out = f"loaded\n{full_port.OUTPUT_MARK} {hashlib.sha256(record).hexdigest()}\n".encode()
    if fails_at is not None:      # a check that failed at request `fails_at`, with the failure line `port_repair.read_failure` reads
        out = ("FAILED: x\n" + port_repair.FAILURE_TOKEN + json.dumps({"kind": "port", "op": fails_at}) + "\n").encode()
    (stage / full_port.OUTPUT_FILE).write_bytes(out)
    code = 0 if fails_at is None else 2
    marker = json.dumps({"returncode": code, "timed_out": False,
                         "output": {"path": f"stages/2/{full_port.OUTPUT_FILE}", "sha256": hashlib.sha256(out).hexdigest()}}).encode()
    (stage / port_repair.MARKER).write_bytes(marker)
    led = Ledger(run / "ledger.jsonl")
    led.append("run_start", None, model="m")
    led.append("stage_start", 2, escalated=False, resumed=False)
    if vouched:
        # What `_run_test` writes before the test starts: the whole port directory, every regular file (review of PR 497, round 4).
        led.append("decision", 2, decision=port_repair.PORT_AT_TEST, sources=port_repair.fingerprint(stage / "port"))
        led.append("evidence", 2, path="stages/2/" + port_repair.MARKER, sha256=hashlib.sha256(marker).hexdigest(), returncode=code, timed_out=False)
    s = object.__new__(Supervisor)
    s.run_dir, s.ledger, s.repair_help, s.helper_drives, s.telegraph = run, led, True, True, object()
    return s, stage


def test_a_check_that_ran_to_the_end_is_kept_as_the_best_port_with_its_supporting_files(tmp_path):
    """Finding: only failures were recorded, so a running port that the gate refused for its agreement was lost to the next attempt."""
    s, stage = _passed_check_stub(tmp_path, vouched=True)
    s._record_best_passed(2, stage)
    (best,) = [e["data"] for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]
    assert best["rank"] == rh.PASSED_RANK > rh.best_rank("decode") and best["passed"] is True
    assert [f["path"] for f in best["files"]] == ["helper.py", "port_model.py"]
    s._record_best_passed(2, stage)                                    # the same port again adds nothing
    assert len([e for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]) == 1


def test_a_test_record_the_supervisor_did_not_write_is_not_a_passed_check(tmp_path):
    s, stage = _passed_check_stub(tmp_path, vouched=False)
    s._record_best_passed(2, stage)
    assert not [e for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]


def test_a_port_edited_after_the_passing_check_does_not_displace_the_best_port(tmp_path):
    """Finding (round 3): on a resume after a successful check, an edited port kept the authenticated success marker and was saved at the passing
    rank without being tested."""
    s, stage = _passed_check_stub(tmp_path, vouched=True)
    s._record_best_passed(2, stage)
    (first,) = [e["data"] for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]
    (stage / "port" / "port_model.py").write_text("x = 1\n# edited after the check\n")
    s._record_best_passed(2, stage)
    bests = [e["data"] for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]
    assert bests == [first], "an untested snapshot was recorded as the best port"
    assert any("not the one the passing check ran on" in e["data"].get("reason", "") for e in s.ledger.read())


def test_a_file_added_to_the_port_after_the_passing_check_is_also_refused(tmp_path):
    s, stage = _passed_check_stub(tmp_path, vouched=True)
    (stage / "port" / "extra.py").write_text("z = 3\n")
    s._record_best_passed(2, stage)
    assert not [e for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]


def test_a_record_rewritten_to_list_the_edited_port_is_refused_because_the_output_mark_no_longer_matches(tmp_path):
    import hashlib
    import json
    from orchard import full_port
    s, stage = _passed_check_stub(tmp_path, vouched=True)
    (stage / "port" / "port_model.py").write_text("x = 1\n# edited\n")
    rec = json.loads((stage / full_port.RECORD_FILE).read_text())
    for item in rec["port_sources"]:
        item["sha256"] = hashlib.sha256((stage / item["path"].replace("stages/2/", "")).read_bytes()).hexdigest()
    (stage / full_port.RECORD_FILE).write_text(json.dumps(rec))          # the coder's rewrite: the output's printed hash is the old record's
    s._record_best_passed(2, stage)
    assert not [e for e in s.ledger.read() if e["data"].get("decision") == rh.BEST]


def _draft_stub(tmp_path, *, with_best):
    """A supervisor at the start of an attempt: the port directory holds the real skeleton, and the ledger holds (or not) a best port that an
    earlier attempt saved. `_place_standing_draft` is replaced by a recorder."""
    from types import SimpleNamespace
    from orchard import port_facts
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    skeleton = Path(port_facts.SKELETON).read_bytes()
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_bytes(b"best_port = 1\n")
    led = Ledger(run / "ledger.jsonl")
    led.append("run_start", None, model="m")
    led.append("stage_start", 2, escalated=False, resumed=False)
    if with_best:
        saved = rh.save_best(stage)
        led.append("decision", 2, decision=rh.BEST, op="prefill", rank=3, **saved, attempt=1)
    (stage / "port" / "port_model.py").write_bytes(skeleton)              # the next attempt's fresh skeleton
    s = object.__new__(Supervisor)
    s.run_dir, s.ledger, s.repair_help, s.helper_drives, s.telegraph = run, led, True, True, object()
    calls = []
    s._pin_full_port = lambda spec, d, **k: None
    s._seed_full_port = lambda spec, d: None
    s._place_standing_draft = lambda spec, d: calls.append("standing") or None
    return s, stage, SimpleNamespace(number=2, draft="full_port"), calls


def test_the_best_port_is_restored_before_the_standing_draft_and_the_draft_is_then_not_placed(tmp_path):
    """Finding (round 3): `_draft` placed the standing draft first, `restore_best` then refused the changed file, and the attempt started from an
    older draft although a further-tested snapshot existed."""
    s, stage, spec, calls = _draft_stub(tmp_path, with_best=True)
    s._draft(spec, stage)
    assert (stage / "port" / "port_model.py").read_bytes() == b"best_port = 1\n"
    assert calls == [], "the standing draft was considered after the best port was restored"
    assert s._best_restore == (str(stage), "The files under `stages/2/port/` already hold the furthest version an earlier attempt reached "
                                           "(its check failed at the 'prefill' request). Start from it: read it, correct what the check shows "
                                           "to be wrong and keep what works.")


def test_without_a_best_port_the_standing_draft_is_still_considered(tmp_path):
    s, stage, spec, calls = _draft_stub(tmp_path, with_best=False)
    s._draft(spec, stage)
    assert calls == ["standing"]
