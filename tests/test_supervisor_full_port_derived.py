# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 456: stage 2 of a full port. When the hardware test passed and the check's record is complete, the supervisor writes result.json
from the record and no finish agent step runs. The chip, the coder and the model are fakes; the supervisor, the gate and the check are real."""
import json

import pytest

from orchard import full_port, full_port_result, supervisor
from orchard.supervisor import EXIT_BLOCKED, Control
from port_check_fakes import HONEST, WRONG
from test_supervisor_full_port import coder_writes, failure, incoherent, make, never_wait, rig, stage_2   # noqa: F401  (rig is a fixture)

pytestmark = pytest.mark.usefixtures("stub_tools")


def finish_steps(rig):
    return [e for e in rig.entries() if e["event"] == "decision" and e["stage"] == 2
            and e["data"].get("decision") == "agent step" and e["data"].get("phase") == "finish"]


def derived_decisions(rig):
    return [d for d in rig.decisions() if d.get("decision") == "result derived by the supervisor"]


def not_derived(rig):
    return [d for d in rig.decisions() if d.get("decision") == "result not derived by the supervisor"]


def abort_at_the_pause(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def test_a_passing_hardware_test_finishes_stage_2_without_a_finish_agent_step(rig):
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert finish_steps(rig) == []
    (d,) = derived_decisions(rig)
    assert d["file"] == "stages/2/result.json" and d["coherent"] is True and "form only" in d["coherent_basis"]
    result = json.loads((stage_2(rig) / "result.json").read_text())
    assert result["derived_by"] == "supervisor" and result["runs"] is True
    ev = [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == "result.json derived by the supervisor"]
    assert len(ev) == 1 and ev[0]["path"] == "stages/2/result.json" and len(ev[0]["sha256"]) == 64


def test_a_finish_script_that_would_write_a_bad_result_is_never_asked(rig):
    make(rig, over={(2, "finish"): lambda good: {"result.json": {**good, "top1_agreement": 0.5}}})
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and finish_steps(rig) == []


def test_a_wrong_port_is_never_passed_by_a_derived_result(rig):
    """A port that gets every token wrong also free-runs one repeated word, which is below the readable floor: nothing is derived, the agent
    path runs as before and the gate fails the stage on the recomputed agreement."""
    coder_writes(rig, {}, port=WRONG)
    assert rig.run() == EXIT_BLOCKED
    assert "below the minimum" in failure(rig)["reason"]
    assert derived_decisions(rig) == [] and "readable floor" in not_derived(rig)[0]["why"]


def test_text_below_the_readable_floor_is_left_to_the_agent_and_the_ledger_says_why(rig, monkeypatch):
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "forced by the test"))
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert len(finish_steps(rig)) == 1 and derived_decisions(rig) == []
    assert "readable floor" in not_derived(rig)[0]["why"]
    assert "derived_by" not in json.loads((stage_2(rig) / "result.json").read_text())          # the agent's result


def test_a_failed_hardware_test_is_left_to_the_agent_as_before(rig):
    broken = HONEST.replace("def load(model_dir, mesh_device):", "def load(model_dir, mesh_device):\n    raise RuntimeError('boom')")
    coder_writes(rig, {}, port=broken)
    rig.run()
    assert len(finish_steps(rig)) >= 1 and derived_decisions(rig) == []
    assert "did not pass" in not_derived(rig)[0]["why"]


def test_a_derived_result_that_the_check_refuses_is_deleted_and_the_agent_writes_it(rig, monkeypatch):
    real = full_port_result.derive

    def tampered(stage_dir, run_dir, **kw):
        result, why = real(stage_dir, run_dir, **kw)
        return ({**result, "top1_agreement": 0.1234} if result else None), why
    monkeypatch.setattr(full_port_result, "derive", tampered)
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and len(finish_steps(rig)) == 1 and derived_decisions(rig) == []
    assert "did not pass the gate's check" in not_derived(rig)[0]["why"]
    assert "derived_by" not in json.loads((stage_2(rig) / "result.json").read_text())


def test_a_rehearsal_run_is_derived_with_coherent_false_when_the_text_is_not_readable(rig, monkeypatch):
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "random weights"))
    rig.args.full_port_rehearsal = True
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and finish_steps(rig) == []
    (d,) = derived_decisions(rig)
    assert d["coherent"] is False and "rehearsal" in d["coherent_basis"]
    assert any(d.get("decision") == "stage 2 waived a field" or "waived" in json.dumps(d) for d in rig.decisions())


def test_the_derived_file_is_the_one_the_gate_recomputes_every_number_from(rig):
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    rec = json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())
    result = json.loads((stage_2(rig) / "result.json").read_text())
    assert result["n_tokens"] == len(rec["chip_ids"]) and result["load_s"] == rec["load_s"]
    assert result["top1_agreement"] == pytest.approx(full_port.agreement(rec["reference_ids"], rec["chip_ids"]))


def test_a_derived_result_that_the_ledger_vouching_refuses_ends_the_stage_and_no_agent_is_asked(rig, monkeypatch):
    """The gate also asks the ledger whether the hardware-test marker is the file the supervisor recorded. When that fails for a result the
    supervisor derived, nothing an agent could write would change it: the stage fails at once."""
    from orchard import supervisor
    monkeypatch.setattr(supervisor, "recorded_by_supervisor", lambda *a, **k: False)
    make(rig)
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert "not the file the supervisor recorded" in failure(rig)["reason"] and failure(rig)["code"] == "stage-failed"
    assert finish_steps(rig) == [] and len(derived_decisions(rig)) >= 1


# ---- the derived result and the rehearsal waiver: the audit trail is the one the agent path leaves (review of PR 462 with PR 455) ----------


def waivers(rig):
    return [e for e in rig.entries() if e["event"] == "decision" and e["data"].get("decision") == supervisor.REHEARSAL_WAIVED]


def test_a_rehearsal_result_derived_with_unreadable_text_records_the_waiver_once_for_stage_2(rig, monkeypatch):
    """The gate waives `coherent` for a rehearsal model and the supervisor writes the ledger decision from the gate's own read. That decision
    has to exist exactly once, with the field, the flag and the note the agent path leaves, whoever wrote result.json."""
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "random weights"))
    rig.args.full_port_rehearsal = True
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and finish_steps(rig) == []
    (w,) = waivers(rig)
    assert w["stage"] == 2 and w["data"]["field"] == "coherent" and w["data"]["by"] == "--full-port-rehearsal"
    assert w["data"]["note"] == "the model is a rehearsal with random weights; every other stage 2 check still applied"


def test_a_rehearsal_result_derived_with_readable_text_waives_nothing_because_nothing_needs_it(rig):
    rig.args.full_port_rehearsal = True
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    (d,) = derived_decisions(rig)
    assert d["coherent"] is True and waivers(rig) == [] and finish_steps(rig) == []


def test_without_the_rehearsal_option_unreadable_text_is_never_derived_waived_or_passed(rig, monkeypatch):
    """The derivation must not turn into a way round the coherent check: nothing is written, nothing is waived, the agent path runs, and an
    agent that says the text is incoherent fails the stage exactly as it did before issue 456."""
    monkeypatch.setattr(full_port_result, "readable", lambda text: (False, "forced by the test"))
    make(rig, over={(2, "finish"): incoherent})
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert derived_decisions(rig) == [] and waivers(rig) == []
    assert "readable floor" in not_derived(rig)[0]["why"] and len(finish_steps(rig)) >= 1
    assert failure(rig)["code"] == "stage-failed" and "coherent" in failure(rig)["reason"]
    written = stage_2(rig) / "result.json"
    assert not written.is_file() or "derived_by" not in json.loads(written.read_text())


def test_the_derived_check_runs_the_stages_own_gate_and_not_a_copy_of_its_arguments(rig, monkeypatch):
    """Stage 3 of a full port (another branch) gives its gate the chip count; a derived result there must be judged by that gate. So the
    derived check asks `Supervisor._gate(spec)` for the gate: once for the derived result and once for the stage's own check."""
    calls = []
    real = supervisor.Supervisor._gate

    def spy(self, spec):
        calls.append(spec.number)
        return real(self, spec)
    monkeypatch.setattr(supervisor.Supervisor, "_gate", spy)
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and len(derived_decisions(rig)) == 1
    assert calls.count(2) >= 2, calls


def test_a_gate_that_refuses_without_a_reason_still_refuses_the_derived_result(rig, monkeypatch):
    """`GateResult(ok=False)` with no reasons must not read as 'no reasons, so it passed': the derived file is deleted and the agent runs."""
    from orchard.stages import GateResult
    real = supervisor.Supervisor._gate
    asked = []

    def refuse_first(self, spec):
        asked.append(spec.number)
        if spec.number == 2 and asked.count(2) == 1:        # the derived check is the first gate call of stage 2
            return lambda stage_dir, run_dir: GateResult(False, ())
        return real(self, spec)
    monkeypatch.setattr(supervisor.Supervisor, "_gate", refuse_first)
    make(rig)
    abort_at_the_pause(rig)
    rig.run()
    assert derived_decisions(rig) == [] and "refused" in not_derived(rig)[0]["why"]
    assert len(finish_steps(rig)) == 1 and [d["result"] for d in rig.ends(2)] == ["pass"]
    assert "derived_by" not in json.loads((stage_2(rig) / "result.json").read_text())

