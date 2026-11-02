# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 and the stack freshness record (orchard/stack_pins.py, orchard/supervisor.py). The supervisor, the stage machine, the ledger and the
gate are the real ones; the comparison with upstream is a fake that returns a record the test chooses (no git, no gh, no interpreter)."""
import hashlib
import json

import pytest

from orchard import stack_pins as sp
from orchard.ledger import Ledger
from orchard.supervisor import Control, build
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

QWEN4_HAZARDS = [sp._entry(sp.H_BELOW, "transformers 4.57.6 is installed in the reference interpreter and the checkpoint needs 5.19.0 or later"),
                 sp._entry(sp.H_CLASS, "transformers 4.57.6 has no class Qwen4ExpForCausalLM that the checkpoint's architectures name")]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def fake_check(calls, *, hazards=(), advisories=()):
    """A comparison that records each call and writes a record the way make_check does."""
    def check(stage_dir, new_run):
        calls.append(new_run)
        h = list(hazards) if new_run else []
        a = ([dict(e, would_be_critical=True) for e in hazards] if not new_run else []) + list(advisories)
        rec = sp.unchecked_record("fake", new_run=new_run)
        rec.update(hazards=h, advisories=a, severity="critical" if h else ("advisory" if a else "ok"), label="fake")
        return rec, sp.write_record(stage_dir, rec)
    return check


def run(rig, check, pid=100):
    rig.m.owner_pid = pid
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=pid), coder=rig.coder_cls(rig.m), versions={"tt_model": "test"},
                    clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers,
                    stack_pins=check)
        return sup.run()


def evidence(rig):
    return [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == "stack pins"]


def abort_when_waiting(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def test_a_new_run_with_a_critical_hazard_blocks_stage_0_with_the_named_reasons_and_starts_no_agent_step(rig):
    calls = []
    abort_when_waiting(rig)
    run(rig, fake_check(calls, hazards=QWEN4_HAZARDS))
    pause = next(d["reason"] for d in rig.decisions() if d.get("decision") == "pause")
    assert pause.startswith("blocked: the stack is stale for a new run") and sp.H_BELOW in pause and sp.H_CLASS in pause and "5.19.0" in pause
    (entry,) = evidence(rig)
    assert entry["severity"] == "critical" and entry["mode"] == "new" and entry["hazards"] == [sp.H_BELOW, sp.H_CLASS]
    assert not [e for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]
    notice = next(e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked"))
    assert notice["evidence"]["hazards"] == [sp.H_BELOW, sp.H_CLASS]


def test_the_ledger_holds_the_hash_of_the_bytes_that_were_written(rig):
    abort_when_waiting(rig)
    run(rig, fake_check([], hazards=QWEN4_HAZARDS))
    (entry,) = evidence(rig)
    data = (rig.run_dir / entry["path"]).read_bytes()
    assert entry["path"] == "stages/0/stack-pins.json" and entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_running_the_command_again_is_a_resume_and_the_same_findings_are_advisories(rig):
    calls = []
    check = fake_check(calls, hazards=QWEN4_HAZARDS)
    abort_when_waiting(rig)
    run(rig, check)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    # the abort ends the first run; a second command resumes it after the abort
    rig.args.resume_aborted = True
    run(rig, check)
    assert calls[0] is True and calls[-1] is False
    last = evidence(rig)[-1]
    assert last["mode"] == "resume" and last["severity"] == "advisory" and last["hazards"] == []
    assert last["advisories"] == [sp.H_BELOW, sp.H_CLASS]


def test_a_clean_new_run_goes_through_stage_0_and_the_gate_reads_the_record(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, fake_check([], advisories=[sp._entry("tt_metal_checkout_stale_unchecked", "upstream could not be reached")]))
    (end,) = [e["data"] for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]
    assert end["result"] == "pass"
    (entry,) = evidence(rig)
    assert entry["severity"] == "advisory" and entry["advisories"] == ["tt_metal_checkout_stale_unchecked"]


def test_no_supplied_check_writes_an_unchecked_record_that_never_claims_a_pass(rig):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, None)
    rec = json.loads((rig.run_dir / "stages" / "0" / sp.FILE).read_text())
    assert rec["severity"] == "advisory" and rec["label"] == "no comparison was made" and rec["hazards"] == []
    assert all(p["state"] == "unknown" for k, p in rec["parts"].items() if k != "skills")


def test_the_stage_0_gate_refuses_a_record_rewritten_after_the_ledger_entry(rig):
    """The agent's shell could edit the file after the supervisor recorded it; the gate compares with the ledger's hash."""
    from orchard import supervisor
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, fake_check([]))
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        spec = next(s for s in __import__("orchard.stages", fromlist=["STAGES"]).STAGES if s.number == 0)
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m), home=rig.home, containers=rig.containers)
        stage_dir = rig.run_dir / "stages" / "0"
        before = sup._with_stack_pins(lambda s, r: supervisor.GateResult(True))(stage_dir, rig.run_dir)
        assert before.ok
        (stage_dir / sp.FILE).write_text(json.dumps({"schema": 1, "severity": "ok", "mode": "new", "hazards": [], "advisories": [], "parts": {}}))
        after = sup._with_stack_pins(lambda s, r: supervisor.GateResult(True))(stage_dir, rig.run_dir)
        assert not after.ok and any("hash changed" in r for r in after.reasons)
        (stage_dir / sp.FILE).unlink()
        gone = sup._with_stack_pins(lambda s, r: supervisor.GateResult(True))(stage_dir, rig.run_dir)
        assert not gone.ok and any("cannot be used" in r for r in gone.reasons)
    del spec


def test_a_record_that_cannot_be_written_is_a_decision_and_the_gate_fails_the_stage(rig):
    def broken(stage_dir, new_run):
        raise OSError("disk full")
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, broken)
    assert any(d.get("decision") == "stack pins not written" for d in rig.decisions())
    assert not [e for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0 and e["data"].get("result") == "pass"]


def test_an_attended_run_that_the_operator_resumes_in_the_same_process_holds_the_findings_as_advisories(rig):
    """The block waits for the operator; `resume` goes on in the same supervisor, which has no earlier run behind it. The earlier evidence entry
    for this stage is what makes the second attempt a resume."""
    calls, sent = [], []

    def operator():
        sent.append(1)
        Control(rig.run_dir).write("resume" if len(sent) == 1 else "abort")
    rig.on_sleep = operator
    run(rig, fake_check(calls, hazards=QWEN4_HAZARDS))
    assert calls[:2] == [True, False]
    assert evidence(rig)[-1]["mode"] == "resume" and evidence(rig)[-1]["hazards"] == []
