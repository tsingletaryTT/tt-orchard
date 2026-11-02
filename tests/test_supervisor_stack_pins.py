# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 and the stack freshness record (orchard/stack_pins.py, orchard/supervisor.py). The supervisor, the stage machine, the ledger and the
gate are the real ones; the comparison with upstream is a fake that returns a record the test chooses (no git, no gh, no interpreter)."""
import hashlib
import json

import pytest

from orchard import blocked, narrate
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


def test_a_record_that_cannot_be_written_is_a_decision_and_blocks_stage_0_at_once(rig):
    def broken(stage_dir, new_run):
        raise OSError("disk full")
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, broken)
    assert any(d.get("decision") == "stack pins not written" for d in rig.decisions())
    pause = next(d["reason"] for d in rig.decisions() if d.get("decision") == "pause")
    assert "could not be written" in pause and blocked.block_code({"reason": pause}) == "stack-stale"
    assert not [e for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]


def test_an_error_message_with_a_credential_is_scrubbed_before_it_reaches_the_ledger(rig):
    def broken(stage_dir, new_run):
        raise OSError("cannot reach https://user:realtoken123@github.com/x/y.git with hf_abcdefgh12345678")
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, broken)
    ledger_text = (rig.run_dir / "ledger.jsonl").read_text()
    assert "stack pins not written" in ledger_text and "realtoken123" not in ledger_text and "hf_abcdefgh12345678" not in ledger_text


def test_a_resumed_stage_does_not_keep_the_last_attempts_record_when_the_new_write_fails(rig):
    """The stage directory survives a resume with the old record and its ledger hash. If this attempt's write fails, the gate must not
    accept the old evidence: the old file is removed before the write, and the stage blocks."""
    abort_when_waiting(rig)
    run(rig, fake_check([], hazards=QWEN4_HAZARDS))
    record = rig.run_dir / "stages" / "0" / sp.FILE
    assert record.is_file()
    rig.args.resume_aborted = True

    def broken(stage_dir, new_run):
        assert not record.exists()          # removed before the check ran
        raise OSError("disk full")
    run(rig, broken)
    assert not record.exists()
    assert not [e for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]
    pause = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"][-1]
    assert "could not be written" in pause


def test_a_kept_stage_directory_with_an_old_record_and_ledger_hash_is_not_accepted_when_the_new_write_fails(rig):
    """The case the full run above does not reach: the stage directory is kept (a stage that resumes with RESUME.md), so the old record is
    still there and the ledger still holds its hash. The old file goes before the check, the failed write blocks, and the gate refuses."""
    from orchard import supervisor
    from orchard.handoff import Blocked
    abort_when_waiting(rig)
    run(rig, fake_check([]))
    stage_dir = rig.run_dir / "stages" / "0"
    assert (stage_dir / sp.FILE).is_file()
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m), home=rig.home,
                    containers=rig.containers)

        def broken(d, new_run):
            raise OSError("disk full")
        sup.stack_pins = broken
        old_hash_in_ledger = sup._stack_pins_sha(stage_dir)
        assert old_hash_in_ledger
        with pytest.raises(Blocked):
            sup._stage_0_stack_pins(stage_dir)
        assert not (stage_dir / sp.FILE).exists()
        gate = sup._with_stack_pins(lambda s, r: supervisor.GateResult(True))(stage_dir, rig.run_dir)
        assert not gate.ok


def test_stage_0_does_not_pass_when_the_file_is_not_the_one_the_ledger_recorded(rig):
    """The gate is wired into the stage: a file whose hash is not the ledger's (swapped after the supervisor recorded it) fails the stage."""
    def swapped(stage_dir, new_run):
        rec, _ = fake_check([])(stage_dir, new_run)
        return rec, "0" * 64
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, swapped)
    ends = [e["data"] for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]
    assert ends and ends[0]["result"] != "pass" and any("hash changed" in r for r in ends[0]["reasons"])


STALE_PATH_REASON = [sp._entry(sp.H_TT_METAL, "tt-metal is 65 commits and 42 days behind upstream, more than 7 days. To update: git -C /data/chips/tt-metal pull --ff-only origin main.")]


def test_a_stack_block_has_its_own_code_even_when_its_text_holds_hardware_words(rig):
    abort_when_waiting(rig)
    run(rig, fake_check([], hazards=STALE_PATH_REASON))
    pause = next(d["reason"] for d in rig.decisions() if d.get("decision") == "pause")
    assert "chips" in pause and "tt-metal" in pause
    code = blocked.block_code({"reason": pause})
    assert code == "stack-stale" and code in blocked.REASONS and code not in blocked.AUTO_RETRY
    assert not blocked.retryable(code, pause) and narrate.UNBLOCK[code]


def test_the_real_check_looks_where_the_run_says(monkeypatch, tmp_path):
    """PR 381 review: the tt-metal tree is under the run's operator home (as preflight reads it), and the reference interpreter is the run's
    `reference_python` input before the environment variable."""
    from types import SimpleNamespace
    from orchard import supervisor
    got = []
    kws = []
    monkeypatch.setattr(sp, "make_check", lambda sources, **kw: (got.append(sources), kws.append(kw)))
    sup = SimpleNamespace(paths=SimpleNamespace(operator_home=str(tmp_path / "ops-home")), home=tmp_path / "invoking-home",
                          inputs={"model": "/m", "reference_python": "/cfg/python"}, run_dir=tmp_path / "run", extra_env={"FOO": "bar"})
    args = SimpleNamespace(skills_dir=[tmp_path / "skills"])
    monkeypatch.delenv("ORCHARD_TT_METAL_DIR", raising=False)
    supervisor.stack_check_for(sup, args, environ={"ORCHARD_REFERENCE_PYTHON": "/env/python"})
    sup.inputs = {"model": "/m"}
    supervisor.stack_check_for(sup, args, environ={"ORCHARD_REFERENCE_PYTHON": "/env/python", "ORCHARD_TT_METAL_DIR": "/named/tree"})
    first, second = got
    assert first.tt_metal_dir == str(tmp_path / "ops-home" / "tt-metal") and first.ref_python == "/cfg/python"
    assert second.tt_metal_dir == "/named/tree" and second.ref_python == "/env/python"
    assert first.skills_dirs == (str(tmp_path / "skills"),) and first.model_dir == "/m"
    # the probe runs under the environment the reference step uses (agent_env, reused: HOME inside the run directory, the run's --env values)
    from orchard.agent import agent_env
    assert kws[0]["probe_env"] == agent_env(tmp_path / "run", extra={"FOO": "bar"}) and kws[0]["probe_env"]["HOME"].startswith(str(tmp_path / "run"))
    assert first.exclude_paths == (str(tmp_path / "run"),) and str(tmp_path / "ops-home") in first.homes


def test_the_keyword_rules_still_classify_other_blocks_the_same_way():
    assert blocked.block_code({"reason": "blocked: a chip is held by another lease"}) == "hardware-unhealthy"
    assert blocked.block_code({"reason": "blocked: the model does not fit this box"}) == "model-does-not-fit"


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
