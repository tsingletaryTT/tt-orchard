# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of the full-port route in the supervisor (issue 329): the port that passed stage 2 sits behind the chat server, the acceptance corpus is
replayed at boot, and the passkey and canary are asked. The chip, the coder and the model are fakes; the supervisor, the stage machine, the gate and
the hardware-test runner are the real ones, and the supervisor's template runs for real on the CPU against a fake `ttnn`. The run reaches stage 5 by
an attended resume past the pause before stage 4 (stage 4 and the stages after stage 5 are the generic scripted ones)."""
import json

import pytest

from orchard import full_port, full_port_serve
from orchard.stages import FULL_PORT_STAGE_5, spec_for
from orchard.supervisor import EXIT_BLOCKED, EXIT_READY, EXIT_ABORTED, REHEARSAL_WAIVED, Control
import port_serve_fakes as fk
from run_fakes import note
from test_supervisor import Rig, step_skills
from test_supervisor_full_port import FULL_PORT, PREPARE, make, started, with_snapshot

pytestmark = pytest.mark.usefixtures("stub_tools")

KEYS = [{"length": 256, "key": "48213"}, {"length": 1984, "key": "90417"}]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    # The keys come from the corpus hash, which holds the accepted record's hash, which holds a load time: they differ from run to run, and the fake
    # tokenizer needs them in its vocabulary. The plan function is patched where the supervisor and the gate both read it.
    monkeypatch.setattr(full_port_serve, "plan_passkeys", lambda corpus_sha, max_len: [dict(k) for k in KEYS])
    fk.tokenizer_with(tmp_path / "snapshot" / "tokenizer.json", [k["key"] for k in KEYS])
    yield r
    r.close()


def go(rig, port=fk.ANSWERING, *, over=None):
    """Script the coder's stage 2 prepare with `port` (a template of fakes) and resume past the pause before stage 4."""
    src = fk.port_source(port, rig.tmp / "snapshot")
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": src}, **(over or {})})
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")


def stage_5_tests(rig):
    return [e["data"] for e in rig.entries() if e["stage"] == 5 and e["event"] == "decision" and e["data"].get("decision") == "hardware test started"]


def decided(rig, n, name):
    return [d for e in rig.entries() if e["stage"] == n and e["event"] == "decision" for d in [e["data"]] if d.get("decision") == name]


def test_the_table_gives_a_full_port_stage_5_its_own_skill_gate_and_no_coder():
    s = spec_for(5, "full-port", None, "full-port", full_port=True)
    assert s is FULL_PORT_STAGE_5 and s.skill == "full-port-serving" and s.draft == "full_port_serve" and s.needs_coder is False and s.boards == 1
    assert spec_for(5, "full-port", None, "full-port").skill == "serving-check"                  # without --full-port the generic stage stays
    assert spec_for(5, "full-port", None, "unknown", full_port=True).skill == "serving-check"
    assert spec_for(5, "weights-only", None, "weights-only", full_port=True).skill != "full-port-serving"


def test_the_skill_file_exists_and_is_a_marked_draft_with_the_tables_name():
    text = (full_port_serve.TEMPLATE.parent.parent / "full-port-serving.md").read_text()
    assert text.startswith("---\nname: full-port-serving\n") and "status: draft" in text


def test_stage_5_serves_the_port_replays_the_corpus_and_passes(rig):
    go(rig)
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(5)] == ["pass"]
    stage = rig.run_dir / "stages" / "5"
    marker = json.loads((stage / "test-result.json").read_text())
    rec = json.loads((stage / "evidence" / "port-serve.json").read_text())
    assert marker["returncode"] == 0 and marker["command"] == full_port_serve.COMMAND
    assert sorted(rec["chips"]) == sorted(marker["chips"]) and len(rec["chips"]) == 2          # stage 2's one board
    assert rec["boot"]["ready_line"].startswith("READY orchard-port on 127.0.0.1:")
    result = json.loads((stage / "result.json").read_text())
    assert result["corpus_positions"] == result["corpus_matched"] == 80 and result["coherent"] is True and result["derived_by"] == "supervisor"
    ev = {e["path"] for e in rig.ends(5)[0]["evidence"]}
    assert {"stages/5/evidence/port-serve.json", "stages/5/corpus.json", "stages/5/evidence/hw-test-output.txt", "stages/5/result.json"} <= ev


def test_stage_5_asks_no_agent_and_needs_no_coder_step(rig):
    go(rig)
    rig.run()
    assert step_skills(rig, 5) == []
    assert len(stage_5_tests(rig)) == 1 and len(stage_5_tests(rig)[0]["chips"]) == 2


def test_the_ledger_holds_the_corpus_the_copied_port_and_the_templates(rig):
    go(rig)
    rig.run()
    corpus = decided(rig, 5, "serving corpus recorded")[0]
    stage = rig.run_dir / "stages" / "5"
    import hashlib
    assert corpus["sha256"] == hashlib.sha256((stage / "corpus.json").read_bytes()).hexdigest() and corpus["positions"] == 80 and corpus["source_stage"] == 2
    copied = decided(rig, 5, "serving port copied")[0]
    assert [f["path"] for f in copied["files"]] == ["stages/5/port/port_model.py"]
    assert copied["files"][0]["sha256"] == hashlib.sha256((stage / "port" / "port_model.py").read_bytes()).hexdigest()
    pinned = decided(rig, 5, "serving check templates copied")[0]
    assert pinned["template"] == full_port_serve.template_sha256() and pinned["worker"] == full_port_serve.worker_sha256()
    assert (stage / "hw_test.json").read_text().strip() == json.dumps({"command": full_port_serve.COMMAND, "deadline_s": full_port_serve.DEADLINE_S})


def test_the_served_port_is_byte_for_byte_the_stage_2_port(rig):
    go(rig)
    rig.run()
    a, b = (rig.run_dir / "stages" / n / "port" / "port_model.py" for n in ("2", "5"))
    assert a.read_bytes() == b.read_bytes()


def test_a_port_that_answers_differently_behind_the_server_is_refused_and_the_run_does_not_reach_stage_6(rig):
    """Stage 2's check passes (the flag file is absent). Before stage 5 the file appears; the port, run behind the server, answers 7 at the first
    position, which differs from the corpus, so the server is refused at boot, on both attempts."""
    flag = rig.tmp / "flip"
    flipping = fk.ANSWERING.replace("def prefill(model, ids):\n", f"def prefill(model, ids):\n    if __import__('os').path.exists({str(flag)!r}):\n        return _peak(7)\n")
    go(rig, flipping)
    calls = []

    def on_sleep():
        calls.append(1)
        if len(calls) == 1:
            flag.write_text("x")
            Control(rig.run_dir).write("resume")
        else:
            Control(rig.run_dir).write("abort")
    rig.on_sleep = on_sleep
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and [d["result"] for d in rig.ends(5)] == ["escalate", "fail"]
    out = (rig.run_dir / "stages" / "5" / "evidence" / "hw-test-output.txt").read_text()
    assert "FAILED: the corpus replay matched" in out and "READY" not in out
    assert step_skills(rig, 5) == [] and not started(rig, 6)
    failing = " ".join(rig.ends(5)[0]["reasons"])
    assert "exited with code 2" in failing


def test_a_model_with_random_weights_fails_stage_5_without_the_rehearsal_option(rig):
    go(rig, fk.RANDOM_WEIGHTS)
    sleeps = []
    rig.on_sleep = lambda: (sleeps.append(1), Control(rig.run_dir).write("resume" if len(sleeps) == 1 else "abort"))      # resume past the pause before stage 4, then stop at the failure
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(5)] == ["escalate", "fail"]
    assert any("coherent must be true" in r for r in rig.ends(5)[0]["reasons"])
    assert not started(rig, 6)


def test_the_rehearsal_option_lets_the_random_weight_model_pass_stage_5_and_the_ledger_says_what_was_waived(rig):
    go(rig, fk.RANDOM_WEIGHTS)
    rig.args.full_port_rehearsal = True
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(5)] == ["pass"]
    waived = decided(rig, 5, REHEARSAL_WAIVED)
    assert len(waived) == 1 and waived[0]["field"] == "coherent" and "stage 5" in waived[0]["note"]
    result = json.loads((rig.run_dir / "stages" / "5" / "result.json").read_text())
    assert result["coherent"] is False and result["corpus_matched"] == result["corpus_positions"] == 80
    assert result["checks"]["passkey"]["answered"] == [False, False]


def test_the_operator_bundle_lists_the_stage_5_checks_with_their_labels(rig):
    go(rig, fk.RANDOM_WEIGHTS)
    rig.args.full_port_rehearsal = True
    assert rig.run() == EXIT_READY
    text = (rig.run_dir / "stages" / "8" / "bundle" / "RESULTS.md").read_text()
    assert all(f"| {name} | yes |" in text for name in ("boots", "corpus", "passkey", "canary"))
    assert any(row.startswith("| ready_s | ") and "| measured |" in row for row in text.splitlines())
    assert any(row.startswith("| corpus_positions | 80 | measured |") for row in text.splitlines())


def test_stage_5_serves_the_stage_3_port_on_the_four_configured_chips_when_stage_3_ran(rig, monkeypatch):
    # The generic stage 4 would ask for a 4-chip mesh-shrink entry that the scripted coder does not write; stage 4 is not this slice's, so it is skipped.
    import dataclasses
    from orchard import supervisor as sup_mod
    real = sup_mod.spec_for
    monkeypatch.setattr(sup_mod, "spec_for", lambda n, *a, **k: dataclasses.replace(real(n, *a, **k), skip="skipped by the test") if n == 4 else real(n, *a, **k))
    rig.args.required_chips = (2, 4)
    go(rig)
    src = fk.port_source(fk.ANSWERING, rig.tmp / "snapshot")
    from test_supervisor_full_port_stage3 import PREPARE_3
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": src}, (3, "prepare"): {**PREPARE_3, "port/port_model.py": src}})
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(3)] == ["pass"] and [d["result"] for d in rig.ends(5)] == ["pass"]
    ran = stage_5_tests(rig)
    assert len(ran) == 1 and len(ran[0]["chips"]) == 4
    assert decided(rig, 5, "serving corpus recorded")[0]["source_stage"] == 3
    assert json.loads((rig.run_dir / "stages" / "5" / "evidence" / "port-serve.json").read_text())["source_stage"] == 3


def test_a_missing_model_snapshot_fails_the_stage_with_the_reason_and_runs_no_test(rig):
    go(rig)
    args = rig.args
    args.input = [i for i in (args.input or []) if not str(i).startswith("model=")]
    from orchard import full_port_serve_stage
    out = full_port_serve_stage.prepare(rig.tmp, rig.tmp, [], snapshot=None, orchard_dir=fk.REPO)
    assert "no model snapshot" in out.problems[0]


def test_the_gate_reads_what_it_judges_against_from_the_ledger_and_never_from_the_stage_directory(rig):
    go(rig)
    rig.run()
    from orchard import ledger as ledger_mod
    from orchard.supervisor import Supervisor
    entries = rig.entries()
    assert Supervisor._serve_corpus_sha(entries) == decided(rig, 5, "serving corpus recorded")[0]["sha256"]
    assert full_port_serve.source_stage(entries) == 2
    # a corpus decision written by an agent for another stage is not taken
    forged = entries + [{"event": "decision", "stage": 2, "data": {"decision": "serving corpus recorded", "sha256": "f" * 64}}]
    assert Supervisor._serve_corpus_sha(forged) != "f" * 64
