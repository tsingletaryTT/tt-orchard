# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 15: `--agreement-reference`, the operator's way to ask stage 1 for the agreement set's CPU reference.

Like `--fidelity-tokens` (issue 789), the supervisor cannot write stage 1's `reference_config.json` (the stage agent does, from the skill), so the
choice travels as a run fact: the supervisor records it in run_start, the stage contexts list it, and the reference-gate skill tells the agent to
set `"agreement_set": true` and to copy the shipped prompt file next to the script. A run that does not name the option has the files it had
before; a resume keeps the recorded choice and refuses a different one. The stage 1 gate then requires the file the run asked for, so a run that
asked for the set never falls back to the 32-token check without saying so."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from orchard import bringup_config as bc
from orchard import context, full_port, stages, supervisor
from orchard.ledger import Ledger
from orchard.supervisor import Control, parse
from run_fakes import argv
from test_bringup_config import load
from test_cli import conf, run as cli_run  # noqa: F401  (the `conf` fixture)
from test_supervisor import Rig

REPO = Path(__file__).resolve().parent.parent
SHIPPED = REPO / "orchard" / "skills" / "weights-swap-templates" / "agreement_prompts.json"


def test_parse_has_the_flag_default_none(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).agreement_reference is None
    assert parse(a + ["--agreement-reference"]).agreement_reference is True


def test_the_bringup_config_adds_the_flag_only_when_named(tmp_path):
    cfg = load(tmp_path)
    assert "--agreement-reference" not in bc.supervisor_argv(cfg, "Cloudflare/clef")
    assert "--agreement-reference" not in bc.supervisor_argv(cfg, "Cloudflare/clef", agreement_reference=None)
    out = bc.supervisor_argv(cfg, "Cloudflare/clef", agreement_reference=True)
    assert "--agreement-reference" in out and supervisor.parse(out).agreement_reference is True


def test_the_cli_passes_the_flag_to_the_supervisor_only_when_given(conf):
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef"])
    assert "--agreement-reference" not in rec.argv
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--agreement-reference"])
    assert code == 0 and "--agreement-reference" in rec.argv


def test_the_run_facts_list_the_choice_only_when_the_run_names_it(tmp_path):
    assert "agreement reference" not in context.facts_from({"model": "m"}, tmp_path)
    assert context.facts_from({"model": "m", "agreement_reference": True}, tmp_path)["agreement reference"] == "on"


def test_the_skill_tells_the_stage_agent_what_to_do_with_the_fact():
    text = (REPO / "orchard" / "skills" / "reference-gate.md").read_text()
    assert '"agreement_set": true' in text and "agreement reference" in text and "agreement_prompts.json" in text


def test_the_readme_names_the_flag():
    assert "--agreement-reference" in (REPO / "README.md").read_text()


# ---- the supervisor ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def run_start(rig):
    return next(e for e in rig.entries() if e["event"] == "run_start")["data"]


@pytest.fixture
def built(monkeypatch):
    import test_supervisor
    from orchard.supervisor import build
    made = []
    monkeypatch.setattr(test_supervisor, "build", lambda *a, **k: made.append(build(*a, **k)) or made[-1])
    return made


def start(rig, **kw):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(**kw)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_run_with_no_option_records_nothing_new(rig, built):
    start(rig)
    assert "agreement_reference" not in run_start(rig) and built[-1].agreement_reference is False


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_the_choice_is_recorded_in_run_start_when_named(rig, built):
    rig.args.agreement_reference = True
    start(rig)
    assert run_start(rig)["agreement_reference"] is True and built[-1].agreement_reference is True


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_resume_keeps_the_recorded_choice(rig, built):
    rig.args.agreement_reference = True
    start(rig)
    rig.args.agreement_reference = None
    start(rig, pid=200)
    assert built[-1].agreement_reference is True


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_run_that_started_without_it_refuses_a_resume_that_names_it(rig, built):
    start(rig)
    rig.args.agreement_reference = True
    with pytest.raises(ValueError, match="agreement reference"):
        rig.run(pid=200)


# ---- the stage 1 gate --------------------------------------------------------------------------------------------------------

def shipped_ids():
    return [p["id"] for p in json.loads(SHIPPED.read_text())["prompts"]]


def make_run(tmp_path, *, asked, reference="good"):
    """A run directory whose ledger says the run asked (or did not ask) for the agreement reference, with a stage 1 directory."""
    run = tmp_path / "run"
    stage = run / "stages" / "1"
    (stage / "evidence" / "reference").mkdir(parents=True)
    led = Ledger(run / "ledger.jsonl")
    try:
        led.append("run_start", None, model="m", **({"agreement_reference": True} if asked else {}))
    finally:
        led.close()
    shutil.copy(SHIPPED, stage / "agreement_prompts.json")
    if reference is not None:
        doc = {"prompt_set_sha256": hashlib.sha256(SHIPPED.read_bytes()).hexdigest(), "n_new_tokens": 32,
               "entries": [{"id": i, "kind": "chat", "prompt_ids": [1, 2, 3], "generated_ids": list(range(32))} for i in shipped_ids()]}
        if reference == "short":
            doc["entries"][2]["generated_ids"] = list(range(10))
        elif reference == "wrong-set":
            doc["prompt_set_sha256"] = "0" * 64
        elif reference == "missing-prompt":
            doc["entries"].pop()
        elif reference == "not-json":
            (stage / "evidence" / "reference" / "agreement-reference.json").write_text("{")
            return stage, run
        (stage / "evidence" / "reference" / "agreement-reference.json").write_text(json.dumps(doc))
    return stage, run


def reasons(stage, run):
    return stages._agreement_reference_reasons(stage, run)


def test_a_run_that_did_not_ask_is_not_checked_even_without_the_file(tmp_path):
    stage, run = make_run(tmp_path, asked=False, reference=None)
    assert reasons(stage, run) == []


def test_a_run_that_asked_and_has_a_good_file_passes(tmp_path):
    stage, run = make_run(tmp_path, asked=True)
    assert reasons(stage, run) == []


def test_a_run_that_asked_and_has_no_file_fails_and_says_to_look_at_the_error_file(tmp_path):
    stage, run = make_run(tmp_path, asked=True, reference=None)
    out = reasons(stage, run)
    assert len(out) == 1 and "agreement-reference.json" in out[0] and "agreement-reference-error.json" in out[0]


@pytest.mark.parametrize("kind,word", [("short", "32"), ("wrong-set", "prompt set"), ("missing-prompt", "missing"), ("not-json", "cannot be read")])
def test_a_run_that_asked_and_has_a_bad_file_fails_with_the_reason(tmp_path, kind, word):
    stage, run = make_run(tmp_path, asked=True, reference=kind)
    out = reasons(stage, run)
    assert out and any(word in r for r in out), out


def test_a_prompt_file_that_is_not_the_shipped_one_fails(tmp_path):
    stage, run = make_run(tmp_path, asked=True)
    (stage / "agreement_prompts.json").write_text('{"prompts": []}')
    assert any("agreement_prompts.json" in r for r in reasons(stage, run))


def test_the_file_is_read_through_safeio_so_a_link_is_refused(tmp_path):
    stage, run = make_run(tmp_path, asked=True)
    target = stage / "evidence" / "reference" / "agreement-reference.json"
    real = tmp_path / "elsewhere.json"
    real.write_text(target.read_text())
    target.unlink()
    target.symlink_to(real)
    assert any("cannot be read" in r for r in reasons(stage, run))


def test_the_stage_1_gate_itself_refuses_a_run_that_asked_and_has_no_file(tmp_path):
    """Through gate_reference, so the check is wired in and not only correct when called."""
    stage, run = make_run(tmp_path, asked=True, reference=None)
    (stage / "reference.json").write_text(json.dumps({"verdict": "pass", "checks": []}))
    gate = stages.gate_reference(stage, run)
    assert not gate.ok and any("agreement-reference.json" in r for r in gate.reasons), gate.reasons


def test_the_stage_1_gate_does_not_mention_the_reference_for_a_run_that_did_not_ask(tmp_path):
    stage, run = make_run(tmp_path, asked=False, reference=None)
    (stage / "reference.json").write_text(json.dumps({"verdict": "pass", "checks": []}))
    assert not any("agreement" in r for r in stages.gate_reference(stage, run).reasons)
