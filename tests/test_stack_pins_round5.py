# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's fourth review (the findings the first summary left out): duplicate architectures, the expected origin before any network
call, the measured gap in the reason, the process group on an interrupt, a class and not any attribute, the flat qwen4_exp_text type, and the
finding details in the ledger and the narration."""
from __future__ import annotations

import importlib.machinery
import json
import os
import subprocess
import sys
import time
import types

import pytest

from orchard import narrate, skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, QWEN4, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)
from test_stack_pins_round2 import real_runner


# ---- duplicates -------------------------------------------------------------------------------------------------------

def test_an_architecture_named_twice_is_checked_once_and_its_missing_class_is_still_reported(tmp_path, py):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM", "Qwen4ExpForCausalLM"], transformers_version="4.0.0")
    src, fake = full(tmp_path, py, model=model)
    probe = probe_says("5.19.0", {"Qwen4ExpForCausalLM": False})
    rec = build(src, fake, probe)
    assert names(rec["hazards"]) == [sp.H_CLASS] and probe.calls[0][4:] == ["Qwen4ExpForCausalLM"]
    assert rec["parts"]["transformers"]["architectures"] == ["Qwen4ExpForCausalLM"]


def test_the_probe_sends_each_name_once_whoever_calls_it(py):
    seen = []
    answer = (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {"A": True, "B": True}}), "")
    out = sp.probe_interpreter(py, ["A", "B", "A", "B", "A"], run=lambda argv, t, e=None: (seen.append(argv), answer)[1])
    assert seen[0][4:] == ["A", "B"] and out["state"] == "answered"


# ---- the expected origin before any network call --------------------------------------------------------------------------

def test_a_checkout_of_another_project_makes_no_network_call(tmp_path, py):
    other = repo(tmp_path, "tt-metal", behind=65, days=42, origin="https://github.com/someone/unrelated.git")
    skills = repo(tmp_path, "skills", behind=65, days=42, origin="https://github.com/someone/skills-fork.git")
    src, fake = full(tmp_path, py, metal=other, skills=skills)
    rec = build(src, fake, GOOD())
    touched = [c for c in fake.calls if c[0] == "gh" or "ls-remote" in c]
    assert not any(other[0] in c or skills[0] in c for c in touched)                 # neither unrelated clone was contacted
    assert any("ls-remote" in c for c in touched)                                  # the plugin, which is the right project, was
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN and "someone/unrelated" in rec["parts"]["tt_metal"]["detail"]


def test_the_expected_repository_is_checked_case_insensitively_before_the_network(tmp_path):
    d, facts = repo(tmp_path, "c", behind=0, origin="https://github.com/Tenstorrent/TT-Metal.git")
    from test_stack_pins import FakeUpstream
    fake = FakeUpstream({d: facts})
    f = skillsfresh.check_clone(d, source="s", runner=skillsfresh.ReadOnlyRunner(fake), cache=skillsfresh.Cache(write=False),
                                update=skillsfresh.skills_update, expected_repo="tenstorrent/tt-metal")
    assert f.state == skillsfresh.CURRENT and f.repo == "Tenstorrent/TT-Metal"


# ---- the measured gap --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("days,shown", [(7.1, "7.1"), (7.04, "7.04"), (42, "42"), (8.5, "8.5")])
def test_the_reason_and_the_record_show_the_measured_gap(tmp_path, py, days, shown):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=3, days=days))
    rec = build(src, fake, GOOD())
    assert rec["parts"]["tt_metal"]["behind_days"] == round(days, 2)
    assert f" {shown} days behind upstream, more than 7 days" in rec["hazards"][0]["reason"]


# ---- an interrupt leaves nothing running ----------------------------------------------------------------------------------

def test_an_interrupt_during_a_call_kills_the_process_group(tmp_path, monkeypatch):
    raw = real_runner()
    pidfile = tmp_path / "pid"
    real_communicate = subprocess.Popen.communicate
    state = {"raised": False}

    def interrupted(self, *a, **k):
        if not state["raised"]:
            state["raised"] = True
            for _ in range(100):                                                  # wait until the child has started its sleep
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                time.sleep(0.05)
            raise KeyboardInterrupt
        return real_communicate(self, *a, **k)
    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        raw(["sh", "-c", f"echo $$ > {pidfile}; sleep 120"], 600, dict(os.environ))
    pid = int(pidfile.read_text().strip())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("the child was still running after the interrupt")


# ---- a class, not any attribute -------------------------------------------------------------------------------------------

def _run_script(monkeypatch, capsys, top, argv, extra_modules=()):
    top.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    monkeypatch.setitem(sys.modules, "transformers", top)
    for name, mod in extra_modules:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "argv", ["-c", *argv])
    exec(compile(sp.PROBE_SCRIPT, "<probe>", "exec"), {})
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])["classes"]


def test_a_string_attribute_such_as___version___is_not_a_model_class(monkeypatch, capsys):
    top = types.ModuleType("transformers")
    top.__version__ = "5.0.0"
    top.RealForCausalLM = type("RealForCausalLM", (), {})
    top.helper = lambda: None
    assert _run_script(monkeypatch, capsys, top, ["-", "__version__", "RealForCausalLM", "helper"]) == {"__version__": False, "RealForCausalLM": True, "helper": False}


def test_a_top_level_attribute_that_is_not_a_class_still_tries_the_model_module(monkeypatch, capsys):
    top = types.ModuleType("transformers")
    top.__version__ = "5.0.0"
    top.Thing = "a string"
    mod = types.ModuleType("transformers.models.m.modeling_m")
    mod.Thing = type("Thing", (), {})
    assert _run_script(monkeypatch, capsys, top, ["m", "Thing"], [("transformers.models.m.modeling_m", mod)]) == {"Thing": True}


def test_a_non_class_attribute_of_the_model_module_is_not_a_model_class_either(monkeypatch, capsys):
    top = types.ModuleType("transformers")
    top.__version__ = "5.0.0"
    mod = types.ModuleType("transformers.models.m.modeling_m")
    mod.Thing = "a string"
    assert _run_script(monkeypatch, capsys, top, ["m", "Thing"], [("transformers.models.m.modeling_m", mod)]) == {"Thing": False}


# ---- the flat qwen4_exp text type -------------------------------------------------------------------------------------------

def test_the_flat_text_form_of_qwen4_exp_has_the_same_minimum_without_a_stated_version(tmp_path, py):
    model = config(tmp_path, model_type="qwen4_exp_text", architectures=["Qwen4ExpForCausalLM"])
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.0.0", {"Qwen4ExpForCausalLM": True}))
    assert names(rec["hazards"]) == [sp.H_BELOW] and "5.19.0" in rec["hazards"][0]["reason"]
    assert build(src, fake, probe_says("5.19.0", {"Qwen4ExpForCausalLM": True}))["severity"] == "ok"


# ---- the details reach the ledger and the narration -----------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path):
    from test_supervisor import Rig
    r = Rig(tmp_path)
    yield r
    r.close()


def test_the_ledger_entry_and_the_narration_carry_the_redacted_reasons(rig):
    from test_supervisor_stack_pins import evidence, run
    from orchard.supervisor import Control
    reason = "transformers 4.57.6 is installed in the reference interpreter and the checkpoint needs 5.19.0 or later"

    def advisory_only(stage_dir, new_run):
        rec = sp.unchecked_record("fake", new_run=new_run)
        rec["advisories"] = [dict(sp._entry(sp.H_BELOW, reason), would_block=True)]
        return rec, sp.write_record(stage_dir, rec)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, advisory_only)
    (entry,) = evidence(rig)
    assert entry["findings"] == [{"name": sp.H_BELOW, "reason": reason}]
    ledger_entry = next(e for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == "stack pins")
    text = " ".join(l.text for l in narrate.lines_for(ledger_entry))
    assert "4.57.6" in text and "5.19.0" in text and sp.H_BELOW in text
