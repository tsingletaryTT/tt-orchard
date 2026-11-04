# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 393: the findings of Copilot's seventh review of PR 381 that were left open (stack freshness record, orchard/stack_pins.py).

1. The probe does not run an interpreter that lies inside the run directory (an agent can write there).
2. A part is not `current` when one of its checks was not made (no minimum, an unreadable one, no class to check).
3. A malformed `transformers_version` is reported even when the table supplies a minimum.
4. A resumed run's findings keep their reasons in the ledger entry and in the narration.
5. The narration keeps the reasons of findings that share a name (several stale skills clones).
"""
from __future__ import annotations

import json
import os

import pytest

from orchard import narrate, skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)


# ---- 1. an interpreter the agent can write is not run ---------------------------------------------------------------------

def _inside_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "venv" / "bin").mkdir(parents=True)
    return run_dir


def test_an_interpreter_inside_the_run_directory_is_not_run(tmp_path):
    run_dir = _inside_run_dir(tmp_path)
    py_in = run_dir / "venv" / "bin" / "python"
    py_in.write_text("#!/bin/sh\n")
    py_in.chmod(0o755)
    probe = GOOD()
    out = sp.probe_interpreter(str(py_in), ["LlamaForCausalLM"], probe, "llama", None, (str(run_dir),))
    assert out["state"] == "unknown" and "run directory" in out["why"] and probe.calls == []


def test_a_link_outside_the_run_directory_that_points_into_it_is_not_run(tmp_path):
    run_dir = _inside_run_dir(tmp_path)
    real = run_dir / "venv" / "bin" / "python"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    link = tmp_path / "python-link"
    os.symlink(real, link)
    probe = GOOD()
    out = sp.probe_interpreter(str(link), ["LlamaForCausalLM"], probe, "llama", None, (str(run_dir),))
    assert out["state"] == "unknown" and "run directory" in out["why"] and probe.calls == []


def test_an_interpreter_reached_through_a_linked_directory_that_points_into_the_run_directory_is_not_run(tmp_path):
    run_dir = _inside_run_dir(tmp_path)
    exe = run_dir / "venv" / "bin" / "python"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    linked_bin = tmp_path / "linked-bin"
    os.symlink(run_dir / "venv" / "bin", linked_bin)
    probe = GOOD()
    out = sp.probe_interpreter(str(linked_bin / "python"), ["LlamaForCausalLM"], probe, "llama", None, (str(run_dir),))
    assert out["state"] == "unknown" and probe.calls == []


def test_an_interpreter_outside_the_run_directory_is_still_run(tmp_path, py):
    run_dir = _inside_run_dir(tmp_path)
    probe = GOOD()
    out = sp.probe_interpreter(py, ["LlamaForCausalLM"], probe, "llama", None, (str(run_dir),))
    assert out["state"] == "answered" and len(probe.calls) == 1


def test_a_sibling_directory_with_the_same_prefix_is_not_the_run_directory(tmp_path, py):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sibling = tmp_path / "run-tools"
    sibling.mkdir()
    exe = sibling / "python"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert sp.probe_interpreter(str(exe), ["A"], probe_says("1.0.0", {"A": True}), "m", None, (str(run_dir),))["state"] == "answered"


def test_the_record_says_why_an_interpreter_in_the_run_directory_was_not_checked(tmp_path):
    run_dir = _inside_run_dir(tmp_path)
    exe = run_dir / "venv" / "bin" / "python"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    src, fake = full(tmp_path, str(exe))
    src = sp.Sources(**{**src.__dict__, "exclude_paths": (str(run_dir),)})
    rec = build(src, fake, GOOD())
    assert rec["parts"]["transformers"]["state"] == skillsfresh.UNKNOWN
    assert "transformers_unchecked" in names(rec["advisories"])


# ---- 2. a part is not current when a check was not made ---------------------------------------------------------------------

def test_a_config_with_a_class_and_no_readable_minimum_is_unknown_with_its_own_finding(tmp_path, py):
    model = config(tmp_path, model_type="llama", architectures=["LlamaForCausalLM"])        # no transformers_version, no table entry
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("4.57.6", {"LlamaForCausalLM": True}))
    t = rec["parts"]["transformers"]
    assert t["state"] == skillsfresh.UNKNOWN and "transformers_minimum_missing" in names(rec["advisories"])
    assert rec["severity"] == "advisory"


def test_a_part_with_every_check_made_is_still_current(tmp_path, py):
    src, fake = full(tmp_path, py)
    rec = build(src, fake, GOOD())
    assert rec["parts"]["transformers"]["state"] == skillsfresh.CURRENT and rec["advisories"] == []


def test_an_unchecked_advisory_never_sits_next_to_a_current_part(tmp_path, py):
    model = config(tmp_path, model_type="llama", architectures=["LlamaForCausalLM", "not a class!"], transformers_version="4.40.0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("4.57.6", {"LlamaForCausalLM": True}))
    assert "transformers_architectures_unchecked" in names(rec["advisories"])
    assert rec["parts"]["transformers"]["state"] == skillsfresh.UNKNOWN


# ---- 3. a malformed transformers_version is reported ------------------------------------------------------------------------

@pytest.mark.parametrize("value", [5, 5.19, True, "5." * 40, ["5.19.0"]])
def test_a_malformed_transformers_version_is_reported_even_when_the_table_gives_a_minimum(tmp_path, py, value):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM"], transformers_version=value)
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.19.0", {"Qwen4ExpForCausalLM": True}))
    assert "transformers_minimum_unreadable" in names(rec["advisories"])
    assert rec["parts"]["transformers"]["state"] == skillsfresh.UNKNOWN


def test_the_facts_keep_a_marker_for_a_value_that_was_dropped(tmp_path):
    facts = sp.config_facts(config(tmp_path, model_type="m", architectures=["A"], transformers_version=5))
    assert facts["transformers_version"] is None and facts["transformers_version_problem"]
    assert sp.config_facts(config(tmp_path, model_type="m", architectures=["A"]))["transformers_version_problem"] is None


# ---- 4. a resumed run keeps its reasons ---------------------------------------------------------------------------------

def _resume_record(tmp_path, py):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM"])
    src, fake = full(tmp_path, py, model=model)
    return build(src, fake, probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}), new_run=False)


def test_the_ledger_entry_of_a_resumed_run_carries_the_reasons_of_the_demoted_findings(tmp_path, py):
    rec = _resume_record(tmp_path, py)
    assert [a["name"] for a in rec["advisories"] if a.get("would_be_critical")] == [sp.H_BELOW, sp.H_CLASS]
    entry = sp.ledger_fields(rec)
    assert entry["would_be_critical"] == [sp.H_BELOW, sp.H_CLASS]
    assert {f["name"] for f in entry["findings"]} >= {sp.H_BELOW, sp.H_CLASS} and all(f["reason"] for f in entry["findings"])


def test_the_narration_of_a_resumed_run_shows_the_demoted_findings_as_resume_advisories():
    data = {"what": "stack pins", "severity": "advisory", "mode": "resume", "hazards": [], "advisories": ["a", "b"], "would_block": [],
            "would_be_critical": ["a"], "findings": [{"name": "a", "reason": "alpha is stale"}]}
    (line,) = narrate.lines_for({"event": "evidence", "data": data, "stage": 0, "ts": "2026-11-04T00:00:00Z", "seq": 1})
    assert "resume" in line.text and "alpha is stale" in line.text and "would have been critical" in line.text


# ---- 5. findings that share a name keep their own reasons ---------------------------------------------------------------------

def test_the_narration_keeps_the_reason_of_each_stale_clone_that_shares_a_name():
    data = {"what": "stack pins", "severity": "advisory", "mode": "new", "hazards": [], "advisories": [sp.H_SKILLS, sp.H_SKILLS],
            "would_block": [sp.H_SKILLS, sp.H_SKILLS],
            "findings": [{"name": sp.H_SKILLS, "reason": "clone ONE is 40 days behind"}, {"name": sp.H_SKILLS, "reason": "clone TWO is 90 days behind"}]}
    lines = narrate.lines_for({"event": "evidence", "data": data, "stage": 0, "ts": "2026-11-04T00:00:00Z", "seq": 1})
    text = " ".join(line.text for line in lines)
    assert "clone ONE" in text and "clone TWO" in text


# ---- the wiring: the supervisor writes the helper's fields to the ledger ------------------------------------------------------

from orchard.supervisor import Control                                                      # noqa: E402
from test_supervisor_stack_pins import (QWEN4_HAZARDS, abort_when_waiting, evidence, fake_check, rig, run)    # noqa: E402,F401



@pytest.mark.usefixtures("stub_tools")
def test_the_ledger_of_a_resumed_run_names_the_demoted_findings_and_their_reasons(rig):
    calls = []
    check = fake_check(calls, hazards=QWEN4_HAZARDS)
    abort_when_waiting(rig)
    run(rig, check)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.args.resume_aborted = True
    run(rig, check)
    last = evidence(rig)[-1]
    assert last["mode"] == "resume" and last["would_be_critical"] == [sp.H_BELOW, sp.H_CLASS]
    assert {f["name"] for f in last["findings"]} == {sp.H_BELOW, sp.H_CLASS} and all("5.19.0" in f["reason"] or "Qwen4Exp" in f["reason"] for f in last["findings"])
