# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A stage whose evidence comes from a shipped template script is only as trustworthy as that script. The gate compares the stage's copy with
the shipped file.

On the pplx-decider run (2026-10-19) the stage 1 agent edited its copy of reference_gate.py: it replaced the class lookup with
`cls = transformers.AutoModelForCausalLM` ("to ensure we get a model with a language modeling head") and changed a test string. The edited
script then loaded none of a headless checkpoint's weights, and the evidence it wrote described that. The shipped template would have
loaded the model correctly. Stages 0 and 1 run templates whose output the gates trust, so a copy that differs from the shipped file fails
the gate and says how to fix it.
"""
import json
import shutil
from pathlib import Path

import pytest

from orchard import stages
from orchard.stages import gate_delta, gate_reference

SKILLS = Path(stages.__file__).with_name("skills")
SHIPPED = {"delta_triage.py": SKILLS / "delta-triage-templates" / "delta_triage.py",
           "reference_gate.py": SKILLS / "reference-gate-templates" / "reference_gate.py"}


def stage_with(tmp_path, number, script, text=None):
    d = tmp_path / "stages" / str(number)
    (d / "evidence").mkdir(parents=True)
    (tmp_path / "evidence").mkdir(exist_ok=True)
    (tmp_path / "evidence" / "e.json").write_text("{}")
    if script:
        (d / script).write_text(text if text is not None else SHIPPED[script].read_text())
    return d


def good_reference(stage):
    checks = [{"name": n, "pass": True, "evidence": ["evidence/e.json"]}
              for n in ("loads", "tokenizer round trip", "decodes forward", "matches the card")]
    (stage / "reference.json").write_text(json.dumps({"verdict": "pass", "checks": checks}))
    (stage / "evidence" / "summary.json").write_text(json.dumps({"verdict": "pass", "checks": {c["name"]: True for c in checks},
                                                                  "template_sha256": sha("reference_gate.py")}))


def good_delta(stage):
    (stage / "delta.json").write_text(json.dumps({
        "model": "a/b", "nearest_model": "c/d", "path": "weights-only", "class": "weights-only", "sidecars": [],
        "template_sha256": sha("delta_triage.py"),
        "differences": [{"area": "config", "finding": "x", "evidence": ["evidence/e.json"]}], "hazards": []}))


def test_an_unedited_reference_template_passes(tmp_path):
    s1 = stage_with(tmp_path, 1, "reference_gate.py")
    good_reference(s1)
    g = gate_reference(s1, tmp_path)
    assert g.ok, g.reasons


def test_an_edited_reference_template_fails_the_gate_and_says_how_to_fix_it(tmp_path):
    edited = SHIPPED["reference_gate.py"].read_text().replace("cls = resolve_class(transformers, config, arch)",
                                                              "cls = transformers.AutoModelForCausalLM")
    assert edited != SHIPPED["reference_gate.py"].read_text()
    s1 = stage_with(tmp_path, 1, "reference_gate.py", edited)
    good_reference(s1)
    g = gate_reference(s1, tmp_path)
    assert not g.ok
    text = " ".join(g.reasons)
    assert "reference_gate.py" in text and "differs from the shipped template" in text and "sha256" in text
    assert "Copy" in text and "config file" in text


def test_a_stage_with_no_template_copy_fails_the_gate(tmp_path):
    """Without the copy, an agent could run an edited script under another name, or delete the copy after running it."""
    s1 = stage_with(tmp_path, 1, None)
    good_reference(s1)
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("reference_gate.py is not in this stage directory" in r for r in g.reasons)


def test_a_stage_0_with_no_triage_copy_fails_the_gate(tmp_path):
    s0 = stage_with(tmp_path, 0, None)
    good_delta(s0)
    g = gate_delta(s0, tmp_path)
    assert not g.ok and any("delta_triage.py is not in this stage directory" in r for r in g.reasons)


def sha(name):
    import hashlib
    return hashlib.sha256(SHIPPED[name].read_bytes()).hexdigest()


def test_a_summary_without_the_templates_stamp_fails(tmp_path):
    """The copy can be unedited while the evidence came from another script."""
    s1 = stage_with(tmp_path, 1, "reference_gate.py")
    good_reference(s1)
    summary = json.loads((s1 / "evidence" / "summary.json").read_text())
    summary.pop("template_sha256")
    (s1 / "evidence" / "summary.json").write_text(json.dumps(summary))
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("template_sha256" in r and "summary.json" in r for r in g.reasons)


def test_a_summary_stamped_by_another_script_fails(tmp_path):
    s1 = stage_with(tmp_path, 1, "reference_gate.py")
    good_reference(s1)
    summary = json.loads((s1 / "evidence" / "summary.json").read_text())
    summary["template_sha256"] = "0" * 64
    (s1 / "evidence" / "summary.json").write_text(json.dumps(summary))
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("template_sha256" in r and "000000000000" in r for r in g.reasons)


def test_a_delta_without_or_with_a_wrong_stamp_fails(tmp_path):
    s0 = stage_with(tmp_path, 0, "delta_triage.py")
    good_delta(s0)
    d = json.loads((s0 / "delta.json").read_text())
    assert gate_delta(s0, tmp_path).ok
    for stamp in (None, "0" * 64):
        e = dict(d)
        e.pop("template_sha256", None)
        if stamp:
            e["template_sha256"] = stamp
        (s0 / "delta.json").write_text(json.dumps(e))
        g = gate_delta(s0, tmp_path)
        assert not g.ok and any("template_sha256" in r and "delta.json" in r for r in g.reasons), stamp


def test_an_unedited_triage_template_passes_and_an_edited_one_fails(tmp_path):
    s0 = stage_with(tmp_path, 0, "delta_triage.py")
    good_delta(s0)
    assert gate_delta(s0, tmp_path).ok
    (s0 / "delta_triage.py").write_text(SHIPPED["delta_triage.py"].read_text() + "\n# a change\n")
    g = gate_delta(s0, tmp_path)
    assert not g.ok and any("delta_triage.py" in r and "differs from the shipped template" in r for r in g.reasons)


def test_one_changed_character_is_enough(tmp_path):
    s1 = stage_with(tmp_path, 1, "reference_gate.py", SHIPPED["reference_gate.py"].read_text().replace("N_NEW", "N_NEW ", 1))
    good_reference(s1)
    assert not gate_reference(s1, tmp_path).ok


def test_the_shipped_templates_exist_where_the_gate_looks():
    for name, path in SHIPPED.items():
        assert path.is_file() and stages.SHIPPED_TEMPLATES[name] == path
