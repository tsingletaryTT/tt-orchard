# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 1's gate reads the evidence the reference template wrote, not only the file the stage agent wrote about it.

On the pplx-decider run (2026-10-19) the template's evidence/summary.json said `fail` (`loads` false: all 851 model keys missing, so
the CPU model ran on weights that never loaded, and its text was noise) and the agent wrote reference.json with `verdict: pass` and the
note "expected and does not affect functionality". The gate checked reference.json against itself and passed it. These tests pin the
fix: the template's evidence decides; a check the evidence failed cannot be passed by the agent's file; missing backbone keys fail;
and an evidence file that is missing (where the template was used), unreadable, oversized or a link out of the run fails closed.
"""
import hashlib
import json

import pytest

from orchard.stages import SHIPPED_TEMPLATES, gate_reference

CHECKS = ("loads", "tokenizer round trip", "decodes forward", "matches the card")


def stage(tmp_path, *, claimed="pass", claimed_checks=None, summary="pass", summary_checks=None, missing=(), keys=851,
          ignore=("^model.visual.*", "^mtp.*"), template=True):
    s1 = tmp_path / "stages" / "1"
    (s1 / "evidence").mkdir(parents=True)
    (tmp_path / "evidence").mkdir(exist_ok=True)
    (tmp_path / "evidence" / "e.json").write_text("{}")
    claimed_checks = claimed_checks or {c: True for c in CHECKS}
    (s1 / "reference.json").write_text(json.dumps({"verdict": claimed, "checks": [
        {"name": n, "pass": ok, "evidence": ["evidence/e.json"]} for n, ok in claimed_checks.items()]}))
    if summary is not None:
        (s1 / "evidence" / "summary.json").write_text(json.dumps(
            {"verdict": summary, "checks": summary_checks or {c: summary == "pass" for c in CHECKS},
             "template_sha256": hashlib.sha256(SHIPPED_TEMPLATES["reference_gate.py"].read_bytes()).hexdigest()}))
    (s1 / "evidence" / "load-report.json").write_text(json.dumps(
        {"missing_keys": list(missing), "model_key_count": keys, "ignore_patterns": list(ignore)}))
    if template:
        (s1 / "reference_gate.py").write_text(SHIPPED_TEMPLATES["reference_gate.py"].read_text())      # an unedited copy
    return s1


def test_a_reference_the_evidence_agrees_with_passes(tmp_path):
    s1 = stage(tmp_path)
    g = gate_reference(s1, tmp_path)
    assert g.ok, g.reasons


def test_a_pass_the_template_failed_does_not_pass(tmp_path):
    s1 = stage(tmp_path, claimed="pass", summary="fail", missing=[f"model.layers.{i}.w" for i in range(851)])
    g = gate_reference(s1, tmp_path)
    assert not g.ok
    text = " ".join(g.reasons)
    assert "evidence/summary.json" in text and "'fail'" in text and "reference.json" in text
    assert "851 of 851" in text and "did not load" in text


def test_one_check_the_evidence_failed_cannot_be_passed_by_the_agents_file(tmp_path):
    s1 = stage(tmp_path, summary="pass", summary_checks={**{c: True for c in CHECKS}, "loads": False})
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("'loads'" in r and "evidence/summary.json" in r for r in g.reasons)


def test_missing_weights_fail_even_when_both_files_say_pass(tmp_path):
    s1 = stage(tmp_path, missing=["model.layers.0.w"])
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("1 of 851" in r for r in g.reasons)


def test_missing_keys_under_an_ignore_pattern_do_not_fail_the_gate(tmp_path):
    s1 = stage(tmp_path, missing=["model.visual.blocks.0.w", "mtp.fc.weight"])
    assert gate_reference(s1, tmp_path).ok


def test_the_template_was_used_but_its_summary_is_gone(tmp_path):
    s1 = stage(tmp_path, summary=None)
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("evidence/summary.json is missing" in r for r in g.reasons)


def test_a_stage_that_never_used_the_template_no_longer_passes(tmp_path):
    """Before, a stage with no copy was judged on reference.json alone, which let an agent delete the copy after running an edited one."""
    s1 = stage(tmp_path, summary=None, template=False)
    (s1 / "evidence" / "load-report.json").unlink()
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("reference_gate.py is not in this stage directory" in r for r in g.reasons)


def test_an_evidence_file_that_is_oversized_fails_closed(tmp_path):
    """Valid JSON over the cap, so only the cap can be what stops it from being read as clean."""
    s1 = stage(tmp_path)
    (s1 / "evidence" / "load-report.json").write_text(
        json.dumps({"missing_keys": [], "model_key_count": 851, "ignore_patterns": [], "pad": "x" * (3 * 1024 * 1024)}))
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("evidence/load-report.json" in r and "too large" in r for r in g.reasons)


def test_an_evidence_file_that_links_out_of_the_run_is_not_read(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"verdict": "pass", "checks": {c: True for c in CHECKS}}))
    s1 = stage(tmp_path, summary="fail")
    (s1 / "evidence" / "summary.json").unlink()
    (s1 / "evidence" / "summary.json").symlink_to(outside)
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("evidence/summary.json" in r and "outside" in r for r in g.reasons)


def test_an_evidence_file_that_is_not_json_fails_closed(tmp_path):
    s1 = stage(tmp_path)
    (s1 / "evidence" / "summary.json").write_text("not json")
    g = gate_reference(s1, tmp_path)
    assert not g.ok and any("evidence/summary.json" in r and "not readable" in r for r in g.reasons)
