# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 440: the operator bundle's RISKS.md says what the GGUF files showed (evidence/gguf-header.json), and says nothing when stage 0 wrote none."""
from __future__ import annotations

import json
from pathlib import Path

GGUF_REL = "stages/0/evidence/gguf-header.json"


def row(item, gguf, config, verdict, note=None):
    r = {"item": item, "gguf": gguf, "config": config, "verdict": verdict}
    if note:
        r["note"] = note
    return r


OK = {"file": "Model-Q4_K_M.gguf", "size": 2708804096, "status": "ok", "role": "language model", "architecture": "qwen35", "file_type": 15,
      "tensor_count": 426,
      "agreement": [row("block count", 32, 32, "agree"), row("hidden size", 2560, 2560, "agree"),
                    row("multi-token prediction layers", 0, 1, "differ", "the config declares a head and the GGUF ships none: a converter drops it"),
                    row("rope sections", [11, 11, 10, 0], None, "not comparable")],
      "quantization": {"full_precision": [{"kind": "blk.N.ssm_a", "type": "F32", "count": 24}, {"kind": "output_norm.weight", "type": "F32", "count": 1}]}}
BAD = {"file": "mmproj.gguf", "size": 5, "status": "unreadable", "reason": "the file does not start with GGUF"}


def _bundle(tmp_path, gguf):
    from test_operator_bundle_template import build, bundle_of, make_run, section
    run = make_run(tmp_path)
    if gguf is not None:
        p = run / GGUF_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(gguf))
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    risks = (bundle_of(run) / "RISKS.md").read_text()
    return (section(risks, "## What the GGUF files show") if "## What the GGUF files show\n" in risks else None), risks


def test_the_section_names_the_reading_the_differences_and_the_full_precision_hint(tmp_path):
    sec, _ = _bundle(tmp_path, {"schema": 1, "files": [OK]})
    assert "independent reading of the architecture" in sec and GGUF_REL in sec
    assert "language model, architecture qwen35, 426 tensors, GGUF file type 15" in sec
    assert "2 agree, 1 differ, 1 not comparable" in sec
    assert "multi-token prediction layers: GGUF 0; config 1." in sec and "a converter drops it" in sec
    assert "blk.N.ssm_a (F32, 24)" in sec and "output_norm.weight (F32, 1)" in sec
    assert "not a measurement on Tenstorrent hardware" in sec and "Used by a later stage: Not shown" in sec
    assert "safetensors" in sec and "no stage uses these files as weights" in sec


def test_an_unreadable_file_is_listed_with_its_reason(tmp_path):
    sec, _ = _bundle(tmp_path, {"schema": 1, "files": [OK, BAD]})
    assert "### mmproj.gguf" in sec and "The header could not be read: the file does not start with GGUF." in sec


def test_a_run_with_no_gguf_evidence_gets_the_same_risks_as_before(tmp_path):
    sec, risks = _bundle(tmp_path, None)
    assert sec is None and "GGUF" not in risks


def test_the_results_file_shows_the_finding_through_the_delta_differences(tmp_path):
    from test_operator_bundle_template import bundle_of, make_run, build
    run = make_run(tmp_path)
    d = json.loads((run / "stages/0/delta.json").read_text())
    d["differences"].append({"area": "other", "finding": "The repository ships GGUF files. Model-Q4_K_M.gguf: 2 rows agree, 1 differ.",
                             "evidence": [GGUF_REL]})
    (run / "stages/0/delta.json").write_text(json.dumps(d))
    assert build(run).returncode == 0
    assert "GGUF files" in (bundle_of(run) / "RESULTS.md").read_text()


import pytest  # noqa: E402


@pytest.mark.parametrize("bad", [[], "x", 5, None, {"files": "nope"}, {"files": [5, None, "x", {"file": 5}, {"file": "a", "status": "weird"}]}])
def test_a_malformed_evidence_file_never_stops_the_bundle(tmp_path, bad):
    sec, _ = _bundle(tmp_path, bad if bad is not None else {})
    assert sec is None or "could not be read" in sec or "holds no usable record" in sec


def test_hostile_values_are_redacted_one_line_and_capped(tmp_path):
    token = "hf_" + "A" * 30
    evil = dict(OK, file="a.gguf", role=f"language\nmodel {token}", architecture="x" * 5000,
                agreement=[row("block count\n# injected heading", "/home/someone/secret " + "y" * 4000, token, "differ", "line one\nline two")] * 60,
                quantization={"full_precision": [{"kind": f"k{i}.{token}", "type": "F32", "count": i} for i in range(200)]})
    sec, risks = _bundle(tmp_path, {"schema": 1, "files": [evil]})
    assert token not in risks and "/home/someone" not in sec
    assert "\n# injected heading" not in sec and "30 more differing rows" in sec
    assert "and 170 more" in sec
    assert max(len(line) for line in sec.splitlines()) < 2600 and len(sec) < 12000        # 200 kinds in, 30 shown; 60 rows in, 30 shown


def test_the_delta_triage_skill_tells_the_agent_about_the_file():
    text = " ".join((Path(__file__).parent.parent / "orchard/skills/delta-triage.md").read_text().split())
    assert "evidence/gguf-header.json" in text and "never changes the class or the path" in text
    assert "Do not edit the file, and do not load or dequantize a GGUF tensor" in text
