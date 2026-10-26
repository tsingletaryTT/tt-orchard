# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 writes the scores the model card reports as a draft, and one hazard that says nobody has confirmed them.

The draft lives in stage 0's evidence (`reported-results.draft.json`). It is never read as `reported-results.json`: a
person accepts rows with `tt-orchard reported accept` (tests/test_reported_cmd.py). The path and class do not depend on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_delta_triage_template import (delta, evidence, run_triage, text_case, tokenizer_json,  # noqa: E402,F401
                                        template_module)

from orchard.package import read_reported_file  # noqa: E402
from orchard.stages import gate_delta  # noqa: E402

CARD = ("---\nlicense: apache-2.0\n---\n# Hemmingway-1\n\n## Evaluation\n\n"
        "| Model | MMLU | GSM8K |\n|---|---|---|\n| Other-9B | 70.1 | 80.0 |\n| Hemmingway-1 | 71.5 | 82.25 |\n")


def snapshot_of(stage: Path) -> Path:
    return Path(json.loads((stage / "triage_config.json").read_text())["model_snapshot"])


def draft_of(stage: Path) -> dict:
    return evidence(stage, "reported-results.draft.json")


def reported_hazards(d: dict) -> list:
    return [h for h in d["hazards"] if "reported" in h["finding"].lower() and "model card" in h["finding"].lower()]


def test_a_card_table_gives_a_draft_and_one_hazard(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text(CARD)
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr
    draft = draft_of(stage)
    assert {x["name"]: x["value"] for x in draft["rows"]} == {"MMLU": "71.5", "GSM8K": "82.25"}
    assert draft["model"].startswith("Altworld/Hemmingway-1")
    assert 0 < len(draft["source"]) <= 200 and "model card" in draft["source"]
    assert len(draft["readme_sha256"]) == 64
    assert draft["other_model_rows"] == 1 and draft["truncated"] is False
    [h] = reported_hazards(delta(stage))
    assert h["area"] == "other"
    assert "2 " in h["finding"] and "nobody has confirmed" in h["finding"]
    assert "reported-results.draft.json" in " ".join(h["evidence"])
    assert "tt-orchard reported accept" in h["finding"]


def test_the_draft_is_not_a_reported_results_file_of_the_run(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text(CARD)
    run_triage(stage)
    assert not (stage.parent.parent / "reported-results.json").exists()
    assert not (stage / "reported-results.json").exists()


def test_the_draft_rows_pass_the_rules_reported_results_files_pass(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text(CARD)
    run_triage(stage)
    rows, source = read_reported_file(stage / "evidence" / "reported-results.draft.json")
    assert [r.name for r in rows] == ["MMLU", "GSM8K"] and source


def test_the_stage_0_gate_still_passes_with_the_hazard(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text(CARD)
    run_triage(stage)
    assert gate_delta(stage, stage.parent.parent).ok


def test_a_card_with_no_table_writes_nothing_and_adds_no_hazard(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text("# Hemmingway-1\n\nIt scores 84 on MMLU.\n")
    run_triage(stage)
    assert not (stage / "evidence" / "reported-results.draft.json").exists()
    assert reported_hazards(delta(stage)) == []


def test_no_readme_writes_nothing(text_case):
    stage = text_case()
    readme = snapshot_of(stage) / "README.md"
    if readme.exists():
        readme.unlink()
    assert run_triage(stage).returncode == 0
    assert not (stage / "evidence" / "reported-results.draft.json").exists()


def test_a_card_whose_rows_were_all_skipped_keeps_the_draft_for_the_reasons_but_raises_no_hazard(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text("| Benchmark | Hemmingway-1 |\n|---|---|\n| MMLU | 71 (est.) |\n")
    run_triage(stage)
    draft = draft_of(stage)
    assert draft["rows"] == [] and len(draft["skipped"]) == 1
    assert reported_hazards(delta(stage)) == []


def test_the_hazard_counts_what_was_left_out(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_text(CARD + "\n| Benchmark | Hemmingway-1 |\n|---|---|\n| ARC | 5 (est.) |\n")
    run_triage(stage)
    [h] = reported_hazards(delta(stage))
    assert "1 more entry was left out" in h["finding"]


def test_the_path_and_class_do_not_depend_on_the_card(text_case):
    stage = text_case()
    readme = snapshot_of(stage) / "README.md"
    if readme.exists():
        readme.unlink()
    run_triage(stage)
    a = delta(stage)
    readme.write_text(CARD)
    run_triage(stage)
    b = delta(stage)
    assert reported_hazards(a) == [] and len(reported_hazards(b)) == 1
    assert (a["path"], a["class"], a["path_reasons"]) == (b["path"], b["class"], b["path_reasons"])


def test_a_readme_larger_than_the_limit_is_read_only_to_the_limit(text_case):
    stage = text_case()
    t = template_module()
    (snapshot_of(stage) / "README.md").write_text(CARD + "x" * (t.MAX_README_BYTES + 100) + "\n| Hemmingway-1 | 1 | 2 |\n")
    run_triage(stage)
    assert draft_of(stage)["truncated"] is True


def test_a_readme_that_is_not_valid_utf8_does_not_stop_the_triage(text_case):
    stage = text_case()
    (snapshot_of(stage) / "README.md").write_bytes(b"\xff\xfe" + CARD.encode())
    r = run_triage(stage)
    assert r.returncode == 0, r.stderr


def test_the_skill_tells_the_agent_to_leave_the_reported_scores_alone():
    skill = " ".join((Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage.md").read_text().split())
    assert "reported-results.draft.json" in skill
    assert "Do not copy those rows into any other file" in skill
    assert "do not edit the draft" in skill
