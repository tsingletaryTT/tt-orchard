# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Resume markers the supervisor writes (the weights-swap plan and the text and audio stage 1 reference files) need a ledger entry
with the same path and sha256 before a resume trusts them (issue 344)."""
import json

import pytest

from orchard.ledger import Ledger
from orchard.stages import AUDIO_STAGE_1, TEXT_STAGE_1, WEIGHTS_ONLY_STAGE_4, evidence_record, open_stage_dir

CASES = [
    # (spec, marker path inside the stage directory, stage number, `what` of the entry the supervisor appends)
    (WEIGHTS_ONLY_STAGE_4, "tests/plan.json", 4, "hardware test list"),
    (TEXT_STAGE_1, "text_reference.json", 1, "text stage record"),
    (AUDIO_STAGE_1, "audio_reference.json", 1, "audio stage record"),
]
IDS = [c[1] for c in CASES]


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield led


def write_marker(d, rel, text='{"tests": []}'):
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text(text)


def vouch(ledger, run, n, what, path):
    ledger.append("evidence", n, what=what, **evidence_record(run, path))


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_marker_an_agent_wrote_is_not_a_resume_point(tmp_path, ledger, spec, rel, n, what):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    again, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed and not (again / rel).exists()
    assert (run / "stages" / f"{n}.partial-1" / rel).exists()
    assert any(e["data"].get("decision") == "marker not written by the supervisor" for e in ledger.read())


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_marker_the_supervisor_recorded_resumes_after_restarts(tmp_path, ledger, spec, rel, n, what):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    vouch(ledger, run, n, what, d / rel)
    for _ in range(2):
        again, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
        assert resumed and again == d


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_marker_changed_after_the_record_is_not_a_resume_point(tmp_path, ledger, spec, rel, n, what):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel, '{"tests": [], "a": 1}')
    vouch(ledger, run, n, what, d / rel)
    write_marker(d, rel, '{"tests": [], "a": 2}')
    _, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_record_from_an_earlier_attempt_does_not_vouch_for_the_retry(tmp_path, ledger, spec, rel, n, what):
    """Attempt 1 was recorded and retried. Attempt 2 (or an agent in it) wrote the same bytes and was killed before its own record."""
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    vouch(ledger, run, n, what, d / rel)
    d, resumed = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    assert not resumed
    write_marker(d, rel)
    _, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_an_evidence_file_entry_from_an_agent_step_does_not_vouch(tmp_path, ledger, spec, rel, n, what):
    """AgentStep appends "evidence file" entries for files an agent names. Those carry a path and sha256 too."""
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    vouch(ledger, run, n, "evidence file", d / rel)
    _, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed


def test_the_text_reference_entry_does_not_vouch_for_the_audio_marker(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, AUDIO_STAGE_1, resuming=False, ledger=ledger)
    write_marker(d, "audio_reference.json")
    vouch(ledger, run, 1, "text stage record", d / "audio_reference.json")
    _, resumed = open_stage_dir(run, AUDIO_STAGE_1, resuming=True, ledger=ledger)
    assert not resumed
