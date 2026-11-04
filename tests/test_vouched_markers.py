# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Resume markers the supervisor writes (the weights-swap plan and the text and audio stage 1 reference files) need a ledger entry
with the same path and sha256 before a resume trusts them (issue 344)."""
import json

import pytest

from orchard.ledger import Ledger
from orchard.stages import AUDIO_STAGE_1, STAGES, TEXT_STAGE_1, WEIGHTS_ONLY_STAGE_4, evidence_record, open_stage_dir

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


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_kill_between_the_move_aside_and_its_ledger_entry_leaves_no_voucher(tmp_path, ledger, spec, rel, n, what):
    """Kill points of the marker rule. Attempt 1 wrote its marker and the supervisor's record. The directory was renamed aside and the
    process died before the move-aside entry. Attempt 2 starts a fresh directory (stage_start with resumed=False), an agent writes the same
    bytes, and the process dies again. The record of attempt 1 must not vouch for the file."""
    import os
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    vouch(ledger, run, n, what, d / rel)
    os.rename(d, d.with_name(f"{n}.partial-1"))                 # the rename happened, the entry did not
    (run / "stages" / str(n) / "evidence").mkdir(parents=True)
    ledger.append("stage_start", n, escalated=False, resumed=False)
    write_marker(run / "stages" / str(n), rel)
    _, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed


@pytest.mark.parametrize("spec,rel,n,what", CASES, ids=IDS)
def test_a_kill_between_the_marker_and_its_record_is_not_a_resume_point(tmp_path, ledger, spec, rel, n, what):
    """The marker is written, then the supervisor dies before it appends the record. A resume cannot tell that file from an agent's."""
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, spec, resuming=False, ledger=ledger)
    write_marker(d, rel)
    again, resumed = open_stage_dir(run, spec, resuming=True, ledger=ledger)
    assert not resumed and not (again / rel).exists()


# ---- a port repair rerun in flight (review of PR 432) -----------------------------------------------------------------------------

def test_a_test_record_no_entry_vouches_for_is_not_a_resume_point_by_default(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    write_marker(d, "test-result.json", '{"returncode": 0}')
    again, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)
    assert not resumed and not (again / "test-result.json").exists()
    assert any(e["data"].get("decision") == "marker not written by the supervisor" for e in ledger.read())


def test_with_a_rerun_owed_an_unvouched_test_record_keeps_the_directory_and_is_marked_untrusted(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    write_marker(d, "test-result.json", '{"returncode": 0}')
    (d / "port").mkdir()
    (d / "port" / "port_model.py").write_text("repaired")
    again, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger, rerun_owed=True)
    assert resumed and again == d and (d / "port" / "port_model.py").read_text() == "repaired"
    kept = [e["data"] for e in ledger.read() if e["data"].get("decision") == "resume from marker"]
    assert kept and kept[-1]["unvouched"] is True
    assert not any(e["data"].get("decision") == "marker not written by the supervisor" for e in ledger.read())


def test_a_rerun_owed_does_not_vouch_for_a_marker_of_another_kind(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, WEIGHTS_ONLY_STAGE_4, resuming=False, ledger=ledger)
    write_marker(d, "tests/plan.json")
    again, resumed = open_stage_dir(run, WEIGHTS_ONLY_STAGE_4, resuming=True, ledger=ledger, rerun_owed=True)
    assert not resumed and not (again / "tests" / "plan.json").exists()

