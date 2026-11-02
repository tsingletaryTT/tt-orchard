# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The rehearsal option of the full-port stage 2 gate (issue 357). A rehearsal model has random weights, so its free-run text is not
readable and `coherent` cannot be true. The option waives that one field. The operator sets it with --full-port-rehearsal when the run
starts; the ledger holds it; nothing the coder writes can turn it on."""
import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from test_full_port import build, one, reasons


def rehearsal_reasons(tmp_path, **kw):
    stage_dir, run = build(tmp_path, **kw)
    return gate_full_port(stage_dir, run, rehearsal=True)


def test_rehearsal_waives_the_coherent_field(tmp_path):
    g = rehearsal_reasons(tmp_path, result={"coherent": False})
    assert g.ok, g.reasons


def test_the_gate_is_unchanged_by_default(tmp_path):
    g = reasons(tmp_path, result={"coherent": False})
    assert not g.ok and one(g, "coherent")


@pytest.mark.parametrize("result", [{"top1_agreement": 0.5}, {"n_tokens": 4}, {"runs": False}, {"load_s": 0}])
def test_rehearsal_still_fails_every_other_result_field(tmp_path, result):
    assert not rehearsal_reasons(tmp_path, result={"coherent": False, **result}).ok


def test_rehearsal_still_fails_the_test_exit_the_text_floor_and_the_template(tmp_path):
    assert not rehearsal_reasons(tmp_path, result={"coherent": False}, marker={"returncode": 1}).ok
    assert not rehearsal_reasons(tmp_path, result={"coherent": False, "free_run_text": "x"}, record={"free_run_text": "x"}).ok
    assert not rehearsal_reasons(tmp_path, result={"coherent": False}, record={"template_sha256": "0" * 64}).ok


def test_a_field_in_result_json_waives_nothing(tmp_path):
    g = reasons(tmp_path, result={"coherent": False, "rehearsal": True, "waive_coherent": True})
    assert not g.ok and one(g, "coherent")


def test_only_run_start_turns_the_option_on():
    from orchard.stages import full_port_rehearsal
    start = {"event": "run_start", "data": {"full_port": True}}
    claims = [{"event": "decision", "data": {"decision": "full-port rehearsal", "rehearsal": True, "full_port_rehearsal": True}},
              {"event": "evidence", "data": {"full_port_rehearsal": True}}]
    assert full_port_rehearsal([start, *claims]) is False
    assert full_port_rehearsal([{**start, "data": {**start["data"], "full_port_rehearsal": True}}]) is True


def test_the_gate_reports_when_it_waived_the_field(tmp_path):
    stage_dir, run = build(tmp_path, result={"coherent": False})
    assert full_port.coherence_waived(stage_dir, rehearsal=True) is True
    assert full_port.coherence_waived(stage_dir, rehearsal=False) is False
    good_dir, _ = build(tmp_path / "ok")
    assert full_port.coherence_waived(good_dir, rehearsal=True) is False          # coherent was true: nothing was waived
