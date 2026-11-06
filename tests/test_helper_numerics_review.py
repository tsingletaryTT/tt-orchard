# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A port that runs but disagrees: the findings of the Copilot review of PR 498. Each test names the finding."""
import json

import pytest

from orchard import port_repair, repair_help as rh
from orchard.supervisor import EXIT_ABORTED
from test_helper_numerics_rounds import LOW, MID, PASSES, checked, decs, record_path, rig, stage_ends, start, wire_helper  # noqa: F401
from test_supervisor import Rig
from fakes import Crash


def test_a_record_whose_ids_were_rewritten_to_agree_is_not_a_pass(tmp_path):
    """Finding: with only chip_ids set equal to reference_ids the reference and port hashes stay valid, so the reader called it passed."""
    run, stage = checked(tmp_path, LOW)
    rec = json.loads(record_path(stage).read_text())
    rec["chip_ids"] = list(rec["reference_ids"])
    record_path(stage).write_text(json.dumps(rec))
    assert port_repair.numerics_state(run, stage) == ("unreadable", None)


def test_an_untouched_record_is_still_read(tmp_path):
    run, stage = checked(tmp_path, LOW)
    assert port_repair.numerics_state(run, stage)[0] == "below"


def test_the_request_outside_a_rehearsal_does_not_say_the_weights_are_random():
    """Finding: --helper-drives-port also runs outside rehearsals, where the checkpoint is trained and its text means something."""
    real, _ = rh.build_task(facts="F", excerpts="E", port_text=LOW, listing="p", homes=(), report="top1_agreement 0.5", rehearsal=False)
    assert "RANDOM-WEIGHT" not in real and "meaningless" not in real and "random" not in real.lower().split("## ")[0]
    assert "teacher forced" in real and "What the check measured" in real
    rehearsal, _ = rh.build_task(facts="F", excerpts="E", port_text=LOW, listing="p", homes=(), report="top1_agreement 0.5", rehearsal=True)
    assert "RANDOM-WEIGHT" in rehearsal


def test_a_helper_driven_run_that_is_not_a_rehearsal_sends_the_plain_numerics_request(rig, monkeypatch, tmp_path):
    rig.args.full_port_rehearsal = False
    rig.args.helper_drives_port = True
    tel, wire = wire_helper(monkeypatch, tmp_path, [MID, PASSES])
    start(rig, LOW, ["# c\n"])
    rig.run()
    first = wire.requests[0]["body"]["messages"][-1]["content"]
    assert "What the check measured" in first and "RANDOM-WEIGHT" not in first


def test_a_restart_after_a_numerics_repair_edit_and_before_its_rerun_runs_the_check_again(rig, monkeypatch, tmp_path):
    """Finding: the edited port no longer matches the old record, the numbers read as None, and the early return skipped the pending-edit rule, so
    the repaired port went to the finish step with evidence from the port before it."""
    tel, wire = wire_helper(monkeypatch, tmp_path, [MID, PASSES])
    start(rig, LOW, ["# c\n"])
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["stage"] == 2 and e["data"].get("what") == "port repair round output")
    assert not decs(rig, port_repair.RERUN)
    rig.run(pid=200)
    assert decs(rig, port_repair.RERUN), "the repaired port was never tested again"
    assert stage_ends(rig) == ["pass"]


# ---- round 3: every source the check ran is compared, not only port_model.py --------------------------------------------------------------

def _checked_with_helper(tmp_path):
    from port_check_fakes import make_run, run_template
    run, stage, *_ = make_run(tmp_path, PASSES, extra_files={"port/helper.py": "H = 1\n", "port/sub/deep.py": "D = 2\n"})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    return run, stage


def test_untouched_sources_leave_the_check_readable_as_a_pass(tmp_path):
    run, stage = _checked_with_helper(tmp_path)
    assert port_repair.numerics_state(run, stage)[0] == "passed"


def test_a_supporting_file_changed_added_or_removed_after_the_check_makes_the_record_unreadable(tmp_path):
    """Finding: only port_model.py was compared, so the old agreement stayed valid for a port nobody had tested."""
    for action in ("changed", "added", "removed", "nested-changed"):
        run, stage = _checked_with_helper(tmp_path / action)
        port = stage / "port"
        if action == "changed":
            (port / "helper.py").write_text("H = 2\n")
        elif action == "added":
            (port / "extra.py").write_text("E = 3\n")
        elif action == "removed":
            (port / "helper.py").unlink()
        else:
            (port / "sub" / "deep.py").write_text("D = 9\n")
        assert port_repair.numerics_state(run, stage) == ("unreadable", None), action
