# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 3 of the full-port route on a lab box (issue 326, review of PR 455, round 1).

With `--lab HOST --full-port --required-chips 2,4` the hardware phase went to the lab path before it computed stage 3's width. The lab path
leased and ran on `spec.boards` boards (two chips) while stage 3's gate expects the four configured chips, so a working port failed every
attempt. The lab path now takes the width stage 3 needs (`_phase_boards`), leases that many chips there and runs the test on them; a lab that
has fewer chips than the port's configured count is refused up front, as the lab path already does for a stage 4 configuration, and is not
left to wait out the stage budget. Other stages on a lab are unchanged. The chips, the coder and the model are fakes; the supervisor, the
stage machine and the gate are the real ones."""
import pytest

from orchard.stages import SKIP_3_FULL_PORT
from orchard.supervisor import EXIT_ABORTED
from port_check_fakes import install_fake_ttnn, python_path, tokenizer_json
from test_supervisor_full_port_stage3 import abort_at_pause, make3
from test_supervisor_lab import FakeLab, LabRig, lab_args

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")


class Stage3LabRig(LabRig):
    """A lab run of a full port configured for four chips (both boards)."""

    def __init__(self, tmp_path, required="2,4"):
        super().__init__(tmp_path)
        snap = self.root / "snapshot"                  # the lab requires the model input under its root
        snap.mkdir()
        tokenizer_json(snap / "tokenizer.json")
        install_fake_ttnn(tmp_path)
        extra = ["--input", f"model={snap}", "--env", f"PYTHONPATH={python_path(tmp_path)}", "--full-port"]
        if required:
            extra += ["--required-chips", required]
        self.args, self.run_dir = lab_args(self, self.root, extra=extra)


@pytest.fixture
def rig(tmp_path):
    r = Stage3LabRig(tmp_path)
    yield r
    r.close()


def lab_acquires(rig):
    return [c[1] for c in rig.lab.adapter.calls if c[0] == "acquire"]


def stage_tests(rig, n):
    return [e["data"] for e in rig.entries() if e["stage"] == n and e["data"].get("decision") == "hardware test started"]


def test_stage_3_on_a_lab_leases_and_runs_on_the_four_configured_chips(rig):
    make3(rig)
    abort_at_pause(rig)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(2)] == ["pass"] and [d["result"] for d in rig.ends(3)] == ["pass"]
    assert lab_acquires(rig) == [2, 4], "stage 2's board, then stage 3's four chips, both on the lab"
    ran = stage_tests(rig, 3)
    assert len(ran) == 1 and len(ran[0]["chips"]) == 4 and ran[0]["where"] == "lab node4", ran
    ev = [e for e in rig.lab.events if e[0] == "run"]
    assert len(ev[-1][2].split(",")) == 4, "the stage 3 test saw four visible chips"
    assert rig.lab.m.leases == {}, "every lab lease went back"


def test_the_lab_decision_for_stage_3_records_the_width(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    lab = [d for d in rig.decisions() if d.get("decision") == "hardware phase" and d.get("action") == "lab"]
    assert [d.get("chips") for d in lab] == [2, 4]


def test_stage_2_on_a_lab_still_leases_one_board(rig):
    make3(rig)
    abort_at_pause(rig)
    rig.run()
    assert lab_acquires(rig)[0] == 2 and len(stage_tests(rig, 2)[0]["chips"]) == 2


def test_a_lab_with_fewer_chips_than_the_configured_count_is_refused_before_a_lease(rig):
    rig.lab.chips = 2
    make3(rig)
    abort_at_pause(rig)                      # a block pauses the run for an operator; the hook ends it
    rig.run()
    assert lab_acquires(rig) == [2], "stage 3 asked the lab for nothing it cannot hold"
    assert not stage_tests(rig, 3)
    pauses = [str(d.get("reason", "")) for d in rig.decisions() if d.get("decision") == "pause"]
    assert any("4 chips" in r and "lab node4 has 2 chips" in r for r in pauses), pauses


def test_a_run_without_a_wider_configured_count_still_skips_stage_3_on_a_lab(tmp_path):
    r = Stage3LabRig(tmp_path, required=None)
    try:
        make3(r)
        abort_at_pause(r)
        r.run()
        assert [d["result"] for d in r.ends(3)] == ["skipped"] or not stage_tests(r, 3)
        assert lab_acquires(r) == [2]
        assert any(SKIP_3_FULL_PORT in str(d.get("reason", "")) or "skipped" in str(d) for d in r.decisions())
    finally:
        r.close()
