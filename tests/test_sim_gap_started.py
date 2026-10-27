"""Review of PR 112, second finding: an `hw` block exists for attempts whose hardware run never started (the checked copy
failed a preflight check, or gozer queued the request). Those are not hardware judgments: the block says `started: false`,
the report does not count them as hardware runs and says why they were not judged."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import kernelprobe as kp
from test_kernelprobe import GOOD_HW_SET, fake_kenv, hw_writer, _passed_attempt  # noqa: F401
from test_sim_gap import report_for, row_of, sim_gap_attempt

def test_hardware_that_ran_is_marked_started(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    assert attempt["hw"]["started"] is True


def test_a_kernel_run_that_hung_is_marked_started(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)

    def hangs(argv, **kw):
        if "--out" in argv:
            return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)
    with pytest.raises(kp.ProbeBlocked):
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=hangs)
    assert attempt["hw"]["category"] == "hang" and attempt["hw"]["started"] is True


def test_a_rejected_checked_copy_is_not_a_hardware_run(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    clean = kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py"
    clean.write_text(clean.read_text() + "\n# edited after the simulator check\n")   # still a valid kernel
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    assert attempt["hw"]["started"] is False and attempt["hw"]["category"] == "harness_error"
    md = report_for(tmp_path, fake_kenv, [attempt])
    cells = row_of(md)
    assert cells["hardware"].startswith("not run") and "(hardware only)" in cells["hardware"]
    assert "ran on hardware passed" not in md                          # nothing is counted as a hardware run
    assert "attempt 1 was not judged on hardware" in md and "harness_error" in md


def test_a_forbidden_kernel_is_not_a_hardware_run(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel="forbidden_import.py")
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    assert attempt["hw"]["started"] is False and attempt["hw"]["category"] == "forbidden_code"
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "ran on hardware passed" not in md and "attempt 1 was not judged on hardware" in md
    assert row_of(md)["hardware"] == "not run"


def test_a_queued_request_is_not_a_hardware_run(tmp_path, fake_kenv):
    from test_kernelprobe import Canceller, queued_fake
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    with pytest.raises(kp.ProbeBlocked):
        kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=queued_fake([]),
                          cancel=Canceller())
    assert attempt["hw"]["category"] == "queued" and attempt["hw"]["started"] is False
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "ran on hardware passed" not in md
    assert "attempt 1 was not judged on hardware: queued" in md
    assert "hardware only: 1 attempt" not in md                        # no hardware result exists
    assert "was not run on hardware" in md                             # the unjudged explanation is there


def test_a_record_without_the_started_key_counts_as_a_hardware_run(tmp_path, fake_kenv):
    """Reports of earlier probes have hw blocks with no `started`; they all came from a run."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    attempt["hw"] = {"passed": True, "category": None, "pcc_min": 0.999, "sets": [], "first_line": ""}
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "1 of 1 attempts that ran on hardware passed" in md


def test_one_run_and_one_rejection_are_counted_separately(tmp_path, fake_kenv):
    a1, first = _passed_attempt(tmp_path, fake_kenv, n=1)
    a2, second = _passed_attempt(tmp_path, fake_kenv, n=2)
    (kp.verdict_dir(tmp_path, "add", 2) / kp.CLEAN_DIR / "kernel.py").write_text("# changed\n")
    kp.run_hardware([first, second], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    md = report_for(tmp_path, fake_kenv, [first, second])
    assert "1 of 1 attempts that ran on hardware passed" in md
    assert "attempt 2 was not judged on hardware" in md and "attempt 1 was not judged" not in md


def test_a_missing_checked_copy_is_not_a_hardware_run(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    (kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py").unlink()
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    assert attempt["hw"]["started"] is False and attempt["hw"]["category"] == "no_kernel"


def test_a_rejected_attempt_is_part_of_the_probe_stopped_before_the_rest_count(tmp_path, fake_kenv):
    a1, first = _passed_attempt(tmp_path, fake_kenv, n=1)
    a2, second = _passed_attempt(tmp_path, fake_kenv, n=2)
    clean = kp.verdict_dir(tmp_path, "add", 2) / kp.CLEAN_DIR / "kernel.py"
    clean.write_text(clean.read_text() + "\n# edited\n")
    kp.run_hardware([first, second], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    header = kp.report_header(env=fake_kenv, coder_model="m", probe_id="p", ops=["add"], tt_lang=kp_facts(),
                              hw={"available": True, "smoke_tail": ["PASSED!"]})
    header["stopped"] = "attempt 2 crashed"
    kp.write_report(tmp_path / "out", [first, second], header)
    md = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "1 of 1 attempts that ran on hardware passed; the probe stopped before the rest" in md


def kp_facts():
    from test_kernelprobe import FACTS
    return FACTS
