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
            with open(kw["log_path"], "a") as f:
                f.write("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n")
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


# ---- review of PR 112, third round: `started` comes from evidence that the kernel was reached -----------------------

def scripted_run(log_text, result=None, returncode=0, killed=False):
    """A fake run_watched: the smoke test passes; the kernel run writes `log_text`, and `result` when it is not None."""
    def fake(argv, **kw):
        if "--out" in argv:
            with open(kw["log_path"], "a") as f:
                f.write(log_text)
            if result is not None:
                Path(argv[argv.index("--out") + 1]).write_text(json.dumps(result))
            return kp.Watched(-9 if killed else returncode, "stalled: no output" if killed else None, killed, 2.0)
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\n")
        return kp.Watched(0, None, False, 1.0)
    return fake


RUN_MARKS = "PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n"


def hw_of(tmp_path, fake_kenv, runner, *, blocked=False):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    if blocked:
        with pytest.raises(kp.ProbeBlocked):
            kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=runner)
    else:
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=runner)
    return attempt


def test_a_gozer_failure_before_the_harness_ran_is_not_a_hardware_run(tmp_path, fake_kenv):
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("gozer: could not reach the chip\n", returncode=1))
    assert attempt["hw"]["category"] == "harness_error" and attempt["hw"]["started"] is False
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "ran on hardware passed" not in md and "hardware only: 1 attempt" not in md
    assert "attempt 1 was not judged on hardware: harness_error" in md


def test_a_setup_failure_before_the_kernel_was_reached_is_not_a_hardware_run(tmp_path, fake_kenv):
    result = {"phase": "hw", "sets": [], "error": {"phase": "setup", "type": "RuntimeError", "text": "no device"}}
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("PROGRESS import\nPROGRESS open_device\n", result))
    assert attempt["hw"]["category"] == "harness_error" and attempt["hw"]["started"] is False


def test_a_kernel_that_failed_to_import_never_reached_the_compiler(tmp_path, fake_kenv):
    result = {"phase": "hw", "sets": [], "error": {"phase": "import", "type": "SyntaxError", "text": "bad"}}
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("PROGRESS import\n", result))
    assert attempt["hw"]["category"] == "import_error" and attempt["hw"]["started"] is False


def test_a_run_marker_in_the_log_means_the_kernel_was_run(tmp_path, fake_kenv):
    result = {"phase": "hw", "sets": [], "error": {"phase": "run", "type": "TTLangCompileError", "text": "boom"}}
    attempt = hw_of(tmp_path, fake_kenv, scripted_run(RUN_MARKS, result))
    assert attempt["hw"]["started"] is True


def test_a_run_phase_error_in_the_result_means_the_kernel_was_run_even_without_the_log_line(tmp_path, fake_kenv):
    result = {"phase": "hw", "sets": [], "error": {"phase": "run", "type": "TTLangCompileError", "text": "boom"}}
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("", result))
    assert attempt["hw"]["started"] is True


def test_measured_sets_mean_the_kernel_was_run_even_without_the_log_line(tmp_path, fake_kenv):
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("", {"phase": "hw", "sets": [GOOD_HW_SET], "error": None}))
    assert attempt["hw"]["started"] is True and attempt["hw"]["passed"] is True


def test_a_hang_after_the_run_marker_is_a_started_run(tmp_path, fake_kenv):
    attempt = hw_of(tmp_path, fake_kenv, scripted_run(RUN_MARKS, killed=True), blocked=True)
    assert attempt["hw"]["category"] == "hang" and attempt["hw"]["started"] is True


def test_a_hang_before_the_kernel_was_reached_is_not_a_started_run(tmp_path, fake_kenv):
    attempt = hw_of(tmp_path, fake_kenv, scripted_run("PROGRESS import\nPROGRESS open_device\n", killed=True),
                    blocked=True)
    assert attempt["hw"]["category"] == "hang" and attempt["hw"]["started"] is False


def test_a_marker_the_kernel_prints_in_an_earlier_log_does_not_count(tmp_path, fake_kenv):
    """The log is appended to across runs; only the text this run wrote is read."""
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    log = kp.verdict_dir(tmp_path, "add", 1) / "hw-output.txt"
    log.write_text(RUN_MARKS)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=scripted_run("gozer: could not reach the chip\n", returncode=1))
    assert attempt["hw"]["started"] is False


def test_an_earlier_marker_does_not_make_a_later_hang_a_started_run(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    (kp.verdict_dir(tmp_path, "add", 1) / "hw-output.txt").write_text(RUN_MARKS)
    with pytest.raises(kp.ProbeBlocked):
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                        run_watched=scripted_run("PROGRESS import\n", killed=True))
    assert attempt["hw"]["category"] == "hang" and attempt["hw"]["started"] is False
