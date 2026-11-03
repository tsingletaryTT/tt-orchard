# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 420: repair rounds for a full-port stage 2 whose port check fails in seconds. This file holds the pieces that need no supervisor:
the failure line the check template prints, the reader that decides whether a failed test earns a repair round, the round counter, and
the fingerprint that shows what a repair step changed. tests/test_supervisor_port_repair.py runs the stage end to end."""
import hashlib
import json
import os

import pytest

from orchard import full_port, port_repair
from port_check_fakes import HONEST, make_run, run_template

BROKEN_LOAD = HONEST.replace('return {"dir": model_dir, "device": mesh_device}', "raise KeyError('model.layers.2.ple.conv1d.weight')")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- the failure line the template prints ----------------------------------------------------------------------------------------

def failure_line(done):
    return [ln for ln in done.stdout.splitlines() if ln.startswith(port_repair.FAILURE_TOKEN)]


def test_a_port_that_fails_at_load_prints_a_structured_failure_line_after_the_text(tmp_path):
    run, stage, _ = make_run(tmp_path, BROKEN_LOAD)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "FAILED: the port failed at 'load'" in done.stdout
    assert port_repair.parse_failure(done.stdout) == {"kind": "port", "op": "load"}


def test_a_check_that_could_not_start_says_setup_and_not_port(tmp_path):
    run, stage, _ = make_run(tmp_path)
    done = run_template(tmp_path, run, stage, chips=[])          # TT_VISIBLE_DEVICES names no chip
    assert done.returncode == 2 and port_repair.parse_failure(done.stdout)["kind"] == "setup"


def test_a_missing_stage_1_reference_says_setup(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (run / "stages" / "1" / "evidence" / "reference" / "generated-ids.json").write_text(json.dumps({"generated_ids": "x"}))
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and port_repair.parse_failure(done.stdout)["kind"] == "setup"


def test_a_missing_port_module_is_a_port_failure_the_coder_can_repair(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (stage / "port" / "port_model.py").unlink()
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and port_repair.parse_failure(done.stdout) == {"kind": "port", "op": None}


def test_a_line_the_port_prints_cannot_be_a_failure_line(tmp_path):
    # The worker's lines carry the prefix `port| `; only the template's own print starts a line with the token.
    spoof = BROKEN_LOAD.replace("raise KeyError", "print('PORT_CHECK_FAILURE {\"kind\": \"setup\", \"op\": null}')\n    raise KeyError")
    run, stage, _ = make_run(tmp_path, spoof)
    done = run_template(tmp_path, run, stage)
    assert port_repair.parse_failure(done.stdout) == {"kind": "port", "op": "load"}


def test_a_passing_check_prints_no_failure_line(tmp_path):
    run, stage, _ = make_run(tmp_path)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not failure_line(done)


# ---- parse_failure ---------------------------------------------------------------------------------------------------------------

def test_parse_takes_the_last_well_formed_line_at_the_start_of_a_line():
    text = ('port| PORT_CHECK_FAILURE {"kind": "setup", "op": null}\n'
            'PORT_CHECK_FAILURE {"kind": "port", "op": "load"}\n'
            'PORT_CHECK_FAILURE not json\n'
            'PORT_CHECK_FAILURE {"kind": "setup", "op": null}\n')
    assert port_repair.parse_failure(text) == {"kind": "setup", "op": None}


@pytest.mark.parametrize("line", ['PORT_CHECK_FAILURE {"kind": "elsewhere", "op": "load"}', 'PORT_CHECK_FAILURE {"op": "load"}',
                                  'PORT_CHECK_FAILURE {"kind": "port", "op": 5}', 'PORT_CHECK_FAILURE []', 'PORT_CHECK_FAILURE {"kind": 1}'])
def test_parse_refuses_a_line_with_the_wrong_shape(line):
    assert port_repair.parse_failure(line + "\n") is None


def test_parse_returns_none_for_no_line_or_a_non_string():
    assert port_repair.parse_failure("FAILED: x\n") is None and port_repair.parse_failure(None) is None


# ---- read_failure: what earns a repair round -------------------------------------------------------------------------------------

def record(tmp_path, *, returncode=2, timed_out=False, text='FAILED: the port failed at \'load\': "KeyError"\n'
           'PORT_CHECK_FAILURE {"kind": "port", "op": "load"}\n', output_sha=None, output_path="stages/2/evidence/hw-test-output.txt"):
    """A run directory with a stage 2 test-result.json and the output file it names, as the supervisor leaves them."""
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "evidence").mkdir(parents=True)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(text)
    marker = {"command": full_port.COMMAND, "returncode": returncode, "timed_out": timed_out, "seconds": 9.0, "chips": ["c"],
              "output": {"path": output_path, "sha256": output_sha or sha(text.encode())}}
    (stage / "test-result.json").write_text(json.dumps(marker))
    return run, stage


def test_a_port_failure_with_exit_code_2_earns_a_round(tmp_path):
    run, stage = record(tmp_path)
    assert port_repair.read_failure(run, stage) == {"kind": "port", "op": "load", "returncode": 2}


@pytest.mark.parametrize("over", [{"returncode": 0}, {"returncode": 1}, {"returncode": 4}, {"returncode": None}, {"timed_out": True},
                                  {"returncode": True}, {"returncode": 2.0}])
def test_only_exit_code_2_that_did_not_time_out_earns_a_round(tmp_path, over):
    run, stage = record(tmp_path, **over)
    assert port_repair.read_failure(run, stage) is None


def test_a_setup_failure_earns_no_round(tmp_path):
    run, stage = record(tmp_path, text='FAILED: x\nPORT_CHECK_FAILURE {"kind": "setup", "op": null}\n')
    assert port_repair.read_failure(run, stage) is None


def test_an_output_file_that_changed_after_the_record_earns_no_round(tmp_path):
    run, stage = record(tmp_path)
    (stage / "evidence" / "hw-test-output.txt").write_text('PORT_CHECK_FAILURE {"kind": "port", "op": "close"}\n')
    assert port_repair.read_failure(run, stage) is None


def test_a_record_that_names_another_output_file_earns_no_round(tmp_path):
    run, stage = record(tmp_path, output_path="stages/2/evidence/other.txt")
    assert port_repair.read_failure(run, stage) is None


@pytest.mark.parametrize("what", ["missing marker", "marker not json", "marker a list", "output missing", "output a link"])
def test_an_unreadable_record_earns_no_round(tmp_path, what):
    run, stage = record(tmp_path)
    marker, out = stage / "test-result.json", stage / "evidence" / "hw-test-output.txt"
    if what == "missing marker":
        marker.unlink()
    elif what == "marker not json":
        marker.write_text("{")
    elif what == "marker a list":
        marker.write_text("[]")
    elif what == "output missing":
        out.unlink()
    else:
        text = out.read_text()
        out.unlink()
        real = stage / "evidence" / "real.txt"
        real.write_text(text)
        os.symlink(real, out)
    assert port_repair.read_failure(run, stage) is None


# ---- rounds_used -----------------------------------------------------------------------------------------------------------------

def entry(event, stage, **data):
    return {"event": event, "stage": stage, "data": data}


def round_(stage, k):
    return entry("decision", stage, decision=port_repair.DECISION, round=k)


def test_rounds_count_from_the_attempts_own_start():
    entries = [entry("stage_start", 2, escalated=False, resumed=False), round_(2, 1), round_(2, 2)]
    assert port_repair.rounds_used(entries, 2) == 2


def test_a_resume_keeps_the_count_and_a_fresh_attempt_starts_again():
    entries = [entry("stage_start", 2, escalated=False, resumed=False), round_(2, 1), round_(2, 2),
               entry("stage_start", 2, escalated=False, resumed=True)]
    assert port_repair.rounds_used(entries, 2) == 2
    entries += [entry("escalate", 2), entry("stage_start", 2, escalated=True, resumed=False)]
    assert port_repair.rounds_used(entries, 2) == 0


def test_a_moved_aside_directory_starts_the_count_again():
    entries = [entry("stage_start", 2, escalated=False, resumed=False), round_(2, 1),
               entry("decision", 2, decision="moved the partial stage directory aside"), round_(2, 1)]
    assert port_repair.rounds_used(entries, 2) == 1


def test_another_stages_rounds_are_not_counted():
    assert port_repair.rounds_used([entry("stage_start", 2, resumed=False), round_(4, 1)], 2) == 0


# ---- the options -----------------------------------------------------------------------------------------------------------------

def test_the_default_is_three_and_the_limit_is_bounded():
    assert port_repair.DEFAULT_ROUNDS == 3 and port_repair.MAX_ROUNDS >= 3
    assert port_repair.check_rounds("2") == 2 and port_repair.check_rounds(0) == 0


@pytest.mark.parametrize("bad", [-1, 11, 1.5, True, "x", None, float("nan")])
def test_a_round_count_outside_zero_to_the_limit_is_refused(bad):
    with pytest.raises(ValueError, match="rounds"):
        port_repair.check_rounds(bad)


# ---- fingerprint -----------------------------------------------------------------------------------------------------------------

def test_the_fingerprint_shows_a_changed_added_and_removed_file(tmp_path):
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("2")
    before = port_repair.fingerprint(tmp_path)
    assert before == port_repair.fingerprint(tmp_path)
    (tmp_path / "a.py").write_text("3")
    (tmp_path / "sub" / "c.py").write_text("4")
    (tmp_path / "sub" / "b.py").unlink()
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["a.py", "sub/b.py", "sub/c.py"]


def test_the_fingerprint_of_a_missing_directory_is_empty_and_a_link_counts_as_a_change(tmp_path):
    assert port_repair.fingerprint(tmp_path / "none") == {}
    (tmp_path / "a.py").write_text("1")
    before = port_repair.fingerprint(tmp_path)
    (tmp_path / "a.py").unlink()
    os.symlink("/etc/hostname", tmp_path / "a.py")
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["a.py"]
