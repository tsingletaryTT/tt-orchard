# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 420: repair rounds for a full-port stage 2 whose port check fails in seconds. This file holds the pieces that need no supervisor:
the failure line the check template prints, the reader that decides whether a failed test earns a repair round, the round counter, and
the fingerprint that shows what a repair step changed. tests/test_supervisor_port_repair.py runs the stage end to end."""
import hashlib
import json
import os
import subprocess
import sys

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


# ---- a runtime failure at init is setup, a port import failure is not (review of PR 425) -----------------------------------------

def test_a_device_open_failure_at_init_is_setup_and_earns_no_round(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "fake-ttnn" / "ttnn.py").write_text(
        "class FabricConfig:\n    FABRIC_1D = 1\n    DISABLED = 0\ndef set_fabric_config(c): pass\n"
        "class MeshShape:\n    def __init__(self, r, c): pass\n"
        "def open_mesh_device(shape):\n    raise RuntimeError('no such device')\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "no such device" in done.stdout
    assert port_repair.parse_failure(done.stdout) == {"kind": "setup", "op": "init"}


def test_a_runtime_that_cannot_be_imported_at_init_is_setup_and_earns_no_round(tmp_path):
    # `import ttnn` is the runtime's own import (a missing dependency, a broken install). It used to run before the worker's first reply, so
    # the worker died and the parent called that a port failure at init. No edit to stages/2/port/ can fix it (review of PR 425, round 3).
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "fake-ttnn" / "ttnn.py").write_text("raise ModuleNotFoundError(\"No module named 'torch'\")\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "No module named 'torch'" in done.stdout
    assert port_repair.parse_failure(done.stdout) == {"kind": "setup", "op": "init"}


def straddling(secret: str, prefix: str = "KeyError: '") -> str:
    """An exception text whose password starts a few characters before the worker's 2,000-character cut (the worker puts `prefix` in front of the
    text), with the `@` far beyond it."""
    pad = 2000 - len(prefix) - len("https://bob:") - 11
    return "x" * pad + f"https://bob:{secret}@host/model"


def test_a_secret_cut_by_the_workers_2000_character_limit_never_leaves_the_worker(tmp_path):
    # The worker used to cut the message first, so the closing `@` was gone and no pattern downstream could tell what the prefix was (review of PR 432).
    secret = "S3CRETpassword" + "p" * 400
    run, stage, _ = make_run(tmp_path, BROKEN_LOAD.replace("model.layers.2.ple.conv1d.weight", straddling(secret)))
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "KeyError" in done.stdout
    assert "S3CRET" not in done.stdout + done.stderr and "bob" not in done.stdout


def test_a_secret_cut_by_the_limit_in_a_setup_error_never_leaves_the_worker_either(tmp_path):
    secret = "S3CRETpassword" + "p" * 400
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "fake-ttnn" / "ttnn.py").write_text(f"raise ModuleNotFoundError({straddling(secret, 'ModuleNotFoundError: ')!r})\n")
    done = run_template(tmp_path, run, stage)
    assert port_repair.parse_failure(done.stdout) == {"kind": "setup", "op": "init"}
    assert "S3CRET" not in done.stdout + done.stderr


def test_the_workers_clip_takes_secrets_out_before_the_cut_and_stays_fast_on_hostile_text():
    # Loaded by path in a child process with a hard timeout: a pattern that went quadratic would hang here, not in a worker the supervisor waits for.
    code = r"""
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("port_check_under_test", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(m)
worst = 0.0
for text in ("a" * 300000, "ab://" * 60000, "https://" + "u" * 300000, "a+." * 100000, "x" * 1989 + " hf_" + "A" * 30, "x" * 1975 + "https://bob:SECRETSECRET@host"):
    t0 = time.time(); out = m.clip(text); worst = max(worst, time.time() - t0)
    assert len(out) <= m.REPLY_CHARS
    assert "hf_A" not in out and "SECRET" not in out, out[-60:]
print("ok", round(worst, 3))
"""
    done = subprocess.run([sys.executable, "-I", "-c", code, str(full_port.TEMPLATE)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.startswith("ok"), done.stdout + done.stderr
    assert float(done.stdout.split()[1]) < 5.0


def test_a_runtime_that_exits_while_it_is_imported_is_setup_too(tmp_path):
    # A broken install can end the interpreter from inside the import (SystemExit). That is the runtime's failure as well.
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "fake-ttnn" / "ttnn.py").write_text("import sys\nsys.exit('runtime library not found')\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and port_repair.parse_failure(done.stdout) == {"kind": "setup", "op": "init"}


def test_a_runtime_that_cannot_be_imported_does_not_stand_in_for_a_port_import_failure(tmp_path):
    # Both failures happen at init. The runtime's is setup; the port module's own import error stays a port failure.
    run, stage, _ = make_run(tmp_path, "import module_that_does_not_exist\n" + HONEST)
    done = run_template(tmp_path, run, stage)
    assert port_repair.parse_failure(done.stdout) == {"kind": "port", "op": "init"}


def test_a_port_that_cannot_be_imported_at_init_stays_repairable(tmp_path):
    run, stage, _ = make_run(tmp_path, "import module_that_does_not_exist\n" + HONEST)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and port_repair.parse_failure(done.stdout) == {"kind": "port", "op": "init"}


# ---- the evidence entry that says a rerun ran (review of PR 432) ------------------------------------------------------------------

def entry(event, **data):
    return {"event": event, "stage": 2, "data": data}


HW = dict(what="hardware test", path="stages/2/test-result.json", sha256="a" * 64, returncode=0, timed_out=False)


def test_a_hardware_test_entry_after_the_rerun_decision_says_the_rerun_ran():
    entries = [entry("stage_start", resumed=False), entry("decision", decision=port_repair.RERUN, round=1, record=None), entry("evidence", **HW)]
    assert port_repair.rerun_recorded(entries, 2) == HW


@pytest.mark.parametrize("name,evidence", [
    ("an agent's evidence file", dict(what="evidence file", path="stages/2/test-result.json", sha256="a" * 64)),
    ("a hardware test entry with no exit code", {k: v for k, v in HW.items() if k != "returncode"}),
    ("a hardware test entry with no timeout flag", {k: v for k, v in HW.items() if k != "timed_out"}),
    ("another kind of entry", dict(HW, what="transcript"))])
def test_only_the_supervisors_hardware_test_entry_counts(name, evidence):
    entries = [entry("stage_start", resumed=False), entry("decision", decision=port_repair.RERUN, round=1, record=None), entry("evidence", **evidence)]
    assert port_repair.rerun_recorded(entries, 2) is None, name


def test_an_entry_before_the_rerun_decision_or_after_its_tested_decision_does_not_count():
    before = [entry("stage_start", resumed=False), entry("evidence", **HW), entry("decision", decision=port_repair.RERUN, round=1, record=None)]
    assert port_repair.rerun_recorded(before, 2) is None
    closed = before[:1] + before[2:] + [entry("evidence", **HW), entry("decision", decision=port_repair.TESTED, sources="x")]
    assert port_repair.rerun_recorded(closed, 2) is None


def test_a_hardware_test_entry_with_no_rerun_decision_at_all_does_not_count():
    assert port_repair.rerun_recorded([entry("stage_start", resumed=False), entry("evidence", **HW)], 2) is None


def test_an_entry_of_an_earlier_attempt_or_another_stage_does_not_count():
    earlier = [entry("stage_start", resumed=False), entry("decision", decision=port_repair.RERUN, round=1, record=None), entry("evidence", **HW),
               entry("stage_start", resumed=False)]
    assert port_repair.rerun_recorded(earlier, 2) is None
    other = [dict(e, stage=4) for e in earlier[:3]]
    assert port_repair.rerun_recorded(other, 2) is None


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


def test_a_prefixed_line_after_the_real_one_does_not_replace_it():
    text = 'PORT_CHECK_FAILURE {"kind": "port", "op": "load"}\nport| PORT_CHECK_FAILURE {"kind": "setup", "op": null}\n'
    assert port_repair.parse_failure(text) == {"kind": "port", "op": "load"}


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


# ---- the failure text is redacted whole, then cut (review of PR 425) --------------------------------------------------------------

def test_safe_failure_text_removes_user_information_whose_at_sign_is_past_the_cut():
    secret = "S3CRETpassword" + "p" * 400
    raw = f'FAILED: the port failed at \'load\': "OSError: https://bob:{secret}@host/model"\nPORT_CHECK_FAILURE {{"kind": "port", "op": "load"}}\n'
    text = port_repair.safe_text(raw, ("/home/x",))
    assert "S3CRET" not in text and "S3CRET" not in port_repair.failed_line(text)
    assert "https://host/model" in text


def test_safe_failure_text_hides_a_home_path_and_a_token():
    text = port_repair.safe_text("FAILED: /home/x/a hf_abcdefgh12345678\n", ("/home/x",))
    assert "/home/x" not in text and "hf_abcdefgh" not in text


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


def test_a_link_that_points_somewhere_else_is_a_change(tmp_path):
    os.symlink("/etc/hostname", tmp_path / "a.py")
    before = port_repair.fingerprint(tmp_path)
    (tmp_path / "a.py").unlink()
    os.symlink("/etc/passwd", tmp_path / "a.py")
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["a.py"]


def _big(path, size, fill=b"x"):
    """A file one byte over the fingerprint's read cap, written sparsely so the test stays fast."""
    with open(path, "wb") as fh:
        fh.write(fill * 16)
        fh.truncate(size)


def test_two_different_files_over_the_read_cap_do_not_fingerprint_alike(tmp_path):
    # issue 453 item 2: every file over the cap used to become the same string, so an edit to one counted as no change
    cap = full_port.JSON_MAX
    _big(tmp_path / "w.bin", cap + 1)
    before = port_repair.fingerprint(tmp_path)
    assert not before["w.bin"].startswith("unreadable")
    # an edit at the start, one in the last block, and a length change are each a change
    for edit in ("head", "tail", "size"):
        _big(tmp_path / "w.bin", cap + 1)
        with open(tmp_path / "w.bin", "r+b") as fh:
            if edit == "head":
                fh.write(b"EDIT")
            elif edit == "tail":
                fh.seek(cap - 8)
                fh.write(b"EDIT")
            else:
                fh.truncate(cap + 2)
        after = port_repair.fingerprint(tmp_path)
        assert port_repair.changed(before, after) == ["w.bin"], edit
    # the same untouched file twice is the same fingerprint (a rewrite with identical bytes changes the modification time and counts as a change: the sample can
    # notice a change and cannot prove there was none)
    _big(tmp_path / "w.bin", cap + 1)
    again = port_repair.fingerprint(tmp_path)
    assert again == port_repair.fingerprint(tmp_path)


def test_an_edit_to_a_large_file_is_seen_even_when_the_modification_time_is_put_back(tmp_path):
    # the first and last blocks carry the check on their own: an edit that restores the size and the time is still a change
    cap = full_port.JSON_MAX
    _big(tmp_path / "w.bin", cap + 1)
    st = os.stat(tmp_path / "w.bin")
    before = port_repair.fingerprint(tmp_path)
    for offset in (0, cap - 8):
        with open(tmp_path / "w.bin", "r+b") as fh:
            fh.seek(offset)
            fh.write(b"EDIT")
        os.utime(tmp_path / "w.bin", ns=(st.st_atime_ns, st.st_mtime_ns))
        assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["w.bin"], offset
        with open(tmp_path / "w.bin", "r+b") as fh:
            fh.seek(offset)
            fh.write(b"\0\0\0\0")
        os.utime(tmp_path / "w.bin", ns=(st.st_atime_ns, st.st_mtime_ns))


def test_a_link_named_pycache_is_still_a_difference(tmp_path):
    (tmp_path / "a.py").write_text("1")
    before = port_repair.fingerprint(tmp_path)
    os.symlink("/etc", tmp_path / "__pycache__")
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["__pycache__"]


def test_sample_blocks_refuses_a_fifo_and_a_link_and_reads_only_two_blocks(tmp_path):
    from orchard import safeio
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(safeio.SafeReadError):
        safeio.sample_blocks(tmp_path / "pipe", block=4)
    with pytest.raises(safeio.NotRegular):  # a device reads without an error, so only the type check refuses it
        safeio.sample_blocks("/dev/null", block=4)
    (tmp_path / "f").write_bytes(b"0123456789")
    os.symlink(tmp_path / "f", tmp_path / "l")
    with pytest.raises(safeio.SafeReadError):
        safeio.sample_blocks(tmp_path / "l", block=4)
    size, _, head, tail = safeio.sample_blocks(tmp_path / "f", block=4)
    assert (size, head, tail) == (10, b"0123", b"6789")
    (tmp_path / "s").write_bytes(b"ab")
    assert safeio.sample_blocks(tmp_path / "s", block=4)[2:] == (b"ab", b"ab")
    (tmp_path / "e").write_bytes(b"")
    assert safeio.sample_blocks(tmp_path / "e", block=4)[2:] == (b"", b"")
    with pytest.raises(ValueError):
        safeio.sample_blocks(tmp_path / "f", block=0)


def test_a_large_file_that_is_not_regular_is_still_unreadable_not_followed(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    fp = port_repair.fingerprint(tmp_path)
    assert fp["pipe"].startswith("unreadable")


def test_bytecode_directories_are_not_part_of_the_fingerprint(tmp_path):
    # issue 453 item 3: a coder running python against the port makes __pycache__/*.pyc appear; that is not an edit
    (tmp_path / "a.py").write_text("1")
    before = port_repair.fingerprint(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0\1")
    (tmp_path / "sub" / "__pycache__").mkdir(parents=True)
    (tmp_path / "sub" / "__pycache__" / "b.pyc").write_bytes(b"x")
    assert port_repair.fingerprint(tmp_path) == before
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == []


def test_a_source_file_next_to_the_bytecode_directory_still_counts(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"x")
    (tmp_path / "a.py").write_text("1")
    before = port_repair.fingerprint(tmp_path)
    (tmp_path / "a.py").write_text("2")
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == ["a.py"]


# ---- the narration ---------------------------------------------------------------------------------------------------------------

def narrated_lines(tmp_path, *decisions):
    from orchard.ledger import Ledger
    from test_supervisor_graft import narrated
    run = tmp_path / "run"
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="x/y")
        for data in decisions:
            led.append("decision", 2, **data)
    return narrated(run)


def test_a_round_is_narrated_with_the_failure_and_what_happens_next(tmp_path):
    out = narrated_lines(tmp_path, {"decision": port_repair.DECISION, "round": 1, "of": 3, "op": "load", "failure": "the port failed at 'load': KeyError"})
    [line] = [x for x in out if "port repair round 1 of 3" in x]
    assert "at 'load'" in line and "KeyError" in line and "same check runs again" in line


def test_the_end_of_the_rounds_says_why(tmp_path):
    out = narrated_lines(tmp_path, {"decision": port_repair.STOPPED, "reason": "spent", "rounds": 3, "of": 3},
                         {"decision": port_repair.STOPPED, "reason": "unchanged", "rounds": 1, "of": 3},
                         {"decision": "port repair refused", "round": 1, "files": ["test-result.json"]})
    joined = "\n".join(out)
    assert "all the rounds were used" in joined and "changed nothing in the port" in joined and "test-result.json, which only the supervisor writes" in joined


# ---- Copilot's second review of PR 425: the ledger readers for a rerun that was started ----------------------------------------------------

def entry(event, stage=2, **data):
    return {"event": event, "stage": stage, "data": data}


RERUN = entry("decision", decision=port_repair.RERUN, round=1, record={"path": "stages/2/test-result.json", "sha256": "a" * 64})
TESTED = entry("decision", decision=port_repair.TESTED, sources="b" * 64)
KEPT = entry("evidence", what="port repair round output", round=1, path="x", sha256="c" * 64)


def test_a_started_rerun_is_in_flight_until_a_tested_entry_follows_it():
    assert port_repair.rerun_inflight([RERUN], 2) == RERUN["data"]
    assert port_repair.rerun_inflight([RERUN, TESTED], 2) is None
    assert port_repair.rerun_inflight([RERUN, TESTED, {**RERUN, "data": {**RERUN["data"], "round": 2}}], 2)["round"] == 2


def test_an_in_flight_rerun_of_another_stage_or_another_attempt_does_not_count():
    assert port_repair.rerun_inflight([{**RERUN, "stage": 4}], 2) is None
    fresh = entry("stage_start", resumed=False)
    assert port_repair.rerun_inflight([RERUN, fresh], 2) is None                              # a new attempt starts without it
    assert port_repair.rerun_inflight([RERUN, entry("stage_start", resumed=True)], 2) == RERUN["data"]      # a resume keeps it
    moved = entry("decision", decision=sorted(port_repair.MOVE_ASIDE_DECISIONS)[0])
    assert port_repair.rerun_inflight([RERUN, moved], 2) is None


def test_a_kept_output_belongs_to_its_round_stage_and_attempt():
    assert port_repair.output_kept([KEPT], 2, 1) and not port_repair.output_kept([KEPT], 2, 2)
    assert not port_repair.output_kept([{**KEPT, "stage": 4}], 2, 1)
    assert not port_repair.output_kept([KEPT, entry("stage_start", resumed=False)], 2, 1)
    assert port_repair.output_kept([KEPT, entry("stage_start", resumed=True)], 2, 1)
