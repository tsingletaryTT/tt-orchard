# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/can_cad/check_cq.py and cq_sandbox.py (review of PR 629): the CadQuery checker runs model-written code only after the static checks pass,
runs it in a confined child, and takes its verdict from a file the trusted epilogue exported, never from what the code prints.

The static and sandbox tests need only the standard library. The geometry tests need cadquery and are skipped without it."""
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "can_cad"
sys.path.insert(0, str(SCRIPTS))
import check_cq  # noqa: E402
import cq_sandbox  # noqa: E402

HAS_CQ = importlib.util.find_spec("cadquery") is not None


def reply(code: str, header='{"summary":"x","units":"mm","assumptions":[],"open_questions":[]}', extra="") -> str:
    return f"{header}\n```python\n{code}\n```\n{extra}"


GOOD = "import cadquery as cq\nimport math\nresult = cq.Workplane('XY').box(1, 1, 1)"


# ---- static checks: nothing runs ------------------------------------------------------------------------------------------------------

def never_run(code, epilogue=""):
    raise AssertionError("generated code must not run for this reply")


def test_a_good_reply_passes_every_static_check():
    row = check_cq.static_checks(reply(GOOD))
    assert row["header_json"] and row["one_python_block"] and row["parses"] and row["imports_ok"] and row["ends_with_result"]


def test_two_python_blocks_fail_and_nothing_runs():
    text = reply(GOOD, extra="```python\nprint(1)\n```\n")
    row = check_cq.check_reply(text, run_code=never_run)
    assert row["one_python_block"] is False and row["python_blocks"] == 2 and row["executed"] is False


def test_every_alias_of_an_import_is_checked():
    row = check_cq.check_reply(reply("import math, os\nresult = 1"), run_code=never_run)
    assert row["imports_ok"] is False and row["executed"] is False
    assert "imports_ok" in row["not_executed_because"]


@pytest.mark.parametrize("line", ["from os import path", "import os.path", "from . import x", "import cadquery, subprocess", "from cadquery.occ_impl import shapes\nimport socket"])
def test_other_import_shapes(line):
    ok = line.startswith("from cadquery.occ_impl") and False
    row = check_cq.static_checks(reply(line + "\nresult = 1"))
    assert row["imports_ok"] is ok


def test_cadquery_submodules_and_math_are_allowed():
    assert check_cq.static_checks(reply("import cadquery.occ_impl as o\nfrom math import pi\nresult = 1"))["imports_ok"] is True


def test_the_last_top_level_statement_must_assign_result():
    assert check_cq.static_checks(reply("result = 1\nprint('later')"))["ends_with_result"] is False
    assert check_cq.static_checks(reply("def f():\n    result = 1\n"))["ends_with_result"] is False
    assert check_cq.static_checks(reply("x = 1\nresult = x"))["ends_with_result"] is True
    assert check_cq.static_checks(reply("result = 1\nresult"))["ends_with_result"] is False


def test_a_reply_that_does_not_parse_is_not_run():
    row = check_cq.check_reply(reply("import cadquery as cq\nresult = ("), run_code=never_run)
    assert row["parses"] is False and row["executed"] is False


# ---- the sandbox -----------------------------------------------------------------------------------------------------------------------

def test_the_environment_has_no_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_FOR_TEST")
    monkeypatch.setenv("SOME_API_KEY", "k")
    r = cq_sandbox.run_code("import os\nopen('env.txt','w').write(repr(sorted(os.environ.items())))\n")
    assert r["returncode"] == 0 and "env.txt" in r["files"]
    r2 = cq_sandbox.run_code("import os\nopen('home.txt','w').write(os.environ['HOME'])\n")
    assert r2["returncode"] == 0
    env = cq_sandbox.scrubbed_env("/w")
    assert set(env) == {"PATH", "HOME", "TMPDIR", "LANG", "PYTHONDONTWRITEBYTECODE"} and "HF_TOKEN" not in env


def test_generated_code_cannot_see_the_parents_token_even_when_it_asks(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_FOR_TEST")
    out = tmp_path / "seen.txt"
    r = cq_sandbox.run_code("import os\nprint('TOKEN=' + str(os.environ.get('HF_TOKEN')))\nraise SystemExit(7 if os.environ.get('HF_TOKEN') else 0)\n")
    assert r["returncode"] == 0, "the token reached the child"


def landlock(r):
    if not r["layers"].get("landlock"):
        pytest.skip("this kernel has no Landlock")
    return r["layers"]


def test_writes_outside_the_work_directory_are_refused(tmp_path):
    target = tmp_path / "outside.txt"
    r = cq_sandbox.run_code(f"open({str(target)!r}, 'w').write('x')\n")
    landlock(r)
    assert r["returncode"] != 0 and not target.exists()
    assert "Permission" in r["stderr"]


def test_writes_inside_the_work_directory_are_allowed():
    r = cq_sandbox.run_code("open('inside.txt','w').write('x')\n")
    assert r["returncode"] == 0 and r["files"].get("inside.txt") == 1


def test_tcp_connections_are_refused():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    try:
        r = cq_sandbox.run_code(f"import socket\ns = socket.socket()\ns.connect(('127.0.0.1', {port}))\n")
        layers = landlock(r)
        if not layers.get("tcp"):
            pytest.skip("Landlock here has no TCP rules")
        assert r["returncode"] != 0 and "Permission" in r["stderr"]
    finally:
        srv.close()


def test_a_runaway_script_is_killed_at_the_timeout():
    r = cq_sandbox.run_code("while True:\n    pass\n", timeout_s=1.0)
    assert r["timed_out"] is True


def test_the_layers_that_applied_are_reported():
    r = cq_sandbox.run_code("pass\n")
    assert r["layers"]["env"] == "scrubbed" and "limits" in r["layers"] and "landlock" in r["layers"]


# ---- the verdict comes from the exported file ------------------------------------------------------------------------------------------

needs_cq = pytest.mark.skipif(not HAS_CQ, reason="cadquery is not installed")


@needs_cq
def test_a_valid_solid_passes_and_its_volume_is_measured():
    row = check_cq.check_reply(reply(GOOD))
    assert row["executes"] and row["valid_solid"] is True and abs(row["volume"] - 1.0) < 1e-6


@needs_cq
def test_printing_VALID_does_not_make_a_wire_a_solid():
    code = "import cadquery as cq\nprint('VALID 1.0')\nresult = cq.Workplane('XY').circle(5)"
    row = check_cq.check_reply(reply(code))
    assert row["executes"] is True and row["valid_solid"] is False


@needs_cq
def test_a_result_that_is_not_a_shape_does_not_execute():
    row = check_cq.check_reply(reply("import cadquery as cq\nresult = 'VALID'"))
    assert row["executes"] is False and row["valid_solid"] is False


@needs_cq
def test_a_script_that_raises_is_not_valid():
    row = check_cq.check_reply(reply("import cadquery as cq\nraise ValueError('boom')\nresult = 1"))
    assert row["ends_with_result"] is True and row["executes"] is False


def test_verify_step_refuses_an_empty_or_broken_file():
    if not HAS_CQ:
        pytest.skip("cadquery is not installed")
    assert check_cq.verify_step(b"not a step file")["valid_solid"] is False


def test_main_counts_distinct_prompts_from_the_saved_rows(tmp_path):
    saved = {"decode": [{"prompt_index": i % 2, "repeat": i // 2, "text": reply("import os\nresult = 1")} for i in range(4)]}
    (tmp_path / "in.json").write_text(json.dumps(saved))
    assert check_cq.main(["x", str(tmp_path / "in.json"), str(tmp_path / "out.json")]) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["n"] == 4 and out["distinct_prompts"] == 2 and out["summary"]["imports_ok"] == 0 and out["executed"] == 0


def test_landlock_is_applied_whenever_the_kernel_has_it():
    """The other Landlock tests skip when the report says it is off, so this one asks the kernel itself."""
    import ctypes
    kernel_has = ctypes.CDLL(None, use_errno=True).syscall(444, None, 0, 1) >= 1
    if not kernel_has:
        pytest.skip("this kernel has no Landlock")
    assert cq_sandbox.run_code("pass\n")["layers"]["landlock"] is True


# ---- Landlock ABI table and the fail-closed path (second review of PR 629) --------------------------------------------------------------

FS_ABI3 = (1 << 15) - 1


@pytest.mark.parametrize("abi,fs_bits,net", [(1, 13, 0), (2, 14, 0), (3, 15, 0), (4, 15, 3), (5, 16, 3), (6, 16, 3), (7, 16, 3), (8, 16, 3)])
def test_the_filesystem_and_network_masks_follow_the_abi(abi, fs_bits, net):
    assert cq_sandbox.masks_for_abi(abi) == ((1 << fs_bits) - 1, net)


def test_abi_4_asks_for_no_ioctl_bit_and_the_same_files_as_abi_3():
    """On ABI 4 the IOCTL_DEV bit (15) is unknown: asking for it makes ruleset creation fail with EINVAL."""
    fs, net = cq_sandbox.masks_for_abi(4)
    assert fs == FS_ABI3 and not fs & (1 << 15) and net == 3
    assert cq_sandbox.masks_for_abi(0) == (0, 0)


class FakeLibc:
    """A libc whose syscall answers the Landlock calls from a table and records the ruleset attribute it was asked to create."""

    def __init__(self, abi, create_fails=False, restrict_fails=False):
        self.abi, self.create_fails, self.restrict_fails, self.attrs = abi, create_fails, restrict_fails, []
        self.fds = []

    def syscall(self, num, *args):
        import ctypes
        if num == cq_sandbox.SYS_CREATE:
            ref, size, flags = args
            if flags == 1:
                return self.abi
            self.attrs.append((ref._obj.handled_access_fs, ref._obj.handled_access_net, size))
            if self.create_fails:
                ctypes.set_errno(22)
                return -1
            fd = os.open(os.devnull, os.O_RDONLY)
            self.fds.append(fd)
            return fd
        if num == cq_sandbox.SYS_ADD_RULE:
            return 0
        if num == cq_sandbox.SYS_RESTRICT:
            return -1 if self.restrict_fails else 0
        raise AssertionError(num)

    def prctl(self, *args):
        return 0


def test_on_an_abi_4_kernel_the_ruleset_is_created_with_masks_the_kernel_accepts(tmp_path):
    libc = FakeLibc(abi=4)
    out = cq_sandbox._landlock(str(tmp_path), libc=libc)
    assert out["landlock"] is True and out["abi"] == 4 and out["tcp"] is True
    fs, net, _ = libc.attrs[0]
    assert fs == FS_ABI3 and net == 3


def test_a_failed_ruleset_creation_reports_landlock_false_and_require_enforced_refuses(tmp_path):
    out = cq_sandbox._landlock(str(tmp_path), libc=FakeLibc(abi=4, create_fails=True))
    assert out["landlock"] is False and "create failed" in out["reason"]
    with pytest.raises(cq_sandbox.SandboxUnavailable):
        cq_sandbox.require_enforced(out)
    out2 = cq_sandbox._landlock(str(tmp_path), libc=FakeLibc(abi=8, restrict_fails=True))
    assert out2["landlock"] is False and "restrict_self" in out2["reason"]
    with pytest.raises(cq_sandbox.SandboxUnavailable):
        cq_sandbox.require_enforced({"landlock": False, "reason": "kernel has no Landlock"})


def forced_failure_python(tmp_path):
    forced = tmp_path / "forced.py"
    forced.write_text(f"import sys\nsys.path.insert(0, {str(SCRIPTS)!r})\nimport cq_sandbox\n"
                      "cq_sandbox._landlock = lambda workdir, libc=None: {'landlock': False, 'reason': 'forced for test'}\n"
                      "sys.exit(cq_sandbox.main(sys.argv[1:]))\n")
    wrapper = tmp_path / "py"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {forced} \"$@\"\n")
    wrapper.chmod(0o755)
    return str(wrapper)


def test_when_the_sandbox_cannot_be_enforced_the_reply_is_not_run(tmp_path):
    r = cq_sandbox.run_code("open('ran.txt', 'w').write('x')\n", python=forced_failure_python(tmp_path))
    assert r["sandbox_unavailable"] is True and r["returncode"] == cq_sandbox.UNAVAILABLE_EXIT
    assert "ran.txt" not in r["files"] and "forced for test" in r["stderr"]


def test_check_reply_reports_sandbox_unavailable_and_counts_nothing_as_executed():
    fake = lambda code, epi="": {"returncode": 97, "timed_out": False, "sandbox_unavailable": True, "stderr": "sandbox unavailable: x", "layers": {"landlock": False}, "files": {}, "step_bytes": {}}  # noqa: E731
    row = check_cq.check_reply(reply(GOOD), run_code=fake)
    assert row["executed"] is False and row["sandbox_unavailable"] is True and row["executes"] is False and row["valid_solid"] is False


# ---- only the epilogue's own result.step is judged -------------------------------------------------------------------------------------

EPI = "open('result.step', 'w').write('FINAL')\n"


def test_other_step_files_are_ignored_and_result_step_is_the_artifact():
    r = cq_sandbox.run_code("open('a_first.step', 'w').write('INTERMEDIATE')\nopen('b.step', 'w').write('OTHER')\n", EPI)
    assert r["step_bytes"] == {"result.step": b"FINAL"} and "a_first.step" in r["files"]


def test_a_result_step_written_by_the_generated_code_is_not_the_artifact():
    r = cq_sandbox.run_code("open('result.step', 'w').write('FORGED')\n", EPI)
    assert r["step_bytes"] == {"result.step": b"FINAL"}


def test_a_script_that_writes_result_step_and_exits_early_has_no_artifact():
    r = cq_sandbox.run_code("open('result.step', 'w').write('FORGED')\nraise SystemExit(0)\n", EPI)
    assert r["step_bytes"] == {}


def test_an_epilogue_that_fails_leaves_no_artifact():
    r = cq_sandbox.run_code("open('result.step', 'w').write('FORGED')\n", "raise ValueError('export failed')\n")
    assert r["step_bytes"] == {} and r["returncode"] != 0


@needs_cq
def test_an_intermediate_solid_exported_first_does_not_decide_the_verdict():
    code = ("import cadquery as cq\ncq.exporters.export(cq.Workplane('XY').box(2, 2, 2), 'a_part.step')\n"
            "result = cq.Workplane('XY').circle(5)")
    row = check_cq.check_reply(reply(code))
    assert row["executes"] is True and row["valid_solid"] is False


# ---- output cannot fill the parent's memory --------------------------------------------------------------------------------------------

def test_a_stdout_flood_is_discarded_not_buffered_in_the_parent():
    import resource
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    r = cq_sandbox.run_code("import sys\nchunk = 'x' * (1 << 20)\nfor _ in range(300):\n    sys.stdout.write(chunk)\n", timeout_s=60)
    grown_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert r["returncode"] == 0 and grown_mb < 100, f"the parent grew by {grown_mb:.0f} MB"


def test_a_stderr_flood_is_stopped_by_the_file_size_limit_and_only_its_tail_is_read():
    r = cq_sandbox.run_code("import sys\nchunk = 'e' * (1 << 20)\nfor _ in range(300):\n    sys.stderr.write(chunk)\n", file_bytes=1 << 20, timeout_s=60)
    assert r["returncode"] != 0 and len(r["stderr"]) <= 300


def test_an_epilogue_that_writes_nothing_does_not_promote_a_result_step_the_script_wrote():
    r = cq_sandbox.run_code("open('result.step', 'w').write('FORGED')\n", "pass\n")
    assert r["step_bytes"] == {} and "result.step" not in r["files"]


def test_check_reply_judges_only_result_step_even_when_the_runner_returns_other_artifacts():
    fake = lambda code, epi="": {"returncode": 0, "timed_out": False, "sandbox_unavailable": False, "stderr": "", "layers": {"landlock": True},  # noqa: E731
                                 "files": {"part.step": 5}, "step_bytes": {"part.step": b"intermediate"}}
    row = check_cq.check_reply(reply(GOOD), run_code=fake)
    assert row["executes"] is False and row["valid_solid"] is False
