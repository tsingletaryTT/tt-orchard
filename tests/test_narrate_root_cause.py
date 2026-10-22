# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The first real error in a server log, quoted in BLOCKED.md and on the screen.

A server that dies at start ends its log with a wrapper ("Engine core initialization failed. See root cause
above."). The cause is lines earlier. BAAI/AREX-2's stage 2 failed twice on `no eos_token_id found for the served
checkpoint`, and the block said only the wrapper. The fixture is a trimmed copy of that log."""
import json
import os
from pathlib import Path

import pytest

from orchard import blocked, narrate
from orchard.ledger import Ledger, read_entries

FIXTURE = Path(__file__).parent / "fixtures" / "server_log_no_eos.txt"
CAUSE = "RuntimeError: no eos_token_id found for the served checkpoint"


def test_the_real_log_gives_its_first_error_and_its_line_number():
    found = narrate.root_cause(FIXTURE.read_text())
    assert found == (40, CAUSE)


def test_the_wrapper_alone_is_not_a_cause():
    text = "INFO starting\n(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}\n"
    assert narrate.root_cause(text) is None


@pytest.mark.parametrize("line, expect", [
    ("(EngineCore pid=7) ERROR 10-21 16:59:10 [core.py:1330] ValueError: bad: thing", "ValueError: bad: thing"),
    ("(APIServer pid=2) KeyError: 'layers'", "KeyError: 'layers'"),
    ("torch.OutOfMemoryError: out of memory", "torch.OutOfMemoryError: out of memory"),
    ("huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error", "huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error"),
    ("RuntimeError: Read 0xffffffff over PCIe ID 2: the board should be reset", "RuntimeError: Read 0xffffffff over PCIe ID 2: the board should be reset"),
    ("AssertionError", "AssertionError"),
])
def test_exception_lines_are_found_after_their_prefixes_are_stripped(line, expect):
    assert narrate.root_cause("INFO ok\n" + line + "\n") == (2, expect)


@pytest.mark.parametrize("line", [
    '  File "/x/y.py", line 3, in run',
    '    raise RuntimeError("boom")',
    "(EngineCore pid=1)     raise RuntimeError(\"boom\")",
    "INFO 10-21 [x:1] no error here",
    "Error when re-initializing chips!",
    "the word Error: in the middle of text",
    "Traceback (most recent call last):",
    "(EngineCore pid=1) ERROR 10-21 16:59:10 [core.py:1330]     ValueError: indented, so not a statement",
    "  ValueError: indented",
    "(EngineCore pid=1)     ValueError: indented after a process tag",
])
def test_frames_code_and_prose_are_not_causes(line):
    assert narrate.root_cause(line + "\n") is None


def test_the_first_error_wins_even_when_a_later_one_is_more_specific():
    text = "ValueError: first\nRuntimeError: second\n"
    assert narrate.root_cause(text) == (1, "ValueError: first")


def test_the_wrapper_before_the_cause_is_skipped():
    text = "RuntimeError: Engine core initialization failed. See root cause above.\nKeyError: 'x'\n"
    assert narrate.root_cause(text) == (2, "KeyError: 'x'")


def test_a_kernel_build_failure_counts():
    text = "trisc1 build failed ... 'vLut8si' is not a member of 'sfpi'\n"
    assert narrate.root_cause(text) == (1, "trisc1 build failed ... 'vLut8si' is not a member of 'sfpi'")


@pytest.mark.parametrize("prefix", ["", "(EngineCore pid=7) ", "(EngineCore pid=7) ERROR 10-21 16:59:10 [core.py:1330] "])
def test_an_indented_source_line_that_mentions_a_build_failure_is_not_the_cause(prefix):
    # A traceback frame shows the source line that raised: its text says "build failed" and it is indented.
    # The cause is the exception that follows it.
    text = (prefix + '  File "jit.py", line 9, in compile\n'
            + prefix + '    raise RuntimeError("trisc1 build failed: " + compiler_stderr)\n'
            + prefix + "RuntimeError: trisc1 build failed: 'vLut8si' is not a member of 'sfpi'\n")
    assert narrate.root_cause(text) == (3, "RuntimeError: trisc1 build failed: 'vLut8si' is not a member of 'sfpi'")


def test_a_very_long_line_is_cut():
    found = narrate.root_cause("RuntimeError: " + "x" * 2000 + "\n")
    assert found[0] == 1 and len(found[1]) <= 300 and found[1].endswith("...")


def test_a_log_over_the_read_limit_is_read_up_to_the_limit_only(tmp_path):
    big = tmp_path / "server.log"
    with open(big, "w") as f:
        f.write("ValueError: early\n")
        f.write("x" * (narrate.ROOT_CAUSE_MAX_BYTES + 10) + "\n")
        f.write("KeyError: late\n")
    assert narrate.root_cause_in_file(big) == (1, "ValueError: early")
    only_late = tmp_path / "late.log"
    only_late.write_text("x" * (narrate.ROOT_CAUSE_MAX_BYTES + 10) + "\nKeyError: late\n")
    assert narrate.root_cause_in_file(only_late) is None


def test_a_missing_or_unreadable_file_gives_none(tmp_path):
    assert narrate.root_cause_in_file(tmp_path / "nope.log") is None
    assert narrate.root_cause_in_file(tmp_path) is None


# ---- in the digest and in BLOCKED.md ---------------------------------------------------------------------

def write_ledger(run, *events):
    with Ledger(run / "ledger.jsonl") as led:
        for ev, stage, data in events:
            led.append(ev, stage, **data)


FAILED_TEST = [("stage_start", 2, {"escalated": False, "resumed": False}),
               ("evidence", 2, {"what": "hardware test", "returncode": 1, "timed_out": False, "path": "p", "sha256": "s"})]


def put_logs(stage_dir: Path, server_log: str, output: str = "serve_and_compare: the server exited with code 1\n"):
    ev = stage_dir / "evidence"
    ev.mkdir(parents=True)
    (ev / "server.log").write_text(server_log)
    (ev / "hw-test-output.txt").write_text(output)
    return ev / "server.log"


def test_the_digest_quotes_the_first_error_before_the_tests_last_output(tmp_path):
    put_logs(tmp_path / "stages/2", FIXTURE.read_text(), "tail line one\nRuntimeError: Engine core initialization failed. See root cause above.\n")
    write_ledger(tmp_path, *FAILED_TEST)
    lines = narrate.what_was_tried(tmp_path, stage=2)
    text = "\n".join(lines)
    assert "(stages/2/evidence/server.log, line 40)" in text and CAUSE in text
    assert text.index(CAUSE) < text.index("The hardware test's last output")


def test_no_cause_in_the_log_leaves_the_digest_as_it_was(tmp_path):
    put_logs(tmp_path / "stages/2", "INFO fine\nINFO still fine\n", "loading\nlast words\n")
    write_ledger(tmp_path, *FAILED_TEST)
    text = "\n".join(narrate.what_was_tried(tmp_path, stage=2))
    assert "first error" not in text and "last words" in text


def test_a_passing_test_gets_no_quote(tmp_path):
    put_logs(tmp_path / "stages/2", FIXTURE.read_text())
    write_ledger(tmp_path, ("stage_start", 2, {"escalated": False, "resumed": False}),
                 ("evidence", 2, {"what": "hardware test", "returncode": 0, "timed_out": False, "path": "p", "sha256": "s"}))
    assert CAUSE not in "\n".join(narrate.what_was_tried(tmp_path, stage=2))


def test_only_the_newest_attempts_log_is_quoted(tmp_path):
    old = put_logs(tmp_path / "stages/2.partial-1", FIXTURE.read_text())
    new = put_logs(tmp_path / "stages/2", "INFO fine\n")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    write_ledger(tmp_path, *FAILED_TEST)
    assert CAUSE not in "\n".join(narrate.what_was_tried(tmp_path, stage=2))


def test_a_stage_4_configuration_log_is_found(tmp_path):
    put_logs(tmp_path / "stages/4/configs/2", FIXTURE.read_text())
    write_ledger(tmp_path, ("stage_start", 4, {"escalated": False, "resumed": False}),
                 ("evidence", 4, {"what": "hardware test", "returncode": 1, "timed_out": False, "path": "p", "sha256": "s"}))
    text = "\n".join(narrate.what_was_tried(tmp_path, stage=4))
    assert "(stages/4/configs/2/evidence/server.log, line 40)" in text and CAUSE in text


def test_blocked_md_quotes_the_first_error(tmp_path):
    put_logs(tmp_path / "stages/2", FIXTURE.read_text())
    write_ledger(tmp_path, ("run_start", None, {"model": "BAAI/AREX-2"}), *FAILED_TEST,
                 ("decision", 2, {"decision": "pause", "reason": "stage 2 failed after escalation"}))
    blocked.write_bundle(tmp_path, read_entries(tmp_path / "ledger.jsonl"), "stage-failed", "stage 2 failed", now=0)
    md = (tmp_path / "BLOCKED.md").read_text()
    assert CAUSE in md.split("## What was tried")[1].split("## How to unblock")[0]


# ---- on the screen, as the run goes ----------------------------------------------------------------------

import io

from datetime import timezone

from orchard import ui


def live(run, *, replay=True):
    out = io.StringIO()
    n = narrate.Narrator(run, ui.PLAIN, out, replay=replay,
                         clock=lambda: 0.0, heartbeat_s=60, tz=timezone.utc)
    n.poll()
    return " ".join(out.getvalue().split())          # the narrator wraps long lines


def test_a_failed_test_prints_the_servers_first_error_before_the_last_output(tmp_path):
    put_logs(tmp_path / "stages/2", FIXTURE.read_text(), "loading\nRuntimeError: Engine core initialization failed. See root cause above.\n")
    write_ledger(tmp_path, *FAILED_TEST)
    text = live(tmp_path)
    assert "first error in the server log (stages/2/evidence/server.log, line 40)" in text and CAUSE in text
    assert text.index(CAUSE) < text.index("last output")


def test_a_passing_test_prints_no_first_error(tmp_path):
    put_logs(tmp_path / "stages/2", FIXTURE.read_text())
    write_ledger(tmp_path, ("stage_start", 2, {"escalated": False, "resumed": False}),
                 ("evidence", 2, {"what": "hardware test", "returncode": 0, "timed_out": False, "path": "p", "sha256": "s"}))
    assert "first error" not in live(tmp_path)


def test_a_failed_test_with_no_error_in_the_log_prints_only_the_last_output(tmp_path):
    put_logs(tmp_path / "stages/2", "INFO fine\n", "loading\nlast words here\n")
    write_ledger(tmp_path, *FAILED_TEST)
    text = live(tmp_path)
    assert "first error" not in text and "last words here" in text
