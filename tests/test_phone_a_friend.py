# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/phone_a_friend.py: when the helper is due, what is sent, and what is done with the reply. No supervisor, no network.

The supervisor wiring (the ledger entries, the coder's prepare note, the verdict) is in tests/test_supervisor_phone_a_friend.py."""
import os
import stat

import pytest

from orchard import phone_a_friend as paf


def entry(seq, event, stage=2, **data):
    return {"seq": seq, "event": event, "stage": stage, "data": data}


def decision(seq, name, stage=2, **data):
    return entry(seq, "decision", stage, decision=name, **data)


WATCHDOG_ESCALATE = {"agent": "stage-agent", "findings": [{"detector": "no_file_written", "summary": "20 turns and no file"}]}


# ---- when -----------------------------------------------------------------------------------------------------------------------------------

def test_not_due_before_any_attempt_has_failed():
    entries = [entry(1, "stage_start", escalated=False)]
    ok, why = paf.due(paf.attempt_state(entries, 2))
    assert not ok and "no attempt" in why


def test_due_after_a_failed_attempt_and_the_reason_says_a_watchdog_ended_it():
    entries = [entry(1, "stage_start", escalated=False), entry(2, "escalate", **WATCHDOG_ESCALATE),
               entry(3, "stage_end", result="escalate", reasons=["x"]), entry(4, "stage_start", escalated=True)]
    state = paf.attempt_state(entries, 2)
    ok, why = paf.due(state)
    assert ok and state["watchdog_ended"] and "watchdog" in why


def test_a_failed_gate_makes_it_due_without_a_watchdog_reason():
    # The stage machine's own escalate entry (by="stage machine") is not the watchdog's.
    entries = [entry(1, "stage_start"), entry(2, "escalate", by="stage machine", reasons=["gate"]),
               entry(3, "stage_end", result="escalate", reasons=["gate"]), entry(4, "stage_start", escalated=True)]
    state = paf.attempt_state(entries, 2)
    ok, why = paf.due(state)
    assert ok and not state["watchdog_ended"] and "failed" in why


def test_a_pass_is_not_a_failed_attempt_and_another_stage_does_not_count():
    entries = [entry(1, "stage_start", stage=1), entry(2, "stage_end", stage=1, result="escalate"),
               entry(3, "stage_start"), entry(4, "stage_end", result="pass")]
    assert paf.attempt_state(entries, 2)["failed"] == 0


def test_one_call_per_attempt_from_the_ledger():
    entries = [entry(1, "stage_start"), entry(2, "stage_end", result="escalate"), entry(3, "stage_start"),
               decision(4, paf.ASKED, k=1)]
    state = paf.attempt_state(entries, 2)
    ok, why = paf.due(state)
    assert not ok and "already asked in this attempt" in why


def test_two_calls_per_stage_from_the_ledger():
    entries = [entry(1, "stage_start"), decision(2, paf.ASKED, k=1), entry(3, "stage_end", result="escalate"),
               entry(4, "stage_start"), decision(5, paf.ASKED, k=2), entry(6, "stage_end", result="escalate"), entry(7, "stage_start")]
    state = paf.attempt_state(entries, 2)
    assert state["asked_in_stage"] == 2 and state["asked_in_attempt"] == 0
    ok, why = paf.due(state)
    assert not ok and "2 times" in why


# ---- what is sent -------------------------------------------------------------------------------------------------------------------------

HOMES = ("/home/ttuser",)
TOKEN = "hf_" + "A" * 30


def test_the_task_is_scrubbed_whole_and_every_part_is_capped():
    brief = f"read /home/ttuser/orchard-runs/x and use {TOKEN} at https://user:secret@host/path"
    task, sent = paf.build_task(brief=brief, facts="F" * (paf.FACTS_CAP + 50), excerpts="E" * (paf.EXCERPTS_CAP + 50),
                                listing="port_model.py  2341 bytes", homes=HOMES)
    assert "/home/ttuser" not in task and TOKEN not in task and "secret" not in task
    assert sent["facts"] == {"chars": paf.FACTS_CAP + len("\n[facts cut at 20000 characters]"), "cut": True}
    assert sent["excerpts"]["cut"] is True
    assert "port_model.py  2341 bytes" in task and task.startswith(paf.REQUEST)


def test_an_absent_excerpts_file_is_named_absent_and_adds_no_section():
    task, sent = paf.build_task(brief="b", facts="f", excerpts=None, listing="(empty)", homes=HOMES)
    assert sent["excerpts"]["present"] is False and "reference excerpts" not in task


def test_the_failure_text_is_scrubbed_and_keeps_the_end_of_the_output():
    out = "x" * 10_000 + f"\nFAILED: KeyError at /home/ttuser/run {TOKEN}"
    text = paf.failure_text(["the finish step ended: escalate"], ["20 turns and no file"], out, HOMES)
    assert TOKEN not in text and "/home/ttuser" not in text and "FAILED: KeyError" in text
    assert "20 turns and no file" in text and len(text) <= paf.FAILURE_CAP + 40


def test_read_input_names_a_missing_and_a_linked_file(tmp_path):
    assert paf.read_input(tmp_path / "nope", 100) == (None, "not present")
    real = tmp_path / "real.txt"
    real.write_text("hello")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    text, note = paf.read_input(link, 100)
    assert text is None and note.startswith("not read")
    assert paf.read_input(real, 100) == ("hello", "whole")
    big = tmp_path / "big.txt"
    big.write_text("z" * 500)
    text, note = paf.read_input(big, 100)
    assert note == "cut" and text.startswith("z" * 100) and "cut at 100" in text


def test_the_listing_names_files_and_links_without_opening_them(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").symlink_to("/etc/passwd")
    listing = paf.port_listing(tmp_path)
    assert "a.py  1 bytes" in listing and "b.py" in listing and "(a link)" in listing
    assert paf.port_listing(tmp_path / "missing").startswith("(the port directory")


# ---- what comes back ----------------------------------------------------------------------------------------------------------------------

def test_the_longest_python_block_is_taken():
    text = "Here:\n```python\nx = 1\n```\nand\n```python\ndef f():\n    return 2\n```\n```bash\nrm -rf /\n```"
    assert paf.extract_draft(text) == "def f():\n    return 2\n"


def test_a_reply_with_no_block_is_taken_whole_only_when_it_parses():
    assert paf.extract_draft("x = 1\ny = 2\n") == "x = 1\ny = 2\n"
    assert paf.extract_draft("You should check the index convention.") is None
    assert paf.extract_draft("```python\n```") is None


def test_an_unclosed_fence_gives_nothing_and_a_long_run_of_backticks_is_scanned_in_linear_time():
    import time
    assert paf.extract_draft("```python\nx = 1\n") is None
    started = time.monotonic()
    assert paf.extract_draft("```" * 100_000) is None
    assert paf.extract_draft("```python\n" * 20_000) is None
    assert time.monotonic() - started < 3


def test_parse_check_names_the_error_and_never_runs_the_text(tmp_path):
    marker = tmp_path / "ran"
    ok, why = paf.parses(f"open({str(marker)!r}, 'w').write('x')\n")
    assert ok and not marker.exists()          # a parse is not an execution
    assert paf.parses("def (:") == (False, "SyntaxError")
    assert paf.parses("(" * 100_000)[0] is False
    assert paf.parses("x" * (paf.DRAFT_MAX + 1)) == (False, "too long")


def test_the_draft_is_written_with_a_header_mode_0600_and_is_never_replaced(tmp_path):
    path = paf.write_draft(tmp_path, 1, "x = 1", "abc123", "big-model")
    assert path == tmp_path / "telegraph" / "draft-1.py"
    text = path.read_text()
    assert text.startswith("# A draft from an outside helper model (telegraph call abc123, model big-model).") and text.endswith("x = 1\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        paf.write_draft(tmp_path, 1, "x = 2", "abc123", "big-model")
    assert "x = 1" in path.read_text()


def test_a_link_where_the_directory_belongs_is_refused_and_nothing_is_written_through_it(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / paf.DIR).symlink_to(outside)
    with pytest.raises(OSError, match="not a plain directory"):
        paf.write_draft(stage, 1, "x = 1", "c", "m")
    assert os.listdir(outside) == []


def test_a_file_where_the_directory_belongs_is_refused(tmp_path):
    (tmp_path / paf.DIR).write_text("not a directory")
    with pytest.raises(OSError):
        paf.write_draft(tmp_path, 1, "x = 1", "c", "m")


def test_the_coder_note_gives_the_path_the_warning_and_the_parse_problem():
    note = paf.coder_note("stages/2/telegraph/draft-1.py", "abc", True, "")
    assert "stages/2/telegraph/draft-1.py" in note and "may be wrong" in note and "follow the facts file" in note
    assert "does not parse" not in note
    assert "does not parse as Python (SyntaxError)" in paf.coder_note("p", "abc", False, "SyntaxError")


# ---- the verdict --------------------------------------------------------------------------------------------------------------------------

def test_unreported_lists_this_attempts_answers_that_have_no_verdict_yet():
    entries = [entry(1, "stage_start"), decision(2, paf.ANSWERED, ok=True, call_id="old", cached=False),
               entry(3, "stage_start"), decision(4, paf.ANSWERED, ok=True, call_id="a", cached=False),
               decision(5, paf.ANSWERED, ok=True, call_id="b", cached=False), decision(6, paf.REPORTED, call_id="b", helped=False),
               decision(7, paf.ANSWERED, ok=False, call_id="c", reason="cap"), decision(8, paf.ANSWERED, ok=True, call_id="d", cached=True)]
    assert paf.unreported(entries, 2) == ["a"]
