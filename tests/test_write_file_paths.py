"""write_file path forms (orchard/agent.py, Tools.target).

write_file is documented as relative to the stage directory, but a model that reads files with run-relative
paths (`stages/1/evidence/a.txt`, which read_file takes) writes the same form. Before this fix such a write
landed in `stages/1/stages/1/...` and the agent spent turns moving files, then repeated the same write until
the repeated-call detector escalated the stage (Cloudflare/clef, stage 1, 2026-10-06). A run-relative path
into the agent's own stage directory now means that file."""
import os
from pathlib import Path

import pytest

from orchard.agent import Tools


@pytest.fixture
def tools(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "1"
    stage.mkdir(parents=True)
    return Tools(run, stage, {"PATH": os.environ["PATH"]}), run, stage


def write(t, path, content="x"):
    import json
    return t.call("write_file", json.dumps({"path": path, "content": content}))


def test_a_stage_relative_path_still_writes_into_the_stage_directory(tools):
    t, run, stage = tools
    assert write(t, "result.json", "a").startswith("wrote")
    assert (stage / "result.json").read_text() == "a"


def test_a_nested_stage_relative_path_still_works(tools):
    t, run, stage = tools
    write(t, "evidence/a.txt", "a")
    assert (stage / "evidence" / "a.txt").read_text() == "a"


def test_a_run_relative_path_into_this_stage_writes_that_file_not_a_nested_copy(tools):
    t, run, stage = tools
    out = write(t, "stages/1/reference.json", "ok")
    assert out.startswith("wrote") and "stages/1/reference.json" in out
    assert (stage / "reference.json").read_text() == "ok"
    assert not (stage / "stages").exists()


def test_a_run_relative_evidence_path_works_too(tools):
    t, run, stage = tools
    write(t, "stages/1/evidence/notes.txt", "n")
    assert (stage / "evidence" / "notes.txt").read_text() == "n"
    assert not (stage / "stages").exists()


def test_an_absolute_path_inside_the_stage_directory_works(tools):
    t, run, stage = tools
    write(t, str(stage / "a.txt"), "abs")
    assert (stage / "a.txt").read_text() == "abs"


@pytest.mark.parametrize("path", ["stages/2/x.txt", "stages/0/delta.json", "../x.txt",
                                  "stages/1/../2/x.txt", "stages/1/../../x.txt", "/etc/x.txt"])
def test_a_path_into_another_stage_or_outside_is_still_refused(tools, path):
    t, run, stage = tools
    out = write(t, path)
    assert out.startswith("refused:"), out
    assert not (run / "stages" / "2").exists()


def test_the_stage_directory_itself_is_not_a_file(tools):
    t, run, stage = tools
    assert write(t, "stages/1").startswith("refused:")
    assert write(t, ".").startswith("refused:")


def test_the_ledger_stays_protected_under_either_form(tools):
    t, run, stage = tools
    assert "ledger" in write(t, "ledger.jsonl")
    assert "ledger" in write(t, "stages/1/ledger.jsonl")


def test_a_link_in_the_stage_directory_that_leaves_it_is_refused_under_either_form(tools, tmp_path):
    t, run, stage = tools
    (tmp_path / "outside").mkdir()
    (stage / "link").symlink_to(tmp_path / "outside")
    assert write(t, "link/x.txt").startswith("refused:")
    assert write(t, "stages/1/link/x.txt").startswith("refused:")
    assert not (tmp_path / "outside" / "x.txt").exists()


def test_the_reply_names_the_file_relative_to_the_run_directory(tools):
    t, run, stage = tools
    assert "stages/1/evidence/a.txt" in write(t, "stages/1/evidence/a.txt")
    assert "stages/1/evidence/b.txt" in write(t, "evidence/b.txt")


def test_a_path_that_does_not_start_with_stages_is_always_stage_relative(tools):
    t, run, stage = tools
    write(t, "other/x.txt", "o")
    assert (stage / "other" / "x.txt").read_text() == "o"
