# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 572, 576, 577 and 581: what a run leaves behind, and what its coder is told.

572: a run that blocks and then recovers (an automatic retry or a resume) ended 'ready for operator review' with the old BLOCKED.md and
     blocked.json still in the run directory, saying the harness stopped. The pair is now moved to numbered names when the run ends ready.
576: the stage 2 task names the idioms file only when the copy of the shipped file succeeded for this stage; a stale copy is removed.
577: the idioms snippet kept integer checkpoint tensors by converting them with .float(), which loses large integers.
581: the agent step logs of every attempt (including the stage directories moved aside) can be counted per tool from the run directory.
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import blocked, full_port, port_facts, supervisor, tool_use
from orchard.supervisor import EXIT_READY

# The real end-to-end rig of the full-port route (its fixture and its scripted coder).
from test_full_port_chain import go, rig  # noqa: F401  (rig is a fixture)

pytestmark = pytest.mark.usefixtures("stub_tools")


# ---- 572 --------------------------------------------------------------------------------------------------------------------------------

def plant_block(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "BLOCKED.md").write_text("# Blocked: agent-stuck\n\nThe harness stopped at stage 3.\n")
    (run_dir / "blocked.json").write_text(json.dumps({"code": "agent-stuck", "stage": 3}) + "\n")


def test_a_run_that_ends_ready_does_not_leave_the_old_block_pair_in_place(rig):  # noqa: F811
    go(rig)
    plant_block(rig.run_dir)                       # what the first, blocked attempt left behind
    assert rig.run() == EXIT_READY
    assert not (rig.run_dir / "BLOCKED.md").exists() and not (rig.run_dir / "blocked.json").exists()
    assert json.loads((rig.run_dir / "blocked.1.json").read_text())["code"] == "agent-stuck"       # kept, under a numbered name
    assert "agent-stuck" in (rig.run_dir / "BLOCKED.1.md").read_text()


def test_the_archive_numbers_each_block_and_never_overwrites_an_earlier_one(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / "blocked.json").write_text(json.dumps({"code": f"c{i}"}))
        (tmp_path / "BLOCKED.md").write_text(f"md {i}")
        blocked.archive_bundle(tmp_path)
    assert [json.loads((tmp_path / f"blocked.{i}.json").read_text())["code"] for i in (1, 2, 3)] == ["c1", "c2", "c3"]
    assert (tmp_path / "BLOCKED.2.md").read_text() == "md 2"


def test_archiving_with_no_block_files_does_nothing_and_a_link_is_moved_not_followed(tmp_path):
    blocked.archive_bundle(tmp_path)               # nothing there: no error
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    (tmp_path / "blocked.json").symlink_to(secret)
    blocked.archive_bundle(tmp_path)
    assert (tmp_path / "blocked.1.json").is_symlink() and secret.read_text() == "outside"


# ---- 576 --------------------------------------------------------------------------------------------------------------------------------

def pin_stage_2(tmp_path, orchard_dir):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    sd.mkdir(parents=True)
    entries = []
    stub = SimpleNamespace(inputs={"model": str(run)}, run_dir=run, paths=SimpleNamespace(orchard_dir=str(orchard_dir)),
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    stub._run_json = lambda *a: supervisor.Supervisor._run_json(stub, *a)
    return stub, sd, entries


def test_a_failed_idioms_copy_removes_the_stale_copy_and_the_task_does_not_name_it(tmp_path, monkeypatch):
    monkeypatch.setattr(full_port, "IDIOMS_SOURCE", Path("no-such-idioms-file.md"))       # the shipped idioms file is not there
    stub, sd, entries = pin_stage_2(tmp_path, Path(supervisor.__file__).resolve().parent.parent)
    (sd / full_port.IDIOMS_NAME).write_text("an earlier copy the coder edited")
    supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd)
    assert not (sd / full_port.IDIOMS_NAME).exists()
    assert stub._idioms_ok is False
    assert any(k.get("decision") == "idioms file not copied" for a, k in entries)
    assert not supervisor.idioms_advertised(stub._idioms_ok, sd)


def test_a_good_idioms_copy_is_advertised(tmp_path):
    orchard_dir = Path(supervisor.__file__).resolve().parent.parent
    stub, sd, _ = pin_stage_2(tmp_path, orchard_dir)
    supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd)
    assert stub._idioms_ok is True and (sd / full_port.IDIOMS_NAME).is_file()
    assert supervisor.idioms_advertised(stub._idioms_ok, sd)
    assert full_port.IDIOMS_NAME in port_facts.prepare_note(facts=True, skeleton=True, excerpts=True, idioms=True)


# ---- 577 --------------------------------------------------------------------------------------------------------------------------------

def test_the_idioms_snippet_keeps_integer_tensors_as_they_are():
    text = (full_port.TEMPLATE.parent / "example_decoder_idioms.md").read_text()
    line = next(l for l in text.splitlines() if l.strip().startswith("host[name] ="))
    expr = line.split("=", 1)[1].split("#")[0].strip()
    torch = pytest.importorskip("torch")
    ints = torch.tensor([16777217, 3], dtype=torch.int64)
    kept = eval(expr, {"t": ints, "torch": torch})              # the right side of the else branch, run on an integer table
    assert kept.dtype == torch.int64 and int(kept[0]) == 16777217
    floats = eval(expr, {"t": torch.ones(2, dtype=torch.bfloat16), "torch": torch})
    assert floats.dtype == torch.float32                          # floating tensors still become float32


# ---- 581 --------------------------------------------------------------------------------------------------------------------------------

def log_line(turn, *calls):
    return json.dumps({"turn": turn, "sent": [], "received": {"role": "assistant", "content": "text with write_file(x) in it",
                                                              "tool_calls": [{"id": f"c{turn}{i}", "type": "function",
                                                                              "function": {"name": n, "arguments": json.dumps(a)}}
                                                                             for i, (n, a) in enumerate(calls)]}}) + "\n"


def make_logs(run: Path):
    for stage_dir, name, lines in (
            ("2", "prepare-00050.jsonl", [log_line(1, ("read_file", {"path": "a"}), ("shell", {"command": "python3 stages/2/stage_tools.py last-check"}))]),
            ("2.partial-1", "prepare-00010.jsonl", [log_line(1, ("write_file", {"path": "b", "content": "x"})),
                                                     log_line(2, ("shell", {"command": "cd stages/2 && ls -la"}))]),
            ("3", "repair-00070.jsonl", [log_line(1, ("shell", {"command": "python3 stages/3/stage_tools.py facts --find x"}))])):
        d = run / "stages" / stage_dir / "log"
        d.mkdir(parents=True)
        (d / name).write_text("".join(lines))


def test_tool_use_counts_every_attempt_including_stage_directories_moved_aside(tmp_path):
    make_logs(tmp_path)
    got = tool_use.count(tmp_path)
    assert got["2"]["write_file"] == 1 and got["2"]["read_file"] == 1                # the moved-aside attempt (2.partial-1) counts toward stage 2
    assert got["2"]["stage_tools last-check"] == 1 and got["2"]["shell"] == 1        # a stage_tools call is named by its subcommand, other shell by count
    assert got["3"]["stage_tools facts"] == 1


def test_tool_use_counts_calls_made_and_not_words_in_the_text(tmp_path):
    make_logs(tmp_path)
    assert tool_use.count(tmp_path)["2"]["write_file"] == 1            # three replies say 'write_file(x)' in their text; one made the call


def test_tool_use_skips_a_link_a_pipe_and_a_garbled_line_and_says_so(tmp_path):
    make_logs(tmp_path)
    d = tmp_path / "stages" / "2" / "log"
    (tmp_path / "outside.jsonl").write_text(log_line(1, ("shell", {"command": "ls"})))
    (d / "linked.jsonl").symlink_to(tmp_path / "outside.jsonl")
    import os
    os.mkfifo(d / "pipe.jsonl")
    (d / "garbled.jsonl").write_text("not json\n" + log_line(2, ("read_file", {"path": "z"})))
    got = tool_use.count(tmp_path)
    assert got["2"]["read_file"] == 2                                   # the good line in the garbled file still counts
    assert got["skipped"] >= 3                                          # link, pipe and the bad line are reported, never followed or opened


def test_the_tool_use_report_prints_one_block_per_stage(tmp_path, capsys):
    make_logs(tmp_path)
    assert tool_use.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "stage 2" in out and "stage 3" in out and "stage_tools last-check" in out


def test_the_task_does_not_name_the_idioms_file_when_the_stale_copy_cannot_be_removed(tmp_path, monkeypatch):
    """If the copy fails and the old file cannot be removed either (a directory that is not writable), the flag still keeps the task from
    advertising it."""
    monkeypatch.setattr(full_port, "IDIOMS_SOURCE", Path("no-such-idioms-file.md"))
    stub, sd, _ = pin_stage_2(tmp_path, Path(supervisor.__file__).resolve().parent.parent)
    (sd / full_port.IDIOMS_NAME).write_text("an earlier copy")
    real_unlink = Path.unlink

    def refuse(self, *a, **k):
        if self.name == full_port.IDIOMS_NAME:
            raise PermissionError("read-only")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", refuse)
    supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd)
    assert (sd / full_port.IDIOMS_NAME).is_file()                      # it is still there
    assert not supervisor.idioms_advertised(stub._idioms_ok, sd)      # and the task does not name it


def test_a_tool_name_that_is_not_plain_text_is_reported_as_other(tmp_path):
    d = tmp_path / "stages" / "2" / "log"
    d.mkdir(parents=True)
    weird = json.dumps({"turn": 1, "sent": [], "received": {"tool_calls": [
        {"function": {"name": "bad\ud800name", "arguments": "{}"}}, {"function": {"name": "x" * 200, "arguments": "{}"}},
        {"function": {"name": "read_file", "arguments": "{}"}}]}}) + "\n"
    (d / "prepare-00001.jsonl").write_text(weird, encoding="utf-8", errors="surrogatepass")
    got = tool_use.count(tmp_path)
    assert got["2"] == {"other": 2, "read_file": 1}
    assert not any(0xD800 <= ord(c) <= 0xDFFF for name in got["2"] for c in name)
