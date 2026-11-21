# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audit of the model's own code survives a hang or a kill of the stage 2 construction helper (follow-up to PR 786).

When transformers refuses a repository with custom code, stage 1's `allow_remote_code` lists every .py file with its sha256 and the names of the
scrubbed variables, writes that list to disk, and only then lets the model's code import. In the stage 2 helper that file went to a temporary
directory that belongs to the helper process, so the supervisor never knew where it was and a clean exit removed it. A model whose code hangs
or is killed left no audit that anyone could read. The supervisor now makes an audit directory beneath the stage directory, passes it to the
helper, and records the file in the ledger after the helper ends, whatever the outcome. The helper writes it atomically, so a kill never leaves
a half-written file.

Limits that stay: code that ran with the user's permissions could change the file afterwards, so the ledger entry carries the sha256 of the file
as the supervisor found it and says when the helper wrote it.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import port_construction as pc
from orchard import supervisor
from orchard.commands import CommandResult
from test_port_construction import _reference_python, load_script
from test_port_construction_remote import install_fakes, snapshot_with_code

@pytest.fixture(autouse=True)
def restore_environment():
    """The route changes os.environ directly (it moves HOME, which the reference interpreter lookup reads), so the whole environment is saved and
    restored around every test of this file."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


HASH = "a" * 64
AUDIT = {"used": True, "files": {"modeling_x.py": HASH, "sub/configuration_x.py": "b" * 64}, "scrubbed": ["HF_TOKEN", "OPENAI_API_KEY"]}


class Ledger:
    def __init__(self):
        self.entries = []

    def append(self, kind, stage=None, **fields):
        self.entries.append({"kind": kind, "stage": stage, **fields})

    def decisions(self, text):
        return [e for e in self.entries if text in str(e.get("decision", ""))]


def stage_of(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    return run, stage


def audit_dir_of(argv):
    """The audit directory the supervisor passed: the helper's last argument."""
    assert len(argv) == 5, argv                       # python, -I, script, snapshot, audit directory
    return Path(argv[4])


def runner_that(write=None, *, result=None, raises=None):
    """A stand-in for commands.run_command: runs `write(audit_directory)` as the helper would before the model's code, then ends as asked."""
    seen = {}

    def run(argv, timeout, env=None):
        seen["argv"] = list(argv)
        if write is not None and len(argv) > 4:                  # a helper that is given no directory writes none
            write(Path(argv[4]))
        if raises is not None:
            raise raises
        return result or CommandResult(tuple(argv), None, "", "", timed_out=True)           # a timeout, the case this follow-up is about
    run.seen = seen
    return run


def write_audit(data):
    def write(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / pc.AUDIT_FILE).write_text(data if isinstance(data, str) else json.dumps(data))
    return write


# ---- the argument -------------------------------------------------------------------------------------------------------------------

def test_run_passes_the_audit_directory_as_the_last_argument_and_only_when_given(tmp_path):
    seen = []
    runner = lambda argv, timeout, env=None: (seen.append(list(argv)), CommandResult(tuple(argv), 1, "", ""))[1]
    pc.run("/x/python", tmp_path, runner=runner, audit_dir=tmp_path / "audit")
    pc.run("/x/python", tmp_path, runner=runner)
    assert seen[0][-1] == str(tmp_path / "audit") and len(seen[0]) == 5
    assert len(seen[1]) == 4 and str(tmp_path / "audit") not in seen[1]               # a caller that names none is served as before


# ---- the supervisor records the audit whatever the helper did -----------------------------------------------------------------------------

def test_the_audit_is_recorded_in_the_ledger_after_a_timeout(tmp_path):
    run, stage = stage_of(tmp_path)
    ledger = Ledger()
    runner = runner_that(write_audit(AUDIT))                         # the helper wrote the audit, then the model's code hung until the timeout
    written, summary = supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python", runner=runner)
    assert written is False and summary is None                      # no construction file: the run is advisory
    assert ledger.decisions("effective construction not written")
    (entry,) = ledger.decisions("remote-code audit recorded")
    rel = "stages/2/evidence/construction-remote-code/remote-code-audit.json"
    assert entry["audit"]["path"] == rel
    assert entry["audit"]["sha256"] == hashlib.sha256((run / rel).read_bytes()).hexdigest()
    assert entry["files_total"] == 2 and entry["files"] == AUDIT["files"] and entry["scrubbed"] == AUDIT["scrubbed"]
    assert "before the model's code" in entry["note"] and "could have changed" in entry["note"]


def test_the_audit_is_recorded_when_the_helper_raises_or_exits_with_an_error(tmp_path):
    for kind, runner in (("raises", runner_that(write_audit(AUDIT), raises=OSError("gone"))),
                         ("exit 3", runner_that(write_audit(AUDIT), result=CommandResult(("x",), 3, "", "")))):
        run, stage = stage_of(tmp_path / kind)
        ledger = Ledger()
        supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python", runner=runner)
        assert len(ledger.decisions("remote-code audit recorded")) == 1, kind


def test_a_helper_that_never_used_the_route_leaves_no_audit_decision(tmp_path):
    run, stage = stage_of(tmp_path)
    ledger = Ledger()
    supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python", runner=runner_that())
    assert not ledger.decisions("remote-code audit")
    assert (stage / "evidence" / "construction-remote-code").is_dir()                    # the directory was offered to the helper


def test_an_audit_left_by_an_earlier_attempt_is_removed_before_the_helper_runs(tmp_path):
    run, stage = stage_of(tmp_path)
    old = stage / "evidence" / "construction-remote-code"
    old.mkdir(parents=True)
    (old / pc.AUDIT_FILE).write_text(json.dumps(AUDIT))
    ledger = Ledger()
    seen_at_start = {}

    def helper(argv, timeout, env=None):
        seen_at_start["present"] = (audit_dir_of(argv) / pc.AUDIT_FILE).exists()
        return CommandResult(tuple(argv), 1, "", "")

    supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python", runner=helper)
    assert seen_at_start["present"] is False
    assert not ledger.decisions("remote-code audit")                  # nothing from this run, and the old file is not reported as this run's


def test_a_link_where_the_audit_directory_goes_is_not_followed(tmp_path):
    run, stage = stage_of(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (stage / "evidence").mkdir()
    (stage / "evidence" / "construction-remote-code").symlink_to(outside)
    ledger = Ledger()
    runner = runner_that(write_audit(AUDIT))
    supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python", runner=runner)
    assert list(outside.iterdir()) == []                                # nothing was written through the link
    assert not ledger.decisions("remote-code audit recorded")


@pytest.mark.parametrize("content", ["not json", "[]", "{}", json.dumps({"used": False, "files": {}, "scrubbed": []}),
                                     json.dumps({"used": True, "files": [], "scrubbed": []}),
                                     json.dumps({"used": True, "files": {"../../etc/passwd": HASH}, "scrubbed": []}),
                                     json.dumps({"used": True, "files": {"a.py": "zz"}, "scrubbed": []}),
                                     json.dumps({"used": True, "files": {}, "scrubbed": ["has space"]}),
                                     json.dumps({"used": True, "files": {}, "scrubbed": ["X" * 500]}),
                                     json.dumps({"used": True, "files": {f"f{i}.py": HASH for i in range(5000)}, "scrubbed": []}),
                                     json.dumps({"used": True, "files": {f"f{i}.py": HASH for i in range(10001)}, "scrubbed": []}),
                                     '{"used": true, "files": {}, "scrubbed": [], "x": ' + "9" * 400 + "}",
                                     "\ud800", "x" * (2 << 20)],
                         ids=["text", "list", "empty", "not-used", "files-list", "path-escape", "bad-hash", "name-space", "name-long",
                              "many-files-cut", "files-over-cap", "huge-int", "surrogate", "oversize"])
def test_a_hostile_audit_file_is_reported_as_unreadable_and_never_raises(tmp_path, content):
    run, stage = stage_of(tmp_path)
    ledger = Ledger()
    supervisor.write_effective_construction(ledger, run, SimpleNamespace(number=2), stage, str(tmp_path), "/x/python",
                                            runner=runner_that(write_audit(content)))
    (entry,) = ledger.decisions("remote-code audit")
    assert entry["decision"] in ("remote-code audit unreadable", "remote-code audit recorded")
    text = json.dumps(entry)
    assert "../../etc/passwd" not in text and "has space" not in text and "X" * 100 not in text      # a reason never quotes the file
    over_cap = isinstance(content, str) and content.count('"f') > pc.MAX_AUDIT_FILES
    if over_cap:
        assert entry["decision"] == "remote-code audit unreadable"
    if entry["decision"] == "remote-code audit recorded":                 # the only hostile shape that is accepted is a long file list, cut
        assert len(entry["files"]) <= pc.MAX_AUDIT_FILES_SHOWN and entry["files_total"] == 5000


# ---- the helper's writer and arguments ---------------------------------------------------------------------------------------------

def test_the_atomic_writer_puts_a_whole_file_in_place_and_keeps_the_old_one_when_a_write_fails(tmp_path, monkeypatch):
    script = load_script()
    path = tmp_path / "d" / "a.json"
    before = os.umask(0o077)                                          # a strict umask would otherwise leave the file at 0o600
    try:
        script.atomic_write_json(path, {"v": 1})
    finally:
        os.umask(before)
    assert json.loads(path.read_text()) == {"v": 1} and oct(path.stat().st_mode & 0o777) == "0o644"
    real_write = os.write

    def failing(fd, data):
        real_write(fd, data[:5])                                      # half of the text reaches the new file
        raise OSError("disk full")

    monkeypatch.setattr(os, "write", failing)
    with pytest.raises(OSError):
        script.atomic_write_json(path, {"v": 2})
    monkeypatch.undo()
    assert json.loads(path.read_text()) == {"v": 1}                   # the old file, whole
    assert [p.name for p in path.parent.iterdir()] == ["a.json"]      # and no half-written file beside it


def test_the_atomic_writer_does_not_follow_a_link_standing_where_its_temporary_file_goes(tmp_path):
    script = load_script()
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    path = tmp_path / "a.json"
    (tmp_path / f".a.json.{os.getpid()}.tmp").symlink_to(target)
    with pytest.raises(FileExistsError):                              # O_EXCL | O_NOFOLLOW: it refuses and writes nothing through the link
        script.atomic_write_json(path, {"v": 1})
    assert target.read_text() == "keep"


def test_the_route_writes_its_audit_through_the_atomic_writer_into_the_directory_it_was_given(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    install_fakes(monkeypatch, refuse=True, calls=[])
    script = load_script()
    audit = tmp_path / "audit"
    script._ROUTE["audit_dir"] = str(audit)
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append(Path(b)), real_replace(a, b))[1])
    script.build_model(str(snap))
    assert replaced == [audit / pc.AUDIT_FILE]
    assert json.loads((audit / pc.AUDIT_FILE).read_text())["used"] is True


@pytest.mark.parametrize("extra", [["relative/dir"], ["", ], ["a", "b"]], ids=["relative", "empty", "two-extra"])
def test_the_script_refuses_an_audit_directory_that_is_not_an_absolute_path(tmp_path, extra, capsys):
    script = load_script()
    assert script.main(["effective_construction.py", str(tmp_path), *extra]) == 2
    assert "usage" in capsys.readouterr().err


# ---- the helper script, run for real, killed while the model's code hangs --------------------------------------------------------------

HANG_CONFIGURATION = '''
import time
time.sleep(3600)                  # the model's own code never finishes importing
'''


def limited():
    """Limits for the child, set before it starts: a runaway child cannot use more than 120 s of CPU, write a file over 64 MiB or hold more than
    16 GiB of address space (torch needs several; the cap stops only a runaway)."""
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def hanging_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({
        "model_type": "hang_remote", "architectures": ["HangRemoteForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_hang_remote.HangRemoteConfig", "AutoModelForCausalLM": "modeling_hang_remote.HangRemoteForCausalLM"}}))
    (snap / "configuration_hang_remote.py").write_text(HANG_CONFIGURATION)
    (snap / "modeling_hang_remote.py").write_text("class HangRemoteForCausalLM: pass\n")
    return snap


@pytest.mark.skipif(_reference_python() is None, reason="needs the reference interpreter ($ORCHARD_REFERENCE_PYTHON or the default venv)")
@pytest.mark.timeout(300)
def test_the_audit_is_on_disk_when_the_helper_is_killed_inside_the_models_code(tmp_path):
    snap = hanging_snapshot(tmp_path)
    audit = tmp_path / "audit"
    audit.mkdir()
    env = dict(pc.default_env(), HF_TOKEN="a-secret-value", TOKENIZERS_PARALLELISM="false")
    proc = subprocess.Popen([_reference_python(), "-I", str(pc.SCRIPT), str(snap), str(audit)], env=env, cwd=tmp_path, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=limited)          # hardening: ok a bounded child of the test
    path = audit / pc.AUDIT_FILE
    try:
        end = time.monotonic() + 200
        while not path.exists() and proc.poll() is None and time.monotonic() < end:
            time.sleep(0.5)
        assert path.exists(), "the helper wrote no audit before the model's code ran"
        time.sleep(1.0)                                              # inside the model's code by now
        assert proc.poll() is None, "the helper ended on its own: the model's code did not hang"
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)                      # this child's own group, by its recorded pid
        except ProcessLookupError:
            pass
        proc.communicate(timeout=30)
    data = json.loads(path.read_text())                              # a whole file, never half of one
    assert data["used"] is True and "HF_TOKEN" in data["scrubbed"] and "TOKENIZERS_PARALLELISM" not in data["scrubbed"]
    assert data["files"]["configuration_hang_remote.py"] == hashlib.sha256(HANG_CONFIGURATION.encode()).hexdigest()
    assert "a-secret-value" not in path.read_text()
    assert [p.name for p in audit.iterdir()] == [pc.AUDIT_FILE]      # the atomic write left no temporary file behind
    assert oct(path.stat().st_mode & 0o777) == "0o644"
