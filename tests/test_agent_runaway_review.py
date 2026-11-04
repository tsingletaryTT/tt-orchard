# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 462 (issue 458): the shell guard for the port directory must act only inside the real port directory, keep the latest
accepted version of a file for the rollback, and bound every scan it makes.

The first three tests put a link, or a swap of a directory for a link, where the guard deletes or restores a file; the outside file
must come out untouched. The swap is made by a hook on `os.unlink`, so the test is deterministic and uses no thread or sleep."""
import os
import types

import pytest

from orchard.agent import AgentStep, Tools, agent_env
from orchard.ledger import Ledger


class Control:
    def __init__(self):
        self.nudges, self.stop = [], None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop = "escalate"

    def pause(self, reason, evidence):
        self.stop = "pause"

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        return self.stop


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    (tmp_path / "run" / "stages" / "0" / "port").mkdir(parents=True)
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


def tools_for(run_dir):
    sd = run_dir / "stages" / "0"
    return Tools(run_dir, sd, agent_env(run_dir)), sd


# ---- a link where the guard acts ----

MAKE_LINK = ("import os, sys\nos.rmdir('stages/0/port')\nos.symlink(sys.argv[1], 'stages/0/port')\n")


def test_a_command_that_makes_port_a_link_to_another_directory_changes_nothing_there(run, tmp_path):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.txt").write_text("hello world\n" * 50_000)                     # 600,000 bytes: over the cap
    (outside / "keep.py").write_text("x = 1\n")
    tools.write_file("mklink.py", MAKE_LINK)
    out = tools.shell(f"python3 stages/0/mklink.py {outside}")
    assert (outside / "big.txt").stat().st_size == 600_000, "a file outside the port was removed"
    assert (outside / "keep.py").read_text() == "x = 1\n"
    assert "link" in out and "port" in out                                         # the agent is told why nothing was checked


def test_a_link_inside_the_port_directory_is_never_followed(run, tmp_path):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.py").write_text("x = 1\n" * 100_000)
    tools.write_file("link.py", f"import os\nos.symlink('{outside}', 'stages/0/port/sub')\n")
    out = tools.shell("python3 stages/0/link.py")
    assert (outside / "big.py").stat().st_size == 600_000
    assert "refused" not in out


def test_a_directory_swapped_for_a_link_before_the_removal_does_not_reach_the_outside_file(run, tmp_path, monkeypatch):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "big.py"
    victim.write_text("x = 1\n" * 100_000)
    tools.write_file("grow.py", "import os\nos.makedirs('stages/0/port/sub', exist_ok=True)\n"
                                "open('stages/0/port/sub/big.py', 'w').write('x = 1\\n' * 100000)\n")
    real_unlink, swapped = os.unlink, []

    def unlink(path, *args, **kw):
        if "big.py" in str(path) and not swapped:                                  # between the scan and the removal
            swapped.append(1)
            os.rename(sd / "port" / "sub", sd / "port" / "sub-moved")
            os.symlink(outside, sd / "port" / "sub")
        return real_unlink(path, *args, **kw)

    monkeypatch.setattr(os, "unlink", unlink)
    out = tools.shell("python3 stages/0/grow.py")
    assert swapped, "the hook never ran: the guard did not remove the new file"
    assert victim.exists() and victim.stat().st_size == 600_000, "the removal followed the swapped directory out of the port"
    assert "refused" in out


# ---- the backup follows accepted edits ----

def writer(name: str, text: str) -> str:
    return f"open('stages/0/port/port_model.py', 'w').write({text!r})\n"


def test_a_rejected_edit_restores_the_latest_accepted_version_and_not_an_older_one(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    port = sd / "port" / "port_model.py"
    tools.write_file("port/port_model.py", "def f():\n    return 0\n")
    tools.write_file("e1.py", writer("e1", "def f():\n    return 1\n"))
    tools.write_file("e2.py", writer("e2", "def f():\n    return 2\n"))
    tools.write_file("e3.py", "open('stages/0/port/port_model.py', 'w').write('x = 1\\n' * 200000)\n")   # makes the file, not holds it
    assert "refused" not in tools.shell("python3 stages/0/e1.py")                    # accepted
    assert "refused" not in tools.shell("python3 stages/0/e2.py")                    # accepted
    out = tools.shell("python3 stages/0/e3.py")                                      # rejected
    assert "refused and restored" in out
    assert port.read_text() == "def f():\n    return 2\n", "the rollback went to a version older than the last accepted edit"


def test_a_file_created_by_an_accepted_command_is_restored_to_that_version_when_a_later_one_is_rejected(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    extra = sd / "port" / "extra.py"
    tools.write_file("n1.py", "open('stages/0/port/extra.py', 'w').write('a = 1\\n')\n")
    tools.write_file("n2.py", "open('stages/0/port/extra.py', 'w').write('x = 1\\n' * 200000)\n")
    assert "refused" not in tools.shell("python3 stages/0/n1.py")
    out = tools.shell("python3 stages/0/n2.py")
    assert "refused and restored" in out and extra.read_text() == "a = 1\n"            # not "removed": it held accepted work


# ---- the listing for a restart is bounded ----

def test_the_restart_listing_stops_at_the_entry_bound_and_does_not_sort_the_directory(run, monkeypatch):
    from orchard.agent import PORT_GUARD_ENTRIES
    run_dir, led = run
    tools, sd = tools_for(run_dir)
    consumed = []

    class Entry:
        def __init__(self, i):
            self.name, self.path = f"f{i}.py", f"/nonexistent/f{i}.py"
            consumed.append(i)

        def is_symlink(self):
            return False

        def is_dir(self, follow_symlinks=True):
            return False

        def is_file(self, follow_symlinks=True):
            return True

        def stat(self, follow_symlinks=True):
            return types.SimpleNamespace(st_size=1, st_dev=1, st_ino=1, st_mtime_ns=1)

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return (Entry(i) for i in range(10_000))

        def close(self):
            pass

    s = AgentStep(agent="stage-agent", endpoint="http://127.0.0.1:1", model="fake", tools=tools, ledger=led, stage=0, phase="run",
                  feed=lambda e: None, control=Control(), run_dir=run_dir, evidence_dir=sd / "evidence",
                  log_path=sd / "log" / "run-1.jsonl", clock=lambda: 1000.0)
    s.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    monkeypatch.setattr(os, "scandir", lambda d: Listing())
    s._restart_fresh(1)
    assert len(consumed) <= 2 * (PORT_GUARD_ENTRIES + 1), f"the listing read {len(consumed)} entries"
    assert s.messages[1]["content"].count("\n- ") <= 40


# ---- the other guards of the anchored scan ----

def test_a_missing_port_directory_gives_no_note_and_no_error(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    os.rmdir(sd / "port")
    out = tools.shell("echo hi")
    assert out.startswith("exit 0") and "refused" not in out and "link" not in out


def test_the_scan_goes_four_levels_down_and_no_further(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    tools.write_file("deep.py", "import os\nfor d in ('a/b/c/d', 'a/b/c/d/e'):\n    os.makedirs('stages/0/port/' + d, exist_ok=True)\n"
                                "    open('stages/0/port/' + d + '/big.py', 'w').write('x = 1\\n' * 100000)\n")
    out = tools.shell("python3 stages/0/deep.py")
    assert not (sd / "port/a/b/c/d/big.py").exists()                       # level 4: seen and removed
    assert (sd / "port/a/b/c/d/e/big.py").exists()                         # level 5: beyond the bound, left alone
    assert "refused and removed" in out


def test_a_file_replaced_after_the_listing_is_not_trusted_when_the_bytes_are_kept(run, monkeypatch):
    """Between the scan and the read another file takes the name. The bytes read would not be the file that was listed, so neither the
    stamp nor the backup is recorded for it."""
    from orchard import port_guard
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    target = sd / "port" / "a.py"
    target.write_text("a = 1\n")
    other = sd / "port-other.py"
    other.write_text("b = 2\n")
    real_open, swapped = port_guard.safeio.open_beneath, []

    def open_beneath(fd, parts, **kw):
        if kw.get("file_flags") is not None and parts[-1:] == ("a.py",) and not swapped:
            swapped.append(1)
            os.replace(other, target)                                       # a different inode under the same name
        return real_open(fd, parts, **kw)

    monkeypatch.setattr(port_guard.safeio, "open_beneath", open_beneath)
    guard = tools._guard
    guard.remember()
    assert swapped
    assert ("a.py",) not in guard._stamp and ("a.py",) not in guard._backup


def test_a_file_replaced_before_the_removal_is_left_alone_and_the_agent_is_told(run, monkeypatch):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    tools.write_file("grow.py", "open('stages/0/port/big.py', 'w').write('x = 1\\n' * 100000)\n")
    clean = sd / "port-other.py"
    clean.write_text("y = 3\n")
    real_stat, swapped = os.stat, []

    def stat(path, *args, **kw):
        if str(path) == "big.py" and kw.get("dir_fd") is not None and not swapped:   # the identity check of the removal
            swapped.append(1)
            os.replace(clean, sd / "port" / "big.py")
        return real_stat(path, *args, **kw)

    monkeypatch.setattr(os, "stat", stat)
    out = tools.shell("python3 stages/0/grow.py")
    assert swapped
    assert (sd / "port" / "big.py").read_text() == "y = 3\n", "a file that was never judged was removed"
    assert "refused" in out and "could not be put back" in out


def test_a_leftover_temporary_name_does_not_stop_a_restore(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    tools.write_file("port/port_model.py", "def f():\n    return 1\n")
    tools.write_file("both.py", "open('stages/0/port/.port-guard-tmp', 'w').write('left over\\n')\n"
                                "open('stages/0/port/port_model.py', 'w').write('x = 1\\n' * 200000)\n")
    out = tools.shell("python3 stages/0/both.py")
    assert "refused and restored" in out
    assert (sd / "port" / "port_model.py").read_text() == "def f():\n    return 1\n"


def test_a_file_swapped_for_a_link_just_before_the_restore_is_replaced_and_the_target_is_not_written(run, tmp_path, monkeypatch):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    tools.write_file("port/port_model.py", "def f():\n    return 1\n")
    tools.write_file("grow.py", "open('stages/0/port/port_model.py', 'w').write('x = 1\\n' * 200000)\n")
    real_rename, swapped = os.rename, []

    def rename(src, dst, *args, **kw):
        if src == ".port-guard-tmp" and not swapped:
            swapped.append(1)
            os.unlink(sd / "port" / "port_model.py")
            os.symlink(outside, sd / "port" / "port_model.py")
        return real_rename(src, dst, *args, **kw)

    monkeypatch.setattr(os, "rename", rename)
    tools.shell("python3 stages/0/grow.py")
    assert swapped
    assert outside.read_text() == "outside\n", "the restore wrote through a link"
    assert not (sd / "port" / "port_model.py").is_symlink()
    assert (sd / "port" / "port_model.py").read_text() == "def f():\n    return 1\n"


def test_a_pycache_directory_is_not_entered(run):
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    tools.write_file("cache.py", "import os\nos.makedirs('stages/0/port/__pycache__', exist_ok=True)\n"
                                 "open('stages/0/port/__pycache__/m.pyc', 'wb').write(b'x' * 600000)\n")
    out = tools.shell("python3 stages/0/cache.py")
    assert (sd / "port" / "__pycache__" / "m.pyc").stat().st_size == 600_000 and "refused" not in out


def test_an_oversize_port_file_that_was_there_before_the_command_is_left_alone(run):
    """A resumed stage can hold a large file from an earlier step. The command did not make it, so the guard does not touch it."""
    run_dir, _ = run
    tools, sd = tools_for(run_dir)
    (sd / "port" / "old_big.py").write_text("x = 1\n" * 200_000)
    out = tools.shell("echo hi")
    assert "refused" not in out and (sd / "port" / "old_big.py").stat().st_size == 1_200_000


def test_the_restart_listing_stops_at_the_entry_bound_even_when_no_entry_is_a_file(run, monkeypatch):
    from orchard.agent import PORT_GUARD_ENTRIES
    run_dir, led = run
    tools, sd = tools_for(run_dir)
    consumed = []

    class Entry:
        def __init__(self, i):
            self.name = f"x{i}"
            consumed.append(i)

        def is_symlink(self):
            return False

        def is_dir(self, follow_symlinks=True):
            return False

        def is_file(self, follow_symlinks=True):
            return False

    class Listing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return (Entry(i) for i in range(10_000))

        def close(self):
            pass

    monkeypatch.setattr(os, "scandir", lambda d: Listing())
    assert tools.restart_files(40) == []
    assert len(consumed) <= 2 * (PORT_GUARD_ENTRIES + 1), f"the listing read {len(consumed)} entries"
