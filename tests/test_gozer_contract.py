# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""GozerAdapter against the real gozer command line, with fake roots.

gozer runs from ORCHARD_GOZER, or ~/code/tt-gozer/bin/gozer, with a fake sysfs, a fake /proc, a
fake history directory and a fake reset command. No device is opened and the real gozer state
under /tmp/tt-gozer is never read. When gozer is not present the tests skip. A skip here is not
evidence that the adapter matches gozer.

Which gozer ran: the default is the working tree of ~/code/tt-gozer, which can hold uncommitted
edits (on 2026-10-02 it did: gozer/cli.py and gozer/__init__.py, with a 0.3.3 version bump). A
green run then checks those edits and not a committed release, which may differ from the 0.3.2
that the adapter's docstring cites. To pin a release, make a clean checkout of it (for example
`git worktree add /tmp/gozer-pin <commit>`) and run with ORCHARD_GOZER=/tmp/gozer-pin/bin/gozer.
`gozer_identity()` names the commit and says whether the tree is dirty; tests/conftest.py adds it
to the report of every failing test in this module.
"""

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard.adapters import Queued, Refused, TicketGone
from orchard.adapters.gozer import GozerAdapter

GOZER = Path(os.environ.get("ORCHARD_GOZER", str(Path.home() / "code/tt-gozer/bin/gozer")))


def gozer_identity() -> str:
    """The commit of the gozer under test, and whether its working tree has uncommitted edits."""
    root = GOZER.resolve().parent.parent
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return f"{GOZER}: not a git checkout, so the version is unknown"
    state = "uncommitted edits in the working tree" if dirty else "clean"
    return f"{GOZER} at commit {commit} ({state})"


pytestmark = pytest.mark.skipif(
    not GOZER.exists(),
    reason=f"gozer not found at {GOZER}; set ORCHARD_GOZER. A skip is not evidence.")

# Two boards of two chips, as on this Quietbox 2 (copied from tests/test_hardware_check.py).
QUIETBOX = [
    {"dev_index": 0, "bdf": "0000:01:00.0", "serial": "0000000000000002", "asic_id": "1111111111111111", "card": "p300c"},
    {"dev_index": 1, "bdf": "0000:02:00.0", "serial": "0000000000000002", "asic_id": "2222222222222222", "card": "p300c"},
    {"dev_index": 2, "bdf": "0000:03:00.0", "serial": "0000000000000001", "asic_id": "3333333333333333", "card": "p300c"},
    {"dev_index": 3, "bdf": "0000:04:00.0", "serial": "0000000000000001", "asic_id": "4444444444444444", "card": "p300c"},
]
BOARD1 = ["0000:03:00.0", "0000:04:00.0"]


def build_sysfs(root, chips):
    """A fake /sys/class/tenstorrent tree, as the gozer test conftest builds it."""
    for c in chips:
        base = os.path.join(root, "class", "tenstorrent", f"tenstorrent!{c['dev_index']}")
        os.makedirs(base, exist_ok=True)
        for name, val in (("tt_serial", c["serial"]), ("tt_asic_id", c["asic_id"]),
                          ("tt_card_type", c["card"]), ("tt_heartbeat", "12345")):
            with open(os.path.join(base, name), "w") as f:
                f.write(val)
        pci = os.path.join(root, "bus", "pci", "devices", c["bdf"])
        os.makedirs(pci, exist_ok=True)
        link = os.path.join(base, "device")
        if not os.path.lexists(link):
            os.symlink(pci, link)
    return os.path.join(root, "class", "tenstorrent")


def make_proc_dir(proc_root, pid, ppid=1, devs=()):
    """What gozer reads for a live process: status (PPid), comm, and fd links to devices."""
    d = Path(proc_root) / str(pid)
    (d / "fd").mkdir(parents=True, exist_ok=True)
    (d / "comm").write_text("python\n")
    (d / "status").write_text(f"Name:\tpython\nPPid:\t{ppid}\n")
    for n, dev in enumerate(devs):
        os.symlink(f"/dev/tenstorrent/{dev}", d / "fd" / str(n + 3))
    return d


def script(path, body):
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def fake_gozer(tmp_path, monkeypatch):
    """(path of a gozer wrapper, fake proc root, file the fake reset command appends to)."""
    proc = tmp_path / "proc"
    proc.mkdir()
    marker = tmp_path / "resets"
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", build_sysfs(str(tmp_path / "sys"), QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", str(proc))
    monkeypatch.setenv("GOZER_HISTORY_ROOT", str(tmp_path / "history"))
    monkeypatch.setenv("GOZER_RESET_CMD", str(script(tmp_path / "reset.sh",
                                                     f'echo "$@" >> {marker}\nexit 0\n')))
    make_proc_dir(proc, os.getpid())     # the owner pid must look alive to gozer
    wrapper = script(tmp_path / "gozer", f'exec {sys.executable} {GOZER} "$@"\n')
    return wrapper, proc, marker


def resets(marker):
    return marker.read_text().splitlines() if marker.exists() else []


def test_acquire_reset_release_round_trip(fake_gozer):
    gz, _, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "contract test", exact="0000:03:00.0")
    # gozer leases whole boards: asking for 1 chip grants the board's 2 (spec section 3).
    assert sorted(lease.chips) == BOARD1 and sorted(lease.dev_indices) == [2, 3]
    assert lease.env["TT_VISIBLE_DEVICES"] == ",".join(lease.chips)
    assert lease.units == ("0000000000000001",)
    ours = [c for c in a.status() if c.bdf in lease.chips]
    assert [c.state for c in ours] == ["CLAIMED", "CLAIMED"]
    assert all(c.lease_pid == os.getpid() and c.board == "0000000000000001" for c in ours)
    a.reset(lease)
    assert len(resets(marker)) == 1 and sorted(resets(marker)[0].split()[1].split(",")) == BOARD1
    a.release(lease)
    assert len(resets(marker)) == 2            # release resets the chips once more
    assert all(c.state == "FREE" for c in a.status())


def test_a_busy_box_refuses_or_queues_and_the_ticket_can_be_cancelled(fake_gozer):
    gz, _, _ = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    both = a.acquire(4, "orchard:contract", "both boards")
    with pytest.raises(Refused):
        a.acquire(2, "orchard:contract", "no queue")
    with pytest.raises(Queued) as q:
        a.acquire(2, "orchard:contract", "queue", queue=True)
    a.cancel(q.value.ticket)
    with pytest.raises(TicketGone):
        a.claim(q.value.ticket, 2, "orchard:contract", "claim after cancel")
    a.release(both)


def test_reset_refuses_while_a_device_is_open(fake_gozer):
    gz, proc, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "refusal", exact="0000:03:00.0")
    holder = make_proc_dir(proc, 999999, ppid=os.getpid(), devs=[2])   # our child holds chip 2
    assert {c.bdf: c.state for c in a.status()}["0000:03:00.0"] == "HELD"
    with pytest.raises(Refused, match="still open"):
        a.reset(lease)
    assert resets(marker) == []
    shutil.rmtree(holder)
    a.reset(lease)
    a.release(lease)


def test_a_dead_owners_lease_is_stale_and_the_next_acquire_reaps_it_without_a_reset(fake_gozer):
    # This is the gozer behavior recovery relies on (orchard/handoff.py, recover): the lease of a
    # dead supervisor shows STALE with that pid, and the next acquire reaps it and grants the board.
    gz, proc, marker = fake_gozer
    dead = make_proc_dir(proc, 777777)                 # the supervisor that is about to die
    old = GozerAdapter(gozer=str(gz), owner_pid=777777).acquire(1, "orchard:old", "coder",
                                                               exact="0000:03:00.0")
    shutil.rmtree(dead)                                # it dies; no device is open
    states = {c.bdf: c for c in GozerAdapter(gozer=str(gz)).status() if c.bdf in old.chips}
    assert {c.state for c in states.values()} == {"STALE"}
    assert {c.lease_pid for c in states.values()} == {777777}
    new = GozerAdapter(gozer=str(gz)).acquire(1, "orchard:new", "after restart",
                                              exact="0000:03:00.0")
    assert sorted(new.chips) == BOARD1 and new.lease_id != old.lease_id
    assert resets(marker) == []                        # reaped without a reset
    GozerAdapter(gozer=str(gz)).release(new)


def test_a_holder_outside_the_owners_tree_is_held_foreign_and_blocks_the_reset(fake_gozer):
    # A tt-model container's server is never the supervisor's descendant (spec section 6, step 3).
    gz, proc, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "foreign", exact="0000:03:00.0")
    holder = make_proc_dir(proc, 888888, ppid=1, devs=[2])
    assert {c.bdf: c.state for c in a.status()}["0000:03:00.0"] == "HELD-FOREIGN"
    with pytest.raises(Refused):
        a.reset(lease)
    assert resets(marker) == []
    shutil.rmtree(holder)
    a.release(lease)


def test_gozer_identity_names_the_commit_and_a_dirty_tree(tmp_path, monkeypatch):
    repo = tmp_path / "gz"
    (repo / "bin").mkdir(parents=True)
    (repo / "bin" / "gozer").write_text("x\n")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, env=env)
    monkeypatch.setattr(sys.modules[__name__], "GOZER", repo / "bin" / "gozer")
    assert "(clean)" in gozer_identity() and "at commit " in gozer_identity()
    (repo / "bin" / "gozer").write_text("edited\n")
    assert "uncommitted edits" in gozer_identity()
    monkeypatch.setattr(sys.modules[__name__], "GOZER", tmp_path / "nowhere" / "bin" / "gozer")
    assert "version is unknown" in gozer_identity()
