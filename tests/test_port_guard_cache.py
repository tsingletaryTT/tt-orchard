# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 462, round 3 (issue 458): the port guard keeps the bytes of each accepted file so a later rejected edit can be put back.
That cache must stay bounded across the shell commands of a long step.

Entries for files that are gone are dropped when the scan was complete; a total byte budget evicts the oldest backups; and a file whose
backup was evicted is never rolled back to a wrong version and never removed: the agent is told the previous version was not kept."""
import os

import pytest

from orchard import port_guard, runaway
from orchard.port_guard import PortGuard


@pytest.fixture
def stage(tmp_path):
    sd = tmp_path / "run" / "stages" / "0"
    (sd / "port").mkdir(parents=True)
    return sd


def put(sd, name, text):
    p = sd / "port" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def varied(kilobytes: int, tag: str) -> str:
    """Python-looking text of about `kilobytes` KB that the content check accepts (one distinct line per value; a repeated line is a loop)."""
    n = kilobytes * 1000 // 45
    return "\n".join(f"value_{tag}_{i} = {i * 7919 % 10007}  # {i * 2654435761 % 2**32:08x} {i * 40503 % 2**32:08x} {i * 97 % 65521:04x}"
                     for i in range(n)) + "\n"


def settle(guard):
    """What the shell tool does around one command: remember before, judge after."""
    guard.remember()
    return guard.judge()


def test_entries_for_files_that_are_gone_are_dropped_after_a_complete_scan(stage):
    guard = PortGuard(stage)
    for batch in range(30):
        names = [f"batch{batch}_{i}.py" for i in range(10)]
        for n in names:
            put(stage, n, f"x = {batch}\n" * 40)
        guard.remember()
        guard.judge()
        for n in names:
            (stage / "port" / n).unlink()
    guard.remember()
    guard.judge()
    entries, size = guard.cache_stats()
    assert entries == 0 and size == 0, "backups and stamps of deleted files stayed in memory"


def test_the_total_backup_bytes_stay_under_the_budget(stage, monkeypatch):
    monkeypatch.setattr(port_guard, "PORT_GUARD_BACKUP_BYTES", 600_000)
    guard = PortGuard(stage)
    for i in range(12):
        put(stage, f"f{i}.py", varied(150, f"f{i}"))                       # about 150 KB each: well under the per-file cap
        settle(guard)
    entries, size = guard.cache_stats()
    assert size <= 600_000, f"{size} bytes of backups are kept; the budget is 600000"
    assert entries >= 1


def test_a_file_whose_backup_was_evicted_is_neither_restored_to_a_wrong_version_nor_removed(stage, monkeypatch):
    monkeypatch.setattr(port_guard, "PORT_GUARD_BACKUP_BYTES", 400_000)
    guard = PortGuard(stage)
    first = put(stage, "first.py", varied(120, "first"))                   # about 120 KB
    settle(guard)
    for i in range(6):                                                       # newer files push the first one's backup out
        put(stage, f"later{i}.py", varied(160, f"later{i}"))
        settle(guard)
    assert guard.backup_evicted(("first.py",)), "the test needs the first backup to have been evicted"
    guard.remember()
    first.write_text("def f():\n    return 1\n" * 1500)                      # a loop: rejected by the content check
    notes = guard.judge()
    assert first.exists(), "an accepted file whose backup was evicted must not be removed"
    assert first.read_text().startswith("def f():"), "it must not be rolled back to a wrong version either"
    assert any("first.py" in n and "not kept" in n for n in notes), notes


def test_a_file_whose_backup_is_kept_is_still_restored(stage):
    guard = PortGuard(stage)
    p = put(stage, "port_model.py", "def f():\n    return 1\n")
    settle(guard)
    guard.remember()
    p.write_text("def f():\n    return 1\n" * 1500)
    notes = guard.judge()
    assert any("restored" in n for n in notes) and p.read_text() == "def f():\n    return 1\n"


def test_a_truncated_scan_does_not_forget_the_backups_of_files_it_did_not_list(stage, monkeypatch):
    guard = PortGuard(stage)
    for i in range(4):
        put(stage, f"keep{i}.py", f"c = {i}\n")
    settle(guard)
    assert guard.cache_stats()[0] == 4
    monkeypatch.setattr(port_guard, "PORT_GUARD_ENTRIES", 2)                 # the next scan stops early
    guard.remember()
    guard.judge()
    assert guard.cache_stats()[0] == 4, "a scan that stopped at its bound is not evidence that the other files are gone"


def test_the_stamp_table_is_bounded_when_the_scan_never_completes(stage, monkeypatch):
    monkeypatch.setattr(port_guard, "PORT_GUARD_STAMPS", 5)
    monkeypatch.setattr(port_guard, "PORT_GUARD_ENTRIES", 3)                 # every scan is truncated: nothing is ever pruned by listing
    guard = PortGuard(stage)
    for i in range(40):
        put(stage, f"n{i}.py", f"d = {i}\n")
        settle(guard)
        os.unlink(stage / "port" / f"n{i}.py")
    assert len(guard._stamp) <= 5, "the stamp table grew without limit"
