# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""park_check against the real gozer CLI with fake roots, and real fake-server processes.

Nothing here opens a device: gozer's reset command is a stub that writes a marker file. The
gozer tests skip when gozer is not present, and a skip is not evidence.
"""
import json
import os
import socket
import subprocess
import sys

import pytest

from orchard import park_check
from orchard.ledger import Ledger
from test_gozer_contract import GOZER, QUIETBOX, build_sysfs, make_proc_dir, script

needs_gozer = pytest.mark.skipif(not GOZER.exists(),
                                 reason=f"gozer not found at {GOZER}; a skip is not evidence")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_box(tmp_path, monkeypatch):
    """(gozer wrapper, fake proc root, reset marker file) for a fake two-board box."""
    proc = tmp_path / "proc"
    proc.mkdir()
    marker = tmp_path / "resets"
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", build_sysfs(str(tmp_path / "sys"), QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", str(proc))
    monkeypatch.setenv("GOZER_HISTORY_ROOT", str(tmp_path / "history"))
    monkeypatch.setenv("GOZER_RESET_CMD",
                       str(script(tmp_path / "reset.sh", f'echo "$@" >> {marker}\n')))
    make_proc_dir(proc, os.getpid())
    gz = script(tmp_path / "gozer", f'exec {sys.executable} {GOZER} "$@"\n')
    return gz, proc, marker


def argv(tmp_path, gz):
    return ["--board", "0000:03:00.0", "--out-dir", str(tmp_path / "out"), "--gozer", str(gz),
            "--port", str(free_port()), "--standin-port", str(free_port()), "--ready-poll", "0.1",
            "--allow-gozer-env"]


def tripwire(tmp_path):
    """A gozer that must never run: it writes a marker and fails.

    Every test that calls park_check.main passes an explicit --gozer. Without one, a broken
    preflight would run the real gozer, against an empty GOZER_ROOT (so no real lease is seen)
    and with the default reset command, which is a real `tt-smi -r`.
    """
    marker = tmp_path / "tripwire-ran"
    return script(tmp_path / "gozer-tripwire", f'echo "$@" >> {marker}\nexit 1\n'), marker


def test_preflight_refuses_when_gozer_variables_are_set(tmp_path, monkeypatch):
    gz, marker = tripwire(tmp_path)
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    code = park_check.main(["--board", "0000:03:00.0", "--out-dir", str(tmp_path / "out"),
                            "--gozer", str(gz)])
    assert code == park_check.EXIT_PREFLIGHT
    assert not marker.exists()          # no gozer command ran


@needs_gozer
def test_preflight_refuses_while_the_other_board_is_busy(tmp_path, fake_box):
    gz, proc, marker = fake_box
    # A process outside any lease holds chip 0, as tt-smi -r does during another board's reset.
    make_proc_dir(proc, 888888, ppid=1, devs=[0])
    assert park_check.main(argv(tmp_path, gz)) == park_check.EXIT_PREFLIGHT
    assert not marker.exists()


@needs_gozer
def test_a_full_park_check_on_a_fake_box(tmp_path, fake_box):
    gz, _, marker = fake_box
    code = park_check.main(argv(tmp_path, gz))
    out = tmp_path / "out"
    with Ledger(out / "ledger.jsonl") as led:
        entries = led.read()
    stops = [e["data"] for e in entries if e["event"] == "notice" and not e["data"].get("ok", True)]
    assert code == park_check.EXIT_PASS, stops
    steps = [e["data"]["step"] for e in entries if e["event"] in ("park", "restore")]
    assert steps == ["note", "canary_before", "standin_started", "standin", "stop_sent", "stopped", "reset",
                     "reset", "serve", "ready", "canary", "resumed"]
    stop = next(e["data"] for e in entries
                if e["event"] == "park" and e["data"]["step"] == "stop_sent")
    # A server started while SIGTERM is blocked inherits the mask and ignores SIGTERM.
    assert stop["result"]["how"] == "SIGTERM"
    assert len(marker.read_text().splitlines()) == 3      # park, restore, release
    assert (out / "summary.md").exists()
    status = json.loads(subprocess.run([str(gz), "status", "--json"], capture_output=True,
                                       text=True).stdout)
    assert all(c["state"] == "FREE" for c in status["chips"])
