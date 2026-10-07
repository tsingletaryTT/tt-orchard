# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run one external program for supervisor code and keep what it printed.

This module owns the one place where supervisor code (the lease adapters and server control)
starts an external program. Agent commands go through orchard/runner.py and its denials. This
helper has none, because only supervisor code calls it, with fixed argument lists.

Each command runs in its own session, so a Ctrl-C at the terminal reaches only the supervisor.
On a timeout the command is killed unless the caller says not to. A chip reset must never be
cut short: killing gozer would leave its `tt-smi -r` running with nobody watching.
"""
from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int | None          # None when the command timed out
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    left_running: bool = False      # timed out and deliberately not killed
    pid: int | None = None

    def record(self, limit: int = 2000) -> dict:
        """The ledger form: the first `limit` characters of each stream."""
        return {"argv": list(self.argv), "returncode": self.returncode,
                "stdout": self.stdout[:limit], "stderr": self.stderr[:limit],
                "timed_out": self.timed_out, "left_running": self.left_running}


Run = Callable[..., CommandResult]


def run_command(argv, timeout: float, *, env: dict | None = None,
                kill_on_timeout: bool = True) -> CommandResult:
    argv = tuple(str(a) for a in argv)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True, env=env)
    except FileNotFoundError as exc:
        return CommandResult(argv, 127, "", f"not found: {exc}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if not kill_on_timeout:
            # Left running on purpose. The Popen object is dropped and its pipes are never read.
            # CPython's Popen.__del__ puts a child that is still running on subprocess._active,
            # and the next Popen reaps it once it exits. gozer prints little, so its pipes cannot
            # fill. `tt-model serve --detach` also uses this path, and nobody has measured how much
            # it prints. If it writes more than the 64 KiB pipe buffer before the container is up,
            # it blocks on the write and the start stalls.
            return CommandResult(argv, None, timed_out=True, left_running=True, pid=proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return CommandResult(argv, None, out, err, timed_out=True, pid=proc.pid)
    return CommandResult(argv, proc.returncode, out, err, pid=proc.pid)
