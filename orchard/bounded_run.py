# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a helper program with a deadline and a cap on each output stream, in a session of its own (PR 422, round 3).

`subprocess.run(capture_output=True)` collects everything a child writes before the caller can look at the size, and its timeout kills only
the child it started. A wrapper (a shell script that execs Python, an interpreter shim) can leave the real worker running. This module
reads stdout and stderr in chunks and stops reading at the cap, starts the child in a new session, and on a timeout, an over-long stream
or an exception out of the wait it sends SIGTERM to the whole process group, then SIGKILL after a grace period, and reaps the child.
The standard library only, because the supervisor uses nothing else.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

CHUNK = 64 * 1024            # the most a stream can hold beyond its cap: the chunk that crossed it
GRACE_S = 5.0                # between SIGTERM and SIGKILL


@dataclass
class Result:
    """`problem` is None for a child that ended by itself, else "timeout", "stdout", "stderr" or "start" (with `detail`)."""
    problem: str | None
    returncode: int | None
    stdout: str
    stderr: str
    detail: str = ""


def _read(stream, cap: int, sink: list, over: threading.Event) -> None:
    """Read `stream` in chunks into `sink` until the end or the cap. Crossing the cap sets `over` and stops reading, so at most one chunk
    beyond the cap is held. The thread ends at the end of the stream, which the group's stop brings about."""
    held = 0
    try:
        while True:
            chunk = stream.read1(CHUNK)
            if not chunk:
                return
            sink.append(chunk)
            held += len(chunk)
            if held > cap:
                over.set()
                return
    except (OSError, ValueError):
        return


def _write(stream, data: bytes) -> None:
    try:
        stream.write(data)
    except (OSError, ValueError):                  # a helper that exits before it reads its request closes the pipe
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _signal_group(proc, sig) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _stop(proc) -> None:
    """End the child's whole process group (SIGTERM, then SIGKILL after the grace period) and reap the child."""
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(proc, signal.SIGKILL)            # also reaches a grandchild that ignored SIGTERM after its parent had gone
    try:
        proc.wait(timeout=GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_end(proc, deadline: float, over_out: threading.Event, over_err: threading.Event) -> str | None:
    """None when the child ended by itself, else the reason it must be stopped."""
    while True:
        if over_out.is_set():
            return "stdout"
        if over_err.is_set():
            return "stderr"
        if proc.poll() is not None:
            return None
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(0.02)


def run(argv, request: str, *, timeout_s: float, out_max: int, err_max: int, env: dict | None = None) -> Result:
    """Run `argv` with `request` on stdin. Never raises for a child that fails; an exception out of the wait (for example an interrupt)
    stops the child's group first and then propagates."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                start_new_session=True)
    except (OSError, ValueError) as exc:
        return Result("start", None, "", "", type(exc).__name__)
    out, err = [], []
    over_out, over_err = threading.Event(), threading.Event()
    threads = [threading.Thread(target=_read, args=(proc.stdout, out_max, out, over_out), daemon=True),
               threading.Thread(target=_read, args=(proc.stderr, err_max, err, over_err), daemon=True),
               threading.Thread(target=_write, args=(proc.stdin, request.encode("utf-8")), daemon=True)]
    for t in threads:
        t.start()
    try:
        problem = _wait_for_end(proc, time.monotonic() + timeout_s, over_out, over_err)
    finally:
        # Every way out (the child ended, a deadline, a cap, an exception such as an interrupt) leaves no process of the child's group behind.
        if proc.poll() is None:
            _stop(proc)
    # The child has ended (by itself or by the stop). Let the readers drain what is already in the pipes, then close them.
    for t in threads[:2]:
        t.join(timeout=2.0)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    return Result(problem, proc.returncode, b"".join(out).decode("utf-8", "replace"), b"".join(err).decode("utf-8", "replace"))
