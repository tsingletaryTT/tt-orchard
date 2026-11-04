# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a helper program with a deadline and a cap on each output stream, in a session of its own (PR 422, rounds 3 and 4).

`subprocess.run(capture_output=True)` collects everything a child writes before the caller can look at the size, and its timeout kills only
the child it started. A wrapper (a shell script that execs Python, an interpreter shim) can leave the real worker running. This module
reads stdout and stderr in chunks and stops reading at the cap, starts the child in a new session, and on every way out (the child ended, a
timeout, an over-long stream, an exception out of the wait) sends SIGTERM to the whole process group, then SIGKILL after a grace period, and
reaps the child. The standard library only, because the supervisor uses nothing else.

What `run` guarantees, and what it does not:
- It returns at most `timeout_s + 3 * GRACE_S + JOIN_S` seconds (plus a few polls) after it started, however the child behaves: it ended by
  itself, it ignores SIGTERM, a descendant keeps the output pipes open and writes nothing, or the output floods. (Review of round 3: a child
  that ended while a descendant held the pipes used to hold `run` until the descendant ended, because the readers were blocked in a buffered
  read and `BufferedReader.close()` waits for the lock that read holds. The readers now poll raw descriptors and look at a stop flag every
  `POLL_S`, and nothing closes a descriptor a reader may still use.)
- The request writer polls a non-blocking descriptor and looks at the stop flag, and `run` joins it with the readers. A descendant that kept
  stdin without reading it (round 5) cannot hold the writer thread, the pipe or the request after `run` has returned.
- Every process of the child's group is gone when it returns, unless one ignores SIGKILL (it cannot) or sits in uninterruptible sleep, in which
  case `run` returns after the bound and the process stays. A descendant that left the group (its own `setsid`) is not reached.
- The group is signalled only while the leader has not been reaped: the child's end is detected with `waitid(WNOWAIT)`, so its pid, which is
  also the group id, stays reserved until the last signal has gone. After the leader is reaped no signal is sent (a pid that was freed could
  name another group). What remains: a group id is named by number, so a caller that signals the same group some other way after `run`
  returns has no such protection.
- The verdict comes from what the readers saw: a stream that crossed its cap is a problem even when the child had already ended by itself.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

CHUNK = 64 * 1024            # the most a stream can hold beyond its cap: the chunk that crossed it
GRACE_S = 5.0                # between SIGTERM and SIGKILL, and again between SIGKILL and giving up on the group
POLL_S = 0.05                # how often a reader looks at its stop flag
JOIN_S = 2.0                 # in all, to let the readers finish after the group has been stopped


@dataclass
class Result:
    """`problem` is None for a child that ended by itself, else "timeout", "stdout", "stderr" or "start" (with `detail`)."""
    problem: str | None
    returncode: int | None
    stdout: str
    stderr: str
    detail: str = ""


def _read(fd: int, cap: int, sink: list, over: threading.Event, stop: threading.Event) -> None:
    """Read the raw descriptor `fd` in chunks into `sink` until the end, the cap, or the stop flag. Crossing the cap sets `over` and stops
    reading, so at most one chunk beyond the cap is held. The descriptor is polled with a timeout, so the thread never blocks for good: when
    the stop flag is set it still reads what is already in the pipe (up to the cap) and then ends."""
    held = 0
    poller = select.poll()
    try:
        poller.register(fd, select.POLLIN)
        while True:
            if not poller.poll(POLL_S * 1000):
                if stop.is_set():
                    return
                continue
            chunk = os.read(fd, CHUNK)
            if not chunk:
                return
            sink.append(chunk)
            held += len(chunk)
            if held > cap:
                over.set()
                return
    except (OSError, ValueError):
        return


def _write(stream, data: bytes, stop: threading.Event) -> None:
    """Write the request to the helper's stdin without ever blocking for good. The descriptor is non-blocking and polled, so a helper (or a
    descendant that kept the pipe and never reads it) cannot hold this thread, the pipe and the request bytes after `run` has returned: the stop
    flag is looked at every POLL_S. The pipe is closed when the request has been written, so a helper that reads to the end sees the end."""
    fd = stream.fileno()
    view = memoryview(data)
    try:
        os.set_blocking(fd, False)
        poller = select.poll()
        poller.register(fd, select.POLLOUT)
        while view:
            if stop.is_set():
                return
            if not poller.poll(POLL_S * 1000):
                continue
            try:
                view = view[os.write(fd, view[:CHUNK]):]
            except BlockingIOError:
                continue
    except (OSError, ValueError):                  # a helper that exits before it reads its request closes the pipe
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _signal_group(proc, sig) -> None:
    """Signal the child's process group. Never after the child has been reaped: its pid is then free and could name another group."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _ended(proc) -> bool:
    """True when the child has ended, WITHOUT reaping it: it stays a zombie, so its pid (the group id) stays reserved until `proc.wait()`."""
    if proc.returncode is not None:
        return True
    try:
        return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:                      # reaped by somebody else
        return True


def _proc_stat(name: str) -> bytes | None:
    """The first 4 KiB of /proc/<name>/stat, or None. The kernel's own file: its size reads as 0, so `safeio` (which compares sizes) is not used."""
    try:
        fd = os.open(f"/proc/{name}/stat", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        return os.read(fd, 4096)
    except OSError:
        return None
    finally:
        os.close(fd)


def _live_members(pgid: int) -> list[int] | None:
    """Live (not zombie) processes of the process group `pgid`, from /proc, or None when /proc cannot be listed."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    found = []
    for name in names:
        if not name.isdigit():
            continue
        data = _proc_stat(name)
        if data is None:
            continue
        try:
            fields = data[data.rindex(b")") + 2:].split()
            state, pgrp = fields[0], int(fields[2])
        except (ValueError, IndexError):
            continue
        if pgrp == pgid and state not in (b"Z", b"X"):
            found.append(int(name))
    return found


def _wait_group_gone(proc, seconds: float) -> bool:
    """Wait until no live process is left in the child's group (a zombie leader does not count), at most `seconds`. Without /proc the leader
    alone is waited for."""
    end = time.monotonic() + seconds
    while True:
        members = _live_members(proc.pid)
        if members is None:
            members = [] if _ended(proc) else [proc.pid]
        if not members:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(0.02)


def _stop(proc) -> None:
    """End the child's whole process group (SIGTERM, then SIGKILL after the grace period) and reap the child. The same whether the child is
    still running or ended by itself and left descendants. The leader is reaped last, so every signal goes out while its pid is reserved."""
    _signal_group(proc, signal.SIGTERM)
    _wait_group_gone(proc, GRACE_S)
    _signal_group(proc, signal.SIGKILL)            # also reaches anything that ignored SIGTERM
    _wait_group_gone(proc, GRACE_S)
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
        if _ended(proc):
            return None
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(0.02)


def run(argv, request: str, *, timeout_s: float, out_max: int, err_max: int, env: dict | None = None) -> Result:
    """Run `argv` with `request` on stdin. Never raises for a child that fails; an exception out of the wait (for example an interrupt)
    stops the child's group first and then propagates. See the module docstring for the time bound and what is guaranteed."""
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                start_new_session=True)
    except (OSError, ValueError) as exc:
        return Result("start", None, "", "", type(exc).__name__)
    out, err = [], []
    over_out, over_err, stop = threading.Event(), threading.Event(), threading.Event()
    threads = [threading.Thread(target=_read, args=(proc.stdout.fileno(), out_max, out, over_out, stop), daemon=True),
               threading.Thread(target=_read, args=(proc.stderr.fileno(), err_max, err, over_err, stop), daemon=True),
               threading.Thread(target=_write, args=(proc.stdin, request.encode("utf-8"), stop), daemon=True)]
    for t in threads:
        t.start()
    problem = None
    try:
        problem = _wait_for_end(proc, time.monotonic() + timeout_s, over_out, over_err)
    finally:
        # Every way out (the child ended, a deadline, a cap, an exception such as an interrupt) stops the group, also when the child had ended
        # by itself: a descendant of it may still hold the pipes.
        try:
            _stop(proc)
        finally:
            stop.set()
    # The group is gone. The readers take what is already in the pipes (up to the cap) and end within POLL_S; only then are the descriptors closed.
    end = time.monotonic() + JOIN_S
    for t in threads:                              # the writer too: it sees the stop flag within POLL_S, so it ends and closes stdin
        t.join(timeout=max(0.0, end - time.monotonic()))
    for t, stream in zip(threads[:2], (proc.stdout, proc.stderr)):
        if t.is_alive():                           # never close a descriptor a thread may still read
            continue
        try:
            stream.close()
        except (OSError, ValueError):
            pass
    if problem is None:                            # the verdict comes from what the readers saw, also when the child had ended first
        if over_out.is_set():
            problem = "stdout"
        elif over_err.is_set():
            problem = "stderr"
    return Result(problem, proc.returncode, b"".join(out).decode("utf-8", "replace"), b"".join(err).decode("utf-8", "replace"))
