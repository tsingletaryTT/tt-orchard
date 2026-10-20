# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Watch a child process for the signs that it is stuck, so a long run does not wait on a timer alone.

A timer says "this has taken long". These signs say "nothing is happening": the output has stopped growing and the
process tree has used no CPU, or one output line keeps repeating. A hard deadline stays as the last backstop, because
a process can be busy and still wrong (a spin that prints nothing).

`run_watched` samples about once a second. The sampling interval is not the decision. The decision comes from what the
child did: bytes written, CPU ticks used by it and everything it started, and the lines it printed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


class StallWatcher:
    """Decides from samples whether a process has stopped making progress. No clock of its own: the caller passes `now`."""

    def __init__(self, *, quiet_s: float = 60.0, min_cpu_ticks: int = 2, repeat_lines: int = 20):
        self.quiet_s, self.min_cpu_ticks, self.repeat_lines = quiet_s, min_cpu_ticks, repeat_lines
        self._mark_at = self._mark_bytes = self._mark_ticks = None
        self._line, self._run = None, 0
        # What the watcher saw, for setting the limits from a measurement: the longest stretch with no progress (to
        # the sample that ended it, or to the last sample) and the longest run of one repeated line.
        self.max_quiet_s, self.max_repeat = 0.0, 0

    def observe(self, now: float, out_bytes: int, cpu_ticks: int) -> str | None:
        if self._mark_at is None:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        # A drop in the tree CPU total means a busy descendant exited and its ticks left the sum. The tree changed;
        # no progress was lost. Rebase the mark. The drop does not count as progress and does not reset the clock.
        if cpu_ticks < self._mark_ticks:
            self._mark_ticks = cpu_ticks
        # Progress is measured against the last time progress was seen, so slow CPU use still adds up.
        quiet = now - self._mark_at
        self.max_quiet_s = max(self.max_quiet_s, quiet)
        if out_bytes > self._mark_bytes or cpu_ticks - self._mark_ticks >= self.min_cpu_ticks:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        if quiet >= self.quiet_s:
            return f"stalled: no output and no CPU use for {quiet:.0f} s"
        return None

    def feed_lines(self, lines) -> str | None:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            self._run = self._run + 1 if line == self._line else 1
            self._line = line
            self.max_repeat = max(self.max_repeat, self._run)
            if self._run >= self.repeat_lines:
                return f"repeating: {line[:80]!r} x {self._run}"
        return None


def tree_cpu_ticks(pid: int, proc_root: str = "/proc") -> int:
    """User plus system CPU ticks of `pid` and every descendant, read from /proc. 0 when the pid is gone."""
    stats: dict[int, tuple[int, int]] = {}            # pid -> (ppid, ticks)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return 0
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, name, "stat"), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # The command name sits in parentheses and may hold spaces and parentheses, so split after the last ")".
        rest = text[text.rfind(")") + 2:].split()
        try:
            stats[int(name)] = (int(rest[1]), int(rest[11]) + int(rest[12]))   # ppid; utime + stime
        except (IndexError, ValueError):
            continue
    children: dict[int, list[int]] = {}
    for p, (pp, _) in stats.items():
        children.setdefault(pp, []).append(p)
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        if p in stats:
            total += stats[p][1]
        todo.extend(children.get(p, []))
    return total


@dataclass
class Watched:
    returncode: int | None
    reason: str | None        # why it was killed, or None when it ended by itself
    killed: bool
    elapsed_s: float
    max_pending_bytes: int = 0   # the most bytes of an unfinished line the watcher held at one time
    max_quiet_s: float = 0.0     # the longest stretch with no output and no CPU use that the watcher saw
    max_repeat: int = 0          # the longest run of one repeated line that the watcher saw


# The watcher keeps the bytes after the last newline until their line is complete. A child that prints without
# newlines would make that buffer grow with its output, so a piece longer than this is cut off and fed to the repeat
# detector as a line of its own (the detector needs one line at a time). Identical output with no newline then shows
# up as a repeating line.
PENDING_MAX_BYTES = 64 * 1024


WAIT_AFTER_KILL_S = 10.0     # longest wait for the leader to be reaped after the kill


TERM_POLL_S = 0.05           # how often the group is checked during the SIGTERM grace


def _signalable(pgid) -> bool:
    """The cheap part of the guard: an id that is a real pid and not the caller's own group."""
    return isinstance(pgid, int) and pgid > 1 and pgid != os.getpgrp()


def _own_group(pgid) -> bool:
    """True only when `pgid` is a process group the child leads itself (run_watched starts it with
    start_new_session=True, so the group id equals the child's pid).

    os.killpg(0, sig) signals the caller's own group, and ids 1 and below or the caller's group would hit the test
    runner, the shell or init. A pid that does not lead a group of its own (a reused pid, a fake) would hit
    someone else's processes. None of these may be signalled, so every killpg call checks here first.
    """
    if not _signalable(pgid):
        return False
    try:
        return os.getpgid(pgid) == pgid
    except (ProcessLookupError, PermissionError):
        return False


def _group_gone(pgid: int) -> bool:
    """True when no process is left in the group. A group owned by someone else (a reused id) counts as gone."""
    if not _signalable(pgid):
        return True
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return True
    return False


def _sigkill_group(pgid) -> None:
    """SIGKILL the group unless it is already gone. The group is probed first, so a group that ended in the last
    moment is not signalled. No leadership check here: the reaped leader no longer answers getpgid, and Linux keeps
    the group id reserved while any member lives. The only unsafe window needs the last member to die and the pid
    counter to wrap all the way round between the probe and the signal; the exposure is the same as with a plain
    SIGKILL of the group."""
    if not _signalable(pgid) or _group_gone(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_group(proc, term_grace_s: float = 5.0) -> None:
    """Stop the child's whole process group: SIGTERM first, then SIGKILL for whatever is left after the grace.

    A wrapper such as `gozer run` releases its chip lease when it gets SIGTERM; SIGKILL would leave the lease held by
    a dead process. The group is judged gone by the group itself, so a leader that exits on SIGTERM while a
    grandchild keeps running still gets the SIGKILL for the grandchild. The leader is reaped on each check, because
    a zombie leader keeps its group alive. A group that holds only zombies (an init that does not reap) also looks
    alive, because signal 0 succeeds on a zombie; that kill waits out the grace and then sends a harmless SIGKILL.

    Any exception during the grace, such as a second interrupt or abort signal, sends the SIGKILL before it
    propagates, so a child that ignores SIGTERM cannot keep its lease because of an interrupt.

    Only a group the child leads is signalled (see `_own_group`).
    """
    if not _own_group(proc.pid):
        return                                    # never signal a group that is not the child's own
    try:
        os.killpg(proc.pid, signal.SIGTERM)       # the child leads its own session, so this reaches grandchildren
    except (ProcessLookupError, PermissionError):
        return                                    # no such group (already gone, or the pid belongs to someone else)
    deadline = time.monotonic() + term_grace_s
    try:
        while True:
            try:
                proc.poll()                       # reaps the leader once it has exited
            except Exception:
                pass
            if _group_gone(proc.pid):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(min(TERM_POLL_S, max(0.0, deadline - time.monotonic())))
    except BaseException:
        _sigkill_group(proc.pid)
        raise
    _sigkill_group(proc.pid)


def run_watched(argv, *, cwd, env, log_path, watcher: StallWatcher, backstop_s: float, poll_s: float = 1.0,
                proc_root: str = "/proc", popen=subprocess.Popen, clock=time.monotonic,
                term_grace_s: float = 5.0) -> Watched:
    """Run `argv` with stdout and stderr appended to `log_path`, and stop it when the watcher says it is stuck or
    when `backstop_s` has passed. The child gets its own session so the whole group can be stopped: SIGTERM first,
    SIGKILL after `term_grace_s` for anything still running."""
    start = clock()
    with open(log_path, "ab") as log:
        # Start reading where this child's output begins: an appended log may hold an earlier attempt's lines.
        read_pos = os.path.getsize(log_path)
        pending = b""                                  # bytes after the last newline, kept until the line is complete
        max_pending = 0

        def seen():
            return (max_pending, getattr(watcher, "max_quiet_s", 0.0), getattr(watcher, "max_repeat", 0))
        proc = popen(list(argv), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    return Watched(proc.wait(timeout=poll_s), None, False, clock() - start, *seen())
                except subprocess.TimeoutExpired:
                    pass
                now = clock()
                size = os.path.getsize(log_path)
                with open(log_path, "rb") as f:
                    f.seek(read_pos)
                    data = pending + f.read(size - read_pos)
                read_pos = size
                cut = data.rfind(b"\n") + 1                # decode whole lines only, so no line or character is split
                fresh = data[:cut].decode("utf-8", errors="replace").splitlines()
                pending = data[cut:]
                while len(pending) > PENDING_MAX_BYTES:   # a line this long is cut, and each piece is fed as a line
                    fresh.append(pending[:PENDING_MAX_BYTES].decode("utf-8", errors="replace"))
                    pending = pending[PENDING_MAX_BYTES:]
                max_pending = max(max_pending, len(pending))
                reason = watcher.feed_lines(fresh) or watcher.observe(now, size, tree_cpu_ticks(proc.pid, proc_root))
                if reason is None and now - start >= backstop_s:
                    reason = f"backstop: still running after {backstop_s:.0f} s"
                if reason is not None:
                    if proc.poll() is not None:        # it ended by itself between the wait and this decision
                        return Watched(proc.returncode, None, False, clock() - start, *seen())
                    _kill_group(proc, term_grace_s=term_grace_s)
                    try:
                        code = proc.wait(timeout=WAIT_AFTER_KILL_S)
                    except subprocess.TimeoutExpired:
                        code = None
                    return Watched(code, reason, True, clock() - start, *seen())
        except BaseException:
            # Any error or interrupt (including the supervisor's abort signals) must not leave the child running.
            _kill_group(proc, term_grace_s=term_grace_s)
            try:
                proc.wait(timeout=WAIT_AFTER_KILL_S)
            except subprocess.TimeoutExpired:
                pass
            raise
