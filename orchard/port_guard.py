# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The guard for the port directory of a full-port stage (issue 458, review of PR 462).

A shell command run by the coder agent can write anything under `stages/<N>/port/`. After the command, a file that grew past the size
caps or became a repetition loop is put back as it was, or removed when it is new (orchard/runaway.py judges the content). Everything
here happens through directory descriptors:

- the stage directory is pinned, the `port` directory is opened beneath it with `O_NOFOLLOW`, and every subdirectory is opened beneath
  the descriptor before it, so a link anywhere on the way is refused and nothing outside the real port directory is read, restored or
  removed. If `port` itself is a link, nothing is looked at and the agent is told;
- a file is read through a descriptor opened with `O_NOFOLLOW | O_NONBLOCK`, must be a regular file, and must be the file the scan
  listed (same device and inode), so a file swapped for a link or for another file after the scan is not read;
- a restore writes a temporary file in the directory by descriptor and renames it over the name; a removal unlinks the name beneath the
  descriptor of its directory after checking the identity. A name swapped for a link is replaced or removed as a link and never
  followed; a directory swapped for a link stays where it was opened;
- the scan is lazy and bounded: at most PORT_GUARD_ENTRIES entries of any kind, PORT_GUARD_FILES files and PORT_GUARD_DEPTH levels,
  with no sorting of a directory listing.

What it cannot do: stop another process that owns a directory on the way from renaming things inside it (the same limit as
`safeio.open_beneath`). The backup of each accepted file is kept in memory, at most MAX_FILE_BYTES per file and PORT_GUARD_FILES files.
"""
from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from orchard import runaway, safeio

PORT_GUARD_FILES = 200
PORT_GUARD_ENTRIES = 2000
PORT_GUARD_DEPTH = 4
_TMP_NAME = ".port-guard-tmp"


@dataclass(frozen=True)
class PortFile:
    """A regular file the scan listed: its path parts beneath the port directory, and the identity and size it had then."""
    parts: tuple[str, ...]
    dev: int
    ino: int
    size: int
    mtime_ns: int

    @property
    def stamp(self) -> tuple[int, int, int, int]:
        return (self.dev, self.ino, self.size, self.mtime_ns)


class PortGuard:
    def __init__(self, stage_dir, run_dir=None):
        self.stage_dir = Path(stage_dir)
        self.run_dir = Path(run_dir) if run_dir is not None else self.stage_dir
        self._stamp: dict[tuple[str, ...], tuple[int, int, int, int]] = {}
        self._backup: dict[tuple[str, ...], bytes] = {}

    # ---- anchoring ----

    @contextmanager
    def _pin(self):
        """The port directory pinned beneath the stage directory, or a SafeReadError when it is missing or is a link."""
        with safeio.pinned_root(self.stage_dir) as stage:
            port = stage.child(("port",))
            try:
                yield port
            finally:
                port.close()

    def _linked_root(self) -> bool:
        """Whether `<stage>/port` exists as a link (looked at by the stage descriptor, never followed)."""
        try:
            with safeio.pinned_root(self.stage_dir) as stage:
                return stat.S_ISLNK(os.stat("port", dir_fd=stage.fd, follow_symlinks=False).st_mode)
        except (safeio.SafeReadError, OSError):
            return False

    def _display(self, parts) -> str:
        return os.path.join(os.path.relpath(self.stage_dir, self.run_dir), "port", *parts)

    # ---- the scan ----

    def _scan(self, pin) -> list[PortFile]:
        """The regular files beneath the pinned port directory (links, caches and anything unreadable left out), lazily and bounded."""
        out: list[PortFile] = []
        stack: list[tuple[tuple[str, ...], int]] = [((), 0)]
        seen = 0
        while stack and len(out) < PORT_GUARD_FILES:
            parts, depth = stack.pop()
            try:
                dirpin = pin.child(parts)
            except safeio.SafeReadError:
                continue                                           # a directory swapped for a link, or one that vanished
            try:
                with os.scandir(dirpin.fd) as entries:
                    for e in entries:
                        seen += 1
                        if seen > PORT_GUARD_ENTRIES or len(out) >= PORT_GUARD_FILES:
                            return out
                        try:
                            if e.is_symlink():
                                continue
                            if e.is_dir(follow_symlinks=False):
                                if depth < PORT_GUARD_DEPTH and e.name != "__pycache__":
                                    stack.append((parts + (e.name,), depth + 1))
                            elif e.is_file(follow_symlinks=False):
                                st = e.stat(follow_symlinks=False)
                                out.append(PortFile(parts + (e.name,), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns))
                        except OSError:
                            continue
            except OSError:
                continue
            finally:
                dirpin.close()
        return out

    def files(self) -> list[PortFile]:
        """The files of the port directory now, or none when it is missing or is a link."""
        try:
            with self._pin() as pin:
                return self._scan(pin)
        except safeio.SafeReadError:
            return []

    # ---- reading, restoring, removing ----

    @staticmethod
    def _read(pin, f: PortFile) -> bytes | None:
        """The bytes of the file the scan listed, at most MAX_FILE_BYTES, or None (too large, changed, a link, a FIFO, a device)."""
        if f.size > runaway.MAX_FILE_BYTES:
            return None
        fd = -1
        try:
            fd = safeio.open_beneath(pin.fd, f.parts, file_flags=os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (f.dev, f.ino) or st.st_size > runaway.MAX_FILE_BYTES:
                return None
            chunks, total = [], 0
            while total <= runaway.MAX_FILE_BYTES:
                block = os.read(fd, min(65536, runaway.MAX_FILE_BYTES + 1 - total))
                if not block:
                    break
                chunks.append(block)
                total += len(block)
            return b"".join(chunks) if total <= runaway.MAX_FILE_BYTES else None
        except (safeio.SafeReadError, OSError):
            return None
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    @staticmethod
    def _current(pin, parts: tuple[str, ...]):
        """(dev, ino, size, mtime_ns) of the name `parts` as it is now, looked at without following a link, or None."""
        parent = -1
        try:
            parent = safeio.open_beneath(pin.fd, parts[:-1])
            st = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        except (safeio.SafeReadError, OSError):
            return None
        finally:
            if parent >= 0:
                try:
                    os.close(parent)
                except OSError:
                    pass

    @staticmethod
    def _restore(pin, f: PortFile, data: bytes) -> None:
        """Put `data` back as the file `f.parts`: a temporary file made in its directory by descriptor, renamed over the name."""
        parent = safeio.open_beneath(pin.fd, f.parts[:-1])
        tmp_made = False
        try:
            try:
                os.unlink(_TMP_NAME, dir_fd=parent)              # a leftover of a restore that was interrupted: this name is the guard's own
            except FileNotFoundError:
                pass
            fd = os.open(_TMP_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o644, dir_fd=parent)
            tmp_made = True
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(_TMP_NAME, f.parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
            tmp_made = False
        finally:
            if tmp_made:
                try:
                    os.unlink(_TMP_NAME, dir_fd=parent)
                except OSError:
                    pass
            os.close(parent)

    @staticmethod
    def _remove(pin, f: PortFile) -> None:
        """Unlink the name `f.parts` beneath the descriptor of its directory, if it is still the file the scan listed."""
        parent = safeio.open_beneath(pin.fd, f.parts[:-1])
        try:
            st = os.stat(f.parts[-1], dir_fd=parent, follow_symlinks=False)
            if (st.st_dev, st.st_ino) != (f.dev, f.ino):
                raise OSError("the file was replaced after it was listed")
            os.unlink(f.parts[-1], dir_fd=parent)
        finally:
            os.close(parent)

    # ---- before and after a shell command ----

    def remember(self) -> None:
        """Before a shell command: keep the bytes of each port file that changed since the last look, when the file is acceptable."""
        try:
            with self._pin() as pin:
                for f in self._scan(pin):
                    if self._stamp.get(f.parts) == f.stamp:
                        continue
                    self._stamp[f.parts] = f.stamp
                    raw = self._read(pin, f)
                    if raw is not None and runaway.file_problem(raw.decode("utf-8", errors="replace")) is None:
                        self._backup[f.parts] = raw
                    else:
                        self._backup.pop(f.parts, None)
        except safeio.SafeReadError:
            return

    def judge(self) -> list[str]:
        """After a shell command: a port file that grew past the caps or became a loop is put back as it was (or removed, when it is new),
        and the agent is told. A file that is acceptable becomes the version a later rejection restores. Nothing outside the real port
        directory is looked at; a port directory that is a link is reported and left alone."""
        notes: list[str] = []
        try:
            pin_cm = self._pin()
            pin = pin_cm.__enter__()
        except safeio.SafeReadError:
            if self._linked_root():
                notes.append(f"refused: {self._display(())} is a link, and a port must be a real directory; nothing under it was read or "
                             "changed. Remove the link and put the port files in a directory.")
            return notes
        try:
            for f in self._scan(pin):
                if self._stamp.get(f.parts) == f.stamp:
                    continue
                raw = None
                if f.size > runaway.MAX_FILE_BYTES:
                    problem = f"it is {f.size:,} bytes; the limit for a file you write is {runaway.MAX_FILE_BYTES:,}"
                else:
                    raw = self._read(pin, f)
                    problem = None if raw is None else runaway.file_problem(raw.decode("utf-8", errors="replace"))
                rel = self._display(f.parts)
                if problem is None:
                    self._stamp[f.parts] = f.stamp
                    if raw is not None:
                        self._backup[f.parts] = raw                    # the accepted bytes are what a later rejection puts back
                    else:
                        self._backup.pop(f.parts, None)
                    continue
                backup = self._backup.get(f.parts)
                try:
                    if backup is not None:
                        self._restore(pin, f, backup)
                        what = "restored"
                    else:
                        self._remove(pin, f)
                        what = "removed"
                except (safeio.SafeReadError, OSError) as exc:
                    notes.append(f"refused: {rel} is not acceptable ({problem}) and could not be put back ({exc.__class__.__name__}); "
                                 "delete it or write a shorter file")
                    continue
                now = self._current(pin, f.parts)
                if now is not None:
                    self._stamp[f.parts] = now
                else:
                    self._stamp.pop(f.parts, None)
                notes.append(f"refused and {what}: {rel} {problem}. Write a shorter file: a port is tens of KB, so do not repeat blocks "
                             "and do not generate text with a loop.")
        finally:
            pin_cm.__exit__(None, None, None)
        return notes
