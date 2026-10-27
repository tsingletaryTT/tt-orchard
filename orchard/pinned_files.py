# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read and write files relative to an open directory descriptor.

A check on a path ("is this a link?", "does this resolve inside the run?") says nothing about the next open of that path: the directory can be
replaced in between, and `O_NOFOLLOW` covers only the last component. The helpers here open a directory once and do every later open relative to that
descriptor (`dir_fd=`), so a directory renamed or replaced by a link afterwards cannot redirect them. The file itself is opened without following a
link and without blocking on a FIFO, and its type is checked on the open descriptor, never on the name.

No earlier helper in this repository did this: `kernelprobe._read_regular` and `ttlang_env._read_reference` open by name and protect the last
component only (the second checks the middle components with lstat first, which is the check-then-open pattern this module removes).
"""
from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path

NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
CHUNK = 1 << 20


class PinnedError(Exception):
    """A name, directory or file that cannot be used safely. The message says what and why."""


def check_name(name) -> str:
    """A single plain file name: not empty, not `.` or `..`, no slash and no NUL."""
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\0" in name:
        raise PinnedError(f"{name!r} is not a plain file name")
    return name


def _explain(exc: OSError, label: str) -> PinnedError:
    if exc.errno == errno.ELOOP:
        return PinnedError(f"{label} is a link; put the real file or directory there")
    if exc.errno == errno.ENOENT:
        return PinnedError(f"{label} is missing")
    if exc.errno == errno.ENOTDIR:
        return PinnedError(f"{label} is not a directory")
    if exc.errno == errno.EEXIST:
        return PinnedError(f"{label} already exists")
    return PinnedError(f"{label} cannot be opened: {exc.strerror or exc}")


def open_dir(path, *, dir_fd: int | None = None, follow: bool = True) -> int:
    """A descriptor for a directory. `path` is a name relative to `dir_fd` when one is given. With `follow=False` the final component must not
    be a link (a link to a directory is refused). Returns a descriptor the caller closes."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | (0 if follow else NOFOLLOW)
    label = str(path)
    try:
        fd = os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR and not follow:
            # Linux answers ENOTDIR for a link opened with O_DIRECTORY | O_NOFOLLOW. The lstat only picks the message; the open already refused.
            try:
                if stat.S_ISLNK(os.stat(path, dir_fd=dir_fd, follow_symlinks=False).st_mode):
                    raise PinnedError(f"{label} is a link; put the real file or directory there") from exc
            except OSError:
                pass
        raise _explain(exc, label) from exc
    return fd


def _open_regular(dir_fd: int, name: str) -> int:
    check_name(name)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | NOFOLLOW | getattr(os, "O_NONBLOCK", 0), dir_fd=dir_fd)
    except OSError as exc:
        raise _explain(exc, name) from exc
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise PinnedError(f"{name} is not a regular file")
    return fd


def read_regular(dir_fd: int, name: str, limit: int) -> bytes:
    """The bytes of a regular file `name` in the directory, at most `limit`. A file over the limit, including one that grows while it is read,
    is refused."""
    fd = _open_regular(dir_fd, name)
    with os.fdopen(fd, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise PinnedError(f"{name} is larger than {limit} bytes")
    return data


def sha256_regular(dir_fd: int, name: str, limit: int) -> tuple[str, int]:
    """(sha256 hex digest, size) of a regular file, read in counted chunks. A file over `limit` bytes is refused before more is hashed."""
    fd = _open_regular(dir_fd, name)
    total = 0
    with os.fdopen(fd, "rb") as f:
        if os.fstat(f.fileno()).st_size > limit:
            raise PinnedError(f"{name} is larger than {limit} bytes")
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(CHUNK), b""):
            total += len(chunk)
            if total > limit:
                raise PinnedError(f"{name} is larger than {limit} bytes")
            h.update(chunk)
    return h.hexdigest(), total


def create_exclusive(dir_fd: int, name: str, mode: int = 0o644) -> int:
    """A new file `name` opened for writing. It must not exist, and a link of that name is never followed. Returns a descriptor."""
    check_name(name)
    try:
        return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | NOFOLLOW, mode, dir_fd=dir_fd)
    except OSError as exc:
        raise _explain(exc, name) from exc


def publish_no_replace(dir_fd: int, tmp_name: str, final_name: str) -> None:
    """Give the finished temporary file its final name, failing when the final name exists (a hard link, which never replaces), then drop the
    temporary name. The temporary file is left in place when publishing fails."""
    check_name(tmp_name)
    check_name(final_name)
    try:
        os.link(tmp_name, final_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False)
    except OSError as exc:
        raise _explain(exc, final_name if exc.errno == errno.EEXIST else tmp_name) from exc
    os.unlink(tmp_name, dir_fd=dir_fd)


def unlink_quietly(dir_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError:
        pass


def fsync_dir(dir_fd: int) -> None:
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
