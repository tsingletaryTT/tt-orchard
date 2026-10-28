# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A per-entry lock for the audio reference cache (`<cache>/audio-reference/<model>`), standard library only.

A run that finds a valid entry validates it and then copies the directory by its path (orchard/audio_stages.py `run_references`). A run that made a
fresh reference replaces a stale entry by moving it aside and renaming the new copy in (orchard/supervisor.py `save_reference`). Without a lock a reader
could validate one version and copy another, or find no directory in the instant between the two renames. Readers take a shared lock for validate and
copy, the replacement takes an exclusive one, both on the file `.<model>.lock` in the cache directory. The lock is advisory and best effort: when the lock
file cannot be opened (a read-only cache, no `fcntl`) the caller goes on without it, as it did before the lock existed, because the cache only saves time.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

try:
    import fcntl
except ImportError:                                   # a platform with no flock: no locking, as before
    fcntl = None

LOCK_SUFFIX = ".lock"


def lock_path(cache_dir, name: str) -> Path:
    return Path(cache_dir) / f".{name}{LOCK_SUFFIX}"


@contextlib.contextmanager
def entry_lock(cache_dir, name: str, *, exclusive: bool):
    """Hold the lock of cache entry `name` for the `with` block: exclusive for a writer, shared for a reader. Waits for a conflicting holder. Does nothing
    when there is no cache directory yet (nothing to protect) or the lock file cannot be opened."""
    fd = None
    if cache_dir and fcntl is not None and Path(cache_dir).is_dir() and name and "/" not in name:
        try:
            fd = os.open(lock_path(cache_dir, name), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except OSError:
            if fd is not None:
                os.close(fd)
            fd = None
    try:
        yield fd is not None
    finally:
        if fd is not None:
            os.close(fd)                              # closing the descriptor releases the lock
