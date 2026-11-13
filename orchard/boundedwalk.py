# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A lazy, bounded walk of a directory that an agent writes (issues 326 and 457).

The port directory of a full port and the evidence directory of a stage belong to agents. A runaway or hostile agent can leave millions of
entries or a very deep tree there. A walk that lists the whole tree first (`rglob`, `os.walk`, `sorted(...)`) makes the supervisor spend memory
and time on all of it before any limit applies. `Walk` asks the operating system for one entry at a time and raises `Limit` as soon as the entry
count or the depth passes its bound, so a directory of a million entries is refused after about `max_entries` reads.

The walk was first written for the stage 3 seed (`_SeedScan` in `orchard/supervisor.py`, PR 455). It lives here so that `full_port` and
`port_repair` can use it without importing the supervisor.
"""
from __future__ import annotations

import os

from orchard import safeio

PORT_ENTRIES = 1000     # a port directory scan stops after this many entries of any kind (files, directories, links): a file count alone
                        # would let a directory of a million subdirectories be read to the end before anything refused it
PORT_DEPTH = 8          # and at this many directory levels below the port directory
CACHE_DIR = "__pycache__"


class Limit(Exception):
    """A bound of the walk was passed; the text is the reason a caller records (for example "holds more than 1000 entries")."""


class Walk:
    """The entries of a directory, read lazily, depth first, through descriptors, with every bound enforced as the entries arrive.

    The root is opened once and held (`safeio.pinned_root`: a link standing where the root should be is refused), and every directory below it is
    opened beneath that descriptor with no link followed and listed from its own descriptor. A directory renamed or replaced by a link after it
    was classified is therefore still the directory that is listed, and a link in its place is refused when the walk reaches it. Nothing is listed or
    sorted first: an entry is asked of the operating system only when the caller wants the next one, so a directory of millions of entries is
    refused after about `max_entries` reads and holds one descriptor per level (at most `max_depth` + 2). A link is reported and never followed.
    With `skip_caches` (the default) a cache directory (`__pycache__`) is reported nowhere and never entered; a link with that name is still
    reported as a link. Yields `(parts, kind, identity)` with `parts` the names from the root, `kind` one of `file`, `link` or `other` (a device,
    a pipe, a socket) and `identity` the `(st_dev, st_ino)` of a file as listed (None otherwise); directories are walked and not yielded.
    `read` opens a listed file beneath the same root descriptor and refuses a file that is no longer the file that was listed. `seen` is the
    number of entries read so far. The order is the operating system's: callers that need a stable order sort what they kept. Close the
    iterator (`contextlib.closing`) to release the descriptors. Opening the root raises `safeio.SafeReadError` (missing, or a link)."""

    def __init__(self, root, max_entries: int = PORT_ENTRIES, max_depth: int = PORT_DEPTH, *, anchor=None, parts=(), skip_caches: bool = True):
        """With `anchor` (a `safeio.Pin` on the run directory) and `parts` (the names from it to the directory), the root is opened beneath the
        anchor with no link followed at any name, so a link standing where `stages/2` should be is refused (review of PR 455, round 4). Without
        them `root` is opened by path, and only its last name is protected."""
        self.root, self.max_entries, self.max_depth, self.seen = os.fspath(root), max_entries, max_depth, 0
        self.anchor, self.anchor_parts = anchor, tuple(parts)
        self.skip_caches = skip_caches
        self._pin = None

    def read(self, parts, identity, max_bytes: int) -> bytes:
        """The bytes of the file `parts` as it was listed (identity `(st_dev, st_ino)`), at most `max_bytes`; only while the iterator is open."""
        if self._pin is None:
            raise safeio.PathRefused("the directory is not open")
        return safeio.read_beneath(self._pin.fd, parts, max_bytes=max_bytes, expect_identity=identity)

    def __iter__(self):
        with (self.anchor.child(self.anchor_parts) if self.anchor is not None else safeio.pinned_root(self.root)) as pin:
            self._pin = pin
            try:
                stack = [((), 0)]
                while stack:
                    rel, depth = stack.pop()
                    with safeio.list_beneath(pin.fd, rel) as listing:
                        for entry in listing:
                            self.seen += 1
                            if self.seen > self.max_entries:
                                raise Limit(f"holds more than {self.max_entries} entries")
                            parts = rel + (entry.name,)
                            if entry.is_symlink():
                                yield parts, "link", None
                            elif entry.is_dir(follow_symlinks=False):
                                if self.skip_caches and entry.name == CACHE_DIR:
                                    continue
                                if depth + 1 > self.max_depth:
                                    raise Limit(f"is nested deeper than {self.max_depth} levels")
                                stack.append((parts, depth + 1))
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                yield parts, "file", (st.st_dev, st.st_ino)
                            else:
                                yield parts, "other", None
            finally:
                self._pin = None
