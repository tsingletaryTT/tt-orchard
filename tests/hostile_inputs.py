# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One corpus of hostile inputs for every reader of a file or record that something else wrote.

Two users:
  * `tests/test_safeio.py` feeds the corpus to `orchard/safeio.py` and proves each guard.
  * `tests/test_hostile_readers.py` feeds the same corpus to every reader registered in `READERS`
    (the registry `READERS` is at the end of this module) and states what each reader must do.

An item builds one hostile thing at a path the caller chooses: a file with hostile content, a directory, a FIFO, a link to a
regular file, a link to /dev/zero, a file nobody may read. Items that need a sequence of actions between a check and an
open (a directory swapped for a link, a file that grows while it is read) are not in the corpus. They are tests of the
reader's open path and live in test_safeio.py, where the hook that makes the swap is installed.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------------------------------------- byte payloads

BIG_STRING_BYTES = 50 * 1024 * 1024          # the size of the large string item


def _deep(n: int) -> bytes:
    return b"[" * n + b"]" * n


def _deep_object(n: int) -> bytes:
    return b'{"a":' * n + b"1" + b"}" * n


# Name -> content. Each is something a writer could leave in a JSON file, accidentally or on purpose. Some are valid JSON of an
# unexpected type or value ([], {}, null, 0, -1, true, "x", ""), some are not JSON at all, some are JSON that Python accepts and
# a reader must still refuse (NaN, Infinity, 1e999, a repeated key, a lone surrogate).
BYTES_CORPUS: dict[str, bytes] = {
    "empty_array": b"[]",
    "empty_object": b"{}",
    "null": b"null",
    "empty_string": b'""',
    "string_x": b'"x"',
    "zero": b"0",
    "minus_one": b"-1",
    "huge_exponent": b"1e999",
    "nan": b"NaN",
    "infinity": b"Infinity",
    "negative_infinity": b"-Infinity",
    "nan_in_object": b'{"a": NaN, "b": [Infinity]}',
    "true": b"true",
    "nest_1000": _deep(1000),
    "nest_1000_object": _deep_object(1000),
    "nest_100000": _deep(100_000),
    "duplicate_keys": b'{"a": 1, "a": 2}',
    "lone_surrogate": b'"\\ud800"',
    "lone_surrogate_key": b'{"\\udc00": 1}',
    "invalid_utf8": b'"\xff\xfe"',
    "bom": b"\xef\xbb\xbf{}",
    "truncated": b'{"a": [1, 2',
    "empty": b"",
    "whitespace_only": b" \n\t",
    "trailing_garbage": b"{} x",
    "huge_integer": b"1" + b"0" * 5000,
    "null_byte_in_string": b'{"a": "\x00"}',
    "negative_zero_count": b'{"count": -0.0}',
    "float_count": b'{"count": 5.0}',
    "negative_count": b'{"count": -3}',
    "string_count": b'{"count": "5"}',
    "bool_count": b'{"count": true}',
}


def big_string() -> bytes:
    """A JSON string of 50 MB. Built on demand; held by no module-level name."""
    return b'"' + b"a" * (BIG_STRING_BYTES - 2) + b'"'


# --------------------------------------------------------------------------------------------------------------- items

@dataclass(frozen=True)
class Item:
    """One hostile thing. `build(dest)` creates it at the path `dest` (its parent exists) and may leave other files beside it."""
    name: str
    build: Callable[[Path], None]
    kind: str                        # "bytes" (a regular file with these bytes), "special" (not a regular file), "denied", "big"
    content: bytes | None = None     # for kind "bytes"


def _write_bytes(data: bytes) -> Callable[[Path], None]:
    def build(dest: Path) -> None:
        dest.write_bytes(data)
    return build


def _directory(dest: Path) -> None:
    dest.mkdir()
    (dest / "inner.json").write_text("{}")


def _fifo(dest: Path) -> None:
    os.mkfifo(dest)


def _link_to_regular(dest: Path) -> None:
    target = dest.with_name(dest.name + ".real")
    target.write_text("{}")
    dest.symlink_to(target.name)


def _link_to_dev_zero(dest: Path) -> None:
    dest.symlink_to("/dev/zero")


def _dangling_link(dest: Path) -> None:
    dest.symlink_to("no-such-file-anywhere")


def _link_loop(dest: Path) -> None:
    dest.symlink_to(dest.name)


def _denied(dest: Path) -> None:
    dest.write_text("{}")
    dest.chmod(0)


def _big_string_file(dest: Path) -> None:
    with open(dest, "wb") as f:
        f.write(b'"')
        chunk = b"a" * (1 << 20)
        for _ in range(BIG_STRING_BYTES // (1 << 20) - 1):
            f.write(chunk)
        f.write(b'"')


def running_as_root() -> bool:
    return os.geteuid() == 0


ITEMS: list[Item] = (
    [Item(name, _write_bytes(data), "bytes", data) for name, data in BYTES_CORPUS.items()]
    + [
        Item("directory", _directory, "special"),
        Item("fifo", _fifo, "special"),
        Item("link_to_regular_file", _link_to_regular, "special"),
        Item("link_to_dev_zero", _link_to_dev_zero, "special"),
        Item("dangling_link", _dangling_link, "special"),
        Item("link_loop", _link_loop, "special"),
        Item("permission_denied", _denied, "denied"),
        Item("big_string_50mb", _big_string_file, "big"),
    ]
)

# Items that can make a reader read without end or hold a lot of memory. They run in a child process with an address-space limit.
HEAVY = frozenset({"link_to_dev_zero", "big_string_50mb"})


def make_item(item: Item, dest: Path) -> None:
    """Build `item` at `dest`. A permission-denied item is built as an ordinary file when the process is root, since root is
    never denied; the test that needs the denial skips then."""
    item.build(dest)


def restore_permissions(root: Path) -> None:
    """Give every file under `root` read and write permission again so a temporary directory can be removed."""
    for path in Path(root).rglob("*"):
        try:
            if not path.is_symlink() and not stat.S_ISFIFO(path.lstat().st_mode):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass


# -------------------------------------------------------------------------------------------------------- child process

REPO = Path(__file__).resolve().parent.parent


def run_limited(code: str, *, timeout: float = 60, address_space: int = 2 << 30):
    """Run Python `code` in a child with an address-space limit and a timeout (the repo's rule for a hostile-input test: a reader
    that reads /dev/zero to the end would exhaust memory, and the limit turns that into a MemoryError in the child alone).
    The repository root is on the child's sys.path. Returns the CompletedProcess; a timeout raises subprocess.TimeoutExpired."""
    import subprocess
    import sys
    prologue = (f"import resource, sys; resource.setrlimit(resource.RLIMIT_AS, ({address_space}, {address_space})); "
                f"sys.path.insert(0, {str(REPO)!r})\n")
    return subprocess.run([sys.executable, "-c", prologue + code], capture_output=True, text=True, timeout=timeout)
