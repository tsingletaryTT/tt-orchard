# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Safe reading and writing of files that something else produced.

Why this module exists. Review after review found narrower versions of the same defects in code that reads a file or a
record another program wrote: JSON of the wrong type, NaN and Infinity, negative or fractional counts, repeated keys, a
file or directory swapped for a link between the check and the open, a path opened by name after it was checked, a read
with no size limit, a safetensors header whose tensors overlap or run past the data, an atomic write that can be raced,
and a limit that was hit without the result saying so. This module holds one reviewed version of each guard. A reader
that needs one of them calls it, and `tests/test_safeio.py` plus the hostile-input registry in `tests/hostile_inputs.py`
prove each guard. `docs/safe-reading.md` says when to use which function and lists what none of them cover.

Standard library only. The supervisor must keep running with no third-party package.

What every function here guarantees, and what none of them does:

  * A failure to read or to validate is a `SafeReadError` (or a subclass) with a short `reason`. A raw `OSError`,
    `ValueError`, `UnicodeDecodeError`, `RecursionError` or `json.JSONDecodeError` never leaves a read function, and the
    message never holds the content of the file (a file can hold a secret, and a message is logged).
  * Nothing here is a security boundary against a user who already controls the process or the directory in which the
    file lives. The guards cover files written by other programs, by an earlier run, by a model or by a download.
  * A path given as a plain string is opened by name. `O_NOFOLLOW` refuses a link only in the last component. Only the
    functions that take an open directory descriptor (`anchor`, `open_beneath`, `pinned_root`, `scan_beneath`) check
    every component. A path an operator typed cannot be anchored, so it keeps the last-component guard only.

`append_line` is included: the run ledger (`orchard/ledger.py`) and the helper ledger (`orchard/telegraph.py`) both append
one JSON line to a file under an exclusive lock, set aside or heal a torn last line, and cut a failed append back. The
two differ in who owns the lock (a lock file held for the life of the process, against a lock taken around each call) and
in what happens to a torn tail (set aside in a sidecar file, against a leading newline). `append_line` covers the part
they share, the single write with its rollback, and leaves the lock and the torn-tail recovery policy to the caller.
"""
from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import stat
import struct
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

__all__ = [
    "SafeReadError", "ReadFailed", "PathRefused", "NotRegular", "TooLarge", "ChangedWhileReading", "NotText", "NotJson",
    "WrongType", "BadValue", "BadHeader", "IncompleteScan", "AppendBroken",
    "read_bytes", "read_json", "read_jsonl", "parse_json_bytes", "open_beneath", "Pin", "pinned_root", "scan_beneath", "ScanEntry",
    "ScanResult", "atomic_write", "append_line", "finite_number", "whole_number", "bounded_str", "safetensors_header",
    "SafetensorsHeader", "TensorEntry", "DTYPE_BITS",
]

# Default limits. Each is a keyword argument of the function that uses it.
MAX_JSON_DEPTH = 64             # containers nested deeper than this are refused
MAX_INT_DIGITS = 100            # a JSON integer with more digits than this is refused
MAX_SCAN_ENTRIES = 100_000      # directory entries taken in one scan, across every directory in it
MAX_SCAN_DEPTH = 8              # directory levels entered by one scan
MAX_SHAPE_RANK = 64
MAX_NUMEL = (1 << 63) - 1       # the largest element count or single dimension a header may claim
MAX_TENSOR_NAME = 4096
MAX_METADATA_STR = 1 << 20
_READ_CHUNK = 1 << 16

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
# O_NONBLOCK keeps the open of a FIFO with no writer from blocking forever; it has no effect on a regular file.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC
_LINK_ERRNOS = (errno.ELOOP, errno.EMLINK, errno.ENOTDIR)


# ---------------------------------------------------------------------------------------------------------- errors

class SafeReadError(Exception):
    """A file or value was refused. `reason` is a short plain sentence that never includes file content."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ReadFailed(SafeReadError):
    """The operating system refused the open or the read (missing file, permission denied, I/O error)."""


class PathRefused(SafeReadError):
    """The path was refused before or while it was opened: a link, a `..`, an empty or absolute part, a null byte."""


class NotRegular(SafeReadError):
    """The opened descriptor is not a regular file (a directory, a FIFO, a device, a socket)."""


class TooLarge(SafeReadError):
    """The file or the header is larger than the limit the caller set."""


class ChangedWhileReading(SafeReadError):
    """The file did not hold the size it had when the read began, so the bytes read are not one version of it."""


class NotText(SafeReadError):
    """The bytes are not valid UTF-8."""


class NotJson(SafeReadError):
    """The text is not acceptable JSON: syntax, NaN or Infinity, a repeated key, a number out of range, nesting too deep."""


class WrongType(SafeReadError):
    """The JSON is valid but its top-level value is not the type the caller expected."""


class BadValue(SafeReadError):
    """A value failed `finite_number`, `whole_number` or `bounded_str`."""


class BadHeader(SafeReadError):
    """A safetensors header is malformed."""


class IncompleteScan(SafeReadError):
    """A scan hit a limit or could not read an entry, and the caller asked for a complete result."""


class AppendBroken(OSError):
    """An append failed and the file could not be cut back to its earlier length. Writing more would corrupt the file."""


def _errname(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno, "error") if exc.errno else "error"


def _os_failure(exc: OSError, what: str = "the file") -> SafeReadError:
    """Map an OSError to a SafeReadError. Only the errno name is kept; the message of an OSError can hold a path."""
    if exc.errno in _LINK_ERRNOS:
        return PathRefused(f"{what} is a link or has a part that is not a directory ({_errname(exc)})")
    return ReadFailed(f"{what} cannot be read ({_errname(exc)})")


# ------------------------------------------------------------------------------------------------ number and string guards

def finite_number(x, *, lo=None, hi=None, what: str = "value"):
    """`x` itself when it is an int or a float, finite, and within `lo` and `hi` (both inclusive, both optional).

    A bool is not a number. A string is not a number. NaN and the infinities are refused. An int of any size passes the
    finite test and is compared with `lo` and `hi` as an int, so a huge int never overflows a float conversion. Raises
    BadValue. Does not convert: 5 stays 5 and 5.0 stays 5.0."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise BadValue(f"{what} must be a number")
    if isinstance(x, float) and not math.isfinite(x):
        raise BadValue(f"{what} must be a finite number")
    if lo is not None and x < lo:
        raise BadValue(f"{what} must be at least {lo}")
    if hi is not None and x > hi:
        raise BadValue(f"{what} must be at most {hi}")
    return x


def whole_number(x, *, lo: int = 0, hi: int | None = None, what: str = "value") -> int:
    """`x` when it is an int (not a bool) with `lo <= x <= hi`. `lo` defaults to 0, so a count cannot be negative.

    A float is refused even when it has a whole value (5.0 is not accepted: a count that arrives as a float came
    through arithmetic or a different writer, and the caller should hear about it). A string is refused. Raises BadValue."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise BadValue(f"{what} must be a whole number")
    if lo is not None and x < lo:
        raise BadValue(f"{what} must be at least {lo}")
    if hi is not None and x > hi:
        raise BadValue(f"{what} must be at most {hi}")
    return x


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in text)


def bounded_str(x, *, max_len: int, min_len: int = 0, what: str = "value") -> str:
    """`x` when it is a str of `min_len` to `max_len` characters with no lone surrogate (which cannot be encoded as
    UTF-8 and breaks a later write or print). It does not strip, normalise or check the characters otherwise. Raises BadValue."""
    if not isinstance(x, str):
        raise BadValue(f"{what} must be text")
    if len(x) > max_len:
        raise BadValue(f"{what} is longer than {max_len} characters")
    if len(x) < min_len:
        raise BadValue(f"{what} is shorter than {min_len} characters")
    if _has_surrogate(x):
        raise BadValue(f"{what} holds a character that is not valid Unicode text")
    return x


# --------------------------------------------------------------------------------------------------------------- paths

def _check_parts(parts) -> tuple[str, ...]:
    """The parts as a tuple of plain names. A `..`, `.`, empty, absolute (contains `/`) or null-holding part is refused."""
    if isinstance(parts, (str, bytes)):
        raise PathRefused("parts must be a sequence of names; a single string is refused")
    out = []
    for part in parts:
        if not isinstance(part, str) or part in ("", ".", "..") or "/" in part or "\0" in part:
            raise PathRefused("a path part is empty, absolute, `.`, `..` or holds a null byte")
        out.append(part)
    return tuple(out)


def open_beneath(root_fd: int, parts: Sequence[str], *, file_flags: int | None = None) -> int:
    """A new descriptor for `parts` beneath the open directory `root_fd`, or a SafeReadError.

    Every directory on the way is opened with `O_DIRECTORY | O_NOFOLLOW` relative to the descriptor before it, so a link
    anywhere in `parts` is refused and `root_fd` is never left by name. A `..`, `.`, empty, absolute or null-holding part is refused
    before any open. With `file_flags=None` the last part is opened as a directory too, and `parts == ()` returns a copy of
    `root_fd`. With `file_flags` set (for example `os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK`) the last part is opened
    with those flags, `O_NOFOLLOW` is added if missing, and `parts` must not be empty.

    The caller closes the returned descriptor. No descriptor is left open on failure. This does not check that the result
    is a regular file: `fstat` the descriptor. It cannot stop another process that owns a directory on the way from
    renaming things inside it; it only guarantees that no link is followed."""
    names = _check_parts(parts)
    if file_flags is not None and not names:
        raise PathRefused("a file needs at least one path part")
    dirs = names[:-1] if file_flags is not None else names
    cur, owned, result = root_fd, False, -1
    try:
        for part in dirs:
            nxt = os.open(part, _DIR_FLAGS, dir_fd=cur)
            prev, prev_owned = cur, owned
            cur, owned = nxt, True
            if prev_owned:
                os.close(prev)
        if file_flags is not None:
            result = os.open(names[-1], file_flags | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=cur)
        else:
            result = cur if owned else os.dup(root_fd)
        return result
    except OSError as exc:
        raise _os_failure(exc, "a path part") from None
    except ValueError:                      # UnicodeEncodeError: a lone surrogate cannot be passed to the operating system
        raise PathRefused("a path part cannot be encoded for the operating system") from None
    finally:
        # A directory opened on the way is closed unless it is the descriptor being returned.
        if owned and cur != result:
            try:
                os.close(cur)
            except OSError:
                pass


class Pin:
    """A directory held open by descriptor. Everything opened through it is opened beneath it, so a directory swapped for a
    link after the pin was taken is refused and never followed. Use `pinned_root` to make one."""

    def __init__(self, fd: int, name: str):
        self.fd = fd
        self.name = name                 # for messages only; never used to open anything

    def close(self) -> None:
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def __enter__(self) -> "Pin":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def child(self, parts: Sequence[str]) -> "Pin":
        """A pin on the directory `parts` beneath this one (see `open_beneath`). The caller closes it."""
        return Pin(open_beneath(self.fd, parts), os.path.join(self.name, *parts))

    def read_json(self, parts: Sequence[str], **kw):
        """`read_json` of the file `parts` beneath this pin."""
        names = _check_parts(parts)
        if not names:
            raise PathRefused("a file needs at least one path part")
        return read_json(os.path.join(*names), anchor=self.fd, **kw)


def _root_name(path) -> str:
    """The root path as text without trailing separators. A path that ends in a separator or a dot component makes the kernel follow a final
    link that O_NOFOLLOW refuses, so separators are stripped (the filesystem root stays `/`) and a final `.` or `..` after another
    component is refused. Bytes and path-like objects that give bytes are decoded, so a pin's name is always text."""
    try:
        raw = os.fspath(path)
    except TypeError:
        raise PathRefused("the root directory path is not usable") from None
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if raw == "" or "\0" in raw:
        raise PathRefused("the root directory path is empty or holds a null byte")
    name = raw.rstrip("/") or "/"
    if "/" in name and name.rsplit("/", 1)[-1] in (".", ".."):
        raise PathRefused("the root directory path ends in a dot component")
    return name


@contextmanager
def pinned_root(path) -> Iterator[Pin]:
    """Open the directory `path` without following a link in its last component and hold it for the `with` block.

    A link in an earlier component of `path` is followed (a path typed by a person cannot be anchored); the directory
    reached is then the anchor for everything opened through it. Raises SafeReadError when it cannot be opened."""
    name = _root_name(path)
    try:
        fd = os.open(name, _DIR_FLAGS)
    except OSError as exc:
        raise _os_failure(exc, "the root directory") from None
    except ValueError:
        raise PathRefused("the root directory path is not usable") from None
    pin = Pin(fd, name)
    try:
        yield pin
    finally:
        pin.close()


# --------------------------------------------------------------------------------------------------------- reading

def _read_fd(fd: int, max_bytes: int, what: str) -> bytes:
    """All the bytes of the regular file behind `fd`, at most `max_bytes` of them. Never asks for more than max_bytes + 1."""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise NotRegular(f"{what} is not a regular file")
        if st.st_size > max_bytes:
            raise TooLarge(f"{what} is larger than {max_bytes} bytes")
        chunks, total = [], 0
        while total <= max_bytes:
            chunk = os.read(fd, min(_READ_CHUNK, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            raise TooLarge(f"{what} is larger than {max_bytes} bytes")
        after = os.fstat(fd)
        if total != st.st_size or after.st_size != st.st_size or after.st_mtime_ns != st.st_mtime_ns:
            raise ChangedWhileReading(f"{what} changed while it was read")
        return b"".join(chunks)
    except OSError as exc:
        raise _os_failure(exc, what) from None


def _open_for_read(path, anchor: int | None) -> int:
    try:
        raw = os.fspath(path)
    except TypeError:
        raise PathRefused("the path is not a path") from None
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if "\0" in raw or raw == "":
        raise PathRefused("the path is empty or holds a null byte")
    if raw.endswith("/") or raw.rsplit("/", 1)[-1] in (".", ".."):
        # A path that ends in a separator or a dot names a directory, and the kernel follows a final link there that O_NOFOLLOW would refuse.
        raise PathRefused("the path ends in a separator or a dot")
    if anchor is None:
        try:
            return os.open(raw, _FILE_FLAGS)
        except OSError as exc:
            raise _os_failure(exc) from None
        except ValueError:                  # UnicodeEncodeError: a lone surrogate cannot be passed to the operating system
            raise PathRefused("the path cannot be encoded for the operating system") from None
    if raw.startswith("/"):
        raise PathRefused("a path given with an anchor must be relative")
    return open_beneath(anchor, raw.split("/"), file_flags=_FILE_FLAGS)


def read_bytes(path, *, max_bytes: int, anchor: int | None = None) -> bytes:
    """The content of the regular file at `path`, never more than `max_bytes` bytes.

    Guarantees: the file is opened with `O_NOFOLLOW | O_NONBLOCK` (a link in the last component is refused and a FIFO
    does not block the open); the same descriptor is `fstat`-ed and must be a regular file; at most `max_bytes + 1`
    bytes are ever requested from it, and a larger file is `TooLarge` and never a truncated result; the size and
    modification time are compared before and after the read, and a file that grew, shrank or was rewritten during the
    read is `ChangedWhileReading`.

    With `anchor` (an open directory descriptor) the path must be relative and is opened component by component beneath
    it with `O_NOFOLLOW` on each (`open_beneath`). Without `anchor` only the last component is protected from a link.

    Does not guarantee: that the bytes are one atomic snapshot (a writer that rewrites the same bytes count in place is
    not seen), or anything about the content. Raises only SafeReadError."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a whole number, zero or more")
    fd = _open_for_read(path, anchor)
    try:
        return _read_fd(fd, max_bytes, "the file")
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------------------------- JSON

class _Refuse(Exception):
    """Raised inside a json hook; turned into NotJson with its own reason."""


def _no_constant(name: str):
    raise _Refuse(f"{name} is not allowed in JSON")


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _Refuse("an object repeats a key")
        out[key] = value
    return out


def _parse_float(text: str):
    value = float(text)
    if not math.isfinite(value):
        raise _Refuse("a number is too large for a float")
    return value


def _make_parse_int(max_digits: int) -> Callable[[str], int]:
    def parse_int(text: str) -> int:
        if len(text.lstrip("-")) > max_digits:       # the limit counts digits; the sign is not one
            raise _Refuse("an integer has too many digits")
        return int(text)
    return parse_int


def _check_tree(value, max_depth: int) -> None:
    """Refuse nesting deeper than `max_depth` and any lone surrogate in a string or key. Iterative, so it cannot recurse."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, str):
            if _has_surrogate(item):
                raise _Refuse("a string holds a character that is not valid Unicode text")
        elif isinstance(item, dict):
            if depth > max_depth:
                raise _Refuse("nesting is too deep")
            for key, child in item.items():
                if _has_surrogate(key):
                    raise _Refuse("a key holds a character that is not valid Unicode text")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            if depth > max_depth:
                raise _Refuse("nesting is too deep")
            stack.extend((child, depth + 1) for child in item)


def parse_json_bytes(data: bytes | str, *, max_bytes: int, max_depth: int = MAX_JSON_DEPTH,
                     max_int_digits: int = MAX_INT_DIGITS, expect=None):
    """Parse JSON that is already in memory (a line of a ledger, an HTTP body) with every guard `read_json` applies.

    Refuses: more than `max_bytes` bytes (`TooLarge`); bytes that are not UTF-8, and a leading byte order mark
    (`NotText`, `NotJson`); NaN, Infinity and -Infinity; a number that does not fit a float (1e999); an integer with more than
    `max_int_digits` digits; an object that repeats a key; a lone surrogate in a string or key; nesting deeper than
    `max_depth`; text after the value. `expect` is a type or a tuple of types the top-level value must have (a bool does not
    count as an int); a mismatch is `WrongType`. Raises only SafeReadError, and no message holds the input."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a whole number, zero or more")
    # The size is judged before any copy is made: a str of more characters than the cap has more bytes than the cap, and a buffer is
    # measured where it is. The byte length of the encoded text is checked again below.
    if isinstance(data, str):
        if len(data) > max_bytes:
            raise TooLarge(f"the JSON is larger than {max_bytes} bytes")
        try:
            raw = data.encode("utf-8")
        except UnicodeEncodeError:
            raise NotText("the text holds a character that cannot be encoded as UTF-8") from None
    elif isinstance(data, (bytes, bytearray, memoryview)):
        if (data.nbytes if isinstance(data, memoryview) else len(data)) > max_bytes:
            raise TooLarge(f"the JSON is larger than {max_bytes} bytes")
        raw = bytes(data)
    else:
        raise TypeError("data must be bytes or str")
    if len(raw) > max_bytes:
        raise TooLarge(f"the JSON is larger than {max_bytes} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise NotText("the bytes are not valid UTF-8") from None
    if text.startswith("﻿"):
        raise NotJson("the text starts with a byte order mark")
    try:
        value = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_duplicates,
                           parse_float=_parse_float, parse_int=_make_parse_int(max_int_digits))
        _check_tree(value, max_depth)
    except _Refuse as exc:
        raise NotJson(str(exc)) from None
    except RecursionError:
        raise NotJson("nesting is too deep") from None
    except json.JSONDecodeError as exc:
        # `exc.msg` is the parser's own wording and holds no input; the position helps a person find the fault.
        raise NotJson(f"not valid JSON ({exc.msg} at character {exc.pos})") from None
    except ValueError:
        raise NotJson("not valid JSON") from None
    if expect is not None:
        types = expect if isinstance(expect, tuple) else (expect,)
        # bool is a subclass of int, so an expected int must not accept true or false
        if isinstance(value, bool) and bool not in types or not isinstance(value, types):
            raise WrongType("the top-level JSON value has the wrong type")
    return value


def read_json(path, *, max_bytes: int, anchor: int | None = None, max_depth: int = MAX_JSON_DEPTH,
              max_int_digits: int = MAX_INT_DIGITS, expect=None):
    """The JSON value in the regular file at `path`, or a SafeReadError.

    Reads with `read_bytes` (`O_NOFOLLOW`, the same descriptor `fstat`-ed as a regular file, at most `max_bytes` read,
    a larger file refused and never parsed in part, a file that changed during the read refused) and parses with
    `parse_json_bytes` (strict UTF-8, no NaN or Infinity, no repeated key, no out-of-range number, no lone surrogate,
    bounded nesting and integer length, optional `expect` type). `anchor` is an open directory descriptor; see `read_bytes`.

    A valid `{}`, `[]`, `null`, `0` or `""` is returned as it is: whether an empty value is acceptable is the caller's
    decision, and `expect` makes the type part of it. Every failure is one SafeReadError subclass with a reason that
    does not hold the file's content."""
    return parse_json_bytes(read_bytes(path, max_bytes=max_bytes, anchor=anchor), max_bytes=max_bytes,
                            max_depth=max_depth, max_int_digits=max_int_digits, expect=expect)


def read_jsonl(path, *, max_bytes: int, anchor: int | None = None, max_depth: int = MAX_JSON_DEPTH,
               max_int_digits: int = MAX_INT_DIGITS, expect=None) -> list:
    """The JSON values on the lines of the regular file at `path`, in order. Blank lines are skipped.

    The file is read with `read_bytes` (so a FIFO, a link, a device and a file over `max_bytes` are refused before anything is
    parsed) and split on the byte `\\n` only (`str.splitlines` also cuts at U+2028, U+2029 and U+0085, which a writer leaves
    raw inside a JSON string). Each line is parsed with `parse_json_bytes`, so every guard of `read_json` applies to every
    line, and `expect` is the type each line's value must have. The first bad line fails the whole read with a SafeReadError
    whose reason names the line number and never holds its content."""
    data = read_bytes(path, max_bytes=max_bytes, anchor=anchor)
    rows = []
    for number, line in enumerate(data.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(parse_json_bytes(line, max_bytes=max_bytes, max_depth=max_depth, max_int_digits=max_int_digits,
                                         expect=expect))
        except SafeReadError as exc:
            raise type(exc)(f"line {number}: {exc.reason}") from None
    return rows


# ------------------------------------------------------------------------------------------------------ directory scan

@dataclass(frozen=True)
class ScanEntry:
    parts: tuple[str, ...]      # the path beneath the pinned root, one name per part
    kind: str                   # "file", "dir", "link" or "other"; a link is listed and never entered or read


@dataclass
class ScanResult:
    """What a scan found and whether it is the whole answer. Check `incomplete`, or call `require_complete`."""
    entries: list[ScanEntry] = field(default_factory=list)      # empty when a `visit` callback was given
    seen: int = 0                                               # entries taken from directories, counted against the budget
    entries_cut: bool = False                                   # the entry budget ran out: later entries were never listed
    depth_cut: bool = False                                     # a directory was not entered because of the depth limit
    unreadable: list[tuple[tuple[str, ...], str]] = field(default_factory=list)   # (parts, reason) of a directory not listed

    @property
    def incomplete(self) -> bool:
        return self.entries_cut or self.depth_cut or bool(self.unreadable)

    def require_complete(self) -> "ScanResult":
        """self, or IncompleteScan naming what was cut."""
        if self.incomplete:
            cuts = [name for name, hit in (("the entry limit", self.entries_cut), ("the depth limit", self.depth_cut),
                                           ("an unreadable directory", bool(self.unreadable))) if hit]
            raise IncompleteScan("the scan is incomplete: it hit " + " and ".join(cuts))
        return self


def scan_beneath(pin: Pin, visit: Callable[[ScanEntry], None] | None = None, *, max_entries: int = MAX_SCAN_ENTRIES,
                 max_depth: int = MAX_SCAN_DEPTH) -> ScanResult:
    """Walk the directory tree under `pin` by descriptor, depth first, one entry at a time.

    Guarantees: directories are listed with `os.scandir` on a descriptor, which streams and never reads a whole directory
    into memory; each subdirectory is opened beneath its parent with `O_DIRECTORY | O_NOFOLLOW`, so one swapped for a
    link after it was listed is refused (it appears in `unreadable`, never followed); a link is reported with kind `link`
    and never entered; every entry taken counts against a single `max_entries` budget shared by all directories, and when
    it is spent the walk stops with `entries_cut` set; a directory found at level `max_depth` (the pinned root is level 1)
    is not entered and sets `depth_cut`; a directory that cannot be listed or opened is recorded in `unreadable` with an
    errno name. The result is `incomplete` when any of those happened, and `require_complete()` turns that into an error.

    `visit` is called with each entry when given, so a large tree is never held in a list; otherwise the entries are collected
    (at most `max_entries` of them). The order within a directory is the order the file system gives. Does not open or
    read any file: a caller that reads an entry opens it with `pin.read_json(parts)` or `open_beneath`, which repeat the link
    refusal at the moment of the read. A file created after its directory was listed is not seen."""
    for name, value in (("max_entries", max_entries), ("max_depth", max_depth)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a whole number, zero or more")
    result = ScanResult()
    if max_depth < 1:
        result.depth_cut = True
        return result
    budget = max_entries
    # Each frame is (descriptor, iterator, parts, level, owned). The root frame's descriptor belongs to the pin.
    try:
        root_fd = os.dup(pin.fd)
    except OSError as exc:
        result.unreadable.append(((), _errname(exc)))
        return result
    stack: list[tuple[int, object, tuple[str, ...], int]] = []
    try:
        try:
            stack.append((root_fd, os.scandir(root_fd), (), 1))
        except OSError as exc:
            os.close(root_fd)
            result.unreadable.append(((), _errname(exc)))
            return result
        while stack:
            fd, it, parts, level = stack[-1]
            try:
                entry = next(it)
            except StopIteration:
                it.close()
                os.close(fd)
                stack.pop()
                continue
            except OSError as exc:
                result.unreadable.append((parts, _errname(exc)))
                it.close()
                os.close(fd)
                stack.pop()
                continue
            if budget <= 0:
                result.entries_cut = True
                break
            budget -= 1
            result.seen += 1
            name = entry.name
            try:
                if entry.is_symlink():
                    kind = "link"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "dir"
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    kind = "other"
            except OSError as exc:
                kind = "other"
                result.unreadable.append((parts + (name,), _errname(exc)))     # its type is unknown, so the scan is incomplete
            item = ScanEntry(parts + (name,), kind)
            if visit is not None:
                visit(item)
            else:
                result.entries.append(item)
            if kind != "dir":
                continue
            if level >= max_depth:
                result.depth_cut = True
                continue
            try:
                child_fd = os.open(name, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                result.unreadable.append((item.parts, _errname(exc)))
                continue
            try:
                stack.append((child_fd, os.scandir(child_fd), item.parts, level + 1))
            except OSError as exc:
                os.close(child_fd)
                result.unreadable.append((item.parts, _errname(exc)))
    finally:
        for fd, it, _parts, _level in stack:
            try:
                it.close()
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
    return result


# ------------------------------------------------------------------------------------------------------------- writing

def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError(errno.EIO, "a write made no progress")
        view = view[n:]


def _fsync_dir(directory: str) -> None:
    fd = os.open(directory or ".", os.O_RDONLY | os.O_DIRECTORY | _O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, data: bytes | str, *, mode: int = 0o600, replace: bool = True) -> None:
    """Write `data` to `path` so that a reader sees the old content or the whole new content and never a part.

    Steps: a temporary file with a random name is created exclusively (`O_EXCL | O_NOFOLLOW`) in the same directory, so the
    final rename stays on one file system; the data is written in full (a short write is retried); the file gets exactly
    `mode` (the process umask does not weaken or widen it); the file is `fsync`-ed; then `os.replace` swaps it in
    (`replace=True`) or `os.link` plus unlink publishes it only if `path` does not exist (`replace=False`, raising
    FileExistsError and leaving the existing file untouched); finally the directory is `fsync`-ed so the rename survives a crash.
    The temporary file is removed on every failure path, including KeyboardInterrupt.

    Does not guarantee: that two writers cannot both succeed (the last rename wins; use a lock if only one may); that a
    link at `path` is followed (a link at `path` is replaced, never written through); durability on a file system that
    ignores `fsync`. If the final directory `fsync` fails after the rename, the new content is in place and OSError is
    raised, so the caller knows durability is unconfirmed. Raises OSError (and FileExistsError for a taken `path` with
    `replace=False`) because it is a write; the message is the operating system's."""
    raw = os.fspath(path)
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not isinstance(data, (bytes, bytearray, memoryview, str)):
        raise TypeError("data must be bytes or str")
    payload = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    directory, base = os.path.split(raw)
    if not base:
        raise IsADirectoryError(errno.EISDIR, "no file name in the path")
    tmp = os.path.join(directory, f".{base}.{uuid.uuid4().hex}.tmp")
    fd = -1
    created = False                 # only a name this call created is ever removed
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC, mode)
        created = True
        os.fchmod(fd, mode)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if replace:
            os.replace(tmp, raw)
        else:
            os.link(tmp, raw)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        # After os.replace the temporary name is gone; after os.link it is a second name that must go.
        if created:
            try:
                os.unlink(tmp)
            except OSError:
                pass                # a stranded temporary name is harmless and must not hide the real error
    _fsync_dir(directory)


def append_line(fd: int, line: bytes | str, *, heal_torn: bool = False, max_bytes: int | None = None) -> int:
    """Append one line (the text plus a newline) to the file behind `fd` with one write, fsync it, and cut it back on failure.

    The caller owns the descriptor and the exclusive lock: open the file with `O_RDWR | O_APPEND` (read access is needed
    to look at the last byte), hold an `flock` for the whole call, and close the descriptor afterwards. This function
    checks that the descriptor is a regular file opened with `O_APPEND`, and refuses a line that contains a newline (a
    line holding one would be two records) or is longer than `max_bytes`.

    With `heal_torn=True` a last line that has no newline (a write cut off by a crash) gets a newline before the new line,
    so the new record cannot be glued to the fragment. The fragment stays in the file; a caller that wants it set aside
    (`orchard/ledger.py`) does that before calling, under the same lock, and leaves `heal_torn` off.

    If the write or the `fsync` fails, the file is cut back to its length before the call with `ftruncate`, and the error
    is raised. If that cut fails as well, AppendBroken is raised and the caller must stop writing. Returns the number of
    bytes appended. Does not take a lock, does not create the file and does not recover a torn tail on its own."""
    if not isinstance(line, (bytes, bytearray, str)):
        raise TypeError("line must be bytes or str")
    payload = line.encode("utf-8") if isinstance(line, str) else bytes(line)
    if b"\n" in payload:
        raise ValueError("a line must not contain a newline")
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError(f"a line must be at most {max_bytes} bytes")
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("the descriptor is not a regular file")
    if not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND:
        raise ValueError("the descriptor must be opened with O_APPEND")
    start = st.st_size
    prefix = b""
    if heal_torn and start > 0 and os.pread(fd, 1, start - 1) != b"\n":
        prefix = b"\n"
    data = prefix + payload + b"\n"
    try:
        _write_all(fd, data)
        os.fsync(fd)
    except BaseException:
        try:
            os.ftruncate(fd, start)
        except OSError as cut:
            raise AppendBroken(cut.errno, "an append failed and could not be undone") from None
        raise
    return len(data)


# ---------------------------------------------------------------------------------------------------- safetensors

# Bits per element of each dtype the safetensors format defines. A dtype not listed is refused: when the format gains a
# dtype, add it here (and its test); do not loosen the check.
DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}
_ENTRY_KEYS = {"dtype", "shape", "data_offsets"}


@dataclass(frozen=True)
class TensorEntry:
    dtype: str
    shape: tuple[int, ...]
    start: int                  # byte offsets relative to the start of the data section
    end: int


@dataclass(frozen=True)
class SafetensorsHeader:
    entries: dict[str, TensorEntry]
    metadata: dict[str, str]
    data_start: int             # file offset where the data section begins (8 + header length)
    data_size: int              # bytes in the data section
    file_size: int


def _numel(shape: list[int], what: str) -> int:
    """The element count of a shape whose dimensions are already known to be whole numbers. A zero dimension makes the count
    zero whatever the other dimensions are; otherwise the running product must stay within MAX_NUMEL."""
    if any(d > MAX_NUMEL for d in shape):
        raise BadHeader(f"{what} has a dimension that is too large")
    if 0 in shape:
        return 0
    n = 1
    for d in shape:
        n *= d
        if n > MAX_NUMEL:
            raise BadHeader(f"{what} claims too many elements")
    return n


def _read_exact(fd: int, offset: int, size: int) -> bytes:
    chunks, have = [], 0
    while have < size:
        chunk = os.pread(fd, min(_READ_CHUNK, size - have), offset + have)
        if not chunk:
            break
        chunks.append(chunk)
        have += len(chunk)
    return b"".join(chunks)


def safetensors_header(path_or_fd, *, max_header_bytes: int, contiguous: bool = True) -> SafetensorsHeader:
    """The validated header of a safetensors file. Only the header is read; no tensor data is touched.

    The file is opened once (with `O_NOFOLLOW` when given a path; a descriptor the caller opened is used as it is and left
    open), `fstat`-ed as a regular file, and the 8-byte length and the JSON are read from that one descriptor with
    `pread`, so the size the checks use is the size of the file read. Checks:

      * the length N is at least 2 (an empty object), at most `max_header_bytes`, and 8 + N fits in the file;
      * the JSON passes every `parse_json_bytes` guard (no NaN, no repeated tensor name, strict UTF-8, bounded depth) and is an object;
      * `__metadata__`, when present, is an object whose keys and values are all strings;
      * each other entry is an object with exactly `dtype`, `shape` and `data_offsets`; `dtype` is in `DTYPE_BITS`; `shape` is a list
        of at most 64 whole numbers, none negative, with an element count no larger than 2**63 - 1; `data_offsets` is two whole
        numbers with `0 <= start <= end <= data size`; and `end - start` is exactly the element count times the dtype size
        (a sub-byte dtype must fill a whole number of bytes);
      * no two ranges overlap (an empty range may touch another), and with `contiguous=True` (the default) the ranges also
        tile the data section with no gap before, between or after them: bytes no tensor claims could hide data. Pass
        `contiguous=False` to allow gaps; overlap stays refused.

    Does not check: that the tensor bytes hold sensible values, that tensor names match any model, or that the file is not
    truncated beyond the offsets it declares (the offsets are checked against the file size, which is the truncation check).
    A file that changes between the `fstat` and the reads is not detected. Raises only SafeReadError."""
    if isinstance(max_header_bytes, bool) or not isinstance(max_header_bytes, int) or max_header_bytes < 2:
        raise ValueError("max_header_bytes must be a whole number of at least 2")
    owned = not isinstance(path_or_fd, int) or isinstance(path_or_fd, bool)
    fd = path_or_fd
    if owned:
        fd = _open_for_read(path_or_fd, None)
    try:
        return _read_safetensors(fd, max_header_bytes, contiguous)
    except OSError as exc:
        raise _os_failure(exc) from None
    finally:
        if owned:
            os.close(fd)


def _read_safetensors(fd: int, max_header_bytes: int, contiguous: bool) -> SafetensorsHeader:
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        raise NotRegular("the file is not a regular file")
    size = st.st_size
    lead = _read_exact(fd, 0, 8)
    if len(lead) != 8:
        raise BadHeader("the file is shorter than 8 bytes")
    (n,) = struct.unpack("<Q", lead)
    if n < 2:
        raise BadHeader("the header length is too small")
    if n > max_header_bytes:
        raise TooLarge(f"the header is larger than {max_header_bytes} bytes")
    if 8 + n > size:
        raise BadHeader("the header is cut short by the end of the file")
    blob = _read_exact(fd, 8, n)
    if len(blob) != n:
        raise BadHeader("the header is cut short")
    try:
        header = parse_json_bytes(blob, max_bytes=n, expect=dict)
    except SafeReadError as exc:
        raise BadHeader(f"the header is not acceptable JSON: {exc.reason}") from None
    data_size = size - 8 - n
    metadata: dict[str, str] = {}
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise BadHeader("__metadata__ is not an object of strings")
    if sum(len(k) + len(v) for k, v in meta.items()) > MAX_METADATA_STR:
        raise BadHeader("__metadata__ is too large")
    metadata.update(meta)
    entries: dict[str, TensorEntry] = {}
    for name, v in header.items():
        if name == "__metadata__":
            continue
        if len(name) > MAX_TENSOR_NAME:
            raise BadHeader("a tensor name is too long")
        if not isinstance(v, dict) or set(v) != _ENTRY_KEYS:
            raise BadHeader("a tensor entry must be an object with dtype, shape and data_offsets")
        dtype, shape, offsets = v["dtype"], v["shape"], v["data_offsets"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise BadHeader("a tensor has a dtype that is not known")
        if (not isinstance(shape, list) or len(shape) > MAX_SHAPE_RANK
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape)):
            raise BadHeader("a tensor shape must be a short list of non-negative whole numbers")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(isinstance(o, bool) or not isinstance(o, int) for o in offsets)):
            raise BadHeader("a tensor's data_offsets must be two whole numbers")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise BadHeader("a tensor's data_offsets do not lie inside the data section")
        bits = _numel(shape, "a tensor") * DTYPE_BITS[dtype]
        if bits % 8 != 0:
            raise BadHeader("a tensor of a packed dtype does not fill a whole number of bytes")
        if end - start != bits // 8:
            raise BadHeader("a tensor's byte span does not match its shape and dtype")
        entries[name] = TensorEntry(dtype, tuple(shape), start, end)
    cursor = 0
    for entry in sorted(entries.values(), key=lambda e: (e.start, e.end)):
        if entry.start < cursor:
            raise BadHeader("two tensors share bytes")
        if contiguous and entry.start > cursor:
            raise BadHeader("the data section has bytes that no tensor claims")
        cursor = max(cursor, entry.end)
    if contiguous and cursor != data_size:
        raise BadHeader("the data section has bytes after the last tensor that no tensor claims")
    return SafetensorsHeader(entries, metadata, 8 + n, data_size, size)
