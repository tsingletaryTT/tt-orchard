# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Proofs for orchard/safeio.py. Each guard in the module has a test here that goes red when the guard is removed
(the mutation table is in the log entry for version 0.9.34)."""
from __future__ import annotations

import ast
import errno
import json
import os
import stat
import struct
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from orchard import safeio
from orchard.safeio import (BadHeader, BadValue, ChangedWhileReading, IncompleteScan, NotJson, NotRegular, NotText, PathRefused,
                            ReadFailed, SafeReadError, TooLarge, WrongType)
import hostile_inputs as hi

CAP = 1 << 20
PROFILE = dict(max_examples=60, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------------------------------------------ the module itself

def test_the_module_imports_only_the_standard_library():
    """The supervisor must run with no third-party package."""
    tree = ast.parse(Path(safeio.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add((node.module or "").split(".")[0])
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names


# ------------------------------------------------------------------------------------------------------------ read_json

VALID = {"empty_array": [], "empty_object": {}, "null": None, "empty_string": "", "string_x": "x", "zero": 0, "minus_one": -1,
         "true": True, "negative_zero_count": {"count": -0.0}, "float_count": {"count": 5.0}, "negative_count": {"count": -3},
         "string_count": {"count": "5"}, "bool_count": {"count": True}}

# What the reader must refuse and why. The values that are valid JSON (a wrong type, a negative count) are not here: telling
# those apart is the caller's job, with `expect`, `whole_number` and `finite_number`.
REFUSED = {
    "huge_exponent": NotJson, "nan": NotJson, "infinity": NotJson, "negative_infinity": NotJson, "nan_in_object": NotJson,
    "nest_1000": NotJson, "nest_1000_object": NotJson, "nest_100000": NotJson, "duplicate_keys": NotJson,
    "lone_surrogate": NotJson, "lone_surrogate_key": NotJson, "invalid_utf8": NotText, "bom": NotJson, "truncated": NotJson,
    "empty": NotJson, "whitespace_only": NotJson, "trailing_garbage": NotJson, "huge_integer": NotJson,
    "null_byte_in_string": NotJson,
}


def test_the_two_tables_cover_every_byte_item():
    assert set(VALID) | set(REFUSED) == set(hi.BYTES_CORPUS)


@pytest.mark.parametrize("name", sorted(VALID))
def test_valid_json_is_returned_as_it_is(tmp_path, name):
    got = safeio.read_json(write(tmp_path / "f.json", hi.BYTES_CORPUS[name]), max_bytes=CAP)
    assert got == VALID[name] and type(got) is type(VALID[name])


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_hostile_json_is_one_safe_error(tmp_path, name):
    with pytest.raises(REFUSED[name]):
        safeio.read_json(write(tmp_path / "f.json", hi.BYTES_CORPUS[name]), max_bytes=CAP)


def test_the_fifty_megabyte_string_is_refused_by_size_and_never_parsed(tmp_path):
    path = tmp_path / "big.json"
    hi.ITEMS[[i.name for i in hi.ITEMS].index("big_string_50mb")].build(path)
    with pytest.raises(TooLarge):
        safeio.read_json(path, max_bytes=CAP)


def test_an_error_message_never_holds_the_files_content(tmp_path):
    secret = "sk-this-is-a-secret-token"
    for payload in (f'{{"{secret}": NaN}}', f'{{"{secret}": 1, "{secret}": 2}}', f'{{"{secret}": [', f'"{secret}', f'["{secret}"] x',
                    f'{{"k": "{secret}"}}'.replace('"k"', '"\\ud800"')):
        path = write(tmp_path / "s.json", payload.encode())
        with pytest.raises(SafeReadError) as info:
            safeio.read_json(path, max_bytes=CAP, expect=list)
        assert secret not in str(info.value) and secret not in repr(info.value) and secret not in info.value.reason
    path = write(tmp_path / "s.json", f'{{"{secret}": 1}}'.encode())
    with pytest.raises(WrongType) as info:
        safeio.read_json(path, max_bytes=CAP, expect=list)
    assert secret not in str(info.value)


def test_a_file_of_exactly_the_cap_is_read_and_one_more_byte_is_not(tmp_path):
    body = b'"' + b"a" * 98 + b'"'
    assert len(body) == 100
    assert safeio.read_json(write(tmp_path / "f.json", body), max_bytes=100) == "a" * 98
    with pytest.raises(TooLarge):
        safeio.read_json(write(tmp_path / "g.json", body + b" "), max_bytes=100)


def test_a_file_larger_than_the_cap_is_refused_before_any_byte_is_read(tmp_path, monkeypatch):
    path = write(tmp_path / "big.json", b'"' + b"a" * 5000 + b'"')
    asked = []
    real = os.read
    monkeypatch.setattr(safeio.os, "read", lambda fd, n: (asked.append(n), real(fd, n))[1])
    with pytest.raises(TooLarge):
        safeio.read_json(path, max_bytes=100)
    assert asked == []


def test_the_read_never_asks_for_more_than_the_cap_plus_one(tmp_path, monkeypatch):
    """A file that grows while it is read (here: by the first read) must not be read to its end. The spy records what is asked
    of the descriptor, so a loop with no bound goes red."""
    path = write(tmp_path / "grow.json", b'"' + b"a" * 20 + b'"')
    asked, real_read = [], os.read

    def read(fd, n):
        asked.append(n)
        if len(asked) == 1:
            with open(path, "ab") as f:                       # the file is growing beneath the reader
                f.write(b"a" * 50_000)
        return real_read(fd, n)

    monkeypatch.setattr(safeio.os, "read", read)
    with pytest.raises(TooLarge):
        safeio.read_json(path, max_bytes=100)
    assert sum(asked) <= 101, asked


def test_a_file_that_changes_size_during_the_read_is_refused_even_within_the_cap(tmp_path, monkeypatch):
    path = write(tmp_path / "grow.json", b"12")
    real_read, state = os.read, {"n": 0}

    def read(fd, n):
        state["n"] += 1
        data = real_read(fd, n)
        if state["n"] == 1:
            with open(path, "ab") as f:
                f.write(b"3")                                  # "12" was read; the file now holds "123", also valid JSON
        return data

    monkeypatch.setattr(safeio.os, "read", read)
    with pytest.raises(ChangedWhileReading):
        safeio.read_json(path, max_bytes=CAP)


def test_a_file_rewritten_to_the_same_size_during_the_read_is_refused(tmp_path, monkeypatch):
    path = write(tmp_path / "f.json", b"[1]")
    real_read, state = os.read, {"n": 0}

    def read(fd, n):
        data = real_read(fd, n)
        state["n"] += 1
        if state["n"] == 1:
            st = os.stat(path)
            with open(path, "r+b") as f:
                f.write(b"[2]")
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        return data

    monkeypatch.setattr(safeio.os, "read", read)
    with pytest.raises(ChangedWhileReading):
        safeio.read_json(path, max_bytes=CAP)


def test_a_size_change_is_caught_even_when_the_modification_time_is_put_back(tmp_path, monkeypatch):
    path = write(tmp_path / "f.json", b"12")
    real_read, state = os.read, {"n": 0}

    def read(fd, n):
        data = real_read(fd, n)
        state["n"] += 1
        if state["n"] == 1:
            st = os.stat(path)
            with open(path, "ab") as f:
                f.write(b"3")
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        return data

    monkeypatch.setattr(safeio.os, "read", read)
    with pytest.raises(ChangedWhileReading):
        safeio.read_json(path, max_bytes=CAP)


def test_a_short_read_is_caught_even_when_the_file_looks_unchanged(tmp_path, monkeypatch):
    """The descriptor returns half the bytes and then end of file, as a file truncated and restored would."""
    path = write(tmp_path / "f.json", b'["abcdef"]')
    real_read, state = os.read, {"n": 0}

    def read(fd, n):
        state["n"] += 1
        data = real_read(fd, n)
        return data[:3] if state["n"] == 1 else b""

    monkeypatch.setattr(safeio.os, "read", read)
    with pytest.raises(ChangedWhileReading):
        safeio.read_json(path, max_bytes=CAP)


def test_a_read_error_is_a_read_failure(tmp_path, monkeypatch):
    path = write(tmp_path / "f.json", b"{}")
    monkeypatch.setattr(safeio.os, "read", lambda fd, n: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
    with pytest.raises(ReadFailed) as info:
        safeio.read_json(path, max_bytes=CAP)
    assert "EIO" in info.value.reason


def test_a_bytes_path_is_read(tmp_path):
    path = write(tmp_path / "f.json", b"[1]")
    assert safeio.read_json(os.fsencode(path), max_bytes=CAP) == [1]


def test_a_json_error_from_the_parser_is_one_safe_error_with_a_position(tmp_path, monkeypatch):
    with pytest.raises(NotJson, match="character"):
        safeio.parse_json_bytes(b"[1,", max_bytes=CAP)
    monkeypatch.setattr(safeio.json, "loads", lambda *a, **k: (_ for _ in ()).throw(ValueError("injected")))
    with pytest.raises(NotJson):
        safeio.parse_json_bytes(b"[1]", max_bytes=CAP)


def test_an_object_nest_is_read_at_the_limit_and_refused_one_deeper():
    ok = safeio.parse_json_bytes(hi._deep_object(64), max_bytes=CAP)
    assert isinstance(ok, dict)
    with pytest.raises(NotJson):
        safeio.parse_json_bytes(hi._deep_object(65), max_bytes=CAP)


def test_a_link_in_the_last_component_is_refused(tmp_path):
    (tmp_path / "real.json").write_text("{}")
    (tmp_path / "link.json").symlink_to("real.json")
    with pytest.raises(PathRefused):
        safeio.read_json(tmp_path / "link.json", max_bytes=CAP)


def test_a_directory_is_not_a_regular_file(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(NotRegular):
        safeio.read_json(tmp_path / "d", max_bytes=CAP)


def test_a_fifo_is_refused_and_does_not_block(tmp_path):
    os.mkfifo(tmp_path / "f")
    import signal

    def timed_out(signum, frame):
        raise TimeoutError("the open of a FIFO blocked")

    old = signal.signal(signal.SIGALRM, timed_out)        # a blocking open is interrupted so that it cannot hang the suite
    signal.alarm(5)
    try:
        with pytest.raises(NotRegular):
            safeio.read_json(tmp_path / "f", max_bytes=CAP)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_a_missing_file_is_a_read_failure(tmp_path):
    with pytest.raises(ReadFailed) as info:
        safeio.read_json(tmp_path / "nope.json", max_bytes=CAP)
    assert "ENOENT" in info.value.reason


@pytest.mark.skipif(hi.running_as_root(), reason="root is never denied")
def test_permission_denied_is_a_read_failure(tmp_path):
    path = write(tmp_path / "f.json", b"{}")
    path.chmod(0)
    try:
        with pytest.raises(ReadFailed) as info:
            safeio.read_json(path, max_bytes=CAP)
        assert "EACCES" in info.value.reason
    finally:
        path.chmod(0o600)


def test_a_link_to_dev_zero_is_refused_without_reading_it(tmp_path):
    """Run in a child with a 2 GiB address-space limit and a timeout: a reader that followed the link and read to the end
    would exhaust memory."""
    link = tmp_path / "zero.json"
    link.symlink_to("/dev/zero")
    done = hi.run_limited("from orchard import safeio\n"
                          f"try:\n    safeio.read_json({str(link)!r}, max_bytes=1 << 20)\n"
                          "except safeio.PathRefused as e:\n    print('refused')\n")
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "refused"


def test_dev_zero_opened_directly_is_not_a_regular_file():
    """In a limited child: a reader that did not check the file type would read the device without end."""
    done = hi.run_limited("from orchard import safeio\n"
                          "try:\n    safeio.read_json('/dev/zero', max_bytes=1 << 20)\n"
                          "except safeio.NotRegular:\n    print('not regular')\n")
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "not regular"


@pytest.mark.parametrize("bad", [-1, True, 1.5, "7", None])
def test_a_bad_cap_is_a_programming_error(tmp_path, bad):
    with pytest.raises(ValueError):
        safeio.read_json(write(tmp_path / "f.json", b"{}"), max_bytes=bad)


@pytest.mark.parametrize("bad", ["", "a\0b"])
def test_an_unusable_path_is_refused(bad):
    with pytest.raises(PathRefused):
        safeio.read_json(bad, max_bytes=CAP)


def test_a_path_that_is_not_a_path_is_refused():
    with pytest.raises(PathRefused):
        safeio.read_json(12, max_bytes=CAP)


def test_nesting_at_the_depth_limit_is_read_and_one_deeper_is_refused(tmp_path):
    assert safeio.read_json(write(tmp_path / "a.json", hi._deep(64)), max_bytes=CAP) is not None
    with pytest.raises(NotJson):
        safeio.read_json(write(tmp_path / "b.json", hi._deep(65)), max_bytes=CAP)
    assert safeio.read_json(write(tmp_path / "c.json", hi._deep(10)), max_bytes=CAP, max_depth=10) is not None
    with pytest.raises(NotJson):
        safeio.read_json(write(tmp_path / "d.json", hi._deep(11)), max_bytes=CAP, max_depth=10)


def test_a_repeated_key_in_a_nested_object_is_refused(tmp_path):
    with pytest.raises(NotJson):
        safeio.read_json(write(tmp_path / "f.json", b'{"a": [{"b": 1, "b": 2}]}'), max_bytes=CAP)
    assert safeio.read_json(write(tmp_path / "g.json", b'{"a": {"b": 1}, "b": {"b": 2}}'), max_bytes=CAP)["b"] == {"b": 2}


def test_integer_length_is_bounded(tmp_path):
    assert safeio.read_json(write(tmp_path / "a.json", b"1" * 100), max_bytes=CAP) == int("1" * 100)
    with pytest.raises(NotJson):
        safeio.read_json(write(tmp_path / "b.json", b"1" * 101), max_bytes=CAP)


@pytest.mark.parametrize("text,expect,ok", [
    (b"{}", dict, True), (b"{}", list, False), (b"[]", (dict, list), True), (b"1", int, True), (b"true", int, False),
    (b"true", bool, True), (b"1.5", int, False), (b"1", float, False), (b"null", dict, False), (b'"x"', str, True),
])
def test_expect_checks_the_top_level_type(tmp_path, text, expect, ok):
    path = write(tmp_path / "f.json", text)
    if ok:
        safeio.read_json(path, max_bytes=CAP, expect=expect)
    else:
        with pytest.raises(WrongType):
            safeio.read_json(path, max_bytes=CAP, expect=expect)


def test_parse_json_bytes_takes_text_and_enforces_the_cap():
    assert safeio.parse_json_bytes('{"a": 1}', max_bytes=100) == {"a": 1}
    with pytest.raises(TooLarge):
        safeio.parse_json_bytes(b'{"a": 1}', max_bytes=3)
    with pytest.raises(NotText):
        safeio.parse_json_bytes("\ud800", max_bytes=100)
    with pytest.raises(TypeError):
        safeio.parse_json_bytes(5, max_bytes=100)


# ------------------------------------------------------------------------------------------------------ the anchored read

def test_an_anchored_read_opens_beneath_the_descriptor(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.json").write_text('{"k": 1}')
    with safeio.pinned_root(tmp_path) as pin:
        assert safeio.read_json("a/f.json", anchor=pin.fd, max_bytes=CAP) == {"k": 1}
        assert pin.read_json(("a", "f.json"), max_bytes=CAP) == {"k": 1}


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x.json", "a/../x.json", "a//x.json", "./x.json"])
def test_an_anchored_read_refuses_absolute_and_dotted_paths(tmp_path, bad):
    with safeio.pinned_root(tmp_path) as pin, pytest.raises(PathRefused):
        safeio.read_json(bad, anchor=pin.fd, max_bytes=CAP)


def test_a_directory_swapped_for_a_link_after_the_pin_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.json").write_text('{"escaped": true}')
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.json").write_text("{}")
    with safeio.pinned_root(root) as pin:
        (root / "sub" / "f.json").unlink()
        (root / "sub").rmdir()
        (root / "sub").symlink_to(outside)                 # swapped after the pin and after discovery
        with pytest.raises(PathRefused):
            safeio.read_json("sub/f.json", anchor=pin.fd, max_bytes=CAP)
    # The unanchored read has no such guard for a link in an earlier component. The docs say so.
    assert safeio.read_json(root / "sub" / "f.json", max_bytes=CAP) == {"escaped": True}


def test_a_file_replaced_by_a_link_between_discovery_and_open_is_refused(tmp_path):
    (tmp_path / "secret.json").write_text('{"secret": true}')
    (tmp_path / "log").mkdir()
    target = tmp_path / "log" / "a.json"
    target.write_text("{}")
    with safeio.pinned_root(tmp_path / "log") as pin:
        found = [e for e in safeio.scan_beneath(pin).entries if e.kind == "file"]
        assert [e.parts for e in found] == [("a.json",)]
        target.unlink()
        target.symlink_to("../secret.json")
        with pytest.raises(PathRefused):
            pin.read_json(found[0].parts, max_bytes=CAP)


def test_no_descriptor_leaks_from_the_read_path(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "ok.json").write_text("{}")
    (tmp_path / "d" / "bad.json").write_text("[NaN]")
    (tmp_path / "d" / "link.json").symlink_to("ok.json")
    before = fd_count()
    with safeio.pinned_root(tmp_path) as pin:
        for rel, exc in (("d/ok.json", None), ("d/bad.json", NotJson), ("d/link.json", PathRefused), ("d/none.json", ReadFailed),
                         ("d", NotRegular), ("e/x.json", ReadFailed)):
            if exc is None:
                safeio.read_json(rel, anchor=pin.fd, max_bytes=CAP)
            else:
                with pytest.raises(exc):
                    safeio.read_json(rel, anchor=pin.fd, max_bytes=CAP)
    assert fd_count() == before


# ------------------------------------------------------------------------------------------------------- open_beneath

@pytest.mark.parametrize("parts", [("..",), ("a", ".."), ("",), (".",), ("/abs",), ("a/b",), ("a\0b",), ("a", "", "b"), (5,), ("..", "a")])
def test_open_beneath_refuses_bad_parts_before_opening_anything(tmp_path, parts):
    (tmp_path / "a").mkdir()
    before = fd_count()
    with safeio.pinned_root(tmp_path) as pin, pytest.raises(PathRefused):
        safeio.open_beneath(pin.fd, parts)
    assert fd_count() == before


@pytest.mark.parametrize("text", ["a/b", "ab", "a"])
def test_open_beneath_refuses_one_string_for_the_parts(tmp_path, text):
    (tmp_path / "a" / "b").mkdir(parents=True)
    with safeio.pinned_root(tmp_path) as pin, pytest.raises(PathRefused):
        safeio.open_beneath(pin.fd, text)


def test_open_beneath_closes_every_directory_it_passed_through(tmp_path):
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "f").write_text("x")
    with safeio.pinned_root(tmp_path) as pin:
        before = fd_count()
        os.close(safeio.open_beneath(pin.fd, ("a", "b", "c")))
        os.close(safeio.open_beneath(pin.fd, ("a", "b", "c", "f"), file_flags=os.O_RDONLY))
        assert fd_count() == before


def test_open_beneath_opens_nested_directories_and_returns_a_new_descriptor(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    with safeio.pinned_root(tmp_path) as pin:
        fd = safeio.open_beneath(pin.fd, ("a", "b"))
        try:
            assert fd != pin.fd and os.path.samestat(os.fstat(fd), os.stat(tmp_path / "a" / "b"))
        finally:
            os.close(fd)
        same = safeio.open_beneath(pin.fd, ())
        try:
            assert same != pin.fd and os.path.samestat(os.fstat(same), os.fstat(pin.fd))
        finally:
            os.close(same)


def test_open_beneath_refuses_a_link_at_any_level(tmp_path):
    (tmp_path / "real" / "inner").mkdir(parents=True)
    (tmp_path / "mid").symlink_to("real")
    (tmp_path / "real" / "ilink").symlink_to("inner")
    with safeio.pinned_root(tmp_path) as pin:
        for parts in (("mid",), ("mid", "inner"), ("real", "ilink")):
            before = fd_count()
            with pytest.raises(PathRefused):
                safeio.open_beneath(pin.fd, parts)
            assert fd_count() == before


def test_open_beneath_with_file_flags_opens_a_file_and_refuses_a_file_link(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("hi")
    (tmp_path / "d" / "l").symlink_to("f")
    flags = os.O_RDONLY | os.O_NONBLOCK
    with safeio.pinned_root(tmp_path) as pin:
        before = fd_count()
        fd = safeio.open_beneath(pin.fd, ("d", "f"), file_flags=flags)
        assert os.read(fd, 10) == b"hi"
        os.close(fd)
        with pytest.raises(PathRefused):
            safeio.open_beneath(pin.fd, ("d", "l"), file_flags=flags)      # O_NOFOLLOW is added when the caller leaves it out
        with pytest.raises(PathRefused):
            safeio.open_beneath(pin.fd, (), file_flags=flags)
        assert fd_count() == before


def test_open_beneath_does_not_find_a_directory_where_a_file_is_wanted(tmp_path):
    (tmp_path / "f").write_text("x")
    with safeio.pinned_root(tmp_path) as pin, pytest.raises(PathRefused):
        safeio.open_beneath(pin.fd, ("f", "g"))


# ------------------------------------------------------------------------------------------------------------ pinned_root

def test_pinned_root_refuses_a_link_and_a_missing_directory(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to("real")
    for path, exc in ((tmp_path / "link", PathRefused), (tmp_path / "none", ReadFailed), (tmp_path / "real" / "x" / "y", ReadFailed)):
        with pytest.raises(exc), safeio.pinned_root(path):
            pass


def test_pinned_root_refuses_a_file_and_bad_paths(tmp_path):
    (tmp_path / "f").write_text("x")
    with pytest.raises(SafeReadError), safeio.pinned_root(tmp_path / "f"):
        pass
    for bad in ("a\0b", 5):
        with pytest.raises(PathRefused), safeio.pinned_root(bad):
            pass


def test_pinned_root_closes_its_descriptor(tmp_path):
    before = fd_count()
    with safeio.pinned_root(tmp_path) as pin:
        fd = pin.fd
        os.fstat(fd)
    with pytest.raises(OSError):
        os.fstat(fd)
    assert fd_count() == before
    with pytest.raises(RuntimeError), safeio.pinned_root(tmp_path):
        raise RuntimeError("inside")
    assert fd_count() == before


def test_a_pin_follows_the_directory_it_opened_when_the_path_is_swapped(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.json").write_text('{"v": 1}')
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.json").write_text('{"v": 2}')
    with safeio.pinned_root(root) as pin:
        root.rename(tmp_path / "moved")
        root.symlink_to(other)
        assert pin.read_json(("a.json",), max_bytes=CAP) == {"v": 1}


def test_a_child_pin_is_a_pin_on_the_subdirectory(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "f.json").write_text("[1]")
    with safeio.pinned_root(tmp_path) as pin:
        with pin.child(("a", "b")) as sub:
            assert sub.read_json(("f.json",), max_bytes=CAP) == [1]
        with pytest.raises(PathRefused):
            pin.read_json((), max_bytes=CAP)


# ----------------------------------------------------------------------------------------------------------- scan_beneath

def tree(root: Path, dirs=(), files=()):
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("{}")


def test_a_complete_scan_lists_every_entry_with_its_kind(tmp_path):
    tree(tmp_path, dirs=["a/b"], files=["x.json", "a/y.json", "a/b/z.json"])
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "ln").symlink_to("a")
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin)
    got = {e.parts: e.kind for e in res.entries}
    assert got == {("x.json",): "file", ("a",): "dir", ("a", "y.json"): "file", ("a", "b"): "dir", ("a", "b", "z.json"): "file",
                   ("fifo",): "other", ("ln",): "link"}
    assert not res.incomplete and not res.entries_cut and not res.depth_cut and res.unreadable == []
    assert res.require_complete() is res


def test_a_link_to_a_directory_is_listed_and_never_entered(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.json").write_text("{}")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "ln").symlink_to("../outside")
    with safeio.pinned_root(tmp_path / "root") as pin:
        res = safeio.scan_beneath(pin)
    assert [(e.parts, e.kind) for e in res.entries] == [(("ln",), "link")]


def test_the_entry_budget_is_shared_by_all_directories_and_a_cut_is_reported(tmp_path):
    for d in "abc":
        tree(tmp_path, files=[f"{d}/f{i}" for i in range(5)])
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin, max_entries=7)
    assert res.seen == 7 and len(res.entries) == 7
    assert res.entries_cut and res.incomplete
    with pytest.raises(IncompleteScan):
        res.require_complete()


def test_a_scan_that_exactly_fits_the_budget_is_not_cut(tmp_path):
    tree(tmp_path, files=["a", "b", "c"])
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin, max_entries=3)
        assert res.seen == 3 and not res.entries_cut and not res.incomplete
        assert safeio.scan_beneath(pin, max_entries=2).entries_cut


def test_a_big_directory_is_read_only_up_to_the_budget(tmp_path):
    for i in range(3000):
        (tmp_path / f"f{i}").touch()
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin, max_entries=100)
    assert res.seen == 100 and res.entries_cut


def test_the_depth_limit_stops_the_walk_and_says_so(tmp_path):
    tree(tmp_path, dirs=["a/b/c"], files=["a/b/c/deep.json", "top.json"])
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin, max_depth=2)
        assert res.depth_cut and res.incomplete
        assert {e.parts for e in res.entries} == {("a",), ("a", "b"), ("top.json",)}
        full = safeio.scan_beneath(pin, max_depth=4)
        assert not full.depth_cut and ("a", "b", "c", "deep.json") in {e.parts for e in full.entries}
        assert safeio.scan_beneath(pin, max_depth=1).depth_cut


def test_a_depth_of_zero_lists_nothing_and_says_so(tmp_path):
    tree(tmp_path, files=["a", "b"])
    with safeio.pinned_root(tmp_path) as pin:
        res = safeio.scan_beneath(pin, max_depth=0)
    assert res.entries == [] and res.depth_cut and res.incomplete


def test_a_directory_that_fails_to_open_for_listing_or_during_listing_is_reported(tmp_path, monkeypatch):
    tree(tmp_path, dirs=["a", "b"], files=["a/f", "b/g", "top"])
    real = os.scandir
    calls = {"n": 0}

    class Breaks:
        def __init__(self, it):
            self.it, self.n = it, 0

        def __iter__(self):
            return self

        def __next__(self):
            self.n += 1
            if self.n > 1:
                raise OSError(errno.EIO, "injected")
            return next(self.it)

        def close(self):
            self.it.close()

    def scandir(fd):
        calls["n"] += 1
        if calls["n"] == 1:
            return real(fd)                                   # the root lists normally
        if calls["n"] == 2:
            raise OSError(errno.EIO, "injected")              # the first subdirectory cannot be listed
        return Breaks(real(fd))                               # the second fails part way through

    monkeypatch.setattr(safeio.os, "scandir", scandir)
    with safeio.pinned_root(tmp_path) as pin:
        before = fd_count()
        res = safeio.scan_beneath(pin)
        assert fd_count() == before
    assert len(res.unreadable) == 2 and {r for _, r in res.unreadable} == {"EIO"} and res.incomplete


def test_a_directory_at_the_last_level_with_no_subdirectory_is_not_a_cut(tmp_path):
    tree(tmp_path, files=["a/f.json"])
    with safeio.pinned_root(tmp_path) as pin:
        assert not safeio.scan_beneath(pin, max_depth=2).depth_cut


@pytest.mark.skipif(hi.running_as_root(), reason="root can list any directory")
def test_a_directory_that_cannot_be_listed_is_reported_and_the_rest_is_still_listed(tmp_path):
    tree(tmp_path, files=["ok/f.json", "locked/g.json"])
    (tmp_path / "locked").chmod(0)
    try:
        with safeio.pinned_root(tmp_path) as pin:
            res = safeio.scan_beneath(pin)
    finally:
        (tmp_path / "locked").chmod(0o700)
    assert [p for p, _ in res.unreadable] == [("locked",)] and res.unreadable[0][1] == "EACCES"
    assert ("ok", "f.json") in {e.parts for e in res.entries}
    assert res.incomplete and not res.entries_cut
    with pytest.raises(IncompleteScan):
        res.require_complete()


def test_a_directory_swapped_for_a_link_after_it_was_listed_is_not_entered(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.json").write_text("{}")
    root = tmp_path / "root"
    tree(root, dirs=["victim"], files=["victim/inner.json"])
    seen = []

    def visit(entry):
        seen.append(entry.parts)
        if entry.parts == ("victim",):                     # the swap happens after the directory was listed, before it is entered
            (root / "victim" / "inner.json").unlink()
            (root / "victim").rmdir()
            (root / "victim").symlink_to("../outside")

    with safeio.pinned_root(root) as pin:
        res = safeio.scan_beneath(pin, visit)
    assert ("victim", "secret.json") not in seen
    assert [p for p, _ in res.unreadable] == [("victim",)] and res.incomplete
    assert res.entries == []                                # with a callback nothing is collected


def test_a_second_scan_on_the_same_pin_sees_everything_again(tmp_path):
    tree(tmp_path, files=["a", "b/c"])
    with safeio.pinned_root(tmp_path) as pin:
        first = {e.parts for e in safeio.scan_beneath(pin).entries}
        assert first == {e.parts for e in safeio.scan_beneath(pin).entries} and len(first) == 3


def test_a_scan_leaks_no_descriptor_even_when_it_stops_early_or_the_callback_raises(tmp_path):
    tree(tmp_path, dirs=["a/b/c"], files=[f"a/b/c/f{i}" for i in range(10)])
    with safeio.pinned_root(tmp_path) as pin:
        before = fd_count()
        safeio.scan_beneath(pin, max_entries=5)
        safeio.scan_beneath(pin, max_depth=2)

        def boom(entry):
            if len(entry.parts) == 3:
                raise RuntimeError("stop")
        with pytest.raises(RuntimeError):
            safeio.scan_beneath(pin, boom)
        assert fd_count() == before


@pytest.mark.parametrize("kw", [{"max_entries": -1}, {"max_entries": True}, {"max_depth": 1.5}])
def test_scan_limits_must_be_whole_numbers(tmp_path, kw):
    with safeio.pinned_root(tmp_path) as pin, pytest.raises(ValueError):
        safeio.scan_beneath(pin, **kw)


# ---------------------------------------------------------------------------------------------------------- atomic_write

def stray(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


def test_atomic_write_writes_replaces_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "out.json"
    safeio.atomic_write(target, b"one")
    assert target.read_bytes() == b"one"
    safeio.atomic_write(target, "two")
    assert target.read_bytes() == b"two" and stray(tmp_path) == []


@pytest.mark.parametrize("mode,umask", [(0o600, 0o000), (0o644, 0o077), (0o640, 0o022)])
def test_atomic_write_sets_exactly_the_requested_mode(tmp_path, mode, umask):
    old = os.umask(umask)
    try:
        safeio.atomic_write(tmp_path / "f", b"x", mode=mode)
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(tmp_path / "f").st_mode) == mode


def test_atomic_write_without_replace_refuses_an_existing_file_and_keeps_it(tmp_path):
    target = tmp_path / "f"
    safeio.atomic_write(target, b"first", replace=False)
    with pytest.raises(FileExistsError):
        safeio.atomic_write(target, b"second", replace=False)
    assert target.read_bytes() == b"first" and stray(tmp_path) == []


def test_atomic_write_cleans_up_when_a_step_fails(tmp_path, monkeypatch):
    target = tmp_path / "f"
    safeio.atomic_write(target, b"old")
    for name in ("write", "fsync", "replace", "fchmod"):
        real = getattr(os, name)

        def boom(*a, _name=name, **k):
            raise OSError(errno.EIO, "injected")

        monkeypatch.setattr(safeio.os, name, boom)
        with pytest.raises(OSError):
            safeio.atomic_write(target, b"new")
        monkeypatch.setattr(safeio.os, name, real)
        assert target.read_bytes() == b"old" and stray(tmp_path) == [], name
    before = fd_count()
    monkeypatch.setattr(safeio.os, "write", lambda *a: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
    with pytest.raises(OSError):
        safeio.atomic_write(target, b"new")
    assert fd_count() == before                               # the descriptor of the temporary file is closed on failure


def test_atomic_write_never_reuses_an_existing_temporary_name(tmp_path, monkeypatch):
    import types
    fixed = types.SimpleNamespace(hex="abc123")
    monkeypatch.setattr(safeio.uuid, "uuid4", lambda: fixed)
    squatter = tmp_path / ".f.abc123.tmp"
    squatter.write_text("someone else's file")
    with pytest.raises(FileExistsError):
        safeio.atomic_write(tmp_path / "f", b"x")
    assert squatter.read_text() == "someone else's file" and not (tmp_path / "f").exists()


def test_atomic_write_cleans_up_when_the_link_step_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(safeio.os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "injected")))
    with pytest.raises(OSError):
        safeio.atomic_write(tmp_path / "f", b"x", replace=False)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_cleans_up_on_an_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(safeio.os, "fsync", lambda fd: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        safeio.atomic_write(tmp_path / "f", b"x")
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_onto_a_directory_fails_and_cleans_up(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(OSError):
        safeio.atomic_write(tmp_path / "d", b"x")
    assert stray(tmp_path) == [] and (tmp_path / "d").is_dir()


def test_atomic_write_replaces_a_link_and_never_writes_through_it(tmp_path):
    (tmp_path / "victim").write_text("keep me")
    (tmp_path / "ln").symlink_to("victim")
    safeio.atomic_write(tmp_path / "ln", b"new")
    assert (tmp_path / "victim").read_text() == "keep me"
    assert not (tmp_path / "ln").is_symlink() and (tmp_path / "ln").read_bytes() == b"new"


def test_atomic_write_survives_short_writes(tmp_path, monkeypatch):
    real = os.write
    monkeypatch.setattr(safeio.os, "write", lambda fd, data: real(fd, bytes(data[:3])))
    safeio.atomic_write(tmp_path / "f", b"0123456789")
    assert (tmp_path / "f").read_bytes() == b"0123456789"


def test_atomic_write_syncs_the_file_and_then_the_directory(tmp_path, monkeypatch):
    kinds, real = [], os.fsync

    def fsync(fd):
        kinds.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real(fd)

    monkeypatch.setattr(safeio.os, "fsync", fsync)
    safeio.atomic_write(tmp_path / "f", b"x")
    assert kinds == ["file", "dir"]


def test_atomic_write_puts_its_temp_file_beside_the_target(tmp_path, monkeypatch):
    opened, real = [], os.open
    monkeypatch.setattr(safeio.os, "open", lambda p, *a, **k: (opened.append(str(p)), real(p, *a, **k))[1])
    (tmp_path / "sub").mkdir()
    safeio.atomic_write(tmp_path / "sub" / "f", b"x")
    assert any(os.path.dirname(p) == str(tmp_path / "sub") and p.endswith(".tmp") for p in opened)


def test_atomic_write_rejects_other_data_types_and_a_directory_path(tmp_path):
    with pytest.raises(TypeError):
        safeio.atomic_write(tmp_path / "f", 5)
    with pytest.raises(OSError):
        safeio.atomic_write(str(tmp_path) + "/", b"x")


# ------------------------------------------------------------------------------------------------------------ append_line

def open_append(path: Path) -> int:
    return os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)


def test_append_line_adds_one_line_and_returns_the_bytes_written(tmp_path):
    fd = open_append(tmp_path / "l")
    try:
        assert safeio.append_line(fd, '{"a": 1}') == 9
        assert safeio.append_line(fd, b'{"b": 2}') == 9
    finally:
        os.close(fd)
    assert (tmp_path / "l").read_bytes() == b'{"a": 1}\n{"b": 2}\n'


def test_append_line_refuses_a_newline_inside_a_line_and_a_line_over_the_cap(tmp_path):
    fd = open_append(tmp_path / "l")
    try:
        with pytest.raises(ValueError):
            safeio.append_line(fd, "a\nb")
        with pytest.raises(ValueError):
            safeio.append_line(fd, "x" * 11, max_bytes=10)
        safeio.append_line(fd, "x" * 10, max_bytes=10)
        with pytest.raises(TypeError):
            safeio.append_line(fd, 5)
    finally:
        os.close(fd)
    assert (tmp_path / "l").read_bytes() == b"x" * 10 + b"\n"


def test_append_line_needs_a_regular_file_opened_for_append(tmp_path):
    fd = os.open(tmp_path / "l", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with pytest.raises(ValueError):
            safeio.append_line(fd, "x")
    finally:
        os.close(fd)
    dfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError):
            safeio.append_line(dfd, "x")
    finally:
        os.close(dfd)


def test_append_line_refuses_a_pipe_even_with_append_set(tmp_path):
    import fcntl
    r, w = os.pipe()
    try:
        fcntl.fcntl(w, fcntl.F_SETFL, fcntl.fcntl(w, fcntl.F_GETFL) | os.O_APPEND)
        with pytest.raises(ValueError):
            safeio.append_line(w, "x")
        os.set_blocking(r, False)
        with pytest.raises(BlockingIOError):
            os.read(r, 10)                                    # nothing was written
    finally:
        os.close(r)
        os.close(w)


def test_append_line_heals_a_torn_tail_only_when_asked(tmp_path):
    path = tmp_path / "l"
    path.write_bytes(b'{"a": 1}\n{"b"')
    fd = open_append(path)
    try:
        safeio.append_line(fd, '{"c": 3}')
        assert path.read_bytes() == b'{"a": 1}\n{"b"{"c": 3}\n'
        path.write_bytes(b'{"a": 1}\n{"b"')
        safeio.append_line(fd, '{"c": 3}', heal_torn=True)
        assert path.read_bytes() == b'{"a": 1}\n{"b"\n{"c": 3}\n'
        safeio.append_line(fd, '{"d": 4}', heal_torn=True)           # a whole last line gets no extra newline
        assert path.read_bytes().endswith(b'{"c": 3}\n{"d": 4}\n')
    finally:
        os.close(fd)


def test_append_line_cuts_the_file_back_when_the_sync_fails(tmp_path, monkeypatch):
    path = tmp_path / "l"
    path.write_bytes(b'{"a": 1}\n')
    fd = open_append(path)
    try:
        monkeypatch.setattr(safeio.os, "fsync", lambda f: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
        with pytest.raises(OSError):
            safeio.append_line(fd, '{"b": 2}', heal_torn=True)
    finally:
        os.close(fd)
    assert path.read_bytes() == b'{"a": 1}\n'


def test_append_line_cuts_back_a_part_written_line(tmp_path, monkeypatch):
    path = tmp_path / "l"
    path.write_bytes(b"A\n")
    fd = open_append(path)
    real, calls = os.write, []

    def write(f, data):
        calls.append(1)
        if len(calls) == 1:
            return real(f, bytes(data[:2]))                          # half the line lands
        raise OSError(errno.ENOSPC, "injected")

    try:
        monkeypatch.setattr(safeio.os, "write", write)
        with pytest.raises(OSError):
            safeio.append_line(fd, "BBBBBB")
    finally:
        os.close(fd)
    assert path.read_bytes() == b"A\n"


def test_append_line_says_so_when_it_cannot_cut_back(tmp_path, monkeypatch):
    fd = open_append(tmp_path / "l")
    try:
        monkeypatch.setattr(safeio.os, "fsync", lambda f: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
        monkeypatch.setattr(safeio.os, "ftruncate", lambda f, n: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
        with pytest.raises(safeio.AppendBroken):
            safeio.append_line(fd, "x")
    finally:
        os.close(fd)


def test_append_line_retries_a_short_write(tmp_path, monkeypatch):
    fd = open_append(tmp_path / "l")
    real = os.write
    monkeypatch.setattr(safeio.os, "write", lambda f, data: real(f, bytes(data[:2])))
    try:
        safeio.append_line(fd, "abcdef")
    finally:
        os.close(fd)
    assert (tmp_path / "l").read_bytes() == b"abcdef\n"


# ------------------------------------------------------------------------------------------------ the three value guards

@pytest.mark.parametrize("x,kw,ok", [
    (1, {}, True), (1.5, {}, True), (-3, {}, True), (0, {}, True), (-0.0, {}, True), (10 ** 400, {}, True),
    (True, {}, False), (False, {}, False), ("1", {}, False), (None, {}, False), ([1], {}, False),
    (float("nan"), {}, False), (float("inf"), {}, False), (float("-inf"), {}, False),
    (5, {"lo": 5}, True), (4, {"lo": 5}, False), (5, {"hi": 5}, True), (6, {"hi": 5}, False), (0.5, {"lo": 0, "hi": 1}, True),
    (10 ** 400, {"hi": 10 ** 500}, True), (10 ** 400, {"hi": 10}, False),
])
def test_finite_number(x, kw, ok):
    if ok:
        got = safeio.finite_number(x, **kw)
        assert got == x and type(got) is type(x)
    else:
        with pytest.raises(BadValue):
            safeio.finite_number(x, **kw)


@pytest.mark.parametrize("x,kw,ok", [
    (0, {}, True), (7, {}, True), (-1, {}, False), (5.0, {}, False), (True, {}, False), ("5", {}, False), (None, {}, False),
    (float("nan"), {}, False), (3, {"lo": 3}, True), (2, {"lo": 3}, False), (-2, {"lo": -5}, True), (9, {"hi": 9}, True),
    (10, {"hi": 9}, False), (10 ** 30, {}, True), (1, {"lo": None}, True),
])
def test_whole_number(x, kw, ok):
    if ok:
        got = safeio.whole_number(x, **kw)
        assert got == x and type(got) is int
    else:
        with pytest.raises(BadValue):
            safeio.whole_number(x, **kw)


@pytest.mark.parametrize("x,kw,ok", [
    ("abc", {"max_len": 3}, True), ("abcd", {"max_len": 3}, False), ("", {"max_len": 3}, True), ("", {"max_len": 3, "min_len": 1}, False),
    ("ab", {"max_len": 3, "min_len": 3}, False), (b"abc", {"max_len": 5}, False), (None, {"max_len": 5}, False), (5, {"max_len": 5}, False),
    ("a\ud800", {"max_len": 5}, False), ("\u00e9\u4e2d", {"max_len": 2}, True),
])
def test_bounded_str(x, kw, ok):
    if ok:
        assert safeio.bounded_str(x, **kw) == x
    else:
        with pytest.raises(BadValue):
            safeio.bounded_str(x, **kw)


@settings(**PROFILE)
@given(st.one_of(st.integers(), st.floats(allow_nan=True, allow_infinity=True), st.booleans(), st.text(max_size=5), st.none()))
def test_finite_number_accepts_only_real_finite_numbers(x):
    try:
        got = safeio.finite_number(x)
    except BadValue:
        assert isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and x != x or x in (float("inf"), float("-inf")))
    else:
        assert not isinstance(x, bool) and isinstance(x, (int, float)) and got == x


@settings(**PROFILE)
@given(st.one_of(st.integers(), st.floats(allow_nan=True, allow_infinity=True), st.booleans(), st.text(max_size=5), st.none()),
       st.integers(min_value=-5, max_value=5))
def test_whole_number_accepts_only_ints_at_or_above_lo(x, lo):
    try:
        got = safeio.whole_number(x, lo=lo)
    except BadValue:
        assert not (type(x) is int and x >= lo)
    else:
        assert type(x) is int and x >= lo and got == x


# -------------------------------------------------------------------------------------------------- the JSON properties

def has_bad_float(value) -> bool:
    if isinstance(value, float):
        return value != value or value in (float("inf"), float("-inf"))
    if isinstance(value, dict):
        return any(has_bad_float(v) for v in value.values())
    if isinstance(value, list):
        return any(has_bad_float(v) for v in value)
    return False


def depth_of(value) -> int:
    best, stack = 0, [(value, 1)]
    while stack:
        item, d = stack.pop()
        if isinstance(item, dict):
            best = max(best, d)
            stack.extend((v, d + 1) for v in item.values())
        elif isinstance(item, list):
            best = max(best, d)
            stack.extend((v, d + 1) for v in item)
    return best


JSON_VALUES = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(min_value=-10 ** 30, max_value=10 ** 30),
              st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=8)),
    lambda inner: st.one_of(st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=4), inner, max_size=4)), max_leaves=20)


@settings(**PROFILE)
@given(JSON_VALUES)
def test_any_json_value_is_refused_or_comes_back_equal_and_finite(value):
    text = json.dumps(value, allow_nan=True)
    try:
        got = safeio.parse_json_bytes(text, max_bytes=1 << 20)
    except NotText:
        assert any(0xD800 <= ord(c) <= 0xDFFF for c in text) or "\\ud" in text.lower()
    except NotJson:
        pass
    else:
        assert not has_bad_float(got) and depth_of(got) <= 64
        assert got == value
    assert not (has_bad_float(value) and "got" in locals())


@settings(**PROFILE)
@given(st.binary(max_size=200))
def test_arbitrary_bytes_never_raise_anything_but_a_safe_error(data):
    try:
        safeio.parse_json_bytes(data, max_bytes=1 << 10)
    except SafeReadError:
        pass


@settings(**PROFILE)
@given(st.text(alphabet=st.sampled_from(list('[]{}:,"0123456789.eE+-NaInfty \n')), max_size=60))
def test_json_shaped_noise_never_raises_anything_but_a_safe_error(text):
    try:
        got = safeio.parse_json_bytes(text, max_bytes=1 << 10)
    except SafeReadError:
        return
    assert not has_bad_float(got)


@settings(**PROFILE)
@given(st.integers(min_value=0, max_value=300), st.sampled_from(["list", "dict"]))
def test_nesting_is_accepted_up_to_the_limit_exactly(n, kind):
    text = ("[" * n + "]" * n) if kind == "list" else ('{"a":' * n + "1" + "}" * n)
    if n == 0:
        text = "1" if kind == "dict" else "[]"
    try:
        got = safeio.parse_json_bytes(text, max_bytes=1 << 20)
    except NotJson:
        assert n > 64
    else:
        assert depth_of(got) <= 64 and (n <= 64 or n == 0)


# ------------------------------------------------------------------------------------------------------ safetensors

def build_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]] | None = None, *, header=None, data=None,
                      length=None, metadata=None, pad_to=None) -> Path:
    """Write a safetensors file. `tensors` are (name, dtype, shape, raw bytes) laid out back to back; `header` overrides the JSON
    object; `data` overrides the data section; `length` overrides the 8-byte length."""
    entries, blob, pos = {}, b"", 0
    for name, dtype, shape, raw in (tensors or []):
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + len(raw)]}
        blob += raw
        pos += len(raw)
    if metadata is not None:
        entries["__metadata__"] = metadata
    obj = entries if header is None else header
    text = (json.dumps(obj) if not isinstance(obj, (bytes, str)) else obj)
    text = text.encode() if isinstance(text, str) else text
    if pad_to:
        text = text + b" " * (pad_to - len(text))
    n = len(text) if length is None else length
    path.write_bytes(struct.pack("<Q", n) + text + (blob if data is None else data))
    return path


def ok_tensors():
    return [("a", "F32", [2, 3], b"\0" * 24), ("b", "BF16", [4], b"\0" * 8), ("c", "U8", [0, 5], b"")]


def test_a_valid_header_is_returned_with_offsets_relative_to_the_data(tmp_path):
    path = build_safetensors(tmp_path / "m.safetensors", ok_tensors(), metadata={"format": "pt"})
    h = safeio.safetensors_header(path, max_header_bytes=1 << 20)
    assert h.entries["a"] == safeio.TensorEntry("F32", (2, 3), 0, 24)
    assert h.entries["b"] == safeio.TensorEntry("BF16", (4,), 24, 32)
    assert h.entries["c"].end - h.entries["c"].start == 0
    assert h.metadata == {"format": "pt"} and h.data_size == 32 and h.file_size == h.data_start + 32


def test_a_file_descriptor_is_used_as_given_and_left_open(tmp_path):
    path = build_safetensors(tmp_path / "m.safetensors", ok_tensors())
    fd = os.open(path, os.O_RDONLY)
    try:
        assert safeio.safetensors_header(fd, max_header_bytes=1 << 20).entries["a"].end == 24
        os.fstat(fd)
    finally:
        os.close(fd)


def test_an_empty_header_object_is_valid(tmp_path):
    h = safeio.safetensors_header(build_safetensors(tmp_path / "e", header={}), max_header_bytes=100)
    assert h.entries == {} and h.data_size == 0


def entry(dtype="F32", shape=(1,), offsets=(0, 4), **extra):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets), **extra}


# (header, size of the data section). Each header has one fault, and the data section is sized so that no other check would refuse it:
# without the guard named by the case, the header would be accepted.
BAD_HEADERS = {
    "array_header": ([], 0),
    "metadata_not_object": ({"__metadata__": []}, 0),
    "metadata_value_not_string": ({"__metadata__": {"a": 1}}, 0),
    "entry_not_object": ({"a": []}, 0),
    "entry_missing_dtype": ({"a": {"shape": [1], "data_offsets": [0, 4]}}, 4),
    "entry_missing_shape": ({"a": {"dtype": "F32", "data_offsets": [0, 4]}}, 4),
    "entry_missing_offsets": ({"a": {"dtype": "F32", "shape": [1]}}, 4),
    "entry_extra_key": ({"a": entry(x=1)}, 4),
    "unknown_dtype": ({"a": entry(dtype="F33")}, 4),
    "dtype_not_text": ({"a": entry(dtype=7)}, 4),
    "shape_not_list": ({"a": {"dtype": "F32", "shape": 1, "data_offsets": [0, 4]}}, 4),
    "negative_dims_with_a_positive_product": ({"a": entry(shape=(-1, -1))}, 4),
    "float_dim": ({"a": entry(shape=(1.0,))}, 4),
    "bool_dim": ({"a": entry(shape=(True,))}, 4),
    "string_dim": ({"a": entry(shape=("1",))}, 4),
    "rank_65": ({"a": entry(shape=[1] * 65)}, 4),
    "dimension_too_large_beside_a_zero": ({"a": entry(dtype="U8", shape=(0, 2 ** 70), offsets=(0, 0))}, 0),
    "three_offsets": ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4, 4]}}, 4),
    "one_offset": ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4]}}, 4),
    "float_offsets": ({"a": entry(offsets=(0.0, 4.0))}, 4),
    "bool_offsets": ({"a": entry(dtype="U8", offsets=(False, True))}, 1),
    "span_too_small": ({"a": entry(shape=(2,), offsets=(0, 4))}, 4),
    "span_too_large": ({"a": entry(shape=(1,), offsets=(0, 8))}, 8),
    "span_for_an_empty_shape": ({"a": entry(shape=(0,), offsets=(0, 4))}, 4),
    "packed_dtype_odd_count": ({"a": entry(dtype="F4", shape=(3,), offsets=(0, 1))}, 1),
    "offsets_past_the_data": ({"a": entry(offsets=(0, 4))}, 3),
    "start_after_end": ({"a": entry(dtype="U8", shape=(0,), offsets=(2, 1))}, 4),
}


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("name", sorted(BAD_HEADERS))
def test_a_malformed_header_is_refused(tmp_path, name, contiguous):
    header, size = BAD_HEADERS[name]
    with pytest.raises(BadHeader):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=header, data=b"\0" * size), max_header_bytes=1 << 20,
                                  contiguous=contiguous)


def test_the_good_version_of_each_bad_header_is_accepted(tmp_path):
    """The bad cases are one fault away from this: it shows that each case differs from a valid header by one fault."""
    assert safeio.safetensors_header(build_safetensors(tmp_path / "m", header={"a": entry()}, data=b"\0" * 4), max_header_bytes=1 << 20)


def test_a_zero_dimension_decides_the_count_whatever_the_other_dimensions_are(tmp_path):
    for shape in ((0, 2 ** 40), (2 ** 40, 2 ** 40, 0), (0,), (3, 0, 5)):
        hdr = {"a": entry(dtype="U8", shape=shape, offsets=(0, 0))}
        assert safeio.safetensors_header(build_safetensors(tmp_path / "m", header=hdr, data=b""), max_header_bytes=1 << 20).entries["a"].end == 0


def test_a_header_holding_an_enormous_element_count_is_refused(tmp_path):
    hdr = {"a": entry(dtype="U8", shape=(2 ** 40, 2 ** 40), offsets=(0, 4))}
    with pytest.raises(BadHeader):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=hdr, data=b"\0" * 4), max_header_bytes=1 << 20)


def test_a_packed_dtype_that_fills_whole_bytes_is_accepted(tmp_path):
    good = {"a": entry(dtype="F4", shape=(4,), offsets=(0, 2))}
    assert safeio.safetensors_header(build_safetensors(tmp_path / "m", header=good, data=b"\0\0"), max_header_bytes=1 << 20)


def test_metadata_over_a_megabyte_is_refused(tmp_path):
    big = {"__metadata__": {"k": "v" * (1 << 20)}}
    with pytest.raises(BadHeader, match="__metadata__"):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=big), max_header_bytes=1 << 22)


def test_a_failure_leaves_no_descriptor_open_and_a_read_error_is_a_read_failure(tmp_path, monkeypatch):
    path = build_safetensors(tmp_path / "m", ok_tensors())
    bad = build_safetensors(tmp_path / "bad", header=[], data=b"")
    before = fd_count()
    with pytest.raises(BadHeader):
        safeio.safetensors_header(bad, max_header_bytes=1 << 20)
    assert fd_count() == before
    monkeypatch.setattr(safeio.os, "pread", lambda *a: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
    with pytest.raises(ReadFailed):
        safeio.safetensors_header(path, max_header_bytes=1 << 20)
    assert fd_count() == before


def test_the_length_prefix_is_checked(tmp_path):
    good = build_safetensors(tmp_path / "g", ok_tensors())
    raw = good.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    cases = {
        "shorter_than_eight": (raw[:5], BadHeader),
        "length_zero": (struct.pack("<Q", 0) + raw[8:], BadHeader),
        "length_one": (struct.pack("<Q", 1) + raw[8:], BadHeader),
        "length_past_end": (struct.pack("<Q", len(raw)) + raw[8:], BadHeader),
        "length_huge": (struct.pack("<Q", 2 ** 63) + raw[8:], TooLarge),
        "header_cut_by_eof": (raw[:8 + n - 3], BadHeader),
        "header_not_json": (struct.pack("<Q", 5) + b"{abc}" + b"\0" * 40, BadHeader),
        "header_invalid_utf8": (struct.pack("<Q", 4) + b'{"\xff"' + b"\0" * 40, BadHeader),
        "header_nan": (struct.pack("<Q", 9) + b'{"a":NaN}' + b"\0" * 40, BadHeader),
    }
    for name, (data, exc) in cases.items():
        p = tmp_path / name
        p.write_bytes(data)
        with pytest.raises(exc):
            safeio.safetensors_header(p, max_header_bytes=1 << 20)


def test_a_header_over_the_limit_is_too_large_and_is_not_read(tmp_path, monkeypatch):
    path = build_safetensors(tmp_path / "m", ok_tensors(), pad_to=5000)
    reads = []
    real = os.pread
    monkeypatch.setattr(safeio.os, "pread", lambda fd, n, off: (reads.append(n), real(fd, n, off))[1])
    with pytest.raises(TooLarge):
        safeio.safetensors_header(path, max_header_bytes=4000)
    assert sum(reads) == 8
    assert safeio.safetensors_header(path, max_header_bytes=5000).entries


def test_a_repeated_tensor_name_is_refused(tmp_path):
    text = ('{"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, '
            '"a": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}')
    with pytest.raises(BadHeader):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=text, data=b"\0\0"), max_header_bytes=1 << 20)


def test_overlapping_ranges_are_refused_even_when_gaps_are_allowed(tmp_path):
    hdr = {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}, "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]}}
    p = build_safetensors(tmp_path / "m", header=hdr, data=b"\0" * 6)
    for contiguous in (True, False):
        with pytest.raises(BadHeader, match="share"):
            safeio.safetensors_header(p, max_header_bytes=1 << 20, contiguous=contiguous)


def test_two_tensors_over_the_same_bytes_are_refused(tmp_path):
    hdr = {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}, "b": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    with pytest.raises(BadHeader, match="share"):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=hdr, data=b"\0" * 4), max_header_bytes=1 << 20)


def test_an_empty_tensor_inside_another_tensors_range_is_refused(tmp_path):
    hdr = {"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}, "b": {"dtype": "U8", "shape": [0], "data_offsets": [2, 2]}}
    with pytest.raises(BadHeader, match="share"):
        safeio.safetensors_header(build_safetensors(tmp_path / "m", header=hdr, data=b"\0" * 4), max_header_bytes=1 << 20)


def test_gaps_and_unclaimed_bytes_are_refused_unless_allowed(tmp_path):
    gap = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}, "b": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]}}
    lead = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]}}
    tail = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}
    for name, hdr, size in (("gap", gap, 6), ("lead", lead, 4), ("tail", tail, 5)):
        p = build_safetensors(tmp_path / name, header=hdr, data=b"\0" * size)
        with pytest.raises(BadHeader, match="claims|claim"):
            safeio.safetensors_header(p, max_header_bytes=1 << 20)
        assert safeio.safetensors_header(p, max_header_bytes=1 << 20, contiguous=False).entries


def test_a_file_that_is_not_regular_or_is_a_link_is_refused(tmp_path):
    good = build_safetensors(tmp_path / "g", ok_tensors())
    (tmp_path / "ln").symlink_to("g")
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "d").mkdir()
    with pytest.raises(PathRefused):
        safeio.safetensors_header(tmp_path / "ln", max_header_bytes=1 << 20)
    for p in (tmp_path / "fifo", tmp_path / "d"):
        with pytest.raises(NotRegular):
            safeio.safetensors_header(p, max_header_bytes=1 << 20)
    with pytest.raises(ReadFailed):
        safeio.safetensors_header(tmp_path / "missing", max_header_bytes=1 << 20)
    assert good.exists()


def test_the_header_and_the_size_come_from_one_descriptor(tmp_path, monkeypatch):
    """The path is opened once. Swapping the path after the open must not change which file the checks use."""
    good = build_safetensors(tmp_path / "m", ok_tensors())
    other = build_safetensors(tmp_path / "other", [("z", "U8", [1], b"\0")])
    real = os.open

    def swap(path, flags, *a, **k):
        fd = real(path, flags, *a, **k)
        if str(path) == str(good):
            os.replace(other, good)                         # the path now names another file
        return fd

    monkeypatch.setattr(safeio.os, "open", swap)
    h = safeio.safetensors_header(good, max_header_bytes=1 << 20)
    assert set(h.entries) == {"a", "b", "c"}


@pytest.mark.parametrize("bad", [0, 1, True, -5, 2.5, None])
def test_a_bad_header_limit_is_a_programming_error(tmp_path, bad):
    with pytest.raises(ValueError):
        safeio.safetensors_header(tmp_path / "x", max_header_bytes=bad)


def test_the_dtype_table_matches_the_format():
    assert safeio.DTYPE_BITS["F32"] == 32 and safeio.DTYPE_BITS["BF16"] == 16 and safeio.DTYPE_BITS["F4"] == 4
    assert all(isinstance(v, int) and v > 0 for v in safeio.DTYPE_BITS.values())


# ----- the property tests: an independent oracle states what an accepted header must satisfy

BITS = safeio.DTYPE_BITS


def oracle_ok(header, data_size) -> bool:
    """True when `header` (a dict as parsed from JSON) is a valid safetensors header for a data section of `data_size` bytes, by a
    second and separate statement of the rules: contiguous tiling, exact spans, known dtypes."""
    if not isinstance(header, dict):
        return False
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values()):
        return False
    spans = []
    for name, v in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(v, dict) or set(v) != {"dtype", "shape", "data_offsets"} or v["dtype"] not in BITS:
            return False
        shape, off = v["shape"], v["data_offsets"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape) or len(shape) > 64:
            return False
        if not isinstance(off, list) or len(off) != 2 or any(type(o) is not int for o in off):
            return False
        n = 0 if 0 in shape else 1
        for d in shape:
            n *= d if n else 1
        bits = n * BITS[v["dtype"]]
        if bits % 8 or off[1] - off[0] != bits // 8 or not 0 <= off[0] <= off[1] <= data_size:
            return False
        spans.append((off[0], off[1]))
    spans.sort()
    cursor = 0
    for s, e in spans:
        if s != cursor:
            return False
        cursor = e
    return cursor == data_size


DTYPES = st.sampled_from(sorted(BITS))
SHAPES = st.lists(st.integers(min_value=0, max_value=5), max_size=3)


@st.composite
def valid_tensor_list(draw):
    items, pos = [], 0
    for i in range(draw(st.integers(min_value=0, max_value=5))):
        dtype, shape = draw(DTYPES), draw(SHAPES)
        n = 0 if 0 in shape else 1
        for d in shape:
            n *= d if n else 1
        bits = n * BITS[dtype]
        if bits % 8:
            shape = [0]
            bits = 0
        items.append((f"t{i}", dtype, shape, pos, pos + bits // 8))
        pos += bits // 8
    return items, pos


@settings(**PROFILE)
@given(valid_tensor_list(), st.randoms(use_true_random=False))
def test_a_valid_header_is_accepted_in_any_key_order(tmp_path_factory, data, rnd):
    items, size = data
    order = list(items)
    rnd.shuffle(order)
    header = {n: {"dtype": d, "shape": s, "data_offsets": [a, b]} for n, d, s, a, b in order}
    path = build_safetensors(tmp_path_factory.mktemp("st") / "m", header=header, data=b"\0" * size)
    got = safeio.safetensors_header(path, max_header_bytes=1 << 20)
    assert {n: (e.dtype, list(e.shape), e.start, e.end) for n, e in got.entries.items()} == {n: (d, s, a, b) for n, d, s, a, b in items}


def mutate(header: dict, kind: int, name: str, k: int):
    """One small change to a valid header; returns the changed header."""
    h = json.loads(json.dumps(header))
    e = h[name]
    if kind == 0:
        e["data_offsets"][1] += 1 + k
    elif kind == 1:
        e["data_offsets"][0] += 1 + k
    elif kind == 2:
        e["shape"] = e["shape"] + [2 + k]
    elif kind == 3:
        e["dtype"] = "NOPE"
    elif kind == 4:
        e["shape"] = [-1 - k]
    elif kind == 5:
        e["data_offsets"] = [e["data_offsets"][0]]
    elif kind == 6:
        e["extra"] = k
    elif kind == 7:
        e["data_offsets"] = [e["data_offsets"][0] + 0.0, e["data_offsets"][1]]
    elif kind == 8:
        del e["dtype"]
    return h


@settings(**PROFILE)
@given(valid_tensor_list().filter(lambda d: d[0]), st.integers(min_value=0, max_value=8), st.integers(min_value=0, max_value=3))
def test_a_header_with_one_change_is_judged_as_the_oracle_judges_it(tmp_path_factory, data, kind, k):
    items, size = data
    header = {n: {"dtype": d, "shape": s, "data_offsets": [a, b]} for n, d, s, a, b in items}
    changed = mutate(header, kind, items[0][0], k)
    path = build_safetensors(tmp_path_factory.mktemp("st") / "m", header=changed, data=b"\0" * size)
    try:
        safeio.safetensors_header(path, max_header_bytes=1 << 20)
        accepted = True
    except BadHeader:
        accepted = False
    assert accepted == oracle_ok(changed, size)


WILD = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(min_value=-5, max_value=40), st.text(max_size=3), st.sampled_from(sorted(BITS))),
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(st.sampled_from(["dtype", "shape", "data_offsets", "x"]), inner, max_size=4)),
    max_leaves=12)


@settings(**PROFILE)
@given(st.dictionaries(st.sampled_from(["a", "b", "__metadata__"]), WILD, max_size=3), st.integers(min_value=0, max_value=40))
def test_whatever_a_header_holds_it_is_refused_or_satisfies_the_oracle(tmp_path_factory, header, size):
    path = build_safetensors(tmp_path_factory.mktemp("st") / "m", header=header, data=b"\0" * size)
    try:
        got = safeio.safetensors_header(path, max_header_bytes=1 << 20)
    except SafeReadError:
        assert not oracle_ok(header, size)
    else:
        assert oracle_ok(header, size)
        assert got.data_size == size


@settings(**PROFILE)
@given(st.binary(max_size=64), st.integers(min_value=0, max_value=100))
def test_arbitrary_file_bytes_never_raise_anything_but_a_safe_error(tmp_path_factory, blob, length):
    p = tmp_path_factory.mktemp("st") / "m"
    p.write_bytes(struct.pack("<Q", length) + blob)
    try:
        safeio.safetensors_header(p, max_header_bytes=1 << 10)
    except SafeReadError:
        pass
