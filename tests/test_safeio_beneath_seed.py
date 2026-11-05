# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`safeio.list_beneath`, `read_beneath` and `write_beneath` (issue 326, review of PR 455, round 3).

They are the anchored versions of "list a directory", "read a file in it" and "write a file into another one", for a caller that holds a
directory open by descriptor and cannot let a link swapped in by someone else lead outside it. Each test uses real directories and a real link;
the swap is done by the test at the point it names, never by a race.
"""
import os
from pathlib import Path

import pytest

from orchard import safeio


def pin(path):
    return safeio.pinned_root(path)


def test_list_beneath_lists_the_directory_it_opened_even_if_the_name_is_swapped_afterwards(tmp_path):
    (tmp_path / "root" / "sub").mkdir(parents=True)
    (tmp_path / "root" / "sub" / "inside.py").write_text("x = 1\n")
    out = tmp_path / "outside"
    out.mkdir()
    (out / "outside.py").write_text("y = 2\n")
    with pin(tmp_path / "root") as root, safeio.list_beneath(root.fd, ("sub",)) as listing:
        os.rename(tmp_path / "root" / "sub", tmp_path / "root" / "sub.orig")
        os.symlink(out, tmp_path / "root" / "sub")
        assert sorted(e.name for e in listing) == ["inside.py"]


def test_list_beneath_refuses_a_link_standing_where_a_directory_should_be(tmp_path):
    (tmp_path / "root").mkdir()
    out = tmp_path / "outside"
    out.mkdir()
    (tmp_path / "root" / "sub").symlink_to(out)
    with pin(tmp_path / "root") as root, pytest.raises(safeio.PathRefused):
        with safeio.list_beneath(root.fd, ("sub",)):
            pytest.fail("a link was listed")


def test_list_beneath_with_no_parts_lists_the_root_and_a_dot_dot_is_refused(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "a.py").write_text("")
    with pin(tmp_path / "root") as root:
        with safeio.list_beneath(root.fd) as listing:
            assert [e.name for e in listing] == ["a.py"]
        with pytest.raises(safeio.PathRefused), safeio.list_beneath(root.fd, ("..",)):
            pass


def test_read_beneath_reads_a_file_and_refuses_a_link_in_any_part(tmp_path):
    (tmp_path / "root" / "sub").mkdir(parents=True)
    (tmp_path / "root" / "sub" / "x.py").write_text("x = 1\n")
    out = tmp_path / "outside"
    out.mkdir()
    (out / "x.py").write_text("OUTSIDE\n")
    with pin(tmp_path / "root") as root:
        assert safeio.read_beneath(root.fd, ("sub", "x.py"), max_bytes=100) == b"x = 1\n"
        os.rename(tmp_path / "root" / "sub", tmp_path / "root" / "sub.orig")
        os.symlink(out, tmp_path / "root" / "sub")
        with pytest.raises(safeio.PathRefused):
            safeio.read_beneath(root.fd, ("sub", "x.py"), max_bytes=100)
        (tmp_path / "root" / "link.py").symlink_to(out / "x.py")
        with pytest.raises(safeio.PathRefused):
            safeio.read_beneath(root.fd, ("link.py",), max_bytes=100)


def test_read_beneath_refuses_a_file_that_is_not_the_one_that_was_listed(tmp_path):
    (tmp_path / "root").mkdir()
    f = tmp_path / "root" / "a.py"
    f.write_text("one\n")
    st = os.stat(f)
    with pin(tmp_path / "root") as root:
        assert safeio.read_beneath(root.fd, ("a.py",), max_bytes=100, expect_identity=(st.st_dev, st.st_ino)) == b"one\n"
        new = tmp_path / "root" / "a.py.new"
        new.write_text("two\n")
        os.replace(new, f)
        with pytest.raises(safeio.ChangedWhileReading, match="changed"):
            safeio.read_beneath(root.fd, ("a.py",), max_bytes=100, expect_identity=(st.st_dev, st.st_ino))


def test_read_beneath_refuses_a_file_over_the_limit_and_a_bad_limit(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "big.py").write_bytes(b"x" * 50)
    with pin(tmp_path / "root") as root:
        with pytest.raises(safeio.TooLarge):
            safeio.read_beneath(root.fd, ("big.py",), max_bytes=10)
        for bad in (True, -1, "5"):
            with pytest.raises(ValueError):
                safeio.read_beneath(root.fd, ("big.py",), max_bytes=bad)


def test_write_beneath_creates_directories_writes_the_file_and_leaves_no_temporary(tmp_path):
    (tmp_path / "root").mkdir()
    with pin(tmp_path / "root") as root:
        safeio.write_beneath(root.fd, ("a", "b", "x.py"), b"x = 1\n", mode=0o640)
        safeio.write_beneath(root.fd, ("a", "b", "x.py"), b"x = 2\n", mode=0o640)       # replaces, does not fail
    assert (tmp_path / "root" / "a" / "b" / "x.py").read_text() == "x = 2\n"
    assert oct((tmp_path / "root" / "a" / "b" / "x.py").stat().st_mode & 0o777) == "0o640"
    assert [p.name for p in (tmp_path / "root" / "a" / "b").iterdir()] == ["x.py"], "a temporary file was left behind"


def test_write_beneath_does_not_write_through_a_link_standing_where_a_directory_should_be(tmp_path):
    (tmp_path / "root").mkdir()
    out = tmp_path / "outside"
    out.mkdir()
    (tmp_path / "root" / "sub").symlink_to(out)
    with pin(tmp_path / "root") as root, pytest.raises(safeio.PathRefused):
        safeio.write_beneath(root.fd, ("sub", "x.py"), b"x = 1\n")
    assert list(out.iterdir()) == []


def test_write_beneath_replaces_a_link_at_the_final_name_and_does_not_follow_it(tmp_path):
    (tmp_path / "root").mkdir()
    out = tmp_path / "outside.txt"
    out.write_text("keep me\n")
    (tmp_path / "root" / "x.py").symlink_to(out)
    with pin(tmp_path / "root") as root:
        safeio.write_beneath(root.fd, ("x.py",), b"x = 1\n")
    assert out.read_text() == "keep me\n"
    assert not (tmp_path / "root" / "x.py").is_symlink() and (tmp_path / "root" / "x.py").read_text() == "x = 1\n"


def test_write_beneath_removes_its_temporary_file_when_the_rename_fails(tmp_path):
    (tmp_path / "root" / "x.py").mkdir(parents=True)
    (tmp_path / "root" / "x.py" / "held").write_text("a directory is already here\n")
    with pin(tmp_path / "root") as root, pytest.raises(safeio.SafeReadError):
        safeio.write_beneath(root.fd, ("x.py",), b"x = 1\n")
    assert sorted(p.name for p in (tmp_path / "root").iterdir()) == ["x.py"], "the temporary file was left in the directory"


@pytest.mark.parametrize("parts", [(), ("..", "x.py"), (".", "x.py"), ("a/b",), ("",), ("a\0b",)])
def test_write_beneath_refuses_bad_parts_before_touching_anything(tmp_path, parts):
    (tmp_path / "root").mkdir()
    with pin(tmp_path / "root") as root, pytest.raises(safeio.PathRefused):
        safeio.write_beneath(root.fd, parts, b"x")
    assert list((tmp_path / "root").iterdir()) == []


def test_write_beneath_refuses_data_that_is_not_bytes(tmp_path):
    (tmp_path / "root").mkdir()
    with pin(tmp_path / "root") as root, pytest.raises(ValueError):
        safeio.write_beneath(root.fd, ("x.py",), "text")
