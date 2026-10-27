# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/pinned_files.py: reading and writing files relative to a directory descriptor, so a path that is replaced after it was checked cannot
redirect a later read or write. (No earlier helper in this repository anchors by descriptor: kernelprobe._read_regular and ttlang_env._read_reference
open by name and protect the last component only.)"""
import errno
import os
import stat

import pytest

from orchard import pinned_files as pf


@pytest.fixture
def d(tmp_path):
    fd = pf.open_dir(tmp_path)
    yield tmp_path, fd
    os.close(fd)


def test_a_regular_file_is_read_relative_to_the_descriptor(d):
    path, fd = d
    (path / "a.bin").write_bytes(b"hello")
    assert pf.read_regular(fd, "a.bin", 100) == b"hello"


def test_a_file_over_the_limit_is_refused_even_when_it_grows(d):
    path, fd = d
    (path / "a.bin").write_bytes(b"x" * 11)
    with pytest.raises(pf.PinnedError, match="larger than 10"):
        pf.read_regular(fd, "a.bin", 10)
    (path / "b.bin").write_bytes(b"x" * 10)
    assert pf.read_regular(fd, "b.bin", 10) == b"x" * 10


def test_a_link_is_refused_with_its_own_message(d):
    path, fd = d
    (path / "real").write_bytes(b"x")
    (path / "link").symlink_to(path / "real")
    with pytest.raises(pf.PinnedError, match="is a link"):
        pf.read_regular(fd, "link", 10)


def test_a_missing_file_is_refused_as_missing(d):
    _, fd = d
    with pytest.raises(pf.PinnedError, match="is missing"):
        pf.read_regular(fd, "nope", 10)


def test_a_directory_a_fifo_and_a_device_are_not_regular_files(d):
    path, fd = d
    (path / "dir").mkdir()
    os.mkfifo(path / "fifo")
    for name in ("dir", "fifo"):
        with pytest.raises(pf.PinnedError, match="not a regular file"):
            pf.read_regular(fd, name, 10)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "/abs", "a\0b", "../x", "x/", 3, None, b"a", ["a"]])
def test_only_a_single_plain_name_is_accepted(d, name):
    _, fd = d
    with pytest.raises(pf.PinnedError, match="not a plain file name"):
        pf.read_regular(fd, name, 10)
    with pytest.raises(pf.PinnedError, match="not a plain file name"):
        pf.sha256_regular(fd, name, 10)
    with pytest.raises(pf.PinnedError, match="not a plain file name"):
        pf.create_exclusive(fd, name)


def test_a_directory_replaced_by_a_link_after_the_descriptor_was_opened_does_not_redirect_reads(tmp_path):
    real = tmp_path / "run"
    real.mkdir()
    (real / "a.bin").write_bytes(b"inside")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.bin").write_bytes(b"outside")
    fd = pf.open_dir(real)
    try:
        real.rename(tmp_path / "moved")
        real.symlink_to(elsewhere)
        assert pf.read_regular(fd, "a.bin", 10) == b"inside"
    finally:
        os.close(fd)


def test_opening_a_directory_refuses_a_link_unless_it_is_the_given_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(pf.PinnedError, match="is a link"):
        pf.open_dir(link, follow=False)
    os.close(pf.open_dir(link, follow=True))


def test_a_subdirectory_that_is_a_link_is_refused_relative_to_a_descriptor(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "sub").symlink_to(tmp_path / "real")
    fd = pf.open_dir(tmp_path)
    try:
        with pytest.raises(pf.PinnedError, match="is a link"):
            pf.open_dir("sub", dir_fd=fd, follow=False)
        with pytest.raises(pf.PinnedError, match="is missing"):
            pf.open_dir("nothere", dir_fd=fd, follow=False)
    finally:
        os.close(fd)


def test_opening_something_that_is_not_a_directory_is_refused(tmp_path):
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(pf.PinnedError, match="not a directory"):
        pf.open_dir(tmp_path / "f")


def test_the_hash_is_counted_and_capped(d):
    import hashlib
    path, fd = d
    (path / "a.bin").write_bytes(b"abc" * 100)
    assert pf.sha256_regular(fd, "a.bin", 1000) == (hashlib.sha256(b"abc" * 100).hexdigest(), 300)
    with pytest.raises(pf.PinnedError, match="larger than 299"):
        pf.sha256_regular(fd, "a.bin", 299)
    assert pf.sha256_regular(fd, "a.bin", 300)[1] == 300


def test_the_hash_refuses_a_link_and_a_directory(d):
    path, fd = d
    (path / "real").write_bytes(b"x")
    (path / "link").symlink_to(path / "real")
    (path / "dir").mkdir()
    with pytest.raises(pf.PinnedError, match="is a link"):
        pf.sha256_regular(fd, "link", 10)
    with pytest.raises(pf.PinnedError, match="not a regular file"):
        pf.sha256_regular(fd, "dir", 10)


def test_a_file_is_created_exclusively_and_never_through_a_link(d):
    path, fd = d
    f = pf.create_exclusive(fd, "new")
    os.close(f)
    assert stat.S_ISREG(os.stat(path / "new").st_mode)
    with pytest.raises(pf.PinnedError, match="already exists"):
        pf.create_exclusive(fd, "new")
    (path / "real").write_bytes(b"x")
    (path / "link").symlink_to(path / "real")
    with pytest.raises(pf.PinnedError):
        pf.create_exclusive(fd, "link")
    assert (path / "real").read_bytes() == b"x"


def test_publishing_never_replaces_an_existing_target(d):
    path, fd = d
    f = pf.create_exclusive(fd, "tmp1")
    os.write(f, b"new")
    os.close(f)
    (path / "final").write_bytes(b"old")
    with pytest.raises(pf.PinnedError, match="already exists"):
        pf.publish_no_replace(fd, "tmp1", "final")
    assert (path / "final").read_bytes() == b"old" and (path / "tmp1").exists()


def test_publishing_moves_the_temporary_file_to_the_target(d):
    path, fd = d
    f = pf.create_exclusive(fd, "tmp1")
    os.write(f, b"new")
    os.close(f)
    pf.publish_no_replace(fd, "tmp1", "final")
    assert (path / "final").read_bytes() == b"new" and not (path / "tmp1").exists()


def test_publishing_a_missing_temporary_file_is_an_error(d):
    _, fd = d
    with pytest.raises(pf.PinnedError, match="is missing"):
        pf.publish_no_replace(fd, "nope", "final")


def test_the_descriptor_is_not_inherited_by_children(d):
    _, fd = d
    import fcntl
    assert fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC


def test_a_file_opened_as_a_directory_without_following_links_says_it_is_not_a_directory(tmp_path):
    (tmp_path / "f").write_bytes(b"x")
    with pytest.raises(pf.PinnedError, match="not a directory"):
        pf.open_dir(tmp_path / "f", follow=False)
    fd = pf.open_dir(tmp_path)
    try:
        with pytest.raises(pf.PinnedError, match="not a directory"):
            pf.open_dir("f", dir_fd=fd, follow=False)
    finally:
        os.close(fd)
