# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 587 (review of PR 580): the holder scan names every device node it checks, and the retry decision looks only at the leased chips.

(1) `fuser` does not look inside a directory, so with no holder found through /proc the scan used to hand it `/dev/tenstorrent` and learn
    nothing; with some holders found it passed only their nodes and missed holders of the other chips. The scan lists the device nodes
    itself and gives fuser all of them (bounded). When the listing fails the report says so and fuser is not given the directory.
(2) A container on chips 2 and 3 is not a reason to refuse a retry for a coder that leased chips 0 and 1. `any_container` takes the leased
    device indices; the full report stays whole."""
import os

from orchard import device_holders
from orchard.commands import CommandResult


def fake_proc(tmp_path, procs):
    """A /proc with one directory per pid; each holds fd links to the named device nodes and a cgroup text."""
    root = tmp_path / "proc"
    for pid, (devices, cgroup) in procs.items():
        fd = root / str(pid) / "fd"
        fd.mkdir(parents=True)
        for i, dev in enumerate(devices):
            os.symlink(dev, fd / str(i))
        (root / str(pid) / "cgroup").write_text(cgroup)
        (root / str(pid) / "comm").write_text("proc\n")
    root.mkdir(exist_ok=True)
    return str(root)


class FuserSpy:
    def __init__(self):
        self.argvs = []

    def __call__(self, argv, timeout):
        self.argvs.append(list(argv))
        return CommandResult(tuple(argv), 1, "", "")


def scan(tmp_path, procs, nodes=("0", "1", "2", "3"), **kw):
    spy = FuserSpy()
    found = device_holders.scan(proc_root=fake_proc(tmp_path, procs), dev_prefix="/dev/tenstorrent", which=lambda n: "/usr/bin/fuser",
                                run=spy, listdir=lambda d: list(nodes), **kw)
    return found, spy


def test_fuser_is_given_every_device_node_even_when_no_holder_was_found_through_proc(tmp_path):
    found, spy = scan(tmp_path, {})
    assert spy.argvs == [["/usr/bin/fuser", "-v", "/dev/tenstorrent/0", "/dev/tenstorrent/1", "/dev/tenstorrent/2", "/dev/tenstorrent/3"]]
    assert found["nodes"] == ["/dev/tenstorrent/0", "/dev/tenstorrent/1", "/dev/tenstorrent/2", "/dev/tenstorrent/3"]


def test_fuser_still_checks_the_other_chips_when_proc_showed_a_holder_of_one(tmp_path):
    found, spy = scan(tmp_path, {41: (["/dev/tenstorrent/0"], "0::/user.slice")})
    assert spy.argvs[0][2:] == ["/dev/tenstorrent/0", "/dev/tenstorrent/1", "/dev/tenstorrent/2", "/dev/tenstorrent/3"]


def test_a_device_listing_that_fails_is_reported_and_fuser_never_gets_the_directory(tmp_path):
    spy = FuserSpy()

    def broken(d):
        raise PermissionError(d)
    found = device_holders.scan(proc_root=fake_proc(tmp_path, {}), dev_prefix="/dev/tenstorrent", which=lambda n: "/usr/bin/fuser", run=spy,
                                listdir=broken)
    assert "PermissionError" in found["nodes_error"] and found["nodes"] == []
    assert spy.argvs == [] and found["fuser"]["skipped"]


def test_the_node_list_is_bounded_and_ignores_names_that_are_not_nodes(tmp_path):
    names = [str(i) for i in range(500)] + ["by-id", "../x"]
    found, spy = scan(tmp_path, {}, nodes=names)
    assert len(found["nodes"]) <= device_holders.MAX_NODES and all(n.rsplit("/", 1)[1].isdigit() for n in found["nodes"])
    assert len(spy.argvs[0]) <= device_holders.MAX_NODES + 2


def test_a_container_on_other_chips_is_not_a_holder_of_the_leased_chips():
    found = {"holders": [{"pid": 7, "comm": "docker", "devices": ["/dev/tenstorrent/2", "/dev/tenstorrent/3"], "container": True}]}
    assert device_holders.any_container(found) is True                   # the whole report still says a container holds something
    assert device_holders.any_container(found, dev_indices=(0, 1)) is False
    assert device_holders.any_container(found, dev_indices=(1, 2)) is True


def test_a_holder_whose_node_cannot_be_read_as_a_chip_number_counts_for_every_lease():
    found = {"holders": [{"pid": 7, "comm": "x", "devices": ["/dev/tenstorrent"], "container": True}]}
    assert device_holders.any_container(found, dev_indices=(0, 1)) is True
