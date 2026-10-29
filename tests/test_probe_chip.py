# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chip that the kernel probe's hardware phase asks gozer for.

It was chip 1 in three places. On this machine the coder's container holds board 0, which is chips 0 and 1, so every hardware check of a
probe that runs beside the coder queued behind the coder and gave up: six hardware calls of one run ended `busy` while board 1 was free.
`ORCHARD_PROBE_CHIP` names another chip; the default stays 1.
"""
import pytest

from orchard import hwtool, kernelprobe, ttnn_dump


def test_the_default_is_chip_1_as_before():
    assert hwtool.probe_chip({}) == "1"


@pytest.mark.parametrize("value", ["0", "2", "3", "12"])
def test_a_whole_number_names_the_chip(value):
    assert hwtool.probe_chip({"ORCHARD_PROBE_CHIP": value}) == value


@pytest.mark.parametrize("bad", ["", " ", "-1", "1,2", "two", "1; rm", "--chips", "0x2", "٣", "1\n2"])
def test_anything_else_is_refused(bad):
    with pytest.raises(ValueError, match="ORCHARD_PROBE_CHIP"):
        hwtool.probe_chip({"ORCHARD_PROBE_CHIP": bad})


def _exact(argv):
    return argv[argv.index("--exact") + 1]


def test_the_probe_lease_follows_the_variable(monkeypatch):
    monkeypatch.delenv("ORCHARD_PROBE_CHIP", raising=False)
    assert _exact(kernelprobe._gozer("r", "python", "x.py")) == "1"
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", "2")
    assert _exact(kernelprobe._gozer("r", "python", "x.py")) == "2"


def test_the_inventory_dump_lease_follows_the_variable(monkeypatch):
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", "3")
    assert _exact(ttnn_dump.dump_command("python")) == "3"


def test_the_agents_hardware_tool_lease_follows_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", "2")
    seen = []

    def sh(argv, timeout):
        seen.append(argv)
        return 1, "", "stop here"

    hwtool.run_wrapper(who="w", reason="r", wait_s=1, result_path=tmp_path / "r.json", command=["true"], sh=sh)
    acquire = next(a for a in seen if "acquire" in a)
    assert _exact(acquire) == "2"
