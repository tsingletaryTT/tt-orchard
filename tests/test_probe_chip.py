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


# The agent's hardware_check launches the wrapper through child_env, whose allow-list used to drop the variable (issue 218): the
# wrapper then leased chip 1, beside the coder, and queued behind it.

class _Seen(Exception):
    pass


def _env_the_wrapper_is_launched_with(tmp_path):
    seen = {}

    def run_watched(argv, **kw):
        seen["env"] = kw["env"]
        raise _Seen

    adir, vdir = tmp_path / "a", tmp_path / "v"
    adir.mkdir()
    vdir.mkdir()
    kernel = adir / "kernel.py"
    kernel.write_text("x = 1\n")
    wrapper = {"who": "w", "wait_s": 1, "sha256": "0" * 64, "call_timeout_s": 5}
    with pytest.raises(_Seen):
        kernelprobe._hw_run(adir, vdir, kernel, op="add", reason="r", shared_python="python3", run_watched=run_watched,
                            cancel=lambda t: None, wrapper=wrapper)
    return seen["env"]


def test_the_wrapper_launch_forwards_the_chip_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", "2")
    assert _env_the_wrapper_is_launched_with(tmp_path)["ORCHARD_PROBE_CHIP"] == "2"


def test_the_wrapper_launch_adds_nothing_when_the_variable_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("ORCHARD_PROBE_CHIP", raising=False)
    assert "ORCHARD_PROBE_CHIP" not in _env_the_wrapper_is_launched_with(tmp_path)


@pytest.mark.parametrize("bad", ["", "-1", "1,2", "two", "1; rm", "--chips"])
def test_a_bad_selection_is_refused_before_the_wrapper_starts(monkeypatch, tmp_path, bad):
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", bad)
    with pytest.raises(ValueError, match="ORCHARD_PROBE_CHIP"):
        _env_the_wrapper_is_launched_with(tmp_path)


def test_a_child_that_is_not_leased_gets_no_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCHARD_PROBE_CHIP", "2")
    assert "ORCHARD_PROBE_CHIP" not in kernelprobe.child_env(tmp_path, hardware=False)
