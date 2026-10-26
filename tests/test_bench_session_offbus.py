# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 25, second half, for the benchmark session: a phase that fails is checked against the PCIe bus, and a board that left
the bus ends the session with exit 7 and the cause in summary.json where it used to look like a model failure.

The bus is a fake (the default reader is replaced), the sleep and clock are fakes, and `lm_eval` and the server are the stand-ins
of tests/test_bench_session.py. Nothing opens a chip."""
from __future__ import annotations

import json

import pytest

from orchard import bench_session as bs
from orchard import bus
from test_bench_session import harness, run_session  # noqa: F401  (harness is a fixture)
from test_supervisor_offbus import ALL_CHIPS, FakeBus

HELD = ",".join(ALL_CHIPS[:2])


class Time:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s


@pytest.fixture
def fake_bus(monkeypatch):
    fb = FakeBus()
    monkeypatch.setattr(bus, "_default_read", fb)
    t = Time()
    monkeypatch.setattr(bs, "_sleep", t.sleep)
    monkeypatch.setattr(bs, "_clock", t.clock)
    monkeypatch.setenv("TT_VISIBLE_DEVICES", HELD)
    fb.time = t
    return fb


def fail_lm_eval(h):
    (h["tmp"] / "fail").write_text("")          # the stand-in lm_eval exits 1 when this file exists


def summary(out):
    return json.loads((out / "summary.json").read_text())


def test_a_failed_phase_on_a_dead_board_ends_the_session_with_exit_7(harness, fake_bus):
    fail_lm_eval(harness)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    code, out = run_session(harness, "gsm8k,bfcl")
    assert code == bs.EXIT_OFF_BUS == 7
    s = summary(out)
    assert [p["name"] for p in s["phases"]] == ["gsm8k"], "the later phases were not run on a dead board"
    p = s["phases"][0]
    assert p["ok"] is False and bus.OFF_BUS in p["problem"] and "D3cold" in p["problem"] and ALL_CHIPS[0] in p["problem"]
    assert "exit 1" in p["problem"], "the failure that came first is kept"
    assert s["off_bus"].startswith(bus.OFF_BUS)


def test_a_failed_phase_on_a_healthy_bus_is_still_a_model_failure_and_the_session_goes_on(harness, fake_bus):
    fail_lm_eval(harness)
    code, out = run_session(harness, "gsm8k,bfcl")
    assert code == 1
    s = summary(out)
    assert [p["name"] for p in s["phases"]] == ["gsm8k", "bfcl"]
    assert "off_bus" not in s and bus.OFF_BUS not in (s["phases"][0]["problem"] or "")
    assert fake_bus.reads >= 1


def test_a_phase_that_passes_does_not_read_the_bus(harness, fake_bus):
    code, out = run_session(harness, "gsm8k")
    assert code == 0 and fake_bus.reads == 0


def test_a_failed_phase_with_no_chips_named_cannot_be_judged_and_stays_a_model_failure(harness, fake_bus, monkeypatch):
    monkeypatch.delenv("TT_VISIBLE_DEVICES")
    fail_lm_eval(harness)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    code, out = run_session(harness, "gsm8k")
    assert code == 1 and fake_bus.reads == 0


def test_only_the_chips_the_session_holds_are_judged(harness, fake_bus):
    fail_lm_eval(harness)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS[2:]}
    code, out = run_session(harness, "gsm8k")
    assert code == 1 and "off_bus" not in summary(out)


def test_the_reset_window_is_waited_out_first(harness, fake_bus):
    fail_lm_eval(harness)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    run_session(harness, "gsm8k")
    assert fake_bus.time.now >= bus.SETTLE_S and fake_bus.reads > 1


def test_a_server_that_never_became_healthy_on_a_dead_board_is_reported_as_off_the_bus(harness, fake_bus, monkeypatch, tmp_path):
    class Swap:
        def guard_cache(self, cache, model_id):
            pass

        def wait_healthy(self, proc, port, timeout, log):
            raise SystemExit(4)

        def stop_server(self, proc):
            pass
    monkeypatch.setattr(bs, "_swap", lambda: Swap())
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    run_sh = tmp_path / "run.sh"
    run_sh.write_text("exit 0\n")
    out = tmp_path / "out"
    code = bs.main(["--label", "t", "--out-dir", str(out), "--phases", "gsm8k", "--run-sh", str(run_sh), "--model-id", "x/y", "--model-dir", str(tmp_path),
                    "--tt-cache", str(tmp_path / "c"), "--hf-home", str(tmp_path / "h")])
    assert code == bs.EXIT_OFF_BUS
    assert bus.OFF_BUS in summary(out)["off_bus"]


def test_a_server_that_never_became_healthy_on_a_healthy_bus_keeps_its_own_exit(harness, fake_bus, monkeypatch, tmp_path):
    class Swap:
        def guard_cache(self, cache, model_id):
            pass

        def wait_healthy(self, proc, port, timeout, log):
            raise SystemExit(4)

        def stop_server(self, proc):
            pass
    monkeypatch.setattr(bs, "_swap", lambda: Swap())
    run_sh = tmp_path / "run.sh"
    run_sh.write_text("exit 0\n")
    with pytest.raises(SystemExit) as e:
        bs.main(["--label", "t", "--out-dir", str(tmp_path / "out"), "--phases", "gsm8k", "--run-sh", str(run_sh), "--model-id", "x/y",
                 "--model-dir", str(tmp_path), "--tt-cache", str(tmp_path / "c"), "--hf-home", str(tmp_path / "h")])
    assert e.value.code == 4


def test_device_indices_in_the_environment_are_not_pci_addresses_and_name_no_chips(harness, fake_bus, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1")
    fail_lm_eval(harness)
    fake_bus.bad = {c: "D3cold" for c in ALL_CHIPS}
    code, out = run_session(harness, "gsm8k")
    assert code == 1 and fake_bus.reads == 0
