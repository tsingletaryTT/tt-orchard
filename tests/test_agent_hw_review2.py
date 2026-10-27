# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for Copilot's second review of PR 116: a relative output directory, a partial attempt saved before the first
check, and the chip time of an interrupted run. Fakes only: no gozer, no chip."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from fake_model import FakeModel
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import (CLOCK, fake_leases, FACTS, GOOD_HW_SET, KERNELS, World, _report_md, _row_attempt, attempt_dir,  # noqa: F401
                           fake_kenv, good_hw, hw_summary, make_check, run_attempt, scripted_hw_agent,
                           skills_current, ttlang_skill)


# ---- 1. a relative output directory ------------------------------------------------------------------------------


def strict_harness(seen):
    """A fake gozer run that behaves like the real child: it resolves --kernel and --out against its working
    directory, so a relative path that already contains that directory is read from a doubled path and fails."""
    def fake(argv, **kw):
        cwd = Path(kw["cwd"])
        seen.append({"cwd": cwd, "argv": list(argv)})
        kernel = cwd / argv[argv.index("--kernel") + 1]
        out = cwd / argv[argv.index("--out") + 1]
        if not kernel.is_file() or not out.parent.is_dir():
            return kp.Watched(1, None, False, 1.0)                # the harness could not load or write: no result file
        out.write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
        return kp.Watched(0, None, False, 1.0)
    return fake


def test_the_tool_works_with_the_default_relative_output_directory(tmp_path, fake_kenv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adir = Path("kernel-probe") / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    seen = []
    text = kp.make_hardware_check(adir, op="add", n=1, shared_python="/p", policy=hwtool.HwPolicy(),
                                  run_watched=strict_harness(seen), leases=fake_leases()).call({})
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS"), text
    argv = seen[0]["argv"]
    assert Path(argv[argv.index("--kernel") + 1]).is_absolute() and Path(argv[argv.index("--out") + 1]).is_absolute()
    assert seen[0]["cwd"].is_absolute()


def test_the_hardware_phase_works_with_a_relative_output_directory(tmp_path, fake_kenv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adir = Path("kernel-probe") / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    clean = kp.verdict_dir(Path("kernel-probe"), "add", 1) / kp.CLEAN_DIR
    clean.mkdir(parents=True)
    (clean / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    sha = hashlib.sha256((clean / "kernel.py").read_bytes()).hexdigest()
    attempt = {"op": "add", "attempt": 1, "passed_sim": True, "category": None, "hw": None,
               "provenance": kp.provenance(kernel_sha256=sha, tt_lang=FACTS, coder_model="c", probe_id="t",
                                           simulator_pcc_min=0.9999)}
    seen = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=strict_harness(seen))
    assert hw["passed"] is True, hw
    argv = seen[0]["argv"]
    assert Path(argv[argv.index("--kernel") + 1]).is_absolute() and Path(argv[argv.index("--out") + 1]).is_absolute()


def test_the_child_environment_names_an_absolute_attempt_directory(tmp_path, fake_kenv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adir = Path("kernel-probe") / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
    seen = []

    def fake(argv, **kw):
        seen.append(kw)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None}))
        return kp.Watched(0, None, False, 1.0)
    kp.make_hardware_check(adir, op="add", n=1, shared_python="/p", policy=hwtool.HwPolicy(), run_watched=fake,
                           leases=fake_leases()).call({})
    assert seen[0]["env"]["ORCHARD_RUN_DIR"] == str((tmp_path / adir).resolve())
    assert Path(seen[0]["log_path"]).is_absolute()


# ---- 2. a partial attempt saved before the first authoritative check ----------------------------------------------


def test_an_interrupt_during_the_agent_loop_still_saves_the_partial_attempt(tmp_path, fake_kenv):
    """On a fresh run `.verdict/<op>-<n>/` does not exist until the authoritative check makes it."""
    good = (KERNELS / "add_good.py").read_text()

    def interrupt_on_the_chip(argv, **kw):
        raise KeyboardInterrupt
    with FakeModel(scripted_hw_agent(good)) as server:
        with pytest.raises(KeyboardInterrupt) as info:
            run_attempt(tmp_path, fake_kenv, server, interrupt_on_the_chip, hw_policy=hwtool.HwPolicy(),
                        shared_python="/p")
    partial = info.value.partial_attempt
    probe = tmp_path / "probe"
    assert partial["agent_hw"]["calls"][0]["outcome"] == "interrupted"
    assert json.loads((probe / ".verdict" / "add-1" / "attempt.json").read_text()) == partial
    assert json.loads((probe / "add" / "attempt-1" / "attempt.json").read_text()) == partial
    records, ignored = kp.load_attempts(probe, ops=["add"], probe_id="t")
    assert [r["attempt"] for r in records] == [1] and ignored == 0


def test_the_verdict_directory_is_made_without_following_a_link(tmp_path, fake_kenv):
    probe = tmp_path / "probe"
    probe.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (probe / ".verdict").symlink_to(elsewhere)               # a link where the probe's directory belongs
    adir = probe / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    kp._save_attempt(adir, {"op": "add", "attempt": 1})
    assert not (probe / ".verdict").is_symlink() and (probe / ".verdict" / "add-1" / "attempt.json").is_file()
    assert list(elsewhere.iterdir()) == []                   # nothing was written through the link


def test_a_link_in_place_of_the_attempts_verdict_directory_is_replaced_not_followed(tmp_path, fake_kenv):
    probe = tmp_path / "probe"
    (probe / ".verdict").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (probe / ".verdict" / "add-1").symlink_to(elsewhere)
    adir = probe / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    kp._save_attempt(adir, {"op": "add", "attempt": 1})
    assert not (probe / ".verdict" / "add-1").is_symlink() and list(elsewhere.iterdir()) == []


# ---- 3. the chip time of an interrupted run -----------------------------------------------------------------------


def test_an_interrupted_run_counts_the_time_it_held_the_board_as_an_upper_bound():
    w = World()

    def interrupted(k, timeout_s, wait_s, on_grant, ctx):
        on_grant()
        w.now += 30.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=interrupted, clock=w.clock, sleep=w.sleep)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "interrupted" and rec["lease_s"] == 30.0 and rec["total_s"] == 30.0
    assert rec["lease_upper_bound"] is True
    assert c.summary()["lease_s_used"] == 30.0


def test_the_chip_time_of_an_interrupted_run_counts_toward_the_cap():
    w = World()

    def interrupted(k, timeout_s, wait_s, on_grant, ctx):
        on_grant()
        w.now += 500.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(lease_s=600.0), prepare=w.prepare, run=interrupted, clock=w.clock,
                             sleep=w.sleep)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    c.run = w.run
    assert c.call({}).startswith("hardware_check refused")      # 100 s left, 120 reserved: refused


def test_the_report_says_the_chip_time_of_an_interrupted_call_is_an_upper_bound(tmp_path):
    rec = {"outcome": "interrupted", "passed": False, "lease_s": 30.0, "total_s": 30.0, "lease_upper_bound": True}
    md = _report_md(tmp_path, [_row_attempt(agent_hw=hw_summary(rec))])
    row = next(l for l in md.splitlines() if "| interrupted |" in l)
    assert "upper bound" in row and "30" in row
    assert "30 s of chip time" in md


