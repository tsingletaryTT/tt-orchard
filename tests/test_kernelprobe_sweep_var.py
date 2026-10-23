# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""sweep_strays and scan_strays can look for another environment variable than ORCHARD_RUN_DIR (the supervisor's graft trial uses
ORCHARD_TRIAL_DIR), and can record the identity each process had. The defaults are unchanged."""
import os
import subprocess
import sys

from orchard import kernelprobe

CODE = "import time; time.sleep(600)"


def start(tmp_path, **env):
    return subprocess.Popen([sys.executable, "-c", CODE], env={**os.environ, **{k: str(v) for k, v in env.items()}},
                            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def test_scan_finds_a_process_by_the_named_variable_and_not_by_the_default_one(tmp_path):
    a = start(tmp_path, ORCHARD_TRIAL_DIR=tmp_path)
    try:
        assert a.pid in kernelprobe.scan_strays(tmp_path, var="ORCHARD_TRIAL_DIR")
        assert a.pid not in kernelprobe.scan_strays(tmp_path)                              # the default variable is not set on it
    finally:
        a.kill()
        a.wait()


def test_the_named_variable_must_hold_exactly_the_directory(tmp_path):
    a = start(tmp_path, ORCHARD_TRIAL_DIR=str(tmp_path) + "0")
    try:
        assert a.pid not in kernelprobe.scan_strays(tmp_path, var="ORCHARD_TRIAL_DIR")
    finally:
        a.kill()
        a.wait()


def test_sweep_stops_the_process_it_found_by_the_named_variable(tmp_path):
    a = start(tmp_path, ORCHARD_TRIAL_DIR=tmp_path)
    recs = kernelprobe.sweep_strays(tmp_path, var="ORCHARD_TRIAL_DIR", grace_s=0.5)
    a.wait(timeout=10)
    assert [r["pid"] for r in recs] == [a.pid] and recs[0]["stopped"] is True


def test_a_sweep_by_the_default_variable_leaves_a_process_that_only_has_the_named_one(tmp_path):
    a = start(tmp_path, ORCHARD_TRIAL_DIR=tmp_path)
    try:
        assert kernelprobe.sweep_strays(tmp_path, grace_s=0.2) == []
        assert a.poll() is None
    finally:
        a.kill()
        a.wait()


def test_the_default_records_carry_no_identity_and_the_identity_option_adds_it(tmp_path):
    a = start(tmp_path, ORCHARD_RUN_DIR=tmp_path)
    b = start(tmp_path, ORCHARD_RUN_DIR=tmp_path)
    try:
        plain = kernelprobe.sweep_strays(tmp_path, grace_s=0.5)
        assert plain and all(set(r) == {"pid", "cmdline", "stopped"} for r in plain)
        c = start(tmp_path, ORCHARD_RUN_DIR=tmp_path)
        rich = kernelprobe.sweep_strays(tmp_path, grace_s=0.5, identity=True)
        assert [r["pid"] for r in rich] == [c.pid] and isinstance(rich[0]["start"], int) and rich[0]["boot_id"]
        c.wait(timeout=10)
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()
            p.wait()
