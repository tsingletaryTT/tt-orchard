# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 447: the three deterministic guards of the driver's command-timeout rule drive a fake child and use no gozer at all. They sit in
tests/test_hardware_check.py, whose other tests need the tt-gozer checkout. A module-wide skip hid them on a machine without the checkout, so they
could go red or be deleted unseen. This test runs them in a child pytest with the checkout pointed at a path that does not exist and requires that
each one PASSES (a skip is a failure here)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARDS = (
    "test_a_protected_call_that_outlives_one_timeout_gets_one_more_full_timeout_and_is_not_killed",
    "test_a_protected_call_that_outlives_two_timeouts_is_left_running_and_waited_for_no_third_time",
    "test_a_call_that_is_not_protected_is_killed_at_its_first_timeout",
)


def test_the_fake_child_timeout_guards_run_and_pass_when_the_gozer_checkout_is_absent(tmp_path):
    env = {**os.environ, "ORCHARD_TEST_GOZER": str(tmp_path / "no-such-gozer")}
    done = subprocess.run([sys.executable, "-m", "pytest", "tests/test_hardware_check.py", "-k", " or ".join(GUARDS), "-q", "-rA", "-p", "no:cacheprovider",
                           f"--basetemp={tmp_path / 'bt'}"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180, start_new_session=True)
    for name in GUARDS:
        assert f"PASSED tests/test_hardware_check.py::{name}" in done.stdout, done.stdout[-1500:]
    assert "SKIPPED" not in done.stdout and done.returncode == 0, done.stdout[-1500:]


def test_a_test_that_needs_the_gozer_checkout_is_skipped_without_it_and_not_failed(tmp_path):
    env = {**os.environ, "ORCHARD_TEST_GOZER": str(tmp_path / "no-such-gozer")}
    done = subprocess.run([sys.executable, "-m", "pytest", "tests/test_hardware_check.py", "-q", "-rs", "-p", "no:cacheprovider",
                           f"--basetemp={tmp_path / 'bt'}"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180, start_new_session=True)
    assert done.returncode == 0 and " failed" not in done.stdout.splitlines()[-1] and "skipped" in done.stdout.splitlines()[-1], done.stdout[-1500:]
    assert "is not present" in done.stdout
