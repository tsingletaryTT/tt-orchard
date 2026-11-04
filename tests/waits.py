# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Waiting for a condition in a test (issues 395 and 405).

A test that starts a child process or a helper and then looks for what it did has to wait for it. A fixed number of seconds that is enough on
an idle box is not enough on one with a load average of 80, where the suite takes six times as long (32 minutes against 5). The wait here
ends as soon as the condition holds, so a passing test pays nothing for a large limit. Only a test that is going to fail waits the whole
`PATIENCE_S`, and then says what it waited for.
"""
from __future__ import annotations

import time
from typing import Callable

# Long enough for a child that waits for its first time slice on a box at load 80, and short enough that a real failure ends the run.
PATIENCE_S = 90.0


def wait_until(condition: Callable[[], object], what: str, *, timeout: float = PATIENCE_S, interval: float = 0.05):
    """Poll `condition` until it returns something true, and return that value. Raises AssertionError naming `what` after `timeout`
    seconds, which are measured on the monotonic clock, so a change of the wall clock does not end the wait early or late."""
    deadline = time.monotonic() + timeout
    while True:
        value = condition()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(interval)
