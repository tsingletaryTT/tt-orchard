# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The refresh button of the Models view's Updates panel (issue 231).

`GET /api/updates` (orchard/updates_view.py) stays read only: it reads the cache file and starts nothing. This module is the one place
where the web server runs `tt-orchard updates --refresh` for the page. That command queries public repositories and the Hugging Face Hub, so
this is a process that makes network requests, started from a button. What holds it down:

- The argument list is fixed (`ARGV`) and has no shell. `RefreshJob.start()` takes no argument, so nothing the page sends reaches the command.
- The child gets an allow-listed environment (`ENV_ALLOW`: what an agent shell gets, plus the home directory for the cache file and the
  checkout setting for the tt-metal tree). No token, credential file path, proxy setting or device variable is passed on.
- Its input is closed and its output is discarded (nothing is captured, so there is no size to cap and no stderr to show). The page learns
  how it ended from a closed vocabulary: `ok`, `failed` with the exit code, `timed-out` or `could-not-start`.
- It has a timeout (`TIMEOUT_S`); on expiry the process group is killed.
- One run at a time: a second start while one runs starts nothing (`already-running`).
- A minimum interval (`MIN_INTERVAL_S`) between the starts of two runs (`too-soon`, with the seconds to wait).

The route that calls this (orchard/webui.py) is a POST, so it needs the page's token and an origin that matches the host, and it is refused
while the UI listens on the network (`--lan`), where anyone who reaches the port could start it. The result is the cache file the panel reads
anyway: the page asks `GET /api/updates` again when the run has ended.
"""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

ARGV = [sys.executable, "-m", "orchard.cli", "updates", "--refresh"]
# `orchard` is not installed in the venv: `python -m orchard.cli` imports it from the working directory, so the child starts in the directory
# that holds the package, whatever directory the web server was started in (found in the review of PR 683).
CWD = str(pathlib.Path(__file__).resolve().parent.parent)
TIMEOUT_S = 240.0
MIN_INTERVAL_S = 60.0
# What an agent shell gets (orchard/agent.py ENV_ALLOW, held equal or smaller by a test), plus HOME and XDG_CACHE_HOME for the cache file
# and ORCHARD_TT_METAL_DIR for the tt-metal tree the command inspects. No name here carries a credential.
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TERM", "HOME", "XDG_CACHE_HOME", "ORCHARD_TT_METAL_DIR")
OUTCOMES = ("ok", "failed", "timed-out", "could-not-start")
MESSAGES = {
    "ok": "The check finished. The table shows the new rows.",
    "failed": "The check failed (exit {code}). Run `tt-orchard updates --refresh` in a terminal to see why.",
    "timed-out": "The check took too long and was stopped. Run `tt-orchard updates --refresh` in a terminal.",
    "could-not-start": "The check could not be started on this machine.",
}


class RefreshJob:
    """One `tt-orchard updates --refresh` at a time, in a thread; `state()` is what the page polls."""

    def __init__(self, *, popen=subprocess.Popen, clock=time.time, environ=None, timeout_s: float = TIMEOUT_S,
                 min_interval_s: float = MIN_INTERVAL_S):
        self._popen, self._clock = popen, clock
        self._environ = os.environ if environ is None else environ
        self._timeout_s, self._min_interval_s = timeout_s, min_interval_s
        self._lock = threading.Lock()
        self._running = False
        self._last_start: float | None = None
        self._finished: float | None = None
        self._outcome: str | None = None
        self._exit: int | None = None

    def _child_env(self) -> dict:
        return {k: self._environ[k] for k in ENV_ALLOW if k in self._environ}

    def start(self) -> str:
        """"started", "already-running" or "too-soon". Starts at most one process; takes nothing from the caller."""
        with self._lock:
            if self._running:
                return "already-running"
            now = self._clock()
            if self._last_start is not None and now - self._last_start < self._min_interval_s:
                return "too-soon"
            self._running, self._last_start = True, now
            self._finished = self._outcome = self._exit = None
        threading.Thread(target=self._work, daemon=True).start()
        return "started"

    def _work(self) -> None:
        outcome, code = "ok", None
        try:
            proc = self._popen(list(ARGV), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=self._child_env(), cwd=CWD, start_new_session=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            outcome = "could-not-start"
        else:
            try:
                code = proc.wait(timeout=self._timeout_s)
                if code != 0:
                    outcome = "failed"
            except subprocess.TimeoutExpired:
                outcome = "timed-out"
                self._kill(proc)
            except Exception:                        # a failure of the wait itself is reported like a failed run
                outcome, code = "failed", None
                self._kill(proc)
        with self._lock:
            self._running, self._finished, self._outcome, self._exit = False, self._clock(), outcome, code

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)       # the command's own children (git, tt-model) are in its session
        except (OSError, ProcessLookupError, TypeError):
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    def state(self) -> dict:
        with self._lock:
            now = self._clock()
            wait = 0 if self._last_start is None else max(0, int(round(self._min_interval_s - (now - self._last_start))))
            message = None
            if self._outcome == "failed":
                message = MESSAGES["failed"].format(code=self._exit if self._exit is not None else "unknown")
            elif self._outcome:
                message = MESSAGES[self._outcome]
            return {"running": self._running, "started": self._last_start, "finished": self._finished, "outcome": self._outcome,
                    "exit": self._exit, "message": message, "retry_in_s": wait}
