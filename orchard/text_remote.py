# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runs the numeric verbs of the text stages (orchard/text_cli.py) in the configured interpreter, from a supervisor that needs only the standard library.
`call` starts the interpreter in its own session, so a timeout or an interrupt kills the verb and everything it started."""
from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

from orchard.text_runtime import TextStageError

CLI = Path(__file__).with_name("text_cli.py")


def call(python: str, verb: str, args: dict, *, timeout_s: float, env: dict | None = None) -> object:
    """The `result` of the verb. Raises TextStageError with the verb's own message when it reports an error, and with the end of its error output when it dies
    without an answer."""
    request = json.dumps({"verb": verb, "args": args})
    try:
        proc = subprocess.Popen([python, "-I", str(CLI)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=env if env is not None else dict(os.environ), start_new_session=True)
    except OSError as exc:
        raise TextStageError(f"cannot start {python}: {exc}") from exc
    try:
        out, err = proc.communicate(request, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill(proc)
        raise TextStageError(f"the text verb {verb} ran past {timeout_s:.0f} s and was stopped") from None
    except BaseException:
        _kill(proc)
        raise
    lines = [line for line in out.splitlines() if line.strip()]
    try:
        answer = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise TextStageError(f"the text verb {verb} gave no answer (exit {proc.returncode}): {err.strip()[-300:]}") from None
    if not isinstance(answer, dict):
        raise TextStageError(f"the text verb {verb} gave an answer that is not an object")
    if "error" in answer:
        raise TextStageError(str(answer["error"]))
    if "result" not in answer:
        raise TextStageError(f"the text verb {verb} gave no result")
    return answer["result"]


def _kill(proc, grace_s: float = 10.0) -> None:
    """SIGTERM to the verb's session first (the verb turns it into an exception whose cleanup kills the CPU runs it started in sessions of their own), then
    SIGKILL after the grace period."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.communicate()
