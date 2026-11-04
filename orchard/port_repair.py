# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Repair rounds for a full-port stage 2 (issue 420).

The port check (orchard/skills/full-port-templates/port_check.py) can fail in seconds on a mistake that its failure text names: a weight key the
port expects and the checkpoint lacks, an import that is missing, a layer index off by one. Before this module the stage ended at the first
failed test and the next attempt began again from a fresh prepare step. A person porting a model fixes the line and runs the 9-second check
again. This module holds the pieces the supervisor needs to let the coder do that, a bounded number of times, without giving the coder a way to
change what the check reports:

- `read_failure` says whether the supervisor's own record of the failed test shows a failure of the PORT (the check's structured failure line,
  kind `port`) and not of the run around it (kind `setup`, a timeout, any other exit code). It trusts nothing the coder can write: the test
  record and the output file are the supervisor's, the output file must match the sha256 in that record, and the failure line is only read
  where the template prints it, at the start of a line (the port's own output carries the prefix `port| `).
- `rounds_used` counts the rounds a stage attempt has already used from the ledger, so a resume does not start the count again.
- `fingerprint` and `changed` show what a repair step wrote.

The supervisor side is `Supervisor._port_repair_rounds` (orchard/supervisor.py). A round is a repair step by the coder (it edits only the port
directory), then the same check on the chips, with the record written by the supervisor as for any test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import json

from orchard import full_port, safeio
from orchard.skillsfresh import scrub_secrets
from orchard.stack_pins import redact
from orchard.stages import MOVE_ASIDE_DECISIONS

DEFAULT_ROUNDS = 3                          # `--port-repair-rounds`: repair rounds per stage attempt
MAX_ROUNDS = 10                             # the largest value the option takes: each round is a model step and a chip test
DECISION = "port repair round"              # the ledger decision written BEFORE a round's repair step starts; `round` and `of` say which
TESTED = "port repair tested"              # the ledger decision after each hardware test: `sources` is the digest of the port directory it ran against
RERUN = "port repair rerun started"         # the ledger decision written just BEFORE a round's hardware rerun: `round`, and `record`, the seal of the failed test's record
STOPPED = "port repair rounds ended"        # the ledger decision that says why the rounds stopped (spent, unchanged, off)
FAILURE_TOKEN = "PORT_CHECK_FAILURE "       # starts the structured line the template prints after `FAILED: <text>` (it starts with the template's MARK)
OUTPUT = "evidence/hw-test-output.txt"      # the supervisor's capture of the test's output, relative to the stage directory
MARKER = "test-result.json"                 # the supervisor's record of the test, relative to the stage directory
KINDS = ("port", "setup")                   # `port`: the port did not do what a request asked. `setup`: the run around it was wrong
PORT_DIR = "port"                           # the directory the coder writes, relative to the stage directory


def check_rounds(value) -> int:
    """The repair-round count from an option or a ledger value: a whole number from 0 (the old behaviour: no rounds) to MAX_ROUNDS."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if type(value) is not int or not 0 <= value <= MAX_ROUNDS:
        raise ValueError(f"the number of port repair rounds must be a whole number from 0 to {MAX_ROUNDS}; got {value!r}")
    return value


def parse_failure(text) -> dict | None:
    """The last well-formed failure line in the check's output, as {"kind", "op"}, or None. Only a line that STARTS with the token counts: the
    template prints its own lines unprefixed and every line of the port's worker with `port| `, so a port cannot write one."""
    if not isinstance(text, str):
        return None
    found = None
    for line in text.splitlines():
        if not line.startswith(FAILURE_TOKEN):
            continue
        try:
            doc = json.loads(line[len(FAILURE_TOKEN):])
        except ValueError:
            continue
        if not isinstance(doc, dict) or doc.get("kind") not in KINDS:
            continue
        op = doc.get("op")
        if op is not None and not isinstance(op, str):
            continue
        found = {"kind": doc["kind"], "op": op}
    return found


def read_failure(run_dir, stage_dir) -> dict | None:
    """{"kind": "port", "op", "returncode": 2} when the supervisor's record of stage 2's test shows a failure of the port, else None.

    Needs all of: a test record that is a JSON object; an exit code of exactly 2 (a bool or a float is not one) with `timed_out` false (the
    check exits 2 only for a failure it named); an `output` entry in the record that names the stage's output file; that file read through the
    bounded reader with the sha256 the record holds; and a failure line of kind `port` in it. Anything else is None, so a record or an output
    file the coder rewrote earns no round."""
    stage_dir = Path(stage_dir)
    try:
        data = safeio.read_json(stage_dir / MARKER, max_bytes=full_port.JSON_MAX, expect=dict)
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    code = data.get("returncode")
    if type(code) is not int or code != 2 or data.get("timed_out") is not False:
        return None
    named = data.get("output")
    out = stage_dir / OUTPUT
    try:
        raw = safeio.read_bytes(out, max_bytes=full_port.JSON_MAX)         # one read: the digest and the text come from the same bytes
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    rel = os.path.relpath(os.path.realpath(out), os.path.realpath(run_dir))
    if not isinstance(named, dict) or named.get("path") != rel or named.get("sha256") != hashlib.sha256(raw).hexdigest():
        return None
    failure = parse_failure(raw.decode("utf-8", "replace"))
    if failure is None or failure["kind"] != "port":
        return None
    return {**failure, "returncode": code}


def rounds_used(entries: list[dict], n: int) -> int:
    """The repair rounds stage `n`'s current attempt has used. The count starts again at a stage_start that is not a resume and at a decision
    that moved the stage directory aside (the attempt's files are gone); a resume's stage_start keeps it."""
    used = 0
    for e in entries:
        if e.get("stage") != n:
            continue
        data = e.get("data") or {}
        if e.get("event") == "stage_start" and data.get("resumed") is False:
            used = 0
        elif e.get("event") == "decision" and data.get("decision") in MOVE_ASIDE_DECISIONS:
            used = 0
        elif e.get("event") == "decision" and data.get("decision") == DECISION:
            used += 1
    return used


def stopped_noted(entries: list[dict], n: int) -> bool:
    """Has the current attempt of stage `n` already written the `port repair rounds ended` decision? A resume that reaches the same end
    does not write it again."""
    seen = False
    for e in entries:
        if e.get("stage") != n:
            continue
        data = e.get("data") or {}
        if e.get("event") == "stage_start" and data.get("resumed") is False:
            seen = False
        elif e.get("event") == "decision" and data.get("decision") in MOVE_ASIDE_DECISIONS:
            seen = False
        elif e.get("event") == "decision" and data.get("decision") == STOPPED:
            seen = True
    return seen


LARGE_BLOCK = 1024 * 1024  # bytes sampled from each end of a file over the read cap


def fingerprint(directory) -> dict[str, str]:
    """{relative path: sha256, or "link: <target>", or "unreadable: <reason>"} for every file under `directory` (none when it is missing).
    A link is a difference, never a file that is followed: the check refuses a link under the port directory."""
    root = Path(directory)
    out: dict[str, str] = {}
    if not root.is_dir() or root.is_symlink():
        return out
    for here, dirs, files in os.walk(root):
        # bytecode a coder's own python run leaves behind is not an edit (issue 453); a link named __pycache__ is still listed below
        dirs[:] = [d for d in dirs if d != "__pycache__" or os.path.islink(os.path.join(here, d))]
        dirs.sort()
        for name in sorted(files + [d for d in dirs if os.path.islink(os.path.join(here, d))]):
            path = Path(here) / name
            rel = os.path.relpath(path, root)
            if path.is_symlink():
                out[rel] = f"link: {os.readlink(path)}"
                continue
            try:
                out[rel] = hashlib.sha256(safeio.read_bytes(path, max_bytes=full_port.JSON_MAX)).hexdigest()
            except safeio.TooLarge:
                # a file over the read cap (a weights file in port/) is told apart by its size, modification time and first and last block,
                # so an edit to it is a change; it is a notice of a change, never a proof of none (issue 453)
                try:
                    size, mtime_ns, head, tail = safeio.sample_blocks(path, block=LARGE_BLOCK)
                    out[rel] = f"large: size={size} mtime_ns={mtime_ns} head={hashlib.sha256(head).hexdigest()} tail={hashlib.sha256(tail).hexdigest()}"
                except (safeio.SafeReadError, OSError, ValueError) as exc:
                    out[rel] = f"unreadable: {type(exc).__name__}"
            except (safeio.SafeReadError, OSError, ValueError) as exc:
                out[rel] = f"unreadable: {type(exc).__name__}"
    return out


def changed(before: dict, after: dict) -> list[str]:
    """The paths added, removed or changed between two fingerprints, sorted."""
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def safe_text(raw: str, homes: tuple = ()) -> str:
    """The check's output with home paths, URL user information and token-shaped words removed from the WHOLE text. Every cut (the failure line
    in the ledger, the tail in the prompt) is taken from this, never from `raw`: a cut between a password and its `@` would hide the user
    information from the pattern (review of PR 425)."""
    return scrub_secrets(redact(raw, homes))


def digest(sources: dict) -> str:
    """One sha256 for a `fingerprint` of the port directory."""
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode("utf-8")).hexdigest()


def tested_digest(entries: list[dict], n: int) -> str | None:
    """The digest of the port directory the stage's latest hardware test ran against, from the `port repair tested` decisions of the current
    attempt (same boundaries as `rounds_used`), or None. A resume that finds the directory different from this one knows the coder edited it
    after the test, and that the edit still has to be tested."""
    found = None
    for e in entries:
        if e.get("stage") != n:
            continue
        data = e.get("data") or {}
        if e.get("event") == "stage_start" and data.get("resumed") is False:
            found = None
        elif e.get("event") == "decision" and data.get("decision") in MOVE_ASIDE_DECISIONS:
            found = None
        elif e.get("event") == "decision" and data.get("decision") == TESTED and isinstance(data.get("sources"), str):
            found = data["sources"]
    return found


def _attempt_events(entries: list[dict], n: int) -> list[tuple]:
    """The (event, data) pairs of stage `n`'s current attempt: after the last stage_start that is not a resume and the last decision that moved
    the stage directory aside (the same boundaries as `rounds_used`)."""
    out: list[tuple] = []
    for e in entries:
        if e.get("stage") != n:
            continue
        data = e.get("data") or {}
        if (e.get("event") == "stage_start" and data.get("resumed") is False) or (
                e.get("event") == "decision" and data.get("decision") in MOVE_ASIDE_DECISIONS):
            out = []
        out.append((e.get("event"), data))
    return out


def rerun_inflight(entries: list[dict], n: int) -> dict | None:
    """The `port repair rerun started` decision of stage `n`'s current attempt that no `port repair tested` decision has followed, or None.

    A rerun opens the test's output file fresh, so from then until the record is written the old record's output hash does not match and
    `read_failure` says nothing. This entry is what a resume trusts instead: the ledger is the supervisor's own. Its `record` is the seal
    of the failed test's record at that moment; a record that no longer matches it was written by the rerun."""
    open_ = None
    for event, data in _attempt_events(entries, n):
        if event != "decision":
            continue
        if data.get("decision") == RERUN:
            open_ = data
        elif data.get("decision") == TESTED:
            open_ = None
    return open_


def rerun_recorded(entries: list[dict], n: int) -> dict | None:
    """The supervisor's own `hardware test` evidence entry that follows the in-flight rerun's `port repair rerun started` decision, or None.

    `Supervisor._run_test` writes this entry (the only one with `returncode` and `timed_out`) right after it writes the test's record, so an
    entry after the rerun decision says the rerun ran and was recorded. The record's bytes cannot say it: an edit that leaves the same failure
    output, with both durations rounding to the same millisecond, gives a record byte for byte equal to the sealed one (review of PR 432)."""
    started, found = False, None
    for event, data in _attempt_events(entries, n):
        if event == "decision" and data.get("decision") == RERUN:
            started, found = True, None
        elif event == "decision" and data.get("decision") == TESTED:
            started, found = False, None
        elif started and event == "evidence" and data.get("what") == "hardware test" and "returncode" in data and "timed_out" in data:
            found = data
    return found


def output_kept(entries: list[dict], n: int, k: int) -> bool:
    """Has this attempt already copied round `k`'s failed output to the evidence directory and recorded it? A rerun that was cut short has
    replaced the output file, so a resume must not copy it over the copy the repair step was shown."""
    return any(event == "evidence" and data.get("what") == "port repair round output" and data.get("round") == k
               for event, data in _attempt_events(entries, n))


def failed_line(text: str, limit: int = 300) -> str:
    """The check's last `FAILED: <text>` line (unprefixed, so the template's own), cut to `limit` characters, for a ledger entry."""
    last = ""
    for line in text.splitlines():
        if line.startswith("FAILED: "):
            last = line[len("FAILED: "):]
    return last[:limit]


def tail(text: str, limit: int = 1500) -> str:
    """The last `limit` characters of a failure text, for a prompt and the ledger."""
    return text if len(text) <= limit else "[cut]\n" + text[-limit:]
