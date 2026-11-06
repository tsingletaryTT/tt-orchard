# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixed-name tools for the coder of a full-port stage (issue 505). The supervisor copies this file to stages/<N>/stage_tools.py (next to
port_check.py) and the task text names it. Run it from anywhere:

    python3 stages/2/stage_tools.py last-check            which check failed, where, the numbers and the evidence file

It reads only files in its own stage directory, and writes nothing there except what a tool says it writes. It is standard library only: it must
run in the coder's shell, where the supervisor's own package is not importable. Everything it prints is advice built from files the coder could
also have edited; the supervisor's verdict is the gate's, taken from its own ledger. Each tool says what it read.

Why fixed names. In run 18 of the tiny Qwen4 rehearsal the coder spent its repair turns on one-off `cat | head`, `grep` and `python3 -c` lines. Each
was a new command for the watchdog to guess about. A tool call has a name and arguments, which the watchdog (orchard/watchdog.py `command_shape`)
counts exactly.
"""
import json
import os
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # stages/<N>, or stage 4's entry directory
OUTPUT_REL = "evidence/hw-test-output.txt"           # the supervisor's capture of the check's output
MARKER_REL = "test-result.json"                      # the supervisor's record of the test
RECORD_REL = "evidence/port-check.json"              # the check's own record
OUTPUT_MAX = 2_000_000                               # bytes read from the output file; larger is refused
JSON_MAX = 1_000_000
TOP1_MIN = 0.85                                      # the gate's bar (orchard/full_port.py TOP1_MIN; a test keeps the two equal)
FAILURE_TOKEN = "PORT_CHECK_FAILURE "                # starts the check's structured failure line (orchard/port_repair.py)
MAX_LINE = 300
ROWS = 8
ADVICE_NOTE = "advice from files in this directory; the gate's verdict is the supervisor's"


def read_file(path: Path, cap: int):
    """(bytes, None) or (None, reason). A regular file only: a link, a FIFO or a device is refused without being opened for blocking, and a file
    over `cap` is refused. The size is checked on the open descriptor, so a file swapped after a check is still bounded."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"cannot be opened ({exc.__class__.__name__}: a link or an unreadable file)"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
        if st.st_size > cap:
            return None, f"is larger than {cap} bytes"
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(cap + 1)
        if len(data) > cap:
            return None, f"is larger than {cap} bytes"
        return data, None
    finally:
        os.close(fd)


def read_json(path: Path, expect=dict):
    """(value, None) or (None, reason): JSON of the expected type from a regular file."""
    data, why = read_file(path, JSON_MAX)
    if data is None:
        return None, why
    try:
        value = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        return None, "is not JSON"
    if not isinstance(value, expect):
        return None, f"holds {type(value).__name__}, not {expect.__name__}"
    return value, None


def clean(text, limit: int = MAX_LINE) -> str:
    """One printable line: control characters removed, cut to `limit`."""
    text = "".join(c if c.isprintable() else " " for c in str(text))
    return text if len(text) <= limit else text[:limit - 3] + "..."


def rel(path: Path) -> str:
    """The path as the task text names it: relative to the run directory when the stage sits under `stages/`."""
    parts = path.parts
    if "stages" in parts:
        return "/".join(parts[len(parts) - 1 - parts[::-1].index("stages"):])
    return str(path)


def whole(x) -> bool:
    return type(x) is int


def ids_ok(x) -> bool:
    return isinstance(x, list) and 0 < len(x) <= 100_000 and all(whole(i) and 0 <= i < 10**12 for i in x)


def failure_line(text: str):
    """The last well-formed structured failure, {"kind", "op"}, and the last `FAILED:` message, from the check's output."""
    found, message = None, None
    for line in text.splitlines():
        if line.startswith("FAILED: "):
            message = line[len("FAILED: "):]
        if line.startswith(FAILURE_TOKEN):
            try:
                doc = json.loads(line[len(FAILURE_TOKEN):])
            except ValueError:
                continue
            if isinstance(doc, dict) and doc.get("kind") in ("port", "setup") and (doc.get("op") is None or isinstance(doc.get("op"), str)):
                found = doc
    return found, message


def where(position: int) -> str:
    return ("the token after the prompt: the prefill logits are wrong" if position == 0 else
            "a decode step: the prefill token matched, so look at the decode path, the cache position and the state updates")


def last_check(stage: Path) -> list[str]:
    """The summary of the last check in `stage`, as lines (fewer than 40)."""
    import hashlib
    stage = Path(stage)
    lines: list[str] = []
    marker, why = read_json(stage / MARKER_REL)
    if marker is None:
        lines.append("no check has run in this directory yet." if why == "missing" else f"not checked: {MARKER_REL} {why}.")
        return lines
    code = marker.get("returncode")
    if not whole(code):
        return [f"not checked: {MARKER_REL} has no whole-number returncode."]
    command = marker.get("command") if isinstance(marker.get("command"), str) else "?"
    lines.append(f"last check: {clean(command, 100)} (exit {code}{', timed out' if marker.get('timed_out') is True else ''})")
    data, why = read_file(stage / OUTPUT_REL, OUTPUT_MAX)
    text = data.decode("utf-8", "replace") if data is not None else ""
    if data is None:
        lines.append(f"not checked: {OUTPUT_REL} {why}.")
    else:
        out = marker.get("output")
        want = out.get("sha256") if isinstance(out, dict) else None
        if want != hashlib.sha256(data).hexdigest():
            lines.append(f"warning: {OUTPUT_REL} does not match the sha256 the supervisor recorded for it; it may have changed since the test.")
    if code == 0 and marker.get("timed_out") is not True:
        lines += numerics(stage)
    elif data is not None:
        kind_op, message = failure_line(text)
        if kind_op is not None:
            kind, op = kind_op["kind"], kind_op["op"]
            lines.append(f"result: FAILED, kind {kind}" + (f", at the '{op}' request" if op else ", before any request reached the port"))
            if kind == "setup":
                lines.append("  the run around the port was wrong (chips, stage 1 files, runtime), not the port: an edit to the port cannot fix it.")
        else:
            lines.append("result: FAILED, with no structured failure line (a crash or a timeout before the check could say what failed).")
        lines.append("message: " + clean(message if message is not None else "(none printed)"))
        lines.append(f"evidence: {rel(stage / OUTPUT_REL)}")
    lines.append(f"({ADVICE_NOTE})")
    return lines


def numerics(stage: Path) -> list[str]:
    """The numbers of a check that ran to the end, from its record: recomputed from the two id lists."""
    record, why = read_json(stage / RECORD_REL)
    path = rel(stage / RECORD_REL)
    if record is None:
        return [f"not checked: {RECORD_REL} {why}, so the agreement cannot be read."]
    ref, chip = record.get("reference_ids"), record.get("chip_ids")
    if not ids_ok(ref) or not ids_ok(chip) or len(ref) != len(chip):
        return [f"not checked: {RECORD_REL} has no two equal-length lists of whole-number ids."]
    n = len(ref)
    rows = [(i, a, b) for i, (a, b) in enumerate(zip(ref, chip)) if a != b]
    agreement = (n - len(rows)) / n
    lines = []
    verdict = "meets" if agreement >= TOP1_MIN else "is BELOW"
    lines.append(f"result: ran to the end. top1_agreement {agreement:.4f} ({n - len(rows)} of {n} ids match) {verdict} the gate's {TOP1_MIN}.")
    stated = record.get("top1_agreement")
    if isinstance(stated, (int, float)) and not isinstance(stated, bool) and abs(stated - agreement) > 1e-6:
        lines.append(f"warning: the record's own top1_agreement {stated} differs from the ids' {agreement:.4f}; the ids are used.")
    if rows:
        lines.append(f"{len(rows)} of {n} ids differ; first mismatch at position {rows[0][0]}: {where(rows[0][0])}.")
        lines.append("position | reference id | port id")
        lines += [f"{p} | {a} | {b}" for p, a, b in rows[:ROWS]]
        if len(rows) > ROWS:
            lines.append(f"({len(rows) - ROWS} more not listed)")
        lines.append("this output names no layer or feature; `layer-diff` compares the port with the reference layer by layer.")
    lines.append(f"evidence: {path}")
    return lines


COMMANDS = {"last-check": "which check failed, where, the numbers and the evidence file"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python3 stage_tools.py <tool> [arguments]\n" + "\n".join(f"  {k}: {v}" for k, v in COMMANDS.items()))
        return 2
    if argv[0] == "last-check":
        print("\n".join(last_check(HERE)))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
