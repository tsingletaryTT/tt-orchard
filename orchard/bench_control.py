# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The base control for a benchmark: its run.sh is the model's own run.sh with the model path swapped, and its board comes from `gozer status`.

    python3 -m orchard.bench_control build --model-run-sh RUN.SH --model-dir DIR --base-snapshot SNAP --base-id ORG/NAME@REV --out base-run.sh
    python3 -m orchard.bench_control board --chips 2 [--chips 2] [--exclude BOARD] [--wait-s 120] [--who W --reason R]

A comparison of a fine-tune with its base is fair only when both are served the same way. The first control for a model (bytkim) was served with the drafter on
and device sampling while the fine-tune was served with the drafter off and host sampling, and the two could not be compared. `build` writes the control's
run.sh from the model's run.sh by replacing the model's directory with the base snapshot wherever the script uses it as a whole value (the `HF_MODEL` line
and the `--model` flag are required; a path inside the directory, such as a cache, is left alone), quoted for the place it goes. It
then reads both files again and refuses unless the only lines that differ are lines where that replacement alone explains the difference. It writes a record
(`<out>.control.json`) with both hashes, the swapped line numbers and the drafter and sampling mode of each script; `bench_session --control-record` copies
that record into the session's summary.json, so a reader can check the comparison without the scripts. The session also refuses a record whose
`base_snapshot` is not its `--model-dir`, because the runtime reads that directory before the script's own path.

`board` reads `gozer status --json` and names a board whose chips are all FREE. `gozer run --exact N` selects the unit that holds chip N, which is a whole board
on this machine, so the command gives the device index of the board's first chip. Two sessions asked for together get two different boards. A board that has just
been released shows BUSY-UNTRACKED for about a minute (its reset opens every device), so `--wait-s` polls until one is free.

Neither command opens a chip. Two launchers that call `board` at the same moment can still pick the same board; the second `gozer run` then queues behind the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 10.0          # choice: a released board takes about a minute to settle, so a ten-second poll finds it within a poll of the end
SUFFIX = ".control.json"


class ControlError(Exception):
    """The control script cannot be built from the model's script, or is not comparable with it."""


class NoFreeBoard(Exception):
    """gozer shows no board with all its chips free, or its status cannot be read."""


_OPERATORS = ";|&()<>"          # outside quotes these end a word, so `--model /path)` and `"$@")` are read as words
_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=|--[A-Za-z][\w-]*=")   # NAME= or --flag= at the front of a word; the value is what follows


def _words(text: str) -> list[tuple[int, int]]:
    """The (start, end) of each shell word in the script: split on unquoted blanks, newlines and operators, with '...' and "..." read as one piece, a backslash
    protecting the next character, `#` starting a comment where a word could start, and a backslash before a newline joining lines. A quote that is never
    closed is an error, because the script cannot be read safely."""
    def line(pos: int) -> int:
        return text.count("\n", 0, pos) + 1
    n, i, spans = len(text), 0, []
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == "\\" and text[i + 1:i + 2] == "\n":
            i += 2
        elif c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in _OPERATORS:
            i += 1
        else:
            start = i
            while i < n and text[i] not in " \t\r\n" and text[i] not in _OPERATORS:
                c = text[i]
                if c == "\\":
                    i = min(i + 2, n)
                elif c == "'":
                    j = text.find("'", i + 1)
                    if j < 0:
                        raise ControlError(f"line {line(i)} opens a single quote that is never closed, so the script cannot be read safely")
                    i = j + 1
                elif c == '"':
                    j = i + 1
                    while j < n and text[j] != '"':
                        j += 2 if text[j] == "\\" else 1
                    if j >= n:
                        raise ControlError(f"line {line(i)} opens a double quote that is never closed, so the script cannot be read safely")
                    i = j + 1
                else:
                    i += 1
            spans.append((start, i))
    return spans


def _unquote(raw: str) -> str:
    """What the shell makes of one word's quoting: the characters of the word with quotes and protecting backslashes taken out."""
    out, i, n = [], 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "'":
            j = raw.index("'", i + 1)
            out.append(raw[i + 1:j])
            i = j + 1
        elif c == '"':
            i += 1
            while raw[i] != '"':
                if raw[i] == "\\" and raw[i + 1] in '$`"\\\n':
                    out.append("" if raw[i + 1] == "\n" else raw[i + 1])
                    i += 2
                else:
                    out.append(raw[i])
                    i += 1
            i += 1
        elif c == "\\":
            out.append("" if raw[i + 1:i + 2] == "\n" else raw[i + 1:i + 2])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _references(text: str, path: str) -> list[tuple[int, int, str]]:
    """Each place the script uses `path` as a whole value: a word, or the value of NAME= or --flag=, that the shell turns into exactly `path`. A path inside
    it (`path/cache`), a longer path that contains it, and the same text in a comment are not references. (start, end, kind) with kind `HF_MODEL`, `--model`
    (the flag, with `=` or as the word before) or `other`."""
    refs, prev = [], None
    for s, e in _words(text):
        m = _PREFIX.match(text[s:e])
        vs = s + (m.end() if m else 0)
        if _unquote(text[vs:e]) == path:
            name = m.group(0) if m else None
            kind = "HF_MODEL" if name == "HF_MODEL=" else "--model" if name == "--model=" or (name is None and prev == "--model") else "other"
            refs.append((vs, e, kind))
        prev = _unquote(text[s:e])
    return refs


def _named(text: str, path: str) -> bool:
    """Whether some word still holds `path` as a whole path (not as the front or the end of a longer one), for example inside a quoted string. A control
    that still names the model's directory somewhere it cannot be swapped would load the model."""
    hole = re.compile(r"(?<![\w./\-])" + re.escape(path) + r"(?![\w./\-])")
    return any(hole.search(text[s:e]) for s, e in _words(text))


def _quoted_like(old: str, new: str) -> str:
    """`new` written so the shell reads it back as one word, in the quoting style of `old` (the model's own spelling) where that style can carry it."""
    if re.fullmatch(r'"(?:[^"\\]|\\.)*"', old, re.S):
        return '"' + re.sub(r'(["\\$`])', r"\\\1", new) + '"'
    if re.fullmatch(r"'[^']*'", old):
        return "'" + new.replace("'", "'\"'\"'") + "'"
    return shlex.quote(new)


def _rewrite(model_text: str, model_dir: str, base_snapshot: str) -> str:
    """The model's run.sh with each whole-value use of `model_dir` replaced by `base_snapshot`, quoted for where it stands. Refuses when the script does
    not name the model in both the `HF_MODEL` line and the `--model` flag, because a control that swapped one of them would serve one model and load
    another, and when the model's directory is still named somewhere the swap cannot reach."""
    if "\n" in base_snapshot or "\0" in base_snapshot:
        raise ControlError("the base snapshot's name has a newline in it")
    refs = _references(model_text, model_dir)
    if not refs and not _named(model_text, model_dir):
        raise ControlError(f"the model's run.sh does not contain {model_dir}")
    if not any(k == "HF_MODEL" for _, _, k in refs):
        raise ControlError(f"the model's run.sh has no HF_MODEL line that names {model_dir}")
    if not any(k == "--model" for _, _, k in refs):
        raise ControlError(f"the model's run.sh has no --model flag that names {model_dir}")
    text = model_text
    for s, e, _ in reversed(refs):
        text = text[:s] + _quoted_like(text[s:e], base_snapshot) + text[e:]
    if _named(text, model_dir):
        raise ControlError(f"the script still names {model_dir} somewhere that is not a plain value (for example inside a longer string), "
                           "so it cannot be swapped safely")
    return text


def differing_lines(a: str, b: str) -> list[tuple[int, str, str]]:
    """(line number, line in a, line in b) for each line that differs; the texts must have the same number of lines."""
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        raise ControlError(f"the two scripts have {len(la)} and {len(lb)} lines")
    return [(i + 1, x, y) for i, (x, y) in enumerate(zip(la, lb)) if x != y]


def control_run_sh(model_text: str, model_dir: str, base_snapshot: str) -> str:
    """The model's run.sh with `model_dir` replaced by `base_snapshot` wherever the script uses it as a whole value (see `_rewrite`)."""
    if model_dir == base_snapshot:
        raise ControlError("the base snapshot is the same directory as the model's")
    return _rewrite(model_text, model_dir, base_snapshot)


def verify_control(model_text: str, control_text: str, model_dir: str, base_snapshot: str) -> list[str]:
    """Why the control is not comparable with the model's script; an empty list when it is. The control must be what `_rewrite` makes of the model's
    script, line for line, so every difference is the path swap and the swap is quoted for its place."""
    try:
        diff = differing_lines(model_text, control_text)
        expected = _rewrite(model_text, model_dir, base_snapshot)
        off = differing_lines(expected, control_text)
    except ControlError as exc:
        return [str(exc)]
    if not diff:
        return ["no line differs, so the control serves the model again"]
    problems = [f"line {n} differs by more than the model path" for n, _, _ in off]
    if _named(control_text, model_dir):
        problems.append(f"the control still names {model_dir}")
    return problems


def serving_facts(text: str) -> dict:
    """The drafter and sampling mode a run.sh asks for: the drafter is off when `QWEN36_DRAFTER` is exported empty, and sampling is on the device when
    the script sets `sample_on_device_mode`."""
    m = re.search(r"^\s*export QWEN36_DRAFTER=(.*)$", text, re.M)
    if m is None:
        drafter = "not set"
    else:
        drafter = "off" if m.group(1).strip().strip("\"'") == "" else "on"
    return {"drafter": drafter, "sampling": "device" if "sample_on_device_mode" in text else "host"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_control(*, model_run_sh, model_dir: str, base_snapshot: str, base_id: str, out) -> dict:
    """Write the control run.sh and its record beside it; returns the record. Nothing is written when the control cannot be built or fails its own check."""
    model_path, out = Path(model_run_sh), Path(out)
    model_text = model_path.read_text(encoding="utf-8")
    text = control_run_sh(model_text, model_dir, base_snapshot)
    problems = verify_control(model_text, text, model_dir, base_snapshot)
    if problems:
        raise ControlError("; ".join(problems))
    sm, sc = serving_facts(model_text), serving_facts(text)
    record = {"built_from": str(model_path), "built_from_sha256": _sha(model_text.encode()), "control": str(out), "control_sha256": _sha(text.encode()),
              "model_dir": model_dir, "base_snapshot": base_snapshot, "base_id": base_id,
              "swapped_lines": [n for n, _, _ in differing_lines(model_text, text)], "serving_model": sm, "serving_control": sc, "same_serving": sm == sc}
    out.write_text(text, encoding="utf-8")
    Path(str(out) + SUFFIX).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def _boards(status: dict) -> dict:
    boards: dict = {}
    for c in status.get("chips") or []:
        boards.setdefault(c.get("board"), []).append(c)
    return {b: sorted(cs, key=lambda c: c["dev_index"]) for b, cs in boards.items()}


def plan_boards(status: dict, wants: list[int], exclude=()) -> list[dict]:
    """One board per entry of `wants` (the chips each session needs), from the boards whose chips are all FREE and that hold at least that many chips.
    Each plan is {board, chips, exact}; `exact` is the device index of the board's first chip, which is what `gozer run --exact` takes. The plans come back
    in the order of `wants`. Each request takes the smallest free board that holds it (the lowest one on a tie), so a small request does not use up the only
    board a larger one fits; with that rule the requests can be placed in any order and an assignment is found whenever one exists."""
    boards = _boards(status)
    if not boards:
        raise NoFreeBoard("gozer status lists no chips")
    free, seen = [], []
    for name, cs in sorted(boards.items(), key=lambda kv: kv[1][0]["dev_index"]):
        states = sorted({c.get("state", "?") for c in cs})
        if name in exclude:
            continue
        if states == ["FREE"]:
            free.append((name, cs))
        else:
            seen.append(f"board {name}: {', '.join(states)}")
    biggest = max((len(cs) for cs in boards.values()), default=0)
    if any(w > biggest for w in wants):
        raise NoFreeBoard(f"a session asks for more chips than one board has ({max(wants)} against {biggest})")
    if len(free) < len(wants):
        saw = "; ".join(seen) or "every board is excluded"
        raise NoFreeBoard(f"{len(wants)} session{'s' if len(wants) != 1 else ''} asked for {len(wants)} board{'s' if len(wants) != 1 else ''} "
                          f"and {len(free)} free ({saw})")
    left = sorted(free, key=lambda b: (len(b[1]), b[1][0]["dev_index"]))
    plan: list = [None] * len(wants)
    for i in range(len(wants)):
        fit = next((b for b in left if len(b[1]) >= wants[i]), None)
        if fit is None:
            sizes = ", ".join(f"{n} ({len(cs)} chip{'s' if len(cs) != 1 else ''})" for n, cs in left) or "none"
            raise NoFreeBoard(f"no free board with at least {wants[i]} chips is left for session {i + 1}; free boards not yet used: {sizes}")
        left.remove(fit)
        name, cs = fit
        plan[i] = {"board": name, "chips": [c["dev_index"] for c in cs], "exact": cs[0]["dev_index"]}
    return plan


def gozer_run_args(plan: dict, *, who: str, reason: str, chips: int | None = None, gozer: str = "gozer") -> list[str]:
    """`gozer run` up to the `--` that precedes the command."""
    n = chips if chips is not None else len(plan["chips"])
    return [gozer, "run", "--chips", str(n), "--exact", str(plan["exact"]), "--who", who, "--reason", reason, "--"]


def read_status(gozer: str, run=subprocess.run) -> dict:
    res = run([gozer, "status", "--json"], capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise NoFreeBoard(f"gozer status failed (exit {res.returncode}): {(res.stderr or '').strip()[:200]}")
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        raise NoFreeBoard("gozer status printed no JSON") from None


def choose_boards(wants: list[int], *, gozer: str = "gozer", exclude=(), wait_s: float = 0.0, run=subprocess.run, sleep=time.sleep, now=time.monotonic) -> list[dict]:
    """`plan_boards` over a fresh `gozer status`, asked again every POLL_S for up to `wait_s` seconds. The error of the last look is raised when time runs out."""
    deadline = now() + wait_s
    while True:
        try:
            return plan_boards(read_status(gozer, run), wants, exclude)
        except NoFreeBoard:
            if now() >= deadline:
                raise
        sleep(POLL_S)


def main(argv=None, run=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the control run.sh and its record")
    b.add_argument("--model-run-sh", required=True)
    b.add_argument("--model-dir", required=True, help="the directory the model's run.sh serves")
    b.add_argument("--base-snapshot", required=True, help="the base model's snapshot directory")
    b.add_argument("--base-id", required=True, help="ORG/NAME@REVISION of the base")
    b.add_argument("--out", required=True)
    d = sub.add_parser("board", help="name free boards and print the gozer run command for each")
    d.add_argument("--chips", type=int, action="append", required=True, help="chips one session needs; repeat for several sessions")
    d.add_argument("--exclude", action="append", default=[], help="a board id not to use")
    d.add_argument("--wait-s", type=float, default=0.0)
    d.add_argument("--who", default="claude:bench")
    d.add_argument("--reason", default="benchmark session")
    d.add_argument("--gozer", default="gozer")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            rec = build_control(model_run_sh=a.model_run_sh, model_dir=a.model_dir, base_snapshot=a.base_snapshot, base_id=a.base_id, out=a.out)
            print(json.dumps(rec, indent=1))
            return 0
        plans = choose_boards(a.chips, gozer=a.gozer, exclude=tuple(a.exclude), wait_s=a.wait_s, run=run)
        print(json.dumps({"plans": plans, "gozer": [gozer_run_args(p, who=a.who, reason=a.reason, chips=c, gozer=a.gozer) for p, c in zip(plans, a.chips)]}, indent=1))
        return 0
    except ControlError as exc:
        print(f"bench_control: {exc}", file=sys.stderr)
        return 2
    except NoFreeBoard as exc:
        print(f"bench_control: no free board: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
