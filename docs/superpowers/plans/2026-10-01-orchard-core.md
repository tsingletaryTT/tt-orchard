# tt-orchard core (plan 1 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the parts of the supervisor that need no hardware: the run ledger, the command runner with its denial list, the tier config loader, and the sizing tool that measures CPU models.

**Architecture:** A stdlib-only Python package `orchard`. `ledger.py` is an append-only, hash-chained JSONL file with replay. `runner.py` is the only way supervisor-launched commands execute and refuses a fixed set of commands. `tiers.py` validates a local TOML file that maps stages to model tiers. `sizing.py` measures load time, prefill and decode speed, and resident size of a model served by a local ollama, and records the result in the ledger.

**Tech Stack:** Python 3.12 (stdlib: `json`, `hashlib`, `fcntl`, `shlex`, `subprocess`, `tomllib`, `http.server`, `urllib`), pytest 8.

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md` (approved 2026-10-01). Sections 5.1, 9, 10 and 12 drive this plan.

## Plan series

This spec covers several independent subsystems, so it gets four plans, each producing working, testable software:

1. **This plan.** Ledger, command runner, tier config, sizing tool.
2. `tt-gozer` changes: `yield`, `redeem`, reservation expiry. Lives in `~/code/tt-gozer`. Needs a read of `gatekeeper.py` allocation first (spec section 8).
3. Supervisor behavior: gozer and single-tenant adapters, park and restore, watchdog.
4. Stage state machine, the new skills (`delta-triage`, `reference-gate`, `operator-bundle`, `gozer-park`), operator bundle and scrub check.

Plans 2 to 4 are written after this one is reviewed, because the sizing results from Task 5 decide the model tiers they use.

## Global Constraints

- Runtime code uses only the Python 3.12 standard library. pytest is the only dev dependency.
- No network access and no hardware access in any test. Tests use temporary directories and a local fake HTTP server.
- Comments explain why, in plain English, and are revised when the code changes (the owner's standing rule: well-documented, deeply commented code).
- Tier endpoints must be on this machine (`127.0.0.1`, `localhost` or `::1`). The run uses only open-source models served locally.
- The denial list in `runner.py` is never loosened without a spec change.
- Default branch is `main`. Commit messages are plain English, one idea per message.
- Config that names models (`config/tiers.toml`) is never committed.

## Rulings carried from the spec

- Config is TOML (`tiers.toml`) because the standard library reads TOML and has no YAML reader. The spec was updated to match.
- The ledger has a `run_start` event for the resolved versions. The spec's event list names the versions record but not its event type.
- The denial list also covers `tt-model publish` (catalog listing, spec section 2) and `hf upload` / `huggingface-cli upload` (another way to publish).
- A second process opening a live ledger is refused with a lock. The spec says nothing about two supervisors on one run. Appending from two processes would break the hash chain.

## Review Focus

Each line has a test in the task that owns the code.

1. A crash that leaves half a line at the end of the ledger, followed by a resume that keeps appending (Task 1: fault injection over every crash point, with and without a torn tail).
2. Two supervisors opening one ledger at once (Task 1: lock test).
3. Denied commands hidden behind wrappers, quotes, compound commands, `cd`, or `xargs` (Task 2: parametrized table).
4. A tier config that points at a remote endpoint, still holds the `CHANGE-ME` sentinel, or leaves a stage unassigned (Task 3).
5. An ollama reply with no prompt-processing fields, which ollama sends when the prompt was cached (Task 4).

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | pytest config and package metadata |
| `.gitignore` | ignores the local tier config, run directories and caches |
| `CLAUDE.md` | project log: original prompt, key decisions, notable moments |
| `orchard/__init__.py` | package marker and version |
| `orchard/ledger.py` | append-only ledger, replay, evidence hashing |
| `orchard/runner.py` | denial rules and the two execution entry points |
| `orchard/tiers.py` | tier config loader and validator |
| `orchard/sizing.py` | ollama measurement, ceiling arithmetic, ledger recording, CLI |
| `config/tiers.example.toml` | example tier config with `CHANGE-ME` sentinels |
| `tests/test_ledger.py`, `tests/test_runner.py`, `tests/test_tiers.py`, `tests/test_sizing.py` | one test file per module |

---

### Task 1: Ledger (includes project scaffold)

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `CLAUDE.md`, `orchard/__init__.py`, `orchard/ledger.py`, `tests/test_ledger.py`

**Interfaces:**
- Produces:
  - `class Ledger(path)`: opens or creates the file; takes an exclusive lock; recovers a torn tail. Methods `append(event: str, stage: int | None = None, **data) -> dict`, `read() -> list[dict]`, `close()`. Context manager.
  - `replay_state(entries: list[dict]) -> dict` with keys `stage`, `stage_status`, `parked`, `completed`.
  - `file_evidence(path) -> {"path": str, "sha256": str}`.
  - `EVENTS`, `LABELS`, `GENESIS`; exceptions `LedgerCorrupt`, `LedgerLocked`.
- Consumes: nothing.

- [ ] **Step 1: Scaffold the project files**

`pyproject.toml`:

```toml
[project]
name = "tt-orchard"
version = "0.0.1"
description = "Supervisor that brings new models up on a Tenstorrent Quietbox with local open-source models."
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`.gitignore`:

```
__pycache__/
.pytest_cache/
config/tiers.toml
runs/
*.torn-*
*.lock
```

`orchard/__init__.py`:

```python
"""tt-orchard: supervisor for autonomous model bring-up on a Tenstorrent Quietbox."""

__version__ = "0.0.1"
```

`CLAUDE.md`:

```markdown
# tt-orchard project log

## Original prompt (2026-10-01)
"What would it take to have a suite of skills that can handle moving from 4 chips back down to 2
chips or even 1 chip on the Quietbox 2? ... hand off to a CPU-based model ... 'hold my beer while I
test this new model'." Then widened: take a new model release (for example Qwen3.7) and bring it up
using the skills, with open-source models only. The run ends at an operator review bundle.

## Key decisions
- Supervisor is a state machine in code. Each stage starts a fresh short model context from the ledger.
- Lease handling goes through an adapter. tt-gozer owns leases and gains `yield` and `redeem`.
- The watchdog reports on agents it did not launch and acts only on agents it launched.
- The supervisor never publishes. A command runner refuses push, publish and reset commands.
- Config is TOML because the supervisor uses only the standard library.

## Layout
Spec: `docs/superpowers/specs/`. Plans: `docs/superpowers/plans/`. Code: `orchard/`. Tests: `tests/`.

## Log
- 2026-10-01: spec approved; plan 1 (ledger, runner, tiers, sizing) written.
```

- [ ] **Step 2: Write the failing tests**

`tests/test_ledger.py`:

```python
"""Ledger tests. The fault-injection test stops the writer at every point in a scripted run
and checks that the resumed run ends in the same state as an uninterrupted one."""
import pytest

from orchard.ledger import (GENESIS, Ledger, LedgerCorrupt, LedgerLocked,
                            file_evidence, replay_state)

SCRIPT = [
    ("run_start", None, {"tt_metal": "abc123"}),
    ("stage_start", 0, {}),
    ("stage_end", 0, {"result": "pass"}),
    ("stage_start", 4, {}),
    ("park", 4, {"board": "0000:01:00.0"}),
    ("restore", 4, {"canary": "ok"}),
    ("stage_end", 4, {"result": "pass"}),
]


def run_script(ledger, events):
    for event, stage, data in events:
        ledger.append(event, stage, **data)


def comparable(ledger):
    """Entries without timestamps, which differ between runs."""
    return [(e["event"], e["stage"], e["data"]) for e in ledger.read()]


def test_round_trip_and_chain(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        run_script(led, SCRIPT)
        entries = led.read()
    assert [e["seq"] for e in entries] == list(range(1, len(SCRIPT) + 1))
    assert entries[0]["prev"] == GENESIS
    assert comparable_from(entries) == SCRIPT


def comparable_from(entries):
    return [(e["event"], e["stage"], e["data"]) for e in entries]


def test_missing_file_is_an_empty_ledger(tmp_path):
    with Ledger(tmp_path / "new" / "ledger.jsonl") as led:
        assert led.read() == []


def test_unknown_event_is_rejected(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        with pytest.raises(ValueError):
            led.append("made_up_event")


def test_measurement_needs_a_label(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        with pytest.raises(ValueError):
            led.append("measurement", 6, tok_s=40)
        led.append("measurement", 6, tok_s=40, label="measured")
        led.append("measurement", 6, tok_s=None, label="TODO")
        assert len(led.read()) == 2


def test_tampering_with_an_earlier_line_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path) as led:
        run_script(led, SCRIPT)
    path.write_text(path.read_text().replace('"result":"pass"', '"result":"fail"', 1))
    with pytest.raises(LedgerCorrupt):
        Ledger(path)


def test_torn_tail_is_set_aside_not_deleted(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path) as led:
        run_script(led, SCRIPT[:3])
    with open(path, "ab") as f:
        f.write(b'{"seq":4,"prev":"abc","ev')  # a write cut off mid-line
    with Ledger(path) as led:
        assert len(led.read()) == 3
    sidecars = list(tmp_path.glob("ledger.jsonl.torn-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_bytes().startswith(b'{"seq":4')


def test_second_opener_is_refused(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with Ledger(path):
        with pytest.raises(LedgerLocked):
            Ledger(path)
    Ledger(path).close()  # the lock is released on close


def test_replay_state_tracks_stage_and_parking():
    entries = [
        {"event": "stage_start", "stage": 4, "data": {}},
        {"event": "park", "stage": 4, "data": {}},
    ]
    state = replay_state(entries)
    assert state["stage"] == 4 and state["stage_status"] == "running" and state["parked"]
    entries += [
        {"event": "restore", "stage": 4, "data": {}},
        {"event": "stage_end", "stage": 4, "data": {"result": "pass"}},
    ]
    state = replay_state(entries)
    assert not state["parked"] and state["completed"] == [4] and state["stage_status"] == "pass"


@pytest.mark.parametrize("torn_tail", [False, True])
@pytest.mark.parametrize("crash_after", range(len(SCRIPT) + 1))
def test_crash_at_every_point_resumes_to_the_same_state(tmp_path, crash_after, torn_tail):
    # Reference: the whole script, never interrupted.
    with Ledger(tmp_path / "ref" / "ledger.jsonl") as ref:
        run_script(ref, SCRIPT)
        want = comparable(ref)
        want_state = replay_state(ref.read())

    # Crash: write part of the script, optionally leave half a line, drop the writer.
    path = tmp_path / "run" / "ledger.jsonl"
    led = Ledger(path)
    run_script(led, SCRIPT[:crash_after])
    led.close()
    if torn_tail:
        with open(path, "ab") as f:
            f.write(b'{"seq":99,"pr')

    # Resume: reopen and finish the script.
    with Ledger(path) as led:
        run_script(led, SCRIPT[crash_after:])
        assert comparable(led) == want
        assert replay_state(led.read()) == want_state


def test_file_evidence_hashes_content(tmp_path):
    f = tmp_path / "e.txt"
    f.write_bytes(b"abc")
    ev = file_evidence(f)
    assert ev["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert ev["path"] == str(f)
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `cd ~/code/tt-orchard && python3 -m pytest tests/test_ledger.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchard.ledger'`.

- [ ] **Step 4: Implement the ledger**

`orchard/ledger.py`:

```python
"""Append-only run ledger.

One JSON object per line. Each line carries the sha256 of the previous line's text, so editing
an earlier line breaks the chain and is detected on the next open. (The last line has no
successor, so a change to it alone cannot be detected. The chain covers every line except the last.)

A line only counts once its trailing newline is on disk. Opening the ledger sets aside any bytes
after the last newline (a write cut off by a crash) in a `.torn-<time>` file next to the ledger.
Nothing is deleted.

The current state of a run is computed from the ledger by `replay_state`. No other state file
exists, so nothing can disagree with the ledger.

One process may hold a ledger open at a time. Two writers would break the chain, so a second
opener gets LedgerLocked.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

GENESIS = "0" * 64

EVENTS = frozenset({
    "run_start", "stage_start", "stage_end", "park", "restore",
    "retry", "escalate", "notice", "measurement", "decision",
})

# Every number in a measurement entry says whether it was measured or is still to do.
LABELS = frozenset({"measured", "TODO"})


class LedgerCorrupt(Exception):
    """The file is not a valid chain. A person has to look at it."""


class LedgerLocked(Exception):
    """Another process has this ledger open."""


def _digest(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # flock belongs to the open file, so it is released when close() runs or the
        # process dies. That is what lets a restarted supervisor reopen a crashed run.
        self._lock_file = open(self.path.with_name(self.path.name + ".lock"), "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise LedgerLocked(f"another process holds {self.path}") from None
        self._seq = 0
        self._prev = GENESIS
        self._recover()

    def close(self) -> None:
        self._lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _lines(self) -> list[str]:
        if not self.path.exists():
            return []
        # After _recover the file is empty or ends in a newline, so the last split piece is "".
        return self.path.read_text(encoding="utf-8").split("\n")[:-1]

    def _recover(self) -> None:
        if self.path.exists():
            raw = self.path.read_bytes()
            good_end = raw.rfind(b"\n") + 1  # offset just past the last complete line
            torn = raw[good_end:]
            if torn:
                # Keep the cut-off bytes for a person to inspect, then drop them from the chain.
                sidecar = self.path.with_name(f"{self.path.name}.torn-{int(time.time())}")
                sidecar.write_bytes(torn)
                with open(self.path, "r+b") as f:
                    f.truncate(good_end)
                    f.flush()
                    os.fsync(f.fileno())
        entries = self.read()  # verifies the whole chain
        if entries:
            self._seq = entries[-1]["seq"]
            self._prev = _digest(self._lines()[-1])

    def read(self) -> list[dict]:
        entries, prev = [], GENESIS
        for i, line in enumerate(self._lines(), start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerCorrupt(f"line {i} is not JSON") from exc
            if entry.get("seq") != i:
                raise LedgerCorrupt(f"line {i}: expected sequence {i}, found {entry.get('seq')}")
            if entry.get("prev") != prev:
                raise LedgerCorrupt(f"line {i}: hash chain broken")
            entries.append(entry)
            prev = _digest(line)
        return entries

    def append(self, event: str, stage: int | None = None, **data) -> dict:
        if event not in EVENTS:
            raise ValueError(f"unknown ledger event {event!r}")
        if event == "measurement" and data.get("label") not in LABELS:
            raise ValueError("a measurement entry needs label='measured' or label='TODO'")
        entry = {
            "seq": self._seq + 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prev": self._prev,
            "event": event,
            "stage": stage,
            "data": data,
        }
        line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        with open(self.path, "ab") as f:
            f.write(line.encode("utf-8") + b"\n")
            f.flush()
            os.fsync(f.fileno())  # the entry exists only once this returns
        self._seq += 1
        self._prev = _digest(line)
        return entry


def replay_state(entries: list[dict]) -> dict:
    """Current run state, computed by walking the entries in order."""
    state = {"stage": None, "stage_status": None, "parked": False, "completed": []}
    for e in entries:
        event, stage = e["event"], e["stage"]
        if event == "stage_start":
            state["stage"], state["stage_status"] = stage, "running"
        elif event == "stage_end":
            result = e["data"].get("result")
            state["stage_status"] = result
            if result == "pass":
                state["completed"].append(stage)
        elif event == "park":
            state["parked"] = True
        elif event == "restore":
            state["parked"] = False
    return state


def file_evidence(path) -> dict:
    """The path and sha256 that an evidence entry records."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python3 -m pytest tests/test_ledger.py -q`
Expected: all pass (7 named tests plus 16 crash-point cases).

- [ ] **Step 6: Watch two guards fail, then restore them**

In `_recover`, comment out the `f.truncate(good_end)` line. Run `python3 -m pytest tests/test_ledger.py -q`. Expected: `test_torn_tail_is_set_aside_not_deleted` and the `torn_tail=True` crash cases fail. Restore the line.
In `read()`, comment out the `prev` check. Run again. Expected: `test_tampering_with_an_earlier_line_is_detected` fails. Restore the check and rerun to confirm green.

- [ ] **Step 7: Commit**

```bash
cd ~/code/tt-orchard
git add pyproject.toml .gitignore CLAUDE.md orchard tests docs
git commit -m "Add the run ledger: hash-chained, crash-safe, single writer"
```

---

### Task 2: Command runner with denial list

**Files:**
- Create: `orchard/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Produces:
  - `class Denied(Exception)` with `.rule` and `.command`.
  - `RULES: list[tuple[str, Callable]]`; each rule is `(name, fn(name, argv, ctx) -> bool)`.
  - `check_argv(argv, run_dir, cwd: str | None) -> str | None` (`cwd` is the real path the command runs in, None when unknowable; returns the directory after the command), `check_string(text, run_dir) -> None`.
  - `run_argv(argv, run_dir, **kw) -> CompletedProcess`, `run_shell(text, run_dir, **kw) -> CompletedProcess`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:

```python
"""Runner tests. Every rule has denied forms in DENY. A structural test fails when a rule is added
without one, so a rule cannot exist untested."""
import subprocess

import pytest

from orchard.runner import RULES, Denied, check_string, run_argv, run_shell

ALLOW = [
    "ls -la",
    "git status",
    "git -C {run} log --oneline",
    "git commit -m 'push it to the shelf'",
    "tt-smi -s",
    "gozer status --json",
    "tt-model serve some/model",
    "python3 -c 'print(1)'",
    "rm -rf {run}/scratch/old",
    "rm {run}/out.txt",
    "echo hi > {run}/out.txt",
    "cd {run}/sub; rm -rf cache",
]

DENY = {
    "tt-model-push": [
        "tt-model push episod/x",
        "/usr/local/bin/tt-model push .",
        "sudo tt-model push x",
        "sudo -u bob tt-model push x",
        "env A=1 tt-model push x",
        "FOO=1 tt-model push x",
        "timeout 10 tt-model push x",
        "bash -c 'tt-model push x'",
        "ls; tt-model push x",
        "true && tt-model push x",
        "false || tt-model push x",
        "echo a | tt-model push x",
    ],
    "tt-model-publish": ["tt-model publish episod/x"],
    "git-push": [
        "git push",
        "git -C /tmp/x push origin main",
        "git -c user.name=a push",
        "nice -n 5 git push",
        "/usr/bin/git push --force",
    ],
    "gh-repo-create": ["gh repo create foo --public", "gh --hostname x repo create foo"],
    "hf-upload": ["hf upload a b", "huggingface-cli upload a b"],
    "tt-smi-reset": ["tt-smi -r", "tt-smi -r 0,1", "tt-smi --reset", "sudo tt-smi -r", "tt-smi -r0"],
    "rm-outside-run-dir": [
        "rm -rf /home/someone/.cache/x",
        "rm {run}/../escaped",
        "rm -rf $HOME/cache",
        "rm -rf ~/code",
        "rm -rf {run}",
        "rm -rf {run}/*/../..",
        "cd /; rm -rf *",
        "cd $SOMEWHERE; rm file",
        "rmdir /tmp/x",
        "unlink /tmp/x",
    ],
    "rm-ledger": ["rm {run}/ledger.jsonl", "rm -f {run}/ledger*"],
    "rm-xargs": ["find . -name x | xargs rm", "ls | xargs -n 1 rm -rf"],
    "substitution": ["echo $(date)", "echo `date`", "diff <(ls) <(ls)"],
}


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    (d / "sub").mkdir(parents=True)
    return d


@pytest.mark.parametrize("cmd", ALLOW)
def test_allowed_commands_pass(cmd, run_dir):
    check_string(cmd.format(run=run_dir), run_dir)


@pytest.mark.parametrize(
    "rule,cmd", [(r, c) for r, cmds in DENY.items() for c in cmds])
def test_denied_commands_name_their_rule(rule, cmd, run_dir):
    with pytest.raises(Denied) as exc:
        check_string(cmd.format(run=run_dir), run_dir)
    assert exc.value.rule == rule


def test_every_rule_has_denied_examples():
    assert {name for name, _ in RULES} | {"substitution"} == set(DENY)


def test_run_argv_executes_allowed_and_refuses_denied(run_dir):
    done = run_argv(["echo", "hello"], run_dir, capture_output=True, text=True)
    assert done.stdout.strip() == "hello"
    with pytest.raises(Denied):
        run_argv(["git", "push"], run_dir)


def test_run_shell_checks_before_running(run_dir):
    marker = run_dir / "marker"
    with pytest.raises(Denied):
        run_shell(f"touch {marker}; git push", run_dir)
    assert not marker.exists()  # nothing ran, including the harmless first command
    run_shell(f"touch {marker}", run_dir)
    assert marker.exists()
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 -m pytest tests/test_runner.py -q`
Expected: collection error, `No module named 'orchard.runner'`.

- [ ] **Step 3: Implement the runner**

`orchard/runner.py`:

```python
"""The command runner: the only way the supervisor runs commands for a stage agent.

It refuses a fixed list of commands (see RULES): publishing, pushing, hand-run chip resets, and
deletion outside the run directory. The block sits here, where commands execute, so an ignored
prompt cannot bypass it.

WHAT THIS GUARDS, AND WHAT IT DOES NOT. The rules judge the commands they name, after stripping
wrappers (sudo, env, timeout, nice, nohup, time, xargs, ...), splitting compound commands at
`;`, `&&`, `||`, `|` and `&`, and looking inside `bash -c '...'`. They do not stop arbitrary
code (for example `python3 -c 'shutil.rmtree(...)'`) or deletion by other tools (`find -delete`).
Command substitution and process substitution are refused outright because the inner command
cannot be judged. Treat this as a guard against the named mistakes. The run's user account and the
lease tool remain the real limits.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass


class Denied(Exception):
    def __init__(self, rule: str, command: str):
        super().__init__(f"denied by rule {rule!r}: {command}")
        self.rule = rule
        self.command = command


SHELLS = {"bash", "sh", "zsh", "dash"}

# Commands that run another command. Options that take a value are listed so the value is not
# mistaken for the command (for example `sudo -u bob git push`).
WRAPPERS = {"sudo", "nohup", "time", "command", "exec", "env", "nice", "timeout", "stdbuf", "xargs"}
VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-T", "-U"},
    "env": {"-u", "-C", "-S"},
    "nice": {"-n"},
    "timeout": {"-k", "-s"},
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {"-n", "-I", "-L", "-P", "-d", "-s", "-E", "-a"},
}
SEPARATOR_CHARS = set(";&|()")
DELETERS = {"rm", "rmdir", "unlink"}


@dataclass
class Ctx:
    run_dir: str          # real path of the run directory
    cwd: str | None       # directory the command runs in; None when a `cd` target was unknowable
    via_xargs: bool       # the command's operands come from a pipe we cannot see


# ---- parsing ------------------------------------------------------------------------------

def _simple_commands(text: str) -> list[list[str]]:
    """Split a shell string into simple commands at ; && || | & and newlines."""
    lexer = shlex.shlex(text.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    commands, current = [], []
    for tok in lexer:
        if tok and set(tok) <= SEPARATOR_CHARS:
            if current:
                commands.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        commands.append(current)
    return commands


def _is_assignment(tok: str) -> bool:
    name, eq, _ = tok.partition("=")
    return bool(eq) and name.isidentifier()


def _unwrap(argv: list[str]) -> tuple[list[str], bool]:
    """Strip leading assignments and wrapper commands. Returns (argv, saw_xargs)."""
    argv, saw_xargs = list(argv), False
    while argv:
        if _is_assignment(argv[0]):
            argv.pop(0)
            continue
        name = os.path.basename(argv[0])
        if name not in WRAPPERS:
            break
        argv.pop(0)
        saw_xargs = saw_xargs or name == "xargs"
        while argv and argv[0].startswith("-"):
            opt = argv.pop(0)
            if opt in VALUE_OPTIONS.get(name, ()) and argv:
                argv.pop(0)
        if name == "timeout" and argv:
            argv.pop(0)  # the duration operand
    return argv, saw_xargs


def _operands(argv: list[str]) -> list[str]:
    """Arguments that are not options. Everything after `--` is an operand."""
    out, after_dd = [], False
    for a in argv[1:]:
        if after_dd or not a.startswith("-"):
            out.append(a)
        elif a == "--":
            after_dd = True
    return out


def _git_subcommand(argv: list[str]) -> str | None:
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"} else 1
    return argv[i] if i < len(argv) else None


# ---- rules --------------------------------------------------------------------------------
# Each rule is (name, fn(name, argv, ctx) -> bool); True means "deny".

def _tt_model_push(name, argv, ctx):
    return name == "tt-model" and "push" in argv[1:]


def _tt_model_publish(name, argv, ctx):
    return name == "tt-model" and "publish" in argv[1:]


def _git_push(name, argv, ctx):
    return name == "git" and _git_subcommand(argv) == "push"


def _gh_repo_create(name, argv, ctx):
    rest = argv[1:]
    return name == "gh" and "repo" in rest and "create" in rest[rest.index("repo"):]


def _hf_upload(name, argv, ctx):
    return name in {"hf", "huggingface-cli"} and "upload" in argv[1:]


def _tt_smi_reset(name, argv, ctx):
    if name != "tt-smi":
        return False
    for a in argv[1:]:
        if a in {"-r", "--reset"} or a.startswith("--reset="):
            return True
        if a.startswith("--") and "reset" in a:
            return True
        if a.startswith("-r") and not a.startswith("--"):  # -r0, -r0,1
            return True
    return False


def _rm_outside_run_dir(name, argv, ctx):
    if name not in DELETERS:
        return False
    for a in _operands(argv):
        if "$" in a:
            return True  # unknown path
        path = os.path.expanduser(a)
        if not os.path.isabs(path):
            if ctx.cwd is None:
                return True  # relative path in an unknown directory
            path = os.path.join(ctx.cwd, path)
        if any(c in os.path.dirname(path) for c in "*?["):
            return True  # a glob in a directory part cannot be judged
        if any(c in os.path.basename(path) for c in "*?["):
            path = os.path.dirname(path)  # a glob is judged by the directory it expands in
        target = os.path.realpath(path)
        if target == ctx.run_dir or not target.startswith(ctx.run_dir + os.sep):
            return True
    return False


def _rm_ledger(name, argv, ctx):
    return name in DELETERS and any(
        os.path.basename(a).startswith("ledger") for a in _operands(argv))


def _rm_xargs(name, argv, ctx):
    return name in DELETERS and ctx.via_xargs


RULES = [
    ("tt-model-push", _tt_model_push),
    ("tt-model-publish", _tt_model_publish),
    ("git-push", _git_push),
    ("gh-repo-create", _gh_repo_create),
    ("hf-upload", _hf_upload),
    ("tt-smi-reset", _tt_smi_reset),
    ("rm-outside-run-dir", _rm_outside_run_dir),
    ("rm-ledger", _rm_ledger),
    ("rm-xargs", _rm_xargs),
]


# ---- checking and running -----------------------------------------------------------------

def check_argv(argv: list[str], run_dir, cwd: str | None) -> str | None:
    """Raise Denied if the command breaks a rule. Returns the directory after the command.

    `cwd` is the directory the command runs in, as a real path. None means the directory is
    unknowable (an earlier `cd $VAR`), and relative paths are then refused by the delete rules.
    """
    run_real = os.path.realpath(run_dir)
    unwrapped, via_xargs = _unwrap(argv)
    if not unwrapped:
        return cwd
    name = os.path.basename(unwrapped[0])
    if name in SHELLS and "-c" in unwrapped:
        i = unwrapped.index("-c")
        if i + 1 < len(unwrapped):
            check_string(unwrapped[i + 1], run_dir)
    ctx = Ctx(run_dir=run_real, cwd=cwd, via_xargs=via_xargs)
    for rule_name, deny in RULES:
        if deny(name, unwrapped, ctx):
            raise Denied(rule_name, " ".join(argv))
    if name in {"cd", "pushd"}:
        target = next((a for a in unwrapped[1:] if not a.startswith("-")), "~")
        if "$" in target or target == "-" or cwd is None:
            return None
        return os.path.realpath(os.path.join(cwd, os.path.expanduser(target)))
    return cwd


def check_string(text: str, run_dir) -> None:
    if any(marker in text for marker in ("$(", "`", "<(", ">(")):
        raise Denied("substitution", text)
    cwd: str | None = os.path.realpath(run_dir)  # every command starts in the run directory
    for argv in _simple_commands(text):
        cwd = check_argv(argv, run_dir, cwd)


def run_argv(argv: list[str], run_dir, **kw) -> subprocess.CompletedProcess:
    """Run one command, no shell. Refused commands never start."""
    check_argv(argv, run_dir, os.path.realpath(run_dir))
    return subprocess.run(argv, shell=False, cwd=run_dir, **kw)


def run_shell(text: str, run_dir, **kw) -> subprocess.CompletedProcess:
    """Run a shell string after judging every command in it. Nothing runs if any part is refused."""
    check_string(text, run_dir)
    return subprocess.run(["bash", "-c", text], cwd=run_dir, **kw)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_runner.py -q`
Expected: all pass. The tests in `ALLOW` and `DENY` define the behavior. If a case fails, fix the code and leave the table alone unless the case itself is wrong, and say which in the commit message.

- [ ] **Step 5: Watch each rule's tests fail without it**

For each name in `RULES`, remove that tuple, run `python3 -m pytest tests/test_runner.py -q`, confirm the denied cases for that rule fail (and `test_every_rule_has_denied_examples` fails), then restore. Do this for at least `git-push`, `tt-smi-reset`, `rm-outside-run-dir` and `rm-xargs`.

- [ ] **Step 6: Commit**

```bash
git add orchard/runner.py tests/test_runner.py
git commit -m "Add the command runner and its denial rules"
```

---

### Task 3: Tier config

**Files:**
- Create: `orchard/tiers.py`, `config/tiers.example.toml`, `tests/test_tiers.py`

**Interfaces:**
- Produces: `load(path) -> TierConfig` with `.tiers: dict[str, dict]` and `.stages: dict[int, dict]` (each stage dict has `run` and optional `diagnose`); `class TierConfigError(ValueError)`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

`tests/test_tiers.py`:

```python
import pytest

from orchard.tiers import TierConfigError, load

GOOD = """
[tiers.large]
role = "plan and diagnose"
endpoint = "http://127.0.0.1:8000/v1"
model = "big-model"
placement = "chips"
context_tokens = 262144

[tiers.small]
role = "routine steps"
endpoint = "http://localhost:8001/v1"
model = "small-model"
placement = "cpu"

[stages.0]
run = "large"
[stages.1]
run = "small"
[stages.2]
run = "small"
diagnose = "large"
[stages.3]
run = "small"
diagnose = "large"
[stages.4]
run = "small"
diagnose = "large"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"
"""


def write(tmp_path, text):
    p = tmp_path / "tiers.toml"
    p.write_text(text)
    return p


def test_good_config_loads(tmp_path):
    cfg = load(write(tmp_path, GOOD))
    assert set(cfg.tiers) == {"large", "small"}
    assert cfg.stages[2] == {"run": "small", "diagnose": "large"}
    assert cfg.stages[7] == {"run": "none"}


@pytest.mark.parametrize("old,new,why", [
    ('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "https://api.example.com/v1"', "remote"),
    ('model = "big-model"', 'model = "CHANGE-ME"', "sentinel"),
    ('placement = "chips"', 'placement = "gpu"', "placement"),
    ('role = "plan and diagnose"\n', '', "missing key"),
    ('[stages.8]\nrun = "small"\n', '', "stage missing"),
    ('[stages.7]\nrun = "none"', '[stages.7]\nrun = "small"', "stage 7 must be none"),
    ('[stages.5]\nrun = "small"', '[stages.5]\nrun = "none"', "only stage 7 may be none"),
    ('[stages.1]\nrun = "small"', '[stages.1]\nrun = "huge"', "unknown tier"),
    ('diagnose = "large"\n[stages.3]', 'diagnose = "nobody"\n[stages.3]', "unknown diagnose tier"),
    ('context_tokens = 262144', 'context_tokens = 0', "context must be positive"),
])
def test_bad_configs_are_refused(tmp_path, old, new, why):
    assert old in GOOD, why  # the edit must hit something, or the case proves nothing
    with pytest.raises(TierConfigError):
        load(write(tmp_path, GOOD.replace(old, new, 1)))


def test_a_cpu_tier_is_required(tmp_path):
    text = GOOD.replace('placement = "cpu"', 'placement = "chips"')
    with pytest.raises(TierConfigError):
        load(write(tmp_path, text))


def test_example_config_is_refused_until_edited():
    with pytest.raises(TierConfigError):
        load("config/tiers.example.toml")
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 -m pytest tests/test_tiers.py -q`
Expected: collection error, `No module named 'orchard.tiers'`.

- [ ] **Step 3: Implement**

`orchard/tiers.py`:

```python
"""Tier config: which local model fills each role, and which tier runs each stage.

The file is local to the machine and never committed. It names models, and the choice of models
depends on measurements (see sizing.py). The validator refuses anything that would send a run
somewhere unintended: a remote endpoint, an unedited example, or a stage nobody owns.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from urllib.parse import urlparse

STAGES = range(9)                 # stages 0..8 in the spec
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
PLACEMENTS = {"chips", "cpu"}
REQUIRED_TIER_KEYS = ("role", "endpoint", "model", "placement")
SENTINEL = "CHANGE-ME"            # example config values; a run must not start on them
NO_MODEL_STAGE = 7                # image build: the supervisor waits and no model is loaded


class TierConfigError(ValueError):
    pass


@dataclass
class TierConfig:
    tiers: dict
    stages: dict


def load(path) -> TierConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    tiers = raw.get("tiers", {})
    stages = {int(k): v for k, v in raw.get("stages", {}).items()}

    for name, tier in tiers.items():
        for key in REQUIRED_TIER_KEYS:
            if key not in tier:
                raise TierConfigError(f"tier {name!r} is missing {key!r}")
            if tier[key] == SENTINEL:
                raise TierConfigError(f"tier {name!r} still has {SENTINEL} in {key!r}")
        if tier["placement"] not in PLACEMENTS:
            raise TierConfigError(f"tier {name!r}: placement must be one of {sorted(PLACEMENTS)}")
        host = urlparse(tier["endpoint"]).hostname
        if host not in LOCAL_HOSTS:
            raise TierConfigError(
                f"tier {name!r}: endpoint host {host!r} is not on this machine; runs use local models only")
        if "context_tokens" in tier and not (isinstance(tier["context_tokens"], int) and tier["context_tokens"] > 0):
            raise TierConfigError(f"tier {name!r}: context_tokens must be a positive integer")

    if not any(t["placement"] == "cpu" for t in tiers.values()):
        raise TierConfigError("a tier with placement 'cpu' is required (the stand-in during parking)")

    for n in STAGES:
        if n not in stages:
            raise TierConfigError(f"stage {n} has no entry")
        stage = stages[n]
        run = stage.get("run")
        if n == NO_MODEL_STAGE:
            if run != "none":
                raise TierConfigError(f"stage {n} must have run = 'none'")
        elif run not in tiers:
            raise TierConfigError(f"stage {n}: run tier {run!r} is not defined")
        if "diagnose" in stage and stage["diagnose"] not in tiers:
            raise TierConfigError(f"stage {n}: diagnose tier {stage['diagnose']!r} is not defined")

    return TierConfig(tiers=tiers, stages=stages)
```

`config/tiers.example.toml`:

```toml
# Copy to config/tiers.toml and edit. The loader refuses any value left as CHANGE-ME.
# Choose models after running orchard/sizing.py (spec section 5.1); do not guess.

[tiers.large]
role = "plan and diagnose"
endpoint = "http://127.0.0.1:8000/v1"
model = "CHANGE-ME"
placement = "chips"
context_tokens = 262144

[tiers.small]
role = "routine steps while the large model is parked or busy"
endpoint = "http://127.0.0.1:8001/v1"
model = "CHANGE-ME"
placement = "cpu"

# Stage owner. "diagnose" is the tier that takes over when the run tier escalates.
[stages.0]
run = "large"
[stages.1]
run = "small"
[stages.2]
run = "small"
diagnose = "large"
[stages.3]
run = "small"
diagnose = "large"
[stages.4]
run = "small"
diagnose = "large"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_tiers.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add orchard/tiers.py config/tiers.example.toml tests/test_tiers.py
git commit -m "Add the tier config loader and its checks"
```

---

### Task 4: Sizing tool

**Files:**
- Create: `orchard/sizing.py`, `tests/test_sizing.py`

**Interfaces:**
- Consumes: `Ledger.append("measurement", ..., label="measured")` from Task 1.
- Produces:
  - `theoretical_gb_s(mt_s, channels=2, bus_bytes=8) -> float`
  - `ceiling_tok_s(bandwidth_gb_s, gb_per_token) -> float`
  - `measure(host, model, prompt, num_predict=128, timeout=900) -> dict` with keys `model`, `load_s`, `prompt_tokens`, `prefill_tok_s`, `decode_tokens`, `decode_tok_s`, `resident_bytes`.
  - `main(argv=None) -> int` (CLI, `python -m orchard.sizing`).

- [ ] **Step 1: Write the failing tests**

`tests/test_sizing.py`:

```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from orchard.ledger import Ledger
from orchard.sizing import ceiling_tok_s, main, measure, theoretical_gb_s


class Fake(BaseHTTPRequestHandler):
    """Stands in for ollama's /api/generate and /api/ps."""

    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.server.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(self.server.generate_reply)

    def do_GET(self):
        self._send(self.server.ps_reply)


@pytest.fixture
def fake():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    server.generate_reply = {
        "load_duration": 3_000_000_000,
        "prompt_eval_count": 50, "prompt_eval_duration": 500_000_000,
        "eval_count": 100, "eval_duration": 2_000_000_000,
    }
    server.ps_reply = {"models": [{"name": "m:1", "size": 18_000_000_000}]}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def host(server):
    return f"http://127.0.0.1:{server.server_address[1]}"


def test_ceiling_arithmetic():
    assert theoretical_gb_s(3600) == pytest.approx(57.6)
    assert theoretical_gb_s(5600) == pytest.approx(89.6)
    assert ceiling_tok_s(57.6, 1.8) == pytest.approx(32.0)


def test_measure_parses_ollama_timings(fake):
    r = measure(host(fake), "m:1", "hello", num_predict=100)
    assert r["decode_tok_s"] == pytest.approx(50.0)
    assert r["prefill_tok_s"] == pytest.approx(100.0)
    assert r["load_s"] == pytest.approx(3.0)
    assert r["resident_bytes"] == 18_000_000_000
    assert fake.seen["options"]["temperature"] == 0  # the same prompt gives the same tokens


def test_cached_prompt_reply_has_no_prefill_fields(fake):
    del fake.generate_reply["prompt_eval_count"], fake.generate_reply["prompt_eval_duration"]
    r = measure(host(fake), "m:1", "hello")
    assert r["prefill_tok_s"] is None and r["decode_tok_s"] == pytest.approx(50.0)


def test_model_missing_from_ps_gives_unknown_size(fake):
    fake.ps_reply = {"models": []}
    assert measure(host(fake), "m:1", "hello")["resident_bytes"] is None


def test_cli_records_a_measured_entry(fake, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hello")
    ledger = tmp_path / "ledger.jsonl"
    rc = main(["--host", host(fake), "--model", "m:1", "--prompt-file", str(prompt),
               "--ledger", str(ledger), "--gb-per-token", "1.8", "--note", "coder idle"])
    assert rc == 0
    with Ledger(ledger) as led:
        (entry,) = led.read()
    assert entry["event"] == "measurement" and entry["data"]["label"] == "measured"
    assert entry["data"]["decode_tok_s"] == pytest.approx(50.0)
    assert entry["data"]["ceiling_tok_s"] == pytest.approx(32.0)
    assert entry["data"]["note"] == "coder idle"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 -m pytest tests/test_sizing.py -q`
Expected: collection error, `No module named 'orchard.sizing'`.

- [ ] **Step 3: Implement**

`orchard/sizing.py`:

```python
"""Measure a CPU-served model so the tier choice rests on numbers (spec sections 5.1 and 12).

Talks to a local ollama over HTTP. It measures load time, prefill speed, decode speed and the
resident size ollama reports. It does not pull or start anything. The caller runs the model server
and decides which models to download.

The ceiling is arithmetic: each generated token reads every active weight once, so tokens per second
cannot exceed memory bandwidth divided by the gigabytes read per token. A measured speed far below
the ceiling points at compute or contention. The theoretical bandwidth is the DIMM rate times the bus
width times the channels, and the real figure is lower.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from orchard.ledger import Ledger


def theoretical_gb_s(mt_s: float, channels: int = 2, bus_bytes: int = 8) -> float:
    return mt_s * bus_bytes * channels / 1000


def ceiling_tok_s(bandwidth_gb_s: float, gb_per_token: float) -> float:
    return bandwidth_gb_s / gb_per_token


def _request(host: str, path: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(host + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _rate(count, duration_ns):
    """Tokens per second, or None when ollama did not report the pair."""
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def measure(host: str, model: str, prompt: str, num_predict: int = 128,
            timeout: float = 900) -> dict:
    reply = _request(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        # temperature 0 and a fixed seed make the output reproducible between runs.
        "options": {"num_predict": num_predict, "temperature": 0, "seed": 1},
    }, timeout)
    running = _request(host, "/api/ps", None, timeout).get("models", [])
    resident = next((m.get("size") for m in running
                     if model in (m.get("name"), m.get("model"))), None)
    return {
        "model": model,
        "load_s": reply.get("load_duration", 0) / 1e9,
        "prompt_tokens": reply.get("prompt_eval_count"),
        # ollama leaves the prompt fields out when it reused a cached prompt.
        "prefill_tok_s": _rate(reply.get("prompt_eval_count"), reply.get("prompt_eval_duration")),
        "decode_tokens": reply.get("eval_count"),
        "decode_tok_s": _rate(reply.get("eval_count"), reply.get("eval_duration")),
        "resident_bytes": resident,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measure a model served by a local ollama.")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--model", action="append", required=True, help="repeat for several models")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--mt-s", type=float, default=3600,
                   help="configured DIMM rate; the host reports 3600 (rated 5600)")
    p.add_argument("--gb-per-token", type=float,
                   help="gigabytes read per token (about the active weights); enables the ceiling")
    p.add_argument("--note", default="", help="conditions, for example 'coder serving'")
    p.add_argument("--ledger", help="append each result to this ledger as a measured entry")
    args = p.parse_args(argv)

    prompt = open(args.prompt_file, encoding="utf-8").read()
    bandwidth = theoretical_gb_s(args.mt_s)
    ledger = Ledger(args.ledger) if args.ledger else None
    try:
        for model in args.model:
            result = measure(args.host, model, prompt, args.num_predict)
            result.update(note=args.note, theoretical_gb_s=bandwidth)
            if args.gb_per_token:
                result["ceiling_tok_s"] = ceiling_tok_s(bandwidth, args.gb_per_token)
            print(json.dumps(result, indent=2))
            if ledger:
                ledger.append("measurement", None, label="measured", **result)
    finally:
        if ledger:
            ledger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `python3 -m pytest tests/test_sizing.py -q` then `python3 -m pytest -q`
Expected: all pass in both runs.

- [ ] **Step 5: Watch the cached-prompt guard fail, then restore it**

In `_rate`, replace the guard with `return count / (duration_ns / 1e9)`. Run `python3 -m pytest tests/test_sizing.py -q`. Expected: `test_cached_prompt_reply_has_no_prefill_fields` errors. Restore the guard.

- [ ] **Step 6: Commit**

```bash
git add orchard/sizing.py tests/test_sizing.py
git commit -m "Add the sizing tool for CPU model measurements"
```

---

### Task 5: Measure candidate CPU models (operator gate, no code)

This task downloads multi-gigabyte models and loads the host. **Stop here and ask the operator** before running anything. The executor does not choose models.

- [ ] **Step 1: Ask the operator**, in one message: which models to pull (the spec names the class to try first: a mixture-of-experts model with a few billion active parameters, plus one dense 8B as a baseline), how much disk to spend (`df -h ~` first; space was tight earlier), and whether the large model should be serving during the measurement. Also ask when a measurement run is safe: it loads the host CPU and memory bus, and the operator has recorded on this machine.
- [ ] **Step 2: Start the model server at low priority.** Run `nice -n 19 ollama serve` in the background (ollama is installed and was not running on 2026-10-01).
- [ ] **Step 3: Measure each approved model twice**, once with the large model idle and once with it serving a long generation, using a fixed 500-token prompt file:

```bash
python3 -m orchard.sizing --model <approved-model> --prompt-file prompts/sizing.txt \
  --gb-per-token <active-weights-in-GB> --note "coder idle" --ledger runs/sizing/ledger.jsonl
```

Expected: one JSON result per model with `decode_tok_s`, `ceiling_tok_s`, `load_s`, `resident_bytes`.
- [ ] **Step 4: Write the results into spec section 5.1** as a measured table beside the ceiling table. Each decode speed is compared with its ceiling. A model whose agent steps would take too long at that speed is ruled out in writing. Record the choice in `CLAUDE.md`.
- [ ] **Step 5: Commit** the spec and `CLAUDE.md` changes.

```bash
git add docs CLAUDE.md
git commit -m "Record measured CPU model speeds and the small-tier choice"
```

---

## Self-review

- **Spec coverage.** Section 9 (ledger): Task 1. Section 10 (denials, hostname scrub is plan 4): Task 2. Section 5 tier config and 5.1 sizing: Tasks 3 to 5. Section 12 measurements: Task 5 covers the CPU speed and resident memory rows. The canary-stability and transcript-replay rows belong to plans 3 and 4. Sections 6, 7, 8 and 11 are plans 2 to 4.
- **Placeholders.** The only sentinel is `CHANGE-ME` in the example config, and a test proves the loader refuses it.
- **Type consistency.** `Ledger.append(event, stage, **data)` is called the same way in Tasks 1 and 4. `check_argv` and `check_string` signatures match between the interfaces block, code and tests.
- **Not executed.** The code in this plan has not been run. The tests are written against the intended behavior, and Steps 2 and 4 of each task are where a mistake in either shows up. The `cd` tracking in Task 2 and the shlex tokenizing of `;`, `&&` and `|` are the parts most likely to need a fix. I checked `cwd` handling by reading it twice and fixed one bug that way, so expect more.
