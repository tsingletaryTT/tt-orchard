# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard kernel-probe`: can the local coder write a tt-lang kernel? (spec section 5, Part B)

An attempt is one fresh agent conversation in `DIR/<op>/attempt-<n>/`. The directory holds copies of the kernel-loop
templates, the op's spec as `kernel_spec.json`, `check_config.json` (the simulator and the tt-lang facts) and
`reference/eltwise_add.py`, copied from the tt-lang checkout (it is not vendored into this repository). The agent
writes `kernel.py` and runs `python3 kernel_check.py` as often as it likes.

The verdict never comes from a file the agent can write. The agent's shell can change any file in its directory,
including its copy of the check and the spec. After the agent ends, `authoritative_check` copies kernel.py alone into
a fresh directory and runs the SHIPPED kernel_check.py (the template under orchard/skills/kernel-loop-templates/) on it, with the
SHIPPED spec, under the stall watcher. When the agent's own check.json says it passed and the authoritative check
says it did not, or its hashes do not match the shipped files, the attempt records `agent_check_untrusted: true`.

The hardware phase runs only on an attempt whose authoritative simulator check passed, and only after a smoke test
(upstream's example on one chip) passed. Every hardware command is a `gozer run --chips 1 --exact 1`; the probe never
resets a chip. A hardware run that the stall watcher kills is a hang: the probe stops with ProbeBlocked naming the
attempt. Before any gozer call the hardware path applies the kernel check's static gate again.

The probe keeps its own records outside the attempt directory, in `DIR/.verdict/<op>-<n>/`: the clean copy of
kernel.py, the authoritative check's result and output, the hardware result and output, and the probe's attempt.json.
A run's report is built from the records run_attempt returned in that run (and the hardware phase added to). Each run
gets a fresh probe id. load_attempts reads the probe-owned records back for a later resume and is not used to build a
report. kernel.py is copied only when it is a regular file (not a link, a
directory or a device) of at most MAX_KERNEL_BYTES; anything else ends the attempt with no check and no hardware run.
The sha256 of the copied bytes is recorded, and the hardware phase refuses a copy whose hash has changed. A
convenience copy of attempt.json is written into the attempt directory after whatever was at that path is removed,
so the probe never writes through a link the agent left.

These measures stop shortcuts and accidents: a model that edits its check, plants a link, or leaves a file in the
way. They do not stop an agent determined to attack the operator's account. Agent shells run as the operator. A
process the agent starts in the background carries ORCHARD_RUN_DIR=<attempt directory>; when the agent step ends
(also after a crash or an interrupt) and before any verdict, `sweep_strays` stops every process of the operator's
that carries that exact value, and the attempt record lists them under `swept`. A process whose /proc environ no
longer holds the value (an exec with a new environment, or a program that overwrote that memory) escapes the sweep
and can write anywhere the operator can, including the `.verdict` directory. The static gate and this boundary are
best effort.

The budget (BUDGET) and the watchdog's detectors bound each attempt. Temperature 0 (the default) asks the server for
greedy decoding. Attempts then often repeat each other but are not guaranteed to be identical, so the report counts
the distinct kernels per op and temperature; it gives counts with no rates.

`main` is the command `tt-orchard kernel-probe`. It starts no model server: the coder must already answer at its
endpoint (starting it is `tt-orchard bringup`'s job), and the simulator environment must pass
`tt-orchard kernel-env --check`. Exit 0 means a report was written, whatever the attempts' results; 2 is a refusal
before anything ran; 3 means an attempt or the hardware phase had a crash; 4 means an interrupt (Ctrl-C or SIGTERM)
stopped the run; 5 means a hardware run hung. With 3, 4 and 5 the report covers what ran before the stop.
"""
from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from orchard import defaults, hwtool, ttlang_env
from orchard.agent import DEVICE_VARS, SECRET_NAME, TOOL_SCHEMAS, AgentStep, Tools, agent_env, probe_model
from orchard.canary import post_json
from orchard.ledger import Ledger
from orchard.stallwatch import StallWatcher, Watched, run_watched
from orchard.watchdog import (IdenticalResponses, Ladder, NoFileWritten, RepeatedToolCall, RetryGuard,
                              ThinkingWithoutAction, TurnRepeat, Watchdog)

__all__ = ["Watched"]       # re-exported: tests and callers build fake run_watched results with it

TEMPLATES = Path(__file__).resolve().parent / "skills" / "kernel-loop-templates"
SKILL = Path(__file__).resolve().parent / "skills" / "kernel-probe.md"
OPS = defaults.KERNEL_PROBE_OPS
# Provisional: none of these has been measured (spec section 5.3).
# max_tokens caps each reply of the coder. Choice, from 27 recorded attempts: the longest reply was 2706 tokens, and a
# request that timed out (6 of 14 attempts at temperature 0.7, 1 of 14 at 0) was a reply of about 8K tokens decoding at
# about 9 tok/s. At 4096 a degenerate reply ends as finish_reason "length" in about 7 minutes at that speed, which the
# agent loop nudges, and not as a 900 s request timeout (twice). The generic loop's 16384 is for reasoning models.
BUDGET = {"max_turns": 40, "wall_s": 1800.0, "sim_timeout_s": 300.0, "hw_backstop_s": 600.0, "max_tokens": 4096}
MAX_WALL_S = 7200.0       # choice: the largest --wall-s; two hours for one attempt is more likely a typo than a plan
GOZER_WHO = "claude:orchard-kernel-probe"
# How long a `gozer run` gets after the watcher's SIGTERM before the SIGKILL. gozer forwards the SIGTERM to its child,
# waits for it, and then releases its lease in a `finally`; the release resets the chip, measured at 41.7 s on this
# box, and gozer also scans /proc and revalidates the lease. A grace shorter than that kills gozer during the reset and
# leaves the lease under a dead pid. The watcher stops waiting as soon as the group is gone, so the normal path does
# not wait out the whole grace.
GOZER_TERM_GRACE_S = 120.0
# The stall watcher's limits for the smoke test and each hardware run. Not measured: the report gives the longest
# quiet stretch and the longest repeat the watcher saw on each run, so these can be set from a measurement.
HW_QUIET_S = 60.0
HW_REPEAT_LINES = 20
AGENT = "kernel-probe"
# One shell command of the agent may run this long: one simulator check plus a margin. The wall budget is checked
# between turns, so an attempt can overrun it by up to this much.
AGENT_TOOL_TIMEOUT_S = BUDGET["sim_timeout_s"] + 120
# The probe's own directory, beside the ops' attempt directories: DIR/.verdict/<op>-<n>/.
VERDICT_DIR = ".verdict"
# Inside a verdict directory, the fresh directory where the authoritative check runs. It holds the copy of kernel.py
# and the files the check writes. The hardware phase runs this same copy.
CLEAN_DIR = "authoritative"
MAX_KERNEL_BYTES = 256 * 1024     # choice: the fixtures are under 4 KiB; a kernel this large is not a kernel
TEMPLATE_FILES = ("kernel_check.py", "kernel_harness.py", "references.py")
# Every category an attempt can end in. The check and the harness produce all but `hang`, which the probe sets when
# the stall watcher kills a hardware run.
CATEGORIES = ("no_kernel", "import_error", "tt_lang_error", "shape_mismatch", "numeric_mismatch", "timeout",
              "compile_error", "hang", "hw_numeric_mismatch", "harness_error", "forbidden_code", "queued", "sim_gap")
SMOKE_TAIL_LINES = 20


class ProbeBlocked(Exception):
    """A hardware run hung. The probe stops and names the attempt; nothing is reset by hand.

    `hw` is the hardware block for the report header when the smoke test hung (None for a kernel hang, which is
    recorded in that attempt's own record)."""

    def __init__(self, message: str, *, hw: dict | None = None):
        super().__init__(message)
        self.hw = hw


def load_spec(op: str) -> dict:
    """The shipped spec of `op`. An op not in OPS is refused, so a name cannot reach outside the ops directory."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; the ops are {', '.join(OPS)}")
    return json.loads((TEMPLATES / "ops" / f"{op}.json").read_text(encoding="utf-8"))


def _template(name: str):
    """A template script loaded by path, for its functions (classify, the static gate, the bars)."""
    spec = importlib.util.spec_from_file_location(f"kernel_loop_{name}", TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def tt_lang_facts(env) -> dict:
    """What a report says about the tt-lang the kernels ran on, including whether the wheel carries the local patch."""
    target = ttlang_env.diagnostics_target(env)
    patched = bool(target and target.is_file() and ttlang_env.sha256_file(target) == defaults.TTLANG_DIAGNOSTICS_SHA256)
    return {"sim_version": ttlang_env.sim_version(env), "source_sha": ttlang_env.head_sha(env),
            "diagnostics_patched": patched, "patch_sha256": defaults.TTLANG_DIAGNOSTICS_SHA256}


LIBRARY = TEMPLATES / "library"
LIBRARY_SOURCE = "orchard worked-kernel library"
LIBRARY_MODES = ("all", "others", "none")


def library_references(op: str, mode: str = "all") -> list[dict]:
    """The worked kernels that go into an attempt's reference/ as `library_<op>.py`, in the shape of
    ttlang_env.read_references. `all` gives every kernel, `others` leaves out the one for `op` (so a pass measures
    writing a kernel from other worked kernels), `none` gives nothing."""
    if mode not in LIBRARY_MODES:
        raise ValueError(f"unknown library mode {mode!r}; the modes are {', '.join(LIBRARY_MODES)}")
    if mode == "none":
        return []
    out = []
    for path in sorted(LIBRARY.glob("*.py")):
        if mode == "others" and path.stem == op:
            continue
        data = path.read_bytes()
        out.append({"name": f"library_{path.name}", "source": LIBRARY_SOURCE, "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(), "data": data})
    return out


def library_note(op: str, mode: str) -> str | None:
    """The sentence the report carries when the coder was given a worked kernel for the op it was asked to write."""
    if mode == "all" and (LIBRARY / f"{op}.py").is_file():
        return (f"The library held a worked kernel for {op}, so a pass here shows that the coder can adapt a worked "
                "kernel. It does not show that the coder can write the kernel from the specification alone.")
    return None


def write_references(adir: Path, references: list[dict]) -> list[dict]:
    """Write the reference files (ttlang_env.read_references) into `adir/reference/` as new regular files and return
    what was written: [{"name": "reference/<name>", "source", "path", "sha256"}], the sha256 of the bytes written."""
    ref_dir = Path(adir) / "reference"
    _real_dir(ref_dir)
    out = []
    for r in references:
        data = r["data"]
        _write_new(ref_dir / r["name"], data)
        out.append({"name": f"reference/{r['name']}", "source": r["source"], "path": r["path"],
                    "sha256": hashlib.sha256(data).hexdigest()})
    return out


def prepare_attempt(adir: Path, *, op: str, env, tt_lang: dict | None = None,
                    references: list[dict] | None = None) -> list[dict]:
    """Fill an EMPTY attempt directory with what the agent reads and runs. `tt_lang` defaults to tt_lang_facts(env).
    Returns the reference files written, with their sha256 (write_references).

    `references` are the files ttlang_env.read_references read once for the whole probe (the tt-lang skill's guide and
    examples, the checkout's specification and two examples). `main` always passes them, after refusing to start when
    any is missing. Without them (unit tests of other parts) only the checkout's eltwise_add.py is copied, as before.

    Whatever is at the attempt's path is removed first, without following a link (a real directory is removed with its
    contents, a link or a file is unlinked), so nothing an earlier run left there (a kernel.py, a check.json) can be
    picked up."""
    load_spec(op)                                       # refuses an unknown op before anything is written
    adir = Path(adir)
    _real_dir(adir.parent)
    _remove(adir)
    adir.mkdir()
    for name in TEMPLATE_FILES:
        shutil.copy2(TEMPLATES / name, adir / name)
    shutil.copy2(TEMPLATES / "ops" / f"{op}.json", adir / "kernel_spec.json")
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    (adir / "check_config.json").write_text(json.dumps({"sim": str(env.sim), "timeout_s": BUDGET["sim_timeout_s"],
                                                        "tt_lang": facts}, indent=2) + "\n", encoding="utf-8")
    if references is not None:
        return write_references(adir, references)
    ref = Path(env.src_dir) / "examples" / "eltwise_add.py"
    if ref.is_file():
        (adir / "reference").mkdir(exist_ok=True)
        shutil.copy2(ref, adir / "reference" / "eltwise_add.py")
    return []


CHECK_AFTER_WRITE_CHARS = 1500          # the most of the check's output that is shown after a write


class ProbeTools(Tools):
    """The agent's tools, with one addition: after every write of kernel.py the shipped check runs and its output is
    appended to the write's result.

    Pilot 4 (2026-10-17): the coder wrote kernel.py three times and never ran kernel_check.py, although the skill says
    to after each write, so it never saw the failing line or the hint. This is the same `python3 kernel_check.py` the
    agent may run itself, so the authoritative check, which runs in a clean directory, is not affected."""

    @property
    def schemas(self) -> list:
        """The tool schemas the model is sent: the generic ones with read_file's offset in lines and a limit in lines."""
        out = []
        for schema in TOOL_SCHEMAS:
            if schema["function"]["name"] == "read_file":
                schema = copy.deepcopy(schema)
                schema["function"]["description"] = (
                    "Read a text file inside the run directory (read only). The path is relative to the run directory, "
                    "or to your stage directory when it is not found there. A long file comes back one page at a time: "
                    "the page ends with the offset to pass to read the next one.")
                props = schema["function"]["parameters"]["properties"]
                props["offset"] = {"type": "integer", "description": "first line to read, counting from 1 (the number "
                                   "grep -n prints); default 1"}
                props["limit"] = {"type": "integer", "description": "how many lines to read (default: as many as fit)"}
            out.append(schema)
        if self.hardware is not None:
            out.append(self.hardware.schema())
        return out

    hardware = None                      # the attempt's hwtool.HardwareCheck, or None when the tool is not offered

    def call(self, name: str, arguments: str) -> str:
        if name == "read_file":
            try:
                args = json.loads(arguments or "{}")
            except ValueError:
                return "error: the arguments are not JSON"
            if isinstance(args, dict):
                unknown = self.unknown_read_args(args, ("path", "offset", "limit"), "read_file takes path, offset "
                                                 "(the first line, from 1) and limit (a number of lines)")
                if unknown:
                    return unknown
                return self.read_lines(args.get("path"), args.get("offset"), args.get("limit"))
        if name == "hardware_check" and self.hardware is not None:
            try:
                args = json.loads(arguments or "{}")
            except ValueError:
                return "error: the arguments are not JSON"
            if not isinstance(args, dict):
                return "error: the arguments must be a JSON object"
            return self.hardware.call(args)
        return super().call(name, arguments)

    def read_lines(self, path, offset=None, limit=None) -> str:
        """read_file for the probe: `offset` is the first line (from 1) and `limit` a number of lines. The coder kept
        passing grep -n line numbers as the generic tool's character offsets (pilot 1 and 5, 2026-10-17). A number
        sent as a string, as the coder sends `limit`, is accepted. A page ends on a whole line, within the output limit."""
        def whole(v, default, low):
            if v is None:
                return default
            if isinstance(v, bool) or not isinstance(v, (int, str)) or (isinstance(v, str) and not v.strip().isdigit()):
                return None
            v = int(v)
            return v if v >= low else None
        first, count = whole(offset, 1, 0), whole(limit, None, 1)
        if first is None or (limit is not None and count is None):
            return "error: offset is the first line to read (1 or more) and limit a number of lines (1 or more)"
        first = max(first, 1)
        found, error = self.readable(path)
        if error:
            return error
        text = Path(found).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if first == 1 and count is None and len(text) <= self.limit:
            return text
        if first > len(lines):
            return f"error: offset {first} is past the end of the file ({len(lines)} lines)"
        chosen, size = [], 0
        for line in lines[first - 1:first - 1 + (count or len(lines))]:
            if chosen and size + len(line) > self.limit:
                break
            chosen.append(line)
            size += len(line)
        last = first + len(chosen) - 1
        body = "".join(chosen).rstrip("\n")
        if last >= len(lines):
            return f"{body}\n[lines {first} to {last} of {len(lines)}: the end of the file]"
        return f"{body}\n[lines {first} to {last} of {len(lines)}; read on with offset={last + 1}]"

    last_failure = None                  # the FAIL line of the latest check this tool ran, None after a PASS or before any

    def write_file(self, path, content) -> str:
        result = super().write_file(path, content)
        target = self.target(path) if isinstance(path, str) else None
        if target is None or target != Path(os.path.realpath(self.stage_dir)) / "kernel.py":
            return result
        if result.startswith("unchanged:") and self.last_failure:
            # The generic reply says "end the step now". After a failing check that sends the coder away from the fix:
            # in pilot 6 it rewrote the same text three times and the watchdog ended the attempt at 18 of 40 turns.
            return ("unchanged: kernel.py already holds exactly this content, and its last check failed "
                    f"({self.last_failure[:300]}). Change the code; writing the same text again does nothing.")
        if not result.startswith("wrote "):
            return result
        shown = self.shell("python3 kernel_check.py")
        lines = [l for l in shown.splitlines() if l.strip()]
        fails = [l for l in lines if l.startswith("FAIL:")]
        # Anything but a PASS is a failure: a check that was killed or crashed prints no FAIL line.
        self.last_failure = fails[0] if fails else (None if any(l == "PASS" for l in lines) else " | ".join(lines[:2]))
        return f"{result}\nkernel_check.py ran after this write:\n{shown[:CHECK_AFTER_WRITE_CHARS]}"


def attempt_tools(adir: Path, hardware=None) -> Tools:
    """The agent's tools. The attempt directory is both the run and the stage directory, so the agent reads and writes
    only there. The environment is agent_env's: the no-chip device mask, no credentials, HOME inside the attempt. Every
    shell command goes through the runner's rules (no pip, no gozer writes, no resets, no uploads). A shell command may
    run as long as one simulator check plus a margin. `hardware` (make_hardware_check) adds the `hardware_check` tool,
    the one way an attempt reaches a chip."""
    adir = Path(adir)
    tools = ProbeTools(adir, adir, agent_env(adir), timeout=AGENT_TOOL_TIMEOUT_S)
    tools.hardware = hardware
    return tools


class ProbeControl:
    """The supervisor's actuator reduced to what a probe attempt needs: the three things the watchdog's Ladder calls
    (nudge, escalate, pause) and the two things AgentStep asks (take_nudges, stop_reason). The watchdog's findings are
    the main stop; the wall clock is the backstop.

    stop_reason() is short, because it becomes the attempt's status: "escalate", "pause" or "wall-clock".
    `detail` says why in words."""

    def __init__(self, *, wall_s: float, clock=time.monotonic):
        self.deadline, self.clock = clock() + wall_s, clock
        self.nudges: list[str] = []
        self.stop: str | None = None
        self.detail: str | None = None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop, self.detail = "escalate", "the watchdog saw the agent loop again after a nudge"

    def pause(self, reason, evidence):
        self.stop, self.detail = "pause", reason

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        if self.stop:
            return self.stop
        if self.clock() >= self.deadline:
            self.detail = self.detail or "the attempt's wall-clock budget is spent"
            return "wall-clock"
        return None


def verdict_dir(probe_dir, op: str, n) -> Path:
    """The probe-owned directory of attempt `n` of `op`: DIR/.verdict/<op>-<n>/."""
    return Path(probe_dir) / VERDICT_DIR / f"{op}-{n}"


def _verdict_dir_of(adir) -> Path:
    """The verdict directory of an attempt directory DIR/<op>/attempt-<n>."""
    adir = Path(adir)
    return verdict_dir(adir.parent.parent, adir.parent.name, adir.name.removeprefix("attempt-"))


def _remove(path: Path) -> None:
    """Remove whatever is at `path` without following a link: a link or a file is unlinked, a real directory is
    removed with its contents (shutil.rmtree unlinks links inside it and does not follow them)."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _real_dir(path: Path) -> None:
    """Make `path` a real directory: anything else at that path (a link, a file) is removed first."""
    try:
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode):
            _remove(path)
    except FileNotFoundError:
        pass
    path.mkdir(parents=True, exist_ok=True)


def _fresh_dir(path: Path) -> None:
    _remove(path)
    path.mkdir(parents=True)


def _write_new(path: Path, data: bytes) -> None:
    """Write a new file at `path`. Whatever was there is removed first, and the open refuses to follow a link or to
    reuse a file that appeared in between (O_CREAT | O_EXCL | O_NOFOLLOW)."""
    _remove(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _read_regular(path: Path, limit: int) -> bytes | None:
    """The bytes of `path` if it is a regular file (checked on the open file, which is opened without following a link
    and without blocking on a FIFO), at most `limit` + 1 of them. None otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None
        return f.read(limit + 1)


def copy_kernel(adir: Path, clean: Path) -> tuple[str | None, str | None, str]:
    """Copy the agent's kernel.py into `clean`. (category or None, sha256 of the copied bytes, first line).

    Only a regular file of at most MAX_KERNEL_BYTES is copied. A missing file is `no_kernel`. A link, a directory, a
    device or a larger file is `forbidden_code`, and nothing of its content is read into any record."""
    src = Path(adir) / "kernel.py"
    try:
        st = os.lstat(src)
    except FileNotFoundError:
        return "no_kernel", None, "kernel.py does not exist"
    if not stat.S_ISREG(st.st_mode):
        return "forbidden_code", None, "kernel.py is not a regular file (it is a link, a directory or a device)"
    if st.st_size > MAX_KERNEL_BYTES:
        return "forbidden_code", None, f"kernel.py is larger than 256 KiB ({st.st_size} bytes)"
    data = _read_regular(src, MAX_KERNEL_BYTES)
    if data is None:
        return "forbidden_code", None, "kernel.py is not a regular file (it changed while it was opened)"
    if len(data) > MAX_KERNEL_BYTES:
        return "forbidden_code", None, "kernel.py is larger than 256 KiB"
    _write_new(Path(clean) / "kernel.py", data)
    return None, hashlib.sha256(data).hexdigest(), ""


# ---- the environment of the probe's children --------------------------------------------------------------------
#
# The simulator check, the smoke test and the hardware run import or run code the probe did not write (the agent's
# kernel, upstream's example). They get agent_env's environment, as the agent's own shell does: an allow-list, a HOME
# that holds no credentials, a token path that does not exist. A child that inherited the probe's environment would see
# the operator's HOME, any token variable and SSH_AUTH_SOCK.

# Variables a hardware child needs from the probe's environment: where gozer keeps its leases and the tt-metal
# locations. A name that looks like a credential is never passed, whatever its prefix.
PASS_PREFIXES = ("GOZER_", "TT_METAL_CACHE", "TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "ARCH_NAME")


def child_env(home_run_dir, adir=None, *, hardware: bool, source=None) -> dict:
    """The explicit environment of one child of the probe.

    HOME is inside `home_run_dir`, a directory the probe owns (a verdict directory, or the scratch directory), so the
    child's user site-packages are not a directory the agent can write. ORCHARD_RUN_DIR names the attempt directory
    (`adir`) when there is one, so sweep_strays finds what the child leaves behind. A leased (`hardware`) child gets no
    device mask: gozer sets TT_VISIBLE_DEVICES for it, and the no-chip value in TT_METAL_VISIBLE_DEVICES would hide the
    leased chip. A child that is not leased keeps the mask."""
    source = os.environ if source is None else source
    extra = {k: v for k, v in source.items()
             if k.startswith(PASS_PREFIXES) and not SECRET_NAME.search(k) and k not in DEVICE_VARS}
    env = agent_env(home_run_dir, extra=extra, source=source)
    if adir is not None:
        env["ORCHARD_RUN_DIR"] = str(Path(adir).resolve())
    if hardware:
        for k in DEVICE_VARS:
            env.pop(k, None)
    return env


FILE_RULE_REMEDY = "write kernel.py with write_file as a plain file of at most 256 KiB"


def authoritative_check(adir: Path, *, op: str, env, run_watched=run_watched, tt_lang: dict | None = None,
                        vdir: Path | None = None) -> dict:
    """Run the SHIPPED check on a copy of the agent's kernel.py and return its result (check.json's fields plus
    `kernel_sha256`, the sha256 of the copied bytes).

    Everything happens in the probe-owned verdict directory, made fresh here: the clean copy of kernel.py
    (CLEAN_DIR), `check-authoritative.json` and `authoritative-output.txt`. Nothing is read back from the attempt
    directory except kernel.py, and that only when copy_kernel accepts it; otherwise no check runs. Every input the
    check would read from check_config.json is passed on the command line. A run the stall watcher killed is
    `timeout`, whatever file it left."""
    vdir = Path(vdir) if vdir is not None else _verdict_dir_of(adir)
    _real_dir(vdir.parent)
    _fresh_dir(vdir)
    clean = vdir / CLEAN_DIR
    clean.mkdir()
    category, sha, line = copy_kernel(adir, clean)
    if category is not None:
        rule = [{"rule": "file", "line": 0, "detail": line, "remedy": FILE_RULE_REMEDY}] \
            if category == "forbidden_code" else []
        return {"passed": False, "category": category, "first_line": line, "sets": [], "violations": rule,
                "kernel_sha256": None}
    out = vdir / "check-authoritative.json"
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    argv = [sys.executable, str(TEMPLATES / "kernel_check.py"), "--attempt-dir", str(clean),
            "--spec", str(TEMPLATES / "ops" / f"{op}.json"), "--sim", str(env.sim),
            "--timeout-s", str(BUDGET["sim_timeout_s"]), "--tt-lang-json", json.dumps(facts),
            "--out", str(out)]
    got = run_watched(argv, cwd=clean, env=child_env(vdir, adir, hardware=False), log_path=vdir / "authoritative-output.txt",
                      watcher=StallWatcher(quiet_s=BUDGET["sim_timeout_s"] / 2),
                      backstop_s=BUDGET["sim_timeout_s"] + 120)
    if got.killed:
        return {"passed": False, "category": "timeout", "first_line": got.reason, "sets": [], "violations": [],
                "kernel_sha256": sha}
    raw = _read_regular(out, 64 * 1024 * 1024)
    if raw is None:
        return {"passed": False, "category": "harness_error", "sets": [], "violations": [], "kernel_sha256": sha,
                "first_line": f"kernel_check.py wrote no result (exit {got.returncode})"}
    try:
        result = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        result = None
    if not isinstance(result, dict):
        return {"passed": False, "category": "harness_error", "sets": [], "violations": [], "kernel_sha256": sha,
                "first_line": "kernel_check.py wrote a result that is not a JSON object"}
    result["kernel_sha256"] = sha
    return result


def set_measures(sets: list) -> dict:
    """The worst value of each measure over the input sets: the smallest PCC, the range of slopes, the largest relative
    L2 error. A null value (the harness could not compute it) is left out; a measure with no values is null."""
    def values(key):
        return [s[key] for s in sets if isinstance(s, dict) and isinstance(s.get(key), (int, float))
                and not isinstance(s.get(key), bool)]
    pcc, slope, rel = values("pcc"), values("slope"), values("rel_l2")
    return {"pcc_min": min(pcc) if pcc else None, "slope_range": [min(slope), max(slope)] if slope else None,
            "rel_l2_max": max(rel) if rel else None}


def provenance(*, kernel_sha256, tt_lang, coder_model, probe_id, simulator_pcc_min, hardware_pcc_min=None,
               sampling: dict | None = None, references: list[dict] | None = None, skills: dict | None = None) -> dict:
    """The provenance block of a probe attempt (spec section 6), with the sampling the coder was asked for, the
    reference files it was given (name, source, path and sha256 of each) and the skills freshness check (each
    source's state and shas, and whether a stale source was allowed)."""
    return {"origin": "kernel", "kernel_sha256": kernel_sha256, "tt_lang": tt_lang, "coder_model": coder_model,
            "probe_id": probe_id, "simulator_pcc_min": simulator_pcc_min, "hardware_pcc_min": hardware_pcc_min,
            "sampling": sampling, "references": references, "skills": skills}


def _shipped_hashes(op: str) -> dict:
    """The sha256 values a check.json written by the shipped, unedited check records for its own files."""
    out = {name: _sha256(TEMPLATES / name) for name in TEMPLATE_FILES}
    out["spec"] = _sha256(TEMPLATES / "ops" / f"{op}.json")
    return out


def agent_check_untrusted(agent_check, authoritative: dict, op: str) -> bool:
    """Whether the agent's own check.json disagrees with the shipped check. `agent_check` is the parsed file, None when
    the agent left none (never untrusted), or "unreadable" when the file exists and is not a JSON object.

    Untrusted means: it says passed and the authoritative check does not, or any of its hashes of the check, the
    harness, the references or the spec differs from the shipped file (a missing hash counts as different: the
    shipped check always writes them)."""
    if agent_check is None:
        return False
    if not isinstance(agent_check, dict):
        return True
    if agent_check.get("passed") and not authoritative.get("passed"):
        return True
    recorded = agent_check.get("sha256") if isinstance(agent_check.get("sha256"), dict) else {}
    return any(recorded.get(name) != want for name, want in _shipped_hashes(op).items())


AGENT_CHECK_MAX_BYTES = 1024 * 1024     # choice: a real check.json is a few KiB


def _read_agent_check(path: Path):
    """The agent's own check.json: the parsed object, None when there is none, or "unreadable" when it is not a regular
    file (a link, a FIFO, a device), is larger than AGENT_CHECK_MAX_BYTES, or is not a JSON object. It is read through
    _read_regular, so a FIFO cannot block the probe and a link to /dev/zero is never read."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(st.st_mode):
        return "unreadable"
    raw = _read_regular(path, AGENT_CHECK_MAX_BYTES)
    if raw is None or len(raw) > AGENT_CHECK_MAX_BYTES:
        return "unreadable"
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return "unreadable"
    return data if isinstance(data, dict) else "unreadable"


def _write_json(path: Path, data) -> None:
    """Write a probe-owned JSON file (never through a link)."""
    _write_new(Path(path), (json.dumps(data, indent=2) + "\n").encode("utf-8"))


def _save_attempt(adir: Path, attempt: dict) -> None:
    """Write the attempt record. The verdict lives in the probe-owned directory; the copy in the attempt directory is a
    convenience for a person reading it, written after whatever was at that path is removed, and is never read back."""
    vdir = _verdict_dir_of(adir)
    _real_dir(vdir.parent)                              # a fresh run has no .verdict yet; a link there is replaced
    _real_dir(vdir)
    _write_json(vdir / "attempt.json", attempt)
    try:
        _write_json(Path(adir) / "attempt.json", attempt)
    except OSError:
        pass                                            # the convenience copy is optional


RECORD_MAX_BYTES = 64 * 1024 * 1024
ATTEMPT_NUMBER = re.compile(r"[1-9][0-9]*")         # a positive whole number in ASCII digits, no leading zero


def _load_record(entry: Path, ops, probe_id) -> dict | None:
    """One `.verdict/<op>-<n>/attempt.json`, or None when it does not belong to this probe (see load_attempts)."""
    op, sep, number = entry.name.rpartition("-")
    # ASCII digits only: str.isdigit also accepts superscripts and other scripts' digits ("²", "٥").
    if not sep or op not in ops or not ATTEMPT_NUMBER.fullmatch(number):
        return None
    try:
        if not stat.S_ISDIR(os.lstat(entry).st_mode) or not stat.S_ISREG(os.lstat(entry / "attempt.json").st_mode):
            return None
    except OSError:
        return None
    raw = _read_regular(entry / "attempt.json", RECORD_MAX_BYTES)
    if raw is None or len(raw) > RECORD_MAX_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("op") != op:
        return None
    n = data.get("attempt")
    if not isinstance(n, int) or isinstance(n, bool) or n != int(number):
        return None
    prov = data.get("provenance")
    if not isinstance(prov, dict) or prov.get("probe_id") != probe_id:
        return None
    return data


def load_attempts(probe_dir, *, ops, probe_id) -> tuple[list[dict], int]:
    """(this probe's attempt records, the number of other entries ignored), for resuming a probe. A run's own report is
    built from the records run_attempt returned in that run.

    Only `.verdict/<op>-<n>/attempt.json` counts, where <op> is one of `ops`, <n> is a positive whole number without a
    leading zero, the directory and the file are real (no links), the file is a regular file of at most
    RECORD_MAX_BYTES, its `op` is <op>, its `attempt` is the int <n>, and its provenance names `probe_id`. Every other
    entry under `.verdict` (an earlier run's records, a planted one, a malformed one) is skipped and counted."""
    root = Path(probe_dir) / VERDICT_DIR
    try:
        if not stat.S_ISDIR(os.lstat(root).st_mode):
            return [], 0
        entries = sorted(root.iterdir())
    except OSError:
        return [], 0
    out, ignored = [], 0
    for entry in entries:
        rec = _load_record(entry, ops, probe_id)
        if rec is None:
            ignored += 1
        else:
            out.append(rec)
    order = {op: i for i, op in enumerate(ops)}
    return sorted(out, key=lambda a: (order[a["op"]], a["attempt"])), ignored


# ---- stray agent processes --------------------------------------------------------------------------------------
#
# An agent's shell command can start a process in the background and exit. spawn_checked stops a shell's session only
# on a timeout, so such a process would keep running as the operator through the verdict and the hardware phase. Every
# agent process carries ORCHARD_RUN_DIR=<attempt directory> (agent_env), so the probe finds them by that value and stops
# them right after the agent step ends, before any verdict.

SWEEP_GRACE_S = 2.0          # how long the strays get after SIGTERM before the SIGKILL
SWEEP_POLL_S = 0.05


RUN_DIR_VAR = "ORCHARD_RUN_DIR"


def _run_dir_entry(run_dir, var: str = RUN_DIR_VAR) -> bytes:
    return os.fsencode(var) + b"=" + os.fsencode(str(Path(run_dir).resolve()))


def _environ_has(proc_root: str, pid: int, entry: bytes) -> bool:
    """Whether /proc/<pid>/environ holds exactly `entry` as one of its variables. False for a process that has ended,
    a zombie (its environ reads empty) or one whose environ cannot be read."""
    try:
        with open(os.path.join(proc_root, str(pid), "environ"), "rb") as f:
            return entry in f.read().split(b"\0")
    except OSError:
        return False


def scan_strays(run_dir, *, proc_root: str = "/proc", uid: int | None = None, var: str = RUN_DIR_VAR) -> list[int]:
    """The pids of the processes owned by `uid` (default: ours) whose environment has ORCHARD_RUN_DIR equal to the
    resolved `run_dir`. The comparison is exact, so attempt-1 does not match attempt-10. An entry that cannot be read
    (the process ended during the scan, or it belongs to someone else) is skipped."""
    uid = os.getuid() if uid is None else uid
    entry = _run_dir_entry(run_dir, var)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    found = []
    for name in names:
        if not name.isdigit():
            continue
        try:
            if os.stat(os.path.join(proc_root, name)).st_uid != uid:
                continue
        except OSError:
            continue
        if _environ_has(proc_root, int(name), entry):
            found.append(int(name))
    return sorted(found)


def _protected_pids(proc_root: str = "/proc") -> set[int]:
    """Our own pid, our parent's, every further ancestor that /proc shows, and 0 and 1. None of them is signalled."""
    out = {0, 1, os.getpid(), os.getppid()}
    pid = os.getppid()
    for _ in range(4096):                                # a bound in case /proc shows a loop
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8", errors="replace") as f:
                text = f.read()
            pid = int(text[text.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid in out:
            break
        out.add(pid)
    return out


def _boot_id(proc_root: str) -> str | None:
    try:
        with open(os.path.join(proc_root, "sys", "kernel", "random", "boot_id"), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _cmdline(proc_root: str, pid: int) -> str:
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
            return f.read(4096).replace(b"\0", b" ").decode("utf-8", errors="replace").strip()[:120]
    except OSError:
        return ""


def _proc_state(proc_root: str, pid: int) -> tuple[str, int] | None:
    """(state letter, start time in clock ticks) of a process from /proc/<pid>/stat, or None when it does not exist."""
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8", errors="replace") as f:
            text = f.read()
        rest = text[text.rfind(")") + 2:].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def _gone(proc_root: str, pid: int, start: int | None) -> bool:
    """True when the process no longer runs: no /proc entry, a zombie, or a different process under the same pid
    (another start time)."""
    st = _proc_state(proc_root, pid)
    if st is None or st[0] == "Z":
        return True
    return start is not None and st[1] != start


def sweep_strays(run_dir, *, proc_root: str = "/proc", kill=os.kill, killpg=os.killpg, grace_s: float = SWEEP_GRACE_S,
                 clock=time.monotonic, sleep=time.sleep, var: str = RUN_DIR_VAR, identity: bool = False) -> list[dict]:
    """Stop every process that carries this attempt's ORCHARD_RUN_DIR: SIGTERM, up to `grace_s` for them to go, then a
    second scan and SIGKILL for everything it finds, so a process a stray started during the grace is caught too.
    Returns [{"pid", "cmdline", "stopped"}]. `var` names the environment variable to look for (ORCHARD_RUN_DIR for an agent's
    strays; the supervisor uses its own for a hardware test's servers). With `identity` each record also has `boot_id` and
    `start` (field 22 of /proc/<pid>/stat), taken before the process is signalled: a pid alone can be reused. `stopped` is True only when the process is gone afterwards (no /proc
    entry, a zombie, or another process under the pid); a process that still runs, for example under a new program
    without the variable, is recorded with `stopped` False. Never raises an Exception. An interrupt at any point after
    the first scan (during the SIGTERM pass or the grace) still gets the SIGKILL pass before it goes on.

    Our own pid and our ancestors are never signalled. Each signal goes to the pid itself (os.kill), except for a
    process that leads its own group, whose group is signalled so its children go too; a group is never signalled when
    its id is 1 or less or is our own group (the stallwatch guards). Before each signal the process's environment is
    read again, so a pid that ended and was reused by another process in between is not signalled.

    What it cannot see: a process whose /proc/<pid>/environ no longer holds the value (an exec with a new environment,
    or a program that overwrote that memory; clearenv() alone does not change it), and a process whose /proc entry
    cannot be read (a non-dumpable process, for example one that ran a setuid program, has a root-owned entry)."""
    records: list[dict] = []
    starts: dict[int, int | None] = {}
    try:
        protected = _protected_pids(proc_root)
        entry = _run_dir_entry(run_dir, var)

        def found() -> list[int]:
            return [pid for pid in scan_strays(run_dir, proc_root=proc_root, var=var)
                    if isinstance(pid, int) and pid > 1 and pid not in protected]

        def note(pids) -> None:
            for pid in pids:
                if pid not in starts:
                    st = _proc_state(proc_root, pid)
                    starts[pid] = st[1] if st else None
                    rec = {"pid": pid, "cmdline": _cmdline(proc_root, pid), "stopped": False}
                    if identity:
                        rec["boot_id"] = _boot_id(proc_root)
                        rec["start"] = starts[pid]
                    records.append(rec)

        def send(pid: int, sig) -> None:
            if not _environ_has(proc_root, pid, entry):
                return                                   # gone, a zombie, or no longer one of ours
            try:
                pgid = os.getpgid(pid)
                if pgid == pid and pgid > 1 and pgid != os.getpgrp() and pgid not in protected:
                    killpg(pgid, sig)
                else:
                    kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        def kill_pass() -> None:
            """The second scan and the SIGKILLs. Every step has its own guard, so this never raises: an error here
            cannot replace an interrupt that is on its way out."""
            try:
                note(found())                            # a second scan: new processes started during the grace
            except Exception:
                pass                                     # the processes already known still get their SIGKILL
            for r in records:
                try:
                    send(r["pid"], signal.SIGKILL)
                except Exception:
                    pass

        targets = found()
        note(targets)
        try:
            for pid in targets:
                send(pid, signal.SIGTERM)
            deadline = clock() + grace_s
            while clock() < deadline and any(_environ_has(proc_root, pid, entry) for pid in targets):
                sleep(SWEEP_POLL_S)
        except BaseException:
            kill_pass()                                  # never raises; an interrupt must not skip the SIGKILL
            raise
        kill_pass()
        deadline = clock() + 1.0                         # a killed process leaves /proc within moments
        while clock() < deadline and not all(_gone(proc_root, r["pid"], starts.get(r["pid"])) for r in records):
            sleep(SWEEP_POLL_S)
        for r in records:
            r["stopped"] = _gone(proc_root, r["pid"], starts.get(r["pid"]))
    except Exception:
        pass
    return records


def build_watchdog(control: ProbeControl, ledger) -> Watchdog:
    """The supervisor's loop detectors with their default thresholds, answering through the probe's control object.
    NoNewEvidence is left out: an attempt has no evidence directory the agent is asked to fill, and the wall-clock
    budget covers a slow agent."""
    return Watchdog([IdenticalResponses(), RepeatedToolCall(), TurnRepeat(), NoFileWritten(), ThinkingWithoutAction()],
                    Ladder(control, ledger, {AGENT}))


def _partial_attempt(box: dict, kw: dict, exc: BaseException) -> dict | None:
    """The record of an attempt that was cut short (an interrupt or a crash) after the agent called `hardware_check`, so
    that those calls stay on disk and in the report. None when the agent made no call. It says only what is known: the
    agent's hardware calls. The record is saved like any attempt; a failure to save it does not hide the exception."""
    hwcheck = box.get("hwcheck")
    if hwcheck is None or not hwcheck.records:
        return None
    sampling = {"temperature": kw.get("temperature", 0.0), "top_p": kw.get("top_p"), "seed": kw.get("seed")}
    attempt = {
        "op": kw["op"], "attempt": kw["n"], "status": "interrupted",
        "status_detail": f"{type(exc).__name__} before the attempt finished; the agent's hardware calls so far are kept",
        "turns": None, "agent_wall_s": None, "passed_sim": False, "category": None, "first_line": "",
        "pcc_min": None, "sets": [], "violations": [], "hw": None, "agent_check_untrusted": False, "swept": [],
        "sampling": sampling, "coder_speed": kw.get("coder_speed"), "agent_hw": hwcheck.summary(),
        "provenance": provenance(kernel_sha256=None, tt_lang=box.get("facts"), coder_model=kw.get("coder_model"),
                                 probe_id=kw.get("probe_id"), simulator_pcc_min=None, sampling=dict(sampling),
                                 skills=kw.get("skills"))}
    try:
        _save_attempt(Path(kw["probe_dir"]) / kw["op"] / f"attempt-{kw['n']}", attempt)
    except Exception:
        pass
    return attempt


def run_attempt(**kw) -> dict:
    """One attempt (see _run_attempt for the arguments). When an interrupt or a crash ends it after the agent called
    `hardware_check`, the exception carries `partial_attempt`, the saved record of what is known, so the caller can put
    it in the report."""
    box: dict = {}
    try:
        return _run_attempt(_box=box, **kw)
    except BaseException as exc:
        partial = _partial_attempt(box, kw, exc)
        if partial is not None:
            try:
                exc.partial_attempt = partial
            except Exception:
                pass
        raise


def _run_attempt(*, op, n, probe_dir, env, endpoint, model, coder_model, probe_id, max_turns=BUDGET["max_turns"],
                 wall_s=BUDGET["wall_s"], http=post_json, clock=time.time, watchdog=None, run_watched=run_watched,
                 tt_lang: dict | None = None, temperature: float = 0.0, top_p: float | None = None,
                 seed: int | None = None, references: list[dict] | None = None, skills: dict | None = None,
                 coder_speed: dict | None = None, hw_policy: hwtool.HwPolicy | None = None, shared_python=None, hw_status=None,
                 _box: dict | None = None) -> dict:
    """One attempt: prepare the directory, run the agent, run the authoritative check, write and return attempt.json.
    `hw_policy` (hwtool.HwPolicy) offers the agent the `hardware_check` tool, which runs its current kernel.py on one
    leased chip under the policy's limits with `shared_python`; None or a policy with 0 calls offers no tool. The calls
    are recorded under `agent_hw`. They are not the attempt's hardware verdict (`hw`).
    `coder_speed` is the decode-speed check made just before the attempt (speedwatch.probe_check), kept in the record.
    The sampling (temperature, top_p, seed) goes to the coder and into the record as `sampling`; the reference files
    written into the attempt (with their sha256) go into the provenance block."""
    sampling = {"temperature": temperature, "top_p": top_p, "seed": seed}
    adir = Path(probe_dir) / op / f"attempt-{n}"
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    copied = prepare_attempt(adir, op=op, env=env, tt_lang=facts, references=references)
    control = ProbeControl(wall_s=wall_s)
    ledger = Ledger(adir / "ledger.jsonl")
    hwcheck = None
    if hw_policy is not None and hw_policy.enabled:
        hwcheck = make_hardware_check(adir, op=op, n=n, shared_python=shared_python or default_shared_python(),
                                      policy=hw_policy, run_watched=run_watched, status=hw_status)
        if _box is not None:
            _box["hwcheck"], _box["facts"] = hwcheck, facts
    started = time.monotonic()
    try:
        dog = watchdog if watchdog is not None else build_watchdog(control, ledger)
        step = AgentStep(agent=AGENT, endpoint=endpoint, model=model, tools=attempt_tools(adir, hardware=hwcheck),
                         ledger=ledger,
                         stage=0, phase="probe", feed=dog.feed, control=control, run_dir=adir,
                         evidence_dir=adir / "evidence", log_path=adir / "log" / "probe-00001.jsonl", http=http,
                         clock=clock, guard=RetryGuard(), max_turns=max_turns, temperature=temperature, top_p=top_p,
                         seed=seed, max_tokens=BUDGET["max_tokens"])
        user = (f"Write kernel.py for the op `{op}` in kernel_spec.json. "
                "Start by reading kernel_spec.json and reference/eltwise_add.py, then the reference files the skill "
                "names for this kind of op.")
        outcome = step.run(SKILL.read_text(encoding="utf-8"), user)
    finally:
        ledger.close()
        # Before any verdict, and on a crash or an interrupt too: no process the agent started may keep running.
        swept = sweep_strays(adir)
    agent_wall_s = round(time.monotonic() - started, 2)
    try:
        check = authoritative_check(adir, op=op, env=env, run_watched=run_watched, tt_lang=facts)
    finally:
        # The check imports the agent's kernel, which can start a process that outlives the check. The children carry
        # ORCHARD_RUN_DIR=<attempt directory>, so the same sweep finds them.
        swept = swept + sweep_strays(adir)
    sets = check.get("sets") if isinstance(check.get("sets"), list) else []
    measures = set_measures(sets)
    passed = check.get("passed") is True
    attempt = {
        "op": op, "attempt": n, "status": outcome.status, "status_detail": outcome.detail or control.detail,
        "turns": outcome.turns, "agent_wall_s": agent_wall_s, "passed_sim": passed,
        "category": None if passed else check.get("category"), "first_line": check.get("first_line") or "",
        **measures, "violations": check.get("violations") or [], "sets": sets, "hw": None,
        "agent_check_untrusted": agent_check_untrusted(_read_agent_check(adir / "check.json"), check, op),
        "swept": swept, "sampling": dict(sampling), "coder_speed": coder_speed,
        "agent_hw": hwcheck.summary() if hwcheck is not None else {"enabled": False},
        "provenance": provenance(kernel_sha256=check.get("kernel_sha256"), tt_lang=facts,
                                 coder_model=coder_model, probe_id=probe_id, simulator_pcc_min=measures["pcc_min"],
                                 sampling=dict(sampling), references=copied or None, skills=skills),
    }
    _save_attempt(adir, attempt)
    return attempt


# ---- the hardware phase -----------------------------------------------------------------------------------------

def _gozer(reason: str, shared_python: str, *args: str) -> list[str]:
    """A one-chip gozer lease around a command run by the shared venv's python. The lease's release resets the chip."""
    return ["gozer", "run", "--chips", "1", "--exact", "1", "--who", GOZER_WHO, "--reason", reason, "--",
            shared_python, *args]


def _new_text(path: Path, offset: int) -> str:
    """What was appended to `path` after byte `offset` (run_watched appends to its log)."""
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _hw_watcher() -> StallWatcher:
    return StallWatcher(quiet_s=HW_QUIET_S, repeat_lines=HW_REPEAT_LINES)


def _watch_facts(got) -> dict:
    """What the stall watcher saw on one hardware run, beside the limits it applied."""
    return {"max_quiet_s": round(float(getattr(got, "max_quiet_s", 0.0) or 0.0), 1),
            "max_repeat": int(getattr(got, "max_repeat", 0) or 0),
            "quiet_limit_s": HW_QUIET_S, "repeat_limit": HW_REPEAT_LINES}


def _watch_text(w: dict) -> str:
    return (f"longest quiet stretch {w['max_quiet_s']:.1f} s, longest repeat {w['max_repeat']} lines; "
            f"watcher limits {w['quiet_limit_s']:.0f} s and {w['repeat_limit']} lines")


GOZER_QUEUED_EXIT = 10      # `gozer run` prints "queued: ticket X" and exits 10 when it could not lease at once
_TICKET = re.compile(r"queued: ticket ([0-9A-Za-z]+)")


def cancel_ticket(ticket: str, run=subprocess.run) -> str | None:
    """Drop a gozer queue ticket (`gozer cancel`, which touches no chip). None when it worked, else the reason it did not.
    A ticket that could pass as an option is not handed to gozer."""
    if not re.fullmatch(r"[0-9A-Za-z]+", ticket or ""):
        return f"not a ticket name: {ticket!r}"
    try:
        got = run(["gozer", "cancel", ticket], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if got.returncode != 0:
        return ((got.stdout or "") + (got.stderr or "")).strip() or f"gozer cancel exited {got.returncode}"
    return None


def _queued(got, text: str, cancel) -> dict | None:
    """When `gozer run` did not run the command because the chips were busy (it queued a ticket and exited 10), cancel
    the ticket and return {"ticket", "cancel_error"}. None for any other result. A ticket left in the queue would be
    granted to a process that is not there, and the queue would wait on it until it expired."""
    if got.killed or got.returncode != GOZER_QUEUED_EXIT:
        return None
    m = _TICKET.search(text)
    if m is None:
        return {"ticket": None, "cancel_error": "no ticket in gozer's output, so nothing was cancelled"}
    return {"ticket": m.group(1), "cancel_error": cancel(m.group(1))}


def _queued_text(q: dict) -> str:
    cancelled = (f"the ticket {q['ticket']} was cancelled" if q["ticket"] and q["cancel_error"] is None
                 else f"the ticket could not be cancelled: {q['cancel_error']}; look with `gozer status`")
    return f"gozer queued the request because the chips were busy, so the command was queued, not run; {cancelled}"


def smoke_test(*, env, shared_python, run_watched=run_watched, record: dict | None = None,
               cancel=cancel_ticket) -> tuple[bool, list[str]]:
    """Upstream's examples/eltwise_add.py with the shared venv's tt-lang and ttnn, on one leased chip (spec 5.4).
    (ok, the last lines of its output). ok needs exit 0 and `PASSED!` in the output. A run the stall watcher
    killed raises ProbeBlocked. `record`, when given, receives `watch`: what the stall watcher saw (_watch_facts), and
    `queued` when gozer queued the request instead of running it (the ticket is cancelled; see _queued)."""
    Path(env.scratch_dir).mkdir(parents=True, exist_ok=True)
    log = Path(env.scratch_dir) / "smoke.txt"
    start = _size(log)
    argv = _gozer("kernel-probe smoke test", str(shared_python), str(Path(env.src_dir) / "examples" / "eltwise_add.py"))
    got = run_watched(argv, cwd=env.scratch_dir, env=child_env(env.scratch_dir, hardware=True), log_path=log, watcher=_hw_watcher(),
                      backstop_s=BUDGET["hw_backstop_s"], term_grace_s=GOZER_TERM_GRACE_S)
    text = _new_text(log, start)
    tail = text.splitlines()[-SMOKE_TAIL_LINES:]
    if record is not None:
        record["watch"] = _watch_facts(got)
    queued = _queued(got, text, cancel)
    if queued is not None:
        if record is not None:
            record["queued"] = queued
        return False, tail
    if got.killed:
        # A hang blocks the probe. The watcher's SIGTERM reaches gozer too, and gozer normally releases its lease
        # (with the chip reset) within GOZER_TERM_GRACE_S. Only a gozer that had to be killed after the grace can
        # leave its lease under a dead process.
        hw = {"available": False, "hung": True, "reason": got.reason, "smoke_tail": tail, "watch": _watch_facts(got)}
        last = " / ".join(tail[-5:]) or "(no output)"
        raise ProbeBlocked(f"the smoke test hung: {got.reason}. Its last lines: {last}. The watcher stopped it, and "
                           "gozer normally releases the lease when it stops. If gozer itself had to be killed (after "
                           f"the {GOZER_TERM_GRACE_S:.0f} s grace, or at once by a second interrupt during the grace), "
                           "the lease may still be held by a dead process: look with `gozer status` and use the "
                           "gozer-gatekeeper skill. Never reset a chip by hand", hw=hw)
    return (got.returncode == 0 and "PASSED!" in text), tail


def kernel_reached(text: str, result: dict | None = None) -> bool:
    """True when the hardware run got as far as the kernel: the harness printed `PROGRESS run <set>` (it prints that
    after the device is open and the inputs are built, just before it calls the operation), or its result file holds
    measured sets or an error from the run phase. A gozer failure, a setup failure, a kernel that fails to import and
    a stop before the harness reached the kernel give no such evidence. `text` is what this run wrote to the log."""
    if any(line.strip().startswith("PROGRESS run ") for line in text.splitlines()):
        return True
    if isinstance(result, dict):
        err = result.get("error")
        if isinstance(result.get("sets"), list) and result["sets"]:
            return True
        if isinstance(err, dict) and err.get("phase") == "run":
            return True
    return False


def hw_started(attempt: dict) -> bool:
    """True when the attempt has a hardware block from a run that started. A block with `started: false` records an
    attempt that never reached the chip (the checked copy failed a preflight check, or gozer queued the request). A
    block from before the key existed came from a run."""
    hw = attempt.get("hw")
    return isinstance(hw, dict) and hw.get("started") is not False


def hardware_only(attempt: dict) -> bool:
    """True for an attempt the simulator could not judge because it lacks a documented `ttl` function (the check's
    category `sim_gap`). Such a kernel goes to hardware without a simulator pass, and its hardware result says so."""
    return attempt.get("passed_sim") is not True and attempt.get("category") == "sim_gap"


def _record_hw(adir: Path, attempt: dict, hw: dict) -> dict:
    if hardware_only(attempt):
        hw = {**hw, "hardware_only": True}
    attempt["hw"] = hw
    if isinstance(attempt.get("provenance"), dict):
        attempt["provenance"]["hardware_pcc_min"] = hw.get("pcc_min")
    _save_attempt(adir, attempt)
    return hw


def _hw_gate(kernel: Path, recorded) -> dict | None:
    """The checks that come before any gozer command, on the copy of kernel.py that will run. None when the copy may run,
    else the hw block that says why not: no kernel, a rule of the static gate broken (`forbidden_code`), or a copy whose
    sha256 differs from `recorded`, the one taken when it was copied (`harness_error`)."""
    check = _template("kernel_check")
    if not kernel.is_file() or kernel.is_symlink():
        return {"passed": False, "started": False, "category": "no_kernel", "pcc_min": None, "sets": [],
                "violations": [], "first_line": "the checked copy of kernel.py is missing"}
    violations = check.kernel_source_violations(kernel)
    ok, reason = check.check_kernel_source(kernel)
    if violations or not ok:
        category = "no_kernel" if reason.startswith("no-operation") else "forbidden_code"
        # The same first line the simulator check gives (kernel_check.py strips the gate's "no-operation: " tag).
        return {"passed": False, "started": False, "category": category, "pcc_min": None, "sets": [],
                "violations": violations, "first_line": reason.removeprefix("no-operation: ")}
    data = _read_regular(kernel, MAX_KERNEL_BYTES)
    if data is None or recorded is None or hashlib.sha256(data).hexdigest() != recorded:
        return {"passed": False, "started": False, "category": "harness_error", "pcc_min": None, "sets": [],
                "violations": [], "first_line": "the clean kernel copy changed since the simulator check; it was not run"}
    return None


def _abs_executable(path) -> str:
    """`path` for a child that runs in another directory: a path with a directory part is made absolute (the child's
    working directory is the copy's, so a relative path would be read from there); a bare command name is left for the
    path lookup."""
    path = str(path)
    return os.path.abspath(path) if "/" in path and not os.path.isabs(path) else path


def _lease_facts(lease: dict | None, elapsed_s: float, call_timeout_s: float) -> dict:
    """What the wrapper's own record says about the lease, as hw-block fields. The chip time (`lease_s`) is the wrapper's seconds
    from its grant to the end of its release, from its own clock. A record that never finished (no end stamp: the wrapper was
    killed, or wrote nothing) is charged as an upper bound (`lease_upper_bound`): the larger of the full call timeout plus the release
    grace and the seconds the run took up to now (hwtool.unfinished_charge_s), whether or not a grant was recorded. A record that
    finished and was never granted is zero."""
    num = hwtool._number
    if not isinstance(lease, dict) or not num(lease.get("ended_at")):
        facts = {"lease_s": round(hwtool.unfinished_charge_s(call_timeout_s, elapsed_s), 1), "lease_upper_bound": True,
                 "waited_s": 0.0}
        if isinstance(lease, dict):
            started, granted = lease.get("started_at"), lease.get("granted_at")
            facts.update(wrapper_pid=lease.get("pid") if isinstance(lease.get("pid"), int) else None,
                         lease_id=lease.get("lease_id") if isinstance(lease.get("lease_id"), str) else None)
            if num(started) and num(granted):
                facts["waited_s"] = round(max(granted - started, 0.0), 1)
            facts["release_error"] = "the wrapper ended before it released the lease"
        return facts
    started, granted, ended = lease.get("started_at"), lease.get("granted_at"), lease.get("ended_at")
    facts: dict = {"wrapper_pid": lease.get("pid") if isinstance(lease.get("pid"), int) else None,
                   "lease_id": lease.get("lease_id") if isinstance(lease.get("lease_id"), str) else None}
    if not num(granted):                                          # finished without a grant: the time went on waiting
        waited = (ended - started) if num(started) else elapsed_s
        return {**facts, "lease_s": 0.0, "waited_s": round(max(waited, 0.0), 1)}
    waited = max(granted - started, 0.0) if num(started) else 0.0
    facts["waited_s"] = round(waited, 1)
    facts["lease_s"] = round(max(ended - granted, 0.0), 1)
    if lease.get("release_ok") is False:
        facts["release_error"] = str(lease.get("release_text") or "gozer release did not succeed")
    elif lease.get("release_ok") is None:
        facts["release_error"] = "the wrapper ended before it released the lease"
    return facts


def _hw_run(adir: Path, vdir: Path, kernel: Path, *, op: str, reason: str, shared_python, run_watched, cancel,
            backstop_s: float = BUDGET["hw_backstop_s"], wrapper: dict | None = None) -> tuple[dict, str | None, str]:
    """One gozer run of the shipped harness on the clean copy `kernel` (already through _hw_gate), judged with the
    hardware bars. Returns (hw block, event, output). `event` is None for a run that finished, "queued" when the chips
    were busy (the ticket is cancelled and the kernel never ran) and "hang" when the stall watcher or the backstop
    stopped it. The results and the output log are written in `vdir`. A process the kernel left behind is stopped
    (finding 5). `wrapper`, when given (`who`, `wait_s`), runs the harness under the short-lived wrapper of orchard/hwtool.py
    instead of `gozer run`: the wrapper holds the lease, waits for a busy board and releases, and writes its own record
    (hw-lease.json in `vdir`), which is read back into the hw block (`lease_s`, `waited_s`, `lease_id`, `wrapper_pid`)."""
    check = _template("kernel_check")
    spec = load_spec(op)
    adir, vdir, kernel = (Path(os.path.abspath(x)) for x in (adir, vdir, kernel))     # cwd is the copy's directory
    out = vdir / "hw-result.json"
    _remove(out)
    log = vdir / "hw-output.txt"
    start = _size(log)
    harness = [str(TEMPLATES / "kernel_harness.py"), "--spec", str(TEMPLATES / "ops" / f"{op}.json"),
               "--kernel", str(kernel), "--out", str(out), "--phase", "hw"]
    env = child_env(vdir, adir, hardware=True)
    lease_file = vdir / "hw-lease.json"
    if wrapper is None:
        argv = _gozer(reason, _abs_executable(shared_python), *harness)
    else:
        _remove(lease_file)
        # hwtool.py is stdlib only, so it runs as a script from any directory with the child's small environment.
        argv = [sys.executable, str(Path(hwtool.__file__).resolve()), "run", "--who", wrapper["who"], "--reason", reason,
                "--wait-s", repr(float(wrapper["wait_s"])), "--result", str(lease_file),
                "--sha256", str(wrapper["sha256"]), "--file", str(kernel),
                "--call-timeout-s", repr(float(wrapper["call_timeout_s"])), "--",
                _abs_executable(shared_python), *harness]
    try:
        got = run_watched(argv, cwd=Path(kernel).parent, env=env, log_path=log,
                          watcher=_hw_watcher(), backstop_s=backstop_s, term_grace_s=GOZER_TERM_GRACE_S)
    finally:
        # gozer has released the lease by now. A process the kernel left behind is stopped here (finding 5).
        swept = sweep_strays(adir)
    extra = {"swept": swept} if swept else {}
    text = _new_text(log, start)
    lease = hwtool.read_wrapper_record(lease_file) if wrapper is not None else None
    extra.update(_lease_facts(lease, got.elapsed_s, float(wrapper["call_timeout_s"])) if wrapper is not None else {})
    if wrapper is None:
        queued = _queued(got, text, cancel)
    else:
        queued = ({"ticket": lease.get("ticket"), "cancel_error": lease.get("cancel_error")}
                  if got.returncode == hwtool.WRAP_BUSY and not got.killed and lease and lease.get("outcome") == "busy" else None)
    if queued is not None:
        # The chips were busy and the kernel never ran: not a failure of the kernel.
        return ({"passed": False, "started": False, "category": "queued", "pcc_min": None, "sets": [], "violations": [],
                 "first_line": _queued_text(queued), "queued": queued, "watch": _watch_facts(got),
                 "elapsed_s": round(got.elapsed_s, 1), **extra}, "queued", text)
    if wrapper is not None and lease and lease.get("outcome") in ("no_lease", "changed") and not got.killed:
        why = (f"gozer could not give a lease: {lease.get('release_text')}" if lease["outcome"] == "no_lease"
               else str(lease.get("release_text")))
        return ({"passed": False, "started": False, "category": "harness_error", "pcc_min": None, "sets": [], "violations": [],
                 "first_line": why, "watch": _watch_facts(got),
                 "elapsed_s": round(got.elapsed_s, 1), **extra}, None, text)
    if wrapper is not None and lease and lease.get("outcome") == "timeout" and not got.killed:
        # The wrapper stopped the harness at the call timeout, counted from the grant: a hang, like a watcher stop.
        return ({"passed": False, "started": kernel_reached(text), "category": "hang", "pcc_min": None, "sets": [], "violations": [],
                 "first_line": f"call timeout: the harness was still running {wrapper['call_timeout_s']:.0f} s after the grant and was stopped",
                 "elapsed_s": round(got.elapsed_s, 1), "watch": _watch_facts(got), **extra}, "hang", text)
    if got.killed:
        # The first line is the watcher's own reason ("backstop: ..." or "stalled: ..." or "repeating: ..."), so a
        # watcher stop is not read as a device hang without saying which rule stopped it.
        return ({"passed": False, "started": kernel_reached(text), "category": "hang", "pcc_min": None, "sets": [], "violations": [],
                 "first_line": got.reason, "elapsed_s": round(got.elapsed_s, 1), "watch": _watch_facts(got), **extra},
                "hang", text)
    try:
        raw = _read_regular(out, 64 * 1024 * 1024)
        result = json.loads(raw.decode("utf-8", errors="replace")) if raw is not None else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        result = None                                   # classify reads a missing or broken result file the same way
    category, line = check.classify(kernel_present=True, timed_out=False, returncode=got.returncode,
                                    text=text, result=result,
                                    bar=spec["pcc_bar_hw"], phase="hw", bars=check.bars_from_spec(spec, "hw"),
                                    diagnostics_present=None)
    sets = result.get("sets") if result and isinstance(result.get("sets"), list) else []
    device = result.get("device_s") if result else None
    block = {"passed": category is None, "started": kernel_reached(text, result), "category": category, "first_line": line,
             **set_measures(sets), "sets": sets, "violations": [],
             "elapsed_s": round(got.elapsed_s, 1), "watch": _watch_facts(got), **extra}
    if isinstance(device, (int, float)) and not isinstance(device, bool) and 0 <= device < 1e7:
        block["device_s"] = round(float(device), 2)
    return block, None, text


def hardware_phase(adir: Path, attempt: dict, *, env, shared_python, run_watched=run_watched,
                   cancel=cancel_ticket) -> dict:
    """Run one attempt's checked kernel on one chip and judge it with the hardware bars. Call it only for an attempt
    whose authoritative simulator check passed, after a passing smoke test (run_hardware does both).

    It runs the copy of kernel.py the authoritative check judged (the verdict directory's CLEAN_DIR), with the
    shipped harness and spec. The static gate runs again first; a kernel that breaks a rule ends `forbidden_code` and
    no gozer command runs. Then the copy is hashed again: when it differs from the sha256 recorded at copy time, the
    attempt ends `harness_error` and no gozer command runs. A run the stall watcher kills is a hang: it is recorded,
    and ProbeBlocked names the attempt. Returns the `hw` block, which is also written into attempt.json."""
    adir = Path(os.path.abspath(adir))              # the child runs in another directory: every path it gets is absolute
    vdir = _verdict_dir_of(adir)
    op, n = attempt["op"], attempt["attempt"]
    load_spec(op)
    kernel = vdir / CLEAN_DIR / "kernel.py"
    blocked = _hw_gate(kernel, (attempt.get("provenance") or {}).get("kernel_sha256"))
    if blocked is not None:
        return _record_hw(adir, attempt, blocked)
    hw, event, _ = _hw_run(adir, vdir, kernel, op=op, reason=f"kernel-probe {op} attempt {n}",
                           shared_python=shared_python, run_watched=run_watched, cancel=cancel)
    if event == "queued":
        # The chips were busy and the kernel never ran: not a failure of the kernel, and the later attempts would be
        # queued too, so the probe stops.
        _record_hw(adir, attempt, hw)
        raise ProbeBlocked(f"{op} attempt {n}: {_queued_text(hw['queued'])}")
    if event == "hang":
        _record_hw(adir, attempt, hw)
        raise ProbeBlocked(f"{op} attempt {n}: {hw['first_line']}")
    return _record_hw(adir, attempt, hw)


# ---- the agent's hardware check ---------------------------------------------------------------------------------

HWCHECK_DIR = ".hwcheck"


def default_shared_python() -> str:
    return str(Path.home() / ".tenstorrent-venv" / "bin" / "python")


def hwcheck_calls_path(probe_dir, op: str, n) -> Path:
    """Where the calls of attempt `n` are written as they finish: DIR/.hwcheck/<op>-<n>-calls.json."""
    return Path(probe_dir) / HWCHECK_DIR / f"{op}-{n}-calls.json"


def hwcheck_dir(probe_dir, op: str, n, k) -> Path:
    """The probe-owned directory of call `k` of attempt `n` of `op`: DIR/.hwcheck/<op>-<n>-<k>/. It is not under
    `.verdict`, which the authoritative check empties and load_attempts reads."""
    return Path(probe_dir) / HWCHECK_DIR / f"{op}-{n}-{k}"


def make_hardware_check(adir: Path, *, op: str, n, shared_python, policy: hwtool.HwPolicy, run_watched=run_watched,
                        who: str | None = None, status=None, clock=time.monotonic,
                        sleep=time.sleep) -> hwtool.HardwareCheck:
    """The attempt's `hardware_check` tool (hwtool.HardwareCheck) wired to the probe's own hardware path.

    Each call copies the attempt's current kernel.py into a fresh probe-owned directory (copy_kernel: a regular file of
    at most 256 KiB) and runs the static gate and the sha256 check on the copy (_hw_gate). It then starts the short-lived
    wrapper of orchard/hwtool.py under the stall watcher (_hw_run). The wrapper holds the lease: it takes it with `gozer
    acquire --owner-pid <the wrapper's pid>`, waits on one ticket with `gozer wait` when the board is busy, checks that the
    copy still has the recorded sha256, runs the shipped harness under the lease, releases it and exits. The probe does not hold
    the lease and repairs nothing. On any exception the watcher kills the wrapper's process group. After every call that
    reached the wrapper the probe reads `gozer status` once (hwtool.held_by_us): a chip still held under this probe's tag, or a
    status that cannot be read, stops the tool, and the probe ends with exit 5. The agent supplies no command text and no path.

    `who` tags every lease of this probe process (default `GOZER_WHO:<pid>`), so the check can tell its own chips from another
    agent's. `status` is the runner for the one `gozer status` read (tests pass a fake)."""
    adir = Path(os.path.abspath(adir))              # absolute before any call directory is derived from it
    probe_dir = adir.parent.parent
    who = who or f"{GOZER_WHO}:{os.getpid()}"
    state: dict = {}

    def prepare(k):
        # The agent's shell can leave a process running that edits files. Stop every process of this attempt before the
        # kernel is copied, as run_attempt does before its authoritative check.
        sweep_strays(adir)
        vdir = hwcheck_dir(probe_dir, op, n, k)
        _real_dir(vdir.parent)
        _fresh_dir(vdir)
        clean = vdir / CLEAN_DIR
        clean.mkdir()
        category, sha, line = copy_kernel(adir, clean)
        if category is not None:
            rule = [{"rule": "file", "line": 0, "detail": line, "remedy": FILE_RULE_REMEDY}] \
                if category == "forbidden_code" else []
            return {"passed": False, "started": False, "category": category, "first_line": line, "pcc_min": None,
                    "sets": [], "violations": rule}, None
        blocked = _hw_gate(clean / "kernel.py", sha)
        if blocked is not None:
            return blocked, sha
        state.update(vdir=vdir, kernel=clean / "kernel.py", sha=sha)
        return None, sha

    def judged(hw, event):
        if event is None and not hw.get("passed") and hw.get("sets"):
            spec = load_spec(op)
            check = _template("kernel_check")
            line = check.numeric_detail(hw["sets"], spec["pcc_bar_hw"], check.bars_from_spec(spec, "hw"))
            if line:
                hw["numbers"] = line
        return hw, event

    def run(k, timeout_s, wait_s):
        """One wrapper run of the prepared copy: the wrapper takes the lease (waiting up to `wait_s` for a busy board), runs the
        harness and releases. An exception reaches the caller after run_watched has killed the wrapper's process group."""
        sweep_strays(adir)                              # before the run, so nothing interruptible follows a ticket
        blocked = _hw_gate(state["kernel"], state["sha"])
        if blocked is not None:                         # the copy changed after prepare: nothing is started
            return {**blocked, "elapsed_s": 0.0}, None, ""
        hw, event, text = _hw_run(adir, state["vdir"], state["kernel"], op=op, reason=f"kernel-probe {op} attempt {n} agent check {k}",
                                  shared_python=shared_python, run_watched=run_watched, cancel=None,
                                  backstop_s=hwtool.outer_backstop_s(wait_s, timeout_s),
                                  wrapper={"who": who, "wait_s": wait_s, "sha256": state["sha"], "call_timeout_s": timeout_s})
        return (*judged(hw, event), text)
    return hwtool.HardwareCheck(policy, prepare=prepare, run=run, clock=clock, sleep=sleep,
                                on_record=lambda summary: (_real_dir(probe_dir / HWCHECK_DIR),
                                                           _write_json(hwcheck_calls_path(probe_dir, op, n), summary)),
                                check=lambda pids: hwtool.held_by_us(who, pids, sh=status))


def _hw_int(args, name: str, default: int, low: int, high: int) -> int:
    v = getattr(args, name, None)
    if v is None:
        return default
    try:
        n = int(v) if not isinstance(v, bool) and float(v) == int(float(v)) else None
    except (TypeError, ValueError, OverflowError):
        n = None
    if n is None or not low <= n <= high:
        raise _Refused(f"--{name.replace('_', '-')} must be a whole number from {low} to {high}")
    return n


def _hw_float(args, name: str, default: float, low: float, high: float, *, open_low: bool) -> float:
    v = getattr(args, name, None)
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = None
    if x is None or isinstance(v, bool) or x != x or x > high or (x <= low if open_low else x < low):
        bound = f"more than {low:g}" if open_low else f"at least {low:g}"
        raise _Refused(f"--{name.replace('_', '-')} must be {bound} and at most {high:g}")
    return x


def hw_policy_from_args(args) -> hwtool.HwPolicy:
    """The agent's hardware-check policy from the command line (defaults: 6 calls, 300 lease seconds, 240 s per call,
    120 s of waiting for a busy board). `--no-agent-hw` and `--hw-checks 0` turn the tool off. A value out of range is
    refused (_Refused). The ranges are choices: a larger number is more likely a typo."""
    base = hwtool.HwPolicy()
    checks = _hw_int(args, "hw_checks", base.max_calls, 0, 50)
    lease = _hw_float(args, "hw_lease_s", base.lease_s, 0.0, 3600.0, open_low=True)
    timeout = _hw_float(args, "hw_call_timeout_s", base.call_timeout_s, 0.0, 1800.0, open_low=True)
    wait = _hw_float(args, "hw_wait_s", base.wait_s, 0.0, 3600.0, open_low=False)
    if getattr(args, "no_agent_hw", False):
        checks = 0
    return hwtool.HwPolicy(max_calls=checks, lease_s=lease, call_timeout_s=timeout, wait_s=wait)


def run_hardware(attempts: list[dict], *, probe_dir, env, shared_python, run_watched=run_watched,
                 state: dict | None = None, cancel=cancel_ticket) -> dict:
    """The hardware phase of a probe: one smoke test, then hardware_phase for each attempt that passed the simulator.

    A failed smoke test ends the phase with no further gozer call, and the report says `hw: unavailable` with the
    smoke test's last lines. A hung smoke test or kernel run raises ProbeBlocked, so no later attempt runs. Returns
    {"available", "smoke_tail"}.

    Results are kept as they happen, so an exception part way through loses nothing that finished: hardware_phase
    writes each attempt's `hw` block into its record in place, and the smoke test's result is stored in `state["smoke"]`
    before any kernel runs."""
    seen: dict = {}
    ok, tail = smoke_test(env=env, shared_python=shared_python, run_watched=run_watched, record=seen, cancel=cancel)
    extra = {"queued": seen["queued"]} if seen.get("queued") else {}
    if state is not None:
        state["smoke"] = {"available": ok, "smoke_tail": tail, "watch": seen.get("watch"), **extra}
    if not ok:
        return {"available": False, "smoke_tail": tail, "watch": seen.get("watch"), **extra}
    for attempt in attempts:
        if attempt.get("passed_sim") is True or hardware_only(attempt):
            adir = Path(probe_dir) / attempt["op"] / f"attempt-{attempt['attempt']}"
            hardware_phase(adir, attempt, env=env, shared_python=shared_python, run_watched=run_watched,
                           cancel=cancel)
    return {"available": True, "smoke_tail": tail, "watch": seen.get("watch")}


# ---- the report -------------------------------------------------------------------------------------------------

def report_header(*, env, coder_model, probe_id, ops, hw=None, tt_lang: dict | None = None,
                  ignored_records: int = 0, wall_s: float | None = None) -> dict:
    """The report's header: the tt-lang facts, the coder's model id, the probe id, the provisional bars of each op,
    the budget, and the hardware phase's result (None when it was not run)."""
    check = _template("kernel_check")
    wall = BUDGET["wall_s"] if wall_s is None else float(wall_s)       # --wall-s replaces the default wall budget
    bars = {}
    for op in ops:
        spec = load_spec(op)
        bars[op] = {phase: check.bars_from_spec(spec, phase) for phase in ("sim", "hw")}
    return {"tt_lang": tt_lang if tt_lang is not None else tt_lang_facts(env), "coder_model": coder_model,
            "probe_id": probe_id, "ops": list(ops), "bars": bars, "provisional": True, "budget": {**BUDGET, "wall_s": wall},
            "inputs": "synthetic", "hw": hw, "ignored_records": ignored_records,
            "budget_note": (f"The wall budget ({wall:.0f} s) is checked between turns, so an attempt can "
                            f"overrun it by up to one tool timeout ({AGENT_TOOL_TIMEOUT_S:.0f} s at most, for the "
                            "kernel check).")}


def _first_pass(rows: list[dict], passed) -> int | None:
    for a in sorted(rows, key=lambda a: a["attempt"]):
        if passed(a):
            return a["attempt"]
    return None


def _hw_passed(a: dict) -> bool:
    return isinstance(a.get("hw"), dict) and a["hw"].get("passed") is True


def _cell(value) -> str:
    """A table cell: no pipe and no line break may leak into the Markdown table."""
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")[:160]


def _pcc_cell(value) -> str:
    """A PCC cell. When six significant digits would round a value near 1 to `1`, the distance from 1 is printed with
    its sign (`1 - 6.0e-08`, or `1 + 1.0e-07` for a value a little above 1 from floating-point error), so the table
    never shows a perfect PCC for an imperfect one."""
    if isinstance(value, float) and value != 1.0 and f"{value:.6g}" == "1":
        return f"1 - {1.0 - value:.1e}" if value < 1.0 else f"1 + {value - 1.0:.1e}"
    return _cell(value)


def _status_cell(a: dict) -> str:
    """How the agent step ended. A step that did not end `done` also gets the first line of the reason."""
    status = a.get("status")
    detail = str(a.get("status_detail") or "").strip().splitlines()
    if status and status != "done" and detail:
        return _cell(f"{status}: {detail[0]}")
    return _cell(status)


def _agent_cell(a: dict) -> str:
    """The agent column: whether its own check.json was untrusted, and how many of its processes the sweep stopped."""
    parts = ["untrusted"] if a.get("agent_check_untrusted") else []
    swept = [r for r in a.get("swept") or [] if isinstance(r, dict)]
    stopped = [r for r in swept if r.get("stopped") is not False]
    left = [r for r in swept if r.get("stopped") is False]
    if stopped:
        parts.append(f"{len(stopped)} stray agent process{'es' if len(stopped) != 1 else ''} stopped")
    if left:
        pids = ", ".join(str(r.get("pid")) for r in left)
        parts.append(f"{len(left)} stray agent process{'es' if len(left) != 1 else ''} could not be confirmed stopped "
                     f"(pid{'s' if len(left) != 1 else ''} {pids})")
    return _cell("; ".join(parts))


def _agent_hw(a: dict) -> dict | None:
    """The attempt's record of the agent's hardware checks when the tool was offered, else None (off, or a record from
    before the tool existed)."""
    rec = a.get("agent_hw")
    return rec if isinstance(rec, dict) and rec.get("enabled") is True and isinstance(rec.get("calls"), list) else None


def _counted_calls(rec: dict) -> list[dict]:
    return [c for c in rec["calls"] if isinstance(c, dict) and c.get("counted") is True]


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v else 0.0


def _agent_hw_cell(a: dict) -> str:
    """The agent hw column: how many hardware checks the agent called, how many passed, and the chip seconds they took."""
    rec = _agent_hw(a)
    if rec is None:
        return _cell(None)
    calls = _counted_calls(rec)
    passed = sum(1 for c in calls if c.get("outcome") == "pass")
    seconds = sum(_num(c.get("lease_s")) for c in calls)
    return _cell(f"{len(calls)} call{'s' if len(calls) != 1 else ''} ({passed} passed), {seconds:.0f} s")


def _call_note(c: dict) -> str:
    parts = []
    if c.get("timed_out"):
        parts.append("timed out")
    if c.get("lease_upper_bound"):
        parts.append("chip time is an upper bound (interrupted while the run held the board)")
    if c.get("queued_polls"):
        state = "granted" if c.get("cancelled") is None else ("cancelled" if c["cancelled"] else "NOT cancelled")
        parts.append(f"{c['queued_polls']} ticket{'s' if c['queued_polls'] != 1 else ''}, {state}")
    if c.get("release_error"):
        parts.append(f"release failed: {c['release_error']}")
    if c.get("outcome") not in ("pass", "busy") and c.get("first_line"):
        parts.append(str(c["first_line"]))
    if c.get("note"):
        parts.append(f"note: {c['note']}")
    return "; ".join(parts)


def _agent_hw_lines(attempts: list[dict]) -> list[str]:
    """The report's section on the hardware checks the agents called: the policy, every call, and the chip time they cost.
    Left out when no record has an `agent_hw` field (a report of attempts from before the tool)."""
    recorded = [a for a in attempts if isinstance(a.get("agent_hw"), dict)]
    if not recorded:
        return []
    on = [a for a in recorded if _agent_hw(a) is not None]
    if not on:
        return ["## Hardware checks called by the coder", "",
                "Hardware checks called by the coder: off (the tool was not offered, so these attempts used the "
                "simulator alone).", ""]
    pol = on[0]["agent_hw"].get("policy") or {}
    lines = ["## Hardware checks called by the coder", "",
             f"Each attempt could call `hardware_check`: up to {pol.get('max_calls')} calls and "
             f"{_num(pol.get('lease_s')):.0f} lease seconds, {_num(pol.get('call_timeout_s')):.0f} s per call, "
             f"{_num(pol.get('wait_s')):.0f} s of waiting for a busy board. The probe judges each call; the coder's "
             "claim counts for nothing. These runs are not the attempt's hardware result (the `hardware` column).", "",
             "| op | attempt | call | outcome | total s | device s | chip s | waited s | notes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    counted, chip, waited = [], 0.0, 0.0
    for a in sorted(on, key=lambda a: (str(a.get("op")), a.get("attempt") or 0)):
        for c in a["agent_hw"]["calls"]:
            if not isinstance(c, dict):
                continue
            lines.append("| " + " | ".join([
                _cell(a.get("op")), _cell(a.get("attempt")), _cell(c.get("k")), _cell(c.get("outcome")),
                _cell(c.get("total_s")), _cell(c.get("device_s")), _cell(c.get("lease_s")), _cell(c.get("waited_s")),
                _cell(_call_note(c))]) + " |")
            if c.get("counted") is True:
                counted.append(c)
            chip += _num(c.get("lease_s"))
            waited += _num(c.get("waited_s"))
    n_att = sum(1 for a in on if _counted_calls(a["agent_hw"]))
    by = {k: sum(1 for c in counted if c.get("outcome") == k) for k in ("pass", "busy", "hang")}
    failed = len(counted) - sum(by.values())
    n = len(counted)
    lines += ["", f"{n} call{'s' if n != 1 else ''} in {n_att} attempt{'s' if n_att != 1 else ''} "
                  f"({by['pass']} passed, {failed} failed, {by['busy']} busy, {by['hang']} hang); "
                  f"{chip:.0f} s of chip time (the seconds gozer held a lease, the reset included), "
                  f"{waited:.0f} s waited for a board.",
              ""]
    for a in sorted(on, key=lambda a: (str(a.get("op")), a.get("attempt") or 0)):
        if a["agent_hw"].get("stopped"):
            lines += [f"STOPPED after {a.get('op')} attempt {a.get('attempt')}: {a['agent_hw']['stopped']}", ""]
    if len(on) != len(recorded):
        lines += [f"{len(recorded) - len(on)} attempt(s) had the tool off.", ""]
    return lines


def _sampling_text(sp: dict) -> str:
    """The run's sampling in one sentence, with the seed base needed to repeat a sampled run."""
    temps = ", ".join(f"{t:g}" for t in sp.get("temperatures") or [])
    top_p = "the server's default" if sp.get("top_p") is None else f"{sp['top_p']:g}"
    text = (f"Sampling: temperature {temps} ({sp.get('attempts_per_temperature')} attempts each), top_p {top_p}. "
            "No seed is sent at temperature 0.")
    if any(t != 0 for t in sp.get("temperatures") or []):       # only then does any attempt send a seed
        text += (f" Above 0, attempt k sends seed {sp.get('seed_base')} + k (base from {sp.get('seed_base_from')}; "
                 f"send the same requests again with --seed {sp.get('seed_base')}).")
    return text


def _group_lines(g: dict, rows: list[dict]) -> list[str]:
    """One temperature group of an op: a heading, the attempts table and the distinct-kernel lines."""
    lines = ["", f"### temperature {g['temperature']:g}", "",
             "| attempt | simulator | hardware | status | PCC min (sim) | PCC min (hw) | turns | wall s | category | "
             "first line | agent check | simulator detail | agent hw |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in sorted(rows, key=lambda a: a["attempt"]):
        h = a.get("hw") if isinstance(a.get("hw"), dict) else None
        category = a.get("category") or (h or {}).get("category")
        first_line = a.get("first_line") if a.get("category") else (h or {}).get("first_line")
        detail = "-"
        if h is not None and hardware_only(a):
            # The simulator could not judge this attempt, so its own category (`sim_gap`) is not the result. The
            # hardware category and diagnostic are; the simulator's go in their own column.
            category, first_line = h.get("category"), h.get("first_line")
            detail = f"{a.get('category')}: {a.get('first_line')}"
        lines.append("| " + " | ".join([
            _cell(a["attempt"]), "yes" if a.get("passed_sim") else "no",
            "-" if h is None else (("yes" if h.get("passed") else "no") if hw_started(a) else "not run")
            + (" (hardware only)" if h.get("hardware_only") else ""),
            _status_cell(a),
            _pcc_cell(a.get("pcc_min")), _pcc_cell((h or {}).get("pcc_min")), _cell(a.get("turns")),
            _cell(a.get("agent_wall_s")), _cell(category), _cell(first_line), _agent_cell(a), _cell(detail),
            _agent_hw_cell(a)]) + " |")
    return lines + [""] + _distinct_lines(g)


def _sim_gap_lines(gap: list[dict]) -> list[str]:
    """What the report says about attempts the simulator could not judge (category `sim_gap`): a hardware result is
    the only check they got, and a simulator pass is not claimed for them."""
    ran = [a for a in gap if hw_started(a)]
    unrun = len(gap) - len(ran)
    lines = []
    if ran:
        n = len(ran)
        passed = sum(1 for a in ran if _hw_passed(a))
        lines.append(f"hardware only: {n} attempt{'s' if n != 1 else ''} ({passed} passed on hardware); the simulator did "
                     "not check these, because it lacks a ttl function they use. Each result is a hardware result alone.")
    if unrun:
        many = unrun != 1
        lines.append(f"{unrun} attempt{'s' if many else ''} used a ttl function the simulator lacks and "
                     f"{'were' if many else 'was'} not run on hardware (the probe ran with --phase sim, or stopped first), "
                     f"so no check judged {'them' if many else 'it'}.")
    return lines


def _bars_text(b: dict) -> str:
    lo, hi = b["slope"]
    return (f"pcc >= {b['pcc']}, rel_l2 <= {b['rel_l2']}, slope in [{lo}, {hi}], "
            f"max_abs_err <= {b['max_abs_atol']} + {b['max_abs_rtol']} * max|expected|")


def _temperature_of(a: dict) -> float:
    """The attempt's sampling temperature; a record from before sampling was recorded was greedy (0)."""
    sp = a.get("sampling") if isinstance(a.get("sampling"), dict) else {}
    t = sp.get("temperature")
    return float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else 0.0


def kernel_groups(attempts: list[dict], ops: list[str], seed_base: int | None = None) -> list[dict]:
    """One group per op and temperature, in the order they appear: the attempt numbers, the group's sampling
    ({"temperature", "top_p", "seed_base", "seeds"}: the seed each attempt sent, null at temperature 0), and how many
    distinct kernels the attempts wrote (by provenance.kernel_sha256). Attempts with no kernel copied count apart."""
    groups: dict[tuple, dict] = {}
    for op in ops:
        for a in sorted((a for a in attempts if a["op"] == op), key=lambda a: a["attempt"]):
            key = (op, _temperature_of(a))
            sp = a.get("sampling") if isinstance(a.get("sampling"), dict) else {}
            g = groups.setdefault(key, {"op": op, "temperature": key[1], "attempts": [], "shas": [],
                                        "sampling": {"temperature": key[1], "top_p": sp.get("top_p"),
                                                     "seed_base": seed_base, "seeds": []}})
            g["attempts"].append(a["attempt"])
            g["sampling"]["seeds"].append(sp.get("seed"))
            g["shas"].append((a.get("provenance") or {}).get("kernel_sha256"))
    out = []
    for g in groups.values():
        shas = [x for x in g.pop("shas") if x]
        out.append({**g, "with_kernel": len(shas), "distinct_kernels": len(set(shas)),
                    "no_kernel": len(g["attempts"]) - len(shas)})
    return out


def _distinct_lines(g: dict) -> list[str]:
    """The distinct-kernel line of a group, and a warning when every attempt that wrote a kernel wrote the same one."""
    line = f"distinct kernels: {g['distinct_kernels']} of {g['with_kernel']} attempts"
    if g["no_kernel"]:
        line += f"; {g['no_kernel']} wrote no kernel"
    lines = [line]
    if g["with_kernel"] > 1 and g["distinct_kernels"] == 1:
        none = f" ({g['no_kernel']} wrote none)" if g["no_kernel"] else ""
        if g["temperature"] == 0:
            lines.append(f"All {g['with_kernel']} attempts that wrote a kernel produced the same kernel{none}: "
                         "temperature 0 asks the server for greedy decoding, so attempts at one prompt often give one "
                         "sample. Raise --temperature to explore.")
        else:
            lines.append(f"All {g['with_kernel']} attempts that wrote a kernel produced the same kernel at temperature "
                         f"{g['temperature']:g}{none}: the temperature may be too low for this prompt, or the server "
                         "may be ignoring it.")
    return lines


def _speed_lines(attempts: list[dict]) -> list[str]:
    """The coder's decode speed before each attempt, from the records; nothing when no attempt was measured."""
    rows = [(a["op"], a["attempt"], a["coder_speed"]) for a in sorted(attempts, key=lambda a: (a["op"], a["attempt"]))
            if isinstance(a.get("coder_speed"), dict)]
    if not rows:
        return []
    lines = ["## Coder decode speed", "",
             "Coder decode speed, measured just before each attempt (against the best seen for this server):", ""]
    for op, n, sp in rows:
        text = (f"{sp['tok_s']:.1f} tok/s ({sp.get('state')})" if sp.get("tok_s") is not None
                else f"not measured ({sp.get('message')})")
        lines.append(f"- {op} attempt {n}: {text}")
    if any(sp.get("state") in ("busy", "suspect", "slow") for _, _, sp in rows):
        lines += ["", "For some attempts the coder was busy with another client or below its best speed. Their turns, "
                      "wall times and request timeouts are not comparable with attempts that ran at full speed."]
    return lines + [""]


def _skills_lines(header: dict) -> list[str]:
    """The report's section on the skill sources: each one's state against upstream with both shas, and a line when a
    stale source was allowed."""
    sk = header.get("skills")
    if not isinstance(sk, dict) or not isinstance(sk.get("freshness"), list):
        return []
    lines = ["## Skill sources", ""]
    if sk.get("stale_allowed"):
        lines += [f"Stale skills allowed (--allow-stale-skills): {', '.join(str(x) for x in sk.get('stale') or [])}. "
                  "The coder was given sources that are not the latest upstream has.", ""]
    for f in sk["freshness"]:
        if isinstance(f, dict):
            lines.append(f"- {f.get('source')} ({f.get('path')}): {f.get('state')}; local {f.get('local')}, "
                         f"upstream {f.get('remote')}. {f.get('detail')}")
            for w in f.get("warnings") or []:
                lines.append(f"  - warning: {w}")
    return lines + [""]


def _reference_lines(header: dict) -> list[str]:
    """The report's list of the files the coder was given, each with its source and sha256."""
    refs = header.get("references")
    if not isinstance(refs, list) or not refs:
        return []
    lines = ["## Reference files", "",
             f"Each attempt's reference/ directory held these files (tt-lang skill directory: "
             f"{header.get('tt_lang_skill_dir')}):", ""]
    for r in refs:
        if isinstance(r, dict):
            lines.append(f"- {r.get('name')}: from the {r.get('source')}, {r.get('path')}, sha256 {r.get('sha256')}")
    lib = header.get("library")
    if isinstance(lib, dict):
        lines += ["", f"Worked-kernel library mode: {lib.get('mode')}."]
        if lib.get("note"):
            lines.append(str(lib["note"]))
    return lines + [""]


def write_report(probe_dir, attempts: list[dict], header: dict) -> dict:
    """Write kernel-probe-report.json (header, attempts, attempts to first pass per op, and per op and temperature the
    distinct-kernel counts) and kernel-probe-report.md."""
    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    ops = list(header.get("ops") or [])
    for a in attempts:
        if a["op"] not in ops:
            ops.append(a["op"])
    by_op = {op: [a for a in attempts if a["op"] == op] for op in ops}
    first = {op: {"sim": _first_pass(rows, lambda a: a.get("passed_sim") is True), "hw": _first_pass(rows, _hw_passed)}
             for op, rows in by_op.items()}
    groups = kernel_groups(attempts, ops, (header.get("sampling") or {}).get("seed_base")
                           if isinstance(header.get("sampling"), dict) else None)
    data = {"header": header, "attempts": attempts, "first_pass": first, "groups": groups}
    _write_json(probe_dir / "kernel-probe-report.json", data)

    t = header.get("tt_lang") or {}
    lines = ["# Kernel probe report", "",
             f"Probe id: {header.get('probe_id')}. Coder model: {header.get('coder_model')}.", "",
             "Temperature 0 asks the server for greedy decoding. Attempts then often repeat each other (the report "
             "counts distinct kernels) but are not guaranteed to be identical. Attempts at temperature above 0 with "
             "different seeds sample independently, if the server honours the temperature. The report gives counts and no rates. The inputs are synthetic: seeded random values plus a heavy-tailed set with a few scaled "
             "outliers. A kernel that passes them can still fail on real activations.", "",
             "The simulator promotes bf16 to f32 (tt-lang-sim's default) and the references compute in f32, so on a "
             "simple op a correct kernel shows a max_abs_err of 0.0 and a PCC of 1.0. Rounding on hardware is "
             "coarser (bf16) than the simulator's f32.", "",
             *(["`add` is a calibration op: its working kernel is upstream's eltwise_add.py, the reference example "
                "the agent reads. A pass of `add` shows the loop works; it is not evidence that the coder can write a "
                "kernel.", ""] if "add" in ops else []),
             *([header["budget_note"], ""] if header.get("budget_note") else []),
             *([_sampling_text(header["sampling"]), ""] if isinstance(header.get("sampling"), dict) else []),
             *([f"{header['ignored_records']} unrelated or unreadable records ignored under {VERDICT_DIR}.", ""]
               if header.get("ignored_records") else []),
             *_speed_lines(attempts),
             "## tt-lang", "",
             f"- simulator version: {t.get('sim_version')}",
             f"- source SHA: {t.get('source_sha')}",
             f"- diagnostics patch applied: {'yes' if t.get('diagnostics_patched') else 'no'} "
             f"(a local patch of the tt-lang-sim wheel; patch sha256 {t.get('patch_sha256')})", "",
             *_skills_lines(header),
             *_reference_lines(header),
             "## Bars (provisional: none has been measured)", ""]
    for op, b in (header.get("bars") or {}).items():
        lines.append(f"- {op}, simulator: {_bars_text(b['sim'])}")
        lines.append(f"- {op}, hardware: {_bars_text(b['hw'])}")
    lines += ["", "## Hardware", ""]
    hw = header.get("hw")
    if header.get("stopped") or header.get("blocked"):
        lines += [f"The probe stopped: {header.get('stopped') or header['blocked']}", ""]
    if isinstance(hw, dict) and isinstance(hw.get("watch"), dict):
        lines += [f"smoke test: {_watch_text(hw['watch'])}", ""]
    if hw is None:
        lines.append("hw: not run")
    elif hw.get("available"):
        lines.append("hw: available (the smoke test passed)")
    elif hw.get("queued"):
        lines += [f"hw: not run. {_queued_text(hw['queued'])}. "
                  "Run the probe again when `gozer status` shows free chips. The smoke test's last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    elif hw.get("hung"):
        lines += [f"hw: blocked. The probe stopped: the smoke test hung ({hw.get('reason')}). Its last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    else:
        lines += ["hw: unavailable. The smoke test (upstream's eltwise_add.py on one chip) failed. Its last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    section = _agent_hw_lines(attempts)
    if section:
        lines += [""] + section
    for op, rows in by_op.items():
        lines += ["", f"## {op}"]
        for g in (g for g in groups if g["op"] == op):
            lines += _group_lines(g, [a for a in rows if a["attempt"] in g["attempts"]])
        f = first[op]
        lines.append("")
        lines.append(f"attempts to first pass in the simulator: {f['sim']}" if f["sim"] is not None
                     else f"none of {len(rows)} attempts passed in the simulator")
        ran = [a for a in rows if hw_started(a)]
        gap = [a for a in rows if hardware_only(a)]
        if gap:
            lines += _sim_gap_lines(gap)
        for a in sorted(rows, key=lambda a: a["attempt"]):
            h = a.get("hw")
            if isinstance(h, dict) and h.get("started") is False:
                lines.append(f"attempt {a['attempt']} was not judged on hardware: {h.get('category')}: {h.get('first_line')}")
        if ran:
            # Count only the attempts that ran on hardware. When the probe stopped early, simulator passes that never
            # reached a chip are said to be left out.
            left = [a for a in rows if a.get("passed_sim") is True and not hw_started(a)]
            count = f"{sum(1 for a in ran if _hw_passed(a))} of {len(ran)} attempts that ran on hardware passed"
            if left and (header.get("stopped") or header.get("blocked")):
                count += "; the probe stopped before the rest"
            lines.append(count)
            if f["hw"] is not None:
                lines.append(f"attempts to first pass on hardware: {f['hw']}")
            for a in sorted(ran, key=lambda a: a["attempt"]):
                if isinstance(a["hw"].get("watch"), dict):
                    lines.append(f"attempt {a['attempt']} on hardware: {_watch_text(a['hw']['watch'])}")
    (probe_dir / "kernel-probe-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


# ---- the command ------------------------------------------------------------------------------------------------

# The same codes as orchard/cli.py (a test checks they agree): 3 a crash, 4 an interrupt, 5 a hardware hang.
EXIT_OK, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED, EXIT_BLOCKED, EXIT_SLOW = 0, 2, 3, 4, 5, 6
CODER_STAGE = 2              # the probe's agent is the coder that writes stage 2's code, so it uses stage 2's tier


class _Refused(Exception):
    """The command stops before anything runs. The message says what to do."""


def default_env(args) -> ttlang_env.KEnv:
    """The tt-lang environment the flags or the ORCHARD_TTLANG_* variables name, as `tt-orchard kernel-env` sees it."""
    # Like ttlang_env.main, this reads os.environ and Path.home() directly.
    home = Path.home()
    return ttlang_env.KEnv(
        src_dir=Path(getattr(args, "src_dir", None) or os.environ.get("ORCHARD_TTLANG_SRC")
                     or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(getattr(args, "venv_dir", None) or os.environ.get("ORCHARD_TTLANG_VENV")
                      or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        environ=dict(os.environ))


def coder_endpoint(args, tier_config, probe) -> tuple[str, str]:
    """(endpoint, model) of the coder the attempts talk to.

    --endpoint and --model always win. A value that neither flag gives comes from the tier stage 2 runs on, in the tier
    config. With both flags missing, the tier's endpoint is found as the supervisor finds it (resolve_endpoint: another
    chip tier serving the same model may stand in). With no tier config, the missing flag is refused by name."""
    from orchard.stages import TierUnavailable, resolve_endpoint, tier_for
    endpoint, model = getattr(args, "endpoint", None), getattr(args, "model", None)
    if endpoint and model:
        return endpoint, model
    if tier_config is None:
        raise _Refused("no coder endpoint is known: pass --endpoint URL and --model NAME, or keep a bringup.toml "
                       "whose tier config names the coder")
    tier = tier_for(tier_config, CODER_STAGE, phase="main", escalated=False)
    if endpoint:
        return endpoint, tier_config.tiers[tier]["model"]
    if model:
        return tier_config.tiers[tier]["endpoint"], model
    try:
        used, endpoint, _note = resolve_endpoint(tier_config, tier, probe)
    except TierUnavailable:
        t = tier_config.tiers[tier]
        raise _Refused(f"tier {tier} ({t['model']} at {safe_url(t['endpoint'])}) is not serving. The probe starts no server: start the coder with `tt-orchard bringup`, or pass "
                       "--endpoint and --model for a server that runs") from None
    return endpoint, tier_config.tiers[used]["model"]


def _attempt_line(a: dict) -> str:
    """One line for an attempt that has ended: op, number, result, first line, turns, wall time."""
    result = "passed in the simulator" if a.get("passed_sim") else (a.get("category") or "failed")
    first = f" ({a['first_line']})" if a.get("first_line") and not a.get("passed_sim") else ""
    return f"{a['op']} {a['attempt']}: {result}{first}, {a.get('turns')} turns, {a.get('agent_wall_s')} s"


def _speed_line(n: int, speed: dict, allow_slow: bool) -> str:
    """One line before an attempt: what the speed check found."""
    if speed.get("state") == "error":
        return f"could not measure the coder's speed: {speed.get('message')}"
    line = f"coder decode speed before attempt {n}: {speed.get('tok_s'):.1f} tok/s ({speed.get('state')})"
    if speed.get("state") == "slow" and allow_slow:
        line += "; the coder is slow, continuing (--allow-slow-coder)"
    return line


def _hw_line(a: dict) -> str:
    h = a["hw"]
    result = "passed on hardware" if h.get("passed") else (h.get("category") or "failed")
    first = f" ({h['first_line']})" if h.get("first_line") and not h.get("passed") else ""
    return f"{a['op']} {a['attempt']}: {result}{first}"


MAX_TEMPERATURE = 2.0
MAX_TEMPERATURES = 10        # choice: a plan larger than this is more likely a typo than a measurement
MAX_TOTAL_ATTEMPTS = 200     # choice: attempts x temperatures; at about 200 s an attempt this is 11 hours
SEED_MAX = 2 ** 63 - 1       # the largest seed an OpenAI-compatible server takes (a signed 64-bit integer)


def _number(text, flag: str) -> float:
    try:
        value = float(str(text).strip())
    except ValueError:
        raise _Refused(f"{flag}: {text!r} is not a number") from None
    return value


def parse_sampling(args, attempts: int = 1) -> dict:
    """The run's sampling from --temperature, --temperatures, --top-p and --seed, checked; raises _Refused.

    {"temperatures": [...], "top_p": float or None, "seed": int or None}. The default is one group at temperature 0
    (the server is asked for greedy decoding; attempts often repeat each other but are not guaranteed to be identical).
    Every temperature must be in [0, 2] and
    listed once, at most MAX_TEMPERATURES of them, with at most MAX_TOTAL_ATTEMPTS attempts in all (`attempts` per
    temperature); top-p in (0, 1]; the seed a whole number of 0 or more whose last attempt (base + total) is at most
    2**63 - 1."""
    one, many = getattr(args, "temperature", None), getattr(args, "temperatures", None)
    if one is not None and many is not None:
        raise _Refused("give --temperature or --temperatures, not both")
    texts = [one] if one is not None else (str(many).split(",") if many is not None else ["0"])
    temperatures = []
    for text in texts:
        t = _number(text, "--temperatures" if many is not None else "--temperature")
        if not (0.0 <= t <= MAX_TEMPERATURE):                 # NaN fails this too
            raise _Refused(f"temperature {text.strip()} is not between 0 and {MAX_TEMPERATURE:g}")
        t = t + 0.0                                            # -0.0 becomes 0.0, so it prints as 0
        if t in temperatures:
            raise _Refused(f"temperature {t:g} is listed twice")
        temperatures.append(t)
    if len(temperatures) > MAX_TEMPERATURES:
        raise _Refused(f"at most {MAX_TEMPERATURES} temperatures (got {len(temperatures)})")
    total = attempts * len(temperatures)
    if total > MAX_TOTAL_ATTEMPTS:
        sum_text = f"{attempts} x {len(temperatures)} = {total}" if len(temperatures) > 1 else f"{total}"
        raise _Refused(f"at most {MAX_TOTAL_ATTEMPTS} attempts in total ({sum_text})")
    top_p = getattr(args, "top_p", None)
    if top_p is not None:
        try:
            top_p = float(str(top_p).strip())
        except ValueError:
            raise _Refused(f"--top-p: {top_p!r} is not a number") from None
        if not (0.0 < top_p <= 1.0):
            raise _Refused(f"--top-p {top_p:g} is not in (0, 1]")
    seed = getattr(args, "seed", None)
    if seed is not None:
        try:
            seed = int(str(seed).strip())
        except ValueError:
            raise _Refused(f"--seed: {seed!r} is not a whole number") from None
        if seed < 0:
            raise _Refused(f"--seed {seed} is below 0")
        if seed + total > SEED_MAX:
            raise _Refused(f"--seed {seed}: the last attempt would send {seed} + {total}, above the largest seed "
                           f"2**63 - 1 ({SEED_MAX})")
    return {"temperatures": temperatures, "top_p": top_p, "seed": seed}


def seed_base_from_probe_id(probe_id: str) -> int:
    """A seed base derived from the probe id, for a run without --seed. The report records it, so the run can be
    repeated with --seed <base> (if the server honours seeds)."""
    return int(hashlib.sha256(probe_id.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def attempt_seed(base: int, k: int, temperature: float) -> int | None:
    """The seed of the k-th attempt of the run (k counts every attempt from 1): none at temperature 0, else base + k."""
    return None if temperature == 0 else base + k


def new_probe_id() -> str:
    """A token no earlier run used: the UTC time and 8 random hex digits."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def safe_url(url) -> str:
    """`url` with any `user:password@` removed, for printing. The request itself still uses the URL as given."""
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return "<an unparseable URL>"
    if "@" not in parts.netloc:
        return str(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


LOCK_NAME = "probe.lock"


class ProbeLock:
    """The held lock: the open file (whose flock is the lock) and its path."""

    def __init__(self, fd: int, path: Path):
        self.fd, self.path = fd, path


def _holder_note(fd: int) -> str:
    """` (pid N)` from the holder's lock file, or "" when the file is empty, garbage or names no sane pid. The text is
    information for the refusal message only; the flock decides who holds the lock."""
    try:
        rec = json.loads(os.pread(fd, 4096, 0).decode("utf-8"))
        pid = rec.get("pid") if isinstance(rec, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and 0 < pid < (1 << 32):
            return f" (pid {pid})"
    except Exception:
        pass
    return ""


def acquire_lock(probe_dir: Path) -> ProbeLock:
    """Take an exclusive flock on `probe_dir/probe.lock` for this process, or raise _Refused with a plain message.

    Two probes on one directory would write the same attempt directories, and each one's sweep would stop the other's
    agent processes. The kernel releases a flock when its holder dies (SIGKILL included), so a lock left by an ended
    probe needs no staleness check: the next probe simply gets it. The file is opened with O_NOFOLLOW (a link is
    refused) and must be a regular file. After the flock, the file at the path must be the file we locked; if another
    probe removed it and made a new one in between, the lock is taken again once. The pid written into the file is
    information for a refused probe's message."""
    path = Path(probe_dir) / LOCK_NAME
    try:
        Path(probe_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _Refused(f"cannot create the output directory {probe_dir}: {exc.strerror or exc}") from None
    for _ in range(2):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise _Refused(f"{path} is a link; remove it if no probe is using {probe_dir}") from None
            raise _Refused(f"cannot open the lock {path}: {exc.strerror or exc}") from None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise _Refused(f"{path} is not a regular file; remove it if no probe is using {probe_dir}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise _Refused(f"another probe is using {probe_dir}{_holder_note(fd)}; wait for it to end or "
                                   "pass another --out") from None
                raise _Refused(f"cannot lock {path}: {exc.strerror or exc}") from None
            try:
                st = os.stat(path, follow_symlinks=False)
                same = (st.st_ino, st.st_dev) == (os.fstat(fd).st_ino, os.fstat(fd).st_dev)
            except OSError:
                same = False
            if not same:
                os.close(fd)                             # we locked a file nobody else can reach: take it again
                continue
            try:
                os.ftruncate(fd, 0)
                os.pwrite(fd, json.dumps({"pid": os.getpid()}).encode("utf-8"), 0)
            except OSError:
                pass                                     # the pid is information only; the flock is the lock
            return ProbeLock(fd, path)
        except BaseException as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            if isinstance(exc, OSError):                 # fstat, stat or another call on the lock file failed
                raise _Refused(f"cannot use the lock {path}: {exc.strerror or exc}") from None
            raise
    raise _Refused(f"another probe is using {probe_dir}: its lock file kept changing while this probe took it")


def release_lock(lock: ProbeLock) -> None:
    """Remove the lock file if it is still the file we hold, then close it (which releases the flock). Never raises.
    After a SIGKILL nothing runs here: the kernel releases the flock, and the file that stays is harmless."""
    try:
        st = os.stat(lock.path, follow_symlinks=False)
        if (st.st_ino, st.st_dev) == (os.fstat(lock.fd).st_ino, os.fstat(lock.fd).st_dev):
            os.unlink(lock.path)
    except OSError:
        pass
    try:
        os.close(lock.fd)
    except OSError:
        pass


def _raise_interrupt(signum, frame):
    """SIGTERM handler while the probe runs: stop the way Ctrl-C does, so the report is still written."""
    raise KeyboardInterrupt


def _crash(where: str, exc: BaseException) -> str:
    return f"{where} crashed: {type(exc).__name__}: {str(exc)[:200]}"


def main(args, *, env=None, out=None, http=post_json, probe=probe_model, run_watched_fn=run_watched,
         tier_config=None, speed_check=None) -> int:
    """`tt-orchard kernel-probe --op NAME [--attempts N] [--phase sim|hw] [--out DIR] [--endpoint URL] [--model NAME]
    [--src-dir D] [--venv-dir D] [--shared-python P]`.

    Order: the op is checked; the simulator environment must pass `ttlang_env.ensure(check_only=True)`; the coder's
    reference files must all be readable (ttlang_env.read_references); the tt-lang checkout and the tt-lang skill must
    be the latest upstream has (ttlang_env.skills_status; --allow-stale-skills runs anyway and the report records
    it); the coder's endpoint and model are resolved and must answer `probe`. Each refusal returns 2 before any attempt directory
    exists. Then N attempts run one after another, each printing one line as it ends. With --phase hw, run_hardware
    runs one smoke test and then each attempt that passed the simulator, each on one leased chip. The report is
    written from the records this run's attempts returned. `tier_config` is the TierConfig the coder comes from when a
    flag is missing (the CLI loads it from bringup.toml).

    The output directory is locked (an flock on `probe.lock`) from the first write to the end. A second probe on the
    same directory is refused with exit 2. The kernel releases the lock when this process dies, however it dies; the
    file may remain after a SIGKILL and is harmless. A clean exit removes it.

    A run that stops early still writes its report from what finished, with "The probe stopped: <reason>" in it:
    a hung hardware run (ProbeBlocked) exits 5, an exception in an attempt or in the hardware phase exits 3, and Ctrl-C
    or SIGTERM exits 4. No hardware runs after a crash or an interrupt. If the report itself cannot be written, the
    records are printed to stderr as JSON and the exit is 3. URLs are printed without any user:password part.

    `speed_check(endpoint, model)` (speedwatch.probe_check; the command line passes it) measures the coder's decode speed
    before each attempt and returns a dict with a `state`. The dict goes into the attempt's record. A `slow` coder stops
    the probe before the attempt (exit 6, the report written) unless --allow-slow-coder; `error` and `suspect` never stop
    it. With no `speed_check` nothing is measured."""
    say = (lambda line: print(line, file=out)) if out is not None else print

    def refuse(message: str) -> int:
        print(f"refused: {message}", file=sys.stderr)
        return EXIT_REFUSED

    if getattr(args, "ladder", False):
        # The capability ladder (issue 175): one ordinary run of this function per rung, each in its own directory.
        from orchard import kernelladder
        return kernelladder.run(
            args, lambda sub: main(sub, env=env, out=out, http=http, probe=probe, run_watched_fn=run_watched_fn,
                                   tier_config=tier_config, speed_check=speed_check),
            default_wall_s=BUDGET["wall_s"], say=say)
    if getattr(args, "op", None) not in OPS:
        return refuse(f"unknown op {getattr(args, 'op', None)!r}; the ops are {', '.join(OPS)} "
                      "(or give --ladder to run the capability ladder)")
    attempts = int(getattr(args, "attempts", 5) or 0)
    if attempts < 1:
        return refuse("--attempts must be at least 1")
    wall_s = getattr(args, "wall_s", None)
    if wall_s is None:
        wall_s = BUDGET["wall_s"]
    else:
        try:
            wall_s = float(wall_s)
        except (TypeError, ValueError):
            return refuse(f"--wall-s: {wall_s!r} is not a number")
        if not (0 < wall_s <= MAX_WALL_S):                          # NaN fails this too
            return refuse(f"--wall-s {wall_s:g} is not above 0 and at most {MAX_WALL_S:g} seconds")
    try:
        plan = parse_sampling(args, attempts)
        hw_policy = hw_policy_from_args(args)
    except _Refused as exc:
        return refuse(str(exc))
    shared = getattr(args, "shared_python", None) or default_shared_python()
    env = env or default_env(args)
    results = ttlang_env.ensure(env, check_only=True, say=say)
    if not results or any(r.status != ttlang_env.OK for r in results):
        return refuse("the tt-lang simulator environment is not ready: run `tt-orchard kernel-env`")
    skill_dir = ttlang_env.skill_dir_from(args)
    references, problems = ttlang_env.read_references(env, skill_dir)
    library_mode = getattr(args, "library", None) or "all"
    if library_mode not in LIBRARY_MODES:
        return refuse(f"unknown --library {library_mode!r}; the modes are {', '.join(LIBRARY_MODES)}")
    if not problems:
        references = references + library_references(args.op, library_mode)
    if problems:
        return refuse("the coder's reference files are not ready: " + "; ".join(problems) + ". The tt-lang skill is "
                      "installed by `tt-warp fastmodels sync`; name another directory with --tt-lang-skill-dir or "
                      f"{ttlang_env.TTLANG_SKILL_ENV}")
    reference_record = [{"name": f"reference/{r['name']}", "source": r["source"], "path": r["path"],
                         "sha256": r["sha256"]} for r in references]
    # The operator's rule: the probe uses the latest tt-lang checkout and tt-lang skill upstream has. A stale source
    # (or one that could not be compared and has no recent current check) refuses the probe unless overridden.
    fresh = ttlang_env.skills_status(env, skill_dir, preread=ttlang_env.skill_bytes(references))
    for line in ttlang_env.skills_lines(fresh):
        say(line)
    stale = [f for f in fresh if f.stale]
    allow_stale = bool(getattr(args, "allow_stale_skills", False))
    if stale and not allow_stale:
        return refuse("these sources are not the latest upstream has: " + " ".join(
            f"{f.source} ({f.path}): {f.detail} {f.update_text()}"
            for f in stale) + " Pass --allow-stale-skills to run anyway; the report then records it")
    if stale:
        say("stale skills allowed (--allow-stale-skills): " + ", ".join(f.source for f in stale))
    skills_record = {"freshness": [f.to_dict() for f in fresh], "stale": [f.source for f in stale],
                     "stale_allowed": bool(stale)}
    try:
        endpoint, model = coder_endpoint(args, tier_config, probe)
    except _Refused as exc:
        return refuse(str(exc))
    if not probe(endpoint, model):
        return refuse(f"the coder at {safe_url(endpoint)} does not answer with the model {model!r}. The probe starts "
                      "no server: start the coder with `tt-orchard bringup`, or pass --endpoint and --model for a "
                      "server that runs")

    probe_dir = Path(getattr(args, "out", None) or "kernel-probe")
    try:
        lock = acquire_lock(probe_dir)
    except _Refused as exc:
        return refuse(str(exc))
    try:
        probe_id = new_probe_id()
        base = plan["seed"] if plan["seed"] is not None else seed_base_from_probe_id(probe_id)
        sampling = {"temperatures": plan["temperatures"], "top_p": plan["top_p"], "seed_base": base,
                    "seed_base_from": "--seed" if plan["seed"] is not None else "probe id",
                    "attempts_per_temperature": attempts}
        facts = tt_lang_facts(env)
        temps = ", ".join(f"{t:g}" for t in plan["temperatures"])
        n_temps = len(plan["temperatures"])
        how_many = (f"{attempts} attempts at temperature {temps}" if n_temps == 1 else
                    f"{attempts} attempts per temperature ({temps}): {attempts * n_temps} in total")
        say(f"kernel probe {probe_id}: op {args.op}, {how_many}, coder {model} at {safe_url(endpoint)}")
        records: list[dict] = []
        hw, stopped, blocked, code = None, None, None, EXIT_OK
        state: dict = {}
        hw_started = False
        try:
            previous = signal.signal(signal.SIGTERM, _raise_interrupt)
        except ValueError:                                  # not the main thread: no handler can be installed
            previous = None
        try:
            where = "attempt 1"
            try:
                # n counts every attempt of the run from 1, across the temperature groups.
                jobs = [t for t in plan["temperatures"] for _ in range(attempts)]
                allow_slow = bool(getattr(args, "allow_slow_coder", False))
                for n, temperature in enumerate(jobs, start=1):
                    where = f"attempt {n}"
                    speed = speed_check(endpoint, model) if speed_check is not None else None
                    if speed is not None:
                        say(_speed_line(n, speed, allow_slow))
                        if speed.get("state") == "slow" and not allow_slow:
                            stopped, code = f"coder slow before attempt {n}: {speed.get('message')}", EXIT_SLOW
                            break
                    record = run_attempt(op=args.op, n=n, probe_dir=probe_dir, env=env, endpoint=endpoint,
                                         model=model, coder_model=model, probe_id=probe_id, http=http,
                                         run_watched=run_watched_fn, tt_lang=facts, temperature=temperature,
                                         top_p=plan["top_p"], seed=attempt_seed(base, n, temperature), wall_s=wall_s,
                                         references=references, skills=skills_record, coder_speed=speed,
                                         hw_policy=hw_policy, shared_python=shared)
                    records.append(record)
                    say(_attempt_line(record))
                    # The tool stops itself when the check after a call finds a chip still held (or cannot tell): end the run,
                    # so no later attempt and no hardware phase starts beside it. Nothing is repaired here.
                    held = (record.get("agent_hw") or {}).get("stopped")
                    if held:
                        stopped, code = f"an agent's hardware call left a chip held: {held}", EXIT_BLOCKED
                        break
                if getattr(args, "phase", "sim") == "hw" and not stopped:
                    where, hw_started = "the hardware phase", True
                    try:
                        hw = run_hardware(records, probe_dir=probe_dir, env=env, shared_python=shared,
                                          run_watched=run_watched_fn, state=state)
                    except ProbeBlocked as exc:
                        blocked = stopped = str(exc)
                        code = EXIT_BLOCKED
                        # A hung smoke test carries its own hardware block. A kernel hang came after a passing
                        # smoke test.
                        hw = exc.hw if exc.hw is not None else state.get("smoke")
            except KeyboardInterrupt as exc:
                stopped, code = f"interrupted during {where}", EXIT_ABORTED
                if isinstance(getattr(exc, "partial_attempt", None), dict):
                    records.append(exc.partial_attempt)
            except Exception as exc:
                stopped, code = _crash(where, exc), EXIT_ERROR
                if isinstance(getattr(exc, "partial_attempt", None), dict):
                    records.append(exc.partial_attempt)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        if hw_started and hw is None:
            hw = state.get("smoke")                         # a crash or an interrupt after the smoke test ran
        try:
            for a in records:
                if isinstance(a.get("hw"), dict):
                    say(_hw_line(a))
            if hw and hw.get("queued"):
                say(f"hardware busy: {_queued_text(hw['queued'])}")
            elif hw and not hw.get("available") and not hw.get("hung"):
                say("hardware unavailable: the smoke test failed (the report has its last lines)")
            header = report_header(env=env, coder_model=model, probe_id=probe_id, ops=[args.op], hw=hw, tt_lang=facts,
                                   ignored_records=0, wall_s=wall_s)
            if stopped:
                header["stopped"] = stopped
            if blocked:
                header["blocked"] = blocked
            header["sampling"] = sampling
            header["tt_lang_skill_dir"] = str(skill_dir)
            header["references"] = reference_record
            header["library"] = {"mode": library_mode, "note": library_note(args.op, library_mode)}
            header["skills"] = skills_record
            header["agent_hw_policy"] = hw_policy.as_dict() | {"enabled": hw_policy.enabled}
            write_report(probe_dir, records, header)
            say(f"report: {probe_dir / 'kernel-probe-report.md'}")
            if stopped:
                say(f"stopped: {stopped}" + (" (pass --allow-slow-coder to run anyway)" if code == EXIT_SLOW else ""))
        except Exception as exc:                            # never let the last step lose the records
            try:
                print(f"the report could not be written ({type(exc).__name__}: {exc}); the attempt records follow",
                      file=sys.stderr)
                print(json.dumps({"stopped": stopped, "attempts": records}, default=str), file=sys.stderr)
            except Exception:
                pass
            return EXIT_ERROR
        return code
    finally:
        release_lock(lock)
