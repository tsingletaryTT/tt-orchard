# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Agent steps: one fresh model context per stage phase, and the tool-call loop that serves it.

This module owns three things.

1. The agent's environment (`agent_env`). It starts from an allow-list, so GitHub and Hugging
   Face tokens, SSH agent sockets and every other variable the supervisor happens to have are
   absent. HOME points to an empty directory inside the run, so `gh`, git's HTTPS helpers and
   `huggingface_hub` do not look in the operator's home. OpenSSH ignores HOME and reads ~/.ssh
   from the passwd entry, so GIT_SSH_COMMAND gives git's ssh no config and no key; plain ssh is
   refused by the runner. Any process can still read a credential file by absolute path. The
   supervisor's preflight refuses to start while known ones exist, unless the operator accepts
   that. Extra variables the operator passes are refused when the name looks like a credential.
2. The tools (`Tools`): `shell`, which runs every command through orchard/runner.py's checks with
   the run directory as its working directory, and `write_file`, which writes only inside the
   stage directory.
3. The loop (`AgentStep`): it sends the conversation to a tier's OpenAI-compatible endpoint
   (stdlib urllib, non-streaming), runs each tool call, records each new file under the stage's
   `evidence/` directory in the ledger with its sha256, and feeds the watchdog an Event for every
   response, tool call and tool result (a shell call's event carries its command's shape, which
   the watchdog's TurnRepeat uses to match near-duplicate commands). A nudge the watchdog's ladder sends is added to the next
   request as a user message. Transport failures are retried once, through the watchdog's
   RetryGuard (spec section 3).

   A reply with finish_reason "length" and no tool calls ran into max_tokens, usually because a
   reasoning model spent the whole budget thinking. The loop does not take it as a final answer.
   It leaves the empty reply out of the conversation, writes a `notice` ledger entry, adds a user
   message that asks for a short think and a tool call, and goes on; the attempt uses up a turn.
   A reply with no tool calls and no text (empty or whitespace only, whatever its finish_reason)
   is handled the same way, with its own `notice` ("empty reply") and its own user message. A
   reasoning model can return one after it finishes thinking: the live Qwen3.8 run did, at turn
   49 of stage 1. Cut-off and empty replies share one count. A second reply in a row of either
   kind ends the step with status "error", and the detail names the kind of each. A reply with
   finish_reason "length" that holds tool calls runs as usual, and a missing finish_reason is
   treated like "stop". A reply with text and no tool calls is the final answer. Every record in
   the step's log carries the reply's finish_reason.

   The step keeps its conversation in `messages`. `continue_with` adds one user message and runs
   the loop again with its own turn cap and its own log file. The supervisor calls it with the
   exit gate's reasons when a step ended "done" and the gate failed, or as a wrap-up when a step
   used up its turns with evidence on disk and its deliverable not written. A step gets at most
   one of the two.

   The environment also sets TT_VISIBLE_DEVICES and TT_METAL_VISIBLE_DEVICES to NO_CHIP, a
   device mask that matches no chip. This is a request to the runtime. Nothing in orchard stops a
   process from opening a device: the same user can open /dev/tenstorrent/* directly, and code
   can clear the variables. Checked on hardware 2026-10-02: a device open with this mask fails (UMD raises RuntimeError: BDF pattern 0000:ff:00.0 did not match any devices).

What it does not do: streaming; a proxy in front of agents the supervisor did not launch (the
watchdog only reads their transcripts); read-only mounts. The same user account runs the agent,
so a file outside the run directory that holds a token can still be read by absolute path, and a
shell redirect can still write outside the run directory. The real limits there are a separate
user account or a mount namespace, which plan 4 does not set up.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from orchard import runaway, safeio
from orchard.canary import CanaryError, post_json
from orchard.port_guard import PORT_GUARD_DEPTH, PORT_GUARD_ENTRIES, PORT_GUARD_FILES, PortGuard  # noqa: F401  (names stay importable from here)
from orchard.defaults import (AGENT_MAX_TOKENS, AGENT_MAX_TURNS, AGENT_REQUEST_TIMEOUT_S,
                              AGENT_THINKING, TOOL_OUTPUT_CHARS, TOOL_TIMEOUT_S)
from orchard.runner import Denied, check_string
from orchard.stages import evidence_record
from orchard.watchdog import Event, RetryGuard, command_shape

# ---- environment --------------------------------------------------------------------------------

TELEGRAPH_DIR_VAR = "ORCHARD_TELEGRAPH_DIR"     # the ledger directory agent shells use (orchard/telegraph_cmd.py)
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TERM")
# The device mask for agent shells: a PCI address that matches no chip on this box, set in both
# TT_VISIBLE_DEVICES and TT_METAL_VISIBLE_DEVICES. Checked on hardware 2026-10-02: a device open
# with this mask fails with a RuntimeError (the BDF matched no device). Code that clears the
# variables still gets every chip. The supervisor's hardware test replaces it with the leased chips.
NO_CHIP = "0000:ff:00.0"
GIT_SSH_COMMAND = "ssh -F /dev/null -o IdentitiesOnly=yes -o IdentityFile=/dev/null -o BatchMode=yes"
DEVICE_VARS = ("TT_VISIBLE_DEVICES", "TT_METAL_VISIBLE_DEVICES")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|_KEY$|^KEY$|COOKIE", re.IGNORECASE)


def agent_env(run_dir, *, extra: dict | None = None, source=None) -> dict:
    """The whole environment of an agent's shell. Nothing is inherited outside ENV_ALLOW."""
    source = os.environ if source is None else source
    env = {k: source[k] for k in ENV_ALLOW if k in source}
    # The telegraph ledger directory belongs to the operator and is shared by every shell: HOME below is inside the run directory, so a shell that
    # resolved `~` would open a second ledger with its own caps (issue 133). The supervisor's value is passed on when it is an
    # absolute path; `tt-orchard telegraph` refuses to guess a directory in an agent shell that has none.
    tele = source.get(TELEGRAPH_DIR_VAR)
    if isinstance(tele, str) and tele.startswith("/") and "\0" not in tele:
        env[TELEGRAPH_DIR_VAR] = tele
    home = Path(run_dir).resolve() / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "GH_CONFIG_DIR": str(home / ".config" / "gh"),
        "HF_TOKEN_PATH": str(home / "no-hf-token"),      # huggingface_hub reads its token here
        "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        # OpenSSH reads ~/.ssh from the passwd entry and ignores HOME, so git over ssh would find
        # the operator's keys. This gives git's ssh no config file and no key.
        "GIT_SSH_COMMAND": GIT_SSH_COMMAND,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "ORCHARD_RUN_DIR": str(Path(run_dir).resolve()),
    })
    env.update({k: NO_CHIP for k in DEVICE_VARS})
    for k, v in (extra or {}).items():
        if k in DEVICE_VARS:
            raise ValueError(f"{k} is the agent shells' device mask; the operator cannot set it")
        if SECRET_NAME.search(k):
            raise ValueError(f"{k} looks like a credential; agent shells never get one")
        env[k] = str(v)
    return env


# ---- running a checked command ------------------------------------------------------------------

SPAWN_TAG_VAR = "ORCHARD_SPAWN_TAG"       # in every spawned command's environment: a value of its own, so a sweep finds what the command started


def spawn_checked(command: str, run_dir, env: dict, timeout: float, stdout) -> tuple[int | None, bool]:
    """Run one shell string after the runner's checks. Returns (exit code or None, timed out).

    Denied propagates: a refused string never starts. The command runs in its own session, so a
    timeout kills everything it started. So does anything that interrupts the wait (the
    supervisor's SIGINT or SIGTERM handler, or an error): a terminal's Ctrl-C does not reach a
    process in another session, so without this the command would keep running, possibly with a
    device open, after the supervisor had released its lease.

    A helper that the command put in a session of its own (`start_new_session=True`, a double fork
    with `setsid`) is not in the command's group, so the group kill misses it (issue 390). Every
    command carries ORCHARD_SPAWN_TAG with a value of its own, and after a kill the processes that
    carry it are stopped and named in `stdout`. A process that replaced its environment is not found.
    """
    check_string(command, run_dir)
    tag = str(Path(run_dir).resolve() / f".spawn-{uuid.uuid4().hex}")       # a value to match, never created as a path
    proc = subprocess.Popen(["bash", "-c", command], cwd=run_dir, env={**env, SPAWN_TAG_VAR: tag}, stdin=subprocess.DEVNULL,
                            stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        late = _cleanup(proc, tag, stdout)
        if late is not None:
            raise late                       # a stop signal arrived during the cleanup; nothing else is on its way, so it goes on now
        return None, True
    except BaseException:
        _cleanup(proc, tag, stdout)          # the interrupt already on its way is the one that propagates, not one that arrives meanwhile
        raise


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)      # the supervisor's handlers for these raise a BaseException
CLEANUP_TRIES = 3                                   # a step that an interrupt cut short is run again, this many times in all


class _HeldStopSignals:
    """While held, SIGINT and SIGTERM are recorded and not acted on. The handlers are swapped, not the signal mask: a mask covers one thread, and
    in a process with other threads the signal can reach any of them while Python still runs the handler in the main thread. A handler can be
    swapped in the main thread only, so in another thread nothing is held and the cleanup relies on its retries.

    Both transitions can be cut short by a stop signal, because until a signal's handler is swapped the original one still runs and raises. So
    the constructor never raises: each original handler is read before anything is swapped, an interrupt that arrives during the swapping is
    kept in `exc` and the swapping goes on (a second try for each signal, CLEANUP_TRIES in all), and the cleanup then runs. `release` puts every
    original handler back, repeating the whole restoration when an interrupt cuts it short (putting a handler back is idempotent), and returns
    the first exception that came up (the one kept at the start, one raised while restoring, or one raised by the handler of a signal that
    arrived while held, which is run as the signal would have been); None when nothing did."""

    def __init__(self):
        self.old: dict = {}
        self.arrived: list = []
        self.exc: BaseException | None = None
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in STOP_SIGNALS:
            for _ in range(CLEANUP_TRIES):
                try:
                    if signum not in self.old:
                        self.old[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._hold)
                    break
                except (ValueError, OSError):
                    self.old.pop(signum, None)
                    break
                except BaseException as exc:          # a stop signal arrived before this one was held: the cleanup still runs
                    self.exc = self.exc or exc

    def _hold(self, signum, frame) -> None:
        self.arrived.append((signum, frame))

    def release(self) -> BaseException | None:
        first = self.exc
        for _ in range(CLEANUP_TRIES):
            try:
                for signum, handler in self.old.items():
                    try:
                        signal.signal(signum, handler)
                    except (ValueError, OSError, TypeError):
                        pass
                break
            except BaseException as exc:              # cut short by a signal: every handler is put back again from the start
                first = first or exc
        for signum, frame in self.arrived:
            handler = self.old.get(signum)
            try:
                if callable(handler):
                    handler(signum, frame)
                elif handler == signal.SIG_DFL:
                    os.kill(os.getpid(), signum)          # the default action of the signal, after the cleanup
            except BaseException as exc:
                return first or exc
        return first


def _cleanup(proc, tag: str, stdout) -> BaseException | None:
    """Kill the command's group, then sweep the helpers that carry `tag`, so that no interrupt skips either step.

    The stop signals are held back while this runs (_HeldStopSignals): one that arrives meanwhile is acted on when the cleanup is done. A step
    that an exception still stops (a thread that is not the main one) is run again, up to CLEANUP_TRIES times. Returns the first exception that
    came up during the cleanup, or None; the caller decides whether it replaces anything."""
    held = _HeldStopSignals()                # never raises: an interrupt during its setup is kept, and release() returns it after the cleanup
    late = None
    try:
        for step in (lambda: _kill_session(proc), lambda: _sweep_helpers(tag, stdout)):
            for _ in range(CLEANUP_TRIES):
                try:
                    step()
                    break
                except BaseException as exc:
                    late = late or exc
    finally:
        replayed = held.release()
        late = late or replayed
    return late


def _kill_session(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


_SURVIVORS: contextvars.ContextVar = contextvars.ContextVar("orchard_spawn_survivors", default=None)


class Survivors(list):
    """What collect_survivors yields: the records (a list of dicts), and in `problems` what stopped a sweep from being complete (a process list
    that could not be read, a scan that never settled, an error). An empty list with no problems is a sweep that found nothing; an empty list
    with problems is a sweep nobody can vouch for, and a caller that guards hardware has to treat the two differently."""

    def __init__(self):
        super().__init__()
        self.problems: list[str] = []


@contextlib.contextmanager
def collect_survivors():
    """Within this block, every spawn_checked call in this context adds what its sweep stopped to the list this yields: one record per process
    ({"pid", "cmdline", "stopped", "boot_id", "start"}), with the identity it had before it was signalled. The sweep removes a process before
    any later scan can see it, so the structured audit paths (the supervisor's ledger, a kernel-probe attempt's `swept` list) take the records
    from here. `stopped` False means the process is still running. The records are added when the sweep ends, also when an interrupt cuts the
    call short, so a caller that reads them in a `finally` sees them on every way out. `problems` lists what made a sweep incomplete."""
    records = Survivors()
    token = _SURVIVORS.set(records)
    try:
        yield records
    finally:
        _SURVIVORS.reset(token)


def _sweep_helpers(tag: str, stdout) -> None:
    """Stop the processes that carry `tag` (SIGTERM, a short grace, then SIGKILL) and say which in `stdout`. This runs on the way out of a kill,
    so it does not raise an Exception: an error here must not replace the result or the interrupt that is already on its way. A collector
    (collect_survivors) also gets the records, with the identity each process had before it was signalled, and the reasons the sweep could not be
    completed (an error, a process list that could not be read, a scan that never settled), so the caller can tell a failed sweep from an empty one.
    The records reach the collector and `stdout` when an interrupt cuts the sweep short as well; the interrupt then goes on."""
    sink = _SURVIVORS.get()
    records: list = []
    status: dict = {}
    try:
        try:
            from orchard import kernelprobe              # imported here: kernelprobe imports this module
            result = kernelprobe.sweep_strays(tag, var=SPAWN_TAG_VAR, identity=sink is not None, into=records, status=status)
            if isinstance(result, list) and result is not records:        # a sweep that returns its records and does not fill `into`
                records.extend(r for r in result if r not in records)
        finally:
            _publish_sweep(sink, records, status, stdout)
    except Exception as exc:
        if sink is not None:
            sink.problems.append(f"the sweep of the command's helpers failed ({type(exc).__name__})")


def _publish_sweep(sink, records: list, status: dict, stdout) -> None:
    """Hand a sweep's records and problems to the collector and name the helpers in `stdout`. Never raises an Exception."""
    try:
        if sink is not None:
            sink.extend(r for r in records if isinstance(r, dict))
            sink.problems.extend(str(p) for p in (status.get("problems") or []))
        if records:
            lines = "".join(f"orchard: a helper outlived the command and was stopped: pid {r['pid']} {r['cmdline']}"
                            f"{'' if r['stopped'] else ' (still running)'}\n" for r in records)
            try:
                stdout.write(lines.encode("utf-8") if "b" in getattr(stdout, "mode", "b") else lines)
                stdout.flush()
            except (OSError, ValueError, TypeError):
                pass
    except Exception:
        pass


READ_FILE_MAX_BYTES = 16 * 1024 * 1024    # choice: read_file loads a file whole to page through it


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters cut ...]\n{text[-half:]}"


TRUNCATION_NUDGE = ("Your last reply was cut off before you ran any command. "
                    "Think briefly, then call a tool.")
EMPTY_NUDGE = ("Your last reply was empty: it had no text and no command. Say what you will do next, "
               "then call a tool, or state that the stage is finished and name the output files you "
               "wrote.")
# The nudge after a capped reply that repeated itself (orchard/runaway.py): asking it to think briefly again invites the same loop.
LOOP_NUDGE = ("Your last reply repeated itself until it was cut off. Do not analyse or explain further. "
              "Run one command or write one file now, and keep the reply short.")
RESTART_NOTE = ("The conversation was restarted because earlier replies repeated themselves until they were cut off. Everything you "
                "did is on disk; the files are listed below. Read only what you need, do one thing at a time, and keep every reply short.")
# What a restart carries over (review of PR 462, round 3): the supervisor's own instructions to this step (a gate's feedback, a wrap-up
# request, a watchdog nudge) are not part of the loop that caused the restart, and dropping them would send the step back to a task it
# had already been told to change. At most this many are carried, each cut to this many characters; the oldest nudges go first.
CARRIED_INSTRUCTIONS = 4
CARRIED_CHARS = 8_000
# The fields of a request that the transcript records, so a reader of the log can tell which sampling a turn used.
LOGGED_SAMPLING = ("temperature", "max_tokens", "top_p", "seed", "frequency_penalty", "repetition_penalty")
# The files a stage directory holds that the shell guard looks at: only the port, and no more than this many files and entries
# (orchard/port_guard.py holds the guard and the bounds; the names stay importable from here).
# How many replies in a row may run no command (cut off or empty) before the step ends as "error".
BAD_REPLIES_IN_A_ROW = 2
# The words each kind of bad reply gets in the ledger notice and in the error detail.
BAD_KIND_TEXT = {"truncated": "cut off at max_tokens", "empty": "empty (no text and no command)"}


def call_complete(call: dict) -> bool:
    """Whether a tool call's arguments are a complete JSON object. A reply cut off at max_tokens can end in the middle of the arguments
    (a long file in `write_file`); such a call must not run, and it must not stay in the conversation without an answer."""
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    args = fn.get("arguments")
    if isinstance(args, dict):
        return True
    if not isinstance(args, str) or not args.strip():
        return False
    try:
        return isinstance(json.loads(args), dict)
    except (ValueError, RecursionError):
        return False


def call_arguments_text(call: dict) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    args = fn.get("arguments")
    return args if isinstance(args, str) else (json.dumps(args) if isinstance(args, dict) else "")

# ---- tools --------------------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": ("Run one bash command. It starts in the run directory. Heredocs, command "
                        "substitution, subshells, eval and publishing commands are refused, with a "
                        "message that says how to rewrite the command."),
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": ("Read a text file inside the run directory (read only). The path is relative to "
                        "the run directory, or to your stage directory when it is not found there. A long "
                        "file comes back one page at a time: the page ends with the offset to pass to "
                        "read the next one."),
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "offset": {"type": "integer",
                                                                   "description": "first character to read "
                                                                                  "(default 0)"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a text file. The path is relative to your stage directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]


class Tools:
    def __init__(self, run_dir, stage_dir, env: dict, *, timeout: float = TOOL_TIMEOUT_S,
                 limit: int = TOOL_OUTPUT_CHARS, scratch=None):
        self.run_dir, self.stage_dir = Path(run_dir), Path(stage_dir)
        # <run>/tt-metal-scratch, offered only to stages 2 to 4 of a stock-runtime run (orchard.stages.scratch_stage): a copy of the
        # host tt-metal tree's models/ that those stages may change. write_file reaches it by a path that starts with its name.
        self.scratch = Path(scratch) if scratch is not None else None
        self.env, self.timeout, self.limit = env, timeout, limit
        # What the sweeps after this tool's shell commands stopped (see collect_survivors); a kernel-probe attempt lists them under `swept`.
        self.swept: list[dict] = []
        self.sweep_problems: list[str] = []         # what made a sweep incomplete (see Survivors); a sweep nobody can vouch for
        # What the port files held before the last shell command, so a command that makes one a runaway can be undone (issue 458). The
        # guard works only through descriptors beneath the real port directory (orchard/port_guard.py).
        self._guard = PortGuard(self.stage_dir, self.run_dir)

    def call(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            return "error: the arguments are not JSON"
        if not isinstance(args, dict):
            return "error: the arguments must be a JSON object"
        if name == "shell":
            return self.shell(args.get("command"))
        if name == "read_file":
            unknown = self.unknown_read_args(args, ("path", "offset"),
                                             "read_file takes path and offset (offset counts characters)")
            if unknown:
                return unknown
            return self.read_file(args.get("path"), args.get("offset", 0))
        if name == "write_file":
            return self.write_file(args.get("path"), args.get("content"))
        return f"error: there is no tool named {name!r}; the tools are shell, read_file and write_file"

    @staticmethod
    def unknown_read_args(args: dict, accepted: tuple, accepts: str):
        """An error text when `args` holds a name `accepted` lacks, else None. read_file used to drop such an argument
        without a word, so a model that sent `limit` (a count of lines) to a reader in characters was never told."""
        unknown = sorted(k for k in args if k not in accepted)
        if not unknown:
            return None
        names = ", ".join(repr(k) for k in unknown)
        return f"error: {accepts}; it has no argument named {names}"

    def shell(self, command) -> str:
        if not isinstance(command, str) or not command.strip():
            return "error: command must be a non-empty string"
        out_path = self.stage_dir / "log" / "last-shell-output.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._guard.remember()
        try:
            with open(out_path, "wb") as out, collect_survivors() as swept:
                try:
                    code, timed_out = spawn_checked(command, self.run_dir, self.env, self.timeout, out)
                finally:                       # an interrupt that cut the wait short still leaves what its sweep stopped
                    self.swept.extend(swept)
                    self.sweep_problems.extend(swept.problems)
        except Denied as exc:
            return f"refused: {exc}"
        text = out_path.read_text(encoding="utf-8", errors="replace")
        head = f"killed after {self.timeout} s" if timed_out else f"exit {code}"
        notes = self._guard.judge()
        return f"{head}\n{clip(text, self.limit)}" + ("\n" + "\n".join(notes) if notes else "")

    def _port_files(self) -> list:
        """The regular files under <stage>/port/ the guard looks at (orchard.port_guard.PortFile: links and caches left out, bounded)."""
        return self._guard.files()

    def restart_files(self, limit: int = 40) -> list[str]:
        """Lines naming the files of the stage directory and of its port that a restarted step is told about, at most `limit`, read
        with a bounded scan (at most PORT_GUARD_ENTRIES entries of the stage directory, and the guard's own bounds for the port). The port
        is read beneath its pinned directory, so a link in place of it lists nothing."""
        found: list[tuple[str, int]] = []
        try:
            with safeio.pinned_root(self.stage_dir) as stage, os.scandir(stage.fd) as entries:
                for seen, e in enumerate(entries, 1):
                    if seen > PORT_GUARD_ENTRIES or len(found) >= limit:
                        break
                    try:
                        if e.is_file(follow_symlinks=False):
                            found.append((os.path.relpath(self.stage_dir / e.name, self.run_dir), e.stat(follow_symlinks=False).st_size))
                    except OSError:
                        continue
        except (safeio.SafeReadError, OSError):
            pass
        for f in self._guard.files():
            if len(found) >= limit:
                break
            found.append((self._guard._display(f.parts), f.size))
        return [f"- {name} ({size} bytes)" for name, size in sorted(found)[:limit]]

    def target(self, path) -> Path | None:
        """The real path `path` names inside the stage directory, or None (links resolved).

        A path is relative to the stage directory, except that one starting with `stages/` is read the
        way read_file reads it, relative to the run directory, and must land in this stage. Models that
        read `stages/1/evidence/a.txt` write the same form, and read as stage-relative it made
        `stages/1/stages/1/evidence/a.txt`. A `stages/` path into another stage is refused."""
        if not isinstance(path, str) or not path.strip():
            return None
        if self.scratch is not None:
            sroot = os.path.realpath(self.scratch)
            first = path.replace("\\", "/").split("/", 1)[0]
            cand = os.path.realpath(os.path.join(os.path.realpath(self.run_dir), path) if first == self.scratch.name else path)
            if (first == self.scratch.name or os.path.isabs(path)) and cand != sroot and os.path.commonpath([sroot, cand]) == sroot:
                return Path(cand)
        root = os.path.realpath(self.stage_dir)
        base = os.path.realpath(self.run_dir) if path.replace("\\", "/").split("/", 1)[0] == "stages" else root
        p = os.path.realpath(os.path.join(base, path))
        if p == root or os.path.commonpath([root, p]) != root:
            return None
        return Path(p)

    def readable(self, path) -> tuple[str | None, str | None]:
        """(real path, None) of a file `path` names inside the run directory, or (None, the message to give the model).
        The path is tried against the run directory first and the stage directory second. Links are resolved before the
        check, so a link out of the run directory is refused. A directory, and a file over READ_FILE_MAX_BYTES, are
        errors."""
        if not isinstance(path, str) or not path.strip():
            return None, "error: path must be a non-empty string"
        root = os.path.realpath(self.run_dir)
        found = None
        for base in (root, os.path.realpath(self.stage_dir)):
            p = os.path.realpath(os.path.join(base, path))
            if os.path.commonpath([root, p]) != root:
                return None, f"refused: {path!r} is outside the run directory {self.run_dir}"
            if os.path.exists(p):
                found = p
                break
        if found is None:
            return None, (f"error: {path!r} does not exist in the run directory or in your stage directory "
                          f"{self.stage_dir}")
        if os.path.isdir(found):
            return None, f"error: {path!r} is a directory; use shell with ls to list it"
        size = os.path.getsize(found)
        if size > READ_FILE_MAX_BYTES:
            return None, (f"error: {path} is {size} bytes, too large to read page by page; use shell to read "
                          "parts of it (grep, head -c, tail -c, jq)")
        return found, None

    def read_file(self, path, offset=0) -> str:
        """Read a text file that lies inside the run directory. A path is tried against the run directory
        first and the stage directory second, because models write both. Links are resolved before the
        check, so a link out of the run directory is refused. Nothing is written.

        A file longer than the output limit comes back one page from `offset`, with a footer that says
        where the page sits in the file and the offset of the next page. (It used to be cut in the
        middle, and an agent that needed the middle read the same view again and again.)"""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return "error: offset must be a whole number of characters, 0 or more"
        found, error = self.readable(path)
        if error:
            return error
        with open(found, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        if offset == 0 and len(text) <= self.limit:
            return text
        if offset >= len(text):
            return f"error: offset {offset} is at or past the end of the file ({len(text)} characters)"
        end = min(offset + self.limit, len(text))
        if end == len(text):
            footer = f"[characters {offset} to {end} of {len(text)}: the end of the file]"
        else:
            footer = (f"[characters {offset} to {end} of {len(text)}; read on with read_file offset={end}, "
                      "or use shell (grep, jq) to find one field]")
        return f"{text[offset:end]}\n{footer}"

    def write_file(self, path, content) -> str:
        if not isinstance(path, str) or not path.strip():
            return ("error: write_file needs a path (a file in your stage directory, such as handoff.json) "
                    "and the content")
        if not isinstance(content, str):
            return "error: content must be a string"
        p = self.target(path)
        if p is None:
            return f"refused: {path!r} is outside your stage directory {self.stage_dir}"
        if p.name.startswith("ledger"):
            return "refused: the ledger is written only by the supervisor"
        rel = os.path.relpath(p, os.path.realpath(self.run_dir))
        problem = runaway.file_problem(content)
        if problem:
            return (f"refused: {rel} was not written: {problem}. A file you write is a few KB to tens of KB: write a shorter file, "
                    "do not repeat blocks, and split a long file into parts you write one at a time.")
        # A repeat of the last write is answered, not done: an agent that has finished sometimes writes
        # the same file again and again instead of ending the step (the first lab run).
        if p.is_file() and p.read_text(encoding="utf-8", errors="replace") == content:
            return (f"unchanged: {rel} already holds exactly this content; nothing was written. If every "
                    "file this step needs is written, end the step now: reply with a short summary and no "
                    "tool call.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {rel}"


# ---- the model endpoint -------------------------------------------------------------------------

class AgentError(Exception):
    """The model endpoint did not give a usable answer."""


def probe_model(endpoint: str, model: str, timeout: float = 5.0) -> bool:
    """Does the OpenAI-compatible server at `endpoint` (ending in /v1) list `model`?"""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and any(isinstance(m, dict) and m.get("id") == model
                                          for m in data.get("data") or [])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Outcome:
    status: str          # done, escalate, pause, operator-pause, abort, turns, error
    turns: int
    final_text: str = ""
    detail: str = ""


class AgentStep:
    """One agent step: a fresh conversation for one stage phase, run until the model stops calling
    tools, the turn limit is reached, or the supervisor says stop.

    `control` is the supervisor's actuator: `take_nudges(agent) -> list[str]` and
    `stop_reason() -> str | None`. `feed` is the watchdog's `feed(Event)`.
    """

    def __init__(self, *, agent: str, endpoint: str, model: str, tools: Tools, ledger, stage: int,
                 phase: str, feed, control, run_dir, evidence_dir, log_path, http=post_json,
                 clock=time.time, guard: RetryGuard | None = None, max_turns: int = AGENT_MAX_TURNS,
                 max_tokens: int = AGENT_MAX_TOKENS, timeout: float = AGENT_REQUEST_TIMEOUT_S,
                 thinking: bool = AGENT_THINKING, finished=None, temperature: float = 0,
                 top_p: float | None = None, seed: int | None = None,
                 frequency_penalty: float | None = None, repetition_penalty: float | None = None,
                 unstick: "runaway.Unstick | None" = None, restart_on_caps: int = 0):
        self.agent, self.endpoint, self.model, self.tools = agent, endpoint, model, tools
        self.ledger, self.stage, self.phase = ledger, stage, phase
        self.feed, self.control, self.http, self.clock = feed, control, http, clock
        self.run_dir, self.evidence_dir = Path(run_dir), Path(evidence_dir)
        self.log_path = Path(log_path)
        self.guard = guard if guard is not None else RetryGuard()
        self.max_turns, self.max_tokens, self.timeout = max_turns, max_tokens, timeout
        self.thinking = thinking     # False sends enable_thinking=false (see defaults.AGENT_THINKING)
        # Sampling. The default is greedy decoding (temperature 0), sent exactly as before. top_p and seed are sent
        # only when a caller sets them; whether a server honours the seed is up to the server.
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        # Issue 458. The penalties are sent on every request only when a caller sets them. `unstick` is the sampling of the request that
        # follows a reply cut off at max_tokens (orchard/runaway.py), None to send nothing different. `restart_on_caps` is how many times a
        # step may drop its conversation and start again from the task after two capped replies in a row, and goes on.
        self.frequency_penalty, self.repetition_penalty = frequency_penalty, repetition_penalty
        self.unstick, self.restart_on_caps = unstick, restart_on_caps
        self._after_cap, self._caps, self._loops, self._restarts = False, 0, 0, 0
        # The supervisor's instructions that a restart must carry over, as (kind, text), and the task as it was first given.
        self._carry: list[tuple[str, str]] = []
        self._task = ""
        self._sent_request: dict = {}
        # finished() -> bool: are this step's files written and valid (the supervisor's check)? When it
        # is true and the agent writes a file again unchanged, the step ends as done: an agent that has
        # finished sometimes repeats its last write instead of ending (the first lab run).
        self.finished = finished
        self._seen: dict[str, tuple[int, int]] = {}
        # The model turn number on every Event this step feeds the watchdog. It counts up through
        # `run` and `continue_with`, so a continuation's turns never reuse an earlier number.
        self._turn = 0
        # The conversation, kept on the step so `continue_with` can add to it. None until `run`.
        self.messages: list[dict] | None = None

    # ---- helpers ----------------------------------------------------------------------------

    def _event(self, kind: str, **fields) -> None:
        self.feed(Event(ts=self.clock(), agent=self.agent, kind=kind, stage=self.stage,
                        turn=self._turn or None, **fields))

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _snapshot(self) -> dict[str, tuple[int, int]]:
        out = {}
        if self.evidence_dir.is_dir():
            root = os.path.realpath(self.evidence_dir)
            for f in sorted(self.evidence_dir.rglob("*")):
                # A link that leaves evidence/ (to test-result.json, say) is not evidence the agent produced there.
                if f.is_file() and os.path.realpath(f).startswith(root + os.sep):
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime_ns)
        return out

    def _record_new_evidence(self) -> None:
        now = self._snapshot()
        for path, stamp in now.items():
            if self._seen.get(path) != stamp:
                try:
                    rec = evidence_record(self.run_dir, path)
                except safeio.SafeReadError as exc:
                    # The file stays out of the ledger's evidence; the reason is named, and the stage gate refuses a path it cannot use.
                    self.ledger.append("decision", self.stage, decision="an evidence file was refused and not recorded", phase=self.phase,
                                       file=os.path.relpath(path, self.run_dir), reason=exc.reason)
                    continue
                self.ledger.append("evidence", self.stage, what="evidence file", phase=self.phase, **rec)
                self._event("evidence", name=rec["path"])
        self._seen = now

    def _send(self, request: dict) -> dict:
        url = self.endpoint.rstrip("/") + "/chat/completions"
        last = None
        # The guard allows the call and one retry of the identical request (spec section 3). The
        # range is only a backstop; the guard is what stops a third send.
        for attempt in range(1, 4):
            if not self.guard.allow(self.agent, request):
                break
            try:
                return self.http(url, request, self.timeout)
            except (OSError, ValueError, CanaryError) as exc:
                last = exc
                self.ledger.append("retry", self.stage, what="model request", phase=self.phase,
                                   attempt=attempt, error=str(exc)[:300])
        raise AgentError(f"the model request to {url} failed and was retried once: {last}")

    def _end(self, status: str, turns: int, final_text: str = "", detail: str = "") -> Outcome:
        if self.log_path.exists():
            try:
                rec = evidence_record(self.run_dir, self.log_path)
            except safeio.SafeReadError as exc:
                rec = {"unreadable": exc.reason}
            self.ledger.append("evidence", self.stage, what="transcript", phase=self.phase, status=status, turns=turns, **rec)
        return Outcome(status, turns, final_text, detail)

    @staticmethod
    def _shape(name: str, args: str) -> str | None:
        """The shape of a shell call's command (orchard.watchdog.command_shape), else None."""
        if name != "shell":
            return None
        try:
            command = json.loads(args).get("command")
        except (ValueError, AttributeError):
            return None
        return command_shape(command) if isinstance(command, str) else None

    @staticmethod
    def _bad_detail(kinds: list[str], tokens) -> str:
        """The error detail when too many replies in a row ran no command. It names each kind."""
        if kinds == ["truncated", "truncated"]:
            return (f"the reply was cut off at max_tokens twice in a row ({tokens} tokens each); "
                    "no action was taken")
        if kinds == ["empty", "empty"]:
            return "the reply was empty (no text and no command) twice in a row; no action was taken"
        return ("two replies in a row ran no command: "
                + ", then ".join(BAD_KIND_TEXT[k] for k in kinds) + "; no action was taken")

    def _send_profiled(self, request: dict, plain: dict) -> dict:
        """Send `request`. When it carries the retry profile and the server refuses it, send `plain` instead (once) and say so in the
        ledger: not every server takes every sampling parameter, and the step must go on."""
        self._sent_request = request
        if request is plain or self.unstick is None:
            return self._send(request)
        try:
            return self._send(request)
        except AgentError as exc:
            self.ledger.append("notice", self.stage, watchdog=True, agent=self.agent, phase=self.phase,
                               what="retry sampling refused by the server", retry_sampling=self.unstick.record(),
                               error=str(exc)[:300])
            self._after_cap = False
            self._sent_request = plain                          # the transcript records the request that was actually served
            return self._send(plain)

    def _capped_in_stage(self) -> int:
        """Capped replies the ledger holds for this stage, from every step so far."""
        try:
            return sum(1 for e in self.ledger.read() if e.get("stage") == self.stage and e.get("event") == "notice"
                       and (e.get("data") or {}).get("reply_capped"))
        except Exception:
            return 0

    def _note_capped(self, content: str, tokens) -> tuple[dict, str | None]:
        """One reply was cut off at max_tokens: count it, find out whether it was a repetition loop, and arm the retry profile. Returns
        the fields for the ledger notice and the loop reason (None when it was not a loop)."""
        loop = runaway.loop_reason(content)
        self._caps += 1
        self._loops += 1 if loop else 0
        self._after_cap = True
        fields = {"reply_capped": True, "repetition": bool(loop), "capped_in_step": self._caps,
                  "capped_in_stage": self._capped_in_stage() + 1}
        if loop:
            fields["repetition_detail"] = loop[:200]
        if self.unstick is not None:
            fields["retry_sampling"] = self.unstick.record()
        return fields, loop

    def _loop_detail(self) -> str:
        """What the error detail adds when some of the capped replies were repetition loops: the coder's output is the cause."""
        if not self._loops:
            return ""
        return (f"; {self._loops} of the {self._caps} capped replies in this step were repetition loops: "
                "the coder's replies are the likely cause, so the port may be fine")

    def _carry_instruction(self, kind: str, text: str) -> None:
        """Remember an instruction the supervisor gave this step, so a restart can carry it into the fresh context. Bounded: each is cut,
        and past CARRIED_INSTRUCTIONS the oldest nudge goes first (a continuation, the supervisor's request to change the work, goes last)."""
        text = text if isinstance(text, str) else str(text)
        if len(text) > CARRIED_CHARS:
            text = text[:CARRIED_CHARS] + "\n(the rest of this instruction was cut)"
        self._carry.append((kind, text))
        while len(self._carry) > CARRIED_INSTRUCTIONS:
            drop = next((i for i, (k, _) in enumerate(self._carry) if k == "nudge"), 0)
            del self._carry[drop]

    def _restart_fresh(self, turn: int) -> None:
        """Drop the conversation and start again from the task, with the list of files on disk. Called after two capped replies in a row,
        at most `restart_on_caps` times in a step. The loops built up in a context of 35,000 tokens and more; a short context does not
        carry them."""
        messages = self.messages
        files = self.tools.restart_files(40)
        # The task as first given (a second restart must not nest the first restart's text), then the instructions the supervisor added
        # since: gate feedback, a wrap-up request, nudges. Without them the fresh context would send the step back to what it was told to change.
        task = self._task if self._task else (messages[1]["content"] if len(messages) > 1 and isinstance(messages[1].get("content"), str) else "")
        carried = "".join(f"\n\n## Earlier instruction from the supervisor ({kind})\n{text}" for kind, text in self._carry)
        text = f"{task}{carried}\n\n## From the supervisor\n{RESTART_NOTE}\n" + ("\n".join(files) if files else "(no files yet)")
        messages[:] = [messages[0], {"role": "user", "content": text}]
        self._restarts += 1
        self.ledger.append("decision", self.stage, decision="context restarted after capped replies", phase=self.phase, turn=turn,
                           capped_in_step=self._caps, repetition_loops=self._loops, restarts=self._restarts,
                           files_listed=len(files), instructions_carried=len(self._carry))

    # ---- the loop ---------------------------------------------------------------------------

    def run(self, system: str, user: str) -> Outcome:
        """Start the conversation and run it until a final answer, the turn limit or a stop."""
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self._task = user if isinstance(user, str) else ""
        self._carry = []
        self._seen = self._snapshot()       # files already here (a resumed stage) are not new
        return self._loop(self.max_turns, logged=0)

    def continue_with(self, text: str, *, max_turns: int, log_path) -> Outcome:
        """Add one user message to this step's conversation and run the loop again.

        The supervisor uses this once per step: for gate feedback, when a step ended "done" and
        the stage's exit gate failed, or for a wrap-up, when a step used up its turns with
        evidence on disk and its deliverable not written. The continuation has its own turn cap and writes its own log
        at `log_path`, so the first run's log stays as it was.
        """
        if self.messages is None:
            raise RuntimeError("continue_with needs a step that has run")
        self.log_path = Path(log_path)
        logged = len(self.messages)          # the continuation's log starts at the new message
        self.messages.append({"role": "user", "content": text})
        self._carry_instruction("continuation", text)
        return self._loop(max_turns, logged=logged)

    def _loop(self, max_turns: int, *, logged: int) -> Outcome:
        messages = self.messages
        bad_run: list[str] = []             # kinds of the replies in a row that ran no command
        last_reply = ""                     # the visible text of the last reply that called a tool
        for turn in range(1, max_turns + 1):
            self._turn += 1
            for text in self.control.take_nudges(self.agent):
                messages.append({"role": "user", "content": text})
                self._carry_instruction("nudge", text)
            reason = self.control.stop_reason()
            if reason:
                return self._end(reason, turn - 1)
            request = {"model": self.model, "messages": messages, "tools": getattr(self.tools, "schemas", TOOL_SCHEMAS),
                       "temperature": 0 if self.temperature == 0 else self.temperature,
                       "max_tokens": self.max_tokens, "stream": False}
            if self.top_p is not None:
                request["top_p"] = self.top_p
            if self.seed is not None:
                request["seed"] = self.seed
            if not self.thinking:
                request["chat_template_kwargs"] = {"enable_thinking": False}
            if self.frequency_penalty is not None:
                request["frequency_penalty"] = self.frequency_penalty
            if self.repetition_penalty is not None:
                request["repetition_penalty"] = self.repetition_penalty
            plain = request
            if self._after_cap and self.unstick is not None:
                request = self.unstick.apply(plain)       # the request after a capped reply: not greedy, a lower cap
            try:
                data = self._send_profiled(request, plain)
                msg = data["choices"][0]["message"]
                finish = data["choices"][0].get("finish_reason")
                if not isinstance(msg, dict):
                    raise TypeError("message is not an object")
            except AgentError as exc:
                return self._end("error", turn - 1, detail=str(exc))
            except (KeyError, IndexError, TypeError) as exc:
                return self._end("error", turn - 1, detail=f"no choices[0].message: {exc}")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
            # A reply cut off at max_tokens can end inside a tool call's arguments. Such a call must not run and must not stay in the
            # conversation without an answer; the calls in the same reply that are complete do run. Either way the cap is counted below.
            raw_calls, cut_calls = calls, []
            if finish == "length" and calls:
                cut_calls = [c for c in calls if not call_complete(c)]
                calls = [c for c in calls if call_complete(c)]
            cut_text = "".join(call_arguments_text(c) for c in cut_calls)
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            details = usage.get("completion_tokens_details") or {}
            assistant = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = calls
            received = {**assistant, "tool_calls": raw_calls} if cut_calls else assistant      # the transcript shows what the coder sent
            # Two kinds of reply ran no command and are not a final answer either. "truncated": it
            # hit max_tokens before it ran any tool (a reasoning model can spend the whole budget
            # thinking). "empty": it ended normally with no text and no tool call (a reasoning model
            # can finish its thinking and then say nothing). Neither is added to the conversation.
            if calls:
                bad = None
            elif finish == "length":
                bad = "truncated"
            elif not content.strip():
                bad = "empty"
            else:
                bad = None
            if bad is None:
                messages.append(assistant)
            self._log({"turn": turn, "sent": messages[logged:-1] if bad is None else messages[logged:],
                       "received": received, "finish_reason": finish, "usage": usage,
                       "request": {k: self._sent_request[k] for k in LOGGED_SAMPLING if k in self._sent_request}})
            logged = len(messages)
            self._event("response", input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        thinking_tokens=details.get("reasoning_tokens") if isinstance(details, dict) else None,
                        text_hash=sha(content) if content else None, had_tool_call=bool(calls))
            if bad:
                bad_run.append(bad)
                tokens = usage.get("completion_tokens")
                loop = None
                capped: dict = {}
                if bad == "truncated":
                    tokens = tokens or self.max_tokens
                    capped, loop = self._note_capped(content if content.strip() else cut_text, tokens)
                    if cut_calls:
                        capped["incomplete_tool_calls"] = len(cut_calls)
                self.ledger.append("notice", self.stage, watchdog=True, agent=self.agent,
                                   what="reply cut off at max_tokens" if bad == "truncated" else "empty reply",
                                   phase=self.phase, turn=turn, finish_reason=finish,
                                   completion_tokens=tokens, in_a_row=len(bad_run), **capped)
                if len(bad_run) >= BAD_REPLIES_IN_A_ROW:
                    if "truncated" in bad_run and self._restarts < self.restart_on_caps:
                        self._restart_fresh(turn)
                        logged = 0                      # the replacement conversation is logged from its start: the next row shows its prompt
                        bad_run = []
                        continue
                    return self._end("error", turn, detail=self._bad_detail(bad_run, tokens) + self._loop_detail())
                messages.append({"role": "user",
                                 "content": (LOOP_NUDGE if loop else TRUNCATION_NUDGE) if bad == "truncated" else EMPTY_NUDGE})
                continue
            if finish == "length":
                # The reply hit max_tokens but ran at least one complete call. Some calls may have been cut inside their arguments; or the
                # server's parser dropped what it could not finish and every call left is complete, so no cut call is visible. The cap
                # counts either way and the next request gets the retry profile (review of PR 462, round 4).
                tokens = usage.get("completion_tokens") or self.max_tokens
                capped, _ = self._note_capped(content if content.strip() else cut_text, tokens)
                self.ledger.append("notice", self.stage, watchdog=True, agent=self.agent,
                                   what=("reply cut off at max_tokens (a tool call's arguments were incomplete)" if cut_calls
                                         else "reply cut off at max_tokens (every tool call in it was complete and ran)"),
                                   phase=self.phase, turn=turn, finish_reason=finish, completion_tokens=tokens, in_a_row=0,
                                   incomplete_tool_calls=len(cut_calls), complete_tool_calls=len(calls), **capped)
            else:
                self._after_cap = False
            bad_run = []
            if not calls:
                return self._end(self.control.stop_reason() or "done", turn, final_text=content)
            repeated = False
            for call in calls:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name, args = str(fn.get("name", "")), fn.get("arguments") or "{}"
                args = args if isinstance(args, str) else json.dumps(args)
                self._event("tool_call", tool=name, args_hash=sha(args), shape=self._shape(name, args))
                result = self.tools.call(name, args)
                messages.append({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": result})
                # wrote: did this write_file call write its file? None for other tools.
                self._event("tool_result", tool=name, output_hash=sha(result),
                            wrote=result.startswith("wrote ") if name == "write_file" else None)
                self._record_new_evidence()
                repeated = repeated or (name == "write_file" and result.startswith("unchanged:"))
            # The same visible text again, with another tool call: on lab run 2 the stage 8 agent said its
            # summary was done, then ran `echo "Done"`, `echo "Finished"` ... with that same text for 16
            # turns, until the identical-responses watchdog escalated a stage whose files had passed.
            same_reply = bool(content.strip()) and content == last_reply
            last_reply = content
            if (repeated or same_reply) and self.finished is not None and self._finished():
                if repeated:
                    self.ledger.append("decision", self.stage, decision="step finished: repeated write",
                                       phase=self.phase, turn=turn)
                    return self._end("done", turn, detail="the step's files were written and valid, and the "
                                                          "agent wrote them again unchanged instead of ending")
                self.ledger.append("decision", self.stage, decision="step finished: repeated reply",
                                   phase=self.phase, turn=turn)
                return self._end("done", turn, detail="the step's files were written and valid, and the agent "
                                                      "gave the same reply again instead of ending")
        return self._end("turns", max_turns, detail=f"no final answer after {max_turns} turns")

    def _finished(self) -> bool:
        try:
            return bool(self.finished())
        except Exception:                # a check that fails is not a finished step
            return False
