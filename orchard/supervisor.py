# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor loop: run a bring-up stage by stage, from the ledger (spec sections 5 to 7, 10).

This module owns the run. At start it records the run (model, resolved versions, inputs) or, on a
restart, replays the ledger. It keeps the coder (the chip tier's model server) running under a
lease this process owns: it starts it, re-leases it after a restart, restarts it once if it dies,
and finishes a park that a crash interrupted (orchard/handoff.py). For each stage it checks the
run-wide caps and the free disk, opens the stage directory (orchard/stages.py), runs the agent
steps (orchard/agent.py) under the watchdog with a real actuator, runs the hardware test on a
leased board, parking the coder when no board is free, and writes the stage's end with the exit
gate's result. A failed stage is escalated once to its diagnose tier (or the [escalation]
default); a second failure pauses the run.

Each attempt at a stage starts its agent steps from a fresh context built from the ledger
(orchard/context.py), so a long run never carries a long conversation. There is one exception.
When a step ends "done" and the exit gate fails, the same conversation gets one more turn budget
(AGENT_CONTINUATION_TURNS): a user message lists the gate's reasons word for word and names the file
the gate reads. The live Qwen3.8 run showed why. Stage 1's agent worked 48 turns and wrote real
evidence, then ended without reference.json, and the escalation started a fresh context that had
lost all of that work. The ledger records a "gate feedback" decision before the continuation runs,
and the continuation writes its own log. It happens at most once per run of the stage body, and
only after status "done". If the gate passes afterwards the stage passes; otherwise it is escalated
or fails as before. A kill during the continuation loses the conversation: the resumed stage starts
a fresh step, which may use its own one continuation.

A second exception covers a step that runs out of turns. When a step ends with status "turns",
the step has not written its deliverable (the gate file for a run or finish step, hw_test.json
or hw_tests.json for a prepare step: missing, or unchanged since the step started), and its stage
directory holds at least one evidence file, the same
conversation gets one wrap-up (AGENT_WRAPUP_TURNS). The message names the missing file, lists the
evidence files, and allows no new investigation. The second-model run showed why: stage 0's agent
wrote 8 evidence files, used all 60 turns and never wrote delta.json. The ledger records a "wrap-up"
decision before the continuation runs. There is at most one wrap-up per run of the stage body, and
a step that was wrapped up gets no gate feedback: if its file fails the gate, the stage fails or
escalates as before. A kill during the wrap-up loses the conversation, as with gate feedback.

The first start of the coder in a run is also asked a
known-answer question (7 times 6); a server that answers without 42 blocks the run, because the
canary alone would accept noise. When stage 0 finds that the model needs new model code
(a full port), the run pauses before stage 2 for the operator. When stage 0 finds that only the
weights differ (weights-only), stage 2 runs the weights-swap-check skill and its gate in place of
the functional decoder, and stages 3 (full model), 5 (serving) and 6 (qualitative check and
benchmark) are recorded as skipped. Stage 0's passing
stage_end records that path, and every later choice is
read from there (orchard/stages.py, run_path). An unknown path keeps the plan 4 table.

Skills name machine paths through placeholders such as {{CACHE_ROOT}} (orchard/paths.py). The
run_start entry records the values (from --cache-root, --hf-home, --operator-home and their
defaults), the context fills them in, and a resumed run keeps them.

Operator commands go through a one-word control file in the run directory: pause, resume, abort.
SIGINT (Ctrl-C) and SIGTERM take the same path as abort: the running command is killed, the coder
is stopped, every lease this process holds is released, the ledger records the abort and the
process exits 4. Any other error that ends the run (an adapter error, a corrupt ledger, a bug) also
stops the coder and releases the leases before it is reported, with exit 3; that run is not
aborted, so running the same command again resumes it. When the ledger cannot be read, the
release works from what this process remembers. A SIGKILL, a power cut or a crash of Python
itself runs no handler: the coder then keeps its chips under a dead pid's lease until the run is
resumed (which re-leases it) or aborted.
An abort itself is terminal: running the same command again returns exit 4 and starts nothing. `--resume-aborted` is the operator's explicit
way to carry on after an operator's or a signal's abort (the web page's `Resume run`); it records `resume` with `after: abort`.
`python3 -m orchard.supervisor status --run-dir DIR [--json]` prints the state of a run without
taking the ledger lock or writing anything (orchard/status.py). The supervisor writes its pid to
`<run dir>/supervisor.pid` at start so that command can tell a live supervisor from a dead one.
A paused supervisor keeps its leases and waits. Abort stops the coder, releases its lease and
closes the ledger. The run ends at "ready for operator review". The supervisor never publishes.
Agents are kept from publishing in three ways, none of them complete: the command runner refuses
the common publish, push and upload spellings; agent shells get no tokens in their environment;
and a preflight refuses to start while known credential files exist in the operator's home,
unless the operator passes --accept-credentials-visible (the ledger records that). Agent shells
run as the same user, so code an agent runs can still read any file that user can read.

Plan 4 runs stages 0 to 6 and 8; stages 3, 5 and 6 are recorded as skipped on the weights-only
path. Stage 7 (plan 5) packages the model only on the weights-only path of a run started with
--package-format v6 and --package-namespace; otherwise it is recorded as skipped. It is supervisor
code (orchard/package.py) with no agent: it stages a v6 thin bundle per profile, scrubs each one,
installs a copy of the required profile, boots that copy on a leased board as the stage's hardware
test (parking the coder when no board is free, as stage 2 does), writes the cards and the publish
commands as text, and runs gate_package. A failure pauses the run and is not escalated, because no
model ran. The run's ledger records the package options in run_start, and a resumed run keeps them.
A run that started without them can be given them on a resume before stage 7 starts; the ledger
then records a "package options set" decision. --package-format v5.1 is refused at start
(orchard/defaults.py, PACKAGE_DEFERRED). Stage 7 looks for other chip counts of the nearest model
in --package-models-root, which defaults to the run's {{TT_MODEL_ROOT}} (orchard/paths.py).

On the weights-only path stage 4 runs a list of hardware tests, one per chip configuration
(orchard/hwtests.py). The prepare step writes hw_tests.json; the supervisor validates it, writes
tests/plan.json (the resume marker) and runs the tests in order of chip count, each under its own
lease, then the finish step writes result.json. Each test's record is written as soon as it ends,
so a crash resumes at the first configuration with no record. For each test:
- enough free boards: lease them (one board: the free one, by its first chip) and leave the coder
  loaded;
- otherwise the coder's boards are needed: first lease any further boards the test needs (so a
  board that cannot be had leaves the coder untouched), then park the coder, run the test on the
  coder's chips plus the further boards, release the further boards and restore the coder. A
  test that fails is recorded, and the coder is restored all the same. With the coder on two
  chips, the 4-chip test is the only one that parks. Measured costs of one park and restore on
  this box: `tt-model stop` 1 to 2 s, a gozer reset 41.7 s (twice: park and restore), the 2-chip
  coder back to ready in about 120 s. The 4-chip coder's restart was not measured.
After every test the supervisor stops and removes any container that still carries this run's test
label (`run_label`); a container that survives that blocks the stage before any lease is released
or reset. Before each test it checks the free disk again (TEST_DISK_GB), and it moves aside a
tensor cache whose last test did not exit 0: a conversion that stopped part way leaves a cache
that reads back without an error.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import hashlib
import json
from types import SimpleNamespace
import os
import pwd
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from orchard.adapters import AdapterError, Lease
from orchard.agent import AgentStep, Tools, agent_env, probe_model, spawn_checked
from orchard import audio_reference_env, audio_runtime, bus, nearest
from orchard.audio_facts import serving_of
from orchard.blocked import block_code, last_pause
from orchard.blocked import write_bundle as write_blocked_bundle
from orchard.canary import CanaryError, compare, post_json
from orchard.canary import ask as canary_ask
from orchard.commands import run_command
from orchard.context import build_messages, facts_from
from orchard.defaults import (AGENT_CONTINUATION_TURNS, AGENT_WRAPUP_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, PACKAGE_DEFERRED, PACKAGE_FORMATS,
                              GRAFT_TRIAL_DEADLINE_S, GRAFT_TRIAL_DISK_GB, RUN_CANARY_PROMPT, SAMPLING_TRIAL_DEADLINE_S, SAMPLING_TRIAL_DISK_GB, SAMPLING_TRIAL_RUN_DISK_GB, STOP_TIMEOUT_S, TEST_DISK_GB, FAST_FAIL_S)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (failed_tests, load_plan, move_aside, pending, read_plan, suspect_caches,
                             unrecorded, write_plan, write_record, write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.package import PackageError, stage_all
from orchard.package import finish as package_finish
from orchard import stock_runtime
from orchard.paths import (PATHS_RECORDED, RunPaths, UnknownPlaceholder, absolute_path,
                           recorded_paths)
from orchard.runner import Denied, check_string
from orchard import graft_trial as graft
from orchard import sampling_choice as sampling
from orchard import kept_coder, kernelprobe, swap_draft
from orchard.server import ServerControl, ServerError, ServerNotOwned, ServerSpec, StopCheck
from orchard.stages import (AUDIO_STAGE_1, AUDIO_STAGE_2, PACKAGE_OPTIONS_SET, GateResult, TierUnavailable, abort_can_resume, attempt_started_ts,
                            budget_cap, check_disk, coder_state, delta_class, delta_facts, delta_path, run_facts, evidence_record,
                            open_stage_dir, package_format, package_options, resolve_endpoint,
                            resolve_skill, run_class, run_path, run_progress, scratch_stage, spec_for, stock_params, tier_for)
from orchard.tiers import TierConfigError, load
from orchard.watchdog import (Event, IdenticalResponses, Ladder, NoFileWritten, NoNewEvidence,
                              RepeatedToolCall, RetryGuard, StageOverBudget, ThinkingWithoutAction,
                              TurnRepeat, Watchdog)

WHO = "orchard:supervisor"
AGENT = "stage-agent"                 # the launched agent's name; the ladder counts its rungs per stage
EXIT_READY, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED = 0, 2, 3, 4
EXIT_BLOCKED = 5     # an unattended run could not go on and named why (orchard/blocked.py)
FULL_PORT = ("stage 0 found a full port (new model code is needed); plan 4 runs weights-only "
             "bring-ups, so the operator decides whether to go on")
RESTRICTED_HEAD = ("stage 0 found the nearest model's backbone with a restricted head (its lm_head is replaced by sidecar "
                   "rows); the harness does not serve that, so a person decides whether to go on")
AUDIO_UNBUILT = ("stage 0 found an audio model; the audio path (its stage 1 and stage 2) is not built yet, and the text stages would "
                 "measure the wrong thing, so a person decides whether to go on")
AUDIO_ENV_MISSING = ("stage 0 found an audio model that the harness's own audio runtime serves, but the audio path's environment is not on "
                     "this machine, so a person decides whether to go on: ")
AUDIO_CHIP_DEADLINE_S = 3600.0         # the audio stage 2 hardware test: two chip runs, a kernel compile on the first, and the model loads
SKILLS_DIR = Path(__file__).with_name("skills")


# ---- operator control ---------------------------------------------------------------------------

class Interrupted(BaseException):
    """SIGINT or SIGTERM arrived. A BaseException, so an `except Exception` on the way does not
    swallow it."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


def _raise_interrupted(signum, frame):
    raise Interrupted(signal.Signals(signum).name)


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class Control:
    """One word in `<run dir>/control`: pause, resume or abort.

    `take` moves the file aside to `control.done-<k>` (never deleted), so each command acts once
    and a stale "resume" cannot end a later pause.
    """
    COMMANDS = ("pause", "resume", "abort")

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "control"

    def write(self, command: str) -> None:
        if command not in self.COMMANDS:
            raise ValueError(f"command must be one of {self.COMMANDS}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name("control.tmp")
        tmp.write_text(command + "\n")
        os.replace(tmp, self.path)

    def take(self) -> str | None:
        try:
            text = self.path.read_text().strip()
        except FileNotFoundError:
            return None
        k = 1
        while (done := self.path.with_name(f"control.done-{k}")).exists():
            k += 1
        os.replace(self.path, done)
        return text if text in self.COMMANDS else None


def write_pid_file(run_dir) -> None:
    """Record this process's pid in `<run dir>/supervisor.pid`, replacing any earlier one.

    `python3 -m orchard.supervisor status` reads it to tell a running supervisor from a dead one.
    The file is written by rename, so a reader never sees half a number. A pid alone can be reused
    by another process after a crash, so the status command also checks the command line.
    """
    path = Path(run_dir) / "supervisor.pid"
    tmp = path.with_name("supervisor.pid.tmp")
    tmp.write_text(f"{os.getpid()}\n")
    os.replace(tmp, path)


class RunActuator:
    """The watchdog's Actuator for the launched stage agent, plus the agent step's stop question.

    A nudge is queued and added to the agent's next request. Escalate and pause end the current
    step; the supervisor then acts on the result. Operator commands are read here too.
    """

    def __init__(self, control: Control):
        self.control = control
        self._nudges: dict[str, list[str]] = {}
        self._stop: str | None = None

    def nudge(self, agent: str, message: str) -> None:
        self._nudges.setdefault(agent, []).append(message)

    def escalate(self, agent: str, stage) -> None:
        self._stop = self._stop or "escalate"

    def pause(self, reason: str, evidence: dict) -> None:
        self._stop = "pause"

    def take_nudges(self, agent: str) -> list[str]:
        return self._nudges.pop(agent, [])

    def stop_reason(self) -> str | None:
        cmd = self.control.take()
        if cmd == "abort":
            self._stop = "abort"
        elif cmd == "pause" and self._stop is None:
            self._stop = "operator-pause"
        return self._stop

    def clear(self) -> None:
        self._stop = None
        self._nudges.clear()


class LabelledContainers:
    """Containers this run's hardware tests started, found by their docker label.

    A test script stops its own container in a `finally`. A test killed at its deadline runs no
    `finally`, and a container started with `docker run --detach` is not in the killed session, so
    it keeps the chips. The supervisor looks for the label after every test of a list.
    """

    def __init__(self, run=run_command, docker: str = "docker"):
        self.run, self.docker = run, docker

    def list(self, label: str) -> list[str] | None:
        """The ids of containers with `label`, or None when docker could not be asked."""
        r = self.run([self.docker, "ps", "--all", "--quiet", "--filter", f"label={label}"], CMD_TIMEOUT_S)
        return r.stdout.split() if r.returncode == 0 else None

    def remove(self, cid: str) -> None:
        self.run([self.docker, "stop", "-t", "60", cid], STOP_TIMEOUT_S)
        self.run([self.docker, "rm", "--force", cid], CMD_TIMEOUT_S)


class ExternalStandIn:
    """The CPU tier as the park's stand-in: a server the operator runs (ollama), never this process.

    Starting and stopping it belongs to the operator, so spawn, wait_ready and stop do nothing and
    the stop check always passes. The park still requires it to answer the canary before the coder stops
    (spec section 6, step 2).
    """

    def __init__(self, endpoint: str, model: str, http=post_json):
        self.endpoint, self.model, self.http = endpoint, model, http
        base = endpoint.rstrip("/")
        self.base = base[:-3] if base.endswith("/v1") else base

    def spawn(self) -> None:
        pass

    def wait_ready(self) -> None:
        pass

    def ask(self, prompt: str) -> str:
        return canary_ask(self.base, self.model, prompt, http=self.http)

    def stop(self) -> None:
        pass

    def record(self) -> dict:
        return {"kind": "external", "endpoint": self.endpoint, "model": self.model}

    def adopt(self, record: dict) -> str | None:
        return None

    def confirm_stopped(self) -> StopCheck:
        return StopCheck(True, {"external": True}, {})


# ---- versions and command-line helpers -----------------------------------------------------------

def resolve_versions(spec: ServerSpec, run=run_command) -> dict:
    """What this host can say about the versions the run uses (spec section 9).

    tt-metal and vLLM live inside the coder's image or bundle, so they are recorded as TODO.
    """
    v = {"coder_target": spec.target, "coder_profile": spec.profile, "coder_image_id": spec.image_id,
         "tt_metal": "TODO: inside the coder image; plan 4 does not read it",
         "vllm": "TODO: inside the coder image; plan 4 does not read it"}
    r = run(["tt-model", "--version"], CMD_TIMEOUT_S)
    v["tt_model"] = r.stdout.strip() if r.returncode == 0 else f"unknown (exit {r.returncode})"
    r = run(["tt-smi", "-s"], CMD_TIMEOUT_S)
    try:
        devices = json.loads(r.stdout)["device_info"]
        v["firmware"] = sorted({d["firmwares"]["fw_bundle_version"] for d in devices})
    except (ValueError, KeyError, TypeError):
        v["firmware"] = f"unknown (tt-smi -s exit {r.returncode})"
    return v


def pairs(items: list[str], what: str) -> dict:
    out = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--{what} takes NAME=VALUE, got {item!r}")
        out[name] = value
    return out


# Files in the operator's home that hold credentials an agent could use, relative to that home.
# SSH private keys (~/.ssh/id_*, without .pub) are found by name as well.
CREDENTIAL_PATHS = (".cache/huggingface/token", ".config/gh/hosts.yml", ".netrc",
                    ".docker/config.json")


def visible_credentials(home) -> list[Path]:
    """Credential files that exist under `home`. Only existence is checked; nothing is read."""
    home = Path(home)
    found = [home / rel for rel in CREDENTIAL_PATHS if (home / rel).exists()]
    ssh = home / ".ssh"
    if ssh.is_dir():
        found += sorted(p for p in ssh.glob("id_*") if not p.name.endswith(".pub"))
    return found


def _discard(path: Path) -> None:
    """Remove a file, a link or a directory tree, quietly (a link is removed, never followed)."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def save_reference(src: Path, cache: Path, name: str) -> bool:
    """Put the CPU reference made in `src` into the reference cache as `cache/name`, replacing an entry that is already there. The caller runs
    this only for a reference that was made fresh, and a fresh one is made only when the cache had no entry that `audio_stages.cached_run` accepts
    for the run's identity (model revision, configuration, library commit, settings), so an entry found here is stale and must not block the save
    (issue 179). The replacement is safe: the new copy is made completely beside the entry first, the old entry is moved aside, the new one is
    renamed into place, and only then is the old one deleted. When the rename fails the old entry is put back. Returns whether the cache now holds
    the new copy; a failure costs a repeat of a CPU run and nothing else."""
    from orchard import cache_lock
    dest = cache / name
    tmp, old = cache / f".{name}.tmp-{os.getpid()}", cache / f".{name}.old-{os.getpid()}"
    moved = False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        _discard(tmp)
        _discard(old)
        shutil.copytree(src, tmp, symlinks=False)                  # complete beside the entry, before anything is moved
        # The exclusive lock covers the two renames. A run that validates and copies the entry holds the shared lock (audio_stages.copy_cached_run), so it
        # sees the old entry or the new one and never the gap between the renames.
        with cache_lock.entry_lock(cache, name, exclusive=True):
            if os.path.lexists(dest):
                os.rename(dest, old)
                moved = True
            try:
                os.rename(tmp, dest)
            except OSError:
                if moved:
                    os.rename(old, dest)
                raise
        _discard(old)
        return True
    except OSError:
        _discard(tmp)
        return False


def operator_home() -> Path:
    """The home OpenSSH uses: the passwd entry's, which ignores HOME."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def boot_unfinished(entries: list[dict]) -> bool:
    """The last coder start ("coder starting", or "coder adopted" for a kept coder) has no "coder
    started" after it."""
    last = None
    for e in entries:
        if e["event"] == "decision" and e["data"].get("decision") in ("coder starting", "coder adopted",
                                                                       "coder started"):
            last = e["data"]["decision"]
    return last in ("coder starting", "coder adopted")


def hardware_test_failure(stage_dir: Path) -> dict | None:
    """None when the stage's hardware test succeeded: test-result.json holds an exit code of
    exactly 0 and `timed_out` is false. Otherwise a dict with `returncode`, `timed_out` and a
    plain `problem` sentence. A missing or unreadable record counts as a failure, because nothing
    shows the test succeeded."""
    path = stage_dir / "test-result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"returncode": None, "timed_out": None,
                "problem": f"test-result.json could not be read: {exc}"}
    if not isinstance(data, dict):
        return {"returncode": None, "timed_out": None,
                "problem": "test-result.json does not hold a JSON object"}
    code, timed_out = data.get("returncode"), data.get("timed_out")
    # bool is a subclass of int in Python, so False would otherwise compare equal to 0.
    if type(code) is int and code == 0 and timed_out is False:
        return None
    if timed_out:
        problem = "the hardware test ran past its deadline and was stopped"
    elif code is None:
        problem = "the hardware test has no exit code (it was refused or did not start)"
    else:
        problem = f"the hardware test exited with code {code!r}"
    return {"returncode": code, "timed_out": timed_out, "problem": problem}


GATE_FEEDBACK_HEAD = "The supervisor checked your stage's output and it does not pass yet."
WRAPUP_HEAD = "Your turn budget for this step is used up."


def deliverable(spec, phase: str) -> str | None:
    """The file a step of this phase must leave in the stage directory."""
    if phase == "prepare":
        return "hw_tests.json" if spec.tests else "hw_test.json"
    return spec.gate_file


def file_stamp(path: Path) -> tuple | None:
    """(size, mtime_ns) of a file, or None when it does not exist."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def evidence_files(run_dir: Path, stage_dir: Path) -> list[str]:
    """Every file under the stage's evidence/ directory, relative to the run directory."""
    ev = stage_dir / "evidence"
    if not ev.is_dir():
        return []
    return sorted(os.path.relpath(f, run_dir) for f in ev.rglob("*") if f.is_file())


def wrapup_text(spec, name: str, files: list[str]) -> str:
    """The user message of a wrap-up: the missing file, the evidence, and the rules."""
    n = spec.number
    lines = [WRAPUP_HEAD, f"This step has not written stages/{n}/{name}. These evidence files are "
                          "in your stage directory:"]
    lines += [f"- {f}" for f in files]
    lines += [f"Write stages/{n}/{name} now, from these files, in the shape the skill describes. "
              "No new investigation is allowed: do not search, list or read anything except the "
              f"files above. You have {AGENT_WRAPUP_TURNS} turns. Where the evidence does not cover "
              "a field, say so in that field. When the file is written, reply with a short summary "
              "and no tool call."]
    return "\n".join(lines)


def gate_feedback_text(spec, reasons) -> str:
    """The user message a continuation starts from: the gate's reasons, word for word, and the
    file the gate reads."""
    n, name = spec.number, spec.gate_file
    lines = [GATE_FEEDBACK_HEAD, f"The check reads stages/{n}/{name} and found these problems:"]
    lines += [f"- {r}" for r in reasons]
    lines += [f"Fix each problem. Write {name} in your stage directory (stages/{n}/{name}) as the "
              f"skill describes, with the files that back it under stages/{n}/evidence/. When it is "
              "done, reply with a short summary that names the output files you wrote, and no tool "
              "call."]
    return "\n".join(lines)


def quiet_substitute(cfg, coder: str | None, asked: str, used: str) -> bool:
    """A substitution the layout means: the step asked for a tier that is never started (Coder-Next's
    large tier) and the coder serves it with the same model. Recording it on every stage told the
    operator nothing; any other substitution is still recorded."""
    return (coder is not None and used == coder and used != asked
            and cfg.tiers[used]["model"] == cfg.tiers[asked]["model"])


def coder_tier(cfg, port: int) -> str:
    """The chip tier whose endpoint uses the coder's port."""
    names = [n for n, t in cfg.tiers.items()
             if t["placement"] == "chips" and urlparse(t["endpoint"]).port == port]
    if len(names) != 1:
        raise ValueError(f"exactly one chip tier must use port {port}; found {names}")
    return names[0]


# ---- the supervisor -----------------------------------------------------------------------------

BUNDLE_BUILD_TIMEOUT_S = 600            # build_bundle.py reads files and copies packages; no network
SWAP_TEMPLATES = ("prepare_swap.py", "serve_and_compare.py", "prompt_suite.json")      # copied with a drafted swap_config.json
STAGE4_TEMPLATES = (*SWAP_TEMPLATES, "serve_and_compare_container.py")     # into each drafted configs/<N>
GRAFT_TRIAL_ENV = "ORCHARD_TRIAL_DIR"      # in the environment of the trial's test: the trial directory, so its servers can be found
GRAFT_TRIAL_TEMPLATE = "graft_trial.py"     # copied into stages/2/graft-trial/ when the trial starts (orchard/graft_trial.py)
SAMPLING_TRIAL_TEMPLATE = "sampling_trial.py"   # copied into stages/2/sampling-trial/ when the drafter-off trial starts (orchard/sampling_choice.py)


class Supervisor:
    def __init__(self, *, run_dir, ledger, cfg, model_id: str, adapter, coder, coder_chips: int,
                 standin, skills_dirs, inputs: dict | None = None, extra_env: dict | None = None,
                 versions: dict | None = None, http=post_json, probe=probe_model, clock=time.time,
                 sleep=time.sleep, budgets: Budgets = Budgets(), disk_usage=shutil.disk_usage,
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None, home=None, containers=None,
                 paths: RunPaths | None = None, package: dict | None = None,
                 package_added: bool = False, unattended: bool = False, lab=None,
                 four_chip_package: str | None = None, coder_tier: str | None = None,
                 keep_up: bool = False, coder_identity: dict | None = None, graft_trial: bool = True,
                 resume_aborted: bool = False, sampling: str = "auto"):
        self.run_dir = Path(run_dir).resolve()
        # Whether the model's purpose needs sampling: "auto" reads it from stage 0's files, the others are the operator's word (orchard/sampling_choice.py).
        self.sampling = sampling
        # An abort is terminal unless the operator asks for this (`--resume-aborted`): then an operator's or a signal's abort is
        # followed by a `resume` decision marked `after: abort`, and the run goes on from the ledger (orchard/stages.py, abort_can_resume).
        self.resume_aborted = resume_aborted
        # Stage 2 serves a grafted drafter once beside the safe configuration when stage 0 found a candidate; False skips that test.
        self.graft_trial = graft_trial
        # Lab mode only: in local mode tests park the coder for its board, so a kept coder saves little.
        self.keep_up = keep_up and lab is not None
        self.coder_identity = coder_identity      # what kept_coder.mismatch compares
        self.coder_tier = coder_tier                # the tier whose endpoint is the coder's port
        self.ledger, self.cfg, self.model_id = ledger, cfg, model_id
        self.adapter, self.coder, self.coder_chips, self.standin = adapter, coder, coder_chips, standin
        # With a lab box (orchard/labclient.py), every hardware test runs there under the lab's own
        # leases, and the coder on this box is never parked. Without one, tests share this box.
        self.lab = lab
        self.test_adapter = lab.adapter if lab is not None else adapter
        self.skills_dirs = [Path(d) for d in skills_dirs]
        self.inputs, self.extra_env = dict(inputs or {}), dict(extra_env or {})
        self.versions = dict(versions or {})
        self.http, self.probe, self.clock, self.sleep = http, probe, clock, sleep
        self.budgets, self.disk_usage = budgets, disk_usage
        self.credentials_visible = list(credentials_visible or [])
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.four_chip_package = four_chip_package      # the container stage 4's 4-chip draft uses, if several fit
        self.package = dict(package) if package else None    # stage 7: format, namespace, models_root
        self.package_added = package_added    # given on a resume of a run that started without them
        self.unattended = unattended          # a pause becomes a named block, never a wait (orchard/blocked.py)
        self.home = Path(home) if home is not None else operator_home()   # whose shared caches to refuse
        self.containers = containers if containers is not None else LabelledContainers()
        # The machine paths the skills' placeholders name (orchard/paths.py).
        self.paths = paths if paths is not None else RunPaths.resolve(self.run_dir, home=self.home)
        # The docker label every test container of this run carries (ORCHARD_TEST_LABEL).
        self.run_label = "orchard.test=" + hashlib.sha256(str(self.run_dir).encode()).hexdigest()[:12]
        self.surviving_containers: list = []         # set by _sweep when a labelled container outlives docker stop and rm
        self.control = Control(self.run_dir)
        self.actuator = RunActuator(self.control)
        self.guard = RetryGuard()
        self.coder_lease: Lease | None = None
        self.test_lease: Lease | None = None      # a stage's own test lease, while it is held
        self.watchdog: Watchdog | None = None
        self.bus_read = None                       # the reader of the PCIe bus; None means orchard.bus's own (tests set one)
        self.stop_message = ""                     # what the shutdown did, for main to print
        agent_env(self.run_dir, extra=self.extra_env)     # refuse a credential before anything runs

    # ---- small helpers ----------------------------------------------------------------------

    def _block(self, stage, reason: str, **evidence):
        self.ledger.append("notice", stage, blocked=True, reason=reason, evidence=evidence)
        raise Blocked(reason, **evidence)

    def _off_bus(self, held) -> str | None:
        """A sentence naming each chip this run holds that has left the PCIe bus (or cannot be read) and why, or None when the
        chips are fine or the bus cannot be listed. Reads sysfs only, takes no lease and never opens a chip (orchard/bus.py).
        It re-reads for bus.SETTLE_S seconds first, because a board being reset is unreadable for part of a minute.
        The coder always runs on this box, lab run or not, so its chips are read here. A lab's own tests run on the lab's chips, which
        are not on this bus: `_spawn` does not call this for them."""
        held = [c for c in (held or ()) if c]
        return bus.diagnose(held=held or None, needed=0 if held else int(self.coder_chips), read=self.bus_read,
                            sleep=self.sleep, clock=self.clock)

    def _block_if_off_bus(self, stage, reason: str, held, **evidence) -> None:
        """After a failure described by `reason`: block when a chip this run holds has left the bus, and return when none has.
        A dead board is not a model failure and a retry would meet the same board, so that block says so, names the chip and
        cause, and keeps `reason` as what came first."""
        dead = self._off_bus(held)
        if dead:
            self._block(stage, f"{dead}. It was found after this failure: {reason}", off_bus=True, **evidence)

    def _restore_coder(self, stage, h) -> None:
        """Bring the parked coder back (Handoff.restore). A failure is checked against the bus first, as a failed coder boot is
        (_block_if_off_bus): the board that was reset may have left the bus, and a retry would meet the same board."""
        try:
            h.restore()
        except Blocked as exc:
            self._block_if_off_bus(stage, exc.reason, h.lease.chips)
            raise

    def _evidence_file(self, stem: str, text: str) -> Path:
        d = self.run_dir / "evidence"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stem}-{len(self.ledger.read()) + 1:05d}.txt"
        with open(path, "x", encoding="utf-8") as f:           # one file per attempt
            f.write(text)
        return path

    def _handoff(self, stage, lease: Lease) -> Handoff:
        return Handoff(ledger=self.ledger, stage=stage, adapter=self.adapter, server=self.coder,
                       standin=self.standin, lease=lease, canary_prompt=RUN_CANARY_PROMPT,
                       note_path=self.run_dir / "stages" / str(stage) / "handoff.json",
                       evidence_dir=self.run_dir / "handoff", budgets=self.budgets,
                       clock=self.clock, sleep=self.sleep)

    # ---- the run ----------------------------------------------------------------------------

    def run(self) -> int:
        """Run until ready, aborted or stopped. Every way out of the loop that this process
        survives releases the hardware first.

        The release sits in `except` clauses. A bare `finally` would also run on the tests'
        Crash, which stands for a SIGKILL, after which no code runs. Releasing there would hide
        the crash recovery that the kill test checks."""
        write_pid_file(self.run_dir)    # first, so `status` sees this process as soon as it starts
        try:
            with self._signals():
                try:
                    return self._run()
                except Interrupted as exc:
                    return self._stop_on_signal(exc.name)
                except KeyboardInterrupt:              # SIGINT when the handler is not installed
                    return self._stop_on_signal("SIGINT")
                except Exception as exc:
                    self._stop_on_error(exc)
                    raise
        finally:
            # Closing the lab connection is safe on every way out, a crash included: a SIGKILL drops
            # the ssh connection, and the lab helper then does what close() asks of it (stop its tests,
            # give back the leases it holds).
            if self.lab is not None:
                self.lab.close()

    @contextlib.contextmanager
    def _signals(self):
        """SIGINT and SIGTERM raise Interrupted while the run is in progress. Handlers can only be
        installed from the main thread; elsewhere the run keeps the process's handlers."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        old = {s: signal.getsignal(s) for s in STOP_SIGNALS}
        for s in STOP_SIGNALS:
            signal.signal(s, _raise_interrupted)
        try:
            yield
        finally:
            for s, handler in old.items():
                signal.signal(s, handler)

    def _ignore_signals(self) -> None:
        """A second Ctrl-C during the shutdown would leave the hardware half released, so the
        shutdown ignores it. `_signals` puts the old handlers back when the run returns."""
        if threading.current_thread() is threading.main_thread():
            for s in STOP_SIGNALS:
                signal.signal(s, signal.SIG_IGN)

    def _stop_on_signal(self, name: str) -> int:
        self._ignore_signals()
        print(f"{name}: stopping the coder and releasing the hardware; this can take a few "
              "minutes", file=sys.stderr)
        try:
            # Recorded first, as in _abort: a crash during the release still leaves an aborted
            # run, and the next start finishes the release and refuses to go on.
            self.ledger.append("decision", None, decision="abort", by=f"signal {name}")
            ok = self._release_everything()
        except Exception:
            ok = self._release_from_memory()
        self.stop_message = f"Aborted by {name}. " + self._release_summary(ok)
        return EXIT_ABORTED

    def _stop_on_error(self, exc: Exception) -> None:
        self._ignore_signals()
        try:
            self.ledger.append("notice", None, what="the supervisor stopped on an error",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            ok = self._release_everything()
        except Exception:
            ok = self._release_from_memory()
        self.stop_message = self._release_summary(ok) + (
            " Fix the cause, then run the same command again to resume." if ok else "")

    def _release_everything(self) -> bool:
        """Release the test lease and then the coder's, and report whether both went back. Both are tried whatever the first returns: a
        test lease kept under a surviving container (_release_test_lease) must not leave the coder, which has a board of its own,
        serving under the exiting supervisor's lease."""
        test_lease_back = self._release_test_lease()
        coder_back = self._release_all()
        return test_lease_back and coder_back

    @staticmethod
    def _release_summary(ok: bool) -> str:
        if ok:
            return "The coder was stopped and every lease this run held was released."
        return ("Releasing the hardware failed, so the chips may still be held. Read the last "
                "notice in the ledger and `gozer status` before doing anything else.")

    def _release_test_lease(self) -> bool:
        """Release a stage's own test lease if one is held. The test command is already dead:
        spawn_checked kills its session on the way out."""
        lease = self.test_lease
        if lease is None:
            return True
        if self.surviving_containers:
            # A container the sweep could not remove still has the chips open. Releasing the lease would reset them under it and
            # let another agent in, so the lease stays and the ledger says why. An operator may have cleaned up since (an attended
            # run resumes after a pause), so docker is asked again, and the remembered state is cleared only when it answers with
            # an empty list; an answer that cannot be had keeps the lease.
            now = self.containers.list(self.run_label)
            if now:
                self.surviving_containers = list(now)
            if now is None or now:
                self.ledger.append("notice", None, what="the test lease was kept: a test container is still running"
                                   if now else "the test lease was kept: docker could not be asked about the test containers",
                                   lease_id=lease.lease_id, containers=list(self.surviving_containers))
                return False
            self._cleared(None, list(self.surviving_containers))
        try:
            self.test_adapter.release(lease)
        except AdapterError as exc:
            self.ledger.append("notice", None, what="releasing the test lease failed",
                               lease_id=lease.lease_id, error=str(exc))
            return False
        self.test_lease = None
        self.ledger.append("decision", None, decision="test lease released", lease_id=lease.lease_id)
        return True

    def _release_from_memory(self) -> bool:
        """The release when the ledger cannot be read or written: use the leases this process
        remembers. Nothing is recorded, because the ledger is what failed."""
        ok = True
        for lease, is_coder in ((self.test_lease, False), (self.coder_lease, True)):
            if lease is None:
                continue
            if not is_coder and self.surviving_containers:
                ok = False             # a container still uses these chips: its lease stays, the coder's is still released
                continue
            try:
                if is_coder and getattr(self.coder, "owned", None) is False:
                    # Another process owns the container: no stop, and no reset of its chips.
                    self.adapter.release(lease, keep_running=True)
                    continue
                if is_coder and not self.coder.confirm_stopped().stopped:
                    self.coder.stop()
                if is_coder:
                    check = wait_stopped(self.coder, self.adapter, lease,
                                         quiet_wait_s=self.budgets.quiet_wait_s,
                                         poll_s=self.budgets.poll_s, clock=self.clock,
                                         sleep=self.sleep, accept=("CLAIMED", "STALE", "FREE"))
                    if not check["ok"]:
                        ok = False
                        continue
                (self.adapter if is_coder else self.test_adapter).release(lease)
            except (AdapterError, ServerError, OSError):
                ok = False
        return ok

    def _run(self) -> int:
        p = run_progress(self.ledger.read())
        if p.aborted and self.resume_aborted and abort_can_resume(self.ledger.read()):
            # The operator asked to carry on after an abort. Recorded once, before any work; replay then clears the abort and any
            # pause the run was waiting in, and the run starts its coder again as after any stop.
            self.ledger.append("decision", None, decision="resume", by="retry", after="abort")
            self.actuator.clear()
            p = run_progress(self.ledger.read())
        if p.aborted or p.finished:
            if p.aborted and p.started:
                # The abort may have kept a test lease under a container that was still running. Sweep and give that lease back now if
                # the container is gone; the run stays aborted and no stage starts, whatever the sweep finds.
                try:
                    self._recover_test_leases()
                except Blocked:
                    pass                  # the notice is in the ledger; the lease stays held
            self._release_all()           # finishes a release a crash interrupted; else does nothing
            return EXIT_ABORTED if p.aborted else EXIT_READY
        if not p.started:
            self.ledger.append("run_start", None, model=self.model_id, versions=self.versions,
                               inputs=self.inputs, required_chips=list(self.required_chips or ()) or None,
                               graft_trial=self.graft_trial, sampling=self.sampling,
                               paths=self.paths.record(), package=self.package,
                               coder=self.coder.record(),
                               tiers={k: dict(v) for k, v in self.cfg.tiers.items()},
                               mode="lab" if self.lab is not None else "local",
                               **({"lab": self.lab.info} if self.lab is not None else {}))
        elif recorded_paths(self.ledger.read()) is None:
            # The run started under a supervisor that did not record paths. Record them once now,
            # so every later resume is held to the same values.
            self.ledger.append("decision", None, decision=PATHS_RECORDED, paths=self.paths.record(),
                               note="the run_start entry predates the paths record")
        while p.started:
            try:
                self._recover_test_leases()
                break
            except Blocked as exc:
                # A container the earlier supervisor left is still running (_sweep wrote the notice). Nothing starts over it. An
                # unattended run ends blocked (running the command again sweeps once more); an attended run waits for the operator
                # and, on `resume`, goes through the recovery again, because a resume that went straight to the stages would skip
                # the second sweep and the lease the first attempt never gave back.
                self.ledger.append("decision", None, decision="pause", reason=f"blocked: {exc.reason}")
                if self.unattended:
                    return self._end_blocked(block_code(last_pause(self.ledger.read())) or "blocked", str(exc.reason))
                if self._wait_for_operator() == "abort":
                    return self._abort()
        if self.package_added and package_options(self.ledger.read()) != self.package:
            # Options given on a resume of a run that started without them (build checked that
            # stage 7 has not started). Recorded once; a later resume reads them from here.
            self.ledger.append("decision", None, decision=PACKAGE_OPTIONS_SET, package=self.package)
        if self.credentials_visible:
            # Recorded at every start, because each start is a fresh acceptance by the operator.
            self.ledger.append("decision", None, decision="operator accepted visible credentials",
                               paths=self.credentials_visible, by="--accept-credentials-visible")
        if self.unattended:
            p = run_progress(self.ledger.read())
            if p.paused is not None:
                # Running the command again is the operator's retry of an earlier block. Whatever it was,
                # the checks run again from the ledger, and a block that cannot be fixed blocks again.
                self.ledger.append("decision", None, decision="resume", by="retry")
                self.actuator.clear()
        while True:
            p = run_progress(self.ledger.read())
            if p.paused is not None:
                if self.unattended:
                    pause = last_pause(self.ledger.read())
                    code = block_code(pause) if pause is not None else None
                    if code is not None:
                        return self._end_blocked(code, str(pause.get("reason") or "paused"))
                if self._wait_for_operator() == "abort":
                    return self._abort()
                continue
            if p.next_stage is None:
                break
            cap = budget_cap(p, self.clock())
            if cap:
                self.ledger.append("decision", p.next_stage, decision="pause", reason=cap)
                continue
            if p.next_stage == 1 and self._select_audio_runtime() == "paused":
                continue
            if p.next_stage == 1 and self._audio_unacknowledged():
                self.ledger.append("decision", 1, decision="pause", reason=AUDIO_UNBUILT)
                continue
            if p.next_stage == 1 and self._restricted_head_unacknowledged():
                self.ledger.append("decision", 1, decision="pause", reason=RESTRICTED_HEAD)
                continue
            if p.next_stage == 2 and self._full_port_unacknowledged():
                self.ledger.append("decision", 2, decision="pause", reason=FULL_PORT)
                continue
            # The spec follows the path stage 0 chose, read from the ledger each time, so a
            # resumed run picks the same skill and gate as the run that crashed.
            entries = self.ledger.read()
            spec = spec_for(p.next_stage, run_path(entries, self.run_dir), package_format(entries),
                            run_class(entries, self.run_dir), stock=stock_params(entries) is not None,
                            audio_runtime=audio_runtime.selected(entries) is not None)
            try:
                if spec.needs_coder or self._coder_in_use():
                    self._ensure_coder()
                result = self._run_stage(spec, resuming=p.open_stage == p.next_stage,
                                         escalated=p.next_stage in p.escalated)
            except Blocked as exc:
                # The notice is already in the ledger. The run waits for the operator.
                self.ledger.append("decision", p.next_stage, decision="pause",
                                   reason=f"blocked: {exc.reason}")
                continue
            if result == "abort":
                return self._abort()
        self.ledger.append("decision", None, decision="ready for operator review",
                           bundle="stages/8/bundle")
        # Nothing owns the coder once this process exits. gozer would then show a lease whose owner
        # pid is dead while the container still has the devices open. So the run gives the
        # hardware back, or (keep_up) hands the coder's chips to a lease its container owns.
        self._release_or_keep()
        return EXIT_READY

    def _full_port_unacknowledged(self) -> bool:
        """Stage 0 chose a full port, and the run has not yet paused for it."""
        entries = self.ledger.read()
        if run_path(entries, self.run_dir) != "full-port":
            return False
        # Unattended, a full port always blocks: an operator's resume is what lets an attended run go on
        # into the full-port stages, and no unattended retry may do that. A pause for a restricted head
        # counts as the same acknowledgement, so one resume is enough.
        return self.unattended or not any(
            e["event"] == "decision" and e["data"].get("reason") in (FULL_PORT, RESTRICTED_HEAD) for e in entries)

    def _audio_unacknowledged(self) -> bool:
        """Stage 0 recorded modality audio, and the run has not yet paused for it. The audio stages are not built (plan
        docs/superpowers/plans/2026-10-28-audio-bringup.md), so the run stops before stage 1. Unattended, it always blocks."""
        entries = self.ledger.read()
        facts = run_facts(entries, self.run_dir)
        if not facts or facts.get("modality") != "audio":
            return False
        if audio_runtime.selected(entries) is not None:
            return False                  # the harness has a runtime for this model, so the audio path is built for it
        return self.unattended or not any(
            e["event"] == "decision" and e["data"].get("reason") == AUDIO_UNBUILT for e in entries)

    def _select_audio_runtime(self) -> str:
        """Stage 1 is next. When stage 0 found an audio model and the harness's own runtime implements it (orchard/audio_runtime.py), record that
        once ("selected"), so a resume keeps the choice. "paused" when the runtime exists and its environment is missing (a pause decision was
        written). "none" in every other case, and the audio stop applies as before."""
        entries = self.ledger.read()
        if audio_runtime.selected(entries) is not None:
            return "selected"
        facts = run_facts(entries, self.run_dir)
        model_dir = self.inputs.get("model")
        from orchard import audio_plan
        if (not facts or facts.get("modality") != "audio" or not model_dir or audio_plan.library_name(self.model_id) is None
                or audio_runtime.runtime_problems(model_dir)):
            return "none"          # the planners know two repositories; another one with a supported configuration is not selected
        paths = audio_runtime.audio_paths(self.paths.operator_home)
        missing = audio_runtime.environment_problems(paths)
        if missing:
            self.ledger.append("decision", 1, decision="pause", reason=AUDIO_ENV_MISSING + "; ".join(missing))
            return "paused"
        self.ledger.append("decision", 1, decision=audio_runtime.AUDIO_RUNTIME_SELECTED, runtime=audio_runtime.RUNTIME, paths=paths.record(),
                           scripts=audio_runtime.script_hashes(), model=self.model_id)
        return "selected"

    def _coder_in_use(self) -> bool:
        """The ledger holds a coder lease this run has not released: a coder is part of the run even when the next stage does not need it, so
        it is adopted (or found dead and handled) as before."""
        entries = self.ledger.read()
        lease_rec, _, _ = coder_state(entries)
        if lease_rec is None:
            return False
        released = {e["data"].get("lease_id") for e in entries
                    if e["event"] == "decision" and e["data"].get("decision") in ("hardware released", "coder left up")}
        return lease_rec["lease_id"] not in released

    def _restricted_head_unacknowledged(self) -> bool:
        """Stage 0 filed the model as a restricted head, and the run has not yet paused for it. The harness does not serve
        such a model (the operator's ruling, 2026-10-19), so the run stops before stage 1 instead of spending a coder boot and
        a CPU reference on it. Unattended, it always blocks."""
        entries = self.ledger.read()
        if run_class(entries, self.run_dir) != "restricted-head":
            return False
        return self.unattended or not any(
            e["event"] == "decision" and e["data"].get("reason") == RESTRICTED_HEAD for e in entries)

    def _wait_for_operator(self) -> str:
        while True:
            cmd = self.control.take()
            if cmd == "resume":
                self.ledger.append("decision", None, decision="resume", by="operator")
                self.actuator.clear()
                return "resume"
            if cmd == "abort":
                return "abort"
            self.sleep(CONTROL_POLL_S)

    def _end_blocked(self, code: str, reason: str) -> int:
        """End an unattended run that cannot go on: name why, write the bundle, give the hardware back.
        The decision is written first, so a crash during the release still leaves a blocked run, and
        running the command again retries."""
        entries = self.ledger.read()
        self.ledger.append("decision", None, decision="blocked", code=code, reason=reason)
        write_blocked_bundle(self.run_dir, entries, code, reason, now=self.clock())
        # A stage can block while its own test lease is held (a refused reset, for one). That lease goes back first,
        # or the automatic retry would run against chips this process still held.
        self._release_test_lease()
        self._release_or_keep()
        return EXIT_BLOCKED

    def _abort(self) -> int:
        # Recorded first: a crash during the release still leaves an aborted run, and a restart
        # finishes the release and refuses to go on.
        self.ledger.append("decision", None, decision="abort", by="operator")
        self._release_everything()                 # a test lease kept at a pause (a further board) goes back with the coder's
        return EXIT_ABORTED

    def _release_all(self) -> bool:
        """Stop the coder and release its lease. Safe to call again: a released lease is skipped.
        Returns False when the lease could not be released."""
        entries = self.ledger.read()
        lease_rec, server_rec, _ = coder_state(entries)
        released = {e["data"].get("lease_id") for e in entries
                    if e["event"] == "decision" and e["data"].get("decision") in ("hardware released", "coder left up")}
        if lease_rec is None or lease_rec["lease_id"] in released:
            return True
        lease = Lease.from_record(lease_rec)
        adopted = lease.lease_id in self._adopted_ids(entries)
        if server_rec:
            self.coder.adopt(server_rec)
        if getattr(self.coder, "owned", None) is False:
            return self._release_foreign_lease(lease, adopted)
        try:
            if not self.coder.confirm_stopped().stopped:
                self.ledger.append("decision", None, decision="stopping the coder",
                                   result=self.coder.stop())
            check = wait_stopped(self.coder, self.adapter, lease,
                                 quiet_wait_s=self.budgets.quiet_wait_s, poll_s=self.budgets.poll_s,
                                 clock=self.clock, sleep=self.sleep, accept=("CLAIMED", "STALE", "FREE"))
            if not check["ok"]:
                self.ledger.append("notice", None, blocked=True,
                                   reason="the coder is not confirmed stopped; the lease was not "
                                          "released", evidence=check)
                return False
            self.adapter.release(lease)
            self.ledger.append("decision", None, decision="hardware released", lease_id=lease.lease_id)
            if adopted:
                kept_coder.forget(self.paths.cache_root, lease.lease_id)
            return True
        except (AdapterError, ServerError, OSError) as exc:
            self.ledger.append("notice", None, what="releasing the hardware failed", error=str(exc))
            return False

    def _release_foreign_lease(self, lease: Lease, adopted: bool) -> bool:
        """The coder's container belongs to another process (the start was refused). Nothing is
        stopped. The lease goes back without a reset, because that container may use these chips."""
        self.ledger.append("notice", None, what="the coder container belongs to another process; it was "
                           "not stopped and the lease was released without a reset", lease_id=lease.lease_id)
        try:
            self.adapter.release(lease, keep_running=True)
        except (AdapterError, OSError) as exc:
            self.ledger.append("notice", None, what="releasing the hardware failed", error=str(exc))
            return False
        self.ledger.append("decision", None, decision="hardware released", lease_id=lease.lease_id,
                           reset=False)
        if adopted:
            kept_coder.forget(self.paths.cache_root, lease.lease_id)
        return True

    # ---- a coder kept up between lab runs (orchard/kept_coder.py) ------------------------------

    @staticmethod
    def _adopted_ids(entries: list[dict]) -> set[str]:
        """Coder leases that a container owns (gozer adopt), not this supervisor."""
        ids = set()
        for e in entries:
            d = e["data"]
            if e["event"] == "decision" and d.get("decision") == "coder adopted":
                ids.add((d.get("lease") or {}).get("lease_id"))
            if e["event"] == "decision" and d.get("decision") == "coder left up":
                ids.add(d.get("kept_lease_id"))
        return ids - {None}

    def _release_or_keep(self) -> bool:
        if self.keep_up:
            kept = self._leave_coder_up()
            if kept is not None:
                return kept
        return self._release_all()

    def _leave_coder_up(self) -> bool | None:
        """Hand a healthy coder's chips to a lease its container owns, and write the record the next run
        adopts. None when the coder cannot be kept (the caller releases as usual)."""
        entries = self.ledger.read()
        lease_rec, server_rec, canary = coder_state(entries)
        released = {e["data"].get("lease_id") for e in entries if e["event"] == "decision"
                    and e["data"].get("decision") in ("hardware released", "coder left up")}
        if (lease_rec is None or lease_rec["lease_id"] in released or boot_unfinished(entries)
                or not isinstance(canary, dict) or not hasattr(self.adapter, "adopt")):
            return None
        lease = Lease.from_record(lease_rec)
        if len(lease.units) != 1:
            self.ledger.append("notice", None, what="the coder was not kept up",
                               reason=f"it holds {len(lease.units)} lease units; a kept coder must be on one")
            return None
        if server_rec:
            self.coder.adopt(server_rec)
        try:
            if self.coder.confirm_stopped().stopped:
                return None
            canary_text = (self.run_dir / canary["path"]).read_text(encoding="utf-8")
        except (ServerError, OSError):
            return None
        holders: list[int] = []
        kept_id = lease.lease_id
        if lease.lease_id not in self._adopted_ids(entries):
            self.ledger.append("decision", None, decision="handing the coder's chips to its container",
                               lease_id=lease.lease_id)
            try:
                self.adapter.release(lease, keep_running=True)
            except AdapterError as exc:
                self.ledger.append("notice", None, what="the coder was not kept up", reason=str(exc))
                return None
            try:
                kept_id, holders = self.adapter.adopt(lease.units[0], who=kept_coder.KEPT_WHO,
                                                      reason="coder kept up for the next run")
            except AdapterError as exc:
                # The supervisor's lease is gone and nothing owns the chips: stop the coder rather than
                # leave it serving on chips gozer could grant to someone else.
                self.ledger.append("notice", None, what="the coder was not kept up; stopping it",
                                   reason=str(exc), result=self.coder.stop())
                self.ledger.append("decision", None, decision="hardware released", lease_id=lease.lease_id,
                                   reset=False)
                return False
        else:
            holders = sorted({p for c in self.adapter.status() if c.bdf in lease.chips
                              for p in (c.lease_pid, *c.pids_holding) if p})
        kept = dataclasses.replace(lease, lease_id=kept_id)
        kept_coder.write(self.paths.cache_root, {
            "identity": self.coder_identity, "lease": kept.record(), "holder_pids": holders,
            "server": self.coder.record(), "canary": canary_text, "run_dir": str(self.run_dir),
            "left_at": self.clock()})
        self.ledger.append("decision", None, decision="coder left up", lease_id=lease.lease_id,
                           kept_lease_id=kept_id, holder_pids=holders,
                           record=str(kept_coder.path(self.paths.cache_root)))
        return True

    def _adopt_kept_coder(self) -> bool:
        """Use the coder an earlier run left up, if every check holds (orchard/kept_coder.py). It never
        stops anything: a coder that fails a check is left for the operator (`tt-orchard coder stop`)."""
        kept = kept_coder.read(self.paths.cache_root)
        if kept is None:
            return False
        why = kept_coder.mismatch(kept, self.coder_identity)
        lease = None
        if why is None:
            try:
                lease = Lease.from_record(kept["lease"])
            except (KeyError, TypeError, ValueError):
                why = "its record has no usable lease"
        if why is None:
            t = self.cfg.tiers[self.coder_tier]
            if not self.probe(t["endpoint"], t["model"]):
                why = f"nothing lists {t['model']} at {t['endpoint']}"
        if why is None:
            try:
                chips = {c.bdf: c for c in self.adapter.status()}
            except AdapterError as exc:
                why = f"gozer status failed: {exc}"
            else:
                holders = set(kept.get("holder_pids") or [])
                bad = [b for b in lease.chips if not (b in chips and chips[b].state == "HELD"
                                                      and chips[b].who == kept_coder.KEPT_WHO
                                                      and chips[b].lease_pid in holders)]
                if bad:
                    why = f"gozer does not show {', '.join(bad)} held by the kept coder"
        if why is not None:
            self.ledger.append("decision", None, decision="kept coder not used", reason=why,
                               from_run=kept.get("run_dir"))
            return False
        self.ledger.append("decision", None, decision="coder adopted", lease=lease.record(),
                           server=kept["server"], from_run=kept.get("run_dir"), left_at=kept.get("left_at"))
        self.coder.adopt(kept["server"])
        self.coder_lease = lease
        baseline = evidence_record(self.run_dir, self._evidence_file("coder-canary-kept", str(kept.get("canary", ""))))
        self._finish_boot(lease, baseline)
        return True

    # ---- the coder ----------------------------------------------------------------------------

    def _ensure_coder(self) -> None:
        """The coder serves under a lease this process owns, before any stage runs."""
        entries = self.ledger.read()
        hp = progress(entries)
        if hp.phase != "idle":
            # A park or restore was interrupted (a crash, or a block the operator resumed).
            rec = recover(entries, adapter=self.adapter, server=self.coder, standin=self.standin)
            h = self._handoff(hp.stage, Lease.from_record(hp.lease))
            try:
                h.recover_and_restore(rec, chips=self.coder_chips, who=WHO, reason="coder",
                                      wait_budget_s=COLD_BOOT_BUDGET_S)
            except Blocked as exc:
                self._block_if_off_bus(hp.stage, exc.reason, h.lease.chips)
                raise
            if rec.action != "abandon":
                self.coder_lease = h.lease
                return
            entries = self.ledger.read()      # abandoned: the coder serves under the old lease
        lease_rec, server_rec, canary = coder_state(entries)
        if lease_rec is None:
            if self.keep_up and self._adopt_kept_coder():
                return
            self._start_coder(None)
            return
        if server_rec:
            self.coder.adopt(server_rec)
        lease = Lease.from_record(lease_rec)
        if getattr(self.coder, "owned", None) is False:
            # The last start was refused: the container of that name is another process's, even if
            # it answers. Give back any lease this run still holds (no reset) and try to start again;
            # a second refusal blocks with the same reason.
            released = {e["data"].get("lease_id") for e in entries if e["event"] == "decision"
                        and e["data"].get("decision") in ("hardware released", "coder left up")}
            if (lease.lease_id not in released
                    and not self._release_foreign_lease(lease, lease.lease_id in self._adopted_ids(entries))):
                self._block(None, f"the coder container belongs to another process, and this run's lease "
                                  f"{lease.lease_id} could not be released to try again")
            self._start_coder(canary)
            return
        mine = [c for c in self.adapter.status() if c.bdf in lease.chips]
        if lease.lease_id in self._adopted_ids(entries):
            # A kept coder's lease is owned by its container's processes, not by this supervisor.
            ours = mine and all(c.state == "HELD" and c.who == kept_coder.KEPT_WHO for c in mine)
        else:
            ours = mine and all(c.lease_pid == self.adapter.owner_pid for c in mine)
        if not ours:
            self._relaunch_coder(lease, canary)
            return
        self.coder_lease = lease
        if self.coder.confirm_stopped().stopped:
            self._coder_died(lease, canary)
        elif boot_unfinished(entries):
            # The last start never reached "coder started": it did not become ready, or its
            # canary did not match and the operator resumed. Check it again before any stage.
            self.ledger.append("decision", None, decision="finishing an unfinished coder boot",
                               lease_id=lease.lease_id)
            self._finish_boot(lease, canary)

    def _start_coder(self, baseline: dict | None, lease: Lease | None = None) -> None:
        if lease is None:
            lease = reacquire(self.adapter, chips=self.coder_chips, who=WHO, reason="coder",
                              ledger=self.ledger, stage=None, wait_budget_s=COLD_BOOT_BUDGET_S,
                              clock=self.clock, sleep=self.sleep)
        self.coder_lease = lease          # known from here on, so a shutdown can release it
        # Recorded before the start: a crash during the boot leaves the lease and server in the
        # ledger, so the restart can stop that server and re-lease.
        # The pending record says "unknown" for this attempt. The live value changes only when the
        # start begins (ServerControl.start), so a failed append here cannot lose an earlier refusal.
        self.ledger.append("decision", None, decision="coder starting", lease=lease.record(),
                           server={**self.coder.record(), "owned": None})
        try:
            self.coder.start(lease)
        except ServerNotOwned as exc:
            # Another process owns the container of this name. Record that, so no later stop (this
            # run's cleanup, a resume) touches it, and block with the usual coder reason.
            self.ledger.append("decision", None, decision="coder not started", lease_id=lease.lease_id,
                               server=self.coder.record(), reason=str(exc))
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
        except (ServerError, OSError) as exc:
            self._block_if_off_bus(None, f"the coder did not start and answer; nothing is retried: {exc}", lease.chips)
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
        # The boot takes up to 45 minutes and is the likeliest time for a kill. This entry gives
        # the kill test a kill point while the container runs and has not answered.
        self.ledger.append("decision", None, decision="coder container started",
                           lease_id=lease.lease_id, server=self.coder.record())
        self._finish_boot(lease, baseline)

    def _finish_boot(self, lease: Lease, baseline: dict | None) -> None:
        """Wait for the coder to answer, compare its canary with `baseline` (the answer recorded
        at an earlier start, if any) and record it as started. With no baseline (the first start of
        the run) it also asks FIRST_BOOT_PROMPT and blocks unless the answer contains
        FIRST_BOOT_EXPECTED; a resume after that block asks it again. Until "coder started" is in the
        ledger, a restart or a resume comes back here."""
        try:
            seconds = self.coder.wait_ready(self.budgets.cold_boot_s)
            answer = self.coder.ask(RUN_CANARY_PROMPT)
            # With no baseline this is the first start of the run. A server that answers with
            # noise still passes the canary, because the canary only needs some answer.
            sanity = self.coder.ask(FIRST_BOOT_PROMPT) if baseline is None else None
        except (ServerError, CanaryError, OSError) as exc:
            self._block_if_off_bus(None, f"the coder did not start and answer; nothing is retried: {exc}", lease.chips)
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
        if sanity is not None and FIRST_BOOT_EXPECTED not in sanity:
            saved = evidence_record(self.run_dir, self._evidence_file("coder-sanity", sanity))
            self._block(None, f"the coder's server answers wrongly: asked {FIRST_BOOT_PROMPT!r}, expected "
                              f"text containing {FIRST_BOOT_EXPECTED!r}, got {sanity[:200]!r}. It was not "
                              "retried; check the package and the server before resuming",
                        question=FIRST_BOOT_PROMPT, answer=sanity[:500], answer_file=saved)
        after = evidence_record(self.run_dir, self._evidence_file("coder-canary", answer))
        if baseline is not None:
            before = (self.run_dir / baseline["path"]).read_text(encoding="utf-8")
            result = compare(before, answer)
            if not result.match:
                self._block(None, "the coder's canary answer changed after it was started again",
                            whitespace_only=result.whitespace_only, before=before[:500],
                            after=answer[:500], before_file=baseline, after_file=after)
        self.ledger.append("decision", None, decision="coder started", lease=lease.record(),
                           server=self.coder.record(), canary=baseline or after, canary_after=after,
                           ready_s=seconds)
        self.coder_lease = lease

    def _relaunch_coder(self, old: Lease, canary: dict | None) -> None:
        """After a restart the coder's lease belongs to a dead pid, so nobody can reset it. Stop
        the coder, let the lease tool reap the orphan, take a new lease and start again."""
        self.ledger.append("decision", None, decision="relaunch the coder under this supervisor's lease",
                           old_lease_id=old.lease_id)
        if getattr(self.coder, "owned", None) is False:
            # The old start was refused: that container is another process's. Leave it, and start
            # again; a second refusal blocks with the same reason.
            self._start_coder(canary)
            return
        if not self.coder.confirm_stopped().stopped:
            self.ledger.append("decision", None, decision="stopping the coder", result=self.coder.stop())
        check = wait_stopped(self.coder, self.adapter, old, quiet_wait_s=self.budgets.quiet_wait_s,
                             poll_s=self.budgets.poll_s, clock=self.clock, sleep=self.sleep,
                             accept=("CLAIMED", "STALE", "FREE"))
        if not check["ok"]:
            self._block(None, "the coder could not be confirmed stopped for the relaunch", **check)
        if old.lease_id in self._adopted_ids(self.ledger.read()):
            self._release_kept_lease(old)
        self._start_coder(canary)

    def _release_kept_lease(self, lease: Lease) -> None:
        """A kept coder's lease stays (STALE) after its container stops; give it back before a new start."""
        try:
            self.adapter.release(lease)
        except AdapterError as exc:
            self._block(None, f"the kept coder's lease {lease.lease_id} could not be released: {exc}")
        self.ledger.append("decision", None, decision="hardware released", lease_id=lease.lease_id)
        kept_coder.forget(self.paths.cache_root, lease.lease_id)

    def _coder_died(self, lease: Lease, canary: dict | None) -> None:
        """Spec section 10: one restart with the same config and a canary check; a second death
        blocks."""
        if run_progress(self.ledger.read()).coder_deaths >= 1:
            self._block(None, "the coder died a second time since the last resume")
        self.ledger.append("decision", None, decision="coder died; restarting it once")
        if lease.lease_id in self._adopted_ids(self.ledger.read()):
            self._release_kept_lease(lease)    # the release resets the chips; then a lease of our own
            self._start_coder(canary)
            return
        try:
            self.adapter.reset(lease)          # a dead server can leave the chips dirty
        except AdapterError as exc:
            self._block(None, f"resetting the coder's chips after it died failed: {exc}")
        self._start_coder(canary, lease=lease)

    # ---- one stage ----------------------------------------------------------------------------

    def _watchdog(self, spec) -> Watchdog:
        wd = Watchdog([IdenticalResponses(), ThinkingWithoutAction(), RepeatedToolCall(),
                       NoNewEvidence(), NoFileWritten(), TurnRepeat(),
                       StageOverBudget({spec.number: spec.budget_s})],
                      Ladder(self.actuator, self.ledger, {AGENT}))
        t0 = attempt_started_ts(self.ledger.read(), spec.number) or self.clock()
        wd.feed(Event(ts=t0, agent="supervisor", kind="ledger", name="stage_start", stage=spec.number))
        return wd

    def _run_stage(self, spec, *, resuming: bool, escalated: bool) -> str:
        n = spec.number
        if spec.skip:
            self.ledger.append("stage_start", n, skip=True)
            self.ledger.append("stage_end", n, result="skipped", reason=spec.skip)
            return "skipped"
        # A resumed stage 2 whose sampling trial already ran only has results to read. The trial's own tensor cache may have taken the space the entry
        # requirement asks for, and nothing is converted again, so the requirement is for a stage that still has a cache to make.
        reads_finished_trial = (resuming and n == 2 and (self.run_dir / "stages" / "2" / sampling.TRIAL_DIR / "test-result.json").is_file())
        ok, free = (True, None) if reads_finished_trial else check_disk(self.run_dir, spec.disk_gb, usage=self.disk_usage)
        if not ok:
            self._block(n, f"stage {n} needs {spec.disk_gb} GB free on the run directory's disk; "
                           f"{free} GB is free", need_gb=spec.disk_gb, free_gb=free)
        marker = self.run_dir / "stages" / str(n) / "test-result.json"
        if resuming and escalated and spec.boards and marker.is_file() and self._test_failure(spec, marker.parent):
            # A kill came after the escalate entry and before the stage_end. The escalated attempt
            # must run a fresh prepare and hardware test: its finish step cannot pass on a test
            # that failed, and it gets no gate feedback for one. So the failed record is not a
            # resume point, and open_stage_dir moves the directory aside.
            self.ledger.append("decision", n, decision="not resuming from a failed hardware test",
                               marker=evidence_record(self.run_dir, marker))
            resuming = False
        stage_dir, resumed = open_stage_dir(self.run_dir, spec, resuming=resuming, ledger=self.ledger)
        self.ledger.append("stage_start", n, escalated=escalated, resumed=resumed)
        self.actuator.clear()
        self.watchdog = self._watchdog(spec)
        status, reasons, gate = self._stage_body(spec, stage_dir, escalated, resumed)
        return self._end_stage(spec, stage_dir, status, reasons, gate, escalated)

    def _gate(self, spec):
        """The stage's gate. Stage 4's also gets the run's required chip counts; no other gate
        changes its call."""
        if spec.number == 4 and self.required_chips:
            return functools.partial(spec.gate, required=self.required_chips)
        return spec.gate

    def _stage_body(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        self._wrapup_used = False           # at most one wrap-up per run of the stage body
        if spec.harness:
            if spec.number == 0:
                return self._stock_triage_body(spec, stage_dir)
            if spec.number == 8:
                return self._bundle_body(spec, stage_dir)
            if spec in (AUDIO_STAGE_1, AUDIO_STAGE_2):
                return self._audio_body(spec, stage_dir, resumed)
            return self._package_body(spec, stage_dir, resumed)
        if spec.boards == 0:
            out, step = self._step(spec, "run", stage_dir, escalated, resumed)
            out = self._wrap_up(spec, stage_dir, step, out)
            if out.status != "done":
                return out.status, [f"the agent step ended: {out.status} {out.detail}".strip()], None
        else:
            if spec.tests:
                ended = self._run_test_list(spec, stage_dir, escalated, resumed)
                if ended:
                    return ended
            elif not (resumed and (stage_dir / "test-result.json").is_file()):
                self._draft(spec, stage_dir)
                out, prep = self._step(spec, "prepare", stage_dir, escalated, resumed)
                out = self._wrap_up(spec, stage_dir, prep, out)
                if out.status != "done":
                    return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
                test, problems = self._read_test(stage_dir)
                if problems:
                    return "fail", problems, None
                # A container package's test leaves a container behind when its client is killed (a timeout, an abort): stop it by
                # label before the chips are given back, and block when one survives (_sweep).
                self._hardware_phase(spec, stage_dir, test,
                                     after_test=(lambda: self._sweep(spec.number)) if self._container_config(stage_dir) else None)
            out, step = self._step(spec, "finish", stage_dir, escalated, resumed)
            out = self._wrap_up(spec, stage_dir, step, out)
            if out.status != "done":
                return out.status, [f"the finish step ended: {out.status} {out.detail}".strip()], None
        gate = self._check_gate(spec, stage_dir)
        if not gate.ok and spec.boards and (failed := self._test_failure(spec, stage_dir)):
            # The finish step was told to record a failed test honestly. Its result cannot pass
            # the gate, and a continuation would ask the agent to make it pass. So the stage ends
            # here through the usual fail or escalate path, and the next attempt runs a fresh
            # prepare and hardware test.
            self.ledger.append("decision", spec.number,
                               decision="no gate feedback: the hardware test failed", **failed)
            return "fail", list(gate.reasons), gate
        if not gate.ok and getattr(step, "wrapped_up", False):
            # The step already had its one continuation, the wrap-up.
            return "fail", list(gate.reasons), gate
        if not gate.ok:
            # One continuation per run of the stage body; there is no loop here on purpose. It is
            # for a missing or malformed output file after a run step, or after a hardware test
            # that succeeded.
            out = self._gate_feedback(spec, stage_dir, step, gate)
            if out.status != "done":
                return out.status, [f"the {step.phase} step's continuation ended: "
                                    f"{out.status} {out.detail}".strip(), *gate.reasons], None
            gate = self._check_gate(spec, stage_dir)
        if gate.ok and spec.number == 2 and spec.draft == "swap_config":
            # The safe configuration passed. A model with a graft candidate is tried once more with the grafted drafter. The trial cannot
            # fail the stage: a failed or slow trial leaves the safe result as it is.
            changed = self._graft_trial(spec, stage_dir)
            # The sampling choice judges the configuration that is now served (the graft when it was chosen), so it runs after the graft trial.
            changed = self._sampling_choice(spec, stage_dir) or changed
            if changed:
                gate = self._check_gate(spec, stage_dir)       # the gate also reads the decision files
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _wrap_up(self, spec, stage_dir: Path, step: AgentStep, out):
        """A step that used up its turns with evidence on disk and its deliverable missing gets one
        wrap-up continuation of the same conversation. Any other outcome is returned as it is."""
        if out.status != "turns" or self._wrapup_used:
            return out
        name = deliverable(spec, step.phase)
        files = evidence_files(self.run_dir, stage_dir)
        # "Not written by this step": missing, or unchanged since the step started. A kill after an
        # earlier attempt's wrap-up can leave that attempt's file behind in a kept stage directory.
        if not name or not files or file_stamp(stage_dir / name) != step.deliverable_stamp:
            return out
        self._wrapup_used = True
        n = spec.number
        entry = self.ledger.append("decision", n, decision="wrap-up", phase=step.phase,
                                   file=f"stages/{n}/{name}", evidence=files, turns=AGENT_WRAPUP_TURNS,
                                   after=f"{out.status} {out.detail}".strip())
        step.wrapped_up = True
        log = stage_dir / "log" / f"{step.phase}-wrapup-{entry['seq']:05d}.jsonl"
        done = step.continue_with(wrapup_text(spec, name, files), max_turns=AGENT_WRAPUP_TURNS,
                                  log_path=log)
        if done.status == "done":
            return done
        return dataclasses.replace(done, detail=f"(in the wrap-up) {done.detail}".strip())

    def _test_failure(self, spec, stage_dir: Path) -> dict | None:
        """Why the stage's hardware testing failed, or None. One test: its test-result.json. A list
        of tests: the records of the tests that must pass (`failed_tests`), so an optional
        configuration that failed still leaves the gate feedback for a malformed result file."""
        if spec.tests:
            return failed_tests(stage_dir, self.required_chips)
        return hardware_test_failure(stage_dir)

    def _package_body(self, spec, stage_dir: Path, resumed: bool):
        """Stage 7 as supervisor code (orchard/package.py): stage and scrub every profile, install a
        copy of the required one, boot it under a lease (the same one-test hardware phase as stage
        2), then record the result, write the cards and publish commands, and run the gate.

        package-thin and install.sh get the agent shells' environment: no tokens, and HOME inside
        the run directory. So even a call that tried to upload would find no credentials."""
        n = spec.number
        opts = package_options(self.ledger.read())
        if not (resumed and (stage_dir / "test-result.json").is_file()):
            try:
                staged = stage_all(self.run_dir, stage_dir, namespace=opts["namespace"],
                                   models_root=opts["models_root"],
                                   env=agent_env(self.run_dir, extra=self.extra_env))
            except PackageError as exc:
                return "fail", [f"packaging: {exc}"], None
            self.ledger.append("evidence", n, what="package staged",
                               profiles=[p["name"] for p in staged["profiles"]],
                               **evidence_record(self.run_dir, stage_dir / "package.json"))
            test, problems = self._read_test(stage_dir)
            if problems:
                return "fail", problems, None
            self._hardware_phase(spec, stage_dir, test)
        try:
            package_finish(self.run_dir, stage_dir)
        except PackageError as exc:
            return "fail", [f"packaging: {exc}"], None
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _stock_triage_body(self, spec, stage_dir: Path):
        """Stage 0 of a run started with stock parameters, as supervisor code (orchard/stock_runtime.py): compare the model's config
        and tensor layout with tt-transformers' `model_params/<name>`, write delta.json and its evidence, run the gate. The class
        is the measurement's: stock-runtime when they agree, full-port with the reasons when they do not."""
        n = spec.number
        stage_dir.mkdir(parents=True, exist_ok=True)
        entries = self.ledger.read()
        params = Path(stock_params(entries) or "")
        snapshot = self.inputs.get("model")
        if not params.name or not snapshot:
            return "fail", ["a stock-runtime run needs the inputs model and stock_params"], None
        try:
            delta = stock_runtime.triage(
                stage_dir=stage_dir, run_dir=self.run_dir, model_id=self.model_id, snapshot=snapshot, params_dir=params,
                support=stock_runtime.support_row(self.run_dir, params.name),
                tt_metal=stock_runtime.tt_metal_tree(params.parents[3]))
        except OSError as exc:
            return "fail", [f"stock triage could not read its inputs: {exc}"], None
        self.ledger.append("evidence", n, what="stock triage", delta_class=delta["class"],
                           **evidence_record(self.run_dir, stage_dir / "delta.json"))
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    # ---- the audio runtime path (orchard/audio_runtime.py, orchard/audio_stages.py) ---------------------------------------------

    def _audio_models(self) -> tuple[str | None, str | None]:
        """The model and its base as stage 0 wrote them (`<repo>@<revision>`)."""
        try:
            delta = json.loads((self.run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.model_id, None
        return delta.get("model") or self.model_id, delta.get("nearest_model")

    def _audio_env(self) -> dict:
        """The environment of every child of the audio stages that reads or makes model data: the agent shells' (no token, HOME in the run directory,
        no chip) plus the Hugging Face cache directory, with the Hub offline."""
        child_env = agent_env(self.run_dir, extra=self.extra_env)
        child_env.update({"HF_HOME": self.paths.hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        return child_env

    def _audio_expected(self, model_id, base_id) -> tuple[dict, str | None]:
        """The identity (revision and configuration hash) of the snapshots this run was started on, for the model and for its base, and the base's
        id when its snapshot is known (a base without one has no control). Raises AudioStageError when a snapshot cannot be read."""
        from orchard import audio_plan
        expected = {}
        for repo, key in ((model_id, "model"), (base_id, "base")):
            name, snapshot = audio_plan.library_name(repo), self.inputs.get(key)
            if name and snapshot and name not in expected:       # a base that is another revision of the model's own repository has the same name; the model's entry stays
                expected[name] = audio_plan.snapshot_identity(snapshot)
        base = base_id if (audio_plan.library_name(base_id) in expected and audio_plan.library_name(base_id) != audio_plan.library_name(model_id)) else None
        return expected, base

    def _audio_body(self, spec, stage_dir: Path, resumed: bool):
        """Stage 1 (CPU references) and stage 2 (the clip on one chip) of an audio run, as supervisor code. The numeric work runs in the configured
        reference interpreter (orchard/audio_remote.py), so the supervisor needs only the standard library. Stage 2 uses the one-test hardware phase
        of stages 2 and 3: a lease (no coder to park when none was booted), one command, the lease back."""
        from orchard import audio_plan, audio_remote
        from orchard.audio_plan import AudioStageError
        n = spec.number
        if self.lab is not None:
            return "fail", ["the audio runtime does not run in lab mode"], None
        paths = audio_runtime.selection_paths(self.ledger.read())
        if paths is None:
            return "fail", ["the ledger holds no audio runtime selection with its paths"], None
        model_id, base_id = self._audio_models()
        stage_dir.mkdir(parents=True, exist_ok=True)
        try:
            expected, base_id = self._audio_expected(model_id, base_id)
            if n == 1:
                if not (resumed and (stage_dir / audio_runtime.REFERENCE_FILE).is_file()):
                    self._audio_stage_1(stage_dir, paths, model_id, base_id, expected)
            else:
                if not (resumed and (stage_dir / "test-result.json").is_file()):
                    plan = audio_plan.chip_plan(model_id, base_id)
                    audio_plan.write_chip_config(stage_dir, paths, self.paths.hf_home, plan)
                    command = audio_plan.chip_test_command(paths, stage_dir.resolve(), audio_runtime.TEMPLATES / "audio_chip_test.py")
                    try:
                        check_string(command, self.run_dir)
                    except Denied as exc:
                        return "fail", [f"the test command is refused: {exc}"], None
                    test = {"command": command, "deadline_s": AUDIO_CHIP_DEADLINE_S}
                    (stage_dir / "hw_test.json").write_text(json.dumps(test) + "\n", encoding="utf-8")
                    (stage_dir / "handoff.json").write_text(json.dumps({
                        "goal": "make one Stable Audio 3 clip with the diffusion transformer on one chip", "stage": 2,
                        "evidence": "stages/2/audio_chip_config.json", "next_action": "the supervisor runs the clip test under its own lease",
                        "check_on_return": "stages/2/test-result.json"}) + "\n", encoding="utf-8")
                    self._hardware_phase(spec, stage_dir, test)
                failed = hardware_test_failure(stage_dir)
                if failed:
                    return "fail", [failed["problem"] + "; see stages/2/evidence/hw-test-output.txt and stages/2/log"], None
                audio_remote.call(paths.reference_python, "finish", {
                    "stage_dir": str(stage_dir), "run_dir": str(self.run_dir), "ref_stage_dir": str(self.run_dir / "stages" / "1"),
                    "plan": audio_plan.chip_plan(model_id, base_id),
                    "roles": {audio_plan.ROLE_CHIP: audio_plan.ROLE_REFERENCE, audio_plan.ROLE_CONTROL_CHIP: audio_plan.ROLE_CONTROL_REFERENCE}},
                    timeout_s=1800.0, env={"PATH": os.environ.get("PATH", "")})
        except AudioStageError as exc:
            return "fail", [str(exc)], None
        self.ledger.append("evidence", n, what="audio stage record", **evidence_record(self.run_dir, stage_dir / spec.gate_file))
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _audio_stage_1(self, stage_dir: Path, paths, model_id, base_id, expected: dict) -> None:
        from orchard import audio_plan, audio_remote
        plan = audio_plan.reference_plan(model_id, base_id)
        cache = Path(self.paths.cache_root) / "audio-reference"
        result = audio_remote.call(paths.reference_python, "references", {
            "stage_dir": str(stage_dir), "run_dir": str(self.run_dir), "paths": paths.record(), "plan": plan, "expected": expected,
            "library_commit": audio_reference_env.LIBRARY_COMMIT, "cache_dir": str(cache), "env": self._audio_env(), "timeout_s": 3600.0},
            timeout_s=4 * 3600.0, env={"PATH": os.environ.get("PATH", "")})
        rec = result["record"]
        for note in result.get("notes", []):
            self.ledger.append("decision", 1, decision="audio reference", note=note)
        # A reference made fresh goes into the cache, so the next run of the same model, revision and library commit reuses it.
        for role in (audio_plan.ROLE_REFERENCE, audio_plan.ROLE_CONTROL_REFERENCE):
            run = rec["runs"].get(role)
            if run is None or run["from_cache"]:
                continue
            save_reference(self.run_dir / run["dir"], cache, run["model"])

    def _bundle_body(self, spec, stage_dir: Path):
        """Stage 8 as supervisor code: copy build_bundle.py into the stage, write its one-line config,
        run it, and run the gate. The script rebuilds the bundle from scratch each time, so a resume
        simply runs it again. It gets the agent shells' environment (no tokens, HOME in the run)."""
        n = spec.number
        stage_dir.mkdir(parents=True, exist_ok=True)
        script = stage_dir / "build_bundle.py"
        # An audio run's other stages already use the scripts of the checkout that is running (audio_runtime.TEMPLATES), so its bundle builder is that
        # checkout's too: a run started under an older checkout can then finish with the audio section. Other runs keep the recorded checkout.
        orchard_dir = (Path(__file__).resolve().parent.parent if audio_runtime.selected(self.ledger.read()) is not None
                       else Path(self.paths.orchard_dir))
        shutil.copyfile(orchard_dir / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py", script)
        (stage_dir / "bundle_config.json").write_text(
            json.dumps({"run_dir": str(self.run_dir), "orchard_dir": str(orchard_dir)}, indent=2)
            + "\n", encoding="utf-8")
        (stage_dir / "log").mkdir(exist_ok=True)
        log = stage_dir / "log" / "build_bundle.log"
        try:
            proc = subprocess.run([sys.executable, str(script)], cwd=self.run_dir, capture_output=True,
                                  text=True, timeout=BUNDLE_BUILD_TIMEOUT_S,
                                  env=agent_env(self.run_dir, extra=self.extra_env))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return "fail", [f"build_bundle.py did not finish: {exc}"], None
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            last = (proc.stderr.strip() or proc.stdout.strip() or "no output").splitlines()[-1]
            return "fail", [f"build_bundle.py exited {proc.returncode}: {last}"[:500]], None
        self.ledger.append("evidence", n, what="bundle built",
                           **evidence_record(self.run_dir, stage_dir / "evidence" / "bundle-build.json"))
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _check_gate(self, spec, stage_dir: Path):
        if spec.number == 8:
            bundle = stage_dir / "bundle"
            bundle.mkdir(exist_ok=True)
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
            self._copy_package(bundle)
        gate = self._gate(spec)(stage_dir, self.run_dir)
        if spec.tests:
            # The gate reads the test records from the stage directory, where the agent can write
            # too. The ledger holds the sha256 of each record the supervisor wrote.
            forged = unrecorded(self.ledger.read(), stage_dir, self.run_dir, spec.number)
            if forged:
                return GateResult(False, gate.reasons + tuple(forged), gate.evidence)
        return gate

    def _copy_package(self, bundle: Path) -> None:
        """Put stage 7's record, publish commands and cards in the operator bundle's package/
        folder, so the operator reads them with the rest. Nothing is copied when no package was
        staged. The bundle scrub then searches these copies too."""
        s7 = self.run_dir / "stages" / "7"
        if not (s7 / "package.json").is_file():
            return
        dest = bundle / "package"
        dest.mkdir(exist_ok=True)
        shutil.copyfile(s7 / "package.json", dest / "package.json")
        if (s7 / "PUBLISH_COMMANDS.txt").is_file():
            shutil.copyfile(s7 / "PUBLISH_COMMANDS.txt", dest / "PUBLISH_COMMANDS.txt")
        for card in sorted((s7 / "package").glob("*/README.md")):
            shutil.copyfile(card, dest / f"{card.parent.name}-README.md")

    def _gate_feedback(self, spec, stage_dir: Path, step: AgentStep, gate):
        """The step ended "done" and the exit gate failed. Tell the same conversation what the gate
        found and run it once more, with its own turn cap and its own log."""
        n = spec.number
        entry = self.ledger.append("decision", n, decision="gate feedback", phase=step.phase,
                                   reasons=list(gate.reasons), file=f"stages/{n}/{spec.gate_file}")
        text = gate_feedback_text(spec, gate.reasons)
        log = stage_dir / "log" / f"{step.phase}-continuation-{entry['seq']:05d}.jsonl"
        return step.continue_with(text, max_turns=AGENT_CONTINUATION_TURNS, log_path=log)

    def _stage_0_facts(self) -> dict:
        """`{"facts": {...}}` for stage 0's passing end, or {} when delta.json holds none (a text model: the default applies).
        The template decides form and modality from the files; `serving` is a property of this machine, so it comes from the
        installed bundles and container packages (orchard/audio_facts.py)."""
        facts, delta = delta_facts(self.run_dir), None
        if facts is None:
            return {}
        try:
            delta = json.loads((self.run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        installed = nearest.read_all_installed(self.paths.tt_model_root, self.paths.operator_home)
        return {"facts": {"form": facts.get("form"), "serving": serving_of(delta.get("nearest_model"), installed),
                          "modality": facts.get("modality")}}

    def _end_stage(self, spec, stage_dir: Path, status: str, reasons: list, gate, escalated: bool) -> str:
        n = spec.number
        if status == "pass":
            ev = [evidence_record(self.run_dir, self.run_dir / rel) for rel in sorted(set(gate.evidence))]
            ev.append(evidence_record(self.run_dir, stage_dir / spec.gate_file))
            if n == 6:
                self._record_numbers(stage_dir)
            # Stage 0's pass records the path it chose. Later stages read it from here
            # (orchard/stages.py, run_path), so an edit to delta.json cannot change the run's path.
            extra = ({"path": delta_path(self.run_dir), "class": delta_class(self.run_dir), **self._stage_0_facts()}
                     if n == 0 else {})
            self.ledger.append("stage_end", n, result="pass", evidence=ev, **extra)
            return "pass"
        if status == "pause":
            return "pause"              # the ladder already wrote the pause decision
        if status == "operator-pause":
            self.ledger.append("decision", n, decision="pause", reason="operator")
            return "pause"
        if status == "abort":
            return "abort"
        if spec.harness:
            # Supervisor code failed: another model cannot fix it, so it is not escalated.
            self.ledger.append("decision", n, decision="pause",
                               reason=f"stage {n} ({spec.name}) failed; it is supervisor code and "
                                      "is not escalated: " + "; ".join(reasons)[:500])
            self.ledger.append("stage_end", n, result="fail", reasons=reasons)
            return "fail"
        if status == "escalate":
            # The watchdog's ladder escalated and wrote the escalate entry before acting.
            self.ledger.append("stage_end", n, result="escalate", reasons=reasons)
            return "escalate"
        if not escalated:
            # The escalate entry comes first: a crash between the two resumes the stage escalated.
            self.ledger.append("escalate", n, by="stage machine", reasons=reasons)
            self.ledger.append("stage_end", n, result="escalate", reasons=reasons)
            return "escalate"
        self.ledger.append("decision", n, decision="pause",
                           reason=f"stage {n} failed after escalation: " + "; ".join(reasons)[:500])
        self.ledger.append("stage_end", n, result="fail", reasons=reasons)
        return "fail"

    def _record_numbers(self, stage_dir: Path) -> None:
        data = json.loads((stage_dir / "result.json").read_text(encoding="utf-8"))
        for num in data["numbers"]:
            self.ledger.append("measurement", 6, name=num["name"], value=num.get("value"),
                               unit=num["unit"], label=num["label"], evidence=num.get("evidence"))

    def _step(self, spec, phase: str, stage_dir: Path, escalated: bool, resumed: bool):
        n = spec.number
        tier = tier_for(self.cfg, n, phase=phase, escalated=escalated)
        try:
            used, endpoint, note = resolve_endpoint(self.cfg, tier, self.probe)
        except TierUnavailable as exc:
            self._block(n, str(exc))
        if note and not quiet_substitute(self.cfg, self.coder_tier, tier, used):
            self.ledger.append("decision", n, decision="tier substituted", note=note)
        skill = resolve_skill(spec.skill, self.skills_dirs)
        if skill is None:
            self._block(n, f"the skill {spec.skill!r} is not in {[str(d) for d in self.skills_dirs]}")
        refs = {r: resolve_skill(r, self.skills_dirs) for r in spec.refs}
        entries = self.ledger.read()
        try:
            system, user = build_messages(spec=spec, phase=phase, run_dir=self.run_dir,
                                          stage_dir=stage_dir, skill_path=skill, refs=refs,
                                          facts=facts_from(run_progress(entries).run_start, self.run_dir),
                                          entries=entries, resumed=resumed, paths=self.paths)
        except UnknownPlaceholder as exc:     # a typo in a skill: the operator fixes the skill
            self._block(n, str(exc))
        model = self.cfg.tiers[used]["model"]
        self.ledger.append("decision", n, decision="agent step", phase=phase, tier=used, model=model,
                           escalated=escalated, skill=str(skill))
        step = AgentStep(agent=AGENT, endpoint=endpoint, model=model,
                         tools=Tools(self.run_dir, stage_dir, agent_env(self.run_dir, extra={
                             **self.extra_env, "TT_MODEL_MODELS_DIR": str(self.paths.tt_model_root)}),
                             scratch=(self.run_dir / "tt-metal-scratch"
                                      if scratch_stage(n, run_path(entries, self.run_dir)) else None)),
                         ledger=self.ledger, stage=n, phase=phase, feed=self.watchdog.feed,
                         control=self.actuator, run_dir=self.run_dir,
                         evidence_dir=stage_dir / "evidence",
                         log_path=stage_dir / "log" / f"{phase}-{len(entries) + 1:05d}.jsonl",
                         http=self.http, clock=self.clock, guard=self.guard,
                         finished=self._finished_check(spec, phase, stage_dir))
        name = deliverable(spec, phase)
        step.deliverable_stamp = file_stamp(stage_dir / name) if name else None
        return step.run(system, user), step

    def _draft(self, spec, stage_dir: Path) -> None:
        """Write the config a stage's skill starts from, when the run already holds every fact in it
        (orchard/swap_draft.py), and copy the templates that read it. A config already in the stage
        directory (a resumed stage) is kept. When a fact is missing or ambiguous nothing is written,
        the ledger says why, and the agent finds the facts as its skill describes."""
        if spec.draft == "stage4":
            return self._draft_stage4(spec, stage_dir)
        if spec.draft == "stock_config":
            return self._draft_stock(spec, stage_dir)
        if spec.draft != "swap_config" or (stage_dir / "swap_config.json").exists():
            return
        cfg, problems = swap_draft.draft(run_dir=self.run_dir, tt_model_root=self.paths.tt_model_root,
                                         cache_root=self.paths.cache_root, hf_home=self.paths.hf_home,
                                         inputs=self.inputs, chips=spec.boards * CHIPS_PER_BOARD,
                                         operator_home=self._container_home(spec))
        if cfg is None:
            self.ledger.append("decision", spec.number, decision="swap config not drafted", problems=problems)
            return
        target = stage_dir / "swap_config.json"
        target.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        templates = Path(self.paths.orchard_dir) / "orchard" / "skills" / "weights-swap-templates"
        copied = []
        for name in (STAGE4_TEMPLATES if "package" in cfg else SWAP_TEMPLATES):      # a container package also needs its script
            shutil.copy2(templates / name, stage_dir / name)
            copied.append(name)
        self.ledger.append("decision", spec.number, decision="swap config drafted",
                           config=evidence_record(self.run_dir, target), bundle_dir=cfg.get("bundle_dir"),
                           copied=copied, **({"package": cfg["package"]} if "package" in cfg else {}))

    def _draft_stock(self, spec, stage_dir: Path) -> None:
        """stock_config.json for stage 2 of a stock-runtime run (orchard/stock_runtime.py, draft_config), and the template that reads it.
        A config already in the stage directory (a resumed stage) is kept."""
        target = stage_dir / "stock_config.json"
        if target.exists():
            return
        try:
            delta = json.loads((self.run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            delta = None
        cfg, problems = stock_runtime.draft_config(run_dir=self.run_dir, delta=delta, inputs=self.inputs,
                                                   chips=spec.boards * CHIPS_PER_BOARD, cache_root=self.paths.cache_root)
        if cfg is None:
            self.ledger.append("decision", spec.number, decision="stock config not drafted", problems=problems)
            return
        target.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(Path(self.paths.orchard_dir) / "orchard" / "skills" / "stock-runtime-templates" / "stock_check.py",
                     stage_dir / "stock_check.py")
        self.ledger.append("decision", spec.number, decision="stock config drafted", config=evidence_record(self.run_dir, target),
                           copied=["stock_check.py"], mesh_device=cfg["mesh_device"])

    def _container_home(self, spec):
        """The operator home to look for container packages in, or None when stage 2 may not use one: the sidecar parity workflow
        (its skill and scripts need a bundle), and a lab run (the containers would run on the lab, where this supervisor's label
        search and sweep cannot see them)."""
        return self.paths.operator_home if spec.skill == "weights-swap-check" and self.lab is None else None

    @staticmethod
    def _container_config(stage_dir: Path) -> bool:
        """Does the stage's swap_config.json name a container package (issue 60)? An unreadable or missing config is not one."""
        try:
            cfg = json.loads((stage_dir / "swap_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(cfg, dict) and "package" in cfg

    def _draft_stage4(self, spec, stage_dir: Path) -> None:
        """Stage 4's configs (swap_draft.draft_stage4) for the counts this run plans: the required ones, or
        1, 2 and 4, and never more chips than the lab (or this box) has. hw_tests.json is written only when
        every count got a config; otherwise the agent writes it as its skill describes."""
        if (stage_dir / "hw_tests.json").exists() or (stage_dir / "configs").exists():
            return
        cap = self._chip_cap(spec)
        counts = [n for n in (self.required_chips or (1, 2, 4)) if n <= cap]
        missing: list[int] = []
        configs, tests, notes = swap_draft.draft_stage4(
            run_dir=self.run_dir, tt_model_root=self.paths.tt_model_root, cache_root=self.paths.cache_root,
            hf_home=self.paths.hf_home, operator_home=self.paths.operator_home, inputs=self.inputs,
            counts=counts, four_chip_package=self.four_chip_package, missing=missing)
        # A required count that nothing serves cannot pass whatever an agent writes (issue 68: the agent wrote the same file three times
        # and the run blocked 20 minutes of chip time later). Say so before any agent step.
        short = list(missing) if self.required_chips else []        # counts holds only the required ones when they are set
        if short:
            rest = [n for n in self.required_chips if n not in short]
            self.ledger.append("decision", spec.number, decision="stage 4 configs not drafted", counts=counts, problems=notes)
            self._block(spec.number,
                        f"stage 4 requires {', '.join(f'{n}-chip' for n in short)} "
                        f"but no installed bundle or container package serves the nearest model on {'that count' if len(short) == 1 else 'those counts'}. "
                        + (f"A run keeps the counts it started with, so move this run's directory aside and start a new run with "
                           f"required_chips = \"{','.join(str(n) for n in rest)}\" in bringup.toml (stages 0 to 2 run again), "
                           if rest else "No required count can be served; ")
                        + "or install a package for the missing count",
                        required=sorted(self.required_chips), missing=short)
        if not configs:
            self.ledger.append("decision", spec.number, decision="stage 4 configs not drafted", counts=counts,
                               problems=notes)
            return
        templates = Path(self.paths.orchard_dir) / "orchard" / "skills" / "weights-swap-templates"
        for n, cfg in configs.items():
            cdir = stage_dir / "configs" / str(n)
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "swap_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            for name in STAGE4_TEMPLATES:
                shutil.copy2(templates / name, cdir / name)
        if tests is not None:
            (stage_dir / "hw_tests.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
        self.ledger.append("decision", spec.number, decision="stage 4 configs drafted", counts=counts,
                           configs={str(n): c.get("bundle_dir") or c.get("package") for n, c in configs.items()},
                           caches={str(n): c["tt_cache"] for n, c in configs.items()},
                           hw_tests=tests is not None, tests=tests, cap=cap, notes=notes)

    def _chip_cap(self, spec) -> int:
        """The most chips one test here can have: the lab's, or this stage's boards on this box."""
        if self.lab is not None and getattr(self.lab, "chips", None):
            return self.lab.chips
        return spec.boards * CHIPS_PER_BOARD

    def _keep_drafted_tests(self, spec, stage_dir: Path) -> None:
        """When the supervisor drafted every count, its hw_tests.json is the list. On lab run 2 the agent
        added a 4-chip test on a 2-chip lab (and built its config by hand); the test could only be
        recorded as not run. The drafted list goes back, and the ledger says what was dropped."""
        drafted = next((e["data"] for e in reversed(self.ledger.read())
                        if e["stage"] == spec.number and e["data"].get("decision") == "stage 4 configs drafted"), {})
        want = drafted.get("tests")
        if not isinstance(want, dict):
            return
        path = stage_dir / "hw_tests.json"
        try:
            have = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            have = None
        if have == want:
            return

        def counts(d):
            items = d.get("tests") if isinstance(d, dict) else None
            return sorted({t.get("chips") for t in items if isinstance(t, dict) and isinstance(t.get("chips"), int)}
                          if isinstance(items, list) else set())
        agent, kept = counts(have), counts(want)
        where = f"the lab {self.lab.info.get('host')}" if self.lab is not None else "this box"
        self.ledger.append("decision", spec.number, decision="kept the drafted test list", agent=agent,
                           drafted=kept, dropped=[n for n in agent if n not in kept],
                           reason=f"{where} has {self._chip_cap(spec)} chips for a test; the supervisor "
                                  f"drafted {kept}")
        path.write_text(json.dumps(want, indent=2) + "\n", encoding="utf-8")

    def _keep_drafted_caches(self, spec, stage_dir: Path) -> None:
        """The tensor cache of a configuration the supervisor drafted is the drafted one. On the first run
        with drafts the agent rewrote the configs with its own spelling of the cache names, so no later run
        would have found them. The cache is read only when the test runs, so setting it back after the
        prepare step changes nothing the agent built."""
        drafted = next((e["data"].get("caches") or {} for e in reversed(self.ledger.read())
                        if e["stage"] == spec.number and e["data"].get("decision") == "stage 4 configs drafted"), {})
        for n, cache in drafted.items():
            path = stage_dir / "configs" / str(n) / "swap_config.json"
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(cfg, dict) or cfg.get("tt_cache") == cache:
                continue
            self.ledger.append("decision", spec.number, decision="kept the drafted tensor cache", config=int(n),
                               agent=cfg.get("tt_cache"), drafted=cache)
            cfg["tt_cache"] = cache
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    def _read_plan(self, spec, stage_dir: Path):
        tests, problems = read_plan(stage_dir, required=self.required_chips, max_chips=spec.boards * CHIPS_PER_BOARD,
                                    budget_s=spec.budget_s, home=self.home)
        return tests, problems + swap_draft.stage4_nearest_problems(self.run_dir, stage_dir, tt_model_root=self.paths.tt_model_root,
                                                                       operator_home=self.paths.operator_home, counts=[t.chips for t in tests])

    def _finished_check(self, spec, phase: str, stage_dir: Path):
        """What says a step's files are written and valid (AgentStep's `finished`): for one hardware
        test's prepare step, hw_test.json and handoff.json as the hardware phase reads them; for a run or
        finish step, the stage's gate. A list of tests (stage 4) and stage 7 have no such check."""
        if phase == "prepare":
            if spec.tests:
                return lambda: not self._read_plan(spec, stage_dir)[1]
            return lambda: not self._read_test(stage_dir)[1]
        return lambda: self._gate(spec)(stage_dir, self.run_dir).ok

    # ---- the hardware test ----------------------------------------------------------------------

    def _read_test(self, stage_dir: Path) -> tuple[dict | None, list[str]]:
        problems = []
        try:
            test = json.loads((stage_dir / "hw_test.json").read_text(encoding="utf-8"))
            note = json.loads((stage_dir / "handoff.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return None, [f"hw_test.json and handoff.json must exist and be JSON: {exc}"]
        if not isinstance(test, dict) or not isinstance(test.get("command"), str) or not test["command"].strip():
            return None, ["hw_test.json needs a command"]
        d = test.get("deadline_s")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
            problems.append("hw_test.json needs a positive deadline_s")
        if (wrong := swap_draft.nearest_problem(self.run_dir, stage_dir / "swap_config.json", tt_model_root=self.paths.tt_model_root,
                                                operator_home=self.paths.operator_home)):
            problems.append(wrong)
        if not isinstance(note, dict) or any(note.get(k) in (None, "") for k in NOTE_KEYS):
            problems.append(f"handoff.json needs {', '.join(NOTE_KEYS)}")
        try:
            check_string(test["command"], self.run_dir)
        except Denied as exc:
            problems.append(f"the test command is refused: {exc}")
        return test, problems

    @staticmethod
    def _cleanup_after_test(after_test) -> None:
        """Run the cleanup callback from a `finally`. When an exception is already on its way out (an Interrupted from SIGTERM, for
        one) and the cleanup blocks, the original exception keeps going: the block is already in the ledger and the surviving container
        keeps its lease (_sweep), and the run must still end the way the signal asked. With no exception in flight the block propagates."""
        if after_test is None:
            return
        in_flight = sys.exc_info()[1]
        try:
            after_test()
        except Blocked:
            if in_flight is None:
                raise

    def _hardware_phase(self, spec, stage_dir: Path, test: dict, *, after_test=None, **record) -> None:
        """Lease the chips (parking the coder when its board is needed), run `test`, give them back. `record` names where the run's record
        and output go (see _run_test); the graft trial uses it to leave stage 2's own test record alone. `after_test` runs when the test
        ends however it ends (a result, a timeout, an interrupt, an error) and before the coder is restored or the lease released: the
        trial uses it to stop the servers its helpers started, which a kill of the test's own session does not reach."""
        n = spec.number
        if self.lab is not None:
            return self._lab_hardware_phase(spec, stage_dir, test, after_test=after_test)
        boards = self._phase_boards(spec, stage_dir)
        if boards > spec.boards:
            return self._wide_hardware_phase(spec, stage_dir, test, boards, after_test=after_test, **record)
        chips = self.adapter.status()
        # A run that has not booted a coder (the audio stages need none) has no coder boards to park: a free board is leased, or the lease waits.
        d = decide_park(self.coder_lease.units if self.coder_lease is not None else (), chips, spec.boards)
        self.ledger.append("decision", n, decision="hardware phase", action=d.action,
                           free_boards=list(d.free_boards), server_boards=list(d.server_boards))
        if d.park_needed:
            h = self._handoff(n, self.coder_lease)
            lease = h.park()
            try:
                self._run_test(spec, stage_dir, test, lease, **record)
            finally:
                self._cleanup_after_test(after_test)
            self._restore_coder(n, h)
            self.coder_lease = h.lease
            return
        exact = None
        if d.action == "use_free":
            exact = next(c.bdf for c in chips if c.board == d.free_boards[0])
        lease = reacquire(self.adapter, chips=spec.boards * CHIPS_PER_BOARD, who=WHO,
                          reason=f"stage {n} hardware test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep, exact=exact)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", test_lease=lease.record())
        try:
            self._run_test(spec, stage_dir, test, lease, **record)
        finally:
            self._cleanup_after_test(after_test)
        try:
            self.adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _graft_trial(self, spec, stage_dir: Path) -> bool:
        """Stage 2's graft trial (orchard/graft_trial.py). Returns True when it wrote a decision file now.

        Only for a model whose stage 0 delta has a graft candidate (the nearest model's mtp.* head fits and the model has none). One more
        hardware test, under the same lease rules as stage 2's own, runs skills/weights-swap-templates/graft_trial.py: it times the safe
        configuration, serves the grafted variant and times that. The decision is code (graft.decide): the graft is shipped only when
        the variant served, agreed with the CPU reference and decoded enough faster. A test that fails, times out or leaves no result is
        recorded and the safe configuration stands. graft-decision.json is the marker: a finished stage 2 is not tried again, and a kill
        after the test's own record resumes at the decision without booting again."""
        n = spec.number
        if (stage_dir / graft.DECISION_FILE).exists():
            return False
        try:
            delta = json.loads((self.run_dir / "stages" / "0" / "delta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        cand = graft.candidate(delta)
        if cand is None:
            return False
        skip = None
        if not self.graft_trial:
            skip = "the graft trial was skipped: --no-graft-trial"
        elif self.lab is not None:
            skip = "the graft trial was skipped: it runs on this box's chips and this run's tests run on a lab"
        elif self._container_config(stage_dir):
            skip = ("the graft trial was skipped: stage 2 was served through a container package, and the trial's scripts "
                    "build and run a bundle's run.sh, which a container package does not have")
        else:
            enough, free = check_disk(self.run_dir, GRAFT_TRIAL_DISK_GB, usage=self.disk_usage)
            if not enough:
                skip = (f"the graft trial was skipped: the grafted variant converts a second tensor cache and needs "
                        f"{GRAFT_TRIAL_DISK_GB:g} GB free on the run directory's disk; {free} GB is free")
        if skip:
            self.ledger.append("decision", n, decision="graft trial skipped", reason=skip)
            decision = {"choice": "safe", "reason": skip, "numbers": {}, "thresholds": {}}
            graft.write_decision(stage_dir, decision, candidate=cand, trial=False)
            self._record_graft_decision(stage_dir, decision)
            return True
        trial_dir = stage_dir / graft.TRIAL_DIR
        trial_dir.mkdir(exist_ok=True)
        marker = trial_dir / "test-result.json"
        if not marker.is_file():
            shutil.copy2(Path(self.paths.orchard_dir) / "orchard" / "skills" / "weights-swap-templates" / GRAFT_TRIAL_TEMPLATE,
                         trial_dir / GRAFT_TRIAL_TEMPLATE)
            (trial_dir / "trial_config.json").write_text(
                json.dumps({"stage_dir": str(stage_dir), "orchard_dir": str(self.paths.orchard_dir)}, indent=2) + "\n", encoding="utf-8")
            self.ledger.append("decision", n, decision="graft trial started", candidate=cand,
                               note="serving the nearest model's mtp.* head grafted onto this model, beside the safe configuration")
            test = {"command": f"python3 stages/{n}/{graft.TRIAL_DIR}/{GRAFT_TRIAL_TEMPLATE}", "deadline_s": GRAFT_TRIAL_DEADLINE_S}
            self._hardware_phase(spec, stage_dir, test, record=f"{graft.TRIAL_DIR}/test-result.json",
                                 output=f"{graft.TRIAL_DIR}/evidence/hw-test-output.txt", what="graft trial test",
                                 env_extra={GRAFT_TRIAL_ENV: str(trial_dir.resolve())},
                                 after_test=lambda: self._stop_trial_servers(n, trial_dir, "after the trial test"))
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        trial = None
        if record.get("returncode") == 0 and not record.get("timed_out"):
            try:
                trial = json.loads((trial_dir / graft.TRIAL_RESULT).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                trial = None
        try:
            safe_top1 = json.loads((stage_dir / "result.json").read_text(encoding="utf-8")).get("top1_agreement")
        except (OSError, ValueError, AttributeError):
            safe_top1 = None
        decision = graft.decide(trial, safe_top1=safe_top1, sampling=graft.sampling_input(stage_dir, self.run_dir))
        if trial is None:
            if record.get("timed_out"):
                why = "the graft trial timed out and was stopped"
            elif record.get("returncode") not in (0, None):
                why = f"the graft trial did not finish (exit code {record['returncode']})"
            else:
                why = f"the graft trial wrote no {graft.TRIAL_RESULT}"
            decision = {**decision, "reason": f"{why}; the safe configuration stands"}
        graft.write_decision(stage_dir, decision, candidate=cand)
        self._record_graft_decision(stage_dir, decision)
        return True

    def _sampling_choice(self, spec, stage_dir: Path) -> bool:
        """Stage 2's sampling choice (orchard/sampling_choice.py, issue 107). Returns True when it wrote a decision file now.

        A fast decoding path can be greedy only: the Qwen3.8 DFlash2 drafter accepts a sampled request and samples only its first token. The rule
        reads whether the model needs sampling (stage 0's files and the operator's --sampling flag), what the served configuration's sampling probe
        showed, and the speed of each configuration. Only when sampling is needed, the drafter is on and the probe does not show sampling applied does
        it need a second measurement: one hardware test, skills/weights-swap-templates/sampling_trial.py, serves the drafter-off variant. The decision
        is code (sampling.decide). A trial that fails, times out or leaves no result is not a stage failure: the served configuration stays and the
        decision says so. sampling-decision.json is the marker, and the trial's own test record is a second one, so a kill after the test resumes at
        the decision without booting again. A stage 2 with no swap check (it was not served by the swap template) makes no decision."""
        n = spec.number
        if (stage_dir / sampling.DECISION_FILE).exists():
            return False
        current = sampling.served_side(stage_dir)
        if current is None:
            return False
        need = sampling.need(self.run_dir, self.sampling)
        variant, note, ran = None, None, False
        if sampling.needs_variant(need, current):
            skip = None
            if self.lab is not None:
                skip = "the sampling trial was skipped: it runs on this box's chips and this run's tests run on a lab"
            elif self._container_config(stage_dir):
                skip = ("the sampling trial was skipped: stage 2 was served through a container package, and the trial's scripts build and "
                        "run a bundle's run.sh, which a container package does not have")
            else:
                # The disk is a prerequisite for a trial that still has to run. One that already ran (its test record exists) may itself have
                # filled the disk with its new tensor cache, and a restart must read its result and not skip it for that.
                already_ran = (stage_dir / sampling.TRIAL_DIR / "test-result.json").is_file()
                if not already_ran:
                    # The new tensor cache is written beside the configured `tt_cache`, which can be on another filesystem than the run
                    # directory, so the conversion is checked there (the nearest existing parent of that path), and the run directory is
                    # checked apart for the trial's evidence.
                    cache_path = self._cache_check_path(stage_dir)
                    enough, free = check_disk(cache_path, SAMPLING_TRIAL_DISK_GB, usage=self.disk_usage)
                    if not enough:
                        where = "the configured tensor cache's disk" if cache_path != self.run_dir else "the run directory's disk"
                        skip = (f"the sampling trial was skipped: the drafter-off variant converts a second tensor cache and needs "
                                f"{SAMPLING_TRIAL_DISK_GB:g} GB free on {where}; {free} GB is free")
                    else:
                        enough, free = check_disk(self.run_dir, SAMPLING_TRIAL_RUN_DISK_GB, usage=self.disk_usage)
                        if not enough:
                            skip = (f"the sampling trial was skipped: its evidence needs {SAMPLING_TRIAL_RUN_DISK_GB:g} GB free on the run "
                                    f"directory's disk; {free} GB is free")
            if skip:
                self.ledger.append("decision", n, decision="sampling trial skipped", reason=skip)
                note = skip
            else:
                variant, note = self._run_sampling_trial(spec, stage_dir, need, current)
                ran = True
        decision = sampling.decide(need, current, variant, variant_note=note)
        sampling.write_decision(stage_dir, decision, trial=variant is not None)
        path = stage_dir / sampling.DECISION_FILE
        self.ledger.append("decision", n, decision="sampling decision", choice=decision["choice"], flag=decision["flag"], reason=decision["reason"],
                           needed=decision["needed"], need_source=decision["need_source"], numbers=decision["numbers"],
                           decision_file=evidence_record(self.run_dir, path), trial=ran)
        return True

    def _cache_check_path(self, stage_dir: Path) -> Path:
        """The path whose filesystem the sampling trial's new tensor cache lands on: the nearest existing parent of the stage's configured `tt_cache`
        (the cache is made later, so the path itself may not exist). The run directory when the config names none."""
        try:
            cfg = json.loads((stage_dir / "swap_config.json").read_text(encoding="utf-8"))
            cache = Path(str(cfg["tt_cache"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return self.run_dir
        if not cache.is_absolute():
            cache = self.run_dir / cache
        while not cache.exists() and cache != cache.parent:
            cache = cache.parent
        return cache if cache.exists() else self.run_dir

    def _run_sampling_trial(self, spec, stage_dir: Path, need: dict, current: dict):
        """Run (or, after a kill, read) the drafter-off trial. Returns (the variant's side or None, a note saying why there is none)."""
        n = spec.number
        trial_dir = stage_dir / sampling.TRIAL_DIR
        trial_dir.mkdir(exist_ok=True)
        marker = trial_dir / "test-result.json"
        if not marker.is_file():
            shutil.copy2(Path(self.paths.orchard_dir) / "orchard" / "skills" / "weights-swap-templates" / SAMPLING_TRIAL_TEMPLATE,
                         trial_dir / SAMPLING_TRIAL_TEMPLATE)
            (trial_dir / "trial_config.json").write_text(
                json.dumps({"stage_dir": str(stage_dir), "orchard_dir": str(self.paths.orchard_dir)}, indent=2) + "\n", encoding="utf-8")
            self.ledger.append("decision", n, decision="sampling trial started", needed=need["needed"], need_source=need["source"],
                               served_verdict=current.get("verdict"),
                               note="serving this model with the speculative drafter off and host sampling, beside the configuration stage 2 served")
            test = {"command": f"python3 stages/{n}/{sampling.TRIAL_DIR}/{SAMPLING_TRIAL_TEMPLATE}",
                    "deadline_s": SAMPLING_TRIAL_DEADLINE_S}
            self._hardware_phase(spec, stage_dir, test, record=f"{sampling.TRIAL_DIR}/test-result.json",
                                 output=f"{sampling.TRIAL_DIR}/evidence/hw-test-output.txt", what="sampling trial test",
                                 env_extra={GRAFT_TRIAL_ENV: str(trial_dir.resolve())},
                                 after_test=lambda: self._stop_trial_servers(n, trial_dir, "after the sampling trial test"))
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        if record.get("returncode") == 0 and not record.get("timed_out"):
            variant = sampling.variant_side(stage_dir, self.run_dir)
            if variant is not None:
                return variant, None
            return None, f"the sampling trial wrote no {sampling.TRIAL_RESULT}"
        if record.get("timed_out"):
            return None, "the sampling trial timed out and was stopped"
        return None, f"the sampling trial did not finish (exit code {record.get('returncode')})"

    def _stop_trial_servers(self, n: int, trial_dir: Path, when: str) -> None:
        """Stop what the graft trial's helpers left running. measure_serving.py and serve_and_compare.py start run.sh in a session of their
        own, so the kill of the trial's session on a deadline or an interrupt does not reach a server, and a server that stays up holds the
        chips. Every process of this user that carries the trial directory in ORCHARD_TRIAL_DIR is stopped (SIGTERM, then SIGKILL), with the
        identity it had (pid, boot id, start time) written to the ledger. A server that is still there afterwards blocks the run: the coder
        is not restored over it. The trial does not release the lease itself; the run's wind-down does, and gozer refuses to reset a
        chip that a server holds."""
        records = kernelprobe.sweep_strays(trial_dir, var=GRAFT_TRIAL_ENV, identity=True)
        if records:
            self.ledger.append("decision", n, decision="graft trial servers stopped", when=when, servers=records)
        left = [r["pid"] for r in records if not r.get("stopped")]
        if left:
            self._block(n, "a server of the graft trial is still running (pid " + ", ".join(str(p) for p in left) + "), so the coder is "
                           "not restored over it; stop it and run the command again", pids=left)

    def _record_graft_decision(self, stage_dir: Path, decision: dict) -> None:
        """The ledger's copy of the decision, with the decision file's sha256 so a later stage can tell the file was not changed."""
        path = stage_dir / graft.DECISION_FILE
        self.ledger.append("decision", 2, decision="graft decision", choice=decision["choice"], reason=decision["reason"],
                           numbers=decision["numbers"], decision_file=evidence_record(self.run_dir, path))

    def _phase_boards(self, spec, stage_dir: Path) -> int:
        """Boards the stage's single test holds: the stage's own, or more when the drafted swap config names a container package that
        needs more chips. Catalog container packages mostly report 2 boards (4 chips), so stage 2 serves them on that many, as
        stage 4 does for its 4-chip test (issue 60)."""
        try:
            cfg = json.loads((stage_dir / "swap_config.json").read_text(encoding="utf-8"))
            chips = int(cfg["chips"]) if cfg.get("package") else 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return spec.boards
        return max(spec.boards, -(-chips // CHIPS_PER_BOARD))

    def _wide_hardware_phase(self, spec, stage_dir: Path, test: dict, boards: int, *, after_test=None, **record) -> None:
        """The hardware phase of a single test that needs `boards` boards (more than the stage's), the way stage 4 runs its larger
        configurations: a further board is leased first (the coder is untouched when there is none), then the coder is parked and its
        board joins the test; afterwards the further board goes back and the coder is restored."""
        n = spec.number
        # A test lease left from a block (a refused pre-test reset, a container that would not stop) goes back first, so an attended
        # resume does not find its own further board still claimed.
        if not self._release_test_lease():
            self._block(n, "releasing a test lease left from an earlier test failed")
        chips = self.adapter.status()
        have = len({c.board for c in chips})
        if boards > have:
            self._block(n, f"the container package needs {boards * CHIPS_PER_BOARD} chips and this box has {have * CHIPS_PER_BOARD}",
                        need_chips=boards * CHIPS_PER_BOARD)
        d = decide_park(self.coder_lease.units, chips, boards)
        self.ledger.append("decision", n, decision="hardware phase", action=d.action, chips=boards * CHIPS_PER_BOARD,
                           free_boards=list(d.free_boards), server_boards=list(d.server_boards))
        shape = SimpleNamespace(chips=boards * CHIPS_PER_BOARD, boards=boards)
        first_chip = {c.board: c.bdf for c in reversed(chips)}
        if d.action == "use_free":
            lease = self._take_test_lease(spec, shape, boards, None)
            try:
                self._run_test(spec, stage_dir, test, lease, boards=boards, **record)
            finally:
                self._cleanup_after_test(after_test)
            self._give_back_test_lease(n, lease)
            return
        more = boards - len(self.coder_lease.units)
        extra = None
        if more > 0:
            extra = self._take_test_lease(spec, shape, more, first_chip[d.free_boards[0]] if d.free_boards else None)
        h = self._handoff(n, self.coder_lease)
        try:
            lease = h.park()
        except Blocked:
            self._release_test_lease()
            raise
        try:
            self._run_test(spec, stage_dir, test, lease, extra=extra, boards=boards, **record)
        finally:
            self._cleanup_after_test(after_test)
        if extra is not None:
            self._give_back_test_lease(n, extra)
        self._restore_coder(n, h)
        self.coder_lease = h.lease

    def _lab_hardware_phase(self, spec, stage_dir: Path, test: dict, after_test=None) -> None:
        """The hardware phase with a lab box: lease the chips there, run the test there, give the
        chips back. The coder on this box keeps serving the agents throughout."""
        n = spec.number
        self.ledger.append("decision", n, decision="hardware phase", action="lab", where=self._where())
        self._lab_files_up(n)
        lease = reacquire(self.test_adapter, chips=spec.boards * CHIPS_PER_BOARD, who=WHO,
                          reason=f"stage {n} hardware test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", test_lease=lease.record(),
                           where=self._where())
        try:
            self._run_test(spec, stage_dir, test, lease)
        finally:
            self._cleanup_after_test(after_test)           # before the lab lease goes back, as on this box (the sweep looks at this box's docker)
        self._audit_lab_caches(n, stage_dir, self._swap_cache(stage_dir))
        try:
            self.test_adapter.release(lease)
        except AdapterError as exc:
            self._block(n, f"releasing the lab test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _recover_test_leases(self) -> None:
        """Give back test leases an earlier supervisor of this run took and never released (it died, and nothing
        released them). Only leases this run's ledger records, and only those taken where this supervisor runs: a lab
        run gives back the lab's, a local run this box's. A lease that is already gone is noted as not released."""
        # Both trials start servers in sessions of their own, so a SIGKILL of the supervisor (which skips the after-test sweep) can leave either one
        # holding the chips. Stop them before any lease is given back or the coder is recovered.
        for name in (graft.TRIAL_DIR, sampling.TRIAL_DIR):
            trial_dir = self.run_dir / "stages" / "2" / name
            if self.lab is None and trial_dir.is_dir():
                self._stop_trial_servers(2, trial_dir, "when the run started again")
        taken, released = {}, set()
        for e in self.ledger.read():
            d = e["data"]
            if e["event"] != "decision":
                continue
            if d.get("decision") == "test lease taken" and str(d.get("where", "")).startswith("lab") == (self.lab is not None):
                rec = d.get("test_lease") or {}
                if rec.get("lease_id"):
                    taken[rec["lease_id"]] = rec
            elif d.get("decision") == "test lease released":
                released.add(d.get("lease_id"))
        if self.lab is None:
            # A container test the earlier supervisor started can still be running (its client died, the container did not). That
            # includes a test run under the coder's own lease after a park, which records no test lease. Stop it first, and block
            # here when one survives: no lease is given back, and the coder is not recovered or started over it.
            self._sweep(None)
        for lease_id, rec in taken.items():
            if lease_id in released:
                continue
            try:
                self.test_adapter.release(Lease.from_record(rec))
                outcome = "released"
            except AdapterError as exc:
                outcome = f"not released: {exc}"
            self.ledger.append("decision", None, decision="test lease released", lease_id=lease_id,
                               note=f"left by an earlier supervisor of this run; {outcome}", where=self._where())

    def _where(self) -> str:
        return f"lab {self.lab.info.get('host')}" if self.lab is not None else "this box"

    @staticmethod
    def _swap_cache(stage_dir: Path) -> list[str]:
        try:
            cache = json.loads((stage_dir / "swap_config.json").read_text()).get("tt_cache")
        except (OSError, ValueError, AttributeError):
            return []
        return [cache] if isinstance(cache, str) and cache else []

    def _audit_lab_caches(self, n: int, stage_dir: Path, caches: list[str]) -> None:
        """What each tensor cache on the lab holds, kept with the stage's evidence: the caches stay on
        the lab, and stage 8's hazard check reads this file in their place."""
        if self.lab is None or not caches:
            return
        try:
            audit = self.lab.cache_audit(caches)
        except Exception as exc:                       # the audit is evidence, not a gate
            self.ledger.append("notice", n, what="the lab cache audit failed", error=str(exc))
            return
        path = stage_dir / "evidence" / "lab-caches.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            known = json.loads(path.read_text())
        except (OSError, ValueError):
            known = {}
        known.update(audit)
        path.write_text(json.dumps(known, indent=2, sort_keys=True))

    def _run_test(self, spec, stage_dir: Path, test: dict, lease: Lease, *, record: str = "test-result.json",
                  output: str = "evidence/hw-test-output.txt", what: str = "hardware test", env_extra: dict | None = None,
                  extra: Lease | None = None, boards: int | None = None) -> dict:
        n = spec.number
        k = (boards or spec.boards) * CHIPS_PER_BOARD       # a wider test (issue 60) holds more chips than the stage's own
        deadline = min(float(test["deadline_s"]), spec.budget_s)
        out_path = stage_dir / output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if extra is None:
            result = self._spawn(n, test["command"], lease.chips[:k], lease.dev_indices[:k], deadline, out_path,
                                 env_extra=env_extra, lease_id=lease.lease_id)
        else:
            # A test on more boards than the stage's: every chip of both leases, in device order.
            held = sorted((i, c) for each in (lease, extra) for c, i in zip(each.chips, each.dev_indices))
            result = self._spawn(n, test["command"], [c for _, c in held], [i for i, _ in held], deadline, out_path,
                                 env_extra=env_extra, lease_ids=[lease.lease_id, extra.lease_id])
        marker = stage_dir / record
        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what=what, returncode=result["returncode"],
                           timed_out=result["timed_out"], **evidence_record(self.run_dir, marker))
        return result

    def _spawn(self, n: int, command: str, chips, ids, deadline: float, out_path: Path,
               env_extra: dict | None = None, **record) -> dict:
        """Run one hardware test command on `chips` (device indices `ids`) and return its record.
        `record` adds keys to the "hardware test started" decision; `env_extra` adds variables to the test's environment."""
        env = agent_env(self.run_dir, extra=self.extra_env)
        env.update(env_extra or {})
        # The leased chips replace the agent shells' no-chip mask. TT_METAL_VISIBLE_DEVICES takes
        # device indices in another form, so the hardware test gets none and the runtime follows
        # TT_VISIBLE_DEVICES. A container test needs the /dev/tenstorrent indices of the same
        # chips (ORCHARD_DEVICE_IDS) and the label its containers carry (ORCHARD_TEST_LABEL).
        env["TT_VISIBLE_DEVICES"] = ",".join(chips)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        env["ORCHARD_DEVICE_IDS"] = ",".join(str(i) for i in ids)
        env["ORCHARD_TEST_LABEL"] = self.run_label
        # One kernel cache for every run (tt-metal keys it by build). Under the run's HOME it was new for
        # every run, and each configuration's first boot recompiled every kernel (25+ minutes).
        env["TT_METAL_CACHE"] = str(self._kernel_cache())
        if self.lab is None:
            self._kernel_cache().mkdir(parents=True, exist_ok=True)
        self.ledger.append("decision", n, decision="hardware test started", command=command,
                           deadline_s=deadline, chips=list(chips), **record,
                           **({"where": self._where()} if self.lab is not None else {}))
        t0 = self.clock()
        with open(out_path, "wb") as out:
            try:
                if self.lab is not None:
                    code, timed_out = self._spawn_on_lab(n, command, env, deadline, out)
                else:
                    code, timed_out = spawn_checked(command, self.run_dir, env, deadline, out)
            except Denied as exc:
                code, timed_out = None, False
                out.write(f"refused: {exc}\n".encode("utf-8"))
        if (timed_out or code not in (0, None)) and self.lab is None:
            # A test that ran and did not succeed is checked against the bus before anyone calls it a model failure. A refused
            # command (code None and no timeout) never reached a chip.
            how = "timed out" if timed_out else f"exited {code}"
            self._block_if_off_bus(n, f"the hardware test {how}", chips)
        return {"command": command, "returncode": code, "timed_out": timed_out,
                "seconds": round(self.clock() - t0, 3), "chips": list(chips),
                "output": evidence_record(self.run_dir, out_path)}

    def _kernel_cache(self) -> Path:
        return Path(self.paths.cache_root) / "kernels"

    def _lab_files_up(self, n: int) -> None:
        """Copy what stage `n`'s test reads to the lab, before its lease is taken: the run directory, an
        exact copy of stages/<n>, and only the HF repos the test reads (labclient.lab_hf_repos). The
        first copy of a model's weights takes minutes; the lab's chips stay free for others meanwhile."""
        t0 = self.clock()
        stage = self.run_dir / "stages" / str(n)
        self.ledger.append("decision", n, decision="copying files to the lab", where=self._where())
        self.lab.sync_up(self.run_dir)
        self.lab.mirror_up(stage)                                 # no earlier attempt's files in it
        from orchard.labclient import lab_hf_repos
        repos = lab_hf_repos(stage, self.paths.hf_home, self.inputs)
        for repo in repos:
            self.lab.sync_up(repo)
        self.lab.ensure_dir(self._kernel_cache())
        self.ledger.append("decision", n, decision="files copied to the lab", seconds=round(self.clock() - t0, 1),
                           repos=[r.name for r in repos], where=self._where())

    def _spawn_on_lab(self, n: int, command: str, env: dict, deadline: float, out) -> tuple[int | None, bool]:
        """The lab's side of a test, after _lab_files_up has copied what it reads. The command passes
        the runner's checks here first, as a local test's does. The test runs on the lab with its output
        streamed into `out`, and the stage's directory comes back, so the finish step, the gates and the
        ledger hashes read it here."""
        check_string(command, self.run_dir)
        stage = self.run_dir / "stages" / str(n)
        # The output streams into `out` here; the lab's copy of that file is the empty one the sync up
        # carried, so the sync down leaves it out.
        streamed = os.path.relpath(os.path.realpath(out.name), os.path.realpath(stage))
        try:
            return self.lab.run_test(command, cwd=self.run_dir, env=env, timeout=deadline, stdout=out)
        finally:
            self.lab.sync_down(stage, exclude=[] if streamed.startswith("..") else [streamed])

    # ---- a list of hardware tests (stage 4 on the weights-only path) ---------------------------

    def _run_test_list(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        """Prepare the list (unless a resumed stage has its plan), then run every test that has
        no record. Returns (status, reasons, gate) when the stage ends here, or None when the
        finish step should run."""
        n = spec.number
        tests = load_plan(stage_dir) if resumed else None
        if tests is not None and (wrong := swap_draft.stage4_nearest_problems(self.run_dir, stage_dir, tt_model_root=self.paths.tt_model_root,
                                                                              operator_home=self.paths.operator_home,
                                                                              counts=[t.chips for t in tests])):
            return "fail", wrong, None             # a plan saved before the check existed can still name the wrong model's package
        if tests is None:
            self._draft(spec, stage_dir)
            out, prep = self._step(spec, "prepare", stage_dir, escalated, resumed)
            out = self._wrap_up(spec, stage_dir, prep, out)
            if out.status != "done":
                return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
            self._keep_drafted_tests(spec, stage_dir)
            self._keep_drafted_caches(spec, stage_dir)
            tests, problems = self._read_plan(spec, stage_dir)
            if problems:
                return "fail", problems, None
            plan = write_plan(stage_dir, tests)
            self.ledger.append("evidence", n, what="hardware test list",
                               configs=[t.chips for t in tests], **evidence_record(self.run_dir, plan))
        # A test lease left from a block (a container that would not stop) goes back first.
        if not self._release_test_lease():
            self._block(n, "releasing a test lease left from an earlier test failed")
        for test in pending(stage_dir, tests):
            self._run_listed(spec, stage_dir, test)
        write_summary(stage_dir, tests)
        return None

    def _run_listed(self, spec, stage_dir: Path, test) -> None:
        n = spec.number
        if self.lab is not None:
            return self._run_listed_on_lab(spec, stage_dir, test)
        ok, free = check_disk(self.run_dir, TEST_DISK_GB, usage=self.disk_usage)
        if not ok:
            self._block(n, f"the {test.chips}-chip test needs {TEST_DISK_GB} GB free on the run "
                           f"directory's disk; {free} GB is free", need_gb=TEST_DISK_GB, free_gb=free)
        if test.cache in suspect_caches(self.ledger.read(), n):
            aside = move_aside(test.cache)
            if aside is not None:
                self.ledger.append("decision", n, decision="moved a tensor cache aside",
                                   reason="its last test did not exit 0, so it may be part-written",
                                   cache=test.cache, aside=str(aside))
        chips = self.adapter.status()
        d = decide_park(self.coder_lease.units, chips, test.boards)
        self.ledger.append("decision", n, decision="hardware phase", config=test.chips,
                           action=d.action, free_boards=list(d.free_boards),
                           server_boards=list(d.server_boards))
        first_chip = {c.board: c.bdf for c in reversed(chips)}       # each board's first chip
        if d.action == "use_free":
            exact = first_chip[d.free_boards[0]] if test.boards == 1 else None
            lease = self._take_test_lease(spec, test, test.boards, exact)
            try:
                self._run_listed_test(spec, stage_dir, test, [lease])
            finally:
                # Also when the test ends in a block (a board off the PCIe bus found by _spawn): its container may still run.
                self._cleanup_after_test(lambda: self._sweep(n))
            self._give_back_test_lease(n, lease)
            return
        # The coder's boards are needed. Further boards first: if one cannot be had, the coder
        # has not been touched.
        extra = None
        more = test.boards - len(self.coder_lease.units)
        if more > 0:
            exact = first_chip[d.free_boards[0]] if d.free_boards else None
            extra = self._take_test_lease(spec, test, more, exact)
        h = self._handoff(n, self.coder_lease)
        try:
            lease = h.park()
        except Blocked:
            self._release_test_lease()
            raise
        try:
            self._run_listed_test(spec, stage_dir, test, [lease] + ([extra] if extra else []))
        finally:
            self._cleanup_after_test(lambda: self._sweep(n))
        if extra is not None:
            self._give_back_test_lease(n, extra)
        self._restore_coder(n, h)
        self.coder_lease = h.lease

    def _run_listed_on_lab(self, spec, stage_dir: Path, test) -> None:
        """One configuration's test on the lab: the cache checks there, a lease of exactly the chips
        the test needs (a lab's boards may hold one chip each), the test, the sweep, the release."""
        n = spec.number
        lab_chips = getattr(self.lab, "chips", None)
        if lab_chips and test.chips > lab_chips:
            host = self.lab.info.get("host")
            if self.required_chips and test.chips in self.required_chips:
                self._block(n, f"this run requires a {test.chips}-chip test, and the lab {host} has "
                               f"{lab_chips} chips", config=test.chips, lab_chips=lab_chips)
            # An optional configuration the lab cannot hold: recorded as not run, never leased (a lease
            # for more chips than the lab has would wait until the stage budget ran out).
            reason = f"not run: the lab {host} has {lab_chips} chips and this configuration needs {test.chips}"
            path = write_record(stage_dir, test.chips, {"config": test.chips, "command": test.command(n),
                                                        "returncode": None, "timed_out": False, "seconds": 0,
                                                        "chips": [], "not_run": reason})
            self.ledger.append("evidence", n, what="hardware test", config=test.chips, cache=test.cache,
                               returncode=None, timed_out=False, not_run=reason,
                               **evidence_record(self.run_dir, path))
            return
        free = self.lab.disk_free_gb(self.paths.cache_root)
        if free < TEST_DISK_GB:
            self._block(n, f"the {test.chips}-chip test needs {TEST_DISK_GB} GB free on the lab's cache disk; "
                           f"{free} GB is free", need_gb=TEST_DISK_GB, free_gb=free, where=self._where())
        if test.cache in suspect_caches(self.ledger.read(), n):
            aside = self.lab.move_aside(test.cache)
            if aside is not None:
                self.ledger.append("decision", n, decision="moved a tensor cache aside",
                                   reason="its last test did not exit 0, so it may be part-written",
                                   cache=test.cache, aside=str(aside), where=self._where())
        self.ledger.append("decision", n, decision="hardware phase", config=test.chips, action="lab",
                           where=self._where())
        self._lab_files_up(n)
        lease = self._take_test_lease(spec, test, None, None, chips=test.chips)
        self._run_listed_test(spec, stage_dir, test, [lease])
        self._audit_lab_caches(n, stage_dir, [test.cache])
        self._sweep(n)
        self._give_back_test_lease(n, lease)

    def _take_test_lease(self, spec, test, boards, exact, *, chips: int | None = None) -> Lease:
        n = spec.number
        lease = reacquire(self.test_adapter, chips=chips if chips is not None else boards * CHIPS_PER_BOARD, who=WHO,
                          reason=f"stage {n} {test.chips}-chip test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep,
                          exact=exact)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", config=test.chips,
                           test_lease=lease.record(), **({"where": self._where()} if self.lab is not None else {}))
        if test.boards >= 2:
            self._reset_test_chips(n, test, lease)
        return lease

    def _reset_test_chips(self, n: int, test, lease: Lease) -> None:
        """Reset a multi-board test's chips in place, so the test starts on chips reset after their last use.

        The park resets only the coder's board. On 2026-10-13 the 4-chip test failed twice with fabric router
        timeouts on the chips the smaller tests had used, and passed once all four were reset. The cause was
        not isolated, so this is a precaution that costs about 42 s per board, and a refused reset blocks."""
        try:
            self.test_adapter.reset(lease)
        except AdapterError as exc:
            self._block(n, f"resetting the chips for the {test.chips}-chip test failed: {exc}",
                        lease_id=lease.lease_id, config=test.chips)
        self.ledger.append("decision", n, decision="test chips reset", config=test.chips,
                           lease_id=lease.lease_id)

    def _give_back_test_lease(self, n: int, lease: Lease) -> None:
        try:
            self.test_adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _run_listed_test(self, spec, stage_dir: Path, test, leases: list[Lease]) -> None:
        """Run one configuration's test on the first `test.chips` chips of `leases`, in device
        order, and write its record."""
        n = spec.number
        held = sorted((i, c) for lease in leases for c, i in zip(lease.chips, lease.dev_indices))
        chosen = held[:test.chips]
        out_dir = stage_dir / "tests" / str(test.chips)
        out_dir.mkdir(parents=True, exist_ok=True)
        def spawn() -> dict:
            return self._spawn(n, test.command(n), [c for _, c in chosen], [i for i, _ in chosen],
                               min(test.deadline_s, spec.budget_s), out_dir / "output.txt",
                               config=test.chips, cache=test.cache,
                               lease_ids=[lease.lease_id for lease in leases])
        result = spawn()
        if (test.boards >= 2 and result["returncode"] != 0 and not result["timed_out"]
                and result["seconds"] < FAST_FAIL_S):
            # A multi-board test that dies this early failed to start (the 4-chip test's `Fabric Router Sync:
            # Timeout`, seen after 13 to 57 s and gone after the chips were reset together). The cause is not
            # isolated, so reset every chip the test holds and run it once more. Its record is the second try's.
            # The second try writes output.txt again. The first try's text says why it failed, so it is kept beside it.
            first = out_dir / "output-first-try.txt"
            os.replace(out_dir / "output.txt", first)
            self.ledger.append("decision", n, decision="test failed at startup; resetting the test chips and trying once more",
                               config=test.chips, returncode=result["returncode"], seconds=result["seconds"],
                               first_output=evidence_record(self.run_dir, first))
            for lease in leases:
                self._reset_test_chips(n, test, lease)
            result = spawn()
        path = write_record(stage_dir, test.chips, {"config": test.chips, **result,
                                                    "device_ids": [i for i, _ in chosen]})
        self.ledger.append("evidence", n, what="hardware test", config=test.chips, cache=test.cache,
                           returncode=result["returncode"], timed_out=result["timed_out"],
                           **evidence_record(self.run_dir, path))

    def _ledger_survivors(self) -> list[str]:
        """The test containers an earlier sweep of this run could not remove and nothing has since cleared (the ledger's last
        "test containers survived" / "test containers cleared" decision). It lets a restart keep the protection of a lease whose
        container was known to be running, when docker cannot be asked again."""
        ids: list[str] = []
        for e in self.ledger.read():
            if e["event"] != "decision":
                continue
            what = e["data"].get("decision")
            if what == "test containers survived":
                ids = [str(c) for c in (e["data"].get("containers") or [])]
            elif what == "test containers cleared":
                ids = []
        return ids

    def _survived(self, n, ids: list[str], reason: str) -> None:
        self.surviving_containers = list(ids)      # _release_test_lease and _release_from_memory keep the lease they run under
        self.ledger.append("decision", n, decision="test containers survived", containers=list(ids), label=self.run_label)
        self._block(n, reason, containers=list(ids))

    def _cleared(self, n, known: list[str]) -> None:
        self.surviving_containers = []
        if known:
            self.ledger.append("decision", n, decision="test containers cleared", containers=list(known), label=self.run_label)

    def _sweep(self, n: int) -> None:
        """Stop and remove any container this run's tests left running. A container that is
        still listed afterwards blocks the stage: releasing or resetting chips under it is
        refused, and the operator decides. Containers an earlier sweep could not remove are
        remembered in the ledger, so a restart that cannot ask docker blocks on them instead
        of releasing their lease."""
        known = self._ledger_survivors()
        found = self.containers.list(self.run_label)
        if found is None:
            if known:
                self._survived(n, known, "docker could not be asked whether the test containers an earlier sweep could not remove are "
                                         "gone; no test lease was released and the coder was not restored")
            self.ledger.append("notice", n, what="docker could not be asked for test containers",
                               label=self.run_label)
            return
        if not found:
            self._cleared(n, known)
            return
        self.ledger.append("notice", n, what="a hardware test left containers running; stopping them",
                           containers=found, label=self.run_label)
        # Recorded before anything is removed: a signal that lands in the middle of the removal must still find these containers
        # remembered, in memory and in the ledger, until docker confirms an empty listing (_cleared).
        self.surviving_containers = list(found)
        self.ledger.append("decision", n, decision="test containers survived", containers=list(found), label=self.run_label)
        for cid in found:
            self.containers.remove(cid)
        left = self.containers.list(self.run_label)
        if left is None:
            # Docker found containers and then could not say whether they are gone. Nothing confirms the chips are free, so the
            # originals stay remembered (_release_test_lease keeps the lease) and the run blocks.
            self._survived(n, list(found), "docker could not confirm that the test containers stopped after docker stop and rm; no "
                                           "test lease was released and the coder was not restored")
        if left:
            self._survived(n, list(left), "test containers are still running after docker stop and rm; no test "
                                          "lease was released and the coder was not restored")
        self._cleared(n, known or list(found))


# ---- the command line ---------------------------------------------------------------------------

def chip_counts(text: str) -> tuple[int, ...]:
    """A `--required-chips` value: comma-separated positive integers with no repeats."""
    try:
        counts = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated list of integers") from None
    if any(c < 1 for c in counts):
        raise argparse.ArgumentTypeError(f"{text!r} holds a chip count below 1")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError(f"{text!r} repeats a chip count")
    return counts


def parse(argv=None):
    p = argparse.ArgumentParser(prog="python3 -m orchard.supervisor", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start or resume a run")
    r.add_argument("--model", required=True, help="Hugging Face model id")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--tiers", required=True, help="tier config (config/tiers.toml)")
    r.add_argument("--coder-target", required=True, help="tt-model package or bundle the coder serves")
    r.add_argument("--coder-kind", choices=("container", "bundle"), default="container")
    r.add_argument("--coder-profile", default="default")
    r.add_argument("--coder-port", type=int, required=True,
                   help="the port of the chip tier the coder serves (matches its endpoint in --tiers)")
    r.add_argument("--coder-chips", type=int, required=True)
    r.add_argument("--coder-image-id", default=None)
    r.add_argument("--coder-keep-up", action="store_true",
                   help="lab mode: leave the coder serving at the end of the run, and adopt a coder an "
                        "earlier run left up (orchard/kept_coder.py)")
    r.add_argument("--skills-dir", action="append", default=[],
                   help="more skill directories, searched after orchard/skills")
    r.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    r.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                   help="a variable for agent shells (never a credential)")
    r.add_argument("--four-chip-package", default=None, metavar="ORG/NAME",
                   help="the container package stage 4 drafts its 4-chip configuration with, when several "
                        "installed ones serve the nearest model")
    r.add_argument("--required-chips", type=chip_counts, default=None, metavar="N,N",
                   help="chip counts stage 4 must pass, such as 2,4. Any other count it records is "
                        "optional. Without this, every configuration stage 4 lists must pass. The "
                        "ledger records it, and a resumed run keeps it")
    r.add_argument("--cache-root", default=None, metavar="DIR",
                   help="where per-model tensor caches go; skills name it as {{CACHE_ROOT}}. "
                        "Default: <parent of --run-dir>/cache. The ledger records it, and a "
                        "resumed run keeps it")
    r.add_argument("--hf-home", default=None, metavar="DIR",
                   help="the operator's Hugging Face cache; skills name it as {{HF_HOME}}. "
                        "Default: $HF_HOME, else <operator home>/.cache/huggingface. Recorded and "
                        "kept like --cache-root")
    r.add_argument("--operator-home", default=None, metavar="DIR",
                   help="the operator's home, where tt-model keeps its packages; skills name it "
                        "as {{OPERATOR_HOME}}. Default: this user's home from the passwd entry. "
                        "Recorded and kept like --cache-root")
    r.add_argument("--package-format", choices=PACKAGE_FORMATS, default=None,
                   help="stage 7 builds this package on the weights-only path: v6 (a thin bundle). "
                        "v5.1 is refused at start: it needs a container image build. Without this, "
                        "stage 7 is skipped. The ledger records it, and a resumed run keeps it. A "
                        "run that started without it can be given it on a resume before stage 7 "
                        "starts")
    r.add_argument("--package-namespace", default=None,
                   help="the operator's Hugging Face namespace, used in the card and in the "
                        "publish commands (text only; the run never publishes)")
    r.add_argument("--package-models-root", default=None, metavar="DIR",
                   help="where tt-model installs bundles; stage 7 looks here for other chip "
                        "counts of the nearest model. Default: the run's {{TT_MODEL_ROOT}}, "
                        "<operator home>/.cache/tt-model/models")
    r.add_argument("--gozer", default="gozer")
    r.add_argument("--tt-model-root", default=None, metavar="DIR",
                   help="where tt-model installs bundles; skills name it as {{TT_MODEL_ROOT}}. Default: "
                        "<operator home>/.cache/tt-model/models. Recorded and kept like --cache-root")
    r.add_argument("--lab", default=None, metavar="HOST",
                   help="run every hardware test on this ssh host (a lab box) instead of here; the coder "
                        "stays here and is never parked. Needs --lab-root, and every run path under it on "
                        "both boxes. The ledger records it, and a resumed run keeps it")
    r.add_argument("--lab-root", default=None, metavar="DIR",
                   help="the directory that has the same absolute path on this box and the lab (such as "
                        "/srv/orchard); the run directory, caches, Hugging Face cache and bundles live under it")
    r.add_argument("--lab-gozer", default="gozer", help="the gozer command on the lab")
    r.add_argument("--lab-path", action="append", default=[], metavar="DIR",
                   help="a directory to put first on PATH on the lab (repeatable), such as ~/.local/bin")
    r.add_argument("--lab-python", default="python3",
                   help="the Python on the lab that runs the lab helper; hardware tests get the directory "
                        "of --lab-test-python first on PATH")
    r.add_argument("--lab-test-python", default=None, metavar="PATH",
                   help="the interpreter tests run with on the lab (its directory goes first on PATH), such as "
                        "the reference venv under --lab-root")
    r.add_argument("--no-graft-trial", action="store_true",
                   help="skip stage 2's graft trial: with a graft candidate (the model has no mtp.* tensors and the nearest model has "
                        "a head that fits) stage 2 serves the grafted drafter once and keeps it when it is faster; this leaves the "
                        "safe configuration without that extra test (a cold boot adds 4 to 25 minutes). The run records the choice")
    r.add_argument("--sampling", choices=sampling.OVERRIDES, default=None,
                   help="whether this model's purpose needs sampling (issue 107). A speculative drafter can ignore sampled requests, so with 'auto' "
                        "(the default) stage 2 reads the model's own generation_config.json, the model card's recommended settings and its warnings "
                        "against greedy decoding, and for a model that needs sampling whose served configuration does not show sampling applied it "
                        "serves a drafter-off variant once and chooses by measurement. 'needed' and 'not-needed' are the operator's word. The run "
                        "records the choice and a resume keeps it")
    r.add_argument("--resume-aborted", action="store_true",
                   help="go on with a run that an operator or a signal aborted: the ledger records `resume` after the abort and the run starts its coder again and continues from the ledger. Without this flag an aborted run stays aborted and running the command again starts nothing. An abort of any other kind, and a finished run, are left as they are")
    r.add_argument("--unattended", action="store_true",
                   help="never wait for an operator: when the run would pause, name the reason, write "
                        "BLOCKED.md, release the hardware and exit 5. Running the command again retries. "
                        "A pause the operator asked for with `control pause` still waits")
    r.add_argument("--accept-credentials-visible", action="store_true",
                   help="start even though credential files exist in the operator's home "
                        "(agent shells can read them); the ledger records this")
    c = sub.add_parser("control", help="send pause, resume or abort to a running supervisor")
    c.add_argument("--run-dir", required=True)
    c.add_argument("command", choices=Control.COMMANDS)
    st = sub.add_parser("status", help="print the state of a run (read-only; no lock, no writes)")
    st.add_argument("--run-dir", required=True)
    st.add_argument("--json", action="store_true", help="print the facts as one JSON object")
    st.add_argument("--style", choices=("auto", "pretty", "plain"), default=None,
                    help="auto (default): colour and emoji on a capable terminal, plain text otherwise")
    return p.parse_args(argv)


def package_option(args) -> dict | None:
    """The stage 7 options as the ledger records them, or None. Refuses a deferred format and a
    format without a namespace. `models_root` is None when --package-models-root was not given;
    build fills in the run's tt_model_root (orchard/paths.py)."""
    fmt = getattr(args, "package_format", None)
    if fmt is None:
        return None
    if fmt in PACKAGE_DEFERRED:
        raise ValueError(f"--package-format {fmt} is deferred: {PACKAGE_DEFERRED[fmt]}")
    if not args.package_namespace:
        raise ValueError("--package-format needs --package-namespace, the operator's Hugging Face "
                         "namespace for the card and the publish commands")
    root = args.package_models_root
    return {"format": fmt, "namespace": args.package_namespace,
            "models_root": absolute_path(root) if root else None}


def lab_problems(args, run_dir: Path, paths: RunPaths, inputs: dict) -> list[str]:
    """Why a run with a lab cannot start: every path a hardware test touches must be the same path on
    both boxes, so each one must sit under --lab-root."""
    if not getattr(args, "lab_root", None):
        return ["--lab needs --lab-root, the directory with the same absolute path on both boxes"]
    root = absolute_path(args.lab_root).rstrip("/") + "/"
    named = {"the run directory": str(run_dir), "--cache-root": paths.cache_root, "--hf-home": paths.hf_home,
             "--tt-model-root": paths.tt_model_root,
             **{f"--input {k}": v for k, v in inputs.items() if str(v).startswith(("/", "~"))}}
    return [f"{what} {path} is not under the lab root {root.rstrip('/')}; a lab run needs every path it "
            "shares with the lab under the lab root" for what, path in named.items()
            if not (absolute_path(path) + "/").startswith(root)]


def build(args, ledger, *, adapter=None, coder=None, versions=None, http=post_json,
          probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(),
          disk_usage=shutil.disk_usage, home=None, containers=None, environ=None, lab=None) -> Supervisor:
    """A Supervisor from parsed `run` arguments. Tests pass fakes for the machine. `environ`
    supplies $HF_HOME for the default --hf-home (default os.environ)."""
    # Everything that can be refused is checked before any external command runs.
    cfg = load(args.tiers)
    tier = coder_tier(cfg, args.coder_port)
    run_dir = Path(args.run_dir).resolve()
    inputs, extra_env = pairs(args.input, "input"), pairs(args.env, "env")
    agent_env(run_dir, extra=extra_env)
    package = package_option(args)
    found = [str(p) for p in visible_credentials(operator_home() if home is None else home)]
    if found and not args.accept_credentials_visible:
        raise ValueError(
            "agent shells run as this user and could read these credential files: "
            + ", ".join(found) + ". Move them aside for the run, or pass "
            "--accept-credentials-visible to start anyway (the ledger records it)")
    spec = ServerSpec(target=args.coder_target, kind=args.coder_kind, port=args.coder_port,
                      model=cfg.tiers[tier]["model"], profile=args.coder_profile,
                      image_id=args.coder_image_id)
    if adapter is None:
        from orchard.adapters.gozer import GozerAdapter
        adapter = GozerAdapter(gozer=args.gozer)
    if coder is None:
        coder = ServerControl(spec, log_path=str(run_dir / "coder.log"))
    cpu = next(t for t in cfg.tiers.values() if t["placement"] == "cpu")
    entries = ledger.read()
    progress = run_progress(entries)
    if versions is None and not progress.started:
        versions = resolve_versions(spec)       # a resumed run keeps the versions it started with
    required = args.required_chips
    if progress.started:                        # a resumed run keeps the counts it started with
        recorded = tuple((progress.run_start or {}).get("required_chips") or ()) or None
        if required is not None and required != recorded:
            raise ValueError(f"this run started with required chips {recorded}; --required-chips "
                             f"{required} differs. Leave the option out to resume with the recorded value, or move this run's directory "
                             f"aside and start a new run with the counts you want")
        required = recorded
    graft_on = not getattr(args, "no_graft_trial", False)
    if progress.started and (progress.run_start or {}).get("graft_trial") is False:
        graft_on = False                        # a run that opted out of the graft trial stays opted out
    sampling_mode = getattr(args, "sampling", None)
    if progress.started:                        # a resumed run keeps the sampling setting it started with
        recorded_mode = (progress.run_start or {}).get("sampling") or "auto"
        if sampling_mode is not None and sampling_mode != recorded_mode:
            raise ValueError(f"this run started with --sampling {recorded_mode}; --sampling {sampling_mode} differs. Leave the option out to resume "
                             "with the recorded value, or move this run's directory aside and start a new run")
        sampling_mode = recorded_mode
    sampling_mode = sampling_mode or "auto"
    home = operator_home() if home is None else home
    paths = run_paths(args, run_dir, recorded_paths(entries) if progress.started else None,
                      home=home, environ=environ)
    if package is not None and package["models_root"] is None:
        package["models_root"] = paths.tt_model_root
    lab_host = getattr(args, "lab", None)
    if progress.started:                        # a resumed run keeps the lab it started with
        recorded_lab = (progress.run_start or {}).get("lab")
        recorded_host = (recorded_lab or {}).get("host")
        if recorded_host != lab_host:
            was = f"lab mode (lab {recorded_host})" if recorded_host else "local mode"
            now = f"lab mode (lab {lab_host})" if lab_host else "local mode"
            raise ValueError(f"this run started in {was} and would resume in {now}. A run keeps its mode: "
                             f"resume it with --mode {'lab' if recorded_host else 'local'}"
                             + (f" and the lab {recorded_host} in the config" if recorded_host else ""))
    if lab_host:
        from orchard.bringup_config import _SSH_HOST
        if not _SSH_HOST.fullmatch(lab_host):
            raise ValueError(f"--lab {lab_host!r} is not an ssh host such as node4 or user@node4")
        if package is not None:
            raise ValueError("--package-format cannot be used with --lab yet: stage 7's boot check and its "
                             "install would have to run on the lab. Leave it out, or run without --lab")
        problems = lab_problems(args, run_dir, paths, inputs)
        if problems:
            raise ValueError("; ".join(problems))
        if lab is None:
            from orchard.labclient import LabError, LabSide
            try:
                lab = LabSide.connect(host=lab_host, root=absolute_path(args.lab_root), gozer=args.lab_gozer,
                                      path=args.lab_path, python=args.lab_python, test_python=args.lab_test_python)
            except (LabError, OSError) as exc:
                raise ValueError(f"the lab {lab_host} could not be reached or started: {exc}. Check it with "
                                 "`tt-orchard lab setup --check`") from exc
        if containers is None:
            containers = LabelledContainers(run=lab.run_argv)
    added = False
    if progress.started:                        # a resumed run keeps the package it started with
        recorded = package_options(entries)
        stage7_started = any(e["event"] == "stage_start" and e["stage"] == 7 for e in entries)
        if package is not None and package != recorded:
            if recorded is not None or stage7_started:
                raise ValueError(f"this run has package options {recorded}; the options given now "
                                 f"({package}) differ. Leave them out to resume with the recorded "
                                 "ones. Options can be added to a run that has none only before "
                                 "stage 7 starts")
            added = True                        # the run records them before its next stage
        else:
            package = recorded
    return Supervisor(run_dir=run_dir, ledger=ledger, cfg=cfg, model_id=args.model, adapter=adapter,
                      coder=coder, coder_chips=args.coder_chips,
                      standin=ExternalStandIn(cpu["endpoint"], cpu["model"], http=http),
                      skills_dirs=[SKILLS_DIR, *args.skills_dir], inputs=inputs,
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required,
                      home=home, containers=containers, paths=paths, package=package,
                      package_added=added, unattended=getattr(args, "unattended", False), graft_trial=graft_on,
                      sampling=sampling_mode,
                      resume_aborted=bool(getattr(args, "resume_aborted", False)),
                      lab=lab if lab_host else None,
                      four_chip_package=getattr(args, "four_chip_package", None), coder_tier=tier,
                      keep_up=bool(getattr(args, "coder_keep_up", False)),
                      coder_identity={"target": args.coder_target, "kind": args.coder_kind,
                                      "port": args.coder_port, "profile": args.coder_profile,
                                      "image_id": args.coder_image_id, "model": cfg.tiers[tier]["model"],
                                      "chips": args.coder_chips})


PATH_FLAGS = {"--cache-root": "cache_root", "--hf-home": "hf_home", "--operator-home": "operator_home",
              "--tt-model-root": "tt_model_root"}


def run_paths(args, run_dir, recorded: dict | None, *, home, environ=None) -> RunPaths:
    """The run's paths (orchard/paths.py). A new run resolves them from the flags and defaults.
    A resumed run uses the recorded values, and a flag that names a different path is refused,
    as --required-chips is."""
    if recorded is None:
        return RunPaths.resolve(run_dir, home=home, environ=environ, cache_root=args.cache_root,
                                hf_home=args.hf_home, operator_home=args.operator_home,
                                tt_model_root=getattr(args, "tt_model_root", None))
    paths = RunPaths.from_record(recorded)
    for flag, field in PATH_FLAGS.items():
        given, kept = getattr(args, field, None), getattr(paths, field)
        if given is not None and absolute_path(given) != kept:
            raise ValueError(f"this run started with {flag} {kept}; {flag} {given} differs. Leave "
                             "the option out to resume with the recorded value")
    return paths


def main(argv=None, *, home=None) -> int:
    args = parse(argv)
    if args.cmd == "control":
        Control(args.run_dir).write(args.command)
        print(f"wrote {args.command!r} to {Path(args.run_dir) / 'control'}")
        return 0
    if args.cmd == "status":
        # Read-only: orchard/status.py never opens the ledger writer, so it works next to a live run.
        from orchard import status
        return status.main(["--run-dir", args.run_dir] + (["--json"] if args.json else [])
                           + (["--style", args.style] if args.style else []))
    run_dir = Path(args.run_dir)
    try:
        with Ledger(run_dir / "ledger.jsonl") as ledger:
            # Refusals happen here, before anything is recorded or started.
            try:
                sup = build(args, ledger, home=home)
            except (TierConfigError, ValueError, LedgerCorrupt) as exc:
                print(f"refused: {exc}", file=sys.stderr)
                return EXIT_REFUSED
            # From here an error is not a refusal: the run had started, and run() has already
            # tried to release the hardware.
            try:
                code = sup.run()
            except Exception as exc:
                print(f"error: the run stopped on {type(exc).__name__}: {exc}. {sup.stop_message}",
                      file=sys.stderr)
                return EXIT_ERROR
    except (LedgerLocked, LedgerCorrupt) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if sup.stop_message:
        print(sup.stop_message, file=sys.stderr)
    print({EXIT_READY: "ready for operator review", EXIT_ABORTED: "aborted",
           EXIT_BLOCKED: "blocked"}.get(code, code))
    return code


if __name__ == "__main__":
    sys.exit(main())
