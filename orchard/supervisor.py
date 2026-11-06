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
import re
from types import SimpleNamespace
import os
import pwd
import shutil
import stat
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from orchard.adapters import AdapterError, Lease
from orchard.agent import AgentStep, Tools, agent_env, collect_survivors, probe_model, spawn_checked
from orchard import audio_reference_env, audio_runtime, bus, floor_report, nearest, port_cache, safeio, served_chat, stage0_scan
from orchard import full_port_result, runaway, text_runtime
from orchard import stack_pins as stack_pins_mod
from orchard.audio_facts import serving_of
from orchard.blocked import block_code, last_pause
from orchard.blocked import write_bundle as write_blocked_bundle
from orchard.canary import CanaryError, compare, post_json
from orchard.canary import ask as canary_ask
from orchard.commands import run_command
from orchard.context import build_messages, facts_from
from orchard.defaults import (AGENT_CONTINUATION_TURNS, AGENT_MAX_TOKENS, AGENT_RESTARTS_ON_CAPS, AGENT_TOOL_DEDUPE, AGENT_WRAPUP_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, PACKAGE_DEFERRED, PACKAGE_FORMATS,
                              GRAFT_TRIAL_DEADLINE_S, GRAFT_TRIAL_DISK_GB, RUN_CANARY_PROMPT, SAMPLING_TRIAL_DEADLINE_S, SAMPLING_TRIAL_DISK_GB, SAMPLING_TRIAL_RUN_DISK_GB, STOP_TIMEOUT_S, TEST_DISK_GB, FAST_FAIL_S)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (PLAN, failed_tests, load_plan, move_aside, plan_refusal, pending, read_plan, suspect_caches,
                             unrecorded, write_plan, write_record, write_summary, write_test_record)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.package import PackageError, stage_all
from orchard.package import finish as package_finish
from orchard import stock_runtime
from orchard.paths import (PATHS_RECORDED, RunPaths, UnknownPlaceholder, absolute_path,
                           recorded_paths)
from orchard.runner import Denied, check_string
from orchard import graft_trial as graft
from orchard.classes import CLASS_UNKNOWN
from orchard import sampling_choice as sampling
from orchard import kept_coder, kernelprobe, swap_draft, swap_prepare
from orchard.server import ServerChipsHeld, ServerControl, ServerError, ServerNotOwned, ServerSpec, StopCheck
from orchard.stages import (TEXT_MAX_BYTES, coder_start_blocked, AGREEMENT_LEDGER_DECISION, MASK_INPUTS_DECISION, mask_input_entries, REFERENCE_IDS_LEDGER_DECISION, agreement_reference_record,
                            reference_ids_record, AUDIO_STAGE_1, AUDIO_STAGE_2, PACKAGE_OPTIONS_SET, TEXT_SPECS, GateResult, TierUnavailable, abort_can_resume, attempt_started_ts,
                            budget_cap, check_disk, coder_state, delta_class, delta_facts, delta_path, delta_path_reasons, run_facts, evidence_record,
                            open_stage_dir, marker_state, package_format, package_options, resolve_endpoint, SUPERVISOR_MARKER,
                            resolve_skill, run_class, run_path, run_progress, scratch_stage, spec_for, stock_params, tier_for,
                            FULL_PORT_ALLOWED, FULL_PORT_STAGE_2, FULL_PORT_STAGE_3, FULL_PORT_STAGE_4, SKIP_3_FULL_PORT, SKIP_4_FULL_PORT, SCORING_STAGE_1, SCORING_STAGE_2, full_port_enabled, full_port_rehearsal,
                            recorded_by_supervisor, run_model_kind, scoring_routed)
from orchard import full_port, full_port_entries, full_port_quality, full_port_quality_stage, full_port_serve, full_port_serve_stage, full_port_stage4, model_kind, phone_a_friend, port_excerpts, port_facts, port_repair, residency, scoring_check
from orchard import repair_help            # the outside helper inside the port repair rounds (issue 427, run11)
from orchard import token_check            # bounded reasons for the ledger (bound_reasons)
from orchard import raw_evidence            # hashes of the raw files the gates recompute from (issue 445)
from orchard.stages import UNVERIFIED_LEDGER_DECISION
from orchard.tiers import TierConfigError, load
from orchard.watchdog import (DedupeStreak, Event, IdenticalResponses, Ladder, NoFileWritten, NoNewEvidence,
                              RepeatedToolCall, RetryGuard, StageOverBudget, ThinkingWithoutAction,
                              TurnRepeat, Watchdog)

def _put(path, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` by replacing the path atomically (a temporary file beside it, then `os.replace`). A stage directory is
    writable by the agent, so a FIFO or a link may stand where the supervisor writes. `Path.write_text` opens that path and blocks on the
    FIFO or follows the link and truncates its target; this never opens the path, so a FIFO or link there is replaced by the file."""
    safeio.atomic_write(path, text.encode(encoding), mode=0o644)


def _open_fresh(path):
    """Open `path` for writing as a new regular file: whatever stands there (a FIFO, a link) is removed first, never opened."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return open(path, "xb")


RUN_JSON_MAX = 16 * 1024 * 1024          # the largest JSON file of a run (a marker, a record, a trial result) the supervisor reads
MAX_SWAP_CONFIG_BYTES = 1024 * 1024    # choice: a stage's swap_config.json is a few kilobytes (issue 137)

WHO = os.environ.get("ORCHARD_LEASE_WHO") or "orchard:supervisor"       # the name gozer shows for the supervisor's leases; an operator's session may set its own
AGENT = "stage-agent"                 # the launched agent's name; the ladder counts its rungs per stage
EXIT_READY, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED = 0, 2, 3, 4
EXIT_BLOCKED = 5     # an unattended run could not go on and named why (orchard/blocked.py)
FULL_PORT = ("stage 0 found a full port (new model code is needed); plan 4 runs weights-only "
             "bring-ups, so the operator decides whether to go on")
# CLASS_UNKNOWN (the fixed start of the unknown-class pause reason) lives in orchard/classes.py, so status.py can read it first without this module.


def class_unknown_reason(reasons) -> str:
    """The pause reason for stage 0's class `unknown` (issue 437): it says so, and quotes the first reasons from delta.json."""
    shown = "; ".join(" ".join(str(r).split())[:200] for r in list(reasons)[:3])
    return CLASS_UNKNOWN + (f": {shown}" if shown else "") + "; a person decides whether to go on"


def is_full_port_pause(reason) -> bool:
    """Whether a pause reason is the full-port pause (the older text) or the unknown-class pause."""
    return reason == FULL_PORT or (isinstance(reason, str) and reason.startswith(CLASS_UNKNOWN))


STAGE_3_SEED_FILES = 200                  # stage 3 of a full port starts from stage 2's passed port: at most this many files are copied,
STAGE_3_SEED_FILE_BYTES = 2 * 1024 * 1024  # each at most this big (the gate reads a source file up to the same size, full_port.SOURCE_MAX)
STAGE_3_SEED_ENTRIES = 1000               # the scan stops after this many entries of any kind (files, directories, links, caches): the file count alone
                                          # would let a directory of a million subdirectories be read to the end before anything refused it
STAGE_3_SEED_DEPTH = 8                    # and at this many directory levels below the port directory
REHEARSAL_WAIVED = "full-port rehearsal: coherence waived"      # the ledger decision written when stage 2 passed with `coherent` not true
FULL_PORT_UNBUILT = ()      # the stages of a full port that are not built yet: empty, because stages 3 to 6 all exist (issues 326, 454, 329). A non-empty
                            # tuple makes a full-port run pause before stage 5 once (unattended: end blocked), as the first slices did; tests set it
FULL_PORT_STAGES_UNBUILT = ("stage 4 of the full-port route passed or was skipped; stages 5 and 6 for a full-port model are not built yet "
                            "(docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md: stages 5 and 6 are issue 329), so a person "
                            "reviews the result and decides whether to go on")
SCORING_STAGES_UNBUILT = ("stage 2 of the scoring route passed (the model is a reranker or an embedding model: stage 1 recorded Hugging Face scores or "
                          "vectors and stage 2's check compared the port with them); stages 3 to 6 for a scoring model are not built yet "
                          "(docs/superpowers/specs/2026-11-02-scoring-model-check-design.md), so a person reviews the result and decides whether to go on")
SCORING_NOT_ROUTED = ("stage 0 found a scoring model (a reranker or an embedding model) that has no scoring route yet; the route runs for "
                      "class weights-only, and for class full-port with --full-port, when the run has a model= input and no text runtime serves it, "
                      "and the generic stages would compare generated tokens that this model does not produce, so a person decides whether to go on")
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
        # The bounded reader refuses a FIFO or a device link instead of blocking on it (the finish step can replace this file).
        data = safeio.read_json(path, max_bytes=1 << 20)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
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
    if phase == "repair":
        return None                     # a repair round has no output file: the port directory is what changes (orchard/port_repair.py)
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


def bound_reasons(reasons) -> list[str]:
    """The reasons of one gate, as text that is safe to write to the ledger and into a message: each reason cut at `token_check.MAX_REASON_CHARS`
    with the count of what was cut, a value that is not text named by its shape, and at most `token_check.MAX_REASONS_WRITTEN` reasons, then
    one line that says how many reasons and characters were left out. A list that is already short is returned as it is. A gate quotes values
    from files an agent or a model wrote, the ledger refuses to read a file over 16 MiB, and a refusal that makes the ledger unreadable ends
    the run (review of PR 444, limit round)."""
    items = list(reasons)
    keep = items[:token_check.MAX_REASONS_WRITTEN]
    out = [token_check.cut_text(r) if isinstance(r, str) else token_check.show(r) for r in keep]
    if len(items) > len(keep):
        left = items[len(keep):]
        chars = sum(len(r) if isinstance(r, str) else 0 for r in left)
        out.append(f"{len(left)} more reasons ({chars} characters) are not shown; the gate gave {len(items)} reasons in all")
    return out


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


PCI_ADDRESS = re.compile(r"^([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$")


def lease_bdfs(lease) -> tuple:
    """The PCI addresses of a lease's chips as sortable numbers, or () when the lease has none or any is not an address.
    A lease already carries them (`chips`), so no device is opened."""
    out = []
    for chip in getattr(lease, "chips", ()) or ():
        m = PCI_ADDRESS.match(str(chip))
        if m is None:
            return ()
        out.append(tuple(int(g, 16) for g in m.groups()))
    return tuple(out)


def reset_order(leases) -> tuple[list, str]:
    """The order in which a multi-board test's separate leases are reset, and the key it used: the lease whose chips have the
    lowest PCI address goes last ("pci_address"). When a lease has no usable address the lowest device index decides
    ("device_index"), and with neither the list order is kept ("list_order"). One key is used for every lease.

    Issue 26, measured on 2026-11-03 on one two-board box (a 1x4 mesh open with the 1-D fabric and strict init, from a release and
    a fresh lease each time): two leases with the lowest PCI address reset last failed 0 of 12 trials, a single lease reset once
    failed 0 of 11, and two leases with the lowest PCI address reset first failed 6 of 12 with `Fabric Router Sync: Timeout after
    10000 ms on Device 0`. On that box the lowest device index, the lowest PCI address and the higher serial first all give the
    same order, so the trials do not tell them apart. The PCI address is used because it follows the physical slot. A device
    index is a position the driver assigns in probe order. No reboot and no second box has been tried (tt-gozer issue 4, stable
    names for boards and chips, is the longer-term fix). The list it is given is not changed."""
    leases = list(leases)
    bdfs = [lease_bdfs(lease) for lease in leases]
    if all(bdfs):
        keyed, key = [min(b) for b in bdfs], "pci_address"
    elif all(getattr(lease, "dev_indices", ()) for lease in leases):
        keyed, key = [min(lease.dev_indices) for lease in leases], "device_index"
    else:
        return leases, "list_order"
    order = sorted(range(len(leases)), key=lambda i: keyed[i], reverse=True)
    return [leases[i] for i in order], key


# ---- the supervisor -----------------------------------------------------------------------------

BUNDLE_BUILD_TIMEOUT_S = 600            # build_bundle.py reads files and copies packages; no network
SWAP_TEMPLATES = ("prepare_swap.py", "serve_and_compare.py", "prompt_suite.json")      # copied with a drafted swap_config.json
AGREEMENT_TEMPLATES = ("agreement_core.py", "agreement_prompts.json")      # also copied when the config sets `agreement_set` (issue 15)
STAGE4_TEMPLATES = (*SWAP_TEMPLATES, "serve_and_compare_container.py")     # into each drafted configs/<N>
GRAFT_TRIAL_ENV = "ORCHARD_TRIAL_DIR"      # in the environment of the trial's test: the trial directory, so its servers can be found
GRAFT_TRIAL_TEMPLATE = "graft_trial.py"     # copied into stages/2/graft-trial/ when the trial starts (orchard/graft_trial.py)
SAMPLING_TRIAL_TEMPLATE = "sampling_trial.py"   # copied into stages/2/sampling-trial/ when the drafter-off trial starts (orchard/sampling_choice.py)


def write_reference_excerpts(ledger, run_dir: Path, spec, stage_dir: Path, snapshot, root) -> tuple:
    """Write `stages/2/reference_excerpts.md` (orchard/port_excerpts.py) and say so in the ledger. Returns (written, the summary the facts file
    points at). Advisory: a config that cannot be read, a missing or hostile reference module and a write error each leave no file, a ledger
    decision that names the reason, and the coder works from the facts file and its skill as before."""
    try:
        result = port_excerpts.for_root(root, port_facts.read_config(snapshot))
        if not result.text:
            ledger.append("decision", spec.number, decision="reference excerpts not written", problem=result.reason[:300])
            return False, None
        record = evidence_record(run_dir, port_excerpts.write(stage_dir, result.text))
    except Exception as exc:
        ledger.append("decision", spec.number, decision="reference excerpts not written",
                      problem=f"{exc.__class__.__name__}: an unexpected error while writing the excerpts")
        return False, None
    ledger.append("decision", spec.number, decision="reference excerpts written", excerpts=record, bytes=len(result.text.encode("utf-8")),
                  classes=result.classes[:port_excerpts.MAX_CLASSES], left_out=len(result.left_out))
    return True, result.summary()


class _SeedLimit(Exception):
    """A bound of the port scan was passed; the text is the reason the ledger records."""


class _SeedScan:
    """The entries of a port directory, read lazily, depth first, through descriptors, with every bound enforced as the entries arrive (issue 326).

    The root is opened once and held (`safeio.pinned_root`: a link standing where the root should be is refused), and every directory below it is
    opened beneath that descriptor with no link followed and listed from its own descriptor. A directory renamed or replaced by a link after it
    was classified is therefore still the one that is listed, and a link in its place is refused when the walk reaches it. Nothing is listed or
    sorted first: an entry is asked of the operating system only when the caller wants the next one, so a directory of millions of entries is
    refused after about `max_entries` reads and holds one descriptor per level (at most `max_depth` + 2). A link is reported and never followed;
    a cache directory (`__pycache__`) is reported nowhere and never entered. Yields `(parts, kind, identity)` with `parts` the names from the
    root, `kind` one of `file`, `link` or `other` (a device, a pipe, a socket) and `identity` the `(st_dev, st_ino)` of a file as listed (None
    otherwise); directories are walked, not yielded. `read` opens a listed file beneath the same root descriptor and refuses a file that is
    no longer the one that was listed. `seen` is the number of entries read so far. The order is the operating system's: callers that need a
    stable order sort what they kept. Close the iterator (`contextlib.closing`) to release the descriptors."""

    def __init__(self, root, max_entries: int = STAGE_3_SEED_ENTRIES, max_depth: int = STAGE_3_SEED_DEPTH, *, anchor=None, parts=()):
        """With `anchor` (a `safeio.Pin` on the run directory) and `parts` (the names from it to the port directory), the root is opened
        beneath the anchor with no link followed at any name, so a link standing where `stages/2` should be is refused (review of PR 455,
        round 4). Without them `root` is opened by path, and only its last name is protected."""
        self.root, self.max_entries, self.max_depth, self.seen = os.fspath(root), max_entries, max_depth, 0
        self.anchor, self.anchor_parts = anchor, tuple(parts)
        self._pin = None

    def read(self, parts, identity, max_bytes: int) -> bytes:
        """The bytes of the file `parts` as it was listed (identity `(st_dev, st_ino)`), at most `max_bytes`; only while the iterator is open."""
        if self._pin is None:
            raise safeio.PathRefused("the port directory is not open")
        return safeio.read_beneath(self._pin.fd, parts, max_bytes=max_bytes, expect_identity=identity)

    def __iter__(self):
        with (self.anchor.child(self.anchor_parts) if self.anchor is not None else safeio.pinned_root(self.root)) as pin:
            self._pin = pin
            try:
                stack = [((), 0)]
                while stack:
                    rel, depth = stack.pop()
                    with safeio.list_beneath(pin.fd, rel) as listing:
                        for entry in listing:
                            self.seen += 1
                            if self.seen > self.max_entries:
                                raise _SeedLimit(f"holds more than {self.max_entries} entries")
                            parts = rel + (entry.name,)
                            if entry.is_symlink():
                                yield parts, "link", None
                            elif entry.is_dir(follow_symlinks=False):
                                if entry.name == "__pycache__":
                                    continue
                                if depth + 1 > self.max_depth:
                                    raise _SeedLimit(f"is nested deeper than {self.max_depth} levels")
                                stack.append((parts, depth + 1))
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                yield parts, "file", (st.st_dev, st.st_ino)
                            else:
                                yield parts, "other", None
            finally:
                self._pin = None


def _holds_python(root, *, anchor=None, parts=()) -> bool | None:
    """Whether a directory holds a `.py` file within the scan bounds; None when the bounds were passed first (too large to say). `anchor` and
    `parts` are those of `_SeedScan`."""
    scan = _SeedScan(root, anchor=anchor, parts=parts)
    try:
        with contextlib.closing(iter(scan)) as entries:
            return any(kind == "file" and parts[-1].endswith(".py") for parts, kind, _identity in entries)
    except _SeedLimit:
        return None
    except (OSError, safeio.SafeReadError):          # a link standing where the directory should be, or one that cannot be opened: it holds no port
        return False


def _clear_port_link(run, stage_parts, *, remove_tree: bool = False) -> None:
    """Remove a link standing at `port` in the stage directory `stage_parts` (names beneath the pin `run` of the run directory), and with
    `remove_tree` a real `port` directory too. The stage directory is opened with no link followed at any name and the removal goes through its
    descriptor, so a link planted at `stages/3` makes this do nothing and nothing outside the run is touched. It never raises: a failure leaves
    what stands there, and the writes that follow refuse it."""
    try:
        fd = safeio.open_beneath(run.fd, stage_parts)
    except safeio.SafeReadError:
        return
    try:
        try:
            mode = os.lstat("port", dir_fd=fd).st_mode
        except OSError:
            return
        try:
            if stat.S_ISLNK(mode):
                os.unlink("port", dir_fd=fd)
            elif remove_tree and stat.S_ISDIR(mode):
                shutil.rmtree("port", ignore_errors=True, dir_fd=fd)
        except OSError:
            pass
    finally:
        os.close(fd)


def _stage_3_port_is_held(sup, run, dst_parts, spec) -> bool:
    """Whether the stage 3 port directory already holds a port (a resumed stage) and so is kept, read beneath the run directory's descriptor:
    a link standing at any name of the path holds nothing. Sets `_port_seed` and writes the ledger line when the answer is yes or when the
    directory is too large to say."""
    held = _holds_python(sup.run_dir / Path(*dst_parts), anchor=run, parts=dst_parts)
    if held is None:                 # the coder's directory is past the scan bounds: it is left as it is, and nothing is copied over it
        sup.ledger.append("decision", spec.number, decision=f"stage {spec.number} port directory too large to scan",
                           limits={"entries": STAGE_3_SEED_ENTRIES, "depth": STAGE_3_SEED_DEPTH})
    if held is None or held:
        sup._port_seed = {"copied": 0, "kept": True}
        return True
    return False


class Supervisor:
    REPAIR_HELP_DEFAULT = True       # the helper is asked inside the port repair rounds when it is configured (a test of the older behaviour turns it off)

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
                 resume_aborted: bool = False, sampling: str = "auto", full_port: bool = False, full_port_added: bool = False,
                 full_port_rehearsal: bool = False, stack_pins=None, kind_override: str | None = None,
                    port_repair_rounds: int = port_repair.DEFAULT_ROUNDS, telegraph=None, tool_dedupe: bool = AGENT_TOOL_DEDUPE, repair_help: bool | None = None):
        self.run_dir = Path(run_dir).resolve()
        # The helper inside the repair rounds (orchard/repair_help.py): on when the helper is, unless `--no-repair-help` or the caller says no.
        self.repair_help = self.REPAIR_HELP_DEFAULT if repair_help is None else bool(repair_help)
        # The outside helper (orchard/telegraph.py, a `Telegraph`), or None. Full-port stage 2 phones it when attempts keep failing
        # (orchard/phone_a_friend.py). None is the earlier behaviour: nothing is sent anywhere.
        self.telegraph = telegraph
        # `--port-repair-rounds` (issue 420): the repair rounds a full-port stage 2 attempt gives the coder after a port check that failed on the
        # port itself. Recorded in run_start; `build` hands over the recorded value on a resume. 0 is the earlier behaviour: no rounds.
        self.port_repair_rounds = port_repair.check_rounds(port_repair_rounds)
        # `--no-tool-dedupe`: the agent's tools answer a repeat of an unchanged read with a note (orchard/agent.py Tools, docs/tool-result-dedupe.md). Recorded in
        # run_start only when off; `build` hands over the recorded value on a resume.
        self.tool_dedupe = bool(tool_dedupe)
        self._repair_baseline: dict | None = None      # the port directory's fingerprint when a repair step starts (the step's `finished` check)
        # Stage 0's stack freshness record (orchard/stack_pins.py): `(stage_dir, new_run) -> (record, sha256)`. None means no comparison was
        # supplied (the tests), and the record says nothing was compared. `_run_resumed` is set when a run starts on a ledger that has begun.
        self.stack_pins, self._run_resumed = stack_pins, False
        # `--model-kind` (orchard/model_kind.py): the operator's word for what the model produces, used when stage 0 passes. The ledger's stage 0
        # `stage_end` holds the answer and its source, so a resume without the flag keeps it.
        self.kind_override = kind_override
        # `--full-port` (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md): the operator lets a class `full-port` model go on
        # into the full-port stage 2. The ledger holds the answer (run_start, or a decision when the flag came on a resume); this is the
        # request. `full_port_added` is set when the flag came on a resume of a run that started without it.
        self.full_port, self.full_port_added = full_port, full_port_added
        # `--full-port-rehearsal` (issue 357): the model is a rehearsal with random weights, so stage 2's `coherent` field is waived. Recorded
        # in run_start only; the gate reads the ledger's answer, never this request.
        self.full_port_rehearsal = full_port_rehearsal
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
        self.surviving_helpers: list = []            # identities (pid, boot id, start) of hardware-test helpers a sweep could not stop
        self.surviving_helper_leases: list = []      # ids of the leases those helpers ran under; empty means every lease
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
        alive = self._helpers_holding(lease.lease_id)
        if alive:
            # A helper of the hardware test that the sweep could not stop may still have a chip open: the same reason as a container.
            self.ledger.append("notice", None, what="the test lease was kept: a helper of the hardware test is still running",
                               lease_id=lease.lease_id, helpers=alive)
            return False
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
            if self._memory_helpers_hold(lease.lease_id):
                ok = False             # a helper of a hardware test still uses these chips: this lease stays, an unrelated one goes back
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
                    self._remove_coder_container(record=False)
                (self.adapter if is_coder else self.test_adapter).release(lease)
            except (AdapterError, ServerError, OSError):
                ok = False
        return ok

    def _run(self) -> int:
        p = run_progress(self.ledger.read())
        self._run_resumed = bool(p.started)
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
                               graft_trial=self.graft_trial, sampling=self.sampling, port_repair_rounds=self.port_repair_rounds,
                               **({"tool_dedupe": False} if not self.tool_dedupe else {}),
                               **({"full_port": True} if self.full_port else {}),
                               **({"full_port_rehearsal": True} if self.full_port_rehearsal else {}),
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
        if self.full_port_added and not full_port_enabled(self.ledger.read()):
            # The flag came on a resume of a run that started without it (build checked that stage 2 has not started). Recorded once.
            self.ledger.append("decision", None, decision=FULL_PORT_ALLOWED, by="--full-port")
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
            if p.next_stage == 1 and self._scoring_unrouted():
                self.ledger.append("decision", 1, decision="pause", reason=SCORING_NOT_ROUTED)
                continue
            if p.next_stage == 2 and self._full_port_unacknowledged():
                reason = (class_unknown_reason(delta_path_reasons(self.run_dir)) if run_class(self.ledger.read(), self.run_dir) == "unknown"
                          else FULL_PORT)
                self.ledger.append("decision", 2, decision="pause", reason=reason)
                continue
            if p.next_stage == 5 and self._full_port_stages_unbuilt():
                self.ledger.append("decision", 5, decision="pause", reason=FULL_PORT_STAGES_UNBUILT)
                continue
            if p.next_stage == 3 and self._scoring_stages_unbuilt():
                self.ledger.append("decision", 3, decision="pause", reason=SCORING_STAGES_UNBUILT)
                continue
            # The spec follows the path stage 0 chose, read from the ledger each time, so a
            # resumed run picks the same skill and gate as the run that crashed.
            entries = self.ledger.read()
            spec = spec_for(p.next_stage, run_path(entries, self.run_dir), package_format(entries),
                            run_class(entries, self.run_dir), stock=stock_params(entries) is not None,
                            audio_runtime=audio_runtime.selected(entries) is not None,
                            text_runtime=text_runtime.selected(entries) is not None, full_port=full_port_enabled(entries),
                            kind=run_model_kind(entries))
            if spec is FULL_PORT_STAGE_3 and self._stage_3_chips() is None:
                spec = dataclasses.replace(spec, skip=SKIP_3_FULL_PORT)       # stage 2's board is the configured chip count: nothing new to show
            if spec is FULL_PORT_STAGE_4 and not self.required_chips:
                spec = dataclasses.replace(spec, skip=SKIP_4_FULL_PORT)       # no chip count was listed: stage 4 has no entry
            try:
                if spec in (FULL_PORT_STAGE_2, SCORING_STAGE_2):
                    # a model that cannot fit the box starts no prepare step (issue 327)
                    self._check_residency(spec, hardware_pending=self._hardware_pending(
                        spec, resuming=p.open_stage == p.next_stage, escalated=p.next_stage in p.escalated))
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
        if text_runtime.selected(entries) is not None:
            return False          # the harness's text runtime is the full port for this model (orchard/text_runtime.py)
        if full_port_enabled(entries) and run_class(entries, self.run_dir) == "full-port":
            return False          # the operator started the run with --full-port; `unknown` and `restricted-head` keep their pauses
        # Unattended, a full port always blocks: an operator's resume is what lets an attended run go on
        # into the full-port stages, and no unattended retry may do that. A pause for a restricted head
        # counts as the same acknowledgement, so one resume is enough.
        return self.unattended or not any(
            e["event"] == "decision" and (is_full_port_pause(e["data"].get("reason")) or e["data"].get("reason") == RESTRICTED_HEAD) for e in entries)

    def _full_port_stages_unbuilt(self) -> bool:
        """Whether a full-port run pauses after stage 4 because the stages after it are not built. Stages 3 (the port on the configured chips),
        4 (each required chip count), 5 (the chat server with the acceptance corpus) and 6 (quality and benchmark) are all built, so no
        full-port run pauses here any more (issues 326, 454 and 329). `FULL_PORT_UNBUILT` is empty; while it is not, a run the operator allowed to
        port (`--full-port`) pauses once after stage 4, so one attended resume goes on, and unattended it always blocks."""
        if not FULL_PORT_UNBUILT:
            return False
        entries = self.ledger.read()
        if not full_port_enabled(entries) or run_class(entries, self.run_dir) != "full-port":
            return False
        if text_runtime.selected(entries) is not None:
            return False
        if self._scoring_route()[1]:
            return False          # a scoring model of class full-port has its own pause after stage 2 (`_scoring_stages_unbuilt`, code scoring-route)
        return self.unattended or not any(
            e["event"] == "decision" and e["data"].get("reason") == FULL_PORT_STAGES_UNBUILT for e in entries)

    def _scoring_route(self) -> tuple[str, bool]:
        """(the run's model kind, whether the scoring route takes it), read from the ledger. A generative model never takes it. A run that
        a text runtime serves (`spec_for` gives that runtime's stages precedence) and a run with no `model=` input (the check has no snapshot
        to load, and the operator's input is the only snapshot it trusts) are not routed either."""
        entries = self.ledger.read()
        kind = run_model_kind(entries)
        routed = (scoring_routed(kind, run_path(entries, self.run_dir), run_class(entries, self.run_dir), full_port_enabled(entries))
                  and text_runtime.selected(entries) is None)
        return kind, routed

    def _scoring_unrouted(self) -> bool:
        """Stage 0 found a reranker or an embedding model whose class has no scoring route, and the run has not yet paused for it. Unattended,
        it always blocks, as the other unbuilt routes do."""
        entries = self.ledger.read()
        kind, routed = self._scoring_route()
        # Without a `model=` input the check has no snapshot to load. This is judged before stage 1 only, from this invocation's inputs: once
        # stage 2 has passed, the route (and the pause after it) follows the ledger, so a resume that omits the input cannot skip the pause.
        if kind not in model_kind.SCORING_KINDS or (routed and self.inputs.get("model")):
            return False
        return self.unattended or not any(e["event"] == "decision" and e["data"].get("reason") == SCORING_NOT_ROUTED for e in entries)

    def _scoring_stages_unbuilt(self) -> bool:
        """A run on the scoring route has passed stage 2, and the stages after it are not built. It pauses once, so one attended resume goes
        on into the generic stages. Unattended, it always blocks."""
        entries = self.ledger.read()
        if not self._scoring_route()[1]:
            return False
        return self.unattended or not any(e["event"] == "decision" and e["data"].get("reason") == SCORING_STAGES_UNBUILT for e in entries)

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
        held = self._helpers_holding(lease.lease_id)
        if held:
            # A hardware test parked the coder and ran under this lease; a helper it left behind may still have a chip open.
            self.ledger.append("notice", None, what="the coder's lease was kept: a helper of a hardware test that ran under it is still running",
                               lease_id=lease.lease_id, helpers=held)
            return False
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
            self._remove_coder_container()
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
        if (lease_rec is None or lease_rec["lease_id"] in released or boot_unfinished(entries) or coder_start_blocked(entries)
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
        except ServerError:
            return None
        read = safeio.read_text_checked(self.run_dir / canary["path"], max_bytes=RUN_JSON_MAX)
        if not read.ok:
            if read.refused:
                self.ledger.append("decision", None, decision="a run file was refused and not used", file=str(canary["path"]), reason=read.reason)
            return None
        canary_text = read.value
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
        alive = self._helpers_running()
        if alive:
            self._block(None, "a helper of an earlier hardware test is still running (pid " + ", ".join(str(r.get("pid")) for r in alive) +
                        "), so the coder is not restored over it; stop it and run the command again", pids=[r.get("pid") for r in alive])
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
        if coder_start_blocked(entries):
            # No coder was launched. Nothing is taken from the server record, so a foreign server on the coder's port is never used.
            self._retry_blocked_start(entries, Lease.from_record(lease_rec), canary)
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

    def _retry_blocked_start(self, entries: list[dict], lease: Lease, canary: dict | None) -> None:
        """The last start was blocked by the granted-board check (`coder start blocked`): retry the check and launch. When this process still
        holds the recorded lease the launch is under it; otherwise (a restart: the lease belonged to a dead pid or was released) a lease is
        taken and checked. No boot is completed, no canary compared and no coder death counted: there is no coder yet."""
        self.ledger.append("decision", None, decision="retrying a blocked coder start", lease_id=lease.lease_id)
        released = {e["data"].get("lease_id") for e in entries if e["event"] == "decision"
                    and e["data"].get("decision") in ("hardware released", "coder left up")}
        mine = [c for c in self.adapter.status() if c.bdf in lease.chips]
        ours = (lease.lease_id not in released and bool(mine) and all(c.lease_pid == self.adapter.owner_pid for c in mine)
                and lease.lease_id not in self._adopted_ids(entries))
        self._start_coder(canary, lease=lease if ours else None, check_granted=True)

    def _start_coder(self, baseline: dict | None, lease: Lease | None = None, *, check_granted: bool = False) -> None:
        """Take a lease if none is given, and start the coder under it. `check_granted` (the relaunch after a resume) asks
        docker about the devices of the lease that was granted before the start: gozer can grant a different board from
        the board the relaunch's stop check looked at (issue 332)."""
        if lease is None:
            lease = reacquire(self.adapter, chips=self.coder_chips, who=WHO, reason="coder",
                              ledger=self.ledger, stage=None, wait_budget_s=COLD_BOOT_BUDGET_S,
                              clock=self.clock, sleep=self.sleep)
        self.coder_lease = lease          # known from here on, so a shutdown can release it
        if check_granted and hasattr(self.coder, "limit_devices"):
            # Before the record, so the pending record pairs the granted lease with the granted board's devices. A stale filter
            # (the board the stop check looked at) would let cleanup release the granted board with a foreign container on it.
            self.coder.limit_devices(lease.dev_indices)
        # Recorded before the start: a crash during the boot leaves the lease and server in the
        # ledger, so the restart can stop that server and re-lease.
        # The pending record says "unknown" for this attempt. The live value changes only when the
        # start begins (ServerControl.start), so a failed append here cannot lose an earlier refusal.
        if check_granted:
            # Before "coder starting": a block here has launched nothing, and a "coder starting" record would tell a resume that a launch
            # happened (it could take a foreign server that answers, or count the missing container as a coder death; issue 351).
            # "coder start pending" is written before the check, so a signal or error during the check still leaves the granted lease in
            # the ledger for cleanup. "coder start blocked" follows a failed check. Both carry owned=False: this attempt owns no container,
            # so cleanup releases the lease without stopping or resetting anything (a container on the granted board is another
            # session's). A resume retries the check and the launch (`_retry_blocked_start`). The check only reads docker.
            unowned = {**self.coder.record(), "owned": False}
            if hasattr(self.coder, "owned"):
                self.coder.owned = False          # ServerControl.start sets it again for a real start
            self.ledger.append("decision", None, decision="coder start pending", lease=lease.record(), server=unowned)
            check = self.coder.confirm_stopped()
            if not check.stopped:
                self.ledger.append("decision", None, decision="coder start blocked", lease=lease.record(), server=unowned,
                                   reason=f"a container holds a device of the granted board ({', '.join(lease.chips)})")
                self._block(None, f"a container holds a device of the granted board ({', '.join(lease.chips)}); the coder was "
                                  "not started. Stop that container or wait for it, then resume",
                            checks=check.checks, evidence=check.evidence)
        self.ledger.append("decision", None, decision="coder starting", lease=lease.record(),
                           server={**self.coder.record(), "owned": None})
        before = self._coder_container_ids()      # the containers under the coder's name before this attempt: none of them is this run's
        try:
            try:
                self.coder.start(lease)
            except ServerChipsHeld as held:
                # `tt-model serve` counts a stopped container as holding its chips. When this run launched a coder before (a retry), a
                # stopped container under the coder's name is that launch's leftover: remove it and start once more. Anything else is
                # not ours to remove, and the block says what holds the chips and how to clear it (issue 417).
                if not self._remove_own_leftover(lease):
                    self._block_chips_held(held, lease)
                self.coder.start(lease)
            self._note_created_container(lease, before)
        except ServerChipsHeld as exc:
            self._block_chips_held(exc, lease, retried=True)
        except ServerNotOwned as exc:
            # Another process owns the container of this name. Record that, so no later stop (this
            # run's cleanup, a resume) touches it, and block with the usual coder reason.
            self.ledger.append("decision", None, decision="coder not started", lease_id=lease.lease_id,
                               server=self.coder.record(), reason=str(exc))
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
        except (ServerError, OSError) as exc:
            self._note_created_container(lease, before)       # tt-model may have made the container before it failed: record it for the cleanup
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
            read = safeio.read_text_checked(self.run_dir / baseline["path"], max_bytes=RUN_JSON_MAX)
            if not read.ok:
                self._block(None, "the coder's baseline canary answer cannot be read, so the answer after the restart cannot be compared "
                                  f"with it ({read.reason or 'the file is missing'})", before_file=baseline)
            before = read.value
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
        # A resume may use a smaller coder profile than the old lease (four chips, now one board).
        # The stop check then covers the chips the new coder will hold: the first `coder_chips` of the
        # old lease. A container that another session runs on one of the rest is left alone. The old
        # lease itself is still released whole below.
        watched = old
        # gozer grants whole boards, so a one-chip request holds a two-chip board: round up to CHIPS_PER_BOARD.
        n = -(-int(self.coder_chips) // CHIPS_PER_BOARD) * CHIPS_PER_BOARD
        if 0 < self.coder_chips and n < len(old.chips):
            watched = Lease(old.lease_id, old.chips[:n], old.dev_indices[:n], old.env, old.units)
            if hasattr(self.coder, "limit_devices"):
                self.coder.limit_devices(watched.dev_indices)
            self.ledger.append("decision", None, decision="stop check narrowed to the new coder's chips",
                               old_chips=list(old.chips), checked_chips=list(watched.chips))
        if not self.coder.confirm_stopped().stopped:
            self.ledger.append("decision", None, decision="stopping the coder", result=self.coder.stop())
        check = wait_stopped(self.coder, self.adapter, watched, quiet_wait_s=self.budgets.quiet_wait_s,
                             poll_s=self.budgets.poll_s, clock=self.clock, sleep=self.sleep,
                             accept=("CLAIMED", "STALE", "FREE"))
        if not check["ok"]:
            self._block(None, "the coder could not be confirmed stopped for the relaunch", **check)
        self._remove_coder_container()
        if old.lease_id in self._adopted_ids(self.ledger.read()):
            self._release_kept_lease(old)
        self._start_coder(canary, check_granted=True)

    def _remove_coder_container(self, record: bool = True) -> None:
        """After the coder's stop is confirmed, remove its stopped container so the next `tt-model serve` does not find it holding the chips
        (issue 417). Best effort: the stop already succeeded and the lease must still be released, so no failure of the removal or of the
        ledger write that records it is allowed to escape. A control without the method (a fake, a bundle) does nothing. `record=False` is the
        ledger-free shutdown (`_release_from_memory`), which writes nothing because the ledger is what failed."""
        remove = getattr(self.coder, "remove_container", None)
        if remove is None:
            return
        try:
            rows = getattr(self.coder, "find_container_rows", None)
            if getattr(self.coder, "owned", None) is False:
                return                      # another process owns the container
            if getattr(self.coder, "owned", None) is True or rows is None:
                result = remove()           # its `tt-model serve` exited 0, so the container under the name is this run's
            else:
                # The owner is unknown (a start that failed, or a record without the field): remove only a container whose id this run
                # recorded as created. One that was there before the attempt is not proven to be this run's.
                found = rows() or []
                created = self._known_created_ids(use_ledger=record)
                if not found or any(cid not in created for _, _, cid in found):
                    return
                result = remove(only_id=found[0][2])
        except Exception as exc:            # best effort: a docker or fake failure must not stop the release that follows
            if record:
                self._note("notice", what="removing the stopped coder container failed", error=str(exc))
            return
        if result and record:
            self._note("decision", decision="coder container removed" if result.get("removed") else "coder container left", result=result)

    def _note(self, event: str, **data) -> None:
        """A ledger entry that may fail without failing the caller (a full disk is when these notes matter least)."""
        try:
            self.ledger.append(event, None, **data)
        except Exception:
            pass

    def _coder_container_ids(self) -> set[str] | None:
        """The ids of the containers under the coder's exact name now, or None when docker cannot say (nothing is then provable)."""
        rows = getattr(self.coder, "find_container_rows", None)
        if rows is None:
            return None
        try:
            found = rows()
        except Exception:
            return None
        return None if found is None else {cid for _, _, cid in found if cid}

    def _known_created_ids(self, use_ledger: bool = True) -> set[str]:
        """The ids this run created under the coder's name: those noted by this process, and with `use_ledger` those the ledger holds from
        an earlier process of the run."""
        known = set(getattr(self, "_created_ids", ()))
        if use_ledger:
            try:
                known |= {e["data"].get("container_id") for e in self.ledger.read()
                          if e["event"] == "decision" and e["data"].get("decision") == "coder container created"}
            except Exception:
                pass
        known.discard(None)
        return known

    def _note_created_container(self, lease: Lease, before: set[str] | None) -> None:
        """Record the id of the container this run's `tt-model serve` made under the coder's name, so a later removal can check that a leftover
        is that container and not one that predates the run (issue 417). Only an id that was not there before the attempt counts: a start that
        failed before it created anything (an image pull, say) leaves a pre-existing container, which is not this run's. With no listing from
        before the attempt nothing is proven, so nothing is recorded. Best effort."""
        now = self._coder_container_ids()
        if before is None or now is None:
            return
        for cid in sorted(now - before):
            if not hasattr(self, "_created_ids"):
                self._created_ids = set()
            self._created_ids.add(cid)
            self._note("decision", decision="coder container created", container_id=cid, lease_id=lease.lease_id)

    def _remove_own_leftover(self, lease: Lease) -> bool:
        """A start was refused because a container holds the chips. True when a stopped container under the coder's exact name was found,
        its id is one this run recorded as created by its own launch ("coder container created"), and it was removed. A container that runs,
        one of another name, and one whose id this run did not record (it predates the run, or took the name since) are left alone, however
        many earlier attempts the ledger shows (issue 417)."""
        rows, remove = getattr(self.coder, "find_container_rows", None), getattr(self.coder, "remove_container", None)
        if rows is None or remove is None:
            return False
        created = self._known_created_ids()
        try:
            found = rows()
            if not found or any(state == "running" for _, state, _ in found) or any(cid not in created for _, _, cid in found):
                return False
            result = remove(even_if_unowned=True, only_id=found[0][2])
        except (ServerError, OSError) as exc:
            self._note("notice", what="removing the leftover coder container failed", error=str(exc))
            return False
        if not result:
            return False
        self._note("decision", decision="coder container removed" if result.get("removed") else "coder container left",
                   result=result, before_start=True, lease_id=lease.lease_id)
        return bool(result.get("removed"))

    def _block_chips_held(self, exc: Exception, lease: Lease, retried: bool = False) -> None:
        """Block after `tt-model serve` refused because a container holds the chips. The reason names the container that is left over and
        the command that removes it, and says the chips are not at fault, so the block is classed as a coder problem and not a hardware one
        (orchard/blocked.py). A chip that has in fact left the bus is still reported first, as for any failed boot."""
        # Recorded before the block, as the not-owned refusal is: a fresh invocation adopts the latest server record, and the record written at
        # "coder starting" says the owner is unknown, which would let a relaunch stop and remove the container that caused this refusal.
        self.ledger.append("decision", None, decision="coder not started", lease_id=lease.lease_id, server=self.coder.record(),
                           reason=str(exc))
        found = None
        try:
            found = self.coder.find_container() if hasattr(self.coder, "find_container") else None
        except (ServerError, OSError):
            found = None
        stopped = [(n, st) for n, st in (found or []) if st != "running"]
        if stopped:
            n, st = stopped[0]
            fix = f"A stopped container named {n} is left over ({st}); remove it with `docker rm {n}`, then resume."
        elif found:
            fix = f"A container named {found[0][0]} is running and holds them; stop it with `tt-model stop`, then resume."
        else:
            fix = ("No container with the coder's name was found; list containers with `docker ps -a`, remove or stop the container that holds "
                   "the chips, then resume.")
        again = " The leftover of this run's earlier launch was removed once and the start was refused again." if retried else ""
        reason = (f"the coder did not start and answer; nothing is retried: {exc}. The chips are not at fault: another container holds "
                  f"them. {fix}{again}")
        self._block_if_off_bus(None, reason, lease.chips)
        self._block(None, reason)

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
                       NoNewEvidence(), NoFileWritten(), TurnRepeat(), DedupeStreak(),
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
        if (resuming and escalated and spec.boards and marker.is_file() and self._test_failure(spec, marker.parent)
                and not (self._repair_pending(spec, marker.parent) and self._latest_start_escalated(n))):
            # A kill came after the escalate entry and before the stage_end. The escalated attempt
            # must run a fresh prepare and hardware test: its finish step cannot pass on a test
            # that failed, and it gets no gate feedback for one. So the failed record is not a
            # resume point, and open_stage_dir moves the directory aside. The exception is a full-port stage 2 whose failed test
            # is one the coder still has repair rounds for (issue 420) and whose latest attempt is already the escalated one: that attempt goes
            # on with them, and its count stands. When the latest `stage_start` is still the original attempt's, the kill came between the
            # escalate entry and the stage_end, and the escalated attempt starts afresh with its own prepare and its own rounds.
            self.ledger.append("decision", n, decision="not resuming from a failed hardware test", marker=self._moved_aside_record(marker))
            resuming = False
        # A port repair rerun in flight keeps the stage directory even when its record has no evidence entry yet (a kill between the file and
        # the entry. A rerun in flight and an `escalate` entry cannot both belong to one attempt, so there is nothing else to ask here.
        owed = (resuming and spec.draft == "full_port" and bool(self.port_repair_rounds)
                and port_repair.rerun_inflight(self.ledger.read(), n) is not None)
        stage_dir, resumed = open_stage_dir(self.run_dir, spec, resuming=resuming, ledger=self.ledger, rerun_owed=owed)
        self.ledger.append("stage_start", n, escalated=escalated, resumed=resumed)
        if n == 0:
            self._stage_0_stack_pins(stage_dir)
        self.actuator.clear()
        self.watchdog = self._watchdog(spec)
        status, reasons, gate = self._body_or_failure(spec, stage_dir, escalated, resumed)
        return self._end_stage(spec, stage_dir, status, reasons, gate, escalated)

    def _body_or_failure(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        """`_stage_body`, with a run file the bounded reader refused in the middle of a stage (a FIFO, a link, an oversize file, NaN) turned into a
        failed attempt that names the reason. It goes through the escalation and pause rules like a failed gate; the run does not abort."""
        try:
            return self._stage_body(spec, stage_dir, escalated, resumed)
        except safeio.SafeReadError as exc:
            return "fail", [f"stage {spec.number}: a run file was refused: {exc.reason}"], None

    def _moved_aside_record(self, marker) -> dict:
        """The ledger record of a marker that is about to be moved aside: its path and sha256, or its path and the named reason when the
        bounded reader refuses it (over the cap, a link that leaves the run directory). The directory is moved aside either way."""
        try:
            return evidence_record(self.run_dir, marker)
        except safeio.SafeReadError as exc:
            return {"path": os.path.relpath(os.path.abspath(marker), os.path.abspath(self.run_dir)), "unreadable": exc.reason}

    def _gate(self, spec):
        """The stage's gate. Stage 4's also gets the run's required chip counts; no other gate
        changes its call."""
        if spec.draft == "full_port_entries":
            # Both are the supervisor's own: the run's --required-chips and the rehearsal flag from the ledger. The coder can write neither.
            return functools.partial(spec.gate, required=self.required_chips, **({"rehearsal": True} if full_port_rehearsal(self.ledger.read()) else {}))
        if spec.number == 4 and self.required_chips:
            return functools.partial(spec.gate, required=self.required_chips)
        if spec.number == 0:
            return self._with_stack_pins(spec.gate)
        if spec.skill == "full-port-mesh":
            # The chip count is the supervisor's own (the run's --required-chips), and so is the rehearsal flag (from the ledger): the coder can
            # write neither.
            kwargs = {"expected_chips": self._stage_3_chips()}
            if full_port_rehearsal(self.ledger.read()):
                kwargs["rehearsal"] = True
            return functools.partial(spec.gate, **kwargs)
        if spec.skill == "full-port-check" and full_port_rehearsal(self.ledger.read()):
            return functools.partial(spec.gate, rehearsal=True)       # from the ledger: the coder cannot write it
        if spec.skill == full_port_serve.SKILL:
            return functools.partial(spec.gate, **self._serve_gate_args())
        if spec.skill == "full-port-quality":
            return full_port_quality_stage.gate(self, spec)           # chip count, copied files and source stage from the supervisor and the ledger
        return spec.gate

    def _serve_gate_args(self) -> dict:
        """What stage 5 of a full port judges against, all from the supervisor's side (issue 329): the stage whose port is served, that stage's chip
        count, the corpus hash the ledger recorded when the corpus was written, and the rehearsal flag. Nothing in the stage directory sets them."""
        entries = self.ledger.read()
        source = full_port_serve.source_stage(entries)
        return {"source": source, "expected_chips": self._serve_chips(source), "corpus_sha256": self._serve_corpus_sha(entries),
                "rehearsal": full_port_rehearsal(entries)}

    def _serve_chips(self, source) -> int:
        """The chips stage 5 runs on: the chips of the stage whose port it serves (stage 3's configured count, else stage 2's one board)."""
        return (self._stage_3_chips() or CHIPS_PER_BOARD) if source == 3 else CHIPS_PER_BOARD

    @staticmethod
    def _serve_corpus_sha(entries) -> str | None:
        """The sha256 of the corpus the supervisor wrote for the latest attempt of stage 5, from its own ledger entry."""
        found = None
        for e in entries:
            if e["event"] == "decision" and e["stage"] == full_port_serve.STAGE and e["data"].get("decision") == "serving corpus recorded":
                found = e["data"].get("sha256")
        return found if isinstance(found, str) else None

    def _stage_3_chips(self) -> int | None:
        """The chip count stage 3 of a full port runs the port on (issue 326): the largest of the run's `--required-chips` when it is more than
        one board's chips, else None. With None, stage 2's one board is the configured chip count and stage 3 is skipped with a recorded reason
        (`SKIP_3_FULL_PORT`). A count that is not a whole number of boards is rounded up, because the lease is whole boards and the gate compares
        the chips the test ran on with this number (a configured 3 is tested on, and expected as, 4). The value comes from the operator's option,
        which a resume keeps (`run_start`); nothing the coder writes is read."""
        top = max(self.required_chips) if self.required_chips else 0
        return -(-top // CHIPS_PER_BOARD) * CHIPS_PER_BOARD if top > CHIPS_PER_BOARD else None

    def _with_stack_pins(self, base):
        """Stage 0's gate plus the stack record's: the file must be the one the ledger recorded (its sha256), well formed, and hold no critical
        hazard. A missing file or a missing ledger entry fails; it does not skip the check."""
        def gate(stage_dir, run_dir):
            result = base(stage_dir, run_dir)
            extra = stack_pins_mod.gate_reasons(stage_dir, self._stack_pins_sha(stage_dir))
            if not extra:
                return result
            return dataclasses.replace(result, ok=False, reasons=tuple(result.reasons) + tuple(extra))
        return gate

    def _homes(self) -> tuple:
        """The home directories to show as <home> in text that reaches the ledger: the run's operator home and the invoking user's."""
        return (str(self.paths.operator_home), str(self.home))

    def _stack_pins_sha(self, stage_dir: Path) -> str | None:
        """The sha256 of the newest `stack pins` evidence entry the ledger holds for this stage directory's record, else None."""
        rel = os.path.relpath(Path(stage_dir) / stack_pins_mod.FILE, self.run_dir)
        found = None
        for e in self.ledger.read():
            d = e.get("data") or {}
            if e["event"] == "evidence" and e["stage"] == 0 and d.get("what") == "stack pins" and d.get("path") == rel:
                found = d.get("sha256")
        return found if isinstance(found, str) else None

    def _stage_0_stack_pins(self, stage_dir: Path) -> None:
        """Write stages/0/stack-pins.json (orchard/stack_pins.py) before the stage's agent starts, and record its hash in the ledger. A new run
        that finds a critical hazard (a checkout more than 7 days behind, a transformers below the checkpoint's minimum or without its
        architecture class) blocks with the named reasons. A resume, and a second attempt at this stage, hold the same findings as advisories.
        The hash is taken from the bytes written, so a file swapped afterwards does not match it."""
        entries = self.ledger.read()
        seen_before = any(e["event"] == "evidence" and e["stage"] == 0 and (e.get("data") or {}).get("what") == "stack pins" for e in entries)
        new_run = not (self._run_resumed or seen_before)
        # A resumed stage keeps its directory, so the record and the ledger hash of the last attempt are still there. Remove the old file first:
        # if this attempt cannot write a new one, the gate must not accept the old evidence.
        try:
            os.unlink(Path(stage_dir) / stack_pins_mod.FILE)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.ledger.append("decision", 0, decision="stack pins not written", reason=f"the old record could not be removed ({type(exc).__name__})")
            self._block(0, "the stack record of an earlier attempt could not be removed, so this attempt cannot write its own. Fix the stage 0 directory, then run the command again")
        try:
            if self.stack_pins is not None:
                rec, sha = self.stack_pins(stage_dir, new_run)
            else:
                rec = stack_pins_mod.unchecked_record("no stack comparison was supplied to this supervisor", new_run=new_run)
                sha = stack_pins_mod.write_record(stage_dir, rec)
        except (OSError, ValueError) as exc:        # no record: nothing may go on without one
            self.ledger.append("decision", 0, decision="stack pins not written", reason=stack_pins_mod.redact(f"{type(exc).__name__}: {exc}", self._homes())[:200])
            self._block(0, f"the stack freshness record could not be written ({type(exc).__name__}). Fix the stage 0 directory, then run the command again")
        self.ledger.append("evidence", 0, what="stack pins", **stack_pins_mod.ledger_fields(rec),
                           path=os.path.relpath(Path(stage_dir) / stack_pins_mod.FILE, self.run_dir), sha256=sha)
        if rec["hazards"]:
            why = "; ".join(f"{h['name']}: {h['reason']}" for h in rec["hazards"])
            self._block(0, "the stack is stale for a new run (" + why[:900] + "). Update the named parts, or run the command again to go on "
                           "with these as advisories", hazards=[h["name"] for h in rec["hazards"]])

    def _stage_body(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        self._wrapup_used = False           # at most one wrap-up per run of the stage body
        if spec in TEXT_SPECS:
            return self._text_body(spec, stage_dir, resumed)
        if spec.harness:
            if spec.number == 0:
                return self._stock_triage_body(spec, stage_dir)
            if spec.number == 8:
                return self._bundle_body(spec, stage_dir)
            if spec in (AUDIO_STAGE_1, AUDIO_STAGE_2):
                return self._audio_body(spec, stage_dir, resumed)
            return self._package_body(spec, stage_dir, resumed)
        if spec.draft == "full_port_entries":
            return full_port_stage4.run(self, spec, stage_dir, escalated, resumed)       # one prepare step, test and record per required chip count
        derived = False         # the supervisor wrote this stage's result itself (issue 456)
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
                prepared = self._prepare_in_code(spec, stage_dir)
                if not prepared.done:
                    out, prep = self._step(spec, "prepare", stage_dir, escalated, resumed, extra=prepared.note)
                    out = self._wrap_up(spec, stage_dir, prep, out)
                    if out.status != "done":
                        return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
                # Whoever prepared, no result may exist before the test has run: the finish step writes the result from the test's output.
                if moved := swap_prepare.set_aside_results(stage_dir):
                    self.ledger.append("decision", spec.number, decision="results written before the test were set aside", files=moved)
                if spec.draft == "full_port":
                    # The coder wrote the port; the check is the supervisor's. Its template, config and command are put back as shipped.
                    if problems := self._pin_full_port(spec, stage_dir, command=True):
                        return "fail", problems, None
                    self._check_port_cache(spec, stage_dir)
                if spec.draft == "full_port_serve" and getattr(self, "_serve_problems", None):
                    return "fail", list(self._serve_problems), None
                if spec.draft == "scoring_check":
                    if problems := self._pin_scoring(spec, stage_dir, command=True):
                        return "fail", problems, None
                if spec.draft == "full_port_quality":
                    if problems := full_port_quality_stage.pin(self, spec, stage_dir):
                        return "fail", problems, None
                    self._check_port_cache(spec, stage_dir)
                test, problems = self._read_test(stage_dir)
                if problems:
                    return "fail", problems, None
                # A container package's test leaves a container behind when its client is killed (a timeout, an abort): stop it by
                # label before the chips are given back, and block when one survives (_sweep).
                self._hardware_phase(spec, stage_dir, test,
                                     after_test=(lambda: self._sweep(spec.number)) if self._container_config(stage_dir) else None)
            if spec.draft == "full_port":
                # A port check that failed on the port itself may earn repair rounds (issue 420). It also runs on a resume, when the stage
                # directory holds a failed record and the attempt has rounds left.
                ended = self._port_repair_rounds(spec, stage_dir, escalated, resumed)
                if ended:
                    return ended
                if self.port_repair_rounds:
                    # The finish step is a conversation of its own. The check that ran after the last repair step can take longer than the
                    # no-progress window, and its evidence is in the ledger, not in the watchdog: the repair step's detector clock would read the
                    # finish step's first reply as an hour with no evidence (review of PR 425, round 3).
                    self.watchdog = self._watchdog(spec)
            if spec.draft == "full_port_serve":
                self._derive_serve_result(spec, stage_dir)
                step = None          # the supervisor wrote the result from the record: stage 5 has no agent step
            elif spec.draft == "scoring_check":
                step = None          # the check's own record is the result: no agent writes a finish step for a scoring model
            elif spec.draft == "full_port_quality":
                step, derived = None, True          # stage 6 of a full port: the supervisor writes result.json from the record, and no agent could add to it
                made, why = full_port_quality_stage.derive(self, spec, stage_dir)
                if not made:
                    return "fail", [why], None
            elif spec.draft == "full_port" and (derived := self._derive_full_port_result(spec, stage_dir)):
                step = None          # the hardware test passed and the record is complete: result.json is copied from it, no agent step
            else:
                out, step = self._step(spec, "finish", stage_dir, escalated, resumed)
                out = self._wrap_up(spec, stage_dir, step, out)
                if out.status != "done":
                    return out.status, [f"the finish step ended: {out.status} {out.detail}".strip()], None
        gate = self._check_gate(spec, stage_dir)
        if not gate.ok and (spec.draft in ("scoring_check", "full_port_serve") or derived):
            return "fail", list(gate.reasons), gate          # nothing an agent could add would change the record the template wrote
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

    def _derive_full_port_result(self, spec, stage_dir: Path) -> bool:
        """Stage 2 of a full port, finish phase (issue 456): when the hardware test passed and the check's record is complete, write
        result.json from the record (orchard/full_port_result.py) so no agent step has to, and say so in the ledger. False leaves the
        finish step to the agent: a failed or missing test or record, text the readable floor does not pass, or a derived result that
        the gate's own check refuses (the file is deleted first, so the agent starts from a clean directory). The check that judges the
        result is the same one the gate runs, so deriving cannot make a result pass that would not."""
        n = spec.number
        rehearsal = full_port_rehearsal(self.ledger.read())
        result, why = full_port_result.derive(stage_dir, self.run_dir, rehearsal=rehearsal)
        if result is None:
            self.ledger.append("decision", n, decision="result not derived by the supervisor", why=why[:300])
            return False
        path = stage_dir / "result.json"
        try:
            safeio.atomic_write(path, (json.dumps(result, indent=2) + "\n").encode("utf-8"), mode=0o644)
        except OSError as exc:
            self.ledger.append("decision", n, decision="result not derived by the supervisor", why=f"could not write it ({exc.__class__.__name__})")
            return False
        # The stage's own gate judges the derived file, with the arguments that stage's gate gets (the rehearsal flag from the ledger, and for
        # a later full-port stage the chip count it is configured for). It is the pure gate: the ledger checks and the waiver decision stay in
        # `_check_gate`, so the decision is written once, by the stage's one gate call, whoever wrote result.json.
        checked = self._gate(spec)(stage_dir, self.run_dir)
        reasons = [] if checked.ok else (list(checked.reasons) or ["the stage's gate refused the derived result"])
        if reasons:
            path.unlink(missing_ok=True)
            self.ledger.append("decision", n, decision="result not derived by the supervisor",
                               why="the derived result did not pass the gate's check: " + "; ".join(reasons[:3])[:300])
            return False
        found = full_port.file_record(self.run_dir, path) or {}
        self.ledger.append("evidence", n, what="result.json derived by the supervisor", **found)
        self.ledger.append("decision", n, decision="result derived by the supervisor", file=f"stages/{n}/result.json",
                           top1_agreement=result["top1_agreement"], n_tokens=result["n_tokens"], coherent=result["coherent"],
                           coherent_basis=result["coherent_basis"][:300])
        return True

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
        if spec.draft == "full_port_entries":
            return full_port_entries.failed(stage_dir, self.required_chips)
        if spec.tests:
            return failed_tests(stage_dir, self.required_chips)
        return hardware_test_failure(stage_dir)

    # ---- repair rounds for a full-port stage 2 (issue 420) -------------------------------------

    def _latest_start_escalated(self, n: int) -> bool:
        """Whether the latest `stage_start` of stage `n` began an escalated attempt (False when there is none, or it began the original one)."""
        latest = None
        for e in self.ledger.read():
            if e.get("event") == "stage_start" and e.get("stage") == n:
                latest = e
        return latest is not None and (latest.get("data") or {}).get("escalated") is True

    def _repair_pending(self, spec, stage_dir: Path) -> bool:
        """Does a full-port stage 2's failed test still owe the coder's work a test, or still have repair rounds? The option is not 0 and the
        attempt has not ended its rounds, and one of these holds (Copilot's second review of PR 425):

        - a rerun was started and its `tested` entry is missing: the supervisor died while the repaired port was on the chips. The rerun opened
          the output file fresh, so the old record may no longer match it; the ledger's entry is what says the test is owed;
        - the record shows a failure of the port (orchard/port_repair.py, `read_failure`) and the attempt's rounds are not all used;
        - the rounds are all used, but the last round's edit was never tested (the port differs from the one the latest test ran against)."""
        if spec.draft != "full_port" or spec.number != 2 or not self.port_repair_rounds:
            return False
        n, entries = spec.number, self.ledger.read()
        if port_repair.stopped_noted(entries, n):
            return False
        if port_repair.rerun_inflight(entries, n) is not None:
            return True
        if port_repair.read_failure(self.run_dir, stage_dir) is None:
            return False
        if port_repair.rounds_used(entries, n) < self.port_repair_rounds:
            return True
        tested = port_repair.tested_digest(entries, n)
        return tested is not None and port_repair.digest(port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)) != tested

    def _seal(self, stage_dir: Path, rel: str) -> dict | None:
        """The path and sha256 of a file of the stage directory through the bounded reader, or None (missing, a link, over the cap)."""
        return full_port.file_record(self.run_dir, Path(stage_dir) / rel)

    def _read_output(self, stage_dir: Path) -> str:
        """The supervisor's capture of the failed test's output, through the bounded reader ("" when it cannot be read)."""
        try:
            return safeio.read_bytes(Path(stage_dir) / port_repair.OUTPUT, max_bytes=RUN_JSON_MAX).decode("utf-8", "replace")
        except (safeio.SafeReadError, OSError, ValueError):
            return ""

    def _repair_text(self, n: int, k: int, limit: int, failure: dict, raw: str) -> str:
        """The supervisor's message to a repair step: which round, where the check failed, and the end of its output."""
        where = f"at the {failure['op']!r} request" if failure.get("op") else "before any request reached the port"
        return "\n".join([f"Port repair round {k} of {limit} for this attempt. The port check failed {where}.",
                          f"The end of the check's output (stages/{n}/{port_repair.OUTPUT}):", "```", port_repair.tail(raw).rstrip(), "```",
                          "The supervisor runs the same check again after this step, on the same chips and with the same reference. "
                          f"Only your edits under stages/{n}/port/ reach it."])

    def _port_repair_rounds(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        """Up to `--port-repair-rounds` repair rounds for a full-port stage 2 whose port check failed on the port itself. Returns
        (status, reasons, gate) when the stage ends here, else None (the finish step follows, with the last test's record).

        A round: its decision goes to the ledger first (so a restart cannot repeat a round it counted), then a repair step by the coder, who may
        write only under stages/2/port/. The supervisor's record of the failed test and its output file must be as they were (the ledger
        holds their hashes), the check's template, config and command are put back as shipped, results written during the step are set
        aside, and the same check runs again under a lease like any test, so the supervisor writes the new record. A round that changed
        nothing in the port ends the rounds: the same files would fail the same way. After the last round the stage goes on as it did before
        this option: the finish step records the failure, the gate fails, and the next attempt starts again."""
        n, limit = spec.number, self.port_repair_rounds
        if not limit or n != 2:         # repair rounds belong to stage 2: stage 3 starts from a port that already passed (issue 326)
            return None
        # Each round takes at most two passes (a catch-up for a rerun whose `tested` entry is missing, then the round) and the last pass reads the
        # end. A loop whose exit depends on ledger entries gets a bound: a mutant of `rerun_inflight` that never closed spun here, appending forever.
        for _ in range(2 * limit + 2):
            entries = self.ledger.read()
            if port_repair.stopped_noted(entries, n):
                return None
            owed = port_repair.rerun_inflight(entries, n)
            ran = port_repair.rerun_recorded(entries, n) if owed is not None else None
            sealed_now = self._seal(stage_dir, port_repair.MARKER) if ran is not None else None
            if ran is not None and sealed_now is not None and sealed_now.get("sha256") == ran.get("sha256"):
                # The rerun's own evidence entry follows its `rerun started` entry and vouches for the record on disk: the test ran, and the
                # supervisor died before it wrote the `tested` entry. Say so and read its record. The entry says it, not the record's bytes, which an
                # unchanged failure can leave equal to the sealed ones. A record that is replaced and has no such entry (a kill between the file and
                # its entry) is not trusted: the owed check runs again below, on the repaired port (review of PR 432).
                self.ledger.append("decision", n, decision=port_repair.TESTED,
                                   sources=port_repair.digest(port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)))
                continue
            failure = port_repair.read_failure(self.run_dir, stage_dir)
            if failure is None and owed is None:
                return None
            used = port_repair.rounds_used(entries, n)
            before = port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)
            tested = port_repair.tested_digest(entries, n)
            if tested is None:
                # The port as the attempt's first test ran it (nothing has edited it since); a later resume compares against this.
                self.ledger.append("decision", n, decision=port_repair.TESTED, sources=port_repair.digest(before))
                tested = port_repair.digest(before)
            # A kill after a repair edit and before its rerun leaves the edited port beside the old failed record. The edit is tested before
            # another repair starts, and before the port can be called unchanged (review of PR 425).
            # A rerun that was started and not finished is owed too: its output file was replaced, so the old record may not read as a failure.
            pending = owed is not None or (port_repair.digest(before) != tested and used > 0)
            if pending:
                k = owed["round"] if owed is not None and type(owed.get("round")) is int else used
            else:
                if used >= limit:
                    self.ledger.append("decision", n, decision=port_repair.STOPPED, reason="spent", rounds=used, of=limit, op=failure["op"])
                    return None
                k = used + 1
                sealed = {rel: self._seal(stage_dir, rel) for rel in (port_repair.MARKER, port_repair.OUTPUT)}
                self._repair_baseline = before
                # The whole output is cleaned before any cut: the ledger's failure line and the prompt's tail come from this text.
                raw = port_repair.safe_text(self._read_output(stage_dir), self._homes())
                self.ledger.append("decision", n, decision=port_repair.DECISION, round=k, of=limit, op=failure["op"], returncode=failure["returncode"],
                                   failure=port_repair.failed_line(raw))
                # Each repair step is a conversation of its own. A watchdog made for the stage would carry the loop detectors' history from one
                # round to the next: six distinct edits of one size give six identical `wrote N characters` replies, which read as a loop and
                # escalated a stage that was making progress (review of PR 425). A new one keeps the attempt's start time, so the stage budget
                # stands, and reads the ladder's rungs from the ledger, so a nudge or an escalation already recorded still counts.
                self.watchdog = self._watchdog(spec)
                # The outside helper at the start of the round (orchard/repair_help.py): in the attempt's first round, and in a round whose failure
                # is no further along than the last one's. Its file, when it is placed, is what the coder's step starts from.
                asked_here, friend_note = False, ""
                if self.repair_help and getattr(self, "telegraph", None) is not None:
                    ask_now, ask_why = repair_help.start_trigger(self.ledger.read(), n, k, failure)
                    if ask_now:
                        friend_note, _placed, asked_here = self._repair_help(spec, stage_dir, k, limit, failure, raw, why=ask_why, coder_failed=False)
                        if _placed:
                            self._repair_baseline = port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)    # the coder's step ends on ITS edit
                try:
                    out, _ = self._step(spec, "repair", stage_dir, escalated, resumed,
                                        extra=self._repair_text(n, k, limit, failure, raw) + (("\n\n" + friend_note) if friend_note else ""))
                finally:
                    self._repair_baseline = None
                if out.status != "done":
                    # A step that stopped in a watchdog pause or escalation, or on capped replies: the helper gets the port as the coder left it,
                    # when it was not asked at the start of this round. A placed answer lets the round go on to its check.
                    helped_on = False
                    if (out.status in ("pause", "escalate", "error") and not asked_here and self.repair_help and getattr(self, "telegraph", None) is not None):
                        _note, helped_on, _ = self._repair_help(spec, stage_dir, k, limit, failure, raw,
                                                                why=f"the coder's repair step ended: {out.status}", coder_failed=True)
                    if not helped_on:
                        return out.status, [f"the repair step ended: {out.status} {out.detail}".strip()], None
                    if out.status in ("pause", "escalate"):
                        self._consume_handled_stop(n, out.status)
                touched = [rel for rel, rec in sealed.items() if self._seal(stage_dir, rel) != rec]
                if touched:
                    self.ledger.append("decision", n, decision="port repair refused", round=k, files=touched)
                    return "fail", [f"the repair step changed {', '.join(f'stages/{n}/{rel}' for rel in touched)}, which only the supervisor writes. "
                                    "The attempt fails, and the next attempt starts again"], None
                if not port_repair.changed(before, port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)):
                    self.ledger.append("decision", n, decision=port_repair.STOPPED, reason="unchanged", rounds=k, of=limit, op=failure["op"])
                    return None
            self._keep_round_record(n, stage_dir, k)
            if moved := swap_prepare.set_aside_results(stage_dir, keep=(port_repair.MARKER, port_repair.OUTPUT)):
                self.ledger.append("decision", n, decision="results written before the test were set aside", files=moved)
            if problems := self._pin_full_port(spec, stage_dir, command=True):
                return "fail", problems, None
            test, problems = self._read_test(stage_dir)
            if problems:
                return "fail", problems, None
            # Before the test opens its output file: from here a kill leaves the old record beside an output that no longer matches its hash.
            self.ledger.append("decision", n, decision=port_repair.RERUN, round=k, record=self._seal(stage_dir, port_repair.MARKER))
            self._hardware_phase(spec, stage_dir, test)
            self.ledger.append("decision", n, decision=port_repair.TESTED,
                               sources=port_repair.digest(port_repair.fingerprint(stage_dir / port_repair.PORT_DIR)))
            self._repair_verdicts(n, stage_dir)
        self.ledger.append("decision", n, decision=port_repair.STOPPED, reason="did not settle", rounds=port_repair.rounds_used(self.ledger.read(), n), of=limit)
        return "fail", [f"the port repair rounds did not settle within {2 * limit + 2} passes, which a ledger that follows the rules cannot cause; "
                        "the attempt fails, and the next attempt starts again"], None

    def _keep_round_record(self, n: int, stage_dir: Path, k: int) -> None:
        """Copy the failed test's output to evidence/port-repair-round-<k>-output.txt before the next test replaces it, and record the copy's
        path and sha256 in the ledger. The next test's capture overwrites the file the repair step read; this keeps what it was shown. A resume
        that finds the copy already recorded leaves it: the output file may by then be a rerun's, cut short."""
        if port_repair.output_kept(self.ledger.read(), n, k):
            return
        try:
            raw = safeio.read_bytes(Path(stage_dir) / port_repair.OUTPUT, max_bytes=RUN_JSON_MAX)
        except (safeio.SafeReadError, OSError, ValueError):
            return
        target = Path(stage_dir) / "evidence" / f"port-repair-round-{k}-output.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        safeio.atomic_write(target, raw, mode=0o644)
        self.ledger.append("evidence", n, what="port repair round output", round=k, **self._moved_aside_record(target))

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

    # ---- the text runtime path (orchard/text_runtime.py, orchard/text_stages.py) ----------------------------------------------------

    def _text_runtime(self):
        """The adapter this run was started with (run_start's input `text_runtime`), or None."""
        return text_runtime.get(text_runtime.selected(self.ledger.read()))

    def _text_env(self) -> dict:
        """The environment of the text stages' children: the agent shells' (no token, HOME in the run directory, no chip) plus the Hugging Face cache directory,
        with the Hub offline."""
        env = agent_env(self.run_dir, extra=self.extra_env)
        env.update({"HF_HOME": self.paths.hf_home, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        return env

    def _text_context(self, rtm, stage_dir: Path) -> "text_runtime.TextContext":
        """The context of a text stage's commands: absolute paths, the stage directory as the output directory, stage 1's reference directory."""
        model_dir = str(Path(self.inputs["model"]).absolute())
        # A resumed run keeps the interpreter its stage 0 recorded: the orchestrator and the gates use it, so the commands must too.
        sel = text_runtime.selection(self.ledger.read())
        recorded = (sel or {}).get("paths") if isinstance((sel or {}).get("paths"), dict) else None
        paths = text_runtime.TextPaths.from_record(recorded) if recorded and "python" in recorded else text_runtime.text_paths(self.paths.operator_home)
        cache = Path(self.paths.cache_root) / "text-runtime" / rtm.name
        weights = rtm.weights_path(model_dir, cache)
        parquet = text_runtime.find_ppl_parquet(self.paths.hf_home, self.inputs.get("text_ppl_parquet"))
        # Once stage 1 has saved its settings, later stages use those: a resume that omits or changes an input must not change the window or the decode length
        # the reference was made with.
        saved = None
        try:
            saved = text_runtime.load_json(self.run_dir / "stages" / "1" / text_runtime.STAGE_CONFIG_FILE, "stage 1's saved settings")
        except text_runtime.TextStageError:
            pass
        if saved and all(isinstance(saved.get(k), int) for k in ("decode_tokens", "ppl_tokens", "max_len")):
            decode, ppl_tokens, max_len = saved["decode_tokens"], saved["ppl_tokens"], saved["max_len"]
            if not (isinstance(saved.get("ppl_parquet"), str) and isinstance(saved.get("ppl_parquet_sha256"), str)):
                # stage 1 of an earlier version recorded only the file name: the window the reference used cannot be checked, so a later stage must not pick one
                raise text_runtime.TextStageError("stage 1's saved settings do not record the perplexity dataset (path and sha256), so the window its reference used cannot be "
                                                  "verified; start a new run")
            else:
                # the perplexity window is cut from this file: the reference was made from it, so later stages use it and refuse a changed copy
                parquet = saved["ppl_parquet"]
                try:
                    now = hashlib.sha256(Path(parquet).read_bytes()).hexdigest()
                except OSError as exc:
                    raise text_runtime.TextStageError(f"the perplexity dataset stage 1 used cannot be read ({exc.__class__.__name__}); the window must stay the same") from exc
                if now != saved["ppl_parquet_sha256"]:
                    raise text_runtime.TextStageError("the perplexity dataset file changed since stage 1 made its reference (its sha256 differs); the window must stay the same")
        else:
            try:
                decode = int(self.inputs.get("text_decode_tokens") or text_runtime.DECODE_TOKENS)
                ppl_tokens = int(self.inputs.get("text_ppl_tokens") or text_runtime.PPL_TOKENS)
            except ValueError as exc:
                raise text_runtime.TextStageError(f"text_decode_tokens and text_ppl_tokens must be whole numbers: {exc}") from exc
            max_len = text_runtime.chip_max_len(decode)
        return text_runtime.TextContext(python=paths.python, model_dir=model_dir, weights=str(weights), out_dir=str(stage_dir.resolve()),
                                        ref_dir=str((self.run_dir / "stages" / "1" / "reference").resolve()),
                                        prompt_file=str((self.run_dir / "stages" / "1" / "prompt.txt").resolve()), ppl_parquet=parquet or "",
                                        decode_tokens=decode, ppl_tokens=ppl_tokens, max_len=max_len)

    def _text_body(self, spec, stage_dir: Path, resumed: bool):
        """Stages 0, 1, 2 and 6 of a text-runtime run, as supervisor code. The numeric work runs in the configured interpreter (orchard/text_remote.py), so the
        supervisor needs only the standard library. Stages 2 and 6 use the one-test hardware phase: a lease (no coder to park when none was booted), one
        command that runs the adapter's chip commands in order, the lease back."""
        from orchard import text_remote
        from orchard.text_runtime import TextStageError
        n = spec.number
        rtm = self._text_runtime()
        if rtm is None:
            return "fail", ["the run was not started with a registered text_runtime input"], None
        if self.lab is not None:
            return "fail", ["the text runtime does not run in lab mode"], None
        stage_dir.mkdir(parents=True, exist_ok=True)
        if n == 0:
            return self._text_triage_body(spec, stage_dir, rtm)
        sel = text_runtime.selection(self.ledger.read())
        if sel is None:
            return "fail", ["the ledger holds no text runtime selection with its paths"], None
        paths = text_runtime.TextPaths.from_record(sel["paths"])
        try:
            if n == 1:
                if not (resumed and (stage_dir / text_runtime.REFERENCE_FILE).is_file()):
                    self._text_stage_1(stage_dir, paths, rtm)
            else:
                ctx = self._text_context(rtm, stage_dir)
                if not (resumed and (stage_dir / "test-result.json").is_file()):
                    commands = text_runtime.chip_config_commands(rtm, ctx, n)
                    text_runtime.write_chip_config(stage_dir, commands)
                    command = text_runtime.chip_test_command(paths, stage_dir.resolve())
                    try:
                        check_string(command, self.run_dir)
                    except Denied as exc:
                        return "fail", [f"the test command is refused: {exc}"], None
                    deadline = text_runtime.CHIP_DEADLINE_S if n == 2 else text_runtime.BENCH_DEADLINE_S
                    test = {"command": command, "deadline_s": deadline}
                    _put((stage_dir / "hw_test.json"), json.dumps(test) + "\n", encoding="utf-8")
                    _put((stage_dir / "handoff.json"), json.dumps({
                        "goal": f"run the {rtm.name} text model on one chip: " + ("teacher-forced agreement and perplexity" if n == 2 else "benchmark with repeats"),
                        "stage": n, "evidence": f"stages/{n}/{text_runtime.CHIP_CONFIG_FILE}", "next_action": "the supervisor runs the chip commands under its own lease",
                        "check_on_return": f"stages/{n}/test-result.json"}) + "\n", encoding="utf-8")
                    self._hardware_phase(spec, stage_dir, test)
                failed = hardware_test_failure(stage_dir)
                if failed:
                    return "fail", [failed["problem"] + f"; see stages/{n}/evidence/hw-test-output.txt and stages/{n}/log"], None
                text_remote.call(paths.python, "finish_chip" if n == 2 else "finish_bench", {
                    "stage_dir": str(stage_dir), "run_dir": str(self.run_dir), "ref_stage_dir": str(self.run_dir / "stages" / "1"), "runtime": rtm.name},
                    timeout_s=1800.0, env={"PATH": os.environ.get("PATH", "")})
        except TextStageError as exc:
            return "fail", [str(exc)], None
        self.ledger.append("evidence", n, what="text stage record", **evidence_record(self.run_dir, stage_dir / spec.gate_file))
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _text_triage_body(self, spec, stage_dir: Path, rtm):
        """Stage 0 of a text-runtime run: read the model's own files, write delta.json and its evidence, record the runtime selection (the interpreter and the script
        hashes), run the gate. No model is asked anything, so no coder boots for it."""
        n = spec.number
        model_dir = self.inputs.get("model")
        if not model_dir:
            return "fail", ["a text-runtime run needs the input model (the snapshot directory)"], None
        paths = text_runtime.text_paths(self.paths.operator_home)
        missing = text_runtime.environment_problems(paths)
        if missing:
            return "fail", missing, None
        try:
            facts = text_runtime.model_facts(model_dir, rtm)
        except (OSError, ValueError, text_runtime.safeio.SafeReadError) as exc:
            return "fail", [f"the model directory cannot be read: {exc}"], None
        (stage_dir / "evidence").mkdir(exist_ok=True)
        evidence = {"runtime": rtm.name, "model": self.model_id, "snapshot": Path(model_dir).resolve().name, "facts": facts, "scripts": text_runtime.script_hashes(rtm), "context_limit_measured": rtm.context_limit}
        _put((stage_dir / "evidence" / "triage.json"), json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        delta = {"model": self.model_id, "nearest_model": f"none: the {rtm.name} runtime (scripts in this repository) serves it",
                 "path": "full-port", "class": "full-port",
                 "path_reasons": [f"the harness's {rtm.name} text runtime implements this architecture; no installed bundle or container package serves it"],
                 "differences": [{"area": "architecture", "evidence": ["stages/0/evidence/triage.json"],
                                  "finding": f"{facts.get('architectures')} ({facts.get('model_type')}), {facts.get('n_layers')} layers; served by the {rtm.name} runtime's own scripts"}],
                 "hazards": [], "facts": {"form": "full", "serving": "none", "modality": "text"},
                 "text_runtime": {"name": rtm.name, "config_problems": facts["config_problems"], "card_license": facts["card_license"],
                                  "expected_license": rtm.license_id}}
        if facts["card_license"] and facts["card_license"] != rtm.license_id:
            delta["hazards"].append({"area": "license", "evidence": ["stages/0/evidence/triage.json"],
                                     "finding": f"the card says license {facts['card_license']}; the adapter expects {rtm.license_id}"})
        _put((stage_dir / "delta.json"), json.dumps(delta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.ledger.append("decision", n, decision=text_runtime.TEXT_RUNTIME_SELECTED, runtime=rtm.name, paths=paths.record(),
                           scripts=text_runtime.script_hashes(rtm), model=self.model_id)
        self.ledger.append("evidence", n, what="text triage", **evidence_record(self.run_dir, stage_dir / "delta.json"))
        gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _text_stage_1(self, stage_dir: Path, paths, rtm) -> None:
        """Write the stage's inputs (the prompt, the configuration the gates read) and run the reference commands."""
        from orchard import text_remote
        ctx = self._text_context(rtm, stage_dir)
        cfgd = json.loads((Path(ctx.model_dir) / "config.json").read_text(encoding="utf-8"))
        facts = text_runtime.model_facts(ctx.model_dir, rtm)
        if not ctx.ppl_parquet:
            raise text_runtime.TextStageError("no WikiText-2 test parquet was found for the perplexity window; give --input text_ppl_parquet=<file> or put the "
                                              "dataset in the Hugging Face cache")
        _put((stage_dir / "prompt.txt"), text_runtime.DEFAULT_PROMPT + "\n", encoding="utf-8")
        try:
            ppl_sha = hashlib.sha256(Path(ctx.ppl_parquet).read_bytes()).hexdigest()
        except OSError as exc:
            raise text_runtime.TextStageError(f"the perplexity dataset file cannot be read ({exc.__class__.__name__}): {Path(ctx.ppl_parquet).name}") from exc
        text_runtime.write_json(stage_dir / text_runtime.STAGE_CONFIG_FILE, {
            "schema": text_runtime.SCHEMA, "runtime": rtm.name, "model_id": self.model_id.split("@", 1)[0], "model_dir": ctx.model_dir, "weights": ctx.weights,
            "config_sha256": facts["config_sha256"], "n_layers": rtm.n_layers(cfgd), "decode_tokens": ctx.decode_tokens, "ppl_tokens": ctx.ppl_tokens, "max_len": ctx.max_len,
            "prompt_sha256": hashlib.sha256(text_runtime.DEFAULT_PROMPT.encode("utf-8")).hexdigest(), "ppl_parquet_name": Path(ctx.ppl_parquet).name,
            "ppl_parquet": str(Path(ctx.ppl_parquet).resolve()), "ppl_parquet_sha256": ppl_sha})
        ctx = dataclasses.replace(ctx, prompt_file=str((stage_dir / "prompt.txt").resolve()), out_dir=str((stage_dir / "reference").resolve()),
                                  ref_dir=str((stage_dir / "reference").resolve()))
        result = text_remote.call(paths.python, "references", {
            "stage_dir": str(stage_dir), "run_dir": str(self.run_dir), "runtime": rtm.name, "context": dataclasses.asdict(ctx), "env": self._text_env(),
            "timeout_s": text_runtime.REFERENCE_TIMEOUT_S}, timeout_s=text_runtime.REFERENCE_TIMEOUT_S + 600, env={"PATH": os.environ.get("PATH", "")})
        for note in result.get("notes", []):
            self.ledger.append("decision", 1, decision="text reference", note=note)

    def _audio_models(self) -> tuple[str | None, str | None]:
        """The model and its base as stage 0 wrote them (`<repo>@<revision>`)."""
        delta = self._run_delta()
        if delta is None:
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
                    # the decoder runs on the chip when the template implements this model's decoder; otherwise on the host, as before
                    model_dir = self.inputs.get("model")
                    on_chip = bool(model_dir) and not audio_runtime.decoder_problems(model_dir)
                    audio_plan.write_chip_config(stage_dir, paths, self.paths.hf_home, plan, chip_decoder=on_chip)
                    command = audio_plan.chip_test_command(paths, stage_dir.resolve(), audio_runtime.TEMPLATES / "audio_chip_test.py")
                    try:
                        check_string(command, self.run_dir)
                    except Denied as exc:
                        return "fail", [f"the test command is refused: {exc}"], None
                    test = {"command": command, "deadline_s": AUDIO_CHIP_DEADLINE_S}
                    _put((stage_dir / "hw_test.json"), json.dumps(test) + "\n", encoding="utf-8")
                    _put((stage_dir / "handoff.json"), json.dumps({
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
        orchard_dir = (Path(__file__).resolve().parent.parent
                       if (audio_runtime.selected(self.ledger.read()) is not None or text_runtime.selected(self.ledger.read()) is not None)
                       else Path(self.paths.orchard_dir))
        shutil.copyfile(orchard_dir / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py", script)
        _put((stage_dir / "bundle_config.json"), 
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
        _put(log, proc.stdout + proc.stderr, encoding="utf-8")
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
        gate = self._run_gate(spec, stage_dir)
        if spec.tests:
            # The gate reads the test records from the stage directory, where the agent can write
            # too. The ledger holds the sha256 of each record the supervisor wrote.
            forged = unrecorded(self.ledger.read(), stage_dir, self.run_dir, spec.number)
            if forged:
                return GateResult(False, gate.reasons + tuple(forged), gate.evidence)
        if spec.skill in ("scoring-reference", "scoring-check"):
            gate = self._scoring_gate(spec, stage_dir, gate)
        if spec.skill == "scoring-check":
            found = full_port.file_record(self.run_dir, stage_dir / "test-result.json")
            if found is None or not recorded_by_supervisor(self.ledger, spec.number, found):
                return GateResult(False, gate.reasons + (
                    "test-result.json is not the file the supervisor recorded when the hardware test ended (the ledger holds no "
                    "matching entry), so its exit code cannot be trusted",), gate.evidence)
        if spec.skill in ("full-port-check", "full-port-mesh", full_port_serve.SKILL, "full-port-quality"):
            # One hardware test, one marker. The finish step can write in the stage directory, so the marker has to be the file the
            # supervisor wrote: its path and sha256 are in the ledger (`_run_test`). A marker edited after the test has no such entry.
            found = full_port.file_record(self.run_dir, stage_dir / "test-result.json")      # bounded read: a FIFO or /dev/zero link is None
            if found is None or not recorded_by_supervisor(self.ledger, spec.number, found):
                return GateResult(False, gate.reasons + (
                    "test-result.json is not the file the supervisor recorded when the hardware test ended (the ledger holds no "
                    "matching entry), so its exit code cannot be trusted",), gate.evidence)
            if gate.ok and "coherent" in gate.waived:      # the gate's own read: result.json is not opened again
                self.ledger.append("decision", spec.number, decision=REHEARSAL_WAIVED, field="coherent", by="--full-port-rehearsal",
                                   note=f"the model is a rehearsal with random weights; every other stage {spec.number} check still applied")
        if spec.draft == "full_port_entries":
            gate = full_port_stage4.vouch(self, spec, stage_dir, gate, waived_decision=REHEARSAL_WAIVED)       # the plan, the markers and the results are the supervisor's (the ledger holds them)
        self._note_unverified(spec.number, gate)
        return gate

    def _scoring_gate(self, spec, stage_dir: Path, gate: GateResult) -> GateResult:
        """What only the supervisor can check for a scoring stage: the run's kind (from stage 0's ledger entry) is the kind of the files, and
        stage 2's reference files are the ones stage 1's gate accepted (their path and sha256 are in the ledger)."""
        kind = run_model_kind(self.ledger.read())
        reasons = list(gate.reasons)
        evidence = gate.evidence
        if spec.number == 1:
            try:
                doc = safeio.read_json(self.run_dir / scoring_check.PAIRS_FILE, max_bytes=scoring_check.JSON_MAX, expect=dict)
            except safeio.SafeReadError:
                doc = {}
            if doc.get("model_kind") != kind:
                reasons.append(f"the pair file is for kind {doc.get('model_kind')!r}; the run's model kind, from stage 0, is {kind!r}")
        else:
            reasons2, evidence = scoring_check.check(stage_dir, self.run_dir, kind)
            reasons += [r for r in reasons2 if r not in reasons]
            accepted = {}
            for e in self.ledger.read():
                if e["event"] == "stage_end" and e["stage"] == 1 and e["data"].get("result") == "pass":
                    accepted = {x.get("path"): x.get("sha256") for x in e["data"].get("evidence", []) if isinstance(x, dict)}
            for rel in (scoring_check.REFERENCE_FILE, scoring_check.PAIRS_FILE):
                try:
                    now = evidence_record(self.run_dir, self.run_dir / rel)["sha256"]
                except (safeio.SafeReadError, OSError, ValueError):
                    now = None
                if accepted.get(rel) is None or accepted[rel] != now:
                    reasons.append(f"{rel} is not the file stage 1's gate accepted (the ledger's stage 1 entry holds a different sha256 or none): "
                                   "it changed after stage 1 passed")
        return GateResult(not reasons, tuple(reasons), tuple(evidence))

    def _copy_package(self, bundle: Path) -> None:
        """Put stage 7's record, publish commands and cards in the operator bundle's package/
        folder, so the operator reads them with the rest. Nothing is copied when no package was
        staged. The bundle scrub then searches these copies too."""
        s7 = self.run_dir / "stages" / "7"
        # Each file is read through a descriptor pinned on the run directory, one component at a time without following a link, with a size
        # cap, and written with safeio.atomic_write. A file or a parent directory swapped for a link after a check, or a FIFO, is refused
        # at the read and the file is not copied (review of PR 403). Nothing is copied when no package.json can be read.
        wanted = [("stages/7/package.json", "package.json")]
        wanted += [("stages/7/PUBLISH_COMMANDS.txt", "PUBLISH_COMMANDS.txt")]
        wanted += [(os.path.relpath(card, self.run_dir), f"{card.parent.name}-README.md")
                   for card in sorted((s7 / "package").glob("*/README.md"))]
        got: dict[str, bytes] = {}
        with safeio.pinned_root(self.run_dir) as pin:
            for rel, name in wanted:
                try:
                    got[name] = safeio.read_bytes(rel, max_bytes=TEXT_MAX_BYTES, anchor=pin.fd)
                except safeio.SafeReadError:
                    continue
        if "package.json" not in got:
            return
        dest = bundle / "package"
        dest.mkdir(exist_ok=True)
        for name, data in got.items():
            safeio.atomic_write(dest / name, data, mode=0o644)

    def _gate_feedback(self, spec, stage_dir: Path, step: AgentStep, gate):
        """The step ended "done" and the exit gate failed. Tell the same conversation what the gate
        found and run it once more, with its own turn cap and its own log."""
        n = spec.number
        reasons = bound_reasons(gate.reasons)
        entry = self.ledger.append("decision", n, decision="gate feedback", phase=step.phase,
                                   reasons=reasons, file=f"stages/{n}/{spec.gate_file}")
        text = gate_feedback_text(spec, reasons)
        log = stage_dir / "log" / f"{step.phase}-continuation-{entry['seq']:05d}.jsonl"
        return step.continue_with(text, max_turns=AGENT_CONTINUATION_TURNS, log_path=log)

    def _served_chat_records(self, n: int, stage_dir: Path) -> None:
        """Issue 17. When stage 2 or stage 4 passes, check what each swap test served: the chat template, special tokens and stop ids of its
        model-dir against the new model's snapshot, and the chat replies its prompt suite recorded (orchard/served_chat.py). One
        served-chat.json per swap test: stage 2's directory, the graft and sampling trials' variant directories when they served (stage 2 may
        ship a variant), and each stage 4 configuration. A stage with no swap_config.json (the stock, text and audio stages) or whose new
        snapshot is not a directory gets none. The check never stops a run: a failure to write is a ledger decision, and the bundle then leaves
        the chat template hazard unverified."""
        if n == 2:
            tests = [(stage_dir, stage_dir)]
            for trial in ("graft-trial", "sampling-trial"):
                v = stage_dir / trial / "variant"
                if (v / "model-dir").is_dir() and (v / "evidence" / "swap-check.json").is_file():
                    tests.append((v, stage_dir))
        else:
            configs = stage_dir / "configs"
            try:
                tests = [(p, p) for p in sorted((p for p in configs.iterdir() if p.name.isdigit() and p.is_dir()), key=lambda p: int(p.name))]
            except OSError:
                tests = []
        for d, config_dir in tests:
            try:
                cfg = safeio.read_json(config_dir / "swap_config.json", max_bytes=1 << 20, expect=dict)
            except safeio.SafeReadError:
                continue
            new = cfg.get("new_snapshot")
            if not isinstance(new, str) or not new:
                continue
            new = str(self.run_dir / new) if not os.path.isabs(new) else new        # the sampling trial resolves a relative path against the run directory
            if not Path(new).is_dir() or not (d / "model-dir").is_dir() or not (d / "evidence" / "swap-check.json").is_file():
                continue
            try:
                rec = served_chat.write_record(d, d / "model-dir", new, d / "evidence" / "swap-check.json")
                self.ledger.append("evidence", n, what="served chat check", ok=rec["ok"], problems=[p["name"] for p in rec["problems"]],
                                   incomplete=len(rec["incomplete"]), **evidence_record(self.run_dir, d / served_chat.FILE))
            except (OSError, ValueError) as exc:
                self.ledger.append("decision", n, decision="served chat check not written", path=os.path.relpath(d / served_chat.FILE, self.run_dir),
                                   reason=f"{type(exc).__name__}: {exc}"[:200])

    def _decode_floor_record(self, n: int, stage_dir: Path) -> None:
        """Issue 181. Stage 0 writes stages/0/decode-floor.json (the DRAM-bandwidth floor for 1, 2 and 4 dies); stages 2 and 4 write
        decode-share.json (each decode speed they measured as a share of that floor). Both are estimates and say so. A model the floor
        cannot count, or a stage with no speed, leaves a record that says it was skipped or no file. The ledger gets an evidence entry."""
        try:
            if n == 0:
                rec = floor_report.write_floor(stage_dir, model_dir=self.inputs.get("model"))
                what, name = "decode floor", floor_report.FLOOR_FILE
                extra = {"scan": rec["scan"]}
            else:
                rec = floor_report.write_share(self.run_dir, stage_dir, n)
                if rec is None:
                    return
                what, name = "decode share", floor_report.SHARE_FILE
                extra = {"shares": [round(r["share"], 4) for r in rec["rows"]]}
            self.ledger.append("evidence", n, what=what, **extra, **evidence_record(self.run_dir, stage_dir / name))
        except (OSError, ValueError) as exc:       # a full disk or a permission failure must not end a passing stage
            self.ledger.append("decision", n, decision="decode record not written", reason=f"{type(exc).__name__}: {exc}"[:200])

    def _classify_kind(self) -> dict:
        """The model kind of this run (orchard/model_kind.py): the operator's `--model-kind`, else the model card's pipeline_tag, else the
        architecture names, else generative. The record names which one decided and goes in stage 0's `stage_end`, where a resume reads it.
        The snapshot is the run's `model=` input, else the one stage 0's triage config names (the directory stage 0 itself read)."""
        model_dir = self.inputs.get("model") or stage0_scan._triage_dirs(self.run_dir / "stages" / "0")[0]
        return model_kind.classify(model_dir, self.kind_override)

    def _stage_0_hazard_scan(self, stage_dir: Path, kind: str = "generative") -> None:
        """Write stages/0/hazard-scan.json (orchard/stage0_scan.py) and record it in the ledger. The scan predicts kernel hazards from the
        model's config; a hit never stops the run, and a model directory it cannot read gives a record that says it was skipped."""
        model_dir = self.inputs.get("model")
        try:
            rec = stage0_scan.write_scan(stage_dir, model_dir=model_dir, kind=kind)
            self.ledger.append("evidence", 0, what="hazard scan", scan=rec["scan"], hazards=[h["name"] for h in rec["hazards"]],
                               **evidence_record(self.run_dir, stage_dir / stage0_scan.FILE))
        except (OSError, ValueError) as exc:       # a full disk or a permission failure must not end a passing stage
            self.ledger.append("decision", 0, decision="hazard scan not written", reason=f"{type(exc).__name__}: {exc}"[:200])

    def _run_json(self, path, what: str, n: int | None = None) -> "safeio.Checked":
        """A JSON file of the run through the bounded safe reader (orchard/safeio.py): no FIFO or device, no link in the last component, a size
        cap, strict JSON. A file that is there and refused is not the same as a missing one, so the refusal is written to the ledger with its
        named reason (a later reader of the ledger, and the operator, see why the file was not used), and the caller gets the `Checked`
        (`.value` is None for a missing or refused file; `.refused` and `.reason` say which). Issue 287."""
        got = safeio.read_json_checked(path, max_bytes=RUN_JSON_MAX)
        if got.refused:
            self.ledger.append("decision", n, decision="a run file was refused and not used", file=what, reason=got.reason)
        return got

    def _run_delta(self) -> dict | None:
        """Stage 0's delta.json as a dict, or None when it is missing, refused (named in the ledger) or not an object."""
        value = self._run_json(self.run_dir / "stages" / "0" / "delta.json", "stages/0/delta.json").value
        return value if isinstance(value, dict) else None

    def _box_chips(self, spec=None) -> int | None:
        """The chips of the machine the hardware tests run on: the lab's stated count, or the boards that machine's adapter lists times
        CHIPS_PER_BOARD (`test_adapter` is the lab's adapter on a lab run, else the coder's). None when it cannot be told (an adapter that
        fails or lists nothing), and then no residency claim is made."""
        if self.lab is not None and getattr(self.lab, "chips", None):
            return int(self.lab.chips)
        try:
            boards = len({c.board for c in self.test_adapter.status()})
        except Exception:                           # an adapter failure is not a residency finding
            return None
        return boards * CHIPS_PER_BOARD or None

    def _stage_0_residency(self, stage_dir: Path) -> None:
        """Write stages/0/residency.json (orchard/residency.py) and record it in the ledger. The estimate comes from the model's safetensors
        headers; a model that needs more chips than the box has is a hazard in the record, and stage 2 of a full port reads it. A model
        directory without headers gives a record that says it was skipped."""
        try:
            # the run's largest required chip count is the sharding it asks for; one chip (or none stated) is no sharding
            sharded = max(self.required_chips) if self.required_chips and max(self.required_chips) > 1 else None
            rec = residency.write_residency(stage_dir, model_dir=self.inputs.get("model"), box_chips=self._box_chips(), sharded_chips=sharded, sharded_basis="required chip counts")
            extra = ({"chips_needed": rec["estimated"]["chips_needed"], "fits_box": rec["verdict"]["fits_box"]} if rec["scan"] == "ran" else {})
            self.ledger.append("evidence", 0, what="residency estimate", scan=rec["scan"], hazards=[h["name"] for h in rec["hazards"]], **extra,
                               **evidence_record(self.run_dir, stage_dir / residency.FILE))
        except (OSError, ValueError) as exc:       # a full disk or a permission failure must not end a passing stage
            self.ledger.append("decision", 0, decision="residency estimate not written", reason=f"{type(exc).__name__}: {exc}"[:200])

    def _hardware_pending(self, spec, *, resuming: bool, escalated: bool) -> bool:
        """Whether the attempt about to start runs the stage's hardware phase: a first attempt, a resume before the test ended, and a retry whose
        failed test is moved aside and rerun. False for a resume after the test ended, which only reads the result (`_stage_body` skips the
        prepare step and the test when `test-result.json` is there). The same rule as `_run_stage` uses to keep or move aside the record:
        the marker counts only when `stages.marker_state` accepts it, because `open_stage_dir` moves aside a marker the supervisor did not
        record (an agent can write the file, and a crash can fall between the write and its evidence entry) or the bounded reader refuses,
        and the hardware phase then runs again, so the machine check must too (review of PR 435, round 2)."""
        if not resuming:
            return True
        marker = self.run_dir / "stages" / str(spec.number) / SUPERVISOR_MARKER
        if marker_state(self.run_dir, spec.number, SUPERVISOR_MARKER, self.ledger)[0] != "ok":
            return True
        return bool(escalated and spec.boards and self._test_failure(spec, marker.parent))

    def _check_residency(self, spec, *, hardware_pending: bool = True) -> None:
        """Before a full-port stage 2: block with the numbers when stage 0's estimate needs more chips than the box has, or more than the
        stage's own lease gives the port check (`spec.boards` boards; the check runs on that lease, not on the whole box). The box is
        read again here, so the lab or board count at this point is the one judged. No record, a damaged or skipped one, or an unknown
        box is no claim and the stage goes on. The first boot's disk and host RAM are judged only when `hardware_pending`: a resume after the
        hardware test ended converts and downloads nothing, and the cache the test made may already fill the disk the check would demand
        again (review of PR 435, round 1). The chip counts are judged either way.

        Known limit (issue 460): MemAvailable is read here, before the hardware phase parks a large-tier coder, so the reading can be smaller than
        the one the conversion will see by about the coder's resident size. The check can then block a little early; it never passes what does not fit."""
        rec, box = residency.read_record(self.run_dir), self._box_chips(spec)
        stage = spec.boards * CHIPS_PER_BOARD
        why = (residency.shortfall(rec, box) or residency.shortfall(rec, stage, what=f"stage {spec.number}'s lease")) if box is not None else None
        if why:
            self._block(spec.number, why, need_chips=rec["estimated"]["chips_needed"], box_chips=box, stage_chips=stage,
                        device_bytes=rec["estimated"]["device_bytes"], label=rec["label"])
        if rec is not None and self.lab is None and hardware_pending and spec.draft == "full_port":
            # Only the full-port route converts the weights here: a scoring port (`scoring_check`, scripts/scoring_check/port_model.py) loads a
            # tokenizer and checks a server that is already running, so it has no conversion budget to be held to (review of PR 435, round 2).
            # The first boot converts the weights on this box (a lab run converts on the lab, which this box cannot read): the converted cache
            # and a download need disk under the cache root, and the conversion needs host RAM. Each is the record's estimate against a reading.
            free, mem = residency.machine_facts(self.paths.cache_root)
            why, notice = residency.machine_check(rec, free_disk_bytes=free, mem_available_bytes=mem)
            fb = rec["estimated"]["first_boot"] if (why or notice) else None
            if notice and not why:
                # The converted cache is a notice here: no port exists yet to say it writes one (review of PR 435, round 3).
                self._note("notice", what="the converted-weight cache may not fit", detail=notice, disk_bytes_needed=fb["disk_bytes"],
                           free_disk_bytes=free, label=rec["label"])
            if why:
                self._block(spec.number, why, disk_bytes_needed=fb["disk_bytes"], free_disk_bytes=free, host_ram_bytes_needed=fb["host_ram_bytes"],
                            mem_available_bytes=mem, label=rec["label"])

    def _check_port_cache(self, spec, stage_dir: Path) -> None:
        """After the coder has written the port and before its hardware test (issue 451): read what the port declares about a converted-weight
        cache (orchard/port_cache.py) and write it to the ledger with the hash of the bytes read, so a later edit cannot change what the check
        used. A port that declares a cache is held to the disk check for it: it blocks with both figures (block code `disk-full`) when the cache
        disk has less free than the first boot needs. A port that declares none, or nothing at all, is not blocked on the cache; a missing or
        unreadable declaration is a notice in the ledger. A lab run converts on the lab, so this box's disk is not read for it."""
        decl = port_cache.read(stage_dir / "port")
        self.ledger.append("decision", spec.number, decision="port cache declaration", state=decl["state"], why=decl["reason"],
                           cache_dir=decl["cache_dir"], sha256=decl["sha256"],
                           path=os.path.relpath(Path(stage_dir) / "port" / port_cache.FILE_NAME, self.run_dir))
        if decl["state"] in ("absent", "unreadable"):
            self._note("notice", what="the port declared no converted-weight cache",
                       detail=(decl["reason"] or f"the port has no {port_cache.FILE_NAME}") + ". The disk is not checked for a cache this port may write "
                       "(a loader that converts straight onto the chips writes none); the estimate was not measured")
        if decl["state"] != "writes" or self.lab is not None:
            return
        rec = residency.read_record(self.run_dir)
        free, _ = residency.machine_facts(self.paths.cache_root)
        why = residency.cache_shortfall(rec, free_disk_bytes=free)
        if why:
            fb = rec["estimated"]["first_boot"]
            self._block(spec.number, why, disk_bytes_needed=fb["disk_bytes"], free_disk_bytes=free, cache_dir=decl["cache_dir"],
                        declaration_sha256=decl["sha256"], label=rec["label"])

    def _stage_0_facts(self) -> dict:
        """`{"facts": {...}}` for stage 0's passing end, or {} when delta.json holds none (a text model: the default applies).
        The template decides form and modality from the files; `serving` is a property of this machine, so it comes from the
        installed bundles and container packages (orchard/audio_facts.py)."""
        facts = delta_facts(self.run_dir)
        if facts is None:
            return {}
        delta = self._run_delta()
        if delta is None:
            return {}
        installed = nearest.read_all_installed(self.paths.tt_model_root, self.paths.operator_home)
        return {"facts": {"form": facts.get("form"), "serving": serving_of(delta.get("nearest_model"), installed),
                          "modality": facts.get("modality")}}

    def _end_stage(self, spec, stage_dir: Path, status: str, reasons: list, gate, escalated: bool) -> str:
        """End a stage (see `_end_stage_body`). A pass records the files the gate accepted as evidence; when one of them can no longer be
        read through the bounded reader (swapped for a FIFO, a link or an oversize file after the gate looked, or a file the gate did not
        read), the stage fails with the named reason, which is the retry's feedback. A crash would give none."""
        try:
            return self._end_stage_body(spec, stage_dir, status, reasons, gate, escalated)
        except safeio.SafeReadError as exc:
            if status != "pass":
                raise
            # The refusal is a failed attempt like any other: it goes through the same path as a gate that failed, so the escalation and the
            # pause rules apply and a refusal that repeats does not rerun the stage without end.
            why = [f"stage {spec.number}: a file the gate accepted could not be read to record it as evidence: {exc.reason}"]
            return self._end_stage_body(spec, stage_dir, "fail", why, gate, escalated)

    def _end_stage_body(self, spec, stage_dir: Path, status: str, reasons: list, gate, escalated: bool) -> str:
        n = spec.number
        if n == 2 and spec.draft == "full_port" and getattr(self, "telegraph", None) is not None and status not in ("abort", "operator-pause"):
            self._report_phone_a_friend(n, helped=(status == "pass"))
        if status == "pass":
            ev = [evidence_record(self.run_dir, self.run_dir / rel) for rel in sorted(set(gate.evidence))]
            ev.append(evidence_record(self.run_dir, stage_dir / spec.gate_file))
            if n in (2, 4):
                self._served_chat_records(n, stage_dir)
            if n == 6 and spec.gate_file == "result.json":
                self._record_numbers(stage_dir)
            if n in (0, 2, 4):
                self._decode_floor_record(n, stage_dir)
            # Stage 0's pass records the path it chose. Later stages read it from here
            # (orchard/stages.py, run_path), so an edit to delta.json cannot change the run's path.
            kind_rec = self._classify_kind() if n == 0 else None
            if n == 0:
                self._stage_0_hazard_scan(stage_dir, kind_rec["kind"])
                self._stage_0_residency(stage_dir)
            extra = ({"path": delta_path(self.run_dir), "class": delta_class(self.run_dir), **self._stage_0_facts(), "model_kind": kind_rec}
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
        reasons = bound_reasons(reasons)
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
        data = safeio.read_json(stage_dir / "result.json", max_bytes=RUN_JSON_MAX, expect=dict)
        for num in data["numbers"]:
            self.ledger.append("measurement", 6, name=num["name"], value=num.get("value"),
                               unit=num["unit"], label=num["label"], evidence=num.get("evidence"))

    def _prepare_in_code(self, spec, stage_dir: Path) -> swap_prepare.PrepareOutcome:
        """Stage 2's prepare step on the weights-only path, done by the supervisor when `_draft` left a swap_config.json and the template
        (orchard/swap_prepare.py). Any other stage, and a stage with no drafted config, gets PrepareOutcome(False) and its agent step."""
        if spec.draft == "full_port_serve":
            return swap_prepare.PrepareOutcome(True)         # the supervisor wrote everything in `_draft`; no agent has a part in stage 5
        if spec.draft == "full_port_quality":
            return swap_prepare.PrepareOutcome(True)          # stage 6 has no prepare agent: the port is copied and the check is the template
        if spec.draft == "full_port":
            seed = getattr(self, "_port_seed", None) or {}
            if spec.number == 3:
                return swap_prepare.PrepareOutcome(False, note=port_facts.stage_3_note(chips=self._stage_3_chips() or 0,
                                                                                       copied=bool(seed.get("copied") or seed.get("kept"))))
            note = port_facts.prepare_note(facts=bool(seed.get("facts")), skeleton=bool(seed.get("skeleton")), excerpts=bool(seed.get("excerpts")))
            if spec.number == 2 and getattr(self, "telegraph", None) is not None:
                friend = self._phone_a_friend(spec, stage_dir)
                if friend:
                    note = note + "\n\n" + friend
            return swap_prepare.PrepareOutcome(False, note=note)
        if spec.skill != "weights-swap-check" or spec.draft != "swap_config":
            return swap_prepare.PrepareOutcome(False)
        outcome = swap_prepare.prepare(run_dir=self.run_dir, stage_dir=stage_dir, env=agent_env(self.run_dir, extra=self.extra_env),
                                       stage=spec.number)
        if outcome.output is not None:
            self.ledger.append("decision", spec.number, decision="prepare run by the supervisor", returncode=outcome.returncode,
                               done=outcome.done, **evidence_record(self.run_dir, stage_dir / outcome.output))
        return outcome

    # ---- phone a friend: the outside helper drafts a port when stage 2 keeps failing (orchard/phone_a_friend.py) -------------------

    def _friend_brief(self, spec, stage_dir: Path, entries: list[dict]) -> str:
        """The messages the coder's own prepare step gets (the skill and the task), as one text for the helper."""
        skill = resolve_skill(spec.skill, self.skills_dirs)
        if skill is None:
            raise ValueError("the skill is not in the skills directories")
        refs = {r: resolve_skill(r, self.skills_dirs) for r in spec.refs}
        system, user = build_messages(spec=spec, phase="prepare", run_dir=self.run_dir, stage_dir=stage_dir, skill_path=skill, refs=refs,
                                      facts=facts_from(run_progress(entries).run_start, self.run_dir), entries=entries, resumed=False,
                                      paths=self.paths)
        return f"{system}\n\n{user}"

    def _friend_failure(self, n: int, stage_dir: Path, entries: list[dict]) -> str:
        """Why the earlier attempts failed, for the helper: the reasons the latest failed attempt ended with, the watchdog's findings since the
        previous end, and the end of the port check's output when this attempt's directory holds one."""
        reasons: list = []
        watchdog: list = []
        for e in entries:
            if e.get("stage") != n:
                continue
            data = e.get("data") if isinstance(e.get("data"), dict) else {}
            if e["event"] == "stage_end" and data.get("result") not in ("pass", "skipped"):
                reasons = list(data.get("reasons") or [])[:8]
        for e in entries:
            data = e.get("data") if isinstance(e.get("data"), dict) else {}
            if e.get("stage") == n and e["event"] in ("retry", "escalate"):
                for f in (data.get("findings") or [])[:4]:
                    if isinstance(f, dict) and isinstance(f.get("summary"), str):
                        watchdog.append(f["summary"][:300])
        output = ""
        try:
            output = safeio.read_bytes(Path(stage_dir) / port_repair.OUTPUT, max_bytes=phone_a_friend.READ_MAX).decode("utf-8", "replace")
        except (safeio.SafeReadError, OSError):
            pass
        return phone_a_friend.failure_text(reasons, watchdog[-6:], output, self._homes())

    def _friend_refused(self, n: int, reason: str) -> None:
        """The helper was due and was not asked. The ledger says why; the stage carries on as before."""
        self.ledger.append("decision", n, decision=phone_a_friend.NOT_ASKED, reason=reason[:300])

    def _phone_a_friend(self, spec, stage_dir: Path) -> str | None:
        """For a full-port stage 2 attempt: when the ledger says the helper is due, ask it once for a draft `port_model.py`, write the draft
        to `stages/2/telegraph/draft-<k>.py` and return the sentence that tells the coder where it is. None in every other case. Nothing here
        fails the stage: a refusal, an unusable reply and a failed write each leave a ledger decision and None."""
        n = spec.number
        entries = self.ledger.read()
        state = phone_a_friend.attempt_state(entries, n)
        is_due, why = phone_a_friend.due(state)
        if not is_due:
            return None
        spent = repair_help.asks_in_stage(entries, n)
        if spent >= repair_help.TOTAL_PER_STAGE:
            # The stage's limit counts the repair-round asks too (review of PR 496): an attempt that follows six repair asks sends no seventh call.
            self._friend_refused(n, f"the helper was already asked {spent} times in this stage, repair rounds included (the limit is {repair_help.TOTAL_PER_STAGE})")
            return None
        k = state["asked_in_stage"] + 1
        stage_dir = Path(stage_dir)
        try:
            brief = self._friend_brief(spec, stage_dir, entries)
            facts, _ = phone_a_friend.read_input(stage_dir / port_facts.FACTS_NAME, phone_a_friend.FACTS_CAP)
            excerpts, _ = phone_a_friend.read_input(stage_dir / "reference_excerpts.md", phone_a_friend.EXCERPTS_CAP)
            task, sent = phone_a_friend.build_task(brief=brief, facts=facts, excerpts=excerpts,
                                                   listing=phone_a_friend.port_listing(stage_dir / port_repair.PORT_DIR), homes=self._homes())
            failure = self._friend_failure(n, stage_dir, entries)
        except Exception as exc:     # the request could not be built (an unreadable skill or file): the helper is not asked, the stage goes on
            self._friend_refused(n, f"the request could not be built ({exc.__class__.__name__})")
            return None
        from orchard import telegraph as tg
        room = self.telegraph.cfg.max_prompt_chars - len(tg.SYSTEM)
        # The helper cuts the middle out of a request over its `max_prompt_chars`; the entry says so, so a cap set for short advice is seen.
        self.ledger.append("decision", n, decision=phone_a_friend.ASKED, k=k, why=why, problem=phone_a_friend.problem_id(self.run_dir), inputs=sent,
                           prompt_chars=len(task) + len(failure), max_prompt_chars=self.telegraph.cfg.max_prompt_chars,
                           over_cap=len(task) + len(failure) > room, failed_attempts=state["failed"])
        try:
            answer = self.telegraph.ask(phone_a_friend.problem_id(self.run_dir), task, failure)
        except Exception as exc:     # `ask` returns its failures; this is for one it did not foresee
            self.ledger.append("decision", n, decision=phone_a_friend.ANSWERED, ok=False, call_id="",
                               reason=f"the helper raised {exc.__class__.__name__}")
            return None
        record = {"ok": bool(answer.ok), "call_id": answer.call_id, "model": answer.model, "tokens_in": answer.tokens_in,
                  "tokens_out": answer.tokens_out, "cost_usd": answer.cost_usd, "cached": bool(answer.cached)}
        if not answer.ok:
            self.ledger.append("decision", n, decision=phone_a_friend.ANSWERED, reason=str(answer.reason)[:300], **record)
            return None
        draft = phone_a_friend.extract_draft(answer.text)
        if draft is None:
            self.ledger.append("decision", n, decision=phone_a_friend.ANSWERED, **record)
            self.ledger.append("decision", n, decision=phone_a_friend.DRAFT_FAILED, call_id=answer.call_id,
                               reason="the reply holds no Python code block")
            return None
        ok_py, why_not = phone_a_friend.parses(draft)
        try:
            path = phone_a_friend.write_draft(stage_dir, k, draft, answer.call_id, answer.model)
            evidence = evidence_record(self.run_dir, path)
        except (OSError, safeio.SafeReadError) as exc:
            self.ledger.append("decision", n, decision=phone_a_friend.ANSWERED, **record)
            self.ledger.append("decision", n, decision=phone_a_friend.DRAFT_FAILED, call_id=answer.call_id,
                               reason=f"the draft could not be written: {exc.__class__.__name__}")
            return None
        self.ledger.append("decision", n, decision=phone_a_friend.ANSWERED, draft=evidence, parses=ok_py, **record)
        placed = False
        if ok_py:
            problems = phone_a_friend.draft_problems(draft)
            if problems:
                self._friend_stub(n, answer.call_id, problems)
            else:
                placed = self._place_friend_draft(n, stage_dir, path, answer.call_id)
        return phone_a_friend.coder_note(evidence["path"], answer.call_id, ok_py, why_not, placed=placed)

    def _place_standing_draft(self, spec, stage_dir: Path) -> None:
        """A later attempt of stage 2 starts from a fresh skeleton. When the helper is not due this attempt (its calls are used up, or no attempt
        has failed) and an earlier attempt left a draft that is real code, put that draft in the fresh skeleton's place, so the attempt starts
        from code the check can run. A due helper is asked instead, and its new draft is placed (`_phone_a_friend`). Advisory: nothing here fails
        the stage, and a port the coder has written is never replaced."""
        n = spec.number
        if n != 2 or getattr(self, "telegraph", None) is None:
            return
        try:
            if phone_a_friend.due(phone_a_friend.attempt_state(self.ledger.read(), n))[0]:
                return
            draft = phone_a_friend.standing_draft(stage_dir)
            if draft is None:
                return
            self._place_friend_draft(n, Path(stage_dir), draft, "")
        except Exception as exc:     # advisory: an error nobody listed leaves the skeleton as it was
            self.ledger.append("decision", n, decision=phone_a_friend.NOT_PLACED, call_id="", reason=f"{exc.__class__.__name__}: an unexpected error")

    def _friend_stub(self, n: int, call_id: str, problems: list[str]) -> None:
        """A draft that parses but is a stub (a required function missing, empty or raising NotImplementedError) helped nobody: say so now, so the
        helper's ladder takes the next question to the next model of its list (`Telegraph._choose_model` moves up one step on a not-helped
        verdict), and the ledger records why. Run9 of the tiny Qwen4 rehearsal: the cheapest model's stub, then eight reads of the excerpts."""
        from orchard import telegraph as tg
        self.ledger.append("decision", n, decision=phone_a_friend.DRAFT_STUB, call_id=call_id, problems=[str(p)[:120] for p in problems[:6]],
                           effect="reported as not helped; the next ask in this stage goes to the next model of the helper's list")
        reason = "stub draft: " + "; ".join(str(p)[:80] for p in problems[:3])
        try:
            self.telegraph.report(call_id, False, note=f"stage {n}: {reason}")
        except (KeyError, OSError, ValueError, *tg.LEDGER_ERRORS) as exc:
            reason += f" (the verdict could not be written: {exc.__class__.__name__})"
        self.ledger.append("decision", n, decision=phone_a_friend.REPORTED, call_id=call_id, helped=False, reason=reason[:300])

    def _place_friend_draft(self, n: int, stage_dir: Path, draft_path, call_id: str) -> bool:
        """Put a draft that parses and is not a stub into `stages/2/port/port_model.py` when that file is still the unchanged skeleton, so the
        supervisor's check runs against real code and its failures become the coder's next input. The ledger holds the draft's hash and the new
        port's hash, or why nothing was placed. A file the coder has written is never replaced."""
        try:
            skeleton = safeio.read_bytes(port_facts.SKELETON, max_bytes=port_facts.SKELETON_MAX)
            ok, why = phone_a_friend.place_draft(stage_dir, draft_path, skeleton)
            if ok:
                port = Path(stage_dir) / "port" / port_facts.SKELETON_NAME
                record = evidence_record(self.run_dir, port)
                self.ledger.append("decision", n, decision=phone_a_friend.PLACED, call_id=call_id, path=record["path"], port_sha256=record["sha256"],
                                   draft_sha256=evidence_record(self.run_dir, Path(draft_path))["sha256"],
                                   skeleton_sha256=hashlib.sha256(skeleton).hexdigest())
                return True
        except (OSError, safeio.SafeReadError) as exc:
            ok, why = False, f"the files could not be read ({exc.__class__.__name__})"
        self.ledger.append("decision", n, decision=phone_a_friend.NOT_PLACED, call_id=call_id, reason=str(why)[:300])
        return False

    # ---- the outside helper inside the port repair rounds (orchard/repair_help.py) -----------------------------------------------------------

    def _repair_help_refused(self, n: int, k: int, reason: str) -> None:
        self.ledger.append("decision", n, decision=repair_help.NOT_ASKED, round=k, reason=reason[:300])

    def _consume_handled_stop(self, n: int, status: str) -> None:
        """The helper's port took over from a repair step the watchdog stopped, so the stop is handled: the actuator forgets it (the next repair
        step would otherwise end before it sent a request) and a recorded watchdog pause is answered by a resume entry (the run would otherwise wait
        for the operator at the stage's end). An operator abort or pause is never consumed here: only these two statuses reach this method
        (review of PR 496)."""
        self.actuator.clear()
        if status == "pause":
            self.ledger.append("decision", n, decision="resume", by="helper rescue",
                                note="a helper-supplied port took over from the repair step the watchdog paused")

    def _repair_help(self, spec, stage_dir: Path, k: int, limit: int, failure: dict, raw: str, *, why: str, coder_failed: bool) -> tuple[str, bool, bool]:
        """Ask the helper for the whole corrected `port_model.py` in repair round `k` and, when the answer is a whole port and the file on disk is
        still the one that was sent (or the coder's step failed), put it in place. Returns (the sentence for the coder, whether a file was
        placed, whether an ask was written). Nothing here fails the stage: every refusal and every unusable answer is a ledger decision."""
        n, stage_dir = spec.number, Path(stage_dir)
        entries = self.ledger.read()
        ok, reason = repair_help.due(entries, n, k)
        if not ok:
            self._repair_help_refused(n, k, reason)
            return "", False, False
        port_bytes, why_not = repair_help.read_port(stage_dir)
        if port_bytes is None:
            self._repair_help_refused(n, k, why_not)
            return "", False, False
        try:
            facts, _ = phone_a_friend.read_input(stage_dir / port_facts.FACTS_NAME, phone_a_friend.FACTS_CAP)
            excerpts, _ = phone_a_friend.read_input(stage_dir / "reference_excerpts.md", phone_a_friend.EXCERPTS_CAP)
            task, sent = repair_help.build_task(facts=facts, excerpts=excerpts, port_text=port_bytes.decode("utf-8", "replace"),
                                                listing=phone_a_friend.port_listing(stage_dir / port_repair.PORT_DIR), homes=self._homes())
            fail = repair_help.failure(k, limit, failure.get("op"), raw, self._homes())
        except Exception as exc:     # the request could not be built: the helper is not asked, the round goes on
            self._repair_help_refused(n, k, f"the request could not be built ({exc.__class__.__name__})")
            return "", False, False
        if task is None:
            self._repair_help_refused(n, k, f"the port is {sent['port']['chars']} characters, more than the {repair_help.PORT_CAP} the request may hold")
            return "", False, False
        from orchard import telegraph as tg
        room = self.telegraph.cfg.max_prompt_chars - len(tg.SYSTEM)
        if len(task) + len(fail) > room:
            # The helper cuts the middle out of a request over its prompt cap, and its answer would then replace the whole file from a partial
            # reading of it: the request is not sent (review of PR 496).
            self._repair_help_refused(n, k, f"the request is {len(task) + len(fail)} characters, more than the {room} the helper's prompt cap leaves room for")
            return "", False, False
        total = repair_help.asks_in_stage(entries, n) + 1
        sent_sha = hashlib.sha256(port_bytes).hexdigest()
        problem = phone_a_friend.problem_id(self.run_dir)
        self.ledger.append("decision", n, decision=repair_help.ASKED, round=k, of=limit, k=total, why=why, problem=problem, inputs=sent,
                           prompt_chars=len(task) + len(fail), max_prompt_chars=self.telegraph.cfg.max_prompt_chars,
                           over_cap=len(task) + len(fail) > room, port_sha256=sent_sha, failure_op=failure.get("op"), coder_failed=coder_failed)
        try:
            answer = self.telegraph.ask(problem, task, fail)
        except Exception as exc:     # `ask` returns its failures; this is for one it did not foresee
            self.ledger.append("decision", n, decision=repair_help.ANSWERED, ok=False, call_id="", reason=f"the helper raised {exc.__class__.__name__}")
            return "", False, True
        record = {"ok": bool(answer.ok), "call_id": answer.call_id, "model": answer.model, "tokens_in": answer.tokens_in,
                  "tokens_out": answer.tokens_out, "cost_usd": answer.cost_usd, "cached": bool(answer.cached)}
        if not answer.ok:
            self.ledger.append("decision", n, decision=repair_help.ANSWERED, reason=str(answer.reason)[:300], **record)
            return "", False, True
        reply = phone_a_friend.extract_draft(answer.text)
        if reply is None:
            self.ledger.append("decision", n, decision=repair_help.ANSWERED, **record)
            self.ledger.append("decision", n, decision=repair_help.NOT_PLACED, call_id=answer.call_id, reason="the reply holds no Python code block")
            return "", False, True
        ok_py, why_py = phone_a_friend.parses(reply)
        try:
            path = phone_a_friend.write_draft(stage_dir, total, reply, answer.call_id, answer.model, prefix=repair_help.PREFIX)
            evidence = evidence_record(self.run_dir, path)
        except (OSError, safeio.SafeReadError) as exc:
            self.ledger.append("decision", n, decision=repair_help.ANSWERED, **record)
            self.ledger.append("decision", n, decision=repair_help.NOT_PLACED, call_id=answer.call_id,
                               reason=f"the reply could not be written: {exc.__class__.__name__}")
            return "", False, True
        self.ledger.append("decision", n, decision=repair_help.ANSWERED, draft=evidence, parses=ok_py, **record)
        if not ok_py:
            self.ledger.append("decision", n, decision=repair_help.NOT_PLACED, call_id=answer.call_id, reason=f"the reply does not parse ({why_py})")
            return "", False, True
        problems = repair_help.reply_problems(reply)
        if problems:
            self.ledger.append("decision", n, decision=repair_help.STUB, call_id=answer.call_id, problems=[str(p)[:120] for p in problems[:6]],
                               effect="reported as not helped; the next ask in this stage goes to the next model of the helper's list")
            self._repair_report(n, k, answer.call_id, failure.get("op"), None, helped=False, outcome="stub")
            return "", False, True
        placed, why_placed, found = repair_help.place(stage_dir, reply, sent_sha, coder_failed=coder_failed, call_id=answer.call_id, model=answer.model)
        if not placed:
            self.ledger.append("decision", n, decision=repair_help.NOT_PLACED, call_id=answer.call_id, reason=why_placed, sent_sha256=sent_sha,
                               found_sha256=found)
            return "", False, True
        port = stage_dir / port_repair.PORT_DIR / port_facts.SKELETON_NAME
        self.ledger.append("decision", n, decision=repair_help.PLACED, call_id=answer.call_id, path=evidence_record(self.run_dir, port)["path"],
                           port_sha256=evidence_record(self.run_dir, port)["sha256"], sent_sha256=sent_sha, found_sha256=found,
                           why="the file was still the one that was sent" if found == sent_sha else "the coder's step failed",
                           draft_sha256=evidence["sha256"])
        return repair_help.coder_note(answer.call_id, True, evidence["path"]), True, True

    def _repair_report(self, n: int, k, call_id: str, before_op, after_op, *, helped: bool, outcome: str) -> None:
        """The verdict on one repair answer, to the helper's ledger and the run's. A verdict that cannot be written is noted and goes on."""
        from orchard import telegraph as tg
        reason = ""
        try:
            self.telegraph.report(call_id, helped, note=f"stage {n} repair round {k}: {outcome}")
        except (KeyError, OSError, ValueError, *tg.LEDGER_ERRORS) as exc:
            reason = f"the verdict could not be written ({exc.__class__.__name__})"
        self.ledger.append("decision", n, decision=repair_help.PROGRESS, call_id=call_id, round=k, before_op=before_op, after_op=after_op,
                           outcome=outcome, helped=helped, **({"reason": reason} if reason else {}))

    def _repair_outcome(self, stage_dir) -> tuple[bool, str | None, bool]:
        """(the check passed, the request it failed at, whether the record could be read) from the supervisor's test record."""
        try:
            data = safeio.read_json(Path(stage_dir) / port_repair.MARKER, max_bytes=full_port.JSON_MAX, expect=dict)
        except (safeio.SafeReadError, OSError, ValueError):
            return False, None, False
        if data.get("returncode") == 0 and data.get("timed_out") is False:
            return True, None, True
        failure = port_repair.read_failure(self.run_dir, stage_dir)
        if failure is None:
            return False, None, False
        return False, failure.get("op"), True

    def _repair_verdicts(self, n: int, stage_dir, *, final_helped: bool | None = None) -> None:
        """Verdicts for the repair answers that have none. After a rerun of the check (`final_helped` None): helped when it passed or now fails
        later in the request chain than where the helper was asked (`repair_help.progressed`); a different message at the same request is not
        progress. At the end of an attempt (`final_helped` given) an answer the check never judged gets the attempt's outcome."""
        entries = self.ledger.read()
        pending = repair_help.unreported(entries, n)
        if not pending:
            return
        passed, after_op, readable = self._repair_outcome(stage_dir) if stage_dir is not None else (False, None, False)
        for item in pending:
            judged = repair_help.tested_after(entries, n, item["seq"]) and readable
            if not judged and final_helped is None:
                continue
            if judged and passed:
                helped, outcome = True, "passed"
            elif judged:
                helped = repair_help.progressed(item["failure_op"], after_op)
                outcome = "further" if helped else ("same request" if repair_help.rank(item["failure_op"]) == repair_help.rank(after_op) else "no progress")
            else:
                helped, outcome = bool(final_helped), "the attempt ended before the check judged it"
            self._repair_report(n, item["round"], item["call_id"], item["failure_op"], after_op if judged else None, helped=helped, outcome=outcome)

    def _report_phone_a_friend(self, n: int, *, helped: bool) -> None:
        """Tell the helper's ledger how the attempt it helped with ended, and note it in the run's ledger. A failed report is noted and goes on."""
        from orchard import telegraph as tg
        self._repair_verdicts(n, None, final_helped=helped)        # repair answers the check never judged get the attempt's outcome
        for call_id in phone_a_friend.unreported(self.ledger.read(), n):
            reason = ""
            try:
                self.telegraph.report(call_id, helped, note=f"stage {n}: the attempt {'passed' if helped else 'did not pass'} after the draft")
            except (KeyError, OSError, ValueError, *tg.LEDGER_ERRORS) as exc:
                reason = f"the verdict could not be written ({exc.__class__.__name__})"
            self.ledger.append("decision", n, decision=phone_a_friend.REPORTED, call_id=call_id, helped=helped,
                               **({"reason": reason} if reason else {}))

    def _step(self, spec, phase: str, stage_dir: Path, escalated: bool, resumed: bool, extra: str | None = None):
        """One agent step. `extra`, when given, is added to the task message (stage 2's prepare gets the script's message)."""
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
        tcfg = self.cfg.tiers[used]
        unstick = runaway.unstick_from_tier(tcfg)
        # What the replies of this step may use (issue 458): the cap and sampling of the tier, and the retry profile after a capped reply.
        step_sampling = {"max_tokens": tcfg.get("max_tokens", AGENT_MAX_TOKENS), "temperature": tcfg.get("temperature", 0),
                    **{k: tcfg[k] for k in ("frequency_penalty", "repetition_penalty", "top_p", "top_k") if k in tcfg},
                    "retry_after_cap": unstick.record(), "restarts_on_caps": AGENT_RESTARTS_ON_CAPS}
        self.ledger.append("decision", n, decision="agent step", phase=phase, tier=used, model=model,
                           escalated=escalated, skill=str(skill), sampling=step_sampling)
        step = AgentStep(agent=AGENT, endpoint=endpoint, model=model,
                         max_tokens=step_sampling["max_tokens"], temperature=step_sampling["temperature"],
                         frequency_penalty=tcfg.get("frequency_penalty"), repetition_penalty=tcfg.get("repetition_penalty"),
                         top_p=tcfg.get("top_p"), top_k=tcfg.get("top_k"),
                         unstick=unstick, restart_on_caps=AGENT_RESTARTS_ON_CAPS,
                         tools=Tools(self.run_dir, stage_dir, agent_env(self.run_dir, extra={
                             **self.extra_env, "TT_MODEL_MODELS_DIR": str(self.paths.tt_model_root)}),
                             scratch=(self.run_dir / "tt-metal-scratch"
                                      if scratch_stage(n, run_path(entries, self.run_dir)) else None),
                             dedupe=self.tool_dedupe),
                         ledger=self.ledger, stage=n, phase=phase, feed=self.watchdog.feed,
                         control=self.actuator, run_dir=self.run_dir,
                         evidence_dir=stage_dir / "evidence",
                         log_path=stage_dir / "log" / f"{phase}-{len(entries) + 1:05d}.jsonl",
                         http=self.http, clock=self.clock, guard=self.guard,
                         finished=self._finished_check(spec, phase, stage_dir))
        name = deliverable(spec, phase)
        step.deliverable_stamp = file_stamp(stage_dir / name) if name else None
        return step.run(system, f"{user}\n\n## From the supervisor\n{extra}" if extra else user), step

    def _draft(self, spec, stage_dir: Path) -> None:
        """Write the config a stage's skill starts from, when the run already holds every fact in it
        (orchard/swap_draft.py), and copy the templates that read it. A config already in the stage
        directory (a resumed stage) is kept. When a fact is missing or ambiguous nothing is written,
        the ledger says why, and the agent finds the facts as its skill describes."""
        if spec.draft == "stage4":
            return self._draft_stage4(spec, stage_dir)
        if spec.draft == "stock_config":
            return self._draft_stock(spec, stage_dir)
        if spec.draft == "full_port":
            self._pin_full_port(spec, stage_dir)
            self._seed_full_port(spec, stage_dir)
            return self._place_standing_draft(spec, stage_dir)
        if spec.draft == "scoring_check":
            return self._pin_scoring(spec, stage_dir)
        if spec.draft == "full_port_serve":
            return self._prepare_serve(spec, stage_dir)
        if spec.draft == "full_port_quality":
            full_port_quality_stage.pin(self, spec, stage_dir)         # a missing snapshot is reported again, as a failure, after the prepare phase
            return self._seed_full_port(spec, stage_dir)
        if spec.draft != "swap_config" or (stage_dir / "swap_config.json").exists():
            return
        # Before the config is drafted and before any agent step of this stage: a bundle that cannot be chosen (the draft gives None) still
        # leaves the prepare agent and the hardware test to run, and the gate must hold the reference ids' hash for those runs too.
        self._pin_reference_ids(spec)
        cfg, problems = swap_draft.draft(run_dir=self.run_dir, tt_model_root=self.paths.tt_model_root,
                                         cache_root=self.paths.cache_root, hf_home=self.paths.hf_home,
                                         inputs=self.inputs, chips=spec.boards * CHIPS_PER_BOARD,
                                         operator_home=self._container_home(spec))
        if cfg is None:
            self.ledger.append("decision", spec.number, decision="swap config not drafted", problems=problems)
            return
        # Stage 1 may have written the CPU reference for the agreement set (issue 15). When it did, the run uses the set: the hash of that file goes
        # into the ledger now, before any stage 2 test, and the gate compares the file with it later.
        reference = agreement_reference_record(self.run_dir)
        if reference is not None:
            cfg["agreement_set"] = True
            self.ledger.append("decision", spec.number, decision=AGREEMENT_LEDGER_DECISION, **reference)
        target = stage_dir / "swap_config.json"
        _put(target, json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        templates = Path(self.paths.orchard_dir) / "orchard" / "skills" / "weights-swap-templates"
        copied = []
        for name in (*(STAGE4_TEMPLATES if "package" in cfg else SWAP_TEMPLATES), *(AGREEMENT_TEMPLATES if cfg.get("agreement_set") else ())):      # a container package also needs its script
            shutil.copy2(templates / name, stage_dir / name)
            copied.append(name)
        self.ledger.append("decision", spec.number, decision="swap config drafted",
                           config=evidence_record(self.run_dir, target), bundle_dir=cfg.get("bundle_dir"),
                           copied=copied, **({"package": cfg["package"]} if "package" in cfg else {}))

    def _pin_reference_ids(self, spec) -> None:
        """The 32-token check's reference ids (issue 433): the hash of stage 1's generated-ids.json goes into the ledger before any stage 2 test,
        and the gate recomputes the check from the raw replies and these ids, so a swap-check.json and a result.json edited together are caught.
        Recorded once: a retried draft finds the record and keeps the first hash, so a file changed after the first pin is never taken as the
        new truth. Nothing is recorded when stage 1 wrote no ids (the gate then judges the run as before)."""
        for e in self.ledger.read():
            if e["event"] == "decision" and e["data"].get("decision") == REFERENCE_IDS_LEDGER_DECISION:
                return
        record = reference_ids_record(self.run_dir)
        if record is not None:
            self.ledger.append("decision", spec.number, decision=REFERENCE_IDS_LEDGER_DECISION, **record)

    def _prepare_serve(self, spec, stage_dir: Path) -> None:
        """Stage 5 of a full port (orchard/full_port_serve_stage.py): copy the port that passed stage 2 or 3, the shipped templates, the acceptance
        corpus and the config into the stage directory, and write the ledger entries that hold their hashes. The problems, if any, are kept for
        `_stage_body`, which fails the attempt with them."""
        done = full_port_serve_stage.prepare(self.run_dir, stage_dir, self.ledger.read(), snapshot=self.inputs.get("model"), orchard_dir=self.paths.orchard_dir)
        self._serve_problems = done.problems
        if done.problems:
            self.ledger.append("decision", spec.number, decision="serving check not prepared", problems=[p[:300] for p in done.problems[:5]])
            return
        self.ledger.append("decision", spec.number, decision="serving corpus recorded", sha256=done.corpus_sha256, positions=done.positions,
                           source_stage=done.source)
        self.ledger.append("decision", spec.number, decision="serving port copied", source_stage=done.source, files=done.port_files)
        self.ledger.append("decision", spec.number, decision="serving check templates copied", **done.pinned)

    def _derive_serve_result(self, spec, stage_dir: Path) -> bool:
        """Stage 5's result.json, written by the supervisor from the check's record (orchard/full_port_serve.py `derive`). The gate that runs next
        judges it; a failed or missing test leaves no file, and the ledger says why."""
        n = spec.number
        result, why = full_port_serve.derive(stage_dir, self.run_dir, rehearsal=full_port_rehearsal(self.ledger.read()))
        if result is None:
            self.ledger.append("decision", n, decision="result not derived by the supervisor", why=why[:300])
            return False
        path = stage_dir / "result.json"
        try:
            safeio.atomic_write(path, (json.dumps(result, indent=2) + "\n").encode("utf-8"), mode=0o644)
        except OSError as exc:
            self.ledger.append("decision", n, decision="result not derived by the supervisor", why=f"could not write it ({exc.__class__.__name__})")
            return False
        self.ledger.append("evidence", n, what="result.json derived by the supervisor", **(full_port.file_record(self.run_dir, path) or {}))
        self.ledger.append("decision", n, decision="result derived by the supervisor", file=f"stages/{n}/result.json", coherent=result["coherent"],
                           corpus_matched=result["corpus_matched"], corpus_positions=result["corpus_positions"])
        return True

    def _seed_full_port(self, spec, stage_dir: Path) -> None:
        """What the coder starts stage 2 of a full port from (orchard/port_facts.py): a facts file read from the checkpoint and a port
        skeleton. Three unattended rehearsal runs showed the coder's first attempt reading for 20 turns and writing nothing. A checkpoint
        or config that cannot be read leaves no facts file, the ledger says why, and the coder works from its skill as before. A port
        already in the stage directory (a resumed stage) is kept."""
        if spec.number in (3, full_port_quality.STAGE):
            return self._seed_stage_3_port(spec, stage_dir)
        snapshot = self.inputs.get("model")
        facts_written = excerpts_written = False
        if snapshot:
            root = port_facts.reference_root(port_facts.default_run, self.inputs.get("reference_python"))
            lookup = (lambda model_type: port_facts.reference_source(root, model_type)) if root is not None else None
            summary = None
            if root is not None:            # reference excerpts (issue 427): the layer classes' source, in a file of its own that the facts point at
                excerpts_written, summary = write_reference_excerpts(self.ledger, self.run_dir, spec, stage_dir, snapshot, root)
            try:
                facts = port_facts.build(snapshot, reference_lookup=lookup, excerpts=summary)
                record = evidence_record(self.run_dir, port_facts.write(stage_dir, facts))
            except (port_facts.PortFactsError, OSError, safeio.SafeReadError) as exc:
                self.ledger.append("decision", spec.number, decision="port facts not written", problem=str(exc)[:300])
            except Exception as exc:         # advisory step: an error nobody listed (a hostile tensor name) must not stop the run; its text may quote the checkpoint
                self.ledger.append("decision", spec.number, decision="port facts not written",
                                   problem=f"{exc.__class__.__name__}: an unexpected error while reading the checkpoint")
            else:
                facts_written = True
                self.ledger.append("decision", spec.number, decision="port facts written", facts=record,
                                   tensor_groups=facts["tensor_groups"], tensors_left_out=facts["tensors_left_out"],
                                   conventions=[c["field"] for c in facts["conventions"]],
                                   reference_found=root is not None)
        try:
            skeleton = port_facts.copy_skeleton(stage_dir)
        except (OSError, safeio.SafeReadError) as exc:       # a full or read-only stage directory, or no shipped skeleton: the coder writes the port from nothing, as before
            skeleton = False
            self.ledger.append("decision", spec.number, decision="port skeleton not copied", problem=exc.__class__.__name__)
        if skeleton:
            self.ledger.append("decision", spec.number, decision="port skeleton copied")
        self._port_seed = {"facts": facts_written, "skeleton": skeleton, **({"excerpts": True} if excerpts_written else {})}

    def _seed_stage_3_port(self, spec, stage_dir: Path, *, source: int = 2) -> None:
        """What the coder starts stage 3 of a full port from: the port that passed stage 2, copied into stages/3/port/ (issue 326). Stage 4's
        entries (issue 454) call this with their own directory as `stage_dir` and the stage whose port passed as `source` (3, or 2 when
        stage 3 was skipped); the copy is the same and its ledger entries name the entry's paths. Regular
        files only, with a count and a size cap, and no `__pycache__`; a link (never followed), a file over the cap, too many files, too many
        entries of any kind or too many directory levels stop the copy, the ledger says why and how many entries were read, and the coder
        writes the port again from stage 2's. The directory is read lazily (`_SeedScan`), so a hostile port directory is refused after about
        `STAGE_3_SEED_FILES` or `STAGE_3_SEED_ENTRIES` reads and is never listed whole. A port already in the stage directory (a resumed stage)
        is kept. The ledger holds the sha256 of each file as copied, so a later reader can tell what the coder changed."""
        if spec.number == full_port_quality.STAGE:
            source = full_port_quality_stage.source_stage(self)       # stage 6 measures stage 3's port, or stage 2's; stage 4's entries pass `source`
        try:
            stage_parts = Path(stage_dir).relative_to(self.run_dir).parts
        except ValueError:
            stage_parts = ()
        if not stage_parts:
            self.ledger.append("decision", spec.number, decision=f"stage {source} port not copied", problem="the stage directory is not inside the run directory")
            self._port_seed = {"copied": 0, "kept": False}
            return
        src_parts, dst_parts = ("stages", str(source), "port"), (*stage_parts, "port")
        copied, problem = [], None
        # Both sides are anchored beneath one descriptor of the run directory. The earlier version pinned only the last name of each path, so
        # a coder that replaced `stages/2` or `stages/3` with a link made the copy read from, or write to, a directory outside the run
        # (review of PR 455, round 4). `open_beneath` refuses a link at every name, and the destination is written and removed only
        # through descriptors.
        with contextlib.ExitStack() as stack:
            try:
                run = stack.enter_context(safeio.pinned_root(self.run_dir))
            except (OSError, safeio.SafeReadError) as exc:
                self.ledger.append("decision", spec.number, decision=f"stage {source} port not copied",
                                   problem=f"the run directory cannot be opened: {exc.__class__.__name__}")
                self._port_seed = {"copied": 0, "kept": False}
                return
            if _stage_3_port_is_held(self, run, dst_parts, spec):
                return
            scan = _SeedScan(self.run_dir / Path(*src_parts), anchor=run, parts=src_parts)
            try:
                _clear_port_link(run, stage_parts)            # a link planted where the port directory goes is removed and never followed
                # Every step is anchored: the source is read through the descriptors of the scan, the destination is written beneath the run
                # directory's descriptor (`write_beneath` makes each directory itself and refuses a link at any name), so a swap between the
                # listing and the read cannot bring in a file from outside stage 2's port or send one out of stages/3/port/. The hash is that
                # of the bytes read, which are the bytes written.
                with contextlib.closing(iter(scan)) as entries:
                    for parts, kind, identity in entries:
                        rel = Path(*parts)
                        if rel.suffix == ".pyc":
                            continue
                        if kind != "file":
                            problem = f"{rel} is not a regular file"
                            break
                        if len(copied) >= STAGE_3_SEED_FILES:
                            problem = f"stages/{source}/port/ holds more than {STAGE_3_SEED_FILES} files (the scan stopped after reading {scan.seen} entries)"
                            break
                        data = scan.read(parts, identity, STAGE_3_SEED_FILE_BYTES)
                        safeio.write_beneath(run.fd, (*dst_parts, *parts), data, mode=0o644)
                        copied.append({"path": "/".join((*dst_parts, rel.as_posix())), "sha256": hashlib.sha256(data).hexdigest()})
            except _SeedLimit as exc:
                problem = f"stages/{source}/port/ {exc} (the scan stopped after reading {scan.seen} entries)"
            except (OSError, safeio.SafeReadError) as exc:
                # The class and the reason only: an OSError's own text carries the absolute path of the file, which does not belong in the ledger.
                problem = f"{exc.__class__.__name__}: {getattr(exc, 'reason', None) or getattr(exc, 'strerror', None) or 'error'}"[:200]
            if problem or not copied:
                _clear_port_link(run, stage_parts, remove_tree=True)       # no half a port: the coder starts from stage 2's files, or from nothing
        problem = problem[:300] if problem else problem
        copied.sort(key=lambda f: f["path"])                  # the walk has no order of its own; the record has one
        if problem or not copied:
            self.ledger.append("decision", spec.number, decision=f"stage {source} port not copied", problem=problem or f"stages/{source}/port/ holds no file",
                               **({"source_stage": source} if source != 2 else {}))
            self._port_seed = {"copied": 0, "kept": False}
            return
        self.ledger.append("decision", spec.number, decision=f"stage {source} port copied", files=copied, **({"source_stage": source} if source != 2 else {}))
        self._port_seed = {"copied": len(copied), "kept": False}

    def _pin_full_port(self, spec, stage_dir: Path, *, command: bool = False, entry: int | None = None) -> list[str]:
        """Stage 2 of a full port (orchard/full_port.py): put the supervisor's template and its config in the stage directory, replacing any
        copy there. Called before the prepare step (so the coder can read the template) and again after it (so an edit made during prepare
        is undone). `entry` is stage 4's chip count when `stage_dir` is that entry's directory (issue 454). With `command` set, hw_test.json is also rewritten to the template's fixed command; the coder's deadline is kept when
        it is a positive number no larger than the default. Returns the problems that stop the stage (no model snapshot in the run)."""
        snapshot = self.inputs.get("model")
        if not snapshot:
            return ["the run has no model snapshot (inputs `model=`), so the port check has nothing to load"]
        shipped = Path(self.paths.orchard_dir) / "orchard" / "skills" / "full-port-templates" / full_port.TEMPLATE_NAME
        safeio.atomic_write(stage_dir / full_port.TEMPLATE_NAME, safeio.read_bytes(shipped, max_bytes=RUN_JSON_MAX), mode=os.stat(shipped).st_mode & 0o777)   # replaces what stands there
        config = stage_dir / full_port.CONFIG_NAME
        _put(config, json.dumps({"model_snapshot": str(Path(snapshot).absolute())}, indent=2) + "\n")
        if command:
            deadline = full_port.DEADLINE_S
            given = self._run_json(stage_dir / "hw_test.json", "hw_test.json", spec.number).value
            given = given.get("deadline_s") if isinstance(given, dict) else None
            if isinstance(given, (int, float)) and not isinstance(given, bool) and 0 < given <= full_port.DEADLINE_S:
                deadline = given
            _put(stage_dir / "hw_test.json", json.dumps({"command": full_port.command(spec.number, entry), "deadline_s": deadline}) + "\n")
        else:
            self.ledger.append("decision", spec.number, decision="port check template copied", copied=[full_port.TEMPLATE_NAME, full_port.CONFIG_NAME])
        return []

    def _pin_scoring(self, spec, stage_dir: Path, *, command: bool = False) -> list[str]:
        """Stage 2 of a scoring model (orchard/scoring_check.py): put the supervisor's template and its config (the snapshot, the run's kind and
        the bounds in force) in the stage directory, replacing any copy there. Called before the prepare step and again after it, so an edit
        made while the coder worked is undone. With `command` set, hw_test.json is rewritten to the template's fixed command (the coder's
        deadline is kept when it is a positive number no larger than the default). The ledger entry holds the sha256 of each file as written.
        Returns the problems that stop the stage (no model snapshot, or a kind the check does not run)."""
        snapshot = self.inputs.get("model")
        if not snapshot:
            return ["the run has no model snapshot (inputs `model=`), so the scoring check has nothing to load"]
        kind = run_model_kind(self.ledger.read())
        if kind not in scoring_check.KINDS:
            return [f"the run's model kind is {kind!r}; the scoring check runs {scoring_check.KINDS}"]
        shipped = scoring_check.TEMPLATE
        safeio.atomic_write(stage_dir / scoring_check.TEMPLATE_NAME, safeio.read_bytes(shipped, max_bytes=RUN_JSON_MAX), mode=os.stat(shipped).st_mode & 0o777)
        config = stage_dir / scoring_check.CONFIG_NAME
        _put(config, json.dumps(scoring_check.draft_config(str(Path(snapshot).absolute()), kind), indent=2) + "\n")
        if command:
            deadline = scoring_check.DEADLINE_S
            given = self._run_json(stage_dir / "hw_test.json", "hw_test.json", spec.number).value
            given = given.get("deadline_s") if isinstance(given, dict) else None
            if isinstance(given, (int, float)) and not isinstance(given, bool) and 0 < given <= scoring_check.DEADLINE_S:
                deadline = given
            _put(stage_dir / "hw_test.json", json.dumps({"command": scoring_check.COMMAND, "deadline_s": deadline}) + "\n")
        else:
            self.ledger.append("decision", spec.number, decision="scoring check template copied", kind=kind,
                               template=evidence_record(self.run_dir, stage_dir / scoring_check.TEMPLATE_NAME),
                               config=evidence_record(self.run_dir, config))
        return []

    def _draft_stock(self, spec, stage_dir: Path) -> None:
        """stock_config.json for stage 2 of a stock-runtime run (orchard/stock_runtime.py, draft_config), and the template that reads it.
        A config already in the stage directory (a resumed stage) is kept."""
        target = stage_dir / "stock_config.json"
        if target.exists():
            return
        delta = self._run_delta()
        cfg, problems = stock_runtime.draft_config(run_dir=self.run_dir, delta=delta, inputs=self.inputs,
                                                   chips=spec.boards * CHIPS_PER_BOARD, cache_root=self.paths.cache_root)
        if cfg is None:
            self.ledger.append("decision", spec.number, decision="stock config not drafted", problems=problems)
            return
        _put(target, json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
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
            cfg = safeio.read_json(stage_dir / "swap_config.json", max_bytes=MAX_SWAP_CONFIG_BYTES)
        except safeio.SafeReadError:
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
            _put((cdir / "swap_config.json"), json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            for name in (*STAGE4_TEMPLATES, *(AGREEMENT_TEMPLATES if cfg.get("agreement_set") else ())):
                shutil.copy2(templates / name, cdir / name)
        if tests is not None:
            _put((stage_dir / "hw_tests.json"), json.dumps(tests, indent=2) + "\n", encoding="utf-8")
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
        have = self._run_json(path, "hw_tests.json", spec.number).value
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
        _put(path, json.dumps(want, indent=2) + "\n", encoding="utf-8")

    def _keep_drafted_caches(self, spec, stage_dir: Path) -> None:
        """The tensor cache of a configuration the supervisor drafted is the drafted one. On the first run
        with drafts the agent rewrote the configs with its own spelling of the cache names, so no later run
        would have found them. The cache is read only when the test runs, so setting it back after the
        prepare step changes nothing the agent built."""
        drafted = next((e["data"].get("caches") or {} for e in reversed(self.ledger.read())
                        if e["stage"] == spec.number and e["data"].get("decision") == "stage 4 configs drafted"), {})
        for n, cache in drafted.items():
            path = stage_dir / "configs" / str(n) / "swap_config.json"
            cfg = self._run_json(path, f"configs/{n}/swap_config.json", spec.number).value
            if not isinstance(cfg, dict) or cfg.get("tt_cache") == cache:
                continue
            self.ledger.append("decision", spec.number, decision="kept the drafted tensor cache", config=int(n),
                               agent=cfg.get("tt_cache"), drafted=cache)
            cfg["tt_cache"] = cache
            _put(path, json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    def _read_plan(self, spec, stage_dir: Path):
        tests, problems = read_plan(stage_dir, required=self.required_chips, max_chips=spec.boards * CHIPS_PER_BOARD,
                                    budget_s=spec.budget_s, home=self.home)
        return tests, problems + swap_draft.stage4_nearest_problems(self.run_dir, stage_dir, tt_model_root=self.paths.tt_model_root,
                                                                       operator_home=self.paths.operator_home, counts=[t.chips for t in tests])

    def _finished_check(self, spec, phase: str, stage_dir: Path):
        """What says a step's files are written and valid (AgentStep's `finished`): for one hardware
        test's prepare step, hw_test.json and handoff.json as the hardware phase reads them; for a run or
        finish step, the stage's gate. A list of tests (stage 4) and stage 7 have no such check. A repair round (issue 420) is finished when
        the port directory differs from what it was when the step started."""
        if phase == "repair":
            return lambda: (self._repair_baseline is not None
                            and bool(port_repair.changed(self._repair_baseline, port_repair.fingerprint(stage_dir / port_repair.PORT_DIR))))
        if phase == "prepare":
            if spec.tests:
                return lambda: not self._read_plan(spec, stage_dir)[1]
            return lambda: not self._read_test(stage_dir)[1]
        return lambda: self._run_gate(spec, stage_dir).ok

    def _run_gate(self, spec, stage_dir: Path) -> GateResult:
        """The stage's gate, as a result that is always returned. A gate reads files an agent can write, and an exception from it (a value
        no one thought of: an integer too large for a float, a type nobody checked) must not escape the supervisor, so it becomes a failed
        gate. The reason and the one ledger notice name the exception's class and never its message, because the message can carry the
        file's own text. An interrupt (KeyboardInterrupt, SystemExit) is not an `Exception` and goes through. The gate is chosen outside the
        try: the choice reads the ledger, and a ledger that cannot be read is a supervisor failure, not a gate refusal (issue 433, review of
        PR 444, limit round)."""
        gate = self._gate(spec)
        try:
            return gate(stage_dir, self.run_dir)
        except Exception as exc:
            name = type(exc).__name__
            self._note("notice", what="a gate raised", for_stage=spec.number, error_class=name)      # `stage` is the ledger's own argument
            return GateResult(False, (f"the gate of stage {spec.number} raised {name} while reading the stage's files, so the stage cannot "
                                      "pass: a value in a file it reads is one the gate does not handle",), ())

    # ---- the hardware test ----------------------------------------------------------------------

    def _read_test(self, stage_dir: Path) -> tuple[dict | None, list[str]]:
        problems = []
        got = {name: safeio.read_json_checked(stage_dir / name, max_bytes=RUN_JSON_MAX) for name in ("hw_test.json", "handoff.json")}
        bad = [g.problem(name) or f"{name} is missing" for name, g in got.items() if not g.ok]
        if bad:
            return None, [f"hw_test.json and handoff.json must exist and be JSON: {'; '.join(bad)}"]
        test, note = got["hw_test.json"].value, got["handoff.json"].value
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
        delta = self._run_delta()
        if delta is None:
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
            _put((trial_dir / "trial_config.json"), 
                json.dumps({"stage_dir": str(stage_dir), "orchard_dir": str(self.paths.orchard_dir)}, indent=2) + "\n", encoding="utf-8")
            self.ledger.append("decision", n, decision="graft trial started", candidate=cand,
                               note="serving the nearest model's mtp.* head grafted onto this model, beside the safe configuration")
            test = {"command": f"python3 stages/{n}/{graft.TRIAL_DIR}/{GRAFT_TRIAL_TEMPLATE}", "deadline_s": GRAFT_TRIAL_DEADLINE_S}
            self._hardware_phase(spec, stage_dir, test, record=f"{graft.TRIAL_DIR}/test-result.json",
                                 output=f"{graft.TRIAL_DIR}/evidence/hw-test-output.txt", what="graft trial test",
                                 env_extra={GRAFT_TRIAL_ENV: str(trial_dir.resolve())},
                                 after_test=lambda: self._stop_trial_servers(n, trial_dir, "after the trial test"))
        record = self._run_json(marker, f"{graft.TRIAL_DIR}/test-result.json", n).value
        record = record if isinstance(record, dict) else {}
        trial = None
        if record.get("returncode") == 0 and not record.get("timed_out"):
            trial = self._run_json(trial_dir / graft.TRIAL_RESULT, f"{graft.TRIAL_DIR}/{graft.TRIAL_RESULT}", n).value
        result_file = self._run_json(stage_dir / "result.json", "result.json", n).value
        safe_top1 = result_file.get("top1_agreement") if isinstance(result_file, dict) else None
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
                self._set_stale_trial_outputs_aside(stage_dir)
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

    @staticmethod
    def _set_stale_trial_outputs_aside(stage_dir: Path) -> None:
        """A skipped decision says no variant was served. A trial killed after its script wrote `trial-result.json` and before the supervisor wrote
        `test-result.json` leaves outputs that the stage 2 gate would read as a variant and then refuse the decision for (issue 194). Move them into
        `sampling-trial/stale/` (or `stale-1`, `stale-2` when an earlier call already made one), which no reader looks at, so the evidence is kept. A trial whose test record exists is not touched."""
        trial_dir = stage_dir / sampling.TRIAL_DIR
        if (trial_dir / "test-result.json").exists():
            return
        leftovers = [trial_dir / sampling.TRIAL_RESULT, trial_dir / "variant"]
        leftovers = [x for x in leftovers if x.exists()]
        if not leftovers:
            return
        # Never remove an archive that an earlier, interrupted call made: it can hold the only copy of the trial result (issue 196). Each call
        # moves into an unused name (stale, stale-1, stale-2, ...), and `mkdir` without `exist_ok` raises when the name is taken.
        k, stale = 0, trial_dir / "stale"
        while True:
            try:
                stale.mkdir()
                break
            except FileExistsError:
                k += 1
                stale = trial_dir / f"stale-{k}"
        for x in leftovers:
            os.replace(x, stale / x.name)

    def _cache_check_path(self, stage_dir: Path) -> Path:
        """The path whose filesystem the sampling trial's new tensor cache lands on: the nearest existing ancestor of `<tt_cache>-nodraft`, the sibling
        the trial writes (issue 126). The sibling is made beside the configured path, so when that path is a link or a mount point the sibling is on
        the filesystem of the link's parent, not of its target. A relative `tt_cache` is taken against the run directory, before the suffix is added.
        The run directory when the config names none."""
        try:
            cfg = safeio.read_json(stage_dir / "swap_config.json", max_bytes=RUN_JSON_MAX, expect=dict)
            value = cfg["tt_cache"]
        except (safeio.SafeReadError, KeyError):
            return self.run_dir
        if not isinstance(value, str) or not value:
            return self.run_dir
        cache = Path(value)
        if not cache.is_absolute():
            cache = self.run_dir / cache
        try:
            cache = cache.with_name(cache.name + sampling.CACHE_SUFFIX)
        except ValueError:                                  # a path with no final name ("" or "/"): nothing to add the suffix to
            return self.run_dir
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
            _put((trial_dir / "trial_config.json"), 
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
        record = self._run_json(marker, f"{sampling.TRIAL_DIR}/test-result.json", n).value
        record = record if isinstance(record, dict) else {}
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
        status: dict = {}
        records = kernelprobe.sweep_strays(trial_dir, var=GRAFT_TRIAL_ENV, identity=True, status=status)
        if records:
            self.ledger.append("decision", n, decision="graft trial servers stopped", when=when, servers=records)
        left = [r["pid"] for r in records if not r.get("stopped")]
        if left:
            self._block(n, "a server of the graft trial is still running (pid " + ", ".join(str(p) for p in left) + "), so the coder is "
                           "not restored over it; stop it and run the command again", pids=left)
        if status.get("problems"):
            self._block(n, "the sweep of the graft trial's servers could not be completed (" + "; ".join(status["problems"]) + "), so the "
                           "coder is not restored until someone has checked that no server of the trial still runs", problems=status["problems"])

    def _record_graft_decision(self, stage_dir: Path, decision: dict) -> None:
        """The ledger's copy of the decision, with the decision file's sha256 so a later stage can tell the file was not changed."""
        path = stage_dir / graft.DECISION_FILE
        self.ledger.append("decision", 2, decision="graft decision", choice=decision["choice"], reason=decision["reason"],
                           numbers=decision["numbers"], decision_file=evidence_record(self.run_dir, path))

    def _phase_boards(self, spec, stage_dir: Path) -> int:
        """Boards the stage's single test holds: the stage's own, or more when the drafted swap config names a container package that
        needs more chips. Catalog container packages mostly report 2 boards (4 chips), so stage 2 serves them on that many, as
        stage 4 does for its 4-chip test (issue 60). Stage 3 of a full port holds the run's configured chips (issue 326)."""
        if spec.skill in ("full-port-mesh", "full-port-quality"):
            chips = self._stage_3_chips()
            return max(spec.boards, -(-chips // CHIPS_PER_BOARD)) if chips else spec.boards
        if spec.skill == full_port_serve.SKILL:          # stage 5 serves the port on the chips the stage it came from used (issue 329)
            return max(spec.boards, -(-self._serve_chips(full_port_serve.source_stage(self.ledger.read())) // CHIPS_PER_BOARD))
        try:
            cfg = safeio.read_json(stage_dir / "swap_config.json", max_bytes=RUN_JSON_MAX, expect=dict)
            chips = int(cfg["chips"]) if cfg.get("package") else 0
        except (safeio.SafeReadError, KeyError, TypeError, ValueError):
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
            what = "the port's configured chip count" if spec.skill in ("full-port-mesh", full_port_serve.SKILL, "full-port-quality") else "the container package"
            self._block(n, f"{what} needs {boards * CHIPS_PER_BOARD} chips and this box has {have * CHIPS_PER_BOARD}",
                        need_chips=boards * CHIPS_PER_BOARD)
        # After an abort the coder's lease is released, and `--resume-aborted` can reach a stage that needs no coder (stage 5 of a full port) with
        # none: there is nothing to park, so every board comes from a lease of its own, which waits while the boards are busy (review of PR 490).
        d = decide_park(self.coder_lease.units if self.coder_lease is not None else (), chips, boards)
        self.ledger.append("decision", n, decision="hardware phase", action=d.action, chips=boards * CHIPS_PER_BOARD,
                           free_boards=list(d.free_boards), server_boards=list(d.server_boards))
        shape = SimpleNamespace(chips=boards * CHIPS_PER_BOARD, boards=boards)
        first_chip = {c.board: c.bdf for c in reversed(chips)}
        if d.action == "use_free" or self.coder_lease is None:
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
        # The width this stage's test needs: stage 3 of a full port runs the port on the run's configured chip count (issue 326), which is
        # more than the stage's own board. The lab path used to lease and run `spec.boards` boards for every stage, so stage 3 got two
        # chips while its gate expects four and a working port failed every attempt (review of PR 455). Other stages keep their own width.
        boards = self._phase_boards(spec, stage_dir) if spec.skill in ("full-port-mesh", full_port_serve.SKILL, "full-port-quality") else spec.boards
        chips = boards * CHIPS_PER_BOARD
        lab_chips = getattr(self.lab, "chips", None)
        if spec.skill in ("full-port-mesh", full_port_serve.SKILL, "full-port-quality") and lab_chips and chips > lab_chips:
            # A lease for more chips than the lab has would wait until the stage budget ran out: refuse it, as a stage 4 configuration is.
            host = self.lab.info.get("host")
            self._block(n, f"the port's configured chip count needs {chips} chips and the lab {host} has {lab_chips} chips",
                        need_chips=chips, lab_chips=lab_chips)
        self.ledger.append("decision", n, decision="hardware phase", action="lab", chips=chips, where=self._where())
        self._lab_files_up(n)
        lease = reacquire(self.test_adapter, chips=chips, who=WHO,
                          reason=f"stage {n} hardware test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", test_lease=lease.record(),
                           where=self._where())
        try:
            self._run_test(spec, stage_dir, test, lease, boards=boards)
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
            held = self._helpers_holding(lease_id)
            if held:
                # The helpers the ledger remembers are checked before any lease goes back (PR 431, round 4): a helper that still runs may have
                # a chip of this lease open.
                self.ledger.append("notice", None, what="the test lease was kept: a helper of the hardware test is still running",
                                   lease_id=lease_id, helpers=held)
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
            cache = safeio.read_json(stage_dir / "swap_config.json", max_bytes=MAX_SWAP_CONFIG_BYTES, expect=dict).get("tt_cache")
        except safeio.SafeReadError:
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
        known = safeio.read_json_checked(path, max_bytes=RUN_JSON_MAX, expect=dict).value
        known = known if isinstance(known, dict) else {}
        known.update(audit)
        _put(path, json.dumps(known, indent=2, sort_keys=True))

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
        self._record_mask_inputs(n, stage_dir)
        marker = stage_dir / record
        tmp = marker.with_name(marker.name + ".tmp")
        _put(tmp, json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what=what, returncode=result["returncode"],
                           timed_out=result["timed_out"], **self._moved_aside_record(marker))
        # The raw files the gates recompute a number from (orchard/raw_evidence.py) are hashed now, before the agent's finish step can write in
        # the stage directory (issue 445). A file the test did not write, or that cannot be read within the cap, gets no entry.
        for rec in raw_evidence.record_after_test(self.run_dir, stage_dir, n):
            self.ledger.append("evidence", n, what=raw_evidence.RAW_EVIDENCE_WHAT, **rec)
        return result

    def _record_mask_inputs(self, n: int, stage_dir: Path) -> None:
        """When a hardware test ends, write the sha256 of the files each `model-dir` under the stage directory holds that skip masks are rebuilt
        from (stages.mask_input_entries). The gate rebuilds a mask only from files that still have these digests (PR 422, round 3), so this has
        to happen before any agent step runs again: the hardware test is the last thing that ran the template, and nothing else runs between it
        and this call."""
        for entry in mask_input_entries(self.run_dir, stage_dir):
            self.ledger.append("decision", n, decision=MASK_INPUTS_DECISION, **entry)

    def _note_unverified(self, n: int, gate) -> None:
        """A passing gate that accepted numbers nothing recomputes (`GateResult.unverified`) leaves one ledger decision that lists them,
        for the operator bundle's thin-evidence section (issue 445). A gate with none leaves nothing."""
        if gate.ok and gate.unverified:
            self.ledger.append("decision", n, decision=UNVERIFIED_LEDGER_DECISION, numbers=list(gate.unverified))

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
        held_leases = list(record.get("lease_ids") or ([record["lease_id"]] if record.get("lease_id") else []))   # the leases this test runs under
        with _open_fresh(out_path) as out, collect_survivors() as survivors:
            try:
                try:
                    if self.lab is not None:
                        code, timed_out = self._spawn_on_lab(n, command, env, deadline, out)
                    else:
                        code, timed_out = spawn_checked(command, self.run_dir, env, deadline, out)
                except Denied as exc:
                    code, timed_out = None, False
                    out.write(f"refused: {exc}\n".encode("utf-8"))
            except BaseException:
                # An interrupt or a crash: spawn_checked swept the helpers on its way out and let the exception go on, so what the sweep
                # stopped (and what it could not stop) is in `survivors` and nowhere else. It is written down and remembered here; the
                # exception that is already on its way is the one that propagates.
                self._note_test_helpers_on_exit(n, survivors, held_leases)
                raise
        self._record_test_helpers(n, survivors, held_leases)
        if (timed_out or code not in (0, None)) and self.lab is None:
            # A test that ran and did not succeed is checked against the bus before anyone calls it a model failure. A refused
            # command (code None and no timeout) never reached a chip.
            how = "timed out" if timed_out else f"exited {code}"
            self._block_if_off_bus(n, f"the hardware test {how}", chips)
        return {"command": command, "returncode": code, "timed_out": timed_out,
                "seconds": round(self.clock() - t0, 3), "chips": list(chips),
                "output": self._moved_aside_record(out_path)}

    def _record_test_helpers(self, n: int, survivors: list[dict], lease_ids=()) -> None:
        """The helpers a hardware test started in sessions of their own, which spawn_checked's sweep stopped after the test ended or was killed:
        a server that run.sh started, for example. The sweep removes them before any later scan, so this is the only record of them: one
        ledger entry with the identity each had (pid, boot id, start time) before it was signalled, and the ids of the leases the test ran
        under (`lease_ids`). A helper that is still running afterwards (it replaced its environment, so the sweep cannot name it again) blocks the
        run the way a server of the graft trial does, so the coder is not restored over a process that holds a chip, and its identity is
        remembered (`surviving_helpers`, and the ledger) so that no release of one of those leases and no restoration of the coder goes ahead
        until that exact process is gone. A sweep that could not be completed (`survivors.problems`: a process list it could not read, a scan
        that never settled, an error) blocks as well: an empty record from such a sweep is not proof that nothing is left.
        The identities are put in memory before the first ledger write, so a ledger that fails cannot leave a running helper unguarded."""
        problems = list(getattr(survivors, "problems", []) or [])
        if not survivors and not problems:
            return
        lease_ids = [str(i) for i in lease_ids or ()]
        left = [r for r in survivors if not r.get("stopped")]
        if left:
            self.surviving_helpers = [dict(r) for r in left]
            self.surviving_helper_leases = list(lease_ids)
        self.ledger.append("decision", n, decision="hardware test helpers stopped", servers=list(survivors), problems=problems,
                           lease_ids=lease_ids)
        if left:
            self._remember_helpers(n, left, problems, lease_ids)
            self._block(n, "a helper of the hardware test is still running (pid " + ", ".join(str(r["pid"]) for r in left) + "), so the coder is "
                           "not restored over it; stop it and run the command again", pids=[r["pid"] for r in left])
        if problems:
            self._block(n, "the sweep of the hardware test's helpers could not be completed (" + "; ".join(problems) + "), so the coder is not "
                           "restored until someone has checked that nothing of the test still runs; then run the command again", problems=problems)

    def _note_test_helpers_on_exit(self, n: int, survivors, lease_ids=()) -> None:
        """What `_record_test_helpers` records, for the way out of a hardware test that an interrupt or an error ended. It never raises an
        Exception and never blocks: the exception already on its way is the one that propagates. The identities of the helpers that are
        still running are remembered, so the lease release and the coder's restoration that follow check them."""
        try:
            problems = list(getattr(survivors, "problems", []) or [])
            if not survivors and not problems:
                return
            lease_ids = [str(i) for i in lease_ids or ()]
            left = [dict(r) for r in survivors if not r.get("stopped")]
            self.surviving_helpers = left                 # in memory first: the ledger may be what failed
            self.surviving_helper_leases = list(lease_ids) if left else []
            self.ledger.append("decision", n, decision="hardware test helpers stopped", servers=[dict(r) for r in survivors],
                               problems=problems, interrupted=True, lease_ids=lease_ids)
            if left:
                self.ledger.append("decision", n, decision="hardware test helpers survived", servers=left, problems=problems,
                                   interrupted=True, lease_ids=lease_ids)
        except Exception:
            pass

    def _remember_helpers(self, n, left: list[dict], problems: list[str], lease_ids=()) -> None:
        self.surviving_helpers = [dict(r) for r in left]
        self.surviving_helper_leases = [str(i) for i in lease_ids or ()]
        self.ledger.append("decision", n, decision="hardware test helpers survived", servers=[dict(r) for r in left], problems=list(problems),
                           lease_ids=list(self.surviving_helper_leases))

    def _ledger_helpers(self) -> tuple[list[dict], list[str]]:
        """The helpers an earlier process of this run recorded as running, and the ids of the leases they ran under, from the ledger: the live
        records of every "hardware test helpers stopped" entry (a record with `stopped: false`, written before the sweep's second entry, which a
        crash or a failed write may have prevented), replaced by the list of a later "survived" entry, and all of it ended by a "cleared" entry.
        A restart keeps the protection of a lease whose helper was known to run."""
        records: list[dict] = []
        leases: list[str] = []

        def ids(data) -> list[str]:
            return [str(i) for i in (data.get("lease_ids") or []) if isinstance(i, (str, int))]
        for e in self.ledger.read():
            if e["event"] != "decision":
                continue
            data = e["data"]
            what = data.get("decision")
            if what == "hardware test helpers stopped":
                live = [r for r in (data.get("servers") or []) if isinstance(r, dict) and r.get("stopped") is False]
                for r in live:
                    if not any(r.get("pid") == k.get("pid") and r.get("start") == k.get("start") for k in records):
                        records.append(r)
                if live:
                    leases = sorted(set(leases) | set(ids(data)))
            elif what == "hardware test helpers survived":
                records = [r for r in (data.get("servers") or []) if isinstance(r, dict)]
                leases = ids(data)
            elif what == "hardware test helpers cleared":
                records, leases = [], []
        return records, leases

    def _helpers_running(self) -> list[dict]:
        """The remembered helpers (in memory, else from the ledger) that still run, judged by the identity each had before it was signalled.
        Helpers that are gone are cleared, in memory and in the ledger."""
        known = self.surviving_helpers
        if not known:
            known, leases = self._ledger_helpers()
            self.surviving_helper_leases = leases
        if not known:
            return []
        alive = [r for r in known if kernelprobe.still_running(r)]
        if alive:
            self.surviving_helpers = alive
            return alive
        self.surviving_helpers = []
        self.surviving_helper_leases = []
        self._note("decision", decision="hardware test helpers cleared", servers=list(known))
        return []

    def _helpers_holding(self, lease_id: str) -> list[dict]:
        """The remembered helpers that still run and ran under the lease `lease_id`. A helper whose lease is not recorded (an older ledger)
        counts for every lease."""
        alive = self._helpers_running()
        leases = self.surviving_helper_leases
        return alive if alive and (not leases or str(lease_id) in leases) else []

    def _memory_helpers_hold(self, lease_id: str) -> bool:
        """`_helpers_holding` from memory alone, for the release that is used when the ledger cannot be read or written."""
        if not any(kernelprobe.still_running(r) for r in self.surviving_helpers):
            return False
        leases = self.surviving_helper_leases
        return not leases or str(lease_id) in leases

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
        if resumed and tests is None and (why := plan_refusal(stage_dir)):
            # A plan that is there and refused (NaN, a link out of the run, over the cap, not a list of tests) is not a missing plan. The
            # prepare step below replaces it, so the file name and the reason go in the ledger first (issue 386).
            self.ledger.append("decision", n, decision="hardware test plan refused, so the prepare step runs again", file=PLAN, reason=why)
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
            if extra is not None:
                # The further lease was reset when it was taken, and the park has since reset the coder's lease. The lease whose
                # chips have the lowest PCI address must be reset last (reset_order, issue 26).
                order, key = reset_order([lease, extra])
                if key != "list_order" and order[-1] is extra:
                    self._note_reset_order(n, test, key)
                    self._reset_test_chips(n, test, extra)
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
        timeouts on the chips the smaller tests had used, and passed once all four were reset. This costs about
        42 s per board, and a refused reset blocks. The order of two separate leases' resets matters (`reset_order`,
        issue 26): the lease whose chips have the lowest PCI address goes last. Whether stale state on the chips also
        plays a part is not isolated."""
        try:
            self.test_adapter.reset(lease)
        except AdapterError as exc:
            self._block(n, f"resetting the chips for the {test.chips}-chip test failed: {exc}",
                        lease_id=lease.lease_id, config=test.chips)
        self.ledger.append("decision", n, decision="test chips reset", config=test.chips,
                           lease_id=lease.lease_id)

    def _note_reset_order(self, n: int, test, key: str) -> None:
        """Record which key ordered the resets, so a run says whether it went by PCI address or fell back (issue 26)."""
        self.ledger.append("decision", n, decision="test chips reset order", config=test.chips, key=key)

    def _give_back_test_lease(self, n: int, lease: Lease) -> None:
        try:
            self.test_adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _run_listed_test(self, spec, stage_dir: Path, test, leases: list[Lease]) -> None:
        """Run one configuration's test (`_run_listed_test_body`). A run file the bounded reader refuses while the test's chips are held
        (an oversize output, a link) ends the test as a record that says so, never as an exception: the callers go on to the sweep, the lease
        release and the restore of the coder, and the gate reads the record."""
        try:
            try:
                self._run_listed_test_body(spec, stage_dir, test, leases)
            finally:
                # Stage 4 runs its configurations here, not through `_run_test`, so the mask-input digests are written here too (PR 422, round 5):
                # once, after the final attempt, before any agent step or the finish step can touch the files. Also after a refused run file.
                self._record_mask_inputs(spec.number, stage_dir)
        except safeio.SafeReadError as exc:
            n = spec.number
            reason = f"not completed: a run file was refused: {exc.reason}"
            path = write_test_record(stage_dir, test, {"config": test.chips, "command": test.command(n), "returncode": None,
                                                       "timed_out": False, "seconds": 0, "chips": [], "not_run": reason})
            self.ledger.append("evidence", n, what="hardware test", config=test.chips, cache=test.cache, returncode=None,
                               timed_out=False, not_run=reason, **self._moved_aside_record(path))

    def _run_listed_test_body(self, spec, stage_dir: Path, test, leases: list[Lease]) -> None:
        """Run one configuration's test on the first `test.chips` chips of `leases`, in device
        order, and write its record."""
        n = spec.number
        held = sorted((i, c) for lease in leases for c, i in zip(lease.chips, lease.dev_indices))
        chosen = held[:test.chips]
        out_path = test.output_path(stage_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        def spawn() -> dict:
            return self._spawn(n, test.command(n), [c for _, c in chosen], [i for i, _ in chosen],
                               min(test.deadline_s, spec.budget_s), out_path,
                               config=test.chips, cache=test.cache,
                               lease_ids=[lease.lease_id for lease in leases])
        result = spawn()
        if (test.boards >= 2 and result["returncode"] != 0 and not result["timed_out"]
                and result["seconds"] < FAST_FAIL_S):
            # A multi-board test that dies this early failed to start (the 4-chip test's `Fabric Router Sync:
            # Timeout`, seen after 13 to 57 s and gone after the chips were reset together). The cause is not
            # isolated, so reset every chip the test holds and run it once more. Its record is the second try's.
            # The second try writes output.txt again. The first try's text says why it failed, so it is kept beside it.
            first = out_path.with_name("output-first-try.txt")
            os.replace(out_path, first)
            self.ledger.append("decision", n, decision="test failed at startup; resetting the test chips and trying once more",
                               config=test.chips, returncode=result["returncode"], seconds=result["seconds"],
                               first_output=self._moved_aside_record(first))
            order, key = reset_order(leases)
            self._note_reset_order(n, test, key)
            for lease in order:
                self._reset_test_chips(n, test, lease)
            result = spawn()
        path = write_test_record(stage_dir, test, {"config": test.chips, **result,
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


def _port_repair_rounds_arg(text: str) -> int:
    """argparse type of `--port-repair-rounds`: a whole number from 0 to port_repair.MAX_ROUNDS."""
    try:
        return port_repair.check_rounds(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


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
    r.add_argument("--port-repair-rounds", type=_port_repair_rounds_arg, default=None, metavar="N",
                   help=f"a full-port stage 2 whose port check failed on the port itself (a missing weight, a bad import, an index off by one) "
                        f"gives the coder up to N repair rounds in the same attempt: it edits only stages/2/port/, and the supervisor runs the same "
                        f"check again (issue 420). 0 is the earlier behaviour, no rounds. Default {port_repair.DEFAULT_ROUNDS}, at most "
                        f"{port_repair.MAX_ROUNDS}. The ledger records the value and a resume keeps it; a resume that names another value is refused")
    r.add_argument("--no-tool-dedupe", dest="no_tool_dedupe", action="store_true",
                   help="send the agent every repeat read in full. By default a repeat read of a file that has not changed since the agent read it in the "
                        "same step, and a repeat of a read-only shell command that printed the same, is answered with a short note naming the turn "
                        "that holds the text (docs/tool-result-dedupe.md). The ledger records the choice when it is off; a resume keeps the recorded "
                        "choice, and a resume that turns it off for a run that started with it on is refused")
    r.add_argument("--telegraph-config", metavar="FILE", default=None,
                   help="a TOML file with the [telegraph] table (bringup passes the config it was started with). When it enables the helper, a "
                        "full-port stage 2 whose attempts keep failing asks it once for a draft port (orchard/phone_a_friend.py); its caps apply. "
                        "Without the option, or with the helper off, nothing is sent")
    r.add_argument("--no-repair-help", action="store_true",
                   help="do not ask the outside helper inside the port repair rounds (orchard/repair_help.py); it is still asked at the start of an "
                        "attempt. By default a configured helper is asked in a round whose check failed on the port and the coder has not moved it on")
    r.add_argument("--full-port", action="store_true",
                   help="let a model that stage 0 classed `full-port` go on: the local coder writes the port in stage 2 and one hardware "
                        "test compares the chip's tokens with the CPU reference (skill full-port-check). Stage 3 runs the port on the largest "
                        "of --required-chips and stage 4 on each other count. Stage 5 serves the port behind the chat server and replays an acceptance "
                        "corpus, and stage 6 measures quality and a benchmark. The ledger records "
                        "the flag and a resume keeps it. It can be added on a resume only before stage 2 has started. It changes nothing for "
                        "any other class, and `unknown` and `restricted-head` still stop")
    r.add_argument("--model-kind", choices=model_kind.KINDS, default=None,
                   help="what the model produces (issue 378): generative (text from tokens, the default for a model whose card and config say "
                        "nothing else), reranker (a yes/no score for a query and document) or embedding (a vector). Without this flag stage 0 reads "
                        "the model card's pipeline_tag (text-ranking is a reranker; feature-extraction and sentence-similarity are embedding "
                        "models), then the architecture names, and records which one decided. A reranker or embedding model of class weights-only, "
                        "or of class full-port with --full-port, takes the scoring route: stage 1 records Hugging Face scores or vectors and stage 2 "
                        "runs the supervisor's scoring check on the coder's port module (skill scoring-check). Stages 3 to 6 are not built for it, so "
                        "the run then pauses. The ledger's stage 0 entry holds the answer and a resume keeps it")
    r.add_argument("--full-port-rehearsal", action="store_true",
                   help="for a rehearsal model with random weights (scripts/tiny_qwen4): stage 2 of a full port waives its `coherent` field and "
                        "no other check, and the ledger says so. Needs --full-port, is given when the run starts, and is recorded in run_start; "
                        "a resume keeps it and cannot add it")
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
          disk_usage=shutil.disk_usage, home=None, containers=None, environ=None, lab=None, stack_pins=None) -> Supervisor:
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
    if progress.started:                        # a resumed run keeps the route it started on (the text runtime is read from the ledger)
        recorded_route = ((progress.run_start or {}).get("inputs") or {}).get(text_runtime.RUNTIME_INPUT)
        given_route = inputs.get(text_runtime.RUNTIME_INPUT)
        if given_route is not None and given_route != recorded_route:
            raise ValueError(f"this run started with text_runtime {recorded_route!r}; --input text_runtime={given_route} differs. Leave the input out to "
                             "resume with the recorded route, or move this run's directory aside and start a new run")
    full_port_req = bool(getattr(args, "full_port", False))
    full_port_added = False
    if progress.started and full_port_req and not full_port_enabled(entries):
        if any(e["event"] == "stage_start" and e["stage"] == 2 for e in entries):
            raise ValueError("--full-port was not given when this run started, and stage 2 has begun without it. The flag can be added on a "
                             "resume only before stage 2 starts. Move this run's directory aside and start a new run with the flag")
        full_port_added = True                  # recorded as a decision by the supervisor's first step
    if progress.started and full_port_enabled(entries):
        full_port_req = False                   # a resumed run keeps what the ledger holds; nothing to request again
    rehearsal_req = bool(getattr(args, "full_port_rehearsal", False))
    if rehearsal_req and not (full_port_enabled(entries) or getattr(args, "full_port", False)):
        raise ValueError("--full-port-rehearsal needs --full-port: the waiver applies only to the full-port stage 2")
    if rehearsal_req and progress.started and not full_port_rehearsal(entries):
        raise ValueError("--full-port-rehearsal waives a stage 2 check, so it can be given only when the run starts. This run started without "
                         "it. Move this run's directory aside and start a new run with the flag")
    if progress.started:
        rehearsal_req = False                   # a resumed run keeps what run_start holds
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
    repair_rounds = getattr(args, "port_repair_rounds", None)
    if progress.started:                        # a resumed run keeps the repair-round count it started with (a run that began before the option: the default)
        recorded_rounds = port_repair.check_rounds((progress.run_start or {}).get("port_repair_rounds", port_repair.DEFAULT_ROUNDS))
        if repair_rounds is not None and repair_rounds != recorded_rounds:
            raise ValueError(f"this run started with --port-repair-rounds {recorded_rounds}; --port-repair-rounds {repair_rounds} differs. Leave the option "
                             "out to resume with the recorded value, or move this run's directory aside and start a new run (the port repair rounds are "
                             "part of what the run's stage 2 attempts did)")
        repair_rounds = recorded_rounds
    repair_rounds = port_repair.DEFAULT_ROUNDS if repair_rounds is None else port_repair.check_rounds(repair_rounds)
    tool_dedupe = not getattr(args, "no_tool_dedupe", False)
    if progress.started:                        # a resumed run keeps the choice it started with (a run that began before the option: on)
        recorded_dedupe = (progress.run_start or {}).get("tool_dedupe", True) is not False
        if not tool_dedupe and recorded_dedupe:
            raise ValueError("this run started with the tool-result dedupe on; --no-tool-dedupe differs. Leave the option out to resume with the recorded "
                             "value, or move this run's directory aside and start a new run (the dedupe changes what the agent was sent)")
        tool_dedupe = recorded_dedupe
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
                      telegraph=_telegraph_from(getattr(args, "telegraph_config", None), environ=environ, home=home),
                      repair_help=False if getattr(args, "no_repair_help", False) else None,
                      coder=coder, coder_chips=args.coder_chips,
                      standin=ExternalStandIn(cpu["endpoint"], cpu["model"], http=http),
                      skills_dirs=[SKILLS_DIR, *args.skills_dir], inputs=inputs,
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required,
                      home=home, containers=containers, paths=paths, package=package,
                      package_added=added, unattended=getattr(args, "unattended", False), graft_trial=graft_on,
                      sampling=sampling_mode, port_repair_rounds=repair_rounds, tool_dedupe=tool_dedupe,
                      resume_aborted=bool(getattr(args, "resume_aborted", False)),
                      full_port=full_port_req, full_port_added=full_port_added, full_port_rehearsal=rehearsal_req, stack_pins=stack_pins,
                      kind_override=getattr(args, "model_kind", None),
                      lab=lab if lab_host else None,
                      four_chip_package=getattr(args, "four_chip_package", None), coder_tier=tier,
                      keep_up=bool(getattr(args, "coder_keep_up", False)),
                      coder_identity={"target": args.coder_target, "kind": args.coder_kind,
                                      "port": args.coder_port, "profile": args.coder_profile,
                                      "image_id": args.coder_image_id, "model": cfg.tiers[tier]["model"],
                                      "chips": args.coder_chips})


def _telegraph_from(path, *, environ=None, home=None):
    """The `Telegraph` the run's `--telegraph-config` names, or None when there is no option or the table leaves the helper off. A file that cannot
    be read, a table that is wrong, or a state directory that cannot be chosen refuses the start with the reason (the operator passed the option)."""
    if not path:
        return None
    from orchard import telegraph as tg
    from orchard import telegraph_cmd
    try:
        tcfg = telegraph_cmd.load_config(path)
        if not tcfg.enabled:
            return None
        directory = telegraph_cmd.resolve_directory(None, tcfg, os.environ if environ is None else environ, home or Path.home())
    except (tg.ConfigError, OSError, ValueError) as exc:
        raise ValueError(f"--telegraph-config {path}: {exc}") from exc
    return tg.Telegraph(tcfg, directory, environ=environ)


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


def _ledger_entries(sup) -> list:
    """The run's ledger entries, or [] when the supervisor has no readable ledger. Setting up the stack check never refuses a run."""
    try:
        entries = sup.ledger.read()
    except Exception:
        return []
    return entries if isinstance(entries, list) else []


def _run_start_inputs(entries: list) -> dict | None:
    """The inputs of the run's first `run_start` entry (the ones the agent works with), or None when there is none or they are not a mapping."""
    for e in entries:
        if isinstance(e, dict) and e.get("event") == "run_start":
            found = (e.get("data") or {}).get("inputs") if isinstance(e.get("data"), dict) else None
            return found if isinstance(found, dict) else None
    return None


def _stack_reference_python(sup, entries: list, inputs: dict, env) -> str | None:
    """The interpreter the probe asks: the one stage 1 uses. A run on the text route (the `text_runtime` input names a registered runtime) uses the
    interpreter stage 0 recorded, else `text_runtime.text_paths` (as `_text_context` does). Any other run uses the `reference_python` input, else
    $ORCHARD_REFERENCE_PYTHON (issue 389, item 1)."""
    name = inputs.get(text_runtime.RUNTIME_INPUT)
    if isinstance(name, str) and name in text_runtime.REGISTRY:
        sel = text_runtime.selection(entries)
        recorded = (sel or {}).get("paths") if isinstance((sel or {}).get("paths"), dict) else None
        if recorded and isinstance(recorded.get("python"), str):
            return recorded["python"]
        return text_runtime.text_paths(sup.paths.operator_home, env).python
    return inputs.get("reference_python") or env.get("ORCHARD_REFERENCE_PYTHON")


def stack_check_for(sup, args, environ=None):
    """The stack freshness comparison for a real run (orchard/stack_pins.py). Where each part lives: the tt-metal tree is $ORCHARD_TT_METAL_DIR
    or tt-metal under the operator's home (as preflight finds it); the TT vLLM plugin checkout is $ORCHARD_TT_VLLM_DIR (branch
    $ORCHARD_TT_VLLM_BRANCH, default main); the reference interpreter is the run's `reference_python` input, else $ORCHARD_REFERENCE_PYTHON; the checkpoint is the run's `model` input;
    the skills clones are the --skills-dir values. $ORCHARD_STACK_PINS_OFFLINE makes no network call. A part that is not named is UNKNOWN."""
    env = os.environ if environ is None else environ
    metal = env.get("ORCHARD_TT_METAL_DIR") or str(Path(sup.paths.operator_home) / "tt-metal")      # the run's operator home, as preflight reads it
    entries = _ledger_entries(sup)
    inputs = _run_start_inputs(entries) or sup.inputs          # a resumed run is judged on the inputs it started with (issue 389, item 2)
    sources = stack_pins_mod.Sources(
        model_dir=inputs.get("model"), ref_python=_stack_reference_python(sup, entries, inputs, env), tt_metal_dir=metal,
        vllm_dir=env.get("ORCHARD_TT_VLLM_DIR"), vllm_branch=env.get("ORCHARD_TT_VLLM_BRANCH") or "main",
        skills_dirs=tuple(str(d) for d in args.skills_dir), homes=(str(sup.paths.operator_home), str(sup.home)),
        exclude_paths=(str(sup.run_dir),))
    return stack_pins_mod.make_check(sources, probe_env=agent_env(sup.run_dir, extra=sup.extra_env), block=env.get(stack_pins_mod.BLOCK_ENV) == "1")


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
            if isinstance(sup, Supervisor):
                # A real Supervisor always has `paths`, `inputs` and `home`; a test may hand main a stand-in that has none. Setting up the stack
                # check must never refuse a run or change how a mid-run error is reported: without it the record says nothing was compared.
                try:
                    sup.stack_pins = stack_check_for(sup, args)
                except Exception:
                    sup.stack_pins = None
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
