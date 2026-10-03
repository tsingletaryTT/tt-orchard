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

The first start of the coder in a run is also asked a
known-answer question (7 times 6); a server that answers without 42 blocks the run, because the
canary alone would accept noise. When stage 0 finds that the model needs new model code
(a full port), the run pauses before stage 2 for the operator. When stage 0 finds that only the
weights differ (weights-only), stage 2 runs the weights-swap-check skill and its gate in place of
the functional decoder, and stage 3 (full model) is recorded as skipped. Stage 0's passing
stage_end records that path, and every later choice is
read from there (orchard/stages.py, run_path). An unknown path keeps the plan 4 table.

Operator commands go through a one-word control file in the run directory: pause, resume, abort.
SIGINT (Ctrl-C) and SIGTERM take the same path as abort: the running command is killed, the coder
is stopped, every lease this process holds is released, the ledger records the abort and the
process exits 4. Any other error that ends the run (an adapter error, a corrupt ledger, a bug) also
stops the coder and releases the leases before it is reported, with exit 3; that run is not
aborted, so running the same command again resumes it. When the ledger cannot be read, the
release works from what this process remembers. A SIGKILL, a power cut or a crash of Python
itself runs no handler: the coder then keeps its chips under a dead pid's lease until the run is
resumed (which re-leases it) or aborted.
A paused supervisor keeps its leases and waits. Abort stops the coder, releases its lease and
closes the ledger. The run ends at "ready for operator review". The supervisor never publishes.
Agents are kept from publishing in three ways, none of them complete: the command runner refuses
the common publish, push and upload spellings; agent shells get no tokens in their environment;
and a preflight refuses to start while known credential files exist in the operator's home,
unless the operator passes --accept-credentials-visible (the ledger records that). Agent shells
run as the same user, so code an agent runs can still read any file that user can read.

Plan 4 runs stages 0 to 6 and 8. Stage 7 (package and container build) is recorded as skipped,
and so is stage 3 on the weights-only path.

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
import functools
import hashlib
import json
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from orchard.adapters import AdapterError, Lease
from orchard.agent import AgentStep, Tools, agent_env, probe_model, spawn_checked
from orchard.canary import CanaryError, compare, post_json
from orchard.canary import ask as canary_ask
from orchard.commands import run_command
from orchard.context import build_messages, facts_from
from orchard.defaults import (AGENT_CONTINUATION_TURNS, CHIPS_PER_BOARD, CMD_TIMEOUT_S,
                              COLD_BOOT_BUDGET_S, CONTROL_POLL_S, FIRST_BOOT_EXPECTED,
                              FIRST_BOOT_PROMPT, RUN_CANARY_PROMPT, STOP_TIMEOUT_S, TEST_DISK_GB)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.hwtests import (failed_tests, load_plan, move_aside, pending, read_plan, suspect_caches,
                             unrecorded, write_plan, write_record, write_summary)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.paths import RunPaths, UnknownPlaceholder
from orchard.runner import Denied, check_string
from orchard.server import ServerControl, ServerError, ServerSpec, StopCheck
from orchard.stages import (GateResult, TierUnavailable, attempt_started_ts, budget_cap, check_disk,
                            coder_state, delta_path, evidence_record, open_stage_dir,
                            resolve_endpoint, resolve_skill, run_path, run_progress, spec_for,
                            tier_for)
from orchard.tiers import TierConfigError, load
from orchard.watchdog import (Event, IdenticalResponses, Ladder, NoFileWritten, NoNewEvidence,
                              RepeatedToolCall, RetryGuard, StageOverBudget, ThinkingWithoutAction,
                              TurnRepeat, Watchdog)

WHO = "orchard:supervisor"
AGENT = "stage-agent"                 # the launched agent's name; the ladder counts its rungs per stage
EXIT_READY, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED = 0, 2, 3, 4
FULL_PORT = ("stage 0 found a full port (new model code is needed); plan 4 runs weights-only "
             "bring-ups, so the operator decides whether to go on")
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


def operator_home() -> Path:
    """The home OpenSSH uses: the passwd entry's, which ignores HOME."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def boot_unfinished(entries: list[dict]) -> bool:
    """The last coder start has a "coder starting" entry and no "coder started" after it."""
    last = None
    for e in entries:
        if e["event"] == "decision" and e["data"].get("decision") in ("coder starting", "coder started"):
            last = e["data"]["decision"]
    return last == "coder starting"


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


def coder_tier(cfg, port: int) -> str:
    """The chip tier whose endpoint uses the coder's port."""
    names = [n for n, t in cfg.tiers.items()
             if t["placement"] == "chips" and urlparse(t["endpoint"]).port == port]
    if len(names) != 1:
        raise ValueError(f"exactly one chip tier must use port {port}; found {names}")
    return names[0]


# ---- the supervisor -----------------------------------------------------------------------------

class Supervisor:
    def __init__(self, *, run_dir, ledger, cfg, model_id: str, adapter, coder, coder_chips: int,
                 standin, skills_dirs, inputs: dict | None = None, extra_env: dict | None = None,
                 versions: dict | None = None, http=post_json, probe=probe_model, clock=time.time,
                 sleep=time.sleep, budgets: Budgets = Budgets(), disk_usage=shutil.disk_usage,
                 credentials_visible: list[str] | None = None,
                 required_chips: tuple[int, ...] | None = None, home=None, containers=None,
                 paths: RunPaths | None = None):
        self.run_dir = Path(run_dir).resolve()
        self.ledger, self.cfg, self.model_id = ledger, cfg, model_id
        self.adapter, self.coder, self.coder_chips, self.standin = adapter, coder, coder_chips, standin
        self.skills_dirs = [Path(d) for d in skills_dirs]
        self.inputs, self.extra_env = dict(inputs or {}), dict(extra_env or {})
        self.versions = dict(versions or {})
        self.http, self.probe, self.clock, self.sleep = http, probe, clock, sleep
        self.budgets, self.disk_usage = budgets, disk_usage
        self.credentials_visible = list(credentials_visible or [])
        self.required_chips = tuple(required_chips) if required_chips else None   # stage 4's required counts
        self.home = Path(home) if home is not None else operator_home()   # whose shared caches to refuse
        self.containers = containers if containers is not None else LabelledContainers()
        # The machine paths the skills' placeholders name (orchard/paths.py).
        self.paths = paths if paths is not None else RunPaths.resolve(self.run_dir, home=self.home)
        # The docker label every test container of this run carries (ORCHARD_TEST_LABEL).
        self.run_label = "orchard.test=" + hashlib.sha256(str(self.run_dir).encode()).hexdigest()[:12]
        self.control = Control(self.run_dir)
        self.actuator = RunActuator(self.control)
        self.guard = RetryGuard()
        self.coder_lease: Lease | None = None
        self.test_lease: Lease | None = None      # a stage's own test lease, while it is held
        self.watchdog: Watchdog | None = None
        self.stop_message = ""                     # what the shutdown did, for main to print
        agent_env(self.run_dir, extra=self.extra_env)     # refuse a credential before anything runs

    # ---- small helpers ----------------------------------------------------------------------

    def _block(self, stage, reason: str, **evidence):
        self.ledger.append("notice", stage, blocked=True, reason=reason, evidence=evidence)
        raise Blocked(reason, **evidence)

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
            ok = self._release_test_lease() and self._release_all()
        except Exception:
            ok = self._release_from_memory()
        self.stop_message = f"Aborted by {name}. " + self._release_summary(ok)
        return EXIT_ABORTED

    def _stop_on_error(self, exc: Exception) -> None:
        self._ignore_signals()
        try:
            self.ledger.append("notice", None, what="the supervisor stopped on an error",
                               error=f"{type(exc).__name__}: {exc}"[:500])
            ok = self._release_test_lease() and self._release_all()
        except Exception:
            ok = self._release_from_memory()
        self.stop_message = self._release_summary(ok) + (
            " Fix the cause, then run the same command again to resume." if ok else "")

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
        try:
            self.adapter.release(lease)
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
            try:
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
                self.adapter.release(lease)
            except (AdapterError, ServerError, OSError):
                ok = False
        return ok

    def _run(self) -> int:
        p = run_progress(self.ledger.read())
        if p.aborted or p.finished:
            self._release_all()           # finishes a release a crash interrupted; else does nothing
            return EXIT_ABORTED if p.aborted else EXIT_READY
        if not p.started:
            self.ledger.append("run_start", None, model=self.model_id, versions=self.versions,
                               inputs=self.inputs, required_chips=list(self.required_chips or ()) or None,
                               coder=self.coder.record(),
                               tiers={k: dict(v) for k, v in self.cfg.tiers.items()})
        if self.credentials_visible:
            # Recorded at every start, because each start is a fresh acceptance by the operator.
            self.ledger.append("decision", None, decision="operator accepted visible credentials",
                               paths=self.credentials_visible, by="--accept-credentials-visible")
        while True:
            p = run_progress(self.ledger.read())
            if p.paused is not None:
                if self._wait_for_operator() == "abort":
                    return self._abort()
                continue
            if p.next_stage is None:
                break
            cap = budget_cap(p, self.clock())
            if cap:
                self.ledger.append("decision", p.next_stage, decision="pause", reason=cap)
                continue
            if p.next_stage == 2 and self._full_port_unacknowledged():
                self.ledger.append("decision", 2, decision="pause", reason=FULL_PORT)
                continue
            # The spec follows the path stage 0 chose, read from the ledger each time, so a
            # resumed run picks the same skill and gate as the run that crashed.
            spec = spec_for(p.next_stage, run_path(self.ledger.read(), self.run_dir))
            try:
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
        # hardware back.
        self._release_all()
        return EXIT_READY

    def _full_port_unacknowledged(self) -> bool:
        """Stage 0 chose a full port, and the run has not yet paused for it."""
        entries = self.ledger.read()
        return run_path(entries, self.run_dir) == "full-port" and not any(
            e["event"] == "decision" and e["data"].get("reason") == FULL_PORT for e in entries)

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

    def _abort(self) -> int:
        # Recorded first: a crash during the release still leaves an aborted run, and a restart
        # finishes the release and refuses to go on.
        self.ledger.append("decision", None, decision="abort", by="operator")
        self._release_all()
        return EXIT_ABORTED

    def _release_all(self) -> bool:
        """Stop the coder and release its lease. Safe to call again: a released lease is skipped.
        Returns False when the lease could not be released."""
        entries = self.ledger.read()
        lease_rec, server_rec, _ = coder_state(entries)
        released = {e["data"].get("lease_id") for e in entries
                    if e["event"] == "decision" and e["data"].get("decision") == "hardware released"}
        if lease_rec is None or lease_rec["lease_id"] in released:
            return True
        lease = Lease.from_record(lease_rec)
        if server_rec:
            self.coder.adopt(server_rec)
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
            return True
        except (AdapterError, ServerError, OSError) as exc:
            self.ledger.append("notice", None, what="releasing the hardware failed", error=str(exc))
            return False

    # ---- the coder ----------------------------------------------------------------------------

    def _ensure_coder(self) -> None:
        """The coder serves under a lease this process owns, before any stage runs."""
        entries = self.ledger.read()
        hp = progress(entries)
        if hp.phase != "idle":
            # A park or restore was interrupted (a crash, or a block the operator resumed).
            rec = recover(entries, adapter=self.adapter, server=self.coder, standin=self.standin)
            h = self._handoff(hp.stage, Lease.from_record(hp.lease))
            h.recover_and_restore(rec, chips=self.coder_chips, who=WHO, reason="coder",
                                  wait_budget_s=COLD_BOOT_BUDGET_S)
            if rec.action != "abandon":
                self.coder_lease = h.lease
                return
            entries = self.ledger.read()      # abandoned: the coder serves under the old lease
        lease_rec, server_rec, canary = coder_state(entries)
        if lease_rec is None:
            self._start_coder(None)
            return
        if server_rec:
            self.coder.adopt(server_rec)
        lease = Lease.from_record(lease_rec)
        mine = [c for c in self.adapter.status() if c.bdf in lease.chips]
        if not (mine and all(c.lease_pid == self.adapter.owner_pid for c in mine)):
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
        self.ledger.append("decision", None, decision="coder starting", lease=lease.record(),
                           server=self.coder.record())
        try:
            self.coder.start(lease)
        except (ServerError, OSError) as exc:
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
        # The boot takes up to 45 minutes and is the likeliest time for a kill. This entry gives
        # the kill test a kill point while the container runs and has not answered.
        self.ledger.append("decision", None, decision="coder container started",
                           lease_id=lease.lease_id)
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
        if not self.coder.confirm_stopped().stopped:
            self.ledger.append("decision", None, decision="stopping the coder", result=self.coder.stop())
        check = wait_stopped(self.coder, self.adapter, old, quiet_wait_s=self.budgets.quiet_wait_s,
                             poll_s=self.budgets.poll_s, clock=self.clock, sleep=self.sleep,
                             accept=("CLAIMED", "STALE", "FREE"))
        if not check["ok"]:
            self._block(None, "the coder could not be confirmed stopped for the relaunch", **check)
        self._start_coder(canary)

    def _coder_died(self, lease: Lease, canary: dict | None) -> None:
        """Spec section 10: one restart with the same config and a canary check; a second death
        blocks."""
        if run_progress(self.ledger.read()).coder_deaths >= 1:
            self._block(None, "the coder died a second time since the last resume")
        self.ledger.append("decision", None, decision="coder died; restarting it once")
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
        ok, free = check_disk(self.run_dir, spec.disk_gb, usage=self.disk_usage)
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
        if spec.boards == 0:
            out, step = self._step(spec, "run", stage_dir, escalated, resumed)
            if out.status != "done":
                return out.status, [f"the agent step ended: {out.status} {out.detail}".strip()], None
        else:
            if spec.tests:
                ended = self._run_test_list(spec, stage_dir, escalated, resumed)
                if ended:
                    return ended
            elif not (resumed and (stage_dir / "test-result.json").is_file()):
                out, _ = self._step(spec, "prepare", stage_dir, escalated, resumed)
                if out.status != "done":
                    return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
                test, problems = self._read_test(stage_dir)
                if problems:
                    return "fail", problems, None
                self._hardware_phase(spec, stage_dir, test)
            out, step = self._step(spec, "finish", stage_dir, escalated, resumed)
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
        if not gate.ok:
            # One continuation per run of the stage body; there is no loop here on purpose. It is
            # for a missing or malformed output file after a run step, or after a hardware test
            # that succeeded.
            out = self._gate_feedback(spec, stage_dir, step, gate)
            if out.status != "done":
                return out.status, [f"the {step.phase} step's continuation ended: "
                                    f"{out.status} {out.detail}".strip(), *gate.reasons], None
            gate = self._check_gate(spec, stage_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _test_failure(self, spec, stage_dir: Path) -> dict | None:
        """Why the stage's hardware testing failed, or None. One test: its test-result.json. A list
        of tests: the records of the tests that must pass (`failed_tests`), so an optional
        configuration that failed still leaves the gate feedback for a malformed result file."""
        if spec.tests:
            return failed_tests(stage_dir, self.required_chips)
        return hardware_test_failure(stage_dir)

    def _check_gate(self, spec, stage_dir: Path):
        if spec.number == 8:
            bundle = stage_dir / "bundle"
            bundle.mkdir(exist_ok=True)
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
        gate = self._gate(spec)(stage_dir, self.run_dir)
        if spec.tests:
            # The gate reads the test records from the stage directory, where the agent can write
            # too. The ledger holds the sha256 of each record the supervisor wrote.
            forged = unrecorded(self.ledger.read(), stage_dir, self.run_dir, spec.number)
            if forged:
                return GateResult(False, gate.reasons + tuple(forged), gate.evidence)
        return gate

    def _gate_feedback(self, spec, stage_dir: Path, step: AgentStep, gate):
        """The step ended "done" and the exit gate failed. Tell the same conversation what the gate
        found and run it once more, with its own turn cap and its own log."""
        n = spec.number
        entry = self.ledger.append("decision", n, decision="gate feedback", phase=step.phase,
                                   reasons=list(gate.reasons), file=f"stages/{n}/{spec.gate_file}")
        text = gate_feedback_text(spec, gate.reasons)
        log = stage_dir / "log" / f"{step.phase}-continuation-{entry['seq']:05d}.jsonl"
        return step.continue_with(text, max_turns=AGENT_CONTINUATION_TURNS, log_path=log)

    def _end_stage(self, spec, stage_dir: Path, status: str, reasons: list, gate, escalated: bool) -> str:
        n = spec.number
        if status == "pass":
            ev = [evidence_record(self.run_dir, self.run_dir / rel) for rel in sorted(set(gate.evidence))]
            ev.append(evidence_record(self.run_dir, stage_dir / spec.gate_file))
            if n == 6:
                self._record_numbers(stage_dir)
            # Stage 0's pass records the path it chose. Later stages read it from here
            # (orchard/stages.py, run_path), so an edit to delta.json cannot change the run's path.
            extra = {"path": delta_path(self.run_dir)} if n == 0 else {}
            self.ledger.append("stage_end", n, result="pass", evidence=ev, **extra)
            return "pass"
        if status == "pause":
            return "pause"              # the ladder already wrote the pause decision
        if status == "operator-pause":
            self.ledger.append("decision", n, decision="pause", reason="operator")
            return "pause"
        if status == "abort":
            return "abort"
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
        if note:
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
                         tools=Tools(self.run_dir, stage_dir, agent_env(self.run_dir, extra=self.extra_env)),
                         ledger=self.ledger, stage=n, phase=phase, feed=self.watchdog.feed,
                         control=self.actuator, run_dir=self.run_dir,
                         evidence_dir=stage_dir / "evidence",
                         log_path=stage_dir / "log" / f"{phase}-{len(entries) + 1:05d}.jsonl",
                         http=self.http, clock=self.clock, guard=self.guard)
        return step.run(system, user), step

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
        if not isinstance(note, dict) or any(note.get(k) in (None, "") for k in NOTE_KEYS):
            problems.append(f"handoff.json needs {', '.join(NOTE_KEYS)}")
        try:
            check_string(test["command"], self.run_dir)
        except Denied as exc:
            problems.append(f"the test command is refused: {exc}")
        return test, problems

    def _hardware_phase(self, spec, stage_dir: Path, test: dict) -> None:
        n = spec.number
        chips = self.adapter.status()
        d = decide_park(self.coder_lease.units, chips, spec.boards)
        self.ledger.append("decision", n, decision="hardware phase", action=d.action,
                           free_boards=list(d.free_boards), server_boards=list(d.server_boards))
        if d.park_needed:
            h = self._handoff(n, self.coder_lease)
            lease = h.park()
            self._run_test(spec, stage_dir, test, lease)
            h.restore()
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
        self._run_test(spec, stage_dir, test, lease)
        try:
            self.adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.test_lease = None
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _run_test(self, spec, stage_dir: Path, test: dict, lease: Lease) -> dict:
        n = spec.number
        k = spec.boards * CHIPS_PER_BOARD
        deadline = min(float(test["deadline_s"]), spec.budget_s)
        result = self._spawn(n, test["command"], lease.chips[:k], lease.dev_indices[:k], deadline,
                             stage_dir / "evidence" / "hw-test-output.txt", lease_id=lease.lease_id)
        marker = stage_dir / "test-result.json"
        tmp = stage_dir / "test-result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what="hardware test", returncode=result["returncode"],
                           timed_out=result["timed_out"], **evidence_record(self.run_dir, marker))
        return result

    def _spawn(self, n: int, command: str, chips, ids, deadline: float, out_path: Path,
               **record) -> dict:
        """Run one hardware test command on `chips` (device indices `ids`) and return its record.
        `record` adds keys to the "hardware test started" decision."""
        env = agent_env(self.run_dir, extra=self.extra_env)
        # The leased chips replace the agent shells' no-chip mask. TT_METAL_VISIBLE_DEVICES takes
        # device indices in another form, so the hardware test gets none and the runtime follows
        # TT_VISIBLE_DEVICES. A container test needs the /dev/tenstorrent indices of the same
        # chips (ORCHARD_DEVICE_IDS) and the label its containers carry (ORCHARD_TEST_LABEL).
        env["TT_VISIBLE_DEVICES"] = ",".join(chips)
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        env["ORCHARD_DEVICE_IDS"] = ",".join(str(i) for i in ids)
        env["ORCHARD_TEST_LABEL"] = self.run_label
        self.ledger.append("decision", n, decision="hardware test started", command=command,
                           deadline_s=deadline, chips=list(chips), **record)
        t0 = self.clock()
        with open(out_path, "wb") as out:
            try:
                code, timed_out = spawn_checked(command, self.run_dir, env, deadline, out)
            except Denied as exc:
                code, timed_out = None, False
                out.write(f"refused: {exc}\n".encode("utf-8"))
        return {"command": command, "returncode": code, "timed_out": timed_out,
                "seconds": round(self.clock() - t0, 3), "chips": list(chips),
                "output": evidence_record(self.run_dir, out_path)}

    # ---- a list of hardware tests (stage 4 on the weights-only path) ---------------------------

    def _run_test_list(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        """Prepare the list (unless a resumed stage has its plan), then run every test that has
        no record. Returns (status, reasons, gate) when the stage ends here, or None when the
        finish step should run."""
        n = spec.number
        tests = load_plan(stage_dir) if resumed else None
        if tests is None:
            out, _ = self._step(spec, "prepare", stage_dir, escalated, resumed)
            if out.status != "done":
                return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
            tests, problems = read_plan(stage_dir, required=self.required_chips,
                                        max_chips=spec.boards * CHIPS_PER_BOARD,
                                        budget_s=spec.budget_s, home=self.home)
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
            self._run_listed_test(spec, stage_dir, test, [lease])
            self._sweep(n)
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
        self._run_listed_test(spec, stage_dir, test, [lease] + ([extra] if extra else []))
        self._sweep(n)
        if extra is not None:
            self._give_back_test_lease(n, extra)
        h.restore()
        self.coder_lease = h.lease

    def _take_test_lease(self, spec, test, boards: int, exact) -> Lease:
        n = spec.number
        lease = reacquire(self.adapter, chips=boards * CHIPS_PER_BOARD, who=WHO,
                          reason=f"stage {n} {test.chips}-chip test", ledger=self.ledger, stage=n,
                          wait_budget_s=spec.budget_s, clock=self.clock, sleep=self.sleep,
                          exact=exact)
        self.test_lease = lease
        self.ledger.append("decision", n, decision="test lease taken", config=test.chips,
                           test_lease=lease.record())
        return lease

    def _give_back_test_lease(self, n: int, lease: Lease) -> None:
        try:
            self.adapter.release(lease)        # the lease tool resets the board as it releases
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
        result = self._spawn(n, test.command(n), [c for _, c in chosen], [i for i, _ in chosen],
                             min(test.deadline_s, spec.budget_s), out_dir / "output.txt",
                             config=test.chips, cache=test.cache,
                             lease_ids=[lease.lease_id for lease in leases])
        path = write_record(stage_dir, test.chips, {"config": test.chips, **result,
                                                    "device_ids": [i for i, _ in chosen]})
        self.ledger.append("evidence", n, what="hardware test", config=test.chips, cache=test.cache,
                           returncode=result["returncode"], timed_out=result["timed_out"],
                           **evidence_record(self.run_dir, path))

    def _sweep(self, n: int) -> None:
        """Stop and remove any container this run's tests left running. A container that is
        still listed afterwards blocks the stage: releasing or resetting chips under it is
        refused, and the operator decides."""
        found = self.containers.list(self.run_label)
        if found is None:
            self.ledger.append("notice", n, what="docker could not be asked for test containers",
                               label=self.run_label)
            return
        if not found:
            return
        self.ledger.append("notice", n, what="a hardware test left containers running; stopping them",
                           containers=found, label=self.run_label)
        for cid in found:
            self.containers.remove(cid)
        left = self.containers.list(self.run_label)
        if left:
            self._block(n, "test containers are still running after docker stop and rm; no test "
                           "lease was released and the coder was not restored", containers=left)


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
    r.add_argument("--skills-dir", action="append", default=[],
                   help="more skill directories, searched after orchard/skills")
    r.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    r.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                   help="a variable for agent shells (never a credential)")
    r.add_argument("--required-chips", type=chip_counts, default=None, metavar="N,N",
                   help="chip counts stage 4 must pass, such as 2,4. Any other count it records is "
                        "optional. Without this, every configuration stage 4 lists must pass. The "
                        "ledger records it, and a resumed run keeps it")
    r.add_argument("--gozer", default="gozer")
    r.add_argument("--accept-credentials-visible", action="store_true",
                   help="start even though credential files exist in the operator's home "
                        "(agent shells can read them); the ledger records this")
    c = sub.add_parser("control", help="send pause, resume or abort to a running supervisor")
    c.add_argument("--run-dir", required=True)
    c.add_argument("command", choices=Control.COMMANDS)
    return p.parse_args(argv)


def build(args, ledger, *, adapter=None, coder=None, versions=None, http=post_json,
          probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(),
          disk_usage=shutil.disk_usage, home=None, containers=None) -> Supervisor:
    """A Supervisor from parsed `run` arguments. Tests pass fakes for the machine."""
    # Everything that can be refused is checked before any external command runs.
    cfg = load(args.tiers)
    tier = coder_tier(cfg, args.coder_port)
    run_dir = Path(args.run_dir).resolve()
    inputs, extra_env = pairs(args.input, "input"), pairs(args.env, "env")
    agent_env(run_dir, extra=extra_env)
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
                             f"{required} differs. Leave the option out to resume with the recorded value")
        required = recorded
    return Supervisor(run_dir=run_dir, ledger=ledger, cfg=cfg, model_id=args.model, adapter=adapter,
                      coder=coder, coder_chips=args.coder_chips,
                      standin=ExternalStandIn(cpu["endpoint"], cpu["model"], http=http),
                      skills_dirs=[SKILLS_DIR, *args.skills_dir], inputs=inputs,
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage,
                      credentials_visible=found, required_chips=required,
                      home=operator_home() if home is None else home, containers=containers)


def main(argv=None, *, home=None) -> int:
    args = parse(argv)
    if args.cmd == "control":
        Control(args.run_dir).write(args.command)
        print(f"wrote {args.command!r} to {Path(args.run_dir) / 'control'}")
        return 0
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
    print({EXIT_READY: "ready for operator review", EXIT_ABORTED: "aborted"}.get(code, code))
    return code


if __name__ == "__main__":
    sys.exit(main())
