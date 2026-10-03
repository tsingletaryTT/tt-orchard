"""The supervisor loop: run a bring-up stage by stage, from the ledger (spec sections 5 to 7, 10).

This module owns the run. At start it records the run (model, resolved versions, inputs) or, on a
restart, replays the ledger. It keeps the coder (the chip tier's model server) running under a
lease this process owns: it starts it, re-leases it after a restart, restarts it once if it dies,
and finishes a park that a crash interrupted (orchard/handoff.py). For each stage it checks the
run-wide caps and the free disk, opens the stage directory (orchard/stages.py), runs the agent
steps (orchard/agent.py) under the watchdog with a real actuator, runs the hardware test on a
leased board, parking the coder when no board is free, and writes the stage's end with the exit
gate's result. A failed stage is escalated once to its diagnose tier (or the [escalation]
default); a second failure pauses the run. When stage 0 finds that the model needs new model code
(a full port), the run pauses before stage 2 for the operator.

Operator commands go through a one-word control file in the run directory: pause, resume, abort.
A paused supervisor keeps its leases and waits. Abort stops the coder, releases its lease and
closes the ledger. The run ends at "ready for operator review" and never publishes: the command
runner refuses publish, push and upload commands, and agents have no credentials.

Plan 4 runs stages 0 to 6 and 8. Stage 7 (package and container build) is recorded as skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from orchard.adapters import AdapterError, Lease
from orchard.agent import AgentStep, Tools, agent_env, probe_model, spawn_checked
from orchard.canary import CanaryError, compare, post_json
from orchard.canary import ask as canary_ask
from orchard.commands import run_command
from orchard.context import build_messages, facts_from
from orchard.defaults import (CHIPS_PER_BOARD, CMD_TIMEOUT_S, COLD_BOOT_BUDGET_S, CONTROL_POLL_S,
                              RUN_CANARY_PROMPT)
from orchard.handoff import (NOTE_KEYS, Blocked, Budgets, Handoff, decide_park, progress, reacquire,
                             recover, wait_stopped)
from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked
from orchard.runner import Denied, check_string
from orchard.server import ServerControl, ServerError, ServerSpec, StopCheck
from orchard.stages import (STAGES, TierUnavailable, attempt_started_ts, budget_cap, check_disk,
                            coder_state, evidence_record, open_stage_dir, resolve_endpoint,
                            resolve_skill, run_progress, tier_for)
from orchard.tiers import TierConfigError, load
from orchard.watchdog import (Event, IdenticalResponses, Ladder, NoNewEvidence, RepeatedToolCall,
                              RetryGuard, StageOverBudget, ThinkingWithoutAction, Watchdog)

WHO = "orchard:supervisor"
AGENT = "stage-agent"                 # the launched agent's name; the ladder counts its rungs per stage
EXIT_READY, EXIT_REFUSED, EXIT_ABORTED = 0, 2, 4
FULL_PORT = ("stage 0 found a full port (new model code is needed); plan 4 runs weights-only "
             "bring-ups, so the operator decides whether to go on")
SKILLS_DIR = Path(__file__).with_name("skills")


# ---- operator control ---------------------------------------------------------------------------

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


def coder_tier(cfg, port: int) -> str:
    """The chip tier whose endpoint uses the coder's port."""
    names = [n for n, t in cfg.tiers.items()
             if t["placement"] == "chips" and urlparse(t["endpoint"]).port == port]
    if len(names) != 1:
        raise ValueError(f"exactly one chip tier must use port {port}; found {names}")
    return names[0]
