# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The blocked end state of an unattended run.

An unattended run (`tt-orchard bringup`, `supervisor run --unattended`) never waits for a person. When the
harness would pause, it names what stopped it, writes a bundle that says so, gives the hardware back and
exits with `EXIT_BLOCKED`. Running the same command again retries: the supervisor records a resume and goes
on from the ledger.

A pause the operator asked for (`control pause`) is not a block: the operator is there and will resume it.

The reasons are a fixed set, so a person and a program can both act on them. The mapping from a pause to a
reason reads the pause decision the supervisor already writes (orchard/status.py, `classify_pause`). A pause
nobody wrote a rule for becomes `unclassified` and keeps its text, so a new kind of pause cannot silently
hang an unattended run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from orchard.stages import STAGES
from orchard.status import classify_pause, stage_rows

REASONS = ("needs-new-model-code", "retry-budget-spent", "stage-failed", "agent-stuck", "disk-full",
           "hardware-unhealthy", "coder-unusable", "blocked", "unclassified")

# Words in a "blocked: ..." reason that say what kind of block it is. The first matching row wins.
_BLOCKED_WORDS = (
    ("disk-full", ("disk", "free space", " gb free")),
    ("hardware-unhealthy", ("lease", "chip", "board", "reset", "gozer", "hardware", "device")),
    ("coder-unusable", ("coder", "canary", "server", "model did not")),
)


def block_code(pause: dict) -> str | None:
    """The reason code for a pause decision, or None when it is not a block (an operator pause)."""
    kind, _ = classify_pause(pause)
    if kind == "operator":
        return None
    if kind == "full_port":
        return "needs-new-model-code"
    if kind == "budget":
        return "retry-budget-spent"
    if kind == "stage_failed":
        return "stage-failed"
    if kind == "watchdog":
        return "agent-stuck"
    if kind == "blocked":
        text = str(pause.get("reason", "")).lower()
        for code, words in _BLOCKED_WORDS:
            if any(w in text for w in words):
                return code
        return "blocked"
    return "unclassified"


def last_pause(entries: list[dict]) -> dict | None:
    """The data of the last pause decision after the last resume, or None."""
    for e in reversed(entries):
        if e["event"] != "decision":
            continue
        decision = e["data"].get("decision")
        if decision == "resume":
            return None
        if decision == "pause":
            return e["data"]
    return None


def _stage_names() -> dict[int, str]:
    return {s.number: s.name for s in STAGES}


def write_bundle(run_dir, entries: list[dict], code: str, reason: str, *, now: float) -> None:
    """Write BLOCKED.md and blocked.json in the run directory, replacing any earlier pair."""
    run_dir = Path(run_dir)
    start = next((e for e in entries if e["event"] == "run_start"), None)
    model = start["data"].get("model") if start else None
    rows = stage_rows(entries, now)
    open_stage = next((r["stage"] for r in reversed(rows) if r["status"] not in ("pass", "skipped")), None)
    at = next((e["stage"] for e in reversed(entries) if e["event"] == "decision"
               and e["data"].get("decision") == "pause"), None)
    stage = at if at is not None else open_stage
    facts = {"code": code, "reason": reason, "model": model, "stage": stage, "stages": rows,
             "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))}
    (run_dir / "blocked.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Blocked: {code}", "",
             f"Model: {model or 'not recorded (the run never started)'}", "",
             f"The harness stopped at stage {stage}." if stage is not None else "The harness stopped before a stage began.",
             "", "What stopped it:", "", f"> {reason}", "", "## Stages", "",
             "| Stage | Name | Status | Wall time (s) | Attempts |", "|---|---|---|---|---|"]
    names = _stage_names()
    for r in rows:
        lines.append(f"| {r['stage']} | {names.get(r['stage'], r['name'])} | {r['status']} | {r['wall_s']} | {r['attempts']} |")
    lines += ["", "## What to do", "",
              f"Fix what the reason names, then run `tt-orchard bringup {model}` again. The run resumes from its "
              "ledger.", "The ledger, each stage's evidence and the agent logs are in this run directory.",
              "Nothing was published, pushed or uploaded."]
    (run_dir / "BLOCKED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
