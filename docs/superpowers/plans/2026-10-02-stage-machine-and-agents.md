# tt-orchard: stage machine, agent steps and the supervisor loop (plan 4 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The smallest loop that lets the harness, with no Claude session in it, bring up a model whose architecture matches one already supported: stages 0 to 6 and a minimal stage 8 bundle, ending at "ready for operator review" and never publishing. The first target is `Altworld/Hemmingway-1`, a creative-writing fine-tune of Qwen3.8-27B with a weights-only delta.

**Architecture:** Five new modules on top of plans 1 to 3. `orchard/stages.py` holds the stage table (owner skill, boards, budget, free disk, gate file, exit gate, resume marker), tier lookup with escalation, skill lookup by name, the stage 0 comparison with a reference answer, and the ledger replay that drives the run (next stage, pauses, escalations, caps, the coder's lease, partial stage directories moved aside). `orchard/agent.py` runs one agent step: a fresh context, a non-streaming OpenAI-compatible call over stdlib urllib, a tool loop whose shell goes through `orchard/runner.py`, evidence files recorded with sha256, watchdog events for every response and tool call, nudges added to the next request, and an environment with no tokens. `orchard/context.py` builds that fresh context. `orchard/supervisor.py` is the loop: run start with versions, the coder kept under the supervisor's own lease (start, re-lease after a restart, one restart after a death, recovery of an interrupted park), each stage's agent steps under the watchdog with a real actuator, the hardware test on a free board or under a park, exit gates, escalation to the diagnose tier, budget caps, and pause, resume and abort through a control file. `orchard/scrub.py` is the bundle scrub. Four draft stage skills live in `orchard/skills/`. The existing tt-model-bringup skills are referenced by name and found at run time.

**Tech Stack:** Python 3.12 standard library only (`urllib.request`, `http.server`, `subprocess`, `json`, `dataclasses`), pytest. Tests run against a scripted OpenAI-shaped fake model server on 127.0.0.1 (`tests/fake_model.py`) and a fake two-board machine (`tests/run_fakes.py`). Agent shell commands in tests are real `bash` processes in `tmp_path`.

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md`, sections 1, 2, 5, 6, 7, 9, 10, 11 and 13. Plan 3 (`docs/superpowers/plans/2026-10-02-supervisor-behavior.md`) supplies the lease adapters, server control, park and restore (`orchard/handoff.py`), and the watchdog with its `Actuator` protocol; its hand-off notes list what plan 4 must do. The Hemmingway-1 reference answer for stage 0 is `/mnt/bonus/models/hemmingway-1/work/stage0-reference.md`; Task 3 copies it into `tests/fixtures/`.

**Checked before hand-over:** every code block below was extracted from this document by a script into a fresh clone of the repo (branch `plan3-supervisor-behavior` at `f1bc1d1`), applied task by task in document order, and run on 2026-10-02: 962 passed and 1 skipped (854 passed before, so 108 new tests). Every mutation step was run on that copy; each named test failed with the mutation and passed after the restore. The implementer still runs every step; this check does not replace them.

## Global Constraints

- Work in `/home/ttuser/code/tt-orchard`. Task 0 creates the branch `plan4-stage-machine` from `plan3-supervisor-behavior`; every later task commits on it. The repo is local only. Never run `git push` and never create a remote.
- Python 3.12, standard library only. Every new module starts with `from __future__ import annotations` and a docstring that says what the module owns. Comments explain why.
- The unit suite uses no hardware and no real model. No test opens `/dev/tenstorrent/*` or runs `tt-smi`, `tt-model`, `docker`, `gozer` or a model server other than `tests/fake_model.py`. Tests build a supervisor only through `build(..., adapter=<fake>, coder=<fake>, versions=<dict>)`. A single test calls `main` with `run`, and that call is refused before any external command runs.
- Do not run the supervisor against hardware, and do not run any gozer command that changes state on the real box. The controller runs the Hemmingway-1 runbook entry (Task 11).
- Change existing modules only where a task says so: `orchard/defaults.py` (append), `orchard/ledger.py` (one event name), one comment in `orchard/watchdog.py`, `tests/test_defaults.py` and `tests/test_ledger.py` (append), and the documents in Task 11. Plan 3's modules are used through their public functions.
- The run never publishes. Nothing in this plan runs a publish, push or upload command. Every agent shell command goes through `orchard/runner.py`'s checks, and agent shells get no credentials.
- Retry rule (spec section 3): never retry the same call with the same inputs more than once. Model requests go through the watchdog's `RetryGuard`.
- Every default value lives in `orchard/defaults.py` as one named constant, with a comment that cites where it was measured or says that it is a choice and was not measured.
- Every guard gets a mutation step: change the guard, run the named test, watch it fail, restore the guard (spec section 13). After each restore, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`) before the confirming run. Plan 3 found that a stale `.pyc` can hide a restore.
- Run the whole suite before each commit: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider`. The baseline is 854 passed and 1 skipped in about 175 s. After this plan: 962 passed and 1 skipped in about 200 s.
- The stage skills in `orchard/skills/` are local drafts, marked `status: draft.` They live in tt-orchard. They never name a lease tool. This plan changes nothing in tenstorrent/skills and opens no PR.
- Out of scope: the request-rewriting proxy for agents the supervisor did not launch, streaming, new model code for architectures that do not match (full ports), stage 7 (package and container build), the `stage-review` skill, read-only mounts, and publishing of any kind.
- Commit messages are plain English, one idea per commit. Writing rules for docs and comments: state the finding; short sentences; no "X, not Y" framing; no aphoristic closers; no metaphors that stand in for a claim.

## Review Focus

1. An agent shell that can reach tokens. Expected: the agent's environment is an allow-list with HOME inside the run directory, so `env` shows no token and `~/.cache/huggingface/token` is not found even when the supervisor has both; the hardware test gets the same environment plus `TT_VISIBLE_DEVICES`; an `--env` name that looks like a credential is refused before anything runs (Task 5: `test_a_shell_command_finds_no_token_in_its_environment_or_home`; Task 9: `test_the_hardware_test_gets_the_leased_chips_and_no_token`, `test_main_refuses_a_credential_before_running_anything`). A file outside the run directory stays readable by absolute path; the docstring and the runbook say so.
2. An escalation that loops. Expected: a failed stage is escalated once to its diagnose tier (or the `[escalation]` default); a failure after escalation pauses the run; three escalations since the last resume pause it too; a watchdog escalation ends the agent step (Task 6: `test_a_second_loop_escalates_and_stops_the_step`; Task 9: `test_a_failed_gate_escalates_once_to_the_diagnose_tier`, `test_a_second_failure_pauses_the_run_until_the_operator_resumes`, `test_the_escalation_cap_pauses_the_run`).
3. A ledger gap on a crash. Expected: a supervisor killed after any ledger event, in either machine layout, restarts, re-leases the coder under its new pid, finishes any interrupted park, and reaches the same final state with no lease left behind (Task 10: `test_a_kill_after_any_ledger_event_reaches_the_same_final_state[4]` and `[2]`). The lease and server are recorded before the coder starts, so a crash during a boot leaves them in the ledger.
4. A park that is needed and skipped. Expected: with the coder on all four chips, every hardware stage's test runs after the park's reset and before the restore's reset, on one board's chips; with the coder on two chips nothing parks and the free board is leased and released (Task 10: `test_with_the_coder_on_four_chips_every_hardware_stage_parks_it`, `test_with_the_coder_on_two_chips_the_free_board_is_used_and_nothing_parks`).
5. An agent that writes outside the run directory. Expected: `write_file` refuses `../`, absolute paths, links that lead out, and ledger names; the shell refuses `rm` outside the run directory before anything in the string runs; a gate refuses evidence that is not a file inside the run directory (Task 5: `test_write_file_stays_inside_the_stage_directory`, `test_a_refused_command_never_starts`; Task 3: `test_evidence_outside_the_run_directory_does_not_count`). A shell redirect or a Python script can still write elsewhere; the agent docstring and the runbook say so.

## File Structure

| File | Responsibility |
|---|---|
| `orchard/defaults.py` | plan 4 budgets, disk needs, agent limits and run caps, one constant each |
| `orchard/ledger.py` | one new event name: `evidence` |
| `orchard/scrub.py` | the bundle scrub: hostname, tokens, absolute home paths |
| `orchard/stages.py` | stage table, gates, tier and skill lookup, disk check, stage 0 comparison and its CLI; ledger replay, stage directories, budget caps, the coder's recorded lease |
| `orchard/agent.py` | agent environment, the `shell` and `write_file` tools, the model call and the tool loop |
| `orchard/context.py` | the fresh context for one agent step |
| `orchard/skills/*.md` | draft stage skills: delta-triage, reference-gate, serving-check, operator-bundle |
| `orchard/supervisor.py` | control file, actuator, external stand-in, versions; the supervisor loop; the CLI |
| `tests/fake_model.py` | scripted OpenAI-shaped model server for tests |
| `tests/run_fakes.py` | fake two-board machine, coder, crashing ledger, scripted bring-up, tier file writer |
| `tests/fixtures/hemmingway_stage0_reference.md` | copy of the hand-written stage 0 reference answer |
| `tests/test_*.py` | one test file per module, plus `test_supervisor_parts.py` and `test_run_e2e.py` |
| `docs/runbooks/hardware-validation.md`, `README.md`, `CLAUDE.md`, the spec | the Hemmingway-1 runbook entry and documentation of what was built |

---

### Task 0: Branch from plan 3

**Files:**
- None changed.

**Interfaces:**
- Consumes: branch `plan3-supervisor-behavior` at `f1bc1d1` or later.
- Produces: the branch `plan4-stage-machine`.

- [ ] **Step 1: Confirm a clean tree on plan 3's branch**

Run: `cd /home/ttuser/code/tt-orchard && git status --short --branch && git log --oneline -1`
Expected: `## plan3-supervisor-behavior` and nothing else from `git status`; the last commit is `f1bc1d1 Docs: tt-gozer merged, park check ran on hardware, recovery abandons an early park` or a later commit on that branch. A later commit can change plan 3's interfaces; if `orchard/server.py`'s `StandIn` protocol or `orchard/handoff.py`'s `Handoff`, `recover` or `Recovery` differ from what Tasks 8 and 9 use, stop and report it. If the tree is not clean, stop and report it.

- [ ] **Step 2: Run the baseline suite**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `854 passed, 1 skipped`.

- [ ] **Step 3: Branch**

Run: `git switch -c plan4-stage-machine && git status --short --branch`
Expected: `## plan4-stage-machine` and nothing else.

---

### Task 1: Plan 4 defaults and the evidence ledger event

**Files:**
- Modify: `orchard/defaults.py` (append), `orchard/ledger.py` (`EVENTS`), `orchard/watchdog.py` (one comment)
- Test: `tests/test_defaults.py` (append), `tests/test_ledger.py` (append)

**Interfaces:**
- Produces in `orchard.defaults`: `STAGE_BUDGET_S: dict[int, float]`, `STAGE_DISK_GB: dict[int, float]`, `LONG_STAGE_S`, `STAGE2_PCC_MIN`, `AGENT_MAX_TURNS`, `AGENT_MAX_TOKENS`, `AGENT_REQUEST_TIMEOUT_S`, `TOOL_TIMEOUT_S`, `TOOL_OUTPUT_CHARS`, `CONTEXT_FILE_CHARS`, `SKILL_CHARS`, `RUN_ESCALATION_CAP`, `RUN_COLD_BOOT_CAP`, `COLD_START_S`, `RUN_WALL_CLOCK_S`, `CONTROL_POLL_S`, `RUN_CANARY_PROMPT`.
- Produces in `orchard.ledger`: the event name `"evidence"` in `EVENTS` (an evidence file's run-relative path and sha256).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_defaults.py`:

```python


def test_every_stage_has_a_budget_and_a_disk_need():
    assert set(d.STAGE_BUDGET_S) == set(range(9)) == set(d.STAGE_DISK_GB)


def test_a_hardware_stage_budget_holds_a_cold_boot():
    # Stages 2 to 6 may boot the model cold (about 30 min) at least once.
    assert all(d.STAGE_BUDGET_S[n] > d.COLD_BOOT_BUDGET_S for n in (2, 3, 4, 5, 6))


def test_a_hardware_stage_disk_need_covers_the_measured_tensor_cache():
    # The base model's 2-chip tensor cache measured 34 GB.
    assert all(d.STAGE_DISK_GB[n] >= 34 for n in (2, 3, 4, 5, 6))


def test_a_model_request_may_outlast_the_measured_prefill():
    assert d.AGENT_REQUEST_TIMEOUT_S > 78


def test_run_caps_allow_one_escalation_and_one_relaunch():
    assert d.RUN_ESCALATION_CAP >= 1 and d.RUN_COLD_BOOT_CAP >= 2


def test_a_cold_boot_is_told_apart_from_a_warm_restart():
    # Warm 2-chip restart 2-3 min, cold first boot about 30 min (spec section 3).
    assert d.WARM_RESTART_2CHIP_S < d.COLD_START_S < d.COLD_BOOT_S
```

Append to `tests/test_ledger.py`:

```python


def test_an_evidence_entry_is_accepted(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        e = led.append("evidence", 0, path="stages/0/evidence/a.txt", sha256="0" * 64)
    assert e["event"] == "evidence"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py tests/test_ledger.py`
Expected: 7 FAIL: six with `AttributeError: module 'orchard.defaults' has no attribute ...` and one with `ValueError: unknown ledger event 'evidence'`.

- [ ] **Step 3: Implement**

Append to `orchard/defaults.py`:

```python

# ---- plan 4: stages, agent steps and the run ---------------------------------------------------
# Per-stage wall-clock budgets (spec section 10). Choices; none is measured. Stage 5 holds one 2-chip
# cold boot (COLD_BOOT_S, about 30 min) plus its checks; stage 4 boots two configurations. Stage 7
# is skipped in plan 4, so its budget is 0.
STAGE_BUDGET_S = {0: 7200.0, 1: 14400.0, 2: 14400.0, 3: 21600.0, 4: 28800.0, 5: 10800.0,
                  6: 14400.0, 7: 0.0, 8: 3600.0}
# Free disk each stage needs on the run directory's filesystem before it starts (spec section 10).
# 40 GB covers one converted 2-chip tensor cache: the base Qwen3.8-27B TP=2 cache measured 34 GB
# (2026-09-30). Stage 4 converts a second (1-chip) cache. The other values are choices.
STAGE_DISK_GB = {0: 1.0, 1: 5.0, 2: 40.0, 3: 40.0, 4: 80.0, 5: 40.0, 6: 40.0, 7: 0.0, 8: 1.0}
LONG_STAGE_S = 3600.0           # spec section 10: a stage with a longer budget must declare a resume marker
STAGE2_PCC_MIN = 0.995          # the functional-decoder skill's default acceptance bar (prefill and decode)
AGENT_MAX_TURNS = 60            # choice: model turns in one agent step before the step counts as failed
AGENT_MAX_TOKENS = 8192         # choice: max_tokens for one model response
AGENT_REQUEST_TIMEOUT_S = 900.0 # choice: one non-streaming response; prefill alone measured 42 to 78 s
                                # at 130K to 204K tokens (spec section 3), and contexts here stay short
TOOL_TIMEOUT_S = 1800.0         # choice: one shell command run for an agent
TOOL_OUTPUT_CHARS = 12000       # choice: the head and tail of a command's output that go back to the model
CONTEXT_FILE_CHARS = 4000       # choice: how much of each earlier stage's result file a new context holds
SKILL_CHARS = 40000             # choice: a skill longer than this is cut, and the context says so
RUN_ESCALATION_CAP = 3          # choice: escalations since the last operator resume before the run pauses
RUN_COLD_BOOT_CAP = 3           # choice: cold coder boots since the last resume before the run pauses
COLD_START_S = 600.0            # choice: a coder start slower than this counts as a cold boot; it sits
                                # between the 2-3 min warm restart and the ~30 min cold boot (spec section 3)
RUN_WALL_CLOCK_S = 259200.0     # choice: 72 h from run start or the last resume, then the run pauses
CONTROL_POLL_S = 10.0           # choice: how often a paused supervisor reads the control file
RUN_CANARY_PROMPT = "What is 17 + 25? Answer with one number."   # choice: short, one greedy answer
```

Replace in `orchard/ledger.py`:

```python
    "retry", "escalate", "notice", "measurement", "decision",
})
```

with:

```python
    "retry", "escalate", "notice", "measurement", "decision",
    # An evidence file and its sha256 (plan 4: agent steps and hardware tests write them).
    "evidence",
})
```

Plan 3 left a comment in `orchard/watchdog.py` that asks plan 4 to add a `test_result` ledger event to the progress events. Plan 4 records test results as `evidence` entries and feeds evidence to the watchdog directly, so the event set stays as it is and the comment says why.

Replace in `orchard/watchdog.py`:

```python
# orchard/ledger.py has no `test_result` event today; plan 4 adds it with the stage test, and
# should add it here at the same time.
```

with:

```python
# Plan 4 records hardware test results and evidence files as `evidence` ledger entries. The agent
# loop (orchard/agent.py) feeds each new evidence file to the watchdog directly as an `evidence`
# event, and no agent runs while a hardware test runs, so no ledger event name is added here.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py tests/test_ledger.py`
Expected: PASS (57 tests: 13 in `test_defaults.py`, 44 in `test_ledger.py`).

- [ ] **Step 5: Mutation check**

1. The evidence event: in `orchard/ledger.py`, delete the line `    "evidence",`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_ledger.py::test_an_evidence_entry_is_accepted`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/defaults.py orchard/ledger.py orchard/watchdog.py tests/test_defaults.py tests/test_ledger.py
git commit -m "Add the plan 4 budgets, agent limits and run caps, and the evidence ledger event"
```

---

### Task 2: The bundle scrub

**Files:**
- Create: `orchard/scrub.py`
- Test: `tests/test_scrub.py`

**Interfaces:**
- Produces in `orchard.scrub`: `TOKEN_PATTERNS`, `HOME_PATH`, `SKIP_NAMES = frozenset({"ledger.jsonl"})`, `scrub_text(text: str, *, hostname: str, home: str) -> list[str]`, `scrub_bundle(bundle, *, hostname: str | None = None, home: str | None = None) -> list[str]` (each hit as `"relative/path: what was found"`; defaults are `socket.gethostname()` and `os.path.expanduser("~")`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scrub.py`:

```python
"""The bundle scrub check (spec section 10)."""
import pytest

from orchard.scrub import scrub_bundle, scrub_text

HOST, HOME = "quietbox-7", "/home/alice"


def test_clean_text_has_no_hits():
    assert scrub_text("Hemmingway-1 decodes at 80 tok/s on 2 chips.", hostname=HOST, home=HOME) == []


@pytest.mark.parametrize("text, what", [
    ("served from quietbox-7 last night", "the hostname 'quietbox-7'"),
    ("token hf_" + "a" * 34, "a Hugging Face token"),
    ("token ghp_" + "b" * 36, "a GitHub token"),
    ("token github_pat_" + "c" * 50, "a GitHub token"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", "a private key"),
    ("weights in /home/alice/models", "an absolute home path"),
    ("weights in /home/bob/models", "an absolute home path"),
    ("cache in /root/.cache", "an absolute home path"),
])
def test_each_kind_of_leak_is_found(text, what):
    assert scrub_text(text, hostname=HOST, home=HOME) == [what]


def test_a_hostname_inside_a_longer_name_is_not_a_hit():
    assert scrub_text("quietbox-70 is another machine", hostname=HOST, home=HOME) == []


def test_the_bundle_scrub_names_the_file_and_skips_the_operator_ledger(tmp_path):
    (tmp_path / "card.md").write_text("Built on quietbox-7.")
    (tmp_path / "ledger.jsonl").write_text('{"path": "/home/alice/run/evidence/x.txt"}\n')
    assert scrub_bundle(tmp_path, hostname=HOST, home=HOME) == ["card.md: the hostname 'quietbox-7'"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.scrub'`.

- [ ] **Step 3: Implement**

Create `orchard/scrub.py`:

```python
"""The bundle scrub check (spec section 10): search for the hostname, tokens and home paths.

The first p150 repo exposed a hostname in an old revision (spec section 3), so the operator bundle
is searched before it is called ready. A hit blocks the bundle. The supervisor runs this check
itself; the stage 8 agent's own scrub notes are not trusted for it.

`ledger.jsonl` is skipped. It is the operator's copy of the run's record, it is never published,
and it holds absolute evidence paths written by the park sequence (orchard/handoff.py). The
operator-bundle skill tells the agent to say so in RESULTS.md.

What this does not find: a token in a format not listed below, a hostname written in another form
(an IP address, a fully qualified name that differs from `socket.gethostname()`), or anything
inside a binary file.
"""
from __future__ import annotations

import os
import re
import socket
from pathlib import Path

TOKEN_PATTERNS = (
    ("a Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
HOME_PATH = re.compile(r"(?:/home/[A-Za-z0-9._-]+|/Users/[A-Za-z0-9._-]+|/root)(?=/|\b)")
SKIP_NAMES = frozenset({"ledger.jsonl"})


def scrub_text(text: str, *, hostname: str, home: str) -> list[str]:
    hits = []
    if hostname and hostname != "localhost" and re.search(rf"\b{re.escape(hostname)}\b", text):
        hits.append(f"the hostname {hostname!r}")
    for what, pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            hits.append(what)
    if (home and home != "/" and home in text) or HOME_PATH.search(text):
        hits.append("an absolute home path")
    return hits


def scrub_bundle(bundle, *, hostname: str | None = None, home: str | None = None) -> list[str]:
    """Every hit in the bundle's text files, as 'relative/path: what was found'."""
    hostname = socket.gethostname() if hostname is None else hostname
    home = os.path.expanduser("~") if home is None else home
    root = Path(bundle)
    hits = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name in SKIP_NAMES:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        hits += [f"{f.relative_to(root)}: {what}" for what in scrub_text(text, hostname=hostname, home=home)]
    return hits
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py`
Expected: PASS (11 tests).

- [ ] **Step 5: Mutation checks**

1. The operator ledger is skipped: in `orchard/scrub.py`, replace `SKIP_NAMES = frozenset({"ledger.jsonl"})` with `SKIP_NAMES = frozenset()`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py::test_the_bundle_scrub_names_the_file_and_skips_the_operator_ledger`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The hostname check: in `orchard/scrub.py`, replace `    if hostname and hostname != "localhost" and re.search(` with `    if False and re.search(`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_scrub.py::test_each_kind_of_leak_is_found`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/scrub.py tests/test_scrub.py
git commit -m "Add the bundle scrub: hostname, tokens and absolute home paths"
```

---

### Task 3: The stage table, the exit gates, tier and skill lookup, and the stage 0 comparison

**Files:**
- Create: `orchard/stages.py`, `tests/fixtures/hemmingway_stage0_reference.md` (a copy)
- Test: `tests/test_stages.py`

**Interfaces:**
- Consumes: `orchard.tiers.TierConfig`; `orchard.scrub.scrub_bundle` (Task 2); the plan 4 defaults (Task 1).
- Produces in `orchard.stages`:
  - `GateResult(ok: bool, reasons: tuple[str, ...] = (), evidence: tuple[str, ...] = ())`.
  - `StageSpec(number, name, skill, refs, boards, gate_file, gate, marker, skip=None)` with properties `budget_s`, `disk_gb`; `STAGES: tuple[StageSpec, ...]` for stages 0 to 8; `validate_table(stages=STAGES)`.
  - `evidence_record(run_dir, path) -> {"path": <run-relative>, "sha256": ...}`; `inside(run_dir, rel) -> Path | None`.
  - Gates `(stage_dir, run_dir) -> GateResult`: `gate_delta` (`delta.json`), `gate_reference` (`reference.json`), `gate_decoder`, `gate_full_model`, `gate_mesh`, `gate_serving`, `gate_numbers` (`result.json`), `gate_bundle` (`bundle/`). Constants `DELTA_AREAS`, `HAZARD_AREAS`, `PATHS`, `SERVING_CHECKS`, `BUNDLE_FILES`.
  - `TierUnavailable`; `tier_for(cfg, stage, *, phase: str, escalated: bool) -> str`; `resolve_endpoint(cfg, tier, probe) -> (tier_used, endpoint, note | None)`; `resolve_skill(name, dirs) -> Path | None`; `check_disk(path, need_gb, usage=shutil.disk_usage) -> (ok, free_gb)`.
  - `reference_items(md) -> (diff_areas, hazard_areas, path)`; `compare_delta(delta, reference_md) -> {"ok", "missing", "path", "reference_path", "path_matches"}`; `main(argv)` for `python3 -m orchard.stages compare-delta DELTA REFERENCE` (exit 0 when it matches, 1 otherwise).
- Phases used across tasks: `"run"` (a stage with no hardware), `"prepare"` and `"finish"` (around a hardware test).

- [ ] **Step 1: Copy the reference answer into the fixtures**

Run: `cp /mnt/bonus/models/hemmingway-1/work/stage0-reference.md tests/fixtures/hemmingway_stage0_reference.md && sha256sum tests/fixtures/hemmingway_stage0_reference.md`
Expected: the file exists. Record the sha256 in the task report. (Normal tests read no files outside the repo, so the test reads this copy. The copy holds model names, revisions and a cache path under `~`; it holds no token or hostname.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_stages.py`:

```python
"""The stage table, tier and skill lookup, the disk check, the exit gates and the stage 0 comparison."""
import dataclasses
import json
import os
import socket
from collections import namedtuple
from pathlib import Path

import pytest

from orchard.stages import (STAGES, TierUnavailable, check_disk, compare_delta, gate_bundle,
                            gate_decoder, gate_delta, gate_full_model, gate_mesh, gate_numbers,
                            gate_reference, gate_serving, main, reference_items, resolve_endpoint,
                            resolve_skill, tier_for, validate_table)
from orchard.tiers import TierConfig

REFERENCE = Path(__file__).parent / "fixtures" / "hemmingway_stage0_reference.md"


def cfg(escalation="large"):
    def tier(port, model, placement="chips"):
        return {"role": "r", "endpoint": f"http://127.0.0.1:{port}/v1", "model": model,
                "placement": placement}
    tiers = {"large": tier(8000, "Qwen/Qwen3.8-27B"), "small": tier(8001, "Qwen/Qwen3.8-27B"),
             "cpu": tier(11434, "qwen3-coder:30b", "cpu")}
    stages = {0: {"run": "large"}, 1: {"run": "small"}, 2: {"run": "small", "diagnose": "large"},
              3: {"run": "small", "diagnose": "large"},
              4: {"run": "small", "plan": "large", "diagnose": "large"},
              5: {"run": "small"}, 6: {"run": "small"}, 7: {"run": "none"}, 8: {"run": "small"}}
    return TierConfig(tiers, stages, escalation)


def write(root, rel, text="x"):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if isinstance(text, str) else json.dumps(text))
    return p


# ---- the table ----------------------------------------------------------------------------------

def test_the_table_names_the_owner_skills_and_hardware_stages():
    assert [s.skill for s in STAGES] == ["delta-triage", "reference-gate", "functional-decoder",
                                         "full-model", "mesh-shrink", "serving-check",
                                         "serving-check", "", "operator-bundle"]
    assert [s.boards for s in STAGES] == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert STAGES[7].skip and all(s.skip is None for s in STAGES if s.number != 7)


def test_a_long_stage_without_a_resume_marker_is_refused():
    table = list(STAGES)
    table[3] = dataclasses.replace(STAGES[3], marker=None)
    with pytest.raises(ValueError, match="resume marker"):
        validate_table(tuple(table))


def test_tier_for_runs_plans_and_escalates():
    c = cfg(escalation="cpu")
    assert tier_for(c, 1, phase="run", escalated=False) == "small"
    assert tier_for(c, 4, phase="prepare", escalated=False) == "large"     # large plans
    assert tier_for(c, 4, phase="finish", escalated=False) == "small"      # small runs
    assert tier_for(c, 2, phase="finish", escalated=True) == "large"       # the diagnose tier
    assert tier_for(c, 5, phase="finish", escalated=True) == "cpu"         # [escalation] default
    with pytest.raises(ValueError):
        tier_for(c, 7, phase="run", escalated=False)


def test_resolve_endpoint_falls_back_only_to_the_same_model():
    c = cfg()
    only_large = lambda endpoint, model: endpoint.endswith(":8000/v1")
    assert resolve_endpoint(c, "large", only_large) == ("large", "http://127.0.0.1:8000/v1", None)
    used, endpoint, note = resolve_endpoint(c, "small", only_large)
    assert (used, endpoint) == ("large", "http://127.0.0.1:8000/v1")
    assert "tier small is not serving" in note
    with pytest.raises(TierUnavailable):
        resolve_endpoint(c, "cpu", only_large)        # no other tier serves qwen3-coder:30b


def test_resolve_skill_finds_flat_files_and_plugin_folders(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "x.md")
    write(b, "y/SKILL.md")
    write(b, "x.md")
    assert resolve_skill("x", [a, b]) == a / "x.md"       # the first directory wins
    assert resolve_skill("y", [a, b]) == b / "y" / "SKILL.md"
    assert resolve_skill("z", [a, b]) is None


def test_check_disk_compares_free_space_with_the_need():
    usage = lambda path: namedtuple("U", "total used free")(0, 0, 5e9)
    assert check_disk(".", 4, usage=usage) == (True, 5.0)
    assert check_disk(".", 6, usage=usage) == (False, 5.0)


# ---- the gates ----------------------------------------------------------------------------------

GOOD_DELTA = {
    "model": "Altworld/Hemmingway-1", "nearest_model": "Qwen/Qwen3.8-27B", "path": "weights-only",
    "differences": [
        {"area": a, "finding": f"{a} checked", "evidence": ["stages/0/evidence/diff.txt"]}
        for a in ("config", "tensors", "tensor_names", "tokenizer", "files", "generation_config")],
    "hazards": [{"area": a, "finding": f"{a} hazard"} for a in ("tensor_cache", "drafter", "disk")],
}


def stage(tmp_path, n, name, data):
    run = tmp_path / "run"
    write(run, f"stages/{n}/evidence/diff.txt", "evidence")
    write(run, f"stages/{n}/{name}", data)
    return run / "stages" / str(n), run


def test_delta_gate_passes_a_complete_delta(tmp_path):
    g = gate_delta(*stage(tmp_path, 0, "delta.json", GOOD_DELTA))
    assert g.ok, g.reasons
    assert "stages/0/evidence/diff.txt" in g.evidence


def test_delta_gate_names_what_is_missing(tmp_path):
    bad = {**GOOD_DELTA, "path": "maybe",
           "differences": [{"area": "vibes", "finding": "x", "evidence": ["a"]}]}
    g = gate_delta(*stage(tmp_path, 0, "delta.json", bad))
    assert not g.ok
    assert any("path must be" in r for r in g.reasons)
    assert any("differences[0] needs an area" in r for r in g.reasons)
    assert not gate_delta(tmp_path / "nothing", tmp_path).ok        # no delta.json at all


def test_evidence_outside_the_run_directory_does_not_count(tmp_path):
    outside = write(tmp_path, "outside.txt")
    sd, run = stage(tmp_path, 0, "delta.json", GOOD_DELTA)
    os.symlink(outside, run / "stages/0/evidence/link.txt")
    for rel in (str(outside), "../outside.txt", "stages/0/evidence/link.txt", "stages/0/evidence"):
        delta = {**GOOD_DELTA, "differences": [{"area": "config", "finding": "x", "evidence": [rel]}]}
        write(run, "stages/0/delta.json", delta)
        g = gate_delta(sd, run)
        assert not g.ok and "not a file inside the run directory" in g.reasons[0], rel


def test_reference_gate_needs_a_pass_verdict_and_passing_checks(tmp_path):
    good = {"verdict": "pass", "checks": [{"name": "greedy matches the card", "pass": True,
                                           "evidence": ["stages/1/evidence/diff.txt"]}]}
    assert gate_reference(*stage(tmp_path, 1, "reference.json", good)).ok
    bad = {"verdict": "pass", "checks": [{"name": "decodes forward", "pass": False,
                                          "evidence": ["stages/1/evidence/diff.txt"]}]}
    g = gate_reference(*stage(tmp_path, 1, "reference.json", bad))
    assert not g.ok and "'decodes forward' did not pass" in g.reasons[0]


def test_decoder_gate_uses_the_functional_decoder_bar_and_not_the_agents(tmp_path):
    ev = ["stages/2/evidence/diff.txt"]
    ok = {"pcc": 0.995, "argmax_match": True, "evidence": ev}
    assert gate_decoder(*stage(tmp_path, 2, "result.json", ok)).ok
    low = {"pcc": 0.9949, "pcc_threshold": 0.9, "argmax_match": True, "evidence": ev}
    assert not gate_decoder(*stage(tmp_path, 2, "result.json", low)).ok


def test_full_model_gate_needs_parity_and_a_top1_fraction(tmp_path):
    ev = ["stages/3/evidence/diff.txt"]
    assert gate_full_model(*stage(tmp_path, 3, "result.json", {"parity": True, "top1": 0.97, "evidence": ev})).ok
    assert not gate_full_model(*stage(tmp_path, 3, "result.json", {"parity": True, "top1": 97, "evidence": ev})).ok


def test_mesh_gate_needs_every_configuration_to_pass(tmp_path):
    ev = ["stages/4/evidence/diff.txt"]
    data = {"configs": [{"chips": 2, "pass": True, "evidence": ev}, {"chips": 1, "pass": False, "evidence": ev}]}
    g = gate_mesh(*stage(tmp_path, 4, "result.json", data))
    assert not g.ok and g.reasons == ("the 1-chip configuration did not pass",)


def test_serving_gate_needs_boot_passkey_and_canary(tmp_path):
    ev = ["stages/5/evidence/diff.txt"]
    data = {"checks": {"boots": {"pass": True, "evidence": ev}, "canary": {"pass": True, "evidence": ev}}}
    g = gate_serving(*stage(tmp_path, 5, "result.json", data))
    assert g.reasons == ("checks.passkey is missing",)


def test_numbers_gate_needs_a_label_on_every_number(tmp_path):
    ev = ["stages/6/evidence/diff.txt"]
    good = {"numbers": [{"name": "decode", "value": 80.1, "unit": "tok/s/user", "label": "measured", "evidence": ev},
                        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
            "qualitative": {"evidence": ev}}
    assert gate_numbers(*stage(tmp_path, 6, "result.json", good)).ok
    guessed = {**good, "numbers": [{**good["numbers"][0], "label": "estimated"}]}
    assert not gate_numbers(*stage(tmp_path, 6, "result.json", guessed)).ok
    todo_only = {**good, "numbers": [good["numbers"][1]]}
    assert "no number is measured" in gate_numbers(*stage(tmp_path, 6, "result.json", todo_only)).reasons


def bundle(tmp_path, **files):
    b = tmp_path / "run" / "stages" / "8" / "bundle"
    b.mkdir(parents=True)
    base = {"RESULTS.md": "Results.", "RISKS.md": "Risks.",
            "PUBLISH_COMMANDS.txt": "tt-model push example/hemmingway-1-p300\n",
            "ledger.jsonl": "{}\n"}
    for name, text in {**base, **files}.items():
        if text is not None:
            (b / name).write_text(text)
    return b.parent, tmp_path / "run"


def test_a_complete_clean_bundle_passes_the_gate(tmp_path):
    g = gate_bundle(*bundle(tmp_path))
    assert g.ok, g.reasons
    assert "stages/8/bundle/PUBLISH_COMMANDS.txt" in g.evidence


def test_a_scrub_hit_or_a_missing_file_blocks_the_bundle(tmp_path):
    g = gate_bundle(*bundle(tmp_path, **{"RESULTS.md": f"Run on {socket.gethostname()}.",
                                         "PUBLISH_COMMANDS.txt": None}))
    assert not g.ok
    assert "bundle/PUBLISH_COMMANDS.txt is missing or empty" in g.reasons
    assert any(r.startswith("scrub: RESULTS.md: the hostname") for r in g.reasons)


# ---- stage 0 against the hand-written reference ---------------------------------------------------

def test_the_hemmingway_reference_is_read_item_by_item():
    diffs, hazards, path = reference_items(REFERENCE.read_text())
    assert diffs == ["config", "tensors", "tensor_names", "tokenizer", "files", "generation_config"]
    assert hazards == ["tensor_cache", "drafter", "disk"]
    assert path == "weights-only"


def test_a_delta_that_covers_the_reference_matches():
    assert compare_delta(GOOD_DELTA, REFERENCE.read_text())["ok"] is True


def test_a_delta_missing_an_item_or_with_the_wrong_path_does_not_match():
    delta = {**GOOD_DELTA, "path": "full-port",
             "differences": [d for d in GOOD_DELTA["differences"] if d["area"] != "tokenizer"],
             "hazards": GOOD_DELTA["hazards"][:2]}
    out = compare_delta(delta, REFERENCE.read_text())
    assert out["ok"] is False and out["path_matches"] is False
    assert out["missing"] == ["difference: tokenizer", "hazard: disk"]


def test_compare_delta_command_line(tmp_path, capsys):
    delta = write(tmp_path, "delta.json", GOOD_DELTA)
    assert main(["compare-delta", str(delta), str(REFERENCE)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    write(tmp_path, "delta.json", {**GOOD_DELTA, "path": "full-port"})
    assert main(["compare-delta", str(delta), str(REFERENCE)]) == 1
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.stages'`.

- [ ] **Step 4: Implement**

The import block already lists what Task 4 adds to this module (`calendar`, `time` and the run caps).

Create `orchard/stages.py`:

```python
"""The stage table, the exit gates and the replay that drives the stage machine (spec sections 5, 10).

This module owns what each stage is: its number, owner skill, the boards its hardware test needs,
its budget and free-disk need (orchard/defaults.py), the file its exit gate reads, the gate itself
and its resume marker. It also owns the questions the supervisor asks the ledger: which stage
runs next, whether the run is paused, how many escalations and coder starts it has used, and
where the coder's lease is recorded. Nothing here starts a process or calls a model.

A gate checks the shape of a stage's result file and that every evidence path it lists is a file
inside the run directory. A gate cannot tell whether a claim is true. The operator reviews the
bundle; the spec's `stage-review` skill is not wired in by plan 4.

Skills are referenced by name. `resolve_skill` finds `<dir>/<name>.md` or `<dir>/<name>/SKILL.md`
in the directories the run is given, so the existing tt-model-bringup skills are used where they
are installed and are not copied here.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard.defaults import (COLD_START_S, LONG_STAGE_S, RUN_COLD_BOOT_CAP, RUN_ESCALATION_CAP,
                              RUN_WALL_CLOCK_S, STAGE2_PCC_MIN, STAGE_BUDGET_S, STAGE_DISK_GB)
from orchard.tiers import TierConfig


class TierUnavailable(Exception):
    """No tier that can serve this stage answers."""


@dataclass(frozen=True)
class GateResult:
    ok: bool
    reasons: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()      # run-directory-relative paths the gate relied on


@dataclass(frozen=True)
class StageSpec:
    number: int
    name: str
    skill: str                          # owner skill, by name
    refs: tuple[str, ...]               # related existing skills, by name
    boards: int                         # boards the hardware test needs; 0 means no hardware phase
    gate_file: str | None               # the file in the stage directory the gate reads
    gate: Callable[[Path, Path], GateResult] | None
    marker: str | None                  # resume marker: a file in the stage directory
    skip: str | None = None             # why plan 4 skips this stage

    @property
    def budget_s(self) -> float:
        return STAGE_BUDGET_S[self.number]

    @property
    def disk_gb(self) -> float:
        return STAGE_DISK_GB[self.number]


# ---- evidence and gate helpers ------------------------------------------------------------------

def evidence_record(run_dir, path) -> dict:
    """The path (relative to the run directory) and sha256 that an evidence entry records.

    Relative paths keep the machine's home directory out of the ledger's evidence lists.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    rel = os.path.relpath(os.path.realpath(path), os.path.realpath(run_dir))
    return {"path": rel, "sha256": h.hexdigest()}


def inside(run_dir, rel) -> Path | None:
    """The file `rel` names, if it is a regular file inside the run directory (links resolved)."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel):
        return None
    root = os.path.realpath(run_dir)
    p = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, p]) != root or not os.path.isfile(p):
        return None
    return Path(p)


def _load(stage_dir, name: str) -> tuple[dict | None, str | None]:
    path = Path(stage_dir) / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{name} is missing"
    except (OSError, ValueError) as exc:
        return None, f"{name} is not readable JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{name} is not a JSON object"
    return data, None


def _evidence(run_dir, items, where: str, reasons: list, seen: list) -> None:
    if not isinstance(items, list) or not items:
        reasons.append(f"{where} lists no evidence")
        return
    for rel in items:
        if inside(run_dir, rel) is None:
            reasons.append(f"{where}: evidence {rel!r} is not a file inside the run directory")
        else:
            seen.append(rel)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _done(reasons: list, seen: list) -> GateResult:
    return GateResult(not reasons, tuple(reasons), tuple(seen))


# ---- the gates (one per stage) ------------------------------------------------------------------

DELTA_AREAS = ("config", "architecture", "tensors", "tensor_names", "tokenizer", "chat_template",
               "files", "generation_config", "license", "other")
HAZARD_AREAS = ("tensor_cache", "drafter", "disk", "license", "other")
PATHS = ("weights-only", "full-port")


def gate_delta(stage_dir, run_dir) -> GateResult:
    """Stage 0: a written list of what differs from the nearest supported model."""
    d, err = _load(stage_dir, "delta.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    for key in ("model", "nearest_model"):
        if not _text(d.get(key)):
            reasons.append(f"delta.json needs {key}")
    if d.get("path") not in PATHS:
        reasons.append(f"delta.json path must be one of {PATHS}")
    diffs = d.get("differences")
    if not isinstance(diffs, list) or not diffs:
        reasons.append("delta.json differences must be a non-empty list")
        diffs = []
    for i, item in enumerate(diffs):
        if not isinstance(item, dict) or item.get("area") not in DELTA_AREAS or not _text(item.get("finding")):
            reasons.append(f"differences[{i}] needs an area from {DELTA_AREAS} and a finding")
            continue
        _evidence(run_dir, item.get("evidence"), f"differences[{i}]", reasons, seen)
    hazards = d.get("hazards", [])
    if not isinstance(hazards, list):
        reasons.append("delta.json hazards must be a list")
        hazards = []
    for i, item in enumerate(hazards):
        if not isinstance(item, dict) or item.get("area") not in HAZARD_AREAS or not _text(item.get("finding")):
            reasons.append(f"hazards[{i}] needs an area from {HAZARD_AREAS} and a finding")
    return _done(reasons, seen)


def gate_reference(stage_dir, run_dir) -> GateResult:
    """Stage 1: the CPU reference reproduces the model card's published behavior."""
    d, err = _load(stage_dir, "reference.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if d.get("verdict") != "pass":
        reasons.append("reference.json verdict is not 'pass'")
    checks = d.get("checks")
    if not isinstance(checks, list) or not checks:
        reasons.append("reference.json checks must be a non-empty list")
        checks = []
    for i, c in enumerate(checks):
        if not isinstance(c, dict) or not _text(c.get("name")):
            reasons.append(f"checks[{i}] needs a name")
            continue
        if c.get("pass") is not True:
            reasons.append(f"check {c['name']!r} did not pass")
        _evidence(run_dir, c.get("evidence"), f"check {c['name']!r}", reasons, seen)
    return _done(reasons, seen)


def gate_decoder(stage_dir, run_dir) -> GateResult:
    """Stage 2: PCC at or above the functional-decoder bar, and argmax agreement."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if not _number(d.get("pcc")) or d["pcc"] < STAGE2_PCC_MIN:
        reasons.append(f"pcc must be a number at or above {STAGE2_PCC_MIN}, got {d.get('pcc')!r}")
    if d.get("argmax_match") is not True:
        reasons.append("argmax_match is not true")
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def gate_full_model(stage_dir, run_dir) -> GateResult:
    """Stage 3: end-to-end parity with the reference, with the top-1 agreement recorded."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    if d.get("parity") is not True:
        reasons.append("parity is not true")
    if not _number(d.get("top1")) or not 0 <= d["top1"] <= 1:
        reasons.append(f"top1 must be a fraction from 0 to 1, got {d.get('top1')!r}")
    _evidence(run_dir, d.get("evidence"), "result.json", reasons, seen)
    return _done(reasons, seen)


def gate_mesh(stage_dir, run_dir) -> GateResult:
    """Stage 4: evidence for each mesh configuration, as mesh-shrink requires."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    configs = d.get("configs")
    if not isinstance(configs, list) or not configs:
        reasons.append("result.json configs must be a non-empty list")
        configs = []
    for i, c in enumerate(configs):
        if not isinstance(c, dict) or isinstance(c.get("chips"), bool) or not isinstance(c.get("chips"), int):
            reasons.append(f"configs[{i}] needs an integer chips")
            continue
        if c.get("pass") is not True:
            reasons.append(f"the {c['chips']}-chip configuration did not pass")
        _evidence(run_dir, c.get("evidence"), f"configs[{i}]", reasons, seen)
    return _done(reasons, seen)


SERVING_CHECKS = ("boots", "passkey", "canary")


def gate_serving(stage_dir, run_dir) -> GateResult:
    """Stage 5: the black-box server checks pass (boot, passkey or needle, canary)."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    checks = d.get("checks") if isinstance(d.get("checks"), dict) else {}
    for name in SERVING_CHECKS:
        c = checks.get(name)
        if not isinstance(c, dict):
            reasons.append(f"checks.{name} is missing")
            continue
        if c.get("pass") is not True:
            reasons.append(f"checks.{name} did not pass")
        _evidence(run_dir, c.get("evidence"), f"checks.{name}", reasons, seen)
    return _done(reasons, seen)


def gate_numbers(stage_dir, run_dir) -> GateResult:
    """Stage 6: every number labelled measured or TODO; measured ones carry evidence."""
    d, err = _load(stage_dir, "result.json")
    if err:
        return GateResult(False, (err,))
    reasons, seen = [], []
    numbers = d.get("numbers")
    if not isinstance(numbers, list) or not numbers:
        reasons.append("result.json numbers must be a non-empty list")
        numbers = []
    measured = 0
    for i, n in enumerate(numbers):
        if not isinstance(n, dict) or not _text(n.get("name")) or not _text(n.get("unit")):
            reasons.append(f"numbers[{i}] needs a name and a unit")
            continue
        if n.get("label") == "measured":
            measured += 1
            if not _number(n.get("value")):
                reasons.append(f"number {n['name']!r} is labelled measured and has no numeric value")
            _evidence(run_dir, n.get("evidence"), f"number {n['name']!r}", reasons, seen)
        elif n.get("label") != "TODO":
            reasons.append(f"number {n['name']!r} must be labelled 'measured' or 'TODO'")
    if numbers and not measured:
        reasons.append("no number is measured")
    qual = d.get("qualitative")
    if not isinstance(qual, dict):
        reasons.append("result.json qualitative is missing")
    else:
        _evidence(run_dir, qual.get("evidence"), "qualitative", reasons, seen)
    return _done(reasons, seen)


BUNDLE_FILES = ("RESULTS.md", "RISKS.md", "PUBLISH_COMMANDS.txt", "ledger.jsonl")


def gate_bundle(stage_dir, run_dir) -> GateResult:
    """Stage 8: results, ledger, open risks, publish commands as text, and a clean scrub."""
    from orchard.scrub import scrub_bundle
    bundle = Path(stage_dir) / "bundle"
    reasons, seen = [], []
    for name in BUNDLE_FILES:
        f = bundle / name
        if not f.is_file() or not f.read_text(encoding="utf-8", errors="replace").strip():
            reasons.append(f"bundle/{name} is missing or empty")
        else:
            seen.append(os.path.relpath(f, run_dir))
    for hit in scrub_bundle(bundle):
        reasons.append(f"scrub: {hit}")
    return _done(reasons, seen)


SKIP_7 = ("plan 4 builds no package or container image; the operator bundle reports the "
          "package as not built")

STAGES: tuple[StageSpec, ...] = (
    StageSpec(0, "intake and delta triage", "delta-triage", ("model-bringup",), 0,
              "delta.json", gate_delta, "RESUME.md"),
    StageSpec(1, "environment and CPU reference", "reference-gate", ("model-bringup",), 0,
              "reference.json", gate_reference, "RESUME.md"),
    StageSpec(2, "functional decoder on one chip", "functional-decoder", ("tt-device-usage",), 1,
              "result.json", gate_decoder, "test-result.json"),
    StageSpec(3, "full model", "full-model", ("tt-device-usage",), 1,
              "result.json", gate_full_model, "test-result.json"),
    StageSpec(4, "multichip, then shrink to fewer chips", "mesh-shrink",
              ("multichip", "tt-device-usage"), 1, "result.json", gate_mesh, "test-result.json"),
    StageSpec(5, "serving integration", "serving-check", ("vllm-integration", "tt-device-usage"), 1,
              "result.json", gate_serving, "test-result.json"),
    StageSpec(6, "qualitative check and benchmark", "serving-check",
              ("qualitative-check", "benchmark-model"), 1, "result.json", gate_numbers,
              "test-result.json"),
    StageSpec(7, "package and container build", "", (), 0, None, None, None, skip=SKIP_7),
    StageSpec(8, "operator bundle", "operator-bundle", (), 0, "bundle/RESULTS.md", gate_bundle, None),
)


def validate_table(stages=STAGES) -> None:
    """Refuse a table the spec forbids: gaps, and long stages with no resume marker."""
    if [s.number for s in stages] != list(range(9)):
        raise ValueError("the stage table must list stages 0 to 8 in order")
    for s in stages:
        if s.skip:
            continue
        if s.gate is None or not s.gate_file or not s.skill:
            raise ValueError(f"stage {s.number} needs a skill, a gate file and a gate")
        if s.budget_s > LONG_STAGE_S and not s.marker:
            raise ValueError(f"stage {s.number} has a budget of {s.budget_s} s and declares no "
                             "resume marker (spec section 10)")


validate_table()


# ---- tiers, skills and disk ---------------------------------------------------------------------

def tier_for(cfg: TierConfig, stage: int, *, phase: str, escalated: bool) -> str:
    """The tier for one agent step. Escalated: the stage's diagnose tier, else [escalation] default.

    A stage with a `plan` tier (stage 4) prepares with it: large plans, small runs.
    """
    entry = cfg.stages[stage]
    if entry["run"] == "none":
        raise ValueError(f"stage {stage} loads no model")
    if escalated:
        return entry.get("diagnose", cfg.escalation)
    if phase == "prepare" and "plan" in entry:
        return entry["plan"]
    return entry["run"]


def resolve_endpoint(cfg: TierConfig, tier: str, probe) -> tuple[str, str, str | None]:
    """(tier used, endpoint, note). `probe(endpoint, model)` says whether a server answers.

    The large and small tiers can name the same model on different chip counts. Only one of them
    can be loaded at a time on this box, so when the named tier is down and another chip tier with
    the same model answers, that one serves the step and the note says so.
    """
    t = cfg.tiers[tier]
    if probe(t["endpoint"], t["model"]):
        return tier, t["endpoint"], None
    for name, other in cfg.tiers.items():
        if (name != tier and other["model"] == t["model"] and other["placement"] == t["placement"]
                and probe(other["endpoint"], other["model"])):
            return name, other["endpoint"], (f"tier {tier} is not serving; tier {name} serves the "
                                             f"same model {t['model']}")
    raise TierUnavailable(f"tier {tier} ({t['model']} at {t['endpoint']}) is not serving")


def resolve_skill(name: str, dirs) -> Path | None:
    for d in dirs:
        for candidate in (Path(d) / f"{name}.md", Path(d) / name / "SKILL.md"):
            if candidate.is_file():
                return candidate
    return None


def check_disk(path, need_gb: float, usage=shutil.disk_usage) -> tuple[bool, float]:
    free_gb = usage(path).free / 1e9
    return free_gb >= need_gb, round(free_gb, 1)


# ---- stage 0 compared with a reference answer ---------------------------------------------------

# Labels used in the hand-written Hemmingway-1 reference (`N. Label: ...`) and the delta area each
# one is about. A label not listed here makes the comparison refuse, so a new reference cannot be
# half-read.
REFERENCE_AREAS = {"text config": "config", "tensors": "tensors", "tensor name prefix": "tensor_names",
                   "tokenizer": "tokenizer", "files": "files", "generation config": "generation_config"}
# Hazard lines are matched by word, in this order ("disk" first: that line also says "caches").
HAZARD_WORDS = (("disk", "disk"), ("drafter", "drafter"), ("cache", "tensor_cache"))


def reference_items(md: str) -> tuple[list[str], list[str], str | None]:
    """The difference areas, hazard areas and expected path a reference answer lists."""
    diffs, hazards, path, section = [], [], None, None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("Differences from"):
            section = "diff"
        elif s.startswith("Hazards"):
            section = "hazard"
        elif s.startswith("Expected path:"):
            path = "weights-only" if "weights-only" in s else "full-port"
            section = None
        elif section == "diff" and (m := re.match(r"\d+\.\s+([^:]+):", s)):
            label = m.group(1).strip().lower()
            if label not in REFERENCE_AREAS:
                raise ValueError(f"reference item {label!r} has no delta area")
            diffs.append(REFERENCE_AREAS[label])
        elif section == "hazard" and s.startswith("- "):
            area = next((a for word, a in HAZARD_WORDS if word in s.lower()), None)
            if area is None:
                raise ValueError(f"reference hazard {s[:60]!r} has no hazard area")
            hazards.append(area)
    return diffs, hazards, path


def compare_delta(delta: dict, reference_md: str) -> dict:
    """Does a stage 0 delta cover every area the reference lists, and agree on the path?

    This compares areas and the path only. Whether each finding is right is for a person.
    """
    want_d, want_h, want_path = reference_items(reference_md)
    have_d = {i.get("area") for i in delta.get("differences") or [] if isinstance(i, dict)}
    have_h = {i.get("area") for i in delta.get("hazards") or [] if isinstance(i, dict)}
    missing = ([f"difference: {a}" for a in want_d if a not in have_d]
               + [f"hazard: {a}" for a in want_h if a not in have_h])
    matches = delta.get("path") == want_path
    return {"ok": not missing and matches, "missing": missing, "path": delta.get("path"),
            "reference_path": want_path, "path_matches": matches}


# ---- command line: compare a stage 0 delta with a reference answer ------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.stages")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare-delta", help="compare a stage 0 delta.json with a reference answer")
    c.add_argument("delta")
    c.add_argument("reference")
    a = p.parse_args(argv)
    delta = json.loads(Path(a.delta).read_text(encoding="utf-8"))
    out = compare_delta(delta, Path(a.reference).read_text(encoding="utf-8"))
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py`
Expected: PASS (21 tests).

- [ ] **Step 6: Mutation checks**

1. Evidence must be inside the run directory: in `orchard/stages.py`, replace `    if os.path.commonpath([root, p]) != root or not os.path.isfile(p):` with `    if not os.path.isfile(p):`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_evidence_outside_the_run_directory_does_not_count`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The PCC bar is the table's: in `orchard/stages.py`, replace `d["pcc"] < STAGE2_PCC_MIN` with `d["pcc"] < d.get("pcc_threshold", STAGE2_PCC_MIN)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_decoder_gate_uses_the_functional_decoder_bar_and_not_the_agents`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. Long stages declare a resume marker: in `orchard/stages.py`, replace `        if s.budget_s > LONG_STAGE_S and not s.marker:` with `        if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_a_long_stage_without_a_resume_marker_is_refused`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. The hazard word order: in `orchard/stages.py`, replace `HAZARD_WORDS = (("disk", "disk"), ("drafter", "drafter"), ("cache", "tensor_cache"))` with `HAZARD_WORDS = (("cache", "tensor_cache"), ("disk", "disk"), ("drafter", "drafter"))`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_the_hemmingway_reference_is_read_item_by_item`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. A substitute tier serves the same model: in `orchard/stages.py`, replace `if (name != tier and other["model"] == t["model"] and other["placement"] == t["placement"]` with `if (name != tier`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_resolve_endpoint_falls_back_only_to_the_same_model`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
6. The bundle gate runs the scrub: in `orchard/stages.py`, replace `    for hit in scrub_bundle(bundle):` with `    for hit in []:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stages.py::test_a_scrub_hit_or_a_missing_file_blocks_the_bundle`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/stages.py tests/test_stages.py tests/fixtures/hemmingway_stage0_reference.md
git commit -m "Add the stage table, exit gates, tier and skill lookup, and the stage 0 comparison"
```

---

### Task 4: Replaying the ledger: next stage, pauses, caps, the coder's lease, stage directories

**Files:**
- Modify: `orchard/stages.py` (insert a section)
- Test: `tests/test_stage_progress.py`

**Interfaces:**
- Consumes: `STAGES`, `evidence_record` (Task 3); a `Ledger`.
- Produces in `orchard.stages`:
  - `RunProgress(started, run_start, clock_start, finished, aborted, paused, done, open_stage, escalated, escalations, cold_boots, coder_deaths, next_stage)`; `run_progress(entries) -> RunProgress`. Decisions it reads: `pause` (with `reason`), `resume`, `abort`, `ready for operator review`, `coder started` (with `ready_s`), `coder died; restarting it once`. A `resume` resets the caps and restarts the wall clock.
  - `ledger_ts(entry) -> float`; `attempt_started_ts(entries, stage) -> float | None`.
  - `budget_cap(progress, now) -> str | None`.
  - `coder_state(entries) -> (lease_record | None, server_record | None, baseline_canary | None)`, read from park and restore entries and the `coder starting` and `coder started` decisions.
  - `open_stage_dir(run_dir, spec, *, resuming: bool, ledger) -> (stage_dir, resumed_from_marker)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage_progress.py`:

```python
"""Replaying the ledger: the next stage, pauses, escalations, caps, the coder's lease, stage dirs."""
import calendar
import time

import pytest

from orchard.ledger import Ledger
from orchard.stages import (STAGES, attempt_started_ts, budget_cap, coder_state, open_stage_dir,
                            run_progress)

T0 = calendar.timegm(time.strptime("2026-10-02T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"))


def entries(*rows):
    """Ledger-shaped dicts. Each row: (event, stage, seconds after T0, data)."""
    return [{"seq": i, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(T0 + t)),
             "event": ev, "stage": st, "data": d} for i, (ev, st, t, d) in enumerate(rows, 1)]


START = ("run_start", None, 0, {"model": "Altworld/Hemmingway-1"})


def test_a_stage_with_no_end_is_open_and_runs_next():
    p = run_progress(entries(START, ("stage_start", 0, 1, {}), ("stage_end", 0, 2, {"result": "pass"}),
                             ("stage_start", 1, 3, {})))
    assert p.started and p.done == (0,) and p.open_stage == 1 and p.next_stage == 1
    assert p.clock_start == T0


def test_a_skipped_stage_counts_as_done():
    rows = [START] + [r for n in range(7) for r in (("stage_start", n, 1, {}),
                                                    ("stage_end", n, 2, {"result": "pass"}))]
    rows += [("stage_start", 7, 3, {"skip": True}), ("stage_end", 7, 3, {"result": "skipped"})]
    assert run_progress(entries(*rows)).next_stage == 8


def test_an_escalated_stage_runs_again_and_is_counted():
    passed = [r for n in (0, 1) for r in (("stage_start", n, 1, {}), ("stage_end", n, 1, {"result": "pass"}))]
    p = run_progress(entries(START, *passed, ("stage_start", 2, 1, {}),
                             ("escalate", 2, 2, {"by": "stage machine"}),
                             ("stage_end", 2, 2, {"result": "escalate"})))
    assert p.next_stage == 2 and p.open_stage is None
    assert p.escalated == {2} and p.escalations == 1


def test_a_pause_lasts_until_a_resume_and_the_resume_resets_the_caps():
    rows = [START, ("escalate", 2, 1, {}), ("restore", 2, 2, {"step": "ready", "seconds": 1800.0}),
            ("restore", 3, 2, {"step": "ready", "seconds": 150.0}),     # a warm restart
            ("decision", None, 2, {"decision": "coder died; restarting it once"}),
            ("decision", 2, 3, {"decision": "pause", "reason": "watchdog: loop"})]
    p = run_progress(entries(*rows))
    assert p.paused == "watchdog: loop" and p.escalations == 1 and p.cold_boots == 1
    assert p.coder_deaths == 1
    p = run_progress(entries(*rows, ("decision", None, 4, {"decision": "resume"})))
    assert p.paused is None and (p.escalations, p.cold_boots, p.coder_deaths) == (0, 0, 0)
    assert p.clock_start == T0 + 4


def test_abort_and_ready_are_read_from_decisions():
    assert run_progress(entries(START, ("decision", None, 1, {"decision": "abort"}))).aborted
    assert run_progress(entries(START, ("decision", None, 1, {"decision": "ready for operator review"}))).finished


def test_a_restart_keeps_the_attempts_clock():
    rows = entries(START, ("stage_start", 3, 100, {}), ("stage_start", 3, 500, {"resumed": True}))
    assert attempt_started_ts(rows, 3) == T0 + 100
    rows = entries(START, ("stage_start", 3, 100, {}), ("stage_end", 3, 200, {"result": "escalate"}),
                   ("stage_start", 3, 300, {}))
    assert attempt_started_ts(rows, 3) == T0 + 300
    assert attempt_started_ts(rows, 4) is None


@pytest.mark.parametrize("rows, words", [
    ([("escalate", 2, 1, {})] * 3, "3 escalations"),
    ([("decision", None, 1, {"decision": "coder started", "ready_s": 1800.0})] * 3, "3 cold coder boots"),
])
def test_budget_caps_pause_the_run(rows, words):
    assert words in budget_cap(run_progress(entries(START, *rows)), now=T0 + 10)


def test_the_wall_clock_cap_counts_from_run_start_or_the_last_resume():
    p = run_progress(entries(START))
    assert budget_cap(p, now=T0 + 3600) is None
    assert "the run has lasted" in budget_cap(p, now=T0 + 72 * 3600)
    # Without the restart at a resume, a resumed run would pause again at once, forever.
    p = run_progress(entries(START, ("decision", None, 72 * 3600, {"decision": "resume"})))
    assert budget_cap(p, now=T0 + 72 * 3600 + 60) is None


def test_coder_state_reads_the_latest_lease_and_the_baseline_canary():
    lease1, lease2 = {"lease_id": "L1"}, {"lease_id": "L2"}
    rows = entries(START,
                   ("decision", None, 1, {"decision": "coder starting", "lease": lease1, "server": {"port": 8000}}),
                   ("decision", None, 2, {"decision": "coder started", "lease": lease1, "canary": {"path": "c1"}}),
                   ("decision", 2, 3, {"decision": "test lease taken", "test_lease": {"lease_id": "T9"}}),
                   ("restore", 2, 4, {"step": "serve", "lease": lease2, "server": {"port": 8000, "pid": 7}}))
    assert coder_state(rows) == (lease2, {"port": 8000, "pid": 7}, {"path": "c1"})
    assert coder_state(entries(START)) == (None, None, None)


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield led


def test_a_partial_stage_directory_is_moved_aside_and_never_deleted(tmp_path, ledger):
    run = tmp_path / "run"
    d, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)
    assert (d / "evidence").is_dir() and not resumed
    (d / "evidence" / "work.txt").write_text("half done")
    d, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)  # no marker yet
    assert not resumed and not (d / "evidence" / "work.txt").exists()
    assert (run / "stages" / "2.partial-1" / "evidence" / "work.txt").read_text() == "half done"
    open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    assert (run / "stages" / "2.partial-2").is_dir()
    moved = [e["data"]["path"] for e in ledger.read() if e["event"] == "decision"]
    assert moved == ["stages/2.partial-1", "stages/2.partial-2"]


def test_a_resumed_stage_with_its_marker_keeps_its_directory(tmp_path, ledger):
    run = tmp_path / "run"
    d, _ = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    (d / "test-result.json").write_text("{}")
    again, resumed = open_stage_dir(run, STAGES[2], resuming=True, ledger=ledger)
    assert again == d and resumed and (d / "test-result.json").exists()
    entry = ledger.read()[-1]["data"]
    assert entry["decision"] == "resume from marker"
    assert entry["marker"]["path"] == "stages/2/test-result.json" and len(entry["marker"]["sha256"]) == 64
    # A new attempt (after an escalation) never resumes, marker or not.
    _, resumed = open_stage_dir(run, STAGES[2], resuming=False, ledger=ledger)
    assert not resumed and (run / "stages" / "2.partial-1" / "test-result.json").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py`
Expected: FAIL with `ImportError: cannot import name 'attempt_started_ts' from 'orchard.stages'`.

- [ ] **Step 3: Implement**

Insert into `orchard/stages.py` immediately above the line that starts `# ---- command line: compare a stage 0 delta`, followed by two blank lines:

```python
# ---- replaying the ledger -----------------------------------------------------------------------

@dataclass(frozen=True)
class RunProgress:
    started: bool
    run_start: dict | None
    clock_start: float | None           # run start, and again at each operator resume
    finished: bool                      # decision "ready for operator review"
    aborted: bool
    paused: str | None                  # the reason, while the latest pause has no resume after it
    done: tuple[int, ...]               # stages that passed or were skipped
    open_stage: int | None              # a stage_start with no stage_end after it
    escalated: frozenset[int]           # stages with an escalate entry
    escalations: int                    # since the last operator resume
    cold_boots: int                     # coder starts slower than COLD_START_S, since the last resume
    coder_deaths: int                   # since the last operator resume
    next_stage: int | None


def ledger_ts(entry: dict) -> float:
    return float(calendar.timegm(time.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%SZ")))


def run_progress(entries: list[dict]) -> RunProgress:
    run_start = clock_start = None
    finished = aborted = False
    paused = None
    done: list[int] = []
    open_stage = None
    escalated: set[int] = set()
    escalations = cold_boots = coder_deaths = 0
    for e in entries:
        ev, stage, d = e["event"], e["stage"], e["data"]
        if ev == "run_start":
            run_start, clock_start = d, ledger_ts(e)
        elif ev == "stage_start":
            open_stage = stage
        elif ev == "stage_end":
            if stage == open_stage:
                open_stage = None
            if d.get("result") in ("pass", "skipped") and stage not in done:
                done.append(stage)
        elif ev == "escalate":
            escalated.add(stage)
            escalations += 1
        elif ev == "restore" and d.get("step") == "ready" and (d.get("seconds") or 0) >= COLD_START_S:
            cold_boots += 1
        elif ev == "decision":
            what = d.get("decision")
            if what == "pause":
                paused = d.get("reason") or "paused"
            elif what == "resume":
                paused, escalations, cold_boots, coder_deaths = None, 0, 0, 0
                clock_start = ledger_ts(e)
            elif what == "abort":
                aborted = True
            elif what == "ready for operator review":
                finished = True
            elif what == "coder started" and (d.get("ready_s") or 0) >= COLD_START_S:
                cold_boots += 1
            elif what == "coder died; restarting it once":
                coder_deaths += 1
    next_stage = next((s.number for s in STAGES if s.number not in done), None)
    return RunProgress(run_start is not None, run_start, clock_start, finished, aborted, paused,
                       tuple(done), open_stage, frozenset(escalated), escalations, cold_boots,
                       coder_deaths, next_stage)


def attempt_started_ts(entries: list[dict], stage: int) -> float | None:
    """When the current attempt at `stage` first started: the first stage_start after its last
    stage_end. A restart keeps the attempt's clock, so a crash does not reset the budget."""
    t = None
    for e in entries:
        if e["stage"] != stage:
            continue
        if e["event"] == "stage_end":
            t = None
        elif e["event"] == "stage_start" and t is None:
            t = ledger_ts(e)
    return t


def budget_cap(p: RunProgress, now: float) -> str | None:
    """Spec section 10: a run-wide cap that pauses the run, or None."""
    if p.escalations >= RUN_ESCALATION_CAP:
        return f"{p.escalations} escalations since the last resume (cap {RUN_ESCALATION_CAP})"
    if p.cold_boots >= RUN_COLD_BOOT_CAP:
        return f"{p.cold_boots} cold coder boots since the last resume (cap {RUN_COLD_BOOT_CAP})"
    if p.clock_start is not None and now - p.clock_start >= RUN_WALL_CLOCK_S:
        return (f"the run has lasted {int(now - p.clock_start)} s since it started or was last "
                f"resumed (cap {int(RUN_WALL_CLOCK_S)} s)")
    return None


def coder_state(entries: list[dict]) -> tuple[dict | None, dict | None, dict | None]:
    """The coder's latest lease record, server record and baseline canary, from the ledger.

    Park and restore entries carry the lease (orchard/handoff.py). The supervisor's own
    "coder starting" and "coder started" decisions carry it too.
    """
    lease = server = canary = None
    for e in entries:
        d = e["data"]
        own = e["event"] == "decision" and d.get("decision") in ("coder starting", "coder started")
        if e["event"] in ("park", "restore") or own:
            if isinstance(d.get("lease"), dict):
                lease = d["lease"]
            if isinstance(d.get("server"), dict):
                server = d["server"]
            if own and isinstance(d.get("canary"), dict):
                canary = d["canary"]
    return lease, server, canary


def open_stage_dir(run_dir, spec: StageSpec, *, resuming: bool, ledger) -> tuple[Path, bool]:
    """The stage directory for this attempt, and whether it resumes from the stage's marker.

    A resumed stage with its marker present keeps its directory. Otherwise an existing directory
    is moved aside to `<n>.partial-<k>` (never deleted) and a fresh one is made (spec section 10).
    """
    d = Path(run_dir) / "stages" / str(spec.number)
    if d.exists():
        marker = d / spec.marker if spec.marker else None
        if resuming and marker is not None and marker.is_file():
            ledger.append("decision", spec.number, decision="resume from marker",
                          marker=evidence_record(run_dir, marker))
            return d, True
        k = 1
        while (aside := d.with_name(f"{spec.number}.partial-{k}")).exists():
            k += 1
        os.rename(d, aside)
        ledger.append("decision", spec.number, decision="moved the partial stage directory aside",
                      path=os.path.relpath(aside, run_dir))
    (d / "evidence").mkdir(parents=True)
    return d, False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py tests/test_stages.py`
Expected: PASS (33 tests).

- [ ] **Step 5: Mutation checks**

1. A partial directory is moved aside: in `orchard/stages.py`, replace `        os.rename(d, aside)` with `        shutil.rmtree(d)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py::test_a_partial_stage_directory_is_moved_aside_and_never_deleted`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. A restart keeps the attempt's clock: in `orchard/stages.py`, replace `        elif e["event"] == "stage_start" and t is None:` with `        elif e["event"] == "stage_start":`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py::test_a_restart_keeps_the_attempts_clock`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A resume restarts the wall clock: in `orchard/stages.py`, replace `                clock_start = ledger_ts(e)` with `                pass`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py::test_the_wall_clock_cap_counts_from_run_start_or_the_last_resume`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. Only a resumed attempt keeps its directory: in `orchard/stages.py`, replace `        if resuming and marker is not None and marker.is_file():` with `        if marker is not None and marker.is_file():`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_stage_progress.py::test_a_resumed_stage_with_its_marker_keeps_its_directory`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/stages.py tests/test_stage_progress.py
git commit -m "Replay the ledger for the stage machine; move partial stage directories aside"
```

---

### Task 5: The agent's environment and tools

**Files:**
- Create: `orchard/agent.py` (first part)
- Test: `tests/test_agent_env.py`

**Interfaces:**
- Consumes: `orchard.runner.check_string`, `Denied`; `evidence_record` (Task 3).
- Produces in `orchard.agent`:
  - `ENV_ALLOW`, `SECRET_NAME`; `agent_env(run_dir, *, extra: dict | None = None, source=None) -> dict` (raises `ValueError` for an extra name that looks like a credential).
  - `spawn_checked(command, run_dir, env, timeout, stdout) -> (exit_code | None, timed_out)` (raises `Denied`; nothing starts when refused).
  - `clip(text, limit) -> str`; `TOOL_SCHEMAS`; `Tools(run_dir, stage_dir, env, *, timeout=TOOL_TIMEOUT_S, limit=TOOL_OUTPUT_CHARS)` with `call(name, arguments_json) -> str`, `shell(command) -> str` (first line `exit N`, `killed after T s` or `refused: ...`), `write_file(path, content) -> str`, `target(path) -> Path | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_env.py`:

```python
"""The agent's environment and tools: no tokens, checked commands, writes only in the stage dir."""
import os
import time

import pytest

from orchard.agent import Tools, agent_env

TOKENS = {"HF_TOKEN": "hf_supervisor_secret", "HUGGING_FACE_HUB_TOKEN": "hf_other_secret",
          "GH_TOKEN": "ghp_supervisor_secret", "GITHUB_TOKEN": "ghs_supervisor_secret",
          "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock", "AWS_SECRET_ACCESS_KEY": "aws_secret"}


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    return tmp_path / "run"


def tools(run, **kw):
    return Tools(run, run / "stages" / "0", agent_env(run), **kw)


def test_the_environment_is_an_allow_list_with_a_home_inside_the_run(run):
    env = agent_env(run, source={"PATH": "/usr/bin", "HOME": "/home/alice", "LANG": "C.UTF-8", **TOKENS})
    assert not set(TOKENS) & set(env)
    assert env["PATH"] == "/usr/bin" and env["LANG"] == "C.UTF-8"
    assert env["HOME"] == str(run.resolve() / "home") and os.path.isdir(env["HOME"])
    assert env["HF_TOKEN_PATH"].startswith(env["HOME"])


def test_extra_variables_that_look_like_credentials_are_refused(run):
    assert agent_env(run, extra={"HF_HOME": "/mnt/models/hf"})["HF_HOME"] == "/mnt/models/hf"
    for name in ("HF_TOKEN", "OPENAI_API_KEY", "GH_AUTH", "DB_PASSWORD"):
        with pytest.raises(ValueError, match="credential"):
            agent_env(run, extra={name: "x"})


def test_a_shell_command_finds_no_token_in_its_environment_or_home(run, tmp_path, monkeypatch):
    for k, v in TOKENS.items():
        monkeypatch.setenv(k, v)
    real_home = tmp_path / "real-home"
    (real_home / ".cache" / "huggingface").mkdir(parents=True)
    (real_home / ".cache" / "huggingface" / "token").write_text("hf_planted_secret")
    monkeypatch.setenv("HOME", str(real_home))
    t = tools(run)
    out = t.shell("env")
    assert out.startswith("exit 0") and "secret" not in out and "ssh-agent" not in out
    out = t.shell("cat ~/.cache/huggingface/token")
    assert out.startswith("exit 1") and "hf_planted_secret" not in out


def test_a_refused_command_never_starts(run):
    t = tools(run)
    out = t.shell("touch started.txt && git push origin main")
    assert out.startswith("refused:") and "git push" in out
    assert not (run / "started.txt").exists()
    assert t.shell("rm -f /nonexistent-orchard-dir/x").startswith("refused:")


def test_shell_runs_in_the_run_directory(run):
    assert os.path.realpath(run) in tools(run).shell("pwd")


def test_a_command_past_its_timeout_is_killed(run):
    t0 = time.monotonic()
    out = tools(run, timeout=0.5).shell("sleep 30")
    assert out.startswith("killed after 0.5 s") and time.monotonic() - t0 < 10


def test_long_output_keeps_its_head_and_tail(run):
    out = tools(run, limit=100).shell("seq 1 5000")
    assert out.splitlines()[1] == "1" and out.splitlines()[-1] == "5000" and "characters cut" in out


def test_write_file_stays_inside_the_stage_directory(run, tmp_path):
    t = tools(run)
    assert t.write_file("evidence/a.txt", "x").startswith("wrote 1 characters to stages/0/evidence/a.txt")
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", run / "stages" / "0" / "link")
    for path in ("../escape.txt", str(tmp_path / "abs.txt"), "link/x.txt", "", "."):
        assert t.write_file(path, "x").startswith("refused:"), path
    assert not list((tmp_path / "outside").iterdir())
    assert not (run / "stages" / "escape.txt").exists()
    assert t.write_file("ledger.jsonl", "{}").startswith("refused: the ledger")


def test_bad_tool_calls_get_an_answer_and_raise_nothing(run):
    t = tools(run)
    assert t.call("shell", "not json").startswith("error:")
    assert t.call("shell", "[1]").startswith("error:")
    assert t.call("browse", "{}").startswith("error: there is no tool")
    assert t.call("shell", '{"command": ""}').startswith("error:")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.agent'`.

- [ ] **Step 3: Implement**

The import block already lists what Task 6 adds to this module.

Create `orchard/agent.py`:

```python
"""Agent steps: one fresh model context per stage phase, and the tool-call loop that serves it.

This module owns three things.

1. The agent's environment (`agent_env`). It starts from an allow-list, so GitHub and Hugging
   Face tokens, SSH agent sockets and every other variable the supervisor happens to have are
   absent. HOME points to an empty directory inside the run, so `gh`, `git` and `huggingface_hub`
   find no stored credentials under the usual paths. Extra variables the operator passes are
   refused when the name looks like a credential.
2. The tools (`Tools`): `shell`, which runs every command through orchard/runner.py's checks with
   the run directory as its working directory, and `write_file`, which writes only inside the
   stage directory.
3. The loop (`AgentStep`): it sends the conversation to a tier's OpenAI-compatible endpoint
   (stdlib urllib, non-streaming), runs each tool call, records each new file under the stage's
   `evidence/` directory in the ledger with its sha256, and feeds the watchdog an Event for every
   response, tool call and tool result. A nudge the watchdog's ladder sends is added to the next
   request as a user message. Transport failures are retried once, through the watchdog's
   RetryGuard (spec section 3).

What it does not do: streaming; a proxy in front of agents the supervisor did not launch (the
watchdog only reads their transcripts); read-only mounts. The same user account runs the agent,
so a file outside the run directory that holds a token can still be read by absolute path, and a
shell redirect can still write outside the run directory. The real limits there are a separate
user account or a mount namespace, which plan 4 does not set up.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from orchard.canary import CanaryError, post_json
from orchard.defaults import (AGENT_MAX_TOKENS, AGENT_MAX_TURNS, AGENT_REQUEST_TIMEOUT_S,
                              TOOL_OUTPUT_CHARS, TOOL_TIMEOUT_S)
from orchard.runner import Denied, check_string
from orchard.stages import evidence_record
from orchard.watchdog import Event, RetryGuard

# ---- environment --------------------------------------------------------------------------------

ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TERM")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|_KEY$|^KEY$|COOKIE", re.IGNORECASE)


def agent_env(run_dir, *, extra: dict | None = None, source=None) -> dict:
    """The whole environment of an agent's shell. Nothing is inherited outside ENV_ALLOW."""
    source = os.environ if source is None else source
    env = {k: source[k] for k in ENV_ALLOW if k in source}
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
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "ORCHARD_RUN_DIR": str(Path(run_dir).resolve()),
    })
    for k, v in (extra or {}).items():
        if SECRET_NAME.search(k):
            raise ValueError(f"{k} looks like a credential; agent shells never get one")
        env[k] = str(v)
    return env


# ---- running a checked command ------------------------------------------------------------------

def spawn_checked(command: str, run_dir, env: dict, timeout: float, stdout) -> tuple[int | None, bool]:
    """Run one shell string after the runner's checks. Returns (exit code or None, timed out).

    Denied propagates: a refused string never starts. The command runs in its own session, so a
    timeout kills everything it started.
    """
    check_string(command, run_dir)
    proc = subprocess.Popen(["bash", "-c", command], cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
                            stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return None, True


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - limit} characters cut ...]\n{text[-half:]}"


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
        "name": "write_file",
        "description": "Write a text file. The path is relative to your stage directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]


class Tools:
    def __init__(self, run_dir, stage_dir, env: dict, *, timeout: float = TOOL_TIMEOUT_S,
                 limit: int = TOOL_OUTPUT_CHARS):
        self.run_dir, self.stage_dir = Path(run_dir), Path(stage_dir)
        self.env, self.timeout, self.limit = env, timeout, limit

    def call(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            return "error: the arguments are not JSON"
        if not isinstance(args, dict):
            return "error: the arguments must be a JSON object"
        if name == "shell":
            return self.shell(args.get("command"))
        if name == "write_file":
            return self.write_file(args.get("path"), args.get("content"))
        return f"error: there is no tool named {name!r}; the tools are shell and write_file"

    def shell(self, command) -> str:
        if not isinstance(command, str) or not command.strip():
            return "error: command must be a non-empty string"
        out_path = self.stage_dir / "log" / "last-shell-output.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "wb") as out:
                code, timed_out = spawn_checked(command, self.run_dir, self.env, self.timeout, out)
        except Denied as exc:
            return f"refused: {exc}"
        text = out_path.read_text(encoding="utf-8", errors="replace")
        head = f"killed after {self.timeout} s" if timed_out else f"exit {code}"
        return f"{head}\n{clip(text, self.limit)}"

    def target(self, path) -> Path | None:
        """The real path `path` names inside the stage directory, or None (links resolved)."""
        if not isinstance(path, str) or not path.strip():
            return None
        root = os.path.realpath(self.stage_dir)
        p = os.path.realpath(os.path.join(root, path))
        if p == root or os.path.commonpath([root, p]) != root:
            return None
        return Path(p)

    def write_file(self, path, content) -> str:
        if not isinstance(content, str):
            return "error: content must be a string"
        p = self.target(path)
        if p is None:
            return f"refused: {path!r} is outside your stage directory {self.stage_dir}"
        if p.name.startswith("ledger"):
            return "refused: the ledger is written only by the supervisor"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {os.path.relpath(p, os.path.realpath(self.run_dir))}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py`
Expected: PASS (9 tests).

- [ ] **Step 5: Mutation checks**

Mutation 3 makes the refused command string run in `tmp_path`: `git push` there fails with "not a git repository", and the test stops at its first assertion.

1. The allow-list: in `orchard/agent.py`, replace `    env = {k: source[k] for k in ENV_ALLOW if k in source}` with `    env = dict(source)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py::test_the_environment_is_an_allow_list_with_a_home_inside_the_run`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. HOME inside the run: in `orchard/agent.py`, replace `        "HOME": str(home),` with `        "HOME": source.get("HOME", str(home)),`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py::test_a_shell_command_finds_no_token_in_its_environment_or_home`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. The runner's check comes first: in `orchard/agent.py`, replace `    check_string(command, run_dir)` with `    pass`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py::test_a_refused_command_never_starts`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. Writes stay in the stage directory: in `orchard/agent.py`, replace `        if p == root or os.path.commonpath([root, p]) != root:` with `        if p == root:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent_env.py::test_write_file_stays_inside_the_stage_directory`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/agent.py tests/test_agent_env.py
git commit -m "Give agent shells an allow-listed environment and checked shell and write_file tools"
```

---

### Task 6: The fake model server and the agent step loop

**Files:**
- Create: `tests/fake_model.py`
- Modify: `orchard/agent.py` (append)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Tools`, `agent_env` (Task 5); `orchard.watchdog.Event`, `RetryGuard`, `Watchdog`, `Ladder`, the detectors; `orchard.canary.post_json`.
- Produces in `tests/fake_model.py`: `FakeModel(script, models=("fake",))` (context manager; attributes `endpoint` ending in `/v1` and `requests`), `call(name, call_id="c1", **arguments) -> dict`, `final(text="done") -> dict`, `turn(request) -> int`.
- Produces in `orchard.agent`: `AgentError`; `probe_model(endpoint, model, timeout=5.0) -> bool`; `sha(text) -> str`; `Outcome(status, turns, final_text="", detail="")` with status `done`, `escalate`, `pause`, `operator-pause`, `abort`, `turns` or `error`; `AgentStep(*, agent, endpoint, model, tools, ledger, stage, phase, feed, control, run_dir, evidence_dir, log_path, http=post_json, clock=time.time, guard=None, max_turns=..., max_tokens=..., timeout=...)` with `run(system, user) -> Outcome`. `control` provides `take_nudges(agent) -> list[str]` and `stop_reason() -> str | None`. Ledger entries it writes: `evidence` (`what="evidence file"` per new file under `evidence_dir`, `what="transcript"` at the end) and `retry` (`what="model request"`).

- [ ] **Step 1: Write the fake model server**

Create `tests/fake_model.py`:

```python
"""A deterministic OpenAI-shaped model server for tests. No model, no network beyond 127.0.0.1.

`FakeModel(script)` serves GET /v1/models and POST /v1/chat/completions. For each chat request
it calls `script(request)`, which returns an assistant message dict, or an int to answer with
that HTTP status. The answer depends only on the request, so a restarted supervisor that sends
the same conversation gets the same answer, as from a greedy server.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def call(name: str, call_id: str = "c1", **arguments) -> dict:
    """An assistant message that asks for one tool call."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def final(text: str = "done") -> dict:
    return {"role": "assistant", "content": text}


def turn(request: dict) -> int:
    """How many assistant messages the conversation already holds: 0 on the first request."""
    return sum(1 for m in request["messages"] if m.get("role") == "assistant")


class FakeModel:
    def __init__(self, script, models=("fake",)):
        self.script, self.models = script, list(models)
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send(200, {"data": [{"id": m} for m in owner.models]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(n))
                owner.requests.append(request)
                answer = owner.script(request)
                if isinstance(answer, int):
                    self._send(answer, {"error": "scripted failure"})
                    return
                self._send(200, {"choices": [{"index": 0, "message": answer,
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 100 + 10 * len(request["messages"]),
                                           "completion_tokens": 20}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        # A short poll interval keeps shutdown (and so each test) fast.
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_agent.py`:

```python
"""The agent step loop against a scripted fake model endpoint."""
import pytest

from fake_model import FakeModel, call, final, turn
from orchard.agent import AgentStep, Tools, agent_env, probe_model
from orchard.ledger import Ledger
from orchard.watchdog import Event, IdenticalResponses, Ladder, StageOverBudget, Watchdog


class Control:
    """The supervisor side of a step: the watchdog's actuator plus the stop question."""

    def __init__(self):
        self.nudges, self.stop = [], None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop = "escalate"

    def pause(self, reason, evidence):
        self.stop = "pause"

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        return self.stop


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run" / "stages" / "0" / "evidence").mkdir(parents=True)
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        yield tmp_path / "run", led


def step(run, ledger, endpoint, *, control=None, feed=None, clock=None, **kw):
    sd = run / "stages" / "0"
    control = control or Control()
    events = []
    return AgentStep(agent="stage-agent", endpoint=endpoint, model="fake",
                     tools=Tools(run, sd, agent_env(run)), ledger=ledger, stage=0, phase="run",
                     feed=feed or events.append, control=control, run_dir=run,
                     evidence_dir=sd / "evidence", log_path=sd / "log" / "run-1.jsonl",
                     clock=clock or (lambda: 1000.0), **kw), events, control


SCRIPT = [call("write_file", path="evidence/a.txt", content="hello"),
          call("shell", command="cat stages/0/evidence/a.txt"), final("stage done")]


def test_a_step_runs_tool_calls_until_a_final_answer(run):
    run_dir, ledger = run
    with FakeModel(lambda r: SCRIPT[turn(r)]) as fm:
        s, events, _ = step(run_dir, ledger, fm.endpoint)
        out = s.run("system text", "user text")
    assert (out.status, out.turns, out.final_text) == ("done", 3, "stage done")
    assert fm.requests[2]["messages"][-1] == {"role": "tool", "tool_call_id": "c1",
                                              "content": "exit 0\nhello"}
    ev = [e["data"] for e in ledger.read() if e["event"] == "evidence"]
    assert ev[0]["path"] == "stages/0/evidence/a.txt" and len(ev[0]["sha256"]) == 64
    assert ev[-1]["what"] == "transcript" and ev[-1]["status"] == "done"
    kinds = [e.kind for e in events]
    assert kinds.count("response") == 3 and kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2 and kinds.count("evidence") == 1


def test_the_request_is_greedy_non_streaming_and_offers_both_tools(run):
    run_dir, ledger = run
    with FakeModel(lambda r: final()) as fm:
        step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    req = fm.requests[0]
    assert req["temperature"] == 0 and req["stream"] is False and req["model"] == "fake"
    assert [t["function"]["name"] for t in req["tools"]] == ["shell", "write_file"]
    assert req["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def repeating(stop_after):
    """The same visible text every turn; the tool arguments change, so only the text repeats."""
    def script(request):
        n = turn(request)
        if n >= stop_after:
            return final("gave up")
        msg = call("shell", command=f"echo {n}")
        msg["content"] = "let me look again"
        return msg
    return script


def test_a_nudge_from_the_ladder_is_added_to_the_next_request(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([IdenticalResponses()], Ladder(control, ledger, {"stage-agent"}))
    with FakeModel(repeating(4)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed)[0].run("s", "u")
    assert out.status == "done"
    users = [m["content"] for m in fm.requests[3]["messages"] if m["role"] == "user"]
    assert len(users) == 2 and "repeating itself" in users[1]
    assert not any("repeating itself" in m.get("content", "") for m in fm.requests[2]["messages"])


def test_a_second_loop_escalates_and_stops_the_step(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([IdenticalResponses()], Ladder(control, ledger, {"stage-agent"}))
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed)[0].run("s", "u")
    assert out.status == "escalate" and out.turns == 6
    assert [e["event"] for e in ledger.read() if e["data"].get("watchdog")] == ["retry", "escalate"]


def test_the_stage_budget_pause_stops_the_step(run):
    run_dir, ledger = run
    control = Control()
    wd = Watchdog([StageOverBudget({0: 60})], Ladder(control, ledger, {"stage-agent"}))
    wd.feed(Event(ts=0, agent="supervisor", kind="ledger", name="stage_start", stage=0))
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, control=control, feed=wd.feed,
                   clock=lambda: 100.0)[0].run("s", "u")
    assert out.status == "pause" and len(fm.requests) == 1


def test_a_failed_request_is_retried_once_and_no_more(run):
    run_dir, ledger = run
    with FakeModel(lambda r: 500) as fm:
        out = step(run_dir, ledger, fm.endpoint)[0].run("s", "u")
    assert out.status == "error" and len(fm.requests) == 2
    assert [e["data"]["attempt"] for e in ledger.read() if e["event"] == "retry"] == [1, 2]


def test_the_turn_limit_ends_the_step(run):
    run_dir, ledger = run
    with FakeModel(repeating(50)) as fm:
        out = step(run_dir, ledger, fm.endpoint, max_turns=2)[0].run("s", "u")
    assert out.status == "turns" and len(fm.requests) == 2


def test_probe_model_reads_the_model_list():
    with FakeModel(lambda r: final(), models=["Qwen/Qwen3.8-27B"]) as fm:
        assert probe_model(fm.endpoint, "Qwen/Qwen3.8-27B") is True
        assert probe_model(fm.endpoint, "other") is False
    assert probe_model("http://127.0.0.1:9/v1", "x", timeout=1) is False
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_agent.py`
Expected: FAIL with `ImportError: cannot import name 'AgentStep' from 'orchard.agent'`.

- [ ] **Step 4: Implement**

Append to `orchard/agent.py`:

```python


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
                 max_tokens: int = AGENT_MAX_TOKENS, timeout: float = AGENT_REQUEST_TIMEOUT_S):
        self.agent, self.endpoint, self.model, self.tools = agent, endpoint, model, tools
        self.ledger, self.stage, self.phase = ledger, stage, phase
        self.feed, self.control, self.http, self.clock = feed, control, http, clock
        self.run_dir, self.evidence_dir = Path(run_dir), Path(evidence_dir)
        self.log_path = Path(log_path)
        self.guard = guard if guard is not None else RetryGuard()
        self.max_turns, self.max_tokens, self.timeout = max_turns, max_tokens, timeout
        self._seen: dict[str, tuple[int, int]] = {}

    # ---- helpers ----------------------------------------------------------------------------

    def _event(self, kind: str, **fields) -> None:
        self.feed(Event(ts=self.clock(), agent=self.agent, kind=kind, stage=self.stage, **fields))

    def _log(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _snapshot(self) -> dict[str, tuple[int, int]]:
        out = {}
        if self.evidence_dir.is_dir():
            for f in sorted(self.evidence_dir.rglob("*")):
                if f.is_file():
                    st = f.stat()
                    out[str(f)] = (st.st_size, st.st_mtime_ns)
        return out

    def _record_new_evidence(self) -> None:
        now = self._snapshot()
        for path, stamp in now.items():
            if self._seen.get(path) != stamp:
                rec = evidence_record(self.run_dir, path)
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
            self.ledger.append("evidence", self.stage, what="transcript", phase=self.phase,
                               status=status, turns=turns, **evidence_record(self.run_dir, self.log_path))
        return Outcome(status, turns, final_text, detail)

    # ---- the loop ---------------------------------------------------------------------------

    def run(self, system: str, user: str) -> Outcome:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logged = 0
        self._seen = self._snapshot()       # files already here (a resumed stage) are not new
        for turn in range(1, self.max_turns + 1):
            for text in self.control.take_nudges(self.agent):
                messages.append({"role": "user", "content": text})
            reason = self.control.stop_reason()
            if reason:
                return self._end(reason, turn - 1)
            request = {"model": self.model, "messages": messages, "tools": TOOL_SCHEMAS,
                       "temperature": 0, "max_tokens": self.max_tokens, "stream": False}
            try:
                data = self._send(request)
                msg = data["choices"][0]["message"]
                if not isinstance(msg, dict):
                    raise TypeError("message is not an object")
            except AgentError as exc:
                return self._end("error", turn - 1, detail=str(exc))
            except (KeyError, IndexError, TypeError) as exc:
                return self._end("error", turn - 1, detail=f"no choices[0].message: {exc}")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            details = usage.get("completion_tokens_details") or {}
            assistant = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            self._log({"turn": turn, "sent": messages[logged:-1], "received": assistant, "usage": usage})
            logged = len(messages)
            self._event("response", input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                        thinking_tokens=details.get("reasoning_tokens") if isinstance(details, dict) else None,
                        text_hash=sha(content) if content else None, had_tool_call=bool(calls))
            if not calls:
                return self._end(self.control.stop_reason() or "done", turn, final_text=content)
            for call in calls:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name, args = str(fn.get("name", "")), fn.get("arguments") or "{}"
                args = args if isinstance(args, str) else json.dumps(args)
                self._event("tool_call", tool=name, args_hash=sha(args))
                result = self.tools.call(name, args)
                messages.append({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": result})
                self._event("tool_result", tool=name, output_hash=sha(result))
                self._record_new_evidence()
        return self._end("turns", self.max_turns, detail=f"no final answer after {self.max_turns} turns")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_agent.py tests/test_agent_env.py`
Expected: PASS (17 tests).

- [ ] **Step 6: Mutation checks**

1. Nudges reach the next request: in `orchard/agent.py`, replace `            for text in self.control.take_nudges(self.agent):` with `            for text in []:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent.py::test_a_nudge_from_the_ladder_is_added_to_the_next_request`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The retry guard limits sends: in `orchard/agent.py`, replace `            if not self.guard.allow(self.agent, request):` with `            if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent.py::test_a_failed_request_is_retried_once_and_no_more`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. New evidence is recorded: in `orchard/agent.py`, replace `                self._record_new_evidence()` with `                pass`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_agent.py::test_a_step_runs_tool_calls_until_a_final_answer`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/agent.py tests/fake_model.py tests/test_agent.py
git commit -m "Add the agent step loop: model calls, tool calls, evidence, watchdog events and nudges"
```

---

### Task 7: The draft stage skills and the stage context

**Files:**
- Create: `orchard/skills/delta-triage.md`, `orchard/skills/reference-gate.md`, `orchard/skills/serving-check.md`, `orchard/skills/operator-bundle.md`, `orchard/context.py`
- Test: `tests/test_skills.py`, `tests/test_context.py`

**Interfaces:**
- Consumes: `STAGES`, `StageSpec`, `evidence_record`, `resolve_skill` (Task 3).
- Produces: the four skills, found by `resolve_skill(name, [orchard/skills])`. Each names its gate file. The existing skills in spec section 11 are referenced by name only (`StageSpec.skill`, `StageSpec.refs`) and found in the directories passed with `--skills-dir`.
- Produces in `orchard.context`: `PHASE_TASKS`, `RULES`, `ledger_excerpt(entries, stage) -> list[str]`, `build_messages(*, spec, phase, run_dir, stage_dir, skill_path, refs, facts, entries, resumed) -> (system, user)`, `facts_from(run_start, run_dir) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills.py`:

```python
"""The local draft stage skills, and the skill names the stage table uses."""
from pathlib import Path

import pytest

from orchard.stages import STAGES, resolve_skill

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "serving-check", "operator-bundle")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
                 "vllm-integration", "qualitative-check", "benchmark-model", "tt-device-usage",
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "serving-check": "result.json", "operator-bundle": "PUBLISH_COMMANDS.txt"}


@pytest.mark.parametrize("name", LOCAL)
def test_each_local_skill_is_a_marked_draft_with_its_gate_file(name):
    path = resolve_skill(name, [SKILLS])
    assert path == SKILLS / f"{name}.md"
    text = path.read_text()
    front = text.split("---")[1]
    assert f"\nname: {name}\n" in front and "\nstatus: draft." in front
    assert GATE_FILES[name] in text


@pytest.mark.parametrize("name", LOCAL)
def test_no_stage_skill_names_a_lease_tool(name):
    # Spec section 11: stage skills say "under whatever lease the machine provides".
    assert "gozer" not in (SKILLS / f"{name}.md").read_text().lower()


def test_every_skill_the_table_names_is_local_or_named_in_the_spec():
    for s in STAGES:
        if s.skip:
            continue
        assert s.skill in LOCAL or s.skill in SPEC_EXISTING, s.skill
        assert set(s.refs) <= SPEC_EXISTING, s.refs


def test_the_bundle_skill_keeps_publishing_with_the_operator():
    text = (SKILLS / "operator-bundle.md").read_text()
    assert "You never run them." in text and "ready for operator review" in text
```

Create `tests/test_context.py`:

```python
"""The fresh context each agent step starts from."""
import json

from orchard.context import build_messages, facts_from
from orchard.stages import STAGES


def setup(tmp_path, skill="Compare the configs."):
    run = tmp_path / "run"
    (run / "stages" / "0").mkdir(parents=True)
    skill_path = tmp_path / "delta-triage.md"
    skill_path.write_text(skill)
    return run, skill_path


def msgs(run, skill_path, n=0, phase="run", entries=(), resumed=False, refs=None):
    return build_messages(spec=STAGES[n], phase=phase, run_dir=run, stage_dir=run / "stages" / str(n),
                          skill_path=skill_path, refs=refs or {}, entries=list(entries),
                          facts={"model": "Altworld/Hemmingway-1"}, resumed=resumed)


def test_a_first_step_holds_the_skill_the_rules_and_the_task(tmp_path):
    run, sp = setup(tmp_path)
    system, user = msgs(run, sp, refs={"model-bringup": None})
    assert "stage 0 (intake and delta triage)" in system and "Compare the configs." in system
    assert "Never publish" in system and "model-bringup: not installed" in system
    assert "Write delta.json in your stage directory" in user
    assert "- model: Altworld/Hemmingway-1" in user and "nothing yet" in user


def test_earlier_results_are_included_and_later_ones_are_not(tmp_path):
    run, sp = setup(tmp_path)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps({"path": "weights-only"}))
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "result.json").write_text('{"pcc": 0.999}')
    _, user = msgs(run, sp, n=1)
    assert "### stages/0/delta.json (sha256 " in user and '"weights-only"' in user
    assert "0.999" not in user


def test_the_finish_step_sees_the_hardware_test_record(tmp_path):
    run, sp = setup(tmp_path)
    sd = run / "stages" / "2"
    sd.mkdir(parents=True)
    (sd / "test-result.json").write_text('{"returncode": 3}')
    _, user = msgs(run, sp, n=2, phase="finish")
    assert "## stages/2/test-result.json" in user and '"returncode": 3' in user
    assert "Write result.json from that evidence" in user


def test_resume_notes_are_given_only_to_a_resumed_step(tmp_path):
    run, sp = setup(tmp_path)
    (run / "stages" / "0" / "RESUME.md").write_text("tensors compared; tokenizer next")
    assert "tokenizer next" in msgs(run, sp, resumed=True)[1]
    assert "tokenizer next" not in msgs(run, sp, resumed=False)[1]


def test_the_ledger_excerpt_says_why_an_attempt_failed(tmp_path):
    run, sp = setup(tmp_path)
    entries = [{"event": "stage_end", "stage": 0, "data": {"result": "escalate",
                                                         "reasons": ["delta.json is missing"]}}]
    assert "- stage 0: escalate (delta.json is missing)" in msgs(run, sp, entries=entries)[1]


def test_a_long_skill_is_cut_and_says_so(tmp_path):
    run, sp = setup(tmp_path, skill="x" * 50000)
    assert "[cut at 40000 characters]" in msgs(run, sp)[0]


def test_facts_list_the_model_and_the_inputs(tmp_path):
    facts = facts_from({"model": "m", "inputs": {"base": "/mnt/base"}}, tmp_path)
    assert facts == {"model": "m", "run directory": str(tmp_path), "input base": "/mnt/base"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_skills.py tests/test_context.py`
Expected: 9 of the 10 tests in `test_skills.py` fail (`assert None == ...` or `FileNotFoundError`: no skill files yet; the table test passes), and `test_context.py` fails with `ModuleNotFoundError: No module named 'orchard.context'`.

- [ ] **Step 3: Write the four draft skills**

Create `orchard/skills/delta-triage.md`:

```markdown
---
name: delta-triage
description: Stage 0 of a tt-orchard run. Compare a new Hugging Face model with the nearest model that already runs on Tenstorrent hardware, write down every difference with evidence, and name the path the run takes.
status: draft. A local tt-orchard copy. It lives in tt-orchard.
---

# Delta triage

## Goal

Write `delta.json` in your stage directory. It lists what differs between the new model and the
nearest supported model, gives evidence for each difference, and names the path:

- `weights-only`: the model code that runs the nearest model can run the new one with new weights.
- `full-port`: new model code is needed. Plan 4 of tt-orchard stops the run for the operator here.

## Inputs

The run lists input paths, for example `model=` and `base=` snapshot directories. Read the files
there. Do not download anything. The run has no Hugging Face token, and the weights are on disk.

## What to compare

Save the output of each comparison under `evidence/`.

1. `config.json`: every key that differs. The text-model keys matter most: layers, hidden size,
   attention heads, KV heads, layer types, vocabulary size, context length. Name keys that differ
   only as bookkeeping (`architectures`, `transformers_version`).
2. Tensors: names from each `model.safetensors.index.json`, and shapes and dtypes from the
   safetensors headers. Report the counts, names found in only one model, and every shape or dtype
   difference. A consistent name prefix difference is its own entry (area `tensor_names`).
3. Tokenizer: vocabulary size, merges, added tokens, chat template and pre-tokenizer pattern. If
   any of them differs, encode a set of test strings with both tokenizers and count the strings
   whose ids differ. Save the strings.
4. Files: the weight shards, their total size, and files only one model has (for example `mtp.*`
   or vision weights).
5. `generation_config.json`: the sampling defaults.
6. The license, from the model card or the LICENSE file.

A safetensors header needs no extra package: the first 8 bytes are a little-endian length, and
that many bytes of JSON follow. Write a short Python script into your stage directory with
write_file and run it with `python3`. The supervisor refuses heredocs.

## The file

    {"model": "<org/name>@<revision>", "nearest_model": "<org/name>@<revision>",
     "path": "weights-only",
     "differences": [{"area": "tokenizer", "finding": "...",
                      "evidence": ["stages/0/evidence/tokenizer-diff.txt"]}],
     "hazards": [{"area": "tensor_cache", "finding": "..."}]}

A difference's `area` is one of: config, architecture, tensors, tensor_names, tokenizer,
chat_template, files, generation_config, license, other. Write one entry for each area you
checked, including areas with no difference (the finding says "identical" and the evidence shows
it). A hazard's `area` is one of: tensor_cache, drafter, disk, license, other. Evidence paths are
relative to the run directory and must be files inside it.

## Hazards to look for

- A converted tensor cache of the nearest model that a later stage could reuse. A cache keyed
  only by layer name serves the old weights silently. Name the cache directory that must stay
  unused.
- A draft model for speculative decoding that was trained against the nearest model. Its
  acceptance rate may drop, so it must be measured.
- Free disk where the weights and caches will go.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
```

Create `orchard/skills/reference-gate.md`:

```markdown
---
name: reference-gate
description: Stage 1 of a tt-orchard run. Build a CPU reference of the new model and show that it reproduces the model's published behavior before any device result is compared with it.
status: draft. A local tt-orchard copy. It lives in tt-orchard.
---

# Reference gate

## Why this stage exists

A failed bring-up on this machine built a CPU reference that decoded the wrong way, and every
later measurement was made against it. This stage checks the reference itself first.

## Goal

Write `reference.json` in your stage directory:

    {"verdict": "pass",
     "environment": {"python": "...", "torch": "...", "transformers": "..."},
     "checks": [{"name": "decodes forward", "pass": true, "note": "...",
                 "evidence": ["stages/1/evidence/decode.txt"]}]}

`verdict` is `pass` only when every check passes.

## Checks

Run each check, save its output under `evidence/`, and write one entry for it.

1. `loads`: the weights load on CPU with no missing or unexpected keys. Save the load report.
2. `tokenizer round trip`: a set of test strings encodes and decodes back to the same text.
3. `decodes forward`: greedy decoding of a short prompt. The argmax of the last position's logits
   equals the first generated token, and each new token is appended at the end. Save the prompt,
   the token ids and the text.
4. `matches the card`: reproduce one behavior the model card states. If the card publishes an
   exact sample, compare with it. If it publishes none, render one prompt through the chat
   template, generate, and check that the output has the form the card describes. Say in `note`
   that the card gives no exact sample.

## Cost

Large models are slow on this CPU. A dense 27B model in bf16 decodes at about 0.7 tokens per
second here (measured 2026-10-02). Keep prompts short and generate at most 32 tokens per check.
Load the model once and run every check in the same Python process.

## Leave for later stages

Save the reference outputs under `evidence/reference/`: the prompt ids, the generated ids and the
top-5 logits of each position. Stages 2 and 3 compare device results with them.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
```

Create `orchard/skills/serving-check.md`:

```markdown
---
name: serving-check
description: Stages 5 and 6 of a tt-orchard run. Check a served model from the outside (boot, passkey, canary), then run a short qualitative check and a benchmark, with every number labelled measured or TODO.
status: draft. A local tt-orchard copy, thinner than the vllm-integration, qualitative-check and benchmark-model skills it points to. It lives in tt-orchard.
---

# Serving check

## How the hardware test works

You do not run the server yourself. In the prepare phase, write a Python script in your stage
directory that does the whole check and exits, and give `python3 stages/<n>/<script>.py` as the
command in `hw_test.json`. The supervisor runs it on a leased board with `TT_VISIBLE_DEVICES`
set, with a deadline. The script must:

- start the server for the new model on the chips in `TT_VISIBLE_DEVICES`, using the recipe the
  earlier stages used (stage 0's path and stage 4's configuration);
- wait for it to be ready, with a time limit;
- run the checks below and write their outputs under `stages/<n>/evidence/`;
- stop the server before it exits, whatever happened. The supervisor checks that the chips are
  quiet afterwards and resets them. Never reset chips yourself.

In the finish phase, read `test-result.json` and the evidence, and write `result.json`.

## Stage 5: does it serve

    {"checks": {"boots":   {"pass": true, "seconds": 41.2, "evidence": ["stages/5/evidence/boot.log"]},
                "passkey": {"pass": true, "lengths": [2048, 32768], "evidence": ["..."]},
                "canary":  {"pass": true, "evidence": ["..."]}}}

- `boots`: the server reaches ready. Record the seconds to ready.
- `passkey`: a passkey (needle) hidden in filler text at two or more prompt lengths, the longest
  near the configured context, is answered exactly.
- `canary`: one fixed greedy prompt, asked twice, gives the same answer both times.

## Stage 6: how well and how fast

    {"numbers": [{"name": "decode", "value": 80.1, "unit": "tok/s/user", "label": "measured",
                  "workload": "prompt 128, generate 128, 1 user", "evidence": ["..."]},
                 {"name": "ttft", "value": null, "unit": "ms", "label": "TODO"}],
     "qualitative": {"summary": "...", "evidence": ["stages/6/evidence/qualitative.md"]}}

- Qualitative: five prompts suited to what the model is for (for a creative-writing model,
  writing tasks). Save the prompts and the full outputs, and write your reading of them in
  `qualitative.md`: wrong language, base-model autocomplete, repetition or broken formatting are
  failures to report.
- Benchmark: decode tokens per second per user and time to first token, at the workload the
  nearest supported model's package reports, so the numbers can be compared. Record the workload
  next to each number.
- Every number is labelled `measured` (the evidence file shows it) or `TODO` with a null value.
  The supervisor writes each one to the ledger as a measurement.

## Related skills

`vllm-integration`, `qualitative-check` and `benchmark-model` describe the full methods. Their
paths are listed in your context when they are installed.
```

Create `orchard/skills/operator-bundle.md`:

```markdown
---
name: operator-bundle
description: Stage 8 of a tt-orchard run. Write the bundle an operator reviews before anything is published - results, open risks and the exact publish commands as text - and never run those commands.
status: draft. A local tt-orchard copy. It lives in tt-orchard.
---

# Operator bundle

## Goal

The run ends at "ready for operator review". The operator decides whether anything is published,
whether it is public, and whether it enters any catalog. Write these files in `bundle/` in your
stage directory:

- `RESULTS.md`: what each stage found, from the stage result files in your context. Every number
  appears with its label (`measured` or `TODO`) and the evidence path behind it. Say that stage 7
  (package and container build) was skipped, so no package exists yet.
- `RISKS.md`: open risks. Include every `TODO` number, every stage 0 hazard and whether a later
  stage dealt with it, anything a gate passed on thin evidence, and what was never tested.
- `PUBLISH_COMMANDS.txt`: the exact commands the operator would run to package and publish, as
  text, one per line, private by default. You never run them. The supervisor refuses publish,
  push and upload commands.
- `card.md` (optional): a draft model card built only from measured numbers.

## The scrub check

After you finish, the supervisor copies the run ledger into `bundle/ledger.jsonl` and searches
every other file in `bundle/` for the machine's hostname, tokens and absolute home paths. A hit
blocks the bundle. Write paths relative to the run directory, and edit command output before you
paste it. Say in `RESULTS.md` that `ledger.jsonl` is the operator's copy of the run record, holds
absolute paths, and is not for publishing.
```

- [ ] **Step 4: Write the context builder**

Create `orchard/context.py`:

```python
"""The fresh, short context each agent step starts from (spec section 5).

A step gets only: the stage skill, the rules of the run, a ledger excerpt (how earlier stages
ended and what happened to this stage so far), the result files of earlier stages, and, for a
resumed or finishing step, the files this stage already wrote. Nothing from an earlier step's
conversation is carried over, so a long run never pays for a long context.
"""
from __future__ import annotations

from pathlib import Path

from orchard.defaults import CONTEXT_FILE_CHARS, SKILL_CHARS
from orchard.stages import STAGES, StageSpec, evidence_record

PHASE_TASKS = {
    "run": ("Do this stage's work as the skill describes. Write {gate} in your stage directory. "
            "Then reply with a short summary and no tool call."),
    "prepare": ("Prepare this stage's hardware test. Do not run it yourself. Write hw_test.json in "
                "your stage directory as {{\"command\": \"<one shell command, run from the run "
                "directory>\", \"deadline_s\": <seconds>}} and handoff.json with the keys goal, "
                "stage, evidence, next_action and check_on_return. The command cannot be a shell "
                "script path; write a Python script and give `python3 stages/{n}/<script>.py`. The "
                "supervisor runs the command on a leased board with TT_VISIBLE_DEVICES set, saves "
                "its output to evidence/hw-test-output.txt and writes test-result.json. Then reply "
                "with a short summary and no tool call."),
    "finish": ("The supervisor ran your hardware test; its record and output are below and in "
               "your stage directory. Write {gate} from that evidence. If the test failed, write "
               "what failed; do not invent a result. Then reply with a short summary and no tool "
               "call."),
}

RULES = """Rules of this run:
- Your tools are shell and write_file. shell starts in the run directory {run_dir}. write_file writes
  only inside your stage directory, stages/{n}.
- Put every file that backs a claim under stages/{n}/evidence/. The supervisor records each one in the
  ledger with its sha256. Evidence paths you cite are relative to the run directory.
- Label every number "measured" (you measured it and the evidence file shows it) or "TODO".
- Never publish, push or upload anything, and never reset chips. The supervisor refuses those
  commands. There are no GitHub or Hugging Face credentials in your environment.
- Do not open Tenstorrent devices from your shell. Hardware work happens only in the hardware test
  that the supervisor runs for you under a lease.
- When you are finished, reply with a short summary and no tool call."""


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n[cut at {limit} characters]"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def ledger_excerpt(entries: list[dict], stage: int) -> list[str]:
    lines = []
    for e in entries:
        d = e["data"]
        if e["event"] == "stage_end":
            why = "; ".join(d.get("reasons") or []) if d.get("result") != "pass" else ""
            lines.append(f"stage {e['stage']}: {d.get('result')}" + (f" ({why})" if why else ""))
        elif e["stage"] == stage and e["event"] in ("escalate", "notice"):
            lines.append(f"stage {stage} {e['event']}: " + str(d.get("reason") or d.get("what")
                                                              or d.get("findings") or "")[:300])
    return lines[-30:]


def build_messages(*, spec: StageSpec, phase: str, run_dir, stage_dir, skill_path: Path | None,
                   refs: dict, facts: dict, entries: list[dict], resumed: bool) -> tuple[str, str]:
    """(system, user) for one agent step."""
    run_dir, stage_dir = Path(run_dir), Path(stage_dir)
    n = spec.number
    skill = _read(skill_path) if skill_path else ""
    system = [f"You are the agent for stage {n} ({spec.name}) of a model bring-up run by tt-orchard.",
              f"Phase: {phase}", "", RULES.format(run_dir=run_dir, n=n), "",
              f"## Skill: {spec.skill} ({skill_path})", clip(skill, SKILL_CHARS)]
    if refs:
        system += ["", "## Related skills (read one with shell if you need it)"]
        system += [f"- {name}: {path or 'not installed on this machine'}" for name, path in refs.items()]

    user = ["## Task", PHASE_TASKS[phase].format(gate=spec.gate_file, n=n), "", "## Run"]
    user += [f"- {k}: {v}" for k, v in facts.items()]
    user += [f"- stage directory: stages/{n}", "", "## Ledger so far"]
    user += [f"- {line}" for line in ledger_excerpt(entries, n)] or ["- nothing yet"]
    user += ["", "## Results of earlier stages"]
    shown = False
    for s in STAGES:
        if s.number >= n or not s.gate_file:
            continue
        f = run_dir / "stages" / str(s.number) / s.gate_file
        if f.is_file():
            rec = evidence_record(run_dir, f)
            user += [f"### {rec['path']} (sha256 {rec['sha256'][:12]})",
                     clip(_read(f), CONTEXT_FILE_CHARS)]
            shown = True
    if not shown:
        user.append("none")
    own = []
    if resumed and spec.marker and (stage_dir / spec.marker).is_file():
        own.append(spec.marker)
    if phase == "finish":
        own += ["hw_test.json", "test-result.json"]
    for name in own:
        f = stage_dir / name
        if f.is_file():
            user += ["", f"## stages/{n}/{name}", clip(_read(f), CONTEXT_FILE_CHARS)]
    return "\n".join(system), "\n".join(user)


def facts_from(run_start: dict, run_dir) -> dict:
    """The run facts every context lists, from the run_start entry."""
    facts = {"model": run_start.get("model"), "run directory": str(run_dir)}
    for name, path in (run_start.get("inputs") or {}).items():
        facts[f"input {name}"] = path
    return facts
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_skills.py tests/test_context.py`
Expected: PASS (17 tests).

- [ ] **Step 6: Mutation checks**

1. No lease tool in a stage skill: in `orchard/skills/serving-check.md`, replace `## Related skills` with `Take a gozer lease first.` followed by a blank line and `## Related skills`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_skills.py::test_no_stage_skill_names_a_lease_tool`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. Resume notes go only to a resumed step: in `orchard/context.py`, replace `    if resumed and spec.marker and (stage_dir / spec.marker).is_file():` with `    if spec.marker and (stage_dir / spec.marker).is_file():`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_context.py::test_resume_notes_are_given_only_to_a_resumed_step`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 7: Sweep the skill text**

Run: `grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what' orchard/skills/*.md orchard/context.py`
Expected: no output.

- [ ] **Step 8: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/skills orchard/context.py tests/test_skills.py tests/test_context.py
git commit -m "Add draft stage skills (delta-triage, reference-gate, serving-check, operator-bundle) and the stage context"
```

---

### Task 8: Supervisor parts: control file, actuator, external stand-in, versions; the run fakes

**Files:**
- Create: `orchard/supervisor.py` (first part), `tests/run_fakes.py`
- Test: `tests/test_supervisor_parts.py`

**Interfaces:**
- Consumes: `orchard.canary.ask`, `post_json`; `orchard.server.ServerSpec`, `StopCheck`; `orchard.commands.run_command`; `orchard.tiers`.
- Produces in `orchard.supervisor`: `WHO`, `AGENT = "stage-agent"`, `EXIT_READY = 0`, `EXIT_REFUSED = 2`, `EXIT_ABORTED = 4`, `FULL_PORT`, `SKILLS_DIR`; `Control(run_dir)` with `COMMANDS`, `write(command)`, `take() -> str | None` (moves the file to `control.done-<k>`); `RunActuator(control)` implementing plan 3's `Actuator` plus `take_nudges`, `stop_reason`, `clear`; `ExternalStandIn(endpoint, model, http=post_json)` implementing plan 3's `StandIn` (`spawn`, `wait_ready`, `ask`, `stop`, `record`, `adopt -> str | None`, `confirm_stopped`); `resolve_versions(spec, run=run_command) -> dict`; `pairs(items, what) -> dict`; `coder_tier(cfg, port) -> str`.
- Produces in `tests/run_fakes.py`: `BOARDS`, `DEV`, `CANARY_ANSWER`, `CrashingLedger(path, crash_after=0, crash_if=None)`, `Machine`, `MachineAdapter`, `MachineCoder`, `DELTA`, `FILES`, `where(request)`, `bringup(request, overrides=None)`, `closed_port()`, `write_tiers(path, chips_endpoint, cpu_endpoint)`, `EXISTING`, `stub_skills(root)`, `argv(tmp_path, tiers, chips_endpoint, chips=4)`, `plenty(path)`, `StuckClock`, `clock()`.

- [ ] **Step 1: Write the run fakes**

`tests/run_fakes.py` is shared by Tasks 8 to 10. Its machine follows gozer's rules as plan 3's `tests/fakes.py` does: a dead owner's lease is reaped at the next acquire once no device is open, and reset and release refuse while a device is open. Its tier file matches the operator's layout: large and small name the same model, only the large one serves (the small endpoint is a closed port), and the CPU tier is a separate server.

Create `tests/run_fakes.py`:

```python
"""Fakes for the supervisor tests: a two-board machine, its coder, and a scripted bring-up.

`Machine` is the state a crash leaves behind: leases (with the pid that owns each), and whether the
coder container runs and on which chips. `MachineAdapter` behaves like gozer through the adapter
interface: a lease whose owner pid is dead is reaped at the next acquire once no device is open,
reset and release refuse while a device is open, and a chip of a running container shows
HELD-FOREIGN. `MachineCoder` behaves like a container coder. `bringup` is a model script that walks
stages 0 to 8 by writing the files each gate reads; it answers from the request alone, as a greedy
server would, so a restarted supervisor gets the same answers.
"""
from __future__ import annotations

import json
import re
import socket
import time

from fake_model import call, final, turn
from fakes import Crash, FakeClock
from orchard.adapters import ChipState, Lease, LeaseLost, Queued, Refused
from orchard.ledger import Ledger
from orchard.server import NotReady, StopCheck

BOARDS = {"B0": ("0000:01:00.0", "0000:02:00.0"), "B1": ("0000:03:00.0", "0000:04:00.0")}
DEV = {c: i for i, c in enumerate(BOARDS["B0"] + BOARDS["B1"])}
CANARY_ANSWER = "42"


class CrashingLedger(Ledger):
    """A ledger whose process dies right after an append is on disk: the k-th append, or the first
    entry for which `crash_if(entry)` is true."""

    def __init__(self, path, crash_after=0, crash_if=None):
        super().__init__(path)
        self.crash_after, self.crash_if, self.appended = crash_after, crash_if, 0

    def append(self, *args, **kwargs):
        entry = super().append(*args, **kwargs)
        self.appended += 1
        if self.appended == self.crash_after or (self.crash_if and self.crash_if(entry)):
            raise Crash(f"killed after ledger event {entry['seq']}: {entry['event']}")
        return entry


class Machine:
    def __init__(self, owner_pid=100):
        self.owner_pid = owner_pid           # the live supervisor; every other owner pid is dead
        self.leases: dict[str, tuple[Lease, int]] = {}
        self.coder_running = False
        self.coder_chips: tuple[str, ...] = ()
        self.coder_starts = 0
        self.coder_answer = CANARY_ANSWER
        self.never_ready = False
        self.resets: list[tuple[str, str]] = []
        self.n = 0

    def held(self) -> set[str]:
        return set(self.coder_chips) if self.coder_running else set()


class MachineAdapter:
    def __init__(self, machine: Machine, owner_pid=None):
        self.m = machine
        self.owner_pid = owner_pid or machine.owner_pid
        self.calls: list[tuple] = []

    def _reap(self):
        for lid, (lease, owner) in list(self.m.leases.items()):
            if owner != self.m.owner_pid and not set(lease.chips) & self.m.held():
                del self.m.leases[lid]

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        self.calls.append(("acquire", chips, exact))
        self._reap()
        busy = {c for lease, _ in self.m.leases.values() for c in lease.chips} | self.m.held()
        free = [b for b, cs in BOARDS.items() if not set(cs) & busy and (exact is None or exact in cs)]
        need = max(1, chips // 2)
        if len(free) < need:
            if queue:
                raise Queued("t1")
            raise Refused("no free board")
        take = free[:need]
        got = tuple(c for b in take for c in BOARDS[b])
        self.m.n += 1
        lease = Lease(f"L{self.m.n}", got, tuple(DEV[c] for c in got),
                      {"TT_VISIBLE_DEVICES": ",".join(got)}, tuple(take))
        self.m.leases[lease.lease_id] = (lease, self.owner_pid)
        return lease

    def claim(self, ticket, chips, who, reason):
        return self.acquire(chips, who, reason, queue=True)

    def cancel(self, ticket):
        self.calls.append(("cancel", ticket))

    def release(self, lease):
        self.calls.append(("release", lease.lease_id))
        if lease.lease_id not in self.m.leases:
            raise LeaseLost(f"no lease {lease.lease_id}")
        if set(lease.chips) & self.m.held():
            raise Refused("device still open")
        del self.m.leases[lease.lease_id]
        self.m.resets.append(("release", lease.lease_id))

    def reset(self, lease):
        self.calls.append(("reset", lease.lease_id))
        entry = self.m.leases.get(lease.lease_id)
        if entry is None or entry[1] != self.owner_pid:
            raise LeaseLost(f"no lease {lease.lease_id} for pid {self.owner_pid}")
        if set(lease.chips) & self.m.held():
            raise Refused("device still open")
        self.m.resets.append(("reset", lease.lease_id))

    def status(self):
        owner_of = {c: o for lease, o in self.m.leases.values() for c in lease.chips}
        held, out = self.m.held(), []
        for board, chips in BOARDS.items():
            for c in chips:
                if c not in owner_of:
                    out.append(ChipState(c, "BUSY-UNTRACKED" if c in held else "FREE", None,
                                         board=board, dev_index=DEV[c]))
                    continue
                owner = owner_of[c]
                state = ("HELD-FOREIGN" if c in held else
                         "CLAIMED" if owner == self.m.owner_pid else "STALE")
                out.append(ChipState(c, state, "orchard", board=board, dev_index=DEV[c],
                                     lease_pid=owner))
        return out


class MachineCoder:
    """A container coder: it outlives the supervisor, and docker is the stop check."""

    def __init__(self, machine: Machine):
        self.m = machine

    def start(self, lease):
        self.m.coder_running, self.m.coder_chips = True, tuple(lease.chips)
        self.m.coder_starts += 1

    def stop(self):
        self.m.coder_running = False
        return {"how": "tt-model stop", "mesh_reset": False}

    def confirm_stopped(self):
        ok = not self.m.coder_running
        return StopCheck(ok, {"docker_ps": ok}, {})

    def wait_ready(self, budget_s):
        if self.m.never_ready:
            raise NotReady(budget_s)
        return 20.0

    def ask(self, prompt):
        if not self.m.coder_running:
            raise ConnectionRefusedError("the coder is down")
        return self.m.coder_answer

    def record(self):
        return {"kind": "container", "target": "fake/qwen-coder", "port": 8000}

    def adopt(self, record):
        return None


# ---- the scripted bring-up ------------------------------------------------------------------------

def ev(n, *names):
    return [f"stages/{n}/evidence/{x}" for x in names or ("notes.txt",)]


DELTA = {"model": "Altworld/Hemmingway-1", "nearest_model": "Qwen/Qwen3.8-27B", "path": "weights-only",
         "differences": [{"area": a, "finding": "checked", "evidence": ev(0)}
                         for a in ("config", "tensors", "tensor_names", "tokenizer", "files",
                                   "generation_config")],
         "hazards": [{"area": a, "finding": "noted"} for a in ("tensor_cache", "drafter", "disk")]}


def note(n):
    return {"goal": "bring up Hemmingway-1", "stage": n, "evidence": ev(n),
            "next_action": "read the test output", "check_on_return": "the test exit code"}


def hw(n):
    return {"evidence/notes.txt": f"stage {n} plan",
            "hw_test.json": {"command": f"printenv > stages/{n}/evidence/devices.txt",
                             "deadline_s": 60},
            "handoff.json": note(n)}


def done(n):
    return ev(n, "hw-test-output.txt", "devices.txt")


FILES = {
    (0, "run"): {"evidence/notes.txt": "configs, tensors and tokenizers compared", "delta.json": DELTA},
    (1, "run"): {"evidence/notes.txt": "greedy decode matches",
                 "reference.json": {"verdict": "pass", "checks": [
                     {"name": n, "pass": True, "evidence": ev(1)}
                     for n in ("loads", "tokenizer round trip", "decodes forward", "matches the card")]}},
    (2, "prepare"): hw(2),
    (2, "finish"): {"result.json": {"pcc": 0.998, "argmax_match": True, "evidence": done(2)}},
    (3, "prepare"): hw(3),
    (3, "finish"): {"result.json": {"parity": True, "top1": 0.97, "evidence": done(3)}},
    (4, "prepare"): hw(4),
    (4, "finish"): {"result.json": {"configs": [{"chips": 2, "pass": True, "evidence": done(4)},
                                                {"chips": 1, "pass": True, "evidence": done(4)}]}},
    (5, "prepare"): hw(5),
    (5, "finish"): {"result.json": {"checks": {k: {"pass": True, "evidence": done(5)}
                                               for k in ("boots", "passkey", "canary")}}},
    (6, "prepare"): hw(6),
    (6, "finish"): {"result.json": {"numbers": [
        {"name": "decode", "value": 80.0, "unit": "tok/s/user", "label": "measured", "evidence": done(6)},
        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
        "qualitative": {"evidence": done(6)}}},
    (8, "run"): {"bundle/RESULTS.md": "Stages 0 to 6 passed. Stage 7 was skipped.",
                 "bundle/RISKS.md": "ttft is TODO.",
                 "bundle/PUBLISH_COMMANDS.txt": "tt-model push example/hemmingway-1-p300\n"},
}


def where(request) -> tuple[int, str]:
    system = request["messages"][0]["content"]
    n = int(re.search(r"agent for stage (\d+) ", system).group(1))
    return n, re.search(r"^Phase: (\w+)$", system, re.M).group(1)


def bringup(request, overrides=None):
    """The scripted model. A request without tools is a canary, answered with CANARY_ANSWER."""
    if "tools" not in request:
        return final(CANARY_ANSWER)
    key = where(request)
    files = (overrides or {}).get(key, FILES[key])
    turns = [call("write_file", path=p, content=c if isinstance(c, str) else json.dumps(c))
             for p, c in files.items()] + [final(f"stage {key[0]} {key[1]} done")]
    return turns[min(turn(request), len(turns) - 1)]


def closed_port() -> int:
    """A local port with nothing listening on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_tiers(path, chips_endpoint: str, cpu_endpoint: str):
    """The operator's tier layout: large and small are the same model on 4 and 2 chips, and only
    the large one is serving (the small endpoint has nothing listening). The CPU tier is separate."""
    text = "\n".join([
        '[tiers.large]', 'role = "plan"', f'endpoint = "{chips_endpoint}"', 'model = "qwen-27b"',
        'placement = "chips"',
        '[tiers.small]', 'role = "run"', f'endpoint = "http://127.0.0.1:{closed_port()}/v1"',
        'model = "qwen-27b"', 'placement = "chips"',
        '[tiers.cpu]', 'role = "stand-in"', f'endpoint = "{cpu_endpoint}"', 'model = "cpu-model"',
        'placement = "cpu"',
        '[stages.0]', 'run = "large"', '[stages.1]', 'run = "small"',
        '[stages.2]', 'run = "small"', 'diagnose = "large"', '[stages.3]', 'run = "small"',
        'diagnose = "large"', '[stages.4]', 'run = "small"', 'plan = "large"', 'diagnose = "large"',
        '[stages.5]', 'run = "small"', '[stages.6]', 'run = "small"', '[stages.7]', 'run = "none"',
        '[stages.8]', 'run = "small"', '[escalation]', 'default = "large"', ""])
    path.write_text(text)
    return path


EXISTING = ("model-bringup", "functional-decoder", "full-model", "mesh-shrink", "multichip",
            "tt-device-usage", "vllm-integration", "qualitative-check", "benchmark-model")


def stub_skills(root):
    """Stand-ins for the installed tt-model-bringup skills, in the plugin's folder layout."""
    for name in EXISTING:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "SKILL.md").write_text(f"# {name}\nStub for tests.\n")
    return root


def argv(tmp_path, tiers, chips_endpoint: str, chips=4):
    port = chips_endpoint.rsplit(":", 1)[1].split("/")[0]
    return ["run", "--model", "Altworld/Hemmingway-1", "--run-dir", str(tmp_path / "run"),
            "--tiers", str(tiers), "--coder-target", "fake/qwen-coder", "--coder-port", port,
            "--coder-chips", str(chips), "--input", "base=/models/base",
            "--skills-dir", str(stub_skills(tmp_path / "plugin-skills"))]


def plenty(path):
    from collections import namedtuple
    return namedtuple("U", "total used free")(0, 0, 10 ** 15)


class StuckClock(FakeClock):
    """A FakeClock that fails the test once the run has waited more than a simulated day. A run
    that waits for an operator nobody plays, or for a lease that never comes, fails here instead
    of hanging."""

    def sleep(self, s):
        super().sleep(s)
        if sum(self.sleeps) > 86400:
            raise AssertionError("the run waited more than a simulated day; it is stuck")


def clock():
    return StuckClock(time.time())
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_supervisor_parts.py`:

```python
"""The supervisor's parts: control file, actuator, external stand-in, coder tier, versions."""
import pytest

from fake_model import FakeModel, final
from fakes import FakeRun
from orchard.server import ServerSpec
from orchard.supervisor import Control, ExternalStandIn, RunActuator, coder_tier, resolve_versions
from orchard.tiers import load
from run_fakes import write_tiers


def test_a_control_command_acts_once_and_is_kept(tmp_path):
    c = Control(tmp_path)
    assert c.take() is None
    c.write("resume")
    assert c.take() == "resume" and c.take() is None
    assert (tmp_path / "control.done-1").read_text() == "resume\n"
    (tmp_path / "control").write_text("explode\n")
    assert c.take() is None and (tmp_path / "control.done-2").exists()
    with pytest.raises(ValueError):
        c.write("explode")


def test_the_actuator_queues_nudges_and_reads_operator_commands(tmp_path):
    a = RunActuator(Control(tmp_path))
    a.nudge("stage-agent", "do something else")
    assert a.take_nudges("stage-agent") == ["do something else"] and a.take_nudges("stage-agent") == []
    assert a.stop_reason() is None
    Control(tmp_path).write("pause")
    assert a.stop_reason() == "operator-pause"
    a.clear()
    a.escalate("stage-agent", 2)
    Control(tmp_path).write("abort")
    assert a.stop_reason() == "abort"          # abort wins over everything


def test_the_external_stand_in_asks_the_cpu_tier_and_never_stops_it():
    with FakeModel(lambda r: final("4"), models=["cpu-model"]) as fm:
        s = ExternalStandIn(fm.endpoint, "cpu-model")
        s.spawn()
        s.wait_ready()
        assert s.ask("What is 2 + 2?") == "4"
        s.stop()
        assert s.confirm_stopped().stopped and s.record()["kind"] == "external"
    assert fm.requests[0]["model"] == "cpu-model" and "tools" not in fm.requests[0]


def test_the_coder_tier_is_the_chip_tier_on_the_coder_port(tmp_path):
    cfg = load(write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1"))
    assert coder_tier(cfg, 8000) == "large"
    with pytest.raises(ValueError, match="exactly one chip tier"):
        coder_tier(cfg, 11434)                  # the CPU tier is never the coder


def test_versions_record_tt_model_and_firmware_and_mark_the_rest_todo():
    devices = {"device_info": [{"firmwares": {"fw_bundle_version": "19.15.0.0"}}] * 4}
    run = FakeRun({"tt-model": [(0, "0.1.0\n", "")], "tt-smi": [(0, devices, "")]}, key=lambda a: a[0])
    v = resolve_versions(ServerSpec("org/coder", "container", 8000, "m"), run=run)
    assert v["tt_model"] == "0.1.0" and v["firmware"] == ["19.15.0.0"]
    assert v["tt_metal"].startswith("TODO") and v["vllm"].startswith("TODO")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_parts.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.supervisor'`.

- [ ] **Step 4: Implement**

The import block already lists what Task 9 adds to this module.

Create `orchard/supervisor.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_parts.py`
Expected: PASS (5 tests).

- [ ] **Step 6: Mutation check**

1. A control command acts once: in `orchard/supervisor.py`, replace `        os.replace(self.path, done)` with `        pass`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor_parts.py::test_a_control_command_acts_once_and_is_kept`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/supervisor.py tests/run_fakes.py tests/test_supervisor_parts.py
git commit -m "Add the supervisor's control file, actuator, external stand-in and version record"
```

---

### Task 9: The supervisor loop and its command line

**Files:**
- Modify: `orchard/supervisor.py` (append)
- Test: `tests/test_supervisor.py`

**Interfaces:**
- Consumes: everything above; plan 3's `decide_park`, `Handoff` (`park`, `restore`, `recover_and_restore`), `recover`, `reacquire`, `wait_stopped`, `progress`, `Blocked`, `Budgets`, `NOTE_KEYS`; `Watchdog`, `Ladder` and the detectors; `GozerAdapter` and `ServerControl` (built only by `main` without fakes).
- Produces in `orchard.supervisor`:
  - `Supervisor(*, run_dir, ledger, cfg, model_id, adapter, coder, coder_chips, standin, skills_dirs, inputs=None, extra_env=None, versions=None, http=post_json, probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(), disk_usage=shutil.disk_usage)` with `run() -> int`.
  - Ledger decisions it writes (names other modules read): `coder starting`, `coder started` (`lease`, `server`, `canary`, `canary_after`, `ready_s`), `relaunch the coder under this supervisor's lease`, `stopping the coder`, `coder died; restarting it once`, `tier substituted`, `agent step`, `hardware phase` (`action`), `test lease taken` (`test_lease`), `hardware test started`, `test lease released`, `pause` (`reason`), `resume`, `abort`, `hardware released`, `ready for operator review`. Stage ends carry `result` `pass` (with `evidence`), `escalate`, `fail` or `skipped`, and `reasons`.
  - `parse(argv)`, `build(args, ledger, *, adapter=None, coder=None, versions=None, http=..., probe=..., clock=..., sleep=..., budgets=..., disk_usage=...) -> Supervisor`, `main(argv=None) -> int`. CLI: `python3 -m orchard.supervisor run --model <hf id> --run-dir <dir> --tiers config/tiers.toml --coder-target <pkg> --coder-port <port> --coder-chips <n> [--coder-kind container|bundle] [--coder-profile P] [--coder-image-id I] [--skills-dir D ...] [--input NAME=PATH ...] [--env NAME=VALUE ...] [--gozer PATH]` and `python3 -m orchard.supervisor control --run-dir <dir> pause|resume|abort`.

How the loop decides, in order, each time round: a paused run waits on the control file; no stage left means "ready for operator review" and the hardware is released; a run-wide cap pauses; stage 2 after a full-port delta pauses once; the coder is made to serve under this process's lease (recover an interrupted park, else relaunch an orphaned coder, else restart a dead one once); then the next stage runs. A `Blocked` from anywhere becomes a pause with the reason.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_supervisor.py`:

```python
"""The supervisor run: escalation, pause and resume, abort, disk, the hardware test, the coder."""
import pytest

from fake_model import FakeModel, turn
from fakes import Crash
from orchard.ledger import Ledger
from orchard.supervisor import EXIT_ABORTED, EXIT_READY, EXIT_REFUSED, Control, build, main, parse
from run_fakes import (BOARDS, FILES, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv,
                       bringup, clock, plenty, where, write_tiers)


class Stop(Exception):
    """Ends a test that would otherwise wait for the operator forever."""


class Rig:
    """A machine, two fake model servers (chip tier and CPU tier) and a run directory."""

    def __init__(self, tmp_path, chips=2):
        self.tmp, self.m = tmp_path, Machine()
        self.script = bringup
        self.chip_server = FakeModel(lambda r: self.script(r), models=["qwen-27b"])
        self.cpu_server = FakeModel(bringup, models=["cpu-model"])
        self.chip_server.__enter__()
        self.cpu_server.__enter__()
        self.tiers = write_tiers(tmp_path / "tiers.toml", self.chip_server.endpoint, self.cpu_server.endpoint)
        self.args = parse(argv(tmp_path, self.tiers, self.chip_server.endpoint, chips=chips))
        self.run_dir = tmp_path / "run"
        self.clock, self.usage, self.on_sleep = clock(), plenty, None

    def sleep(self, s):
        self.clock.sleep(s)
        if self.on_sleep:
            self.on_sleep()

    def run(self, pid=100, crash_if=None):
        self.m.owner_pid = pid
        path = self.run_dir / "ledger.jsonl"
        with (CrashingLedger(path, crash_if=crash_if) if crash_if else Ledger(path)) as led:
            sup = build(self.args, led, adapter=MachineAdapter(self.m, owner_pid=pid),
                        coder=MachineCoder(self.m), versions={"tt_model": "test"}, clock=self.clock,
                        sleep=self.sleep, disk_usage=lambda p: self.usage(p))
            return sup.run()

    def entries(self):
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            return led.read()

    def decisions(self):
        return [e["data"] for e in self.entries() if e["event"] == "decision"]

    def ends(self, n):
        return [e["data"] for e in self.entries() if e["event"] == "stage_end" and e["stage"] == n]

    def close(self):
        self.chip_server.__exit__()
        self.cpu_server.__exit__()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def escalation_aware(stage, bad_files):
    """A script that fails `stage` until the context shows the stage was escalated."""
    def script(request):
        if "tools" in request and where(request)[0] == stage:
            if f"stage {stage}: escalate" not in request["messages"][1]["content"]:
                return bringup(request, overrides=bad_files)
        return bringup(request)
    return script


def test_main_writes_the_control_file(tmp_path, capsys):
    assert main(["control", "--run-dir", str(tmp_path), "pause"]) == 0
    assert (tmp_path / "control").read_text() == "pause\n"


def test_main_refuses_a_credential_before_running_anything(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + ["--env", "HF_TOKEN=hf_x", "--gozer", "/nonexistent"]
    assert main(a) == EXIT_REFUSED
    assert "looks like a credential" in capsys.readouterr().err
    assert not (tmp_path / "run" / "ledger.jsonl").exists()      # nothing was recorded or run


# ---- the run --------------------------------------------------------------------------------------

def test_a_failed_gate_escalates_once_to_the_diagnose_tier(rig):
    bad = {(2, "finish"): {"result.json": {"pcc": 0.9, "argmax_match": True,
                                           "evidence": ["stages/2/evidence/hw-test-output.txt"]}}}
    rig.script = escalation_aware(2, bad)
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(2)] == ["escalate", "pass"]
    esc = [e["data"] for e in rig.entries() if e["event"] == "escalate"]
    assert esc[0]["by"] == "stage machine" and "pcc must be" in esc[0]["reasons"][0]
    steps = [d for d in rig.decisions() if d["decision"] == "agent step" and d["escalated"]]
    assert steps and all(d["tier"] == "large" for d in steps)
    assert (rig.run_dir / "stages" / "2.partial-1" / "result.json").exists()   # kept, never deleted


def test_a_second_failure_pauses_the_run_until_the_operator_resumes(rig):
    bad = {(3, "finish"): {"result.json": {"parity": False, "top1": 0.2,
                                           "evidence": ["stages/3/evidence/hw-test-output.txt"]}}}
    rig.script = lambda r: bringup(r, overrides=bad)

    def fixed_and_resumed():
        rig.script = bringup
        Control(rig.run_dir).write("resume")
    rig.on_sleep = fixed_and_resumed
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(3)] == ["escalate", "fail", "pass"]
    pauses = [d for d in rig.decisions() if d["decision"] in ("pause", "resume")]
    assert pauses[0]["reason"].startswith("stage 3 failed after escalation")
    assert pauses[1] == {"decision": "resume", "by": "operator"}


def test_the_escalation_cap_pauses_the_run(rig):
    def bad(n):
        return {(n, "finish"): {"result.json": {"broken": True}}}

    def script(request):
        if "tools" in request and where(request)[0] in (2, 3, 4):
            n = where(request)[0]
            return escalation_aware(n, bad(n))(request)
        return bringup(request)
    rig.script = script
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    pauses = [e for e in rig.entries() if e["event"] == "decision" and e["data"]["decision"] == "pause"]
    assert [(e["stage"], e["data"]["reason"]) for e in pauses] == [
        (4, "3 escalations since the last resume (cap 3)")]   # before stage 4 runs again


def test_a_full_port_pauses_before_stage_2(rig):
    from run_fakes import DELTA
    full = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "path": "full-port"}}}
    rig.script = lambda r: bringup(r, overrides=full)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(0)] == ["pass"] and rig.ends(2) == []
    assert any(d.get("reason", "").startswith("stage 0 found a full port") for d in rig.decisions())


def test_an_operator_pause_waits_for_resume(rig):
    Control(rig.run_dir).write("pause")
    rig.on_sleep = lambda: Control(rig.run_dir).write("resume")
    assert rig.run() == EXIT_READY
    assert {"decision": "pause", "reason": "operator"} in rig.decisions()


def test_abort_stops_the_coder_releases_its_lease_and_stays_aborted(rig):
    Control(rig.run_dir).write("abort")
    assert rig.run() == EXIT_ABORTED
    assert not rig.m.coder_running and rig.m.leases == {}
    assert [d["decision"] for d in rig.decisions()][-3:] == ["abort", "stopping the coder",
                                                             "hardware released"]
    starts = rig.m.coder_starts
    assert rig.run() == EXIT_ABORTED and rig.m.coder_starts == starts


def test_a_stage_short_of_disk_pauses_until_the_operator_resumes(rig):
    from collections import namedtuple
    rig.usage = lambda p: namedtuple("U", "total used free")(0, 0, 2e9)

    def freed():
        rig.usage = plenty
        Control(rig.run_dir).write("resume")
    rig.on_sleep = freed
    assert rig.run() == EXIT_READY
    notes = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert notes == ["stage 1 needs 5.0 GB free on the run directory's disk; 2.0 GB is free"]


def test_a_refused_test_command_fails_the_stage_before_any_hardware_is_used(rig):
    bad = {(2, "prepare"): {**FILES[(2, "prepare")],
                            "hw_test.json": {"command": "tt-smi -r 0", "deadline_s": 60}}}
    rig.script = escalation_aware(2, bad)
    assert rig.run() == EXIT_READY
    first = rig.ends(2)[0]
    assert first["result"] == "escalate" and "the test command is refused" in first["reasons"][0]
    started = [e for e in rig.entries() if e["event"] == "decision" and e["stage"] == 2
               and e["data"]["decision"] == "hardware test started"]
    assert len(started) == 1                    # only the escalated attempt ran a test


def test_the_hardware_test_gets_the_leased_chips_and_no_token(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supervisor_secret")
    monkeypatch.setenv("GH_TOKEN", "ghp_supervisor_secret")
    assert rig.run() == EXIT_READY
    env = (rig.run_dir / "stages" / "2" / "evidence" / "devices.txt").read_text()
    assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B1'])}" in env
    assert "secret" not in env


def test_numbers_from_stage_6_become_labelled_measurements(rig):
    assert rig.run() == EXIT_READY
    got = [(e["data"]["name"], e["data"]["label"]) for e in rig.entries()
           if e["event"] == "measurement" and e["stage"] == 6]
    assert got[-2:] == [("decode", "measured"), ("ttft", "TODO")]   # after the queue-wait measurement


def test_a_relaunched_coder_with_a_different_canary_answer_blocks(rig):
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    rig.m.coder_answer = "41"                   # the coder answers differently after its restart

    def stop():
        raise Stop()
    rig.on_sleep = stop
    with pytest.raises(Stop):
        rig.run(pid=200)
    d = [x["decision"] for x in rig.decisions()]
    assert "relaunch the coder under this supervisor's lease" in d
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == ["the coder's canary answer changed after it was started again"]


def test_a_coder_that_dies_is_restarted_once_and_a_second_death_blocks(rig):
    killed = set()

    def script(request):
        if "tools" in request:
            n, phase = where(request)
            if n in (1, 3) and turn(request) == 0 and n not in killed:
                killed.add(n)
                rig.m.coder_running = False         # the coder dies while the agent works
        return bringup(request)
    rig.script = script

    def stop():
        raise Stop()
    rig.on_sleep = stop
    with pytest.raises(Stop):
        rig.run()
    d = [x["decision"] for x in rig.decisions()]
    assert d.count("coder died; restarting it once") == 1
    blocked = [e["data"]["reason"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert blocked == ["the coder died a second time since the last resume"]
    assert ("reset", "L1") in rig.m.resets        # the chips were reset before the single restart
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py`
Expected: FAIL with `ImportError: cannot import name 'build' from 'orchard.supervisor'`.

- [ ] **Step 3: Implement**

Append to `orchard/supervisor.py`:

```python


# ---- the supervisor -----------------------------------------------------------------------------

class Supervisor:
    def __init__(self, *, run_dir, ledger, cfg, model_id: str, adapter, coder, coder_chips: int,
                 standin, skills_dirs, inputs: dict | None = None, extra_env: dict | None = None,
                 versions: dict | None = None, http=post_json, probe=probe_model, clock=time.time,
                 sleep=time.sleep, budgets: Budgets = Budgets(), disk_usage=shutil.disk_usage):
        self.run_dir = Path(run_dir).resolve()
        self.ledger, self.cfg, self.model_id = ledger, cfg, model_id
        self.adapter, self.coder, self.coder_chips, self.standin = adapter, coder, coder_chips, standin
        self.skills_dirs = [Path(d) for d in skills_dirs]
        self.inputs, self.extra_env = dict(inputs or {}), dict(extra_env or {})
        self.versions = dict(versions or {})
        self.http, self.probe, self.clock, self.sleep = http, probe, clock, sleep
        self.budgets, self.disk_usage = budgets, disk_usage
        self.control = Control(self.run_dir)
        self.actuator = RunActuator(self.control)
        self.guard = RetryGuard()
        self.coder_lease: Lease | None = None
        self.watchdog: Watchdog | None = None
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
        p = run_progress(self.ledger.read())
        if p.aborted or p.finished:
            self._release_all()           # finishes a release a crash interrupted; else does nothing
            return EXIT_ABORTED if p.aborted else EXIT_READY
        if not p.started:
            self.ledger.append("run_start", None, model=self.model_id, versions=self.versions,
                               inputs=self.inputs, coder=self.coder.record(),
                               tiers={k: dict(v) for k, v in self.cfg.tiers.items()})
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
            try:
                self._ensure_coder()
                result = self._run_stage(STAGES[p.next_stage], resuming=p.open_stage == p.next_stage,
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
        # Nothing owns the coder once this process exits, and a lease judged by a dead pid with a
        # device still open would show as a lease that lies. So the run gives the hardware back.
        self._release_all()
        return EXIT_READY

    def _full_port_unacknowledged(self) -> bool:
        """Stage 0 chose a full port, and the run has not yet paused for it."""
        try:
            path = json.loads((self.run_dir / "stages" / "0" / "delta.json").read_text()).get("path")
        except (OSError, ValueError, AttributeError):
            return False
        return path == "full-port" and not any(
            e["event"] == "decision" and e["data"].get("reason") == FULL_PORT for e in self.ledger.read())

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

    def _release_all(self) -> None:
        """Stop the coder and release its lease. Safe to call again: a released lease is skipped."""
        entries = self.ledger.read()
        lease_rec, server_rec, _ = coder_state(entries)
        released = {e["data"].get("lease_id") for e in entries
                    if e["event"] == "decision" and e["data"].get("decision") == "hardware released"}
        if lease_rec is None or lease_rec["lease_id"] in released:
            return
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
                return
            self.adapter.release(lease)
            self.ledger.append("decision", None, decision="hardware released", lease_id=lease.lease_id)
        except (AdapterError, ServerError, OSError) as exc:
            self.ledger.append("notice", None, what="releasing the hardware failed", error=str(exc))

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

    def _start_coder(self, baseline: dict | None, lease: Lease | None = None) -> None:
        if lease is None:
            lease = reacquire(self.adapter, chips=self.coder_chips, who=WHO, reason="coder",
                              ledger=self.ledger, stage=None, wait_budget_s=COLD_BOOT_BUDGET_S,
                              clock=self.clock, sleep=self.sleep)
        # Recorded before the start: a crash during the boot leaves the lease and server in the
        # ledger, so the restart can stop that server and re-lease.
        self.ledger.append("decision", None, decision="coder starting", lease=lease.record(),
                           server=self.coder.record())
        try:
            self.coder.start(lease)
            seconds = self.coder.wait_ready(self.budgets.cold_boot_s)
            answer = self.coder.ask(RUN_CANARY_PROMPT)
        except (ServerError, CanaryError, OSError) as exc:
            self._block(None, f"the coder did not start and answer; nothing is retried: {exc}")
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
                       NoNewEvidence(), StageOverBudget({spec.number: spec.budget_s})],
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
        stage_dir, resumed = open_stage_dir(self.run_dir, spec, resuming=resuming, ledger=self.ledger)
        self.ledger.append("stage_start", n, escalated=escalated, resumed=resumed)
        self.actuator.clear()
        self.watchdog = self._watchdog(spec)
        status, reasons, gate = self._stage_body(spec, stage_dir, escalated, resumed)
        return self._end_stage(spec, stage_dir, status, reasons, gate, escalated)

    def _stage_body(self, spec, stage_dir: Path, escalated: bool, resumed: bool):
        if spec.boards == 0:
            out = self._step(spec, "run", stage_dir, escalated, resumed)
            if out.status != "done":
                return out.status, [f"the agent step ended: {out.status} {out.detail}".strip()], None
        else:
            if not (resumed and (stage_dir / "test-result.json").is_file()):
                out = self._step(spec, "prepare", stage_dir, escalated, resumed)
                if out.status != "done":
                    return out.status, [f"the prepare step ended: {out.status} {out.detail}".strip()], None
                test, problems = self._read_test(stage_dir)
                if problems:
                    return "fail", problems, None
                self._hardware_phase(spec, stage_dir, test)
            out = self._step(spec, "finish", stage_dir, escalated, resumed)
            if out.status != "done":
                return out.status, [f"the finish step ended: {out.status} {out.detail}".strip()], None
        if spec.number == 8:
            bundle = stage_dir / "bundle"
            bundle.mkdir(exist_ok=True)
            shutil.copyfile(self.ledger.path, bundle / "ledger.jsonl")
        gate = spec.gate(stage_dir, self.run_dir)
        return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)

    def _end_stage(self, spec, stage_dir: Path, status: str, reasons: list, gate, escalated: bool) -> str:
        n = spec.number
        if status == "pass":
            ev = [evidence_record(self.run_dir, self.run_dir / rel) for rel in sorted(set(gate.evidence))]
            ev.append(evidence_record(self.run_dir, stage_dir / spec.gate_file))
            if n == 6:
                self._record_numbers(stage_dir)
            self.ledger.append("stage_end", n, result="pass", evidence=ev)
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
        system, user = build_messages(spec=spec, phase=phase, run_dir=self.run_dir,
                                      stage_dir=stage_dir, skill_path=skill, refs=refs,
                                      facts=facts_from(run_progress(entries).run_start, self.run_dir),
                                      entries=entries, resumed=resumed)
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
        return step.run(system, user)

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
        self.ledger.append("decision", n, decision="test lease taken", test_lease=lease.record())
        self._run_test(spec, stage_dir, test, lease)
        try:
            self.adapter.release(lease)        # the lease tool resets the board as it releases
        except AdapterError as exc:
            self._block(n, f"releasing the test lease failed: {exc}", lease_id=lease.lease_id)
        self.ledger.append("decision", n, decision="test lease released", lease_id=lease.lease_id)

    def _run_test(self, spec, stage_dir: Path, test: dict, lease: Lease) -> dict:
        n = spec.number
        chips = list(lease.chips[:spec.boards * CHIPS_PER_BOARD])
        env = agent_env(self.run_dir, extra=self.extra_env)
        env["TT_VISIBLE_DEVICES"] = ",".join(chips)
        deadline = min(float(test["deadline_s"]), spec.budget_s)
        out_path = stage_dir / "evidence" / "hw-test-output.txt"
        self.ledger.append("decision", n, decision="hardware test started", command=test["command"],
                           deadline_s=deadline, chips=chips, lease_id=lease.lease_id)
        t0 = self.clock()
        with open(out_path, "wb") as out:
            try:
                code, timed_out = spawn_checked(test["command"], self.run_dir, env, deadline, out)
            except Denied as exc:
                code, timed_out = None, False
                out.write(f"refused: {exc}\n".encode("utf-8"))
        result = {"command": test["command"], "returncode": code, "timed_out": timed_out,
                  "seconds": round(self.clock() - t0, 3), "chips": chips,
                  "output": evidence_record(self.run_dir, out_path)}
        marker = stage_dir / "test-result.json"
        tmp = stage_dir / "test-result.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, marker)         # the resume marker appears whole or not at all
        self.ledger.append("evidence", n, what="hardware test", returncode=code, timed_out=timed_out,
                           **evidence_record(self.run_dir, marker))
        return result


# ---- the command line ---------------------------------------------------------------------------

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
    r.add_argument("--gozer", default="gozer")
    c = sub.add_parser("control", help="send pause, resume or abort to a running supervisor")
    c.add_argument("--run-dir", required=True)
    c.add_argument("command", choices=Control.COMMANDS)
    return p.parse_args(argv)


def build(args, ledger, *, adapter=None, coder=None, versions=None, http=post_json,
          probe=probe_model, clock=time.time, sleep=time.sleep, budgets=Budgets(),
          disk_usage=shutil.disk_usage) -> Supervisor:
    """A Supervisor from parsed `run` arguments. Tests pass fakes for the machine."""
    # Everything that can be refused is checked before any external command runs.
    cfg = load(args.tiers)
    tier = coder_tier(cfg, args.coder_port)
    run_dir = Path(args.run_dir).resolve()
    inputs, extra_env = pairs(args.input, "input"), pairs(args.env, "env")
    agent_env(run_dir, extra=extra_env)
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
    if versions is None and not run_progress(entries).started:
        versions = resolve_versions(spec)       # a resumed run keeps the versions it started with
    return Supervisor(run_dir=run_dir, ledger=ledger, cfg=cfg, model_id=args.model, adapter=adapter,
                      coder=coder, coder_chips=args.coder_chips,
                      standin=ExternalStandIn(cpu["endpoint"], cpu["model"], http=http),
                      skills_dirs=[SKILLS_DIR, *args.skills_dir], inputs=inputs,
                      extra_env=extra_env, versions=versions, http=http, probe=probe,
                      clock=clock, sleep=sleep, budgets=budgets, disk_usage=disk_usage)


def main(argv=None) -> int:
    args = parse(argv)
    if args.cmd == "control":
        Control(args.run_dir).write(args.command)
        print(f"wrote {args.command!r} to {Path(args.run_dir) / 'control'}")
        return 0
    run_dir = Path(args.run_dir)
    try:
        with Ledger(run_dir / "ledger.jsonl") as ledger:
            code = build(args, ledger).run()
    except (TierConfigError, ValueError, LedgerLocked, LedgerCorrupt) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print({EXIT_READY: "ready for operator review", EXIT_ABORTED: "aborted"}.get(code, code))
    return code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py tests/test_supervisor_parts.py`
Expected: PASS (19 tests).

- [ ] **Step 5: Mutation checks**

1. The hardware test's environment: in `orchard/supervisor.py`, replace `        env = agent_env(self.run_dir, extra=self.extra_env)` with `        env = dict(os.environ)`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_the_hardware_test_gets_the_leased_chips_and_no_token`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. The relaunch canary comparison: in `orchard/supervisor.py`, replace `            if not result.match:` with `            if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_a_relaunched_coder_with_a_different_canary_answer_blocks`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. A coder is restarted only once: in `orchard/supervisor.py`, replace `        if run_progress(self.ledger.read()).coder_deaths >= 1:` with `        if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_a_coder_that_dies_is_restarted_once_and_a_second_death_blocks`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
4. A stage escalates once: in `orchard/supervisor.py`, replace `        if not escalated:` with `        if True:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_a_second_failure_pauses_the_run_until_the_operator_resumes`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
5. Abort releases the lease: in `orchard/supervisor.py`, replace `            self.adapter.release(lease)` with `            pass`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_abort_stops_the_coder_releases_its_lease_and_stays_aborted`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
6. The run-wide caps: in `orchard/supervisor.py`, replace `            cap = budget_cap(p, self.clock())` with `            cap = None`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_the_escalation_cap_pauses_the_run`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
7. A full port pauses before stage 2: in `orchard/supervisor.py`, replace `            if p.next_stage == 2 and self._full_port_unacknowledged():` with `            if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_supervisor.py::test_a_full_port_pauses_before_stage_2`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

In mutation 5, change the line in `_release_all`, which has no comment after it. The release in `_hardware_phase` carries a comment and stays as it is.

- [ ] **Step 6: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add orchard/supervisor.py tests/test_supervisor.py
git commit -m "Add the supervisor loop: coder lease, stages under the watchdog, hardware tests, escalation, pause and abort"
```

---

### Task 10: End to end, with the supervisor killed after every ledger event

**Files:**
- Test: `tests/test_run_e2e.py`

**Interfaces:**
- Consumes: `build`, `parse`, `EXIT_READY` (Task 9); `tests/run_fakes.py` (Task 8); `orchard.stages.run_progress`; `orchard.ledger.replay_state`.
- Produces: the spec section 13 fault-injection check for the whole loop, in both machine layouts.

This task adds tests and no code. The tests pass on the code from Task 9. The mutation steps show that they can fail.

- [ ] **Step 1: Write the tests**

Create `tests/test_run_e2e.py`:

```python
"""End to end: a scripted Hemmingway-1 bring-up from stage 0 to the operator bundle, against a fake
machine and fake model servers, with the supervisor killed after every ledger event (spec section 13).

The two machine layouts are the operator's: the coder on all four chips (every hardware stage
parks it) and the coder on two chips (every hardware stage uses the free board).
"""

import pytest

from fake_model import FakeModel
from fakes import Crash
from orchard.ledger import Ledger, replay_state
from orchard.stages import run_progress
from orchard.supervisor import EXIT_READY, build, parse
from run_fakes import (BOARDS, CrashingLedger, Machine, MachineAdapter, MachineCoder, argv, bringup,
                       clock, plenty, write_tiers)

FIRST, SECOND = 100, 200          # supervisor pids before and after the kill


@pytest.fixture(scope="module")
def servers():
    with FakeModel(bringup, models=["qwen-27b"]) as chips, FakeModel(bringup, models=["cpu-model"]) as cpu:
        yield chips, cpu


def run(base, servers, machine, *, chips, pid, crash_after=0):
    chip_server, cpu_server = servers
    tiers = base / "tiers.toml"
    base.mkdir(parents=True, exist_ok=True)
    if not tiers.exists():
        write_tiers(tiers, chip_server.endpoint, cpu_server.endpoint)
    args = parse(argv(base, tiers, chip_server.endpoint, chips=chips))
    machine.owner_pid = pid
    c = clock()
    path = base / "run" / "ledger.jsonl"
    with (CrashingLedger(path, crash_after) if crash_after else Ledger(path)) as led:
        return build(args, led, adapter=MachineAdapter(machine, owner_pid=pid),
                     coder=MachineCoder(machine), versions={"tt_model": "test"}, clock=c,
                     sleep=c.sleep, disk_usage=plenty).run()


def entries(base):
    with Ledger(base / "run" / "ledger.jsonl") as led:
        return led.read()


def final_state(base):
    """What must be the same however many times the supervisor died on the way."""
    es = entries(base)
    p = run_progress(es)
    last = {}
    for e in es:
        if e["event"] == "stage_end":
            last[e["stage"]] = e["data"]["result"]
    bundle = base / "run" / "stages" / "8" / "bundle"
    return {"finished": p.finished, "done": p.done, "parked": replay_state(es)["parked"],
            "paused": p.paused, "last_result": last,
            "bundle": {f.name: f.read_text() for f in sorted(bundle.iterdir()) if f.name != "ledger.jsonl"},
            # The stage 6 result numbers (they carry an evidence key); handoff timings vary with kills.
            "numbers": sorted({e["data"]["name"] for e in es if e["event"] == "measurement"
                               and e["stage"] == 6 and "evidence" in e["data"]})}


def test_with_the_coder_on_four_chips_every_hardware_stage_parks_it(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=4, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    for n in (2, 3, 4, 5, 6):
        seq = [(e["event"], e["data"].get("step") or e["data"].get("decision")) for e in es if e["stage"] == n]
        test = seq.index(("decision", "hardware test started"))
        assert seq.index(("park", "reset")) < test < seq.index(("restore", "reset")), n
        assert ("restore", "resumed") in seq[test:], n
        env = (tmp_path / "run" / "stages" / str(n) / "evidence" / "devices.txt").read_text()
        assert f"TT_VISIBLE_DEVICES={','.join(BOARDS['B0'])}\n" in env       # one board of the four chips
    s = final_state(tmp_path)
    assert s["finished"] and s["done"] == (0, 1, 2, 3, 4, 5, 6, 7, 8) and not s["parked"]
    assert s["last_result"][7] == "skipped"
    assert not m.coder_running and m.leases == {}           # the finished run gave the hardware back


def test_with_the_coder_on_two_chips_the_free_board_is_used_and_nothing_parks(tmp_path, servers):
    m = Machine()
    assert run(tmp_path, servers, m, chips=2, pid=FIRST) == EXIT_READY
    es = entries(tmp_path)
    assert not [e for e in es if e["event"] in ("park", "restore")]
    taken = [e["data"]["test_lease"]["chips"] for e in es if e["data"].get("decision") == "test lease taken"]
    assert taken == [list(BOARDS["B1"])] * 5
    released = [e for e in es if e["data"].get("decision") == "test lease released"]
    assert len(released) == 5 and m.leases == {}
    assert final_state(tmp_path)["finished"]


@pytest.mark.parametrize("chips", [4, 2])
def test_a_kill_after_any_ledger_event_reaches_the_same_final_state(tmp_path, servers, chips):
    ref = tmp_path / "reference"
    assert run(ref, servers, Machine(), chips=chips, pid=FIRST) == EXIT_READY
    want = final_state(ref)
    total = len(entries(ref))
    for k in range(1, total + 1):
        base = tmp_path / f"kill-{k:03d}"
        m = Machine()
        with pytest.raises(Crash):
            run(base, servers, m, chips=chips, pid=FIRST, crash_after=k)
        assert run(base, servers, m, chips=chips, pid=SECOND) == EXIT_READY, k
        assert final_state(base) == want, k
        # The hardware is given back, including any lease the killed supervisor held.
        assert not m.coder_running and m.leases == {}, (k, m.leases)
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_run_e2e.py --durations=3`
Expected: PASS (4 tests). The 4-chip kill test runs about 160 kills in about 25 s; the 2-chip one about 105 in about 12 s. Record the two durations and the entry counts in the task report (`len(entries(ref))` in each).

- [ ] **Step 3: Mutation checks**

1. A park when no board is free: in `orchard/supervisor.py`, replace `        if d.park_needed:` with `        if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_run_e2e.py::test_with_the_coder_on_four_chips_every_hardware_stage_parks_it`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
2. An orphaned coder is relaunched: in `orchard/supervisor.py`, replace `            self._relaunch_coder(lease, canary)` with `            self.coder_lease = lease`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_run_e2e.py::test_a_kill_after_any_ledger_event_reaches_the_same_final_state`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.
3. An interrupted park is recovered: in `orchard/supervisor.py`, replace `        if hp.phase != "idle":` with `        if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_run_e2e.py::test_a_kill_after_any_ledger_event_reaches_the_same_final_state`. Expected: FAIL. Restore the line, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`), run it again: PASS.

Mutation 1 does not hang: with no park, the test lease waits in the fake queue, the wait blocks and pauses the run, and `StuckClock` fails the test once a simulated day has passed.

- [ ] **Step 4: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add tests/test_run_e2e.py
git commit -m "Run a scripted bring-up end to end and kill the supervisor after every ledger event"
```

---

### Task 11: The Hemmingway-1 runbook entry and the documents

**Files:**
- Modify: `docs/runbooks/hardware-validation.md` (append), `README.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-10-01-orchard-design.md` (status, sections 4, 9 and 10 only)

**Interfaces:**
- Consumes: the results recorded in Tasks 3, 9 and 10.
- Produces: a runbook entry the controller can follow, and documents that match the code. Nobody runs the runbook entry in this task.

- [ ] **Step 1: Get the facts to write down**

Run: `python3 -m pytest -q -p no:cacheprovider -rs 2>&1 | tail -4`
Expected: 962 passed and 1 skipped (the opt-in replay). A different count means a task added or dropped tests; say which in the README and the log.

- [ ] **Step 2: Append the runbook entry**

Append to `docs/runbooks/hardware-validation.md`:

```markdown

## Hemmingway-1 run (plan 4)

Purpose: the first bring-up driven by the harness and not by Claude by hand. The supervisor takes
`Altworld/Hemmingway-1` (a creative-writing fine-tune of Qwen3.8-27B with the same architecture)
from stage 0 to the operator bundle. It parks the large coder for every hardware stage, records
everything in a ledger, and stops at "ready for operator review". It publishes nothing. The run
also checks stage 0 against the hand-written reference answer.

Who runs it: the controller, with the operator's agreement. An implementer does not. It takes all
four chips for hours.

Before running:
- The operator has answered the open decision about the command runner (spec section 10 and
  section 14, item 5). Do not start without that answer.
- The operator confirms which 4-chip package serves the large tier. `config/tiers.toml` names the
  model (`Qwen/Qwen3.8-27B` on port 8000) and no package. The installed 4-chip candidate is
  `mando2222/qwen3.8-27b-dflash2-p300x2-q4kv` (container, profile `batch8-dflash2`). The model id
  that package serves must equal the large tier's `model`, because the supervisor checks
  `/v1/models` for it. If it differs, the operator edits `config/tiers.toml`.
- `gozer status` shows all four chips `FREE` and no other lease. No `GOZER_*` variable is set.
- ollama serves the CPU tier: `curl -s http://127.0.0.1:11434/v1/models` lists `qwen3-coder:30b`.
- Nothing listens on port 8000: `ss -ltn "( sport = :8000 )"` prints only its header.
- The run directory is on `/mnt/bonus` (404 GB free on 2026-10-02). The root disk is 99% full,
  and stages 2 to 6 each require 40 GB free (stage 4 requires 80 GB).
- Do not give the harness the reference answer, and do not name the base model's tensor cache
  as an input. Stage 0 must find both on its own.

Run, from the repo root:

    python3 -m orchard.supervisor run \
      --model Altworld/Hemmingway-1 \
      --run-dir /mnt/bonus/models/hemmingway-1/orchard-run-1 \
      --tiers config/tiers.toml \
      --coder-target mando2222/qwen3.8-27b-dflash2-p300x2-q4kv --coder-kind container \
      --coder-profile batch8-dflash2 --coder-port 8000 --coder-chips 4 \
      --skills-dir /home/ttuser/code/skills/plugins/tt-model-bringup/skills \
      --input model=/mnt/bonus/models/hemmingway-1/hf/hub/models--Altworld--Hemmingway-1/snapshots/1a5f363a3dd2d1cc456c28b8abbb403b9555efaf \
      --input base=/mnt/bonus/models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
      --env HF_HOME=/mnt/bonus/models/hemmingway-1/hf --env HF_HUB_OFFLINE=1

What happens: the supervisor records the run and the versions it can read (`tt-model --version`,
the firmware from `tt-smi -s`; tt-metal and vLLM are recorded as TODO). It takes a 4-chip lease
under its own pid and starts the coder (a cold boot is about 30 min, budget 45 min). Stages 0 and 1
run on the large server; the small tier is not serving, and the ledger records each substitution.
Each of stages 2 to 6 runs a prepare step, parks the coder (stand-in canary on ollama, stop, reset),
runs the agent's test command on board 0's two chips, restores the coder (reset, start, canary
compared with the pre-park answer) and runs a finish step. Each reset measured 41.7 s and a warm
restart 2 to 3 min. The total run time is not measured. Stage 7 is recorded as skipped. Stage 8
writes the bundle, the supervisor scrubs it, and then it stops the coder and releases the four
chips. Restart the operator's own coder afterwards if it is wanted.

Watch and steer, from another shell:

    tail -n 5 /mnt/bonus/models/hemmingway-1/orchard-run-1/ledger.jsonl
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/hemmingway-1/orchard-run-1 pause
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/hemmingway-1/orchard-run-1 resume
    python3 -m orchard.supervisor control --run-dir /mnt/bonus/models/hemmingway-1/orchard-run-1 abort

A paused run holds its leases and waits. Read the last `notice` and `decision` entries before
resuming. Abort stops the coder and releases the lease. If the supervisor dies, run the same
`run` command again: it replays the ledger, re-leases the coder under its new pid, and resumes.

Compare stage 0 with the reference answer once stage 0 has passed:

    python3 -m orchard.stages compare-delta \
      /mnt/bonus/models/hemmingway-1/orchard-run-1/stages/0/delta.json \
      /mnt/bonus/models/hemmingway-1/work/stage0-reference.md

Exit 0 means every difference area and hazard in the reference is covered and the path agrees
(`weights-only`). The command compares areas only. Read each finding against the reference by hand,
in particular the tokenizer's combining-mark difference and the tensor-cache hazard.

Stop conditions: a pause whose reason you cannot explain, a `blocked` notice from a park or
restore, a canary that changed, or any sign that something was published. Never use `--force`,
never run `tt-smi -r` by hand, and never run the commands in `PUBLISH_COMMANDS.txt`; they are for
the operator.

Record afterwards: the run directory, the exit code, each stage's result, the compare-delta output
and your reading of the findings, every number in stage 6 with its label, every pause with its
reason, and the wall time.
```

- [ ] **Step 3: Update `README.md`**

Replace in `README.md`:

```markdown
| Supervisor loop, stage state machine, model proxy, new bring-up skills, operator bundle | designed, not started (plan 4) |
```

with:

```markdown
| Stage machine, agent steps and supervisor loop (`orchard/stages.py`, `orchard/agent.py`, `orchard/context.py`, `orchard/supervisor.py`, `orchard/scrub.py`) | built, tested against a fake machine and fake model servers, including a kill after every ledger event; not yet run on hardware (Hemmingway-1 entry in `docs/runbooks/hardware-validation.md`) |
| Stage skills (`orchard/skills/`: delta-triage, reference-gate, serving-check, operator-bundle) | local drafts that live in tt-orchard |
| Model proxy for agents the supervisor did not launch, stage 7 (package and image build) | designed; no code yet |
```

Replace in `README.md`:

```markdown
The suite has 854 passing tests and 1 skipped, and runs without hardware or network.
```

with:

```markdown
The suite has 962 passing tests and 1 skipped, and runs without hardware or network. 854 of them predate plan 4.
```

Then add a "What is built" bullet for plan 4, in the style of the existing bullets: "The supervisor (`python3 -m orchard.supervisor run ...`) runs stages 0 to 6 and 8 of a weights-only bring-up with local models, parks the coder for hardware stages, records every step in the ledger, and stops at ready for operator review. It never publishes."

- [ ] **Step 4: Update `CLAUDE.md`**

Replace in `CLAUDE.md`:

```markdown
Plan 4 is not started.
```

with:

```markdown
Plan 4 (stage machine, agent steps, supervisor loop, draft stage skills) is implemented on branch
`plan4-stage-machine`. The Hemmingway-1 run in the runbook has not been run.
```

Replace in `CLAUDE.md`:

```markdown
- The watchdog acts only through an injected actuator, which plan 4 supplies. Each ladder rung is
  written to the ledger before it is acted on, so a restart cannot repeat a rung.
```

with:

```markdown
- The watchdog acts only through an injected actuator, which plan 4 supplies. Each ladder rung is
  written to the ledger before it is acted on, so a restart cannot repeat a rung.
- Plan 4 (2026-10-02): the supervisor starts the coder itself under its own owner-pid lease, because a
  lease cannot pass from the controller's pid to the supervisor's. It stops the coder and releases the
  lease at the end of the run and on abort. The CPU tier (ollama) is an external stand-in that the
  operator runs.
- A hardware stage is three parts: a prepare step writes `hw_test.json` and `handoff.json`, the
  supervisor runs that one command under the lease (parking the coder when no board is free), and a
  finish step writes the result from the test output. `test-result.json` is the resume marker.
- When the named tier is down, a chip tier with the same model serves the step, and the ledger says
  so. On this box only the 4-chip large server runs, so it serves every step.
- Agent shells get an allow-listed environment with HOME inside the run directory. Read-only mounts
  are not built.
```

Replace in `CLAUDE.md`:

```markdown


## Notable moments
```

with:

```markdown

- 2026-10-02: plan 4 written. Prompt: the smallest loop that lets the harness bring up a model whose
  architecture matches a supported one, stage 0 to 6 plus a minimal stage 8 bundle, first target
  Altworld/Hemmingway-1. Every code block was run from the plan text in a scratch copy before hand-over.
  Found while checking it: counting every coder start against a cap paused the run at stage 5, because
  each park restarts the coder; the cap now counts cold boots (a start slower than 10 min). The run
  also left the coder serving under a dead pid's lease after it finished, so the run now releases the
  hardware at the end.

## Notable moments
```

Add to the log any skip, surprise or count recorded in Tasks 3, 9 and 10.

- [ ] **Step 5: Update the spec (status, sections 4, 9 and 10 only)**

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
Plan 1 (the core library: ledger, command runner, tier config, sizing tool) is implemented through
Task 4. Task 5 of plan 1 (running the sizing tool on this machine) and plans 2 to 4 are not
implemented. Every number below is either cited to a
```

with:

```markdown
Plans 1 to 4 are implemented. Not yet done: plan 1 Task 5 (the sizing tool on this machine), plan 3's
park check on hardware, and plan 4's Hemmingway-1 run (docs/runbooks/hardware-validation.md). Every
number below is either cited to a
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
    adapters/gozer.py, adapters/single_tenant.py
```

with:

```markdown
    adapters/gozer.py, adapters/single_tenant.py
    agent.py                agent steps: environment, tools, model calls (plan 4)
    context.py              the fresh context for one agent step (plan 4)
    supervisor.py           the run loop and its command line (plan 4)
    scrub.py                the bundle scrub (plan 4)
    skills/                 draft stage skills until they move to tt-model-bringup (plan 4)
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
- Events: `run_start`, `stage_start`, `stage_end`, `park`, `restore`, `retry`, `escalate`, `notice`,
  `measurement`, `decision`.
```

with:

```markdown
- Events: `run_start`, `stage_start`, `stage_end`, `park`, `restore`, `retry`, `escalate`, `notice`,
  `measurement`, `decision`, `evidence` (an evidence file's run-relative path and sha256; plan 4).
```

Replace in `docs/superpowers/specs/2026-10-01-orchard-design.md`:

```markdown
directory. The runner is one layer among these.
```

with:

```markdown
directory. The runner is one layer among these. Plan 4 built the first outer layer: agent shells get
an allow-listed environment with no tokens and a HOME inside the run directory. Read-only mounts are
not built, so files outside the run directory stay readable and writable by the run's user.
```

Do not change any other section.

- [ ] **Step 6: Sweep the new text and the new code comments**

Run: `git diff -U0 README.md CLAUDE.md docs | grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what'`
Then: `git diff --name-only --diff-filter=A plan3-supervisor-behavior -- orchard tests | xargs grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what'`
Expected: no output from either. Rewrite any hit as a plain statement.

- [ ] **Step 7: Run the whole suite and commit**

Run: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (1 skipped: the opt-in replay).

```bash
git add docs/runbooks/hardware-validation.md README.md CLAUDE.md docs/superpowers/specs/2026-10-01-orchard-design.md
git commit -m "Document plan 4 and add the Hemmingway-1 runbook entry"
```

---

## Hand-off notes

- What plan 4 does not do: streaming; a proxy for agents the supervisor did not launch (their transcripts are still only read); new model code for a full port (the run pauses before stage 2); stage 7; the `stage-review` skill; read-only mounts; killing a hardware test process that outlives a supervisor crash. Such a process keeps its board: inside a park the restore's stop check blocks the run for the operator, and outside a park the restarted stage finds that board busy and parks the coder or waits for a lease.
- The gates check shape and evidence presence. They cannot tell whether a claim is true. The operator reviews the bundle.
- tt-metal and vLLM versions are recorded as TODO, because they live inside the coder image. `tt-model --version` and the firmware bundle from `tt-smi -s` are recorded.
- The watchdog thresholds are plan 3's in-sample defaults. The first real run's ledger and agent transcripts (`stages/<n>/log/`) are the data to check them against.
- The open decision about an adversarial review of the command runner (spec section 14, item 5) is still open. The runbook entry says not to start without the operator's answer.
- The draft skills live in tt-orchard (`orchard/skills/`). They stay there.
