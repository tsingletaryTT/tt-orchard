# tt-orchard: lease adapters, park and restore, watchdog (plan 3 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the supervisor the behavior it needs around the hardware and around its agents. That means lease adapters for tt-gozer and for a machine with no lease tool, the park and restore sequence that swaps the coder off a board and back, and the watchdog that detects model loops and answers them with a capped response ladder.

**Architecture:** Three layers, each in its own modules. `orchard/adapters/` hides the lease tool behind one interface (`LeaseAdapter`). `orchard/server.py` and `orchard/canary.py` start, stop, check and question a model server. `orchard/handoff.py` drives spec section 6 as a function-driven state machine. Each step writes a `park` or `restore` ledger entry, so `progress()` can replay the ledger and say which step completed, and `recover()` compares that with the machine after a restart (the machine wins). `orchard/watchdog.py` holds the normalised `Event` stream, the detectors and the response ladder. `orchard/transcripts.py` turns a qwen-code transcript into that stream. This plan does not cover the stage state machine, agent launching, the stage test inside a park (spec section 6, step 4), or the request-rewriting model proxy. Those are plan 4. The watchdog engine therefore works on a normalised event stream and calls an injected `Actuator` for nudge, escalate and pause. Plan 4 supplies the real actuator and the proxy that feeds the events. Plan 4 also re-leases a coder that is serving when the supervisor restarts outside a park, including after a park that was abandoned before the coder was told to stop, and restarts a model server that dies (spec section 10). The spec's `gozer env <lease>` and the grant's `env` field come from the same gozer function (`Keymaster.env_for`), so the plan uses the grant's `env`.

**Tech Stack:** Python 3.12 standard library only (`dataclasses`, `subprocess`, `json`, `urllib.request`, `http.server`), pytest. Tests use a fake command runner, fake proc trees and fake machine objects. Two test files run real binaries against fake state: the gozer CLI with fake roots (Task 4, Task 15) and `ps`/`pgrep`/`ss`/`curl` against a local fake HTTP server (Task 7).

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md`, sections 3, 4, 6, 7, 9, 10, 12 and 13. Gozer facts are from `~/code/tt-gozer/gozer/cli.py` (read 2026-10-02; committed `main` is gozer 0.3.2 and carries the `reset` command and the owner-pid ownership code; the working tree has an uncommitted 0.3.3 version bump) and the `gozer-park` skill.

**Checked before hand-over:** every code block below was assembled into a scratch copy of the repo (outside the repo) and run on 2026-10-02, first for the original plan and again after the review revision (`.superpowers/plan3-review.md`). After the revision all 839 tests passed (838 plus the opt-in replay); the gozer contract and park-check tests ran against the real gozer CLI with fake roots. The first run found a bug, fixed in Task 15 (a blocked signal mask inherited by the fake servers). The mutation steps added by the revision were each run and each named test failed as the plan says; two weak ones were found that way and strengthened. The implementer still runs every step; this check does not replace them.

## Global Constraints

- Work in `/home/ttuser/code/tt-orchard`. Task 0 commits on `main` and then creates the branch `plan3-supervisor-behavior`; every later task commits on that branch. The repo is local only. Never run `git push` and never create a remote.
- Python 3.12, standard library only. Every new module starts with `from __future__ import annotations` and a docstring that says what the module owns. Comments explain why.
- The unit suite uses no hardware. No test opens `/dev/tenstorrent/*`, runs `tt-smi`, runs `tt-model`, or runs gozer against the real gozer state. gozer runs in tests only with `GOZER_ROOT`, `GOZER_SYSFS_ROOT`, `GOZER_PROC_ROOT`, `GOZER_HISTORY_ROOT` and `GOZER_RESET_CMD` all pointed into `tmp_path`.
- Do not run `orchard/park_check.py` or `orchard/hardware_check.py` against hardware, and do not run any gozer command that changes state on the real box. The controller runs the hardware check (Task 15).
- Normal tests read no files outside the repo. The transcript replay over `~/.qwen/projects` is opt-in (`ORCHARD_REPLAY=1`) and skips loudly otherwise.
- Every test that calls `park_check.main` passes an explicit `--gozer`: a stub, or a wrapper around the real gozer with all five `GOZER_*` roots faked. A test that expects no gozer call passes a tripwire stub that writes a marker and exits non-zero, and asserts the marker is absent. Tests build `GozerAdapter` only with a `FakeRun` or such a wrapper, never with the default `gozer` on PATH.
- Tests in `tests/test_single_tenant.py` build `SingleTenantAdapter` only with a `FakeRun`. The `adapter()` helper defaults `run` to an unscripted `FakeRun()`, which fails the test if it is called. A test that needs a reset call passes its own scripted `FakeRun`. Otherwise a regression in a guard runs a real `tt-smi -r`.
- Do not change existing modules, except `orchard/ledger.py`'s `replay_state` (Task 8) and the documents in Task 16. Do not import from `orchard/hardware_check.py`, except `make_out_dir` in `orchard/park_check.py`.
- The adapter never passes `--force` to gozer, never calls `gozer wait` (it grants a lease with no owner pid), and nothing in this plan runs `tt-smi -r`.
- Retry rule (spec section 3): never retry the same call with the same inputs more than once. A nudge changes the prompt. A queue claim is a poll of gozer's protocol; it is not a retry of a failed call.
- Every default value lives in `orchard/defaults.py` as one named constant, with a comment that cites where it was measured or says that it is a choice and was not measured.
- Every guard gets a mutation step: remove the guard, run the named test, watch it fail, restore the guard (spec section 13). After each restore, delete the bytecode (`find . -name __pycache__ -prune -exec rm -rf {} +`) before the confirming run. A mutation that keeps the file size, restored within the same second, leaves the mutated `.pyc` in use; this happened while checking this plan, and the restored tests failed until the cache was cleared.
- Run the whole suite before each commit: `cd /home/ttuser/code/tt-orchard && python3 -m pytest -q -p no:cacheprovider`. The baseline is 607 passed in about 160 s.
- Commit messages are plain English, one idea per commit. Writing rules for docs and comments: state the finding; short sentences; no "X, not Y" framing; no aphoristic closers; no metaphors that stand in for a claim.

## Review Focus

1. A stop that docker confirms while a leftover vLLM EngineCore worker still holds the chip. Expected: no reset runs, the run blocks after the quiet wait, and the ledger names the holder state (Task 7: `test_bundle_worker_left_in_the_group_is_not_stopped`; Task 9: `test_a_leftover_worker_blocks_the_reset`).
2. A reset refused with exit 15 because a `HELD-FOREIGN` container still holds the board. Expected: the stop is checked again, the reset is retried once, then the stage blocks; `--force` is never used. A reset that ran and failed (exit 17) blocks at once (Task 9: `test_a_refused_reset_is_retried_once_then_blocks`, `test_a_failed_reset_blocks_at_once`).
3. The supervisor killed between `tt-model stop` and the reset. Expected: the restart finds the coder stopped and the lease orphaned, takes a new lease, resets, restores the coder and compares the canary (Task 11: `test_killed_between_tt_model_stop_and_the_reset`, plus the crash-after-every-ledger-event test, which also checks that no coder stops before the stand-in answered and that a park killed before its stop is abandoned with the coder left up).
4. A queue ticket that expires (gozer drops tickets after one hour) while the supervisor waits. Expected: one new ticket, with a ledger notice that the place in the queue was lost; a second expiry blocks (Task 10: `test_an_expired_ticket_is_replaced_once_and_the_lost_place_recorded`, `test_a_second_expiry_blocks`).
5. A canary answer after the restore that differs only in whitespace. Expected: the stage blocks, because the comparison is exact for greedy decoding, and the evidence says `whitespace_only: true` so a person can judge it quickly (Task 6: `test_whitespace_only_difference_is_flagged`; Task 9: `test_a_whitespace_only_canary_difference_still_blocks_and_says_so`).

## File Structure

| File | Responsibility |
|---|---|
| `orchard/defaults.py` | every timing, budget and threshold, one constant each, with its source |
| `orchard/commands.py` | `CommandResult`, `run_command`: the one place supervisor code starts an external program |
| `orchard/adapters/__init__.py` | `Lease`, `ChipState`, the adapter exceptions, the `LeaseAdapter` protocol, `boards_of` |
| `orchard/adapters/gozer.py` | `GozerAdapter`: the gozer CLI through an injected `run` |
| `orchard/adapters/single_tenant.py` | `SingleTenantAdapter`, `device_holders` |
| `orchard/canary.py` | the canary chat call (`ask`) and the exact comparison (`compare`) |
| `orchard/fake_server.py` | a small OpenAI-shaped HTTP server that opens no device |
| `orchard/server.py` | `ServerSpec`, `ServerControl` (start, stop, confirm stopped, wait ready, ask), `ServerStandIn` |
| `orchard/handoff.py` | `decide_park`, `progress`, `Handoff` (park, restore, recovery), `reacquire`, `release_for_idle_phase`, `recover` |
| `orchard/watchdog.py` | `Event`, `Finding`, the six detectors, `Ladder`, `RetryGuard`, `Watchdog`, `replay` |
| `orchard/transcripts.py` | qwen-code transcript to `Event` stream; signature write and load |
| `scripts/extract_qwen_signature.py` | writes a sanitised signature fixture from one transcript |
| `orchard/park_check.py` | hardware check of park and restore with fake servers (run by the controller) |
| `orchard/ledger.py` | `replay_state`: a multi-step restore keeps the run parked until its last step |
| `tests/fakes.py` | `FakeRun`, the shared machine fakes (`World`, `FakeAdapter`, `FakeServer`, `FakeStandIn`), `make_handoff` |
| `tests/fixtures/qwen_loop_signature.jsonl`, `tests/fixtures/qwen_quiet_signature.jsonl` | sanitised event signatures of the recorded loop and of one quiet chat |
| `tests/test_*.py` | one test file per module, plus `test_gozer_contract.py`, `test_server_live.py`, `test_replay_local.py` |
| `docs/runbooks/hardware-validation.md`, `README.md`, `CLAUDE.md`, the spec | documentation of what was built |

---

### Task 0: Commit the existing work and branch

**Files:**
- Commit: `CLAUDE.md`, `README.md`, `docs/runbooks/`, `docs/superpowers/plans/2026-10-01-orchard-core.md`, `docs/superpowers/plans/2026-10-02-gozer-reset-and-ownership.md`, `docs/superpowers/plans/2026-10-02-supervisor-behavior.md` (this plan), `docs/superpowers/specs/2026-10-01-orchard-design.md`, `orchard/hardware_check.py`, `tests/test_hardware_check.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean `main` and the branch `plan3-supervisor-behavior`.

- [ ] **Step 1: Confirm the ignore rules**

Run: `cd /home/ttuser/code/tt-orchard && grep -nx 'runs/' .gitignore && grep -nx 'config/tiers.toml' .gitignore`
Expected: two lines, `runs/` and `config/tiers.toml`. If either is missing, add it to `.gitignore` and include `.gitignore` in the commit below.

- [ ] **Step 2: Look at what is uncommitted**

Run: `git status --short`
Expected, exactly these lines (order may differ):

```
 M CLAUDE.md
 M docs/superpowers/plans/2026-10-01-orchard-core.md
 M docs/superpowers/specs/2026-10-01-orchard-design.md
?? README.md
?? docs/runbooks/
?? docs/superpowers/plans/2026-10-02-gozer-reset-and-ownership.md
?? docs/superpowers/plans/2026-10-02-supervisor-behavior.md
?? orchard/hardware_check.py
?? tests/test_hardware_check.py
```

If anything else appears, stop and report it. Do not add it.

- [ ] **Step 3: Prove that nothing under `runs/` or `config/tiers.toml` would be added**

Run: `git add -n CLAUDE.md README.md docs orchard/hardware_check.py tests/test_hardware_check.py`
Expected: `add '...'` lines for the files in Step 2 and for each file under `docs/runbooks/`. No line names `runs/` or `config/tiers.toml`. (`git add -n` answers "what would be added"; `git check-ignore -v` also prints negation matches, so it is the wrong instrument here.)

- [ ] **Step 4: Run the baseline suite**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `607 passed`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md docs orchard/hardware_check.py tests/test_hardware_check.py
git commit -m "Commit the hardware-check driver, runbooks, plans 2 and 3, and the doc updates since plan 1"
```

- [ ] **Step 6: Confirm a clean tree and branch**

Run: `git status --short && git switch -c plan3-supervisor-behavior && git status --short --branch`
Expected: the first `git status --short` prints nothing. The last command prints `## plan3-supervisor-behavior` and nothing else.

---

### Task 1: Defaults and the command helper

**Files:**
- Create: `orchard/defaults.py`, `orchard/commands.py`
- Test: `tests/test_defaults.py`, `tests/test_commands.py`

**Interfaces:**
- Produces: `orchard.defaults` constants: `CHIPS_PER_BOARD`, `GOZER_RESET_S`, `GOZER_RELEASE_S`, `TT_MODEL_STOP_S`, `CONTAINER_WARM_BOOT_S`, `WARM_RESTART_2CHIP_S`, `COLD_BOOT_S`, `NEIGHBOUR_BUSY_S`, `GOZER_CLAIM_WINDOW_S`, `GOZER_TICKET_MAX_AGE_S`, `CMD_TIMEOUT_S`, `START_TIMEOUT_S`, `RESET_TIMEOUT_S`, `STOP_TIMEOUT_S`, `QUIET_WAIT_S`, `MESH_RESET_EXTRA_S`, `POLL_S`, `READY_POLL_S`, `COLD_BOOT_BUDGET_S`, `STANDIN_READY_S`, `CANARY_TIMEOUT_S`, `CANARY_MAX_TOKENS`, `QUEUE_POLL_S`, `IDLE_RELEASE_S`, `IDENTICAL_N`, `THINKING_CAP`, `REPEAT_TOOL_N`, `NO_EVIDENCE_S`, `LEASE_IDLE_S`, `LEASE_POLL_S`, `RUNG_CAPS`.
- Produces: `orchard.commands.CommandResult` (frozen dataclass: `argv: tuple[str, ...]`, `returncode: int | None`, `stdout: str = ""`, `stderr: str = ""`, `timed_out: bool = False`, `left_running: bool = False`, `pid: int | None = None`; method `record(limit: int = 2000) -> dict`).
- Produces: `orchard.commands.run_command(argv, timeout: float, *, env: dict | None = None, kill_on_timeout: bool = True) -> CommandResult`. Every injected `run` in this plan has this signature.

- [ ] **Step 1: Write the failing tests**

`tests/test_defaults.py`:

```python
"""The defaults keep the relations the measurements require."""
from orchard import defaults as d


def test_queue_poll_stays_well_inside_gozers_claim_window():
    # The head ticket gets a 90 s claim window. A poll that sleeps longer can lose its place.
    assert d.QUEUE_POLL_S * 3 <= d.GOZER_CLAIM_WINDOW_S


def test_quiet_wait_outlasts_a_neighbours_reset():
    # During any reset the other board shows BUSY-UNTRACKED for about 42 s.
    assert d.QUIET_WAIT_S > d.NEIGHBOUR_BUSY_S
    assert d.QUIET_WAIT_S + d.MESH_RESET_EXTRA_S > 2 * d.NEIGHBOUR_BUSY_S


def test_a_container_start_gets_longer_than_a_command():
    assert d.START_TIMEOUT_S > d.CMD_TIMEOUT_S > 10 * d.TT_MODEL_STOP_S


def test_cold_boot_budget_exceeds_the_measured_cold_boot():
    assert d.COLD_BOOT_BUDGET_S > d.COLD_BOOT_S > d.WARM_RESTART_2CHIP_S


def test_reset_timeout_is_far_past_the_measured_reset():
    assert d.RESET_TIMEOUT_S >= 10 * d.GOZER_RESET_S


def test_watchdog_thresholds_sit_between_the_quiet_chats_and_the_loop():
    # Quiet chats: thoughts_token_count p99 16861. The loop: 27939 on each call.
    assert 16861 < d.THINKING_CAP < 27939
    assert d.IDENTICAL_N == 3 and d.REPEAT_TOOL_N == 3


def test_every_rung_has_a_cap_of_at_least_one():
    assert set(d.RUNG_CAPS) == {"nudge", "escalate", "pause"}
    assert all(v >= 1 for v in d.RUNG_CAPS.values())
```

`tests/test_commands.py`:

```python
"""run_command: exit codes, output, timeouts, and leaving a reset running."""
import os
import signal
import sys

import pytest

from orchard.commands import run_command

PY = sys.executable


def test_captures_exit_code_and_both_streams():
    r = run_command([PY, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], 10)
    assert (r.returncode, r.stdout, r.stderr, r.timed_out) == (3, "out\n", "err\n", False)


def test_missing_program_is_exit_127():
    r = run_command(["/nonexistent/orchard-no-such-binary"], 5)
    assert r.returncode == 127 and "not found" in r.stderr


def test_timeout_kills_the_command():
    r = run_command([PY, "-c", "import time; time.sleep(30)"], 0.3)
    assert r.timed_out and r.returncode is None and not r.left_running
    with pytest.raises(ProcessLookupError):
        os.kill(r.pid, 0)


def test_timeout_can_leave_a_reset_running():
    r = run_command([PY, "-c", "import time; time.sleep(30)"], 0.3, kill_on_timeout=False)
    try:
        assert r.timed_out and r.left_running and r.returncode is None
        os.kill(r.pid, 0)          # still alive: a reset must never be cut short
    finally:
        os.killpg(r.pid, signal.SIGKILL)


def test_env_is_passed_through():
    r = run_command([PY, "-c", "import os; print(os.environ['ORCHARD_T'])"], 10,
                    env={**os.environ, "ORCHARD_T": "x"})
    assert r.stdout == "x\n"


def test_record_truncates_each_stream():
    r = run_command([PY, "-c", "print('a' * 5000)"], 10)
    rec = r.record()
    assert rec["returncode"] == 0 and len(rec["stdout"]) == 2000 and rec["argv"][0] == PY
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py tests/test_commands.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.defaults'` and `'orchard.commands'`.

- [ ] **Step 3: Write `orchard/defaults.py`**

```python
"""Every timing, budget and threshold the supervisor uses, one named constant each.

Each constant says where its number comes from. "Measured" means measured on this Quietbox 2 and
recorded in the spec (section 3 or 14) or the gozer-park skill. "Choice" means nobody measured
it; it is a starting value to revisit when a measurement exists.
"""
from __future__ import annotations

# ---- hardware facts -------------------------------------------------------------------------
# A QB2 p300c board has two chips. gozer leases whole boards: lease fb9995 (2026-10-02, H5) asked
# for 1 chip and was granted one unit with chips 0000:01:00.0 and 0000:02:00.0.
CHIPS_PER_BOARD = 2

# ---- measured timings (2026-10-02, this machine) ---------------------------------------------
GOZER_RESET_S = 41.7            # `gozer reset`, 7 runs, 41.6 to 41.7 s (spec section 3, gozer-park)
GOZER_RELEASE_S = 41.7          # `gozer release` including its reset, 4 runs, 41.6 to 41.7 s
TT_MODEL_STOP_S = 1.9           # `tt-model stop`, clean shutdown, 1.6 to 1.9 s (H5 container)
CONTAINER_WARM_BOOT_S = 20.3    # Audio8 container warm boot to ready, 19.6 to 20.3 s (H5)
WARM_RESTART_2CHIP_S = 180.0    # 2-chip Qwen3.8 warm restart, 2 to 3 min (spec section 3)
COLD_BOOT_S = 1800.0            # 2-chip cold first boot, about 30 min (spec section 3)
NEIGHBOUR_BUSY_S = 42.0         # the other board shows BUSY-UNTRACKED during any reset (spec section 8)

# ---- gozer protocol constants (read from tt-gozer gozer/queue.py) ------------------------------
GOZER_CLAIM_WINDOW_S = 90.0     # CLAIM_WINDOW_SECONDS: the head ticket's window to claim
GOZER_TICKET_MAX_AGE_S = 3600.0 # TICKET_MAX_AGE_SECONDS: a ticket expires after one hour

# ---- budgets derived from the measurements (the multipliers are choices) -----------------------
CMD_TIMEOUT_S = 120.0           # choice: gozer status/acquire, docker, ps, ss, curl; each takes < 2 s
START_TIMEOUT_S = 900.0         # choice, not measured: `tt-model serve --detach` for a container. The
                                # 20 s warm boot includes the watch; a start that reloads an image is
                                # unmeasured. On a timeout the start is left running and checked.
RESET_TIMEOUT_S = 600.0         # choice: about 14 times the measured reset; same as hardware_check
STOP_TIMEOUT_S = 120.0          # choice: `tt-model stop` measured under 4 s; docker's SIGKILL path is longer
QUIET_WAIT_S = 60.0             # choice: longer than NEIGHBOUR_BUSY_S, so a neighbour's reset can end
MESH_RESET_EXTRA_S = 60.0       # choice: added to the quiet wait when `tt-model stop` says it reset
                                # the mesh itself (its SIGKILL path), a reset gozer does not run
POLL_S = 2.0                    # choice: re-read interval while waiting for chips to go quiet
READY_POLL_S = 5.0              # choice: health poll interval while a server boots
COLD_BOOT_BUDGET_S = 2700.0     # choice: 1.5 times COLD_BOOT_S; past this the stage blocks (spec section 6)
STANDIN_READY_S = 600.0         # choice, not measured: CPU stand-in load time is open (spec section 12)
CANARY_TIMEOUT_S = 300.0        # choice, not measured: one short greedy answer
CANARY_MAX_TOKENS = 64          # choice: the canary needs a short answer only
QUEUE_POLL_S = 10.0             # choice: well inside GOZER_CLAIM_WINDOW_S
IDLE_RELEASE_S = 900.0          # choice: hold a lease through a phase with no hardware use up to
                                # 15 min. A release costs one reset (42 s) plus a queue wait; an
                                # image build (1.5 to 2.5 h) is far past this.

# ---- watchdog thresholds (replay of ~/.qwen transcripts on 2026-10-02) -------------------------
# The recorded loop: qwen-code 0.24.7, chat 197354ac, five identical calls of about 653 s each
# (input 145299, output 33348, thoughts 27939 tokens). 582 main-agent responses in 31 chats:
# thoughts p50 348, p95 5935, p99 16861, max 27939 (the loop). With these values the detectors
# fire only in the loop chat (tests/test_transcripts.py pins the exact findings).
# The basis is in-sample: the thresholds were chosen on the same 31 chats the tests replay. They
# separate this corpus; nothing yet shows they generalise. Plan 4's proxy is a different
# instrument from qwen-code telemetry, so the thresholds must be checked again on proxy events.
# Plan 4 passes them in from the run config; these constants are the defaults.
IDENTICAL_N = 3                 # measured: fires on the third call (about 33 min) of each 5-call repeat
THINKING_CAP = 20000            # measured: above every quiet response, below the loop's 27939
REPEAT_TOOL_N = 3               # measured: no transcript repeats a tool call; outputs repeat at most twice
NO_EVIDENCE_S = 3600.0          # choice, not measured: transcripts carry no evidence events
LEASE_IDLE_S = 1800.0           # choice, not measured
LEASE_POLL_S = 60.0             # choice: `gozer status` is read at most once a minute by the watchdog
RUNG_CAPS = {"nudge": 1, "escalate": 1, "pause": 1}   # choice: each rung once per agent and stage
```

- [ ] **Step 4: Write `orchard/commands.py`**

```python
"""Run one external program for supervisor code and keep what it printed.

This module owns the one place where supervisor code (the lease adapters and server control)
starts an external program. Agent commands go through orchard/runner.py and its denials. This
helper has none, because only supervisor code calls it, with fixed argument lists.

Each command runs in its own session, so a Ctrl-C at the terminal reaches only the supervisor.
On a timeout the command is killed unless the caller says not to. A chip reset must never be
cut short: killing gozer would leave its `tt-smi -r` running with nobody watching.
"""
from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int | None          # None when the command timed out
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    left_running: bool = False      # timed out and deliberately not killed
    pid: int | None = None

    def record(self, limit: int = 2000) -> dict:
        """The ledger form: the first `limit` characters of each stream."""
        return {"argv": list(self.argv), "returncode": self.returncode,
                "stdout": self.stdout[:limit], "stderr": self.stderr[:limit],
                "timed_out": self.timed_out, "left_running": self.left_running}


Run = Callable[..., CommandResult]


def run_command(argv, timeout: float, *, env: dict | None = None,
                kill_on_timeout: bool = True) -> CommandResult:
    argv = tuple(str(a) for a in argv)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True, env=env)
    except FileNotFoundError as exc:
        return CommandResult(argv, 127, "", f"not found: {exc}")
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if not kill_on_timeout:
            # Left running on purpose. The Popen object is dropped: the process is never reaped
            # and its pipes are never read. gozer prints little, so the pipes cannot fill.
            return CommandResult(argv, None, timed_out=True, left_running=True, pid=proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return CommandResult(argv, None, out, err, timed_out=True, pid=proc.pid)
    return CommandResult(argv, proc.returncode, out, err, pid=proc.pid)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_defaults.py tests/test_commands.py`
Expected: PASS (13 tests).

- [ ] **Step 6: Mutation check of the left-running guard**

In `run_command`, change `if not kill_on_timeout:` to `if False:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_commands.py::test_timeout_can_leave_a_reset_running`. Expected: FAIL (`left_running` is False and `os.kill(r.pid, 0)` raises). Restore the line and run it again: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/defaults.py orchard/commands.py tests/test_defaults.py tests/test_commands.py
git commit -m "Add the defaults module and the command helper that can leave a reset running"
```

---

### Task 2: Adapter types

**Files:**
- Create: `orchard/adapters/__init__.py`
- Test: `tests/test_adapter_types.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all importable from `orchard.adapters`:
  - `STATES = frozenset({"HELD", "HELD-FOREIGN", "CLAIMED", "STALE", "BUSY-UNTRACKED", "FREE"})`
  - `Lease` (frozen dataclass): `lease_id: str`, `chips: tuple[str, ...]` (BDFs), `dev_indices: tuple[int, ...]`, `env: dict[str, str]` (contains `TT_VISIBLE_DEVICES`), `units: tuple[str, ...]` (board serials); methods `record() -> dict` and classmethod `from_record(rec: dict) -> Lease`.
  - `ChipState` (frozen dataclass): `bdf: str`, `state: str`, `who: str | None`, `board: str = ""`, `dev_index: int | None = None`, `lease_pid: int | None = None`, `pids_holding: tuple[int, ...] = ()`. An unknown `state` raises `ValueError`.
  - Exceptions: `AdapterError(Exception)`; `Queued(AdapterError)` with `.ticket: str`, `.position: int | None`; `Refused(AdapterError)` with `.reason: str`, `.permanent: bool` (default False; True when no wait can change the answer); `ResetFailed(AdapterError)` with `.detail: str`, `.left_running: bool`; `LeaseLost(AdapterError)` with `.detail: str`; `TicketGone(AdapterError)` with `.ticket: str`.
  - `LeaseAdapter` (Protocol): attribute `owner_pid: int`; `acquire(chips: int, who: str, reason: str, *, queue: bool = False, exact: str | None = None) -> Lease`; `claim(ticket: str, chips: int, who: str, reason: str) -> Lease`; `cancel(ticket: str) -> None`; `release(lease: Lease) -> None`; `reset(lease: Lease) -> None`; `status() -> list[ChipState]`.
  - `boards_of(chips: list[ChipState]) -> dict[str, list[ChipState]]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Adapter types: the lease record, chip states and the exceptions."""
import pytest

from orchard.adapters import (ChipState, Lease, Queued, Refused, ResetFailed, TicketGone,
                              boards_of)

LEASE = Lease("fb9995", ("0000:01:00.0", "0000:02:00.0"), (0, 1),
              {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, ("0000046131924062",))


def test_lease_record_round_trips_through_json_types():
    rec = LEASE.record()
    assert rec == {"lease_id": "fb9995", "chips": ["0000:01:00.0", "0000:02:00.0"],
                   "dev_indices": [0, 1], "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"},
                   "units": ["0000046131924062"]}
    assert Lease.from_record(rec) == LEASE


def test_lease_is_hashable_even_with_an_env_dict():
    assert hash(LEASE) == hash(Lease.from_record(LEASE.record()))


def test_unknown_chip_state_is_refused():
    with pytest.raises(ValueError, match="unknown chip state"):
        ChipState("0000:01:00.0", "WEDGED", None)


def test_boards_of_groups_by_serial_in_listed_order():
    chips = [ChipState("a", "FREE", None, board="B0"), ChipState("c", "FREE", None, board="B1"),
             ChipState("b", "CLAIMED", "x", board="B0")]
    groups = boards_of(chips)
    assert list(groups) == ["B0", "B1"]
    assert [c.bdf for c in groups["B0"]] == ["a", "b"]


def test_exceptions_carry_what_the_caller_needs():
    q = Queued("t-1", 3)
    assert (q.ticket, q.position) == ("t-1", 3)
    assert TicketGone("t-1").ticket == "t-1"
    assert ResetFailed("x").left_running is False
    assert ResetFailed("x", left_running=True).left_running is True
    assert Refused("busy").permanent is False and Refused("no reset", permanent=True).permanent
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_adapter_types.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.adapters'`.

- [ ] **Step 3: Write `orchard/adapters/__init__.py`**

```python
"""Lease adapters: how the supervisor takes, resets and gives back Tenstorrent boards.

This package owns the interface between the supervisor and whatever controls access to the chips
on a machine (spec section 4). `gozer.py` drives tt-gozer. `single_tenant.py` is for a machine
with no lease tool. Supervisor code depends only on the types in this file, so either adapter can
stand behind it.

A board is named by its serial, which is gozer's unit key at board grain. A chip is named by its
PCI address (BDF). Device indices appear only where a device node is meant (/dev/tenstorrent/N).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# The chip states `gozer status` reports (tt-gozer gozer/gatekeeper.py, reconcile).
STATES = frozenset({"HELD", "HELD-FOREIGN", "CLAIMED", "STALE", "BUSY-UNTRACKED", "FREE"})


@dataclass(frozen=True)
class Lease:
    lease_id: str
    chips: tuple[str, ...]
    dev_indices: tuple[int, ...]
    # A dict cannot be hashed. hash=False keeps the frozen dataclass hashable on its other fields.
    env: dict[str, str] = field(hash=False)
    units: tuple[str, ...]

    def record(self) -> dict:
        return {"lease_id": self.lease_id, "chips": list(self.chips),
                "dev_indices": list(self.dev_indices), "env": dict(self.env),
                "units": list(self.units)}

    @classmethod
    def from_record(cls, rec: dict) -> "Lease":
        return cls(lease_id=str(rec["lease_id"]), chips=tuple(rec["chips"]),
                   dev_indices=tuple(int(i) for i in rec["dev_indices"]),
                   env={str(k): str(v) for k, v in rec["env"].items()}, units=tuple(rec["units"]))


@dataclass(frozen=True)
class ChipState:
    bdf: str
    state: str
    who: str | None
    board: str = ""
    dev_index: int | None = None
    lease_pid: int | None = None          # the pid gozer judges the lease by
    pids_holding: tuple[int, ...] = ()    # processes with the device open, as gozer sees them

    def __post_init__(self):
        if self.state not in STATES:
            # A state this code does not know means the lease tool changed. Fail closed and let a
            # person read what it now reports.
            raise ValueError(f"unknown chip state {self.state!r}")


class AdapterError(Exception):
    """The lease tool did something the adapter cannot interpret."""


class Queued(AdapterError):
    """No chips now. The request waits in the queue under `ticket`."""

    def __init__(self, ticket: str, position: int | None = None):
        super().__init__(f"queued with ticket {ticket} (position {position})")
        self.ticket, self.position = ticket, position


class Refused(AdapterError):
    """The lease tool refused (gozer exit 12 for acquire, 15 for reset or release).

    `permanent` means no wait can change the answer (for example, an adapter with no reset
    command), so the caller blocks at once. gozer's exit 15 is never marked permanent: one of its
    causes is a device that is still open.
    """

    def __init__(self, reason: str, permanent: bool = False):
        super().__init__(reason)
        self.reason, self.permanent = reason, permanent


class ResetFailed(AdapterError):
    """The reset ran and failed, or is still running after its timeout (left_running)."""

    def __init__(self, detail: str, left_running: bool = False):
        super().__init__(detail)
        self.detail, self.left_running = detail, left_running


class LeaseLost(AdapterError):
    """The lease no longer exists, or another lease now holds its chips. Start no server."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class TicketGone(AdapterError):
    """The queue ticket no longer exists. gozer expires tickets after one hour."""

    def __init__(self, ticket: str):
        super().__init__(f"queue ticket {ticket} no longer exists")
        self.ticket = ticket


class LeaseAdapter(Protocol):
    owner_pid: int

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease: ...

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease: ...

    def cancel(self, ticket: str) -> None: ...

    def release(self, lease: Lease) -> None: ...

    def reset(self, lease: Lease) -> None: ...

    def status(self) -> list[ChipState]: ...


def boards_of(chips: list[ChipState]) -> dict[str, list[ChipState]]:
    """Group chip states by board serial, in the order the lease tool listed them."""
    out: dict[str, list[ChipState]] = {}
    for c in chips:
        out.setdefault(c.board, []).append(c)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_adapter_types.py`
Expected: PASS (5 tests).

- [ ] **Step 5: Mutation check of the unknown-state guard**

Delete the `if self.state not in STATES:` block in `ChipState.__post_init__`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_adapter_types.py::test_unknown_chip_state_is_refused`. Expected: FAIL (`DID NOT RAISE`). Restore it: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/adapters/__init__.py tests/test_adapter_types.py
git commit -m "Add the lease adapter types: Lease, ChipState, the exceptions and the protocol"
```

---

### Task 3: The gozer adapter

**Files:**
- Create: `orchard/adapters/gozer.py`, `tests/fakes.py`
- Test: `tests/test_gozer_adapter.py`

**Interfaces:**
- Consumes: `orchard.commands.CommandResult`, `run_command`; `orchard.defaults.CMD_TIMEOUT_S`, `RESET_TIMEOUT_S`; everything in `orchard.adapters` from Task 2.
- Produces: `orchard.adapters.gozer.GozerAdapter(*, gozer: str = "gozer", owner_pid: int | None = None, run=run_command, timeout: float = CMD_TIMEOUT_S, reset_timeout: float = RESET_TIMEOUT_S)`, implementing `LeaseAdapter`. Attribute `in_flight: set[str]` (lease ids whose reset or release is still running). Module constants `EXIT_OK`, `EXIT_QUEUED`, `EXIT_UNAVAILABLE`, `EXIT_NO_LEASE`, `EXIT_REFUSED`, `EXIT_RESET_FAILED`, `EXIT_RESET_CHANGED_HANDS`.
- Produces in `tests/fakes.py`: `FakeRun(script: dict | None = None, key=None)` (callable with `run_command`'s signature; `.calls: list[dict]`; `.argvs() -> list[list[str]]`), constants `OWNER = 4242`, `GRANT` (dict), `LEASE` (`Lease`). Later tasks append to `tests/fakes.py`.

- [ ] **Step 1: Write `tests/fakes.py`**

```python
"""Test doubles shared by the plan 3 tests. Nothing here touches hardware or the network."""
from __future__ import annotations

import json

from orchard.adapters import Lease
from orchard.commands import CommandResult

OWNER = 4242

# Shape copied from a real grant: runs/h5-20261002T201323Z/lease.json (owner pid changed).
GRANT = {"chips": ["0000:01:00.0", "0000:02:00.0"], "dev_indices": [0, 1],
         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, "expanded": True,
         "granted": True, "lease_id": "fb9995", "neighbours": {}, "owner_pid": OWNER,
         "requested": 1, "units": ["0000046131924062"]}

LEASE = Lease("fb9995", ("0000:01:00.0", "0000:02:00.0"), (0, 1),
              {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, ("0000046131924062",))


class FakeRun:
    """Stands in for orchard.commands.run_command.

    script maps a key to a list of answers, used in order; the last answer repeats. The key is
    argv[1] (the gozer subcommand) unless `key` is a function of argv. An answer is
    (returncode, stdout, stderr). stdout may be a dict, sent as JSON. returncode may be
    "timeout". A call with no scripted answer fails the test.
    """

    def __init__(self, script: dict | None = None, key=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.key = key or (lambda argv: argv[1])
        self.calls: list[dict] = []

    def __call__(self, argv, timeout, *, env=None, kill_on_timeout=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "timeout": timeout, "env": env,
                           "kill_on_timeout": kill_on_timeout})
        answers = self.script.get(self.key(argv))
        if not answers:
            raise AssertionError(f"unexpected call: {argv}")
        rc, out, err = answers.pop(0) if len(answers) > 1 else answers[0]
        if rc == "timeout":
            return CommandResult(tuple(argv), None, timed_out=True, left_running=not kill_on_timeout)
        return CommandResult(tuple(argv), rc, out if isinstance(out, str) else json.dumps(out), err)

    def argvs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_gozer_adapter.py`:

```python
"""GozerAdapter against hand-written gozer output (tests/test_gozer_contract.py checks the real CLI)."""
import pytest

from fakes import GRANT, LEASE, OWNER, FakeRun
from orchard.adapters import (AdapterError, ChipState, LeaseLost, Queued, Refused, ResetFailed,
                              TicketGone)
from orchard.adapters.gozer import GozerAdapter


def make(script):
    run = FakeRun(script)
    return GozerAdapter(owner_pid=OWNER, run=run), run


OK_RELEASE = (0, {"released": True, "message": "released fb9995; chips reset"}, "")
OK_RESET = (0, {"reset": True, "status": "reset", "message": "Resetting PCI BDFs"}, "")


def test_acquire_sends_the_documented_flags():
    a, run = make({"acquire": [(0, GRANT, "")]})
    a.acquire(2, "orchard:run1", "stage 2")
    assert run.argvs() == [["gozer", "acquire", "--chips", "2", "--who", "orchard:run1",
                            "--reason", "stage 2", "--owner-pid", str(OWNER), "--json",
                            "--no-queue"]]


def test_acquire_returns_the_granted_lease():
    a, _ = make({"acquire": [(0, GRANT, "")]})
    assert a.acquire(2, "w", "r") == LEASE


def test_exact_board_is_passed():
    a, run = make({"acquire": [(0, GRANT, "")]})
    a.acquire(2, "w", "r", exact="0000:01:00.0")
    argv = run.argvs()[0]
    assert argv[argv.index("--exact") + 1] == "0000:01:00.0"


def test_acquire_with_queue_raises_queued_with_the_ticket():
    a, run = make({"acquire": [(10, {"granted": False, "queued": True, "ticket": "t-7",
                                     "position": 2, "ahead": ["claude:x"]}, "")]})
    with pytest.raises(Queued) as exc:
        a.acquire(4, "w", "r", queue=True)
    assert (exc.value.ticket, exc.value.position) == ("t-7", 2)
    assert "--no-queue" not in run.argvs()[0]


def test_acquire_without_queue_is_refused_when_busy():
    a, _ = make({"acquire": [(12, {"granted": False, "queued": False}, "")]})
    with pytest.raises(Refused):
        a.acquire(4, "w", "r")


def test_a_grant_owned_by_another_pid_is_released_and_refused():
    a, run = make({"acquire": [(0, dict(GRANT, owner_pid=999), "")], "release": [OK_RELEASE]})
    with pytest.raises(AdapterError, match="owner_pid 999"):
        a.acquire(2, "w", "r")
    assert run.argvs()[1] == ["gozer", "release", "fb9995", "--json"]


def test_a_grant_whose_env_does_not_list_its_chips_is_refused():
    bad = dict(GRANT, env={"TT_VISIBLE_DEVICES": "0000:01:00.0"})
    a, run = make({"acquire": [(0, bad, "")], "release": [OK_RELEASE]})
    with pytest.raises(AdapterError, match="env does not list"):
        a.acquire(2, "w", "r")
    # An unusable lease is given back at once, or the board stays held until the supervisor dies.
    assert run.argvs()[1] == ["gozer", "release", "fb9995", "--json"]


def test_claim_passes_the_ticket_and_reports_a_gone_ticket():
    a, run = make({"acquire": [(13, "", "gozer: no such ticket t-7")]})
    with pytest.raises(TicketGone):
        a.claim("t-7", 4, "w", "r")
    argv = run.argvs()[0]
    assert argv[-2:] == ["--ticket", "t-7"] and "--no-queue" not in argv


def test_claim_that_is_still_queued_raises_queued():
    a, _ = make({"acquire": [(10, {"granted": False, "queued": True, "ticket": "t-7",
                                   "position": 1}, "")]})
    with pytest.raises(Queued):
        a.claim("t-7", 4, "w", "r")


@pytest.mark.parametrize("rc,payload,exc", [
    (13, {"reset": False, "status": "not-found", "message": "lease fb9995 not found"}, LeaseLost),
    (15, {"reset": False, "status": "refused", "message": "device still open"}, Refused),
    (17, {"reset": False, "status": "failed", "message": "tt-smi -r failed"}, ResetFailed),
    (18, {"reset": False, "status": "changed-hands", "message": "unit held by another lease"}, LeaseLost),
    (2, "", AdapterError),
])
def test_reset_exit_codes(rc, payload, exc):
    a, _ = make({"reset": [(rc, payload, "")]})
    with pytest.raises(exc):
        a.reset(LEASE)


def test_reset_success_needs_status_reset():
    a, run = make({"reset": [OK_RESET, (0, {"reset": True, "status": "odd"}, "")]})
    a.reset(LEASE)
    assert run.argvs()[0] == ["gozer", "reset", "fb9995", "--json"]
    with pytest.raises(AdapterError):
        a.reset(LEASE)


def test_reset_is_never_killed_and_a_hung_reset_blocks_another():
    a, run = make({"reset": [("timeout", "", "")]})
    with pytest.raises(ResetFailed) as exc:
        a.reset(LEASE)
    assert exc.value.left_running and run.calls[0]["kill_on_timeout"] is False
    with pytest.raises(Refused, match="still running"):
        a.reset(LEASE)
    with pytest.raises(Refused, match="still running"):
        a.release(LEASE)
    assert len(run.calls) == 1


def test_release_whose_reset_failed_raises():
    a, _ = make({"release": [(0, {"released": True,
                                  "message": "released fb9995; chips NOT marked clean"}, "")]})
    with pytest.raises(ResetFailed):
        a.release(LEASE)


def test_release_of_a_gone_lease_is_quiet():
    a, _ = make({"release": [(13, {"released": False, "message": "lease fb9995 not found"}, "")]})
    a.release(LEASE)


def test_release_refused():
    a, _ = make({"release": [(15, {"released": False, "message": "device still open"}, "")]})
    with pytest.raises(Refused, match="still open"):
        a.release(LEASE)


STATUS = {"grain": "board", "queue": [], "chips": [
    {"bdf": "0000:01:00.0", "board": "B0", "card": "p300c", "dev_index": 0,
     "state": "HELD-FOREIGN", "who": "orchard:run1", "pid": OWNER, "reason": "coder",
     "pids_holding": [777], "overstayed": False},
    {"bdf": "0000:03:00.0", "board": "B1", "card": "p300c", "dev_index": 2, "state": "FREE",
     "who": None, "pid": None, "reason": None, "pids_holding": [], "overstayed": False}]}


def test_status_reads_every_field():
    a, _ = make({"status": [(0, STATUS, "")]})
    chips = a.status()
    assert chips[0] == ChipState("0000:01:00.0", "HELD-FOREIGN", "orchard:run1", board="B0",
                                 dev_index=0, lease_pid=OWNER, pids_holding=(777,))
    assert chips[1].state == "FREE" and chips[1].who is None


def test_status_with_an_unknown_state_fails_closed():
    odd = {"chips": [dict(STATUS["chips"][0], state="WEDGED")]}
    a, _ = make({"status": [(0, odd, "")]})
    with pytest.raises(AdapterError):
        a.status()


def test_owner_pid_must_be_above_one():
    with pytest.raises(ValueError):
        GozerAdapter(owner_pid=1, run=FakeRun())


def test_the_adapter_never_forces_and_never_waits():
    refused = (15, {"reset": False, "status": "refused", "message": "device still open"}, "")
    failed = (17, {"reset": False, "status": "failed", "message": "tt-smi -r failed"}, "")
    a, run = make({"acquire": [(0, GRANT, "")], "reset": [OK_RESET, refused, failed],
                   "release": [OK_RELEASE, (15, {"released": False, "message": "still open"}, "")],
                   "status": [(0, STATUS, "")], "cancel": [(0, {"cancelled": True}, "")]})
    a.acquire(2, "w", "r", queue=True)
    a.claim("t", 2, "w", "r")
    a.status()
    a.cancel("t")
    a.reset(LEASE)
    # The refusal and failure paths are where a "--force" retry would appear, so drive them too.
    with pytest.raises(Refused):
        a.reset(LEASE)
    with pytest.raises(ResetFailed):
        a.reset(LEASE)
    a.release(LEASE)
    with pytest.raises(Refused):
        a.release(LEASE)
    assert not any("--force" in argv for argv in run.argvs())
    assert not any(argv[1] == "wait" for argv in run.argvs())
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_gozer_adapter.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.adapters.gozer'`.

- [ ] **Step 4: Write `orchard/adapters/gozer.py`**

```python
"""Lease adapter for tt-gozer.

This module owns the translation between the supervisor's lease calls and the `gozer` command
line. It reads only gozer's `--json` output and exit codes (spec section 8, item 3). The flags and
exit codes below were read from tt-gozer `gozer/cli.py` on 2026-10-02. Committed `main` is gozer
0.3.2 and carries the `reset` command and the owner-pid ownership code (the working tree has an
uncommitted 0.3.3 version bump for its report archive).

Rules this adapter keeps:
- Every lease is taken with `--owner-pid` set to the supervisor's pid. gozer then judges the lease
  by that pid's liveness, so it cannot expire while no device is open during a swap.
- It never calls `gozer wait`. `wait` grants a lease with no owner pid, which falls back to gozer's
  15 minute detached window. A queued ticket is claimed by running `acquire --ticket` again.
- It never passes `--force`.
- `reset` and `release` are never killed on a timeout. Killing gozer would leave its `tt-smi -r`
  running with nobody watching. After such a timeout the adapter refuses a second reset or
  release of that lease.
"""
from __future__ import annotations

import json
import os

from orchard.adapters import (AdapterError, ChipState, Lease, LeaseLost, Queued, Refused,
                              ResetFailed, TicketGone)
from orchard.commands import CommandResult, run_command
from orchard.defaults import CMD_TIMEOUT_S, RESET_TIMEOUT_S

# gozer exit codes (tt-gozer gozer/cli.py).
EXIT_OK = 0
EXIT_QUEUED = 10
EXIT_UNAVAILABLE = 12          # acquire --no-queue found no chips; also a bad argument
EXIT_NO_LEASE = 13             # no such lease or ticket
EXIT_REFUSED = 15              # reset or release refused; nothing was done
EXIT_RESET_FAILED = 17         # reset ran tt-smi and it failed; the lease is untouched
EXIT_RESET_CHANGED_HANDS = 18  # reset ran, and afterwards the lease was gone or taken

# `gozer release` exits 0 even when its reset failed. These phrases in its message say so
# (orchard/hardware_check.py makes the same check).
RELEASE_RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")


class GozerAdapter:
    def __init__(self, *, gozer: str = "gozer", owner_pid: int | None = None, run=run_command,
                 timeout: float = CMD_TIMEOUT_S, reset_timeout: float = RESET_TIMEOUT_S):
        pid = os.getpid() if owner_pid is None else owner_pid
        # gozer counts the owner's descendants as the owner's own work only for owner pids above
        # 1, because every process descends from pid 1 (spec section 8, item 2).
        if pid <= 1:
            raise ValueError(f"owner pid must be above 1, got {pid}")
        self.gozer, self.owner_pid, self.run = gozer, pid, run
        self.timeout, self.reset_timeout = timeout, reset_timeout
        self.in_flight: set[str] = set()

    # ---- plumbing ---------------------------------------------------------------------------

    def _call(self, *args: str, long: bool = False) -> tuple[CommandResult, dict | None]:
        res = self.run([self.gozer, *args], self.reset_timeout if long else self.timeout,
                       kill_on_timeout=not long)
        try:
            payload = json.loads(res.stdout) if res.stdout.strip() else None
        except json.JSONDecodeError:
            payload = None
        return res, payload if isinstance(payload, dict) else None

    @staticmethod
    def _detail(res: CommandResult, payload: dict | None) -> str:
        msg = str((payload or {}).get("message") or (payload or {}).get("error") or "")
        return (msg or res.stderr.strip() or res.stdout.strip())[:500]

    def _acquire_argv(self, chips: int, who: str, reason: str) -> list[str]:
        if chips < 1:
            raise ValueError(f"chips must be at least 1, got {chips}")
        return ["acquire", "--chips", str(chips), "--who", who, "--reason", reason,
                "--owner-pid", str(self.owner_pid), "--json"]

    def _granted(self, res: CommandResult, payload: dict | None, *, op: str,
                 ticket: str | None = None) -> Lease:
        if res.timed_out:
            raise AdapterError(f"gozer {op} timed out after {self.timeout} s")
        rc = res.returncode
        if rc == EXIT_OK and payload and payload.get("granted"):
            return self._lease_from(payload)
        if rc == EXIT_QUEUED and payload and payload.get("queued") and payload.get("ticket"):
            raise Queued(str(payload["ticket"]), payload.get("position"))
        if rc == EXIT_NO_LEASE and ticket is not None:
            raise TicketGone(ticket)
        if rc == EXIT_UNAVAILABLE:
            raise Refused(self._detail(res, payload) or "no chips free")
        raise AdapterError(f"gozer {op} exited {rc}: {self._detail(res, payload)}")

    def _lease_from(self, p: dict) -> Lease:
        lid, chips, devs = p.get("lease_id"), p.get("chips"), p.get("dev_indices")
        env, units = p.get("env"), p.get("units")
        problems = []
        if not isinstance(lid, str) or not lid:
            problems.append("no lease_id")
        if not isinstance(chips, list) or not chips:
            problems.append("no chips")
            chips = []
        if not isinstance(devs, list) or len(devs) != len(chips):
            problems.append("dev_indices do not match chips")
        if not isinstance(units, list) or not units:
            problems.append("no units")
        if not isinstance(env, dict) or env.get("TT_VISIBLE_DEVICES") != ",".join(chips):
            problems.append("env does not list the granted chips")
        # The check that matters most: a lease judged by another pid is not held through a swap.
        if p.get("owner_pid") != self.owner_pid:
            problems.append(f"owner_pid {p.get('owner_pid')!r} is not this supervisor ({self.owner_pid})")
        if problems:
            note = ""
            if isinstance(lid, str) and lid:
                # Keep no lease the supervisor cannot use. The release resets the chips (about 42 s).
                r, _ = self._call("release", lid, "--json", long=True)
                note = f"; released it (gozer exit {r.returncode})"
            raise AdapterError("gozer granted a lease the adapter cannot use: "
                               + "; ".join(problems) + note)
        return Lease(lease_id=lid, chips=tuple(chips), dev_indices=tuple(int(i) for i in devs),
                     env={str(k): str(v) for k, v in env.items()}, units=tuple(units))

    def _not_in_flight(self, lease: Lease) -> None:
        if lease.lease_id in self.in_flight:
            raise Refused(f"a reset or release of lease {lease.lease_id} is still running; "
                          "wait for it to end and read `gozer status`")

    # ---- the LeaseAdapter calls ---------------------------------------------------------------

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease:
        argv = self._acquire_argv(chips, who, reason)
        if exact:
            argv += ["--exact", exact]
        if not queue:
            argv.append("--no-queue")
        return self._granted(*self._call(*argv), op="acquire")

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease:
        argv = self._acquire_argv(chips, who, reason) + ["--ticket", ticket]
        return self._granted(*self._call(*argv), op="claim", ticket=ticket)

    def cancel(self, ticket: str) -> None:
        res, payload = self._call("cancel", ticket, "--json")
        if res.returncode in (EXIT_OK, EXIT_NO_LEASE):   # 13: the ticket is already gone
            return
        raise AdapterError(f"gozer cancel exited {res.returncode}: {self._detail(res, payload)}")

    def release(self, lease: Lease) -> None:
        self._not_in_flight(lease)
        res, payload = self._call("release", lease.lease_id, "--json", long=True)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"gozer release {lease.lease_id} still running after "
                              f"{self.reset_timeout} s; left running", left_running=True)
        if res.returncode == EXIT_OK:
            msg = str((payload or {}).get("message", ""))
            if any(t in msg for t in RELEASE_RESET_FAILED_TEXT):
                raise ResetFailed(f"lease {lease.lease_id} released, but its reset did not "
                                  f"succeed: {msg}")
            return
        if res.returncode == EXIT_NO_LEASE:
            return                       # already gone: nothing is left to release
        if res.returncode == EXIT_REFUSED:
            raise Refused(self._detail(res, payload))
        raise AdapterError(f"gozer release exited {res.returncode}: {self._detail(res, payload)}")

    def reset(self, lease: Lease) -> None:
        self._not_in_flight(lease)
        res, payload = self._call("reset", lease.lease_id, "--json", long=True)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"gozer reset {lease.lease_id} still running after "
                              f"{self.reset_timeout} s; left running", left_running=True)
        rc, detail = res.returncode, self._detail(res, payload)
        if rc == EXIT_OK and (payload or {}).get("status") == "reset":
            return
        if rc == EXIT_NO_LEASE:
            raise LeaseLost(f"gozer has no lease {lease.lease_id}: {detail}")
        if rc == EXIT_REFUSED:
            # Nothing was done. Some refusals leave a valid lease in place (spec section 8,
            # item 1), so the caller reads the chip states and decides.
            raise Refused(detail)
        if rc == EXIT_RESET_FAILED:
            raise ResetFailed(detail)
        if rc == EXIT_RESET_CHANGED_HANDS:
            raise LeaseLost(f"the reset ran, and afterwards lease {lease.lease_id} was gone or "
                            f"taken: {detail}")
        raise AdapterError(f"gozer reset exited {rc}: {detail}")

    def status(self) -> list[ChipState]:
        res, payload = self._call("status", "--json")
        if res.returncode != EXIT_OK or not payload or not isinstance(payload.get("chips"), list):
            raise AdapterError(f"gozer status failed (exit {res.returncode}): "
                               f"{self._detail(res, payload)}")
        out = []
        for c in payload["chips"]:
            try:
                out.append(ChipState(bdf=str(c["bdf"]), state=str(c["state"]), who=c.get("who"),
                                     board=str(c.get("board") or ""), dev_index=c.get("dev_index"),
                                     lease_pid=c.get("pid"),
                                     pids_holding=tuple(c.get("pids_holding") or ())))
            except (KeyError, TypeError, ValueError) as exc:
                raise AdapterError(f"gozer status listed a chip the adapter cannot read: "
                                   f"{c!r} ({exc})") from exc
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_gozer_adapter.py`
Expected: PASS (23 tests).

- [ ] **Step 6: Mutation checks**

1. In `_lease_from`, delete the `if p.get("owner_pid") != self.owner_pid:` check (both lines). Run `python3 -m pytest -q -p no:cacheprovider tests/test_gozer_adapter.py::test_a_grant_owned_by_another_pid_is_released_and_refused`. Expected: FAIL. Restore: PASS.
2. Make `_not_in_flight` return at once (`return` as its first line). Run `tests/test_gozer_adapter.py::test_reset_is_never_killed_and_a_hung_reset_blocks_another`. Expected: FAIL. Restore: PASS.
3. Never force: in `reset`, add `"--force"` to the `_call("reset", ...)` arguments. Run `tests/test_gozer_adapter.py::test_the_adapter_never_forces_and_never_waits`. Expected: FAIL. Restore: PASS.
4. Grant env: delete the two lines of the `env does not list the granted chips` check in `_lease_from`. Run `tests/test_gozer_adapter.py::test_a_grant_whose_env_does_not_list_its_chips_is_refused`. Expected: FAIL (nothing raised, and no release). Restore: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/adapters/gozer.py tests/fakes.py tests/test_gozer_adapter.py
git commit -m "Add the gozer lease adapter, driven through an injected command runner"
```

---

### Task 4: Check the gozer adapter against the real gozer command line

**Files:**
- Test: `tests/test_gozer_contract.py`

**Interfaces:**
- Consumes: `GozerAdapter` (Task 3); the exceptions from Task 2.
- Produces: helpers in this test file (`GOZER`, `QUIETBOX`, `build_sysfs`, `make_proc_dir`, `script`, the `fake_gozer` fixture). Task 15 imports them.

The FakeRun tests check the adapter against JSON written by hand. This file checks that JSON against what gozer really prints, with fake roots, so no device is touched. It also checks the two gozer behaviors recovery depends on (Task 11): the lease of a dead owner shows `STALE` with that pid and the next `acquire` reaps it without a reset, and a holder outside the owner's process tree shows `HELD-FOREIGN` and makes `reset` refuse.

- [ ] **Step 1: Write the test**

```python
"""GozerAdapter against the real gozer command line, with fake roots.

gozer runs from ORCHARD_GOZER, or ~/code/tt-gozer/bin/gozer, with a fake sysfs, a fake /proc, a
fake history directory and a fake reset command. No device is opened and the real gozer state
under /tmp/tt-gozer is never read. When gozer is not present the tests skip. A skip here is not
evidence that the adapter matches gozer.
"""
import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from orchard.adapters import Queued, Refused, TicketGone
from orchard.adapters.gozer import GozerAdapter

GOZER = Path(os.environ.get("ORCHARD_GOZER", str(Path.home() / "code/tt-gozer/bin/gozer")))
pytestmark = pytest.mark.skipif(
    not GOZER.exists(),
    reason=f"gozer not found at {GOZER}; set ORCHARD_GOZER. A skip is not evidence.")

# Two boards of two chips, as on this Quietbox 2 (copied from tests/test_hardware_check.py).
QUIETBOX = [
    {"dev_index": 0, "bdf": "0000:01:00.0", "serial": "0000000000000002", "asic_id": "1111111111111111", "card": "p300c"},
    {"dev_index": 1, "bdf": "0000:02:00.0", "serial": "0000000000000002", "asic_id": "2222222222222222", "card": "p300c"},
    {"dev_index": 2, "bdf": "0000:03:00.0", "serial": "0000000000000001", "asic_id": "3333333333333333", "card": "p300c"},
    {"dev_index": 3, "bdf": "0000:04:00.0", "serial": "0000000000000001", "asic_id": "4444444444444444", "card": "p300c"},
]
BOARD1 = ["0000:03:00.0", "0000:04:00.0"]


def build_sysfs(root, chips):
    """A fake /sys/class/tenstorrent tree, as the gozer test conftest builds it."""
    for c in chips:
        base = os.path.join(root, "class", "tenstorrent", f"tenstorrent!{c['dev_index']}")
        os.makedirs(base, exist_ok=True)
        for name, val in (("tt_serial", c["serial"]), ("tt_asic_id", c["asic_id"]),
                          ("tt_card_type", c["card"]), ("tt_heartbeat", "12345")):
            with open(os.path.join(base, name), "w") as f:
                f.write(val)
        pci = os.path.join(root, "bus", "pci", "devices", c["bdf"])
        os.makedirs(pci, exist_ok=True)
        link = os.path.join(base, "device")
        if not os.path.lexists(link):
            os.symlink(pci, link)
    return os.path.join(root, "class", "tenstorrent")


def make_proc_dir(proc_root, pid, ppid=1, devs=()):
    """What gozer reads for a live process: status (PPid), comm, and fd links to devices."""
    d = Path(proc_root) / str(pid)
    (d / "fd").mkdir(parents=True, exist_ok=True)
    (d / "comm").write_text("python\n")
    (d / "status").write_text(f"Name:\tpython\nPPid:\t{ppid}\n")
    for n, dev in enumerate(devs):
        os.symlink(f"/dev/tenstorrent/{dev}", d / "fd" / str(n + 3))
    return d


def script(path, body):
    path.write_text("#!/bin/bash\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def fake_gozer(tmp_path, monkeypatch):
    """(path of a gozer wrapper, fake proc root, file the fake reset command appends to)."""
    proc = tmp_path / "proc"
    proc.mkdir()
    marker = tmp_path / "resets"
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", build_sysfs(str(tmp_path / "sys"), QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", str(proc))
    monkeypatch.setenv("GOZER_HISTORY_ROOT", str(tmp_path / "history"))
    monkeypatch.setenv("GOZER_RESET_CMD", str(script(tmp_path / "reset.sh",
                                                     f'echo "$@" >> {marker}\nexit 0\n')))
    make_proc_dir(proc, os.getpid())     # the owner pid must look alive to gozer
    wrapper = script(tmp_path / "gozer", f'exec {sys.executable} {GOZER} "$@"\n')
    return wrapper, proc, marker


def resets(marker):
    return marker.read_text().splitlines() if marker.exists() else []


def test_acquire_reset_release_round_trip(fake_gozer):
    gz, _, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "contract test", exact="0000:03:00.0")
    # gozer leases whole boards: asking for 1 chip grants the board's 2 (spec section 3).
    assert sorted(lease.chips) == BOARD1 and sorted(lease.dev_indices) == [2, 3]
    assert lease.env["TT_VISIBLE_DEVICES"] == ",".join(lease.chips)
    assert lease.units == ("0000000000000001",)
    ours = [c for c in a.status() if c.bdf in lease.chips]
    assert [c.state for c in ours] == ["CLAIMED", "CLAIMED"]
    assert all(c.lease_pid == os.getpid() and c.board == "0000000000000001" for c in ours)
    a.reset(lease)
    assert len(resets(marker)) == 1 and sorted(resets(marker)[0].split()[1].split(",")) == BOARD1
    a.release(lease)
    assert len(resets(marker)) == 2            # release resets the chips once more
    assert all(c.state == "FREE" for c in a.status())


def test_a_busy_box_refuses_or_queues_and_the_ticket_can_be_cancelled(fake_gozer):
    gz, _, _ = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    both = a.acquire(4, "orchard:contract", "both boards")
    with pytest.raises(Refused):
        a.acquire(2, "orchard:contract", "no queue")
    with pytest.raises(Queued) as q:
        a.acquire(2, "orchard:contract", "queue", queue=True)
    a.cancel(q.value.ticket)
    with pytest.raises(TicketGone):
        a.claim(q.value.ticket, 2, "orchard:contract", "claim after cancel")
    a.release(both)


def test_reset_refuses_while_a_device_is_open(fake_gozer):
    gz, proc, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "refusal", exact="0000:03:00.0")
    holder = make_proc_dir(proc, 999999, ppid=os.getpid(), devs=[2])   # our child holds chip 2
    assert {c.bdf: c.state for c in a.status()}["0000:03:00.0"] == "HELD"
    with pytest.raises(Refused, match="still open"):
        a.reset(lease)
    assert resets(marker) == []
    shutil.rmtree(holder)
    a.reset(lease)
    a.release(lease)


def test_a_dead_owners_lease_is_stale_and_the_next_acquire_reaps_it_without_a_reset(fake_gozer):
    # This is the gozer behavior recovery relies on (orchard/handoff.py, recover): the lease of a
    # dead supervisor shows STALE with that pid, and the next acquire reaps it and grants the board.
    gz, proc, marker = fake_gozer
    dead = make_proc_dir(proc, 777777)                 # the supervisor that is about to die
    old = GozerAdapter(gozer=str(gz), owner_pid=777777).acquire(1, "orchard:old", "coder",
                                                               exact="0000:03:00.0")
    shutil.rmtree(dead)                                # it dies; no device is open
    states = {c.bdf: c for c in GozerAdapter(gozer=str(gz)).status() if c.bdf in old.chips}
    assert {c.state for c in states.values()} == {"STALE"}
    assert {c.lease_pid for c in states.values()} == {777777}
    new = GozerAdapter(gozer=str(gz)).acquire(1, "orchard:new", "after restart",
                                              exact="0000:03:00.0")
    assert sorted(new.chips) == BOARD1 and new.lease_id != old.lease_id
    assert resets(marker) == []                        # reaped without a reset
    GozerAdapter(gozer=str(gz)).release(new)


def test_a_holder_outside_the_owners_tree_is_held_foreign_and_blocks_the_reset(fake_gozer):
    # A tt-model container's server is never the supervisor's descendant (spec section 6, step 3).
    gz, proc, marker = fake_gozer
    a = GozerAdapter(gozer=str(gz))
    lease = a.acquire(1, "orchard:contract", "foreign", exact="0000:03:00.0")
    holder = make_proc_dir(proc, 888888, ppid=1, devs=[2])
    assert {c.bdf: c.state for c in a.status()}["0000:03:00.0"] == "HELD-FOREIGN"
    with pytest.raises(Refused):
        a.reset(lease)
    assert resets(marker) == []
    shutil.rmtree(holder)
    a.release(lease)
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest -q -p no:cacheprovider -rs tests/test_gozer_contract.py`
Expected: PASS (5 tests). If the file reports `SKIPPED`, gozer was not found; record the skip reason in the task report and do not count it as passing. If a test fails, read the real gozer output (`python3 ~/code/tt-gozer/bin/gozer <cmd> --json` with the same `GOZER_*` variables). Where the adapter misreads it, fix the adapter. Never relax the expectations that gozer leases whole boards and that a reset with a device open is refused. If gozer behaves differently from what this plan read in `cli.py` (for example, `acquire --ticket` for a cancelled ticket does not exit 13), stop and report it before changing any test. Report any adapter change in the task report.

- [ ] **Step 3: Mutation check: watch the contract test catch a wrong key**

In `GozerAdapter.status`, change `lease_pid=c.get("pid")` to `lease_pid=c.get("owner_pid")`. Run `tests/test_gozer_contract.py::test_acquire_reset_release_round_trip`. Expected: FAIL (`lease_pid` is None). The FakeRun suite cannot catch this, because its JSON was written to match the adapter. Restore: PASS.

- [ ] **Step 4: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add tests/test_gozer_contract.py
git commit -m "Check the gozer adapter against the real gozer command line with fake roots"
```

---

### Task 5: The single-tenant adapter

**Files:**
- Create: `orchard/adapters/single_tenant.py`
- Test: `tests/test_single_tenant.py`

**Interfaces:**
- Consumes: Task 2 types; `run_command`, `CommandResult` (Task 1); `RESET_TIMEOUT_S`.
- Produces: `device_holders(proc_root: str = "/proc") -> dict[int, list[int]]` (device index to sorted pids); `SingleTenantAdapter(chips: Sequence[tuple[str, int]], *, board: str = "local", chips_per_board: int = CHIPS_PER_BOARD, proc_root: str = "/proc", reset_argv: Sequence[str] | None = None, run=run_command, reset_timeout: float = RESET_TIMEOUT_S, owner_pid: int | None = None, which=shutil.which, gozer_state_dirs: Sequence[str] = ("/tmp/tt-gozer",))`, implementing `LeaseAdapter`. Leases cover whole boards (units `local0`, `local1`, ...). Building it never refuses, so a restarted supervisor can build it while its old coder still holds a board. `acquire` refuses permanently when `gozer` is on PATH or a gozer state directory exists, and refuses a board whose device is open. A missing `reset_argv` makes `reset` refuse permanently.

- [ ] **Step 1: Write the failing tests**

```python
"""SingleTenantAdapter: one tenant, no lease tool, whole boards, a fake /proc."""
import os

import pytest

from fakes import FakeRun
from orchard.adapters import LeaseLost, Refused, ResetFailed
from orchard.adapters.single_tenant import SingleTenantAdapter, device_holders

# Two p300c boards of two chips, as on this Quietbox 2.
CHIPS = [("0000:01:00.0", 0), ("0000:02:00.0", 1), ("0000:03:00.0", 2), ("0000:04:00.0", 3)]
BOARD0 = ("0000:01:00.0", "0000:02:00.0")
BOARD1 = ("0000:03:00.0", "0000:04:00.0")


def fake_proc(tmp_path, holders):
    """holders: {pid: [link targets]}. Returns the proc root."""
    root = tmp_path / "proc"
    root.mkdir(exist_ok=True)
    for pid, targets in holders.items():
        fd = root / str(pid) / "fd"
        fd.mkdir(parents=True, exist_ok=True)
        for n, t in enumerate(targets):
            os.symlink(t, fd / str(n + 3))
    return str(root)


def adapter(tmp_path, holders=None, **kw):
    # No lease tool on this "machine": the tests never look at the real PATH or /tmp/tt-gozer.
    kw.setdefault("which", lambda name: None)
    kw.setdefault("gozer_state_dirs", ())
    return SingleTenantAdapter(CHIPS, proc_root=fake_proc(tmp_path, holders or {}), **kw)


def test_device_holders_reads_only_tenstorrent_nodes(tmp_path):
    root = fake_proc(tmp_path, {10: ["/dev/tenstorrent/0", "/dev/null"],
                                11: ["/dev/tenstorrent_x", "/dev/tenstorrent/1"]})
    assert device_holders(root) == {0: [10], 1: [11]}


def test_a_restarted_supervisor_can_build_it_while_its_coder_holds_a_board(tmp_path):
    # After a crash the old coder still holds board 0. Building the adapter must still work.
    a = adapter(tmp_path, {321: ["/dev/tenstorrent/0"]})
    assert {c.bdf: c.state for c in a.status()}["0000:01:00.0"] == "BUSY-UNTRACKED"


def test_acquire_refuses_a_board_whose_device_is_open(tmp_path):
    a = adapter(tmp_path, {321: ["/dev/tenstorrent/0"]})
    with pytest.raises(Refused, match="pid 321 holds /dev/tenstorrent/0"):
        a.acquire(2, "w", "r", exact="0000:01:00.0")
    assert a.acquire(2, "w", "r").chips == BOARD1       # the other board is still free


def test_one_chip_grants_the_whole_board(tmp_path):
    # UMD expands TT_VISIBLE_DEVICES to the whole board, as gozer notes, so leases are per board.
    lease = adapter(tmp_path).acquire(1, "orchard:t", "r")
    assert lease.chips == BOARD0 and lease.dev_indices == (0, 1)
    assert lease.env == {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}
    assert lease.units == ("local0",)


def test_two_leases_never_share_a_board(tmp_path):
    a = adapter(tmp_path)
    first = a.acquire(1, "w", "r")
    second = a.acquire(1, "w", "r")
    assert first.chips == BOARD0 and second.chips == BOARD1
    with pytest.raises(Refused, match="0 free"):
        a.acquire(1, "w", "r")


def test_four_chips_take_both_boards(tmp_path):
    lease = adapter(tmp_path).acquire(4, "w", "r")
    assert lease.chips == BOARD0 + BOARD1 and lease.units == ("local0", "local1")


def test_exact_picks_the_board_of_the_named_chip(tmp_path):
    assert adapter(tmp_path).acquire(1, "w", "r", exact="0000:04:00.0").chips == BOARD1


def test_a_chip_count_that_splits_a_board_is_a_config_error(tmp_path):
    with pytest.raises(ValueError, match="whole boards"):
        SingleTenantAdapter(CHIPS[:3], proc_root=fake_proc(tmp_path, {}))


@pytest.mark.parametrize("which,dirs,needle", [
    (lambda name: "/home/u/.local/bin/gozer", (), "gozer is on PATH"),
    (lambda name: None, ("STATE",), "gozer state directory"),
])
def test_acquire_refuses_when_a_lease_tool_is_present(tmp_path, which, dirs, needle):
    state = tmp_path / "tt-gozer"
    state.mkdir()
    dirs = tuple(str(state) if d == "STATE" else d for d in dirs)
    a = adapter(tmp_path, which=which, gozer_state_dirs=dirs)
    with pytest.raises(Refused, match=needle) as exc:
        a.acquire(2, "w", "r")
    assert exc.value.permanent


def test_there_is_no_queue(tmp_path):
    with pytest.raises(Refused, match="no queue"):
        adapter(tmp_path).claim("t", 1, "w", "r")


def test_release_does_not_touch_the_hardware(tmp_path):
    run = FakeRun()          # any call fails the test
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    a.release(a.acquire(2, "w", "r"))
    assert run.calls == []


def test_reset_without_a_configured_command_is_a_permanent_refusal(tmp_path):
    a = adapter(tmp_path)
    with pytest.raises(Refused, match="no reset command") as exc:
        a.reset(a.acquire(2, "w", "r"))
    assert exc.value.permanent


def test_reset_runs_the_configured_command_with_the_lease_bdfs(tmp_path):
    run = FakeRun({"tt-smi": [(0, "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    a.reset(a.acquire(2, "w", "r"))
    assert run.argvs() == [["tt-smi", "-r", "0000:01:00.0,0000:02:00.0"]]
    assert run.calls[0]["kill_on_timeout"] is False


def test_reset_refuses_while_a_device_is_open(tmp_path):
    run = FakeRun({"tt-smi": [(0, "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    fake_proc(tmp_path, {555: ["/dev/tenstorrent/1"]})
    with pytest.raises(Refused, match="still open") as exc:
        a.reset(lease)
    assert run.calls == [] and not exc.value.permanent


def test_failed_or_hung_reset_raises(tmp_path):
    run = FakeRun({"tt-smi": [(1, "", "boom"), ("timeout", "", "")]}, key=lambda argv: argv[0])
    a = adapter(tmp_path, run=run, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    with pytest.raises(ResetFailed, match="boom"):
        a.reset(lease)
    with pytest.raises(ResetFailed) as exc:
        a.reset(lease)
    assert exc.value.left_running
    with pytest.raises(Refused, match="still running"):
        a.reset(lease)


def test_reset_of_an_unknown_lease_is_lost(tmp_path):
    a = adapter(tmp_path, reset_argv=["tt-smi", "-r"])
    lease = a.acquire(2, "w", "r")
    a.release(lease)
    with pytest.raises(LeaseLost):
        a.reset(lease)


def test_status_reports_held_claimed_free_and_untracked(tmp_path):
    a = adapter(tmp_path)
    a.acquire(1, "w", "r")
    fake_proc(tmp_path, {600: ["/dev/tenstorrent/0"], 601: ["/dev/tenstorrent/2"]})
    states = {c.bdf: c.state for c in a.status()}
    assert states == {"0000:01:00.0": "HELD", "0000:02:00.0": "CLAIMED",
                      "0000:03:00.0": "BUSY-UNTRACKED", "0000:04:00.0": "FREE"}
    assert {c.board for c in a.status()} == {"local0", "local1"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_single_tenant.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.adapters.single_tenant'`.

- [ ] **Step 3: Write `orchard/adapters/single_tenant.py`**

```python
"""Lease adapter for a machine with no lease tool.

It assumes one tenant: this supervisor. It keeps nothing on disk. A lease is a record in memory of
which configured boards the supervisor is using. Leases cover whole boards (CHIPS_PER_BOARD chips
in the configured order), because UMD expands TT_VISIBLE_DEVICES to the whole board.

Where it refuses, and why:
- Building the adapter never refuses. After a supervisor crash the old coder can still hold a
  board, and the restarted supervisor must be able to build the adapter to recover.
- `acquire` refuses when a lease tool is present (`gozer` on PATH, or a gozer state directory).
  On such a machine other agents share the boards, and this adapter cannot see their leases.
  That refusal is permanent.
- `acquire` refuses a board while any process holds one of its device nodes.
- `reset` refuses permanently when no reset command was configured, because a reset on a machine
  whose tenants are unknown is a choice a person makes once.

Limits:
- The device scan reads /proc/<pid>/fd. An unprivileged user can read that only for its own
  processes, so a device held by another user's process (a root container) is invisible here, as
  it is to gozer (spec section 8, item 4).
- Release does not reset the chips, unlike gozer's release. The supervisor resets where it needs
  clean chips.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from typing import Sequence

from orchard.adapters import ChipState, Lease, LeaseLost, Refused, ResetFailed
from orchard.commands import run_command
from orchard.defaults import CHIPS_PER_BOARD, RESET_TIMEOUT_S

DEV_LINK = re.compile(r"^/dev/tenstorrent/(\d+)$")
GOZER_STATE_DIRS = ("/tmp/tt-gozer",)        # gozer's default GOZER_ROOT


def device_holders(proc_root: str = "/proc") -> dict[int, list[int]]:
    """Device index to the pids that hold /dev/tenstorrent/<index> open."""
    out: dict[int, set[int]] = {}
    try:
        entries = os.listdir(proc_root)
    except FileNotFoundError:
        return {}
    for name in entries:
        if not name.isdigit():
            continue
        fd_dir = os.path.join(proc_root, name, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue                 # exited, or another user's process
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            m = DEV_LINK.match(target)
            if m:
                out.setdefault(int(m.group(1)), set()).add(int(name))
    return {k: sorted(v) for k, v in out.items()}


def _describe(held: dict[int, list[int]]) -> str:
    return ", ".join(f"pid {p} holds /dev/tenstorrent/{i}"
                     for i, pids in sorted(held.items()) for p in pids)


class SingleTenantAdapter:
    def __init__(self, chips: Sequence[tuple[str, int]], *, board: str = "local",
                 chips_per_board: int = CHIPS_PER_BOARD, proc_root: str = "/proc",
                 reset_argv: Sequence[str] | None = None, run=run_command,
                 reset_timeout: float = RESET_TIMEOUT_S, owner_pid: int | None = None,
                 which=shutil.which, gozer_state_dirs: Sequence[str] = GOZER_STATE_DIRS):
        chips = [(str(b), int(i)) for b, i in chips]
        if not chips or len(chips) % chips_per_board:
            raise ValueError(f"configure whole boards: a multiple of {chips_per_board} chips as "
                             f"(bdf, device index), got {len(chips)}")
        self.cpb = chips_per_board
        # Boards in configured order: local0, local1, ...
        self.boards = {f"{board}{k}": chips[k * chips_per_board:(k + 1) * chips_per_board]
                       for k in range(len(chips) // chips_per_board)}
        self.proc_root = proc_root
        self.reset_argv = list(reset_argv) if reset_argv else None
        self.run, self.reset_timeout = run, reset_timeout
        self.owner_pid = os.getpid() if owner_pid is None else owner_pid
        self.which, self.gozer_state_dirs = which, tuple(gozer_state_dirs)
        self.leases: dict[str, Lease] = {}
        self.in_flight: set[str] = set()
        self._n = 0

    def _no_lease_tool(self) -> None:
        found = self.which("gozer")
        if found:
            raise Refused(f"gozer is on PATH ({found}): this machine has a lease tool; use the "
                          "gozer adapter", permanent=True)
        for d in self.gozer_state_dirs:
            if os.path.isdir(d):
                raise Refused(f"a gozer state directory exists ({d}): this machine has a lease "
                              "tool; use the gozer adapter", permanent=True)

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease:
        self._no_lease_tool()
        taken = {u for lease in self.leases.values() for u in lease.units}
        free = [name for name in self.boards if name not in taken]
        held = device_holders(self.proc_root)
        if exact is not None:
            start = next((k for k, name in enumerate(free)
                          if exact in (b for b, _ in self.boards[name])), None)
            if start is None:
                raise Refused(f"{exact} is not on a free configured board")
            free = free[start:]
        else:
            # Without --exact, skip boards that some process holds open.
            free = [name for name in free if not any(i in held for _, i in self.boards[name])]
        need = math.ceil(chips / self.cpb)
        if need > len(free):
            raise Refused(f"asked for {chips} chips ({need} boards), {len(free)} free"
                          + (f"; {_describe(held)}" if held else ""))
        pick = free[:need]
        pairs = [pair for name in pick for pair in self.boards[name]]
        busy = {i: held[i] for _, i in pairs if i in held}
        if busy:
            raise Refused(_describe(busy))
        self._n += 1
        bdfs = tuple(b for b, _ in pairs)
        lease = Lease(lease_id=f"local-{self._n}", chips=bdfs,
                      dev_indices=tuple(i for _, i in pairs),
                      env={"TT_VISIBLE_DEVICES": ",".join(bdfs)}, units=tuple(pick))
        self.leases[lease.lease_id] = lease
        return lease

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease:
        raise Refused("the single-tenant adapter has no queue", permanent=True)

    def cancel(self, ticket: str) -> None:
        return None

    def release(self, lease: Lease) -> None:
        self.leases.pop(lease.lease_id, None)

    def reset(self, lease: Lease) -> None:
        if lease.lease_id not in self.leases:
            raise LeaseLost(f"no lease {lease.lease_id}")
        if lease.lease_id in self.in_flight:
            raise Refused(f"a reset of lease {lease.lease_id} is still running")
        if self.reset_argv is None:
            raise Refused("the single-tenant adapter has no reset command configured; pass "
                          "reset_argv when creating it", permanent=True)
        held = device_holders(self.proc_root)
        busy = {i: held[i] for i in lease.dev_indices if i in held}
        if busy:
            raise Refused("device still open: " + _describe(busy))
        res = self.run([*self.reset_argv, ",".join(lease.chips)], self.reset_timeout,
                       kill_on_timeout=False)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"{' '.join(res.argv)} still running after {self.reset_timeout} s; "
                              "left running", left_running=True)
        if res.returncode != 0:
            raise ResetFailed(f"{' '.join(res.argv)} exited {res.returncode}: "
                              f"{(res.stderr or res.stdout).strip()[:500]}")

    def status(self) -> list[ChipState]:
        held = device_holders(self.proc_root)
        leased = {u for lease in self.leases.values() for u in lease.units}
        out = []
        for name, pairs in self.boards.items():
            mine = name in leased
            for b, i in pairs:
                if i in held:
                    state = "HELD" if mine else "BUSY-UNTRACKED"
                else:
                    state = "CLAIMED" if mine else "FREE"
                out.append(ChipState(bdf=b, state=state, who="orchard" if mine else None,
                                     board=name, dev_index=i,
                                     lease_pid=self.owner_pid if mine else None,
                                     pids_holding=tuple(held.get(i, ()))))
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_single_tenant.py`
Expected: PASS (18 tests).

- [ ] **Step 5: Mutation checks**

1. In `reset`, delete the `if busy: raise Refused("device still open: ...")` lines. Run `tests/test_single_tenant.py::test_reset_refuses_while_a_device_is_open`. Expected: FAIL. Restore: PASS.
2. Lease tool present: in `acquire`, delete the `self._no_lease_tool()` line. Run `tests/test_single_tenant.py::test_acquire_refuses_when_a_lease_tool_is_present`. Expected: FAIL. Restore: PASS.
3. Board grain: in `acquire`, add `[:chips]` to the end of the `pairs = [...]` line. Run `tests/test_single_tenant.py::test_one_chip_grants_the_whole_board`. Expected: FAIL (one chip granted). Restore: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/adapters/single_tenant.py tests/test_single_tenant.py
git commit -m "Add the single-tenant lease adapter for machines without a lease tool"
```

---

### Task 6: The canary and the fake server

**Files:**
- Create: `orchard/canary.py`, `orchard/fake_server.py`
- Test: `tests/test_canary.py`

**Interfaces:**
- Consumes: `CANARY_TIMEOUT_S`, `CANARY_MAX_TOKENS`.
- Produces: `orchard.canary.CanaryError(Exception)`; `post_json(url: str, body: dict, timeout: float) -> dict`; `ask(endpoint: str, model: str, prompt: str, *, http=post_json, timeout: float = CANARY_TIMEOUT_S, max_tokens: int = CANARY_MAX_TOKENS) -> str`; `CanaryResult` (frozen dataclass: `match: bool`, `whitespace_only: bool`, `before: str`, `after: str`); `compare(before: str, after: str) -> CanaryResult`.
- Produces: `orchard.fake_server.make_server(port: int, answer: str, model: str, host: str = "127.0.0.1") -> ThreadingHTTPServer`; `main(argv=None) -> int` (flags `--port`, `--answer`, `--model`). Run as a script: `python3 orchard/fake_server.py --port N --answer TEXT --model NAME`.

- [ ] **Step 1: Write the failing tests**

```python
"""The canary call, the exact comparison, and the fake server it is tested against."""
import threading

import pytest

from orchard.canary import CanaryError, ask, compare
from orchard.fake_server import make_server


def test_ask_sends_a_greedy_chat_request():
    seen = {}

    def http(url, body, timeout):
        seen.update(url=url, body=body, timeout=timeout)
        return {"choices": [{"message": {"role": "assistant", "content": "4"}}]}

    assert ask("http://127.0.0.1:8000/", "qwen", "2+2?", http=http, timeout=5) == "4"
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["stream"] is False
    assert seen["body"]["messages"] == [{"role": "user", "content": "2+2?"}]


def test_ask_turns_a_connection_error_into_canary_error():
    def http(url, body, timeout):
        raise ConnectionRefusedError("refused")

    with pytest.raises(CanaryError, match="refused"):
        ask("http://127.0.0.1:1", "m", "p", http=http)


def test_ask_refuses_a_reply_without_content():
    with pytest.raises(CanaryError, match="choices"):
        ask("http://x", "m", "p", http=lambda u, b, t: {"error": "overloaded"})


def test_identical_answers_match():
    r = compare("Paris", "Paris")
    assert r.match and not r.whitespace_only


def test_whitespace_only_difference_is_flagged():
    r = compare("Paris\n", "Paris")
    assert not r.match and r.whitespace_only


def test_a_different_word_is_not_whitespace_only():
    r = compare("Paris", "Lyon")
    assert not r.match and not r.whitespace_only


def test_fake_server_answers_health_models_and_chat():
    srv = make_server(0, "forty-two", "fake")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        endpoint = f"http://127.0.0.1:{srv.server_address[1]}"
        assert ask(endpoint, "fake", "anything", timeout=5) == "forty-two"
    finally:
        srv.shutdown()
        srv.server_close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_canary.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.canary'`.

- [ ] **Step 3: Write `orchard/canary.py`**

```python
"""Canary prompts: ask a model server one fixed question and compare the answers.

This module owns the chat call the park sequence uses to check a server (spec section 6, steps 2
and 5) and the comparison rule. Decoding is greedy (temperature 0), so a restarted coder with the
same weights should give the same text, and the comparison is exact. A difference in whitespace
alone still fails. The result says so, so a person can judge it quickly. Whether the answer is
identical across a real restart is not yet measured (spec section 12).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from orchard.defaults import CANARY_MAX_TOKENS, CANARY_TIMEOUT_S


class CanaryError(Exception):
    """The server did not give a usable answer."""


def post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise CanaryError(f"{url} returned JSON that is not an object")
    return data


def ask(endpoint: str, model: str, prompt: str, *, http=post_json,
        timeout: float = CANARY_TIMEOUT_S, max_tokens: int = CANARY_MAX_TOKENS) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False}
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    try:
        data = http(url, body, timeout)
    except (OSError, ValueError) as exc:     # URLError and socket errors are OSError
        raise CanaryError(f"canary request to {url} failed: {exc}") from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CanaryError(f"{url} gave no choices[0].message.content: {str(data)[:300]}") from exc
    if not isinstance(text, str):
        raise CanaryError(f"{url} gave content that is not text: {text!r}")
    return text


@dataclass(frozen=True)
class CanaryResult:
    match: bool
    whitespace_only: bool     # the texts differ, and only in whitespace
    before: str
    after: str


def compare(before: str, after: str) -> CanaryResult:
    match = before == after
    return CanaryResult(match=match, whitespace_only=(not match and before.split() == after.split()),
                        before=before, after=after)
```

- [ ] **Step 4: Write `orchard/fake_server.py`**

```python
"""A small OpenAI-shaped HTTP server that opens no device.

orchard/park_check.py runs two of these, one standing in for the coder and one for the CPU
stand-in, so the park sequence can run on a real board without loading a model. Tests run it too.
It answers GET /health, GET /v1/models and POST /v1/chat/completions, always with the same text.
It imports nothing from orchard, so it can run as a plain script.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    answer = ""
    model = ""

    def log_message(self, *args):        # keep test output quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": self.model}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        json.loads(self.rfile.read(n) or b"{}")          # a malformed body fails loudly
        self._send(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": self.answer}}]})


def make_server(port: int, answer: str, model: str, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"answer": answer, "model": model})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fake_server", description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--model", default="fake")
    a = p.parse_args(argv)
    srv = make_server(a.port, a.answer, a.model)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_canary.py`
Expected: PASS (7 tests).

- [ ] **Step 6: Mutation check of the exact comparison**

In `compare`, change `match = before == after` to `match = before.strip() == after.strip()`. Run `tests/test_canary.py::test_whitespace_only_difference_is_flagged`. Expected: FAIL. Restore: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/canary.py orchard/fake_server.py tests/test_canary.py
git commit -m "Add the canary call with an exact comparison, and a fake server to test against"
```

---

### Task 7: Server control

**Files:**
- Create: `orchard/server.py`
- Modify: `tests/fakes.py` (append `FakeProc`, `FakeClock`)
- Test: `tests/test_server.py`, `tests/test_server_live.py`

**Interfaces:**
- Consumes: `run_command`, `CommandResult` (Task 1); `Lease` (Task 2); `canary.ask`, `post_json`, `CanaryError` (Task 6); `CMD_TIMEOUT_S`, `START_TIMEOUT_S`, `STOP_TIMEOUT_S`, `READY_POLL_S`, `STANDIN_READY_S`.
- Produces in `orchard.server`:
  - `KINDS = ("container", "bundle", "process")`; `ServerError(Exception)`; `ServerStarting(ServerError)` (a container start timed out while the server was coming up); `NotReady(ServerError)` with `.waited_s: float`.
  - `ServerSpec` (frozen dataclass): `target: str`, `kind: str`, `port: int`, `model: str`, `profile: str = "default"`, `image_id: str | None = None`, `argv: tuple[str, ...] = ()`; properties `endpoint` and `container_name`.
  - `StopCheck` (frozen dataclass): `stopped: bool`, `checks: dict[str, bool]`, `evidence: dict[str, dict]`.
  - `spawn_session(argv, env, log_path=None) -> subprocess.Popen`.
  - `ServerControl(spec, *, run=run_command, spawn=spawn_session, killpg=os.killpg, http=post_json, clock=time.monotonic, sleep=time.sleep, tt_model="tt-model", docker="docker", timeout=CMD_TIMEOUT_S, stop_timeout=STOP_TIMEOUT_S, start_timeout=START_TIMEOUT_S, ready_poll_s=READY_POLL_S, log_path=None)` with `start(lease: Lease | None) -> None`, `stop() -> dict` (for a container or bundle it includes `mesh_reset: bool`, true when `tt-model stop` says it reset the mesh itself), `confirm_stopped() -> StopCheck`, `wait_ready(budget_s: float) -> float`, `ask(prompt: str) -> str`, `record() -> dict`, `adopt(record: dict) -> None`; attributes `pid`, `pgid`, `dev_indices`, `proc`.
  - `StandIn` (Protocol): `start()`, `ask(prompt) -> str`, `stop()`, `record() -> dict`, `adopt(record)`, `confirm_stopped() -> StopCheck`.
  - `confirm_stopped()` evidence for a container includes `docker_inspect.others_with_all_devices`: names of other containers that map the whole `/dev/tenstorrent` directory. Such a container blocks the stop confirmation only when it is the coder's own.
  - `ServerStandIn(server: ServerControl, ready_budget_s: float = STANDIN_READY_S)` implementing `StandIn`.
- Produces in `tests/fakes.py`: `FakeProc(pid=4321)` (`.poll()`, `.wait(timeout=None)`, `.returncode`, `.polls`), `FakeClock()` (callable returning `.now`; `.sleep(s)` advances it and records `.sleeps`).

- [ ] **Step 1: Append to `tests/fakes.py`**

```python
class FakeProc:
    """A Popen stand-in. Set .returncode to make it 'exit'."""

    def __init__(self, pid=4321):
        self.pid, self.returncode, self.polls = pid, None, 0

    def poll(self):
        self.polls += 1
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class FakeClock:
    """time.monotonic and time.sleep for tests: sleep advances the clock at once."""

    def __init__(self, now=0.0):
        self.now, self.sleeps = now, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
```

- [ ] **Step 2: Write the failing tests**

`tests/test_server.py`:

```python
"""ServerControl with fake commands: start, stop, the stop checks, readiness."""
import signal

import pytest

from fakes import LEASE, FakeClock, FakeProc, FakeRun
from orchard.server import NotReady, ServerControl, ServerError, ServerSpec, ServerStarting

SS_HEADER = "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process\n"
SS_LISTEN = SS_HEADER + "LISTEN 0      4096   127.0.0.1:20000      0.0.0.0:*\n"
CLEAR = {"docker ps": [(0, "", "")], "docker ps-q": [(0, "", "")], "ss -ltn": [(0, SS_HEADER, "")],
         "curl -sS": [(7, "", "curl: (7) Failed to connect to 127.0.0.1 port 20000")],
         "ps -p": [(1, "", "")], "pgrep -g": [(1, "", "")],
         "tt-model stop": [(0, "stopped 1 container(s)", "")], "tt-model serve": [(0, "", "")]}
# A real `docker ps` line from the H5 check (2026-10-02), with the container name tt-model gives it.
H5_LINE = "3c1d2e4f5a6b f0ed6056d85f tt-model-audio8-asr-infinite-p150-default 0.0.0.0:20000->20000/tcp\n"
CONTAINER = ServerSpec("episod/audio8-asr-infinite-p150", "container", 20000, "audio8",
                       image_id="f0ed6056d85f")
BUNDLE = ServerSpec("org/qwen-bundle", "bundle", 20000, "qwen")


def key(argv):
    if argv[:3] == ["docker", "ps", "-q"]:
        return "docker ps-q"
    return " ".join(argv[:2])


def control(spec=CONTAINER, **over):
    run = FakeRun({**CLEAR, **over}, key=key)
    clock = FakeClock()
    spawned, killed = [], []
    proc = FakeProc()

    def spawn(argv, env, log_path=None):
        spawned.append((argv, env))
        return proc

    def killpg(pgid, sig):
        killed.append((pgid, sig))
        if sig == signal.SIGTERM and getattr(proc, "obey_term", True):
            proc.returncode = 0

    ctl = ServerControl(spec, run=run, spawn=spawn, killpg=killpg, clock=clock, sleep=clock.sleep)
    return ctl, run, spawned, killed, proc, clock


def test_spec_names_the_container_as_tt_model_does():
    assert CONTAINER.container_name == "tt-model-audio8-asr-infinite-p150-default"
    assert CONTAINER.endpoint == "http://127.0.0.1:20000"
    with pytest.raises(ValueError):
        ServerSpec("x", "vm", 1, "m")
    with pytest.raises(ValueError):
        ServerSpec("x", "process", 1, "m")      # a process server needs argv


def test_container_start_pins_the_leased_chips():
    ctl, run, *_ = control()
    ctl.start(LEASE)
    argv = run.argvs()[0]
    assert argv[:3] == ["tt-model", "serve", "episod/audio8-asr-infinite-p150"]
    assert argv[argv.index("--device-id") + 1] == "0,1"
    assert argv[argv.index("--port") + 1] == "20000" and "--detach" in argv
    assert run.calls[0]["env"]["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert ctl.dev_indices == (0, 1)


def test_container_start_failure_raises():
    ctl, *_ = control(**{"tt-model serve": [(1, "", "no such package")]})
    with pytest.raises(ServerError, match="no such package"):
        ctl.start(LEASE)


def test_container_start_has_its_own_budget_and_is_never_killed():
    ctl, run, *_ = control()
    ctl.start(LEASE)
    assert run.calls[0]["timeout"] == ctl.start_timeout > ctl.timeout
    assert run.calls[0]["kill_on_timeout"] is False


def test_a_start_that_times_out_while_the_server_comes_up_says_so():
    ctl, *_ = control(**{"tt-model serve": [("timeout", "", "")], "docker ps": [(0, H5_LINE, "")]})
    with pytest.raises(ServerStarting, match="not starting another"):
        ctl.start(LEASE)


def test_a_start_that_times_out_with_nothing_up_is_a_plain_failure():
    ctl, *_ = control(**{"tt-model serve": [("timeout", "", "")]})
    with pytest.raises(ServerError, match="nothing came up") as exc:
        ctl.start(LEASE)
    assert not isinstance(exc.value, ServerStarting)


def test_bundle_start_spawns_in_its_own_session():
    ctl, _, spawned, *_ = control(BUNDLE)
    ctl.start(LEASE)
    argv, env = spawned[0]
    assert argv[:3] == ["tt-model", "serve", "org/qwen-bundle"]
    assert env["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert ctl.pid == ctl.pgid == 4321


def test_container_stop_runs_tt_model_stop_with_its_profile():
    ctl, run, *_ = control()
    assert ctl.stop()["mesh_reset"] is False
    assert run.argvs()[0] == ["tt-model", "stop", "episod/audio8-asr-infinite-p150",
                              "--profile", "default"]


def test_a_stop_that_reset_the_mesh_itself_is_flagged():
    out = "grace period expired; docker sent SIGKILL\nreset the mesh with tt-smi (throwaway container)"
    ctl, *_ = control(**{"tt-model stop": [(0, out, "")]})
    assert ctl.stop()["mesh_reset"] is True


def test_container_is_stopped_when_docker_and_the_port_are_clear():
    ctl, *_ = control()
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert check.stopped and set(check.checks) == {"docker_ps", "docker_devices", "port_closed",
                                                   "health_refused"}


@pytest.mark.parametrize("line", [
    H5_LINE,                                                            # by name
    "aaa f0ed6056d85f some-other-name \n",                              # by image id
    "bbb other:latest renamed 0.0.0.0:20000->20000/tcp\n",              # by published port
])
def test_a_container_still_listed_by_docker_is_not_stopped(line):
    ctl, *_ = control(**{"docker ps": [(0, line, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["docker_ps"] is False


@pytest.mark.parametrize("devices,stopped", [
    ("abc123 /other /dev/tenstorrent/0 \n", False),
    ("abc123 /other /dev/tenstorrent/3 \n", True),       # a chip this lease does not hold
])
def test_a_container_that_maps_our_device_is_not_stopped(devices, stopped):
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")], "docker inspect": [(0, devices, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().stopped is stopped


def test_another_agents_container_mapping_every_device_is_named_and_does_not_block():
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")],
                         "docker inspect": [(0, "abc123 /tti-server /dev/tenstorrent \n", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert check.stopped
    assert check.evidence["docker_inspect"]["others_with_all_devices"] == ["tti-server"]


def test_our_own_container_mapping_every_device_blocks():
    ctl, *_ = control(**{"docker ps": [(0, H5_LINE, "")], "docker ps-q": [(0, "3c1d2e4f5a6b\n", "")],
                         "docker inspect": [(0, "3c1d2e4f5a6b99 /tt-model-audio8-asr-infinite-p150-default "
                                               "/dev/tenstorrent \n", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["docker_devices"] is False


def test_unknown_chips_make_any_mapped_device_count():
    ctl, *_ = control(**{"docker ps-q": [(0, "abc\n", "")],
                         "docker inspect": [(0, "abc123 /other /dev/tenstorrent/3 \n", "")]})
    assert ctl.dev_indices is None
    assert not ctl.confirm_stopped().stopped


def test_a_listening_port_is_not_stopped():
    ctl, *_ = control(**{"ss -ltn": [(0, SS_LISTEN, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["port_closed"] is False


@pytest.mark.parametrize("rc", [0, 28])
def test_only_a_refused_connection_confirms_the_port_is_closed(rc):
    ctl, *_ = control(**{"curl -sS": [(rc, "", "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["health_refused"] is False


def test_bundle_worker_left_in_the_group_is_not_stopped():
    # The parent is gone; a vLLM worker it started still runs in its process group.
    ctl, *_ = control(BUNDLE, **{"pgrep -g": [(0, "4400\n", "")]})
    ctl.adopt({"pid": 4321, "pgid": 4321, "dev_indices": [0, 1]})
    check = ctl.confirm_stopped()
    assert not check.stopped
    assert check.checks["process"] is True and check.checks["process_group"] is False


def test_a_child_server_is_reaped_before_ps_looks_for_it():
    ctl, run, _, _, proc, _ = control(BUNDLE)
    ctl.start(LEASE)
    proc.returncode = 0
    polls_at_ps = []
    inner = ctl.run

    def watching(argv, timeout, **kw):
        if argv[:2] == ["ps", "-p"]:
            polls_at_ps.append(proc.polls)
        return inner(argv, timeout, **kw)

    ctl.run = watching
    ctl.confirm_stopped()
    assert polls_at_ps and polls_at_ps[0] >= 1    # an unreaped child shows in ps as a zombie


def test_a_server_with_no_recorded_pid_is_never_confirmed_stopped():
    ctl, *_ = control(BUNDLE)
    check = ctl.confirm_stopped()
    assert not check.stopped and check.checks["process"] is False


def test_process_stop_sends_sigterm_to_the_group():
    spec = ServerSpec("t/fake", "process", 20990, "fake", argv=("python3", "fake.py"))
    ctl, _, _, killed, _, _ = control(spec)
    ctl.start(None)
    assert ctl.stop()["how"] == "SIGTERM"
    assert killed == [(4321, signal.SIGTERM)]


def test_process_stop_escalates_to_sigkill():
    spec = ServerSpec("t/fake", "process", 20990, "fake", argv=("python3", "fake.py"))
    ctl, _, _, killed, proc, _ = control(spec)
    proc.obey_term = False
    ctl.start(None)
    assert ctl.stop()["how"] == "SIGKILL"
    assert killed == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_wait_ready_returns_after_health_answers_200():
    ctl, _, _, _, _, clock = control(**{"curl -sS": [(7, "000", ""), (0, "200", "")]})
    assert ctl.wait_ready(60) == 5.0
    assert clock.sleeps == [5.0]


def test_wait_ready_gives_up_at_the_budget():
    ctl, *_ = control(**{"curl -sS": [(7, "000", "")]})
    with pytest.raises(NotReady) as exc:
        ctl.wait_ready(20)
    assert exc.value.waited_s >= 20


def test_wait_ready_stops_when_the_server_exits():
    ctl, _, _, _, proc, _ = control(BUNDLE, **{"curl -sS": [(7, "000", "")]})
    ctl.start(LEASE)
    proc.returncode = 1
    with pytest.raises(ServerError, match="exited"):
        ctl.wait_ready(600)
```

`tests/test_server_live.py`:

```python
"""ServerControl's stop checks with the real ps, pgrep, ss and curl against a fake server.

The FakeRun tests check how ServerControl reads command output written by hand. This file checks
that the real tools print what ServerControl expects. It opens no device. A skip is not evidence.
"""
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest

from orchard.server import ServerControl, ServerSpec

MISSING = [t for t in ("ps", "pgrep", "ss", "curl") if shutil.which(t) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}; a skip is not evidence")
FAKE = Path(__file__).resolve().parent.parent / "orchard" / "fake_server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_tools_see_a_live_server_and_then_see_it_gone(tmp_path):
    port = free_port()
    spec = ServerSpec("test/fake", "process", port, "fake",
                      argv=(sys.executable, str(FAKE), "--port", str(port), "--answer", "forty-two"))
    srv = ServerControl(spec, log_path=str(tmp_path / "server.log"), ready_poll_s=0.1,
                        sleep=time.sleep)
    srv.start(None)
    try:
        srv.wait_ready(20)
        live = srv.confirm_stopped()
        assert live.checks == {"process": False, "process_group": False, "port_closed": False,
                               "health_refused": False}, live.evidence
        assert srv.ask("anything") == "forty-two"
    finally:
        srv.stop()
    gone = srv.confirm_stopped()
    assert gone.stopped, gone.evidence
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_server.py tests/test_server_live.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.server'`.

- [ ] **Step 4: Write `orchard/server.py`**

```python
"""Start, stop, check and question the model server that holds a board.

This module owns the server side of a park (spec section 6, steps 3 and 5): starting the coder,
stopping it, confirming with the server's own tooling that it has stopped before any reset, and
asking it the canary. It never calls tt-smi. tt-model has no command that lists running servers,
so a stop is confirmed with docker, ps, pgrep, ss and curl, as the gozer-park skill describes
(checked on this box on 2026-10-02 against the Audio8 container).

Three kinds of server:
- "container": a v5.1 container package. `tt-model serve --detach` starts it, pinned to the leased
  chips with --device-id; docker shows it; it is never the supervisor's descendant, so gozer shows
  its chips HELD-FOREIGN while it runs.
- "bundle": a v5/v6 bundle. `tt-model serve` runs in the foreground, so it is started as a child in
  its own session. Its pid is then its process group id, and pgrep -g covers the vLLM workers it
  starts. It stays the supervisor's descendant, which gozer counts as the owner's own work.
- "process": any other command, started the same way and stopped with SIGTERM to its group. The
  CPU stand-in, the park check and the tests use it.

tt-model requires the --device-id count to match the profile's chip count. gozer grants whole
boards, so a 1-chip profile on a board lease fails to start; plan 4 picks profiles that use the
whole lease.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Protocol

from orchard.adapters import Lease
from orchard.canary import ask as canary_ask
from orchard.canary import post_json
from orchard.commands import run_command
from orchard.defaults import (CMD_TIMEOUT_S, READY_POLL_S, STANDIN_READY_S, START_TIMEOUT_S,
                              STOP_TIMEOUT_S)

KINDS = ("container", "bundle", "process")
CURL_COULD_NOT_CONNECT = 7     # curl's exit code when nothing accepts the connection
# `tt-model stop` says so when docker had to SIGKILL and it reset the mesh itself, from a
# throwaway container (`tt-model stop --help`). Its clean output has no such word.
MESH_RESET_TEXT = re.compile(r"\breset", re.IGNORECASE)


class ServerError(Exception):
    """The server could not be started, or exited."""


class ServerStarting(ServerError):
    """The start timed out, and the stop checks show something coming up. Start nothing else."""


class NotReady(ServerError):
    def __init__(self, waited_s: float, detail: str = ""):
        super().__init__(f"not ready after {waited_s} s {detail}".strip())
        self.waited_s = waited_s


@dataclass(frozen=True)
class ServerSpec:
    target: str                    # tt-model package or bundle id (org/name); a label for "process"
    kind: str
    port: int
    model: str                     # the model name the OpenAI API expects, for the canary
    profile: str = "default"
    image_id: str | None = None    # what `tt-model list` prints as `image <id>`
    argv: tuple[str, ...] = ()     # "process" only

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"server kind must be one of {KINDS}, got {self.kind!r}")
        if not 0 < self.port < 65536:
            raise ValueError(f"bad port {self.port}")
        if self.kind == "process" and not self.argv:
            raise ValueError("a process server needs argv")

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def container_name(self) -> str:
        # tt-model names the container tt-model-<name>-<profile> (gozer-park skill).
        return f"tt-model-{self.target.rsplit('/', 1)[-1]}-{self.profile}"


@dataclass(frozen=True)
class StopCheck:
    stopped: bool
    checks: dict[str, bool] = field(hash=False)        # True means "shows nothing running"
    evidence: dict[str, dict] = field(hash=False)


def spawn_session(argv, env, log_path=None) -> subprocess.Popen:
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        return subprocess.Popen(list(argv), env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        if log_path:
            out.close()


class ServerControl:
    def __init__(self, spec: ServerSpec, *, run=run_command, spawn=spawn_session,
                 killpg=os.killpg, http=post_json, clock=time.monotonic, sleep=time.sleep,
                 tt_model: str = "tt-model", docker: str = "docker",
                 timeout: float = CMD_TIMEOUT_S, stop_timeout: float = STOP_TIMEOUT_S,
                 start_timeout: float = START_TIMEOUT_S, ready_poll_s: float = READY_POLL_S,
                 log_path: str | None = None):
        self.spec = spec
        self.run, self.spawn, self.killpg, self.http = run, spawn, killpg, http
        self.clock, self.sleep = clock, sleep
        self.tt_model, self.docker = tt_model, docker
        self.timeout, self.stop_timeout, self.ready_poll_s = timeout, stop_timeout, ready_poll_s
        self.start_timeout = start_timeout
        self.log_path = log_path
        self.proc = None                                  # our child, when we started it
        self.pid: int | None = None
        self.pgid: int | None = None
        self.dev_indices: tuple[int, ...] | None = None   # None: unknown, so any device counts

    # ---- ledger form --------------------------------------------------------------------------

    def record(self) -> dict:
        return {"kind": self.spec.kind, "target": self.spec.target, "port": self.spec.port,
                "pid": self.pid, "pgid": self.pgid,
                "dev_indices": None if self.dev_indices is None else list(self.dev_indices)}

    def adopt(self, record: dict) -> None:
        """Take over a server a previous supervisor started, from its ledger record."""
        self.proc = None
        self.pid, self.pgid = record.get("pid"), record.get("pgid")
        devs = record.get("dev_indices")
        self.dev_indices = None if devs is None else tuple(int(i) for i in devs)

    # ---- start and stop -----------------------------------------------------------------------

    def _env(self, lease: Lease | None) -> dict:
        env = dict(os.environ)
        env.pop("TT_VISIBLE_DEVICES", None)      # a server without a lease sees no chip list
        if lease is not None:
            env.update(lease.env)
        return env

    def start(self, lease: Lease | None) -> None:
        s = self.spec
        if s.kind == "container":
            if lease is None:
                raise ServerError("a container server needs a lease")
            # --device-id pins the container to the leased chips. Without it tt-model picks free
            # chips itself, which may be on a board this lease does not hold.
            argv = [self.tt_model, "serve", s.target, "--detach", "--local-only",
                    "--no-update-check", "--port", str(s.port), "--profile", s.profile,
                    "--device-id", ",".join(str(i) for i in lease.dev_indices)]
            # The start has its own budget and is never killed: a killed `tt-model serve` can leave
            # docker bringing the container up while the caller believes it failed.
            self.dev_indices = tuple(lease.dev_indices)
            res = self.run(argv, self.start_timeout, env=self._env(lease), kill_on_timeout=False)
            if res.timed_out:
                check = self.confirm_stopped()
                if not check.stopped:
                    raise ServerStarting(f"tt-model serve still running after {self.start_timeout} s "
                                         f"and the server is coming up ({check.checks}); "
                                         "not starting another")
                raise ServerError(f"tt-model serve still running after {self.start_timeout} s and "
                                  "nothing came up")
            if res.returncode != 0:
                raise ServerError(f"tt-model serve exited {res.returncode}: "
                                  f"{(res.stderr.strip() or res.stdout.strip())[:500]}")
        else:
            argv = (list(s.argv) if s.kind == "process" else
                    [self.tt_model, "serve", s.target, "--local-only", "--no-update-check",
                     "--port", str(s.port)])
            self.proc = self.spawn(argv, self._env(lease), self.log_path)
            # start_new_session made the child a process group leader: its pid is its group id.
            self.pid = self.pgid = self.proc.pid
        self.dev_indices = tuple(lease.dev_indices) if lease is not None else ()

    def stop(self) -> dict:
        s = self.spec
        if s.kind == "process":
            return self._signal_group()
        argv = [self.tt_model, "stop", s.target] + (["--profile", s.profile]
                                                     if s.kind == "container" else [])
        res = self.run(argv, self.stop_timeout)
        if self.proc is not None:
            self.proc.poll()          # reap a bundle's `tt-model serve`
        # A mesh reset by tt-model runs outside gozer, so the caller waits longer for quiet chips.
        return {**res.record(), "mesh_reset": bool(MESH_RESET_TEXT.search(res.stdout + res.stderr))}

    def _signal_group(self) -> dict:
        if self.pgid is None:
            return {"how": "not started"}
        try:
            self.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return {"how": "already gone"}
        if self.proc is None:
            return {"how": "SIGTERM (adopted; the stop checks decide)"}
        deadline = self.clock() + self.stop_timeout
        while self.clock() < deadline:
            if self.proc.poll() is not None:
                return {"how": "SIGTERM"}
            self.sleep(0.2)
        try:
            self.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return {"how": "SIGKILL"}

    # ---- confirming the stop ------------------------------------------------------------------

    def confirm_stopped(self) -> StopCheck:
        checks: dict[str, bool] = {}
        ev: dict[str, dict] = {}
        if self.proc is not None:
            # An exited child stays a zombie until it is waited for, and ps lists zombies.
            self.proc.poll()
        if self.spec.kind == "container":
            self._check_docker(checks, ev)
        else:
            self._check_process(checks, ev)
        self._check_port(checks, ev)
        return StopCheck(all(checks.values()), checks, ev)

    def _is_mine(self, line: str) -> bool:
        parts = line.split(None, 3)
        if len(parts) < 3:
            return False
        image, name = parts[1], parts[2]
        ports = parts[3] if len(parts) > 3 else ""
        return (name == self.spec.container_name
                or (self.spec.image_id is not None and self.spec.image_id in image)
                or f":{self.spec.port}->" in ports)

    def _maps_our_device(self, paths: list[str], mine: bool) -> bool:
        # tt-model maps the whole directory in some cases. Another agent's container can do the
        # same (common for inference-server style runs); that one does not hold our chips any more
        # than the rest of the box, so only our own container's whole-directory mapping blocks.
        if "/dev/tenstorrent" in paths:
            return mine
        if self.dev_indices is None:               # unknown chips: any device counts
            return any(p.startswith("/dev/tenstorrent/") for p in paths)
        return any(p == f"/dev/tenstorrent/{i}" for i in self.dev_indices for p in paths)

    def _check_docker(self, checks: dict, ev: dict) -> None:
        ps = self.run([self.docker, "ps", "--format", "{{.ID}} {{.Image}} {{.Names}} {{.Ports}}"],
                      self.timeout)
        mine = [ln for ln in ps.stdout.splitlines() if self._is_mine(ln)]
        mine_ids = [ln.split()[0] for ln in mine]
        checks["docker_ps"] = ps.returncode == 0 and not mine
        ev["docker_ps"] = {**ps.record(), "matching": mine}
        ids = self.run([self.docker, "ps", "-q"], self.timeout)
        id_list = ids.stdout.split()
        held, others, inspect_ok = [], [], True
        if ids.returncode == 0 and id_list:
            # A privileged container or one with a mounted /dev may list no device, so this check
            # runs in addition to the one above. Each line: full id, name, mapped device paths.
            ins = self.run([self.docker, "inspect", "--format",
                            "{{.Id}} {{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}",
                            *id_list], self.timeout)
            inspect_ok = ins.returncode == 0
            for ln in ins.stdout.splitlines():
                parts = ln.split()
                if len(parts) < 2:
                    continue
                is_mine = any(parts[0].startswith(i) for i in mine_ids)
                if self._maps_our_device(parts[2:], is_mine):
                    held.append(ln)
                elif "/dev/tenstorrent" in parts[2:]:
                    others.append(parts[1].lstrip("/"))
            ev["docker_inspect"] = {**ins.record(), "holding": held,
                                    "others_with_all_devices": others}
        checks["docker_devices"] = ids.returncode == 0 and inspect_ok and not held

    def _check_process(self, checks: dict, ev: dict) -> None:
        if self.pid is None or self.pgid is None:
            checks["process"] = False
            ev["process"] = {"error": "no pid recorded for this server"}
            return
        ps = self.run(["ps", "-p", str(self.pid), "-o", "pid="], self.timeout)
        checks["process"] = ps.returncode == 1 and not ps.stdout.strip()
        # pgrep -g matches process group ids, not command lines, so it cannot match itself.
        pg = self.run(["pgrep", "-g", str(self.pgid)], self.timeout)
        checks["process_group"] = pg.returncode == 1 and not pg.stdout.strip()
        ev["ps"], ev["pgrep"] = ps.record(), pg.record()

    def _check_port(self, checks: dict, ev: dict) -> None:
        ss = self.run(["ss", "-ltn", f"( sport = :{self.spec.port} )"], self.timeout)
        # ss exits 0 whether or not it finds a listener, so read the rows after the header.
        listening = [ln for ln in ss.stdout.splitlines()[1:] if ln.strip()]
        checks["port_closed"] = ss.returncode == 0 and not listening
        curl = self.run(["curl", "-sS", "--max-time", "3", f"{self.spec.endpoint}/health"],
                        self.timeout)
        # 0 means a server answered. 28 means something accepted and did not answer in time.
        # Only "could not connect" confirms that nothing listens.
        checks["health_refused"] = curl.returncode == CURL_COULD_NOT_CONNECT
        ev["ss"], ev["curl"] = ss.record(), curl.record()

    # ---- readiness and the canary -------------------------------------------------------------

    def wait_ready(self, budget_s: float) -> float:
        t0 = self.clock()
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(f"the server exited with code {self.proc.returncode} "
                                  "before it was ready")
            r = self.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "3",
                          f"{self.spec.endpoint}/health"], self.timeout)
            if r.returncode == 0 and r.stdout.strip() == "200":
                return round(self.clock() - t0, 3)
            waited = self.clock() - t0
            if waited >= budget_s:
                raise NotReady(round(waited, 3), r.stderr.strip()[:300])
            self.sleep(self.ready_poll_s)

    def ask(self, prompt: str) -> str:
        return canary_ask(self.spec.endpoint, self.spec.model, prompt, http=self.http)


class StandIn(Protocol):
    def start(self) -> None: ...

    def ask(self, prompt: str) -> str: ...

    def stop(self) -> None: ...

    def record(self) -> dict: ...

    def adopt(self, record: dict) -> None: ...

    def confirm_stopped(self) -> StopCheck: ...


class ServerStandIn:
    """The CPU stand-in: a local server process. It gets no lease and no chip list.

    The spec requires that the stand-in never touch the coder's weight or tensor caches (a cleared
    cache turns a 2-3 min warm restart into a 30 min cold boot). This class cannot check that; the
    stand-in's command line, set in plan 4, must point at its own cache.
    """

    def __init__(self, server: ServerControl, ready_budget_s: float = STANDIN_READY_S):
        if server.spec.kind == "container":
            raise ValueError("the CPU stand-in runs as a process or bundle, never a container")
        self.server, self.ready_budget_s = server, ready_budget_s

    def start(self) -> None:
        self.server.start(None)
        self.server.wait_ready(self.ready_budget_s)

    def ask(self, prompt: str) -> str:
        return self.server.ask(prompt)

    def stop(self) -> None:
        self.server.stop()

    def record(self) -> dict:
        return self.server.record()

    def adopt(self, record: dict) -> None:
        self.server.adopt(record)

    def confirm_stopped(self) -> StopCheck:
        return self.server.confirm_stopped()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider -rs tests/test_server.py tests/test_server_live.py`
Expected: PASS (`test_server.py` 29 tests, `test_server_live.py` 1 test). If the live test is skipped, record the reason in the task report; it is not evidence.

- [ ] **Step 6: Mutation checks**

1. In `_check_process`, replace the `checks["process_group"] = ...` line with `checks["process_group"] = True`. Run `tests/test_server.py::test_bundle_worker_left_in_the_group_is_not_stopped`. Expected: FAIL. Restore: PASS.
2. In `confirm_stopped`, delete the `self.proc.poll()` call (keep the `if`, use `pass`). Run `tests/test_server.py::test_a_child_server_is_reaped_before_ps_looks_for_it`. Expected: FAIL (no poll happened before `ps -p`). Restore: PASS. The live test cannot catch this mutation: there `stop()` polls the child until it exits, so it is reaped before `confirm_stopped` runs.
3. In `_check_port`, change `== CURL_COULD_NOT_CONNECT` to `!= 0`. Run `tests/test_server.py::test_only_a_refused_connection_confirms_the_port_is_closed`. Expected: FAIL for `rc=28`. Restore: PASS.
4. Whole-directory rule: in `_maps_our_device`, change `return mine` to `return True`. Run `tests/test_server.py::test_another_agents_container_mapping_every_device_is_named_and_does_not_block`. Expected: FAIL. Restore: PASS.
5. Start budget: in `start`, change the container `self.run(argv, self.start_timeout, env=self._env(lease), kill_on_timeout=False)` to `self.run(argv, self.timeout, env=self._env(lease))`. Run `tests/test_server.py::test_container_start_has_its_own_budget_and_is_never_killed`. Expected: FAIL. Restore: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/server.py tests/fakes.py tests/test_server.py tests/test_server_live.py
git commit -m "Add server control: start, stop, and confirm the stop with docker, ps, pgrep, ss and curl"
```

---

### Task 8: The park decision, chip quiet check and handoff progress

**Files:**
- Create: `orchard/handoff.py` (first part)
- Modify: `orchard/ledger.py` (`replay_state`, the `restore` branch)
- Test: `tests/test_handoff_progress.py`, `tests/test_ledger.py` (append one test)

**Interfaces:**
- Consumes: `ChipState`, `Lease`, `boards_of` (Task 2); the `Ledger` entries format (`event`, `stage`, `data`).
- Produces in `orchard.handoff`:
  - `PARK_STEPS = ("note", "canary_before", "standin_started", "standin", "stop_sent", "stopped", "reset")`, `ABANDONED = "abandoned"` (the park step that closes a park before the coder was told to stop), `RESTORE_STEPS = ("reset", "serve", "ready", "canary", "resumed")`, `NOTE_KEYS = ("goal", "stage", "evidence", "next_action", "check_on_return")`.
  - `Blocked(Exception)` with `.reason: str`, `.evidence: dict`.
  - `ParkDecision` (frozen dataclass: `action: str` in `{"use_free", "park", "wait"}`, `free_boards: tuple[str, ...]`, `server_boards: tuple[str, ...]`, `boards_needed: int`; property `park_needed: bool`).
  - `decide_park(server_boards, chips: list[ChipState], boards_needed: int) -> ParkDecision`.
  - `chips_quiet(chips: list[ChipState], lease: Lease, accept=("CLAIMED",)) -> tuple[bool, dict[str, str]]`.
  - `Progress` (frozen dataclass: `phase: str` in `{"idle", "parking", "parked", "restoring"}`, `stage`, `park_done: tuple[str, ...]`, `restore_done: tuple[str, ...]`, `lease: dict | None`, `owner_pid: int | None`, `server: dict | None`, `standin: dict | None`, `canary_before: dict | None`, `mesh_reset: bool`).
  - `progress(entries: list[dict]) -> Progress`. An `abandoned` park entry ends the handoff (phase "idle"). Entries with no step (the plan 1 form) are read as `replay_state` reads them.
- Changes `orchard.ledger.replay_state`: a `park` entry with step `abandoned` clears `parked`; a `restore` entry clears `parked` only when its `data.step` is absent or `"resumed"`. `replay_state` is the authority on whether the run is parked; `progress()` derives from the same entries and must agree (parked exactly when the phase is not "idle"; Task 9 tests this after every entry).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ledger.py`:

```python
def test_a_restore_step_before_resumed_keeps_the_run_parked(tmp_path):
    # A restore runs in several steps (orchard/handoff.py). The coder is back only after the last.
    with Ledger(tmp_path / "l.jsonl") as led:
        led.append("park", 2, step="reset")
        led.append("restore", 2, step="serve")
        assert replay_state(led.read())["parked"] is True
        led.append("restore", 2, step="resumed")
        assert replay_state(led.read())["parked"] is False


def test_an_abandoned_park_is_not_parked(tmp_path):
    # A park closed before the coder was told to stop: the coder never left.
    with Ledger(tmp_path / "l.jsonl") as led:
        led.append("park", 2, step="note")
        assert replay_state(led.read())["parked"] is True
        led.append("park", 2, step="abandoned")
        assert replay_state(led.read())["parked"] is False
```

`tests/test_handoff_progress.py`:

```python
"""decide_park, chips_quiet and progress: pure functions over chip states and ledger entries."""
import pytest

from fakes import LEASE
from orchard.adapters import ChipState
from orchard.handoff import chips_quiet, decide_park, progress


def board(serial, first, state, who=None):
    return [ChipState(f"0000:0{first}:00.0", state, who, board=serial),
            ChipState(f"0000:0{first + 1}:00.0", state, who, board=serial)]


# The large tier serves Qwen3.8-27B on all 4 chips; the small tier on 2 (operator, 2026-10-02).
LARGE = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "HELD-FOREIGN", "orchard:run")
SMALL = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "FREE")


@pytest.mark.parametrize("need", [1, 2])
def test_large_tier_on_both_boards_parks_for_every_hardware_stage(need):
    d = decide_park({"B0", "B1"}, LARGE, need)
    assert d.action == "park" and d.park_needed and d.free_boards == ()


def test_small_tier_on_one_board_leaves_the_other_free():
    d = decide_park({"B0"}, SMALL, 1)
    assert d.action == "use_free" and not d.park_needed and d.free_boards == ("B1",)


def test_small_tier_and_a_two_board_stage_parks():
    assert decide_park({"B0"}, SMALL, 2).action == "park"


def test_a_board_leased_by_someone_else_is_not_free():
    chips = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "CLAIMED", "claude:x")
    assert decide_park({"B0"}, chips, 1).action == "park"


def test_a_board_busy_during_a_neighbours_reset_is_not_free():
    chips = board("B0", 1, "HELD-FOREIGN", "orchard:run") + board("B1", 3, "BUSY-UNTRACKED")
    assert decide_park({"B0"}, chips, 1).action == "park"


def test_no_server_board_and_nothing_free_waits():
    chips = board("B0", 1, "CLAIMED", "claude:x") + board("B1", 3, "CLAIMED", "claude:y")
    assert decide_park(set(), chips, 1).action == "wait"


def test_bad_arguments_are_errors():
    with pytest.raises(ValueError, match="not in the chip list"):
        decide_park({"B9"}, SMALL, 1)
    with pytest.raises(ValueError, match="boards_needed"):
        decide_park({"B0"}, SMALL, 3)


def chip(bdf, state):
    return ChipState(bdf, state, "orchard:run")


def test_chips_quiet_needs_every_lease_chip_claimed():
    ok, states = chips_quiet([chip("0000:01:00.0", "CLAIMED"), chip("0000:02:00.0", "CLAIMED")], LEASE)
    assert ok and states == {"0000:01:00.0": "CLAIMED", "0000:02:00.0": "CLAIMED"}


@pytest.mark.parametrize("second", ["HELD", "HELD-FOREIGN", "BUSY-UNTRACKED", "STALE", "FREE"])
def test_chips_quiet_refuses_any_other_state(second):
    ok, _ = chips_quiet([chip("0000:01:00.0", "CLAIMED"), chip("0000:02:00.0", second)], LEASE)
    assert not ok


def test_chips_quiet_refuses_a_lease_chip_missing_from_status():
    ok, _ = chips_quiet([chip("0000:01:00.0", "CLAIMED")], LEASE)
    assert not ok


def test_chips_quiet_accepts_extra_states_when_asked():
    ok, _ = chips_quiet([chip("0000:01:00.0", "STALE"), chip("0000:02:00.0", "STALE")], LEASE,
                        accept=("CLAIMED", "STALE"))
    assert ok


def e(event, step, stage=2, **data):
    return {"event": event, "stage": stage, "data": {"step": step, **data}}


def test_an_empty_ledger_is_idle():
    assert progress([]).phase == "idle"


def test_park_steps_are_collected_in_order():
    entries = [e("park", "note", lease=LEASE.record(), owner_pid=100, server={"kind": "container"}),
               e("park", "canary_before", canary={"path": "c.txt", "sha256": "ab"}),
               {"event": "measurement", "stage": 2, "data": {"label": "measured"}}]
    p = progress(entries)
    assert (p.phase, p.park_done, p.stage) == ("parking", ("note", "canary_before"), 2)
    assert p.lease == LEASE.record() and p.owner_pid == 100
    assert p.canary_before == {"path": "c.txt", "sha256": "ab"} and p.server == {"kind": "container"}


def test_reset_means_parked_and_restore_steps_mean_restoring():
    entries = [e("park", s) for s in ("note", "canary_before", "standin", "stop_sent", "stopped", "reset")]
    assert progress(entries).phase == "parked"
    entries += [e("restore", "reset"), e("restore", "serve", server={"pid": 9})]
    p = progress(entries)
    assert p.phase == "restoring" and p.restore_done == ("reset", "serve") and p.server == {"pid": 9}


def test_restart_clears_the_restore_steps():
    entries = [e("park", "note"), e("park", "reset"), e("restore", "reset"), e("restore", "serve"),
               e("restore", "restart")]
    p = progress(entries)
    assert p.phase == "restoring" and p.restore_done == ()


def test_resumed_ends_the_handoff_and_a_new_note_starts_another():
    entries = [e("park", "note"), e("park", "reset"), e("restore", "resumed")]
    assert progress(entries).phase == "idle"
    p = progress(entries + [e("park", "note", stage=3)])
    assert p.phase == "parking" and p.park_done == ("note",) and p.stage == 3


def test_the_latest_lease_record_wins():
    new = dict(LEASE.record(), lease_id="new")
    p = progress([e("park", "note", lease=LEASE.record()), e("restore", "restart", lease=new)])
    assert p.lease["lease_id"] == "new"


def test_an_abandoned_park_is_idle():
    entries = [e("park", "note"), e("park", "canary_before"), e("park", "standin_started"),
               e("park", "abandoned")]
    assert progress(entries).phase == "idle"


def test_the_plan_1_form_without_steps_is_read_as_replay_state_reads_it():
    # replay_state: a park entry with no step parks, a restore entry with no step ends the park.
    park = {"event": "park", "stage": 4, "data": {}}
    restore = {"event": "restore", "stage": 4, "data": {"canary": "ok"}}
    assert progress([park]).phase == "parking"
    assert progress([park, restore]).phase == "idle"


def test_a_stop_that_reset_the_mesh_is_remembered():
    entries = [e("park", "note"), e("park", "stop_sent", result={"how": "x", "mesh_reset": True})]
    assert progress(entries).mesh_reset is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_progress.py tests/test_ledger.py`
Expected: `test_handoff_progress.py` fails with `ModuleNotFoundError: No module named 'orchard.handoff'`; `test_a_restore_step_before_resumed_keeps_the_run_parked` fails on its first assert.

- [ ] **Step 3: Change `replay_state` in `orchard/ledger.py`**

Replace

```python
        elif event == "park":
            state["parked"] = True
        elif event == "restore":
            state["parked"] = False
```

with

```python
        elif event == "park":
            # replay_state is the authority on "parked"; orchard/handoff.py progress() reads the
            # same entries and agrees. A park closed before the coder was told to stop ends
            # with step "abandoned": the coder never left.
            state["parked"] = e["data"].get("step") != "abandoned"
        elif event == "restore":
            # A restore runs in several steps (orchard/handoff.py). The coder is back only after
            # the last one. An entry with no step (the plan 1 form) still ends the park.
            if e["data"].get("step") in (None, "resumed"):
                state["parked"] = False
```

- [ ] **Step 4: Create `orchard/handoff.py` with the first part**

```python
"""Park and restore: swap the coder off a board and back (spec section 6).

This module owns the decision whether a stage needs a park, the park and restore sequence, and
recovery after a supervisor restart. The sequence is a function-driven state machine. Each step
writes a `park` or `restore` ledger entry when it completes, so `progress()` can replay the
ledger and say which step completed. A resumed `Handoff` skips the steps already recorded.
`recover()` compares the ledger with the machine after a restart, and the machine wins.

Park steps: note (the handoff note exists and parses), canary_before (the coder answers the
canary), standin_started (the CPU stand-in process exists; its pid is recorded before anything
else can fail), standin (the stand-in answers the canary), stop_sent (`tt-model stop`), stopped
(the stop is confirmed by the server's tooling and by the lease tool), reset (the lease is reset
in place and stays held). The stage test runs next; plan 4 owns it.
Restore steps: reset (the test is gone; the chips are reset again), serve, ready, canary (the
answer must equal the pre-park answer exactly), resumed (the stand-in stops and is checked gone).
A `restart` entry makes the restore start again from its reset.

A park that cannot go on before the coder was told to stop ends with an `abandoned` park entry:
the coder never left, the stand-in is stopped, and the run is not parked.

`orchard.ledger.replay_state` is the authority on whether the run is parked. `progress()` derives
the step detail from the same entries and must agree with it: parked exactly when the phase is
not "idle" (tests/test_handoff.py checks this after every entry).

Any condition the spec says needs a person raises `Blocked` after a ledger `notice`. Nothing is
retried blindly. The retry rule (spec section 3) allows one retry of a reset that gozer refused
because a device was busy (exit 15). A reset that ran and failed (exit 17) blocks at once
(spec section 10). Restarting a model server that died is plan 4's job (spec section 10).
"""
from __future__ import annotations

from dataclasses import dataclass

from orchard.adapters import ChipState, Lease, boards_of

PARK_STEPS = ("note", "canary_before", "standin_started", "standin", "stop_sent", "stopped",
              "reset")
ABANDONED = "abandoned"          # closes a park before the coder was told to stop
RESTORE_STEPS = ("reset", "serve", "ready", "canary", "resumed")
NOTE_KEYS = ("goal", "stage", "evidence", "next_action", "check_on_return")


class Blocked(Exception):
    """The stage is blocked and the run pauses for the operator. The ledger has a notice."""

    def __init__(self, reason: str, **evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


@dataclass(frozen=True)
class ParkDecision:
    action: str                       # "use_free", "park" or "wait"
    free_boards: tuple[str, ...]
    server_boards: tuple[str, ...]
    boards_needed: int

    @property
    def park_needed(self) -> bool:
        return self.action == "park"


def decide_park(server_boards, chips: list[ChipState], boards_needed: int) -> ParkDecision:
    """Does a stage that needs `boards_needed` boards have to park the coder? (spec section 6)

    `server_boards` are the board serials the coder's lease holds: both boards for the large tier,
    one for the small tier. A board is free when every chip on it is FREE with no lease. "use_free"
    means: lease a free board (reacquire with exact=its first chip) and leave the coder loaded.
    "park" means the coder's boards are needed; take any further boards with a lease of their own.
    "wait" means parking cannot supply enough boards; wait in the queue.
    """
    boards = boards_of(chips)
    server = set(server_boards)
    unknown = server - set(boards)
    if unknown:
        raise ValueError(f"server boards {sorted(unknown)} are not in the chip list")
    if not 1 <= boards_needed <= len(boards):
        raise ValueError(f"boards_needed must be 1 to {len(boards)}, got {boards_needed}")
    free = tuple(b for b, cs in boards.items()
                 if b not in server and all(c.state == "FREE" and not c.who for c in cs))
    if len(free) >= boards_needed:
        action = "use_free"
    elif server and len(free) + len(server) >= boards_needed:
        action = "park"
    else:
        action = "wait"
    return ParkDecision(action, free, tuple(sorted(server)), boards_needed)


def chips_quiet(chips: list[ChipState], lease: Lease, accept=("CLAIMED",)) -> tuple[bool, dict[str, str]]:
    """Is every chip of the lease in an accepted state? CLAIMED means leased and no device open."""
    states = {c.bdf: c.state for c in chips if c.bdf in lease.chips}
    quiet = set(states) == set(lease.chips) and all(s in accept for s in states.values())
    return quiet, states


@dataclass(frozen=True)
class Progress:
    phase: str                                  # "idle", "parking", "parked" or "restoring"
    stage: int | None = None
    park_done: tuple[str, ...] = ()
    restore_done: tuple[str, ...] = ()
    lease: dict | None = None                   # the latest lease record
    owner_pid: int | None = None                # the supervisor pid that wrote the latest step
    server: dict | None = None                  # the coder's ServerControl record
    standin: dict | None = None
    canary_before: dict | None = None           # file_evidence of the pre-park answer
    mesh_reset: bool = False                    # `tt-model stop` reset the mesh itself


def progress(entries: list[dict]) -> Progress:
    """Replay the ledger: where is the current handoff? The ledger is the only state.

    Entries without a step (the plan 1 form) count as a park that starts and a restore that ends,
    as `replay_state` reads them, so the two always agree.
    """
    cur = None
    restoring = False
    for e in entries:
        ev = e["event"]
        if ev not in ("park", "restore"):
            continue
        d = e["data"]
        step = d.get("step")
        if ev == "park" and step in ("note", None):
            cur = {"stage": e["stage"], "park_done": [], "restore_done": [], "lease": None,
                   "owner_pid": None, "server": None, "standin": None, "canary_before": None,
                   "mesh_reset": False}
            restoring = False
        if cur is None:
            continue
        for k in ("lease", "owner_pid", "server", "standin"):
            if d.get(k) is not None:
                cur[k] = d[k]
        if ev == "park":
            if step == ABANDONED:
                cur = None
                continue
            if step not in cur["park_done"]:
                cur["park_done"].append(step)
            if step == "canary_before":
                cur["canary_before"] = d.get("canary")
            if step == "stop_sent" and isinstance(d.get("result"), dict):
                cur["mesh_reset"] = bool(d["result"].get("mesh_reset"))
            continue
        restoring = True
        if step in ("resumed", None):
            cur = None
        elif step == "restart":
            cur["restore_done"] = []
        elif step not in cur["restore_done"]:
            cur["restore_done"].append(step)
    if cur is None:
        return Progress("idle")
    if restoring:
        phase = "restoring"
    elif "reset" in cur["park_done"]:
        phase = "parked"
    else:
        phase = "parking"
    return Progress(phase, cur["stage"], tuple(cur["park_done"]), tuple(cur["restore_done"]),
                    cur["lease"], cur["owner_pid"], cur["server"], cur["standin"],
                    cur["canary_before"], cur["mesh_reset"])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_progress.py tests/test_ledger.py`
Expected: PASS (all of `test_ledger.py`, including the existing `test_replay_state_tracks_stage_and_parking`, and 25 tests in `test_handoff_progress.py`).

- [ ] **Step 6: Mutation checks**

1. In `decide_park`, drop the `and all(c.state == "FREE" and not c.who for c in cs)` condition. Run `tests/test_handoff_progress.py::test_a_board_leased_by_someone_else_is_not_free`. Expected: FAIL. Restore: PASS.
2. In `progress`, delete the `elif step == "restart": cur["restore_done"] = []` branch. Run `tests/test_handoff_progress.py::test_restart_clears_the_restore_steps`. Expected: FAIL. Restore: PASS.
3. Missing chip: in `chips_quiet`, drop `set(states) == set(lease.chips) and ` from the `quiet = ...` line. Run `tests/test_handoff_progress.py::test_chips_quiet_refuses_a_lease_chip_missing_from_status`. Expected: FAIL. Restore: PASS.
4. Abandoned park: in `replay_state`, change the park line to `state["parked"] = True`. Run `tests/test_ledger.py::test_an_abandoned_park_is_not_parked`. Expected: FAIL. Restore: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/handoff.py orchard/ledger.py tests/test_handoff_progress.py tests/test_ledger.py
git commit -m "Add the park decision and handoff progress; a multi-step restore keeps the run parked"
```

---

### Task 9: Park and restore

**Files:**
- Modify: `orchard/handoff.py` (append `Budgets`, `Handoff`)
- Modify: `tests/fakes.py` (append the machine fakes)
- Test: `tests/test_handoff.py`

**Interfaces:**
- Consumes: Task 2 exceptions (including `Refused.permanent`); `CanaryError`, `CanaryResult`, `compare` (Task 6); `NotReady`, `ServerError`, `ServerStarting`, `StandIn.confirm_stopped` (Task 7); `progress`, `chips_quiet`, `Blocked`, `NOTE_KEYS`, `ABANDONED` (Task 8); `orchard.ledger.file_evidence`; `QUIET_WAIT_S`, `POLL_S`, `COLD_BOOT_BUDGET_S`, `MESH_RESET_EXTRA_S`.
- Produces in `orchard.handoff`:
  - `Budgets` (frozen dataclass: `quiet_wait_s: float = QUIET_WAIT_S`, `poll_s: float = POLL_S`, `cold_boot_s: float = COLD_BOOT_BUDGET_S`, `mesh_reset_extra_s: float = MESH_RESET_EXTRA_S`).
  - `wait_stopped(server, adapter, lease, *, quiet_wait_s, poll_s, clock, sleep, accept=("CLAIMED",)) -> dict` (keys `ok`, `checks`, `chip_states`, and `others_with_all_devices` or `server_evidence`): the two-sided stop check, shared with Task 10.
  - `Handoff(*, ledger, stage, adapter, server, standin, lease, canary_prompt, note_path, evidence_dir, budgets=Budgets(), clock=time.monotonic, sleep=time.sleep)` with `park() -> Lease` and `restore() -> CanaryResult | None`; attributes `ledger`, `stage`, `adapter`, `server`, `standin`, `lease`, `budgets`, `clock`, `sleep`. `server` is anything with `ServerControl`'s methods; `standin` is a `StandIn`.
  - Ledger entries: `park`/`restore` with `step`, `lease` (record), `owner_pid`, plus step data (`standin_started` carries the stand-in's record with its pid, written before its canary); `notice` with `blocked=True`, `reason`, `evidence` before every `Blocked`; a `park` entry with step `abandoned` after a block before `stop_sent`; `retry` with `what="reset"` (only after a busy refusal); `measurement` entries `park_reset_seconds` and `restore_reset_seconds` (the successful reset call only) and `coder_ready_wait_seconds` (the readiness wait only; `tt-model serve` is outside it).
  - Evidence files are named `<stem>-<next ledger sequence number>.txt` (`canary-before`, `standin-canary`, `canary-after`), opened with mode `x`, and never overwritten.
- Produces in `tests/fakes.py`: `BOARD0`, `BOARD1`, `WHO`, `NOTE`, `CANARY`, `Crash(BaseException)`, `World`, `FakeAdapter`, `FakeServer`, `FakeStandIn`, `make_handoff(tmp_path, world=None, *, ledger_cls=Ledger, ledger_kw=None, budgets=None) -> (Handoff, World, Ledger)`, `steps(ledger) -> list[tuple[str, str]]`.

- [ ] **Step 1: Append the machine fakes to `tests/fakes.py`**

```python
# ---- the machine, for the handoff tests (Tasks 9 to 11) ------------------------------------------

from orchard.adapters import ChipState, LeaseLost, Queued, Refused, ResetFailed  # noqa: E402
from orchard.ledger import Ledger  # noqa: E402
from orchard.server import NotReady, ServerStarting, StopCheck  # noqa: E402

BOARD0 = ("0000:01:00.0", "0000:02:00.0")
BOARD1 = ("0000:03:00.0", "0000:04:00.0")
WHO = "orchard:test"
NOTE = {"goal": "bring up model X", "stage": 2, "evidence": ["evidence/pcc.json"],
        "next_action": "run the decoder test", "check_on_return": "PCC above 0.99"}
CANARY = "What is 2 + 2? Answer with one number."


class Crash(BaseException):
    """The supervisor process dies here. BaseException, so no `except Exception` catches it."""


class World:
    """The machine as the fake adapter, server and stand-in see it.

    One shared object, so a test can kill the supervisor and look at what is left. The coder is a
    tt-model container on board 0: while it runs, gozer shows its chips HELD-FOREIGN.
    """

    def __init__(self, owner_pid=100):
        self.owner_pid = owner_pid          # the live supervisor; any other owner pid is dead
        self.leases = {}                    # lease_id -> (Lease, owner pid)
        self.coder_running = False
        self.coder_starts = 0
        self.coder_answer = "4"
        self.coder_answer_after = None      # if set, what the coder says after a restart
        self.never_ready = False
        self.standin_running = False
        self.standin_answer = "4"
        self.standin_fails = False
        self.standin_survives_stop = False  # the stand-in ignores its stop
        self.stop_mesh_reset = False        # tt-model stop says it reset the mesh itself
        self.start_hangs_but_comes_up = False  # tt-model serve times out while the coder boots
        self.worker_left = False            # a vLLM worker that outlives the server
        self.leave_worker_on_stop = False
        self.crash_in_stop = False          # die inside tt-model stop, after the coder stopped
        self.refuse_resets = 0              # the next n resets are refused (exit 15)
        self.lose_lease_on_reset = False    # the next reset finds the lease taken (exit 18)
        self.reset_fails = False            # tt-smi ran and failed (exit 17)
        self.reset_unavailable = False      # the adapter has no reset command (permanent refusal)
        self.reset_hangs = False
        self.foreign_polls = 0              # the next n status reads show our chips HELD-FOREIGN
        self.queue_script = []              # answers for acquire(queue=True)/claim before a grant
        self.resets = []
        self.events = []
        self.n = 0


class FakeAdapter:
    def __init__(self, world, owner_pid=None):
        self.world = world
        self.owner_pid = owner_pid or world.owner_pid
        self.calls = []

    def _holding(self):
        return self.world.coder_running or self.world.worker_left

    def _grant(self):
        w = self.world
        # gozer reaps a lease whose owner is dead once no device is open.
        for lid, (_, owner) in list(w.leases.items()):
            if owner != w.owner_pid and not self._holding():
                del w.leases[lid]
        busy = {b for lease, _ in w.leases.values() for b in lease.chips}
        if busy & set(BOARD0):
            return None
        w.n += 1
        lease = Lease(f"L{w.n}", BOARD0, (0, 1), {"TT_VISIBLE_DEVICES": ",".join(BOARD0)}, ("B0",))
        w.leases[lease.lease_id] = (lease, self.owner_pid)
        return lease

    def _scripted(self):
        if self.world.queue_script:
            ans = self.world.queue_script.pop(0)
            if ans is not None:
                raise ans

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        self.calls.append(("acquire", chips, queue))
        self._scripted()
        lease = self._grant()
        if lease is None:
            raise Queued("t-busy") if queue else Refused("board 0 is leased")
        return lease

    def claim(self, ticket, chips, who, reason):
        self.calls.append(("claim", ticket))
        self._scripted()
        lease = self._grant()
        if lease is None:
            raise Queued(ticket)
        return lease

    def cancel(self, ticket):
        self.calls.append(("cancel", ticket))

    def release(self, lease):
        self.calls.append(("release", lease.lease_id))
        self.world.leases.pop(lease.lease_id, None)

    def reset(self, lease):
        self.calls.append(("reset", lease.lease_id))
        w = self.world
        entry = w.leases.get(lease.lease_id)
        if entry is None or entry[1] != self.owner_pid:
            raise LeaseLost(f"no lease {lease.lease_id} for pid {self.owner_pid}")
        if w.lose_lease_on_reset:
            w.lose_lease_on_reset = False
            raise LeaseLost(f"the reset ran, and afterwards lease {lease.lease_id} was taken")
        if w.reset_unavailable:
            raise Refused("no reset command configured", permanent=True)
        if w.reset_hangs:
            raise ResetFailed("gozer reset still running; left running", left_running=True)
        if w.reset_fails:
            raise ResetFailed("tt-smi -r failed")
        if w.refuse_resets > 0:
            w.refuse_resets -= 1
            raise Refused("device still open (HELD-FOREIGN)")
        if self._holding():
            raise Refused("device still open")
        w.resets.append(lease.lease_id)

    def status(self):
        w = self.world
        foreign = w.foreign_polls > 0
        if foreign:
            w.foreign_polls -= 1
        owner_of = {b: (lease, o) for lease, o in w.leases.values() for b in lease.chips}
        out = []
        for i, b in enumerate(BOARD0 + BOARD1):
            serial = "B0" if b in BOARD0 else "B1"
            holding = b in BOARD0 and self._holding()
            lo = owner_of.get(b)
            if lo is None:
                out.append(ChipState(b, "BUSY-UNTRACKED" if holding else "FREE", None, board=serial,
                                     dev_index=i, pids_holding=(777,) if holding else ()))
                continue
            _, owner = lo
            if holding or foreign:
                state = "HELD-FOREIGN"     # a container is never the supervisor's descendant
            else:
                state = "CLAIMED" if owner == w.owner_pid else "STALE"
            out.append(ChipState(b, state, WHO, board=serial, dev_index=i, lease_pid=owner,
                                 pids_holding=(777,) if holding else ()))
        return out


class FakeServer:
    def __init__(self, world):
        self.world, self.adopted = world, None

    def start(self, lease):
        self.world.coder_starts += 1
        self.world.events.append("coder_start")
        self.world.coder_running = True
        if self.world.start_hangs_but_comes_up:
            raise ServerStarting("tt-model serve still running; the server is coming up")

    def stop(self):
        w = self.world
        w.events.append("coder_stop")
        w.coder_running = False
        if w.leave_worker_on_stop:
            w.worker_left = True
        if w.crash_in_stop:
            w.crash_in_stop = False
            raise Crash("killed inside tt-model stop")
        return {"how": "tt-model stop", "mesh_reset": w.stop_mesh_reset}

    def confirm_stopped(self):
        # docker sees only the container; a leftover worker is invisible to it.
        ok = not self.world.coder_running
        return StopCheck(ok, {"docker_ps": ok}, {})

    def wait_ready(self, budget_s):
        if self.world.never_ready:
            raise NotReady(budget_s)
        return 20.3

    def ask(self, prompt):
        w = self.world
        if not w.coder_running:
            raise ConnectionRefusedError("coder is down")
        if w.coder_starts > 1 and w.coder_answer_after is not None:
            return w.coder_answer_after
        return w.coder_answer

    def record(self):
        return {"kind": "container", "port": 20000}

    def adopt(self, record):
        self.adopted = record


class FakeStandIn:
    def __init__(self, world):
        self.world = world

    def start(self):
        self.world.events.append("standin_start")
        self.world.standin_running = True

    def ask(self, prompt):
        self.world.events.append("standin_ask")
        if self.world.standin_fails or not self.world.standin_running:
            raise ConnectionRefusedError("stand-in is down")
        return self.world.standin_answer

    def stop(self):
        self.world.events.append("standin_stop")
        if not self.world.standin_survives_stop:
            self.world.standin_running = False

    def confirm_stopped(self):
        ok = not self.world.standin_running
        return StopCheck(ok, {"process": ok}, {})

    def record(self):
        return {"kind": "process", "port": 8001, "pid": 5150, "pgid": 5150}

    def adopt(self, record):
        self.adopted = record


def make_handoff(tmp_path, world=None, *, ledger_cls=Ledger, ledger_kw=None, budgets=None):
    """A coder serving on board 0 under lease L1, and a Handoff ready to park it."""
    from orchard.handoff import Budgets, Handoff
    world = world or World()
    adapter = FakeAdapter(world)
    server = FakeServer(world)
    lease = adapter.acquire(2, WHO, "coder")
    server.start(lease)
    note = tmp_path / "note.json"
    note.write_text(json.dumps(NOTE))
    ledger = ledger_cls(tmp_path / "ledger.jsonl", **(ledger_kw or {}))
    clock = FakeClock()
    h = Handoff(ledger=ledger, stage=2, adapter=adapter, server=server, standin=FakeStandIn(world),
                lease=lease, canary_prompt=CANARY, note_path=note,
                evidence_dir=tmp_path / "evidence", budgets=budgets or Budgets(), clock=clock,
                sleep=clock.sleep)
    return h, world, ledger


def steps(ledger):
    return [(e["event"], e["data"]["step"]) for e in ledger.read() if e["event"] in ("park", "restore")]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_handoff.py`:

```python
"""Park and restore against a fake machine (tests/fakes.py)."""
import pytest

from fakes import World, make_handoff, steps
from orchard.handoff import Blocked, Handoff, progress
from orchard.ledger import file_evidence, replay_state

PARK = [("park", s) for s in ("note", "canary_before", "standin_started", "standin", "stop_sent",
                              "stopped", "reset")]
RESTORE = [("restore", s) for s in ("reset", "serve", "ready", "canary", "resumed")]


def ledger_of(h):
    return h.ledger


def notices(ledger):
    return [e["data"] for e in ledger.read() if e["event"] == "notice"]


def test_park_then_restore_runs_every_step_in_order(tmp_path):
    h, world, ledger = make_handoff(tmp_path)
    lease = h.park()
    assert lease.lease_id == "L1" and world.resets == ["L1"] and not world.coder_running
    assert replay_state(ledger.read())["parked"] is True
    result = h.restore()
    assert result.match and steps(ledger) == PARK + RESTORE
    assert world.resets == ["L1", "L1"] and world.coder_running and not world.standin_running
    assert replay_state(ledger.read())["parked"] is False
    names = [e["data"]["name"] for e in ledger.read() if e["event"] == "measurement"]
    assert names == ["park_reset_seconds", "restore_reset_seconds", "coder_ready_wait_seconds"]


def test_coder_is_not_stopped_until_the_standin_answers(tmp_path):
    h, world, _ = make_handoff(tmp_path)
    h.park()
    assert world.events.index("standin_ask") < world.events.index("coder_stop")


def test_a_standin_that_does_not_answer_leaves_the_coder_up(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="stand-in did not answer"):
        h.park()
    assert world.coder_running and "coder_stop" not in world.events
    assert not world.standin_running
    assert notices(ledger)[-1]["blocked"] is True
    # The park is closed: the coder never left, so the run is not parked.
    assert steps(ledger)[-1] == ("park", "abandoned")
    assert replay_state(ledger.read())["parked"] is False and progress(ledger.read()).phase == "idle"


def test_after_an_abandoned_park_a_new_park_starts_from_its_note(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.standin_fails = False
    h.park()
    assert steps(ledger)[-len(PARK):] == PARK and world.resets == ["L1"]


def test_parked_and_progress_agree_after_every_entry(tmp_path):
    # replay_state is the authority on "parked"; progress() must agree after every entry,
    # including an abandoned park.
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.standin_fails = False
    h.park()
    h.restore()
    entries = ledger.read()
    for n in range(len(entries) + 1):
        prefix = entries[:n]
        assert replay_state(prefix)["parked"] == (progress(prefix).phase != "idle"), n


def test_the_standin_pid_is_recorded_before_its_canary(tmp_path):
    world = World()
    world.standin_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    started = [e["data"] for e in ledger.read()
               if e["event"] == "park" and e["data"]["step"] == "standin_started"]
    assert started[0]["standin"]["pid"] == 5150


def test_a_standin_that_survives_its_stop_is_reported(tmp_path):
    world = World()
    world.standin_survives_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    h.restore()
    resumed = [e["data"] for e in ledger.read()
               if e["event"] == "restore" and e["data"]["step"] == "resumed"]
    assert resumed[0]["standin_stopped"] is False
    assert any(n.get("what", "").startswith("the stand-in is still running") for n in notices(ledger))


def test_an_empty_standin_answer_leaves_the_coder_up(tmp_path):
    world = World()
    world.standin_answer = "  "
    h, world, _ = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="empty answer"):
        h.park()
    assert world.coder_running and steps(ledger_of(h))[-1] == ("park", "abandoned")


@pytest.mark.parametrize("content", [None, "not json", '{"goal": "x"}'])
def test_a_missing_or_incomplete_note_blocks_before_anything_runs(tmp_path, content):
    h, world, ledger = make_handoff(tmp_path)
    if content is None:
        h.note_path.unlink()
    else:
        h.note_path.write_text(content)
    with pytest.raises(Blocked, match="handoff note"):
        h.park()
    assert "standin_start" not in world.events and world.coder_running
    assert steps(ledger) == []


def test_a_leftover_worker_blocks_the_reset(tmp_path):
    world = World()
    world.leave_worker_on_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="not confirmed stopped"):
        h.park()
    assert world.resets == [] and not any(c[0] == "reset" for c in h.adapter.calls)
    assert notices(ledger)[-1]["evidence"]["chip_states"]["0000:01:00.0"] == "HELD-FOREIGN"


def test_a_neighbours_reset_that_ends_inside_the_wait_does_not_block(tmp_path):
    world = World()
    world.foreign_polls = 3          # tt-smi -r of the other board opens every device for ~42 s
    h, world, _ = make_handoff(tmp_path, world)
    h.park()
    assert world.resets == ["L1"]


def test_a_refused_reset_is_retried_once_then_blocks(tmp_path):
    world = World()
    world.refuse_resets = 2
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="refused the reset twice"):
        h.park()
    assert [c for c in h.adapter.calls if c[0] == "reset"] == [("reset", "L1"), ("reset", "L1")]
    assert [e["data"]["what"] for e in ledger.read() if e["event"] == "retry"] == ["reset"]
    assert world.resets == []


def test_one_refusal_then_success_records_two_attempts(tmp_path):
    world = World()
    world.refuse_resets = 1
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    reset = [e for e in ledger.read() if e["event"] == "park" and e["data"]["step"] == "reset"]
    assert reset[0]["data"]["attempts"] == 2


def test_a_reset_left_running_is_never_retried(tmp_path):
    world = World()
    world.reset_hangs = True
    h, world, _ = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="reset failed"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1


def test_a_failed_reset_blocks_at_once(tmp_path):
    # Spec section 10: if the reset fails the stage blocks. Only a busy refusal is retried.
    world = World()
    world.reset_fails = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="reset failed"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1
    assert [e for e in ledger.read() if e["event"] == "retry"] == []


def test_an_adapter_that_cannot_reset_blocks_at_once_and_says_why(tmp_path):
    world = World()
    world.reset_unavailable = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked, match="cannot reset these chips"):
        h.park()
    assert len([c for c in h.adapter.calls if c[0] == "reset"]) == 1
    assert [e for e in ledger.read() if e["event"] == "retry"] == []
    assert h.clock() < h.budgets.quiet_wait_s      # no quiet wait was spent on it


@pytest.mark.parametrize("mesh_reset,parks", [(True, True), (False, False)])
def test_a_mesh_reset_by_tt_model_stop_extends_the_quiet_wait(tmp_path, mesh_reset, parks):
    # tt-model's own reset (its SIGKILL path) runs outside gozer: our chips look busy for longer.
    world = World()
    world.stop_mesh_reset = mesh_reset
    world.foreign_polls = 40                   # 80 s at a 2 s poll: past 60 s, inside 120 s
    h, world, _ = make_handoff(tmp_path, world)
    if parks:
        h.park()
        assert world.resets == ["L1"]
    else:
        with pytest.raises(Blocked, match="not confirmed stopped"):
            h.park()


def test_evidence_files_are_never_overwritten(tmp_path):
    world = World()
    world.coder_answer_after = "5"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked):
        h.restore()
    first = notices(ledger)[-1]["evidence"]["after_file"]
    world.coder_answer_after = "4"             # the operator found the cause
    assert h.restore().match
    # The blocked attempt's file is still the file its notice hashed.
    assert file_evidence(first["path"]) == first
    afters = sorted(p.name for p in h.evidence_dir.glob("canary-after-*.txt"))
    assert len(afters) == 2


def test_a_start_that_times_out_while_the_coder_boots_starts_nothing_else(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.start_hangs_but_comes_up = True
    with pytest.raises(Blocked, match="coming up"):
        h.restore()
    world.start_hangs_but_comes_up = False
    with pytest.raises(Blocked, match="not starting another"):
        h.restore()
    assert world.coder_starts == 2             # the first start and the one that timed out


def test_a_lost_lease_blocks_without_starting_a_server(tmp_path):
    h, world, _ = make_handoff(tmp_path)
    h.park()
    world.lose_lease_on_reset = True
    with pytest.raises(Blocked, match="lease is gone"):
        h.restore()
    assert world.coder_starts == 1


def test_a_coder_that_never_returns_blocks_without_a_retry(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.never_ready = True
    with pytest.raises(Blocked, match="cold-boot budget"):
        h.restore()
    assert world.coder_starts == 2               # the first start, and one start in the restore
    assert ("restore", "ready") not in steps(ledger)


def test_a_changed_canary_blocks(tmp_path):
    world = World()
    world.coder_answer_after = "5"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked, match="canary answer changed") as exc:
        h.restore()
    assert exc.value.evidence["whitespace_only"] is False
    assert ("restore", "canary") not in steps(ledger)


def test_a_whitespace_only_canary_difference_still_blocks_and_says_so(tmp_path):
    world = World()
    world.coder_answer_after = "4\n"
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    with pytest.raises(Blocked) as exc:
        h.restore()
    assert exc.value.evidence["whitespace_only"] is True
    assert notices(ledger)[-1]["evidence"]["after"] == "4\n"


def test_park_resumes_after_the_last_recorded_step(tmp_path):
    world = World()
    world.leave_worker_on_stop = True
    h, world, ledger = make_handoff(tmp_path, world)
    with pytest.raises(Blocked):
        h.park()
    world.worker_left = False                   # the operator stopped the worker
    again = Handoff(ledger=ledger, stage=2, adapter=h.adapter, server=h.server, standin=h.standin,
                    lease=h.lease, canary_prompt=h.canary_prompt, note_path=h.note_path,
                    evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    again.park()
    assert world.events.count("standin_start") == 1 and world.events.count("coder_stop") == 1
    assert steps(ledger) == PARK


def test_restore_after_a_block_does_not_repeat_a_finished_step(tmp_path):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    world.never_ready = True
    with pytest.raises(Blocked):
        h.restore()
    world.never_ready = False                   # the operator looked; the coder came up late
    assert h.restore().match
    assert world.coder_starts == 2
    assert steps(ledger) == PARK + RESTORE


def test_restore_with_nothing_parked_is_an_error(tmp_path):
    h, _, _ = make_handoff(tmp_path)
    with pytest.raises(ValueError, match="nothing to restore"):
        h.restore()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff.py`
Expected: FAIL with `ImportError: cannot import name 'Budgets' from 'orchard.handoff'` (raised inside `make_handoff`) and `cannot import name 'Handoff'`.

- [ ] **Step 4: Append `Budgets`, `wait_stopped` and `Handoff` to `orchard/handoff.py`**

Add these imports at the top of the file, after the existing `from orchard.adapters import ChipState, Lease, boards_of` line. They also cover what Tasks 10 and 11 append:

```python
import dataclasses
import json
import time
from pathlib import Path

from orchard.adapters import (AdapterError, LeaseLost, Queued, Refused, ResetFailed,
                              TicketGone)
from orchard.canary import CanaryError, CanaryResult, compare
from orchard.defaults import (COLD_BOOT_BUDGET_S, IDLE_RELEASE_S, MESH_RESET_EXTRA_S, POLL_S,
                              QUEUE_POLL_S, QUIET_WAIT_S)
from orchard.ledger import file_evidence
from orchard.server import NotReady, ServerError, ServerStarting
```

Then append:

```python
@dataclass(frozen=True)
class Budgets:
    quiet_wait_s: float = QUIET_WAIT_S              # how long to wait for chips to go quiet
    poll_s: float = POLL_S
    cold_boot_s: float = COLD_BOOT_BUDGET_S         # the run's cold-boot budget (spec section 6)
    mesh_reset_extra_s: float = MESH_RESET_EXTRA_S  # added when tt-model reset the mesh itself


def wait_stopped(server, adapter, lease: Lease, *, quiet_wait_s: float, poll_s: float, clock,
                 sleep, accept=("CLAIMED",)) -> dict:
    """Poll until the server's own checks and the lease tool both show nothing running.

    Both must agree. docker can show a container gone while a vLLM worker it started still holds
    the device; the lease tool sees that holder. The lease tool can show a chip CLAIMED while
    another user's container still holds it; docker sees that one. During another board's reset
    our chips can look busy for about 42 s, so the wait lasts longer than that.
    """
    deadline = clock() + quiet_wait_s
    while True:
        check = server.confirm_stopped()
        try:
            quiet, states = chips_quiet(adapter.status(), lease, accept)
        except AdapterError as exc:
            quiet, states = False, {"error": str(exc)}
        if check.stopped and quiet:
            others = (check.evidence.get("docker_inspect") or {}).get("others_with_all_devices") or []
            return {"ok": True, "checks": check.checks, "chip_states": states,
                    "others_with_all_devices": others}
        if clock() >= deadline:
            return {"ok": False, "checks": check.checks, "chip_states": states,
                    "server_evidence": check.evidence}
        sleep(poll_s)


class Handoff:
    """One park and restore of the coder, step by step, each step recorded in the ledger."""

    def __init__(self, *, ledger, stage, adapter, server, standin, lease: Lease, canary_prompt: str,
                 note_path, evidence_dir, budgets: Budgets = Budgets(), clock=time.monotonic,
                 sleep=time.sleep):
        self.ledger, self.stage = ledger, stage
        self.adapter, self.server, self.standin = adapter, server, standin
        self.lease, self.canary_prompt = lease, canary_prompt
        self.note_path = Path(note_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.budgets, self.clock, self.sleep = budgets, clock, sleep

    # ---- ledger and evidence helpers ----------------------------------------------------------

    def _record(self, event: str, step: str, **data) -> None:
        self.ledger.append(event, self.stage, step=step, lease=self.lease.record(),
                           owner_pid=self.adapter.owner_pid, **data)

    def _block(self, reason: str, **evidence):
        self.ledger.append("notice", self.stage, blocked=True, reason=reason, evidence=evidence)
        raise Blocked(reason, **evidence)

    def _block_before_stop(self, reason: str, **evidence):
        """Block a park that never told the coder to stop: close it, so the run is not parked."""
        self.ledger.append("notice", self.stage, blocked=True, reason=reason, evidence=evidence)
        stopped = None
        if "standin_started" in progress(self.ledger.read()).park_done:
            stopped = self._stop_standin()
        self._record("park", ABANDONED, reason=reason, standin_stopped=stopped)
        raise Blocked(reason, **evidence)

    def _evidence_path(self, stem: str) -> Path:
        """A new file for every attempt, named by the next ledger sequence number.

        A ledger entry records each file's sha256, so a file is never written twice.
        """
        seq = len(self.ledger.read()) + 1
        n = 0
        while True:
            path = self.evidence_dir / (f"{stem}-{seq:05d}.txt" if n == 0 else
                                        f"{stem}-{seq:05d}-{n}.txt")
            if not path.exists():
                return path
            n += 1

    def _write_evidence(self, stem: str, text: str) -> Path:
        path = self._evidence_path(stem)
        with open(path, "x", encoding="utf-8") as f:       # "x": never overwrite
            f.write(text)
        return path

    def _stop_standin(self) -> bool:
        """Stop the stand-in and check it is gone. It is a process in its own session, so it
        outlives a supervisor crash; a stand-in left running holds host memory and its port."""
        try:
            self.standin.stop()
            check = self.standin.confirm_stopped()
        except Exception as exc:          # cleanup: record it and carry on
            self.ledger.append("notice", self.stage, what="stopping the stand-in failed",
                               error=str(exc))
            return False
        if not check.stopped:
            self.ledger.append("notice", self.stage, what="the stand-in is still running after "
                               "its stop", checks=check.checks, standin=self.standin.record())
        return check.stopped

    # ---- shared checks ------------------------------------------------------------------------

    def _wait_stopped(self, accept=("CLAIMED",), extra_s: float = 0.0) -> dict:
        check = wait_stopped(self.server, self.adapter, self.lease,
                             quiet_wait_s=self.budgets.quiet_wait_s + extra_s,
                             poll_s=self.budgets.poll_s, clock=self.clock, sleep=self.sleep,
                             accept=accept)
        if check["ok"] and check["others_with_all_devices"]:
            # Recorded for the operator; another agent's container does not block our stop.
            self.ledger.append("notice", self.stage,
                               what="other containers map the whole /dev/tenstorrent directory",
                               containers=check["others_with_all_devices"])
        return check

    def _reset(self, event: str) -> None:
        for attempt in (1, 2):
            t0 = self.clock()
            try:
                self.adapter.reset(self.lease)
            except LeaseLost as exc:
                self._block(f"the lease is gone or another tenant holds its chips; no server is "
                            f"started: {exc}")
            except Refused as exc:
                if exc.permanent:
                    self._block(f"the lease adapter cannot reset these chips: {exc}")
                # gozer's refusal ran nothing (spec section 8, item 1). The usual cause is a device
                # still open, so look again before the one retry the retry rule allows.
                if attempt == 2:
                    self._block(f"gozer refused the reset twice: {exc}")
                check = self._wait_stopped()
                if not check["ok"]:
                    self._block(f"the reset was refused and the chips are still in use: {exc}", **check)
                self.ledger.append("retry", self.stage, what="reset", attempt=2, reason=str(exc))
                continue
            except ResetFailed as exc:
                # The reset ran and failed, or is still running: the stage blocks (spec section 10).
                self._block(f"the reset failed: {exc}", left_running=exc.left_running)
            seconds = round(self.clock() - t0, 3)      # the successful call only
            self._record(event, "reset", seconds=seconds, attempts=attempt)
            self.ledger.append("measurement", self.stage, name=f"{event}_reset_seconds",
                               value=seconds, unit="s", label="measured")
            return

    # ---- park (spec section 6, steps 1 to 3) --------------------------------------------------

    def park(self) -> Lease:
        p = progress(self.ledger.read())
        if p.phase == "restoring":
            raise ValueError("a restore is in progress; call restore() or recover first")
        done = set(p.park_done) if p.phase in ("parking", "parked") else set()
        if "note" not in done:
            self._park_note()
        if "canary_before" not in done:
            self._park_canary()
        # Stand-in first (spec section 6, step 2): the coder stops only after the stand-in answered.
        if "standin" not in done:
            self._park_standin(started="standin_started" in done)
        mesh_reset = p.mesh_reset
        if "stop_sent" not in done:
            result = self.server.stop()
            mesh_reset = bool(result.get("mesh_reset"))
            self._record("park", "stop_sent", result=result)
        if "stopped" not in done:
            extra = self.budgets.mesh_reset_extra_s if mesh_reset else 0.0
            check = self._wait_stopped(extra_s=extra)
            if not check["ok"]:
                self._block("the coder is not confirmed stopped; no reset was run", **check)
            self._record("park", "stopped", **check)
        if "reset" not in done:
            self._reset("park")
        return self.lease

    def _park_note(self) -> None:
        path = self.note_path
        try:
            note = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            self._block(f"the handoff note {path} is missing or is not JSON: {exc}")
        if not isinstance(note, dict):
            self._block(f"the handoff note {path} is not a JSON object")
        missing = [k for k in NOTE_KEYS if note.get(k) in (None, "")]
        if missing:
            self._block(f"the handoff note {path} lacks {', '.join(missing)}", note=str(path))
        self._record("park", "note", note=file_evidence(path), server=self.server.record())

    def _park_canary(self) -> None:
        try:
            answer = self.server.ask(self.canary_prompt)
        except (CanaryError, OSError) as exc:
            self._block_before_stop(f"the coder did not answer the canary before the park: {exc}")
        if not answer.strip():
            self._block_before_stop("the coder gave an empty canary answer before the park")
        path = self._write_evidence("canary-before", answer)
        self._record("park", "canary_before", canary=file_evidence(path))

    def _park_standin(self, started: bool) -> None:
        if not started:
            try:
                self.standin.start()
            except (ServerError, OSError) as exc:
                self._record("park", "standin_started", standin=self.standin.record(), ok=False)
                self._block_before_stop(f"the stand-in did not start; the coder stays up: {exc}")
            # Recorded before the canary, so a crash from here on leaves the pid in the ledger.
            self._record("park", "standin_started", standin=self.standin.record())
        try:
            answer = self.standin.ask(self.canary_prompt)
        except (CanaryError, ServerError, OSError) as exc:
            self._block_before_stop(f"the stand-in did not answer; the coder stays up: {exc}")
        if not answer.strip():
            self._block_before_stop("the stand-in gave an empty answer; the coder stays up")
        path = self._write_evidence("standin-canary", answer)
        self._record("park", "standin", canary=file_evidence(path))

    # ---- restore (spec section 6, steps 4 to 6; the stage test itself is plan 4's) ------------

    def restore(self) -> CanaryResult | None:
        p = progress(self.ledger.read())
        if p.phase not in ("parked", "restoring"):
            raise ValueError(f"nothing to restore (phase {p.phase})")
        done = set(p.restore_done)
        if "reset" not in done:
            check = self._wait_stopped()
            if not check["ok"]:
                self._block("the chips are still in use after the stage; no reset was run", **check)
            self._reset("restore")
        if "serve" not in done:
            # Never start a second server while one may be up or coming up on the same port.
            if not self.server.confirm_stopped().stopped:
                self._block("a server is already up or coming up for the coder; not starting another")
            try:
                self.server.start(self.lease)
            except ServerStarting as exc:
                self._block(f"the coder start timed out while it was coming up; nothing else is "
                            f"started: {exc}", may_be_running=True)
            except ServerError as exc:
                self._block(f"starting the coder failed; nothing is retried: {exc}")
            self._record("restore", "serve", server=self.server.record())
        if "ready" not in done:
            try:
                seconds = self.server.wait_ready(self.budgets.cold_boot_s)
            except NotReady as exc:
                self._block("the coder did not return within the cold-boot budget; the run is "
                            "paused and nothing is retried", waited_s=exc.waited_s,
                            budget_s=self.budgets.cold_boot_s)
            except ServerError as exc:
                self._block(f"the coder exited before it was ready; nothing is retried: {exc}")
            self._record("restore", "ready", seconds=seconds)
            # The wait for readiness only: `tt-model serve` itself is not inside this number.
            self.ledger.append("measurement", self.stage, name="coder_ready_wait_seconds",
                               value=seconds, unit="s", label="measured")
        result = None
        if "canary" not in done:
            result = self._restore_canary(p.canary_before)
        if "resumed" not in done:
            stopped = self._stop_standin()
            # Plan 4 hands the note and a summary of the stage test to the coder.
            self._record("restore", "resumed", note=str(self.note_path), standin_stopped=stopped)
        return result

    def _restore_canary(self, before_ev: dict | None) -> CanaryResult | None:
        try:
            after = self.server.ask(self.canary_prompt)
        except (CanaryError, OSError) as exc:
            self._block(f"the coder did not answer the canary after the restore: {exc}")
        after_path = self._write_evidence("canary-after", after)
        if before_ev is None:
            # Only possible after a restart that interrupted the park before its canary.
            self.ledger.append("notice", self.stage,
                               what="canary not compared: no answer was recorded before the park")
            self._record("restore", "canary", compared=False, canary=file_evidence(after_path))
            return None
        try:
            before = Path(before_ev["path"]).read_text()
        except OSError as exc:
            self._block(f"the pre-park canary answer cannot be read: {exc}", before_file=before_ev)
        result = compare(before, after)
        if not result.match:
            self._block("the canary answer changed after the restore",
                        whitespace_only=result.whitespace_only, before=before[:500],
                        after=after[:500], before_file=before_ev,
                        after_file=file_evidence(after_path))
        self._record("restore", "canary", compared=True, match=True, canary=file_evidence(after_path))
        return result
```

Note for the implementer: `self._block` always raises, so code after it in the same branch never runs. Python cannot see that, so a linter may warn that `note` or `answer` "may be unbound". Leave the code as it is.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff.py`
Expected: PASS (29 tests).

- [ ] **Step 6: Mutation checks**

1. Stand-in first: in `park`, move the `if "standin" not in done: self._park_standin(...)` lines below the `stop_sent` lines. Run `tests/test_handoff.py::test_coder_is_not_stopped_until_the_standin_answers` and `::test_a_standin_that_does_not_answer_leaves_the_coder_up`. Expected: both FAIL. Restore: PASS.
2. Two-sided stop check: in `_wait_stopped`, change `if check.stopped and quiet:` to `if check.stopped:`. Run `tests/test_handoff.py::test_a_leftover_worker_blocks_the_reset`. Expected: FAIL (the reset was attempted). Restore: PASS.
3. Retry once: in `_reset`, change `if attempt == 2:` (the Refused branch) to `if attempt == 3:` and the loop to `for attempt in (1, 2, 3):`. Run `tests/test_handoff.py::test_a_refused_reset_is_retried_once_then_blocks`. Expected: FAIL. Restore: PASS.
4. Failed reset: in the `ResetFailed` branch of `_reset`, replace the `self._block(...)` line with `if attempt == 2: self._block(...)` followed by `continue`. Run `tests/test_handoff.py::test_a_failed_reset_blocks_at_once`. Expected: FAIL. Restore: PASS.
5. Permanent refusal: change `if exc.permanent:` to `if False:`. Run `tests/test_handoff.py::test_an_adapter_that_cannot_reset_blocks_at_once_and_says_why`. Expected: FAIL. Restore: PASS.
6. Mesh reset: in `park`, change the `extra = ...` line to `extra = 0.0`. Run `tests/test_handoff.py::test_a_mesh_reset_by_tt_model_stop_extends_the_quiet_wait`. Expected: FAIL for `mesh_reset=True`. Restore: PASS.
7. Never overwrite: in `_evidence_path`, replace `if not path.exists(): return path` with `return self.evidence_dir / f"{stem}.txt"`. Run `tests/test_handoff.py::test_evidence_files_are_never_overwritten`. Expected: FAIL. Restore: PASS.
8. One server at a time: in `restore`, change `if not self.server.confirm_stopped().stopped:` to `if False:`. Run `tests/test_handoff.py::test_a_start_that_times_out_while_the_coder_boots_starts_nothing_else`. Expected: FAIL (a second start). Restore: PASS.
9. Agreement: in `orchard/ledger.py` `replay_state`, change the park line to `state["parked"] = True`. Run `tests/test_handoff.py::test_parked_and_progress_agree_after_every_entry`. Expected: FAIL. Restore: PASS.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/handoff.py tests/fakes.py tests/test_handoff.py
git commit -m "Add park and restore: stand-in first, two-sided stop check, one reset retry, exact canary"
```

---

### Task 10: Release for an idle phase, and waiting in the queue

**Files:**
- Modify: `orchard/handoff.py` (append `release_for_idle_phase`, `reacquire`)
- Test: `tests/test_handoff_queue.py`

**Interfaces:**
- Consumes: `Queued`, `Refused`, `TicketGone`, `AdapterError` (Task 2); `Blocked` (Task 8); `wait_stopped` (Task 9); `IDLE_RELEASE_S`, `QUEUE_POLL_S`, `QUIET_WAIT_S`, `POLL_S`.
- Produces in `orchard.handoff`:
  - `release_for_idle_phase(adapter, server, lease, *, ledger, stage, expected_idle_s: float, budget_s: float = IDLE_RELEASE_S, quiet_wait_s: float = QUIET_WAIT_S, poll_s: float = POLL_S, clock=time.monotonic, sleep=time.sleep) -> bool` (True if released). Before a release it runs `wait_stopped`; if the server is not confirmed stopped it writes a blocked notice and raises `Blocked` without releasing.
  - `reacquire(adapter, *, chips: int, who: str, reason: str, ledger, stage, wait_budget_s: float, clock=time.monotonic, sleep=time.sleep, poll_s: float = QUEUE_POLL_S, exact: str | None = None) -> Lease`. Raises `Blocked`. Ledger: `decision` "queued for lease" (ticket, position), `notice` for an expired ticket, `decision` "lease granted" with `waited_s`, `measurement` `queue_wait_seconds`.

- [ ] **Step 1: Write the failing tests**

```python
"""release_for_idle_phase and reacquire, against the fake machine."""
import pytest

from fakes import WHO, FakeAdapter, FakeClock, FakeServer, World
from orchard.adapters import Queued, Refused, TicketGone
from orchard.defaults import GOZER_CLAIM_WINDOW_S
from orchard.handoff import Blocked, reacquire, release_for_idle_phase
from orchard.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        yield led


def decisions(ledger):
    return [e["data"]["decision"] for e in ledger.read() if e["event"] == "decision"]


def run(ledger, world, budget=600.0):
    clock = FakeClock()
    adapter = FakeAdapter(world)
    lease = reacquire(adapter, chips=2, who=WHO, reason="stage 3", ledger=ledger, stage=3,
                      wait_budget_s=budget, clock=clock, sleep=clock.sleep)
    return lease, adapter, clock


def test_a_short_idle_phase_keeps_the_lease(ledger):
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    assert release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                                  expected_idle_s=600) is False
    assert ("release", "L1") not in adapter.calls
    assert decisions(ledger) == ["hold the lease through a phase with no hardware use"]


def test_a_long_idle_phase_releases_the_lease(ledger):
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    # An image build takes 1.5 to 2.5 h (spec section 3).
    assert release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                                  expected_idle_s=5400) is True
    assert ("release", "L1") in adapter.calls and world.leases == {}


def test_a_long_idle_phase_with_a_server_still_up_keeps_the_lease_and_blocks(ledger):
    # gozer's release resets the chips, and gozer cannot see another user's container.
    world = World()
    adapter = FakeAdapter(world)
    lease = adapter.acquire(2, WHO, "r")
    world.coder_running = True
    clock = FakeClock()
    with pytest.raises(Blocked, match="not released"):
        release_for_idle_phase(adapter, FakeServer(world), lease, ledger=ledger, stage=7,
                               expected_idle_s=5400, clock=clock, sleep=clock.sleep)
    assert ("release", "L1") not in adapter.calls and "L1" in world.leases


def test_a_free_box_grants_at_once(ledger):
    lease, adapter, _ = run(ledger, World())
    assert lease.lease_id == "L1" and adapter.calls == [("acquire", 2, True)]
    granted = [e["data"] for e in ledger.read() if e["event"] == "decision"][-1]
    assert granted["waited_s"] == 0


def test_a_busy_box_waits_on_the_ticket_and_records_the_wait(ledger):
    world = World()
    world.queue_script = [Queued("t1", 2), Queued("t1", 1), Queued("t1", 1)]
    lease, adapter, clock = run(ledger, world)
    assert lease.lease_id == "L1"
    assert adapter.calls == [("acquire", 2, True), ("claim", "t1"), ("claim", "t1"), ("claim", "t1")]
    assert decisions(ledger) == ["queued for lease", "lease granted"]
    wait = [e["data"] for e in ledger.read() if e["event"] == "measurement"][-1]
    assert wait["name"] == "queue_wait_seconds" and wait["value"] == 30.0
    assert all(s * 3 <= GOZER_CLAIM_WINDOW_S for s in clock.sleeps)


def test_an_expired_ticket_is_replaced_once_and_the_lost_place_recorded(ledger):
    world = World()
    world.queue_script = [Queued("t1"), TicketGone("t1"), Queued("t2")]
    lease, adapter, _ = run(ledger, world)
    assert lease.lease_id == "L1"
    lost = [e["data"] for e in ledger.read() if e["event"] == "notice"]
    assert lost[0]["what"].startswith("queue ticket expired") and lost[0]["ticket"] == "t1"
    queued = [e["data"]["ticket"] for e in ledger.read()
              if e["event"] == "decision" and e["data"]["decision"] == "queued for lease"]
    assert queued == ["t1", "t2"]
    assert ("claim", "t2") in adapter.calls


def test_a_second_expiry_blocks(ledger):
    world = World()
    world.queue_script = [Queued("t1"), TicketGone("t1"), Queued("t2"), TicketGone("t2")]
    with pytest.raises(Blocked, match="second queue ticket expired"):
        run(ledger, world)


def test_waiting_past_the_budget_cancels_the_ticket_and_blocks(ledger):
    world = World()
    world.queue_script = [Queued("t1")] * 100
    with pytest.raises(Blocked, match="past the budget"):
        run(ledger, world, budget=25)
    adapter_calls = [c for c in ledger.read() if c["event"] == "notice"]
    assert adapter_calls[-1]["data"]["blocked"] is True


def test_the_cancel_goes_to_the_lease_tool(ledger):
    world = World()
    world.queue_script = [Queued("t1")] * 100
    adapter = FakeAdapter(world)
    clock = FakeClock()
    with pytest.raises(Blocked):
        reacquire(adapter, chips=2, who=WHO, reason="r", ledger=ledger, stage=3, wait_budget_s=25,
                  clock=clock, sleep=clock.sleep)
    assert adapter.calls[-1] == ("cancel", "t1")
    assert len([c for c in adapter.calls if c[0] == "claim"]) == 3


def test_a_refusal_blocks(ledger):
    world = World()
    world.queue_script = [Refused("the single-tenant adapter has no queue")]
    with pytest.raises(Blocked, match="refused"):
        run(ledger, world)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_queue.py`
Expected: FAIL with `ImportError: cannot import name 'reacquire' from 'orchard.handoff'`.

- [ ] **Step 3: Append to `orchard/handoff.py`**

The imports Task 9 added already cover this code. Append:

```python
def release_for_idle_phase(adapter, server, lease: Lease, *, ledger, stage,
                           expected_idle_s: float, budget_s: float = IDLE_RELEASE_S,
                           quiet_wait_s: float = QUIET_WAIT_S, poll_s: float = POLL_S,
                           clock=time.monotonic, sleep=time.sleep) -> bool:
    """Spec section 6, first branch row: release the lease for a long phase with no hardware use.

    A held board is unavailable to everyone else. Holding it costs nothing for a short phase;
    for a long one (an image build of 1.5 to 2.5 h, a CPU-only reference run) the supervisor
    releases it and takes a new lease, through `reacquire`, before the next hardware phase.
    gozer's release resets the chips (about 42 s), and gozer cannot see a container owned by
    another user, so the same two-sided stop check as a park runs first (`wait_stopped`). A
    ResetFailed from the release propagates to the caller.
    """
    info = {"lease_id": lease.lease_id, "expected_idle_s": expected_idle_s, "budget_s": budget_s}
    if expected_idle_s <= budget_s:
        ledger.append("decision", stage,
                      decision="hold the lease through a phase with no hardware use", **info)
        return False
    check = wait_stopped(server, adapter, lease, quiet_wait_s=quiet_wait_s, poll_s=poll_s,
                         clock=clock, sleep=sleep)
    if not check["ok"]:
        reason = "the server is not confirmed stopped; the lease was not released"
        ledger.append("notice", stage, blocked=True, reason=reason, evidence=check)
        raise Blocked(reason, **check)
    adapter.release(lease)
    ledger.append("decision", stage,
                  decision="released the lease for a phase with no hardware use", **info)
    return True


def reacquire(adapter, *, chips: int, who: str, reason: str, ledger, stage, wait_budget_s: float,
              clock=time.monotonic, sleep=time.sleep, poll_s: float = QUEUE_POLL_S,
              exact: str | None = None) -> Lease:
    """Take a lease, waiting in the lease tool's queue if the box is busy (spec section 10).

    The ticket is claimed by repeating the acquire with the ticket (the gozer-park skill); `gozer
    wait` is never used, because it grants a lease with no owner pid. The poll interval stays well
    inside gozer's 90 s claim window. gozer expires a ticket after one hour; the first expiry takes
    a new ticket and records the lost place, and a second one blocks. Waiting past the budget
    cancels the ticket and blocks. The ledger records the wait.
    """
    t0 = clock()

    def waited() -> float:
        return round(clock() - t0, 3)

    def block(why: str, **ev):
        ledger.append("notice", stage, blocked=True, reason=why, evidence=ev)
        raise Blocked(why, **ev)

    def granted(lease: Lease) -> Lease:
        ledger.append("decision", stage, decision="lease granted", lease_id=lease.lease_id,
                      chips=list(lease.chips), waited_s=waited())
        ledger.append("measurement", stage, name="queue_wait_seconds", value=waited(), unit="s",
                      label="measured")
        return lease

    def enqueue() -> tuple[Lease | None, str | None]:
        try:
            return granted(adapter.acquire(chips, who, reason, queue=True, exact=exact)), None
        except Queued as q:
            ledger.append("decision", stage, decision="queued for lease", ticket=q.ticket,
                          position=q.position, chips=chips)
            return None, q.ticket
        except Refused as exc:
            block(f"the lease tool refused the request: {exc}")

    lease, ticket = enqueue()
    if lease is not None:
        return lease
    replaced = False
    while True:
        if clock() - t0 >= wait_budget_s:
            try:
                adapter.cancel(ticket)
            except AdapterError as exc:
                ledger.append("notice", stage, what="cancelling the queue ticket failed",
                              ticket=ticket, error=str(exc))
            block("waited past the budget for a lease; the ticket was cancelled", ticket=ticket,
                  waited_s=waited(), budget_s=wait_budget_s)
        sleep(poll_s)
        try:
            return granted(adapter.claim(ticket, chips, who, reason))
        except Queued:
            continue
        except TicketGone:
            ledger.append("notice", stage, what="queue ticket expired; the place in the queue is lost",
                          ticket=ticket, waited_s=waited())
            if replaced:
                block("a second queue ticket expired", ticket=ticket)
            replaced = True
            lease, ticket = enqueue()
            if lease is not None:
                return lease
        except Refused as exc:
            block(f"the lease tool refused the claim: {exc}", ticket=ticket)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_queue.py`
Expected: PASS (10 tests).

- [ ] **Step 5: Mutation checks**

1. Delete the `if replaced: block(...)` lines. Run `tests/test_handoff_queue.py::test_a_second_expiry_blocks`. Expected: FAIL with `DID NOT RAISE` (a third ticket is taken and granted). Restore: PASS.
2. Delete the `adapter.cancel(ticket)` call (keep the try with `pass`). Run `tests/test_handoff_queue.py::test_the_cancel_goes_to_the_lease_tool`. Expected: FAIL. Restore: PASS.
3. Stop check before release: in `release_for_idle_phase`, change `if not check["ok"]:` to `if False:`. Run `tests/test_handoff_queue.py::test_a_long_idle_phase_with_a_server_still_up_keeps_the_lease_and_blocks`. Expected: FAIL (the lease was released under a live server). Restore: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/handoff.py tests/test_handoff_queue.py
git commit -m "Release the lease for long idle phases and wait in the queue with a bounded ticket loop"
```

---

### Task 11: Recovery after a supervisor restart

**Files:**
- Modify: `orchard/handoff.py` (append `Recovery`, `recover`, `Handoff.recover_and_restore`)
- Test: `tests/test_handoff_recovery.py`

**Interfaces:**
- Consumes: `progress`, `Handoff`, `reacquire` (Tasks 8 to 10); the fakes from Task 9.
- Produces in `orchard.handoff`:
  - `Recovery` (frozen dataclass: `phase: str`, `park_done: tuple[str, ...]`, `restore_done: tuple[str, ...]`, `lease_id: str | None`, `lease_state: str` in `{"none", "ours", "orphaned", "gone", "taken"}`, `coder_running: bool`, `standin_running: bool`, `action: str` in `{"none", "abandon", "restore"}`, `disagreements: tuple[str, ...]`). The action is "abandon" when the coder is running and the park never sent its stop: the park is closed with an `abandoned` entry, the stand-in is stopped and checked, and the coder is left serving (re-leasing it is plan 4's job).
  - `recover(entries: list[dict], *, adapter, server, standin=None) -> Recovery`.
  - `Handoff.recover_and_restore(recovery: Recovery, *, chips: int, who: str, reason: str, wait_budget_s: float) -> CanaryResult | None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Recovery: kill the supervisor after every ledger event and check the final state (spec section 13)."""
from fakes import (CANARY, WHO, Crash, FakeAdapter, FakeServer, FakeStandIn, World, make_handoff,
                   steps)
from orchard.adapters import Lease
from orchard.handoff import Handoff, progress, recover
from orchard.ledger import Ledger, replay_state

NEW_PID = 200


class CrashingLedger(Ledger):
    """A ledger whose process dies right after its k-th append is on disk."""

    def __init__(self, path, crash_after=0):
        super().__init__(path)
        self.crash_after, self.appended = crash_after, 0

    def append(self, *args, **kwargs):
        entry = super().append(*args, **kwargs)
        self.appended += 1
        if self.appended == self.crash_after:
            raise Crash(f"killed after ledger event {self.appended}: {entry['event']} "
                        f"{entry['data'].get('step')}")
        return entry


def restart(tmp_path, h, world):
    """The supervisor died. Start a new one with a new pid and let it recover.

    The stand-in and a bundle coder run in their own sessions (orchard/server.py), so they
    outlive the supervisor. The fake keeps them running across the restart.
    """
    h.ledger.close()
    world.owner_pid = NEW_PID
    ledger = Ledger(tmp_path / "ledger.jsonl")
    adapter, server, standin = FakeAdapter(world, NEW_PID), FakeServer(world), FakeStandIn(world)
    rec = recover(ledger.read(), adapter=adapter, server=server, standin=standin)
    p = progress(ledger.read())
    h2 = Handoff(ledger=ledger, stage=2, adapter=adapter, server=server, standin=standin,
                 lease=Lease.from_record(p.lease) if p.lease else h.lease, canary_prompt=CANARY,
                 note_path=h.note_path, evidence_dir=h.evidence_dir, clock=h.clock, sleep=h.sleep)
    h2.recover_and_restore(rec, chips=2, who=WHO, reason="coder", wait_budget_s=600)
    return rec, ledger


def final_state(world, ledger):
    entries = ledger.read()
    return {"coder_running": world.coder_running, "standin_running": world.standin_running,
            "leases": len(world.leases),
            "lease_owner_is_live": all(o == world.owner_pid for _, o in world.leases.values()),
            "parked": replay_state(entries)["parked"], "phase": progress(entries).phase}


EXPECTED = {"coder_running": True, "standin_running": False, "leases": 1,
            "lease_owner_is_live": True, "parked": False, "phase": "idle"}
# A crash before `stop_sent`: the park is abandoned and the coder, which never left, keeps
# serving under the dead supervisor's lease. Re-leasing it is plan 4's job.
EXPECTED_ABANDONED = dict(EXPECTED, lease_owner_is_live=False)

# A clean park and restore appends 15 entries: park note, canary_before, standin_started,
# standin, stop_sent, stopped, reset, measurement; restore reset, measurement, serve, ready,
# measurement, canary, resumed.
CLEAN_APPENDS = 15
STOP_SENT = 5        # the append number of `park stop_sent`


def stand_in_answered_before_any_stop(world):
    if "coder_stop" not in world.events:
        return True
    return ("standin_ask" in world.events
            and world.events.index("standin_ask") < world.events.index("coder_stop"))


def test_a_clean_run_appends_the_expected_entries(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger)
    h.park()
    h.restore()
    assert ledger.appended == CLEAN_APPENDS
    entries = ledger.read()
    assert entries[STOP_SENT - 1]["data"]["step"] == "stop_sent"
    assert final_state(world, ledger) == EXPECTED


def test_a_crash_after_any_ledger_event_reaches_the_expected_final_state(tmp_path):
    # The last event is `resumed`: after it the handoff is over, and re-leasing a coder that is
    # serving belongs to plan 4 (see test_a_crash_after_resume_leaves_nothing_to_recover).
    failures = []
    for k in range(1, CLEAN_APPENDS):
        d = tmp_path / f"k{k}"
        d.mkdir()
        h, world, ledger = make_handoff(d, ledger_cls=CrashingLedger, ledger_kw={"crash_after": k})
        try:
            h.park()
            h.restore()
            failures.append((k, "no crash happened"))
            continue
        except Crash:
            pass
        rec, ledger2 = restart(d, h, world)
        state = final_state(world, ledger2)
        want = EXPECTED_ABANDONED if k < STOP_SENT else EXPECTED
        if state != want:
            failures.append((k, rec, state))
        if k < STOP_SENT and "coder_stop" in world.events:
            failures.append((k, "a coder whose park never sent its stop was stopped"))
        if not stand_in_answered_before_any_stop(world):
            failures.append((k, "the coder stopped before the stand-in answered", world.events))
        ledger2.close()
    assert failures == []


def test_an_orphaned_standin_is_found_from_the_ledger_and_stopped(tmp_path):
    # Killed right after the stand-in started, before its canary: its pid is in the ledger.
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger, ledger_kw={"crash_after": 3})
    try:
        h.park()
    except Crash:
        pass
    assert world.standin_running                 # it outlived the supervisor
    rec, ledger2 = restart(tmp_path, h, world)
    assert rec.action == "abandon" and rec.standin_running
    assert not world.standin_running
    closing = [e["data"] for e in ledger2.read() if e["event"] == "park"][-1]
    assert closing["step"] == "abandoned" and closing["standin_stopped"] is True


def test_a_crash_after_resume_leaves_nothing_to_recover(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger,
                                    ledger_kw={"crash_after": CLEAN_APPENDS})
    h.park()
    try:
        h.restore()
    except Crash:
        pass
    ledger.close()
    with Ledger(tmp_path / "ledger.jsonl") as led:
        rec = recover(led.read(), adapter=FakeAdapter(world, NEW_PID), server=FakeServer(world))
    assert rec.action == "none" and rec.phase == "idle"


def test_killed_between_tt_model_stop_and_the_reset(tmp_path):
    world = World()
    world.crash_in_stop = True          # the coder stopped, and the ledger never heard of it
    h, world, ledger = make_handoff(tmp_path, world)
    try:
        h.park()
    except Crash:
        pass
    assert ("park", "stop_sent") not in steps(ledger)
    rec, ledger2 = restart(tmp_path, h, world)
    assert rec.phase == "parking" and not rec.coder_running and rec.lease_state == "orphaned"
    assert final_state(world, ledger2) == EXPECTED
    assert world.resets == ["L2"]       # the new lease is reset before the coder starts on it
    decisions = [e["data"]["decision"] for e in ledger2.read() if e["event"] == "decision"]
    assert decisions[0] == "recover after restart; the machine wins"
    assert "new lease after restart" in decisions


def test_the_machine_wins_when_the_ledger_says_stopped_but_docker_shows_the_coder(tmp_path):
    h, world, ledger = make_handoff(tmp_path, ledger_cls=CrashingLedger, ledger_kw={"crash_after": 6})
    try:
        h.park()                        # dies right after the `stopped` entry
    except Crash:
        pass
    world.coder_running = True          # someone started the coder again meanwhile
    rec, ledger2 = restart(tmp_path, h, world)
    assert "the ledger says the coder was stopped; the machine shows it running" in rec.disagreements
    assert final_state(world, ledger2) == EXPECTED


def test_recover_with_nothing_in_progress_does_nothing(tmp_path):
    world = World()
    with Ledger(tmp_path / "ledger.jsonl") as led:
        rec = recover(led.read(), adapter=FakeAdapter(world), server=FakeServer(world))
    assert rec.action == "none" and rec.lease_state == "none"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_recovery.py`
Expected: `test_a_clean_run_appends_the_expected_entries` passes; the others FAIL with `ImportError: cannot import name 'recover' from 'orchard.handoff'`.

- [ ] **Step 3: Append to `orchard/handoff.py`**

The imports Task 9 added include `dataclasses`. Append at module level:

```python
@dataclass(frozen=True)
class Recovery:
    phase: str
    park_done: tuple[str, ...]
    restore_done: tuple[str, ...]
    lease_id: str | None
    lease_state: str           # "none", "ours", "orphaned", "gone" or "taken"
    coder_running: bool
    standin_running: bool
    action: str                # "none", "abandon" or "restore"
    disagreements: tuple[str, ...]


def recover(entries: list[dict], *, adapter, server, standin=None) -> Recovery:
    """After a supervisor restart, compare the ledger with the machine. The machine wins.

    The coder's state comes from the server's own tooling (docker for a container, ps and pgrep
    for a process), because a container server can hold chips that gozer shows as free or as
    another holder (spec section 6, branch table). The lease's state comes from the lease tool:
    "ours" when this process owns it, "orphaned" when the previous supervisor pid still owns it,
    "gone" when its chips are free, "taken" otherwise. Every disagreement is listed.

    The action: "abandon" when the coder is running and the park never sent its stop (the coder
    never left, so it is not stopped; re-leasing a coder that serves outside a park is plan 4's
    job), "restore" for any other interrupted handoff, "none" when no handoff was in progress.
    The stand-in is a process in its own session, so it can outlive the crash; its recorded pid
    is adopted and checked.
    """
    p = progress(entries)
    if p.phase == "idle":
        return Recovery("idle", (), (), None, "none", False, False, "none", ())
    if p.server:
        server.adopt(p.server)
    if standin is not None and p.standin:
        standin.adopt(p.standin)
    running = not server.confirm_stopped().stopped
    standin_running = bool(standin is not None and p.standin
                           and not standin.confirm_stopped().stopped)
    lease_id = (p.lease or {}).get("lease_id")
    lease_chips = set((p.lease or {}).get("chips") or ())
    mine = [c for c in adapter.status() if c.bdf in lease_chips]
    if mine and all(c.lease_pid == adapter.owner_pid for c in mine):
        state = "ours"
    elif not mine or all(c.state == "FREE" for c in mine):
        state = "gone"
    elif p.owner_pid is not None and all(c.lease_pid == p.owner_pid for c in mine):
        state = "orphaned"
    else:
        state = "taken"
    disagreements = []
    if "stopped" in p.park_done and "serve" not in p.restore_done and running:
        disagreements.append("the ledger says the coder was stopped; the machine shows it running")
    if "serve" in p.restore_done and not running:
        disagreements.append("the ledger says the coder was started; the machine shows it stopped")
    if state != "ours":
        disagreements.append(f"the ledger holds lease {lease_id}; the lease tool shows it {state}")
    action = "abandon" if running and "stop_sent" not in p.park_done else "restore"
    return Recovery(p.phase, p.park_done, p.restore_done, lease_id, state, running,
                    standin_running, action, tuple(disagreements))
```

Append inside `class Handoff`:

```python
    # ---- recovery after a restart (spec section 6, branch table; section 10) ----------------

    def recover_and_restore(self, recovery: Recovery, *, chips: int, who: str, reason: str,
                            wait_budget_s: float) -> CanaryResult | None:
        """Finish an interrupted handoff: bring the coder back under a lease this process holds.

        If the park never sent its stop and the coder is serving, the park is abandoned instead
        (see `recover`). Otherwise: a lease is judged by its owner pid, so after a restart the old lease belongs to a dead
        process. While the coder runs, its chips stay HELD-FOREIGN under that lease and nobody
        can reset or re-lease them. So: stop the coder if it runs, take a new lease (gozer reaps
        the orphan once no device is open, without a reset), and run the whole restore again from
        its reset. The stage test, if it was interrupted, is re-run by plan 4's stage machine.
        """
        if recovery.action == "none":
            return None
        self.ledger.append("decision", self.stage, decision="recover after restart; the machine wins",
                           **dataclasses.asdict(recovery))
        if recovery.action == "abandon":
            # The park never sent its stop and the coder is serving: close the park and leave the
            # coder up (spec section 6, step 2). Its lease belongs to the dead supervisor;
            # re-leasing a coder that serves outside a park is plan 4's job.
            stopped = self._stop_standin() if "standin_started" in recovery.park_done else None
            self._record("park", ABANDONED, recovered=True, standin_stopped=stopped)
            return None
        ours = recovery.lease_state == "ours"
        continuing = ours and recovery.coder_running and "serve" in recovery.restore_done
        if recovery.coder_running and not continuing:
            self._record("park", "stop_sent", result=self.server.stop(), recovered=True)
            check = self._wait_stopped(accept=("CLAIMED", "STALE", "FREE"))
            if not check["ok"]:
                self._block("after the restart the coder could not be confirmed stopped", **check)
            self._record("park", "stopped", recovered=True, **check)
        if not ours:
            old = self.lease.lease_id
            self.lease = reacquire(self.adapter, chips=chips, who=who, reason=reason,
                                   ledger=self.ledger, stage=self.stage, wait_budget_s=wait_budget_s,
                                   clock=self.clock, sleep=self.sleep)
            self.ledger.append("decision", self.stage, decision="new lease after restart",
                               old_lease_id=old, lease_id=self.lease.lease_id)
        if not continuing:
            self._record("restore", "restart", recovered=True)
        return self.restore()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_handoff_recovery.py`
Expected: PASS (7 tests). If `test_a_crash_after_any_ledger_event_reaches_the_expected_final_state` fails, the assertion lists each `k` with its `Recovery` and final state; fix the code for each listed `k`. Do not narrow the range of `k`.

- [ ] **Step 5: Mutation checks**

1. In `recover_and_restore`, delete the block that stops a running coder (`if recovery.coder_running and not continuing:` and its body). Run `tests/test_handoff_recovery.py::test_a_crash_after_any_ledger_event_reaches_the_expected_final_state`. Expected: FAIL for the `k` values after `restore serve` (the orphaned lease is never reaped while the coder holds it, so the new lease queues until the budget blocks). Restore: PASS.
2. Delete the `if not continuing: self._record("restore", "restart", recovered=True)` lines. Run the same test. Expected: FAIL for the `k` values after `restore serve` (the restore skips its reset and serve under the new lease). Restore: PASS.
3. Abandon rule: in `recover`, change the `action = ...` line to `action = "restore"`. Run the same test. Expected: FAIL for `k` 1 to 4 (a coder whose park never sent its stop is stopped). Restore: PASS.
4. Stand-in pid first: in `_park_standin`, replace the `self._record("park", "standin_started", standin=self.standin.record())` line after `start()` succeeds with `pass`. Run `tests/test_handoff_recovery.py::test_an_orphaned_standin_is_found_from_the_ledger_and_stopped`. Expected: FAIL (the orphaned stand-in is never found). Restore: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/handoff.py tests/test_handoff_recovery.py
git commit -m "Recover an interrupted handoff after a restart; tested by a crash after every ledger event"
```

---

### Task 12: Watchdog events and detectors

**Files:**
- Create: `orchard/watchdog.py` (first part)
- Test: `tests/test_watchdog_detectors.py`

**Interfaces:**
- Consumes: `ChipState`, `AdapterError` (Task 2); watchdog constants from `orchard.defaults`.
- Produces in `orchard.watchdog`:
  - `KINDS = frozenset({"response", "tool_call", "tool_result", "evidence", "ledger"})`.
  - `Event` (frozen dataclass): `ts: float`, `agent: str`, `kind: str`, `input_tokens: int | None = None`, `output_tokens: int | None = None`, `thinking_tokens: int | None = None`, `text_hash: str | None = None`, `duration_s: float | None = None`, `tool: str | None = None`, `args_hash: str | None = None`, `output_hash: str | None = None`, `had_tool_call: bool = False`, `name: str | None = None` (ledger event name or evidence path), `stage: int | None = None`.
  - `Finding` (frozen dataclass): `detector: str`, `agent: str`, `ts: float`, `summary: str`, `evidence: dict = {}`, `pause: bool = False`, `notice_only: bool = False` (a supervisor matter such as an idle lease: a notice, never a rung); method `record() -> dict`.
  - `PROGRESS_LEDGER_EVENTS = frozenset({"measurement", "stage_end", "test_result"})`. `NoNewEvidence` keeps one clock per agent and counts only evidence events and these ledger events as progress; a progress event from agent "supervisor" moves every agent's clock. Ladder entries and polls never reset it.
  - `LeaseIdle` findings carry `notice_only=True`.
  - Detectors, each with class attribute `name` and `feed(event: Event) -> Finding | None`: `IdenticalResponses(n=IDENTICAL_N)`, `ThinkingWithoutAction(cap=THINKING_CAP)`, `NoNewEvidence(window_s=NO_EVIDENCE_S)`, `RepeatedToolCall(n=REPEAT_TOOL_N)`, `LeaseIdle(status, lease_chips, *, agent, idle_s=LEASE_IDLE_S, poll_s=LEASE_POLL_S, exempt=lambda: False)`, `StageOverBudget(budgets: dict[int, float], *, agent="supervisor")`.
  - `transcript_detectors() -> list` (the three detectors that a transcript can feed), `replay(events, detectors) -> list[Finding]`.

- [ ] **Step 1: Write the failing tests**

```python
"""The detectors on synthetic event streams."""
import pytest

from orchard.adapters import AdapterError, ChipState
from orchard.watchdog import (Event, IdenticalResponses, LeaseIdle, NoNewEvidence, RepeatedToolCall,
                              StageOverBudget, ThinkingWithoutAction, replay)


def resp(ts, i=1000, o=100, think=50, text=None, tool=False, agent="a"):
    return Event(ts=ts, agent=agent, kind="response", input_tokens=i, output_tokens=o,
                 thinking_tokens=think, text_hash=text, had_tool_call=tool)


def times(findings):
    return [f.ts for f in findings]


def test_event_kind_is_checked():
    with pytest.raises(ValueError):
        Event(ts=0, agent="a", kind="chat")


def test_three_identical_responses_fire_once_and_the_detector_re_arms():
    found = replay([resp(t) for t in range(1, 7)], [IdenticalResponses()])
    assert times(found) == [3, 6]
    assert found[0].detector == "identical_responses" and found[0].evidence["count"] == 3


def test_identical_counts_with_a_tool_call_are_not_a_repeat():
    assert replay([resp(t, tool=True) for t in range(5)], [IdenticalResponses()]) == []


def test_the_same_nonempty_text_is_a_repeat_whatever_the_counts():
    evs = [resp(t, i=1000 + t, text="ab" * 32, tool=True) for t in range(3)]
    assert times(replay(evs, [IdenticalResponses()])) == [2]


def test_agents_are_counted_separately():
    evs = [resp(1, agent="a"), resp(2, agent="b"), resp(3, agent="a"), resp(4, agent="b")]
    assert replay(evs, [IdenticalResponses()]) == []


def test_identical_needs_n_of_at_least_two():
    with pytest.raises(ValueError):
        IdenticalResponses(n=1)


@pytest.mark.parametrize("think,tool,fires", [(20001, False, True), (20000, False, False),
                                              (27939, True, False), (None, False, False)])
def test_thinking_without_action(think, tool, fires):
    found = replay([resp(1, think=think, tool=tool)], [ThinkingWithoutAction()])
    assert bool(found) is fires


def test_no_new_evidence_fires_after_the_window_and_resets_on_evidence():
    d = NoNewEvidence(window_s=100)
    evs = [resp(0), resp(99), resp(100), resp(150),
           Event(ts=160, agent="a", kind="evidence", name="evidence/pcc.json"),
           resp(250), resp(260)]
    assert times(replay(evs, [d])) == [100, 260]


@pytest.mark.parametrize("name", ["measurement", "stage_end", "test_result"])
def test_a_named_progress_event_from_the_supervisor_counts_for_every_agent(name):
    d = NoNewEvidence(window_s=100)
    evs = [resp(0, agent="a"), resp(0, agent="b"),
           Event(ts=90, agent="supervisor", kind="ledger", name=name),
           resp(150, agent="a"), resp(150, agent="b")]
    assert replay(evs, [d]) == []


@pytest.mark.parametrize("name", ["retry", "escalate", "notice", "decision", "tick", "stage_start"])
def test_ladder_entries_and_polls_never_reset_the_clock(name):
    d = NoNewEvidence(window_s=100)
    evs = [resp(0), Event(ts=90, agent="supervisor", kind="ledger", name=name), resp(100)]
    assert times(replay(evs, [d])) == [100]


def test_each_agent_has_its_own_clock():
    d = NoNewEvidence(window_s=100)
    evs = [resp(0, agent="a"), resp(0, agent="b"),
           Event(ts=90, agent="b", kind="evidence", name="evidence/b.json"),
           resp(100, agent="a"), resp(100, agent="b")]
    found = replay(evs, [d])
    assert [(f.agent, f.ts) for f in found] == [("a", 100)]


def call(ts, tool="read_file", args="1" * 64):
    return Event(ts=ts, agent="a", kind="tool_call", tool=tool, args_hash=args)


def result(ts, tool="read_file", out="2" * 64):
    return Event(ts=ts, agent="a", kind="tool_result", tool=tool, output_hash=out)


def test_the_same_tool_call_three_times_fires():
    assert times(replay([call(1), call(2), call(3)], [RepeatedToolCall()])) == [3]


def test_the_same_tool_with_new_arguments_is_not_a_repeat():
    assert replay([call(1, args="1" * 64), call(2, args="3" * 64), call(3)], [RepeatedToolCall()]) == []


def test_calls_and_results_are_counted_separately():
    evs = [call(1), result(2), call(3), result(4), call(5)]
    found = replay(evs, [RepeatedToolCall()])
    assert times(found) == [5] and found[0].evidence["what"] == "call"


def test_the_same_output_three_times_fires():
    found = replay([result(1), result(2), result(3)], [RepeatedToolCall()])
    assert times(found) == [3] and found[0].evidence["what"] == "output"


class Status:
    def __init__(self, state="CLAIMED"):
        self.state, self.calls = state, 0

    def __call__(self):
        self.calls += 1
        if self.state == "error":
            raise AdapterError("gozer status failed")
        return [ChipState("0000:01:00.0", self.state, "orchard:run")]


def tick(ts):
    return Event(ts=ts, agent="sup", kind="ledger", name="tick")


def test_a_lease_idle_past_the_limit_fires():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    found = replay([tick(t) for t in range(0, 1861, 60)], [d])
    assert times(found) == [1800] and found[0].agent == "coder"
    assert found[0].notice_only and not found[0].pause      # a notice, never the repeat ladder


def test_an_open_device_resets_the_idle_clock():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    replay([tick(0), tick(900)], [d])
    status.state = "HELD"
    replay([tick(960)], [d])
    status.state = "CLAIMED"
    assert replay([tick(1020), tick(2700)], [d]) == []


def test_lease_idle_reads_status_at_most_once_per_poll():
    status = Status()
    d = LeaseIdle(status, lambda: {"0000:01:00.0"}, agent="coder", idle_s=1800, poll_s=60)
    replay([tick(t) for t in range(0, 600)], [d])
    assert status.calls == 10


def test_an_exempt_lease_and_an_unreadable_status_never_fire():
    d = LeaseIdle(Status(), lambda: {"0000:01:00.0"}, agent="c", idle_s=60, poll_s=1,
                  exempt=lambda: True)
    assert replay([tick(t) for t in range(200)], [d]) == []
    d = LeaseIdle(Status("error"), lambda: {"0000:01:00.0"}, agent="c", idle_s=60, poll_s=1)
    assert replay([tick(t) for t in range(200)], [d]) == []


def stage_event(ts, name, stage=2):
    return Event(ts=ts, agent="supervisor", kind="ledger", name=name, stage=stage)


def test_a_stage_past_its_budget_asks_for_a_pause_once():
    d = StageOverBudget({2: 100})
    found = replay([stage_event(0, "stage_start"), resp(50), resp(101), resp(200)], [d])
    assert times(found) == [101] and found[0].pause is True


def test_a_finished_stage_or_one_without_a_budget_never_fires():
    d = StageOverBudget({2: 100})
    evs = [stage_event(0, "stage_start"), stage_event(50, "stage_end"), resp(500),
           stage_event(600, "stage_start", stage=5), resp(5000)]
    assert replay(evs, [d]) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_watchdog_detectors.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.watchdog'`.

- [ ] **Step 3: Write `orchard/watchdog.py` (first part)**

```python
"""Loop detection and the response ladder (spec section 7).

This module owns:
- `Event`, the normalised record every detector reads. The model proxy (plan 4) and the transcript
  reader (orchard/transcripts.py) both produce it.
- The detectors. Each is a small class whose `feed(event)` returns a `Finding` or None. Each one
  re-arms after it fires, so a loop that continues after a nudge fires again and the ladder climbs.
- The response ladder (added below the detectors).

Default thresholds come from replaying the recorded qwen-code loop and 30 quiet chats
(orchard/defaults.py).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from orchard.adapters import AdapterError
from orchard.defaults import (IDENTICAL_N, LEASE_IDLE_S, LEASE_POLL_S, NO_EVIDENCE_S, REPEAT_TOOL_N,
                              RUNG_CAPS, THINKING_CAP)

KINDS = frozenset({"response", "tool_call", "tool_result", "evidence", "ledger"})


@dataclass(frozen=True)
class Event:
    ts: float                         # seconds since the epoch
    agent: str
    kind: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    text_hash: str | None = None      # sha256 of the visible response text; None when it is empty
    duration_s: float | None = None
    tool: str | None = None
    args_hash: str | None = None
    output_hash: str | None = None
    had_tool_call: bool = False       # the response asked for at least one tool call
    name: str | None = None           # ledger event name, or evidence path
    stage: int | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"event kind must be one of {sorted(KINDS)}, got {self.kind!r}")


@dataclass(frozen=True)
class Finding:
    detector: str
    agent: str
    ts: float
    summary: str
    evidence: dict = field(default_factory=dict, hash=False)
    pause: bool = False               # a budget cap: go straight to pause (spec section 10)
    notice_only: bool = False         # a supervisor matter (an idle lease): a notice, no rung

    def record(self) -> dict:
        return {"detector": self.detector, "agent": self.agent, "ts": self.ts,
                "summary": self.summary, "evidence": self.evidence, "pause": self.pause,
                "notice_only": self.notice_only}


class IdenticalResponses:
    """N responses in a row with the same text, or the same token counts and no tool call.

    The recorded loop was five calls with identical input and output token counts and an empty
    visible text, so the counts carry the signal. An empty text never matches by text: empty
    retries are common in quiet chats.
    """
    name = "identical_responses"

    def __init__(self, n: int = IDENTICAL_N):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self._last: dict[str, Event] = {}
        self._count: dict[str, int] = {}

    @staticmethod
    def _same(a: Event, b: Event) -> bool:
        if a.text_hash is not None and a.text_hash == b.text_hash:
            return True
        return (not a.had_tool_call and not b.had_tool_call
                and a.input_tokens is not None and a.output_tokens is not None
                and (a.input_tokens, a.output_tokens) == (b.input_tokens, b.output_tokens))

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind != "response":
            return None
        prev = self._last.get(ev.agent)
        count = self._count.get(ev.agent, 0) + 1 if prev is not None and self._same(prev, ev) else 1
        self._last[ev.agent] = ev
        if count >= self.n:
            self._count[ev.agent] = 0          # re-arm: the next n repeats fire again
            return Finding(self.name, ev.agent, ev.ts,
                           f"{self.n} identical responses in a row ({ev.input_tokens} input, "
                           f"{ev.output_tokens} output tokens)",
                           {"count": self.n, "input_tokens": ev.input_tokens,
                            "output_tokens": ev.output_tokens, "text_hash": ev.text_hash})
        self._count[ev.agent] = count
        return None


class ThinkingWithoutAction:
    name = "thinking_without_action"

    def __init__(self, cap: int = THINKING_CAP):
        self.cap = cap

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "response" and not ev.had_tool_call and (ev.thinking_tokens or 0) > self.cap:
            return Finding(self.name, ev.agent, ev.ts,
                           f"{ev.thinking_tokens} thinking tokens with no tool call (cap {self.cap})",
                           {"thinking_tokens": ev.thinking_tokens, "cap": self.cap})
        return None


# Ledger events that mean the work moved forward. Ladder entries (retry, escalate, notice,
# decision), polls and anything else never count, so the watchdog's own writes cannot silence it.
PROGRESS_LEDGER_EVENTS = frozenset({"measurement", "stage_end", "test_result"})


class NoNewEvidence:
    """An agent's model responses keep coming for window_s with no progress.

    Progress is an evidence file, or a ledger event named in PROGRESS_LEDGER_EVENTS. Each agent
    has its own clock. A progress event from "supervisor" (a stage_end, a measurement the
    supervisor wrote) moves every agent's clock; one from an agent moves only that agent's.
    """
    name = "no_new_evidence"

    def __init__(self, window_s: float = NO_EVIDENCE_S):
        self.window_s = window_s
        self._since: dict[str, float] = {}

    @staticmethod
    def _is_progress(ev: Event) -> bool:
        return ev.kind == "evidence" or (ev.kind == "ledger" and ev.name in PROGRESS_LEDGER_EVENTS)

    def feed(self, ev: Event) -> Finding | None:
        if self._is_progress(ev):
            if ev.agent == "supervisor":
                for agent in self._since:
                    self._since[agent] = ev.ts
            else:
                self._since[ev.agent] = ev.ts
            return None
        if ev.kind != "response":
            return None
        since = self._since.setdefault(ev.agent, ev.ts)
        if ev.ts - since >= self.window_s:
            self._since[ev.agent] = ev.ts
            return Finding(self.name, ev.agent, ev.ts,
                           f"{int(ev.ts - since)} s of model responses with no new evidence, "
                           "measurement, stage end or test result",
                           {"since": since, "window_s": self.window_s})
        return None


class RepeatedToolCall:
    """The same tool call (tool and arguments), or the same tool output, N times in a row."""
    name = "repeated_tool_call"

    def __init__(self, n: int = REPEAT_TOOL_N):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self._last: dict[tuple, tuple] = {}
        self._count: dict[tuple, int] = {}

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "tool_call":
            key, what = (ev.tool, ev.args_hash), "call"
        elif ev.kind == "tool_result":
            key, what = (ev.tool, ev.output_hash), "output"
        else:
            return None
        slot = (ev.agent, ev.kind)
        count = self._count.get(slot, 0) + 1 if self._last.get(slot) == key else 1
        self._last[slot] = key
        if count >= self.n:
            self._count[slot] = 0
            return Finding(self.name, ev.agent, ev.ts,
                           f"the same tool {what} ({ev.tool}) {self.n} times in a row",
                           {"what": what, "tool": ev.tool, "count": self.n})
        self._count[slot] = count
        return None


class LeaseIdle:
    """A lease held while its chips have no device open, for longer than idle_s.

    `status` is the adapter's status call; `lease_chips` returns the BDFs of the lease to watch;
    `exempt` returns True while idle chips are expected (during a park, between the stop and the
    reset). The status is read at most once per poll_s, on the clock of the events fed in.
    """
    name = "lease_idle"

    def __init__(self, status, lease_chips, *, agent: str, idle_s: float = LEASE_IDLE_S,
                 poll_s: float = LEASE_POLL_S, exempt=lambda: False):
        self.status, self.lease_chips, self.agent = status, lease_chips, agent
        self.idle_s, self.poll_s, self.exempt = idle_s, poll_s, exempt
        self._last_poll: float | None = None
        self._idle_since: float | None = None

    def feed(self, ev: Event) -> Finding | None:
        if self._last_poll is not None and ev.ts - self._last_poll < self.poll_s:
            return None
        self._last_poll = ev.ts
        chips = set(self.lease_chips())
        if not chips or self.exempt():
            self._idle_since = None
            return None
        try:
            states = {c.bdf: c.state for c in self.status() if c.bdf in chips}
        except AdapterError:
            return None                       # unknown is not idle
        if not states or any(s != "CLAIMED" for s in states.values()):
            self._idle_since = None
            return None
        if self._idle_since is None:
            self._idle_since = ev.ts
            return None
        if ev.ts - self._idle_since >= self.idle_s:
            since, self._idle_since = self._idle_since, ev.ts
            # An idle lease is the supervisor's decision (handoff.release_for_idle_phase), so
            # it is reported in a notice and never climbs the repeat ladder.
            return Finding(self.name, self.agent, ev.ts,
                           f"lease chips {sorted(chips)} held with no device open for "
                           f"{int(ev.ts - since)} s; consider release_for_idle_phase",
                           {"chips": sorted(chips), "since": since}, notice_only=True)
        return None


class StageOverBudget:
    """A stage past its wall-clock budget. The finding asks for a pause (spec section 10)."""
    name = "stage_over_budget"

    def __init__(self, budgets: dict[int, float], *, agent: str = "supervisor"):
        self.budgets, self.agent = dict(budgets), agent
        self._start: tuple[int | None, float] | None = None
        self._fired = False

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "ledger" and ev.name == "stage_start":
            self._start, self._fired = (ev.stage, ev.ts), False
        elif ev.kind == "ledger" and ev.name == "stage_end":
            self._start = None
        if self._start is None or self._fired:
            return None
        stage, t0 = self._start
        budget = self.budgets.get(stage)
        if budget is not None and ev.ts - t0 > budget:
            self._fired = True
            return Finding(self.name, self.agent, ev.ts,
                           f"stage {stage} has run {int(ev.ts - t0)} s, past its budget of "
                           f"{int(budget)} s", {"stage": stage, "budget_s": budget}, pause=True)
        return None


def transcript_detectors() -> list:
    """The detectors a session transcript can feed. The other three need ledger, evidence or lease
    events, which a transcript does not carry."""
    return [IdenticalResponses(), ThinkingWithoutAction(), RepeatedToolCall()]


def replay(events, detectors) -> list[Finding]:
    """Feed every event to every detector, in order, and collect the findings."""
    found = []
    for ev in events:
        for d in detectors:
            f = d.feed(ev)
            if f is not None:
                found.append(f)
    return found
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_watchdog_detectors.py`
Expected: PASS (31 tests).

- [ ] **Step 5: Mutation checks**

1. In `IdenticalResponses._same`, change `if a.text_hash is not None and a.text_hash == b.text_hash:` to `if a.text_hash == b.text_hash:`. Run `tests/test_watchdog_detectors.py::test_identical_counts_with_a_tool_call_are_not_a_repeat`. Expected: FAIL (two None hashes now match). Restore: PASS. (Task 14 repeats this mutation on the real-data fixtures.)
2. In `IdenticalResponses.feed`, delete `self._count[ev.agent] = 0` and the comment above it. Run `tests/test_watchdog_detectors.py::test_three_identical_responses_fire_once_and_the_detector_re_arms`. Expected: FAIL (it fires on 3, 4, 5 and 6). Restore: PASS.
3. Named progress only: in `NoNewEvidence._is_progress`, change the return line to `return ev.kind in ("evidence", "ledger")`. Run `tests/test_watchdog_detectors.py::test_ladder_entries_and_polls_never_reset_the_clock`. Expected: FAIL. Restore: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/watchdog.py tests/test_watchdog_detectors.py
git commit -m "Add the watchdog event stream and its six detectors"
```

---

### Task 13: The response ladder, the retry guard and the watchdog

**Files:**
- Modify: `orchard/watchdog.py` (append)
- Test: `tests/test_watchdog_ladder.py`

**Interfaces:**
- Consumes: `Event`, `Finding` (Task 12); `RUNG_CAPS`; a `Ledger`.
- Produces in `orchard.watchdog`:
  - `RUNGS = ("nudge", "escalate", "pause")`.
  - `Actuator` (Protocol): `nudge(agent: str, message: str) -> None`, `escalate(agent: str, stage: int | None) -> None`, `pause(reason: str, evidence: dict) -> None`.
  - `nudge_message(findings: list[Finding]) -> str`.
  - `Ladder(actuator, ledger, launched, *, caps=None)` with `respond(agent: str, findings: list[Finding], stage: int | None) -> str` returning `"nudge"`, `"escalate"`, `"pause"`, `"notice"` or `"none"`. Findings with `notice_only=True` get one `notice` entry and never take a rung. Ledger entries carry `watchdog=True`, `agent`, `findings`, and `rung` for an action: `retry` (nudge, with `message`), `escalate`, `decision` (`decision="pause"`, `reason`); a foreign agent gets `notice` with `report_only=True`.
  - `RetryGuard` with `MAX_SENDS = 2`, `key(agent, request) -> str` (static), `allow(agent: str, request: dict) -> bool`.
  - `Watchdog(detectors, ladder)` with `feed(event: Event) -> list[Finding]` and attribute `stage`.

- [ ] **Step 1: Write the failing tests**

```python
"""The response ladder, the retry guard and the watchdog that joins them."""
import pytest

from orchard.ledger import Ledger
from orchard.watchdog import (Event, Finding, IdenticalResponses, Ladder, RetryGuard,
                              StageOverBudget, Watchdog, nudge_message)


class Actuator:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def nudge(self, agent, message):
        self.calls.append(("nudge", agent))
        if self.fail:
            raise RuntimeError("proxy down")

    def escalate(self, agent, stage):
        self.calls.append(("escalate", agent, stage))

    def pause(self, reason, evidence):
        self.calls.append(("pause", reason))


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / "ledger.jsonl") as led:
        yield led


def finding(agent="coder", pause=False, summary="3 identical responses in a row"):
    return Finding("identical_responses", agent, 1.0, summary, {"count": 3}, pause=pause)


def rungs(ledger):
    return [(e["event"], e["data"].get("rung")) for e in ledger.read() if e["data"].get("watchdog")]


def test_a_launched_agent_climbs_nudge_escalate_pause_once_each(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    got = [ladder.respond("coder", [finding()], 2) for _ in range(5)]
    assert got == ["nudge", "escalate", "pause", "none", "none"]
    assert [c[0] for c in act.calls] == ["nudge", "escalate", "pause"]
    assert rungs(ledger) == [("retry", "nudge"), ("escalate", "escalate"), ("decision", "pause")]


def test_the_ledger_entry_comes_before_the_action(ledger):
    with pytest.raises(RuntimeError):
        Ladder(Actuator(fail=True), ledger, {"coder"}).respond("coder", [finding()], 2)
    # A restarted supervisor reads the nudge from the ledger and does not nudge again.
    act = Actuator()
    assert Ladder(act, ledger, {"coder"}).respond("coder", [finding()], 2) == "escalate"


def test_a_foreign_agent_gets_a_notice_and_no_action(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    assert [ladder.respond("someone", [finding("someone")], 2) for _ in range(4)] == ["notice"] * 4
    assert act.calls == []
    notes = [e["data"] for e in ledger.read() if e["event"] == "notice"]
    assert len(notes) == 4 and all(n["report_only"] is True for n in notes)
    assert notes[0]["findings"][0]["evidence"] == {"count": 3}


def test_a_budget_finding_goes_straight_to_pause(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    assert ladder.respond("supervisor", [finding("supervisor", pause=True)], 2) == "pause"
    assert ladder.respond("supervisor", [finding("supervisor", pause=True)], 2) == "none"
    assert [c[0] for c in act.calls] == ["pause"]


def test_an_idle_lease_gets_a_notice_and_never_a_nudge(ledger):
    act = Actuator()
    ladder = Ladder(act, ledger, {"coder"})
    idle = Finding("lease_idle", "coder", 1.0, "lease chips held idle", {}, notice_only=True)
    assert [ladder.respond("coder", [idle], 2) for _ in range(3)] == ["notice"] * 3
    assert act.calls == []
    assert len([e for e in ledger.read() if e["event"] == "notice"]) == 3
    assert ladder.respond("coder", [idle, finding()], 2) == "nudge"   # the repeat still climbs
    assert "repeating itself" not in str([e["data"] for e in ledger.read()
                                          if e["event"] == "notice"])


def test_rungs_are_counted_per_stage(ledger):
    ladder = Ladder(Actuator(), ledger, {"coder"})
    assert ladder.respond("coder", [finding()], 2) == "nudge"
    assert ladder.respond("coder", [finding()], 3) == "nudge"


def test_caps_can_allow_more_than_one_nudge(ledger):
    ladder = Ladder(Actuator(), ledger, {"coder"}, caps={"nudge": 2, "escalate": 1, "pause": 1})
    assert [ladder.respond("coder", [finding()], 2) for _ in range(3)] == ["nudge", "nudge", "escalate"]


def test_caps_must_name_every_rung(ledger):
    with pytest.raises(ValueError):
        Ladder(Actuator(), ledger, {"coder"}, caps={"nudge": 1})


def test_the_nudge_names_the_repeat_and_asks_for_something_different():
    msg = nudge_message([finding(summary="3 identical responses in a row (145299 input, 33348 output tokens)")])
    assert "3 identical responses in a row" in msg
    assert "greedily" in msg and "different" in msg


def test_retry_guard_allows_one_retry_of_the_same_call():
    g = RetryGuard()
    req = {"messages": [{"role": "user", "content": "x"}], "temperature": 0}
    assert [g.allow("coder", req) for _ in range(3)] == [True, True, False]
    nudged = {"messages": req["messages"] + [{"role": "user", "content": "try another way"}]}
    assert g.allow("coder", nudged) is True
    assert g.allow("other", req) is True


def resp(ts):
    return Event(ts=ts, agent="coder", kind="response", input_tokens=10, output_tokens=5)


def test_the_watchdog_tracks_the_stage_and_answers_once_per_agent(ledger):
    act = Actuator()
    wd = Watchdog([IdenticalResponses(), StageOverBudget({2: 1000})], Ladder(act, ledger, {"coder"}))
    wd.feed(Event(ts=0, agent="supervisor", kind="ledger", name="stage_start", stage=2))
    found = [f for t in (1, 2, 3) for f in wd.feed(resp(t))]
    assert wd.stage == 2 and [f.detector for f in found] == ["identical_responses"]
    assert act.calls == [("nudge", "coder")]
    entries = [e for e in ledger.read() if e["data"].get("watchdog")]
    assert entries[0]["stage"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_watchdog_ladder.py`
Expected: FAIL with `ImportError: cannot import name 'Ladder' from 'orchard.watchdog'`.

- [ ] **Step 3: Append to `orchard/watchdog.py`**

Task 12's imports already include `hashlib`, `json`, `Counter`, `Protocol` and `RUNG_CAPS`. Append:

```python
RUNGS = ("nudge", "escalate", "pause")


class Actuator(Protocol):
    """What the ladder can do to a launched agent. Plan 4's proxy and stage machine provide it."""

    def nudge(self, agent: str, message: str) -> None: ...

    def escalate(self, agent: str, stage: int | None) -> None: ...

    def pause(self, reason: str, evidence: dict) -> None: ...


def nudge_message(findings: list[Finding]) -> str:
    lines = ["The supervisor stopped this step because it is repeating itself:"]
    lines += [f"- {f.summary}" for f in findings]
    lines.append("The model server decodes greedily, so the same request returns the same answer. "
                 "Do something different: run a tool to get new information, write down what you "
                 "have found so far, or say what is blocking you.")
    return "\n".join(lines)


class Ladder:
    """The response ladder (spec section 7): nudge, then escalate, then pause, each capped.

    Rungs are counted per agent and stage, and the counts are read back from the ledger, so a
    restarted supervisor does not repeat a rung. An agent the supervisor did not launch gets one
    `notice` per finding with the evidence, and no action. A finding with pause=True (a budget
    cap) goes straight to the pause rung, for any agent, because pausing the run is the
    supervisor's own action.
    """

    def __init__(self, actuator: Actuator, ledger, launched, *, caps: dict | None = None):
        self.actuator, self.ledger = actuator, ledger
        self.launched = frozenset(launched)
        self.caps = dict(RUNG_CAPS if caps is None else caps)
        missing = set(RUNGS) - set(self.caps)
        if missing:
            raise ValueError(f"caps must name every rung; missing {sorted(missing)}")
        self._taken: Counter = Counter()
        for e in ledger.read():
            d = e["data"]
            if d.get("watchdog") and d.get("rung") in RUNGS:
                self._taken[(d.get("agent"), e["stage"], d["rung"])] += 1

    def _left(self, agent: str, stage, rung: str) -> bool:
        return self._taken[(agent, stage, rung)] < self.caps[rung]

    def _take(self, event: str, agent: str, stage, rung: str, records: list, **extra) -> None:
        # The entry is written before the action. A crash between the two counts the rung as
        # used, so a restart cannot take it twice.
        self.ledger.append(event, stage, watchdog=True, rung=rung, agent=agent, findings=records,
                           **extra)
        self._taken[(agent, stage, rung)] += 1

    def _pause(self, agent: str, stage, records: list) -> str:
        reason = "watchdog: " + "; ".join(r["summary"] for r in records)
        self._take("decision", agent, stage, "pause", records, decision="pause", reason=reason)
        self.actuator.pause(reason, {"agent": agent, "findings": records})
        return "pause"

    def respond(self, agent: str, findings: list[Finding], stage) -> str:
        quiet = [f for f in findings if f.notice_only]
        if quiet:
            # An idle lease and similar supervisor matters: one notice, never a nudge.
            self.ledger.append("notice", stage, watchdog=True, agent=agent,
                               findings=[f.record() for f in quiet])
        findings = [f for f in findings if not f.notice_only]
        if not findings:
            return "notice"
        records = [f.record() for f in findings]
        if any(f.pause for f in findings):
            return self._pause(agent, stage, records) if self._left(agent, stage, "pause") else "none"
        if agent not in self.launched:
            self.ledger.append("notice", stage, watchdog=True, report_only=True, agent=agent,
                               findings=records)
            return "notice"
        if self._left(agent, stage, "nudge"):
            message = nudge_message(findings)
            self._take("retry", agent, stage, "nudge", records, message=message)
            self.actuator.nudge(agent, message)
            return "nudge"
        if self._left(agent, stage, "escalate"):
            self._take("escalate", agent, stage, "escalate", records)
            self.actuator.escalate(agent, stage)
            return "escalate"
        if self._left(agent, stage, "pause"):
            return self._pause(agent, stage, records)
        return "none"                 # every rung used; the run is already paused


class RetryGuard:
    """Spec section 3: never retry the same call with the same inputs more than once.

    Nothing in plan 3 calls it. Plan 4's proxy asks before it forwards a request, and plan 4 must
    test that wiring. The counts live in memory: they are lost on a restart and never forgotten
    while the process lives. An identical request is allowed twice
    (the call and one retry). A nudged request has a different prompt, so it has a new key.
    """
    MAX_SENDS = 2

    def __init__(self):
        self._sends: Counter = Counter()

    @staticmethod
    def key(agent: str, request: dict) -> str:
        blob = json.dumps([agent, request], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def allow(self, agent: str, request: dict) -> bool:
        k = self.key(agent, request)
        if self._sends[k] >= self.MAX_SENDS:
            return False
        self._sends[k] += 1
        return True


class Watchdog:
    """Feeds each event to every detector and answers the findings once per agent."""

    def __init__(self, detectors, ladder: Ladder):
        self.detectors, self.ladder = list(detectors), ladder
        self.stage: int | None = None

    def feed(self, ev: Event) -> list[Finding]:
        if ev.kind == "ledger" and ev.name == "stage_start":
            self.stage = ev.stage
        findings = replay([ev], self.detectors)
        by_agent: dict[str, list[Finding]] = {}
        for f in findings:
            by_agent.setdefault(f.agent, []).append(f)
        for agent, fs in by_agent.items():
            self.ladder.respond(agent, fs, self.stage)
        return findings
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_watchdog_ladder.py`
Expected: PASS (11 tests).

- [ ] **Step 5: Mutation checks**

1. Foreign agents: in `respond`, delete the `if agent not in self.launched:` block. Run `tests/test_watchdog_ladder.py::test_a_foreign_agent_gets_a_notice_and_no_action`. Expected: FAIL. Restore: PASS.
2. Ledger first: in the nudge branch, move `self.actuator.nudge(agent, message)` above `self._take(...)`. Run `tests/test_watchdog_ladder.py::test_the_ledger_entry_comes_before_the_action`. Expected: FAIL (the second ladder nudges again). Restore: PASS.
3. Caps: change `_left` to `return True`. Run `tests/test_watchdog_ladder.py::test_a_launched_agent_climbs_nudge_escalate_pause_once_each`. Expected: FAIL. Restore: PASS.
4. Idle lease routing: in `respond`, delete the line `findings = [f for f in findings if not f.notice_only]`. Run `tests/test_watchdog_ladder.py::test_an_idle_lease_gets_a_notice_and_never_a_nudge`. Expected: FAIL (the idle lease is nudged). Restore: PASS.

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/watchdog.py tests/test_watchdog_ladder.py
git commit -m "Add the capped response ladder, the retry guard and the watchdog loop"
```

---

### Task 14: Transcripts, the loop signature and the replay

**Files:**
- Create: `orchard/transcripts.py`, `scripts/extract_qwen_signature.py`, `tests/fixtures/qwen_loop_signature.jsonl`, `tests/fixtures/qwen_quiet_signature.jsonl`
- Test: `tests/test_transcripts.py`, `tests/test_replay_local.py`

**Interfaces:**
- Consumes: `Event`, `replay`, `transcript_detectors`, `IdenticalResponses` (Tasks 12, 13).
- Produces in `orchard.transcripts`: `parse_ts(s: str) -> float`; `iso(ts: float) -> str` (millisecond UTC, `Z` suffix); `content_hash(obj, salt: bytes = b"") -> str`; `qwen_events(lines: Iterable[str], agent: str | None = None, salt: bytes = b"") -> Iterator[Event]`; `write_signature(events, path) -> int`; `load_signature(path) -> list[Event]`.
- Produces: `scripts/extract_qwen_signature.py TRANSCRIPT OUT` (prints the number of events written). It salts every hash with 16 random bytes that are never written: equality within one file is kept, which is all the detectors compare, and nobody can recover short content from a committed hash by hashing guesses.

The recorded loop is on disk (checked 2026-10-02): `~/.qwen/projects/-home-ttuser-code-tt-agents/chats/197354ac-2d30-4b0f-9d2a-a1518f7b33de.jsonl`. A replay of it with the code below, done while writing this plan, gave seven findings. Five are `thinking_without_action`, one on each call of the known loop (02:05:05 to 02:48:58Z, 27939 thinking tokens each). Two are `identical_responses`. One is on the third call of the known loop. The other is on the third call of a second five-call repeat in the same chat, at 03:40:20 to 03:53:39Z (input 198041, output 5895, thoughts 5281). The tests pin all seven. The other 30 chats give no finding (the coordinator checked the same corpus on 2026-10-02: 31 chats, 582 main-agent responses, and only chat 197354ac has consecutive identical input and output counts). This basis is in-sample: the thresholds were chosen on the same chats the tests replay, so the tests show the detectors separate this corpus and do not show they generalise. Plan 4's proxy is a different instrument from qwen-code telemetry, so plan 4 must check the thresholds again on proxy events. Spec section 12 also asks for a replay of a run that worked, such as the Audio8 bring-up; no transcript of that run is in this corpus, so the quiet fixture is a card-of-the-day chat. If the transcripts are gone when this task runs, stop and report. A synthetic loop is a weaker test (spec section 12), and the plan does not substitute one.

- [ ] **Step 1: Write the failing tests**

`tests/test_transcripts.py`:

```python
"""The qwen-code transcript reader, the committed signatures and what the detectors find in them."""
import json
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from orchard.transcripts import iso, load_signature, parse_ts, qwen_events, write_signature
from orchard.watchdog import KINDS, Event, IdenticalResponses, replay, transcript_detectors

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
SESSION = "abcdef12-0000-0000-0000-000000000000"

LOOP_FINDINGS = [
    ("thinking_without_action", "2026-10-01T02:05:05.444Z"),
    ("thinking_without_action", "2026-10-01T02:16:01.025Z"),
    ("identical_responses", "2026-10-01T02:26:58.738Z"),
    ("thinking_without_action", "2026-10-01T02:26:58.738Z"),
    ("thinking_without_action", "2026-10-01T02:37:57.977Z"),
    ("thinking_without_action", "2026-10-01T02:48:58.884Z"),
    ("identical_responses", "2026-10-01T03:46:56.001Z"),
]


def api(ts, i, o, think, text="", sub=None):
    ev = {"event.name": "qwen-code.api_response", "input_token_count": i, "output_token_count": o,
          "thoughts_token_count": think, "duration_ms": 1500, "response_text": text}
    if sub:
        ev["subagent_name"] = sub
    return {"type": "system", "subtype": "ui_telemetry", "timestamp": ts, "sessionId": SESSION,
            "systemPayload": {"uiEvent": ev}}


def assistant(ts, *calls):
    parts = [{"text": "private reasoning", "thought": True}]
    parts += [{"functionCall": {"name": n, "args": a}} for n, a in calls]
    return {"type": "assistant", "timestamp": ts, "sessionId": SESSION,
            "message": {"role": "model", "parts": parts}}


def tool_result(ts, name, response):
    return {"type": "tool_result", "timestamp": ts, "sessionId": SESSION,
            "message": {"role": "user", "parts": [{"functionResponse": {"id": "x", "name": name,
                                                                         "response": response}}]}}


def lines(*records):
    return [json.dumps(r) for r in records]


def test_a_response_learns_whether_a_tool_call_followed():
    evs = list(qwen_events(lines(
        api("2026-10-01T00:00:00.000Z", 100, 10, 5, text="Looking."),
        assistant("2026-10-01T00:00:00.010Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:00.020Z", "read_file", {"output": "x"}),
        api("2026-10-01T00:00:01.000Z", 200, 20, 30000),
    )))
    assert [e.kind for e in evs] == ["response", "tool_call", "tool_result", "response"]
    assert evs[0].had_tool_call is True and evs[3].had_tool_call is False
    assert evs[0].agent == "qwen:abcdef12" and evs[0].duration_s == 1.5
    assert evs[1].tool == "read_file" and len(evs[1].args_hash) == 64
    assert evs[3].text_hash is None and evs[3].thinking_tokens == 30000


def test_subagent_responses_and_other_records_are_skipped():
    evs = list(qwen_events(lines(
        api("2026-10-01T00:00:00.000Z", 100, 10, 5, sub="managed-auto-memory-dreamer"),
        {"type": "system", "subtype": "attribution_snapshot", "timestamp": "2026-10-01T00:00:00.000Z"},
        {"type": "user", "timestamp": "2026-10-01T00:00:00.000Z", "message": {"parts": []}},
    )))
    assert evs == []


def test_a_torn_last_line_is_ignored():
    good = lines(api("2026-10-01T00:00:00.000Z", 1, 1, 1))
    assert len(list(qwen_events(good + ['{"type": "assis']))) == 1


def test_times_round_trip_to_the_millisecond():
    assert iso(parse_ts("2026-10-01T02:26:58.738Z")) == "2026-10-01T02:26:58.738Z"


def found(path, detectors=None):
    return [(f.detector, iso(f.ts)) for f in replay(load_signature(path),
                                                    detectors or transcript_detectors())]


def test_the_loop_signature_has_the_recorded_shape():
    evs = load_signature(FIX / "qwen_loop_signature.jsonl")
    kinds = [e.kind for e in evs]
    assert (kinds.count("response"), kinds.count("tool_call"), kinds.count("tool_result")) == (74, 76, 76)
    loop = [e for e in evs if e.kind == "response" and e.input_tokens == 145299]
    assert len(loop) == 5 and {(e.output_tokens, e.thinking_tokens) for e in loop} == {(33348, 27939)}
    assert all(640 < e.duration_s < 660 for e in loop)


def test_the_detectors_fire_on_both_repeats_in_the_loop_chat():
    assert found(FIX / "qwen_loop_signature.jsonl") == LOOP_FINDINGS


def test_the_detectors_stay_quiet_on_a_chat_that_worked():
    assert found(FIX / "qwen_quiet_signature.jsonl") == []


def test_n_of_two_would_also_fire_on_a_two_call_retry():
    # The loop chat also has two identical calls at 01:47:42 and 01:50:03. N=3 ignores them.
    hits = found(FIX / "qwen_loop_signature.jsonl", [IdenticalResponses(n=2)])
    assert ("identical_responses", "2026-10-01T01:50:03.988Z") in hits


HEX64 = re.compile(r"[0-9a-f]{64}")
ALLOWED = {f.name for f in fields(Event)}


def test_the_fixtures_carry_no_message_text():
    for name in ("qwen_loop_signature.jsonl", "qwen_quiet_signature.jsonl"):
        for line in (FIX / name).read_text().splitlines():
            d = json.loads(line)
            assert set(d) <= ALLOWED, d
            for k, v in d.items():
                if not isinstance(v, str):
                    continue
                ok = ((k in ("text_hash", "args_hash", "output_hash") and HEX64.fullmatch(v))
                      or (k == "kind" and v in KINDS)
                      or (k == "agent" and re.fullmatch(r"qwen:[0-9a-f]{8}", v))
                      or (k == "tool" and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", v)))
                assert ok, (name, k, v)


def test_the_extract_script_writes_a_loadable_signature(tmp_path):
    src = tmp_path / "chat.jsonl"
    src.write_text("\n".join(lines(api("2026-10-01T00:00:00.000Z", 1, 2, 3, text="Hello there"),
                                   assistant("2026-10-01T00:00:00.001Z", ("glob", {"p": "*"})))) + "\n")
    out = tmp_path / "sig.jsonl"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "extract_qwen_signature.py"),
                        str(src), str(out)], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "2"
    evs = load_signature(out)
    assert [e.kind for e in evs] == ["response", "tool_call"] and evs[0].had_tool_call
    text = out.read_text()
    assert "Hello" not in text and "private reasoning" not in text and '"*"' not in text


def equality_structure(events):
    """Each hash replaced by the index of its first occurrence: what the detectors compare."""
    first = {}
    out = []
    for e in events:
        row = []
        for k in ("text_hash", "args_hash", "output_hash"):
            h = getattr(e, k)
            row.append(None if h is None else first.setdefault(h, len(first)))
        out.append(tuple(row))
    return out


def test_two_extractions_differ_in_hashes_and_agree_on_equality(tmp_path):
    src = tmp_path / "chat.jsonl"
    src.write_text("\n".join(lines(
        api("2026-10-01T00:00:00.000Z", 1, 2, 3, text="same"),
        assistant("2026-10-01T00:00:00.001Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:00.002Z", "read_file", {"output": "x"}),
        api("2026-10-01T00:00:01.000Z", 1, 2, 3, text="same"),
        assistant("2026-10-01T00:00:01.001Z", ("read_file", {"path": "a"})),
        tool_result("2026-10-01T00:00:01.002Z", "read_file", {"output": "x"}))) + "\n")
    script = str(REPO / "scripts" / "extract_qwen_signature.py")
    outs = []
    for name in ("one.jsonl", "two.jsonl"):
        subprocess.run([sys.executable, script, str(src), str(tmp_path / name)], check=True,
                       capture_output=True)
        outs.append(load_signature(tmp_path / name))
    hashes = [{getattr(e, k) for e in evs for k in ("text_hash", "args_hash", "output_hash")} - {None}
              for evs in outs]
    assert hashes[0].isdisjoint(hashes[1])
    assert equality_structure(outs[0]) == equality_structure(outs[1])
    # Within one file, repeated content still hashes equal.
    assert outs[0][0].text_hash == outs[0][3].text_hash


def test_write_then_load_round_trips(tmp_path):
    evs = [Event(ts=1.5, agent="qwen:abcdef12", kind="response", input_tokens=3)]
    assert write_signature(evs, tmp_path / "s.jsonl") == 1
    assert load_signature(tmp_path / "s.jsonl") == evs
```

`tests/test_replay_local.py`:

```python
"""Opt-in replay of every qwen-code transcript on this machine (spec section 12).

Normal test runs read nothing outside the repo, so this test runs only with ORCHARD_REPLAY=1.
It reads ~/.qwen/projects (or ORCHARD_QWEN_PROJECTS). A skipped replay is not evidence that the
detectors fire on the loop or stay quiet elsewhere.
"""
import os
from pathlib import Path

import pytest

from orchard.transcripts import iso, qwen_events
from orchard.watchdog import replay, transcript_detectors
from test_transcripts import LOOP_FINDINGS

QWEN = Path(os.environ.get("ORCHARD_QWEN_PROJECTS", str(Path.home() / ".qwen" / "projects")))
LOOP = "197354ac-2d30-4b0f-9d2a-a1518f7b33de.jsonl"

pytestmark = [
    pytest.mark.skipif(os.environ.get("ORCHARD_REPLAY") != "1",
                       reason="set ORCHARD_REPLAY=1 to replay the local qwen transcripts; "
                              "a skipped replay is not evidence"),
    pytest.mark.skipif(not QWEN.is_dir(),
                       reason=f"no qwen transcripts at {QWEN}; this replay was not run, and a "
                              "skipped replay is not evidence"),
]


def test_the_detectors_fire_on_the_recorded_loop_and_nowhere_else():
    paths = sorted(QWEN.glob("*/chats/*.jsonl"))
    assert any(p.name == LOOP for p in paths), f"{LOOP} is no longer under {QWEN}"
    hits = {}
    for p in paths:
        with p.open(encoding="utf-8", errors="replace") as f:
            found = replay(qwen_events(f), transcript_detectors())
        if found:
            hits[p.name] = [(x.detector, iso(x.ts)) for x in found]
    assert hits == {LOOP: LOOP_FINDINGS}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_transcripts.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchard.transcripts'`.

- [ ] **Step 3: Write `orchard/transcripts.py`**

```python
"""Read qwen-code session transcripts as the watchdog's event stream.

This module owns the mapping from a qwen-code (0.24.7) chat file to `Event`s, and the sanitised
signature files the tests commit. A chat file is JSON lines. The records used:
- `system` with subtype `ui_telemetry` whose uiEvent `event.name` is `qwen-code.api_response`:
  one model call, with input, output and thinking token counts, the duration and the visible
  response text. Calls by memory subagents carry `subagent_name` and are skipped.
- `assistant`: message.parts holds thought text and `functionCall {name, args}` parts. It follows
  the api_response it belongs to. A call retried by qwen-code itself has an api_response and no
  assistant record.
- `tool_result`: message.parts[].functionResponse {name, response}.

A response is held back until the next api_response or tool result, so `had_tool_call` is known
when it is emitted. Only hashes, counts, tool names, kinds and times leave this module; no message
text does.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from orchard.watchdog import Event


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def iso(ts: float) -> str:
    dt = datetime.fromtimestamp(round(ts, 3), tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def content_hash(obj, salt: bytes = b"") -> str:
    """sha256 of the JSON form, after `salt`. A random salt that is then thrown away keeps
    equality within one extraction and stops anyone from recovering short content (a file path,
    a one-word reply) by hashing guesses."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(salt + blob.encode("utf-8")).hexdigest()


def _parts(record: dict) -> list:
    parts = (record.get("message") or {}).get("parts") or []
    return [p for p in parts if isinstance(p, dict)]


def qwen_events(lines: Iterable[str], agent: str | None = None,
                salt: bytes = b"") -> Iterator[Event]:
    pending: Event | None = None
    calls: list[Event] = []

    def flush() -> list[Event]:
        nonlocal pending, calls
        out = [replace(pending, had_tool_call=bool(calls))] if pending is not None else []
        out += calls
        pending, calls = None, []
        return out

    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                          # a torn last line, or a non-JSON line
        if not isinstance(r, dict) or "timestamp" not in r:
            continue
        if agent is None and r.get("sessionId"):
            agent = "qwen:" + str(r["sessionId"])[:8]
        who = agent or "qwen:unknown"
        typ = r.get("type")
        if typ == "system" and r.get("subtype") == "ui_telemetry":
            ev = (r.get("systemPayload") or {}).get("uiEvent") or {}
            if ev.get("event.name") != "qwen-code.api_response" or ev.get("subagent_name"):
                continue
            yield from flush()
            text = ev.get("response_text") or ""
            dur = ev.get("duration_ms")
            pending = Event(ts=parse_ts(r["timestamp"]), agent=who, kind="response",
                            input_tokens=ev.get("input_token_count"),
                            output_tokens=ev.get("output_token_count"),
                            thinking_tokens=ev.get("thoughts_token_count"),
                            duration_s=None if dur is None else dur / 1000,
                            text_hash=content_hash(text, salt) if text else None)
        elif typ == "assistant":
            ts = parse_ts(r["timestamp"])
            for p in _parts(r):
                fc = p.get("functionCall")
                if isinstance(fc, dict):
                    calls.append(Event(ts=ts, agent=who, kind="tool_call", tool=fc.get("name"),
                                       args_hash=content_hash(fc.get("args"), salt)))
            if pending is None:
                yield from flush()
        elif typ == "tool_result":
            yield from flush()
            ts = parse_ts(r["timestamp"])
            for p in _parts(r):
                fr = p.get("functionResponse")
                if isinstance(fr, dict):
                    yield Event(ts=ts, agent=who, kind="tool_result", tool=fr.get("name"),
                                output_hash=content_hash([fr.get("name"), fr.get("response")],
                                                            salt))
    yield from flush()


def write_signature(events, path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            d = {k: v for k, v in asdict(e).items() if v is not None}
            f.write(json.dumps(d, sort_keys=True) + "\n")
            n += 1
    return n


def load_signature(path) -> list[Event]:
    return [Event(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
```

- [ ] **Step 4: Write `scripts/extract_qwen_signature.py`**

```python
#!/usr/bin/env python3
"""Write the sanitised event signature of one qwen-code transcript.

Usage: python3 scripts/extract_qwen_signature.py TRANSCRIPT OUT

OUT gets one JSON object per event: kinds, times, token counts, durations, tool names and
sha256 hashes. No message text is written. Every hash is salted with 16 random bytes that are
discarded when the script ends: equal content within one file still has equal hashes, which is
all the detectors compare, and nobody can test a guess against a committed hash. Prints the
number of events written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The script runs from a checkout without installing the package, so the repo root goes on the
# import path. This affects only this script's own process.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard.transcripts import qwen_events, write_signature  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, out = argv
    with open(src, encoding="utf-8", errors="replace") as f:
        salt = os.urandom(16)                   # never written anywhere
        n = write_signature(qwen_events(f, salt=salt), out)
    print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 5: Generate the two fixtures from the real transcripts**

```bash
mkdir -p tests/fixtures
python3 scripts/extract_qwen_signature.py \
  ~/.qwen/projects/-home-ttuser-code-tt-agents/chats/197354ac-2d30-4b0f-9d2a-a1518f7b33de.jsonl \
  tests/fixtures/qwen_loop_signature.jsonl
python3 scripts/extract_qwen_signature.py \
  ~/.qwen/projects/-home-ttuser-code-card-of-the-day-generator-qwencode-qwen3-8-27b/chats/8333c2b9-1be6-404c-be48-a0c997a42af5.jsonl \
  tests/fixtures/qwen_quiet_signature.jsonl
```

Expected output: `226`, then `265`. If either file is missing, stop and report; do not write a synthetic fixture. The hashes differ on every run because of the salt; generate the fixtures once and commit them. The quiet chat was chosen because it would fire if empty response texts counted as identical (it has empty-text retries), so it catches that mutation.

Read both fixture files whole before committing them, and confirm by eye that no line holds message text.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider -rs tests/test_transcripts.py tests/test_replay_local.py`
Expected: `test_transcripts.py` PASS (12 tests); `test_replay_local.py` SKIPPED with the reason "set ORCHARD_REPLAY=1 ...".

Then run the opt-in replay once and record its result in the task report:

Run: `ORCHARD_REPLAY=1 python3 -m pytest -q -p no:cacheprovider -rs tests/test_replay_local.py`
Expected: PASS (1 test). If a chat added since 2026-10-02 fires, the assertion shows its findings. Report them and do not change the thresholds in this task.

- [ ] **Step 7: Mutation checks against real data**

1. Empty texts must never match. In `orchard/watchdog.py`, `IdenticalResponses._same`, change `if a.text_hash is not None and a.text_hash == b.text_hash:` to `if a.text_hash == b.text_hash:`. Run `tests/test_transcripts.py::test_the_detectors_stay_quiet_on_a_chat_that_worked` and `::test_the_detectors_fire_on_both_repeats_in_the_loop_chat`. Expected: both FAIL on real data (the quiet chat fires at 2026-09-23T21:46:10.146Z; the loop chat gains findings at 01:47:42 and 01:54:10). Restore: PASS.
2. In `orchard/defaults.py`, set `IDENTICAL_N = 2`. Run `tests/test_transcripts.py::test_the_detectors_fire_on_both_repeats_in_the_loop_chat`. Expected: FAIL (extra findings at 01:50:03, 02:16:01, 02:37:57, 03:43:37, 03:50:16). Restore: PASS.
3. Set `THINKING_CAP = 1500`. Run `tests/test_transcripts.py::test_the_detectors_stay_quiet_on_a_chat_that_worked` and, with `ORCHARD_REPLAY=1`, `tests/test_replay_local.py`. Expected: both FAIL (a quiet response with 2034 thinking tokens fires). The loop test still passes. Restore: PASS. The cap cannot be tested at 16000: the 16861-token response in chat 433f3fa4 has a tool call, and `ThinkingWithoutAction` counts only responses without one, so no quiet response without a tool call exceeds 5281 tokens. Found by the implementer of batch 3.
4. Salt: in `scripts/extract_qwen_signature.py`, change `salt = os.urandom(16)` to `salt = b""`. Run `tests/test_transcripts.py::test_two_extractions_differ_in_hashes_and_agree_on_equality`. Expected: FAIL. Restore: PASS.

- [ ] **Step 8: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass (the replay test skipped).

```bash
git add orchard/transcripts.py scripts/extract_qwen_signature.py tests/fixtures/qwen_loop_signature.jsonl tests/fixtures/qwen_quiet_signature.jsonl tests/test_transcripts.py tests/test_replay_local.py
git commit -m "Read qwen-code transcripts as watchdog events; pin the detectors on the recorded loop"
```

---

### Task 15: Hardware validation of park and restore (a runbook entry with its driver)

**Files:**
- Create: `orchard/park_check.py`
- Modify: `docs/runbooks/hardware-validation.md` (append a section)
- Test: `tests/test_park_check.py`

**Interfaces:**
- Consumes: `GozerAdapter` (Task 3); `ServerSpec`, `ServerControl`, `ServerStandIn`, `ServerError` (Task 7); `Handoff`, `Budgets`, `Blocked` (Task 9); `AdapterError` (Task 2); `CHIPS_PER_BOARD`, `READY_POLL_S`; `Ledger`; `make_out_dir` from `orchard.hardware_check`.
- Produces: `orchard.park_check.main(argv=None) -> int` (exit 0 pass, 1 a step failed or blocked, 2 preflight refused, 3 acquire failed); `summary(entries, code) -> str`.

This task writes the driver and the runbook entry. Do not run the driver against hardware. The controller runs it.

- [ ] **Step 1: Write the failing tests**

```python
"""park_check against the real gozer CLI with fake roots, and real fake-server processes.

Nothing here opens a device: gozer's reset command is a stub that writes a marker file. The
gozer tests skip when gozer is not present, and a skip is not evidence.
"""
import json
import os
import socket
import subprocess
import sys

import pytest

from orchard import park_check
from orchard.ledger import Ledger
from test_gozer_contract import GOZER, QUIETBOX, build_sysfs, make_proc_dir, script

needs_gozer = pytest.mark.skipif(not GOZER.exists(),
                                 reason=f"gozer not found at {GOZER}; a skip is not evidence")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_box(tmp_path, monkeypatch):
    """(gozer wrapper, fake proc root, reset marker file) for a fake two-board box."""
    proc = tmp_path / "proc"
    proc.mkdir()
    marker = tmp_path / "resets"
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", build_sysfs(str(tmp_path / "sys"), QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", str(proc))
    monkeypatch.setenv("GOZER_HISTORY_ROOT", str(tmp_path / "history"))
    monkeypatch.setenv("GOZER_RESET_CMD",
                       str(script(tmp_path / "reset.sh", f'echo "$@" >> {marker}\n')))
    make_proc_dir(proc, os.getpid())
    gz = script(tmp_path / "gozer", f'exec {sys.executable} {GOZER} "$@"\n')
    return gz, proc, marker


def argv(tmp_path, gz):
    return ["--board", "0000:03:00.0", "--out-dir", str(tmp_path / "out"), "--gozer", str(gz),
            "--port", str(free_port()), "--standin-port", str(free_port()), "--ready-poll", "0.1",
            "--allow-gozer-env"]


def tripwire(tmp_path):
    """A gozer that must never run: it writes a marker and fails.

    Every test that calls park_check.main passes an explicit --gozer. Without one, a broken
    preflight would run the real gozer, against an empty GOZER_ROOT (so no real lease is seen)
    and with the default reset command, which is a real `tt-smi -r`.
    """
    marker = tmp_path / "tripwire-ran"
    return script(tmp_path / "gozer-tripwire", f'echo "$@" >> {marker}\nexit 1\n'), marker


def test_preflight_refuses_when_gozer_variables_are_set(tmp_path, monkeypatch):
    gz, marker = tripwire(tmp_path)
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    code = park_check.main(["--board", "0000:03:00.0", "--out-dir", str(tmp_path / "out"),
                            "--gozer", str(gz)])
    assert code == park_check.EXIT_PREFLIGHT
    assert not marker.exists()          # no gozer command ran


@needs_gozer
def test_preflight_refuses_while_the_other_board_is_busy(tmp_path, fake_box):
    gz, proc, marker = fake_box
    # A process outside any lease holds chip 0, as tt-smi -r does during another board's reset.
    make_proc_dir(proc, 888888, ppid=1, devs=[0])
    assert park_check.main(argv(tmp_path, gz)) == park_check.EXIT_PREFLIGHT
    assert not marker.exists()


@needs_gozer
def test_a_full_park_check_on_a_fake_box(tmp_path, fake_box):
    gz, _, marker = fake_box
    code = park_check.main(argv(tmp_path, gz))
    out = tmp_path / "out"
    with Ledger(out / "ledger.jsonl") as led:
        entries = led.read()
    stops = [e["data"] for e in entries if e["event"] == "notice" and not e["data"].get("ok", True)]
    assert code == park_check.EXIT_PASS, stops
    steps = [e["data"]["step"] for e in entries if e["event"] in ("park", "restore")]
    assert steps == ["note", "canary_before", "standin_started", "standin", "stop_sent", "stopped", "reset",
                     "reset", "serve", "ready", "canary", "resumed"]
    stop = next(e["data"] for e in entries
                if e["event"] == "park" and e["data"]["step"] == "stop_sent")
    # A server started while SIGTERM is blocked inherits the mask and ignores SIGTERM.
    assert stop["result"]["how"] == "SIGTERM"
    assert len(marker.read_text().splitlines()) == 3      # park, restore, release
    assert (out / "summary.md").exists()
    status = json.loads(subprocess.run([str(gz), "status", "--json"], capture_output=True,
                                       text=True).stdout)
    assert all(c["state"] == "FREE" for c in status["chips"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest -q -p no:cacheprovider tests/test_park_check.py`
Expected: FAIL with `ImportError: cannot import name 'park_check' from 'orchard'`.

- [ ] **Step 3: Write `orchard/park_check.py`**

```python
"""Hardware check for park and restore (docs/runbooks/hardware-validation.md, "Park check").

Run it as: python3 -m orchard.park_check --board <BDF of the board's first chip>

It drives the real handoff code (orchard/handoff.py) through the real gozer adapter on one real
board. Two copies of orchard/fake_server.py stand in for the coder and the CPU stand-in. No model
is loaded and no process opens a device, so the chips stay CLAIMED and each `gozer reset` runs on
idle chips.

What it exercises: the adapter's status, acquire, reset and release against the real gozer and a
real board; the stop checks (ps, pgrep, ss, curl) against a real process; every park and restore
step and its ledger entry; the canary comparison. What it does not: a device held open
(orchard/hardware_check.py covers that), a container server, a real stand-in model, and a real
coder's restart time.

Expected: three resets of the board (park, restore, release), about 42 s each (spec section 3).

Signals: SIGINT and SIGTERM are held back while a reset or release runs, so neither is cut short;
a signal that arrives meanwhile takes effect when the call returns, and the cleanup (stop both
servers, release the lease) runs. They are held only around those calls. A blocked signal mask is
inherited by every child process, so a fake server started while signals were blocked would
ignore its SIGTERM. If the process is killed with SIGKILL, no cleanup runs. The lease is owned by
this pid, so gozer reaps it once the pid is dead (the fake servers hold no device). Stop any fake
server left behind by the pids in the ledger. Never use --force, and never run tt-smi -r by hand.

Exit codes: 0 pass, 1 a step failed or blocked, 2 preflight refused, 3 acquire failed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

from orchard.adapters import AdapterError
from orchard.adapters.gozer import GozerAdapter
from orchard.defaults import CHIPS_PER_BOARD, READY_POLL_S
from orchard.handoff import Blocked, Budgets, Handoff
from orchard.hardware_check import make_out_dir
from orchard.ledger import Ledger
from orchard.server import ServerControl, ServerError, ServerSpec, ServerStandIn

WHO = "orchard:park-check"
REASON = "validate park and restore"
CANARY_PROMPT = "park check: reply with the configured answer"
EXIT_PASS, EXIT_FAIL, EXIT_PREFLIGHT, EXIT_ACQUIRE = 0, 1, 2, 3
GUARDED = {signal.SIGINT, signal.SIGTERM}
FAKE_SERVER = Path(__file__).with_name("fake_server.py")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python3 -m orchard.park_check",
                                description="Park and restore a fake coder on one real board.")
    p.add_argument("--board", required=True, help="BDF of the board's first chip, e.g. 0000:03:00.0")
    p.add_argument("--out-dir", default=None, help="default runs/park-check/<UTC time>-<BDF>/")
    p.add_argument("--gozer", default="gozer", help="a gozer that has `reset` (committed main, 0.3.2, has it)")
    p.add_argument("--port", type=int, default=20990, help="port of the fake coder")
    p.add_argument("--standin-port", type=int, default=20991, help="port of the fake stand-in")
    p.add_argument("--answer", default="the park check answer")
    p.add_argument("--ready-budget", type=float, default=60.0)
    p.add_argument("--ready-poll", type=float, default=READY_POLL_S)
    p.add_argument("--allow-gozer-env", action="store_true",
                   help="test only: run although GOZER_* variables are set")
    return p.parse_args(argv)


def fake_argv(port: int, answer: str) -> tuple[str, ...]:
    return (sys.executable, str(FAKE_SERVER), "--port", str(port), "--answer", answer,
            "--model", "fake")


@contextlib.contextmanager
def signals_held():
    """Hold SIGINT and SIGTERM back for the duration; they arrive when the block ends."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, GUARDED)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


class HeldAdapter:
    """The adapter, with signals held back around reset and release only.

    The mask is inherited by child processes, so it must not be held while a server starts.
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self.owner_pid = adapter.owner_pid

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def reset(self, lease):
        with signals_held():
            self._adapter.reset(lease)

    def release(self, lease):
        with signals_held():
            self._adapter.release(lease)


def _interrupt(signum, frame):
    # SIGTERM's default action ends the process with no cleanup. Raising here runs the finally
    # blocks, which stop the servers and release the lease.
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


def summary(entries: list[dict], code: int) -> str:
    lines = ["# Park check summary", ""]
    for e in entries:
        d = e["data"]
        if e["event"] in ("park", "restore"):
            lines.append(f"{e['event']:<8} {d.get('step')}")
        elif e["event"] == "measurement":
            lines.append(f"measured {d['name']} = {d['value']} {d['unit']}")
        elif e["event"] == "notice" and (d.get("blocked") or d.get("ok") is False):
            lines.append(f"STOP     {d.get('reason') or d.get('check')}")
    lines += ["", f"Exit code: {code}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else make_out_dir(args.board, base="runs/park-check")
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.jsonl")
    old_term = signal.signal(signal.SIGTERM, _interrupt)
    try:
        ledger.append("run_start", None, options=vars(args), pid=os.getpid())
        code = _run(args, out, ledger)
        text = summary(ledger.read(), code)
        (out / "summary.md").write_text(text)
        print(text)
        return code
    finally:
        ledger.close()
        signal.signal(signal.SIGTERM, old_term)


def _run(args, out: Path, ledger) -> int:
    env_set = sorted(k for k in os.environ if k.startswith("GOZER_"))
    if env_set and not args.allow_gozer_env:
        # A GOZER_* variable points gozer at other state or replaces the reset command.
        ledger.append("notice", None, check="P.env", ok=False, gozer_env=env_set)
        return EXIT_PREFLIGHT
    adapter = HeldAdapter(GozerAdapter(gozer=args.gozer))
    try:
        chips = adapter.status()
    except AdapterError as exc:
        ledger.append("notice", None, check="P", ok=False, reason=str(exc))
        return EXIT_PREFLIGHT
    first = next((c for c in chips if c.bdf == args.board), None)
    if first is None:
        ledger.append("notice", None, check="P", ok=False, reason=f"{args.board} not in gozer status")
        return EXIT_PREFLIGHT
    busy = [c.bdf for c in chips if c.board == first.board and (c.state != "FREE" or c.who)]
    # A reset elsewhere makes every board look busy for about 42 s; two resets must not overlap.
    resetting = [c.bdf for c in chips if c.board != first.board and c.state == "BUSY-UNTRACKED"]
    if busy or resetting:
        ledger.append("notice", None, check="P", ok=False, busy=busy, other_board_busy=resetting)
        return EXIT_PREFLIGHT
    ledger.append("notice", None, check="P", ok=True, board=first.board)
    try:
        lease = adapter.acquire(CHIPS_PER_BOARD, WHO, REASON, exact=args.board)
    except AdapterError as exc:
        ledger.append("notice", None, check="A", ok=False, reason=str(exc))
        return EXIT_ACQUIRE
    ledger.append("decision", None, decision="lease taken", lease=lease.record())
    coder = ServerControl(ServerSpec("park-check/coder", "process", args.port, "fake",
                                     argv=fake_argv(args.port, args.answer)),
                          log_path=str(out / "coder.log"), ready_poll_s=args.ready_poll)
    standin = ServerStandIn(ServerControl(ServerSpec("park-check/standin", "process",
                                                     args.standin_port, "fake",
                                                     argv=fake_argv(args.standin_port, args.answer)),
                                          log_path=str(out / "standin.log"),
                                          ready_poll_s=args.ready_poll),
                            ready_budget_s=args.ready_budget)
    code = EXIT_FAIL
    try:
        coder.start(lease)
        coder.wait_ready(args.ready_budget)
        note = out / "handoff-note.json"
        note.write_text(json.dumps({"goal": "park check", "stage": "none", "evidence": [],
                                    "next_action": "restore the fake coder",
                                    "check_on_return": "the canary answer matches"}))
        h = Handoff(ledger=ledger, stage=None, adapter=adapter, server=coder, standin=standin,
                    lease=lease, canary_prompt=CANARY_PROMPT, note_path=note,
                    evidence_dir=out / "evidence", budgets=Budgets(cold_boot_s=args.ready_budget))
        h.park()
        ledger.append("decision", None, decision="no stage test in the park check; the board is "
                                                 "idle between the park reset and the restore reset")
        result = h.restore()
        code = EXIT_PASS if result is not None and result.match else EXIT_FAIL
    except (Blocked, AdapterError, ServerError) as exc:
        ledger.append("notice", None, check="ABORT", ok=False, reason=str(exc))
    finally:
        code = _cleanup(adapter, lease, coder, standin, ledger, code)
    return code


def _cleanup(adapter, lease, coder, standin, ledger, code: int) -> int:
    for name, stop in (("coder", coder.stop), ("standin", standin.stop)):
        try:
            stop()
        except Exception as exc:            # cleanup: record and continue to the release
            ledger.append("notice", None, check=f"cleanup.{name}", ok=False, reason=str(exc))
    try:
        adapter.release(lease)
        ledger.append("notice", None, check="cleanup.release", ok=True)
    except AdapterError as exc:
        ledger.append("notice", None, check="cleanup.release", ok=False, reason=str(exc),
                      recovery=f"once nothing holds the board, run: gozer release {lease.lease_id}; "
                               "never --force, never tt-smi -r by hand")
        return EXIT_FAIL
    return code


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest -q -p no:cacheprovider -rs tests/test_park_check.py`
Expected: PASS (3 tests). If the gozer tests are skipped, record the reason; a skip is not evidence.

- [ ] **Step 5: Append the runbook entry to `docs/runbooks/hardware-validation.md`**

```markdown
## Park check (plan 3)

Purpose: run the park and restore code (`orchard/handoff.py`) through the gozer adapter on one
real board, with fake servers in place of the coder and the CPU stand-in. It checks the adapter
against the real gozer and a real board reset, the stop checks against a real process, and the
ledger steps. It does not check a device held open (the hardware-check driver above does), a
container server, a real stand-in model, or a real restart time.

Who runs it: the controller. An implementer does not. It takes its own gozer lease under its own pid.

Before running:
- `gozer status` shows every chip of the target board `FREE`, and the other board is not
  `BUSY-UNTRACKED` (another reset is running). The driver refuses otherwise (exit 2).
- No `GOZER_*` variable is set.
- Ports 20990 and 20991 are free (`ss -ltn "( sport = :20990 or sport = :20991 )"` prints only
  its header), or pass `--port` and `--standin-port`.

Run, from the repo root:

    python3 -m orchard.park_check --board 0000:03:00.0

Expected, measured on the hardware-check runs: three resets of the board, about 42 s each (the
park reset, the restore reset, and the reset inside the release), about 2.5 min in all. The
ledger under `runs/park-check/<UTC time>-<BDF>/` shows the park steps `note`, `canary_before`,
`standin_started`, `standin`, `stop_sent`, `stopped`, `reset`, then the restore steps `reset`,
`serve`, `ready`, `canary`, `resumed`, and three measurements (`park_reset_seconds`,
`restore_reset_seconds`, `coder_ready_wait_seconds`). Exit 0. Afterwards `gozer status` shows the board `FREE`.

Stop conditions: any exit other than 0. Read `summary.md` and the last `notice` in the ledger.
If the summary says the release failed, wait until nothing holds the board, then run
`gozer release <lease-id>` from the ledger's `lease taken` decision. Never use `--force`, never
run `tt-smi -r` by hand.

Record afterwards: the run directory, the exit code, the three reset times, and anything in the
summary marked STOP.
```

- [ ] **Step 6: Mutation checks**

1. Preflight: in `_run`, change `if busy or resetting:` to `if busy:`. Run `python3 -m pytest -q -p no:cacheprovider tests/test_park_check.py::test_preflight_refuses_while_the_other_board_is_busy`. Expected: FAIL (the run goes ahead and resets the board while the other board is busy). Restore: PASS.
2. Env guard (the tripwire gozer is in place, so a broken guard cannot reach the real gozer): in `_run`, change `if env_set and not args.allow_gozer_env:` to `if False:`. Run `tests/test_park_check.py::test_preflight_refuses_when_gozer_variables_are_set`. Expected: FAIL (the tripwire marker exists). Restore: PASS.
3. Signal mask: in `main`, wrap the `code = _run(args, out, ledger)` line in `with signals_held():`. Run `tests/test_park_check.py::test_a_full_park_check_on_a_fake_box`. Expected: FAIL on `stop["result"]["how"] == "SIGTERM"` (it is `SIGKILL`), after about 6 minutes, because each fake server inherited the blocked mask and was killed only after the 120 s stop timeout. This was found by running the plan's code before the plan was handed over. Restore: PASS in a few seconds.

- [ ] **Step 7: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add orchard/park_check.py tests/test_park_check.py docs/runbooks/hardware-validation.md
git commit -m "Add the park-and-restore hardware check with fake servers and its runbook entry"
```

---

### Task 16: Documentation

**Files:**
- Modify: `README.md` (status table, test count, what is built), `CLAUDE.md` (project log), `docs/superpowers/specs/2026-10-01-orchard-design.md` (sections 8 and 14, only where facts changed)

**Interfaces:**
- Consumes: the results recorded in Tasks 4, 7, 14 and 15 (skips, the opt-in replay result).
- Produces: documents that match the code.

- [ ] **Step 1: Get the facts to write down**

Run: `python3 -m pytest -q -p no:cacheprovider -rs 2>&1 | tail -8`
Write down the passed count and every skipped test with its reason. Expected: 838 passed and 1 skipped (the opt-in replay); 607 of them predate plan 3. A different count means a task added or dropped tests; say which in the README and the log.

- [ ] **Step 2: Update `README.md`**

In the status table, replace the row `| Supervisor loop, park and restore, watchdog | designed, not started |` with:

```markdown
| Lease adapters (`orchard/adapters/`: gozer, single tenant) | built, tested against fakes; the gozer adapter also against the real gozer CLI with fake roots |
| Server control and canary (`orchard/server.py`, `orchard/canary.py`) | built, tested; stop checks also run against real ps, pgrep, ss and curl with a fake server |
| Park and restore (`orchard/handoff.py`) | built, tested against a fake machine, including a crash after every ledger event; not yet run on hardware (`orchard/park_check.py` is written for the controller to run) |
| Watchdog (`orchard/watchdog.py`, `orchard/transcripts.py`) | built, tested; thresholds set from a replay of the recorded qwen-code loop and 30 quiet chats |
| Supervisor loop, stage state machine, model proxy | designed, not started (plan 4) |
```

Replace "The suite has 607 tests" with the count from Step 1, and add one sentence naming the opt-in replay (`ORCHARD_REPLAY=1 python3 -m pytest tests/test_replay_local.py`). Add a "What is built" bullet for each of the four new parts, one or two sentences each, in the style of the existing bullets.

- [ ] **Step 3: Update `CLAUDE.md`**

Under `## Status`, replace "Plans 2 to 4 are not started." with a sentence that plan 2 is merged in tt-gozer and plan 3 is implemented on branch `plan3-supervisor-behavior`, with plan 4 not started. Under `## Key decisions`, add:

```markdown
- Tiers (operator, 2026-10-02): large = Qwen3.8-27B on all 4 chips (both boards); small =
  Qwen3.8-27B on 2 chips (one board); a CPU tier chosen by measurement. While the large tier is
  loaded no board is free, so every hardware stage parks it. `handoff.decide_park` makes that call
  from the boards the coder holds and the boards a stage needs.
- Park and restore is a step-by-step state machine. Each step writes a ledger entry; a restart
  replays the ledger and then believes the machine (docker, ps, gozer status) where they differ.
  After a crash the old lease belongs to a dead pid, so recovery stops the coder, takes a new
  lease, resets it and restores.
- The watchdog acts only through an injected actuator, which plan 4 supplies. Each ladder rung is
  written to the ledger before it is acted on, so a restart cannot repeat a rung.
```

Under `## Log`, add a dated entry for plan 3: the prompt ("write plan 3: supervisor behavior (adapters, park and restore, watchdog)"), that it was executed task by task, and the notable moment from Task 14: the replay found a second five-call repeat in the loop chat (03:40 to 03:53Z, 5281 thinking tokens), so the identical-response detector fires twice there and the thinking cap fires five times. Add any skip or surprise recorded in Tasks 4, 7, 14 and 15. Also record that the plan was revised before execution after a review (`.superpowers/plan3-review.md`, 1 critical and 10 important findings, all accepted): a tripwire gozer in the park-check tests, abandoning a park killed before its stop, recording the stand-in pid first, an explicit abandoned step, a single-tenant adapter that can be rebuilt after a crash and leases whole boards, salted fixture hashes, and per-attempt evidence files.

- [ ] **Step 4: Update the spec, sections 8 and 14 only**

In section 8, item 5, after the sentence about `gozer wait`, add: "The tt-orchard gozer adapter (plan 3) never calls `wait`; it claims a ticket by repeating `acquire --owner-pid ... --ticket ...` every 10 s." In section 14, item 1, append: "Partly answered on 2026-10-02 by the operator: large = Qwen3.8-27B on 4 chips, small = Qwen3.8-27B on 2 chips; the CPU tier is still chosen by measurement." Add a new item 6:

```markdown
6. Watchdog thresholds. **Set on 2026-10-02 from a replay** of the recorded loop (qwen-code
   0.24.7, chat 197354ac) and 30 quiet chats (582 main-agent responses): N = 3 identical responses,
   thinking cap 20000 tokens. The replay found a second five-call repeat in the same chat
   (03:40 to 03:53Z, input 198041, output 5895, thoughts 5281 tokens), which N = 3 also catches.
   No quiet chat fires. T (no new evidence) and the lease-idle limit are not measured, because
   transcripts carry no evidence or lease events.
```

Do not change any other section.

- [ ] **Step 5: Sweep the new text and the new code comments**

Run: `git diff -U0 README.md CLAUDE.md docs/superpowers/specs/2026-10-01-orchard-design.md | grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what'`
Then run the same pattern over the files plan 3 created: `git diff --name-only --diff-filter=A main -- orchard tests scripts | xargs grep -nE ', not |rather than|instead of|reads as|honest|the one|which is why|This is what'`
Expected: no output from the first. Rewrite any hit in either as a plain statement, or justify it in the task report (for example, the `pgrep -g` comment states a contrast that matters: `pgrep -f` matches itself).

- [ ] **Step 6: Run the whole suite and commit**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all pass.

```bash
git add README.md CLAUDE.md docs/superpowers/specs/2026-10-01-orchard-design.md
git commit -m "Document plan 3: adapters, park and restore, watchdog, and the replay findings"
```

---

## Hand-off notes

- Plan 4 consumes: `LeaseAdapter` and `GozerAdapter`; `ServerControl`, `ServerStandIn`; `decide_park`, `Handoff.park/restore/recover_and_restore`, `recover`, `reacquire`, `release_for_idle_phase`, `Blocked`; `Event`, the detectors, `Ladder`, `RetryGuard`, `Watchdog`, and the `Actuator` protocol it must implement.
- Plan 4 must also: run the stage test between `park()` and `restore()` under `TT_VISIBLE_DEVICES` from the held lease, with a deadline; re-lease a coder that is serving when the supervisor restarts outside a park, including after `recover` abandons a park; restart a model server that dies, once, with a canary check (spec section 10); point the CPU stand-in at its own caches.
- Plan 4 notes from the review:
  - `RetryGuard` has no caller in plan 3. Plan 4 wires it into the proxy, tests that wiring, and decides whether its counts must survive a restart.
  - A finding with `pause=True` pauses the run even for a foreign agent. Today only `StageOverBudget` sets it, with agent "supervisor"; keep it that way or route foreign pause findings to a notice.
  - The watchdog thresholds are in-sample defaults; check them again on proxy events (Task 14) and pass N, T and the caps from the run config.
  - tt-model requires the `--device-id` count to match the profile. gozer grants whole boards, so choose profiles that use the whole lease.
  - gozer's exit 15 does not always mean "nothing happened": `release` can reset and then return 15 ("aborted before teardown"), and `reset`'s 15 also covers "no longer locked" and "belongs to another lease". Plan 3 retries a refused reset once and then blocks; the block message can be wrong for those causes. gozer re-checks ownership under its mutex, so no wrong reset follows.
