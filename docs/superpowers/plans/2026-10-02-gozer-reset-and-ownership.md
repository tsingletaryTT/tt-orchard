# tt-gozer: reset in place and descendant ownership (plan 2 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give tt-gozer the two additions the supervisor needs to hold a board lease through a model swap: `gozer reset <lease>` (reset a held lease's chips without releasing it) and an ownership check that counts the owner's child processes. Add the skill and docs that teach the pattern. Make the gozer test suite independent of the machine it runs on.

**Architecture:** All work happens in the git worktree `~/code/tt-gozer-orchard` (branch `orchard-hold-through-swap`, created from `main` at `3e95328`). The live checkout `~/code/tt-gozer` is the `gozer` other agents run through a symlink and is never edited. `procfd.py` gains a parent-pid walk. `gatekeeper.reconcile` uses it when deciding `HELD` versus `HELD-FOREIGN`. `Keymaster.reset` follows the pattern `Keymaster.release` already uses (re-read the gate under the mutex before and after the reset, never hold the mutex across it). `cli.py` gains `reset` and two exit codes. A new skill `gozer-park` documents the supervisor pattern.

**Tech Stack:** Python 3.9+ standard library only (gozer's stated requirement; the repo uses `from __future__ import annotations`), pytest, the existing fake-sysfs and fake-proc test fixtures.

**Spec:** `docs/superpowers/specs/2026-10-01-orchard-design.md`, sections 4, 6, 8 and 11 (revised 2026-10-02). The fallback design (`yield`/`redeem` with a reservation) is not part of this plan.

## Global Constraints

- Work only in `/home/ttuser/code/tt-gozer-orchard`. Never edit `/home/ttuser/code/tt-gozer`. Never run `git push`, never switch branches in the live checkout.
- No hardware. Tests use fake sysfs and fake proc trees. Tests stub the reset command with `GOZER_RESET_CMD` or `km.reset_runner`; nothing runs the real `tt-smi`.
- Standard library only. Keep `from __future__ import annotations` at the top of every new module.
- Do not change `gozer/__version__`. The live checkout holds an uncommitted version change (`0.3.3`); the owner bumps the version when the branch merges.
- Do not edit `gozer/report.py`, `gozer/history_archive.jsonl`, or `docs/index.html`. The live checkout has uncommitted edits to the report code, and touching it here would conflict at merge. A test may call `report.compute_stats`.
- Keep edits to shared files (`cli.py`, `keymaster.py`, `gatekeeper.py`, `procfd.py`, `README.md`) as small appended or localized hunks. The live checkout has uncommitted edits to `cli.py`, `README`-adjacent docs and `CLAUDE.md`.
- Comments explain why, in plain English. Keep comments and docs matching the code.
- Commit messages are plain English, one idea per commit.
- Writing rules for docs and skills: state the finding; short sentences; no aphoristic closers; no "X, not Y" framing; no metaphors that stand in for a claim.

## Baseline

On `3e95328` the suite gives 259 passed and 1 failed (`tests/test_history_root.py::test_history_survives_gozer_root_being_wiped`). The failing test builds a `Gatekeeper` with a fake sysfs but the default real `/proc`. This box has real Tenstorrent devices held open by other agents, so the chips look busy, `acquire` queues instead of granting, and no `granted` event is written. Task 0 fixes it.

## Review Focus

1. A reset that races a release or a reap, so the chips now belong to another tenant (Task 2: monkeypatched race test, before and after the reset).
2. A device holder that is a deep descendant, a cycle in the parent chain, an unreadable or missing `status` file, or a holder whose owner has exited (Task 1).
3. A reset requested while a workload still has the device open (Task 2: refused, nothing reset).
4. Distinct exit codes for "refused, nothing done", "reset ran and failed", and "reset ran, then the lease was gone or taken" (Task 2, CLI).
5. The hold-through-swap sequence end to end: lease survives a backdated `since` with no device open, reset in place keeps the lease, a child server reads as `HELD`, release at the end still works (Task 4).

## File Structure

| File | Responsibility |
|---|---|
| `tests/conftest.py` | add `fake_proc_tree` (fake `/proc` with `status`, `comm`, `fd` per pid) and the guard that stops a test reading the real `/proc` |
| `tests/test_history_root.py` | give each Gatekeeper a fake proc root |
| `gozer/procfd.py` | `parent_pid`, `is_descendant` |
| `gozer/gatekeeper.py` | `reconcile` counts descendants of the owner as owners |
| `gozer/keymaster.py` | `ResetResult`, `Keymaster.reset` |
| `gozer/cli.py` | `cmd_reset`, parser entry, `EXIT_RESET_FAILED` (17), `EXIT_RESET_CHANGED_HANDS` (18) |
| `tests/test_procfd.py`, `tests/test_ownership.py`, `tests/test_reset_keymaster.py`, `tests/test_reset_cli.py`, `tests/test_hold_through_swap.py` | new and extended tests |
| `skills/gozer-park/SKILL.md`, `install.sh`, `tests/test_install.py` | the new skill and its installation |
| `skills/gozer-keymaster/SKILL.md`, `skills/gozer-gatekeeper/SKILL.md`, `README.md`, `docs/superpowers/specs/2026-08-14-tt-gozer-design.md`, `CLAUDE.md` | documentation of the new behavior |

---

### Task 0: Make the suite independent of the machine

**Files:**
- Modify: `tests/conftest.py`, `tests/test_history_root.py`

**Interfaces:**
- Produces: `fake_proc_tree(tmp_path, procs) -> str` in `tests/conftest.py`. `procs` maps pid to a dict with keys `ppid` (int, default 1) and `devs` (list of device indices the process holds open, default empty). It writes `<root>/<pid>/status` (`Name:\tpython` and `PPid:\t<ppid>` lines), `<root>/<pid>/comm`, and one `fd/<n>` symlink to `/dev/tenstorrent/<dev>` per device. Returns the root path as a string.
- Produces: an autouse fixture `no_real_proc` in `tests/conftest.py` that makes `gozer.procfd.holders` raise `AssertionError` when called with `proc_root == "/proc"`.

- [ ] **Step 1: Write the guard and the helper**

Append to `tests/conftest.py`:

```python
def fake_proc_tree(tmp_path, procs):
    """Build a fake /proc. procs: {pid: {"ppid": int, "devs": [device_index, ...]}}.

    Writes what gozer reads: fd/ symlinks to /dev/tenstorrent/<N> (the device
    scan), comm (the process name) and status (the parent pid, for the
    ownership walk). A pid with no entry does not exist, which is how a test
    models a process that has exited.
    """
    root = tmp_path / "proc"
    root.mkdir(parents=True, exist_ok=True)
    for pid, spec in procs.items():
        d = root / str(pid)
        (d / "fd").mkdir(parents=True, exist_ok=True)
        (d / "comm").write_text("python\n")
        (d / "status").write_text(f"Name:\tpython\nPPid:\t{spec.get('ppid', 1)}\n")
        for i, dev in enumerate(spec.get("devs", [])):
            os.symlink(f"/dev/tenstorrent/{dev}", d / "fd" / str(i + 3))
    return str(root)


@pytest.fixture(autouse=True)
def no_real_proc(monkeypatch):
    """Fail a test that scans the real /proc.

    A Gatekeeper built without a proc_root scans the real /proc for open
    Tenstorrent devices. On a shared box other agents hold devices open, so
    the chips look busy and the test's own acquire queues instead of
    granting. The test then passes or fails depending on what else is running.
    Raising here turns that silent dependence into an immediate, named error.
    """
    from gozer import procfd
    real = procfd.holders

    def guarded(proc_root="/proc", *args, **kwargs):
        if proc_root == "/proc":
            raise AssertionError(
                "this test scans the real /proc; give the Gatekeeper a fake "
                "proc_root (see fake_proc_tree in conftest.py)")
        return real(proc_root, *args, **kwargs)

    monkeypatch.setattr(procfd, "holders", guarded)
```

- [ ] **Step 2: Run the suite and confirm the guard makes the hidden dependence loud**

Run: `cd ~/code/tt-gozer-orchard && python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -15`
Expected: `test_history_root.py` tests (and any others that use the real `/proc`) now fail with the `AssertionError` text above. Write down every failing test name in the report. If a test that legitimately needs the real `/proc` fails (for example a test of `procfd.holders` itself with the default argument), give it a fake root instead; do not weaken the guard.

- [ ] **Step 3: Fix each failing test by giving it a fake proc root**

In `tests/test_history_root.py`, add `proc_root=str(tmp_path / "proc")` to every `Gatekeeper(...)` call. A missing proc directory is read as "no process holds any device", which is what these tests mean. In every other failing test, add a fake proc root the same way (a `tmp_path / "proc"` path is enough unless the test needs a process).

- [ ] **Step 4: Run the suite and confirm it is green**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: `260 passed` (259 passing before plus the one that was failing). If the count differs, explain it in the report.

- [ ] **Step 5: Prove the guard can fail**

Temporarily remove `proc_root=...` from one `Gatekeeper(...)` call in `tests/test_history_root.py`, run `python3 -m pytest tests/test_history_root.py -q -p no:cacheprovider`, confirm the `AssertionError` message appears, restore the line.

- [ ] **Step 6: Commit**

```bash
cd ~/code/tt-gozer-orchard
git add tests/conftest.py tests/test_history_root.py   # plus any other test file edited in Step 3
git commit -m "Tests: stop reading the real /proc, so results do not depend on what else is running"
```

---

### Task 1: Descendants of the owner count as the owner

**Files:**
- Modify: `gozer/procfd.py`, `gozer/gatekeeper.py` (the `owned = ...` expression in `reconcile`), `skills/gozer-gatekeeper/SKILL.md`, `docs/superpowers/specs/2026-08-14-tt-gozer-design.md` (the `HELD-FOREIGN` row near line 346), `README.md` (every sentence that defines `HELD-FOREIGN`; find them with `grep -n HELD-FOREIGN README.md`)
- Test: `tests/test_procfd.py` (append), `tests/test_ownership.py` (create)

**Interfaces:**
- Produces: `procfd.parent_pid(pid: int, proc_root: str = "/proc") -> int | None` and `procfd.is_descendant(pid: int, ancestor: int, proc_root: str = "/proc", max_depth: int = 64) -> bool`.
- Consumes: `fake_proc_tree` from Task 0.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_procfd.py`:

```python
from conftest import fake_proc_tree
from gozer.procfd import is_descendant, parent_pid

OWNER = 5_000_100       # far above any real pid_max, so these never name a real process
CHILD = 5_000_200
GRANDCHILD = 5_000_300
STRANGER = 5_000_400


def test_parent_pid_reads_the_ppid_line(tmp_path):
    root = fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER}})
    assert parent_pid(CHILD, root) == OWNER


def test_parent_pid_is_none_for_a_missing_process(tmp_path):
    root = fake_proc_tree(tmp_path, {})
    assert parent_pid(CHILD, root) is None


def test_parent_pid_is_none_for_an_unreadable_or_malformed_status(tmp_path):
    root = fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER}})
    (tmp_path / "proc" / str(CHILD) / "status").write_text("Name:\tpython\nPPid:\tnot-a-number\n")
    assert parent_pid(CHILD, root) is None
    (tmp_path / "proc" / str(CHILD) / "status").write_text("Name:\tpython\n")
    assert parent_pid(CHILD, root) is None


def test_a_child_and_a_grandchild_descend_from_the_owner(tmp_path):
    root = fake_proc_tree(tmp_path, {
        OWNER: {"ppid": 1}, CHILD: {"ppid": OWNER}, GRANDCHILD: {"ppid": CHILD}})
    assert is_descendant(CHILD, OWNER, root) is True
    assert is_descendant(GRANDCHILD, OWNER, root) is True


def test_a_process_is_not_its_own_descendant(tmp_path):
    root = fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}})
    assert is_descendant(OWNER, OWNER, root) is False


def test_an_unrelated_process_does_not_descend_from_the_owner(tmp_path):
    root = fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}, STRANGER: {"ppid": 1}})
    assert is_descendant(STRANGER, OWNER, root) is False


def test_a_chain_that_ends_at_a_missing_process_is_not_a_descendant(tmp_path):
    # CHILD names a parent that no longer exists, so the walk cannot reach OWNER.
    root = fake_proc_tree(tmp_path, {CHILD: {"ppid": STRANGER}})
    assert is_descendant(CHILD, OWNER, root) is False


def test_a_cycle_in_the_parent_chain_terminates(tmp_path):
    root = fake_proc_tree(tmp_path, {CHILD: {"ppid": GRANDCHILD}, GRANDCHILD: {"ppid": CHILD}})
    assert is_descendant(CHILD, OWNER, root) is False


def test_the_walk_stops_at_max_depth(tmp_path):
    procs = {5_001_000 + i: {"ppid": 5_001_000 + i + 1} for i in range(10)}
    procs[5_001_010] = {"ppid": OWNER}
    procs[OWNER] = {"ppid": 1}
    root = fake_proc_tree(tmp_path, procs)
    assert is_descendant(5_001_000, OWNER, root, max_depth=20) is True
    assert is_descendant(5_001_000, OWNER, root, max_depth=3) is False
```

Create `tests/test_ownership.py`:

```python
"""`HELD` versus `HELD-FOREIGN` for a lease taken with an owner pid.

A supervisor that starts the model server as its own child holds the lease under
its pid, while the device is opened by the child. gozer must read that as the
owner's own work (`HELD`) and keep `HELD-FOREIGN` for a holder that has nothing
to do with the owner, because the gatekeeper skill tells agents to investigate
`HELD-FOREIGN` as a lease that is lying.
"""
from conftest import QUIETBOX, fake_proc_tree
from gozer.gatekeeper import Gatekeeper
from gozer.keymaster import Keymaster

OWNER = 5_000_100
CHILD = 5_000_200
GRANDCHILD = 5_000_300
STRANGER = 5_000_400


def board_with_lease(tmp_path, sysfs):
    """Take the lease first, with only the owner running.

    A process that already has a device open would make that board look busy,
    and acquire would put the lease on the other board. The tests add the
    device holders afterwards, with fake_proc_tree.
    """
    gk = Gatekeeper(root=str(tmp_path / "state"), sysfs_root=sysfs(QUIETBOX),
                    proc_root=fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}}))
    grant = Keymaster(gk).acquire("1", who="orchard:test", pid=OWNER)
    assert 0 in grant.dev_indices
    return gk


def state_of(gk, dev_index):
    return {s.chip.dev_index: s.state for s in gk.reconcile(reap=False)}[dev_index]


def test_a_child_of_the_owner_holding_the_device_is_held(tmp_path, sysfs):
    gk = board_with_lease(tmp_path, sysfs)
    fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER, "devs": [0]}})
    assert state_of(gk, 0) == "HELD"


def test_a_grandchild_of_the_owner_holding_the_device_is_held(tmp_path, sysfs):
    gk = board_with_lease(tmp_path, sysfs)
    fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER},
                              GRANDCHILD: {"ppid": CHILD, "devs": [0]}})
    assert state_of(gk, 0) == "HELD"


def test_an_unrelated_holder_is_still_held_foreign(tmp_path, sysfs):
    gk = board_with_lease(tmp_path, sysfs)
    fake_proc_tree(tmp_path, {STRANGER: {"ppid": 1, "devs": [0]}})
    assert state_of(gk, 0) == "HELD-FOREIGN"


def test_a_child_whose_owner_has_exited_is_held_foreign(tmp_path, sysfs):
    # The owner pid is gone and the server was re-parented to init. Someone
    # holds the device and nobody owns the lease, so the alarm must still fire.
    gk = board_with_lease(tmp_path, sysfs)
    fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER, "devs": [0]}})
    import shutil
    shutil.rmtree(tmp_path / "proc" / str(OWNER))
    (tmp_path / "proc" / str(CHILD) / "status").write_text("Name:\tpython\nPPid:\t1\n")
    assert state_of(gk, 0) == "HELD-FOREIGN"


def test_the_owner_pid_itself_holding_the_device_is_still_held(tmp_path, sysfs):
    gk = board_with_lease(tmp_path, sysfs)
    fake_proc_tree(tmp_path, {OWNER: {"ppid": 1, "devs": [0]}})
    assert state_of(gk, 0) == "HELD"


def test_a_detached_lease_never_reports_held_foreign(tmp_path, sysfs):
    # Unchanged behavior, pinned so the new rule cannot widen into it.
    gk = Gatekeeper(root=str(tmp_path / "state"), sysfs_root=sysfs(QUIETBOX),
                    proc_root=fake_proc_tree(tmp_path, {}))
    grant = Keymaster(gk).acquire("1", who="claude:test")   # no pid -> detached
    assert 0 in grant.dev_indices
    fake_proc_tree(tmp_path, {STRANGER: {"ppid": 1, "devs": [0]}})
    assert state_of(gk, 0) == "HELD"
```

- [ ] **Step 2: Run the tests and confirm the right ones fail**

Run: `python3 -m pytest tests/test_procfd.py tests/test_ownership.py -q -p no:cacheprovider 2>&1 | tail -15`
Expected: `ImportError: cannot import name 'is_descendant' from 'gozer.procfd'` (collection error) for test_procfd.py. After Step 3 adds the functions but before Step 4, the first two `test_ownership.py` tests (`child`, `grandchild`) fail with `assert 'HELD-FOREIGN' == 'HELD'`.

- [ ] **Step 3: Implement the walk**

Add to `gozer/procfd.py`, after `pid_alive`:

```python
def parent_pid(pid: int, proc_root: str = "/proc") -> int | None:
    """The parent of `pid`, read from the `PPid:` line of /proc/<pid>/status.

    status is parsed rather than /proc/<pid>/stat because stat's second field
    is the process name in parentheses, and a name can itself contain spaces
    and parentheses. Returns None when the process is gone, the file cannot be
    read, or the line is missing or malformed: the caller treats all of those
    as "no known parent".
    """
    try:
        with open(os.path.join(proc_root, str(pid), "status")) as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def is_descendant(pid: int, ancestor: int, proc_root: str = "/proc",
                  max_depth: int = 64) -> bool:
    """True when `ancestor` appears in the parent chain above `pid`.

    A process is not its own descendant. The walk stops at a missing process,
    at a repeated pid (a cycle), and after `max_depth` steps, so a malformed
    tree cannot make it loop.
    """
    seen = {pid}
    current = pid
    for _ in range(max_depth):
        parent = parent_pid(current, proc_root)
        if parent is None or parent <= 0 or parent in seen:
            return False
        if parent == ancestor:
            return True
        seen.add(parent)
        current = parent
    return False
```

- [ ] **Step 4: Use it in `reconcile`**

In `gozer/gatekeeper.py`, replace the line `owned = any(p == owner or p == owner_pgid for p in pids)` (inside the non-detached branch of `reconcile`) with:

```python
                    # A holder counts as the owner's own when it is the owner
                    # pid, the owner's process-group leader, or any descendant
                    # of the owner. The descendant case is a supervisor that
                    # starts the model server as its child: the lease is held
                    # under the supervisor's pid and the child opens the device.
                    owned = any(
                        p == owner or p == owner_pgid
                        or (owner is not None
                            and procfd.is_descendant(p, owner, self.proc_root))
                        for p in pids)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `python3 -m pytest tests/test_procfd.py tests/test_ownership.py -q -p no:cacheprovider 2>&1 | tail -4`, then `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all new tests pass; the full suite is green.

- [ ] **Step 6: Watch the rule fail without the new check**

Replace the `procfd.is_descendant(...)` term in `reconcile` with `False`, run `python3 -m pytest tests/test_ownership.py -q -p no:cacheprovider`, confirm `test_a_child_of_the_owner_holding_the_device_is_held` and `test_a_grandchild_...` fail with `HELD-FOREIGN`. Restore the term. Then delete the `seen` check in `is_descendant`, confirm the cycle test hangs or fails (use `timeout 20 python3 -m pytest tests/test_procfd.py::test_a_cycle_in_the_parent_chain_terminates -q -p no:cacheprovider`; with `max_depth` still in place it terminates but must still be seen to differ: if it passes, note that `max_depth` alone covers cycles and keep the `seen` check because it ends the walk sooner), restore.

- [ ] **Step 7: Update the documentation**

- `skills/gozer-gatekeeper/SKILL.md`, the `HELD-FOREIGN` bullet: say it means a device holder that is neither the lease owner nor a descendant of it.
- `docs/superpowers/specs/2026-08-14-tt-gozer-design.md`, the table row near line 346: change "by another pid" to "by a pid that is not the owner, the owner's process-group leader or a descendant of the owner".
- `README.md`: edit each sentence found with `grep -n HELD-FOREIGN README.md` the same way. Do not touch lines that do not define the state.

- [ ] **Step 8: Commit**

```bash
git add gozer/procfd.py gozer/gatekeeper.py tests/test_procfd.py tests/test_ownership.py skills/gozer-gatekeeper/SKILL.md docs/superpowers/specs/2026-08-14-tt-gozer-design.md README.md
git commit -m "A device held by a child of the lease owner counts as the owner's own, not HELD-FOREIGN"
```

---

### Task 2: `gozer reset <lease>`

**Files:**
- Modify: `gozer/keymaster.py`, `gozer/cli.py`, `README.md`, `skills/gozer-keymaster/SKILL.md`
- Test: `tests/test_reset_keymaster.py` (create), `tests/test_reset_cli.py` (create)

**Interfaces:**
- Consumes: `fake_proc_tree` (Task 0); `Keymaster._units_held_elsewhere`, `Keymaster.reset_runner`, `reset_mod.reset_chips`, `procfd.holders`, `history.log`, `Gatekeeper.critical_section` (all existing, read in `keymaster.py` `release`).
- Produces:
  - `@dataclass class ResetResult: ok: bool; status: str; message: str`, where `status` is one of `"reset"`, `"not-found"`, `"refused"`, `"failed"`, `"changed-hands"`.
  - `Keymaster.reset(lease_id: str, force: bool = False) -> ResetResult`.
  - CLI `gozer reset <lease> [--force] [--json]`; exit codes: `0` reset ok; `13` no such lease; `15` refused, nothing done; `17` (`EXIT_RESET_FAILED`) the reset ran and failed; `18` (`EXIT_RESET_CHANGED_HANDS`) the reset ran and the lease was then found gone or taken by another lease. JSON keys: `reset` (bool), `status`, `message`.
  - History events: `reset` with fields `lease_id`, `who`, `chips`, `reset_ok`; `refused` with `action="reset"` for a refusal.

- [ ] **Step 1: Write the failing Keymaster tests**

Create `tests/test_reset_keymaster.py`:

```python
"""Keymaster.reset: reset a held lease's chips and keep the lease.

It borrows the revalidation pattern of Keymaster.release. The gate is re-read
under the mutex before the reset and again after it, and the mutex is never held
across the reset itself (a reset takes tens of seconds).
"""
import json
import os

import pytest

from conftest import QUIETBOX, fake_proc_tree
from gozer import history, keymaster as keymaster_mod
from gozer.gatekeeper import Gatekeeper
from gozer.keymaster import Keymaster

OWNER = 5_000_100
CHILD = 5_000_200


class _Ok:
    returncode, stdout, stderr = 0, "reset ok", ""


class _Bad:
    returncode, stdout, stderr = 1, "", "tt-smi: link training failed"


def make(tmp_path, sysfs):
    """A gatekeeper over a fake /proc holding only the owner process.

    Tests that need a device holder add it after the lease is taken: a process
    that already has a device open makes its board look busy, and acquire would
    put the lease on the other board.
    """
    gk = Gatekeeper(root=str(tmp_path / "state"), sysfs_root=sysfs(QUIETBOX),
                    proc_root=fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}}))
    km = Keymaster(gk)
    calls = []
    km.reset_runner = lambda argv, **kw: calls.append(argv) or _Ok()
    return km, gk, calls


def events(gk, kind):
    path = os.path.join(gk.history_root, "history.jsonl")
    with open(path) as f:
        return [r for r in map(json.loads, f) if r["event"] == kind]


def test_reset_runs_tt_smi_on_exactly_the_leased_chips_and_keeps_the_lease(tmp_path, sysfs):
    km, gk, calls = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    result = km.reset(grant.lease_id)
    assert result.ok is True and result.status == "reset"
    assert len(calls) == 1 and calls[0][1] == "-r"
    assert set(calls[0][2].split(",")) == set(grant.bdfs)
    assert gk.read_lease(grant.lease_id) is not None
    assert gk.unit_lease(grant.units[0])["lease_id"] == grant.lease_id


def test_reset_does_not_mark_the_unit_clean(tmp_path, sysfs):
    # The lease continues, and release resets again; marking clean now would
    # let a later --fresh request trust silicon the workload may dirty first.
    km, gk, _ = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    km.reset(grant.lease_id)
    assert gk.is_clean(grant.units[0]) is False


def test_reset_leaves_the_queue_and_claim_window_alone(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    held = km.acquire("1", who="orchard:test", pid=OWNER)
    km.acquire("1", who="claude:other", pid=OWNER)          # takes the other board
    waiting = km.acquire("1", who="claude:waiting", pid=OWNER)   # queues
    assert isinstance(waiting, dict)
    km.reset(held.lease_id)
    assert [e["ticket"] for e in gk.queue_entries()] == [waiting["ticket"]]
    assert gk.expire_and_get_claim_holder() is None


def test_reset_logs_a_reset_event(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    km.reset(grant.lease_id)
    (rec,) = events(gk, "reset")
    assert rec["lease_id"] == grant.lease_id and rec["who"] == "orchard:test"
    assert rec["chips"] == grant.bdfs and rec["reset_ok"] is True


def test_reset_of_an_unknown_lease_is_not_found(tmp_path, sysfs):
    km, _, calls = make(tmp_path, sysfs)
    result = km.reset("nope")
    assert (result.ok, result.status) == (False, "not-found")
    assert "nope" in result.message and calls == []


def test_reset_refuses_while_a_device_is_open_and_runs_nothing(tmp_path, sysfs):
    km, gk, calls = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    assert 0 in grant.dev_indices
    fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER, "devs": [0]}})
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "refused")
    assert "still open" in result.message and str(CHILD) in result.message
    assert calls == []
    (rec,) = events(gk, "refused")
    assert rec["action"] == "reset" and rec["reason"] == "device still open"


def test_reset_force_overrides_an_open_device(tmp_path, sysfs):
    km, _, calls = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    assert 0 in grant.dev_indices
    fake_proc_tree(tmp_path, {CHILD: {"ppid": OWNER, "devs": [0]}})
    result = km.reset(grant.lease_id, force=True)
    assert result.ok is True and len(calls) == 1


def test_failed_reset_keeps_the_lease_and_reports_the_output(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    km.reset_runner = lambda argv, **kw: _Bad()
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "failed")
    assert "link training failed" in result.message
    assert gk.read_lease(grant.lease_id) is not None
    assert events(gk, "reset")[0]["reset_ok"] is False


def test_reset_never_touches_chips_that_now_belong_to_another_lease(tmp_path, sysfs, monkeypatch):
    """The release race, for reset: the lease is reaped and the unit re-claimed
    while the open-device scan runs. A reset here would hit the new tenant."""
    km, gk, calls = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:stale", pid=OWNER)
    unit = grant.units[0]

    def stealing_holders(proc_root, **kw):
        gk.release_unit(unit)
        gk.claim_unit(unit, {"lease_id": "newtenant", "who": "claude:b"})
        return {}

    monkeypatch.setattr(keymaster_mod.procfd, "holders", stealing_holders)
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "refused")
    assert "newtenant" in result.message
    assert calls == [], "reset ran against another lease's chips"
    assert gk.unit_lease(unit)["lease_id"] == "newtenant"


def test_reset_refuses_when_its_own_lock_is_already_gone(tmp_path, sysfs):
    km, gk, calls = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    gk.release_unit(grant.units[0])          # reaped out from under the lease
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "refused")
    assert "no longer locked" in result.message and calls == []


def test_reset_reports_changed_hands_when_the_lease_is_taken_during_the_reset(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    unit = grant.units[0]

    def runner(argv, **kw):
        gk.release_unit(unit)
        gk.claim_unit(unit, {"lease_id": "newtenant", "who": "claude:b"})
        return _Ok()

    km.reset_runner = runner
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "changed-hands")
    assert "newtenant" in result.message
    assert gk.unit_lease(unit)["lease_id"] == "newtenant"     # lock left alone


def test_reset_reports_changed_hands_when_the_lease_is_reaped_during_the_reset(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    grant = km.acquire("1", who="orchard:test", pid=OWNER)

    def runner(argv, **kw):
        gk.release_unit(grant.units[0])
        return _Ok()

    km.reset_runner = runner
    result = km.reset(grant.lease_id)
    assert (result.ok, result.status) == (False, "changed-hands")
    assert "reaped" in result.message


def test_reset_does_not_hold_the_mutex_across_the_reset(tmp_path, sysfs):
    km, gk, _ = make(tmp_path, sysfs)
    held_during_reset = []

    def watching_runner(argv, **kw):
        held_during_reset.append(os.path.isdir(os.path.join(gk.root, "mutex", ".gatekeeper.lock")))
        return _Ok()

    km.reset_runner = watching_runner
    grant = km.acquire("1", who="orchard:test", pid=OWNER)
    km.reset(grant.lease_id)
    assert held_during_reset == [False]
```

Create `tests/test_reset_cli.py`:

```python
"""`gozer reset` through the CLI: exit codes and JSON."""
import json
import os
import stat

import pytest

from conftest import QUIETBOX, fake_proc_tree
from gozer.cli import main

OWNER = 5_000_100
CHILD = 5_000_200


@pytest.fixture
def env(tmp_path, sysfs, monkeypatch):
    marker = tmp_path / "reset-ran"
    script = tmp_path / "reset.sh"
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {marker}\nexit ${{FAKE_RESET_EXIT:-0}}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", sysfs(QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}}))
    monkeypatch.setenv("GOZER_RESET_CMD", str(script))
    return tmp_path, marker


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


def take_lease(capsys):
    code, out = run(["acquire", "--chips", "1", "--who", "orchard:test",
                     "--owner-pid", str(OWNER), "--json"], capsys)
    assert code == 0
    return json.loads(out)["lease_id"]


def test_reset_exits_0_resets_and_keeps_the_lease(env, capsys):
    _, marker = env
    lease = take_lease(capsys)
    code, out = run(["reset", lease, "--json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["reset"] is True and payload["status"] == "reset"
    assert marker.exists()
    code, out = run(["status", "--json"], capsys)
    assert lease in out                               # still listed as held


def test_reset_of_an_unknown_lease_exits_13(env, capsys):
    code, out = run(["reset", "nope", "--json"], capsys)
    assert code == 13 and json.loads(out)["status"] == "not-found"


def test_reset_with_a_device_open_exits_15_and_runs_nothing(env, capsys, monkeypatch):
    tmp_path, marker = env
    lease = take_lease(capsys)
    monkeypatch.setenv("GOZER_PROC_ROOT", fake_proc_tree(
        tmp_path, {OWNER: {"ppid": 1}, CHILD: {"ppid": OWNER, "devs": [0]}}))
    code, out = run(["reset", lease, "--json"], capsys)
    assert code == 15 and json.loads(out)["status"] == "refused"
    assert not marker.exists()


def test_reset_force_with_a_device_open_exits_0(env, capsys, monkeypatch):
    tmp_path, marker = env
    lease = take_lease(capsys)
    monkeypatch.setenv("GOZER_PROC_ROOT", fake_proc_tree(
        tmp_path, {OWNER: {"ppid": 1}, CHILD: {"ppid": OWNER, "devs": [0]}}))
    code, _ = run(["reset", lease, "--force"], capsys)
    assert code == 0 and marker.exists()


def test_a_failed_reset_exits_17_and_keeps_the_lease(env, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_RESET_EXIT", "3")
    lease = take_lease(capsys)
    code, out = run(["reset", lease, "--json"], capsys)
    assert code == 17 and json.loads(out)["status"] == "failed"
    code, out = run(["status", "--json"], capsys)
    assert lease in out


def test_exit_codes_17_and_18_are_distinct_from_the_existing_ones():
    from gozer import cli
    codes = [cli.EXIT_OK, cli.EXIT_QUEUED, cli.EXIT_WAIT_TIMEOUT, cli.EXIT_UNAVAILABLE,
             cli.EXIT_NO_LEASE, cli.EXIT_NO_TOPOLOGY, cli.EXIT_RELEASE_REFUSED,
             cli.EXIT_GATE_STUCK, cli.EXIT_RESET_FAILED, cli.EXIT_RESET_CHANGED_HANDS,
             cli.EXIT_INTERRUPTED]
    assert len(codes) == len(set(codes))
    assert (cli.EXIT_RESET_FAILED, cli.EXIT_RESET_CHANGED_HANDS) == (17, 18)


def test_a_reset_event_does_not_disturb_the_report_statistics():
    from gozer import report
    base = [
        {"ts": "2026-10-02T10:00:00Z", "event": "granted", "lease_id": "aaa",
         "who": "orchard:test", "chips": ["0000:01:00.0"]},
        {"ts": "2026-10-02T11:00:00Z", "event": "released", "lease_id": "aaa",
         "who": "orchard:test", "chips": ["0000:01:00.0"], "duration_s": 3600.0},
    ]
    reset = {"ts": "2026-10-02T10:30:00Z", "event": "reset", "lease_id": "aaa",
             "who": "orchard:test", "chips": ["0000:01:00.0"], "reset_ok": True}
    assert report.compute_stats(base[:1] + [reset] + base[1:]) == report.compute_stats(base)
```

- [ ] **Step 2: Run the tests and confirm they fail for the right reason**

Run: `python3 -m pytest tests/test_reset_keymaster.py tests/test_reset_cli.py -q -p no:cacheprovider 2>&1 | tail -15`
Expected: `AttributeError: 'Keymaster' object has no attribute 'reset'` for the keymaster tests, and `argument command: invalid choice: 'reset'` or `SystemExit: 2` for the CLI tests. The `report.compute_stats` test passes already, since `compute_stats` ignores unknown events; it is a guard for later.

- [ ] **Step 3: Implement `Keymaster.reset`**

In `gozer/keymaster.py`, add `from dataclasses import dataclass` if it is not already imported (check the existing imports; `Grant` is already a dataclass), then add after `Grant`:

```python
@dataclass
class ResetResult:
    """Outcome of Keymaster.reset. `status` is one of: "reset" (ok),
    "not-found", "refused" (nothing was done), "failed" (tt-smi ran and
    failed) and "changed-hands" (the reset ran, and afterwards the lease was
    found gone or taken by another lease)."""
    ok: bool
    status: str
    message: str
```

and add this method to `Keymaster`, after `release`:

```python
    def reset(self, lease_id: str, force: bool = False) -> ResetResult:
        """Reset a held lease's chips and keep the lease.

        Used by a supervisor that stops one model server, runs something else
        on the same board, and starts a server again, without ever letting go
        of the board. Releasing in between would let another agent take it.

        The structure follows `release`: refuse while a device is open; re-read
        the gate under the mutex before the reset so it never fires at chips
        another tenant now holds; drop the mutex for the reset itself (tt-smi
        takes tens of seconds); re-read the gate once more afterwards. It does
        not mark the unit clean, touch the queue, or open a claim window,
        because the lease continues and `release` resets again.
        """
        lease = self.gk.read_lease(lease_id)
        if lease is None:
            return ResetResult(False, "not-found", f"lease {lease_id} not found")
        units = lease.get("units", [])

        fd_map = procfd.holders(self.gk.proc_root)
        still_open = [d for d in lease.get("dev_indices", []) if fd_map.get(d)]
        if still_open and not force:
            history.log(self.gk.history_root, "refused", action="reset",
                       lease_id=lease_id, who=lease.get("who"),
                       reason="device still open", dev_indices=still_open)
            return ResetResult(False, "refused", (
                f"chips {still_open} still open by "
                f"{sorted({p for d in still_open for p in fd_map[d]})} -- "
                "stop the workload first, or pass --force"))

        with self.gk.critical_section():
            foreign = self._units_held_elsewhere(units, lease_id)
            unheld = [u for u in units if self.gk.unit_lease(u) is None]
        if foreign:
            history.log(self.gk.history_root, "refused", action="reset",
                       lease_id=lease_id, who=lease.get("who"),
                       reason="units now belong to another lease", foreign=foreign)
            return ResetResult(False, "refused", (
                "refusing to reset: " +
                ", ".join(f"{unit} now belongs to lease {other}"
                          for unit, other in sorted(foreign.items())) +
                f" -- lease {lease_id} was already reaped or released; nothing was reset"))
        if unheld:
            # Our lock is gone and nobody holds the unit. A reset now would land
            # on whoever takes these chips next, so do nothing.
            history.log(self.gk.history_root, "refused", action="reset",
                       lease_id=lease_id, who=lease.get("who"),
                       reason="unit no longer locked by this lease", unheld=unheld)
            return ResetResult(False, "refused", (
                f"refusing to reset: {', '.join(sorted(unheld))} no longer locked "
                f"by lease {lease_id} (already reaped?); nothing was reset"))

        reset_ok, out = reset_mod.reset_chips(lease["chips"], runner=self.reset_runner)
        message = out or ("reset ok" if reset_ok else "reset failed")

        with self.gk.critical_section():
            foreign = self._units_held_elsewhere(units, lease_id)
            unheld = [u for u in units if self.gk.unit_lease(u) is None]
        history.log(self.gk.history_root, "reset", lease_id=lease_id,
                   who=lease.get("who"), chips=lease.get("chips", []),
                   reset_ok=reset_ok)
        if foreign:
            return ResetResult(False, "changed-hands", "; ".join([message,
                "discovered after the reset: " +
                ", ".join(f"{unit} now belongs to lease {other}"
                          for unit, other in sorted(foreign.items())) +
                " -- leaving the new tenant's lock alone"]))
        if unheld:
            return ResetResult(False, "changed-hands", "; ".join([message,
                f"discovered after the reset: lease {lease_id} was reaped while "
                "the reset ran"]))
        return ResetResult(reset_ok, "reset" if reset_ok else "failed", message)
```

- [ ] **Step 4: Implement the CLI**

In `gozer/cli.py`, add the constants after `EXIT_GATE_STUCK`:

```python
# `gozer reset` ran tt-smi and it failed. The lease is untouched.
EXIT_RESET_FAILED = 17
# `gozer reset` ran, and afterwards the lease was gone or another lease held the
# unit. The reset may have hit chips that were no longer ours. Distinct from 15,
# which means nothing was done.
EXIT_RESET_CHANGED_HANDS = 18
```

add the command after `cmd_release`:

```python
def cmd_reset(args) -> int:
    _, km = _make(args)
    result = km.reset(args.lease, force=args.force)
    _emit({"reset": result.ok, "status": result.status, "message": result.message},
          result.message, args.json)
    if result.ok:
        return EXIT_OK
    if result.status == "not-found":
        return EXIT_NO_LEASE
    if result.status == "failed":
        return EXIT_RESET_FAILED
    if result.status == "changed-hands":
        return EXIT_RESET_CHANGED_HANDS
    return EXIT_RELEASE_REFUSED     # "refused": nothing was done
```

and the parser entry, directly after the `release`/`banish` block and before `rec = add("reconcile", ...)`:

```python
    rs = add("reset", cmd_reset,
             "reset a held lease's chips in place; the lease stays held")
    rs.add_argument("lease")
    rs.add_argument("--force", action="store_true",
                    help="reset even if a device is still open")
```

- [ ] **Step 5: Run the tests and the whole suite**

Run: `python3 -m pytest tests/test_reset_keymaster.py tests/test_reset_cli.py -q -p no:cacheprovider 2>&1 | tail -4`, then `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all new tests pass; the suite is green.

- [ ] **Step 6: Mutation check on the new guards**

With `python -B` and `__pycache__` removed before each run, apply each mutation to `Keymaster.reset` one at a time, run `tests/test_reset_keymaster.py tests/test_reset_cli.py`, confirm a test fails for the right reason, restore. Mutations: skip the open-device refusal; skip the first revalidation (`foreign`); skip the `unheld` refusal; skip the second revalidation; call `self.gk.mark_clean(...)` after a successful reset; hold `critical_section` across the reset; return `ok=True` when `reset_ok` is False; map status `failed` to exit 15; map `changed-hands` to exit 15. Put the table in the report. Any survivor gets a test.

- [ ] **Step 7: Documentation**

- `README.md`: add a row for `reset <lease>` [`--force`] to the commands table, next to the `release` row (`grep -n "release" README.md` shows it near line 121), and add rows for exit codes `17` and `18` to the exit-code table (near line 164). Add `reset` to the list of history events if one exists (near line 274).
- `skills/gozer-keymaster/SKILL.md`: add a short section after the `--owner-pid` section titled "Resetting without releasing". Say: `gozer reset <lease-id>` resets the board's chips and keeps the lease; it refuses while any device is open; `--force` overrides that and should be used only when a person says so; see `gozer-park` for the supervisor pattern.
- `gozer/history.py`: find where the module or `log` docstring lists event types (`grep -n "released" gozer/history.py`) and add `reset` with its fields; if no list exists, add nothing.

- [ ] **Step 8: Commit**

```bash
git add gozer/keymaster.py gozer/cli.py gozer/history.py tests/test_reset_keymaster.py tests/test_reset_cli.py README.md skills/gozer-keymaster/SKILL.md
git commit -m "Add gozer reset: reset a held lease's chips in place and keep the lease"
```

(Leave `gozer/history.py` out of `git add` if Step 7 did not change it.)

---

### Task 3: The `gozer-park` skill

**Files:**
- Create: `skills/gozer-park/SKILL.md`
- Modify: `install.sh`, `tests/test_install.py`, `README.md` (the skills list near line 150)

**Interfaces:**
- Consumes: the commands from Tasks 1 and 2.

- [ ] **Step 1: Write the failing test**

In `tests/test_install.py`, extend `test_installer_links_cli_and_skills` with this assertion after the `gozer-gatekeeper` line:

```python
    assert os.path.islink(home / ".claude" / "skills" / "gozer-park")
```

and add:

```python
def test_gozer_park_skill_exists_and_names_the_commands_it_teaches():
    text = open(os.path.join(REPO, "skills", "gozer-park", "SKILL.md")).read()
    assert text.startswith("---\nname: gozer-park\n")
    for needle in ("--owner-pid", "gozer reset", "gozer env", "HELD-FOREIGN",
                   "gozer release"):
        assert needle in text, needle
```

- [ ] **Step 2: Run and confirm failure**

Run: `python3 -m pytest tests/test_install.py -q -p no:cacheprovider 2>&1 | tail -6`
Expected: `FileNotFoundError` for `skills/gozer-park/SKILL.md`, and the link assertion fails.

- [ ] **Step 3: Write the skill**

Create `skills/gozer-park/SKILL.md`:

````markdown
---
name: gozer-park
description: Use when a supervisor must stop the model server that is using a board, run something else on the same board, and bring the server back - holds the lease through the swap, resets the chips in place, and says when to release instead.
---

# Swapping what runs on a board without letting go of it

A model server holds a board. You need that board for something else, such as a test or a
different model, and you want the first server back afterwards. If you `gozer release` in
between, another agent can take the board in the gap, and your restart queues behind them.
Keep the lease instead.

## Take the lease under your own pid

```bash
gozer acquire --chips 1 --who "orchard:<run-id>" --reason "<what the run is doing>" \
    --owner-pid $$
eval "$(gozer env <lease-id>)"
```

`$$` must be the pid of a process that stays alive for the whole run, such as the supervisor.
A short-lived shell is the wrong owner, because gozer reaps the lease once that pid is gone and
no device is open. A lease owned by a live pid stays `CLAIMED` while nothing has the device
open. The 15 minute grace window of a bare `acquire` does not apply to it.

## Start servers and tests from that process

Start every server and every test from the owner process, with the exports from `gozer env`.
Do not wrap them in `gozer run`: it takes its own lease and releases it when the command ends.

gozer reads a device holder that is the owner pid, or any descendant of it, as the owner's own
work (`HELD`). A holder that is neither is `HELD-FOREIGN`. Treat `HELD-FOREIGN` on your own
board as a real alarm: something outside your process tree has the device open.

## The swap

1. Stop the first server cleanly. For tt-model that is `tt-model stop`. Run `gozer status`
   until your chips are no longer `HELD`.
2. Run `gozer reset <lease-id>`. It resets the board's chips and keeps the lease. It refuses
   while any device is open. Do not pass `--force` unless a person tells you to.
3. Run the other work with the same exports.
4. When it ends, run `gozer reset <lease-id>` again. A failed or hung run can leave the chips
   dirty.
5. Start the first server again with the same exports.

## Exit codes of `gozer reset`

| Code | Meaning | What to do |
|---|---|---|
| `0` | reset ran and succeeded | continue |
| `13` | no such lease | the lease was released or reaped; take a new one |
| `15` | refused, nothing was done | a device is open (stop the workload) or the unit now belongs to another lease |
| `17` | the reset ran and failed | the lease is still yours; read the message, try once more, then stop and tell a person |
| `18` | the reset ran and the lease was then gone or taken | do not start a server; run `gozer status` and take a new lease |

## Release instead for a long phase with no hardware

The board stays unavailable to other agents for as long as you hold it. If the next phase does
not use the chips for a long time, such as an image build or a CPU-only reference run, run
`gozer release <lease-id>` and take a new lease when the hardware phase begins. Expect to queue.

## If the supervisor dies

gozer reaps the lease once the owner pid is dead and no device is open. A server that is still
running keeps its device open, so the lease stays and `gozer status` shows `HELD-FOREIGN`.
Do not use `--force` to clear that. Stop the server first, then run `gozer reconcile`.
````

- [ ] **Step 4: Install the skill**

In `install.sh`, change the loop line to `for skill in gozer-keymaster gozer-gatekeeper gozer-park; do`, update the comment on line 2 ("the two skills" becomes "the three skills"), and in `README.md` add a line for `gozer-park` to the skills list next to the existing `gozer-keymaster` bullet (near line 150), saying: how to hold a board through a model swap with `--owner-pid` and `gozer reset`.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_install.py -q -p no:cacheprovider 2>&1 | tail -4`, then the full suite.
Expected: green.

- [ ] **Step 6: Sweep the skill text**

Run: `grep -nE ", not |rather than|instead of|reads as|honest|the one |which is why|This is what" skills/gozer-park/SKILL.md`. Rewrite any hit as a plain statement.

- [ ] **Step 7: Commit**

```bash
git add skills/gozer-park install.sh tests/test_install.py README.md
git commit -m "Add the gozer-park skill: hold a board lease through a model swap"
```

---

### Task 4: The hold-through-swap sequence, end to end

**Files:**
- Create: `tests/test_hold_through_swap.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above, through the CLI.

- [ ] **Step 1: Write the tests**

Create `tests/test_hold_through_swap.py`:

```python
"""The supervisor's sequence, as the CLI sees it.

A supervisor takes a lease under its own pid, runs a model server as its child,
stops it, resets the board in place, runs something else, resets again, starts
the server again, and finally releases. The lease must survive every step,
including a backdated `since` while no device is open, which is the case that
reaps a lease taken without an owner pid.
"""
import json
import os
import shutil
import stat

import pytest

from conftest import QUIETBOX, fake_proc_tree
from gozer.cli import main
from gozer.gatekeeper import Gatekeeper

OWNER = 5_000_100
SERVER = 5_000_200
TEST_RUN = 5_000_300


@pytest.fixture
def env(tmp_path, sysfs, monkeypatch):
    marker = tmp_path / "resets"
    script = tmp_path / "reset.sh"
    script.write_text(f"#!/bin/sh\necho reset >> {marker}\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("GOZER_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv("GOZER_SYSFS_ROOT", sysfs(QUIETBOX))
    monkeypatch.setenv("GOZER_PROC_ROOT", fake_proc_tree(tmp_path, {OWNER: {"ppid": 1}}))
    monkeypatch.setenv("GOZER_RESET_CMD", str(script))
    return tmp_path, marker


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


def states(capsys):
    code, out = run(["reconcile", "--json"], capsys)
    assert code == 0
    return {c["dev_index"]: c["state"] for c in json.loads(out)["chips"]}


def start_process(tmp_path, pid, ppid, dev=0):
    fake_proc_tree(tmp_path, {pid: {"ppid": ppid, "devs": [dev]}})


def stop_process(tmp_path, pid):
    shutil.rmtree(tmp_path / "proc" / str(pid))


def backdate(lease_id, since="2000-01-01T00:00:00Z"):
    gk = Gatekeeper()
    lease = gk.read_lease(lease_id)
    lease["since"] = since
    gk.write_lease(lease)
    for unit in lease["units"]:
        gk.update_unit_lease(unit, lease)


def test_the_lease_survives_a_full_swap_and_releases_at_the_end(env, capsys):
    tmp_path, marker = env
    code, out = run(["acquire", "--chips", "1", "--who", "orchard:test",
                     "--owner-pid", str(OWNER), "--json"], capsys)
    assert code == 0
    lease_id = json.loads(out)["lease_id"]

    start_process(tmp_path, SERVER, OWNER)                 # the coder starts as the owner's child
    assert states(capsys)[0] == "HELD"

    code, _ = run(["reset", lease_id], capsys)             # refused: the coder still has the device
    assert code == 15 and not marker.exists()

    stop_process(tmp_path, SERVER)                         # the coder is stopped
    backdate(lease_id)                                     # long past the 900 s detached window
    assert states(capsys)[0] == "CLAIMED"                  # no device open, and the lease is still held
    assert Gatekeeper().read_lease(lease_id) is not None

    code, _ = run(["reset", lease_id], capsys)             # reset in place
    assert code == 0 and marker.read_text().count("reset") == 1
    assert Gatekeeper().read_lease(lease_id) is not None

    start_process(tmp_path, TEST_RUN, OWNER)               # the test runs on the same lease
    assert states(capsys)[0] == "HELD"
    stop_process(tmp_path, TEST_RUN)

    code, _ = run(["reset", lease_id], capsys)             # clean the chips after the test
    assert code == 0 and marker.read_text().count("reset") == 2

    start_process(tmp_path, SERVER, OWNER)                 # the coder comes back
    assert states(capsys)[0] == "HELD"
    stop_process(tmp_path, SERVER)

    code, _ = run(["release", lease_id], capsys)
    assert code == 0
    assert Gatekeeper().read_lease(lease_id) is None
    assert marker.read_text().count("reset") == 3          # release resets too


def test_control_a_lease_without_an_owner_pid_is_reaped_in_the_same_gap(env, capsys):
    """Why the owner pid matters: the same swap, with a bare acquire, loses the
    lease as soon as a reconcile runs during the gap with no device open."""
    tmp_path, _ = env
    code, out = run(["acquire", "--chips", "1", "--who", "orchard:test", "--json"], capsys)
    assert code == 0
    lease_id = json.loads(out)["lease_id"]
    backdate(lease_id)
    assert states(capsys)[0] == "FREE"
    assert Gatekeeper().read_lease(lease_id) is None


def test_the_alarm_still_fires_for_a_holder_outside_the_owners_tree(env, capsys):
    tmp_path, _ = env
    code, out = run(["acquire", "--chips", "1", "--who", "orchard:test",
                     "--owner-pid", str(OWNER), "--json"], capsys)
    assert code == 0
    start_process(tmp_path, 5_000_900, 1)                  # not a descendant of the owner
    assert states(capsys)[0] == "HELD-FOREIGN"
```

- [ ] **Step 2: Run the tests**

Run: `python3 -m pytest tests/test_hold_through_swap.py -q -p no:cacheprovider 2>&1 | tail -8`
Expected: all three pass (Tasks 1 and 2 already provide everything). If the first fails at a step, read the failure and fix the code or the test; say in the report which one was wrong.

- [ ] **Step 3: Make sure the first test can fail**

Temporarily replace the descendant check in `reconcile` with `False`, rerun, and confirm the first test fails at the first `HELD` assertion and the third test still passes. Restore. Then comment out the `--owner-pid` argument in the first test's `acquire` call, rerun, and confirm the test fails at the `CLAIMED` assertion (the lease was reaped). Restore.

- [ ] **Step 4: Log it in `CLAUDE.md`**

Append to the "What happened" section of `CLAUDE.md` in the worktree a short entry dated 2026-10-02: tt-orchard (the supervisor in `~/code/tt-orchard`) needs to hold a board through a model swap; gozer gained `reset <lease>`, descendant-aware ownership, and the `gozer-park` skill; the work lives on branch `orchard-hold-through-swap` in a worktree; the test suite was made independent of the real `/proc`; the version was not bumped (the live checkout has an uncommitted bump to 0.3.3, so the owner bumps at merge). Keep it to six sentences.

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q -p no:cacheprovider 2>&1 | tail -3`
Expected: all green, with the count stated in the report.

- [ ] **Step 6: Commit**

```bash
git add tests/test_hold_through_swap.py CLAUDE.md
git commit -m "Test the hold-through-swap sequence end to end, with a control that shows why the owner pid matters"
```

---

## Self-review

- **Spec coverage.** Section 8 item 1 (`reset`) is Task 2. Item 2 (descendant ownership) is Task 1. Item 3 (JSON) is covered by Task 2's CLI tests. Item 4 (known limits) is documented in the skill (Task 3). Section 11 (`gozer-park`) is Task 3. Section 13's concurrency test for reset against release is Task 2's race tests. Task 0 is an addition: the suite could not be trusted on this box without it.
- **Not covered here.** The behavior on the real box is not tested (no hardware in this plan). Spec section 14 item 2 says plan 3 checks it under a lease before anything depends on it.
- **Placeholders.** None. Every test and code block is complete. Two steps tell the implementer to find a line by grep (`README.md` rows, `history.py` event list) because those files have uncommitted siblings in the live checkout and the exact line numbers in the worktree matter.
- **Type consistency.** `ResetResult.status` strings match between `Keymaster.reset`, `cmd_reset` and the tests. `fake_proc_tree` is defined in Task 0 and used by Tasks 1, 2 and 4.
- **Not executed.** The code in this plan has not been run. The parts most likely to need a fix are the `test_reset_leaves_the_queue_and_claim_window_alone` setup (it relies on QUIETBOX having exactly two boards, so the third `acquire` queues) and the `Gatekeeper()` default construction in `test_hold_through_swap.py` (it relies on `GOZER_ROOT` and the fake sysfs coming from the environment, as `tests/test_owner_pid.py` already does).
