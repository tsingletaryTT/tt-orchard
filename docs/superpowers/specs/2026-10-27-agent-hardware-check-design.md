# An agent-callable hardware check in the kernel probe (design note, 2026-10-27)

Branch `agent-hw-check`, on `probe-hw-sim-gap` (pull request 112). Version 0.9.13.

## Decision

The operator's decision: a hardware check is always available to the agent during a kernel-probe attempt. If hardware is the
right choice for the next step, the agent can make that choice. Before this change the probe ran the simulator for every kernel
and ran hardware only after the attempt ended.

The agent cannot call gozer. The command runner refuses gozer writes and the agent's shell has a no-chip device mask. So the
probe offers the check as a tool of the agent loop, next to `shell`, `read_file` and `write_file`. The probe owns the lease.

## The tool

`hardware_check` takes one optional argument, `note` (a string of at most 200 characters, kept in the record). Any other
argument is an error and nothing runs. The agent supplies no command and no path.

Each call does what the probe's own hardware phase does, on the attempt's current `kernel.py`:

1. `copy_kernel` copies the file into a fresh probe-owned directory, `DIR/.hwcheck/<op>-<attempt>-<call>/` (a regular file of at most
   256 KiB; a link, a directory or a larger file is `forbidden_code`, a missing file is `no_kernel`). `.hwcheck` is not under
   `.verdict`, because the authoritative check empties its directory and `load_attempts` reads it.
2. `_hw_gate` runs the static gate on the copy and compares its sha256 with the sha256 taken at copy time.
3. `_hw_run` runs the shipped `kernel_harness.py` with the shipped spec through `gozer run --chips 1 --exact 1` (the probe's own
   chip selection), under the stall watcher, with the probe's hardware environment, and judges the result with the hardware bars
   (`kernel_check.classify`).

Steps 2 and 3 are the code that `hardware_phase` used before, moved into two functions that both paths call. `hardware_phase`
behaves as before: a hang or a queued request still ends the probe. The gate and the sha256 check run again before every gozer
try of a call that waits for a board, because the agent can write anywhere the operator can.

The answer is bounded text: PASS or FAIL with the category and first line, the numbers line for a numeric failure, the last
lines of the output for a compile error or a harness error, the seconds in all and the device seconds, the time waited for a
lease, and the budget (calls left, lease seconds left). The harness now writes `device_s`, the seconds from opening the device to
just before closing it. It covers every input set, the reference output on the CPU and the compile. It is not a benchmark of the
kernel.

## Policy

| Setting | Default | Flag |
|---|---|---|
| calls per attempt | 6 | `--hw-checks N` (0 turns the tool off), `--no-agent-hw` (the same as 0) |
| chip seconds per attempt | 600 (120 reserved for the stop and the reset) | `--hw-lease-s S` |
| one run's hard timeout | 240 s | `--hw-call-timeout-s S` |
| wait for a busy board | 120 s | `--hw-wait-s S` |
| run seconds that must be left for a call, after the 120 s cleanup reserve | 60 | not a flag |

The numbers are choices. No population of attempts was measured.

- A call that waits for a board counts as a call. A call that the probe refused before it reached the lease (no kernel, a rule of
  the gate broken) does not count and uses no chip.
- A run is stopped at the lesser of the call timeout and the lease seconds left. The stall watcher usually stops a hang sooner
  (60 s of silence). A hung run is stopped by SIGTERM to the process group first, and gozer's lease release resets the chip. After a
  hang the tool is off for the rest of the attempt.
- One call at a time. A second call while one runs is answered at once and starts nothing.
- A busy board: the probe asks with `gozer acquire --owner-pid <its pid>`. When no board is free gozer queues one ticket and the probe
  claims that same ticket every 2 s (`gozer acquire --ticket`) for at most `--hw-wait-s` seconds, so the call keeps its place in the
  queue and never asks again in a loop. A grant runs the shipped harness under the lease's `TT_VISIBLE_DEVICES` and the probe releases
  the lease in a `finally`. When the wait runs out the ticket is cancelled and the agent is told the board is busy and to go on with
  the simulator. gozer owns the lease by the probe's pid, so gozer reaps it if the probe dies. See the comparison near the end of
  this note.
- The lease is taken per call and released after it. The release resets the chip (about 40 s).

### Hold mode

An option that keeps the lease for the whole attempt would save the 40 s reset on each call after the first. It was not built.
It needs `gozer acquire` with an owner pid, a way to set `TT_VISIBLE_DEVICES` for the child, and a release on every exit path,
including a crash of the probe, and it keeps a board away from every other client for the attempt's length (5 to 15 minutes). The
default must not hold a board, and a hold mode that is only safe on a one-user box is not worth the code.

## Records and report

Each `attempt.json` has `agent_hw`: `enabled`, the policy, `calls`, `calls_counted`, `bad_calls` and `lease_s_used`. A call
record has the time, the call number, whether it counted, the outcome (`pass`, a failure category, `busy`, `hang`, `refused` or
`interrupted`), the total seconds, the device seconds, the chip seconds, the seconds waited for a lease, the number of tries,
whether every ticket was cancelled, whether it timed out, the note, the first line and the kernel's sha256. An attempt run without
the tool records `{"enabled": false}`. These calls are not the attempt's hardware verdict (`hw`).

The report has a column `agent hw` in the attempt table (calls, passes, chip seconds) and a section "Hardware checks called by the
coder" with the policy, every call, and a totals line that gives the chip time (the seconds gozer held a lease, reset included) and
the seconds waited for a board.

## What the agent is told

`orchard/skills/kernel-probe.md` (100 lines, one fenced block) names the tool, its cost (about 3 seconds to run, about 45 with the
lease and the reset), when it is worth using (a kernel the simulator cannot run, a numeric check at hardware precision after the
simulator passes, a performance question) and that the simulator is much cheaper for everything else. The tool's description states
the same and this attempt's limits.

## Measured on hardware

One chip, the library kernel `orchard/skills/kernel-loop-templates/library/silu_mul.py` and two edits of it, through the tool's own
path (the real gozer, the real harness, the host tt-metal v0.80.0 and the shared virtual environment). The board was free, so no
queue wait was needed first; the tool took its own lease for each call. 2026-10-27.

| Call | Kernel | Answer | Seconds in all | Device seconds |
|---|---|---|---|---|
| 1 | library `silu_mul` | PASS, pcc_min 0.999996, 6 input sets | 47.5 | 0.5 |
| 2 | `*` turned into `+` | FAIL hw_numeric_mismatch, pcc_min 0.128, "got is about 0.499x expected", worst places shown | 47.6 | 0.6 |
| 3 | unknown function `ttl.math.not_a_real_function` | FAIL compile_error, the compiler's line and caret shown | 5.2 | 0.0 |
| 4 (second session) | the same compile error | FAIL compile_error | 46.9 | not reached |
| 5 | library `silu_mul` after call 4 | PASS | 47.4 | 0.5 |
| 6 | board held by another lease (20 s wait; measured with an earlier version of the wait) | BUSY after 20 s, tickets cancelled, queue empty | 20.1 | none |

- A call takes about 47 s in all. The kernel itself takes under a second with the device open. The rest is `gozer run`: the lease,
  the interpreter start, the compile and the reset that the release makes. The skill's "about 3 seconds to run" is from the earlier
  measurement in the project log (about 3 s for a run); here the part with the device open was 0.5 s.
- Call 3 took 5.2 s and `gozer run` printed `chips [0, 1] still open by [pid] -- let the work finish, or pass --force`: the release
  was refused because the harness process still held the device when gozer looked, so the chip was not reset and the lease was left
  as STALE under a dead pid. I cleared it with `gozer reconcile` and took one more lease to reset the chip. The same command line
  in a second session (call 4) took 46.9 s and released normally. So this is a race in gozer's release check and not a rule of
  the compile error path. The tool does not handle it: the next call's `gozer run` reaps a stale lease of the dead pid. The
  probe's own hardware phase has the same exposure.
- The busy path worked against the real gozer: the script held the board with a lease, the tool tried 5 times, cancelled every
  ticket and answered BUSY; `gozer queue` was empty afterwards. (My script read the wrong key of `gozer acquire --json`, so it did not
  release its lease at the end; I released it by its id. It was the script's own lease.)
- The hang and the timeout paths were tested with fakes only. No hang was provoked on the chip.

## Not done

- The policy numbers are not tuned. Whether a coder uses the tool well, and whether it changes the pass counts, is not known: no
  coder round was run.
- A kernel that hangs the device is stopped by the watcher and the lease release resets the chip, but a hang that the watcher
  cannot see (a process that keeps writing) runs until the call timeout.
- The probe's own end-of-attempt hardware phase (`--phase hw`) is unchanged and still needs a simulator pass or a `sim_gap`.

## Can the probe lean on gozer instead of tracking its own state? (round 4, 2026-10-27)

Three review rounds in a row found a new hole in the same place: a lease or ticket that survives an exception, an interrupt or a
timeout. Each time the bookkeeping module grew (an in-memory ledger of tickets and leases, a token in the request's reason to find
a grant in history, a record of overrunning releases, a plan to persist it all). Before patching a fourth time I compared two designs
on the failure paths the reviews raised.

**Design A, the ledger (previous round).** The probe keeps lists of pending tickets and leases in memory, registers each before it
acts, finds an orphaned grant by a token in the request's reason, and asks gozer only to confirm things are gone. To be correct it
also has to be shared across attempts, persisted to the output directory (a restart loses a list), and reconciled with the time
accounting. Its state can disagree with gozer's, and every path that updates it is another place to be interrupted between the
action and the update.

**Design B, gozer keeps the books.** The lease is taken with `gozer acquire --owner-pid <the probe's pid>` and the probe holds
nothing in memory. What the probe holds is whatever gozer shows for that pid: the leases in `gozer history` that were granted to the
pid and not released, and the tickets in `gozer queue` that carry the pid. gozer already reaps a lease whose owner pid is dead, so a
crash or a kill of the probe needs no code at all.

Facts measured against the real gozer on 2026-10-27 (the tool is built on these):
- `gozer acquire --ticket T --owner-pid P` claims the probe's own ticket. While the claim is not yet possible it answers exit 10 with
  the same ticket and position, and it makes no second ticket (the queue shows one entry); when it is possible it grants the lease with
  `owner_pid` P, and `history` shows `detached: false, pid: P`. So polling the same ticket every 2 s keeps the queue place, as `gozer
  wait` does, and the lease is owned by the probe's pid. `gozer wait` grants a detached lease (the owner pid is lost), so it is not used.
- A queue entry has `pid` and `who`; a `granted` event has `lease_id`, `pid`, `who`, `chips`; a `released` event has `lease_id` and
  `duration_s` (the seconds gozer held it, reset included).
- One claim of the ticket answered with an error while another lease's reset was starting (seen once on the real gozer). The wait
  therefore treats an answer that is neither a ticket nor a grant as a moment's trouble: it looks for a lease granted to the pid, and it
  keeps claiming while the queue still shows the ticket. The ticket is cancelled whenever the wait ends without a grant.
- `gozer wait --timeout 5` meant five minutes (bare digits are minutes in gozer); this is why an earlier note here said it ran over.

Comparison on the paths Copilot raised:

| Failure path | Design A | Design B |
|---|---|---|
| A grant whose output was lost (waiter killed, answer cut off) | find it by a token in the reason, in a special timeout branch and an interrupt branch | it is a lease with the probe's pid in history; `mine()` finds it on every path, no token |
| A second attempt, or a restarted probe | the ledger is per call or per process; has to be shared and persisted | gozer's record is the state: all attempts share one pid; a restarted probe's old lease is gozer's to reap |
| A ticket kept before an interruptible step | register it first, cancel on exceptions | nothing to register: the ticket is in `gozer queue` with the pid, and `reclaim()` cancels whatever is there; the request is followed by no interruptible work before the claim loop |
| A release that overruns | track the pid of the left-running command, reconcile the time | not killed; the lease stays in history unreleased, so it blocks the next call; when gozer writes the `released` event its `duration_s` replaces the estimate |
| A recovered grant's chip time | pass timing with the exception | the interrupted call's record carries the lease id; `duration_s` from history settles it at the end of the call |
| Probe crash, SIGKILL | nothing (a leaked lease lives to the grace window) | the lease is owned by the pid: gozer reaps it |

Design B removes the in-memory lists, the token, the persistence and the special timeout and interrupt branches, and it makes the
attempts share state without a shared object. It costs a `gozer history` and a `gozer queue` read per call (about 0.1 s). Its limits
are stated in the code: the history window is the last 200 events, and an overrunning release is judged by gozer's own record. I chose B and replaced the module.

### Every path from the first request to the final release

Each row has a test in `tests/test_agent_hw_paths.py` (through the probe's wiring and a simulated gozer that keeps a queue, a history
and leases by pid), `tests/test_hwlease.py` (the gozer side) or `tests/test_agent_hw_queue.py` (the policy side).

| Id | Path | What happens |
|---|---|---|
| P1 | a free board | `gozer acquire --owner-pid`, the harness under the lease's `TT_VISIBLE_DEVICES`, one release; nothing pending |
| P1b | a busy board | one ticket, claimed every 2 s with `--ticket`, never a second request; granted; run; released |
| P2, P2b | the wait runs out, or the wait is zero | the ticket is cancelled once; BUSY; nothing runs |
| P3, P3b | the cancel fails | the answer names the ticket and the error; the ticket is in `gozer queue` for the pid, so the next call is refused until it is gone |
| P4 | a claim granted but its answer lost | found by the pid in history; used and released |
| P4b | the first request granted but its answer lost | the error path reclaims what gozer shows for the pid (found by this table: the lease leaked before) |
| P5, P5b | an interrupt during the wait | the ticket is cancelled, or a lease granted meanwhile is released; the interrupt goes on |
| P6 | gozer refuses the request | harness_error; nothing held |
| P7 | a grant with no devices | released, never run |
| P8 | an error or an interrupt in the grant callback | released (the callback is inside the release) |
| P9 | an error in the sweep or gate after the grant | released |
| P10, P10b, P10c | an interrupt or error in the harness run, or a hang | released; a hang turns the tool off |
| P11 | `gozer release` exits 0 but did not reset the chips | `release_error` for the agent and the record; the lease is gone, so nothing blocks |
| P12 | the release is refused | the lease is unreleased in history: the answer says so and the next call is refused until history shows it released |
| P13 | the release overruns 90 s | left running, never killed; the call's chip time is an estimate marked incomplete; the next call is refused; when gozer writes `released`, its `duration_s` replaces the estimate and can exhaust the budget |
| P14, P14b | an interrupt during the release | `reclaim()` releases what gozer still shows; if that fails the lease stays visible and blocks |
| P15 | history cannot be read | "cannot confirm": the next call is refused |
| P16, P16b-d | an attempt leaves something | the next attempt's first call is refused (one pid); `main` asks gozer after any attempt that called the tool and stops the probe (exit 5) when something is unresolved; it passes one `GozerLeases` to every attempt |
| P17, P18, P19 | anything unresolved | in the call file on disk, the attempt record and the report (`UNRESOLVED`); a failing check refuses the call |

The budget: the reserve for cleanup is 210 s, gozer's 120 s to stop the harness plus 90 s for the separate release and reset
(`test_the_cleanup_reserve_covers_...`), so a call's worst case, its run deadline plus both phases, fits the lease cap.

### Round 5 additions (2026-10-27)

| Id | Path | What happens |
|---|---|---|
| R1 | `gozer history` fails, `gozer queue` works (or the reverse) | `mine()` keeps what the readable one returned (leases from history, tickets from the queue) and sets `readable` False; `reclaim()` gives back what is known and `unresolved()` names it and says what it cannot confirm, so new work is still refused |
| R2 | an `acquire` granted the lease but its reply was lost (the first request, or a claim) | `reclaim(on_found=...)` calls the callback before it releases: the grant is noted and the chip-time clock starts, the lease id goes into the call record as incomplete, the held time is estimated from the request and replaced by gozer's `duration_s`, and the budget counts it before the next call; what could not be given back is told to the agent |
| R3 | an interrupt lands while `gozer release` (a separate session) is still running, or a release overran | `release()` marks the lease in flight before the command runs and keeps it so if the command is interrupted or left running; `reclaim()` does not start a second release: it waits up to 90 s for history to show the lease released, then leaves it unresolved (the next call is refused) |
| R4 | a short wait | a claim gets the wait that is left (never under 5 s, never over 60 s); no claim starts after the deadline; the first claim is made even for a zero wait; reads made while recovering use a 15 s timeout, so recovery does not stretch the wait |
| R5 | the cleanup reserve | checked in `HardwareCheck._call`: a call is admitted only when `lease_s - used - cleanup_s >= min_call_s`, and the harness backstop handed to `run_watched` is `min(call_timeout_s, lease_s - used - cleanup_s)`; the stop (term grace 120 s) and the separate release (90 s) are what the 210 s reserve is for. `test_the_harness_deadline_and_both_cleanup_phases_fit_inside_the_lease_cap` pins it on the real wiring |

A grant that gozer makes after the wait deadline (a claim that was already running) is not thrown away: it is a real lease, so the
call uses it, and the bounded claim timeout limits how late that can be.

### Round 6: subprocesses and settled values (2026-10-27)

**Sweep of every subprocess the hardware path starts, and whether it is reaped.**

| Command | Started by | If an interrupt lands | Now |
|---|---|---|---|
| `gozer acquire` (request) and `gozer acquire --ticket` (claim) | `hwlease.run_reaping` | the child (its own session) would keep running and could grant a lease after the probe moved on | SIGTERM to its group, 5 s grace, SIGKILL, then `wait`; the interrupt goes on and the closure's `reclaim()` finds any grant in history |
| `gozer cancel`, `gozer history`, `gozer queue` | `hwlease.run_reaping` | harmless but would be left running | reaped the same way |
| `gozer release` | `orchard.commands.run_command(kill_on_timeout=False)` | a reset must never be cut short | left running on purpose; the lease is tracked in `in_flight` and `reclaim()` waits for history instead of starting a second release |
| the harness under the lease | `run_watched` (stall watcher) | stopped by its own SIGTERM and SIGKILL rules | unchanged |
| `gozer run` of `hardware_phase`, `smoke_test`, `cancel_ticket` | the probe's own phase, not the tool | `run_watched`; `subprocess.run` (kills its child on any exception) | unchanged |
| the sweep of the attempt's strays | `sweep_strays` | signals only; it starts nothing | unchanged |

**Every value the agent or the record shows is read after it is settled.** The budget in an answer is formatted by `_left()`, which
settles first (the overlong release scenario answered "540 lease seconds left" while 100 remained). `settle` marks the record changed,
so a settle made while an answer was formatted is written at the end of the call. `HardwareCheck.reconcile()` settles, refreshes the
unresolved check and persists without a new call; the attempt record and the partial attempt call it right before they are built, so
a release gozer finished during the coder or the simulator check is in `attempt.json` and the report (chip time, no `UNRESOLVED`).

Pins for the two older mediums (the wait deadline and the cleanup reserve): see round 5 above. The claim timeouts and the deadline
are in `tests/test_hwlease_recovery.py` (`test_a_claim_is_given_no_more_than_the_wait_that_is_left...`, `test_no_claim_starts_after_the_deadline`);
the reserve is in `tests/test_agent_hw_recovery.py` (`test_the_harness_deadline_and_both_cleanup_phases_fit_inside_the_lease_cap`).

### Round 8: the cleanup as a state machine (2026-10-27)

The round 8 findings were narrower cases of two classes: an interrupt that lands during the cleanup, and a stale or partial read of
gozer's state. Instead of one more patch for each, this section writes the whole cleanup path down with one invariant, and
`tests/test_hwlease_state_machine.py` runs every state and event pair against a fake gozer that keeps its own ground truth.

**Invariant.** After a call's cleanup, either gozer's own record (the fake's ground truth, not what the probe read) shows nothing of
ours, or the probe reports UNRESOLVED and every new call is refused. There is no third outcome. The test checks this after each
scenario, and for the UNRESOLVED outcome it also makes a new call and checks that it is refused.

**States.** N: no lease and no ticket. T: a ticket is held. L: a lease is held and the harness runs. R: releasing (the `gozer release`
command is running). F: a release is in flight (interrupted or overran; it may still be running). U: unresolved.

**One cleanup procedure for every exception path (`GozerLeases.reclaim`).**
1. Read history, the queue and `gozer status`, fresh. A read that fails does not discard the other two. If status shows chips held
   by this pid and the 200-event history window has no lease for them, one wider read (5000 events) is made. Chips still without a
   lease id are `unidentified`.
2. Cancel every ticket the queue shows. A ticket the caller holds in hand is cancelled too when a read failed. A cancel that fails
   or is interrupted is tried once more.
3. Release every lease history shows, and every lease the caller holds in hand when a read failed. The grant callback runs first. A
   lease whose release is already in flight is waited for (up to 90 s) and is never released twice.
4. Read again, fresh, and decide. A lease, a ticket or unidentified chips of ours, or a read that fails, give U.

Every step runs in its own guard. A second KeyboardInterrupt or error inside a step is recorded (`Reclaimed.secondary`), and the next
step still runs. `run_reaping` does the same for the child process: the SIGTERM, the first wait, the SIGKILL and the last wait are
each guarded, the last wait is tried twice, and the second exception is attached to the first as `secondary_interrupt`. The first
exception is the one that is re-raised.

| State | grant | interrupt | second interrupt in the cleanup | timeout | read failure | release done |
|---|---|---|---|---|---|---|
| N (request running) | the reply is lost: step 1 finds the lease by pid (history, or status plus the wider read) and releases it | the acquire child is stopped and reaped, then the cleanup runs | the child is still signalled and waited for; the cleanup still runs | the child is killed and reaped, then the cleanup | history unreadable: status still shows chips held by our pid, so they are unidentified and the state is U; nothing held means N | not applicable |
| T (ticket held) | a claim granted with its reply lost: found by pid and released | the known ticket is cancelled, even when the queue cannot be read | the cancel is retried and the final read still runs | the wait ends and the ticket is cancelled once | the known ticket is still cancelled; the final read fails, so U | not applicable |
| L (lease held) | not applicable | the lease is released by its known id, even when history cannot be read | the release still runs; U if gozer still shows the lease | the harness is stopped, then the release | the known lease is released; the final read decides | the lease is gone: N |
| R (release running) | not applicable | the lease becomes F | the cleanup does not start a second release; it waits for the first | the release overran: F | the wait cannot confirm: U | gozer writes `released`: N |
| F (release in flight) | not applicable | the cleanup waits for history to show it released | the wait goes on | after 90 s: U | the wait cannot confirm: U | N, and gozer's duration settles the call's chip time |
| U (unresolved) | not applicable | not applicable | not applicable | not applicable | the next call is refused, naming what is held or that gozer cannot be read | the next call's check sees nothing held and the call may start |

**The window of history.** `gozer history` shows only its last events, so an unreleased grant can fall outside it. Current ownership
is read from `gozer status` (chips whose `pid` and `who` are ours). History supplies lease ids and timing. Held chips with no lease id
even in the wider read are U, and the message says they cannot be released from here.

**The wait deadline.** It is checked before every claim, including the claim after a failed claim and the claim after a late
wake-up. Only the first claim is made without the check, so a zero wait still makes its immediate claim. One consequence is that a
4 s wait claims at 0 s and 2 s and no longer at 4 s.

**Known ids.** A lease or ticket the caller holds in hand is acted on when gozer shows it, and also when a read failed. When all
three reads worked and do not show it, it is already gone and no second cancel or release is sent. The probe also records that its
own release was confirmed, so an exception after a confirmed release does not release the lease again.

**Where the cleanup reserve is checked.** In `HardwareCheck._call` (orchard/hwtool.py): a call starts only when
`lease_s - used - cleanup_s >= min_call_s`, and its run deadline is `min(call_timeout, lease_s - used - cleanup_s)`. `cleanup_s` is
210 s: 120 s of terminate grace plus the 90 s release.

**Table test.** `tests/test_hwlease_state_machine.py` has one scenario for each cell above (N, T, L and R against grant, interrupt,
second interrupt, timeout, read failure and release done, plus the history window and a hard window cap). Each runs against
`SimGozer`, which keeps ground truth (`truth_held`, `truth_tickets`) apart from what a read returns, and each ends with the
invariant. The first run of the table found one real case: with the history window capped, chips of ours were held and no lease id
could be found, and a refused release left them held while a later call still ran. The status read and the `unidentified` list
close it.
