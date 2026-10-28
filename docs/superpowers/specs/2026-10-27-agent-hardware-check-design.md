# An agent-callable hardware check in the kernel probe (design note, 2026-10-27)

Branch `agent-hw-check`, on `probe-hw-sim-gap` (pull request 112). Version 0.9.44 (the first version was 0.9.13; the lease design changed in round 9).

## Decision

The operator's decision: a hardware check is always available to the agent during a kernel-probe attempt. If hardware is the
right choice for the next step, the agent can make that choice. Before this change the probe ran the simulator for every kernel
and ran hardware only after the attempt ended.

The agent cannot call gozer. The command runner refuses gozer writes and the agent's shell has a no-chip device mask. So the
probe offers the check as a tool of the agent loop, next to `shell`, `read_file` and `write_file`. A short-lived wrapper process
holds the lease (see "A wrapper process holds the lease" below); the probe does not.

## The tool

`hardware_check` takes one optional argument, `note` (a string of at most 200 characters, kept in the record). Any other
argument is an error and nothing runs. The agent supplies no command and no path.

Each call does what the probe's own hardware phase does, on the attempt's current `kernel.py`:

1. `copy_kernel` copies the file into a fresh probe-owned directory, `DIR/.hwcheck/<op>-<attempt>-<call>/` (a regular file of at most
   256 KiB; a link, a directory or a larger file is `forbidden_code`, a missing file is `no_kernel`). `.hwcheck` is not under
   `.verdict`, because the authoritative check empties its directory and `load_attempts` reads it.
2. `_hw_gate` runs the static gate on the copy and compares its sha256 with the sha256 taken at copy time.
3. `_hw_run` runs the shipped `kernel_harness.py` with the shipped spec under the wrapper of `orchard/hwtool.py` (it takes one
   chip with `gozer acquire --chips 1 --exact 1 --owner-pid`), under the stall watcher, with the probe's hardware environment, and judges the result with the hardware bars
   (`kernel_check.classify`).

Steps 2 and 3 are the code that `hardware_phase` used before, moved into two functions that both paths call. `hardware_phase`
behaves as before (it still uses `gozer run`): a hang or a queued request still ends the probe. The gate runs again before the
wrapper starts, and the wrapper checks the sha256 of the copy again after its wait, because the agent can write anywhere the
operator can.

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
- A busy board: the wrapper asks with `gozer acquire --owner-pid <its pid>`. When no board is free gozer queues one ticket and the
  wrapper runs `gozer wait --timeout` once, for at most `--hw-wait-s` seconds. A grant runs the shipped harness under the lease's
  `TT_VISIBLE_DEVICES` and the wrapper releases the lease in a `finally`. When the wait runs out the ticket is cancelled and the
  agent is told the board is busy and to go on with the simulator. A wrapper that dies leaves a lease with a dead owner, which
  gozer reaps.
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
whether the ticket was cancelled, whether it timed out, the note, the first line, the kernel's sha256, the lease id and the
release error if there was one. `agent_hw` has `stopped` (the sentence) when the check after a call stopped the tool. An attempt run without
the tool records `{"enabled": false}`. These calls are not the attempt's hardware verdict (`hw`).

The report has a column `agent hw` in the attempt table (calls, passes, chip seconds) and a section "Hardware checks called by the
coder" with the policy, every call, and a totals line that gives the chip time (the seconds gozer held a lease, reset included) and
the seconds waited for a board, and a line `STOPPED after ...` when a tool was stopped.

## What the agent is told

`orchard/skills/kernel-probe.md` (100 lines, one fenced block) names the tool, its cost (about 3 seconds to run, about 45 with the
lease and the reset), when it is worth using (a kernel the simulator cannot run, a numeric check at hardware precision after the
simulator passes, a performance question) and that the simulator is much cheaper for everything else. The tool's description states
the same and this attempt's limits.

## Measured on hardware

The first table was measured through `gozer run` (rounds 1 to 3). The wrapper path was measured on 2026-10-27 after round 9; see the
last part of this section.

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

### The wrapper against the real gozer (round 9, 2026-10-27)

The wrapper (`orchard/hwtool.py run`) was run as a child process against the real gozer. The wrapped command was `/bin/true` or
`/bin/sleep`: no chip was opened and no ttnn was imported. The board of chips 0 and 1 was free. The other board was held by
another agent and was not touched. Each release resets the board and took about 42 s.

| Run | Result |
|---|---|
| A: free board, `/bin/true` | exit 0 in 41.8 s; lease owned by the wrapper's pid (not detached); the wrapper's own stamps give 41.7 s held; release ok; `held_by_us` shows nothing |
| C: SIGTERM while `/bin/sleep 120` runs under the lease | exit 143 in 43.0 s; the command was ended, the lease released, the record says `interrupted` and `release_ok` true; nothing held afterwards |
| B: board held by another lease, wait 6 s | exit 10 in 6.2 s; the ticket was cancelled and the queue was empty afterwards |
| D: board held, the holder lets go after 8 s, wait 120 s | granted by `gozer wait` after 50.6 s (the holder's own reset took most of it); the lease was detached; `/bin/true` ran; released; five heartbeat lines, all different; nothing held afterwards |

The stall watcher was not in the loop in these runs. Its quiet limit is 60 s and the heartbeat is 10 s, so a wait does not look like
a hang. That is argued from the numbers, and is covered in the tests only as "the lines differ".

## Not done

- The policy numbers are not tuned. Whether a coder uses the tool well, and whether it changes the pass counts, is not known: no
  coder round was run.
- A kernel that hangs the device is stopped by the watcher and the lease release resets the chip, but a hang that the watcher
  cannot see (a process that keeps writing) runs until the call timeout.
- The probe's own end-of-attempt hardware phase (`--phase hw`) is unchanged and still needs a simulator pass or a `sim_gap`.

## A wrapper process holds the lease (round 9, 2026-10-27)

Rounds 4 to 8 tried to make the probe's own process hold the lease and clean up after every exception: a ledger read back from
gozer, a reclaim, a tracker for releases still running, a settle of the chip time, a state machine over all of it. Each round of
review found another interrupt or partial read that the cleanup did not cover. The module is optional for the harness and a wrong
cleanup can leave a real board leased, so the design was cut down to what is needed.

**The design.**

1. The probe copies and gates the kernel as before (`prepare`).
2. The probe starts one short-lived wrapper, `python orchard/hwtool.py run --who TAG --reason R --wait-s N --result PATH --sha256 H
   --file KERNEL -- python kernel_harness.py ...`, in its own session, under the stall watcher (`run_watched`), with an outer timeout
   that is the sum of every phase (`hwtool.outer_backstop_s`: the acquire, the wait and its margin, the call timeout and the release
   grace). The command is the shipped harness. The agent supplies nothing.
3. The wrapper holds the lease. It runs `gozer acquire --owner-pid <the wrapper's pid>`. A grant is used at once. A busy board
   gives one ticket, and the wrapper runs `gozer wait --timeout Ns TICKET` once (it prints a new line every 10 s, so the stall
   watcher sees progress). A wait that ends without a grant runs `gozer cancel TICKET` and the wrapper exits with status 10. After a
   grant the wrapper checks the sha256 of the kernel copy again (the agent can write anywhere the operator can, and a wait can last
   two minutes), runs the harness with `TT_VISIBLE_DEVICES` from the lease (the call-timeout clock starts at the grant, and the harness is stopped at
   that deadline whatever the wait took: exit status 124, outcome `timeout`, a hang for the probe), runs `gozer release` in a `finally`, and exits with the
   harness's status. A SIGTERM or Ctrl-C ends the harness, releases the lease and exits. A second signal during the release is
   ignored, so the reset is not cut short.
4. If the wrapper dies by any signal, including SIGKILL, its lease has a dead owner and gozer reaps it. `gozer wait` grants a lease
   that has no owner pid (gozer's own `wait` does that), which gozer reaps by its grace window if nothing holds the chips; this is
   the one place the wrapper relies on gozer's rule for a detached lease instead of the owner pid.
5. The probe kills the wrapper's process group on any exception (`run_watched` does this) and reads `gozer status --json` once after
   every call that reached the wrapper (`hwtool.held_by_us`). It repairs nothing: no release, no cancel, no reconcile, no second
   request.
6. If that one read shows a chip, or a queue ticket, under this probe's tag (`claude:orchard-kernel-probe:<probe pid>`) or owned by
   a wrapper pid it started, or if the read fails (the check fails closed), the tool is stopped. The answer says so. Every later
   call is refused without asking gozer for anything. `main` ends the run with exit 5 after the attempt and starts no later attempt
   and no hardware phase.

Chip time is the wrapper's own: seconds from its grant stamp to its end stamp (written after the release), from its own clock. A
record that never finished (no end stamp: the wrapper was killed or wrote nothing) and an interrupted call are charged as an upper
bound (`lease_upper_bound`): the larger of the full call timeout plus the release grace and the seconds the call has been running
up to now (`hwtool.unfinished_charge_s`), never zero and never the partial time. A record that finished and was never granted is zero.
Nothing is settled afterwards against gozer's history.

**Timeouts.** The phases each have their own allowance, and the outer timeout is their sum so that none can eat another: acquire
(60 s), wait (`--hw-wait-s` plus a 30 s margin for `gozer wait`), run (the call timeout, enforced by the wrapper from the grant) and
release (a 90 s grace; a cancel, 60 s, uses the same slot). A test checks that the outer timeout is at least the longest path
through the wrapper for several waits and call timeouts. The lease-seconds cap counts chip time only (the wrapper's own stamps), and
the 210 s cleanup reserve still covers the group's 120 s terminate grace and the 90 s release.

**Records after an interrupt.** A call that finished is written to the record, with its chip time, before the status check, and
before any interrupt that lands afterwards is handled. An interrupt in the status check stops the tool (fail closed) and the stop is
written before the interrupt goes on. An interrupt while the record is being made records the call once, charged once. An interrupt
before the wrapper could start is charged the same upper bound (it cannot be told from one after the grant).

**Three states.**

| State | Meaning | Events and what follows |
|---|---|---|
| idle | no call is running and nothing is held | a call: becomes running. A call over the policy (calls, lease seconds, a hung attempt) or a blocked kernel is refused and stays idle |
| running | a call is in progress | a second call is answered at once ("already running"). The call ends in one of: pass or fail, busy (wait ended, ticket cancelled), hang (the watcher killed the group), a wrapper that could not get a lease or saw the copy change, or an exception (the group is killed first). Then the single status read: nothing of ours held, back to idle; held or unreadable, stopped |
| stopped | the check after a call found a chip or ticket of ours still held, or could not tell | every call is refused with the sentence, and gozer is not asked anything; the probe ends with exit 5. There is no way back inside the attempt |

**Invariant.** After a call, either gozer holds nothing of ours, or the tool is stopped and refuses every later call without asking
gozer. `tests/test_agent_hw_states.py` runs every scenario of the table below against a fake gozer that keeps its own record of
leases and tickets, and checks that invariant on that record, which can differ from what the probe read. It also runs the real wrapper as a child
process with a stub `gozer` and sends it SIGTERM.

| Scenario | gozer afterwards | Tool afterwards |
|---|---|---|
| free board, harness passes | clean | idle |
| busy, then the wait grants | clean | idle |
| busy, the wait ends | ticket cancelled, clean | idle (answer BUSY) |
| busy, the wait ends and the cancel fails | ticket left | stopped |
| the stall watcher kills a hung run | released by the wrapper, clean | idle, hardware off for the attempt |
| Ctrl-C at the probe during the run | released by the wrapper, clean | idle; the interrupt goes on |
| the wrapper is SIGKILLed after its grant | lease still shown until gozer reaps it | stopped |
| the release is refused | lease held | stopped |
| gozer says released but the lease stays | lease held | stopped |
| the release works but the reset did not take | clean | idle, the answer carries the release note |
| `gozer status` cannot be read | unknown | stopped |
| `gozer acquire` is refused | clean | idle (a harness error, no run) |
| the kernel copy changed during the wait | released, clean | idle (a harness error, no run) |

**What was removed.** `orchard/hwlease.py` (the lease ledger read back from gozer, `reclaim`, the in-flight release tracker,
`unidentified` chips, the wider history read), the settle of chip time against gozer's history, the reconcile before the attempt
record, and the tests of all of it. The policy object keeps its limits (calls, lease seconds, call timeout, wait, cleanup reserve).
The cleanup reserve is still checked in `HardwareCheck._call`: a call starts only when `lease_s - used - cleanup_s >= min_call_s`,
and the run deadline is capped by the same amount.

**Limits.**

- A lease from `gozer wait` has no owner pid. If the wrapper is SIGKILLed while it holds one, gozer reaps it by its grace window, not
  at once. The status check then shows it, and the tool stops.
- The one status read can run before gozer has reaped the lease of a killed wrapper. The tool then stops, which is the safe side; the
  operator looks with `gozer status`.
- The tag in `--who` is per probe process. Two probes on one box have different tags.
- The wrapper's `release` has no timeout of its own. The bound is the probe's backstop and the 120 s terminate grace of the process
  group, after which the group is killed and gozer reaps a lease with a dead owner.
