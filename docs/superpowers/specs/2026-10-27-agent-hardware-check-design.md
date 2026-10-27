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
- A busy board: `gozer run` queues a ticket and exits 10 when no board is free. The probe keeps that ticket and waits on it with
  `gozer wait --json` for at most `--hw-wait-s` seconds, so the call keeps its place in the queue and never asks again in a loop (an
  earlier version cancelled and asked again every 10 s, which gave up the place each time). gozer ends the wait itself (`--timeout 12s`; bare digits mean minutes to gozer);
  the probe stops the waiter only if it runs 10 s past that. A grant prints JSON with the lease id and
  `TT_VISIBLE_DEVICES`; the probe runs the shipped harness under that environment (no second `gozer run`) and releases the lease with
  `gozer release` on every path, an interrupt included. When the wait runs out, or the answer is not a grant, the ticket is cancelled and
  the agent is told the board is busy and to go on with the simulator. An interrupt during the wait cancels the ticket.
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
| 6 | board held by another lease (20 s wait; measured with the earlier polling version) | BUSY after 20 s, tickets cancelled, queue empty | 20.1 | none |

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

## The wait on a ticket, measured (2026-10-27)

With one board held by a lease of my own: `gozer acquire --json` with queueing answers `{"queued": true, "ticket": ..., "position": 1}`
and exit 10. `gozer wait --json --timeout 5 TICKET` did not return within 25 s because gozer reads `5` as five minutes (`12s` is a 12 s wait). When the holder
released, `gozer wait` returned the grant (`{"granted": true, "lease_id": ..., "env": {"TT_VISIBLE_DEVICES": ...}}`) after the
holder's reset, and `gozer release LEASE` on that lease reset the chips and freed them. The probe's own wait path then ran against the real
gozer with the board held by a lease of mine: with `--hw-wait-s 12` the call answered BUSY after 12.0 s, cancelled its one ticket and the
queue was empty; with the holder released after 8 s the call was granted, waited 50.6 s (the holder's 42 s reset included), ran the
library `silu_mul` (PASS, 0.6 s with the device open), released the lease itself (48.6 s from grant to the end of the release) and left all
four chips free. The tests of the other paths use fakes of `gozer wait` and `gozer release`.

## Every path between a queue ticket and the final release

`orchard/hwlease.py` keeps one rule for all of them: register before acting. A ticket is registered when the probe keeps it, a lease
the moment it is granted (or found in gozer's history by a token in the request's reason), and each is dropped only when gozer
confirms it is gone. A new call does not start while one is pending (`HardwareCheck.health`). The tests are in
`tests/test_hwlease.py` (the bookkeeping) and `tests/test_agent_hw_paths.py` (the same paths through the probe's wiring).

| Id | Path | What happens | Test |
|---|---|---|---|
| P1 | `gozer run` queues, the wait is granted, the harness runs, the lease is released | PASS; one release; nothing pending | `test_p1_...` |
| P2 | gozer ends the wait by its own timeout | ticket cancelled once, nothing runs, BUSY | `test_p2_...` |
| P3 | the cancel fails | the answer names the ticket and the error; the ticket stays pending; the next call is refused while the queue still shows it | `test_p3_...`, `test_p3b_...` |
| P4 | the probe has to stop the waiter (it ran 10 s past gozer's timeout) and the grant is in the output it drained | the grant is used and released | `test_p4_...` |
| P5 | the same, but the grant is only in `gozer history` (granted before it was printed) | found by the token in the reason; used and released | `test_p5_...` |
| P6 | an interrupt during the wait | waiter killed; a grant that arrived meanwhile is released, otherwise the ticket is cancelled; the interrupt goes on | `test_p6_...`, `test_p6b_...` |
| P7 | the waiter cannot start | ticket cancelled, BUSY | `test_p7_...` |
| P8 | a grant with no device list | released, never run | `test_p8_...` |
| P9 | an error or interrupt in the grant callback | released (the callback is inside the release) | `test_p9_...`, `test_p9b_...` |
| P10 | an error in the sweep or the gate after the grant | released | `test_p10_...` |
| P11 | an interrupt or an error in the harness run | released | `test_p11_...`, `test_p11b_...` |
| P12 | the harness hangs (watcher stop) | released; the tool is off for the rest of the attempt | `test_p12_...` |
| P13 | `gozer release` exits 0 but its reset failed (`NOT marked clean`, `not resetting`) | `release_error` for the agent and the record; not held, so the next call may start | `test_p13_...` |
| P14 | the release is refused or fails | lease pending, error told to the agent; the next call is refused until history shows it released | `test_p14_...`, `test_p14b_...` |
| P15 | the release overruns 90 s | left running, never killed (a reset must finish); pending; refused while its process runs; resolved by history | `test_p15_...` |
| P16 | an interrupt during the release | the lease was registered before the command, so it stays pending and blocks the next call | `test_p16_...` |
| P17 | gozer's history or queue cannot be read | cannot confirm, so the next call is refused and says so | `test_p17_...`, `test_hwlease.py` |
| P18, P19 | anything unresolved after a call | in the call record on disk, the attempt record and the report (`UNRESOLVED`) | `test_p18_...`, `test_p19_...` |
| P20 | an interrupt between the kept ticket and the wait | the ticket is already registered | `test_p20_...` |

The reserve for cleanup in the budget is now 210 s: gozer's 120 s grace to stop the harness plus 90 s for the separate release and
reset (`test_the_cleanup_reserve_covers_...`). A call's worst case, its run deadline plus both phases, fits the lease cap.

Correction (2026-10-27): an earlier version of this note said gozer's `--timeout` was seen to run over. It did not: gozer reads bare
digits as minutes, so `--timeout 5` meant five minutes. The probe now writes seconds (`12s`), gozer ends the wait itself, and the
probe stops the waiter only if it runs 10 s past that.
