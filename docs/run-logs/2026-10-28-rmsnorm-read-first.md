# rmsnorm: does reading the specification's block sections first help the coder? (2026-10-28)

## Pre-registration (written 2026-10-28, before arm A was run)

Question. The coder (Qwen3-Coder-Next) writes a tt-lang `rmsnorm` kernel and has passed the simulator in 0 of 19
earlier attempts (reports rms1 to rms5). A read-only analysis of those 19 attempts found that the coder reads on average
306 of the 1,619 lines of `reference/TTLangSpecification.md` (19%), that the section "Block states" (lines 743 to 764)
was read in 0 of 19, and that `pop(): not a valid next dataflow step` ended 4 of 19, with its explanation in that
section. Is the missing information not read, or read and not found?

Change. One. Arm B's skill (`orchard/skills/kernel-probe.md`) tells the coder to read the sections "Dataflow buffer",
"Block" and "Block states" of the specification in full before it writes a kernel (finding them with `grep -n`, reading
them with `read_file` offset and limit), and to read "Block states" again when a check reports a dataflow error. It names
section titles and contains no worked solution or op-specific fix. Arm A is `origin/main` unchanged.

Library mode (a deviation from the brief, which said to use the default). The default `--library all` puts the
orchard library's verified `rmsnorm` kernel into `reference/` as `library_rmsnorm.py`. With it either arm could pass by
copying that file, which says nothing about reading the specification, and the 19 earlier attempts ran without it (their
system prompt has no library paragraph). Both arms therefore use `--library none`, the condition of the 19 attempts.
The skill's sentence about `reference/library_*.py` stays in both arms and then describes files that are absent; that
costs both arms the same.

Fixed in both arms: op `rmsnorm`; `--library none`; `--phase sim`;
temperature 0.7; the same `--seed` base (the probe adds the attempt number, so attempt k has the same seed in both
arms); 5 attempts; 1800 s wall budget and 40 turns per attempt; the same coder boot, same tt-lang pin and same
simulator environment; arm A first, arm B second. No outside model is used.

Measurements, per attempt (taken from the attempt's report and its `log/probe-*.jsonl` with
`2026-10-28-rmsnorm-read-first-analysis.py`):

| Measurement | Definition |
|---|---|
| passed in the simulator | `passed_sim` in the attempt record, from the unedited check |
| failure class at the end | deadlock, pop invalid step (any error whose text is "not a valid next dataflow step": `pop()`, `store()` or `copy`), shape mismatch, numeric, wall-clock (the attempt ended on the wall budget with no simulator result), other. The class comes from the first line of the unedited check's answer; how the attempt ended (`status`) is listed beside it |
| turns to the first simulator result | the turn whose tool call produced the first `kernel_check.py` result |
| "Block states" read | count of `read_file` calls on the specification whose returned lines cover lines 743 to 764 in full, plus shell calls (grep, sed, head) whose output prints the section's body |
| `read_file` calls | all `read_file` calls in the attempt, any file |
| coder decode speed | tok/s the probe measured before the attempt |

Success criteria (fixed now). Arm B succeeds if either:
1. at least 1 of its 5 attempts passes in the simulator, or
2. its `pop()` / `store()` dataflow-step failures are clearly below arm A's, and its deadlock failures and its numeric
   failures are not higher than arm A's.

If neither holds, the report says that reading the section does not help and the failures are design failures.
Counts only, never rates. Five attempts per arm cannot show a small difference: a change of one attempt in a failure
class is inside what a rerun of the same arm could produce.

Baseline for context only (not a control: earlier skill text, so not comparable): rms5, 5 attempts, 0 passed,
failures deadlock 1, `pop()` 1, numeric 2, shape mismatch 1; rms4: 0 passed, deadlock 3, `pop()` 1, shape mismatch 1.

Clarification added after the pilot and before arm A. The pilot (one arm B attempt, left out of the counts) ended with
`Cannot perform copy (as destination): not a valid next dataflow step`. The class "pop invalid step" is therefore
every "not a valid next dataflow step" error, whatever operation it names, and the success criterion's "`pop()` /
`store()` dataflow-step failures" means that class. The pilot ran with the same seed base and library mode as the arms.

## What was run

- Coder: Qwen3-Coder-Next, `p300` profile on one board (two chips), served by `tt-model serve` under a `gozer acquire --owner-pid` lease, port 8011. Decode speed was 27.7 to 38.6 tok/s (see the table); no attempt was flagged slow.
- Probe: `tt-orchard kernel-probe --op rmsnorm --attempts N --temperature 0.7 --seed 2026102800 --phase sim --library none`, run from an exported copy of the code for each arm. Arm A is `origin/main` at the time (4410d9d); arm B is the same tree plus the skill change and version 0.9.36. The two copies differ in `orchard/skills/kernel-probe.md`, the version string and files that the probe does not read.
- Library mode: `--library none` for both arms (see the pre-registration for why this is not the default).
- Pilot: one arm B attempt, watched for its first 12 turns. The coder ran `grep -n "^## "`, then `read_file` with offset and limit on "Dataflow buffer" and "Block states", before it wrote anything. It then wrote `kernel.py`, got a check result, rewrote it, and ended on the wall budget with `copy ... not a valid next dataflow step`. Nothing misbehaved. The pilot is not counted.
- Order: arm A (5 attempts, first coder boot), then arm B attempts 1 and 2 (same boot), then an outside interruption (below), then arm B attempts 3 to 5 on a restarted coder.

### The interruption

At about 03:51Z, while arm B attempt 3 was running, another agent's `tt-orchard bringup` run started a coder package with the same container name as this experiment's coder. Its retry stopped or replaced this experiment's coder container. Arm B attempt 3 (original run) ended after 777 s on a connection reset, and attempts 4 and 5 ran with 0 turns (`Connection refused`). Those three records are in `readfirst-b` and are not counted. Attempts 3 to 5 were rerun on a restarted coder (same command, same lease) as `readfirst-b2` with `--seed 2026102802 --attempts 3`, so the three reruns sent seeds 2026102803 to 2026102805, the seeds the lost attempts would have sent. Arm B below is attempts 1 and 2 from the first run and attempts 3 to 5 from the rerun. The restart is a second coder boot, so arm B's last three attempts differ from the other seven in that one respect. Whether the server honours the seed was not verified, as the probe's help text says.

## Results

Arm A: skill on `origin/main`, 5 attempts. Arm B: skill that says to read three sections first, 5 attempts (1 and 2 first run; 3 to 5 are rerun attempts 1 to 3).

| Arm | Attempt | Passed | Failure class at the end | Last check's first line | Turns | First check at turn | "Block states" read | `read_file` calls | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | no | numeric | pcc 0.9903, bar 0.999 | 36 | 21 | 0 | 21 | 29.9 |
| A | 2 | no | shape mismatch | `broadcast shape mismatch at dimension 1: block has 1, target has 8` | 29 | 16 | 0 | 11 | 27.7 |
| A | 3 | no | pop invalid step | `Cannot perform copy (as destination): not a valid next dataflow step` | 35 | 8 | 0 | 20 | 35.7 |
| A | 4 | no | shape mismatch | `Shape mismatch in store(): source shape (1, 8) ... destination shape (1, 1)` | 34 | 16 | 0 | 24 | 38.6 |
| A | 5 | no | other | `TypeError: unsupported operand type(s) for *: 'Tensor' and 'Block'` | 27 | 22 | 0 | 15 | 38.6 |
| B | 1 | no | pop invalid step | `Cannot perform copy (as destination): not a valid next dataflow step` | 21 | 12 | 1 | 13 | 38.6 |
| B | 2 | no | numeric | pcc 0.9928, bar 0.999 | 21 | 11 | 1 | 19 | 38.6 |
| B | 3 | no | shape mismatch | `Shape mismatch in binary operation: left shape (2, 1) does not match right shape (1, 1)` | 31 | 21 | 1 | 21 | 37.7 |
| B | 4 | no | numeric | rel_l2 0.8232, bar 0.01 | 31 | 19 | 1 | 21 | 38.6 |
| B | 5 | no | deadlock | `Deadlock detected: all generators blocked` | 33 | 15 | 0 | 14 | 38.6 |

Every attempt in both arms ended on the 1800 s wall budget (1808 to 2185 s; the budget is checked between turns). No attempt ended on a pass.

Counts by failure class:

| Class | Arm A | Arm B |
|---|---|---|
| passed in the simulator | 0 | 0 |
| pop invalid step (dataflow step) | 1 | 1 |
| deadlock | 0 | 1 |
| numeric | 1 | 2 |
| shape mismatch | 2 | 1 |
| other | 1 | 0 |

"Block states" read (lines 743 to 764 returned in full by a `read_file`, or printed by a shell call): arm A 0 of 5, arm B 4 of 5. In arm A, attempts 3 and 4 had grep hits that printed one line of the section (a `ttl.copy` row or the intro sentence); none printed it. Arm B attempt 5 read none of the three sections and used 8 greps on `reference/`, so the instruction was not followed in that attempt. The three sections were read in full (Dataflow buffer 105 of 105 lines, Block 381 of 381, Block states 22 of 22) in B attempts 1, 3 and 4; B attempt 2 read 281 of the 381 Block lines.

## Verdict against the pre-registered criteria

1. At least 1 of 5 arm B attempts passes: no (0 of 5).
2. Arm B's dataflow-step failures clearly below arm A's, with deadlock and numeric failures not higher: no. Dataflow-step failures are 1 in each arm; deadlock went from 0 to 1 and numeric from 1 to 2.

Neither criterion holds. In this experiment, reading the sections did not help, and the coder's failures are design failures: it read "Block states" in 4 of 5 attempts and still wrote kernels that fail in the same ways. One `copy` dataflow error occurred in each arm: arm A attempt 3 never read the section, and arm B attempt 1 read all three sections in full and then made the same kind of error.

Five attempts per arm cannot show a small difference. A change of one attempt in a failure class is inside what a rerun of the same arm could produce, and the pre-registered criterion 2 needed a clear drop, which 1 versus 1 is not.

## What could not be determined

- Whether the coder would reach a pass with a different change (a worked example, a design note about what each thread does, or the library). This experiment tested one change and it did not help.
- Whether reading helps at a size smaller than this experiment can see. If arm B's true pass rate were 20%, five attempts would still show zero passes about one time in three (0.8 to the fifth power).
- Whether the coder's decode speed matters. It was 27.7 to 35.7 tok/s in arm A attempts 1 to 3 and 37.7 to 38.6 elsewhere. All attempts ended on the wall budget at 21 to 36 turns, and arm B used its first turns on the long reads, so the arms did not have the same number of turns to iterate. Arm A's early attempts decoded more slowly yet made more turns (36, 29, 35) than arm B's first two (21, 21), so slower decoding does not explain a gap in either direction.
- Whether the restart changed arm B attempts 3 to 5. They ran on a second coder boot; no other difference is known. The `rel_l2 0.823223` of B attempt 4 equals the value in the earlier `rms5` attempt 5, which suggests that the same wrong kernel structure arrives from different prompts.
- Whether the server honours the seed (not verified).
- The class "other" in arm A attempt 5 (`Tensor * Block`) is a type error on a host-side tensor; the pre-registered classes had no name for it.

## Notes on the method

- The analysis script (`2026-10-28-rmsnorm-read-first-analysis.py`) was corrected after arm A finished and before arm B's results were read, in two places, and every number above comes from the final version. First, the wall-clock class applies only to an attempt with no simulator result (all ten attempts had one). Second, a shell call counts as printing "Block states" only when it shows at least 5 of the section's numbered lines or at least 3 of its six state names; a grep that prints one line of the section is counted apart as a grep hit. The first version counted a one-line grep hit, which gave arm A 2 of 5 where the strict version gives 0 of 5.
- Per-attempt records: the probe output directories `readfirst-a`, `readfirst-b` (attempts 1 and 2 valid) and `readfirst-b2` under `~/orchard-runs/kprobe/`, not committed.
- A slip at the start: `gozer acquire --chips 2-3` granted board 0 (chips 0 and 1), which was free at the time and which this experiment was told not to use. The lease was released after about one minute with `--no-reset`, before anything opened a device. The lease for board 1 was then taken with `--exact` and its unit id.
