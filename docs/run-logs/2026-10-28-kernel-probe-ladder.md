# Kernel probe ladder: where do the coder's passes stop? (2026-10-28)

Issue 175. Pre-registration first; the results follow the pre-registration in the same file.

## Pre-registration (written 2026-10-28, before any pilot or batch attempt)

Question. The kernel probe has measured hard ops only (rmsnorm passed 0 of 29 attempts over earlier runs, linear_scan 0
of 3). With only zeros measured, a change to the prompt, the library or the coder cannot show a gain. On a ladder of ops
from easy to hard, at which rung do the coder's simulator passes fall to zero?

The ladder (`defaults.KERNEL_PROBE_LADDER`), one op per rung, easiest first:

| Rung | Op | What the kernel computes | Input shapes |
|---|---|---|---|
| 1 | `add` | out = a + b | 2 x (128, 128) |
| 2 | `mul` | out = a * b | 2 x (128, 128) |
| 3 | `scale_shift` | out = x * 2.5 + 0.5 (two Python scalars) | (128, 128) |
| 4 | `add_row` | out = x + r, where the (32, 256) tensor r is repeated down the rows of x | (64, 256) and (32, 256) |
| 5 | `rsqrt` | out = 1 / sqrt(x), x positive | (128, 128) |
| 6 | `row_sum` | out[i, 0] = sum of row i; the other 31 columns of out are 0 | (64, 256), output (64, 32) |
| 7 | `center` | out = x - mean(x, last axis) (a reduction, then a broadcast back) | (64, 256) |
| 8 | `rmsnorm` | out = x * rsqrt(mean(x^2, last axis) + 1e-6) * w | 2 x (64, 256) |
| 9 | `linear_scan` | out[t] = a[t] * out[t-1] + b[t] down axis 0 | 2 x (64, 128) |

`add` is upstream's own example, so its pass shows the loop works. Each rung has a CPU reference (torch, in
`references.py`), the same simulator bars (PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02, max abs error
within 0.01 + 0.01 |expected|) on 3 seeds with and without a heavy-tailed outlier set (6 input sets), and a working
kernel under `tests/fixtures/kernel-loop/kernels/` that passes on the simulator. For each new rung a wrong kernel (one
changed expression) fails the unedited check, so the bar can tell a wrong kernel from a right one.

Fixed in every rung (the settings the report records):

- Op list and order: as in the table. One op per rung.
- Coder: Qwen3-Coder-Next, `p300` profile on one board (chips 2 and 3), served by `tt-model serve` under a
  `gozer acquire --owner-pid` lease, port 8011. No outside model is used.
- Command: `tt-orchard kernel-probe --ladder --temperature 0.7 --seed 2026110100 --attempts 10 --phase sim --library none --no-agent-hw`
  with `--rung-wall` as set below, run from an exported copy of this branch's head.
- `--no-agent-hw`: the agent is offered no hardware tool, so the experiment takes no lease beyond the coder's board.
- Temperature 0.7, no top-p. Attempt k of every rung sends seed 2026110100 + k (k = 1 to 10). Whether the server honours
  the seed was not verified.
- `--library none`: no worked kernel is in `reference/`, the condition of the rmsnorm read-first run.
- Simulator phase only: a pass is `passed_sim`, from the unedited check. No hardware phase.
- 40 turns per attempt, 4096 tokens per reply, the default tool timeouts.
- Skill text: `orchard/skills/kernel-probe.md` at this branch's head. It differs from the text of earlier runs in one
  way: the "Read first, by kind of op" lines now name the new ops (`mul`, `scale_shift`, `add_row`, `rsqrt`, `row_sum`)
  next to the old ones. This is needed so the skill names every op; it adds no worked solution. The earlier rmsnorm
  numbers were measured with the old text, so they are context and no control.
- Pilots and batches use different seed bases (pilot 2026110000, one attempt per rung), so no pilot attempt repeats a
  batch attempt. Pilot attempts are not counted in the batch table.

Wall budget per rung (the rule, `kernelladder.size_wall_budget`): 4 times the wall time of the slowest PASSING pilot
on that rung, at least 300 s and at most 1800 s. A rung whose pilot did not pass keeps 1800 s, because the question
there is whether more time lets the coder pass. The unedited check judges the kernel alone, so a budget long enough
for the passes cannot lower the pass count and only shortens the time failing attempts spend. The budget is checked
between turns, so an attempt can overrun it by up to one tool timeout (420 s at most). The table counts attempts that
end on the budget with no simulator result as `wall_clock`, so a reader can see how many a longer budget could have
changed. The budgets are written into this file after the pilots and before the batch.

Projection and the stop rule. Projected batch seconds = sum over rungs of 10 x (expected seconds per attempt + 60 s of
probe overhead). The expected seconds of a rung are its pilot's agent wall time (its budget when the pilot failed).
If the projection for the whole ladder is above 6 hours, the batch stops after a rung when the elapsed time plus the
projection for the rungs left is above 6 hours (`--max-hours 6 --rung-expect-s ...`), and the partial table is the result.
Because a stop in ladder order would leave out the hardest rungs, where the answer is, this order is used when the whole
ladder does not fit in 6 hours: the rungs run from the first rung whose pilot failed, one rung at a time downward
(toward easier rungs) and upward, nearest to that rung first, until the projection is used up. Rungs not run are named in
the report. This is a pre-registered deviation from "ladder order", for that case only.

Readout (counts only, never rates). Per rung: attempts, simulator passes, failure class counts (`deadlock`,
`dataflow_step`, `shape_mismatch`, `numeric`, `tt_lang_error`, `import_error`, `forbidden_code`, `no_kernel`,
`wall_clock`, `other`; the class comes from the first line and category of the unedited check, as in the rmsnorm
read-first run log, whose "pop invalid step" is `dataflow_step` here and whose "other" is split into the last four
classes above). The first rung in ladder order with 0 passes in its 10 attempts is the target rung for later experiments.
No pass threshold is pre-registered, because the experiment measures a baseline. Ten attempts per rung cannot tell a
pass rate of 0% from one of 20% (0.8 to the tenth power is 0.11), so a rung with 0 or 1 passes is "at or near zero",
and a rung with 9 or 10 passes is "at or near full".

Not measured here: hardware passes, other temperatures, other coders, other skill texts, other libraries, longer budgets.

## Pilot results and the batch plan (written 2026-10-28, after the pilots and before the first batch attempt)

Pilot: one attempt per rung, `--ladder --attempts 1 --temperature 0.7 --seed 2026110000 --phase sim --library none --no-agent-hw`,
coder Qwen3-Coder-Next as above, default 1800 s wall budget, rungs in ladder order on a coder that served only this probe.
The pilot ran from commit cfe7b5d. Its rsqrt attempt used the earlier rsqrt inputs (a sigmoid of a normal draw, with no heavy-tail
set; the batch uses the positive inputs from the review fix, see the PR). The pilot was stopped by hand during rung 8
(`rmsnorm`, exit 4, no attempt counted), so `rmsnorm` and `linear_scan` have no pilot attempt.

| Rung | Op | Pilot result | Turns | Agent wall (s) | Budget by the rule (s) |
|---|---|---|---|---|---|
| 1 | add | passed | 4 | 206 | 824 |
| 2 | mul | passed | 21 | 361 | 1444 |
| 3 | scale_shift | passed | 29 | 749 | 1800 (4 x 749 = 2996, capped) |
| 4 | add_row | passed | 21 | 1391 | 1800 (capped) |
| 5 | rsqrt | passed | 12 | 281 | 1123 |
| 6 | row_sum | passed | 15 | 570 | 1800 (4 x 570 = 2280, capped) |
| 7 | center | failed, `Deadlock detected: all generators blocked` | 29 | 1914 | 1800 (no pass to scale from) |
| 8 | rmsnorm | not piloted | | | 1800 |
| 9 | linear_scan | not piloted | | | 1800 |

Projection for the whole ladder at 10 attempts per rung: sum of 10 x (pilot wall + 60 s) for rungs 1 to 7 is about 16 hours
(add 0.7, mul 1.2, scale_shift 2.3, add_row 4.0, rsqrt 0.9, row_sum 1.8, center 5.5), and rungs 8 and 9 add about 5.3 hours each
at the wall budget. That is far above 6 hours, so the pre-registered rule for a ladder that does not fit applies: start at the first
rung whose pilot failed (`center`, rung 7) and take the rungs nearest to it.

Batch plan (a deviation from one process at a time, decided here and not before the pilot, with its reason):

- `center` alone takes about 5.5 hours at its budget (a failing attempt runs to the wall budget, and the 1914 s of the pilot shows the
  overrun of a tool call after the budget), so a strictly sequential batch fits one rung in 6 hours.
- Two probe processes therefore run against the one coder at the same time. Process A: `--ladder --rungs center`. Process B:
  `--ladder --rungs row_sum,rsqrt,mul,add --max-hours 5.5` with `--rung-expect-s row_sum=570,rsqrt=281,mul=361,add=206`: the rungs
  below `center`, nearest first, that fit beside it. Both use `--attempts 10 --temperature 0.7 --seed 2026110100 --phase sim
  --library none --no-agent-hw` and the budgets above through `--rung-wall` (row_sum 1800, rsqrt 1123, mul 1444, add 824, center
  1800), so attempt k of every rung sends seed 2026110100 + k.
- `scale_shift` and `add_row` are not in the batch (they would add 2.3 and 4.0 hours); they keep their one pilot attempt, and the
  table says so. `rmsnorm` and `linear_scan` are not in the batch for the same reason; the earlier runs (0 of 29 and 0 of 3, with
  other skill text) are the only data on them.
- Two clients at once slow the coder. The 2026-10-17 measurement in `orchard/defaults.py` is 77% of single-client speed each. Slower
  decoding means fewer turns inside the wall budget, so every rung in the batch is measured under that condition (and under single-client
  speed after the other process ends); the pilot was measured at single-client speed. Each attempt's record has the decode speed measured
  just before it, and the report lists it. A comparison of a pilot wall time with a batch pass time mixes the two conditions.
- Stop rules. Process B stops after the rung it is on when process A has ended, and any rung it did not finish is reported with
  the attempts it has. If the coder is reported slow (3 samples below half its best speed) the probe stops by its own rule and that is
  reported. No attempt is repeated or dropped. Neither process is restarted with different settings during the batch.
- The batch runs from an export of the branch head at the time of the run; the head is named in the results.

## Old coder, stopped at about 50 minutes, partial (not in the results table)

Everything above this line was measured on the earlier package `raahemnabeel/qwen3-coder-next-blackhole` (p300 profile, board 1).
At about 20:49Z the operator decided to move the ladder to the new default coder (PR 185), so both batch processes were stopped
with SIGTERM after about 52 minutes (started 19:56Z). What they had, at 29 to 38 tok/s with two clients (the pilot was measured at
one client):

| Rung | Attempt (seed 2026110100 + k) | Result | Turns | Agent wall (s) |
|---|---|---|---|---|
| center | 1 | failed, `IndexError: row slice 2:4 is outside the tensor, which has 2 tile(s) along it` | 15 | 917 |
| center | 2 | failed, `Tensor shape (32, 32) does not match Block shape (1, 8)` (shape mismatch) | 20 | 871 |
| center | 3 | passed in the simulator | 13 | 532 |
| center | 4 | interrupted by the stop, not counted | | |
| row_sum | 1 | failed, numeric (PCC 0.365 on seed 0, bar 0.999) | 16 | 2124 |
| row_sum | 2 | interrupted by the stop, not counted | | |

This is 3 finished attempts on `center` and 1 on `row_sum`: too few to say anything about a rate. The earlier pilot is above. Neither
is mixed into the table of the new coder.

## New coder: what changed and the plan (written 2026-10-28 before the first attempt on the new coder)

- Coder change: from `raahemnabeel/qwen3-coder-next-blackhole` to `raahemnabeel/qwen3-coder-next` (fused decode), profile `p300` on one
  board (chips 0 and 1) under a new gozer lease, port 8011, started with `tt-model serve --profile p300 --port 8011 --device-id 0,1
  --local-only --detach`. The old container was stopped and its lease released first. Decode speed is recorded by the probe before
  each attempt; the coder's speed with the first measurement is in the results.
- Code: the branch head after merging PR 185 (main at that time), exported for the run.
- Unchanged: the ladder, the ops, the bars, the settings of the first pre-registration (temperature 0.7, `--phase sim --library none
  --no-agent-hw`, 40 turns, seed base 2026110100 for the batch and 2026110000 for the pilot), and the budget rule.
- Re-pilot: faster decoding changes the turns that fit inside the wall budget, so each rung of the plan is piloted once on the new
  coder (seed base 2026110000, one attempt per rung, 1800 s), and the budgets of the rule are recomputed from those walls.
- Batch: process A runs `center`; process B runs the rungs below it nearest first (row_sum, rsqrt, add_row, scale_shift, mul, add) for as
  many as its projection allows (5.5 hours at the pilot walls); the same stop rules as above. Which rungs B takes is written in the
  next section before the first batch attempt.
- Rung 8 (`rmsnorm`), added by the operator: a third process against the same coder, `--ladder --rungs rmsnorm --attempts 10
  --temperature 0.7 --seed 2026110200 --phase sim --library none --no-agent-hw`, wall budget 1800 s. `--library none` because the
  default would put the verified library `rmsnorm` in `reference/`, and the coder could pass by copying it. It has no pilot. Its
  result is reported apart from the ladder table. Three clients at once slow each other; each attempt's decode speed is recorded.
  If none of the 10 pass and the failures repeat one class, the operator authorized helper calls to Telegraph for this problem: first a
  line in the Telegraph decisions file (trigger, what was tried locally at least twice, why it is worth it), the failing kernel and
  the simulator error sent only where the policy allows, Haiku before Sonnet, at most 6 calls, and the report says whether the reply
  changed anything.

## New coder: pilot results and the batch settings (written 2026-10-28 after the new-coder pilot, before the first batch attempt)

New coder pilot (`--ladder --rungs add,...,center --attempts 1 --seed 2026110000`, commit 115a6c8, single client, decode 106 to 107 tok/s
measured by the probe before each attempt; the old coder measured 36 to 38 tok/s):

| Rung | Op | Pilot result | Turns | Agent wall (s) | Budget by the rule (s) |
|---|---|---|---|---|---|
| 1 | add | passed | 13 | 172 | 689 |
| 2 | mul | passed | 12 | 157 | 628 |
| 3 | scale_shift | passed | 33 | 654 | 1800 (capped) |
| 4 | add_row | passed | 19 | 475 | 1800 (4 x 475 = 1901, capped) |
| 5 | rsqrt | passed | 13 | 134 | 536 |
| 6 | row_sum | failed, `Cannot perform copy (as destination): not a valid next dataflow step` | 24 | 563 | 1800 (no pass) |
| 7 | center | passed | 28 | 651 | 1800 (capped) |

Batch (three processes against the one coder, all `--attempts 10 --temperature 0.7 --phase sim --library none --no-agent-hw`, commit 115a6c8):

- A: `--ladder --rungs center --seed 2026110100 --rung-wall center=1800`.
- B: `--ladder --rungs row_sum,rsqrt,add_row,scale_shift,mul,add --seed 2026110100 --max-hours 5.5` with `--rung-wall
  row_sum=1800,rsqrt=536,add_row=1800,scale_shift=1800,mul=628,add=689` and `--rung-expect-s
  row_sum=563,rsqrt=134,add_row=475,scale_shift=654,mul=157,add=172`. The rungs run in that order (nearest to `center` first) and B
  stops by its own projection rule when the rest would pass 5.5 hours; the table says which rungs did not run.
- C (operator's addition): `--ladder --rungs rmsnorm --seed 2026110200 --rung-wall rmsnorm=1800`, as pre-registered above.
- Projection at pilot walls (10 x (wall + 60 s)): center 1.9 h; row_sum 1.7, rsqrt 0.5, add_row 1.5, scale_shift 2.0, mul 0.6, add 0.6 h.
  The whole ladder (rungs 1 to 7) is 8.9 hours at one process, so rungs 8 and 9 and a single process would not fit; three
  processes in parallel are the plan, and a rung that does not fit in B's 5.5 hours is left out and named.
- Three clients slow each other (the old measurement: 77% of single-client speed for two, 60% for three). A pass time in the batch
  is therefore not comparable with a pilot wall.

## Results on the new coder (stopped on the operator's instruction; partial)

Coder: `raahemnabeel/qwen3-coder-next` (fused decode), profile `p300`, one board, started about 20:50Z on 2026-10-28; branch head 115a6c8 for the
runs. Attempts at temperature 0.7, `--phase sim --library none --no-agent-hw`, 40 turns, seeds 2026110100 + k (pilot 2026110000 + k).
The batch was stopped by the operator at about 00:20Z on 2026-10-29, after about 3 hours; no rung reached 10 attempts. The ladder
table below combines the single-client pilot and the three-client batch, which are different load conditions (see below).

| Rung | Op | Pilot (1 attempt, one client) | Batch attempts | Batch passed | Batch failure classes | Wall budget (s) |
|---|---|---|---|---|---|---|
| 1 | add | passed, 172 s | not run | | | |
| 2 | mul | passed, 157 s | not run | | | |
| 3 | scale_shift | passed, 654 s | not run | | | |
| 4 | add_row | passed, 475 s | not run | | | |
| 5 | rsqrt | passed, 134 s | not run | | | |
| 6 | row_sum | failed (`not a valid next dataflow step`), 563 s | 9 | 6 | numeric 1, shape_mismatch 1, dataflow_step 1 | 1800 |
| 7 | center | passed, 651 s | 6 | 1 | numeric 2, tt_lang_error 1, dataflow_step 1, deadlock 1 | 1800 |
| 8 | rmsnorm | not piloted | 3 (see below) | 0 | shape_mismatch 1, dataflow_step 1, tt_lang_error 1 | 1800 |
| 9 | linear_scan | not piloted, not run | | | | |

Counts only, and none is a rate: 6 of 9 and 1 of 6 cannot be told apart from 4 of 9 or 3 of 6 with confidence, and the attempts that
did not finish (the stopped ones) are not counted. Pass walls on the new coder in the batch: row_sum 227 to 1198 s; center 1882 s (the one pass).
Attempts that ended on the wall budget carry its overrun (up to 2124 s on the old coder; center attempt 6 passed at 1882 s on a 1800 s budget).

Where passes stop. On this evidence no rung has zero passes. `center` has the lowest pass count (1 of 6) among the rungs with a batch;
`row_sum` (6 of 9) passed more often than `center`, so the ladder is not monotone in this order: `row_sum`, which was placed below
`center`, failed its own pilot and failed 3 of 9 batch attempts. The first rung in ladder order with a failure in the pilot is `row_sum`.
The rungs `add` to `rsqrt` have one pass in one attempt each and no batch; whether their pass count is near 10 of 10 is not known.

Rung 8 (`rmsnorm`, run separately at the operator's request, `--library none`, seed 2026110200 for attempt 1 and 2026110201 + k for
the restart, 1800 s): 3 finished attempts, 0 passed. It was stopped twice on the operator's instruction ("stop rmsnorm"), and the second
process, which I had started, was stopped again; no further rmsnorm run was made. Partial records: attempt 1 `shape_mismatch` (`Tensor
shape (32, 32) does not match Block shape (1, 8)`, 32 turns, 1830 s), then `dataflow_step` (676 s, 28 turns) and `tt_lang_error`
(1816 s, 25 turns). Telegraph was not used (the condition was 10 failed attempts with a repeated class; there were 3).

Failure classes seen, by name: deadlock, `not a valid next dataflow step`, a numeric PCC miss, shape mismatch, and a type or index error
(`IndexError: row slice ... is outside the tensor`).

Load and decode speed. The probe measured 106 to 107 tok/s before each pilot attempt (one client). With three clients the first
attempt of each process measured 48.7 to 56.5 tok/s ("busy"). Every later attempt measured 8.5 to 12.4 tok/s before it started, with 2 or
3 requests running. I did not find why the speed fell from about 49 to about 12; the coder's log shows prefill chunks of 8K to 9K tokens
and a prefix state with all 16 slots used, and nothing that names a fault. The probe labelled every one of these samples "busy" and none "slow",
because it compares with the load. So the batch attempts ran at about a quarter of the pilot's decode speed, and their pass counts and
wall times must not be compared with the pilot's. That is the largest limit of this table.

## What could not be determined

- Pass counts of rungs 1 to 5 over 10 attempts (not run). Their pilots passed once each on the new coder and once each on the old.
- Whether `row_sum` or `center` is the first rung with a pass count near zero: neither is, on 9 and 6 attempts.
- The effect of the coder change (old versus new) on the same rungs: the old-coder batch had 3 attempts on `center` (1 pass) and 1 on
  `row_sum` (0), which is too little, and the load differs.
- Whether the slowdown from about 49 to about 12 tok/s came from the concurrent clients, from long contexts, or from the server; the pilot
  at one client did not show it.
- Whether the server honours the seed (not verified), and whether `linear_scan` or `rmsnorm` has a pass at 10 attempts.
- The per-attempt hardware result (this ran in the simulator only).
