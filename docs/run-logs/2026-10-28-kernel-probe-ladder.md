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
  numbers were measured with the old text, so they are context, not a control.
- Pilots and batches use different seed bases (pilot 2026110000, one attempt per rung), so no pilot attempt repeats a
  batch attempt. Pilot attempts are not counted in the batch table.

Wall budget per rung (the rule, `kernelladder.size_wall_budget`): 4 times the wall time of the slowest PASSING pilot
on that rung, at least 300 s and at most 1800 s. A rung whose pilot did not pass keeps 1800 s, because the question
there is whether more time lets the coder pass. A pass is judged on the kernel, not on the clock; a budget long enough
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
