# Pinned reference set against the old one, local coder (2026-10-29)

Question: does the pinned reference set (PR 198) help the local coder write a tt-lang kernel that passes the simulator? A cheap
test, not a benchmark. Coder: Qwen3-Coder-Next on port 8011, a client only; its owner restarted it near the end of the run.

Setup. Ops `center` and `rmsnorm`, temperature 0.7, one attempt per probe process, `--library none --phase sim --no-agent-hw`,
interleaved old, new, old, new, one process at a time. Attempt k of every cell used seed 2026110400 + k. Each attempt had a 900 s wall
budget, set with the new `--attempt-wall-s 900` (it is checked between turns, so attempts ran up to 1166 s). `--allow-slow-coder` was
on, so a slow or shared coder never stopped a probe; the speed before each attempt is recorded. Old is the new `--reference-source
skill` (the skill's guide.md and examples.md, no extra kernels). New is the default (`checkout`).

Reference sets, in bytes. Old: guide 34,958, examples 37,187, specification 92,238, two example kernels 12,909: total 177,292.
New: guide 40,934, examples 10,999, specification 92,238, two example kernels 12,909, five extra kernels 34,885: total 191,965.

| op | set | attempts run | simulator passes | failures by class | turns per attempt | decode tok/s before each attempt |
| --- | --- | --- | --- | --- | --- | --- |
| center | old | 5 | 2 | tt_lang_error 3 | 26, 26, 35, 29, 31 | 99.9, 26.2, 69.2, 67.9, 23.6 |
| center | new | 5 | 2 | tt_lang_error 2, shape_mismatch 1 | 18, 26, 31, 25, 24 | 84.1, 67.9, 67.8, 44.9, 72.3 |
| rmsnorm | old | 4 (+1 cut) | 0 | numeric_mismatch 2, tt_lang_error 1, shape_mismatch 1 | 36, 27, 39, 26 | 70.2, 10.5, 12.8, 12.9 |
| rmsnorm | new | 4 (+1 not run) | 2 | numeric_mismatch 1, tt_lang_error 1 | 39, 24, 20, 25 | 12.8, 12.7, 12.3, 12.4 |

Incomplete cells. The fifth rmsnorm old attempt stopped after 11 turns with a failed model request (the coder dropped the
connection while it restarted, the probe's status was `error`, and the probe process then stayed alive for 5.6 hours): it is left out
of the table. The fifth rmsnorm new attempt was refused because the coder was not answering, and the run stopped there. So the two
rmsnorm cells have n=4.

Reading. Over the cells that completed, old passed 2 of 9 attempts (center 2 of 5, rmsnorm 0 of 4) and new passed 4 of 9 (center 2 of 5,
rmsnorm 2 of 4). On `center` the two sets tie; the rmsnorm difference is 0 of 4 against 2 of 4, which at this size is within what
five-attempt cells give by chance, and the coder's load differed between the two ops (shared and slow at about 12 tok/s through the rmsnorm
block, 24 to 100 tok/s in the center block), so each op's own interleaving is the only fair comparison. Most failures in both sets
are tt_lang_error (the simulator rejects the kernel), and most attempts ran to the wall budget whether they passed or not.
