# Staged construction: does building a kernel in checked steps help the coder? (2026-10-28)

Issue 176. Code: pull request 199 (`kernel-probe --staged`, `orchard/kernelstaged.py`).

## Outcome and status

Arms W and S were NOT run: superseded by the operator's decision to attempt a real model instead. The pre-registration below
stays as written; no attempt of either arm exists. The capability check through the Telegraph helper was run and is reported
under "Results". The staged probe itself is in pull request 199 and is tested there; nothing here measures whether it helps
the local coder.

## Pre-registration (written 2026-10-28, before any coder attempt of this experiment)

### Question

The coder (Qwen3-Coder-Next, the p300 fused-decode bundle) passes the elementwise ops and `row_sum` in the capability
ladder (PR 177) and fails `center` (subtract the row mean and broadcast it back: numeric mismatch, PCC about 0) and
`rmsnorm` (shape mismatch "Tensor shape (32, 32) does not match Block shape (1, 8)"; 0 of 29 earlier attempts passed). The
operator's reasoning: staged construction "still seems like how a person would build it". Does building the same kernel in
stages, each ending at a simulator check against a reference for that step, give more passes than asking for the whole
kernel?

### The two arms

Same coder, same server, same tt-lang pin, same simulator, same skill text for the base prompt, `--library none`
(no worked kernel in `reference/`, as in the ladder runs), `--phase sim`, `--no-agent-hw`, temperature 0.7.

| | arm W (whole kernel) | arm S (staged) |
| --- | --- | --- |
| command | `kernel-probe --op OP` | `kernel-probe --op OP --staged` |
| agent conversations per attempt | 1 | one per stage, each a fresh conversation |
| check | the shipped check for the whole op | the shipped check with a spec and reference for the stage; the last stage is the unmodified whole-op check |
| turns | 40 | 40 divided over the stages (center 4 stages of 10; rmsnorm 5 stages of 8) |
| wall clock | 1800 s | 1800 s divided over the stages (center 450 s each; rmsnorm 360 s each) |
| starting file | none | stage 1 none; stage k+1 starts from the exact bytes that passed stage k |
| a failing stage | n/a | ends the attempt; the record names the stage |

The two arms get the same total turns and wall time. A stage that ends early does not hand its unused budget on, so arm S
can run out of turns in a stage where arm W would still have had turns.

Stage tables (instructions are in `orchard/kernelstaged.py`; each names only its step and holds no code):

| op | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 |
| --- | --- | --- | --- | --- | --- |
| center | copy: out = x | scale: out = x / 256 | row_mean: out (64, 32), column 0 holds each row's mean | center: the whole op | (none) |
| rmsnorm | copy: out = x | square: out = x * x | row_sum_of_squares: out (64, 32), column 0 holds each row's sum | scale_rows: out = x * rsqrt(mean(x^2) + 1e-6) | rmsnorm: the whole op, with w |

`center` has four stages because its "broadcast back" step is the whole op. Six fixture kernels, one per intermediate
stage, pass their stage's check on the real simulator (`tests/test_kernelstaged.py`), so each stage's reference and bars can
be met.

### Fixed settings

- Coder: `Qwen/Qwen3-Coder-Next`, p300 fused-decode bundle, served on port 8011 by the ladder agent. This experiment is a
  client of that server only. The container and its lease are not touched.
- Code: both arms run from one exported copy of the branch of PR 199 (commit recorded in the results section).
- Attempts: 10 per arm per op, 40 attempts in all, one attempt per probe invocation (`--attempts 1`).
- Temperature 0.7. Seeds: attempt k (k = 1 to 10) of each arm and op sends seed 2026110300 + k (the probe adds the attempt
  number to `--seed`, so each invocation is passed `--seed 2026110300 + k - 1`). The seed is the same in both arms. Whether the
  server honours the seed is not verified. In arm S every stage of attempt k sends the same seed.
- Order: interleaved, one process at a time against the coder: W-center-1, S-center-1, W-rmsnorm-1, S-rmsnorm-1,
  W-center-2, and so on. The next invocation starts when the previous one exits.
- Decode speed: the probe measures the coder's decode speed before every attempt (`speedwatch.probe_check`) and keeps it
  in the record. `--allow-slow-coder` is passed, so an attempt runs even when the ladder's processes slow the coder; the
  speed is in the table.
- A pilot (one staged `center` attempt with seed 2026110399, watched through its first stage) is run first and not counted.
- Start condition: no `kernel-probe --ladder` process is left (checked with `ps`). If the ladder is still running 3 hours
  after the code is ready, the experiment starts anyway with one process of ours and the load is recorded.

### Measurements per attempt

passed (the unedited whole-op check, `passed_sim`), failure class (the check's category, with `tt_lang_error` split into
deadlock and "dataflow step"), the stage reached (arm S: the last stage started), the stage it failed at, turns, wall
seconds, status, decode speed before the attempt. Analysis: `2026-10-28-staged-construction-analysis.py`.

### Success criteria (fixed now)

Per op, arm S passes the op if either:
1. arm S has at least 1 pass and arm S passes minus arm W passes is at least 3 of 10, or
2. arm S reaches the final stage (stage 5 for `rmsnorm`, stage 4 for `center`) in at least 5 of 10 attempts, and arm W has
   no pass in its 10 attempts (read here as "arm W never gets past its first failing error"; the count of W attempts
   whose failure is a shape mismatch is printed beside it).

If neither holds for an op, the report says staged construction did not help on that op at this sample size. Counts only,
never rates. Ten attempts per arm cannot show a difference of one or two passes: a rerun of the same arm could differ by
that much.

### Capability check through the Telegraph helper (same pre-registration)

Question: can a larger model pass the same op with the same prompt and the same check? Five attempts of arm W on `rmsnorm`
by the Telegraph helper (`anthropic/claude-sonnet-5-5` only; the model list of this check holds that one model).

- Prompt: the whole-kernel probe's system prompt (`orchard/skills/kernel-probe.md`), the user message of arm W, and the
  contents of every file the coder can read in its attempt directory (`kernel_spec.json`, `kernel_check.py`,
  `reference/*`), put into the request text, because the helper has no tools. It is one answer per attempt, with no
  simulator feedback in between. That differs from arm W, where the coder gets the check's answer after each write.
- The answer's first Python code block is written as `kernel.py` and judged by the same unedited check (`--library none`).
- Five requests, each carrying its attempt number so the cache cannot answer one with another. At most 6 calls for this
  problem (the config's cap), cost recorded from the ledger. Before the first call a line is appended to the telegraph
  decisions file saying what triggered it, what was tried locally at least twice and why it is worth it.
- Skipped, with the reason, if the helper is unavailable or a cap refuses.

### Baselines for context

PR 177's ladder run on the same coder: `center` and `rmsnorm` as in the question above; the earlier whole-kernel
`rmsnorm` attempts passed 0 of 29. The arms below are the comparison; the ladder numbers are context only.

---

## Results

Arms W and S: not run (see the status above). No pass counts, stage histograms, failure classes or decode speeds exist for them.

### Capability check through the Telegraph helper (run 2026-10-28)

As pre-registered: five one-shot whole-kernel `rmsnorm` answers from `anthropic/claude-sonnet-5-5` (one model in the list),
temperature 0 (the helper's setting), no tools and no simulator feedback between requests, `--library none`. The prompt held the
probe skill text, arm W's user message and the text of `kernel_spec.json` and every file in `reference/` (the specification, the
guide, the examples and the two checkout examples). A line was added to the telegraph decisions file first. Each answer's first
code block was written as `kernel.py` and judged by the unedited shipped check on the simulator. The helper wraps the prompt in its
own system text, so the request is not byte-for-byte the coder's. The 5 requests differ only by their attempt number.

| attempt | passed | tokens in | tokens out | seconds | cost USD |
| --- | --- | --- | --- | --- | --- |
| 1 | yes | 80798 | 3444 | 24.4 | 0.196036 |
| 2 | yes | 80798 | 2898 | 20.7 | 0.190576 |
| 3 | yes | 80798 | 3004 | 20.2 | 0.191636 |
| 4 | yes | 80798 | 3221 | 21.7 | 0.193806 |
| 5 | yes | 80798 | 2765 | 18.7 | 0.189246 |

Total: 5 of 5 passed, 5 calls, $0.961 (0.961 = sum of the column). Failure classes: none.

Similarity of the five passing kernels to the orchard library kernels and to the checkout's examples
(`2026-10-28-staged-construction-similarity.py`: comments and blank lines removed, difflib ratio over lines, flag at 0.80):

| attempt | best library match | best checkout-example match | flag |
| --- | --- | --- | --- |
| 1 | 0.23 (rmsnorm.py) | 0.18 | ok |
| 2 | 0.20 (rmsnorm.py) | 0.15 | ok |
| 3 | 0.28 (rmsnorm.py) | 0.21 | ok |
| 4 | 0.33 (rmsnorm.py) | 0.22 | ok |
| 5 | 0.33 (rmsnorm.py) | 0.22 | ok |

Not determined: whether the same prompt in an agent loop with check feedback (the next entry) changes this; whether the
local coder would pass with the specification text pasted whole into its prompt (it reads files in parts).

---

## Agent-loop capability check: pre-registration (written 2026-10-29, before the first call of this check)

Operator decision: stop the W/S experiment; instead repeat the capability check with a loop that gives the helper the same
simulator feedback the local coder gets.

Settings, fixed now:
- Targets: `center` (5 attempts) and `linear_scan` (5 attempts). Calls are interleaved (center 1, linear_scan 1, center 2, ...).
- Model: `anthropic/claude-sonnet-5-5` only, through the Telegraph helper under its rules, temperature 0 (the helper's setting).
  Prices from the config: $2.00 in and $10.00 out per million tokens. A line goes into the telegraph decisions file before the
  first call.
- Loop: each turn is one Telegraph `ask`, because the helper has no tools. The harness (`2026-10-28-staged-construction-agent-loop.py`)
  writes the reply's first python code block to `kernel.py` with the probe's own write tool, which runs `python3 kernel_check.py`
  and returns its output (the same text the coder sees after a write). The next request holds the prompt, the files, and every
  earlier kernel with its check output. The helper's own system text differs from the coder's.
- Turns: at most min(8, 40) = 8 per attempt. The config's cap of 6 calls per problem is not raised, and each attempt is its own
  problem (`staged-176-agent-<op>-<k>`), so an attempt can use at most 6 turns. An attempt ends at the first `PASS`, at the turn
  cap, at a refusal or when the spend reaches the budget.
- Verdict: the final `kernel.py` is judged by the unedited authoritative check (`--library none`, the shipped check, same bars).
- References: the skill text, `kernel_spec.json`, `reference/eltwise_add.py`, `reference/eltwise_broadcast_reduce.py` and
  `reference/TTLangSpecification.md`. `reference/guide.md` and `reference/examples.md` (74 KB, written for an older tt-lang, and
  the skill says so) are left out to hold the cost. That is a difference from the coder's directory. `--library none`: no worked
  kernel is given.
- Budget: about $6 in all. The run stops starting calls when its own spend reaches $5.50. Expected spend: under $2 (one request is
  about 30,000 tokens in and 3,000 out, about $0.09).
- Recorded per attempt: passed, failure class (the check's category), turns, calls, tokens, cost, how the loop ended.
- Similarity: each passing kernel is compared with the orchard library kernels and the checkout's examples
  (`2026-10-28-staged-construction-similarity.py`: comments and blank lines removed, difflib ratio over lines, flag at 0.80, a
  near-copy). The files of the passing kernels are kept in the run directory.
- No claims beyond the counts. Five attempts per target cannot show a small difference.

### Agent-loop capability check: results (run 2026-10-29)

As pre-registered: Sonnet 5.5 through Telegraph, one Telegraph call per turn, the probe's write tool running the check after each
write, at most 6 turns per attempt (the config's cap of 6 calls per problem, not raised; one problem per attempt),
`--library none`, the shipped check as the verdict. The harness used an exported copy of the PR 199 code at commit 3f028f4 (its
check and write tool are the whole-kernel probe's).

| target | attempts | passed | failure classes | turns used (per attempt) |
| --- | --- | --- | --- | --- |
| center | 5 | 5 | none | 1, 1, 1, 1, 1 |
| linear_scan | 5 | 0 | sim_gap 3, no_kernel 2 | 6, 6, 0, 0, 6 |

Each `center` attempt passed on its first kernel (the in-loop check printed PASS) and the unedited check agreed.

`linear_scan`, per attempt:

| attempt | turns | how it ended | final check |
| --- | --- | --- | --- |
| 1 | 6 | call cap 6 of 6 reached | sim_gap: the simulator has no `ttl.raw_element_read` |
| 2 | 6 | call cap 6 of 6 reached | sim_gap: the simulator has no `ttl.raw_element_write` |
| 3 | 0 | first answer empty: finish_reason `length` after 8000 output tokens | no_kernel |
| 4 | 0 | first answer empty: finish_reason `length` after 8000 output tokens | no_kernel |
| 5 | 6 | call cap 6 of 6 reached | sim_gap: the simulator has no `ttl.raw_element_read` |

A `sim_gap` verdict means the kernel uses a documented function the simulator lacks, so it can be judged only on hardware; the
probe's check counts it as not passed in the simulator. Hardware was not used. Whether those three kernels would pass on a chip
was not determined (the 2026-10-27 note says data movement kernels cannot compute on the values they read).

Cost (from the helper's per-call usage and prices of $2 per million tokens in and $10 per million out):

| part | calls | cost USD |
| --- | --- | --- |
| center, 5 attempts | 5 | 0.579 (0.116, 0.114, 0.114, 0.118, 0.116) |
| linear_scan, answered calls | 18 | 2.790 (0.922, 0.935, 0.933 for attempts 1, 2, 5) |
| linear_scan, the two empty answers (8000 output tokens each, 49303 in) | 2 | 0.357 |
| total of this check | 25 | 3.726 |

The harness's own running total showed $3.369 because the two empty answers carry no cost in the helper's record. The figure above
adds their token counts at the stated prices (49,303 x $2 per million + 8,000 x $10 per million = $0.1786 each). With the first
check ($0.961) the two capability checks cost $4.69, under the $6 limit. A request is about 49,000 tokens in at turn 1 and grows to
about 56,000 by turn 6.

Similarity of the five passing `center` kernels to the orchard library kernels and the checkout's examples
(`2026-10-28-staged-construction-similarity.py`, flag at 0.80):

| attempt | best library match | best checkout-example match | flag |
| --- | --- | --- | --- |
| 1 | 0.34 (center.py) | 0.18 | ok |
| 2 | 0.35 (center.py) | 0.18 | ok |
| 3 | 0.24 (center.py) | 0.18 | ok |
| 4 | 0.34 (center.py) | 0.18 | ok |
| 5 | 0.34 (center.py) | 0.18 | ok |

No kernel was flagged as a near-copy. The score is a line-level ratio, so a copy with renamed identifiers would score low.

Deviations: `reference/guide.md` and `reference/examples.md` were left out of the prompt (cost); the helper's system text and
temperature 0 differ from the coder's; the helper has no tools, so the harness wrote the kernel and ran the check on its behalf.

Not determined: whether the local coder passes `center` in this loop (it did not in the capability ladder, PR 177); what a
hardware run of the three `sim_gap` kernels gives; whether the two empty answers on `linear_scan` would have been kernels with
a larger output limit.
