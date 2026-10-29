# Staged construction: does building a kernel in checked steps help the coder? (2026-10-28)

Issue 176. Code: pull request 199 (`kernel-probe --staged`, `orchard/kernelstaged.py`).

## Outcome and status

Pre-registered before the first counted attempt. Results are added below the line "Results" after the runs.

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

(to be filled in after the runs)
