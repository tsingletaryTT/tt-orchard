# Benchmark of ProprietaryLegal/Qwen3.8-27B-CJK-Trim against Qwen/Qwen3.8-27B

Date: 2026-10-26 to 2026-10-27 (UTC). Issue: tsingletaryTT/tt-orchard 85. Files: `evidence/2026-10-27-cjk-trim-benchmark/`.
The results sections are added after the full run. This part was written before it.

## Scope

The operator limited the job to about 3 hours of wall time (4 hours at most) and to the questions a vocabulary trim can change. Sets that were not run:
IFEval, HumanEval, MBPP. Sets that were run are small: GSM8K 100 items, BFCL simple and multiple 100 items each at 4 requests at a time and 50 each
at 1 request at a time, a CJK check of 40 prompts, a speed sweep. A difference of a few items on any of them is inside the noise.

## Conditions

Both models served through the 2-chip bundle `episod/qwen3.8-27b-dflash2-p300`, the drafter off and sampling on the host (the trim renumbers its
token ids, so the drafter cannot run). The base control's `run.sh` was built with `orchard.bench_control build` from the trim's `run.sh`; only the
model directory differs (lines 57 and 76), and the control record says the serving mode is the same. One board per model, run side by side, one
boot per model. Thinking as the chat template defaults it, temperature 0, 32768 tokens for GSM8K and BFCL, 8192 for the CJK check,
stop string `</s>` for the lm-eval task. Decode ran with `max_num_seqs` 4. Ports 8102 and 8211 (the first pilot used 8102, which the brief
listed as off limits; nothing else held it).

## Pilot (10 items per set, trim first, then the base)

- Trim: server ready in 270 s on a cold tensor cache (kernel cache warm). The tensor cache is 30 GB. Free disk went from 69 GB to 290 GB while
  the pilot ran (something else on the box freed space), so both models could run side by side. GSM8K 10 items took 117 s, BFCL 20 items
  233 s at 1 at a time and 178 s at 4 at a time. No request failed.
- Single-stream decode in the pilot: trim 20.3 to 20.6 tokens/s; base 16.1 to 17.0 tokens/s, with a time to first token of 1.6 s against 0.3 s.
- The pilot's CJK check (8 of the 40 prompts) already showed the main result: the trim answered Chinese and Japanese prompts with "It looks
  like you've sent a string of CJK characters that don't form a recognizable word", and the base answered them.

Projected full wall time per model: about 2.3 hours for the trim (the CJK check takes about 54 minutes, because the trim spends about four times
the tokens on CJK text) and less for the base. GSM8K stays at 100 items.

## Prediction (written after the pilot, before the full run)

- Tokenizer: English and code give identical pieces; CJK text costs about four times the tokens (measured below, before the pilot).
- GSM8K and BFCL: the trim is within 3 items of 100 of the base on each set. No set shows a difference that a sign test would call clear.
- CJK check: the trim fails the "uses the asked script" measure on most Chinese, Japanese and Korean prompts (at most 30 percent pass), and
  the base passes at least 90 percent. Mixed prompts that contain a few CJK words come out partly right for the trim.
- Speed: the trim decodes at least as fast as the base, because the pilot already showed 20 against 16.5 tokens per second.

## Results (full run, 2026-10-27 UTC)

Both models ran at the same time, the trim on board 0 and the base control on board 1, one boot each (ready in 132 s and 136 s on warm tensor caches).
Wall time of the sessions: trim 6,855 s, base 5,938 s. With the two pilots (1,516 s and 1,133 s) the chip time was about 4.3 hours. From the first pilot launch
to the last full result about 2.6 hours passed.

### Tokenizer (CPU, no chip)

500 strings per set; Wikipedia paragraphs (20231101 dump, first rows) for the four languages and chunks of this repository's Python files for code.
`tokenizer_check.py`, `tokenizer-check.json`.

| Set | tokens, trim | tokens, base | ratio | strings with identical pieces | characters per token, trim / base |
|---|---|---|---|---|---|
| English | 21,036 | 21,036 | 1.00 | 500 of 500 | 4.31 / 4.31 |
| Code | 75,172 | 75,172 | 1.00 | 500 of 500 | 3.85 / 3.85 |
| Chinese | 149,021 | 35,008 | 4.26 | 102 of 500 | 0.44 / 1.86 |
| Japanese | 224,685 | 48,618 | 4.62 | 30 of 500 | 0.36 / 1.65 |
| Korean | 129,468 | 33,996 | 3.81 | 7 of 500 | 0.46 / 1.75 |

Non-CJK text tokenizes to the same pieces. CJK text costs about four times the tokens; the largest single string was 6.6 times.

### CJK check (40 prompts, `cjk_check.py`, defined before the run)

Thinking on, temperature 0, 8192 tokens, 4 requests at a time. 10 prompts each in Chinese, Japanese and Korean and 10 mixed with English.

| Measure | trim | base |
|---|---|---|
| reply uses the asked script (at least 30% of letters; mixed: CJK and Latin both present) | 0 of 40 | 32 of 40 (zh 9, ja 9, ko 6, mixed 8) |
| reply well formed (finished with stop, not empty, no repeated characters) | 22 of 40 | 33 of 40 |
| reply cut off at 8192 tokens with no answer | 15 of 40 | 4 of 40 |
| identical replies | 2 of 40 | |

The trim wrote 44 CJK characters in 40 replies. In at least 14 of 40 replies it told the user that the message looked garbled, random or unrecognizable (counted by those words), in English or
Russian (some replies start with a stray `.sendMessage` or a Russian greeting). On a Japanese prompt asking for a translation of a Japanese sentence it answered
"Hello, world! This is a secret message encoded in multiple layers." In 15 prompts it spent all 8192 tokens thinking. The base model answered most prompts in the
asked language; its 4 cut-offs and 3 other malformed replies are also in `cjk-base.json`. Sample size: 40 prompts, one run each.

### Quality sets (small)

| Set | trim | base | only trim right / only base right (sign test p) |
|---|---|---|---|
| GSM8K, first 100 items, flexible-extract | 95 | 94 | 1 / 0 (1.0) |
| GSM8K strict-match (needs the `####` format) | 81 | 73 | not paired |
| BFCL simple, first 50, 1 at a time | 44 | 44 | 0 / 0 (1.0) |
| BFCL multiple, first 50, 1 at a time | 47 | 46 | 1 / 0 (1.0) |
| BFCL simple, first 100, 4 at a time | 74 | 76 | 12 / 14 (0.845) |
| BFCL multiple, first 100, 4 at a time | 75 | 84 | 9 / 18 (0.122) |

- The GSM8K and BFCL numbers show no difference between the models. 50 to 100 items cannot show a gap of a few points.
- BFCL at 4 requests at a time scores lower than at 1 for both models: the base 76 of 100 and 84 of 100 at 4, against 44 of 50 and 46 of 50 at 1 (the one-at-a-time
  sets are the first 50 items of the four-at-a-time sets, and the two were not paired). The trim shows the same drop (74 and 75 against 44 and 47). Issue 83 found the same drop on
  AREX-2. The base model shows it too, so it does not come from the trim. In each model's four-at-a-time run one request per BFCL set timed out after 900 s.
- One GSM8K base reply was flagged as a stop-string suspect; no request failed in any phase of either model; GSM8K cut-offs 0 for both.
- IFEval, HumanEval and MBPP were not run (the operator limited the job to about 3 hours).

### Speed (single stream, thinking on, 2 repetitions, greedy)

| Prompt (prompt tokens) | trim tok/s, first token | base tok/s, first token |
|---|---|---|
| short chat (64) | 17.0, 1.49 s | 16.8, 1.62 s |
| 700 words (967) | 17.1, 1.56 s | 16.6, 1.60 s |
| 3,000 words (3,892) | 16.8, 2.05 s | 17.1, 2.14 s |
| 12,000 words (15,457) | 15.7, 5.12 s | 13.3, 5.81 s |

Both ran at the same time, so the rows include the same neighbour load. The pilot, taken while the other board was idle, gave the trim 20.3 to 20.6 and the base
16.1 to 17.0 tokens per second with a first-token time of 0.3 s and 1.6 s; that difference in the pilot is not seen in the full run. The prediction "at least as fast"
holds as parity (within 3 percent) except for the longest prompt.

### Prediction against result

- Tokenizer: as predicted (identical pieces for English and code; 3.8 to 4.6 times the tokens for CJK).
- GSM8K and BFCL: as predicted, within 3 items except BFCL multiple at 4 at a time (9 items, p 0.12).
- CJK: as predicted for the trim (0 of 40, below the 30 percent I expected at most) and for the base (32 of 40, below the 90 percent I expected; the misses are
  cut-offs and mixed prompts whose correct answer has few or no CJK characters).
- Speed: parity, with a gap only at 15,000 prompt tokens.

## Card and gate

`tt-orchard package-card` wrote 25 measured rows from `benchmark.json` (identity check passed: the file names both models with revisions); I then added a
section "Benchmark against the base model, and what the vocabulary trim changes", the speed rows, a limitation about CJK text and about 4 requests at a time, and a
Not measured entry (`edit_card.py`). The package gate passes. The card says which sets are small (50 to 100 items) and which were not run (IFEval, HumanEval, MBPP, long
context, agent tasks). Nothing was uploaded, published or listed; `episod/qwen3.8-27b-cjk-trim-p300` on the Hub still has the old card until the operator uploads
`stages/7/package/qwen3.8-27b-cjk-trim-p300/README.md`.

## Slips and notes

- The first pilot used port 8102, which the brief listed as off limits; nothing held it. All later sessions used 8210 and 8211.
- The first launch of the wrapper was refused by the permission classifier; I reported that, and the operator then allowed it.
- The first CJK scoring used a mixed-prompt rule (CJK and Latin both present). A mixed prompt that asks for a translation can pass or fail that rule for reasons
  that do not concern the model, so the mixed row is the weakest of the four.
- Free disk rose from 69 GB to 290 GB during the pilot (not by me), which made the side-by-side run possible. Each tensor cache is 30 to 31 GB and was left in place.
- Files: `evidence/2026-10-27-cjk-trim-benchmark/` (`benchmark.json`, `cjk-trim.json`, `cjk-base.json`, speed and tokenizer files, the scripts). Paths are replaced by placeholders.

## Driver defects found in review (the run used the earlier version)

The benchmark ran with the first version of `session_all.py`. Copilot's review of the pull request found two defects in it, and the committed copy now has both fixes.
1. A failed `orchard.bench_session` step was printed and the driver went on to the next step and exited 0, so `session.sh` could not see a failed phase (including exit 7, a board off the bus).
   The fixed driver ends the session with the failing status; its `finally` stops the server and no later phase starts.
2. `session.sh` runs the driver under `timeout`, and its SIGTERM ended Python without running the `finally`. The server runs in its own session, so it could outlive the timeout and keep the chips
   while gozer released them. The fixed driver turns SIGTERM and SIGINT into `SystemExit`, so the `finally` stops the server's process group (SIGTERM, a grace period, SIGKILL, a check that the group is gone).

Could either have affected the recorded numbers? No. Every `bench_session` step of all four sessions (two pilots, two full runs) logged `exit 0`, and every phase in the recorded summaries has `ok: true`
with no request errors except the one timed-out BFCL request per set at 4 at a time that the results section already reports. No session was ended by `timeout`; all four finished by themselves and wrote `SESSION_ALL end`.
So neither defect was triggered. The risk was to a later run, not to these numbers.

Check of the fix, by hand with a fake checkout (a stub `orchard` package, a fake `run.sh` whose server sleeps, a fake `bench_session` that exits 7 or sleeps): the old driver exits 0 when the phase exits 7, and
after `timeout 4` two server processes are left; the fixed driver exits 7 for the failing phase and leaves no process in the server's group in either case (exit 124 under `timeout`). The repository has no test pattern for
evidence scripts, so this check is described here and not committed as a test.

Evidence note: `peek.py` (the helper that prints a session's phases) first crashed on a phase with no `scores` (lm-eval failed, timed out or wrote nothing). It now prints each phase's
`ok` and `problem` and goes on, and guards `scores`, `proxy` and `wall_s` the same way. Checked by hand on copies of the recorded trim summaries: one with `scores` removed (prints
`scores none`, keeps the proxy counts), one with `scores`, `proxy` and `wall_s` removed and `problem` set to a timeout message (prints `ok False`, the problem, and `none` for the rest),
and a missing summary file (prints `summary unreadable`); the real summaries print as before. `peek.py` only prints; no number in this log or on the card came from it
(they come from `compile_bench.py` and the summaries).
