# BAAI/AREX-2 graft: drafter acceptance, BFCL with and without the graft (2026-10-26)

Issue 14 left four things open: drafter acceptance, IFEval and tool-calling comparisons under one token limit, 4 chips, and a model other than AREX-2.
This log covers the first two. Everything ran on one 2-chip board (a QuietBox 2 board, two Blackhole p300c chips), each session under its own
`gozer run`-style lease, with the code of branch `graft-acceptance` copied to a scratch directory first so edits could not change a session.

## Where acceptance comes from

Issue 14 said the server prints no counters, and `measure_serving.py` carried the same sentence. That was wrong for the DFlash2 serving code: it
logs one `[dflash2-serve] slot N session: ...` line when a request ends, with the iterations, the committed tokens and a histogram of how many drafted
tokens were accepted per iteration. `bench_session.py` already read these lines for each benchmark phase (the bytkim and AREX-2 benchmark runs).
The graft trial, which times the configurations through `measure_serving.py`, did not. It does now (version 0.8.44): the rate is accepted over drafted
tokens for the lines written while the timed prompts ran, and the trial result, the decision file, the package card and the stage 8 bundle carry it.
The choice between graft and safe does not read it.

## Measured, graft on (drafter and on-device sampling, nearest model's 15 `mtp.*` tensors grafted)

| Where | Requests | Iterations | Draft length | Accepted over drafted |
|---|---|---|---|---|
| `measure_serving` prompts (4 prompts, 3 repetitions each, one request at a time) | 12 | 525 | 7 | 0.479 |
| BFCL simple and multiple (300 items, 4 at a time) | 300 | 6,854 | 7 | 0.644 |
| IFEval, full (541 items, earlier run) | 541 | 277,665 | 7 | 0.598 |
| GSM8K, full (1,319 items, earlier run) | 1,319 | 60,378 | 7 | 0.549 |

Accepted tokens per iteration are 3.35 on the measure prompts (the other rows give 3.84 to 4.19 and 4.5 on BFCL). The drafted-token counts are totals over
slots, so each row is one share over all its requests. It is not the average of per-request shares. The 4-prompt row is a small sample: the coding prompt and the chat prompt
have different acceptance. The rows come from one run each. bytkim/Qwen3.8-27B-pi, which ships its own head, was 0.46 to 0.60 over its benchmark phases
(`docs/run-logs/2026-10-20-bytkim-pi-benchmark-full.md`). No new measurement was taken for it here. The safe configuration has the drafter off, so
`measure_serving` reports its acceptance as not measured, which the real run shows.

Single-stream decode speed on the same four prompts (median over repetitions): graft 49.8, 51.1, 50.3 and 28.8 tokens/s, safe configuration 16.4, 16.8,
17.3 and 17.1; the median of the four prompt medians is 50.05 against 16.95 tokens/s, 2.95 times. First token 0.29 to 0.30 s for the graft. Server ready
after 804 s (graft, empty tensor cache) and 326 s (safe configuration, empty tensor cache, warm kernel cache).

## BFCL, graft against no graft, same items

The same 150 `simple` and 150 `multiple` items (fixed seed, dataset revision `61fc0608cfd8`), thinking default, 32,768-token limit, temperature 0, same
prompts. "Plain" is the safe configuration (drafter off, sampling on the host). Counts are passed items; the p value is a two-sided exact sign test on the items
only one side passed.

| Configuration | Requests at a time | simple (150) | multiple (150) |
|---|---|---|---|
| graft | 4 | 136 (0.907, standard error 0.024) | 144 (0.960, 0.016) |
| plain | 4 | 107 (0.713, 0.037) | 136 (0.907, 0.024) |
| plain | 1 | 137 (0.913, 0.023) | 143 (0.953, 0.017) |

- Graft against plain at 4 at a time: simple 32 items only the graft passed, 3 only plain, p = 4e-7; multiple 8 and 0, p = 0.008.
- Graft (4 at a time) against plain at 1 at a time: simple 0 and 1, p = 1; multiple 1 and 0, p = 1. No difference is shown.
- Plain at 1 against plain at 4: simple 32 items only the single-stream run passed, 2 only the 4-at-a-time run, p = 7e-8; multiple 8 and 1, p = 0.04.

The graft is not what separates them. The plain configuration returns damaged tool calls when four requests run together: replies with the argument text
cut and a stray `<tool_call` marker or a `<parameter=...>` tag inside a string (for example `"coord277.62\n\nThe distance between the user asks: ..."` where
a list of two numbers belongs), tool names with the arguments pasted onto them, and arguments left out. The 300 requests produced 79,622 completion tokens
(one reply reached the 32,768-token limit and two had no finish reason) against 37,708 for the graft and 38,312 for plain one at a time. Failure reasons for
plain at 4: 10 `missing_required`, 11 `extra_param` and 4 `wrong_name` on `simple`, against none of those three for the graft. The cause is not known. It
shows up with the drafter off and sampling on the host and does not show up with them on, and it goes away at one request at a time. The first four prompts of
`measure_serving` (one at a time) gave the same tool calls for both configurations. I did not run the graft at one request at a time, or the base model's plain
configuration, so a bug of the base bundle's non-drafted path cannot be told apart from a fault of this fine-tune's serving. The earlier AREX-2 benchmark ran
its plain configuration at 4 at a time too.

## IFEval

The existing comparison (full IFEval, 541 items, strict prompt-level): graft 483 (0.893), plain 459 (0.848); 52 items only the graft passed, 28 only plain,
sign test p = 0.0097. The plain run used an 8,192-token limit and its empty replies were asked again at 32,768 tokens, which gives the same replies as a
32,768-token run for greedy decoding unless the answer takes more than 8,192 tokens, so the limits match in effect. It was not run again from scratch.

Given the BFCL result, I asked the plain configuration at one request at a time for 26 of the 52 graft-only items (every second item by document id: 0, 47, 57,
80, 85, 103, 134, 157, 167, 180, 195, 222, 249, 266, 284, 341, 347, 365, 389, 400, 407, 423, 438, 470, 476, 484; 32,768-token limit, thinking default).
It passed 20 of the 26 (strict prompt level) and failed 0, 157, 347, 389, 438 and 476. One reply reached the token limit. Taken alone this does not say the
IFEval gap is the same effect: the items were picked because plain failed them once, so some would pass a second run of any configuration, and I did not
repeat the 26 items at four at a time or run the graft again as a control. It does mean that most of the graft-only items are not items the safe
configuration cannot answer. The 4-at-a-time GSM8K comparison (0.938 against 0.936, p = 0.77) showed no gap and is not affected by this.

## Not done

- A model other than AREX-2. The kept run directories of the earlier no-`mtp.*` runs (Clef, RAG-Gate-27B) hold no stage directories, model directories
  or tensor caches (deleted for disk space), so neither the trial nor a by-hand equivalent could start without two more cold conversions of 30 to 55 GB each
  and a disk that had 130 GB free. None was brought up under the run directory root with a model that ships no `mtp.*` tensors since.
- 4 chips. One free board at a time: the other board was held by another agent's long job for the whole session. A 4-chip configuration needs both.
- The graft at one request at a time, and the base model's safe configuration at 1 and 4 at a time (see the BFCL section). Issue 83.
- The trial itself on AREX-2 (the supervisor's `graft_trial.py`). The numbers above come from the same `measure_serving.py` the trial calls, run by hand on a
  graft built with the shipped `prepare_swap.py`.
