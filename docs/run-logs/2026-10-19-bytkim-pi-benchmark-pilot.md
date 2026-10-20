# 2026-10-19: benchmark pilot, `bytkim/Qwen3.8-27B-pi` against `Qwen/Qwen3.8-27B`

Plan: `docs/superpowers/plans/2026-10-19-model-card-benchmark.md`. The operator said "don't use the HW yet", then "go"; the first session started at 00:58:53Z on 2026-10-20 (UTC).
Nothing was published, pushed or uploaded by the pilot. Raw files (per-item outputs, per-request logs, server logs) are in
`~/orchard-runs/bytkim--qwen3.8-27b-pi/benchmark-pilot/` and have not been redacted for sharing.

## What ran

Four sessions for the first pass (`pi-default`, `base-default`, `pi-off`, `base-off`) and four for the code pass (`code-*`), each under its own
`gozer run --chips 2 --exact 2` lease with a 4-hour timeout, started by `run_pilot.sh` and `run_pilot_code.sh` in that directory. One server boot per session.
Same bundle for both models (`episod/qwen3.8-27b-dflash2-p300`, bfp8 weights as shipped, DFlash2 drafter on, sampling on the device, `max_num_seqs` 4), the same
`run.sh` text apart from the model path, warm tensor caches. 20 items per task (a fixed seeded subset, the same items for both models), greedy decoding,
concurrency 4, `max_gen_toks` 8192, thinking as the chat template defaults it or turned off through the proxy. lm-evaluation-harness 0.4.13 in
`~/orchard-venvs/bench` (no token, scrubbed environment); the tool-call scorer is `orchard/toolcall_bench.py` on BFCL v3 `simple` and `multiple`, dataset revision
`61fc0608cfd8`.

### Quality, counts of 20 items

| Task (20 items each) | pi thinking default | base thinking default | pi thinking off | base thinking off |
|---|---|---|---|---|
| GSM8K (5-shot chat turns, flexible-extract) | 18/20 | 17/20 | 18/20 | 19/20 |
| GSM8K (strict-match) | 13/20 | 12/20 | 18/20 | 19/20 |
| IFEval, prompt-level strict | 19/20 | 19/20 | 18/20 | 19/20 |
| IFEval, prompt-level loose | 18/20 | 18/20 | 17/20 | 18/20 |
| BFCL simple (tool call) | 17/20 | 17/20 | 19/20 | 19/20 |
| BFCL multiple (tool call) | 19/20 | 19/20 | 20/20 | 20/20 |
| HumanEval (humaneval_chat) | 19/20 | 20/20 | 20/20 | 20/20 |
| MBPP (mbpp_chat) | 19/20 | 20/20 | 18/20 | 19/20 |

### Cost per session (seconds)

| Phase | pi thinking default | base thinking default | pi thinking off | base thinking off |
|---|---|---|---|---|
| gsm8k: wall s (tokens per request) | 92.2 (287) | 52.7 (254) | 37.0 (159) | 31.6 (164) |
| ifeval: wall s (tokens per request) | 246.2 (1545) | 251.0 (1754) | 68.4 (263) | 72.8 (304) |
| bfcl: wall s (tokens per request) | 77.6 (173) | 65.6 (190) | 20.1 (57) | 19.2 (57) |
| humaneval: wall s (tokens per request) | 88.9 (552) | 179.9 (917) | 24.3 (156) | 23.5 (156) |
| mbpp: wall s (tokens per request) | 400.3 (967) | 100.5 (611) | 29.4 (150) | 34.1 (167) |
| server ready s | 110.0 | 222.0 | 128.0 | 114.0 |
| replies cut off at 8192 tokens | 2 | 1 | 0 | 0 |

Counts are of 20 items, so one item is 5 percentage points; the standard error of a score near 0.9 is about 0.07. Perfect scores have a standard error of 0 at
this size, which understates the uncertainty.

## What the numbers show

- At 20 items per task the fine-tune and the base cannot be told apart on any task, with thinking on or off. Every difference is one or two items.
- Turning thinking off cut a session from 7 to 11 minutes to about 3 to 4.5 and did not lower any score by more than one or two items.
- The fine-tune missed one more item than the base on HumanEval and MBPP with thinking on, and one more on MBPP with thinking off. Twenty items cannot say
  whether that is a pattern.
- On the tool-call tasks the two models made the same call on all 40 items with thinking off, and on 37 of 40 with thinking on.

## Findings about the instrument

1. The harness's `humaneval_instruct` scored the fine-tune 0.0 on 20 HumanEval items, with 12 of 20 stored replies empty. It expects a prefilled assistant turn and
   stops at `\ndef`. `humaneval_chat` and `mbpp_chat` (`orchard/lm_eval_tasks/`) replace it. A positive control (the dataset's own solutions) scores 1.0 on 40 items
   of each, a stub scores 0.0. The first run of that control scored 0 because its replies lacked `"index": 0` in each choice, which makes lm-eval turn a reply into an
   empty string without an error.
2. GSM8K strict-match is much lower than flexible-extract with thinking on (13 and 12 of 20 against 18 and 17) and equal with thinking off. The strict filter wants
   the `####` answer format, which a thinking reply does not follow, so strict-match with thinking on measures format.
3. The server prints the drafter's acceptance per request in `server.log` (`[dflash2-serve] ... accept 3.95/7 -> 4.93 tok/iter, 59.0 tok/s in-model`). The packaging
   log said it prints none. These lines are not yet read into the results.
4. Replies cut off at 8192 tokens: 2 for the fine-tune and 1 for the base with thinking on (IFEval and MBPP), none with thinking off.

## What a full run costs

From the pilot's per-item wall times (concurrency 4), for GSM8K 1,319, IFEval 541, HumanEval 164, MBPP 500 and BFCL 600 items: about 7.0 hours for the fine-tune and
4.3 for the base with thinking on, and about 1.6 and 1.5 with thinking off. That is 11.3 hours for both models with thinking on and 3.1 with thinking off. These are
extrapolations from 20 items; the MBPP time for the fine-tune (400 s) comes from two long replies and may be high.

## Not done

- The speed sweep (input length 1K, 8K, 32K by concurrency 1, 2, 4), and drafter acceptance read from the server log.
- A full-size run. The plan needs the operator's choice of thinking setting and task sizes first.
- MMLU-Pro and a long-context task.
- Any change to the package card.
