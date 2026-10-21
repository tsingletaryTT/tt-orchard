# Model card benchmark: plan and pilot on `bytkim/Qwen3.8-27B-pi`

Date: 2026-10-19. Status: plan written; the pilot has not run. The operator said not to use the hardware yet, so everything below that
needs a chip is waiting.

## Why

The `bytkim/Qwen3.8-27B-pi` package card has two kinds of evidence: 32 of 32 tokens matched the CPU reference, and decode speed on four
prompts. The first shows the weights load and decode. The second does not support a claim about the fine-tune, because the text differed
between the fine-tune and the base on three of the four prompts and a speculative drafter makes decode speed depend on the text. The
author's benchmark charts are unverified. A card that offers a model for coding agents needs quality numbers for base and fine-tune
measured the same way on the same server.

## What the card should be able to say after this

1. How the fine-tune scores against the base on coding, tool calling, instruction following and one general task, with sample counts and
   intervals, on the hardware and bundle the package ships for.
2. How fast the package serves at several input lengths and several concurrent requests, with the drafter state stated.
3. What was not measured.

It does not need to say the fine-tune is better. If the numbers are equal within their intervals, the card says that.

## Fixed conditions

These come from the packaging run (`docs/run-logs/2026-10-19-bytkim-pi-packaging.md`) and apply to every number below.

- One 2-chip board (P150x2). Bundle `episod/qwen3.8-27b-dflash2-p300`: bfp8 weights as shipped, DFlash2 drafter on, sampling on the device,
  `max_model_len` 262144, `max_num_seqs` 4. The server runs at most 4 requests at once, so 4 is the top concurrency.
- Base and fine-tune use the same `run.sh` text. Only the model path differs (the two lines the packaging log lists).
- Both tensor caches exist and are warm: the fine-tune's under `~/orchard-runs/cache/bytkim--qwen3.8-27b-pi/2chip-.../tt_cache`, the base's
  under `/mnt/bonus/models/orchard-runs/bytkim-package/base-tt-cache`. A boot takes about 2 to 4 minutes with them. Disk has 38 GB free on
  root and 46 GB on `/mnt/bonus`, so no third cache is built.
- Hardware only through `gozer run --chips 2 --who claude:bench-pilot --reason ... -- timeout <s> <command>`. No `tt-smi -r` by hand.
  A chip stays leased for the whole benchmark of one model, then is released.

## Comparability rules

A comparison is reported only when all of these hold, and the result file records each one.

- Same bundle, same `run.sh` text apart from the model path, same drafter and sampling settings.
- Same harness version, same task versions and same dataset revisions. The harness is `lm-evaluation-harness` in its own venv
  (`~/orchard-venvs/bench`), never the shared Tenstorrent venv.
- Same items in the same order (fixed seed, `--limit` chosen once), same prompt template, same sampling (greedy unless a task needs
  otherwise), same generation limit.
- Base and fine-tune run in the same session, one after the other, so a change on the box does not land on one model only.
- The thinking mode setting is the same for both and is stated. The chat template defaults to thinking, which multiplies the tokens per
  item. The pilot measures that cost before the full run chooses.

## Tasks

| Area | Task | Harness | Why |
|---|---|---|---|
| Coding | HumanEval (164) | `humaneval_chat` (`orchard/lm_eval_tasks/`) | The harness's `humaneval_instruct` scored a chat reply as empty (prefill and stop strings); see the pilot log. |
| Coding | MBPP (500) | `mbpp_chat` (`orchard/lm_eval_tasks/`) | A second coding set with different problems. |
| Instruction following | IFEval (541) | lm-eval `ifeval` | Rule-checked, no judge model. |
| Math / general reasoning | GSM8K (1,319) | lm-eval `gsm8k`, 5-shot as chat turns | Cheap check that the fine-tune did not lose general ability. |
| Tool calling | BFCL v3 `simple` (400) and `multiple` (200), Apache-2.0, dataset revision `61fc0608cfd8` | `orchard/toolcall_bench.py` | The model is sold for agents. lm-eval has no tool-call task. |
| General knowledge | MMLU-Pro subset | lm-eval `mmlu_pro` | Only if the pilot shows time for it. |

Code execution tasks (HumanEval, MBPP) run model-written code, so they run in the bench venv inside a disposable directory with a
timeout and no credentials in the environment (`HF_ALLOW_CODE_EVAL=1` is the harness's own gate).

The tool-calling scorer is the only new code. It reads the BFCL question and function definitions, sends them with `tools`, and scores
the returned call by function name and by each argument value against the ground truth. It does not execute anything. Its tests come
first and are checked by mutation.

## Performance sweep

Fixed prompts built from one text of known length. Input lengths 1K, 8K and 32K tokens, output 256 tokens, concurrency 1, 2 and 4,
three repetitions after one warm-up, medians with ranges. Recorded: time to first token, decode tokens per second per request and in
total, and the drafter state. Drafter acceptance is reported only if `/metrics` shows spec-decode counters; the server did not export
them on 2026-10-19. The existing `orchard/measure_serving.py` (branch `bytkim-package`) is the starting point; it takes one request at
a time and needs a concurrency option and a prompt-length option.

## Pilot (first, small, watched)

Goal: measure what a full run costs and find the failures, with a small sample.

1. No hardware: build the bench venv, run the harness against a stdlib fake OpenAI server (checks the plumbing, the dataset downloads,
   the scorer and the result files), and write the tool-call scorer test first. Nothing here touches a chip.
2. With the operator's go-ahead for hardware: boot the fine-tune under one lease. Run, in order, 20 items each of GSM8K, HumaneEval,
   IFEval and BFCL simple, with thinking as the template defaults it, at concurrency 4. Record items per minute and tokens per item.
3. Repeat the same 20-item set with thinking off, to see how much quality and time the setting changes.
4. Stop the server, release, boot the base, repeat. Same session.
5. From the measured items per minute, write the full-run time estimate and choose `--limit` per task. A full run is not started
   without a time estimate in the result file and the operator's go-ahead.

Pilot stop rules (watchers, not timers alone): the whole pilot has a 4-hour hard timeout under `gozer run`. The client stops a task when
no response arrives for 15 minutes, or when 3 consecutive items return an error, or when two items in a row return the same text
for a different question. Each stop writes its reason.

## Outputs

- `benchmarks/<package>/results.json`: per task and per model, n, score, standard error, tokens per item, wall time, all the
  comparability fields above, harness and dataset versions.
- Raw per-item outputs saved beside it (redacted the way package evidence is), so a number can be recomputed.
- Card rows cite `results.json` through the same card-numbers file the packaging used, so the package gate sees measured values.
- The card's "not measured" list is regenerated from what was skipped.

## Tooling built before the pilot (no hardware used)

Status 2026-10-19. All of it is tested without a chip, and each module's guards were checked by mutation.

- `orchard/toolcall_bench.py`: the BFCL scorer (50 tests, 47 mutations, none survived).
- `orchard/bench_proxy.py`: a loopback proxy between the harness and the server. It adds `chat_template_kwargs` (thinking off), and records tokens,
  finish reason, reasoning length and latency for each request (18 tests, 20 mutations, one equivalent survivor).
- `orchard/bench_session.py`: starts a bundle's `run.sh` with the model's tensor cache, runs the phases `gsm8k`, `humaneval`, `mbpp`, `ifeval`
  (lm-evaluation-harness 0.4.13 in `~/orchard-venvs/bench`, a scrubbed environment with no token and a home inside the output directory) and
  `bfcl`, each through its own proxy under the stall watcher, stops the server's whole process group, and writes `summary.json`
  (33 tests, 41 mutations, none survived).
- A plumbing run of lm-eval against a fake server showed that the datasets download without a token and that the harness's results parse.
  The 2026-10-19 run used lm-eval defaults, which are not the settings the pilot will use (chat template, 5-shot turns for GSM8K).

Chosen but untested on a real server: `max_gen_toks` 8192 with thinking on. A reply that spends all of it reasoning comes back with empty
content and `finish_reason` length. The proxy counts those (`cut_off`), and the pilot reports the count per task.

## Pilot result (2026-10-20 UTC)

Run on the hardware with the operator's go-ahead: `docs/run-logs/2026-10-19-bytkim-pi-benchmark-pilot.md`. At 20 items per task the fine-tune and the base cannot be told
apart, thinking on or off. Sessions took 3 to 11 minutes. A full run is about 11 hours for both models with thinking on and about 3 hours with thinking off.
The pilot also found that the harness's HumanEval and MBPP instruct tasks cannot score a chat reply; the chat versions in this repository replace them.

## Full run result (2026-10-20 UTC)

The full run (every item, thinking on, 8192 token limit) is in `docs/run-logs/2026-10-20-bytkim-pi-benchmark-full.md`. The fine-tune scores at or above the base on every set;
the gap is clear on GSM8K and MBPP. The card was updated and published on that basis.

## Re-run at 32768 tokens (2026-10-21 UTC)

The empty replies of HumanEval, MBPP and IFEval were asked again with a limit of 32768 tokens (`docs/run-logs/2026-10-21-bytkim-pi-benchmark-rerun.md`). The fine-tune and the base are close on those sets at that limit; the MBPP difference found at 8192 tokens is not clear any more.

## Not decided yet

- Whether a benchmark becomes a stage 7 requirement for every package. The bytkim pilot gives the cost; the decision comes after it.
- Whether the public Clef and RAG-Gate cards get benchmarks. Clef has parity numbers only; RAG-Gate has its own gate-accuracy task.
- MMLU-Pro and a long-context task (needle retrieval at 32K to 128K). Added if the pilot leaves time.
- LiveCodeBench or SWE-bench style agent tasks. They need a sandbox and many turns, and are out of scope for this plan.

## Risks to watch

- Thinking mode: items can use thousands of reasoning tokens; at about 50 tokens per second per stream this sets the cost.
- The drafter's speedup depends on the text, so speed ratios between base and fine-tune across different tasks are not comparable
  without the same outputs. The card reports speed per task and does not average across tasks.
- HumanEval and MBPP are public and may be in either model's training data. The card says so.
- Sampling on the device and the bundle's quantization (bfp8) can lower scores below the numbers the author reports for bf16 weights. The
  card reports what this package scores and says it differs from the author's setup.
