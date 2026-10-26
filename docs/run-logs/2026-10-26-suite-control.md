# Prompt suite control, first chip data (issue 15), 2026-10-26

What was run: `tt-orchard suite-control` built a control from the nearest model, and the control was served on one 2-chip board
under `gozer run --chips 2`. This is the first time any chip has served the prompt suite.

## The control run

| Item | Value |
|---|---|
| Model | Qwen/Qwen3.8-27B (the nearest model) |
| Serving | the 2-chip `qwen3.8-27b-dflash2-p300` bundle, drafter off, sampling on the host (the configuration of BAAI/AREX-2, so the control mirrors it) |
| Cold start | 1,942 s to ready (cold tensor cache) |
| 32-token check | top-1 agreement 0.9375 against the AREX-2 reference (30 of 32), coherent |
| Suite | 8 prompts, `max_tokens` 64, chat format (the template exists and mentions tools), suite sha256 `afc956d9...` |
| Raw replies | `evidence/suite-control-qwen3.8-27b/swap-check.json` (paths replaced by `<CONTROL_DIR>` and `<HOME>`) |

The control was made with `drafter: "off"` in its config because the AREX-2 stage serves with the drafter off. The base has `mtp.*`
tensors, so without the key it would have been served with the drafter on, and a control served another way could differ from the run
for that reason alone. This key is new in `prepare_swap.py` (version 0.8.44).

## What the suite found on these replies

`python3 -m orchard.prompt_suite swap-check.json` gives no reasons, no advisory findings and `"findings": []`. On eight replies from
one healthy model, no check fired. That is the whole false-positive evidence: it does not show that the checks are free of false
positives on other prompts or other models.

Finish reasons: six replies reached the 64-token limit. Five (`chat-capital`, `code-fizzbuzz`, `non-english`, `summarize`, `list-format`) ended inside the
reasoning with an empty answer, and `arithmetic` was cut inside its answer. `multi-turn` finished with an answer and `tool-call` returned a call to
`get_weather` with `{"city": "Lisbon"}`. With thinking on, which is this model's default, 64 tokens is not enough for most prompts to reach an answer.
The checks then read the reasoning text. The identical-replies rule also compares the reasoning text when there is no answer, so it still has something to
compare, but a comparison of reasoning that stops mid-sentence is weaker than a comparison of answers. The next section measures other settings.

## Thinking and token limits, one server boot (A/B)

Same server, same 8 prompts, greedy (`temperature` 0, `seed` 1234), one boot of the control's directory (ready in 266 s on a warm cache), five cells
run one after the other on one 2-chip board. Thinking is switched per request with `chat_template_kwargs: {"enable_thinking": false}`. Raw replies:
`evidence/suite-control-qwen3.8-27b/ab-cells.json` (`ab-summary.json` holds the counts below). Findings are what `orchard/prompt_suite.py` reports on the
cell's replies (the gate's reasons and the advisory list, and every finding of every kind); none of the cells had a control to compare with.

| Thinking | max_tokens | Reached the limit | Answer or tool call present | Cut inside an answer | Findings (blocking / advisory) | Completion tokens | Wall time |
|---|---|---|---|---|---|---|---|
| on | 64 | 6 of 8 | 3 of 8 | 1 (arithmetic) | 0 / 0 | 495 | 40.1 s |
| on | 256 | 0 of 8 | 8 of 8 | 0 | 0 / 0 | 841 | 61.2 s |
| on | 1024 | 0 of 8 | 8 of 8 | 0 | 0 / 0 | 841 | 61.6 s |
| off | 64 | 2 of 8 | 8 of 8 | 2 (code-fizzbuzz, arithmetic) | 0 / 0 | 240 | 25.2 s |
| off | 256 | 0 of 8 | 8 of 8 | 0 | 0 / 0 | 338 | 30.9 s |

- Thinking on at 256 and at 1024 gave token-for-token the same 841 tokens, so the extra room was not used. The longest reply at 256 was `code-fizzbuzz`
  with 253 tokens, three below the limit, so 256 has almost no margin on that prompt. 512 was not tried.
- Thinking off gives every prompt a complete reply at 256 in half the time. At 64 it cuts two code and arithmetic answers. The replies it gives are the
  same answers the thinking-on cells give (same capital, same Russian sentence, same list, same tool call).
- No check changed its verdict in any cell: no blocking and no advisory finding anywhere, including at 64 tokens with thinking on, where five replies had
  an empty answer. The empty-answer check did not fire on them, because the check reads the reasoning text when the answer is empty (a reply that has
  reasoning is not empty). So the 64-token cells do not show the thinking-on replies as failures, and they also do not show what the reply would
  have said.

Recommendation (a measurement on one model, not a rule): the cheapest setting that gives readable, complete answers for all eight prompts is thinking off at
256 tokens (30.9 s, 338 tokens). It does not run the model in its default mode, so it cannot show a failure that only appears while reasoning (a loop in the
reasoning, for example). The cheapest setting that keeps the default mode and gives complete answers is thinking on at 256 (61.2 s), with a thin margin on one prompt; 512 would be the
safer value and was not measured. Thinking on at 64, the current default, costs 40.1 s and gives 2 complete replies. The `SUITE_*` defaults are unchanged. Not shown:
whether a fine-tune needs more room than its base for the same prompts; a model that reasons longer would need a larger limit.

## The base against AREX-2's reference

The control, which is the base model and not AREX-2, agreed with AREX-2's CPU reference on 30 of 32 tokens (0.9375). AREX-2's own stage 2 run
agreed on 31 of 32 (0.96875) in the same serving configuration. Both are above the 0.85 bar, so the 32-token bar cannot tell the base standing in
by mistake from the real fine-tune for this model, which is the weakness issue 15 describes. One model pair and one prompt, so this is an
observation and not a measured margin. It does support the rule that the suite's identical-replies check (a comparison with the control) is
needed beside the bar.

## What was not run

The plan was to serve the suite on BAAI/AREX-2 and ThakiCloud/RAG-Gate-27B as well, so that base and fine-tune replies could be
compared. Both boards were held by other agents (`claude:graft-accept` on board 0, `claude:kernel-probe` on board 1) for the whole time
I waited, about two hours. A retry loop of mine left 53 queued requests under `claude:suite-control` (one per attempt: each `gozer run`
queued and exited, and the ticket stayed). I could not cancel them, because the auto mode classifier denied `gozer cancel`. They are
listed by `gozer queue`; each can be removed with `gozer cancel <ticket>`.

So these questions have no data yet:

- whether the identical-replies rule passes a real fine-tune (`SUITE_IDENTICAL_MIN_PROMPTS`);
- what agreement numbers a fine-tune gets against a control, and whether the base's own top-1 against a fine-tune's reference (0.9375 here, for AREX-2's
  reference) separates a mistaken base from a real fine-tune;
- whether a fine-tune's replies show any suite finding the control does not.

## Provisional values

Every `SUITE_*` value in `orchard/defaults.py` stays a choice. The comments there now say which data exists (one model, eight replies).
