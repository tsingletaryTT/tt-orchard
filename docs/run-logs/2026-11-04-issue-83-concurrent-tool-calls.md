# Issue 83: tool calls damaged when requests overlap, reproduced on the safe configuration (2026-11-04)

Issue 83 reports that a model served with the drafter off and sampling on the host (the safe configuration) returns damaged tool calls when four requests run at the same time, and not when they run one at a time. This run repeats the observation on a different model in the same serving configuration and looks for the trigger. The full numbers are also in a comment on issue 83.

## Setup

- One 2-chip Blackhole board, leased through gozer for the length of the run and released afterwards (all four chips free at the end).
- The published package `episod/sensenova-6.8-code2video-p300`: a 27B model, drafter off, sampling on the host, vLLM with the `qwen3_coder` tool-call parser, `max_num_seqs` 4, `--no-async-scheduling`, context 262,144. It serves with the TT model code that the AREX-2 bundle uses, so the serving path is the same. The weights are a different fine-tune.
- Thinking off (`chat_template_kwargs: {"enable_thinking": false}`), 4,096-token limit, one request per prompt, no streaming.
- The server was started from the warm tensor cache and answered after 3 minutes.

## Test

Eight tasks, each with two or three tool definitions and a request for exactly one call with arguments that can be checked exactly: weather with a unit, the area of a rectangle (12.5 by 8), an email with a recipient, subject and body, a flight search (two airports, a date, a passenger count), a table booking, a thermostat value (21.5), a funds transfer (two account ids, 250.75, a currency) and a task with a title, a due date, a priority and a list of two tags. Each condition sent 40 requests (each task five times), with the same prompts in the same order. Every request and reply was kept.

A reply is **damaged** when it hit the token limit, has no tool call, names a function that is not a defined tool, has arguments that are not valid JSON, has a marker such as `<tool_call` or `<parameter` in the arguments or the text, or leaves a required argument out. A reply is **not exact** when the structure is sound but the function or a value is wrong.

## Results

| Condition | Requests in flight | Temperature | Exact | Damaged | Not exact only |
|---|---|---|---|---|---|
| one at a time | 1 | 0 | 40 | 0 | 0 |
| one at a time | 1 | 0.7, seeded | 40 | 0 | 0 |
| continuous | 2 | 0 | 29 | 4 | 7 |
| continuous | 3 | 0 | 17 | 16 | 7 |
| continuous | 4 | 0 | 17 | 19 | 4 |
| continuous | 4 | 0.7, seeded | 17 | 20 | 3 |
| groups of four started together, each group finished before the next | 4 | 0 | 40 | 0 | 0 |

Kinds of damage at four in flight, temperature 0 (a reply can have more than one): arguments left out 15, a marker inside a string 4, a function name with junk pasted on 2. At temperature 0.7: arguments left out 15, markers 8.

Paired by prompt, one at a time against four in flight: 23 prompts exact only in the one-at-a-time run, none exact only at four in flight (exact sign test p = 2.4e-7), at temperature 0 and again at 0.7 with seeds. At two against three in flight: 14 against 2 (p = 0.004).

Examples of damaged replies at four in flight, each with the good reply for the same prompt at one in flight:

| Task | One in flight | Four in flight |
|---|---|---|
| email | `{"to": "ana.ruiz@example.com", "subject": "Q3 budget review", "body": "Numbers attached, please confirm by Friday."}` | `{"to": "ana.ruiz@example.com", "subject": "Q3 budget review"}` (body left out) |
| flights | `{"origin": "SFO", "destination": "JFK", "date": "2026-12-03", "passengers": 2}` | `{"origin": "SFO", "destination": "JFK", "date": "202026-12-03\n</parameter name=\"passengers\""}` |
| transfer | all four arguments | `{"from_account": "CHK-1042"}` |

## What the runs show

1. **It is the same effect as the issue describes**, on a second model, with the same kinds of damage (cut arguments, a stray `<parameter` marker, a name with junk pasted on).
2. **It grows with concurrency:** 0, 4, 16 and 19 damaged replies at 1, 2, 3 and 4 requests in flight.
3. **Four sequences decoding together is not enough.** Groups of four started together and finished together gave 40 of 40 exact and call text identical to the one-at-a-time run.
4. **The trigger tracks a request arriving while others decode.** Counting the requests that started during a given request's decode (after a 1 s allowance for its own prefill, measured from the client side, so approximate): at temperature 0, requests with none were damaged 0 of 25 times across 2, 3 and 4 in flight; requests with one or more were damaged 39 of 95 times. At temperature 0.7 and four in flight: 1 of 5 against 19 of 35.
5. **No cross-request contamination.** None of the 29 wrong values equals a value that belongs to another task. Each damaged reply is its own request's output, cut or garbled.
6. **The server log shows no errors.** One start-up warning: "TT model Qwen36DFlashForCausalLM does not advertise decode_input_update_contract >= 1; preserving its legacy reset_batch reload behavior. Async decode correctness is not guaranteed until the model adapter implements the explicit contract." Async scheduling is off in this package, so this is a lead (batch recomposition when a request joins). The run did not show that it is the cause.
7. **An observation (no finding follows from it):** at one in flight, temperature 0.7 with seeds gave exactly the same call text as temperature 0 in all 40 requests. The tool-call tokens are probably high-confidence. The run did not check whether the temperature is applied on the host sampling path.

## Limits

- One model, one board, 40 requests per condition, eight tasks with short answers (40 to 100 tokens). The original evidence used 300 BFCL items with longer answers; the rate here is not comparable.
- The arrival measure comes from client timestamps and a fixed 1 s allowance for prefill.
- The damage definition above was written before the runs and not changed after.
- Not tested: the graft configuration (drafter on, device sampling) at one and four in flight; a scheduler setting that never mixes a new request's prefill with running decodes; a model adapter that implements `decode_input_update_contract`; the base model without the fine-tune.

## What to do next

- Add the concurrent probe to the harness: eight tasks, four in flight against one at a time, compared by prompt. It takes about 5 minutes on a warm server and would show this in a stage 2 or stage 4 run, which send one request at a time today.
- Run the three untested cases above; each is a restart of the server and about 10 minutes.
- Until the cause is fixed, a card for a model served this way should say that tool-call quality drops when requests overlap.
